"""
rl_scheduler.py
----------------
The advanced / novel component: a reinforcement learning agent (tabular
Q-learning) that learns a general POLICY for deciding when to start a
flexible job, given the current carbon intensity and how much slack time
remains before its deadline.

How this differs from the GA scheduler (scheduler.py):
- The GA searches for a good schedule for ONE specific set of jobs and ONE
  specific carbon curve. Run it again on a new day / new jobs, and it has
  to search again from scratch.
- The RL agent instead learns a reusable DECISION RULE ("if carbon is high
  and I still have plenty of slack, wait; if carbon is low, or I'm almost
  out of slack, start now") from many training episodes. Once trained, it
  can make an instant decision for a brand new job or a brand new carbon
  curve it has never seen, with no re-optimisation needed.
- Trade-off: the RL agent here decides about ONE job in isolation (no
  shared-resource concurrency constraint), whereas the GA jointly
  schedules multiple competing jobs under a concurrency limit. They are
  solving related but different versions of the problem -- which is
  exactly the kind of honest comparison worth making in the report,
  rather than claiming one simply "beats" the other.

This is intentionally a tabular Q-learning agent (not deep RL): with a
small, discretised state space, a Q-table trains fast, is fully
inspectable (you can print the whole learned policy), and avoids the
extra complexity of neural function approximation when it isn't needed
-- itself a defensible design decision to explain in an interview.
"""

from __future__ import annotations

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


ACTION_WAIT = 0
ACTION_START = 1


# ---------------------------------------------------------------------------
# State discretisation
# ---------------------------------------------------------------------------

def compute_carbon_bins(carbon_series: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Quantile bin edges computed once from the historical carbon series."""
    return np.quantile(carbon_series, np.linspace(0, 1, n_bins + 1))


def bucket_carbon(value: float, bin_edges: np.ndarray) -> int:
    return int(np.clip(np.digitize(value, bin_edges[1:-1]), 0, len(bin_edges) - 2))


def bucket_slack(slack: int) -> int:
    """Discretise remaining slack (slots of flexibility left) into 4 levels."""
    if slack <= 0:
        return 0   # forced -- must start now
    elif slack <= 2:
        return 1   # little slack
    elif slack <= 5:
        return 2   # some slack
    else:
        return 3   # plenty of slack


# ---------------------------------------------------------------------------
# Environment: a single flexible job deciding when to start
# ---------------------------------------------------------------------------

class JobSchedulingEnv:
    """
    One episode = one job. At each time slot before the job starts, the
    agent chooses WAIT or START. If it waits until slack runs out, it is
    forced to start (to guarantee the deadline is met). Reward is the
    negative carbon cost incurred once the job starts (spread over its
    duration), scaled down for stable Q-learning updates.
    """

    def __init__(self, carbon_series: np.ndarray, bin_edges: np.ndarray, power_kw: float = 20.0):
        self.carbon_series = carbon_series
        self.bin_edges = bin_edges
        self.power_kw = power_kw

    def reset(self, duration: int, earliest_start: int, deadline: int):
        self.duration = duration
        self.current_slot = earliest_start
        self.deadline = deadline
        return self._get_state()

    def _get_state(self):
        carbon_now = self.carbon_series[self.current_slot]
        slack = (self.deadline - self.duration) - self.current_slot
        return (bucket_carbon(carbon_now, self.bin_edges), bucket_slack(slack))

    def step(self, action: int):
        slack = (self.deadline - self.duration) - self.current_slot
        forced = slack <= 0

        if action == ACTION_START or forced:
            end = min(self.current_slot + self.duration, len(self.carbon_series))
            carbon_cost = self.carbon_series[self.current_slot:end].sum() * self.power_kw * 0.5
            reward = -carbon_cost / 1000.0  # scaled for stable Q-value magnitudes
            done = True
            info = {"started_at": self.current_slot, "carbon_cost": carbon_cost, "forced": forced}
            return None, reward, done, info

        # WAIT: advance one slot, no immediate cost
        self.current_slot += 1
        return self._get_state(), 0.0, False, {}


# ---------------------------------------------------------------------------
# Tabular Q-learning
# ---------------------------------------------------------------------------

def train_q_learning(
    carbon_series: np.ndarray,
    n_episodes: int = 5000,
    alpha: float = 0.1,
    gamma: float = 1.0,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    seed: int = 42,
):
    rng = random.Random(seed)
    bin_edges = compute_carbon_bins(carbon_series)
    env = JobSchedulingEnv(carbon_series, bin_edges)

    n_carbon_bins = len(bin_edges) - 1
    n_slack_bins = 4
    Q = np.zeros((n_carbon_bins, n_slack_bins, 2))  # state -> action values

    rewards_history = []

    for ep in range(n_episodes):
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * (ep / n_episodes)

        # Randomised job for this training episode -- this variety is what
        # lets the learned policy generalise, rather than memorising one job.
        duration = rng.randint(2, 8)
        earliest_start = rng.randint(0, len(carbon_series) - 20)
        slack_room = rng.randint(2, 15)
        deadline = min(earliest_start + duration + slack_room, len(carbon_series))

        state = env.reset(duration, earliest_start, deadline)
        done = False
        total_reward = 0.0

        while not done:
            if rng.random() < epsilon:
                action = rng.choice([ACTION_WAIT, ACTION_START])
            else:
                action = int(np.argmax(Q[state[0], state[1]]))

            next_state, reward, done, info = env.step(action)
            total_reward += reward

            if done:
                target = reward
            else:
                target = reward + gamma * np.max(Q[next_state[0], next_state[1]])

            Q[state[0], state[1], action] += alpha * (target - Q[state[0], state[1], action])
            state = next_state

        rewards_history.append(total_reward)

    return Q, bin_edges, rewards_history


def greedy_action(Q, bin_edges, carbon_now, slack):
    s = (bucket_carbon(carbon_now, bin_edges), bucket_slack(slack))
    return int(np.argmax(Q[s[0], s[1]]))


def run_policy(Q, bin_edges, carbon_series, duration, earliest_start, deadline, power_kw=20.0):
    """Apply the trained greedy policy to one job, returning its chosen start slot and cost."""
    env = JobSchedulingEnv(carbon_series, bin_edges, power_kw=power_kw)
    state = env.reset(duration, earliest_start, deadline)
    while True:
        action = int(np.argmax(Q[state[0], state[1]]))
        next_state, reward, done, info = env.step(action)
        if done:
            return info["started_at"], info["carbon_cost"], info["forced"]
        state = next_state


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_curve(rewards_history, out_path: str, window: int = 100):
    smoothed = pd.Series(rewards_history).rolling(window).mean()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(smoothed)
    ax.set_xlabel("Training episode")
    ax.set_ylabel(f"Reward (rolling {window}-episode average)")
    ax.set_title("RL agent training progress")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved training curve to {out_path}")


def plot_policy_table(Q, bin_edges, out_path: str):
    """Visualise the learned policy: for each (carbon level, slack level), WAIT or START?"""
    n_carbon_bins = Q.shape[0]
    n_slack_bins = Q.shape[1]
    policy = np.argmax(Q, axis=2)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(policy, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n_slack_bins))
    ax.set_xticklabels(["forced", "little", "some", "plenty"])
    ax.set_yticks(range(n_carbon_bins))
    ax.set_yticklabels([f"bucket {i}" for i in range(n_carbon_bins)])
    ax.set_xlabel("Slack remaining")
    ax.set_ylabel("Carbon intensity level (0=lowest)")
    ax.set_title("Learned policy: green=WAIT, red=START")

    for i in range(n_carbon_bins):
        for j in range(n_slack_bins):
            label = "START" if policy[i, j] == 1 else "WAIT"
            ax.text(j, i, label, ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved policy visualisation to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from dataloader import fetch_carbon_intensity
    from scheduler import generate_synthetic_jobs, evaluate_schedule

    # Train on a longer carbon history for variety across episodes
    carbon_df = fetch_carbon_intensity(days=14)
    carbon_series = carbon_df["carbon_intensity"].values
    print(f"Training on {len(carbon_series)} half-hour slots of carbon data...")

    Q, bin_edges, rewards_history = train_q_learning(carbon_series, n_episodes=5000)

    Path("docs").mkdir(exist_ok=True)
    plot_training_curve(rewards_history, "docs/rl_training_curve.png")
    plot_policy_table(Q, bin_edges, "docs/rl_policy_table.png")

    # --- Evaluate the trained policy on a fresh, held-out test window ---
    test_carbon_df = fetch_carbon_intensity(days=2)
    test_carbon = test_carbon_df["carbon_intensity"].values

    jobs = generate_synthetic_jobs(n_jobs=12, n_slots=len(test_carbon), seed=99)

    naive_schedule = [job["earliest_start"] for job in jobs]
    naive_carbon, _ = evaluate_schedule(naive_schedule, jobs, test_carbon, max_concurrent=999)

    rl_schedule = []
    rl_total_carbon = 0.0
    for job in jobs:
        start, cost, forced = run_policy(
            Q, bin_edges, test_carbon,
            job["duration"], job["earliest_start"], job["deadline"],
            power_kw=job["power_kw"],
        )
        rl_schedule.append(start)
        rl_total_carbon += cost

    saving_pct = (naive_carbon - rl_total_carbon) / naive_carbon * 100
    print(f"\nNaive total carbon:  {naive_carbon:,.0f} gCO2")
    print(f"RL policy total carbon: {rl_total_carbon:,.0f} gCO2")
    print(f"Carbon saving vs naive: {saving_pct:.1f}%")

    print("\nNote: this RL agent decides each job independently (no shared")
    print("concurrency limit), unlike the GA scheduler -- see module docstring.")

    Path("data").mkdir(exist_ok=True)
    pd.DataFrame({
        "job_id": [j["job_id"] for j in jobs],
        "duration_slots": [j["duration"] for j in jobs],
        "naive_start": naive_schedule,
        "rl_start": rl_schedule,
    }).to_csv("data/rl_schedule_result.csv", index=False)
    print("Saved RL schedule details to data/rl_schedule_result.csv")
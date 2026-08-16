from __future__ import annotations

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Anchor all output paths to the project root (the folder containing src/,
# models/, docs/, data/) rather than trusting the current working
# directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


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


def save_policy(Q, bin_edges, path: str):
    import joblib
    joblib.dump({"Q": Q, "bin_edges": bin_edges}, path)
    print(f"Saved trained policy to {path}")


def load_policy(path: str):
    import joblib
    saved = joblib.load(path)
    return saved["Q"], saved["bin_edges"]

"""
    Convenience wrapper for interactive use (e.g. from the dashboard):
    loads the saved trained policy and returns a suggested start slot for
    one new job, given the current carbon curve.

    Also returns the "naive" (start immediately) cost for comparison, so
    callers can show the saving.
    """
def suggest_start_time(carbon_series: np.ndarray, duration_slots: int, deadline_slot: int,
                        power_kw: float, policy_path: str = None):
    
    Q, bin_edges = load_policy(policy_path or str(PROJECT_ROOT / "models" / "rl_policy.joblib"))

    suggested_start, suggested_cost, forced = run_policy(
        Q, bin_edges, carbon_series, duration_slots, 0, deadline_slot, power_kw=power_kw
    )

    naive_cost = carbon_series[0:duration_slots].sum() * power_kw * 0.5

    return {
        "suggested_start_slot": suggested_start,
        "suggested_cost_gco2": suggested_cost,
        "naive_cost_gco2": naive_cost,
        "saving_pct": (naive_cost - suggested_cost) / naive_cost * 100 if naive_cost > 0 else 0.0,
        "forced": forced,
    }

"""
    Builds a grounded, plain-English explanation of WHY the policy chose
    the start time it did -- not just what it chose. This showcases our explainablity feature
    """
def explain_suggestion(carbon_series: np.ndarray, duration_slots: int, deadline_slot: int,
                        power_kw: float, result: dict) -> dict:
    
    feasible_starts = range(0, max(deadline_slot - duration_slots, 0) + 1)
    costs = {
        s: carbon_series[s:s + duration_slots].sum() * power_kw * 0.5
        for s in feasible_starts
    }
    best_start = min(costs, key=costs.get)
    best_cost = costs[best_start]
    worst_cost = max(costs.values())

    chosen_start = result["suggested_start_slot"]
    chosen_cost = result["suggested_cost_gco2"]

    gap_to_optimal_pct = (
        (chosen_cost - best_cost) / best_cost * 100 if best_cost > 0 else 0.0
    )
    window_avg_intensity = float(np.mean(carbon_series[0:deadline_slot]))
    chosen_avg_intensity = float(np.mean(carbon_series[chosen_start:chosen_start + duration_slots]))
    pct_below_window_avg = (
        (window_avg_intensity - chosen_avg_intensity) / window_avg_intensity * 100
        if window_avg_intensity > 0 else 0.0
    )

    lines = []

    if result["forced"]:
        if abs(gap_to_optimal_pct) < 1.0:
            lines.append(
                "There wasn't enough slack before the deadline to wait for a better window, "
                "so the job had to start as early as possible -- but this happens to already be "
                "the lowest-carbon option in the whole timeframe, so no carbon was left on the table."
            )
        else:
            lines.append(
                f"There wasn't enough slack before the deadline to wait for a better window, "
                f"so the job had to start as early as possible. The lowest-carbon option in this "
                f"timeframe would actually have been at slot {best_start} ({best_cost:,.0f} gCO2, "
                f"{gap_to_optimal_pct:.1f}% better) -- but the deadline didn't allow reaching it. "
                f"A longer deadline would give the scheduler more room to find a cleaner window."
            )
    else:
        lines.append(
            f"The policy chose to wait and start at slot {chosen_start} rather than immediately, "
            f"because carbon intensity there ({chosen_avg_intensity:.0f} gCO2/kWh average over the "
            f"job's duration) is {pct_below_window_avg:.0f}% below the "
            f"{window_avg_intensity:.0f} gCO2/kWh average across the whole time it was allowed to consider."
        )

        if abs(gap_to_optimal_pct) < 1.0:
            lines.append(
                "This was in fact the lowest-carbon feasible window available in the whole "
                "search range -- the policy found the true best option."
            )
        else:
            lines.append(
                f"The true lowest-carbon option in this window would have started at slot {best_start} "
                f"({best_cost:,.0f} gCO2 total), which is {gap_to_optimal_pct:.1f}% better than what the "
                f"policy chose. This gap is expected: this is a tabular Q-learning agent with a "
                f"coarse, discretised view of carbon levels and remaining slack, so it approximates "
                f"the optimal choice rather than guaranteeing it."
            )

    lines.append(
        f"Compared to starting immediately, this saves {result['saving_pct']:.1f}% "
        f"({result['naive_cost_gco2']:,.0f} gCO2 naive vs {result['suggested_cost_gco2']:,.0f} gCO2 suggested)."
    )

    return {
        "narrative": " ".join(lines),
        "best_possible_start_slot": best_start,
        "best_possible_cost_gco2": best_cost,
        "gap_to_optimal_pct": gap_to_optimal_pct,
        "window_avg_intensity": window_avg_intensity,
        "chosen_avg_intensity": chosen_avg_intensity,
    }


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

    Path(PROJECT_ROOT / "models").mkdir(exist_ok=True)
    save_policy(Q, bin_edges, str(PROJECT_ROOT / "models" / "rl_policy.joblib"))

    Path(PROJECT_ROOT / "docs").mkdir(exist_ok=True)
    plot_training_curve(rewards_history, str(PROJECT_ROOT / "docs" / "rl_training_curve.png"))
    plot_policy_table(Q, bin_edges, str(PROJECT_ROOT / "docs" / "rl_policy_table.png"))

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

    Path(PROJECT_ROOT / "data").mkdir(exist_ok=True)
    pd.DataFrame({
        "job_id": [j["job_id"] for j in jobs],
        "duration_slots": [j["duration"] for j in jobs],
        "naive_start": naive_schedule,
        "rl_start": rl_schedule,
    }).to_csv(str(PROJECT_ROOT / "data" / "rl_schedule_result.csv"), index=False)
    print("Saved RL schedule details to data/rl_schedule_result.csv")

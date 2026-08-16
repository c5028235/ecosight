"""
scheduler.py
------------
Uses a genetic algorithm (via DEAP) to schedule a set of flexible compute
jobs against a carbon intensity time series, choosing start times that
minimise total carbon emitted, subject to:
  - each job having an earliest start time and a deadline
  - a shared resource limit (only N jobs may run concurrently -- e.g. a
    fixed-size compute cluster)

Why a GA rather than just picking the lowest-carbon slot per job:
if jobs were scheduled independently, they'd likely all pile into the same
lowest-carbon window, breaching the concurrency limit. This turns it into
a genuine combinatorial scheduling problem (which jobs share the good
slots, which get pushed to slightly worse ones) -- exactly the kind of
problem evolutionary algorithms are well suited to, where an exact
solution is expensive but a good-enough solution found by evolving
candidate schedules works well in practice.

The result is compared against a "naive" baseline: every job starts
immediately at its earliest possible start time, ignoring carbon cost
entirely (this is what most schedulers do by default today).
"""

from __future__ import annotations

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Anchor all input/output paths to the project root (the folder containing
# src/, models/, docs/, data/) rather than trusting the current working
# directory, so this script works correctly no matter which folder it's
# run from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

from deap import base, creator, tools, algorithms


# ---------------------------------------------------------------------------
# Synthetic job generation
# ---------------------------------------------------------------------------

def generate_synthetic_jobs(n_jobs: int, n_slots: int, seed: int = 42) -> list[dict]:
    """
    Generate synthetic flexible compute jobs.
    Each job has:
      - duration: how many half-hour slots it needs to run for
      - earliest_start: the earliest slot it's allowed to start in
      - deadline: the slot by which it must have finished
      - power_kw: its power draw while running (used to weight carbon cost)
    """
    rng = random.Random(seed)
    jobs = []
    for i in range(n_jobs):
        duration = rng.randint(2, 8)  # 1-4 hours, in 30-min slots
        earliest_start = rng.randint(0, n_slots // 2)
        slack = rng.randint(4, n_slots // 2)  # how much flexibility beyond the minimum
        deadline = min(earliest_start + duration + slack, n_slots)
        power_kw = round(rng.uniform(5, 50), 1)  # e.g. a batch training job on a small cluster
        jobs.append({
            "job_id": i,
            "duration": duration,
            "earliest_start": earliest_start,
            "deadline": deadline,
            "power_kw": power_kw,
        })
    return jobs


# ---------------------------------------------------------------------------
# Schedule evaluation
# ---------------------------------------------------------------------------

def evaluate_schedule(
    start_times: list[int],
    jobs: list[dict],
    carbon: np.ndarray,
    max_concurrent: int,
) -> tuple[float, int]:
    """
    Given a proposed start time for each job, compute:
      - total_carbon: sum of (power_kw * 0.5h * carbon_intensity) across all
        slots each job occupies
      - violations: count of time slots where more than max_concurrent jobs
        are running simultaneously (a constraint violation)
    """
    n_slots = len(carbon)
    occupancy = np.zeros(n_slots)
    total_carbon = 0.0

    for job, start in zip(jobs, start_times):
        end = start + job["duration"]
        end = min(end, n_slots)
        occupancy[start:end] += 1
        # carbon cost: power (kW) * time (0.5h per slot) * intensity (gCO2/kWh) = gCO2
        total_carbon += job["power_kw"] * 0.5 * carbon[start:end].sum()

    violations = int(np.sum(np.maximum(occupancy - max_concurrent, 0)))
    return total_carbon, violations


def make_fitness(jobs, carbon, max_concurrent, penalty_per_violation=50000):
    def fitness(individual):
        total_carbon, violations = evaluate_schedule(individual, jobs, carbon, max_concurrent)
        return (total_carbon + penalty_per_violation * violations,)
    return fitness


# ---------------------------------------------------------------------------
# GA setup
# ---------------------------------------------------------------------------

def run_ga(jobs: list[dict], carbon: np.ndarray, max_concurrent: int,
           pop_size: int = 100, n_generations: int = 60, seed: int = 42):

    random.seed(seed)
    n_slots = len(carbon)

    # Avoid re-creating DEAP classes if this function is called more than once
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    def init_gene(job):
        lo = job["earliest_start"]
        hi = max(job["deadline"] - job["duration"], lo)
        return random.randint(lo, hi)

    def init_individual():
        return creator.Individual([init_gene(job) for job in jobs])

    toolbox = base.Toolbox()
    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", make_fitness(jobs, carbon, max_concurrent))
    toolbox.register("mate", tools.cxUniform, indpb=0.5)

    def mutate(individual, indpb=0.3):
        for i, job in enumerate(jobs):
            if random.random() < indpb:
                lo = job["earliest_start"]
                hi = max(job["deadline"] - job["duration"], lo)
                individual[i] = random.randint(lo, hi)
        return (individual,)

    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    pop, logbook = algorithms.eaSimple(
        pop, toolbox, cxpb=0.6, mutpb=0.3, ngen=n_generations,
        stats=stats, halloffame=hof, verbose=True,
    )

    best_schedule = list(hof[0])
    best_carbon, best_violations = evaluate_schedule(best_schedule, jobs, carbon, max_concurrent)
    return best_schedule, best_carbon, best_violations, logbook


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_schedule_comparison(
    jobs: list[dict], carbon: np.ndarray,
    naive_schedule: list[int], ga_schedule: list[int],
    naive_carbon: float, ga_carbon: float,
    out_path: str,
):
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                              gridspec_kw={"height_ratios": [1, 2, 2]})

    slots = np.arange(len(carbon))
    axes[0].plot(slots, carbon, color="gray")
    axes[0].set_ylabel("Carbon\nintensity")
    axes[0].set_title("Grid carbon intensity over the scheduling window")

    for ax, schedule, title, total in [
        (axes[1], naive_schedule, f"Naive schedule (starts immediately) -- total: {naive_carbon:,.0f} gCO2", naive_carbon),
        (axes[2], ga_schedule, f"GA-optimised schedule -- total: {ga_carbon:,.0f} gCO2", ga_carbon),
    ]:
        for job, start in zip(jobs, schedule):
            ax.barh(job["job_id"], job["duration"], left=start, height=0.6, color="tab:blue", alpha=0.7)
        ax.set_ylabel("Job ID")
        ax.set_title(title)

    axes[-1].set_xlabel("Half-hour time slot")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved schedule comparison plot to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from dataloader import fetch_carbon_intensity

    carbon_df = fetch_carbon_intensity(days=2)
    carbon = carbon_df["carbon_intensity"].values
    n_slots = len(carbon)
    print(f"Carbon intensity series: {n_slots} half-hour slots")

    N_JOBS = 12
    MAX_CONCURRENT = 3

    jobs = generate_synthetic_jobs(n_jobs=N_JOBS, n_slots=n_slots)
    print(f"Generated {N_JOBS} synthetic jobs, max {MAX_CONCURRENT} concurrent.")

    # --- Naive baseline: everyone starts at their earliest possible time ---
    naive_schedule = [job["earliest_start"] for job in jobs]
    naive_carbon, naive_violations = evaluate_schedule(naive_schedule, jobs, carbon, MAX_CONCURRENT)
    print(f"Naive schedule: {naive_carbon:,.0f} gCO2, violations: {naive_violations}")

    # --- GA-optimised schedule ---
    ga_schedule, ga_carbon, ga_violations, logbook = run_ga(
        jobs, carbon, MAX_CONCURRENT, pop_size=100, n_generations=60
    )
    print(f"GA schedule: {ga_carbon:,.0f} gCO2, violations: {ga_violations}")

    saving_pct = (naive_carbon - ga_carbon) / naive_carbon * 100
    print(f"\nCarbon saving vs naive schedule: {saving_pct:.1f}%")

    (PROJECT_ROOT / "docs").mkdir(exist_ok=True)
    plot_schedule_comparison(
        jobs, carbon, naive_schedule, ga_schedule, naive_carbon, ga_carbon,
        out_path=str(PROJECT_ROOT / "docs" / "schedule_comparison.png"),
    )

    (PROJECT_ROOT / "data").mkdir(exist_ok=True)
    pd.DataFrame({
        "job_id": [j["job_id"] for j in jobs],
        "duration_slots": [j["duration"] for j in jobs],
        "power_kw": [j["power_kw"] for j in jobs],
        "naive_start": naive_schedule,
        "ga_start": ga_schedule,
    }).to_csv(str(PROJECT_ROOT / "data" / "schedule_result.csv"), index=False)
    print("Saved schedule details to data/schedule_result.csv")
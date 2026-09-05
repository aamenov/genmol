"""Common-stream simulation of fragment-score estimators.

The oracle is deterministic for every ``(fragment, context)`` pair, while the
same fragment's molecular score varies across contexts.  This isolates the
winner's-curse mechanism without spending PMO oracle calls or requiring a GPU.
It is a diagnostic experiment, not a molecular benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.exps.pmo.main.genmol.fragment_population import (
    FragmentObservation,
    FragmentPopulation,
    FragmentSeed,
)


VARIANTS = {
    "released": {"mode": "released", "min_support": 1, "prior_strength": 0.0},
    "running_mean": {"mode": "mean", "min_support": 1, "prior_strength": 0.0},
    "support3": {"mode": "mean", "min_support": 3, "prior_strength": 0.0},
    "shrink10": {"mode": "bayes", "min_support": 1, "prior_strength": 10.0},
}


@dataclass(frozen=True)
class SimulationConfig:
    fragments: int
    capacity: int
    observations_per_fragment: int
    replicates: int
    context_stds: tuple[float, ...]
    checkpoint_every: int
    seed: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fragments", type=int, default=200)
    parser.add_argument("--capacity", type=int, default=20)
    parser.add_argument("--observations-per-fragment", type=int, default=20)
    parser.add_argument("--replicates", type=int, default=30)
    parser.add_argument("--context-stds", type=float, nargs="+", default=[0.05, 0.2, 0.4])
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _validate(config: SimulationConfig) -> None:
    if config.fragments < 2:
        raise ValueError("fragments must be at least 2")
    if not 1 <= config.capacity < config.fragments:
        raise ValueError("capacity must lie in [1, fragments)")
    if config.observations_per_fragment < 2:
        raise ValueError("at least two observations per fragment are required")
    if config.replicates < 1 or config.checkpoint_every < 1:
        raise ValueError("replicates and checkpoint-every must be positive")
    if not config.context_stds or any(value < 0 for value in config.context_stds):
        raise ValueError("context standard deviations must be nonnegative")


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _one_replicate(
    *,
    config: SimulationConfig,
    context_std: float,
    replicate: int,
) -> list[dict[str, float | int | str]]:
    # Truth and stream order are paired across noise levels; only context
    # offsets depend on ``context_std``.
    truth_rng = np.random.default_rng(np.random.SeedSequence([config.seed, replicate, 0]))
    order_rng = np.random.default_rng(np.random.SeedSequence([config.seed, replicate, 1]))
    context_rng = np.random.default_rng(
        np.random.SeedSequence([config.seed, replicate, 2, int(context_std * 1_000_000)])
    )

    names = [f"fragment_{index:04d}" for index in range(config.fragments)]
    latent_means = {
        name: float(value)
        for name, value in zip(names, truth_rng.uniform(0.1, 0.9, config.fragments))
    }
    scores: dict[tuple[str, int], float] = {}
    observations: list[tuple[str, int, float]] = []
    for fragment in names:
        offsets = context_rng.normal(0.0, context_std, config.observations_per_fragment)
        # Center each finite context panel so ``true_means`` remains the exact
        # per-fragment average before clipping, up to boundary effects.
        offsets -= offsets.mean()
        for context_index, offset in enumerate(offsets):
            score = float(np.clip(latent_means[fragment] + offset, 0.0, 1.0))
            scores[(fragment, context_index)] = score
            observations.append((fragment, context_index, score))

    # Clipping changes boundary-fragment expectations.  The exact empirical
    # panel mean is therefore the ground truth each estimator can recover.
    true_means = {
        fragment: float(
            np.mean(
                [scores[(fragment, index)] for index in range(config.observations_per_fragment)]
            )
        )
        for fragment in names
    }

    initial_fragments = list(order_rng.choice(names, size=config.capacity, replace=False))
    initial_events = [(fragment, 0, scores[(fragment, 0)]) for fragment in initial_fragments]
    initial_set = {(fragment, context) for fragment, context, _ in initial_events}
    remaining = [row for row in observations if (row[0], row[1]) not in initial_set]
    order_rng.shuffle(remaining)

    oracle_top = sorted(true_means, key=true_means.get, reverse=True)[: config.capacity]
    oracle_top_set = set(oracle_top)
    optimum = float(np.mean([true_means[fragment] for fragment in oracle_top]))
    prior_mean = float(np.mean(list(true_means.values())))
    seed_rows = sorted(
        [FragmentSeed(fragment, score) for fragment, _, score in initial_events],
        key=lambda seed: (seed.score, seed.fragment),
        reverse=True,
    )
    fragment_lookup: dict[str, tuple[str, ...]] = {}
    for fragment, context_index, _ in remaining:
        fragment_lookup[f"{fragment}|{context_index}"] = (fragment,)

    rows: list[dict[str, float | int | str]] = []
    for variant, settings in VARIANTS.items():
        kwargs = {
            "capacity": config.capacity,
            "mode": settings["mode"],
            "fragmenter": lambda molecule, lookup=fragment_lookup: lookup[molecule],
            "min_support": settings["min_support"],
        }
        if settings["mode"] in {"mean", "bayes"}:
            kwargs["legacy_seed_count"] = 1
        if settings["mode"] == "bayes":
            kwargs["prior_strength"] = settings["prior_strength"]
            kwargs["prior_mean"] = prior_mean
        population = FragmentPopulation(seed_rows, **kwargs)

        for step, (fragment, context_index, score) in enumerate(remaining, start=1):
            molecule = f"{fragment}|{context_index}"
            population.observe(
                FragmentObservation(
                    observation_id=molecule,
                    child_smiles=molecule,
                    child_score=score,
                )
            )
            if step % config.checkpoint_every != 0 and step != len(remaining):
                continue

            active_rows = population.active_rows()
            active = [active_fragment for _, active_fragment in active_rows]
            active_true_mean = float(np.mean([true_means[item] for item in active]))
            optimism = float(
                np.mean(
                    [estimated - true_means[item] for estimated, item in active_rows]
                )
            )
            rows.append(
                {
                    "context_std": context_std,
                    "replicate": replicate,
                    "variant": variant,
                    "observations": step,
                    "active_true_mean": active_true_mean,
                    "optimum_true_mean": optimum,
                    "regret": optimum - active_true_mean,
                    "top_v_jaccard": len(set(active) & oracle_top_set)
                    / len(set(active) | oracle_top_set),
                    "active_score_optimism": optimism,
                }
            )
    return rows


def _first_threshold(group: pd.DataFrame, threshold: float) -> float:
    reached = group[group["active_true_mean"] >= threshold]
    return math.nan if reached.empty else float(reached["observations"].iloc[0])


def main() -> None:
    args = _parse_args()
    config = SimulationConfig(
        fragments=args.fragments,
        capacity=args.capacity,
        observations_per_fragment=args.observations_per_fragment,
        replicates=args.replicates,
        context_stds=tuple(args.context_stds),
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
    )
    _validate(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    for context_std in config.context_stds:
        for replicate in range(config.replicates):
            rows.extend(
                _one_replicate(
                    config=config,
                    context_std=context_std,
                    replicate=replicate,
                )
            )
    trajectories = pd.DataFrame(rows)
    trajectories.to_csv(args.output_dir / "trajectories.csv", index=False)

    final_observation = trajectories["observations"].max()
    final = trajectories[trajectories["observations"] == final_observation]
    aggregates = (
        final.groupby(["context_std", "variant"])
        .agg(
            regret_mean=("regret", "mean"),
            regret_std=("regret", "std"),
            top_v_jaccard_mean=("top_v_jaccard", "mean"),
            optimism_mean=("active_score_optimism", "mean"),
            optimism_std=("active_score_optimism", "std"),
        )
        .reset_index()
    )
    thresholds = []
    for (context_std, replicate, variant), group in trajectories.groupby(
        ["context_std", "replicate", "variant"]
    ):
        optimum = float(group["optimum_true_mean"].iloc[0])
        thresholds.append(
            {
                "context_std": context_std,
                "replicate": replicate,
                "variant": variant,
                "observations_to_95pct_optimum": _first_threshold(
                    group.sort_values("observations"), 0.95 * optimum
                ),
            }
        )
    threshold_frame = pd.DataFrame(thresholds)
    threshold_summary = (
        threshold_frame.groupby(["context_std", "variant"], dropna=False)
        .agg(
            observations_to_95pct_median=("observations_to_95pct_optimum", "median"),
            threshold_success_rate=("observations_to_95pct_optimum", lambda values: values.notna().mean()),
        )
        .reset_index()
    )
    aggregates = aggregates.merge(threshold_summary, on=["context_std", "variant"])
    aggregates.to_csv(args.output_dir / "summary.csv", index=False)

    metadata = {
        "experiment": "deterministic contextual fragment-score simulation",
        "scientific_status": "mechanism diagnostic; not a molecular or PMO result",
        "git_commit": _git_commit(),
        "config": asdict(config),
        "variants": VARIANTS,
        "outputs": ["trajectories.csv", "summary.csv"],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(aggregates.to_string(index=False))
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

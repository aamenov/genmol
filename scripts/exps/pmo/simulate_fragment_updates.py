"""Common-stream simulation of fragment-score estimators.

The oracle is deterministic for every ``(fragment, context)`` pair, while the
same fragment's molecular score varies across contexts.  This isolates the
winner's-curse mechanism without spending PMO oracle calls or requiring a GPU.
It is a diagnostic experiment, not a molecular benchmark.
"""

from __future__ import annotations

import argparse
import datetime as datetime_module
import fcntl
import hashlib
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
from scripts.exps.pmo.main.genmol.experiment_io import (
    sha256_config,
    sha256_file,
    write_manifest,
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


class OutputDirectoryLock:
    """Prevent concurrent publishers from targeting the same result directory."""

    def __init__(self, destination: Path):
        lock_path = destination.parent / f".{destination.name}.lock"
        self._descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(self._descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(self._descriptor)
            self._descriptor = -1
            raise

    def close(self) -> None:
        if self._descriptor < 0:
            return
        fcntl.flock(self._descriptor, fcntl.LOCK_UN)
        os.close(self._descriptor)
        self._descriptor = -1

    def __enter__(self) -> "OutputDirectoryLock":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


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


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def _git_metadata() -> dict[str, Any]:
    status = _git_output("status", "--porcelain=v1")
    tracked_diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    return {
        "commit": _git_output("rev-parse", "HEAD"),
        "branch": _git_output("branch", "--show-current"),
        "dirty": bool(status),
        "status": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
    }


def _utc_timestamp() -> str:
    return datetime_module.datetime.now(datetime_module.timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")


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
    # Recreate the same standardized context panel for every noise level.  The
    # comparison is therefore paired; ``context_std`` only rescales offsets.
    context_rng = np.random.default_rng(np.random.SeedSequence([config.seed, replicate, 2]))

    names = [f"fragment_{index:04d}" for index in range(config.fragments)]
    latent_means = {
        name: float(value)
        for name, value in zip(names, truth_rng.uniform(0.1, 0.9, config.fragments))
    }
    scores: dict[tuple[str, int], float] = {}
    observations: list[tuple[str, int, float]] = []
    for fragment in names:
        offsets = context_rng.normal(0.0, 1.0, config.observations_per_fragment)
        # Center each finite context panel so ``true_means`` remains the exact
        # per-fragment average before clipping, up to boundary effects.
        offsets = (offsets - offsets.mean()) * context_std
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
    panel_sample_sds = [
        float(
            np.std(
                [
                    scores[(fragment, index)]
                    for index in range(config.observations_per_fragment)
                ],
                ddof=1,
            )
        )
        for fragment in names
    ]
    within_sum_squares = sum(
        sum(
            (scores[(fragment, index)] - true_means[fragment]) ** 2
            for index in range(config.observations_per_fragment)
        )
        for fragment in names
    )
    within_degrees_freedom = config.fragments * (
        config.observations_per_fragment - 1
    )
    achieved_mean_within_sd = float(np.mean(panel_sample_sds))
    achieved_pooled_within_sd = math.sqrt(
        within_sum_squares / within_degrees_freedom
    )
    clipped_score_fraction = sum(
        score in {0.0, 1.0} for score in scores.values()
    ) / len(scores)

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

        def record(observation_count: int) -> None:
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
                    # This is the configured scale before centering/clipping,
                    # not the achieved conditional standard deviation.
                    "context_scale": context_std,
                    "replicate": replicate,
                    "variant": variant,
                    "observations": observation_count,
                    "achieved_mean_within_fragment_sample_sd": achieved_mean_within_sd,
                    "achieved_pooled_within_fragment_sample_sd": achieved_pooled_within_sd,
                    "clipped_score_fraction": clipped_score_fraction,
                    "active_true_mean": active_true_mean,
                    "optimum_true_mean": optimum,
                    "regret": optimum - active_true_mean,
                    "top_v_jaccard": len(set(active) & oracle_top_set)
                    / len(set(active) | oracle_top_set),
                    "active_score_optimism": optimism,
                }
            )

        record(0)

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
            record(step)
    return rows


def _first_threshold(group: pd.DataFrame, threshold: float) -> float:
    reached = group[group["active_true_mean"] >= threshold]
    return math.nan if reached.empty else float(reached["observations"].iloc[0])


def _simulation_frames(config: SimulationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate(config)

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

    final_observation = trajectories["observations"].max()
    final = trajectories[trajectories["observations"] == final_observation]
    aggregates = (
        final.groupby(["context_scale", "variant"])
        .agg(
            regret_mean=("regret", "mean"),
            regret_std=("regret", "std"),
            top_v_jaccard_mean=("top_v_jaccard", "mean"),
            optimism_mean=("active_score_optimism", "mean"),
            optimism_std=("active_score_optimism", "std"),
            achieved_mean_within_fragment_sample_sd=(
                "achieved_mean_within_fragment_sample_sd",
                "mean",
            ),
            achieved_pooled_within_fragment_sample_sd=(
                "achieved_pooled_within_fragment_sample_sd",
                "mean",
            ),
            clipped_score_fraction=("clipped_score_fraction", "mean"),
        )
        .reset_index()
    )
    thresholds = []
    for (context_scale, replicate, variant), group in trajectories.groupby(
        ["context_scale", "replicate", "variant"]
    ):
        optimum = float(group["optimum_true_mean"].iloc[0])
        thresholds.append(
            {
                "context_scale": context_scale,
                "replicate": replicate,
                "variant": variant,
                "observations_to_95pct_optimum": _first_threshold(
                    group.sort_values("observations"), 0.95 * optimum
                ),
            }
        )
    threshold_frame = pd.DataFrame(thresholds)
    threshold_summary = (
        threshold_frame.groupby(["context_scale", "variant"], dropna=False)
        .agg(
            observations_to_95pct_median=("observations_to_95pct_optimum", "median"),
            threshold_successes=("observations_to_95pct_optimum", "count"),
            threshold_trials=("observations_to_95pct_optimum", "size"),
            threshold_success_rate=(
                "observations_to_95pct_optimum",
                lambda values: values.notna().mean(),
            ),
        )
        .reset_index()
    )
    aggregates = aggregates.merge(
        threshold_summary,
        on=["context_scale", "variant"],
    )
    return trajectories, aggregates


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def run_simulation(
    config: SimulationConfig,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, Path]:
    """Run the diagnostic and atomically publish a new immutable directory."""

    _validate(config)
    destination = Path(output_dir).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_lock = OutputDirectoryLock(destination)
    except BlockingIOError as error:
        raise RuntimeError(f"another simulation owns output {destination}") from error

    with output_lock:
        if destination.exists():
            raise FileExistsError(destination)
        temporary = Path(
            tempfile.mkdtemp(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
            )
        )
        started = time.monotonic()
        try:
            trajectories, aggregates = _simulation_frames(config)
            trajectories_path = temporary / "trajectories.csv"
            summary_path = temporary / "summary.csv"
            trajectories.to_csv(trajectories_path, index=False)
            aggregates.to_csv(summary_path, index=False)
            outputs = {
                "trajectories": {
                    "path": trajectories_path.name,
                    "sha256": sha256_file(trajectories_path),
                    "size_bytes": trajectories_path.stat().st_size,
                    "rows": len(trajectories),
                },
                "summary": {
                    "path": summary_path.name,
                    "sha256": sha256_file(summary_path),
                    "size_bytes": summary_path.stat().st_size,
                    "rows": len(aggregates),
                },
            }

            metadata = {
                "schema_version": 2,
                "experiment": "deterministic contextual fragment-score simulation",
                "scientific_status": "mechanism diagnostic; not a molecular or PMO result",
                "created_at": _utc_timestamp(),
                "elapsed_seconds": time.monotonic() - started,
                "context_panel_coupling": (
                    "paired latent means, observation order, and standardized context offsets; "
                    "context_scale only rescales offsets before clipping"
                ),
                "context_scale_semantics": (
                    "Configured pre-clipping offset scale, historically exposed by the "
                    "--context-stds CLI; achieved within-fragment sample SD and clipping "
                    "fraction are reported separately."
                ),
                "shrinkage_prior": (
                    "complete-panel empirical global mean; privileged diagnostic information, "
                    "not an online or PMO-available prior"
                ),
                "config": asdict(config),
                "config_sha256": sha256_config(asdict(config)),
                "variants": VARIANTS,
                "code": {
                    "git": _git_metadata(),
                    "script_path": str(Path(__file__).resolve()),
                    "script_sha256": sha256_file(Path(__file__).resolve()),
                },
                "runtime": {
                    "hostname": socket.gethostname(),
                    "python_version": platform.python_version(),
                    "executable": sys.executable,
                    "numpy_version": np.__version__,
                    "pandas_version": pd.__version__,
                },
                "outputs": outputs,
                "caveats": [
                    "This toy mechanism diagnostic uses no molecules and no PMO oracle.",
                    "Each observation contains exactly one fragment and support is exhaustive and balanced.",
                    "The shrinkage arm uses privileged complete-panel prior information.",
                    "A positive result supports only the variance premise; it cannot establish fewer oracle calls.",
                ],
            }
            write_manifest(temporary / "manifest.json", metadata)
            os.rename(temporary, destination)
            _fsync_directory(destination.parent)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
    return aggregates, destination


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
    aggregates, destination = run_simulation(config, args.output_dir)
    print(aggregates.to_string(index=False))
    print(f"Results: {destination}")


if __name__ == "__main__":
    main()

"""Build an auditable PDF for the fixed QED Bayesian-shrinkage ablation.

The report consumes a completed schema-3 collector manifest.  It never loads a
checkpoint or runs an oracle.  Scalar results come from the collector's
hash-addressed CSV; learning curves come from the already validated run
summaries after their paths and SHA-256 hashes are checked again.
"""

from __future__ import annotations

import argparse
import csv
import datetime as datetime_module
import fcntl
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import statistics
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from pypdf import PdfReader  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4, landscape  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.pdfgen import canvas as reportlab_canvas  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.exps.pmo.main.genmol.experiment_io import (  # noqa: E402
    sha256_file,
    trajectory_auc,
)


REPORT_SCHEMA_VERSION = 1
COLLECTION_SCHEMA_VERSION = 3
EXPECTED_ORACLE = "qed"
EXPECTED_SEEDS = (0, 1, 2)
EXPECTED_VARIANTS = (
    "released",
    "running_mean",
    "shrink1",
    "shrink3",
    "shrink10",
    "shrink30",
)
EXPECTED_PRIOR_STRENGTH = {
    "released": 0.0,
    "running_mean": 0.0,
    "shrink1": 1.0,
    "shrink3": 3.0,
    "shrink10": 10.0,
    "shrink30": 30.0,
}
EXPECTED_POLICY_MODE = {
    "released": "released",
    "running_mean": "mean",
    "shrink1": "bayes",
    "shrink3": "bayes",
    "shrink10": "bayes",
    "shrink30": "bayes",
}
EXPECTED_BUDGET = 1_000
EXPECTED_REPORTING_FREQUENCY = 100
EXPECTED_PRIOR_MEAN = 0.5
EXPECTED_MODEL_SHA256 = (
    "8d00aa47b02f64bf39ff6b0b2e786f213587366fc2c3d29712a00f3f84108dd6"
)
METRICS = (
    "all_auc_top_1",
    "all_auc_top_10",
    "all_auc_top_100",
    "all_top_1",
    "all_top_10",
    "all_top_100",
)
SHARED_CONFIG_FIELDS = (
    "experiment_id",
    "scientific_status",
    "matrix_path",
    "matrix_sha256",
    "oracle",
    "model_path",
    "vocab_path",
    "device",
    "max_oracle_calls",
    "reporting_frequency",
    "checkpoint_every",
    "max_iterations",
    "population_size",
    "warmup",
    "legacy_warmup_off_by_one",
    "gamma",
    "softmax_temp",
    "randomness",
    "guidance_scale",
    "min_mol_size",
    "max_mol_size",
    "min_support",
    "parent_control",
    "legacy_seed_count",
    "delta_attribution",
    "population_sampling_order",
    "statistical_duplicate_policy",
    "released_duplicate_policy",
    "durable_events",
)
REQUIRED_CSV_FIELDS = {
    "experiment_id",
    "run_id",
    "oracle",
    "variant",
    "seed",
    "summary_schema_version",
    "status",
    "oracle_budget",
    "all_oracle_calls",
    "charged_child_count",
    "elapsed_seconds",
    "reporting_frequency",
    "prior_mean",
    "prior_strength",
    "prior_mean_source",
    "model_path",
    "model_sha256",
    "vocabulary_path",
    "vocabulary_sha256",
    "config_sha256",
    "source_summary_sha256",
    *METRICS,
}
PDF_HEADINGS = (
    "GenMol QED Bayesian Shrinkage Ablation",
    "Aggregate scalar metrics",
    "Top-10 learning curves",
    "Bayesian shrinkage dose response",
    "Per-seed results",
    "Integrity and limitations",
)
COLORS = {
    "released": "#4C566A",
    "running_mean": "#0072B2",
    "shrink1": "#009E73",
    "shrink3": "#56B4E9",
    "shrink10": "#E69F00",
    "shrink30": "#D55E00",
}


class ReportError(ValueError):
    """The collection cannot support the fixed Bayesian-ablation report."""


@dataclass(frozen=True)
class RunEvidence:
    run_id: str
    variant: str
    seed: int
    config: dict[str, Any]
    row: dict[str, str]
    values: dict[str, float]
    trajectory: tuple[tuple[int, float], ...]
    summary_path: Path
    summary_sha256: str
    provenance: dict[str, Any]
    launch: dict[str, Any]


@dataclass(frozen=True)
class ReportData:
    collection_path: Path
    collection_sha256: str
    collection: dict[str, Any]
    csv_path: Path
    csv_sha256: str
    matrix_path: Path
    matrix_sha256: str
    runs: tuple[RunEvidence, ...]
    aggregates: dict[str, Any]
    common_config: dict[str, Any]
    caveats: tuple[str, ...]


class _DestinationLock:
    def __init__(self, path: Path):
        self.path = path
        self._handle = path.open("a+b")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self._handle.close()
            raise

    def __enter__(self) -> _DestinationLock:
        return self

    def __exit__(self, *_: Any) -> None:
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()


class _InvariantCanvas(reportlab_canvas.Canvas):
    """ReportLab canvas with fixed document identifiers and timestamps."""

    def __init__(self, *args: Any, **kwargs: Any):
        kwargs["invariant"] = 1
        super().__init__(*args, **kwargs)
        self.setTitle("GenMol QED Bayesian Shrinkage Ablation")
        self.setAuthor("GenMol v2 reproducible experiment harness")
        self.setSubject("Exploratory fixed-budget fragment-vocabulary ablation")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value}")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise ReportError(f"invalid {label} at {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReportError(f"{label} at {path} must contain an object")
    return value


def _lexical_absolute(path: str | os.PathLike[str]) -> Path:
    if not isinstance(path, (str, os.PathLike)):
        raise ReportError("path must be a string or path-like value")
    return Path(os.path.abspath(Path(path).expanduser()))


def _safe_file(
    path: str | os.PathLike[str],
    label: str,
    *,
    contained_by: Path | None = None,
) -> Path:
    try:
        lexical = _lexical_absolute(path)
    except (TypeError, ValueError, OSError) as error:
        raise ReportError(f"{label} must be a valid path") from error
    for component in (lexical, *lexical.parents):
        if component.is_symlink():
            raise ReportError(f"{label} must not contain symlinks: {component}")
    if not lexical.is_file():
        raise ReportError(f"{label} is not a file: {lexical}")
    resolved = lexical.resolve(strict=True)
    if contained_by is not None:
        try:
            resolved.relative_to(contained_by)
        except ValueError as error:
            raise ReportError(
                f"{label} escapes its trusted root {contained_by}: {resolved}"
            ) from error
    return resolved


def _safe_directory(path: str | os.PathLike[str], label: str) -> Path:
    try:
        lexical = _lexical_absolute(path)
    except (TypeError, ValueError, OSError) as error:
        raise ReportError(f"{label} must be a valid path") from error
    for component in (lexical, *lexical.parents):
        if component.is_symlink():
            raise ReportError(f"{label} must not contain symlinks: {component}")
    if not lexical.is_dir():
        raise ReportError(f"{label} is not a directory: {lexical}")
    return lexical.resolve(strict=True)


def _mapping(container: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    value = container.get(key)
    if not isinstance(value, Mapping):
        raise ReportError(f"{context}.{key} must be an object")
    return value


def _sequence(container: Mapping[str, Any], key: str, context: str) -> Sequence[Any]:
    value = container.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ReportError(f"{context}.{key} must be an array")
    return value


def _integer(value: Any, context: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReportError(f"{context} must be an integer")
    if minimum is not None and value < minimum:
        raise ReportError(f"{context} must be at least {minimum}")
    return value


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ReportError(f"{context} must be a finite number")
    return result


def _csv_integer(value: str | None, context: str) -> int:
    try:
        result = int(value or "")
    except ValueError as error:
        raise ReportError(f"{context} must be an integer") from error
    return result


def _csv_number(value: str | None, context: str) -> float:
    try:
        result = float(value or "")
    except ValueError as error:
        raise ReportError(f"{context} must be a finite number") from error
    if not math.isfinite(result):
        raise ReportError(f"{context} must be a finite number")
    return result


def _equal(actual: Any, expected: Any, context: str) -> None:
    if actual != expected:
        raise ReportError(f"{context} is {actual!r}, expected {expected!r}")


def _close(actual: float, expected: float, context: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ReportError(f"{context} is {actual!r}, expected {expected!r}")


def _sha256_digest(value: Any, context: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ReportError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _resolve_summary_path(
    raw_path: Any,
    *,
    experiment_root: Path,
    context: str,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ReportError(f"{context} must be a nonempty path")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = experiment_root / candidate
    return _safe_file(candidate, context, contained_by=experiment_root)


def _read_csv_rows(
    path: Path, expected_hash: str, expected_columns: Sequence[Any]
) -> list[dict[str, str]]:
    _equal(sha256_file(path), expected_hash, "results CSV SHA-256")
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ReportError("results CSV has no header")
            if not all(isinstance(column, str) for column in expected_columns):
                raise ReportError("collection results_csv.columns must contain strings")
            _equal(reader.fieldnames, list(expected_columns), "results CSV columns")
            missing = REQUIRED_CSV_FIELDS - set(reader.fieldnames)
            if missing:
                raise ReportError(f"results CSV is missing columns: {sorted(missing)}")
            rows = [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as error:
        raise ReportError(f"invalid results CSV at {path}: {error}") from error
    _equal(sha256_file(path), expected_hash, "results CSV SHA-256 after reading")
    return rows


def _expected_job_set() -> set[tuple[str, str, int]]:
    return {
        (EXPECTED_ORACLE, variant, seed)
        for variant in EXPECTED_VARIANTS
        for seed in EXPECTED_SEEDS
    }


def _validate_collection_header(
    collection: Mapping[str, Any], collection_path: Path
) -> tuple[Path, str, Path, str, Path]:
    _equal(collection.get("schema_version"), COLLECTION_SCHEMA_VERSION, "collection schema")
    _equal(collection.get("collection_complete"), True, "collection completeness")
    _equal(collection.get("missing_jobs"), [], "collection missing_jobs")
    _equal(collection.get("skipped"), [], "collection skipped jobs")
    _equal(collection.get("run_count"), 18, "collection run_count")

    matrix = _mapping(collection, "matrix", "collection")
    _equal(matrix.get("expected_job_count"), 18, "matrix expected_job_count")
    _equal(
        matrix.get("recorded_by_all_collected_runs"),
        True,
        "matrix provenance completeness",
    )
    jobs = _sequence(matrix, "expected_jobs", "collection.matrix")
    parsed_jobs: list[tuple[str, str, int]] = []
    for index, row in enumerate(jobs):
        if not isinstance(row, Mapping):
            raise ReportError(f"matrix expected_jobs[{index}] must be an object")
        parsed_jobs.append(
            (
                str(row.get("oracle")),
                str(row.get("variant")),
                _integer(row.get("seed"), f"matrix expected_jobs[{index}].seed"),
            )
        )
    _equal(len(set(parsed_jobs)), 18, "unique matrix jobs")
    _equal(set(parsed_jobs), _expected_job_set(), "matrix job set")

    matrix_path = _safe_file(matrix.get("path"), "matrix source")
    matrix_hash = _sha256_digest(matrix.get("sha256"), "matrix SHA-256")
    _equal(sha256_file(matrix_path), matrix_hash, "matrix source SHA-256")

    results = _mapping(collection, "results_csv", "collection")
    _equal(results.get("immutable"), True, "results CSV immutable marker")
    raw_csv_path = results.get("path")
    if not isinstance(raw_csv_path, str) or not raw_csv_path:
        raise ReportError("collection.results_csv.path must be nonempty")
    csv_reference = Path(raw_csv_path)
    if csv_reference.is_absolute() or len(csv_reference.parts) != 1:
        raise ReportError("results CSV must be a contained basename beside the collection")
    csv_hash = _sha256_digest(results.get("sha256"), "results CSV SHA-256")
    _equal(
        csv_reference.name,
        f"results.{csv_hash}.csv",
        "hash-named results CSV path",
    )
    csv_path = _safe_file(
        collection_path.parent / csv_reference,
        "results CSV",
        contained_by=collection_path.parent,
    )
    experiment_root = _safe_directory(
        collection.get("experiment_root"), "collection experiment root"
    )
    return csv_path, csv_hash, matrix_path, matrix_hash, experiment_root


def _validate_config(
    config: Mapping[str, Any],
    *,
    variant: str,
    seed: int,
    experiment_id: str,
) -> None:
    _equal(config.get("experiment_id"), experiment_id, "run config experiment_id")
    _equal(config.get("oracle"), EXPECTED_ORACLE, "run config oracle")
    _equal(config.get("variant"), variant, "run config variant")
    _equal(config.get("seed"), seed, "run config seed")
    _equal(config.get("max_oracle_calls"), EXPECTED_BUDGET, "run oracle budget")
    _equal(
        config.get("reporting_frequency"),
        EXPECTED_REPORTING_FREQUENCY,
        "run reporting frequency",
    )
    _equal(config.get("policy_mode"), EXPECTED_POLICY_MODE[variant], "run policy mode")
    _equal(config.get("min_support"), 1, "run minimum support")
    _equal(config.get("checkpoint_every"), 100, "run checkpoint interval")
    _equal(config.get("population_size"), 100, "run population size")
    _equal(config.get("warmup"), 100, "run warmup")
    _equal(config.get("parent_control"), False, "run parent control")
    _equal(config.get("legacy_seed_count"), 1, "run legacy seed count")
    _equal(config.get("durable_events"), True, "run durable event setting")
    strength = _number(config.get("prior_strength"), "run prior strength")
    _close(strength, EXPECTED_PRIOR_STRENGTH[variant], "run prior strength")
    if variant.startswith("shrink"):
        _close(_number(config.get("prior_mean"), "run prior mean"), 0.5, "run prior mean")
        source = config.get("prior_mean_source")
        if not isinstance(source, str) or not source.strip():
            raise ReportError("Bayesian run prior_mean_source must be nonempty")
    else:
        _equal(config.get("prior_mean"), None, "control prior mean")
        _equal(config.get("prior_mean_source"), None, "control prior mean source")
    model_path = config.get("model_path")
    if not isinstance(model_path, str) or Path(model_path).name != "50000.ckpt":
        raise ReportError("every run must use the final 50000.ckpt model")


def _validate_trajectory(
    summary: Mapping[str, Any],
    *,
    run_id: str,
    csv_values: Mapping[str, float],
    compact_metrics: Mapping[str, Any],
) -> tuple[tuple[int, float], ...]:
    _equal(summary.get("schema_version"), 2, f"{run_id} summary schema")
    _equal(summary.get("status"), "completed", f"{run_id} summary status")
    _equal(summary.get("run_id"), run_id, f"{run_id} summary identity")
    scores = _mapping(summary, "scores", f"{run_id} summary")
    all_scores = _mapping(scores, "all_charged_molecules", f"{run_id} summary.scores")
    _equal(all_scores.get("oracle_calls"), EXPECTED_BUDGET, f"{run_id} summary calls")
    _equal(all_scores.get("oracle_budget"), EXPECTED_BUDGET, f"{run_id} summary budget")
    _equal(
        all_scores.get("reporting_frequency"),
        EXPECTED_REPORTING_FREQUENCY,
        f"{run_id} summary reporting frequency",
    )
    raw_points = _sequence(
        all_scores, "trajectory_top_10", f"{run_id} all-charged metrics"
    )
    points: list[tuple[int, float]] = []
    for index, point in enumerate(raw_points):
        if not isinstance(point, Mapping):
            raise ReportError(f"{run_id} trajectory point {index} must be an object")
        call = _integer(
            point.get("oracle_calls"), f"{run_id} trajectory point {index} calls"
        )
        value = _number(
            point.get("top_k_mean"), f"{run_id} trajectory point {index} value"
        )
        if not 0.0 <= value <= 1.0:
            raise ReportError(f"{run_id} QED trajectory value is outside [0, 1]")
        points.append((call, value))
    expected_calls = tuple(range(0, EXPECTED_BUDGET + 1, EXPECTED_REPORTING_FREQUENCY))
    _equal(tuple(call for call, _ in points), expected_calls, f"{run_id} trajectory calls")
    _close(points[0][1], 0.0, f"{run_id} trajectory origin")
    for previous, current in zip(points, points[1:]):
        if current[1] + 1e-12 < previous[1]:
            raise ReportError(f"{run_id} top-10 trajectory is not monotone")

    stored_top_10 = _number(all_scores.get("top_10"), f"{run_id} summary top-10")
    _close(points[-1][1], stored_top_10, f"{run_id} trajectory endpoint")
    _close(stored_top_10, csv_values["all_top_10"], f"{run_id} CSV endpoint")
    _close(
        stored_top_10,
        _number(compact_metrics.get("top_10"), f"{run_id} collected top-10"),
        f"{run_id} collected endpoint",
    )
    as_dicts = [
        {"oracle_calls": call, "top_k_mean": value} for call, value in points
    ]
    recomputed_auc = trajectory_auc(as_dicts, normalize_by=EXPECTED_BUDGET)
    stored_auc = _number(all_scores.get("auc_top_10"), f"{run_id} summary top-10 AUC")
    _close(recomputed_auc, stored_auc, f"{run_id} trajectory AUC")
    _close(stored_auc, csv_values["all_auc_top_10"], f"{run_id} CSV AUC")
    _close(
        stored_auc,
        _number(compact_metrics.get("auc_top_10"), f"{run_id} collected AUC"),
        f"{run_id} collected AUC",
    )
    return tuple(points)


def _aggregate_runs(runs: Sequence[RunEvidence]) -> dict[str, Any]:
    by_variant = {
        variant: sorted(
            (run for run in runs if run.variant == variant), key=lambda run: run.seed
        )
        for variant in EXPECTED_VARIANTS
    }
    baseline = {run.seed: run for run in by_variant["running_mean"]}
    aggregates: dict[str, Any] = {}
    for variant, variant_runs in by_variant.items():
        _equal(
            [run.seed for run in variant_runs],
            list(EXPECTED_SEEDS),
            f"{variant} seed coverage",
        )
        metric_summary = {}
        for metric in METRICS:
            values = [run.values[metric] for run in variant_runs]
            metric_summary[metric] = {
                "mean": statistics.fmean(values),
                "sample_sd": statistics.stdev(values),
                "by_seed": {
                    str(run.seed): run.values[metric] for run in variant_runs
                },
            }
        paired_auc = [
            run.values["all_auc_top_10"]
            - baseline[run.seed].values["all_auc_top_10"]
            for run in variant_runs
        ]
        paired_final = [
            run.values["all_top_10"] - baseline[run.seed].values["all_top_10"]
            for run in variant_runs
        ]
        calls = [call for call, _ in variant_runs[0].trajectory]
        trajectory = []
        for position, call in enumerate(calls):
            values = [run.trajectory[position][1] for run in variant_runs]
            trajectory.append(
                {
                    "oracle_calls": call,
                    "mean": statistics.fmean(values),
                    "sample_sd": statistics.stdev(values),
                }
            )
        aggregates[variant] = {
            "prior_strength": EXPECTED_PRIOR_STRENGTH[variant],
            "run_count": len(variant_runs),
            "metrics": metric_summary,
            "paired_vs_running_mean": {
                "all_auc_top_10": {
                    "mean": statistics.fmean(paired_auc),
                    "sample_sd": statistics.stdev(paired_auc),
                    "by_seed": {
                        str(run.seed): value
                        for run, value in zip(variant_runs, paired_auc)
                    },
                },
                "all_top_10": {
                    "mean": statistics.fmean(paired_final),
                    "sample_sd": statistics.stdev(paired_final),
                    "by_seed": {
                        str(run.seed): value
                        for run, value in zip(variant_runs, paired_final)
                    },
                },
            },
            "trajectory_top_10": trajectory,
        }
    return aggregates


def load_report_data(collection_manifest: str | os.PathLike[str]) -> ReportData:
    """Load and fully validate the fixed 18-run report input."""

    collection_path = _safe_file(collection_manifest, "collection manifest")
    collection_hash = sha256_file(collection_path)
    collection = _read_json(collection_path, "collection manifest")
    csv_path, csv_hash, matrix_path, matrix_hash, experiment_root = (
        _validate_collection_header(collection, collection_path)
    )
    result_metadata = _mapping(collection, "results_csv", "collection")
    rows = _read_csv_rows(
        csv_path,
        csv_hash,
        _sequence(result_metadata, "columns", "collection.results_csv"),
    )
    _equal(len(rows), 18, "results CSV row count")
    csv_by_id: dict[str, dict[str, str]] = {}
    for index, row in enumerate(rows):
        run_id = row.get("run_id", "")
        if not run_id or run_id in csv_by_id:
            raise ReportError(f"results CSV has invalid or duplicate run_id at row {index + 2}")
        csv_by_id[run_id] = row

    raw_runs = _sequence(collection, "runs", "collection")
    _equal(len(raw_runs), 18, "collection runs length")
    experiment_id = collection.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ReportError("collection experiment_id must be nonempty")
    evidence: list[RunEvidence] = []
    common_config: dict[str, Any] | None = None
    observed_jobs: set[tuple[str, str, int]] = set()
    bayes_prior_sources: set[str] = set()
    model_hashes: set[str] = set()
    vocabulary_hashes: set[str] = set()
    code_identities: set[tuple[Any, ...]] = set()

    for ordinal, raw_run in enumerate(raw_runs):
        if not isinstance(raw_run, Mapping):
            raise ReportError(f"collection run {ordinal} must be an object")
        identity = _mapping(raw_run, "identity", f"collection run {ordinal}")
        variant = identity.get("variant")
        seed = identity.get("seed")
        if variant not in EXPECTED_VARIANTS:
            raise ReportError(f"collection run {ordinal} has unsupported variant {variant!r}")
        seed = _integer(seed, f"collection run {ordinal} seed")
        job = (str(identity.get("oracle")), str(variant), seed)
        if job in observed_jobs:
            raise ReportError(f"collection repeats job {job!r}")
        observed_jobs.add(job)
        run_id = identity.get("run_id")
        expected_run_id = f"{experiment_id}:{EXPECTED_ORACLE}:{variant}:seed{seed}"
        _equal(run_id, expected_run_id, f"collection run {ordinal} identity")
        _equal(identity.get("experiment_id"), experiment_id, f"{run_id} experiment")
        _equal(identity.get("oracle"), EXPECTED_ORACLE, f"{run_id} oracle")
        _equal(raw_run.get("status"), "completed", f"{run_id} status")
        _equal(raw_run.get("summary_schema_version"), 2, f"{run_id} summary schema")
        _equal(raw_run.get("oracle_budget"), EXPECTED_BUDGET, f"{run_id} budget")

        config = dict(_mapping(raw_run, "config", run_id))
        _validate_config(
            config,
            variant=str(variant),
            seed=seed,
            experiment_id=experiment_id,
        )
        selected_common = {key: config.get(key) for key in SHARED_CONFIG_FIELDS}
        if common_config is None:
            common_config = selected_common
        else:
            _equal(selected_common, common_config, f"{run_id} shared configuration")
        if str(variant).startswith("shrink"):
            bayes_prior_sources.add(str(config["prior_mean_source"]))

        row = csv_by_id.pop(str(run_id), None)
        if row is None:
            raise ReportError(f"collection run {run_id} has no CSV row")
        for actual, expected, context in (
            (row.get("experiment_id"), experiment_id, "CSV experiment_id"),
            (row.get("oracle"), EXPECTED_ORACLE, "CSV oracle"),
            (row.get("variant"), variant, "CSV variant"),
            (_csv_integer(row.get("seed"), "CSV seed"), seed, "CSV seed"),
            (row.get("status"), "completed", "CSV status"),
            (
                _csv_integer(row.get("summary_schema_version"), "CSV summary schema"),
                2,
                "CSV summary schema",
            ),
            (
                _csv_integer(row.get("oracle_budget"), "CSV oracle budget"),
                EXPECTED_BUDGET,
                "CSV oracle budget",
            ),
            (
                _csv_integer(row.get("all_oracle_calls"), "CSV all oracle calls"),
                EXPECTED_BUDGET,
                "CSV all oracle calls",
            ),
            (
                _csv_integer(row.get("charged_child_count"), "CSV child count"),
                EXPECTED_BUDGET,
                "CSV charged child count",
            ),
            (
                _csv_integer(row.get("reporting_frequency"), "CSV reporting frequency"),
                EXPECTED_REPORTING_FREQUENCY,
                "CSV reporting frequency",
            ),
            (row.get("model_path"), config.get("model_path"), "CSV model path"),
            (row.get("config_sha256"), raw_run.get("provenance", {}).get("config_sha256"), "CSV config hash"),
        ):
            _equal(actual, expected, f"{run_id} {context}")
        _close(
            _csv_number(row.get("elapsed_seconds"), f"{run_id} CSV elapsed seconds"),
            _number(raw_run.get("elapsed_seconds"), f"{run_id} collected elapsed seconds"),
            f"{run_id} elapsed seconds",
        )
        _close(
            _csv_number(row.get("prior_strength"), f"{run_id} CSV prior strength"),
            EXPECTED_PRIOR_STRENGTH[str(variant)],
            f"{run_id} CSV prior strength",
        )
        if str(variant).startswith("shrink"):
            _close(
                _csv_number(row.get("prior_mean"), f"{run_id} CSV prior mean"),
                EXPECTED_PRIOR_MEAN,
                f"{run_id} CSV prior mean",
            )
            _equal(
                row.get("prior_mean_source"),
                config.get("prior_mean_source"),
                f"{run_id} CSV prior source",
            )
        else:
            _equal(row.get("prior_mean"), "", f"{run_id} control CSV prior mean")
            _equal(
                row.get("prior_mean_source"),
                "",
                f"{run_id} control CSV prior source",
            )

        values = {
            metric: _csv_number(row.get(metric), f"{run_id} CSV {metric}")
            for metric in METRICS
        }
        if any(not 0.0 <= value <= 1.0 for value in values.values()):
            raise ReportError(f"{run_id} contains a QED metric outside [0, 1]")
        if not values["all_top_1"] >= values["all_top_10"] >= values["all_top_100"]:
            raise ReportError(f"{run_id} final top-k metrics are inconsistently ordered")

        metric_groups = _mapping(raw_run, "metrics", run_id)
        compact = _mapping(metric_groups, "all_charged_molecules", f"{run_id}.metrics")
        for csv_name, compact_name in (
            ("all_auc_top_1", "auc_top_1"),
            ("all_auc_top_10", "auc_top_10"),
            ("all_auc_top_100", "auc_top_100"),
            ("all_top_1", "top_1"),
            ("all_top_10", "top_10"),
            ("all_top_100", "top_100"),
        ):
            _close(
                values[csv_name],
                _number(compact.get(compact_name), f"{run_id} collected {compact_name}"),
                f"{run_id} CSV/collection {csv_name}",
            )

        sources = _mapping(raw_run, "sources", run_id)
        summary_source = _mapping(sources, "summary", f"{run_id}.sources")
        summary_path = _resolve_summary_path(
            summary_source.get("path"),
            experiment_root=experiment_root,
            context=f"{run_id} summary source",
        )
        summary_hash = _sha256_digest(
            summary_source.get("sha256"), f"{run_id} summary SHA-256"
        )
        _equal(
            row.get("source_summary_sha256"),
            summary_hash,
            f"{run_id} CSV summary SHA-256",
        )
        _equal(sha256_file(summary_path), summary_hash, f"{run_id} summary SHA-256")
        summary = _read_json(summary_path, f"{run_id} summary")
        trajectory = _validate_trajectory(
            summary,
            run_id=str(run_id),
            csv_values=values,
            compact_metrics=compact,
        )
        _equal(sha256_file(summary_path), summary_hash, f"{run_id} summary hash after reading")

        provenance = dict(_mapping(raw_run, "provenance", run_id))
        _equal(
            provenance.get("durable_events_provenance_complete"),
            True,
            f"{run_id} durable event provenance",
        )
        _equal(
            provenance.get("timing_provenance_complete"),
            True,
            f"{run_id} timing provenance",
        )
        launch = dict(_mapping(raw_run, "launch", run_id))
        for key in (
            "history_complete",
            "policy_complete",
            "matrix_provenance_complete",
            "provenance_complete",
            "durable_events_config_complete",
        ):
            _equal(launch.get(key), True, f"{run_id} launch {key}")
        model_hash = _sha256_digest(row.get("model_sha256"), f"{run_id} model hash")
        _equal(model_hash, EXPECTED_MODEL_SHA256, f"{run_id} 50k model hash")
        vocabulary_hash = _sha256_digest(
            row.get("vocabulary_sha256"), f"{run_id} vocabulary hash"
        )
        _equal(provenance.get("model_sha256"), model_hash, f"{run_id} model hash")
        _equal(
            provenance.get("vocabulary_sha256"),
            vocabulary_hash,
            f"{run_id} vocabulary hash",
        )
        _equal(
            row.get("vocabulary_path"),
            provenance.get("vocabulary_path"),
            f"{run_id} vocabulary path",
        )
        model_hashes.add(model_hash)
        vocabulary_hashes.add(vocabulary_hash)
        code_identities.add(
            (
                provenance.get("git_commit"),
                provenance.get("git_dirty"),
                provenance.get("tracked_diff_sha256"),
            )
        )
        evidence.append(
            RunEvidence(
                run_id=str(run_id),
                variant=str(variant),
                seed=seed,
                config=config,
                row=row,
                values=values,
                trajectory=trajectory,
                summary_path=summary_path,
                summary_sha256=summary_hash,
                provenance=provenance,
                launch=launch,
            )
        )

    _equal(observed_jobs, _expected_job_set(), "collected run job set")
    _equal(csv_by_id, {}, "unreferenced CSV rows")
    _equal(len(bayes_prior_sources), 1, "Bayesian prior source count")
    _equal(len(model_hashes), 1, "model hash count")
    _equal(len(vocabulary_hashes), 1, "vocabulary hash count")
    _equal(len(code_identities), 1, "run code identity count")
    if common_config is None:
        raise ReportError("collection contains no runs")
    evidence.sort(key=lambda run: (EXPECTED_VARIANTS.index(run.variant), run.seed))
    aggregates = _aggregate_runs(evidence)
    sharing = any(run.launch.get("any_gpu_sharing") is True for run in evidence)
    dirty = any(run.provenance.get("git_dirty") is True for run in evidence)
    caveats = [
        "This is an exploratory QED-only ablation (one of 23 PMO tasks), not a paper-scale PMO reproduction.",
        "Each arm has only three paired seeds, and each variant-seed run has 1,000 unique oracle calls; no p-values or confirmatory efficacy claims are reported.",
        "The Bayesian arms use a fixed neutral proxy prior mean of 0.5 because the original ZINC molecule-level prior is unavailable.",
        "Legacy seed fragments use the declared count-one approximation; this is not exact continuation of paper Equation 5 statistics.",
        "The locally trained 50,000-step checkpoint differs from the paper's complete hardware/training protocol.",
        "Whole-molecule scores associated with fragments are adaptive and non-causal; this report measures optimization outcomes, not fragment causality.",
    ]
    if sharing:
        caveats.append(
            "At least one launch shared a GPU; score metrics remain usable, but wall-time comparisons are suppressed."
        )
    if dirty:
        caveats.append(
            "Runs recorded a dirty Git worktree; the shared commit and tracked-diff hash are shown in provenance."
        )
    _equal(sha256_file(collection_path), collection_hash, "collection hash after reading")
    _equal(sha256_file(csv_path), csv_hash, "results CSV hash after reading")
    _equal(sha256_file(matrix_path), matrix_hash, "matrix hash after reading")
    return ReportData(
        collection_path=collection_path,
        collection_sha256=collection_hash,
        collection=dict(collection),
        csv_path=csv_path,
        csv_sha256=csv_hash,
        matrix_path=matrix_path,
        matrix_sha256=matrix_hash,
        runs=tuple(evidence),
        aggregates=aggregates,
        common_config=common_config,
        caveats=tuple(caveats),
    )


def _mean_sd_text(summary: Mapping[str, Any], digits: int = 6) -> str:
    return f"{summary['mean']:.{digits}f} +/- {summary['sample_sd']:.{digits}f}"


def _lambda_text(variant: str) -> str:
    if variant == "released":
        return "n/a"
    return f"{EXPECTED_PRIOR_STRENGTH[variant]:g}"


def _execution_provenance_rows(data: ReportData) -> tuple[tuple[str, str], ...]:
    elapsed: list[float] = []
    started: list[float] = []
    finished: list[float] = []
    utilizations: list[int] = []
    free_memory: list[int] = []
    shared_launches = 0
    resumes = 0
    gpu_uuids: dict[int, str] = {}
    for run in data.runs:
        run_elapsed = _csv_number(
            run.row.get("elapsed_seconds"), f"{run.run_id} elapsed seconds"
        )
        elapsed.append(run_elapsed)
        attempts = _sequence(run.launch, "attempts", f"{run.run_id} launch")
        _equal(len(attempts), 1, f"{run.run_id} launch attempt count")
        attempt = attempts[0]
        if not isinstance(attempt, Mapping):
            raise ReportError(f"{run.run_id} launch attempt must be an object")
        record = _mapping(attempt, "record", f"{run.run_id} launch attempt")
        gpu = _mapping(record, "physical_gpu", f"{run.run_id} launch record")
        gpu_index = _integer(gpu.get("index"), f"{run.run_id} physical GPU")
        gpu_uuid = gpu.get("uuid")
        if not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-"):
            raise ReportError(f"{run.run_id} physical GPU UUID is invalid")
        if gpu_index in gpu_uuids:
            _equal(gpu_uuid, gpu_uuids[gpu_index], f"physical GPU {gpu_index} UUID")
        gpu_uuids[gpu_index] = gpu_uuid
        _equal(
            str(run.provenance.get("cuda_visible_devices")),
            str(gpu_index),
            f"{run.run_id} physical GPU mapping",
        )
        utilization = _integer(
            gpu.get("utilization_percent"), f"{run.run_id} launch utilization"
        )
        threshold = _integer(
            record.get("utilization_threshold"), f"{run.run_id} utilization threshold"
        )
        if utilization >= threshold:
            raise ReportError(f"{run.run_id} was launched above its utilization gate")
        utilizations.append(utilization)
        total_memory = _integer(
            gpu.get("memory_total_mib"), f"{run.run_id} total GPU memory"
        )
        used_memory = _integer(
            gpu.get("memory_used_mib"), f"{run.run_id} used GPU memory"
        )
        minimum_free = _integer(
            record.get("min_free_memory_mib"), f"{run.run_id} free-memory gate"
        )
        available = total_memory - used_memory
        if available < minimum_free:
            raise ReportError(f"{run.run_id} was launched below its free-memory gate")
        free_memory.append(available)
        shared_launches += record.get("sharing_actual") is True
        launch_time = _number(record.get("time_unix"), f"{run.run_id} launch time")
        started.append(launch_time)
        finished.append(launch_time + run_elapsed)
        resumes += _integer(
            run.provenance.get("resume_count"), f"{run.run_id} resume count"
        )

    gpu_text = "; ".join(
        f"{index}={gpu_uuids[index]}" for index in sorted(gpu_uuids)
    )
    controller_span = max(finished) - min(started)
    return (
        ("Physical GPU mapping", gpu_text),
        (
            "Pre-launch gate",
            f"18/18 launches at {min(utilizations)}-{max(utilizations)}% utilization "
            f"(<10%) with at least {min(free_memory)} MiB free; "
            f"{shared_launches}/18 shared with pre-existing low-utilization processes",
        ),
        (
            "Runtime record",
            f"controller span {controller_span:.1f} s; sum of per-run elapsed times "
            f"{sum(elapsed):.1f} s; median {statistics.median(elapsed):.1f} s "
            f"(range {min(elapsed):.1f}-{max(elapsed):.1f}); resumes={resumes}. "
            "Wall-time comparisons are suppressed because GPUs were shared.",
        ),
    )


def _chart_image(payload: bytes, *, width: float, height: float) -> Image:
    stream = io.BytesIO(payload)
    image = Image(stream, width=width, height=height)
    image._genmol_stream = stream  # type: ignore[attr-defined]
    return image


def _save_figure(figure: Any) -> bytes:
    stream = io.BytesIO()
    figure.savefig(
        stream,
        format="png",
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "GenMol deterministic Bayesian report"},
    )
    plt.close(figure)
    return stream.getvalue()


def _learning_curve_chart(data: ReportData) -> bytes:
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    ):
        figure, axis = plt.subplots(figsize=(10.2, 5.0), constrained_layout=True)
        for variant in EXPECTED_VARIANTS:
            runs = [run for run in data.runs if run.variant == variant]
            calls = [point[0] for point in runs[0].trajectory]
            for run in runs:
                axis.plot(
                    calls,
                    [point[1] for point in run.trajectory],
                    color=COLORS[variant],
                    alpha=0.20,
                    linewidth=0.8,
                )
            aggregate = data.aggregates[variant]["trajectory_top_10"]
            means = [point["mean"] for point in aggregate]
            deviations = [point["sample_sd"] for point in aggregate]
            axis.fill_between(
                calls,
                [max(0.0, mean - deviation) for mean, deviation in zip(means, deviations)],
                [min(1.0, mean + deviation) for mean, deviation in zip(means, deviations)],
                color=COLORS[variant],
                alpha=0.09,
                linewidth=0,
            )
            axis.plot(
                calls,
                means,
                color=COLORS[variant],
                linewidth=2.0,
                label=variant,
            )
        axis.set_xlim(0, EXPECTED_BUDGET)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("Unique canonical-molecule oracle calls")
        axis.set_ylabel("Mean of top 10 QED scores")
        axis.set_title("Per-seed traces (thin) and mean +/- 1 sample SD (band)")
        axis.legend(ncol=3, frameon=False, loc="lower right")
        return _save_figure(figure)


def _dose_response_chart(data: ReportData) -> bytes:
    labels = ("0\nrunning mean", "1", "3", "10", "30")
    variants = ("running_mean", "shrink1", "shrink3", "shrink10", "shrink30")
    positions = list(range(len(labels)))
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    ):
        figure, (dose_axis, delta_axis) = plt.subplots(
            1, 2, figsize=(10.2, 4.4), constrained_layout=True
        )
        by_variant_seed = {
            (run.variant, run.seed): run.values["all_auc_top_10"]
            for run in data.runs
        }
        for seed in EXPECTED_SEEDS:
            dose_axis.plot(
                positions,
                [by_variant_seed[(variant, seed)] for variant in variants],
                color="#777777",
                alpha=0.45,
                linewidth=1.0,
                marker="o",
                markersize=3,
            )
        means = [
            data.aggregates[variant]["metrics"]["all_auc_top_10"]["mean"]
            for variant in variants
        ]
        deviations = [
            data.aggregates[variant]["metrics"]["all_auc_top_10"]["sample_sd"]
            for variant in variants
        ]
        dose_axis.errorbar(
            positions,
            means,
            yerr=deviations,
            color="#0072B2",
            linewidth=2.0,
            marker="o",
            capsize=3,
            label="mean +/- sample SD",
        )
        released_mean = data.aggregates["released"]["metrics"]["all_auc_top_10"][
            "mean"
        ]
        dose_axis.axhline(
            released_mean,
            color=COLORS["released"],
            linestyle="--",
            linewidth=1.2,
            label="released mean",
        )
        dose_axis.set_xticks(positions, labels)
        dose_axis.set_xlabel("Bayesian prior strength lambda")
        dose_axis.set_ylabel("Top-10 AUC")
        dose_axis.set_title("Observed dose response")
        dose_axis.legend(frameon=False, fontsize=8)

        shrink_variants = variants[1:]
        delta_positions = list(range(len(shrink_variants)))
        for position, variant in zip(delta_positions, shrink_variants):
            paired = data.aggregates[variant]["paired_vs_running_mean"][
                "all_auc_top_10"
            ]
            seed_values = [paired["by_seed"][str(seed)] for seed in EXPECTED_SEEDS]
            delta_axis.scatter(
                [position - 0.10, position, position + 0.10],
                seed_values,
                color=COLORS[variant],
                alpha=0.70,
                s=20,
            )
            delta_axis.errorbar(
                [position],
                [paired["mean"]],
                yerr=[paired["sample_sd"]],
                color=COLORS[variant],
                marker="D",
                markersize=5,
                capsize=3,
                linewidth=1.5,
            )
        delta_axis.axhline(0.0, color="#333333", linewidth=0.9)
        delta_axis.set_xticks(delta_positions, ("1", "3", "10", "30"))
        delta_axis.set_xlabel("Bayesian prior strength lambda")
        delta_axis.set_ylabel("Paired AUC delta vs running mean")
        delta_axis.set_title("Three paired seeds; diamonds are means")
        return _save_figure(figure)


def _table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    widths: Sequence[float],
    *,
    font_size: float = 7.2,
) -> Table:
    styles = getSampleStyleSheet()
    cell_style = ParagraphStyle(
        "Cell",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=font_size,
        leading=font_size + 1.7,
        textColor=colors.HexColor("#263238"),
    )
    header_style = ParagraphStyle(
        "HeaderCell",
        parent=cell_style,
        fontName="Helvetica-Bold",
        textColor=colors.white,
    )

    def paragraph(value: Any, style: ParagraphStyle = cell_style) -> Paragraph:
        return Paragraph(escape(str(value)).replace("\n", "<br/>"), style)

    data = [[paragraph(header, header_style) for header in headers]]
    data.extend([[paragraph(value) for value in row] for row in rows])
    table = Table(data, colWidths=list(widths), repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#B0BEC5")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7F9")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def render_report_pdf(data: ReportData) -> bytes:
    """Render deterministic PDF bytes from validated report data."""

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#17365D"),
        alignment=TA_CENTER,
        spaceAfter=8 * mm,
    )
    heading = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=19,
        textColor=colors.HexColor("#1F4E79"),
        spaceAfter=4 * mm,
    )
    subheading = ParagraphStyle(
        "ReportSubheading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#365F91"),
        spaceAfter=2 * mm,
    )
    body = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#263238"),
        spaceAfter=2.5 * mm,
    )
    note = ParagraphStyle(
        "ReportNote",
        parent=body,
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#455A64"),
    )
    compact_note = ParagraphStyle(
        "ReportCompactNote",
        parent=note,
        fontSize=7.5,
        leading=9.2,
        spaceAfter=0.8 * mm,
    )

    def p(text: str, style: ParagraphStyle = body) -> Paragraph:
        return Paragraph(escape(text), style)

    def decorate(canvas: Any, document: Any) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#B0BEC5"))
        canvas.line(15 * mm, 13 * mm, 282 * mm, 13 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#546E7A"))
        canvas.drawString(15 * mm, 8 * mm, "GenMol v2 | exploratory QED ablation")
        canvas.drawRightString(282 * mm, 8 * mm, f"Page {document.page}")
        canvas.restoreState()

    stream = io.BytesIO()
    document = SimpleDocTemplate(
        stream,
        pagesize=landscape(A4),
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=14 * mm,
        bottomMargin=17 * mm,
        title="GenMol QED Bayesian Shrinkage Ablation",
        author="GenMol v2 reproducible experiment harness",
    )
    story: list[Any] = []
    story.append(p("GenMol QED Bayesian Shrinkage Ablation", title))
    story.append(
        p(
            "Validated 50,000-step-checkpoint comparison of the released update, "
            "online running mean, and four fixed Bayesian shrinkage strengths."
        )
    )
    story.append(
        _table(
            ("Protocol field", "Locked value"),
            (
                ("Oracle and budget", "QED; 1,000 unique canonical molecules per variant-seed run"),
                ("Arms", ", ".join(EXPECTED_VARIANTS)),
                ("Paired seeds", "0, 1, 2 (18 completed runs)"),
                ("Bayesian prior", "fixed proxy mean 0.5; lambda in {1, 3, 10, 30}"),
                (
                    "Population and warmup",
                    f"population={data.common_config['population_size']}; "
                    f"warmup={data.common_config['warmup']}; legacy off-by-one="
                    f"{data.common_config['legacy_warmup_off_by_one']}",
                ),
                (
                    "Sampling controls",
                    f"temperature={data.common_config['softmax_temp']}; "
                    f"randomness={data.common_config['randomness']}; "
                    f"guidance={data.common_config['guidance_scale']}; "
                    f"gamma={data.common_config['gamma']}",
                ),
                (
                    "Molecule bounds",
                    f"{data.common_config['min_mol_size']} to "
                    f"{data.common_config['max_mol_size']} under the runner convention",
                ),
                (
                    "Loop and persistence",
                    f"max iterations={data.common_config['max_iterations']}; "
                    f"report/checkpoint every {data.common_config['reporting_frequency']}/"
                    f"{data.common_config['checkpoint_every']} calls; durable events="
                    f"{data.common_config['durable_events']}",
                ),
                (
                    "Fragment statistics",
                    f"minimum support={data.common_config['min_support']}; legacy seed count="
                    f"{data.common_config['legacy_seed_count']}; parent control="
                    f"{data.common_config['parent_control']}",
                ),
                (
                    "Sampling and duplicates",
                    f"{data.common_config['population_sampling_order']}; statistical: "
                    f"{data.common_config['statistical_duplicate_policy']}; released: "
                    f"{data.common_config['released_duplicate_policy']}",
                ),
                ("Primary endpoint", "normalized trapezoidal AUC of mean top-10 QED vs oracle calls"),
                ("Scientific status", str(data.common_config["scientific_status"])),
            ),
            (48 * mm, 215 * mm),
            font_size=8.2,
        )
    )
    story.append(PageBreak())

    story.append(p("Aggregate scalar metrics", heading))
    aggregate_auc_rows = []
    aggregate_final_rows = []
    for variant in EXPECTED_VARIANTS:
        aggregate = data.aggregates[variant]
        aggregate_auc_rows.append(
            (
                variant,
                _lambda_text(variant),
                _mean_sd_text(aggregate["metrics"]["all_auc_top_1"]),
                _mean_sd_text(aggregate["metrics"]["all_auc_top_10"]),
                _mean_sd_text(aggregate["metrics"]["all_auc_top_100"]),
                _mean_sd_text(
                    aggregate["paired_vs_running_mean"]["all_auc_top_10"]
                ),
            )
        )
        aggregate_final_rows.append(
            (
                variant,
                _lambda_text(variant),
                _mean_sd_text(aggregate["metrics"]["all_top_1"]),
                _mean_sd_text(aggregate["metrics"]["all_top_10"]),
                _mean_sd_text(aggregate["metrics"]["all_top_100"]),
                _mean_sd_text(
                    aggregate["paired_vs_running_mean"]["all_top_10"]
                ),
            )
        )
    story.append(p("Normalized AUC metrics", subheading))
    story.append(
        _table(
            ("Variant", "lambda", "Top-1 AUC", "Top-10 AUC", "Top-100 AUC", "Paired top-10 AUC delta vs mean"),
            aggregate_auc_rows,
            (34 * mm, 16 * mm, 48 * mm, 48 * mm, 48 * mm, 73 * mm),
            font_size=7.1,
        )
    )
    story.append(Spacer(1, 5 * mm))
    story.append(p("Final-budget top-k metrics", subheading))
    story.append(
        _table(
            ("Variant", "lambda", "Final top-1", "Final top-10", "Final top-100", "Paired final top-10 delta vs mean"),
            aggregate_final_rows,
            (34 * mm, 16 * mm, 48 * mm, 48 * mm, 48 * mm, 73 * mm),
            font_size=7.1,
        )
    )
    story.append(Spacer(1, 4 * mm))
    story.append(
        p(
            "Values are mean +/- sample SD across three paired seeds. The sample size "
            "is intentionally too small for a confirmatory significance claim.",
            note,
        )
    )
    story.append(PageBreak())

    story.append(p("Top-10 learning curves", heading))
    story.append(
        _chart_image(
            _learning_curve_chart(data), width=250 * mm, height=123 * mm
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        p(
            "The origin is fixed at zero and AUC is normalized by the common "
            "1,000-call budget, matching the experiment-I/O convention. All arms "
            "spend the same unique-molecule budget; no parent-scoring arms are included.",
            note,
        )
    )
    story.append(PageBreak())

    story.append(p("Bayesian shrinkage dose response", heading))
    story.append(
        _chart_image(_dose_response_chart(data), width=250 * mm, height=108 * mm)
    )
    story.append(Spacer(1, 3 * mm))
    paired_rows = []
    for variant in EXPECTED_VARIANTS:
        paired = data.aggregates[variant]["paired_vs_running_mean"]
        paired_rows.append(
            (
                variant,
                _lambda_text(variant),
                _mean_sd_text(paired["all_auc_top_10"]),
                _mean_sd_text(paired["all_top_10"]),
                ", ".join(
                    f"s{seed}={paired['all_auc_top_10']['by_seed'][str(seed)]:+.6f}"
                    for seed in EXPECTED_SEEDS
                ),
            )
        )
    story.append(
        _table(
            ("Variant", "lambda", "Paired AUC delta", "Paired final top-10 delta", "AUC deltas by seed"),
            paired_rows,
            (37 * mm, 18 * mm, 53 * mm, 60 * mm, 99 * mm),
            font_size=7.1,
        )
    )
    story.append(PageBreak())

    story.append(p("Per-seed results", heading))
    per_seed_rows = []
    for run in data.runs:
        per_seed_rows.append(
            (
                run.variant,
                run.seed,
                _lambda_text(run.variant),
                f"{run.values['all_auc_top_10']:.6f}",
                f"{run.values['all_top_1']:.6f}",
                f"{run.values['all_top_10']:.6f}",
                f"{run.values['all_top_100']:.6f}",
                str(run.launch.get("any_gpu_sharing")),
                run.run_id,
            )
        )
    story.append(
        _table(
            ("Variant", "Seed", "lambda", "Top-10 AUC", "Top-1", "Top-10", "Top-100", "GPU shared", "Run ID"),
            per_seed_rows,
            (32 * mm, 14 * mm, 15 * mm, 28 * mm, 25 * mm, 25 * mm, 25 * mm, 22 * mm, 81 * mm),
            font_size=6.5,
        )
    )
    story.append(Spacer(1, 4 * mm))
    story.append(
        p(
            "Every scalar above was reconciled across the immutable CSV, collection "
            "manifest, and the collector-validated summary. Individual seeds remain "
            "visible so stability is not hidden by an average.",
            note,
        )
    )
    story.append(PageBreak())

    story.append(p("Integrity and limitations", heading))
    first = data.runs[0]
    provenance_rows = (
        ("Collection manifest", f"{data.collection_path} | sha256 {data.collection_sha256}"),
        ("Immutable results CSV", f"{data.csv_path.name} | sha256 {data.csv_sha256}"),
        ("Matrix", f"{data.matrix_path} | sha256 {data.matrix_sha256}"),
        ("Model", f"{first.config['model_path']} | sha256 {first.provenance['model_sha256']}"),
        ("Vocabulary", f"{first.provenance['vocabulary_path']} | sha256 {first.provenance['vocabulary_sha256']}"),
        ("Code identity", f"commit {first.provenance.get('git_commit')} | dirty={first.provenance.get('git_dirty')} | tracked diff {first.provenance.get('tracked_diff_sha256')}"),
        ("Trajectory evidence", "18 summary files re-hashed; endpoints and normalized trapezoidal AUCs recomputed"),
        *_execution_provenance_rows(data),
    )
    story.append(
        _table(
            ("Provenance field", "Validated value"),
            provenance_rows,
            (48 * mm, 215 * mm),
            font_size=6.8,
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(p("Paper reference point", subheading))
    story.append(
        p(
            "GenMol Table 13 reports QED PMO AUC top-10 of 0.942 +/- "
            "0.000 across three runs at a 10,000-oracle-call budget. That value "
            "is included only as context: this report uses 1,000 calls, a local "
            "50,000-step checkpoint, and a different fragment-vocabulary update "
            "study, so a numeric gap is not an efficacy comparison.",
            note,
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(p("Scientific caveats", subheading))
    for caveat in data.caveats:
        story.append(p(f"- {caveat}", compact_note))
    story.append(Spacer(1, 1.5 * mm))
    story.append(
        p(
            "Interpretation boundary: observed differences describe this locked "
            "exploratory matrix only. They neither reproduce the paper's 23-task, "
            "10,000-call PMO aggregate nor establish that Bayesian shrinkage is "
            "generally superior.",
            compact_note,
        )
    )

    document.build(
        story,
        onFirstPage=decorate,
        onLaterPages=decorate,
        canvasmaker=_InvariantCanvas,
    )
    return stream.getvalue()


def validate_pdf(payload: bytes) -> dict[str, Any]:
    """Check basic PDF structure and required textual sections."""

    try:
        reader = PdfReader(io.BytesIO(payload))
    except Exception as error:
        raise ReportError(f"generated report is not a readable PDF: {error}") from error
    if reader.is_encrypted:
        raise ReportError("generated report must not be encrypted")
    if len(reader.pages) < 5:
        raise ReportError("generated report must contain at least five pages")
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
    missing = [heading for heading in PDF_HEADINGS if heading not in extracted]
    if missing:
        raise ReportError(f"generated report is missing headings: {missing}")
    return {
        "page_count": len(reader.pages),
        "file_size_bytes": len(payload),
        "required_headings_present": True,
        "encrypted": False,
    }


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _publish_immutable_pdf(destination: Path, payload: bytes) -> tuple[Path, str, bool]:
    digest = hashlib.sha256(payload).hexdigest()
    path = destination / f"bayesian_ablation.{digest}.pdf"
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise ReportError(f"hash-named PDF has unexpected content: {path}")
        return path, digest, False
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination, prefix=".bayesian_ablation.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        created = True
        try:
            os.link(temporary, path)
        except FileExistsError:
            created = False
            if path.is_symlink() or path.read_bytes() != payload:
                raise ReportError(f"hash-named PDF has unexpected content: {path}")
        _fsync_directory(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return path, digest, created


def _atomic_json(path: Path, value: Mapping[str, Any], *, overwrite: bool) -> Path:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    payload = (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _utc_timestamp() -> str:
    return (
        datetime_module.datetime.now(datetime_module.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _recheck_report_sources(data: ReportData) -> None:
    _equal(
        sha256_file(data.collection_path),
        data.collection_sha256,
        "collection hash before publication",
    )
    _equal(sha256_file(data.csv_path), data.csv_sha256, "CSV hash before publication")
    _equal(
        sha256_file(data.matrix_path),
        data.matrix_sha256,
        "matrix hash before publication",
    )
    for run in data.runs:
        _equal(
            sha256_file(run.summary_path),
            run.summary_sha256,
            f"{run.run_id} summary hash before publication",
        )


def write_report(
    collection_manifest: str | os.PathLike[str],
    output_dir: str | os.PathLike[str] | None = None,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Validate inputs and atomically publish an immutable PDF and manifest pointer."""

    data = load_report_data(collection_manifest)
    destination_candidate = (
        data.collection_path.parent if output_dir is None else _lexical_absolute(output_dir)
    )
    for component in (destination_candidate, *destination_candidate.parents):
        if component.is_symlink():
            raise ReportError(f"report destination must not contain symlinks: {component}")
    destination_candidate.mkdir(parents=True, exist_ok=True)
    destination = _safe_directory(destination_candidate, "report destination")
    pointer = destination / "bayesian_ablation_report.json"
    lock_path = destination / ".bayesian_ablation_report.lock"
    if pointer.is_symlink() or lock_path.is_symlink():
        raise ReportError("report pointer and lock must not be symlinks")
    try:
        lock = _DestinationLock(lock_path)
    except BlockingIOError as error:
        raise ReportError(f"another report writer owns {destination}") from error
    with lock:
        if pointer.exists() and not overwrite:
            raise FileExistsError(pointer)
        payload = render_report_pdf(data)
        audit = validate_pdf(payload)
        _recheck_report_sources(data)
        pdf_path, pdf_hash, pdf_created = _publish_immutable_pdf(destination, payload)
        reporter_path = Path(__file__).resolve()
        report_manifest = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "created_at": _utc_timestamp(),
            "report_type": "fixed QED Bayesian shrinkage ablation",
            "scientific_status": data.common_config["scientific_status"],
            "design": {
                "oracle": EXPECTED_ORACLE,
                "unique_oracle_calls": EXPECTED_BUDGET,
                "reporting_frequency": EXPECTED_REPORTING_FREQUENCY,
                "seeds": list(EXPECTED_SEEDS),
                "variants": list(EXPECTED_VARIANTS),
                "prior_mean": EXPECTED_PRIOR_MEAN,
                "prior_strengths": [1, 3, 10, 30],
                "run_count": len(data.runs),
            },
            "sources": {
                "collection_manifest": {
                    "path": str(data.collection_path),
                    "sha256": data.collection_sha256,
                },
                "results_csv": {
                    "path": str(data.csv_path),
                    "sha256": data.csv_sha256,
                },
                "matrix": {
                    "path": str(data.matrix_path),
                    "sha256": data.matrix_sha256,
                },
                "summaries": [
                    {
                        "run_id": run.run_id,
                        "path": str(run.summary_path),
                        "sha256": run.summary_sha256,
                    }
                    for run in data.runs
                ],
            },
            "reporter": {
                "path": str(reporter_path),
                "sha256": sha256_file(reporter_path),
                "python": platform.python_version(),
                "packages": {
                    "matplotlib": _package_version("matplotlib"),
                    "pypdf": _package_version("pypdf"),
                    "reportlab": _package_version("reportlab"),
                },
            },
            "pdf": {
                "path": pdf_path.name,
                "sha256": pdf_hash,
                "immutable": True,
                **audit,
            },
            "aggregates": data.aggregates,
            "common_config": data.common_config,
            "caveats": list(data.caveats),
        }
        try:
            _atomic_json(pointer, report_manifest, overwrite=overwrite)
        except BaseException:
            if pdf_created:
                pdf_path.unlink(missing_ok=True)
                _fsync_directory(destination)
            raise
        return pdf_path, pointer


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection_manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    pdf_path, manifest_path = write_report(
        args.collection_manifest,
        args.output_dir,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {"pdf": str(pdf_path), "report_manifest": str(manifest_path)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "EXPECTED_MODEL_SHA256",
    "EXPECTED_PRIOR_STRENGTH",
    "EXPECTED_SEEDS",
    "EXPECTED_VARIANTS",
    "REPORT_SCHEMA_VERSION",
    "ReportData",
    "ReportError",
    "RunEvidence",
    "load_report_data",
    "render_report_pdf",
    "validate_pdf",
    "write_report",
]

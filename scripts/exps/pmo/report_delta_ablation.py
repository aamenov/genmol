"""Build an auditable PDF for the fixed QED delta-y ablation.

The input is a completed schema-3 collection manifest.  The collector has
already replayed oracle accounting; this report independently checks the fixed
12-run design, source hashes, scalar metrics, launch provenance, and the
parent/child attribution evidence used by the two matched arms.  It never runs
the model or oracle.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import re
import statistics
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import yaml
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as reportlab_canvas
from reportlab.platypus import (
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

from scripts.exps.pmo.main.genmol.experiment_io import sha256_file, summarize_scores


REPORT_SCHEMA_VERSION = 2
COLLECTION_SCHEMA_VERSION = 3
REPORTER_PATH = Path(__file__).resolve()
EXPERIMENT_ID = "fragment_vocab_qed_50k_delta_1k_v2"
SCIENTIFIC_STATUS = (
    "Exploratory paired three-seed QED delta-y credit-assignment study using the "
    "final 50k checkpoint, a 1,000-unique-oracle-call budget, warmup parameter "
    "100 (101 frozen iterations under the legacy iteration > warmup boundary), "
    "and neutral zero seed estimates with absolute seed scores used only "
    "for matched tie-breaking. The delta and transition-absolute control arms "
    "share parent/child oracle accounting, domains, transition deduplication, "
    "deterministic attribution, and warmup; only the post-warmup credit target "
    "differs. This QED-only reduced-budget study is not paper-comparable, causal, "
    "or confirmatory."
)
EXPECTED_ORACLE = "qed"
EXPECTED_SEEDS = (0, 1, 2)
EXPECTED_VARIANTS = (
    "released",
    "running_mean",
    "running_mean_delta_control",
    "delta",
)
MATCHED_CONTROL = "running_mean_delta_control"
EXPECTED_MODEL_SHA256 = (
    "8d00aa47b02f64bf39ff6b0b2e786f213587366fc2c3d29712a00f3f84108dd6"
)
EXPECTED_VOCABULARY_SHA256 = (
    "6b98420f3fa88835e50cb28b960225dc6c7857a29f3057fb1ca22b4211c1806c"
)
EXPECTED_BUDGET = 1_000
EXPECTED_POLICY = {
    "released": ("released", False),
    "running_mean": ("mean", False),
    MATCHED_CONTROL: ("delta_control", True),
    "delta": ("delta", True),
}
EXPECTED_SEED_INITIALIZATION = {
    "released": "released_absolute_rows",
    "running_mean": "absolute_seed_statistics",
    MATCHED_CONTROL: "neutral_zero_with_seed_score_tiebreak",
    "delta": "neutral_zero_with_seed_score_tiebreak",
}
EXPECTED_CREDIT_VALUE_POLICY = {
    "released": "absolute_child_score",
    "running_mean": "absolute_child_score",
    MATCHED_CONTROL: "absolute_child_score",
    "delta": "child_score_minus_parent_score",
}
COMMON_CHILD_HORIZON = 500
COMMON_CHILD_VIEW = "first_500_charged_children"
VIEW_PREFIX = {
    "all_charged_molecules": "all",
    "charged_children_total_call_axis": "child_total",
}
REPORT_VIEWS = (*VIEW_PREFIX, COMMON_CHILD_VIEW)
SCALAR_NAMES = (
    "auc_top_1",
    "auc_top_10",
    "auc_top_100",
    "top_1",
    "top_10",
    "top_100",
)
TARGET_VARIANTS = frozenset({MATCHED_CONTROL, "delta"})
CONTEXTUAL_VARIANTS = frozenset({"released", "running_mean"})
VARIANT_LABELS = {
    "released": "released (contextual)",
    "running_mean": "running_mean (contextual)",
    MATCHED_CONTROL: "transition-absolute control",
    "delta": "delta-y",
}
METRIC_DEFINITIONS = {
    "top_k": {
        "k_values": [1, 10, 100],
        "definition": (
            "At a stated horizon, top-k is the arithmetic mean of the highest "
            "min(k, N) QED scores observed in that view, where N is the number "
            "of included scores available by that horizon."
        ),
    },
    "auc_top_k": {
        "definition": (
            "The top-k trajectory starts at (0, 0), is evaluated every 100 axis "
            "units through the stated horizon, is integrated by the trapezoidal "
            "rule, and is divided by that horizon."
        ),
    },
    "views": {
        "all_charged_molecules": (
            "All charged unique parent and child QED evaluations in global "
            "unique-oracle-call order; horizon and AUC divisor are 1,000 calls."
        ),
        "charged_children_total_call_axis": (
            "Only charged child QED scores, located at their original global "
            "unique-oracle-call indices; checkpoints, horizon, and AUC divisor "
            "remain 0, 100, ..., 1,000 calls. This is not an equal-child-evaluation view."
        ),
        COMMON_CHILD_VIEW: (
            "The first 500 charged child QED scores in event order, densely "
            "indexed as charged-child evaluations 1 through 500; checkpoints are "
            "0, 100, ..., 500 and the trapezoidal AUC divisor is 500 charged-child "
            "evaluations."
        ),
    },
}


class ReportError(ValueError):
    """Raised when evidence cannot support this fixed report."""


@dataclass(frozen=True)
class RunEvidence:
    run_id: str
    variant: str
    seed: int
    config: dict[str, Any]
    metrics: dict[str, dict[str, float]]
    child_count: int
    elapsed_seconds: float
    diagnostics: dict[str, Any] | None
    event_accounting: dict[str, int]
    provenance: dict[str, Any]
    launch: dict[str, Any]
    sources: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class ReportData:
    collection_path: Path
    collection_sha256: str
    csv_path: Path
    csv_sha256: str
    matrix_path: Path
    matrix_sha256: str
    experiment_root: Path
    collection: dict[str, Any]
    runs: tuple[RunEvidence, ...]
    aggregates: dict[str, Any]
    paired: dict[str, Any]
    caveats: tuple[str, ...]


class _DestinationLock:
    def __init__(self, path: Path):
        self._handle = path.open("a+b")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self._handle.close()
            raise

    def __enter__(self) -> "_DestinationLock":
        return self

    def __exit__(self, *_: Any) -> None:
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()


class _InvariantCanvas(reportlab_canvas.Canvas):
    def __init__(self, *args: Any, **kwargs: Any):
        kwargs["invariant"] = 1
        super().__init__(*args, **kwargs)
        self.setTitle("GenMol QED Delta-y Credit Ablation")
        self.setAuthor("GenMol v2 reproducible experiment harness")
        self.setSubject("Exploratory fixed-budget fragment-credit ablation")


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


def _safe_file(
    path: str | os.PathLike[str],
    label: str,
    *,
    contained_by: Path | None = None,
) -> Path:
    candidate = Path(path).expanduser()
    lexical = candidate.absolute()
    for component in (lexical, *lexical.parents):
        if component.is_symlink():
            raise ReportError(f"{label} must not traverse a symlink: {component}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file():
        raise ReportError(f"{label} is not a regular file: {resolved}")
    if contained_by is not None:
        root = contained_by.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ReportError(f"{label} escapes {root}: {resolved}") from error
    return resolved


def _safe_directory(path: str | os.PathLike[str], label: str) -> Path:
    candidate = Path(path).expanduser()
    lexical = candidate.absolute()
    for component in (lexical, *lexical.parents):
        if component.is_symlink():
            raise ReportError(f"{label} must not traverse a symlink: {component}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_dir():
        raise ReportError(f"{label} is not a directory: {resolved}")
    return resolved


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReportError(f"{context} must be an object")
    return value


def _sequence(value: Any, context: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ReportError(f"{context} must be an array")
    return value


def _integer(value: Any, context: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReportError(f"{context} must be an integer")
    if minimum is not None and value < minimum:
        raise ReportError(f"{context} must be at least {minimum}")
    return value


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise ReportError(f"{context} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ReportError(f"{context} must be numeric") from error
    if not math.isfinite(result):
        raise ReportError(f"{context} must be finite")
    return result


def _csv_number(value: str | None, context: str) -> float:
    if value in (None, ""):
        raise ReportError(f"{context} is missing")
    return _number(value, context)


def _csv_integer(value: str | None, context: str) -> int:
    if value in (None, ""):
        raise ReportError(f"{context} is missing")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ReportError(f"{context} must be an integer") from error
    return parsed


def _equal(actual: Any, expected: Any, context: str) -> None:
    if actual != expected:
        raise ReportError(f"{context}: expected {expected!r}, observed {actual!r}")


def _close(actual: Any, expected: Any, context: str) -> None:
    observed = _number(actual, context)
    wanted = _number(expected, context)
    if not math.isclose(observed, wanted, rel_tol=1e-10, abs_tol=1e-12):
        raise ReportError(f"{context}: expected {wanted!r}, observed {observed!r}")


def _digest(value: Any, context: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ReportError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _expected_jobs() -> set[tuple[str, str, int]]:
    return {
        (EXPECTED_ORACLE, variant, seed)
        for variant in EXPECTED_VARIANTS
        for seed in EXPECTED_SEEDS
    }


def _validate_matrix(path: Path) -> None:
    try:
        matrix = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ReportError(f"invalid matrix YAML at {path}: {error}") from error
    matrix = _mapping(matrix, "matrix YAML")
    _equal(matrix.get("schema_version"), 1, "matrix schema")
    _equal(matrix.get("experiment_id"), EXPERIMENT_ID, "matrix experiment_id")
    _equal(matrix.get("scientific_status"), SCIENTIFIC_STATUS, "matrix scientific status")
    _equal(matrix.get("model_path"), "outputs/paper_v1/checkpoints/50000.ckpt", "matrix model")
    _equal(matrix.get("variants"), list(EXPECTED_VARIANTS), "matrix variants")
    _equal(matrix.get("seeds"), list(EXPECTED_SEEDS), "matrix seeds")
    _equal(matrix.get("stop_on_failure"), True, "matrix stop_on_failure")
    _equal(
        matrix.get("tasks"),
        [{"oracle": "qed", "gamma": 0.0, "prior_mean": None, "prior_mean_source": None}],
        "matrix tasks",
    )
    common = _mapping(matrix.get("common"), "matrix common")
    expected = {
        "output_root": "output/pmo_ablation",
        "max_oracle_calls": 1000,
        "reporting_frequency": 100,
        "checkpoint_every": 100,
        "max_iterations": 10000,
        "population_size": 100,
        "warmup": 100,
        "legacy_warmup_off_by_one": True,
        "softmax_temp": 1.2,
        "randomness": 2.0,
        "guidance_scale": 2.0,
        "legacy_seed_count": 1,
        "delta_attribution": "novel_vs_parent",
        "durable_events": True,
    }
    _equal(dict(common), expected, "matrix common settings")


def _validate_collection_header(
    collection: Mapping[str, Any], collection_path: Path
) -> tuple[Path, str, Path, str, Path]:
    _equal(collection.get("schema_version"), COLLECTION_SCHEMA_VERSION, "collection schema")
    _equal(collection.get("collection_complete"), True, "collection completeness")
    _equal(collection.get("experiment_id"), EXPERIMENT_ID, "collection experiment_id")
    _equal(collection.get("missing_jobs"), [], "collection missing jobs")
    _equal(collection.get("skipped"), [], "collection skipped jobs")
    _equal(collection.get("run_count"), 12, "collection run count")

    matrix = _mapping(collection.get("matrix"), "collection matrix")
    _equal(matrix.get("schema_version"), 1, "collection matrix schema")
    _equal(matrix.get("expected_job_count"), 12, "matrix job count")
    _equal(
        matrix.get("recorded_by_all_collected_runs"),
        True,
        "matrix provenance completeness",
    )
    raw_jobs = _sequence(matrix.get("expected_jobs"), "matrix expected jobs")
    jobs: list[tuple[str, str, int]] = []
    for index, raw_job in enumerate(raw_jobs):
        job = _mapping(raw_job, f"matrix job {index}")
        jobs.append(
            (
                str(job.get("oracle")),
                str(job.get("variant")),
                _integer(job.get("seed"), f"matrix job {index} seed"),
            )
        )
    _equal(len(jobs), len(set(jobs)), "unique matrix jobs")
    _equal(set(jobs), _expected_jobs(), "matrix job set")
    matrix_path = _safe_file(matrix.get("path"), "matrix source")
    matrix_sha = _digest(matrix.get("sha256"), "matrix SHA-256")
    _equal(sha256_file(matrix_path), matrix_sha, "matrix source SHA-256")
    _validate_matrix(matrix_path)

    results = _mapping(collection.get("results_csv"), "collection results CSV")
    _equal(results.get("immutable"), True, "results CSV immutable marker")
    csv_sha = _digest(results.get("sha256"), "results CSV SHA-256")
    raw_reference = results.get("path")
    if not isinstance(raw_reference, str) or not raw_reference:
        raise ReportError("results CSV path must be nonempty")
    reference = Path(raw_reference)
    if reference.is_absolute() or len(reference.parts) != 1:
        raise ReportError("results CSV must be a basename beside the collection")
    _equal(reference.name, f"results.{csv_sha}.csv", "hash-named results CSV")
    csv_path = _safe_file(
        collection_path.parent / reference,
        "results CSV",
        contained_by=collection_path.parent,
    )
    _equal(sha256_file(csv_path), csv_sha, "results CSV SHA-256")
    experiment_root = _safe_directory(
        collection.get("experiment_root"), "collection experiment root"
    )
    return csv_path, csv_sha, matrix_path, matrix_sha, experiment_root


def _read_csv_rows(
    path: Path,
    expected_columns: Sequence[Any],
) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or ())
            _equal(columns, list(expected_columns), "results CSV columns")
            required = {
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
                "config_sha256",
                "model_sha256",
                "vocabulary_sha256",
                "source_summary_sha256",
                "source_events_sha256",
            }
            for prefix in VIEW_PREFIX.values():
                required.update(f"{prefix}_{metric}" for metric in SCALAR_NAMES)
            missing = required - set(columns)
            if missing:
                raise ReportError(f"results CSV lacks required columns: {sorted(missing)!r}")
            return [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as error:
        if isinstance(error, ReportError):
            raise
        raise ReportError(f"invalid results CSV at {path}: {error}") from error


def _resolve_source(
    source: Mapping[str, Any],
    *,
    experiment_root: Path,
    label: str,
) -> tuple[Path, str]:
    raw_path = source.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ReportError(f"{label} path must be nonempty")
    reference = Path(raw_path)
    if reference.is_absolute():
        raise ReportError(f"{label} path must be relative to the experiment root")
    path = _safe_file(experiment_root / reference, label, contained_by=experiment_root)
    digest = _digest(source.get("sha256"), f"{label} SHA-256")
    _equal(sha256_file(path), digest, f"{label} SHA-256")
    return path, digest


def _validate_config(
    config: Mapping[str, Any], *, variant: str, seed: int, matrix_sha: str
) -> None:
    expected_mode, expected_parent = EXPECTED_POLICY[variant]
    exact = {
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": SCIENTIFIC_STATUS,
        "oracle": EXPECTED_ORACLE,
        "variant": variant,
        "seed": seed,
        "max_oracle_calls": 1000,
        "reporting_frequency": 100,
        "checkpoint_every": 100,
        "max_iterations": 10000,
        "population_size": 100,
        "warmup": 100,
        "legacy_warmup_off_by_one": True,
        "gamma": 0.0,
        "softmax_temp": 1.2,
        "randomness": 2.0,
        "guidance_scale": 2.0,
        "min_mol_size": 10,
        "max_mol_size": 30,
        "min_support": 1,
        "policy_mode": expected_mode,
        "parent_control": expected_parent,
        "prior_strength": 0.0,
        "prior_mean": None,
        "prior_mean_source": None,
        "legacy_seed_count": 1,
        "seed_initialization_policy": EXPECTED_SEED_INITIALIZATION[variant],
        "credit_value_policy": EXPECTED_CREDIT_VALUE_POLICY[variant],
        "delta_attribution": "novel_vs_parent",
        "population_sampling_order": "canonical fragment string before uniform sampling",
        "statistical_duplicate_policy": (
            "one update per unique canonical parent-child transition"
            if variant in TARGET_VARIANTS
            else "one update per unique canonical child"
        ),
        "released_duplicate_policy": "repeat cached-child decomposition, matching release",
        "durable_events": True,
    }
    for key, wanted in exact.items():
        _equal(config.get(key), wanted, f"{variant} seed {seed} config {key}")
    model_path = config.get("model_path")
    if not isinstance(model_path, str) or Path(model_path).name != "50000.ckpt":
        raise ReportError(f"{variant} seed {seed} did not use 50000.ckpt")
    matrix_path = config.get("matrix_path")
    if not isinstance(matrix_path, str) or not matrix_path:
        raise ReportError(f"{variant} seed {seed} lacks matrix path")
    _equal(config.get("matrix_sha256"), matrix_sha, f"{variant} seed {seed} matrix hash")

    # These fields make the matched contrast explicit.  Requiring them avoids
    # accepting an older parent-throughput control that updated during warmup.
    mechanics = {
        "warmup_update_policy": (
            "frozen"
            if variant in TARGET_VARIANTS
            else "standard"
        ),
        "observation_identity": (
            "unique_canonical_parent_child_transition"
            if variant in TARGET_VARIANTS
            else "canonical_child_occurrence"
            if variant == "released"
            else "unique_canonical_child"
        ),
        "parent_domain_policy": (
            "parent_and_child_within_configured_atom_bounds"
            if variant in TARGET_VARIANTS
            else "child_within_configured_atom_bounds"
        ),
        "credit_fragment_policy": (
            "deterministic_cut_all_child_minus_parent"
            if variant in TARGET_VARIANTS
            else "sampled_three_cut_child"
        ),
    }
    for key, wanted in mechanics.items():
        _equal(config.get(key), wanted, f"{variant} seed {seed} config {key}")


def _metric_group(
    run_id: str,
    raw_metrics: Mapping[str, Any],
    view: str,
    row: Mapping[str, str],
) -> dict[str, float]:
    group = _mapping(raw_metrics.get(view), f"{run_id} {view}")
    prefix = VIEW_PREFIX[view]
    result: dict[str, float] = {}
    for name in SCALAR_NAMES:
        value = _number(group.get(name), f"{run_id} {view}.{name}")
        if not 0.0 <= value <= 1.0:
            raise ReportError(f"{run_id} {view}.{name} is outside [0, 1]")
        _close(value, _csv_number(row.get(f"{prefix}_{name}"), f"{run_id} CSV {prefix}_{name}"), f"{run_id} CSV/collection {prefix}_{name}")
        result[name] = value
    if (
        result["top_1"] + 1e-12 < result["top_10"]
        or result["top_10"] + 1e-12 < result["top_100"]
    ):
        raise ReportError(f"{run_id} {view} final top-k ordering is invalid")
    _equal(group.get("axis_budget"), EXPECTED_BUDGET, f"{run_id} {view} axis budget")
    _equal(group.get("reporting_frequency"), 100, f"{run_id} {view} reporting frequency")
    return result


def _outcome_score(value: Any, context: str) -> tuple[float, bool, str] | None:
    if value is None:
        return None
    row = _mapping(value, context)
    score = _number(row.get("score"), f"{context}.score")
    if not 0.0 <= score <= 1.0:
        raise ReportError(f"{context}.score is outside [0, 1]")
    charged = row.get("charged")
    if not isinstance(charged, bool):
        raise ReportError(f"{context}.charged must be Boolean")
    canonical = row.get("canonical_smiles")
    if not isinstance(canonical, str) or not canonical:
        raise ReportError(f"{context}.canonical_smiles must be nonempty")
    return score, charged, canonical


def _fragment_tuple(value: Any, context: str) -> tuple[str, ...]:
    raw = _sequence(value, context)
    result = tuple(raw)
    if any(not isinstance(item, str) or not item for item in result):
        raise ReportError(f"{context} must contain nonempty strings")
    if result != tuple(sorted(set(result))):
        raise ReportError(f"{context} must be sorted and unique")
    return result


def _validate_attribution(
    value: Any,
    *,
    context: str,
    applicable: bool,
    empty_reason: str | None = None,
) -> tuple[tuple[str, ...], float]:
    row = _mapping(value, context)
    expected_keys = {
        "applicable",
        "reason",
        "attribution_mode",
        "parent_all_fragments",
        "child_all_fragments",
        "credited_fragments",
        "mapping_counts",
        "mapping_covered",
        "mapping_coverage",
    }
    _equal(set(row), expected_keys, f"{context} fields")
    _equal(row.get("applicable"), applicable, f"{context}.applicable")
    parent = _fragment_tuple(row.get("parent_all_fragments"), f"{context}.parent_all_fragments")
    child = _fragment_tuple(row.get("child_all_fragments"), f"{context}.child_all_fragments")
    credited = _fragment_tuple(row.get("credited_fragments"), f"{context}.credited_fragments")
    counts = _mapping(row.get("mapping_counts"), f"{context}.mapping_counts")
    _equal(set(counts), {"parent_all", "child_all", "shared", "credited"}, f"{context}.mapping_counts fields")
    shared = set(parent) & set(child)
    expected_counts = {
        "parent_all": len(parent),
        "child_all": len(child),
        "shared": len(shared),
        "credited": len(credited),
    }
    _equal(dict(counts), expected_counts, f"{context}.mapping_counts")
    expected_coverage = len(credited) / len(child) if child else 0.0
    _close(row.get("mapping_coverage"), expected_coverage, f"{context}.mapping_coverage")
    _equal(row.get("mapping_covered"), bool(credited), f"{context}.mapping_covered")
    if applicable:
        _equal(row.get("reason"), "deterministic_mapping", f"{context}.reason")
        _equal(row.get("attribution_mode"), "novel_vs_parent", f"{context}.mode")
        _equal(set(credited), set(child) - set(parent), f"{context} set difference")
    else:
        _equal(row.get("reason"), empty_reason, f"{context}.reason")
        _equal(row.get("attribution_mode"), None, f"{context}.mode")
        _equal(parent, (), f"{context}.parent_all_fragments")
        _equal(child, (), f"{context}.child_all_fragments")
        _equal(credited, (), f"{context}.credited_fragments")
    return credited, expected_coverage


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "sample_sd": None,
            "median": None,
            "minimum": None,
            "maximum": None,
            "positive": 0,
            "zero": 0,
            "negative": 0,
        }
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "positive": sum(value > 0.0 for value in values),
        "zero": sum(value == 0.0 for value in values),
        "negative": sum(value < 0.0 for value in values),
    }


def _event_diagnostics(
    path: Path,
    *,
    run_id: str,
    variant: str,
    expected_events: int,
    expected_child_count: int,
) -> tuple[dict[str, Any] | None, dict[str, float], dict[str, int]]:
    target = variant in TARGET_VARIANTS
    event_count = 0
    charged_parent = charged_child = cached_parent = cached_child = 0
    paired_events = 0
    unique_transitions: set[tuple[str, str]] = set()
    unique_deltas: list[float] = []
    unique_transitions_with_credit = 0
    unique_fragment_coverages: list[float] = []
    credited_fragment_observations = 0
    credited_unique: set[str] = set()
    population_updates = duplicate_transitions = 0
    last_calls = 0
    fragment_stats: dict[str, tuple[float, int]] = {}
    charged_child_scores: list[float] = []
    try:
        handle = path.open(encoding="utf-8")
    except OSError as error:
        raise ReportError(f"cannot read {run_id} events: {error}") from error
    with handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                event = json.loads(line, parse_constant=_reject_json_constant)
            except ValueError as error:
                raise ReportError(f"invalid {run_id} event line {line_number}: {error}") from error
            event = _mapping(event, f"{run_id} event {line_number}")
            index = line_number - 1
            _equal(event.get("event_index"), index, f"{run_id} event index")
            _equal(event.get("iteration"), index, f"{run_id} iteration")
            remask = index > 100
            _equal(event.get("remask_enabled"), remask, f"{run_id} event {index} remask")
            parent = _outcome_score(event.get("parent_oracle"), f"{run_id} event {index} parent")
            child = _outcome_score(event.get("child_oracle"), f"{run_id} event {index} child")
            for outcome, kind in ((parent, "parent"), (child, "child")):
                if outcome is None:
                    continue
                if kind == "parent":
                    if outcome[1]:
                        charged_parent += 1
                    else:
                        cached_parent += 1
                elif outcome[1]:
                    charged_child += 1
                    charged_child_scores.append(outcome[0])
                else:
                    cached_child += 1
            calls = _integer(event.get("oracle_calls"), f"{run_id} event {index} calls")
            if calls < last_calls or calls > EXPECTED_BUDGET:
                raise ReportError(f"{run_id} event {index} has invalid call chronology")
            last_calls = calls
            update = _mapping(event.get("population_update"), f"{run_id} event {index} update")
            updated = update.get("updated")
            if not isinstance(updated, bool):
                raise ReportError(f"{run_id} event {index} update flag must be Boolean")
            if updated:
                population_updates += 1

            if not target:
                _equal(parent, None, f"{run_id} event {index} unexpected parent")
                _equal(event.get("attribution"), None, f"{run_id} event {index} attribution")
                event_count += 1
                continue

            if not remask:
                _equal(parent, None, f"{run_id} event {index} warmup parent")
                _equal(update.get("reason"), "frozen_warmup", f"{run_id} event {index} warmup reason")
                _equal(updated, False, f"{run_id} event {index} warmup update")
                _validate_attribution(
                    event.get("attribution"),
                    context=f"{run_id} event {index} attribution",
                    applicable=False,
                    empty_reason="warmup_frozen",
                )
                event_count += 1
                continue

            if parent is None:
                raise ReportError(f"{run_id} event {index} lacks a post-warmup parent")
            if child is None:
                _equal(update.get("reason"), "budget_after_parent", f"{run_id} event {index} terminal reason")
                _validate_attribution(
                    event.get("attribution"),
                    context=f"{run_id} event {index} attribution",
                    applicable=False,
                    empty_reason="budget_after_parent",
                )
                event_count += 1
                continue

            paired_events += 1
            credited, fragment_coverage = _validate_attribution(
                event.get("attribution"),
                context=f"{run_id} event {index} attribution",
                applicable=True,
            )
            transition = (parent[2], child[2])
            is_new_transition = transition not in unique_transitions
            delta_value = child[0] - parent[0]
            if is_new_transition:
                unique_transitions.add(transition)
                unique_deltas.append(delta_value)
                unique_fragment_coverages.append(fragment_coverage)
                if credited:
                    unique_transitions_with_credit += 1
                credited_fragment_observations += len(credited)
                credited_unique.update(credited)
            else:
                duplicate_transitions += 1
                _equal(updated, False, f"{run_id} event {index} duplicate transition update")

            observed = _fragment_tuple(
                update.get("observed_fragments", []),
                f"{run_id} event {index} observed fragments",
            )
            if updated:
                _equal(is_new_transition, True, f"{run_id} event {index} transition uniqueness")
                _equal(observed, credited, f"{run_id} event {index} credited update fragments")
                snapshots = _mapping(
                    event.get("fragment_statistics_after"),
                    f"{run_id} event {index} fragment statistics",
                )
                _equal(set(snapshots), set(credited), f"{run_id} event {index} statistic fragments")
                credited_value = delta_value if variant == "delta" else child[0]
                credit_label = "delta" if variant == "delta" else "absolute-child"
                for fragment in credited:
                    snapshot = _mapping(
                        snapshots[fragment],
                        f"{run_id} event {index} {fragment} statistics",
                    )
                    previous_total, previous_count = fragment_stats.get(fragment, (0.0, 0))
                    current_count = _integer(
                        snapshot.get("count"),
                        f"{run_id} event {index} {fragment} count",
                    )
                    current_total = _number(
                        snapshot.get("total"),
                        f"{run_id} event {index} {fragment} total",
                    )
                    _equal(
                        current_count,
                        previous_count + 1,
                        f"{run_id} event {index} {fragment} count increment",
                    )
                    _close(
                        current_total,
                        previous_total + credited_value,
                        f"{run_id} event {index} {fragment} {credit_label} increment",
                    )
                    fragment_stats[fragment] = (current_total, current_count)
            elif is_new_transition and credited:
                raise ReportError(f"{run_id} event {index} skipped a creditable unique transition")
            event_count += 1

    _equal(event_count, expected_events, f"{run_id} event count")
    _equal(last_calls, EXPECTED_BUDGET, f"{run_id} final oracle calls")
    _equal(charged_parent + charged_child, EXPECTED_BUDGET, f"{run_id} charged call count")
    _equal(charged_child, expected_child_count, f"{run_id} charged child count")
    if len(charged_child_scores) < COMMON_CHILD_HORIZON:
        raise ReportError(
            f"{run_id} has only {len(charged_child_scores)} charged children; "
            f"{COMMON_CHILD_HORIZON} are required for the common charged-child view"
        )
    common_summary = summarize_scores(
        charged_child_scores[:COMMON_CHILD_HORIZON],
        reporting_frequency=100,
        budget=COMMON_CHILD_HORIZON,
    )
    common_metrics = {
        name: _number(common_summary.get(name), f"{run_id} {COMMON_CHILD_VIEW}.{name}")
        for name in SCALAR_NAMES
    }
    if any(not 0.0 <= value <= 1.0 for value in common_metrics.values()):
        raise ReportError(f"{run_id} {COMMON_CHILD_VIEW} contains a metric outside [0, 1]")
    if (
        common_metrics["top_1"] + 1e-12 < common_metrics["top_10"]
        or common_metrics["top_10"] + 1e-12 < common_metrics["top_100"]
    ):
        raise ReportError(f"{run_id} {COMMON_CHILD_VIEW} final top-k ordering is invalid")
    total_parent_lookups = charged_parent + cached_parent
    event_accounting = {
        "logged_events": event_count,
        "runner_iterations_represented": event_count,
        "charged_parent_evaluations": charged_parent,
        "cached_parent_lookups": cached_parent,
        "charged_child_evaluations": charged_child,
        "cached_child_lookups": cached_child,
    }
    if not target:
        return None, common_metrics, event_accounting
    return {
        "events": event_count,
        "charged_parent_calls": charged_parent,
        "charged_child_calls": charged_child,
        "total_parent_lookups": total_parent_lookups,
        "cached_parent_lookups": cached_parent,
        "cached_parent_lookup_rate": (
            cached_parent / total_parent_lookups if total_parent_lookups else 0.0
        ),
        "cached_child_lookups": cached_child,
        "paired_events": paired_events,
        "unique_parent_child_transitions": len(unique_transitions),
        "duplicate_transition_events": duplicate_transitions,
        "population_updates": population_updates,
        "delta_y_unique_transitions": _distribution(unique_deltas),
        "attribution": {
            "applicable_unique_transitions": len(unique_transitions),
            "unique_transitions_with_any_credit": unique_transitions_with_credit,
            "any_credit_fraction": (
                unique_transitions_with_credit / len(unique_transitions)
                if unique_transitions
                else 0.0
            ),
            "mean_fragment_level_coverage": (
                statistics.fmean(unique_fragment_coverages)
                if unique_fragment_coverages
                else 0.0
            ),
            "credited_fragment_observations": credited_fragment_observations,
            "unique_credited_fragments": len(credited_unique),
        },
    }, common_metrics, event_accounting


def _summary(
    values: Sequence[float], *, by_seed: Mapping[int, float] | None = None
) -> dict[str, Any]:
    if not values:
        raise ReportError("cannot summarize an empty value sequence")
    result: dict[str, Any] = {
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }
    if by_seed is not None:
        result["by_seed"] = {str(seed): by_seed[seed] for seed in sorted(by_seed)}
    return result


def _aggregate(runs: Sequence[RunEvidence]) -> tuple[dict[str, Any], dict[str, Any]]:
    by_variant = {
        variant: sorted(
            (run for run in runs if run.variant == variant), key=lambda item: item.seed
        )
        for variant in EXPECTED_VARIANTS
    }
    aggregates: dict[str, Any] = {}
    for variant, variant_runs in by_variant.items():
        _equal([run.seed for run in variant_runs], list(EXPECTED_SEEDS), f"{variant} seeds")
        views: dict[str, Any] = {}
        for view in REPORT_VIEWS:
            views[view] = {}
            for metric in SCALAR_NAMES:
                by_seed = {
                    run.seed: run.metrics[view][metric] for run in variant_runs
                }
                views[view][metric] = _summary(list(by_seed.values()), by_seed=by_seed)
        child_by_seed = {run.seed: float(run.child_count) for run in variant_runs}
        aggregates[variant] = {
            "run_count": len(variant_runs),
            "metrics": views,
            "charged_child_count": _summary(
                list(child_by_seed.values()), by_seed=child_by_seed
            ),
            "diagnostics_by_seed": {
                str(run.seed): run.diagnostics
                for run in variant_runs
                if run.diagnostics is not None
            },
            "event_accounting_by_seed": {
                str(run.seed): run.event_accounting for run in variant_runs
            },
        }

    controls = {run.seed: run for run in by_variant[MATCHED_CONTROL]}
    targets = {run.seed: run for run in by_variant["delta"]}
    paired: dict[str, Any] = {
        "contrast": f"delta - {MATCHED_CONTROL}",
        "metrics": {},
    }
    for view in REPORT_VIEWS:
        paired["metrics"][view] = {}
        for metric in SCALAR_NAMES:
            by_seed = {
                seed: targets[seed].metrics[view][metric]
                - controls[seed].metrics[view][metric]
                for seed in EXPECTED_SEEDS
            }
            paired["metrics"][view][metric] = _summary(
                list(by_seed.values()), by_seed=by_seed
            )
    child_by_seed = {
        seed: float(targets[seed].child_count - controls[seed].child_count)
        for seed in EXPECTED_SEEDS
    }
    paired["charged_child_count"] = _summary(
        list(child_by_seed.values()), by_seed=child_by_seed
    )
    return aggregates, paired


def _validate_launch(run_id: str, launch: Mapping[str, Any]) -> None:
    for key in (
        "history_complete",
        "policy_complete",
        "matrix_provenance_complete",
        "provenance_complete",
        "durable_events_config_complete",
    ):
        _equal(launch.get(key), True, f"{run_id} launch {key}")
    _equal(launch.get("gpu_migrated"), False, f"{run_id} GPU migration")
    attempts = _sequence(launch.get("attempts"), f"{run_id} launch attempts")
    if not attempts:
        raise ReportError(f"{run_id} has no launch attempt")
    observed_sharing = False
    for index, raw_attempt in enumerate(attempts):
        attempt = _mapping(raw_attempt, f"{run_id} launch attempt {index}")
        record = _mapping(attempt.get("record"), f"{run_id} launch record {index}")
        _equal(record.get("utilization_threshold"), 10, f"{run_id} utilization threshold")
        physical = _mapping(record.get("physical_gpu"), f"{run_id} physical GPU")
        utilization = _number(
            physical.get("utilization_percent"), f"{run_id} launch utilization"
        )
        if utilization >= 10.0:
            raise ReportError(f"{run_id} launched at {utilization}% GPU utilization")
        if record.get("sharing_actual") is True:
            observed_sharing = True
            _equal(record.get("sharing_authorized"), True, f"{run_id} sharing authorization")
        uuid = physical.get("uuid")
        if not isinstance(uuid, str) or not uuid:
            raise ReportError(f"{run_id} launch lacks a GPU UUID")
    _equal(
        launch.get("any_gpu_sharing"),
        observed_sharing,
        f"{run_id} launch sharing summary",
    )


def load_report_data(collection_manifest: str | os.PathLike[str]) -> ReportData:
    """Load and independently audit the fixed delta-y experiment evidence."""

    collection_path = _safe_file(collection_manifest, "collection manifest")
    collection_sha = sha256_file(collection_path)
    collection = _read_json(collection_path, "collection manifest")
    csv_path, csv_sha, matrix_path, matrix_sha, experiment_root = (
        _validate_collection_header(collection, collection_path)
    )
    results_meta = _mapping(collection.get("results_csv"), "collection results CSV")
    rows = _read_csv_rows(
        csv_path,
        _sequence(results_meta.get("columns"), "collection results columns"),
    )
    _equal(len(rows), 12, "results CSV row count")
    rows_by_id: dict[str, dict[str, str]] = {}
    for index, row in enumerate(rows):
        run_id = row.get("run_id")
        if not run_id or run_id in rows_by_id:
            raise ReportError(f"results CSV row {index + 2} has a missing or duplicate run_id")
        rows_by_id[run_id] = row

    raw_runs = _sequence(collection.get("runs"), "collection runs")
    _equal(len(raw_runs), 12, "collection runs length")
    evidence: list[RunEvidence] = []
    observed_jobs: set[tuple[str, str, int]] = set()
    model_hashes: set[str] = set()
    vocabulary_hashes: set[str] = set()
    code_identities: set[tuple[Any, ...]] = set()
    gpu_uuids: set[str] = set()

    for ordinal, raw_run in enumerate(raw_runs):
        run = _mapping(raw_run, f"collection run {ordinal}")
        identity = _mapping(run.get("identity"), f"collection run {ordinal} identity")
        variant = identity.get("variant")
        if variant not in EXPECTED_VARIANTS:
            raise ReportError(f"collection run {ordinal} has unexpected variant {variant!r}")
        seed = _integer(identity.get("seed"), f"collection run {ordinal} seed")
        job = (str(identity.get("oracle")), str(variant), seed)
        if job in observed_jobs:
            raise ReportError(f"collection repeats job {job!r}")
        observed_jobs.add(job)
        run_id = f"{EXPERIMENT_ID}:qed:{variant}:seed{seed}"
        _equal(identity.get("run_id"), run_id, f"collection run {ordinal} run_id")
        _equal(identity.get("experiment_id"), EXPERIMENT_ID, f"{run_id} experiment")
        _equal(identity.get("oracle"), EXPECTED_ORACLE, f"{run_id} oracle")
        _equal(run.get("status"), "completed", f"{run_id} status")
        _equal(run.get("summary_schema_version"), 2, f"{run_id} summary schema")
        _equal(run.get("oracle_budget"), EXPECTED_BUDGET, f"{run_id} budget")

        config = dict(_mapping(run.get("config"), f"{run_id} config"))
        _validate_config(config, variant=str(variant), seed=seed, matrix_sha=matrix_sha)
        row = rows_by_id.pop(run_id, None)
        if row is None:
            raise ReportError(f"{run_id} has no results CSV row")
        expected_strings = {
            "experiment_id": EXPERIMENT_ID,
            "run_id": run_id,
            "oracle": EXPECTED_ORACLE,
            "variant": variant,
            "status": "completed",
            "model_path": config.get("model_path"),
        }
        for key, wanted in expected_strings.items():
            _equal(row.get(key), wanted, f"{run_id} CSV {key}")
        _equal(_csv_integer(row.get("seed"), f"{run_id} CSV seed"), seed, f"{run_id} CSV seed")
        _equal(_csv_integer(row.get("summary_schema_version"), f"{run_id} CSV schema"), 2, f"{run_id} CSV schema")
        _equal(_csv_integer(row.get("oracle_budget"), f"{run_id} CSV budget"), 1000, f"{run_id} CSV budget")
        _equal(_csv_integer(row.get("all_oracle_calls"), f"{run_id} CSV calls"), 1000, f"{run_id} CSV calls")
        child_count = _integer(run.get("charged_child_count"), f"{run_id} child count", minimum=1)
        _equal(_csv_integer(row.get("charged_child_count"), f"{run_id} CSV child count"), child_count, f"{run_id} CSV child count")
        elapsed = _number(run.get("elapsed_seconds"), f"{run_id} elapsed seconds")
        _close(row.get("elapsed_seconds"), elapsed, f"{run_id} CSV elapsed seconds")

        raw_metrics = _mapping(run.get("metrics"), f"{run_id} metrics")
        metrics = {
            view: _metric_group(run_id, raw_metrics, view, row)
            for view in VIEW_PREFIX
        }
        all_group = _mapping(raw_metrics.get("all_charged_molecules"), f"{run_id} all metrics")
        child_group = _mapping(raw_metrics.get("charged_children_total_call_axis"), f"{run_id} child metrics")
        _equal(all_group.get("score_count"), 1000, f"{run_id} all score count")
        _equal(child_group.get("score_count"), child_count, f"{run_id} child score count")

        provenance = dict(_mapping(run.get("provenance"), f"{run_id} provenance"))
        _equal(provenance.get("resume_count"), 0, f"{run_id} resume count")
        _equal(provenance.get("durable_events_provenance_complete"), True, f"{run_id} durable provenance")
        _equal(provenance.get("timing_provenance_complete"), True, f"{run_id} timing provenance")
        _equal(provenance.get("git_dirty"), False, f"{run_id} dirty code state")
        model_sha = _digest(provenance.get("model_sha256"), f"{run_id} model SHA-256")
        vocabulary_sha = _digest(provenance.get("vocabulary_sha256"), f"{run_id} vocabulary SHA-256")
        _equal(model_sha, EXPECTED_MODEL_SHA256, f"{run_id} final checkpoint SHA-256")
        _equal(vocabulary_sha, EXPECTED_VOCABULARY_SHA256, f"{run_id} vocabulary SHA-256")
        _equal(row.get("model_sha256"), model_sha, f"{run_id} CSV model SHA-256")
        _equal(row.get("vocabulary_sha256"), vocabulary_sha, f"{run_id} CSV vocabulary SHA-256")
        _equal(row.get("config_sha256"), provenance.get("config_sha256"), f"{run_id} CSV config SHA-256")
        model_hashes.add(model_sha)
        vocabulary_hashes.add(vocabulary_sha)
        code_identities.add(
            (
                provenance.get("git_commit"),
                provenance.get("git_dirty"),
                provenance.get("tracked_diff_sha256"),
            )
        )

        launch = dict(_mapping(run.get("launch"), f"{run_id} launch"))
        _validate_launch(run_id, launch)
        for raw_attempt in _sequence(launch.get("attempts"), f"{run_id} attempts"):
            physical = _mapping(
                _mapping(raw_attempt, f"{run_id} attempt").get("record"),
                f"{run_id} launch record",
            ).get("physical_gpu")
            gpu_uuids.add(str(_mapping(physical, f"{run_id} physical GPU").get("uuid")))

        raw_sources = _mapping(run.get("sources"), f"{run_id} sources")
        sources: dict[str, dict[str, Any]] = {}
        resolved_sources: dict[str, Path] = {}
        for name in ("manifest", "summary", "events", "checkpoint"):
            raw_source = _mapping(raw_sources.get(name), f"{run_id} {name} source")
            source_path, source_sha = _resolve_source(
                raw_source,
                experiment_root=experiment_root,
                label=f"{run_id} {name} source",
            )
            sources[name] = {"path": str(source_path), "sha256": source_sha}
            resolved_sources[name] = source_path
        _equal(row.get("source_summary_sha256"), sources["summary"]["sha256"], f"{run_id} CSV summary hash")
        _equal(row.get("source_events_sha256"), sources["events"]["sha256"], f"{run_id} CSV event hash")
        summary_doc = _read_json(resolved_sources["summary"], f"{run_id} summary")
        _equal(summary_doc.get("schema_version"), 2, f"{run_id} source summary schema")
        _equal(summary_doc.get("run_id"), run_id, f"{run_id} source summary run_id")
        _equal(summary_doc.get("status"), "completed", f"{run_id} source summary status")
        _equal(summary_doc.get("checkpoint_consistent"), True, f"{run_id} checkpoint consistency")
        summary_scores = _mapping(summary_doc.get("scores"), f"{run_id} summary scores")
        for view in VIEW_PREFIX:
            summary_group = _mapping(summary_scores.get(view), f"{run_id} summary {view}")
            for metric in SCALAR_NAMES:
                _close(summary_group.get(metric), metrics[view][metric], f"{run_id} summary {view}.{metric}")

        diagnostics, common_child_metrics, event_accounting = _event_diagnostics(
            resolved_sources["events"],
            run_id=run_id,
            variant=str(variant),
            expected_events=_integer(run.get("events"), f"{run_id} events", minimum=1),
            expected_child_count=child_count,
        )
        metrics[COMMON_CHILD_VIEW] = common_child_metrics
        for name, source in sources.items():
            _equal(sha256_file(Path(source["path"])), source["sha256"], f"{run_id} {name} hash after reading")

        evidence.append(
            RunEvidence(
                run_id=run_id,
                variant=str(variant),
                seed=seed,
                config=config,
                metrics=metrics,
                child_count=child_count,
                elapsed_seconds=elapsed,
                diagnostics=diagnostics,
                event_accounting=event_accounting,
                provenance=provenance,
                launch=launch,
                sources=sources,
            )
        )

    _equal(observed_jobs, _expected_jobs(), "collected run job set")
    _equal(rows_by_id, {}, "unreferenced CSV rows")
    _equal(len(model_hashes), 1, "model hash count")
    _equal(len(vocabulary_hashes), 1, "vocabulary hash count")
    _equal(len(code_identities), 1, "run code identity count")
    if len(gpu_uuids) > 4:
        raise ReportError(f"experiment used more than four GPUs: {sorted(gpu_uuids)!r}")
    evidence.sort(key=lambda item: (EXPECTED_VARIANTS.index(item.variant), item.seed))
    aggregates, paired = _aggregate(evidence)
    shared_job_count = sum(
        run.launch.get("any_gpu_sharing") is True for run in evidence
    )
    caveats = [
        "This is an exploratory QED-only, 1,000-call, three-seed experiment; it is neither confirmatory nor paper-comparable.",
        "Delta-y is an association for an adaptive parent-child proposal, not a causal fragment contribution.",
        "The deterministic all-cut child-minus-parent mapping is approximate; several credited fragments each receive the full molecular delta.",
        "Parent-scoring arms spend part of the fixed budget on parents and therefore evaluate fewer children than released or ordinary running mean.",
        "Both matched arms start with neutral zero online estimates; absolute seed scores only break ties. The released and ordinary running-mean arms are contextual references, not matched causal controls.",
        "A unique oracle call is a newly scored canonical molecule. A charged child is a unique child evaluation, not an optimizer iteration; logged events and cached-child lookups are reported separately. Cached parent lookups are free under the unique-call budget.",
        "The first-500-child view equalizes charged child evaluations, but those children can occur at different total oracle-call positions and remain adaptively generated.",
    ]
    if shared_job_count == len(evidence):
        caveats.append(
            "All 12 jobs shared a GPU. Score comparisons remain valid, but wall-time comparisons are suppressed."
        )
    elif shared_job_count:
        caveats.append(
            f"{shared_job_count} of 12 jobs shared a GPU. Score comparisons remain valid, but wall-time comparisons are suppressed."
        )
    _equal(sha256_file(collection_path), collection_sha, "collection hash after reading")
    _equal(sha256_file(csv_path), csv_sha, "results CSV hash after reading")
    _equal(sha256_file(matrix_path), matrix_sha, "matrix hash after reading")
    return ReportData(
        collection_path=collection_path,
        collection_sha256=collection_sha,
        csv_path=csv_path,
        csv_sha256=csv_sha,
        matrix_path=matrix_path,
        matrix_sha256=matrix_sha,
        experiment_root=experiment_root,
        collection=dict(collection),
        runs=tuple(evidence),
        aggregates=aggregates,
        paired=paired,
        caveats=tuple(caveats),
    )


def _mean_sd(value: Mapping[str, Any], digits: int = 6) -> str:
    return f"{value['mean']:.{digits}f} +/- {value['sample_sd']:.{digits}f}"


def _paragraph(text: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(str(text)).replace("\n", "<br/>"), style)


def _table(
    rows: Sequence[Sequence[Any]],
    *,
    widths: Sequence[float] | None,
    header: int = 1,
    font_size: float = 7.2,
) -> Table:
    styles = getSampleStyleSheet()
    cell = ParagraphStyle(
        "DeltaCell",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=font_size,
        leading=font_size + 1.6,
        spaceAfter=0,
        spaceBefore=0,
    )
    header_style = ParagraphStyle(
        "DeltaHeaderCell",
        parent=cell,
        fontName="Helvetica-Bold",
        textColor=colors.white,
    )
    rendered = [
        [
            _paragraph(value, header_style if row_index < header else cell)
            for value in row
        ]
        for row_index, row in enumerate(rows)
    ]
    table = Table(rendered, colWidths=widths, repeatRows=header, hAlign="LEFT")
    commands: list[tuple[Any, ...]] = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#AAB2BD")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        commands.append(("BACKGROUND", (0, 0), (-1, header - 1), colors.HexColor("#2F4858")))
    for row_index in range(header, len(rows)):
        if (row_index - header) % 2:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#F3F6F8")))
    table.setStyle(TableStyle(commands))
    return table


def _footer(canvas: reportlab_canvas.Canvas, document: SimpleDocTemplate) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#54606A"))
    canvas.drawString(14 * mm, 8 * mm, "GenMol QED delta-y ablation | exploratory")
    canvas.drawRightString(283 * mm, 8 * mm, f"Page {document.page}")
    canvas.restoreState()


def render_report_pdf(data: ReportData) -> bytes:
    """Render deterministic PDF bytes from validated evidence."""

    stream = io.BytesIO()
    document = SimpleDocTemplate(
        stream,
        pagesize=landscape(A4),
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=12 * mm,
        bottomMargin=14 * mm,
        title="GenMol QED Delta-y Credit Ablation",
        author="GenMol v2 reproducible experiment harness",
        subject="Exploratory fixed-budget fragment-credit ablation",
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "DeltaTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        leading=23,
        textColor=colors.HexColor("#20323D"),
        spaceAfter=7,
    )
    heading = ParagraphStyle(
        "DeltaHeading",
        parent=styles["Heading1"],
        fontSize=14,
        leading=17,
        textColor=colors.HexColor("#20323D"),
        spaceAfter=6,
    )
    subheading = ParagraphStyle(
        "DeltaSubheading",
        parent=styles["Heading2"],
        fontSize=10,
        leading=12,
        textColor=colors.HexColor("#2F4858"),
        spaceBefore=6,
        spaceAfter=4,
    )
    body = ParagraphStyle(
        "DeltaBody",
        parent=styles["BodyText"],
        fontSize=8.5,
        leading=11,
        spaceAfter=5,
    )
    note = ParagraphStyle(
        "DeltaNote",
        parent=body,
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#49545C"),
    )
    story: list[Any] = []
    story.append(_paragraph("GenMol QED Delta-y Credit Ablation", title))
    story.append(_paragraph(data.collection.get("created_at", "Audited collection"), note))
    story.append(Spacer(1, 3 * mm))
    story.append(_paragraph("Design and interpretation", heading))
    story.append(_paragraph(SCIENTIFIC_STATUS, body))
    story.append(
        _paragraph(
            "The primary paired contrast is delta minus running_mean_delta_control. "
            "Both arms freeze vocabulary updates during warmup, score the same parent/child "
            "domains, deduplicate canonical parent-child transitions, and use the same "
            "deterministic changed-fragment mapping. Both initialize online estimates at "
            "zero and use absolute seed scores only to break ties. Their sole configured "
            "difference is post-warmup credit: child minus parent versus absolute child. "
            "Released and ordinary running mean are contextual references only.",
            body,
        )
    )
    story.append(_paragraph("Exact metric definitions", subheading))
    story.append(
        _table(
            [
                ("Quantity", "Definition"),
                ("Final top-k", METRIC_DEFINITIONS["top_k"]["definition"]),
                ("AUC top-k", METRIC_DEFINITIONS["auc_top_k"]["definition"]),
            ],
            widths=(40 * mm, 190 * mm),
            font_size=6.9,
        )
    )
    primary = data.paired["metrics"]["all_charged_molecules"]
    story.append(_paragraph("Observed paired effect (not a claim of efficacy)", subheading))
    story.append(
        _table(
            [
                ("Metric", "Mean delta-control difference", "Seed 0", "Seed 1", "Seed 2"),
                *[
                    (
                        metric,
                        _mean_sd(primary[metric]),
                        *(f"{primary[metric]['by_seed'][str(seed)]:+.6f}" for seed in EXPECTED_SEEDS),
                    )
                    for metric in ("auc_top_1", "auc_top_10", "auc_top_100", "top_1", "top_10", "top_100")
                ],
            ],
            widths=(42 * mm, 55 * mm, 31 * mm, 31 * mm, 31 * mm),
        )
    )
    story.append(Spacer(1, 4 * mm))
    story.append(
        _paragraph(
            "Positive values favor delta. Sample SD describes three paired seeds; no "
            "confidence interval or significance decision is justified here.",
            note,
        )
    )

    story.append(PageBreak())
    story.append(_paragraph("Aggregate metric views", heading))
    story.append(_paragraph("All-call metrics", subheading))
    story.append(
        _paragraph(
            "Every unique parent and child oracle evaluation occupies its true position on "
            "the 1,000-call axis. Values are mean +/- sample SD across seeds 0, 1, and 2.",
            body,
        )
    )
    all_rows: list[Sequence[Any]] = [
        ("Variant", "AUC top-1", "AUC top-10", "AUC top-100", "Final top-1", "Final top-10", "Final top-100")
    ]
    for variant in EXPECTED_VARIANTS:
        values = data.aggregates[variant]["metrics"]["all_charged_molecules"]
        all_rows.append(
            (
                VARIANT_LABELS[variant],
                *(_mean_sd(values[metric]) for metric in SCALAR_NAMES),
            )
        )
    story.append(
        _table(
            all_rows,
            widths=(43 * mm, 32 * mm, 32 * mm, 32 * mm, 32 * mm, 32 * mm, 32 * mm),
            font_size=6.7,
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        _paragraph(
            "Charged-child score subset on the total unique-call axis (not equal child evaluations)",
            subheading,
        )
    )
    child_rows: list[Sequence[Any]] = [
        ("Variant", "Children", "AUC top-1", "AUC top-10", "AUC top-100", "Final top-1", "Final top-10", "Final top-100")
    ]
    for variant in EXPECTED_VARIANTS:
        aggregate = data.aggregates[variant]
        values = aggregate["metrics"]["charged_children_total_call_axis"]
        child_rows.append(
            (
                VARIANT_LABELS[variant],
                _mean_sd(aggregate["charged_child_count"], digits=1),
                *(_mean_sd(values[metric]) for metric in SCALAR_NAMES),
            )
        )
    story.append(
        _table(
            child_rows,
            widths=(41 * mm, 26 * mm, 28 * mm, 28 * mm, 28 * mm, 28 * mm, 28 * mm, 28 * mm),
            font_size=6.4,
        )
    )
    story.append(
        _paragraph(
            "Charged-child scores retain their actual global call indices; parent calls create "
            "gaps. The count column is the number of unique charged-child evaluations reached, "
            "not the number of runner iterations.",
            note,
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(_paragraph("Common first 500 charged child evaluations", subheading))
    common_rows: list[Sequence[Any]] = [
        (
            "Variant",
            "AUC top-1",
            "AUC top-10",
            "AUC top-100",
            "Final top-1",
            "Final top-10",
            "Final top-100",
        )
    ]
    for variant in EXPECTED_VARIANTS:
        values = data.aggregates[variant]["metrics"][COMMON_CHILD_VIEW]
        common_rows.append(
            (
                VARIANT_LABELS[variant],
                *(_mean_sd(values[metric]) for metric in SCALAR_NAMES),
            )
        )
    story.append(
        _table(
            common_rows,
            widths=(43 * mm, 32 * mm, 32 * mm, 32 * mm, 32 * mm, 32 * mm, 32 * mm),
            font_size=6.4,
        )
    )
    story.append(
        _paragraph(METRIC_DEFINITIONS["views"][COMMON_CHILD_VIEW], note)
    )

    story.append(PageBreak())
    story.append(_paragraph("Delta-y and attribution diagnostics", heading))
    story.append(
        _paragraph(
            "Diagnostics are recomputed from hash-checked event streams. Delta distributions "
            "and both attribution summaries use unique canonical parent-child transitions, "
            "matching the estimator's deduplication unit. Fragment coverage is the credited "
            "child-fragment fraction, averaged across those unique transitions.",
            body,
        )
    )
    diagnostic_rows: list[Sequence[Any]] = [
        (
            "Variant / seed",
            "Charged parents",
            "Cached parents / all lookups",
            "Charged children",
            "Unique transitions",
            "Delta-y mean +/- SD",
            "+ / 0 / -",
            "Transitions with any credit",
            "Mean fragment coverage",
            "Credited observations / unique",
        )
    ]
    for run in data.runs:
        if run.diagnostics is None:
            continue
        diagnostic = run.diagnostics
        distribution = diagnostic["delta_y_unique_transitions"]
        attribution = diagnostic["attribution"]
        diagnostic_rows.append(
            (
                (
                    f"{VARIANT_LABELS[run.variant]} / {run.seed}\n"
                    f"events {run.event_accounting['logged_events']}; cached child "
                    f"{run.event_accounting['cached_child_lookups']}"
                ),
                diagnostic["charged_parent_calls"],
                (
                    f"{diagnostic['cached_parent_lookups']}/"
                    f"{diagnostic['total_parent_lookups']} "
                    f"({diagnostic['cached_parent_lookup_rate']:.3f})"
                ),
                diagnostic["charged_child_calls"],
                diagnostic["unique_parent_child_transitions"],
                f"{distribution['mean']:+.5f} +/- {distribution['sample_sd']:.5f}",
                f"{distribution['positive']} / {distribution['zero']} / {distribution['negative']}",
                (
                    f"{attribution['unique_transitions_with_any_credit']}/"
                    f"{attribution['applicable_unique_transitions']} "
                    f"({attribution['any_credit_fraction']:.3f})"
                ),
                f"{attribution['mean_fragment_level_coverage']:.3f}",
                f"{attribution['credited_fragment_observations']} / {attribution['unique_credited_fragments']}",
            )
        )
    story.append(
        _table(
            diagnostic_rows,
            widths=(33 * mm, 19 * mm, 29 * mm, 19 * mm, 24 * mm, 38 * mm, 23 * mm, 29 * mm, 27 * mm, 27 * mm),
            font_size=5.7,
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(
        _paragraph(
            "Matched charged-child subset on total unique-call axis (not equal child evaluations)",
            subheading,
        )
    )
    child_paired = data.paired["metrics"]["charged_children_total_call_axis"]
    story.append(
        _table(
            [
                ("Metric", "Delta minus control", "Seed deltas"),
                *[
                    (
                        metric,
                        _mean_sd(child_paired[metric]),
                        ", ".join(
                            f"s{seed}={child_paired[metric]['by_seed'][str(seed)]:+.6f}"
                            for seed in EXPECTED_SEEDS
                        ),
                    )
                    for metric in SCALAR_NAMES
                ],
            ],
            widths=(48 * mm, 62 * mm, 120 * mm),
            font_size=6.4,
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(_paragraph("Matched first-500-charged-child diagnostic", subheading))
    common_paired = data.paired["metrics"][COMMON_CHILD_VIEW]
    story.append(
        _table(
            [
                ("Metric", "Delta minus control", "Seed deltas"),
                *[
                    (
                        metric,
                        _mean_sd(common_paired[metric]),
                        ", ".join(
                            f"s{seed}={common_paired[metric]['by_seed'][str(seed)]:+.6f}"
                            for seed in EXPECTED_SEEDS
                        ),
                    )
                    for metric in SCALAR_NAMES
                ],
            ],
            widths=(48 * mm, 62 * mm, 120 * mm),
            font_size=6.4,
        )
    )
    story.append(
        _paragraph(
            "This equalizes 500 unique charged-child evaluations per run. Its AUC uses dense "
            "charged-child indices 1..500, the origin (0, 0), 100-evaluation checkpoints, "
            "trapezoids, and a divisor of 500; it does not equalize total oracle calls or "
            "runner iterations.",
            note,
        )
    )

    story.append(PageBreak())
    story.append(_paragraph("Configuration and provenance", heading))
    first = data.runs[0]
    code_identity = (
        f"{first.provenance.get('git_commit')} | dirty={first.provenance.get('git_dirty')} | "
        f"tracked diff={first.provenance.get('tracked_diff_sha256')}"
    )
    gpu_groups: dict[str, dict[str, Any]] = {}
    for run in data.runs:
        for raw_attempt in run.launch.get("attempts", []):
            record = raw_attempt["record"]
            gpu = record["physical_gpu"]
            uuid = str(gpu.get("uuid"))
            group = gpu_groups.setdefault(
                uuid,
                {"indices": set(), "jobs": set(), "utilizations": [], "free_mib": []},
            )
            group["indices"].add(gpu.get("index"))
            group["jobs"].add(f"{run.variant}/s{run.seed}")
            group["utilizations"].append(float(gpu.get("utilization_percent")))
            group["free_mib"].append(
                int(gpu.get("memory_total_mib", 0)) - int(gpu.get("memory_used_mib", 0))
            )
    gpu_rows = []
    for uuid in sorted(gpu_groups):
        group = gpu_groups[uuid]
        indices = ",".join(str(value) for value in sorted(group["indices"]))
        gpu_rows.append(
            f"GPU {indices} {uuid}: {len(group['jobs'])} jobs; launch utilization "
            f"{min(group['utilizations']):g}-{max(group['utilizations']):g}%; "
            f"minimum free {min(group['free_mib'])} MiB"
        )
    shared_job_count = sum(
        run.launch.get("any_gpu_sharing") is True for run in data.runs
    )
    sharing_label = (
        "All 12 jobs shared a GPU; wall-time comparisons suppressed"
        if shared_job_count == len(data.runs)
        else f"{shared_job_count} of 12 jobs shared a GPU; wall-time comparisons suppressed"
        if shared_job_count
        else "No job shared a GPU"
    )
    provenance_rows = [
        ("Item", "Value"),
        ("Model checkpoint", f"50000.ckpt | SHA-256 {EXPECTED_MODEL_SHA256}"),
        ("Initial QED vocabulary", f"SHA-256 {EXPECTED_VOCABULARY_SHA256}"),
        ("Matrix", f"{data.matrix_path.name} | SHA-256 {data.matrix_sha256}"),
        (
            "Collection",
            f"{data.collection_path.name} | SHA-256 {data.collection_sha256}",
        ),
        ("Results CSV", f"{data.csv_path.name} | SHA-256 {data.csv_sha256}"),
        ("Reporter", f"{REPORTER_PATH} | SHA-256 {sha256_file(REPORTER_PATH)}"),
        ("Code identity", code_identity),
        ("GPU sharing", sharing_label),
        ("GPU launches", "\n".join(gpu_rows)),
    ]
    story.append(_table(provenance_rows, widths=(48 * mm, 210 * mm), font_size=6.8))
    story.append(Spacer(1, 2 * mm))
    story.append(_paragraph("Locked hyperparameters", subheading))
    story.append(
        _table(
            [
                ("Budget", "Warmup", "Population", "gamma", "Temperature", "Randomness", "Guidance", "Attribution"),
                ("1,000 unique calls", "parameter 100; 101 frozen (0..100)", "100", "0.0", "1.2", "2.0", "2.0", "novel_vs_parent"),
            ],
            widths=(39 * mm, 52 * mm, 29 * mm, 22 * mm, 31 * mm, 31 * mm, 27 * mm, 40 * mm),
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(_paragraph("Estimator controls", subheading))
    story.append(
        _table(
            [
                ("Arm", "Seed initialization", "Post-warmup credit value"),
                (
                    VARIANT_LABELS["released"],
                    EXPECTED_SEED_INITIALIZATION["released"],
                    EXPECTED_CREDIT_VALUE_POLICY["released"],
                ),
                (
                    VARIANT_LABELS["running_mean"],
                    EXPECTED_SEED_INITIALIZATION["running_mean"],
                    EXPECTED_CREDIT_VALUE_POLICY["running_mean"],
                ),
                (
                    "transition-absolute control",
                    EXPECTED_SEED_INITIALIZATION[MATCHED_CONTROL],
                    EXPECTED_CREDIT_VALUE_POLICY[MATCHED_CONTROL],
                ),
                (
                    "delta-y",
                    EXPECTED_SEED_INITIALIZATION["delta"],
                    EXPECTED_CREDIT_VALUE_POLICY["delta"],
                ),
            ],
            widths=(55 * mm, 105 * mm, 90 * mm),
            font_size=6.7,
        )
    )
    story.append(Spacer(1, 1 * mm))
    story.append(_paragraph("Scientific caveats", heading))
    for caveat in data.caveats:
        story.append(_paragraph(f"- {caveat}", note))

    document.build(
        story,
        onFirstPage=_footer,
        onLaterPages=_footer,
        canvasmaker=_InvariantCanvas,
    )
    return stream.getvalue()


def validate_pdf(payload: bytes) -> dict[str, Any]:
    if not payload.startswith(b"%PDF-"):
        raise ReportError("rendered payload is not a PDF")
    try:
        reader = PdfReader(io.BytesIO(payload))
        if reader.is_encrypted:
            raise ReportError("rendered PDF is encrypted")
        page_count = len(reader.pages)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except ReportError:
        raise
    except Exception as error:
        raise ReportError(f"rendered PDF failed validation: {error}") from error
    if page_count != 4:
        raise ReportError(f"rendered PDF must have exactly 4 pages, observed {page_count}")
    for heading in (
        "GenMol QED Delta-y Credit Ablation",
        "All-call metrics",
        "Common first 500 charged child evaluations",
        "Delta-y and attribution diagnostics",
        "Matched first-500-charged-child diagnostic",
        "Configuration and provenance",
        "Scientific caveats",
    ):
        if heading not in text:
            raise ReportError(f"rendered PDF is missing heading {heading!r}")
    return {"page_count": page_count, "size_bytes": len(payload), "encrypted": False}


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


def _publish_pdf(directory: Path, payload: bytes) -> tuple[Path, str]:
    digest = hashlib.sha256(payload).hexdigest()
    destination = directory / f"delta_ablation.{digest}.pdf"
    if destination.exists():
        if destination.read_bytes() != payload:
            raise ReportError(f"hash-named PDF has unexpected contents: {destination}")
        return destination, digest
    descriptor, temporary_name = tempfile.mkstemp(
        dir=directory,
        prefix=".delta_ablation.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_bytes() != payload:
                raise ReportError(f"concurrent PDF publication disagreed: {destination}")
        _fsync_directory(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination, digest


def _atomic_json(
    destination: Path, value: Mapping[str, Any], *, overwrite: bool
) -> Path:
    payload = (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if destination.exists():
        if destination.read_bytes() == payload:
            return destination
        if not overwrite:
            raise FileExistsError(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination


def _prepare_output_directory(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path).expanduser().absolute()
    for component in (candidate, *candidate.parents):
        if component.is_symlink():
            raise ReportError(f"output directory must not traverse a symlink: {component}")
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate.resolve(strict=True)


def _recheck_sources(data: ReportData) -> None:
    _equal(sha256_file(data.collection_path), data.collection_sha256, "collection recheck")
    _equal(sha256_file(data.csv_path), data.csv_sha256, "results CSV recheck")
    _equal(sha256_file(data.matrix_path), data.matrix_sha256, "matrix recheck")
    for run in data.runs:
        for name, source in run.sources.items():
            _equal(
                sha256_file(Path(source["path"])),
                source["sha256"],
                f"{run.run_id} {name} source recheck",
            )


def write_report(
    collection_manifest: str | os.PathLike[str],
    *,
    output_dir: str | os.PathLike[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate inputs and publish deterministic hash-addressed report artifacts."""

    data = load_report_data(collection_manifest)
    reporter_sha256 = sha256_file(REPORTER_PATH)
    destination = _prepare_output_directory(
        data.collection_path.parent if output_dir is None else output_dir
    )
    lock_path = destination / ".delta_ablation_report.lock"
    try:
        lock = _DestinationLock(lock_path)
    except BlockingIOError as error:
        raise ReportError(f"report destination is active: {destination}") from error
    with lock:
        _recheck_sources(data)
        pdf_payload = render_report_pdf(data)
        # ReportLab's invariant canvas should make identical evidence produce
        # byte-identical output.  Check that property before publication.
        if render_report_pdf(data) != pdf_payload:
            raise ReportError("PDF renderer is not deterministic")
        pdf_validation = validate_pdf(pdf_payload)
        pdf_path, pdf_sha = _publish_pdf(destination, pdf_payload)
        shared_job_count = sum(
            run.launch.get("any_gpu_sharing") is True for run in data.runs
        )
        gpu_uuids = {
            str(raw_attempt["record"]["physical_gpu"]["uuid"])
            for run in data.runs
            for raw_attempt in run.launch.get("attempts", [])
        }
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "report_type": "fixed QED delta-y credit-assignment ablation",
            "experiment_id": EXPERIMENT_ID,
            "scientific_status": SCIENTIFIC_STATUS,
            "collection_created_at": data.collection.get("created_at"),
            "inputs": {
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
                "model_sha256": EXPECTED_MODEL_SHA256,
                "vocabulary_sha256": EXPECTED_VOCABULARY_SHA256,
            },
            "reporter": {
                "path": str(REPORTER_PATH),
                "sha256": reporter_sha256,
            },
            "design": {
                "oracle": EXPECTED_ORACLE,
                "variants": list(EXPECTED_VARIANTS),
                "seeds": list(EXPECTED_SEEDS),
                "unique_oracle_calls_per_run": EXPECTED_BUDGET,
                "run_count": 12,
                "matched_contrast": f"delta - {MATCHED_CONTROL}",
                "matched_variants": [MATCHED_CONTROL, "delta"],
                "contextual_reference_variants": sorted(CONTEXTUAL_VARIANTS),
                "delta_attribution": "novel_vs_parent",
                "common_charged_child_horizon": COMMON_CHILD_HORIZON,
                "calls_vs_iterations": (
                    "Unique oracle calls count newly scored canonical molecules; "
                    "charged children count unique child evaluations, while runner "
                    "iterations/events also include cached lookups."
                ),
            },
            "metric_definitions": METRIC_DEFINITIONS,
            "execution": {
                "unique_gpu_count": len(gpu_uuids),
                "shared_job_count": shared_job_count,
                "all_jobs_shared": shared_job_count == len(data.runs),
                "wall_time_comparisons_reported": False,
            },
            "aggregates": data.aggregates,
            "paired_delta_vs_matched_control": data.paired,
            "runs": [
                {
                    "run_id": run.run_id,
                    "variant": run.variant,
                    "seed": run.seed,
                    "charged_child_count": run.child_count,
                    "event_accounting": run.event_accounting,
                    "metrics": run.metrics,
                    "delta_diagnostics": run.diagnostics,
                    "config": run.config,
                    "provenance": run.provenance,
                    "launch": run.launch,
                    "sources": run.sources,
                }
                for run in data.runs
            ],
            "caveats": list(data.caveats),
            "pdf": {
                "path": str(pdf_path),
                "sha256": pdf_sha,
                **pdf_validation,
            },
        }
        report_path = _atomic_json(
            destination / "delta_ablation_report.json",
            report,
            overwrite=overwrite,
        )
        _recheck_sources(data)
        _equal(
            sha256_file(REPORTER_PATH),
            reporter_sha256,
            "reporter source SHA-256 after publication",
        )
        _equal(sha256_file(pdf_path), pdf_sha, "published PDF SHA-256")
    return {
        "pdf_path": str(pdf_path),
        "pdf_sha256": pdf_sha,
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "page_count": pdf_validation["page_count"],
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    result = write_report(
        args.collection_manifest,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

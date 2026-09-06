"""Re-score immutable legacy MDLM rows with the current metric implementation.

The historical benchmark summaries use schema 2 and are evidence, not mutable
state.  This runner reads them through stable file descriptors, validates their
manifest-pinned byte identities, and sends each seed to a fresh Python
interpreter.  A fresh interpreter is required because PyTDC diversity passes
through set iteration whose order is fixed only at interpreter start by
``PYTHONHASHSEED``.

No checkpoint is loaded and no samples are generated.  The only output is one
schema-1 JSON attestation, published atomically without replacement after all
three workers agree with every historical row, summary, and frozen-manifest
value.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import io
import json
import math
import os
import platform
import socket
import stat
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_SRC = REPOSITORY_ROOT / "src"
for import_root in (REPOSITORY_ROOT, REPOSITORY_SRC):
    while str(import_root) in sys.path:
        sys.path.remove(str(import_root))
    sys.path.insert(0, str(import_root))

from scripts.exps.denovo import benchmark  # noqa: E402
from scripts.exps.denovo import report as denovo_report  # noqa: E402


SCHEMA_VERSION = 1
LEGACY_SUMMARY_SCHEMA_VERSION = 2
EXPECTED_SEEDS = (0, 1, 2)
EXPECTED_SAMPLES_PER_SEED = 1_000
EXPECTED_MANIFEST_SHA256 = (
    "6da46fc615dedbcca436da087a2c1e9145f5d110036e0c15bb431ded3c2e5539"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "8d00aa47b02f64bf39ff6b0b2e786f213587366fc2c3d29712a00f3f84108dd6"
)
EXPECTED_HISTORICAL_RUNNER_SHA256 = (
    "a77d7c84d8628403c6d7a11edebce5bf9b8405e3f8b79a326a6e06bd7f444cea"
)
EXPECTED_SOURCE_AGGREGATE_SHA256 = (
    "b474efc593b665489359425dbe1ed0873f8ae1d44b77478b871aff6d6b555904"
)
NUMERIC_ABSOLUTE_TOLERANCE = 1e-12
WORKER_RESULT_PREFIX = "GENMOL_MDLM_RESCORE_WORKER_RESULT="

MANIFEST_RELATIVE_PATH = Path("experiments/udlm/baselines/mdlm_50000.json")
RESCORE_RELATIVE_PATH = Path("scripts/udlm/rescore_mdlm_baseline.py")
BENCHMARK_RELATIVE_PATH = Path("scripts/exps/denovo/benchmark.py")
REPORT_RELATIVE_PATH = Path("scripts/exps/denovo/report.py")
CHEMISTRY_RELATIVE_PATH = Path("src/genmol/utils/utils_chem.py")

# This dedicated worktree lives at <workspace>/run_sources/<worktree>.  Keep a
# portable fallback for the eventual merged checkout, where the historical
# output directory and repository root coincide.
_outer_workspace_candidate = REPOSITORY_ROOT.parents[1]
WORKSPACE_SCOPE_ROOT = (
    _outer_workspace_candidate
    if (_outer_workspace_candidate / "output/benchmarks/denovo_50000").is_dir()
    else REPOSITORY_ROOT
)
DEFAULT_HISTORICAL_RUNS_DIR = (
    WORKSPACE_SCOPE_ROOT / "output/benchmarks/denovo_50000"
)
DEFAULT_OUTPUT_PATH = REPOSITORY_ROOT / (
    "output/udlm/baseline_rescore_attestation.json"
)

EXPECTED_PROTOCOL = {
    "seeds": [0, 1, 2],
    "samples_per_seed": 1_000,
    "total_requested_samples": 3_000,
    "single_generation_batch_per_seed": True,
    "softmax_temperature": 0.5,
    "randomness": 0.5,
    "minimum_added_length": 40,
    "safe_version": "V1",
    "use_bracket_safe": False,
}

OPTIONAL_TEXT_FIELDS = frozenset(
    {
        "raw_safe",
        "raw_safe_error",
        "strict_smiles",
        "strict_decode_error",
        "released_repaired_smiles",
        "released_smiles",
        "released_decode_error",
    }
)
NUMERIC_ROW_FIELDS = frozenset(
    {"strict_qed", "strict_sa", "released_qed", "released_sa"}
)
OPTIONAL_BOOLEAN_ROW_FIELDS = frozenset(
    {"strict_quality_pass", "released_quality_pass"}
)
REQUIRED_BOOLEAN_ROW_FIELDS = frozenset(
    {
        "strict_is_first_unique",
        "strict_quality_counted",
        "released_is_first_unique",
        "released_quality_counted",
        "released_was_recovered",
        "released_largest_component_applied",
    }
)
OFFLINE_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "NVIDIA_VISIBLE_DEVICES": "",
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "WANDB_MODE": "offline",
    "WANDB_DISABLED": "true",
}
PYTHON_NETWORK_GUARDED_APIS = (
    "socket.create_connection",
    "socket.getaddrinfo",
    "socket.socket.connect",
    "socket.socket.connect_ex",
)
PYTHON_NETWORK_GUARD_LIMITATION = (
    "Python-runtime guard only; this is not OS-level or process-level network "
    "isolation and does not claim to block native extensions, subprocesses, "
    "raw sockets, or any other unguarded socket, name-resolution, or datagram API"
)


class RescoreValidationError(ValueError):
    """Raised when legacy evidence cannot support an exact attestation."""


@dataclass(frozen=True)
class RetainedArtifact:
    """Bytes retained after a stable, no-symlink descriptor read."""

    path: Path
    payload: bytes
    sha256: str
    size_bytes: int

    def provenance(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "read_policy": (
                "regular_file_no_symlink_stable_descriptor_bytes_retained_in_memory"
            ),
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RescoreValidationError(f"{label} must be a JSON object")
    return value


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise RescoreValidationError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise RescoreValidationError(f"{label} must be at least {minimum}")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RescoreValidationError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise RescoreValidationError(f"{label} must be finite")
    return result


def _sha256_value(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RescoreValidationError(
            f"{label} must be 64 lowercase hexadecimal digits"
        )
    return value


def _reject_json_constant(value: str) -> None:
    raise RescoreValidationError(f"non-finite JSON constant is forbidden: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RescoreValidationError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def strict_json_loads(payload: bytes, *, label: str) -> Any:
    """Decode UTF-8 JSON while rejecting duplicate keys and nonfinite values."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RescoreValidationError(f"{label} is not UTF-8") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except RescoreValidationError:
        raise
    except json.JSONDecodeError as error:
        raise RescoreValidationError(f"{label} is not valid JSON") from error


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def read_stable_regular_file(
    path: Path,
    *,
    allowed_root: Path,
    label: str,
    expected_sha256: str | None = None,
) -> RetainedArtifact:
    """Read one file once, rejecting traversal, symlinks, and replacement races."""

    root = allowed_root.resolve(strict=True)
    candidate = path if path.is_absolute() else root / path
    candidate = Path(os.path.abspath(candidate))
    if not _is_within(candidate, root):
        raise RescoreValidationError(f"{label} escapes the allowed project scope")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise RescoreValidationError(f"{label} is unavailable: {candidate}") from error
    if resolved != candidate:
        raise RescoreValidationError(f"{label} must not traverse a symlink")
    try:
        path_before = candidate.stat(follow_symlinks=False)
    except OSError as error:
        raise RescoreValidationError(f"cannot stat {label}: {candidate}") from error
    if not stat.S_ISREG(path_before.st_mode):
        raise RescoreValidationError(f"{label} is not a regular file: {candidate}")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as error:
        raise RescoreValidationError(f"cannot securely open {label}: {candidate}") from error
    try:
        before = os.fstat(descriptor)
        expected_identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        path_identity = (
            path_before.st_dev,
            path_before.st_ino,
            path_before.st_mode,
            path_before.st_size,
            path_before.st_mtime_ns,
            path_before.st_ctime_ns,
        )
        if not stat.S_ISREG(before.st_mode) or path_identity != expected_identity:
            raise RescoreValidationError(f"{label} changed before it was opened")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    try:
        path_after = candidate.stat(follow_symlinks=False)
    except OSError as error:
        raise RescoreValidationError(f"{label} disappeared while being read") from error
    for observed in (after, path_after):
        observed_identity = (
            observed.st_dev,
            observed.st_ino,
            observed.st_mode,
            observed.st_size,
            observed.st_mtime_ns,
            observed.st_ctime_ns,
        )
        if observed_identity != expected_identity:
            raise RescoreValidationError(f"{label} changed while being read")

    payload = b"".join(chunks)
    digest = _sha256_bytes(payload)
    if expected_sha256 is not None and digest != expected_sha256:
        raise RescoreValidationError(
            f"{label} SHA-256 mismatch: {digest} != {expected_sha256}"
        )
    return RetainedArtifact(
        path=candidate,
        payload=payload,
        sha256=digest,
        size_bytes=len(payload),
    )


def _assert_exact_or_close(actual: Any, expected: Any, *, label: str) -> None:
    """Compare nested evidence: counts/types exactly, floats at absolute 1e-12."""

    if isinstance(expected, Mapping):
        actual_mapping = _mapping(actual, label)
        if set(actual_mapping) != set(expected):
            raise RescoreValidationError(
                f"{label} keys differ: {sorted(actual_mapping)} != {sorted(expected)}"
            )
        for key in expected:
            _assert_exact_or_close(
                actual_mapping[key], expected[key], label=f"{label}.{key}"
            )
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise RescoreValidationError(f"{label} list shape differs")
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _assert_exact_or_close(
                actual_item, expected_item, label=f"{label}[{index}]"
            )
        return
    if type(expected) is int or type(actual) is int:
        if type(actual) is not type(expected) or actual != expected:
            raise RescoreValidationError(
                f"{label} count/integer differs: {actual!r} != {expected!r}"
            )
        return
    if isinstance(expected, float) or isinstance(actual, float):
        actual_number = _finite_number(actual, label)
        expected_number = _finite_number(expected, label)
        if not math.isclose(
            actual_number,
            expected_number,
            rel_tol=0.0,
            abs_tol=NUMERIC_ABSOLUTE_TOLERANCE,
        ):
            raise RescoreValidationError(
                f"{label} differs: {actual_number!r} != {expected_number!r} "
                f"within abs {NUMERIC_ABSOLUTE_TOLERANCE}"
            )
        return
    if type(actual) is not type(expected) or actual != expected:
        raise RescoreValidationError(
            f"{label} differs exactly: {actual!r} != {expected!r}"
        )


def _ordered_seed_rows(value: Any, *, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or len(value) != len(EXPECTED_SEEDS):
        raise RescoreValidationError(f"{label} must contain exactly three seed rows")
    by_seed: dict[int, Mapping[str, Any]] = {}
    for index, raw_row in enumerate(value):
        row = _mapping(raw_row, f"{label}[{index}]")
        seed = _integer(row.get("seed"), f"{label}[{index}].seed", minimum=0)
        if seed in by_seed:
            raise RescoreValidationError(f"{label} contains duplicate seed {seed}")
        by_seed[seed] = row
    if tuple(sorted(by_seed)) != EXPECTED_SEEDS:
        raise RescoreValidationError(f"{label} seeds must be {list(EXPECTED_SEEDS)}")
    return [by_seed[seed] for seed in EXPECTED_SEEDS]


def validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the semantics behind the already byte-pinned frozen manifest."""

    if manifest.get("schema_version") != 1:
        raise RescoreValidationError("manifest schema_version must equal 1")
    if manifest.get("purpose") != (
        "frozen audited local MDLM comparator for UDLM; not an exact paper "
        "reproduction"
    ):
        raise RescoreValidationError("manifest purpose is unexpected")
    if manifest.get("protocol") != EXPECTED_PROTOCOL:
        raise RescoreValidationError("manifest protocol is not the frozen protocol")

    checkpoint = _mapping(manifest.get("checkpoint"), "manifest.checkpoint")
    if (
        checkpoint.get("sha256") != EXPECTED_CHECKPOINT_SHA256
        or checkpoint.get("size_bytes") != 1_396_998_679
        or checkpoint.get("global_step") != 50_000
        or checkpoint.get("diffusion_type") != "mdlm"
    ):
        raise RescoreValidationError("manifest checkpoint identity is unexpected")
    source_aggregate = _mapping(
        manifest.get("source_aggregate"), "manifest.source_aggregate"
    )
    if source_aggregate != {
        "original_relative_path": (
            "output/benchmarks/denovo_50000/report/aggregate.json"
        ),
        "sha256": EXPECTED_SOURCE_AGGREGATE_SHA256,
        "schema_version": 1,
        "runner_sha256": EXPECTED_HISTORICAL_RUNNER_SHA256,
    }:
        raise RescoreValidationError("manifest source aggregate identity is unexpected")

    validated_rows: dict[str, list[Mapping[str, Any]]] = {}
    metric_values: dict[str, dict[str, list[float]]] = {}
    for branch_name in ("released_comparable", "strict"):
        branch = _mapping(manifest.get(branch_name), f"manifest.{branch_name}")
        rows = _ordered_seed_rows(
            branch.get("per_seed"), label=f"manifest.{branch_name}.per_seed"
        )
        values = {name: [] for name in ("validity", "uniqueness", "quality", "diversity")}
        for seed, row in zip(EXPECTED_SEEDS, rows, strict=True):
            requested = _integer(
                row.get("requested"), f"manifest.{branch_name}.seed_{seed}.requested"
            )
            valid_count = _integer(
                row.get("valid_count"),
                f"manifest.{branch_name}.seed_{seed}.valid_count",
            )
            unique_count = _integer(
                row.get("unique_count"),
                f"manifest.{branch_name}.seed_{seed}.unique_count",
            )
            quality_count = _integer(
                row.get("quality_count"),
                f"manifest.{branch_name}.seed_{seed}.quality_count",
            )
            if not 0 <= quality_count <= unique_count <= valid_count <= requested:
                raise RescoreValidationError(
                    f"manifest {branch_name} seed {seed} count funnel is invalid"
                )
            expected_rates = {
                "validity": valid_count / requested,
                "uniqueness": unique_count / valid_count if valid_count else None,
                "quality": quality_count / requested,
            }
            for metric, expected in expected_rates.items():
                _assert_exact_or_close(
                    row.get(metric),
                    expected,
                    label=f"manifest.{branch_name}.seed_{seed}.{metric}",
                )
            diversity = _finite_number(
                row.get("diversity"),
                f"manifest.{branch_name}.seed_{seed}.diversity",
            )
            if not 0.0 <= diversity <= 1.0:
                raise RescoreValidationError("manifest diversity must lie in [0, 1]")
            for metric in values:
                values[metric].append(float(row[metric]))
            if branch_name == "released_comparable":
                _sha256_value(
                    row.get("raw_samples_sha256"),
                    f"manifest seed {seed} raw_samples_sha256",
                )
                _sha256_value(
                    row.get("summary_sha256"),
                    f"manifest seed {seed} summary_sha256",
                )
        expected_mean = {
            metric: statistics.fmean(series) for metric, series in values.items()
        }
        expected_sd = {
            metric: statistics.stdev(series) for metric, series in values.items()
        }
        _assert_exact_or_close(
            branch.get("mean"), expected_mean, label=f"manifest.{branch_name}.mean"
        )
        _assert_exact_or_close(
            branch.get("sample_sd"),
            expected_sd,
            label=f"manifest.{branch_name}.sample_sd",
        )
        validated_rows[branch_name] = rows
        metric_values[branch_name] = values

    funnel = _mapping(
        manifest.get("strict_vs_repaired_funnel"),
        "manifest.strict_vs_repaired_funnel",
    )
    expected_funnel_keys = {
        "requested",
        "strict_valid",
        "strict_unique_within_seed",
        "strict_quality",
        "released_valid",
        "released_unique_within_seed",
        "released_quality",
        "released_recovered_strict_failure",
        "released_largest_component_applied",
    }
    if set(funnel) != expected_funnel_keys:
        raise RescoreValidationError("manifest funnel fields are unexpected")
    for name, value in funnel.items():
        _integer(value, f"manifest.strict_vs_repaired_funnel.{name}", minimum=0)

    return {
        "document": dict(manifest),
        "rows": validated_rows,
        "values": metric_values,
        "source_aggregate": dict(source_aggregate),
        "checkpoint": dict(checkpoint),
        "funnel": dict(funnel),
    }


def validate_source_aggregate(
    aggregate: Mapping[str, Any], validated_manifest: Mapping[str, Any]
) -> None:
    """Bind the retained original aggregate to the compact frozen manifest."""

    if aggregate.get("schema_version") != 1 or aggregate.get("status") != "completed":
        raise RescoreValidationError("historical aggregate is not completed schema 1")
    if aggregate.get("runner_sha256") != EXPECTED_HISTORICAL_RUNNER_SHA256:
        raise RescoreValidationError("historical aggregate runner hash is unexpected")
    checkpoint = _mapping(aggregate.get("checkpoint"), "historical aggregate checkpoint")
    for key in ("sha256", "size_bytes", "global_step"):
        if checkpoint.get(key) != validated_manifest["checkpoint"].get(key):
            raise RescoreValidationError(
                f"historical aggregate checkpoint.{key} disagrees with manifest"
            )
    aggregate_metrics = _mapping(
        aggregate.get("aggregate_metrics"), "historical aggregate metrics"
    )
    manifest_document = validated_manifest["document"]
    for branch_name in ("released_comparable", "strict"):
        branch = _mapping(
            aggregate_metrics.get(branch_name),
            f"historical aggregate metrics.{branch_name}",
        )
        for metric in ("validity", "uniqueness", "quality", "diversity"):
            aggregate_metric = _mapping(
                branch.get(metric),
                f"historical aggregate metrics.{branch_name}.{metric}",
            )
            _assert_exact_or_close(
                aggregate_metric.get("mean"),
                manifest_document[branch_name]["mean"][metric],
                label=f"historical aggregate {branch_name}.{metric}.mean",
            )
            _assert_exact_or_close(
                aggregate_metric.get("sample_sd"),
                manifest_document[branch_name]["sample_sd"][metric],
                label=f"historical aggregate {branch_name}.{metric}.sample_sd",
            )


def parse_raw_rows(
    payload: bytes, *, expected_count: int, label: str
) -> list[dict[str, str]]:
    """Parse retained CSV bytes using the current ordered 21-field schema."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RescoreValidationError(f"{label} is not UTF-8") from error
    with io.StringIO(text, newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(benchmark.RAW_SAMPLE_FIELDS):
            raise RescoreValidationError(
                f"{label} has unexpected columns/order: {reader.fieldnames!r}"
            )
        rows = list(reader)
    if len(rows) != expected_count:
        raise RescoreValidationError(
            f"{label} has {len(rows)} rows; expected {expected_count}"
        )
    for expected_index, row in enumerate(rows):
        if None in row or set(row) != set(benchmark.RAW_SAMPLE_FIELDS):
            raise RescoreValidationError(f"{label} row {expected_index} is malformed")
        if row["sample_index"] != str(expected_index):
            raise RescoreValidationError(
                f"{label} sample_index is not ordered at row {expected_index}"
            )
    return rows


def _csv_boolean(value: str, *, optional: bool, label: str) -> bool | None:
    if optional and value == "":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    qualifier = "True, False, or empty" if optional else "True or False"
    raise RescoreValidationError(f"{label} must be {qualifier}; got {value!r}")


def normalize_historical_row(row: Mapping[str, str], *, row_index: int) -> dict[str, Any]:
    """Convert CSV spellings to the exact Python types emitted by the benchmark."""

    if tuple(row) != tuple(benchmark.RAW_SAMPLE_FIELDS):
        raise RescoreValidationError(f"legacy row {row_index} fields/order differ")
    normalized: dict[str, Any] = {}
    for field in benchmark.RAW_SAMPLE_FIELDS:
        value = row[field]
        label = f"legacy row {row_index}.{field}"
        if field == "sample_index":
            try:
                parsed_index = int(value)
            except ValueError as error:
                raise RescoreValidationError(f"{label} is not an integer") from error
            if value != str(parsed_index) or parsed_index != row_index:
                raise RescoreValidationError(f"{label} is not canonical/ordered")
            normalized[field] = parsed_index
        elif field == "raw_model_text":
            normalized[field] = value
        elif field in OPTIONAL_TEXT_FIELDS:
            normalized[field] = value or None
        elif field in NUMERIC_ROW_FIELDS:
            if value == "":
                normalized[field] = None
            else:
                try:
                    number = float(value)
                except ValueError as error:
                    raise RescoreValidationError(f"{label} is not numeric") from error
                if not math.isfinite(number):
                    raise RescoreValidationError(f"{label} is nonfinite")
                normalized[field] = number
        elif field in OPTIONAL_BOOLEAN_ROW_FIELDS:
            normalized[field] = _csv_boolean(value, optional=True, label=label)
        elif field in REQUIRED_BOOLEAN_ROW_FIELDS:
            normalized[field] = _csv_boolean(value, optional=False, label=label)
        else:  # pragma: no cover - guarded by the pinned 21-field schema
            raise RescoreValidationError(f"no type rule exists for {field}")
    return normalized


def _row_to_csv_strings(row: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for field in benchmark.RAW_SAMPLE_FIELDS:
        value = row[field]
        result[field] = "" if value is None else str(value)
    return result


def compare_raw_records(
    historical_rows: Sequence[Mapping[str, str]],
    rescored_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare every one of the 21 fields with field-appropriate semantics."""

    if len(historical_rows) != len(rescored_rows):
        raise RescoreValidationError(
            f"rescored row count {len(rescored_rows)} != historical {len(historical_rows)}"
        )
    field_results: list[dict[str, Any]] = []
    for field in benchmark.RAW_SAMPLE_FIELDS:
        numeric = field in NUMERIC_ROW_FIELDS
        max_absolute_difference: float | None = None
        for row_index, (historical_csv, rescored) in enumerate(
            zip(historical_rows, rescored_rows, strict=True)
        ):
            if set(rescored) != set(benchmark.RAW_SAMPLE_FIELDS):
                raise RescoreValidationError(
                    f"rescored row {row_index} does not contain exactly 21 fields"
                )
            historical = normalize_historical_row(
                historical_csv, row_index=row_index
            )[field]
            actual = rescored[field]
            label = f"row {row_index}.{field}"
            if numeric:
                if historical is None or actual is None:
                    if historical is not None or actual is not None:
                        raise RescoreValidationError(
                            f"{label} nullability differs: {actual!r} != {historical!r}"
                        )
                    continue
                actual_number = _finite_number(actual, label)
                difference = abs(actual_number - historical)
                max_absolute_difference = max(
                    difference,
                    max_absolute_difference if max_absolute_difference is not None else 0.0,
                )
                if difference > NUMERIC_ABSOLUTE_TOLERANCE:
                    raise RescoreValidationError(
                        f"{label} differs by {difference!r}, exceeding absolute "
                        f"tolerance {NUMERIC_ABSOLUTE_TOLERANCE}"
                    )
            elif type(actual) is not type(historical) or actual != historical:
                raise RescoreValidationError(
                    f"{label} differs exactly: {actual!r} != {historical!r}"
                )
        field_results.append(
            {
                "field": field,
                "comparison": (
                    "finite_numeric_absolute_tolerance_1e-12_or_exact_null"
                    if numeric
                    else "exact_value_and_type"
                ),
                "compared_rows": len(historical_rows),
                "mismatch_count": 0,
                "max_absolute_difference": (
                    max_absolute_difference if numeric else None
                ),
            }
        )
    return {
        "all_match": True,
        "row_count": len(historical_rows),
        "field_count": len(benchmark.RAW_SAMPLE_FIELDS),
        "cell_count": len(historical_rows) * len(benchmark.RAW_SAMPLE_FIELDS),
        "numeric_absolute_tolerance": NUMERIC_ABSOLUTE_TOLERANCE,
        "field_results": field_results,
    }


def _derived_failures(
    strict_counts: Mapping[str, int],
    released_counts: Mapping[str, int],
    cross_counts: Mapping[str, int],
    *,
    requested_count: int,
) -> dict[str, int]:
    return {
        **cross_counts,
        "strict_decode_failed": requested_count - strict_counts["valid_count"],
        "released_decode_failed": requested_count - released_counts["valid_count"],
        "strict_duplicates": strict_counts["valid_count"] - strict_counts["unique_count"],
        "released_duplicates": (
            released_counts["valid_count"] - released_counts["unique_count"]
        ),
    }


def validate_legacy_summary_and_rows(
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, str]],
    *,
    seed: int,
    raw_sha256: str,
) -> dict[str, Any]:
    """Validate schema-2 metadata without ever calling a schema-6 mutator/loader."""

    if summary.get("schema_version") != LEGACY_SUMMARY_SCHEMA_VERSION:
        raise RescoreValidationError(
            f"legacy seed {seed} summary must remain schema 2"
        )
    if summary.get("status") != "completed":
        raise RescoreValidationError(f"legacy seed {seed} summary is incomplete")
    if summary.get("seed") != seed or summary.get("num_samples") != len(rows):
        raise RescoreValidationError(f"legacy seed {seed} aliases disagree")
    run = _mapping(summary.get("run"), f"legacy seed {seed}.run")
    if (
        run.get("seed") != seed
        or run.get("requested_sample_count") != len(rows)
        or run.get("one_seed_per_invocation") is not True
        or run.get("single_generation_batch") is not True
    ):
        raise RescoreValidationError(f"legacy seed {seed} run contract disagrees")
    seed_config = _mapping(
        run.get("seed_configuration"), f"legacy seed {seed}.seed_configuration"
    )
    if seed_config.get("python_hash_seed") != str(seed):
        raise RescoreValidationError(
            f"legacy seed {seed} did not record matching PYTHONHASHSEED"
        )
    generation_protocol = _mapping(
        run.get("generation_protocol"), f"legacy seed {seed}.generation_protocol"
    )
    if generation_protocol.get("model_use_bracket_safe") is not False:
        raise RescoreValidationError(
            f"legacy seed {seed} unexpectedly used Bracket-SAFE"
        )

    artifacts = _mapping(summary.get("artifacts"), f"legacy seed {seed}.artifacts")
    raw_artifact = _mapping(
        artifacts.get("raw_samples_csv"),
        f"legacy seed {seed}.artifacts.raw_samples_csv",
    )
    if (
        raw_artifact.get("sha256") != raw_sha256
        or raw_artifact.get("row_count") != len(rows)
        or raw_artifact.get("fields") != list(benchmark.RAW_SAMPLE_FIELDS)
    ):
        raise RescoreValidationError(
            f"legacy seed {seed} raw artifact metadata disagrees with retained bytes"
        )
    git = _mapping(summary.get("git"), f"legacy seed {seed}.git")
    if git.get("runner_sha256") != EXPECTED_HISTORICAL_RUNNER_SHA256:
        raise RescoreValidationError(
            f"legacy seed {seed} historical runner hash is unexpected"
        )

    # These are intentionally the current report implementation's row and
    # metric validators, applied to the retained byte parse rather than a path
    # that could be swapped after hash verification.
    strict_counts = denovo_report._validate_branch_rows(  # noqa: SLF001
        rows, prefix="strict", seed=seed
    )
    released_counts = denovo_report._validate_branch_rows(  # noqa: SLF001
        rows, prefix="released", seed=seed
    )
    cross_counts = denovo_report._validate_cross_branch_rows(  # noqa: SLF001
        rows, seed=seed
    )
    metrics = _mapping(summary.get("metrics"), f"legacy seed {seed}.metrics")
    if set(metrics) != {"released_comparable", "strict"}:
        raise RescoreValidationError(f"legacy seed {seed} metric branches differ")
    denovo_report._validate_metric_branch(  # noqa: SLF001
        _mapping(metrics["strict"], f"legacy seed {seed}.metrics.strict"),
        strict_counts,
        seed=seed,
        branch_name="strict",
    )
    denovo_report._validate_metric_branch(  # noqa: SLF001
        _mapping(
            metrics["released_comparable"],
            f"legacy seed {seed}.metrics.released_comparable",
        ),
        released_counts,
        seed=seed,
        branch_name="released_comparable",
    )
    expected_failures = _derived_failures(
        strict_counts,
        released_counts,
        cross_counts,
        requested_count=len(rows),
    )
    _assert_exact_or_close(
        summary.get("failure_counts"),
        expected_failures,
        label=f"legacy seed {seed}.failure_counts",
    )
    return {
        "metrics": {name: dict(_mapping(value, name)) for name, value in metrics.items()},
        "failure_counts": expected_failures,
        "row_validator_counts": {
            "strict": dict(strict_counts),
            "released_comparable": dict(released_counts),
            "cross_branch": dict(cross_counts),
        },
    }


def _manifest_seed_entry(
    validated_manifest: Mapping[str, Any], branch_name: str, seed: int
) -> Mapping[str, Any]:
    return validated_manifest["rows"][branch_name][EXPECTED_SEEDS.index(seed)]


def _compare_seed_with_manifest(
    *,
    seed: int,
    metrics: Mapping[str, Any],
    raw_sha256: str,
    summary_sha256: str,
    validated_manifest: Mapping[str, Any],
) -> None:
    for branch_name in ("released_comparable", "strict"):
        metric = _mapping(metrics.get(branch_name), f"rescored {branch_name}")
        expected = _manifest_seed_entry(validated_manifest, branch_name, seed)
        actual = {
            "seed": seed,
            "requested": metric["validity_denominator"],
            "valid_count": metric["valid_count"],
            "unique_count": metric["unique_count"],
            "quality_count": metric["quality_count"],
            "validity": metric["validity"],
            "uniqueness": metric["uniqueness"],
            "quality": metric["quality"],
            "diversity": metric["diversity"],
        }
        expected_core = {key: expected[key] for key in actual}
        _assert_exact_or_close(
            actual,
            expected_core,
            label=f"rescored seed {seed} versus manifest.{branch_name}",
        )
    released_entry = _manifest_seed_entry(
        validated_manifest, "released_comparable", seed
    )
    if (
        released_entry.get("raw_samples_sha256") != raw_sha256
        or released_entry.get("summary_sha256") != summary_sha256
    ):
        raise RescoreValidationError(
            f"seed {seed} retained artifact hashes disagree with manifest"
        )


def rescore_retained_seed(
    *,
    seed: int,
    raw_artifact: RetainedArtifact,
    summary_artifact: RetainedArtifact,
    validated_manifest: Mapping[str, Any],
    decode_function: Callable[..., list[dict[str, Any]]],
    evaluate_function: Callable[..., tuple[dict[str, Any], dict[str, int]]],
    oracle_qed: Callable[[Sequence[str]], Any],
    oracle_sa: Callable[[Sequence[str]], Any],
    diversity_evaluator: Callable[[Sequence[str]], Any],
) -> dict[str, Any]:
    """Purely re-decode/re-score one pair of already retained artifact bytes."""

    manifest_entry = _manifest_seed_entry(
        validated_manifest, "released_comparable", seed
    )
    if raw_artifact.sha256 != manifest_entry.get("raw_samples_sha256"):
        raise RescoreValidationError(f"seed {seed} raw bytes are not manifest-pinned")
    if summary_artifact.sha256 != manifest_entry.get("summary_sha256"):
        raise RescoreValidationError(f"seed {seed} summary bytes are not manifest-pinned")

    rows = parse_raw_rows(
        raw_artifact.payload,
        expected_count=EXPECTED_SAMPLES_PER_SEED,
        label=f"seed {seed} raw_samples.csv",
    )
    summary = _mapping(
        strict_json_loads(
            summary_artifact.payload, label=f"seed {seed} summary.json"
        ),
        f"seed {seed} summary.json",
    )
    historical = validate_legacy_summary_and_rows(
        summary,
        rows,
        seed=seed,
        raw_sha256=raw_artifact.sha256,
    )

    decode_start = time.perf_counter()
    decoded = decode_function(
        [row["raw_model_text"] for row in rows], use_bracket_safe=False
    )
    decode_seconds = time.perf_counter() - decode_start
    scoring_start = time.perf_counter()
    rescored_metrics, rescored_failures = evaluate_function(
        decoded,
        requested_count=EXPECTED_SAMPLES_PER_SEED,
        oracle_qed=oracle_qed,
        oracle_sa=oracle_sa,
        diversity_evaluator=diversity_evaluator,
    )
    scoring_seconds = time.perf_counter() - scoring_start

    comparison = compare_raw_records(rows, decoded)
    rescored_csv_rows = [_row_to_csv_strings(row) for row in decoded]
    # Validate newly computed rows with exactly the same current validators too.
    rescored_strict_counts = denovo_report._validate_branch_rows(  # noqa: SLF001
        rescored_csv_rows, prefix="strict", seed=seed
    )
    rescored_released_counts = denovo_report._validate_branch_rows(  # noqa: SLF001
        rescored_csv_rows, prefix="released", seed=seed
    )
    rescored_cross_counts = denovo_report._validate_cross_branch_rows(  # noqa: SLF001
        rescored_csv_rows, seed=seed
    )
    rescored_derived_failures = _derived_failures(
        rescored_strict_counts,
        rescored_released_counts,
        rescored_cross_counts,
        requested_count=EXPECTED_SAMPLES_PER_SEED,
    )
    _assert_exact_or_close(
        rescored_failures,
        rescored_derived_failures,
        label=f"rescored seed {seed}.failure_counts versus current row validators",
    )
    _assert_exact_or_close(
        rescored_metrics,
        historical["metrics"],
        label=f"rescored seed {seed}.metrics versus schema-2 summary",
    )
    _assert_exact_or_close(
        rescored_failures,
        historical["failure_counts"],
        label=f"rescored seed {seed}.failure_counts versus schema-2 summary",
    )
    _compare_seed_with_manifest(
        seed=seed,
        metrics=rescored_metrics,
        raw_sha256=raw_artifact.sha256,
        summary_sha256=summary_artifact.sha256,
        validated_manifest=validated_manifest,
    )

    return {
        "seed": seed,
        "status": "exact_match",
        "inputs": {
            "raw_samples_csv": raw_artifact.provenance(),
            "summary_json": {
                **summary_artifact.provenance(),
                "schema_version": LEGACY_SUMMARY_SCHEMA_VERSION,
                "mutation_policy": "read_only_never_rewritten",
            },
        },
        "row_comparison": comparison,
        "current_report_validator_counts": historical["row_validator_counts"],
        "metrics": rescored_metrics,
        "failure_counts": rescored_failures,
        "manifest_comparison": {
            "all_counts_metrics_and_artifact_hashes_match": True,
        },
        "timings_seconds": {
            "decode_records": decode_seconds,
            "metric_evaluation": scoring_seconds,
            "decode_and_metrics": decode_seconds + scoring_seconds,
        },
    }


@contextmanager
def network_disabled() -> Iterator[None]:
    """Guard selected Python socket connection/name-resolution entry points."""

    def blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(
            "guarded Python network API is disabled for MDLM baseline rescoring"
        )

    previous_create_connection = socket.create_connection
    previous_getaddrinfo = socket.getaddrinfo
    previous_connect = socket.socket.connect
    previous_connect_ex = socket.socket.connect_ex
    socket.create_connection = blocked
    socket.getaddrinfo = blocked
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    try:
        yield
    finally:
        socket.socket.connect_ex = previous_connect_ex
        socket.socket.connect = previous_connect
        socket.getaddrinfo = previous_getaddrinfo
        socket.create_connection = previous_create_connection


def validate_worker_environment(seed: int) -> dict[str, Any]:
    """Require seed-at-start, CPU visibility controls, and offline settings."""

    if os.environ.get("PYTHONHASHSEED") != str(seed):
        raise RescoreValidationError(
            f"worker seed {seed} requires PYTHONHASHSEED={seed} at interpreter start"
        )
    for key, expected in OFFLINE_ENVIRONMENT.items():
        if os.environ.get(key) != expected:
            raise RescoreValidationError(
                f"worker seed {seed} requires {key}={expected!r}"
            )
    return {
        "python_hash_seed": str(seed),
        "device": "cpu",
        "cuda_visible_devices": "",
        "nvidia_visible_devices": "",
        "offline_environment": {
            key: os.environ[key]
            for key in OFFLINE_ENVIRONMENT
            if key not in {"CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"}
        },
        "python_network_guard_during_computation": {
            "guarded_apis": list(PYTHON_NETWORK_GUARDED_APIS),
            "scope_limitation": PYTHON_NETWORK_GUARD_LIMITATION,
        },
        "executable": sys.executable,
        "python": sys.version,
        "platform": platform.platform(),
        "pid": os.getpid(),
    }


def worker_environment(seed: int) -> dict[str, str]:
    """Build the complete deterministic/offline environment for one subprocess."""

    environment = os.environ.copy()
    environment.update(OFFLINE_ENVIRONMENT)
    environment.update(
        {
            "PYTHONHASHSEED": str(seed),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join(
                (str(REPOSITORY_SRC), str(REPOSITORY_ROOT))
            ),
        }
    )
    return environment


def _module_provenance(module_name: str) -> dict[str, Any]:
    module = importlib.import_module(module_name)
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RescoreValidationError(f"runtime module {module_name} has no file")
    artifact = read_stable_regular_file(
        Path(module_file),
        allowed_root=WORKSPACE_SCOPE_ROOT,
        label=f"runtime module {module_name}",
    )
    return {"module": module_name, **artifact.provenance()}


def runtime_module_provenance() -> dict[str, Any]:
    """Fingerprint decoding modules actually loaded by the worker."""

    result = {}
    for name in (
        "scripts.udlm.rescore_mdlm_baseline",
        "scripts.exps.denovo.benchmark",
        "scripts.exps.denovo.report",
        "genmol.utils.utils_chem",
        "safe",
        "safe.converter",
        "rdkit",
        "rdkit.Chem",
        "rdkit.Chem.rdchem",
        "rdkit.Chem.rdmolfiles",
        "rdkit.Chem.rdmolops",
    ):
        result[name] = _module_provenance(name)
    return result


def environment_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for label, candidates in {
        "numpy": ("numpy",),
        "rdkit": ("rdkit", "rdkit-pypi"),
        "safe": ("safe-mol", "safe"),
        "tdc": ("PyTDC", "pytdc"),
        "torch": ("torch",),
    }.items():
        value = None
        for candidate in candidates:
            try:
                value = importlib.metadata.version(candidate)
                break
            except importlib.metadata.PackageNotFoundError:
                continue
        versions[label] = value
    return versions


def _expected_source_hashes_from_args(args: argparse.Namespace) -> dict[str, str]:
    return {
        "rescore_runner": _sha256_value(
            args.expected_rescore_sha256, "expected rescore SHA-256"
        ),
        "benchmark_runner": _sha256_value(
            args.expected_benchmark_sha256, "expected benchmark SHA-256"
        ),
        "report_validator": _sha256_value(
            args.expected_report_sha256, "expected report SHA-256"
        ),
    }


def _verify_worker_sources(expected_hashes: Mapping[str, str]) -> None:
    paths = {
        "rescore_runner": REPOSITORY_ROOT / RESCORE_RELATIVE_PATH,
        "benchmark_runner": REPOSITORY_ROOT / BENCHMARK_RELATIVE_PATH,
        "report_validator": REPOSITORY_ROOT / REPORT_RELATIVE_PATH,
    }
    for name, path in paths.items():
        artifact = read_stable_regular_file(
            path,
            allowed_root=REPOSITORY_ROOT,
            label=name,
            expected_sha256=expected_hashes[name],
        )
        runtime_path = {
            "rescore_runner": Path(__file__).resolve(),
            "benchmark_runner": Path(benchmark.__file__).resolve(),
            "report_validator": Path(denovo_report.__file__).resolve(),
        }[name]
        if artifact.path != runtime_path:
            raise RescoreValidationError(
                f"runtime {name} path {runtime_path} != pinned {artifact.path}"
            )


def _seed_artifacts(
    historical_runs_dir: Path,
    *,
    seed: int,
    validated_manifest: Mapping[str, Any],
) -> tuple[RetainedArtifact, RetainedArtifact]:
    entry = _manifest_seed_entry(validated_manifest, "released_comparable", seed)
    seed_dir = historical_runs_dir / f"seed_{seed}"
    raw = read_stable_regular_file(
        seed_dir / benchmark.RAW_SAMPLES_FILENAME,
        allowed_root=WORKSPACE_SCOPE_ROOT,
        label=f"seed {seed} raw samples",
        expected_sha256=entry["raw_samples_sha256"],
    )
    summary = read_stable_regular_file(
        seed_dir / benchmark.SUMMARY_FILENAME,
        allowed_root=WORKSPACE_SCOPE_ROOT,
        label=f"seed {seed} summary",
        expected_sha256=entry["summary_sha256"],
    )
    return raw, summary


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    """Execute one seed in the interpreter whose hash seed matches that seed."""

    seed = args.worker_seed
    if seed not in EXPECTED_SEEDS:
        raise RescoreValidationError(f"worker seed must be one of {EXPECTED_SEEDS}")
    worker_start = time.perf_counter()
    environment = validate_worker_environment(seed)
    source_before = benchmark.require_clean_pushed_source(
        args.expected_source_revision
    )
    expected_source_hashes = _expected_source_hashes_from_args(args)
    _verify_worker_sources(expected_source_hashes)

    manifest_artifact = read_stable_regular_file(
        args.manifest,
        allowed_root=WORKSPACE_SCOPE_ROOT,
        label="frozen MDLM manifest",
        expected_sha256=EXPECTED_MANIFEST_SHA256,
    )
    validated_manifest = validate_manifest(
        _mapping(
            strict_json_loads(manifest_artifact.payload, label="frozen MDLM manifest"),
            "frozen MDLM manifest",
        )
    )
    raw_artifact, summary_artifact = _seed_artifacts(
        args.historical_runs_dir,
        seed=seed,
        validated_manifest=validated_manifest,
    )

    metric_input_start = time.perf_counter()
    sa_snapshot = benchmark.load_pinned_sa_metric_input()
    metric_input_seconds = time.perf_counter() - metric_input_start
    metric_inputs = dict(sa_snapshot.provenance)
    benchmark.assert_local_genmol_import()
    with network_disabled():
        from tdc import Evaluator, Oracle

        with benchmark.pinned_tdc_sa_oracle(sa_snapshot, Oracle) as sa_oracle:
            result = rescore_retained_seed(
                seed=seed,
                raw_artifact=raw_artifact,
                summary_artifact=summary_artifact,
                validated_manifest=validated_manifest,
                decode_function=benchmark.decode_records,
                evaluate_function=benchmark.evaluate_records,
                oracle_qed=Oracle("qed"),
                oracle_sa=sa_oracle,
                diversity_evaluator=Evaluator("diversity"),
            )
    benchmark.assert_runtime_tdc_metric_provenance(metric_inputs)
    modules = runtime_module_provenance()
    source_after = benchmark.require_clean_pushed_source(
        args.expected_source_revision
    )
    if source_after != source_before:
        raise RescoreValidationError("clean pushed source changed during worker run")
    _verify_worker_sources(expected_source_hashes)

    # Re-read the pinned paths after computation.  The computation itself used
    # only the retained bytes; this additionally proves the historical paths
    # still named the same pinned evidence at worker completion.
    final_raw, final_summary = _seed_artifacts(
        args.historical_runs_dir,
        seed=seed,
        validated_manifest=validated_manifest,
    )
    if (
        final_raw.sha256 != raw_artifact.sha256
        or final_summary.sha256 != summary_artifact.sha256
    ):
        raise RescoreValidationError(f"seed {seed} inputs changed during computation")

    result["environment"] = {
        **environment,
        "versions": environment_versions(),
    }
    result["source_verification"] = {
        "before": source_before,
        "after": source_after,
        "expected_file_sha256": dict(expected_source_hashes),
        "clean_pushed_before_and_after_computation": True,
    }
    result["runtime_modules"] = modules
    result["metric_inputs"] = metric_inputs
    result["metric_inputs_sha256"] = canonical_json_sha256(metric_inputs)
    result["timings_seconds"]["metric_input_load"] = metric_input_seconds
    result["timings_seconds"]["worker_total"] = time.perf_counter() - worker_start
    return result


def _source_file_provenance(
    path: Path, *, expected_revision: str, label: str
) -> dict[str, Any]:
    artifact = read_stable_regular_file(
        path,
        allowed_root=REPOSITORY_ROOT,
        label=label,
    )
    tracked = benchmark.tracked_source_file_provenance(
        artifact.path,
        expected_revision=expected_revision,
        expected_sha256=artifact.sha256,
    )
    return {
        **tracked,
        "size_bytes": artifact.size_bytes,
        "stable_bytes_verified": True,
    }


def source_provenance(expected_revision: str) -> dict[str, Any]:
    paths = {
        "rescore_runner": REPOSITORY_ROOT / RESCORE_RELATIVE_PATH,
        "benchmark_runner": REPOSITORY_ROOT / BENCHMARK_RELATIVE_PATH,
        "report_validator": REPOSITORY_ROOT / REPORT_RELATIVE_PATH,
        "chemistry_utils_module": REPOSITORY_ROOT / CHEMISTRY_RELATIVE_PATH,
    }
    return {
        name: _source_file_provenance(
            path, expected_revision=expected_revision, label=name
        )
        for name, path in paths.items()
    }


def _worker_command(
    args: argparse.Namespace,
    *,
    seed: int,
    source_files: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--manifest",
        str(args.manifest),
        "--historical-runs-dir",
        str(args.historical_runs_dir),
        "--expected-source-revision",
        args.expected_source_revision,
        "--worker-seed",
        str(seed),
        "--expected-rescore-sha256",
        source_files["rescore_runner"]["sha256"],
        "--expected-benchmark-sha256",
        source_files["benchmark_runner"]["sha256"],
        "--expected-report-sha256",
        source_files["report_validator"]["sha256"],
    ]


def _invoke_worker(
    args: argparse.Namespace,
    *,
    seed: int,
    source_files: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    command = _worker_command(args, seed=seed, source_files=source_files)
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=worker_environment(seed),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"seed {seed} rescore worker failed with status {completed.returncode}\n"
            f"stdout:\n{completed.stdout[-4000:]}\n"
            f"stderr:\n{completed.stderr[-8000:]}"
        )
    marker_lines = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith(WORKER_RESULT_PREFIX)
    ]
    if len(marker_lines) != 1:
        raise RuntimeError(
            f"seed {seed} worker emitted {len(marker_lines)} result records"
        )
    payload = marker_lines[0][len(WORKER_RESULT_PREFIX) :].encode("utf-8")
    result = _mapping(
        strict_json_loads(payload, label=f"seed {seed} worker result"),
        f"seed {seed} worker result",
    )
    if result.get("seed") != seed or result.get("status") != "exact_match":
        raise RuntimeError(f"seed {seed} worker returned an invalid completion record")
    return dict(result)


def _aggregate_rescore_metrics(seed_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for branch_name in ("released_comparable", "strict"):
        result[branch_name] = {}
        for metric in ("validity", "uniqueness", "quality", "diversity"):
            values = [float(row["metrics"][branch_name][metric]) for row in seed_results]
            result[branch_name][metric] = {
                "values_by_seed": [
                    {"seed": seed_row["seed"], "value": value}
                    for seed_row, value in zip(seed_results, values, strict=True)
                ],
                "mean": statistics.fmean(values),
                "sample_sd": statistics.stdev(values),
            }
    return result


def _aggregate_funnel(seed_results: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "requested": sum(
            row["metrics"]["strict"]["validity_denominator"] for row in seed_results
        ),
        "strict_valid": sum(
            row["metrics"]["strict"]["valid_count"] for row in seed_results
        ),
        "strict_unique_within_seed": sum(
            row["metrics"]["strict"]["unique_count"] for row in seed_results
        ),
        "strict_quality": sum(
            row["metrics"]["strict"]["quality_count"] for row in seed_results
        ),
        "released_valid": sum(
            row["metrics"]["released_comparable"]["valid_count"]
            for row in seed_results
        ),
        "released_unique_within_seed": sum(
            row["metrics"]["released_comparable"]["unique_count"]
            for row in seed_results
        ),
        "released_quality": sum(
            row["metrics"]["released_comparable"]["quality_count"]
            for row in seed_results
        ),
        "released_recovered_strict_failure": sum(
            row["failure_counts"]["released_recovered_strict_failure"]
            for row in seed_results
        ),
        "released_largest_component_applied": sum(
            row["failure_counts"]["released_largest_component_applied"]
            for row in seed_results
        ),
    }


def _compare_aggregate_with_manifest(
    aggregate: Mapping[str, Any],
    funnel: Mapping[str, int],
    validated_manifest: Mapping[str, Any],
) -> None:
    manifest = validated_manifest["document"]
    for branch_name in ("released_comparable", "strict"):
        for metric in ("validity", "uniqueness", "quality", "diversity"):
            _assert_exact_or_close(
                aggregate[branch_name][metric]["mean"],
                manifest[branch_name]["mean"][metric],
                label=f"rescored aggregate {branch_name}.{metric}.mean",
            )
            _assert_exact_or_close(
                aggregate[branch_name][metric]["sample_sd"],
                manifest[branch_name]["sample_sd"][metric],
                label=f"rescored aggregate {branch_name}.{metric}.sample_sd",
            )
    _assert_exact_or_close(
        funnel,
        validated_manifest["funnel"],
        label="rescored strict-versus-repaired funnel",
    )


def _validate_output_path(output_path: Path, historical_runs_dir: Path) -> Path:
    root = REPOSITORY_ROOT.resolve(strict=True)
    candidate = output_path if output_path.is_absolute() else REPOSITORY_ROOT / output_path
    candidate = Path(os.path.abspath(candidate))
    if not _is_within(candidate, root):
        raise RescoreValidationError("output path escapes the source repository")
    historical = historical_runs_dir.resolve(strict=True)
    if _is_within(candidate, historical):
        raise RescoreValidationError("output must not be placed inside legacy inputs")
    if candidate.suffix != ".json":
        raise RescoreValidationError("output path must end in .json")
    if candidate.exists() or candidate.is_symlink():
        raise FileExistsError(f"refusing to clobber existing output: {candidate}")
    return candidate


def atomic_exclusive_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish complete JSON atomically and fail if the destination exists."""

    root = REPOSITORY_ROOT.resolve(strict=True)
    absolute = Path(os.path.abspath(path))
    if absolute == root or not _is_within(absolute, root) or absolute.suffix != ".json":
        raise RescoreValidationError("output must be an in-repository JSON file")
    try:
        parent = absolute.parent.resolve(strict=True)
    except OSError as error:
        raise RescoreValidationError(
            "output parent must already exist as an in-repository directory"
        ) from error
    if parent != absolute.parent or not _is_within(parent, root) or not parent.is_dir():
        raise RescoreValidationError(
            "output parent must be a real in-repository directory without symlinks"
        )
    absolute = parent / absolute.name
    if os.path.lexists(absolute):
        raise FileExistsError(f"refusing to clobber existing output: {absolute}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{absolute.name}.", suffix=".tmp", dir=parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, absolute, follow_symlinks=False)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to clobber concurrently created output: {absolute}"
            ) from error
        directory_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary_path.unlink(missing_ok=True)


def _revalidate_all_inputs(
    *,
    manifest_path: Path,
    historical_runs_dir: Path,
    validated_manifest: Mapping[str, Any],
) -> None:
    read_stable_regular_file(
        manifest_path,
        allowed_root=WORKSPACE_SCOPE_ROOT,
        label="frozen MDLM manifest final recheck",
        expected_sha256=EXPECTED_MANIFEST_SHA256,
    )
    read_stable_regular_file(
        historical_runs_dir / "report/aggregate.json",
        allowed_root=WORKSPACE_SCOPE_ROOT,
        label="historical aggregate final recheck",
        expected_sha256=EXPECTED_SOURCE_AGGREGATE_SHA256,
    )
    for seed in EXPECTED_SEEDS:
        _seed_artifacts(
            historical_runs_dir,
            seed=seed,
            validated_manifest=validated_manifest,
        )


def run_parent(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    """Coordinate isolated seed workers and publish one no-clobber attestation."""

    total_start = time.perf_counter()
    args.manifest = args.manifest.absolute()
    args.historical_runs_dir = args.historical_runs_dir.absolute()
    output_path = _validate_output_path(args.output, args.historical_runs_dir)

    source_before = benchmark.require_clean_pushed_source(
        args.expected_source_revision
    )
    source_files = source_provenance(args.expected_source_revision)
    manifest_artifact = read_stable_regular_file(
        args.manifest,
        allowed_root=WORKSPACE_SCOPE_ROOT,
        label="frozen MDLM manifest",
        expected_sha256=EXPECTED_MANIFEST_SHA256,
    )
    validated_manifest = validate_manifest(
        _mapping(
            strict_json_loads(manifest_artifact.payload, label="frozen MDLM manifest"),
            "frozen MDLM manifest",
        )
    )
    aggregate_artifact = read_stable_regular_file(
        args.historical_runs_dir / "report/aggregate.json",
        allowed_root=WORKSPACE_SCOPE_ROOT,
        label="historical source aggregate",
        expected_sha256=validated_manifest["source_aggregate"]["sha256"],
    )
    source_aggregate = _mapping(
        strict_json_loads(
            aggregate_artifact.payload, label="historical source aggregate"
        ),
        "historical source aggregate",
    )
    validate_source_aggregate(source_aggregate, validated_manifest)

    worker_start = time.perf_counter()
    seed_results_by_seed: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                _invoke_worker,
                args,
                seed=seed,
                source_files=source_files,
            ): seed
            for seed in EXPECTED_SEEDS
        }
        for future in as_completed(futures):
            seed = futures[future]
            seed_results_by_seed[seed] = future.result()
    worker_wall_seconds = time.perf_counter() - worker_start
    seed_results = [seed_results_by_seed[seed] for seed in EXPECTED_SEEDS]

    metric_input_hashes = {row["metric_inputs_sha256"] for row in seed_results}
    if len(metric_input_hashes) != 1:
        raise RescoreValidationError("seed workers used different metric inputs")
    runtime_module_hashes = {
        canonical_json_sha256(row["runtime_modules"]) for row in seed_results
    }
    if len(runtime_module_hashes) != 1:
        raise RescoreValidationError("seed workers loaded different runtime modules")
    worker_source_hashes = {
        canonical_json_sha256(row["source_verification"]["expected_file_sha256"])
        for row in seed_results
    }
    if len(worker_source_hashes) != 1:
        raise RescoreValidationError("seed workers used different source hashes")

    aggregate_metrics = _aggregate_rescore_metrics(seed_results)
    aggregate_funnel = _aggregate_funnel(seed_results)
    _compare_aggregate_with_manifest(
        aggregate_metrics, aggregate_funnel, validated_manifest
    )

    source_after_computation = benchmark.require_clean_pushed_source(
        args.expected_source_revision
    )
    if source_after_computation != source_before:
        raise RescoreValidationError("source revision changed during parent computation")
    final_source_files = source_provenance(args.expected_source_revision)
    _assert_exact_or_close(
        final_source_files,
        source_files,
        label="source file provenance before versus after computation",
    )
    _revalidate_all_inputs(
        manifest_path=args.manifest,
        historical_runs_dir=args.historical_runs_dir,
        validated_manifest=validated_manifest,
    )

    metric_inputs = seed_results[0].pop("metric_inputs")
    runtime_modules = seed_results[0].pop("runtime_modules")
    for row in seed_results[1:]:
        row.pop("metric_inputs")
        row.pop("runtime_modules")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_exact_match",
        "purpose": (
            "CPU-only current-code re-decode and re-score attestation for the "
            "immutable legacy MDLM baseline; no molecules were regenerated"
        ),
        "created_at_utc": _utc_now(),
        "protocol": {
            "seeds": list(EXPECTED_SEEDS),
            "one_fresh_interpreter_per_seed": True,
            "python_hash_seed_equals_seed": True,
            "parent_max_parallel_workers": args.jobs,
            "device": "cpu",
            "cuda_visible_devices": "",
            "network_controls": {
                "offline_environment_variables": dict(OFFLINE_ENVIRONMENT),
                "python_runtime_guarded_apis": list(PYTHON_NETWORK_GUARDED_APIS),
                "os_or_process_network_isolation": False,
                "scope_limitation": PYTHON_NETWORK_GUARD_LIMITATION,
            },
            "decode_function": "benchmark.decode_records",
            "use_bracket_safe": False,
            "qed": "tdc.Oracle('qed')",
            "sa": "benchmark.pinned_tdc_sa_oracle over retained pinned SA bytes",
            "diversity": "tdc.Evaluator('diversity')",
            "row_fields_compared": list(benchmark.RAW_SAMPLE_FIELDS),
            "nonnumeric_and_count_comparison": "exact value and type",
            "numeric_comparison": "absolute tolerance 1e-12; relative tolerance 0",
            "legacy_summary_policy": "schema-2 input is read-only and never rewritten",
        },
        "source": {
            "repository_root": str(REPOSITORY_ROOT),
            "revision": args.expected_source_revision,
            "upstream": source_before["upstream"],
            "clean_pushed_checks": {
                "before_computation": True,
                "after_computation": True,
                "immediately_before_publication": True,
            },
            "files": source_files,
        },
        "inputs": {
            "historical_runs_dir": str(args.historical_runs_dir),
            "frozen_manifest": {
                **manifest_artifact.provenance(),
                "schema_version": 1,
            },
            "historical_source_aggregate": {
                **aggregate_artifact.provenance(),
                "schema_version": 1,
                "historical_runner_sha256": EXPECTED_HISTORICAL_RUNNER_SHA256,
            },
            "retention_and_mutation_policy": (
                "all legacy CSV/JSON and SA inputs are hashed from stable retained "
                "bytes; schema-2 summaries are never written"
            ),
        },
        "implementation": {
            "benchmark_schema_version": benchmark.SCHEMA_VERSION,
            "report_schema_version": denovo_report.REPORT_SCHEMA_VERSION,
            "raw_sample_field_count": len(benchmark.RAW_SAMPLE_FIELDS),
            "runtime_modules": runtime_modules,
            "runtime_modules_sha256": canonical_json_sha256(runtime_modules),
            "metric_inputs": metric_inputs,
            "metric_inputs_sha256": next(iter(metric_input_hashes)),
        },
        "seed_results": seed_results,
        "aggregate_metrics": aggregate_metrics,
        "strict_vs_repaired_funnel": aggregate_funnel,
        "manifest_comparison": {
            "all_seed_rows_metrics_failures_hashes_and_aggregates_match": True,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        },
        "environment": {
            "parent_executable": sys.executable,
            "parent_python": sys.version,
            "parent_platform": platform.platform(),
            "worker_policy": dict(OFFLINE_ENVIRONMENT),
        },
        "timings_seconds": {
            "worker_wall": worker_wall_seconds,
            "total_before_publication": time.perf_counter() - total_start,
        },
        "output": {
            "path": str(output_path),
            "publication": "temporary_fsync_then_atomic_hard_link_no_replace",
            "clobber_allowed": False,
        },
    }

    # Third source check: deliberately adjacent to the only publication call.
    source_before_publication = benchmark.require_clean_pushed_source(
        args.expected_source_revision
    )
    if source_before_publication != source_before:
        raise RescoreValidationError("source changed before attestation publication")
    atomic_exclusive_write_json(output_path, payload)
    return output_path, payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT / MANIFEST_RELATIVE_PATH,
        help="Frozen MDLM manifest (exact pinned bytes required).",
    )
    parser.add_argument(
        "--historical-runs-dir",
        type=Path,
        default=DEFAULT_HISTORICAL_RUNS_DIR,
        help="Legacy denovo_50000 directory containing seed_*/ and report/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="New schema-1 attestation JSON; existing paths are never replaced.",
    )
    parser.add_argument(
        "--expected-source-revision",
        required=True,
        help="Exact clean pushed Git revision containing this rescorer.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        choices=(1, 2, 3),
        default=3,
        help="Maximum concurrent seed subprocesses (each remains one interpreter).",
    )
    parser.add_argument("--worker-seed", type=int, choices=EXPECTED_SEEDS, help=argparse.SUPPRESS)
    parser.add_argument("--expected-rescore-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--expected-benchmark-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--expected-report-sha256", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker_seed is not None:
        missing = [
            name
            for name in (
                "expected_rescore_sha256",
                "expected_benchmark_sha256",
                "expected_report_sha256",
            )
            if getattr(args, name) is None
        ]
        if missing:
            raise RescoreValidationError(
                f"worker invocation lacks parent-pinned source hashes: {missing}"
            )
        result = run_worker(args)
        print(
            WORKER_RESULT_PREFIX
            + json.dumps(
                result,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        return 0
    output_path, _ = run_parent(args)
    print(f"Wrote immutable MDLM rescore attestation: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

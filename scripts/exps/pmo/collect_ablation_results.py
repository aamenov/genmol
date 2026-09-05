"""Validate and collect completed PMO ablation runs.

The input is one experiment tree containing ``<oracle>/<variant>/seed_<N>``.
Checkpoint and event data are authoritative: every reported metric is
recomputed before publishing.  Checkpoints use pickle and must only be loaded
from trusted local experiment directories.
"""

from __future__ import annotations

import argparse
import csv
import datetime as datetime_module
import fcntl
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.exps.pmo.main.genmol.experiment_io import (  # noqa: E402
    iter_events,
    load_checkpoint,
    sha256_config,
    sha256_file,
    summarize_indexed_scores,
    summarize_scores,
    write_manifest,
)
from scripts.exps.pmo import launch_ablation as ablation_launcher  # noqa: E402


COLLECTION_SCHEMA_VERSION = 3
SEED_DIRECTORY = re.compile(r"seed_(-?\d+)\Z")
SOURCE_FILES = {
    "manifest": Path("manifest.json"),
    "summary": Path("summary.json"),
    "events": Path("events.jsonl"),
    "checkpoint": Path("state/latest.pkl"),
}
METRIC_FIELDS = (
    "top_1",
    "top_10",
    "top_100",
    "auc_top_1",
    "auc_top_10",
    "auc_top_100",
)
CHILD_VIEWS = {
    "total": (
        "charged_children_total_call_axis",
        "child scores at their actual positions on the total oracle-call axis",
    ),
    "count": (
        "charged_children_child_count_axis",
        "dense child-score curve on the charged-child-count axis",
    ),
    "legacy": (
        "charged_children_only",
        "legacy dense child curve padded over the total budget; AUC is not globally aligned",
    ),
}
CSV_CONFIG_FIELDS = (
    "population_size",
    "warmup",
    "gamma",
    "softmax_temp",
    "randomness",
    "guidance_scale",
    "min_mol_size",
    "max_mol_size",
    "max_iterations",
    "checkpoint_every",
    "policy_mode",
    "min_support",
    "parent_control",
    "prior_mean",
    "prior_strength",
    "prior_mean_source",
    "delta_attribution",
    "legacy_seed_count",
    "legacy_warmup_off_by_one",
    "durable_events",
    "population_sampling_order",
)
CSV_FIELDS = (
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
    "child_views",
    "events",
    "iterations_completed",
    "elapsed_seconds",
    "reporting_frequency",
    *(f"all_{name}" for name in METRIC_FIELDS),
    *(
        f"child_{view}_{name}"
        for view in CHILD_VIEWS
        for name in ("score_count", "axis_budget", *METRIC_FIELDS)
    ),
    *CSV_CONFIG_FIELDS,
    "config_sha256",
    "model_path",
    "model_sha256",
    "vocabulary_path",
    "vocabulary_sha256",
    "scientific_status",
    "git_commit",
    "git_branch",
    "git_dirty",
    "tracked_diff_sha256",
    "hostname",
    "python_version",
    "cuda_visible_devices",
    "gpu_name",
    "gpu_logical_index",
    "torch_version",
    "rdkit_version",
    "resume_count",
    "launch_attempt_count",
    "launch_history_complete",
    "launch_policy_complete",
    "launch_matrix_provenance_complete",
    "launch_provenance_complete",
    "launch_gpu_migrated",
    "launch_any_gpu_sharing",
    "launch_wall_time_comparable",
    "durable_events_provenance_complete",
    "timing_provenance_complete",
    "launch_first_time_unix",
    "launch_last_time_unix",
    "launch_gpu_uuids",
    "launch_command_sha256s",
    "launch_logs",
    *(f"source_{name}_sha256" for name in SOURCE_FILES),
)

PAPER_GAMMA = {
    "albuterol_similarity": 0.2,
    "amlodipine_mpo": 0.3,
    "celecoxib_rediscovery": 0.0,
    "deco_hop": 0.2,
    "drd2": 0.0,
    "fexofenadine_mpo": 0.0,
    "gsk3b": 0.0,
    "isomers_c7h8n2o2": 0.5,
    "isomers_c9h10n2o2pf2cl": 0.0,
    "jnk3": 0.5,
    "median1": 0.2,
    "median2": 0.2,
    "mestranol_similarity": 0.0,
    "osimertinib_mpo": 0.0,
    "perindopril_mpo": 0.4,
    "qed": 0.0,
    "ranolazine_mpo": 0.0,
    "scaffold_hop": 0.0,
    "sitagliptin_mpo": 0.2,
    "thiothixene_rediscovery": 0.3,
    "troglitazone_rediscovery": 0.0,
    "valsartan_smarts": 0.4,
    "zaleplon_mpo": 0.4,
}
VARIANT_SETTINGS = {
    "released": ("released", 1, False),
    "running_mean": ("mean", 1, False),
    "support3": ("mean", 3, False),
    "shrink10": ("bayes", 1, False),
    "delta": ("delta", 1, True),
    "running_mean_parent_control": ("mean", 1, True),
}
SMALL_MOLECULE_ORACLES = {
    "albuterol_similarity",
    "isomers_c7h8n2o2",
    "isomers_c9h10n2o2pf2cl",
    "median1",
    "qed",
    "sitagliptin_mpo",
    "zaleplon_mpo",
}
LARGE_MOLECULE_ORACLES = {"gsk3b", "jnk3"}
VALUE_FLAGS = {
    "--oracle",
    "--variant",
    "--model-path",
    "--vocab-path",
    "--device",
    "--seed",
    "--max-oracle-calls",
    "--reporting-frequency",
    "--checkpoint-every",
    "--max-iterations",
    "--population-size",
    "--warmup",
    "--gamma",
    "--softmax-temp",
    "--randomness",
    "--guidance-scale",
    "--min-mol-size",
    "--max-mol-size",
    "--legacy-seed-count",
    "--prior-mean",
    "--prior-mean-source",
    "--delta-attribution",
    "--experiment-id",
    "--scientific-status",
    "--output-root",
    "--matrix-path",
    "--matrix-sha256",
}
BOOLEAN_FLAGS = {
    "--legacy-warmup-off-by-one",
    "--resume",
    "--durable-events",
}
GPU_POLICY_FIELDS = {
    "utilization_threshold",
    "min_free_memory_mib",
    "sharing_authorized",
    "sharing_actual",
    "wall_time_comparable",
}


class CollectionError(ValueError):
    """A run cannot safely be represented in the result collection."""


@dataclass(frozen=True)
class LaunchRecord:
    path: Path
    sha256: str
    line_number: int
    record: dict[str, Any]


@dataclass(frozen=True)
class CollectedRun:
    run_dir: Path
    row: dict[str, Any]
    config: dict[str, Any]
    metrics: dict[str, dict[str, Any]]
    sources: dict[str, dict[str, str]]
    launch_attempts: tuple[LaunchRecord, ...]


@dataclass(frozen=True)
class MatrixPlan:
    path: Path
    sha256: str
    matrix: dict[str, Any]
    jobs: dict[tuple[str, str, int], Any]


class FileLock:
    """Non-blocking advisory lock with deterministic cleanup."""

    def __init__(self, path: Path, operation: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._handle = path.open("a+b")
        try:
            fcntl.flock(self._handle.fileno(), operation | fcntl.LOCK_NB)
        except BaseException:
            self._handle.close()
            raise

    def close(self) -> None:
        if self._handle.closed:
            return
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()

    def __enter__(self) -> FileLock:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _utc_timestamp() -> str:
    now = datetime_module.datetime.now(datetime_module.timezone.utc)
    return now.isoformat(timespec="seconds").replace("+00:00", "Z")


def _lexical_absolute(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(Path(path).expanduser()))


def _trusted_existing_path(
    path: str | os.PathLike[str],
    label: str,
    *,
    kind: str,
    root: Path | None = None,
) -> Path:
    lexical = _lexical_absolute(path)
    for component in (lexical, *lexical.parents):
        if component.is_symlink():
            raise CollectionError(f"{label} must not contain symlinks: {component}")
    if kind == "file" and not lexical.is_file():
        raise CollectionError(f"{label} is not a file: {lexical}")
    if kind == "directory" and not lexical.is_dir():
        raise CollectionError(f"{label} is not a directory: {lexical}")
    resolved = lexical.resolve(strict=True)
    if root is not None:
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise CollectionError(
                f"{label} escapes experiment root {root}: {resolved}"
            ) from error
    return resolved


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value}")


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise CollectionError(f"invalid {label} at {path}: {error}") from error
    if not isinstance(value, dict):
        raise CollectionError(f"{label} at {path} must contain a JSON object")
    return value


def _mapping(container: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    value = container.get(key)
    if not isinstance(value, Mapping):
        raise CollectionError(f"{context}.{key} must be an object")
    return value


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CollectionError(f"{context} must be an integer >= {minimum}")
    return value


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CollectionError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise CollectionError(f"{context} must be a finite number")
    return result


def _nonnegative_number(value: Any, context: str) -> float:
    result = _number(value, context)
    if result < 0:
        raise CollectionError(f"{context} must be nonnegative")
    return result


def _boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise CollectionError(f"{context} must be a boolean")
    return value


def _equal(actual: Any, expected: Any, context: str) -> None:
    if actual != expected:
        raise CollectionError(f"{context} is {actual!r}, expected {expected!r}")


def _metric_equal(actual: Any, expected: Any, context: str) -> None:
    if actual != expected:
        raise CollectionError(
            f"{context} differ from recomputation "
            f"(stored_sha256={sha256_config(actual)}, "
            f"recomputed_sha256={sha256_config(expected)})"
        )


def _parse_launch_command(command: Any) -> dict[str, Any]:
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise CollectionError("launch command must be a list of strings")
    if len(command) < 2:
        raise CollectionError("launch command is missing its executable or script")
    parsed: dict[str, Any] = {"executable": command[0], "script": command[1]}
    index = 2
    while index < len(command):
        flag = command[index]
        if flag in parsed:
            raise CollectionError(f"launch command repeats {flag}")
        if flag in BOOLEAN_FLAGS:
            parsed[flag] = True
            index += 1
            continue
        if flag not in VALUE_FLAGS:
            raise CollectionError(f"launch command contains unknown argument {flag!r}")
        if index + 1 >= len(command) or command[index + 1].startswith("--"):
            raise CollectionError(f"launch command has no value for {flag}")
        parsed[flag] = command[index + 1]
        index += 2
    return parsed


def _launch_identity(record: Mapping[str, Any]) -> tuple[str, str, str, int] | None:
    if record.get("event") != "launch":
        return None
    command = record.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        return None
    try:
        values = {}
        for flag in ("--experiment-id", "--oracle", "--variant", "--seed"):
            positions = [index for index, item in enumerate(command) if item == flag]
            if len(positions) != 1 or positions[0] + 1 >= len(command):
                return None
            values[flag] = command[positions[0] + 1]
        return (
            values["--experiment-id"],
            values["--oracle"],
            values["--variant"],
            int(values["--seed"]),
        )
    except ValueError:
        return None


def load_launch_records(
    logs_dir: str | os.PathLike[str],
) -> dict[tuple[str, str, str, int], tuple[LaunchRecord, ...]]:
    """Index every valid JSON launch record from append-mode log files."""

    root = _trusted_existing_path(logs_dir, "launch log directory", kind="directory")
    indexed: dict[tuple[str, str, str, int], list[LaunchRecord]] = {}
    for candidate in sorted(root.iterdir()):
        if candidate.is_symlink():
            raise CollectionError(f"launch log source must not be a symlink: {candidate}")
        if not candidate.is_file():
            continue
        path = _trusted_existing_path(candidate, "launch log source", kind="file", root=root)
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise CollectionError(f"could not read launch log {path}: {error}") from error
        source_hash = hashlib.sha256(payload).hexdigest()
        lines = payload.splitlines()
        for line_number, raw_line in enumerate(lines, start=1):
            try:
                record = json.loads(
                    raw_line.decode("utf-8"),
                    parse_constant=_reject_json_constant,
                )
            except (UnicodeError, ValueError):
                continue
            if not isinstance(record, dict):
                continue
            identity = _launch_identity(record)
            if identity is not None:
                indexed.setdefault(identity, []).append(
                    LaunchRecord(path, source_hash, line_number, record)
                )

    def sort_key(attempt: LaunchRecord) -> tuple[float, str, int]:
        raw_time = attempt.record.get("time_unix")
        time_value = float(raw_time) if isinstance(raw_time, (int, float)) else -math.inf
        return time_value, str(attempt.path), attempt.line_number

    return {
        identity: tuple(sorted(attempts, key=sort_key))
        for identity, attempts in indexed.items()
    }


def _optional(parsed: Mapping[str, Any], flag: str, converter: type, default: Any) -> Any:
    if flag not in parsed:
        return default
    try:
        return converter(parsed[flag])
    except (TypeError, ValueError) as error:
        raise CollectionError(f"invalid {flag} value in launch command") from error


def _resolved_launch_config(parsed: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "--oracle",
        "--variant",
        "--model-path",
        "--experiment-id",
        "--scientific-status",
        "--output-root",
    )
    missing = [flag for flag in required if flag not in parsed]
    if missing:
        raise CollectionError(f"launch command is missing required flags: {missing}")
    oracle = str(parsed["--oracle"])
    variant = str(parsed["--variant"])
    if oracle not in PAPER_GAMMA or variant not in VARIANT_SETTINGS:
        raise CollectionError("launch command names an unsupported oracle or variant")
    model_path = Path(str(parsed["--model-path"])).expanduser().resolve()
    raw_vocab = parsed.get("--vocab-path")
    vocab_path = (
        Path(str(raw_vocab)).expanduser().resolve()
        if raw_vocab is not None
        else REPOSITORY_ROOT / "scripts" / "exps" / "pmo" / "vocab" / f"{oracle}.csv"
    )
    raw_min = _optional(parsed, "--min-mol-size", int, 20)
    raw_max = _optional(parsed, "--max-mol-size", int, 40)
    if oracle in SMALL_MOLECULE_ORACLES:
        min_size, max_size = 10, 30
    elif oracle in LARGE_MOLECULE_ORACLES:
        min_size, max_size = 30, 80
    else:
        min_size, max_size = raw_min, raw_max
    mode, min_support, parent_control = VARIANT_SETTINGS[variant]
    resolved = {
        "experiment_id": str(parsed["--experiment-id"]),
        "scientific_status": str(parsed["--scientific-status"]),
        "oracle": oracle,
        "variant": variant,
        "policy_mode": mode,
        "parent_control": parent_control,
        "model_path": str(model_path),
        "vocab_path": str(vocab_path),
        "device": _optional(parsed, "--device", str, "cuda:0"),
        "seed": _optional(parsed, "--seed", int, 0),
        "max_oracle_calls": _optional(parsed, "--max-oracle-calls", int, 10_000),
        "reporting_frequency": _optional(parsed, "--reporting-frequency", int, 100),
        "checkpoint_every": _optional(parsed, "--checkpoint-every", int, 100),
        "max_iterations": _optional(parsed, "--max-iterations", int, 30_000),
        "population_size": _optional(parsed, "--population-size", int, 100),
        "warmup": _optional(parsed, "--warmup", int, 1_000),
        "legacy_warmup_off_by_one": bool(
            parsed.get("--legacy-warmup-off-by-one", False)
        ),
        "gamma": _optional(parsed, "--gamma", float, PAPER_GAMMA[oracle]),
        "softmax_temp": _optional(parsed, "--softmax-temp", float, 1.2),
        "randomness": _optional(parsed, "--randomness", float, 2.0),
        "guidance_scale": _optional(parsed, "--guidance-scale", float, 2.0),
        "min_mol_size": min_size,
        "max_mol_size": max_size,
        "min_support": min_support,
        "prior_strength": 10.0 if variant == "shrink10" else 0.0,
        "prior_mean": _optional(parsed, "--prior-mean", float, None),
        "prior_mean_source": _optional(parsed, "--prior-mean-source", str, None),
        "legacy_seed_count": _optional(parsed, "--legacy-seed-count", int, None),
        "delta_attribution": _optional(
            parsed,
            "--delta-attribution",
            str,
            "novel_vs_parent",
        ),
        "statistical_duplicate_policy": "one update per unique canonical child",
        "released_duplicate_policy": "repeat cached-child decomposition, matching release",
    }
    matrix_flags = {"--matrix-path", "--matrix-sha256"} & parsed.keys()
    if matrix_flags:
        if matrix_flags != {"--matrix-path", "--matrix-sha256"}:
            raise CollectionError("launch command must provide both matrix provenance flags")
        matrix_path = _trusted_existing_path(
            str(parsed["--matrix-path"]), "launch matrix path", kind="file"
        )
        matrix_sha256 = str(parsed["--matrix-sha256"])
        if not re.fullmatch(r"[0-9a-f]{64}", matrix_sha256):
            raise CollectionError("launch --matrix-sha256 must be lowercase hexadecimal")
        try:
            actual_matrix_hash = sha256_file(matrix_path)
        except OSError as error:
            raise CollectionError(f"cannot hash launch matrix {matrix_path}: {error}") from error
        _equal(actual_matrix_hash, matrix_sha256, "launch matrix SHA256")
        resolved.update(
            {"matrix_path": str(matrix_path), "matrix_sha256": matrix_sha256}
        )
    return resolved


def _validate_gpu_snapshot(record: Mapping[str, Any]) -> tuple[Mapping[str, Any], bool]:
    gpu = _mapping(record, "physical_gpu", "launch")
    _integer(gpu.get("index"), "launch physical_gpu.index")
    uuid = gpu.get("uuid")
    if not isinstance(uuid, str) or not uuid:
        raise CollectionError("launch physical_gpu.uuid must be a nonempty string")
    total = _integer(
        gpu.get("memory_total_mib"), "launch physical_gpu.memory_total_mib", minimum=1
    )
    used = _integer(gpu.get("memory_used_mib"), "launch physical_gpu.memory_used_mib")
    utilization = _integer(
        gpu.get("utilization_percent"), "launch physical_gpu.utilization_percent"
    )
    if used > total:
        raise CollectionError("launch physical GPU used memory exceeds total memory")
    if utilization > 100:
        raise CollectionError("launch physical GPU utilization exceeds 100 percent")

    processes = record.get("compute_processes")
    if not isinstance(processes, list):
        raise CollectionError("launch compute_processes must be a list")
    for index, process in enumerate(processes):
        if not isinstance(process, Mapping):
            raise CollectionError(f"launch compute_processes[{index}] must be an object")
        _equal(process.get("gpu_uuid"), uuid, f"launch compute_processes[{index}] GPU")
        _integer(process.get("pid"), f"launch compute_processes[{index}].pid", minimum=1)
        name = process.get("process_name")
        if not isinstance(name, str) or not name:
            raise CollectionError(
                f"launch compute_processes[{index}].process_name must be nonempty"
            )
        _integer(
            process.get("used_memory_mib"),
            f"launch compute_processes[{index}].used_memory_mib",
        )

    present = GPU_POLICY_FIELDS & record.keys()
    if present and present != GPU_POLICY_FIELDS:
        missing = sorted(GPU_POLICY_FIELDS - present)
        raise CollectionError(f"launch GPU policy snapshot is missing fields: {missing}")
    legacy_authorized = record.get("shared_gpu_authorized")
    if "shared_gpu_authorized" in record:
        legacy_authorized = _boolean(
            legacy_authorized, "launch shared_gpu_authorized"
        )
    if not present:
        if processes and legacy_authorized is False:
            raise CollectionError("launch shared a GPU without recorded authorization")
        return gpu, False

    threshold = _integer(
        record.get("utilization_threshold"), "launch utilization_threshold", minimum=1
    )
    if threshold > 100:
        raise CollectionError("launch utilization_threshold exceeds 100 percent")
    minimum_free = _integer(
        record.get("min_free_memory_mib"), "launch min_free_memory_mib"
    )
    authorized = _boolean(record.get("sharing_authorized"), "launch sharing_authorized")
    actual = _boolean(record.get("sharing_actual"), "launch sharing_actual")
    comparable = _boolean(
        record.get("wall_time_comparable"), "launch wall_time_comparable"
    )
    if utilization >= threshold:
        raise CollectionError("launch GPU utilization did not satisfy its threshold")
    if total - used < minimum_free:
        raise CollectionError("launch GPU free memory did not satisfy its minimum")
    _equal(actual, bool(processes), "launch sharing_actual")
    if actual and not authorized:
        raise CollectionError("launch shared a GPU without authorization")
    _equal(comparable, not actual, "launch wall_time_comparable")
    if "shared_gpu_authorized" in record:
        _equal(legacy_authorized, authorized, "launch sharing authorization aliases")
    return gpu, True


def _validate_matrix_command(
    command: Sequence[str],
    parsed: Mapping[str, Any],
    identity: tuple[str, str, str, int],
) -> bool:
    if "--matrix-path" not in parsed:
        return False
    matrix_path = _trusted_existing_path(
        parsed["--matrix-path"], "launch matrix path", kind="file"
    )
    matrix_sha256 = str(parsed["--matrix-sha256"])
    try:
        matrix = ablation_launcher._load_matrix(matrix_path)
        matches = [
            job
            for job in ablation_launcher._jobs(matrix)
            if (job.oracle, job.variant, job.seed) == identity[1:]
        ]
        if len(matches) != 1:
            raise CollectionError(
                f"launch matrix contains {len(matches)} matching jobs; expected one"
            )
        expected = ablation_launcher._command(
            matrix_path, matrix_sha256, matrix, matches[0]
        )
    except CollectionError:
        raise
    except Exception as error:
        raise CollectionError(f"cannot reconstruct launch from matrix: {error}") from error

    def normalized(items: Sequence[str]) -> list[str]:
        return [item for index, item in enumerate(items) if index and item != "--resume"]

    _equal(normalized(command), normalized(expected), "launch command versus matrix")
    return True


def _load_matrix_plan(
    matrix_path: str | os.PathLike[str], experiment_root: Path
) -> MatrixPlan:
    path = _trusted_existing_path(matrix_path, "matrix path", kind="file")
    digest = sha256_file(path)
    try:
        matrix = ablation_launcher._load_matrix(path)
        enumerated = ablation_launcher._jobs(matrix)
    except Exception as error:
        raise CollectionError(f"invalid experiment matrix {path}: {error}") from error
    jobs: dict[tuple[str, str, int], Any] = {}
    for job in enumerated:
        identity = (job.oracle, job.variant, job.seed)
        if identity in jobs:
            raise CollectionError(f"matrix repeats job identity {identity!r}")
        expected_dir = experiment_root / job.oracle / job.variant / f"seed_{job.seed}"
        generated_dir = _lexical_absolute(ablation_launcher._run_dir(matrix, job))
        _equal(generated_dir, expected_dir, f"matrix output directory for {identity!r}")
        jobs[identity] = job
    if not jobs:
        raise CollectionError("matrix does not enumerate any jobs")
    return MatrixPlan(path, digest, matrix, jobs)


def _validate_run_matrix(collected: CollectedRun, plan: MatrixPlan) -> bool:
    row = collected.row
    identity = (row["oracle"], row["variant"], row["seed"])
    job = plan.jobs.get(identity)
    if job is None:
        raise CollectionError(f"run {identity!r} is not enumerated by the matrix")
    config = collected.config
    recorded_path = config.get("matrix_path")
    recorded_hash = config.get("matrix_sha256")
    if (recorded_path is None) != (recorded_hash is None):
        raise CollectionError(f"run {identity!r} has partial matrix provenance")
    recorded = recorded_path is not None
    if recorded:
        path = _trusted_existing_path(recorded_path, "run matrix path", kind="file")
        _equal(path, plan.path, f"run {identity!r} matrix path")
        _equal(recorded_hash, plan.sha256, f"run {identity!r} matrix SHA256")

    try:
        command = ablation_launcher._command(plan.path, plan.sha256, plan.matrix, job)
        parsed = _parse_launch_command(command)
        expected = _resolved_launch_config(parsed)
    except Exception as error:
        raise CollectionError(
            f"cannot reconstruct matrix config for run {identity!r}: {error}"
        ) from error
    for key, value in expected.items():
        if key not in {"matrix_path", "matrix_sha256"}:
            _equal(config.get(key), value, f"run {identity!r} matrix config {key}")
    if "durable_events" in config:
        _equal(
            config["durable_events"],
            "--durable-events" in parsed,
            f"run {identity!r} matrix config durable_events",
        )
    return recorded


def _validate_launch(
    attempt: LaunchRecord,
    *,
    identity: tuple[str, str, str, int],
    config: Mapping[str, Any],
    manifest_runtime: Mapping[str, Any],
    expected_output_root: Path,
) -> tuple[dict[str, Any], bool, bool]:
    record = attempt.record
    parsed = _parse_launch_command(record.get("command"))
    _equal(_launch_identity(record), identity, "launch identity")
    _equal(record.get("job"), f"{identity[1]}__{identity[2]}__seed{identity[3]}", "launch job")
    _nonnegative_number(record.get("time_unix"), "launch time_unix")
    _, policy_complete = _validate_gpu_snapshot(record)
    _equal(
        Path(parsed["executable"]).expanduser().resolve(),
        Path(str(manifest_runtime.get("executable"))).expanduser().resolve(),
        "launch executable",
    )
    _equal(
        Path(parsed["script"]).expanduser().resolve(),
        REPOSITORY_ROOT / "scripts" / "exps" / "pmo" / "run_ablation.py",
        "launch runner script",
    )
    _equal(
        Path(parsed["--output-root"]).expanduser().resolve(),
        expected_output_root,
        "launch output root",
    )
    resolved = _resolved_launch_config(parsed)
    if "population_sampling_order" in config:
        resolved["population_sampling_order"] = (
            "canonical fragment string before uniform sampling"
        )
    for key, launched_value in resolved.items():
        _equal(launched_value, config.get(key), f"launch resolved config {key}")
    if "durable_events" in config:
        _equal(
            "--durable-events" in parsed,
            config["durable_events"],
            "launch resolved config durable_events",
        )
    matrix_complete = _validate_matrix_command(record["command"], parsed, identity)
    return parsed, policy_complete, matrix_complete


def _checkpoint_records(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    event_count: int,
    budget: int,
) -> tuple[dict[int, tuple[str, float]], float]:
    try:
        state, metadata = load_checkpoint(path, with_metadata=True)
    except Exception as error:
        raise CollectionError(f"invalid checkpoint at {path}: {error}") from error
    if not isinstance(state, Mapping) or not isinstance(metadata, Mapping):
        raise CollectionError(f"checkpoint at {path} must contain mapping state and metadata")
    _equal(state.get("event_count"), event_count, "checkpoint state event_count")
    _equal(
        state.get("next_iteration"),
        summary.get("iterations_completed"),
        "checkpoint state next_iteration",
    )
    checkpoint_elapsed = _nonnegative_number(
        state.get("elapsed_seconds"), "checkpoint state elapsed_seconds"
    )
    oracle_state = _mapping(state, "oracle", "checkpoint state")
    _equal(oracle_state.get("budget"), budget, "checkpoint oracle budget")
    buffer = _mapping(oracle_state, "buffer", "checkpoint state.oracle")
    _equal(len(buffer), budget, "checkpoint oracle buffer size")
    records: dict[int, tuple[str, float]] = {}
    for canonical, entry in buffer.items():
        if not isinstance(canonical, str) or not canonical:
            raise CollectionError("checkpoint oracle keys must be canonical SMILES strings")
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise CollectionError("checkpoint oracle entries must be [score, call_index]")
        score = _number(entry[0], f"checkpoint score for {canonical}")
        call_index = _integer(entry[1], "checkpoint oracle call index", minimum=1)
        if call_index in records:
            raise CollectionError(f"duplicate checkpoint oracle call index {call_index}")
        records[call_index] = (canonical, score)
    _equal(sorted(records), list(range(1, budget + 1)), "checkpoint oracle call indices")

    extra = _mapping(manifest, "extra", "manifest")
    git = _mapping(extra, "git", "manifest.extra")
    vocabulary = _mapping(extra, "vocabulary", "manifest.extra")
    expected_metadata = {
        "config_sha256": manifest.get("config_sha256"),
        "model_sha256": _mapping(manifest, "model", "manifest").get("sha256"),
        "vocabulary_sha256": vocabulary.get("sha256"),
        "git_commit": git.get("commit"),
        "tracked_diff_sha256": git.get("tracked_diff_sha256"),
    }
    for key, expected in expected_metadata.items():
        _equal(metadata.get(key), expected, f"checkpoint metadata {key}")
    return records, checkpoint_elapsed


def _score_outcome(
    observation: Mapping[str, Any],
    *,
    context: str,
    replayed: Mapping[int, tuple[str, float]],
    charged_canonicals: set[str],
    checkpoint_records: Mapping[int, tuple[str, float]],
    cumulative_calls: int,
    budget: int,
) -> tuple[bool, int | None, float | None]:
    outcome_fields = {
        "raw_smiles",
        "canonical_smiles",
        "valid",
        "score",
        "charged",
        "call_index",
        "reason",
    }
    _equal(set(observation), outcome_fields, f"{context} ScoreOutcome fields")
    raw = observation.get("raw_smiles")
    canonical = observation.get("canonical_smiles")
    valid = observation.get("valid")
    score = observation.get("score")
    charged = observation.get("charged")
    call_index = observation.get("call_index")
    reason = observation.get("reason")
    if raw is not None and not isinstance(raw, str):
        raise CollectionError(f"{context}.raw_smiles must be a string or null")
    if not isinstance(valid, bool) or not isinstance(charged, bool):
        raise CollectionError(f"{context}.valid and .charged must be booleans")
    if not isinstance(reason, str):
        raise CollectionError(f"{context}.reason must be a string")

    if reason == "missing_smiles":
        expected = (None, False, None, False, None)
        _equal(
            (canonical, valid, score, charged, call_index),
            expected,
            f"{context} missing-smiles outcome",
        )
        _equal(raw, None, f"{context} missing-smiles raw_smiles")
        return False, None, None
    if reason == "invalid_smiles":
        if not isinstance(raw, str):
            raise CollectionError(f"{context} invalid-smiles raw_smiles must be a string")
        expected = (None, False, None, False, None)
        _equal(
            (canonical, valid, score, charged, call_index),
            expected,
            f"{context} invalid-smiles outcome",
        )
        return False, None, None
    if not isinstance(raw, str) or not raw:
        raise CollectionError(f"{context}.raw_smiles must be nonempty for {reason!r}")
    if not isinstance(canonical, str) or not canonical:
        raise CollectionError(
            f"{context}.canonical_smiles must be nonempty for {reason!r}"
        )
    if reason == "budget_exhausted":
        _equal(
            (valid, score, charged, call_index),
            (True, None, False, None),
            f"{context} budget-exhausted outcome",
        )
        _equal(cumulative_calls, budget, f"{context} budget-exhausted call count")
        return False, None, None
    if reason == "cache_hit":
        _equal(valid, True, f"{context} cache-hit valid")
        _equal(charged, False, f"{context} cache-hit charged")
        index = _integer(call_index, f"{context}.call_index", minimum=1)
        value = _number(score, f"{context}.score")
        expected = replayed.get(index)
        if expected is None:
            raise CollectionError(f"{context} cache hit references a future call {index}")
        _equal((canonical, value), expected, f"{context} cache replay")
        _equal((canonical, value), checkpoint_records.get(index), f"{context} checkpoint")
        return False, index, value
    if reason == "scored":
        _equal(valid, True, f"{context} scored valid")
        _equal(charged, True, f"{context} scored charged")
        index = _integer(call_index, f"{context}.call_index", minimum=1)
        _equal(index, cumulative_calls + 1, f"{context} chronological call index")
        if index > budget:
            raise CollectionError(f"{context} exceeds oracle budget {budget}")
        value = _number(score, f"{context}.score")
        _equal((canonical, value), checkpoint_records.get(index), f"{context} checkpoint")
        if canonical in charged_canonicals:
            raise CollectionError(f"{context} recharges canonical molecule {canonical!r}")
        return True, index, value
    raise CollectionError(f"{context} has unsupported ScoreOutcome reason {reason!r}")


def _event_records(
    path: Path,
    *,
    checkpoint_records: Mapping[int, tuple[str, float]],
    expected_events: int,
    budget: int,
    summary_schema: int,
) -> tuple[list[tuple[int, float]], bool, float | None]:
    charged_calls: list[int] = []
    children: list[tuple[int, float]] = []
    cumulative_calls = 0
    observed_events = 0
    replayed: dict[int, tuple[str, float]] = {}
    charged_canonicals: set[str] = set()
    elapsed_values: list[float] = []
    timing_complete = True
    for event_index, event in enumerate(iter_events(path)):
        observed_events += 1
        _equal(event.get("event_index"), event_index, f"event {event_index} event_index")
        _equal(event.get("iteration"), event_index, f"event {event_index} iteration")
        outcomes: dict[str, tuple[bool, int | None, float | None] | None] = {}
        for observation_name in ("parent_oracle", "child_oracle"):
            observation = event.get(observation_name)
            if observation is None:
                outcomes[observation_name] = None
                continue
            if not isinstance(observation, Mapping):
                raise CollectionError(
                    f"event {event_index}.{observation_name} must be an object or null"
                )
            outcome = _score_outcome(
                observation,
                context=f"event {event_index}.{observation_name}",
                replayed=replayed,
                charged_canonicals=charged_canonicals,
                checkpoint_records=checkpoint_records,
                cumulative_calls=cumulative_calls,
                budget=budget,
            )
            outcomes[observation_name] = outcome
            charged, validated_index, validated_score = outcome
            if charged:
                if validated_index is None or validated_score is None:
                    raise CollectionError("charged outcome lacks replayable score identity")
                canonical = str(observation["canonical_smiles"])
                charged_calls.append(validated_index)
                cumulative_calls += 1
                replayed[validated_index] = (canonical, validated_score)
                charged_canonicals.add(canonical)
                if observation_name == "child_oracle":
                    children.append((validated_index, validated_score))
        child_outcome = outcomes["child_oracle"]
        update = event.get("population_update")
        update_reason = update.get("reason") if isinstance(update, Mapping) else None
        if child_outcome is None:
            _equal(update_reason, "budget_after_parent", f"event {event_index} null child")
            parent_outcome = outcomes["parent_oracle"]
            if parent_outcome is None or not parent_outcome[0]:
                raise CollectionError(
                    f"event {event_index} null child requires a charged parent"
                )
            _equal(cumulative_calls, budget, f"event {event_index} null child budget")
        elif update_reason == "budget_after_parent":
            raise CollectionError(
                f"event {event_index} budget_after_parent must have a null child"
            )
        _equal(event.get("oracle_calls"), cumulative_calls, f"event {event_index} oracle_calls")
        if "elapsed_seconds" in event:
            elapsed = _nonnegative_number(
                event["elapsed_seconds"], f"event {event_index} elapsed_seconds"
            )
            if elapsed_values and elapsed < elapsed_values[-1]:
                raise CollectionError("event elapsed_seconds are not chronological")
            elapsed_values.append(elapsed)
        else:
            timing_complete = False
            if summary_schema >= 2:
                raise CollectionError(
                    f"schema-{summary_schema} event {event_index} lacks elapsed_seconds"
                )
    _equal(observed_events, expected_events, "event log count")
    _equal(cumulative_calls, budget, "charged event count")
    _equal(charged_calls, list(range(1, budget + 1)), "charged event call order")
    return children, timing_complete, elapsed_values[-1] if elapsed_values else None


def _child_count_summary(
    scores: Sequence[float],
    *,
    reporting_frequency: int,
) -> dict[str, Any]:
    if scores:
        result = summarize_scores(
            scores,
            reporting_frequency=reporting_frequency,
            budget=len(scores),
        )
        result["axis"] = "charged_child_count"
        result["score_count"] = result.pop("oracle_calls")
        result["child_count_horizon"] = result.pop("oracle_budget")
        return result
    result = {
        "axis": "charged_child_count",
        "score_count": 0,
        "child_count_horizon": 0,
        "reporting_frequency": reporting_frequency,
    }
    for k in (1, 10, 100):
        label = f"top_{k}"
        result[label] = None
        result[f"auc_{label}"] = None
        result[f"trajectory_{label}"] = [{"oracle_calls": 0, "top_k_mean": 0.0}]
    return result


def _compact_metrics(
    source_key: str,
    group: Mapping[str, Any],
    semantics: str,
) -> dict[str, Any]:
    if source_key == "charged_children_child_count_axis":
        score_count = group["score_count"]
        axis_budget = group["child_count_horizon"]
    elif source_key == "charged_children_total_call_axis":
        score_count = group["score_count"]
        axis_budget = group["oracle_budget"]
    else:
        score_count = group["oracle_calls"]
        axis_budget = group["oracle_budget"]
    return {
        "source_key": source_key,
        "semantics": semantics,
        "axis": group.get("axis"),
        "score_count": score_count,
        "axis_budget": axis_budget,
        "reporting_frequency": group["reporting_frequency"],
        **{name: group[name] for name in METRIC_FIELDS},
    }


def _validated_metrics(
    summary: Mapping[str, Any],
    *,
    all_scores: Sequence[float],
    child_indexed_scores: Sequence[tuple[int, float]],
    reporting_frequency: int,
    budget: int,
) -> tuple[str, dict[str, dict[str, Any]]]:
    groups = _mapping(summary, "scores", "summary")
    all_expected = summarize_scores(
        all_scores,
        reporting_frequency=reporting_frequency,
        budget=budget,
    )
    _metric_equal(
        groups.get("all_charged_molecules"),
        all_expected,
        "all-charged metrics",
    )
    recognized = {key for key in groups if str(key).startswith("charged_children")}
    child_scores = [score for _, score in child_indexed_scores]
    compact = {
        "all_charged_molecules": _compact_metrics(
            "all_charged_molecules",
            all_expected,
            "all charged molecules on the total oracle-call axis",
        )
    }
    if summary.get("schema_version") == 1:
        legacy_key = CHILD_VIEWS["legacy"][0]
        _equal(recognized, {legacy_key}, "schema-1 child metric keys")
        legacy_expected = summarize_scores(
            child_scores,
            reporting_frequency=reporting_frequency,
            budget=budget,
        )
        _metric_equal(groups.get(legacy_key), legacy_expected, "legacy child metrics")
        compact[legacy_key] = _compact_metrics(
            legacy_key,
            legacy_expected,
            CHILD_VIEWS["legacy"][1],
        )
        return "legacy", compact
    if summary.get("schema_version") == 2:
        total_key, total_semantics = CHILD_VIEWS["total"]
        count_key, count_semantics = CHILD_VIEWS["count"]
        _equal(recognized, {total_key, count_key}, "schema-2 child metric keys")
        total_expected = summarize_indexed_scores(
            child_indexed_scores,
            observed_oracle_calls=budget,
            reporting_frequency=reporting_frequency,
            budget=budget,
        )
        count_expected = _child_count_summary(
            child_scores,
            reporting_frequency=reporting_frequency,
        )
        _metric_equal(groups.get(total_key), total_expected, "total-call child metrics")
        _metric_equal(groups.get(count_key), count_expected, "child-count metrics")
        compact[total_key] = _compact_metrics(total_key, total_expected, total_semantics)
        compact[count_key] = _compact_metrics(count_key, count_expected, count_semantics)
        return "total|count", compact
    raise CollectionError(
        f"unsupported summary schema_version {summary.get('schema_version')!r}"
    )


def collect_run(
    run_dir: str | os.PathLike[str],
    *,
    experiment_root: str | os.PathLike[str] | None = None,
    trust_local_checkpoint: bool = False,
    launch_records: Mapping[
        tuple[str, str, str, int],
        Sequence[LaunchRecord],
    ]
    | None = None,
) -> CollectedRun:
    """Validate one completed run containing an explicitly trusted local pickle."""

    if not trust_local_checkpoint:
        raise CollectionError(
            "checkpoint loading requires explicit trust_local_checkpoint=True"
        )
    directory = _trusted_existing_path(run_dir, "run directory", kind="directory")
    root = (
        _trusted_existing_path(experiment_root, "experiment root", kind="directory")
        if experiment_root is not None
        else directory.parents[2]
    )
    try:
        relative_run = directory.relative_to(root)
    except ValueError as error:
        raise CollectionError(f"run directory escapes experiment root {root}") from error
    if len(relative_run.parts) != 3:
        raise CollectionError(
            f"run is not exactly <oracle>/<variant>/seed_N under {root}: {directory}"
        )
    seed_match = SEED_DIRECTORY.fullmatch(directory.name)
    if seed_match is None or len(directory.parents) < 3:
        raise CollectionError(f"run path does not end in <oracle>/<variant>/seed_N: {directory}")
    path_oracle = directory.parent.parent.name
    path_variant = directory.parent.name
    path_seed = int(seed_match.group(1))
    paths = {
        name: _trusted_existing_path(
            directory / relative,
            f"run {name} source",
            kind="file",
            root=root,
        )
        for name, relative in SOURCE_FILES.items()
    }
    hashes_before = {name: sha256_file(path) for name, path in paths.items()}
    manifest = _read_json_object(paths["manifest"], "manifest")
    summary = _read_json_object(paths["summary"], "summary")
    for manifest_key, source_name in (
        ("summary_path", "summary"),
        ("events_path", "events"),
        ("checkpoint_path", "checkpoint"),
    ):
        if manifest_key in manifest:
            declared = _trusted_existing_path(
                manifest[manifest_key],
                f"manifest {manifest_key}",
                kind="file",
                root=root,
            )
            _equal(declared, paths[source_name], f"manifest {manifest_key}")
    _equal(manifest.get("schema_version"), 1, "manifest schema_version")
    _equal(manifest.get("status"), "completed", "manifest status")
    _equal(summary.get("status"), "completed", "summary status")
    _equal(manifest.get("error"), None, "manifest error")
    _equal(summary.get("error"), None, "summary error")
    _equal(summary.get("checkpoint_consistent"), True, "summary checkpoint_consistent")
    config = dict(_mapping(manifest, "config", "manifest"))
    durable_events_complete = "durable_events" in config
    if durable_events_complete:
        _boolean(config["durable_events"], "manifest config durable_events")
    experiment_id = config.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise CollectionError("manifest.config.experiment_id must be a nonempty string")
    identity = (experiment_id, path_oracle, path_variant, path_seed)
    run_id = f"{experiment_id}:{path_oracle}:{path_variant}:seed{path_seed}"
    for actual, expected, context in (
        (manifest.get("run_id"), run_id, "manifest run_id"),
        (summary.get("run_id"), run_id, "summary run_id"),
        (manifest.get("task"), path_oracle, "manifest task"),
        (manifest.get("variant"), path_variant, "manifest variant"),
        (manifest.get("seed"), path_seed, "manifest seed"),
        (config.get("oracle"), path_oracle, "manifest config oracle"),
        (config.get("variant"), path_variant, "manifest config variant"),
        (config.get("seed"), path_seed, "manifest config seed"),
    ):
        _equal(actual, expected, context)
    config_hash = sha256_config(config)
    _equal(manifest.get("config_sha256"), config_hash, "manifest config_sha256")
    _equal(summary.get("config_sha256"), config_hash, "summary config_sha256")

    model = _mapping(manifest, "model", "manifest")
    model_path = model.get("path")
    model_sha256 = model.get("sha256")
    if not isinstance(model_path, str) or not isinstance(model_sha256, str):
        raise CollectionError("manifest model identity is invalid")
    _equal(config.get("model_path"), model_path, "manifest config model_path")
    _equal(summary.get("model_sha256"), model_sha256, "summary model_sha256")
    extra = _mapping(manifest, "extra", "manifest")
    vocabulary = _mapping(extra, "vocabulary", "manifest.extra")
    _equal(config.get("vocab_path"), vocabulary.get("path"), "vocabulary path")
    budget = _integer(manifest.get("oracle_budget"), "manifest oracle_budget", minimum=1)
    frequency = _integer(
        config.get("reporting_frequency"),
        "manifest config reporting_frequency",
        minimum=1,
    )
    _equal(config.get("max_oracle_calls"), budget, "manifest config max_oracle_calls")
    _equal(manifest.get("oracle_calls"), budget, "manifest oracle_calls")
    event_count = _integer(summary.get("events"), "summary events")
    iterations = _integer(summary.get("iterations_completed"), "summary iterations")
    _equal(iterations, event_count, "summary iterations versus events")
    _equal(summary.get("recoverable_events"), event_count, "summary recoverable events")
    _equal(summary.get("recoverable_oracle_calls"), budget, "summary recoverable calls")
    summary_schema = _integer(summary.get("schema_version"), "summary schema_version", minimum=1)
    summary_elapsed = _nonnegative_number(
        summary.get("elapsed_seconds"), "summary elapsed_seconds"
    )
    manifest_elapsed = _nonnegative_number(
        manifest.get("elapsed_seconds"), "manifest elapsed_seconds"
    )
    _equal(manifest_elapsed, summary_elapsed, "manifest/summary elapsed_seconds")

    checkpoint, checkpoint_elapsed = _checkpoint_records(
        paths["checkpoint"],
        manifest=manifest,
        summary=summary,
        event_count=event_count,
        budget=budget,
    )
    child_scores, timing_complete, last_event_elapsed = _event_records(
        paths["events"],
        checkpoint_records=checkpoint,
        expected_events=event_count,
        budget=budget,
        summary_schema=summary_schema,
    )
    if last_event_elapsed is not None and last_event_elapsed > checkpoint_elapsed:
        raise CollectionError("last event elapsed_seconds exceeds checkpoint elapsed_seconds")
    if checkpoint_elapsed > summary_elapsed:
        raise CollectionError("checkpoint elapsed_seconds exceeds summary elapsed_seconds")
    all_scores = [checkpoint[index][1] for index in range(1, budget + 1)]
    child_views, compact_metrics = _validated_metrics(
        summary,
        all_scores=all_scores,
        child_indexed_scores=child_scores,
        reporting_frequency=frequency,
        budget=budget,
    )

    git = _mapping(extra, "git", "manifest.extra")
    runtime = _mapping(extra, "runtime", "manifest.extra")
    base_runtime = _mapping(manifest, "runtime", "manifest")
    gpu = runtime.get("gpu") if isinstance(runtime.get("gpu"), Mapping) else {}
    hashes_after = {name: sha256_file(path) for name, path in paths.items()}
    _equal(hashes_after, hashes_before, "source hashes after validation")
    sources = {
        name: {"path": str(path), "sha256": hashes_after[name]}
        for name, path in paths.items()
    }

    attempts = tuple((launch_records or {}).get(identity, ()))
    validated_attempts = []
    for attempt in attempts:
        _equal(sha256_file(attempt.path), attempt.sha256, "launch log source hash")
        validated_attempts.append(
            _validate_launch(
                attempt,
                identity=identity,
                config=config,
                manifest_runtime=base_runtime,
                expected_output_root=directory.parents[3],
            )
        )
    parsed_attempts = [validated[0] for validated in validated_attempts]
    resume_count = _integer(manifest.get("resume_count", 0), "manifest resume_count")
    initial_positions = [
        index for index, parsed in enumerate(parsed_attempts) if "--resume" not in parsed
    ]
    initial_position = initial_positions[-1] if initial_positions else None
    lineage_attempts = (
        attempts[initial_position:] if initial_position is not None else ()
    )
    lineage_parsed = (
        parsed_attempts[initial_position:] if initial_position is not None else []
    )
    history_complete = bool(lineage_attempts) and len(lineage_attempts) == resume_count + 1
    if history_complete:
        history_complete = all("--resume" in parsed for parsed in lineage_parsed[1:])
    policy_complete = bool(attempts) and all(
        validated[1] for validated in validated_attempts
    )
    matrix_complete = bool(attempts) and all(
        validated[2] for validated in validated_attempts
    )
    if lineage_attempts:
        first_gpu = lineage_attempts[0].record["physical_gpu"]
        _equal(
            runtime.get("cuda_visible_devices"),
            str(first_gpu["index"]),
            "initial launch physical GPU versus manifest CUDA_VISIBLE_DEVICES",
        )
        first_uuid = first_gpu["uuid"]
        gpu_migrated = any(
            attempt.record["physical_gpu"]["uuid"] != first_uuid
            for attempt in lineage_attempts[1:]
        )
    else:
        gpu_migrated = None
    launch_times = [attempt.record["time_unix"] for attempt in attempts]
    if launch_times != sorted(launch_times):
        raise CollectionError("launch attempts are not chronological")
    launch_gpus = [attempt.record["physical_gpu"]["uuid"] for attempt in attempts]
    any_gpu_sharing = (
        any(bool(attempt.record["compute_processes"]) for attempt in attempts)
        if attempts
        else None
    )
    command_hashes = [sha256_config(attempt.record["command"]) for attempt in attempts]
    launch_logs = list(dict.fromkeys(str(attempt.path) for attempt in attempts))
    child_columns: dict[str, Any] = {}
    for view, (source_key, _) in CHILD_VIEWS.items():
        metrics = compact_metrics.get(source_key, {})
        for name in ("score_count", "axis_budget", *METRIC_FIELDS):
            child_columns[f"child_{view}_{name}"] = metrics.get(name)
    all_metrics = compact_metrics["all_charged_molecules"]
    row = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "oracle": path_oracle,
        "variant": path_variant,
        "seed": path_seed,
        "summary_schema_version": summary_schema,
        "status": "completed",
        "oracle_budget": budget,
        "all_oracle_calls": budget,
        "charged_child_count": len(child_scores),
        "child_views": child_views,
        "events": event_count,
        "iterations_completed": iterations,
        "elapsed_seconds": summary_elapsed,
        "reporting_frequency": frequency,
        **{f"all_{name}": all_metrics[name] for name in METRIC_FIELDS},
        **child_columns,
        **{name: config.get(name) for name in CSV_CONFIG_FIELDS},
        "config_sha256": config_hash,
        "model_path": model_path,
        "model_sha256": model_sha256,
        "vocabulary_path": vocabulary.get("path"),
        "vocabulary_sha256": vocabulary.get("sha256"),
        "scientific_status": config.get("scientific_status"),
        "git_commit": git.get("commit"),
        "git_branch": git.get("branch"),
        "git_dirty": git.get("dirty"),
        "tracked_diff_sha256": git.get("tracked_diff_sha256"),
        "hostname": base_runtime.get("hostname"),
        "python_version": base_runtime.get("python_version"),
        "cuda_visible_devices": runtime.get("cuda_visible_devices"),
        "gpu_name": gpu.get("name"),
        "gpu_logical_index": gpu.get("logical_index"),
        "torch_version": runtime.get("torch"),
        "rdkit_version": runtime.get("rdkit"),
        "resume_count": resume_count,
        "launch_attempt_count": len(attempts),
        "launch_history_complete": history_complete if attempts else None,
        "launch_policy_complete": policy_complete if attempts else None,
        "launch_matrix_provenance_complete": matrix_complete if attempts else None,
        "launch_provenance_complete": (
            history_complete
            and policy_complete
            and matrix_complete
            and durable_events_complete
            if attempts
            else None
        ),
        "launch_gpu_migrated": gpu_migrated,
        "launch_any_gpu_sharing": any_gpu_sharing,
        "launch_wall_time_comparable": (
            all(attempt.record["wall_time_comparable"] for attempt in attempts)
            if policy_complete
            else None
        ),
        "durable_events_provenance_complete": durable_events_complete,
        "timing_provenance_complete": timing_complete,
        "launch_first_time_unix": launch_times[0] if attempts else None,
        "launch_last_time_unix": launch_times[-1] if attempts else None,
        "launch_gpu_uuids": launch_gpus,
        "launch_command_sha256s": command_hashes,
        "launch_logs": launch_logs,
        **{
            f"source_{name}_sha256": source["sha256"]
            for name, source in sources.items()
        },
    }
    _equal(set(row), set(CSV_FIELDS), "CSV row fields")
    return CollectedRun(directory, row, config, compact_metrics, sources, attempts)


def discover_run_dirs(experiment_root: str | os.PathLike[str]) -> list[Path]:
    root = _trusted_existing_path(
        experiment_root, "experiment root", kind="directory"
    )
    result = []
    for path in sorted(root.glob("*/*/seed_*")):
        if path.is_dir() and SEED_DIRECTORY.fullmatch(path.name):
            result.append(
                _trusted_existing_path(
                    path, "run directory", kind="directory", root=root
                )
            )
    return result


def _is_incomplete(run_dir: Path, experiment_root: Path) -> bool:
    paths: dict[str, Path] = {}
    missing = False
    for name, relative in SOURCE_FILES.items():
        candidate = run_dir / relative
        for component in (candidate, *candidate.parents):
            if component.is_symlink():
                raise CollectionError(
                    f"run {name} source must not contain symlinks: {component}"
                )
        if not candidate.is_file():
            missing = True
            continue
        paths[name] = _trusted_existing_path(
            candidate,
            f"run {name} source",
            kind="file",
            root=experiment_root,
        )
    if missing:
        return True
    manifest = _read_json_object(paths["manifest"], "manifest")
    summary = _read_json_object(paths["summary"], "summary")
    return manifest.get("status") != "completed" or summary.get("status") != "completed"


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    return value


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=CSV_FIELDS,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _csv_value(row.get(key)) for key in CSV_FIELDS})
    return stream.getvalue().encode("utf-8")


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


def _publish_immutable_csv(
    destination: Path,
    payload: bytes,
) -> tuple[Path, str, bool]:
    digest = hashlib.sha256(payload).hexdigest()
    path = destination / f"results.{digest}.csv"
    if path.exists():
        if path.read_bytes() != payload:
            raise CollectionError(f"hash-named result file has unexpected content: {path}")
        return path, digest, False
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination,
        prefix=".results.",
        suffix=".tmp",
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
            if path.read_bytes() != payload:
                raise CollectionError(
                    f"hash-named result file has unexpected content: {path}"
                )
        _fsync_directory(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return path, digest, created


def _relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _manifest_run(collected: CollectedRun, experiment_root: Path) -> dict[str, Any]:
    row = collected.row
    sources = {
        name: {
            "path": _relative_or_absolute(Path(source["path"]), experiment_root),
            "sha256": source["sha256"],
        }
        for name, source in collected.sources.items()
    }
    launch_sources: dict[tuple[str, str], dict[str, str]] = {}
    attempts = []
    for attempt in collected.launch_attempts:
        launch_sources[(str(attempt.path), attempt.sha256)] = {
            "path": str(attempt.path),
            "sha256": attempt.sha256,
        }
        attempts.append(
            {
                "path": str(attempt.path),
                "line_number": attempt.line_number,
                "command_sha256": sha256_config(attempt.record["command"]),
                "policy_complete": GPU_POLICY_FIELDS <= attempt.record.keys(),
                "matrix_provenance_complete": (
                    "--matrix-path" in attempt.record["command"]
                    and "--matrix-sha256" in attempt.record["command"]
                ),
                "record": attempt.record,
            }
        )
    launch = None
    if attempts:
        initial_ordinals = [
            index
            for index, attempt in enumerate(attempts, start=1)
            if "--resume" not in attempt["record"]["command"]
        ]
        launch = {
            "attempt_count": len(attempts),
            "initial_attempt_ordinal": (
                initial_ordinals[-1] if initial_ordinals else None
            ),
            "pre_initial_attempt_count": (
                initial_ordinals[-1] - 1 if initial_ordinals else None
            ),
            "history_complete": row["launch_history_complete"],
            "policy_complete": row["launch_policy_complete"],
            "matrix_provenance_complete": row[
                "launch_matrix_provenance_complete"
            ],
            "provenance_complete": row["launch_provenance_complete"],
            "gpu_migrated": row["launch_gpu_migrated"],
            "any_gpu_sharing": row["launch_any_gpu_sharing"],
            "wall_time_comparable": row["launch_wall_time_comparable"],
            "durable_events_config_complete": row[
                "durable_events_provenance_complete"
            ],
            "sources": list(launch_sources.values()),
            "attempts": attempts,
        }
    provenance_fields = (
        "config_sha256",
        "model_path",
        "model_sha256",
        "vocabulary_path",
        "vocabulary_sha256",
        "scientific_status",
        "git_commit",
        "git_branch",
        "git_dirty",
        "tracked_diff_sha256",
        "hostname",
        "python_version",
        "cuda_visible_devices",
        "gpu_name",
        "gpu_logical_index",
        "torch_version",
        "rdkit_version",
        "resume_count",
        "durable_events_provenance_complete",
        "timing_provenance_complete",
    )
    return {
        "identity": {
            "experiment_id": row["experiment_id"],
            "run_id": row["run_id"],
            "oracle": row["oracle"],
            "variant": row["variant"],
            "seed": row["seed"],
        },
        "run_dir": _relative_or_absolute(collected.run_dir, experiment_root),
        "summary_schema_version": row["summary_schema_version"],
        "status": row["status"],
        "oracle_budget": row["oracle_budget"],
        "events": row["events"],
        "iterations_completed": row["iterations_completed"],
        "elapsed_seconds": row["elapsed_seconds"],
        "charged_child_count": row["charged_child_count"],
        "metrics": collected.metrics,
        "config": collected.config,
        "provenance": {name: row[name] for name in provenance_fields},
        "sources": sources,
        "launch": launch,
    }


def _recheck_sources(collected: Sequence[CollectedRun]) -> None:
    checked_logs: set[tuple[Path, str]] = set()
    for run in collected:
        for source in run.sources.values():
            _equal(
                sha256_file(source["path"]),
                source["sha256"],
                f"source hash before publication for {source['path']}",
            )
        for attempt in run.launch_attempts:
            key = (attempt.path, attempt.sha256)
            if key not in checked_logs:
                _equal(
                    sha256_file(attempt.path),
                    attempt.sha256,
                    f"launch log hash before publication for {attempt.path}",
                )
                checked_logs.add(key)


def collect_results(
    experiment_root: str | os.PathLike[str],
    output_dir: str | os.PathLike[str] | None = None,
    *,
    matrix_path: str | os.PathLike[str] | None = None,
    logs_dir: str | os.PathLike[str] | None = None,
    skip_incomplete: bool = False,
    overwrite: bool = False,
    trust_local_checkpoints: bool = False,
) -> tuple[Path, Path]:
    """Validate a complete matrix of trusted local runs and publish its results."""

    if not trust_local_checkpoints:
        raise CollectionError(
            "checkpoint loading requires --trust-local-checkpoints or "
            "trust_local_checkpoints=True; pickle can execute code"
        )
    root = _trusted_existing_path(
        experiment_root, "experiment root", kind="directory"
    )
    if matrix_path is None:
        raise CollectionError("collection requires exactly one explicit matrix path")
    plan = _load_matrix_plan(matrix_path, root)
    destination = root if output_dir is None else _lexical_absolute(output_dir)
    for component in (destination, *destination.parents):
        if component.is_symlink():
            raise CollectionError(
                f"collection destination must not contain symlinks: {component}"
            )
    destination.mkdir(parents=True, exist_ok=True)
    destination = _trusted_existing_path(
        destination, "collection destination", kind="directory"
    )
    run_dirs = discover_run_dirs(root)
    actual_jobs = {
        (path.parent.parent.name, path.parent.name, int(path.name.removeprefix("seed_")))
        for path in run_dirs
    }
    unexpected_jobs = sorted(actual_jobs - plan.jobs.keys())
    if unexpected_jobs:
        raise CollectionError(
            f"experiment tree contains jobs absent from matrix: {unexpected_jobs!r}"
        )
    absent_jobs = sorted(plan.jobs.keys() - actual_jobs)
    if absent_jobs and not skip_incomplete:
        raise CollectionError(f"experiment tree is missing matrix jobs: {absent_jobs!r}")
    destination_lock_path = destination / ".collection.lock"
    if destination_lock_path.is_symlink():
        raise CollectionError(f"collection lock must not be a symlink: {destination_lock_path}")
    try:
        destination_lock = FileLock(destination_lock_path, fcntl.LOCK_EX)
    except BlockingIOError as error:
        raise CollectionError(f"another collector owns destination {destination}") from error
    with destination_lock, ExitStack() as source_locks:
        skipped: list[dict[str, str]] = []
        for oracle, variant, seed in absent_jobs:
            skipped.append(
                {
                    "run_dir": f"{oracle}/{variant}/seed_{seed}",
                    "reason": "missing",
                }
            )
        collectable: list[Path] = []
        for run_dir in run_dirs:
            run_lock_path = run_dir / ".run.lock"
            if run_lock_path.is_symlink():
                raise CollectionError(f"run lock must not be a symlink: {run_lock_path}")
            try:
                run_lock = FileLock(run_lock_path, fcntl.LOCK_SH)
            except BlockingIOError as error:
                if skip_incomplete:
                    skipped.append(
                        {
                            "run_dir": _relative_or_absolute(run_dir, root),
                            "reason": "active",
                        }
                    )
                    continue
                raise CollectionError(f"run directory is active: {run_dir}") from error
            try:
                if skip_incomplete and _is_incomplete(run_dir, root):
                    skipped.append(
                        {
                            "run_dir": _relative_or_absolute(run_dir, root),
                            "reason": "incomplete",
                        }
                    )
                    run_lock.close()
                    continue
            except BaseException:
                run_lock.close()
                raise
            source_locks.enter_context(run_lock)
            collectable.append(run_dir)
        collection_path = destination / "collection_manifest.json"
        legacy_results_path = destination / "results.csv"
        if not overwrite and (collection_path.exists() or legacy_results_path.exists()):
            raise FileExistsError(
                f"collection output already exists in {destination}; pass --overwrite"
        )
        launch_records = load_launch_records(logs_dir) if logs_dir is not None else {}
        collected: list[CollectedRun] = []
        for run_dir in collectable:
            collected.append(
                collect_run(
                    run_dir,
                    experiment_root=root,
                    trust_local_checkpoint=True,
                    launch_records=launch_records,
                )
            )
        if not collected and not skip_incomplete:
            raise CollectionError("no completed runs were available to collect")
        collected.sort(
            key=lambda run: (run.row["oracle"], run.row["variant"], run.row["seed"])
        )
        run_ids = [run.row["run_id"] for run in collected]
        if len(set(run_ids)) != len(run_ids):
            raise CollectionError("duplicate run identities were discovered")
        experiment_ids = {run.row["experiment_id"] for run in collected}
        matrix_experiment_id = str(plan.matrix["experiment_id"])
        if experiment_ids - {matrix_experiment_id}:
            raise CollectionError(
                f"experiment tree mixes experiment IDs: {sorted(experiment_ids)!r}"
            )
        recorded_matrix = [_validate_run_matrix(run, plan) for run in collected]
        collected_jobs = {
            (run.row["oracle"], run.row["variant"], run.row["seed"])
            for run in collected
        }
        missing_jobs = []
        skipped_reasons = {entry["run_dir"]: entry["reason"] for entry in skipped}
        for oracle, variant, seed in sorted(plan.jobs.keys() - collected_jobs):
            relative = f"{oracle}/{variant}/seed_{seed}"
            missing_jobs.append(
                {
                    "oracle": oracle,
                    "variant": variant,
                    "seed": seed,
                    "reason": skipped_reasons.get(relative, "not_collected"),
                }
            )
        collection_complete = not missing_jobs
        if not collection_complete and not skip_incomplete:
            raise CollectionError(f"matrix jobs were not collected: {missing_jobs!r}")
        _recheck_sources(collected)
        _equal(sha256_file(plan.path), plan.sha256, "matrix hash before publication")
        results_path, results_hash, results_created = _publish_immutable_csv(
            destination,
            _csv_bytes([run.row for run in collected]),
        )
        try:
            collection_manifest = {
                "schema_version": COLLECTION_SCHEMA_VERSION,
                "created_at": _utc_timestamp(),
                "experiment_root": str(root),
                "experiment_id": matrix_experiment_id,
                "run_count": len(collected),
                "collection_complete": collection_complete,
                "missing_jobs": missing_jobs,
                "skipped": skipped,
                "matrix": {
                    "path": str(plan.path),
                    "sha256": plan.sha256,
                    "schema_version": plan.matrix["schema_version"],
                    "expected_job_count": len(plan.jobs),
                    "expected_jobs": [
                        {"oracle": oracle, "variant": variant, "seed": seed}
                        for oracle, variant, seed in sorted(plan.jobs)
                    ],
                    "recorded_by_all_collected_runs": bool(collected)
                    and all(recorded_matrix),
                },
                "collector": {
                    "path": str(Path(__file__).resolve()),
                    "sha256": sha256_file(Path(__file__).resolve()),
                },
                "results_csv": {
                    "path": results_path.name,
                    "sha256": results_hash,
                    "immutable": True,
                    "columns": list(CSV_FIELDS),
                },
                "runs": [_manifest_run(run, root) for run in collected],
            }
            write_manifest(collection_path, collection_manifest, overwrite=overwrite)
        except BaseException:
            if results_created:
                results_path.unlink(missing_ok=True)
                _fsync_directory(destination)
            raise
        return results_path, collection_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--logs-dir", type=Path)
    parser.add_argument("--skip-incomplete", action="store_true")
    parser.add_argument(
        "--trust-local-checkpoints",
        action="store_true",
        help="Explicitly trust contained local checkpoint pickles to execute during loading.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="atomically replace an existing collection manifest pointer",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results_path, manifest_path = collect_results(
        args.experiment_root,
        args.output_dir,
        matrix_path=args.matrix,
        logs_dir=args.logs_dir,
        skip_incomplete=args.skip_incomplete,
        overwrite=args.overwrite,
        trust_local_checkpoints=args.trust_local_checkpoints,
    )
    print(
        json.dumps(
            {"collection_manifest": str(manifest_path), "results_csv": str(results_path)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "COLLECTION_SCHEMA_VERSION",
    "CSV_FIELDS",
    "CollectedRun",
    "CollectionError",
    "FileLock",
    "LaunchRecord",
    "collect_results",
    "collect_run",
    "discover_run_dirs",
    "load_launch_records",
]

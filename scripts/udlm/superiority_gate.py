"""Evaluate the frozen UDLM-over-local-GenMol de-novo superiority gate.

The production CLI starts from the three raw benchmark run directories, lets
the existing de-novo reporter revalidate every row, verifies that the exact
candidate lock was already committed at the benchmark revision, and writes one
no-clobber decision record.  It is CPU-only and stream-hashes, but never
deserializes, model checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.exps.denovo import report as denovo_report  # noqa: E402


SCHEMA_VERSION = 1
PROTOCOL_RELATIVE_PATH = Path("experiments/udlm/protocols/de_novo_superiority_v3.json")
PROTOCOL_SHA256 = "27a1f3e4fa66988d77eddeb66025eae64b514c452e089bb5c62fff99060c9f16"
PROTOCOL_CANONICAL_SHA256 = (
    "e7b108dce51cd1445758a9f7dc852532b2ee075a80f783ae009303a7550577ee"
)
PREVIOUS_PROTOCOL_RELATIVE_PATH = Path(
    "experiments/udlm/protocols/de_novo_superiority_v2.json"
)
PREVIOUS_PROTOCOL_SHA256 = (
    "f845429dae7ca889c09aad3af7946d20a5a05c189d2a19ec5ad8da7fff075a66"
)
PREVIOUS_PROTOCOL_CANONICAL_SHA256 = (
    "b3b890ba19368e0caefda7a6ca9b082c9d7c2911ddd92eba1d83d1cf91408396"
)
BASELINE_RELATIVE_PATH = Path("experiments/udlm/baselines/mdlm_50000.json")
BASELINE_SHA256 = "6da46fc615dedbcca436da087a2c1e9145f5d110036e0c15bb431ded3c2e5539"
BASELINE_RESCORE_RELATIVE_PATH = Path(
    "experiments/udlm/baselines/mdlm_50000_rescore_attestation.json"
)
BASELINE_RESCORE_SHA256 = (
    "6326b63c38c7052d0b47282d611618f77637496da2785779af69097fc1441323"
)
BASELINE_RESCORE_CANONICAL_SHA256 = (
    "5aaba90f1ee23a45591eeeb297d0392f36e7ef1ed034263e254f0ecad94c6e96"
)
DENOVO_RESCORE_RELATIVE_PATH = Path("scripts/udlm/rescore_denovo_run.py")
DENOVO_RESCORE_DEPENDENCY_RELATIVE_PATH = Path("scripts/udlm/rescore_mdlm_baseline.py")
DENOVO_LAUNCHER_RELATIVE_PATH = Path("scripts/exps/denovo/launch_benchmark.py")
PILOT_EVIDENCE_WRITER_RELATIVE_PATH = Path("scripts/udlm/write_pilot_evidence.py")
EXPECTED_BASELINE_RESCORE_SOURCE_REVISION = "74482c2742ab5ad15def122c809a6b4e403e94cf"
EXPECTED_PROTOCOL_ID = "genmol_udlm_de_novo_superiority_v3"
EXPECTED_SEEDS = (0, 1, 2)
EXPECTED_SAMPLES_PER_SEED = 1_000
EXPECTED_NFE = 128
EXPECTED_BASELINE_BENCHMARK_SCHEMA_VERSION = 7
EXPECTED_BASELINE_REPORT_SCHEMA_VERSION = 6
EXPECTED_BASELINE_RAW_SAMPLE_FIELD_COUNT = 21
NUMERIC_RAW_SAMPLE_FIELDS = frozenset(
    {"strict_qed", "strict_sa", "released_qed", "released_sa"}
)
EXPECTED_BASELINE_CHECKPOINT_SHA256 = (
    "8d00aa47b02f64bf39ff6b0b2e786f213587366fc2c3d29712a00f3f84108dd6"
)
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
HEX_GIT_REVISION = re.compile(r"[0-9a-f]{40}\Z")
RUN_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
METRICS = ("validity", "uniqueness", "quality", "diversity")
CANDIDATE_LEDGER_SCHEMA_VERSION = 2
CANDIDATE_LOCK_SCHEMA_VERSION = 2
PILOT_EVIDENCE_SCHEMA_VERSION = 2
PILOT_FAILURE_RECEIPT_SCHEMA_VERSION = 1
PILOT_FAILURE_RECEIPT_KIND = "pilot_failure"
LAUNCH_MANIFEST_SCHEMA_VERSION = 2
RUNTIME_CONFIG_SCHEMA_VERSION = 2
TRAINING_SUMMARY_SCHEMA_VERSION = 5
PILOT_EXIT_STATUS_SCHEMA_VERSION = 5
PILOT_EMPIRICAL_UNIFORM_MIX = 0.0002
PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT = {
    "relative_path": (
        "experiments/udlm/prior_geometry/" "floor_selection_train_rows_10001_30000.json"
    ),
    "sha256": "02908dafaf589ca9a49e560aa1eab470a18d6bfe616b781164784c489f54a9f1",
    "source_revision": "6424b323084358ea050ba22d7e13ef8d45962496",
    "scope": "retrospective_training_only_engineering_selection",
}
PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_CANONICAL_SHA256 = (
    "2435a36af83e88a1bb1d602e840bb6ae1e2a97963a48abf68606a37e6320694d"
)
MAX_SAFE_UTILIZATION_PERCENT = 10
MIN_SAFE_FREE_MEMORY_MIB = 30_000
ACTIVE_COMPUTE_PROCESSES_ALLOWED = True
REGISTERED_SELECTION_PILOT_SEEDS = (1000, 1001)
REGISTERED_SELECTION_SAMPLES_PER_SEED = 256
REGISTERED_SELECTION_NFE = 128
REGISTERED_SELECTION_METRIC_BRANCH = "released_comparable"
NONREGISTERED_OPERATING_POINT_REASON = "engineering_or_nonregistered_operating_point"
UNDEFINED_SELECTION_METRIC_REASON = "undefined_released_diversity_no_unique_molecules"
FAILED_PILOT_REASON = "pilot_failed"
CANDIDATE_SELECTION_RULE = (
    "maximize_mean_released_quality_then_mean_released_diversity_"
    "then_lexicographically_smallest_attempt_id"
)
CHECKPOINT_SELECTION_RULE = "last_completed_optimizer_step"
CLAIM_SCOPE_BY_STARTUP = {
    "warm_start": "operational_continuation_only",
    "scratch": "single_training_trajectory_checkpoint_comparison_only",
}


class GateValidationError(ValueError):
    """Raised when evidence cannot support the registered decision."""


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise GateValidationError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise GateValidationError(f"non-finite JSON constant: {value}")


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise GateValidationError(f"non-finite JSON number: {value}")
    return parsed


def strict_json_loads(payload: bytes, *, label: str) -> object:
    """Decode UTF-8 JSON, rejecting duplicate keys and non-finite numbers."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise GateValidationError(f"{label} is not UTF-8") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_float,
        )
    except json.JSONDecodeError as error:
        raise GateValidationError(f"{label} is not valid JSON") from error


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def matched_panel_config_sha256(resolved_config: Mapping[str, Any]) -> str:
    """Recompute the launcher's common R/S/E config digest."""

    normalized = json.loads(
        json.dumps(resolved_config, allow_nan=False, ensure_ascii=False)
    )
    try:
        prior_variant = normalized["training"]["udlm"]["prior_variant"]
        callback_dirpath = normalized["callback"]["dirpath"]
    except (KeyError, TypeError) as error:
        raise GateValidationError(
            "resolved config lacks matched-panel treatment or callback fields"
        ) from error
    if prior_variant not in {
        "release_uniform",
        "schedule_uniform",
        "empirical_frequency",
    }:
        raise GateValidationError("resolved config has an unregistered UDLM treatment")
    if not isinstance(callback_dirpath, str) or not callback_dirpath:
        raise GateValidationError(
            "resolved config callback.dirpath must be a nonempty string"
        )
    normalized["training"]["udlm"]["prior_variant"] = "<REGISTERED_TREATMENT>"
    normalized["callback"]["dirpath"] = "<VARIANT_RUN_DIR>/checkpoints"
    return canonical_json_sha256(normalized)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GateValidationError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise GateValidationError(
            f"{label} fields are invalid: missing={missing}, extra={extra}"
        )


def _integer(value: object, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise GateValidationError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise GateValidationError(f"{label} must be at least {minimum}")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateValidationError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise GateValidationError(f"{label} must be finite")
    return result


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or HEX_SHA256.fullmatch(value) is None:
        raise GateValidationError(f"{label} must be 64 lowercase hexadecimal digits")
    return value


def _git_revision(value: object, label: str) -> str:
    if not isinstance(value, str) or HEX_GIT_REVISION.fullmatch(value) is None:
        raise GateValidationError(f"{label} must be 40 lowercase hexadecimal digits")
    return value


def _selected_gpu_uuids(
    value: object, label: str, *, expected_count: int | None = None
) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(uuid, str)
            or not uuid.startswith("GPU-")
            or len(uuid) <= len("GPU-")
            or "," in uuid
            for uuid in value
        )
        or len(set(value)) != len(value)
    ):
        raise GateValidationError(
            f"{label} must be a nonempty array of unique NVIDIA GPU UUID strings"
        )
    if expected_count is not None and len(value) != expected_count:
        raise GateValidationError(
            f"{label} count {len(value)} disagrees with world size {expected_count}"
        )
    return list(value)


def _required_true(value: object, label: str) -> None:
    if value is not True:
        raise GateValidationError(f"{label} must be true")


def _timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise GateValidationError(f"{label} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise GateValidationError(f"{label} is not valid ISO-8601") from error
    if parsed.tzinfo is None:
        raise GateValidationError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _utc_timestamp(value: object, label: str) -> datetime:
    parsed = _timestamp(value, label)
    if not isinstance(value, str):  # pragma: no cover - rejected by _timestamp
        raise GateValidationError(f"{label} must be an ISO-8601 string")
    original = datetime.fromisoformat(value)
    if original.utcoffset() != timezone.utc.utcoffset(None):
        raise GateValidationError(f"{label} must be expressed in UTC")
    return parsed


def _relative_path(value: object, label: str, *, suffix: str) -> Path:
    if not isinstance(value, str) or not value:
        raise GateValidationError(f"{label} must be a repository-relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != value:
        raise GateValidationError(f"{label} must be a normalized relative POSIX path")
    path = Path(*pure.parts)
    if path.suffix != suffix:
        raise GateValidationError(f"{label} must end in {suffix}")
    return path


def _relative_directory(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise GateValidationError(f"{label} must be a repository-relative directory")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or pure.as_posix() != value
        or len(pure.parts) < 2
        or pure.parts[0] != "output"
    ):
        raise GateValidationError(
            f"{label} must be a normalized repository-relative output directory"
        )
    return Path(*pure.parts)


def _stable_regular_file_bytes(path: Path, *, label: str) -> bytes:
    """Read one file while rejecting symlinks and concurrent replacement."""

    try:
        before_path = path.stat(follow_symlinks=False)
    except OSError as error:
        raise GateValidationError(f"{label} is unavailable: {path}") from error
    if not stat.S_ISREG(before_path.st_mode):
        raise GateValidationError(f"{label} is not a regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise GateValidationError(f"cannot safely open {label}: {path}") from error
    chunks: list[bytes] = []
    try:
        before_fd = os.fstat(descriptor)
        identity = (
            before_fd.st_dev,
            before_fd.st_ino,
            before_fd.st_mode,
            before_fd.st_size,
            before_fd.st_mtime_ns,
            before_fd.st_ctime_ns,
        )
        if (
            not stat.S_ISREG(before_fd.st_mode)
            or (
                before_path.st_dev,
                before_path.st_ino,
                before_path.st_mode,
                before_path.st_size,
                before_path.st_mtime_ns,
                before_path.st_ctime_ns,
            )
            != identity
        ):
            raise GateValidationError(f"{label} changed before open: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_fd = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = path.stat(follow_symlinks=False)
    for observed in (after_fd, after_path):
        if (
            observed.st_dev,
            observed.st_ino,
            observed.st_mode,
            observed.st_size,
            observed.st_mtime_ns,
            observed.st_ctime_ns,
        ) != identity:
            raise GateValidationError(f"{label} changed while being read: {path}")
    return b"".join(chunks)


def _repository_artifact_bytes(relative_path: Path, *, label: str) -> bytes:
    root = REPOSITORY_ROOT.resolve(strict=True)
    path = root.joinpath(relative_path)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise GateValidationError(f"{label} is unavailable: {path}") from error
    if resolved != path or root not in resolved.parents:
        raise GateValidationError(f"{label} must not escape or traverse symlinks")
    return _stable_regular_file_bytes(path, label=label)


def _stable_artifact_snapshot(
    path: Path, *, allowed_root: Path, label: str
) -> dict[str, Any]:
    """Stream-hash one scoped artifact and retain its stable file identity."""

    root = allowed_root.resolve(strict=True)
    path = Path(os.path.abspath(path))
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise GateValidationError(f"{label} is unavailable: {path}") from error
    if resolved != path or root not in resolved.parents:
        raise GateValidationError(f"{label} must not escape or traverse symlinks")
    try:
        before_path = path.stat(follow_symlinks=False)
    except OSError as error:
        raise GateValidationError(f"{label} is unavailable: {path}") from error
    if not stat.S_ISREG(before_path.st_mode):
        raise GateValidationError(f"{label} is not a regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise GateValidationError(f"cannot safely open {label}: {path}") from error
    digest = hashlib.sha256()
    try:
        before_fd = os.fstat(descriptor)
        identity = (
            before_fd.st_dev,
            before_fd.st_ino,
            before_fd.st_mode,
            before_fd.st_nlink,
            before_fd.st_size,
            before_fd.st_mtime_ns,
            before_fd.st_ctime_ns,
        )
        if (
            not stat.S_ISREG(before_fd.st_mode)
            or (
                before_path.st_dev,
                before_path.st_ino,
                before_path.st_mode,
                before_path.st_nlink,
                before_path.st_size,
                before_path.st_mtime_ns,
                before_path.st_ctime_ns,
            )
            != identity
        ):
            raise GateValidationError(f"{label} changed before open: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after_fd = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = path.stat(follow_symlinks=False)
    for observed in (after_fd, after_path):
        if (
            observed.st_dev,
            observed.st_ino,
            observed.st_mode,
            observed.st_nlink,
            observed.st_size,
            observed.st_mtime_ns,
            observed.st_ctime_ns,
        ) != identity:
            raise GateValidationError(f"{label} changed while being read: {path}")
    return {
        "path": str(path),
        "device": int(after_path.st_dev),
        "inode": int(after_path.st_ino),
        "mode": int(after_path.st_mode),
        "link_count": int(after_path.st_nlink),
        "size_bytes": int(after_path.st_size),
        "mtime_ns": int(after_path.st_mtime_ns),
        "ctime_ns": int(after_path.st_ctime_ns),
        "sha256": digest.hexdigest(),
        "stable_regular_file_verified": True,
    }


def _repository_artifact_snapshot(relative_path: Path, *, label: str) -> dict[str, Any]:
    return _stable_artifact_snapshot(
        REPOSITORY_ROOT / relative_path,
        allowed_root=REPOSITORY_ROOT,
        label=label,
    )


def _project_checkpoint_snapshot(path: Path, *, label: str) -> dict[str, Any]:
    repository_root = Path(os.path.abspath(REPOSITORY_ROOT))
    project_root = (
        repository_root.parent.parent
        if repository_root.parent.name == "run_sources"
        else repository_root
    )
    return _stable_artifact_snapshot(path, allowed_root=project_root, label=label)


def load_pinned_json(
    relative_path: Path, expected_sha256: str, *, label: str
) -> tuple[Mapping[str, Any], bytes]:
    payload = _repository_artifact_bytes(relative_path, label=label)
    observed = _sha256_bytes(payload)
    if observed != expected_sha256:
        raise GateValidationError(
            f"{label} SHA-256 mismatch: {observed} != {expected_sha256}"
        )
    parsed = _mapping(strict_json_loads(payload, label=label), label)
    return parsed, payload


def validate_empirical_prior_floor_audit() -> dict[str, Any]:
    """Live-read and validate the exact training-only floor-selection audit."""

    relative_path = Path(PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT["relative_path"])
    audit, _payload = load_pinned_json(
        relative_path,
        PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT["sha256"],
        label="empirical-prior floor audit",
    )
    if (
        canonical_json_sha256(audit)
        != PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_CANONICAL_SHA256
    ):
        raise GateValidationError(
            "empirical-prior floor audit canonical digest is unexpected"
        )
    if audit.get("schema_version") != 1:
        raise GateValidationError(
            "empirical-prior floor audit schema_version must equal 1"
        )
    if audit.get("purpose") != (
        "CPU-only empirical-UDLM uniform-floor training-data audit"
    ):
        raise GateValidationError("empirical-prior floor audit purpose is unexpected")
    if audit.get("claim_scope") != (
        "This audit measures unigram fit on ordered SAFE training blocks. It does "
        "not train a denoiser, generate or score molecules, rank generators, or "
        "establish that UDLM beats GenMol."
    ):
        raise GateValidationError(
            "empirical-prior floor audit claim scope is unexpected"
        )
    if dict(_mapping(audit.get("git"), "empirical-prior floor audit git")) != {
        "commit": PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT["source_revision"],
        "dirty": False,
        "upstream": PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT["source_revision"],
    }:
        raise GateValidationError(
            "empirical-prior floor audit source revision is unexpected"
        )
    if dict(
        _mapping(audit.get("data_use"), "empirical-prior floor audit data use")
    ) != {
        "exploratory_rows": [10_001, 20_000],
        "final_generation_seeds_or_metrics_used": False,
        "formal_preregistration_before_data_access": False,
        "prior_estimation_rows": [1, 10_000],
        "retrospective_replication_rows": [20_001, 30_000],
        "split": "training",
    }:
        raise GateValidationError(
            "empirical-prior floor audit data-use scope is unexpected"
        )
    recommendation = _mapping(
        audit.get("recommendation"), "empirical-prior floor audit recommendation"
    )
    if dict(recommendation) != {
        "block_optima": [0.00016593802382907556, 0.00019334112119092408],
        "both_block_optima_within_0_0001_to_0_0003": True,
        "candidate_minus_current_nll_by_block": [
            -0.00860566722454914,
            -0.008485138280406979,
        ],
        "candidate_nll_strictly_better_than_current_on_both_blocks": True,
        "candidate_uniform_mixture_weight": PILOT_EMPIRICAL_UNIFORM_MIX,
        "current_uniform_mixture_weight": 0.01,
        "qualification": (
            "The rule was formalized after exploratory inspection of these "
            "training blocks. It is suitable only as disclosed pilot "
            "hyperparameter engineering and is not confirmatory "
            "molecular-generation evidence."
        ),
        "recommended_uniform_mixture_weight": PILOT_EMPIRICAL_UNIFORM_MIX,
        "selection_rule": (
            "recommend 0.0002 only when both ordered-block continuous optima lie "
            "in [0.0001, 0.0003], both blocks contain tokens absent from the "
            "first-10000 prefix, and 0.0002 has lower unigram NLL than 0.01 on "
            "both; otherwise retain 0.01"
        ),
        "status": "training_only_retrospective_engineering_recommendation",
    }:
        raise GateValidationError(
            "empirical-prior floor audit recommendation is unexpected"
        )
    return {
        "relative_path": relative_path.as_posix(),
        "raw_sha256": PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT["sha256"],
        "canonical_sha256": PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_CANONICAL_SHA256,
        "source_revision": PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT["source_revision"],
        "selection_scope": PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT["scope"],
        "empirical_uniform_mix": PILOT_EMPIRICAL_UNIFORM_MIX,
    }


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if canonical_json_sha256(protocol) != PROTOCOL_CANONICAL_SHA256:
        raise GateValidationError(
            "superiority protocol content is not the frozen value"
        )
    if protocol.get("schema_version") != 3:
        raise GateValidationError("protocol schema_version must equal 3")
    if protocol.get("protocol_id") != EXPECTED_PROTOCOL_ID:
        raise GateValidationError("unexpected superiority protocol ID")
    if protocol.get("status") != (
        "frozen_after_failed_health_instrumentation_before_scientific_screens"
    ):
        raise GateValidationError("superiority protocol is not frozen")
    amendment = _mapping(protocol.get("amends"), "protocol.amends")
    if dict(amendment) != {
        "relative_path": PREVIOUS_PROTOCOL_RELATIVE_PATH.as_posix(),
        "raw_sha256": PREVIOUS_PROTOCOL_SHA256,
        "canonical_sha256": PREVIOUS_PROTOCOL_CANONICAL_SHA256,
        "reason": (
            "Advance the training-summary schema from 4 to 5 after a completed "
            "ten-update health-training process exposed Lightning's intended "
            "unranked ModelCheckpoint positive-infinity sentinel; require an "
            "exact checkpoint schema, structural sentinel validation, exact "
            "checkpoint/config and loop-progress bindings, exact live trainer/"
            "optimizer/scheduler/sampler/callback bindings, and an exact "
            "independent screen/receipt verifier without changing any scientific "
            "setting or threshold."
        ),
        "gpu_training_process_executed_before_amendment": True,
        "successful_health_panels_before_amendment": False,
        "failed_health_run_denoising_sampling_executed": False,
        "failed_health_run_molecular_scoring_executed": False,
        "registered_candidate_checkpoint_selection_or_ranking_executed_since_v2_freeze": False,
        "candidate_final_evaluation_run_executed_since_v2_freeze": False,
        "pre_v2_ineligible_cpu_smokes_and_audited_baseline_rescoring_remain_disclosed": True,
        "scientific_decision_thresholds_changed": False,
        "failed_health_instrumentation": {
            "run_id": ("health-w1-r-" "12bdce22809f9672dbb6666fa3a6e828b39aadb0"),
            "run_relative_path": (
                "output/udlm/health-w1-r-" "12bdce22809f9672dbb6666fa3a6e828b39aadb0"
            ),
            "source_revision": "12bdce22809f9672dbb6666fa3a6e828b39aadb0",
            "status": "failed_post_training_semantic_audit",
            "optimizer_updates_completed": 10,
            "training_summary_published": False,
            "successful_exit_receipt": False,
            "scientifically_eligible": False,
            "launch_manifest_sha256": (
                "ff0a946c8504b058574e082cadc14441295c4955352adc867e442521bf756f05"
            ),
            "runtime_config_sha256": (
                "14547b9be4772b1c78f5bc639e9d0d2ed17103daf6ea1e3dd8b4a346807bc24a"
            ),
            "checkpoint_sha256": (
                "f85230b09ddacd319082571b59773a9c3d26f61f27cfde8f206eae549cce4cb6"
            ),
            "failed_exit_receipt_sha256": (
                "5336a4791256910632412ae09eec9d20e223c22765bcfd8796d00408248d3b99"
            ),
            "training_log_sha256": (
                "6b6f14759c9d45cb6c0200ce188ccfdb912b91544912e9c01d07cf58a5aeef95"
            ),
            "diagnosis": (
                "The blanket checkpoint finiteness audit rejected Lightning "
                "2.5.1 ModelCheckpoint.kth_value=+inf even though every learned "
                "model, EMA, optimizer, and scheduler tensor was finite."
            ),
        },
    }:
        raise GateValidationError("superiority protocol amendment is unexpected")
    baseline = _mapping(protocol.get("baseline"), "protocol.baseline")
    if baseline.get("manifest_relative_path") != BASELINE_RELATIVE_PATH.as_posix():
        raise GateValidationError("protocol baseline path is unexpected")
    if baseline.get("manifest_sha256") != BASELINE_SHA256:
        raise GateValidationError("protocol baseline digest is unexpected")
    if (
        baseline.get("rescore_attestation_relative_path")
        != BASELINE_RESCORE_RELATIVE_PATH.as_posix()
    ):
        raise GateValidationError(
            "protocol baseline rescore-attestation path is unexpected"
        )
    if baseline.get("rescore_attestation_sha256") != BASELINE_RESCORE_SHA256:
        raise GateValidationError(
            "protocol baseline rescore-attestation digest is unexpected"
        )
    if (
        baseline.get("rescore_source_revision")
        != EXPECTED_BASELINE_RESCORE_SOURCE_REVISION
    ):
        raise GateValidationError("protocol baseline rescore revision is unexpected")
    if baseline.get("rescore_status") != "completed_exact_match":
        raise GateValidationError("protocol baseline rescore status is unexpected")
    if baseline.get("checkpoint_sha256") != EXPECTED_BASELINE_CHECKPOINT_SHA256:
        raise GateValidationError("protocol baseline checkpoint is unexpected")
    if baseline.get("metric_branch") != "released_comparable":
        raise GateValidationError("protocol must gate released-compatible metrics")

    final = _mapping(
        protocol.get("final_operating_point"), "protocol.final_operating_point"
    )
    expected_final = {
        "candidate_diffusion_type": "udlm",
        "generation_seeds": list(EXPECTED_SEEDS),
        "requested_samples_per_seed": EXPECTED_SAMPLES_PER_SEED,
        "seed_count": len(EXPECTED_SEEDS),
        "total_requested_samples": len(EXPECTED_SEEDS) * EXPECTED_SAMPLES_PER_SEED,
        "nfe": EXPECTED_NFE,
        "nfe_definition": "one full backbone forward evaluation per reverse step",
        "inference_weights": "ema",
        "metric_branch": "released_comparable",
        "strict_diagnostic_branch_required": True,
    }
    if dict(final) != expected_final:
        raise GateValidationError("protocol final operating point is unexpected")

    firewall = _mapping(
        protocol.get("selection_firewall"), "protocol.selection_firewall"
    )
    for key in (
        "final_seeds_forbidden_during_selection",
        "candidate_ledger_required",
        "candidate_ledger_must_disclose_all_pilot_attempts",
        "candidate_lock_must_be_a_git_blob_at_benchmark_revision",
        "candidate_must_be_selected_without_final_seed_results",
        "nonregistered_completed_pilots_are_disclosed_but_ineligible",
    ):
        _required_true(firewall.get(key), f"protocol.selection_firewall.{key}")
    if firewall.get("pilot_seed_minimum_inclusive") != 1000:
        raise GateValidationError("pilot seed firewall must begin at 1000")
    if firewall.get("final_attempts_per_seed") != 1:
        raise GateValidationError("final attempt count must be one per seed")
    if firewall.get("selection_rule") != CANDIDATE_SELECTION_RULE:
        raise GateValidationError("protocol pilot selection rule is unexpected")
    if firewall.get("checkpoint_selection_rule") != CHECKPOINT_SELECTION_RULE:
        raise GateValidationError("protocol checkpoint-selection rule is unexpected")
    if firewall.get("eligible_pilot_generation_seeds") != list(
        REGISTERED_SELECTION_PILOT_SEEDS
    ):
        raise GateValidationError("protocol eligible pilot seeds are unexpected")
    if (
        firewall.get("eligible_requested_samples_per_seed")
        != REGISTERED_SELECTION_SAMPLES_PER_SEED
    ):
        raise GateValidationError("protocol eligible pilot sample count is unexpected")
    if firewall.get("eligible_nfe") != REGISTERED_SELECTION_NFE:
        raise GateValidationError("protocol eligible pilot NFE is unexpected")
    if firewall.get("eligible_metric_branch") != REGISTERED_SELECTION_METRIC_BRANCH:
        raise GateValidationError("protocol eligible pilot metric branch is unexpected")

    point = _mapping(protocol.get("point_estimate_gates"), "point gates")
    uncertainty = _mapping(protocol.get("uncertainty_gates"), "uncertainty gates")
    if set(point) != set(METRICS):
        raise GateValidationError("point gates must define exactly four metrics")
    if uncertainty.get("confidence_level_one_sided") != 0.95:
        raise GateValidationError("uncertainty confidence level must be 0.95")
    if not math.isclose(
        _finite(uncertainty.get("normal_quantile"), "normal quantile"),
        1.6448536269514722,
        rel_tol=0,
        abs_tol=1e-15,
    ):
        raise GateValidationError("uncertainty normal quantile is unexpected")
    if uncertainty.get("multiple_metric_decision") != (
        "intersection_union_all_four_point_gates_and_all_four_interval_gates_must_pass"
    ):
        raise GateValidationError("protocol must require every registered gate")
    for metric in METRICS:
        _mapping(uncertainty.get(metric), f"uncertainty gate {metric}")
    lock_requirements = _mapping(
        protocol.get("candidate_lock_requirements"),
        "protocol.candidate_lock_requirements",
    )
    accepted_training_schemas = _mapping(
        lock_requirements.get("accepted_training_artifact_schema_versions"),
        "protocol accepted training artifact schemas",
    )
    if accepted_training_schemas != {
        "launch_manifest": LAUNCH_MANIFEST_SCHEMA_VERSION,
        "runtime_config": RUNTIME_CONFIG_SCHEMA_VERSION,
        "training_summary": TRAINING_SUMMARY_SCHEMA_VERSION,
        "successful_exit_receipt": PILOT_EXIT_STATUS_SCHEMA_VERSION,
    }:
        raise GateValidationError(
            "protocol accepted training artifact schemas are unexpected"
        )
    if (
        lock_requirements.get("candidate_lock_schema_version")
        != CANDIDATE_LOCK_SCHEMA_VERSION
    ):
        raise GateValidationError("protocol candidate-lock schema is unexpected")
    expected_selection_schemas = {
        "candidate_ledger_schema_version": CANDIDATE_LEDGER_SCHEMA_VERSION,
        "pilot_evidence_schema_version": PILOT_EVIDENCE_SCHEMA_VERSION,
        "pilot_failure_receipt_schema_version": PILOT_FAILURE_RECEIPT_SCHEMA_VERSION,
    }
    for field, expected in expected_selection_schemas.items():
        if lock_requirements.get(field) != expected:
            raise GateValidationError(f"protocol {field} is unexpected")
    for field in (
        "matched_r_s_e_registered_order_and_receipt_gated_advancement_machine_enforced",
        "predecessor_receipt_chain_is_machine_enforced_by_each_per_run_launch_manifest",
        "predecessor_artifacts_revalidated_unchanged_before_successful_exit_receipt",
        "predecessor_receipt_must_predate_successor_lock_and_gpu_probe",
        "terminal_e_successful_exit_receipt_hash_and_full_chain_required",
        "terminal_e_receipt_must_predate_candidate_lock",
        "selected_candidate_receipt_must_predate_candidate_lock",
        "selected_candidate_receipt_must_be_exact_member_of_terminal_r_s_e_chain",
        "terminal_e_and_selected_candidate_must_share_matched_panel",
        "candidate_ledger_winner_checkpoint_must_equal_locked_training_checkpoint",
        "per_seed_success_failure_outcomes_and_partial_failure_retention_required",
        "completed_pilot_summary_raw_receipt_hashes_required",
        "completed_pilot_training_receipt_full_validation_required",
        "pilot_quality_diversity_independently_recomputed_from_raw_model_text",
        "producer_authored_failure_receipt_command_source_log_and_partial_hashes_required",
        "pilot_failure_receipt_mode_and_requested_samples_required",
        "all_pilot_outcomes_must_predate_lock_and_source_revisions_be_ancestors",
        "selected_pilot_checkpoint_config_sampling_ema_source_metric_and_receipt_identity_must_equal_lock",
        "final_candidate_qed_sa_diversity_independently_recomputed_from_raw_model_text",
        "independent_rescore_source_and_dependency_hashes_required",
        "gate_report_rescore_launcher_writer_source_hashes_plus_scipy_version_required",
        "training_summary_checkpoint_audit_excludes_only_exact_live_bound_lightning_sentinel_required",
        "training_summary_checkpoint_exact_top_level_schema_and_nonfinite_python_numpy_rejection_required",
        "training_summary_checkpoint_hyperparameters_loop_progress_and_live_trainer_configuration_match_required",
        "training_summary_optimizer_scheduler_sampler_and_model_checkpoint_live_state_matches_required",
        "independent_optimization_screen_exact_closed_schema_and_cross_artifact_bindings_required",
        "failed_health_namespaces_are_immutable_and_never_eligible_for_candidate_selection",
    ):
        if lock_requirements.get(field) is not True:
            raise GateValidationError(f"protocol {field} must be true")
    expected_floor = {
        "empirical_uniform_mix": PILOT_EMPIRICAL_UNIFORM_MIX,
        "consumed_only_by_prior_variant": "empirical_frequency",
        "audit_relative_path": PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT["relative_path"],
        "audit_raw_sha256": PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT["sha256"],
        "audit_source_revision": PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT["source_revision"],
        "selection_scope": PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT["scope"],
    }
    if lock_requirements.get("audited_empirical_prior_floor") != expected_floor:
        raise GateValidationError(
            "protocol audited empirical-prior floor is unexpected"
        )
    validate_empirical_prior_floor_audit()
    previous_protocol, _previous_payload = load_pinned_json(
        PREVIOUS_PROTOCOL_RELATIVE_PATH,
        PREVIOUS_PROTOCOL_SHA256,
        label="previous superiority protocol",
    )
    if canonical_json_sha256(previous_protocol) != PREVIOUS_PROTOCOL_CANONICAL_SHA256:
        raise GateValidationError(
            "previous superiority protocol canonical digest is unexpected"
        )
    previous_lock_requirements = _mapping(
        previous_protocol.get("candidate_lock_requirements"),
        "previous protocol candidate-lock requirements",
    )
    expected_lock_requirements = json.loads(
        json.dumps(
            previous_lock_requirements,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    expected_lock_requirements["accepted_training_artifact_schema_versions"][
        "training_summary"
    ] = TRAINING_SUMMARY_SCHEMA_VERSION
    for field in (
        "training_summary_checkpoint_audit_excludes_only_exact_live_bound_lightning_sentinel_required",
        "training_summary_checkpoint_exact_top_level_schema_and_nonfinite_python_numpy_rejection_required",
        "training_summary_checkpoint_hyperparameters_loop_progress_and_live_trainer_configuration_match_required",
        "training_summary_optimizer_scheduler_sampler_and_model_checkpoint_live_state_matches_required",
        "independent_optimization_screen_exact_closed_schema_and_cross_artifact_bindings_required",
        "failed_health_namespaces_are_immutable_and_never_eligible_for_candidate_selection",
    ):
        expected_lock_requirements[field] = True
    if dict(lock_requirements) != expected_lock_requirements:
        raise GateValidationError(
            "v3 candidate-lock requirements differ beyond the registered "
            "schema-5 instrumentation amendment"
        )
    for field in (
        "primary_claim",
        "baseline",
        "final_operating_point",
        "selection_firewall",
        "point_estimate_gates",
        "uncertainty_gates",
        "decision",
        "claim_boundaries",
    ):
        if protocol.get(field) != previous_protocol.get(field):
            raise GateValidationError(
                f"v3 unexpectedly changes scientific protocol field {field}"
            )


def _sample_sd(values: Sequence[float]) -> float:
    if len(values) != 3:
        raise GateValidationError("registered summaries require exactly three seeds")
    return statistics.stdev(values)


def _close(actual: object, expected: float, label: str) -> None:
    value = _finite(actual, label)
    if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise GateValidationError(f"{label}={value} disagrees with {expected}")


def _ordered_seed_rows(value: object, *, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or len(value) != len(EXPECTED_SEEDS):
        raise GateValidationError(f"{label} must contain exactly three rows")
    by_seed: dict[int, Mapping[str, Any]] = {}
    for index, raw_row in enumerate(value):
        row = _mapping(raw_row, f"{label} row {index}")
        seed = row.get("seed")
        if type(seed) is not int or seed in by_seed:
            raise GateValidationError(f"{label} seeds must be unique integers")
        by_seed[seed] = row
    if tuple(sorted(by_seed)) != EXPECTED_SEEDS:
        raise GateValidationError(
            f"{label} seeds must be exactly {list(EXPECTED_SEEDS)}"
        )
    return [by_seed[seed] for seed in EXPECTED_SEEDS]


def validate_baseline_manifest(baseline: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the compact comparator's counts, means, and sample SDs."""

    if baseline.get("schema_version") != 1:
        raise GateValidationError("baseline manifest schema_version must equal 1")
    checkpoint = _mapping(baseline.get("checkpoint"), "baseline.checkpoint")
    if checkpoint.get("sha256") != EXPECTED_BASELINE_CHECKPOINT_SHA256:
        raise GateValidationError("baseline checkpoint digest is unexpected")
    if checkpoint.get("diffusion_type") != "mdlm":
        raise GateValidationError("baseline must be MDLM")
    baseline_protocol = _mapping(baseline.get("protocol"), "baseline.protocol")
    if baseline_protocol.get("seeds") != list(EXPECTED_SEEDS):
        raise GateValidationError("baseline seeds are unexpected")
    if baseline_protocol.get("samples_per_seed") != EXPECTED_SAMPLES_PER_SEED:
        raise GateValidationError("baseline sample count is unexpected")
    if baseline_protocol.get("total_requested_samples") != 3_000:
        raise GateValidationError("baseline total sample count is unexpected")

    branch = _mapping(
        baseline.get("released_comparable"), "baseline.released_comparable"
    )
    ordered = _ordered_seed_rows(branch.get("per_seed"), label="baseline per-seed rows")
    values: dict[str, list[float]] = {metric: [] for metric in METRICS}
    pooled_valid = 0
    pooled_requested = 0
    for expected_seed, row in zip(EXPECTED_SEEDS, ordered, strict=True):
        if row.get("seed") != expected_seed:
            raise GateValidationError("baseline seed rows are incomplete")
        requested = _integer(row.get("requested"), "baseline requested", minimum=1)
        if requested != EXPECTED_SAMPLES_PER_SEED:
            raise GateValidationError("baseline per-seed request count is unexpected")
        valid_count = _integer(
            row.get("valid_count"), "baseline valid count", minimum=0
        )
        unique_count = _integer(
            row.get("unique_count"), "baseline unique count", minimum=0
        )
        quality_count = _integer(
            row.get("quality_count"), "baseline quality count", minimum=0
        )
        if not 0 <= quality_count <= unique_count <= valid_count <= requested:
            raise GateValidationError("baseline count funnel is inconsistent")
        expected_values = {
            "validity": valid_count / requested,
            "uniqueness": unique_count / valid_count if valid_count else math.nan,
            "quality": quality_count / requested,
        }
        for metric, expected in expected_values.items():
            _close(row.get(metric), expected, f"baseline seed {expected_seed} {metric}")
        diversity = _finite(row.get("diversity"), "baseline diversity")
        if not 0 <= diversity <= 1:
            raise GateValidationError("baseline diversity must lie in [0, 1]")
        for metric in METRICS:
            values[metric].append(float(row[metric]))
        _sha256(row.get("raw_samples_sha256"), "baseline raw CSV digest")
        _sha256(row.get("summary_sha256"), "baseline summary digest")
        pooled_valid += valid_count
        pooled_requested += requested

    means = _mapping(branch.get("mean"), "baseline means")
    sample_sds = _mapping(branch.get("sample_sd"), "baseline sample SDs")
    for metric in METRICS:
        expected_mean = statistics.fmean(values[metric])
        expected_sd = _sample_sd(values[metric])
        _close(means.get(metric), expected_mean, f"baseline mean {metric}")
        _close(sample_sds.get(metric), expected_sd, f"baseline sample SD {metric}")
    return {
        "values": values,
        "means": {
            metric: statistics.fmean(series) for metric, series in values.items()
        },
        "sample_sds": {metric: _sample_sd(series) for metric, series in values.items()},
        "pooled_valid": pooled_valid,
        "pooled_requested": pooled_requested,
        "checkpoint": dict(checkpoint),
    }


_BASELINE_READ_POLICY = (
    "regular_file_no_symlink_stable_descriptor_bytes_retained_in_memory"
)
_BASELINE_OFFLINE_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "NVIDIA_VISIBLE_DEVICES": "",
    "TRANSFORMERS_OFFLINE": "1",
    "WANDB_DISABLED": "true",
    "WANDB_MODE": "offline",
}
_BASELINE_WORKER_OFFLINE_ENVIRONMENT = {
    name: value
    for name, value in _BASELINE_OFFLINE_ENVIRONMENT.items()
    if name not in {"CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"}
}
_BASELINE_GUARDED_NETWORK_APIS = [
    "socket.create_connection",
    "socket.getaddrinfo",
    "socket.socket.connect",
    "socket.socket.connect_ex",
]
_BASELINE_NETWORK_SCOPE_LIMITATION = (
    "Python-runtime guard only; this is not OS-level or process-level network "
    "isolation and does not claim to block native extensions, subprocesses, raw "
    "sockets, or any other unguarded socket, name-resolution, or datagram API"
)
_BASELINE_SOURCE_FILES = {
    "benchmark_runner": (
        "scripts/exps/denovo/benchmark.py",
        "scripts.exps.denovo.benchmark",
    ),
    "chemistry_utils_module": (
        "src/genmol/utils/utils_chem.py",
        "genmol.utils.utils_chem",
    ),
    "report_validator": (
        "scripts/exps/denovo/report.py",
        "scripts.exps.denovo.report",
    ),
    "rescore_runner": (
        "scripts/udlm/rescore_mdlm_baseline.py",
        "scripts.udlm.rescore_mdlm_baseline",
    ),
}
_BASELINE_FAILURE_FIELDS = {
    "raw_safe_conversion_failed",
    "released_decode_failed",
    "released_duplicates",
    "released_largest_component_applied",
    "released_recovered_strict_failure",
    "strict_decode_failed",
    "strict_duplicates",
    "strict_valid_but_released_failed",
}


def _validate_attested_metric_branch(
    value: object,
    *,
    expected: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    branch = _mapping(value, label)
    _exact_keys(
        branch,
        {
            "definition",
            "diversity",
            "diversity_input_count",
            "diversity_undefined_reason",
            "quality",
            "quality_count",
            "quality_denominator",
            "quality_thresholds",
            "unique_count",
            "uniqueness",
            "uniqueness_denominator",
            "valid_count",
            "validity",
            "validity_denominator",
        },
        label,
    )
    if not isinstance(branch.get("definition"), str) or not branch["definition"]:
        raise GateValidationError(f"{label}.definition must be nonempty")
    if branch.get("diversity_undefined_reason") is not None:
        raise GateValidationError(f"{label}.diversity must be defined")
    if branch.get("quality_thresholds") != {
        "qed_min_inclusive": 0.6,
        "sa_max_inclusive": 4.0,
    }:
        raise GateValidationError(f"{label}.quality thresholds are unexpected")

    requested = _integer(expected.get("requested"), f"{label}.expected requested")
    expected_counts = {
        "valid_count": expected["valid_count"],
        "unique_count": expected["unique_count"],
        "quality_count": expected["quality_count"],
        "validity_denominator": requested,
        "uniqueness_denominator": expected["valid_count"],
        "quality_denominator": requested,
        "diversity_input_count": expected["unique_count"],
    }
    for name, expected_value in expected_counts.items():
        actual = _integer(branch.get(name), f"{label}.{name}", minimum=0)
        if actual != expected_value:
            raise GateValidationError(
                f"{label}.{name}={actual} disagrees with frozen manifest "
                f"value {expected_value}"
            )
    for metric in METRICS:
        _close(branch.get(metric), float(expected[metric]), f"{label}.{metric}")
    return dict(branch)


def _validate_attested_seed(
    row: Mapping[str, Any],
    *,
    seed: int,
    baseline_manifest: Mapping[str, Any],
    source_files: Mapping[str, Mapping[str, Any]],
    metric_inputs_sha256: str,
    row_fields: Sequence[str],
) -> dict[str, Any]:
    label = f"baseline rescore seed {seed}"
    if row.get("seed") != seed or row.get("status") != "exact_match":
        raise GateValidationError(f"{label} is not an exact-match result")

    inputs = _mapping(row.get("inputs"), f"{label}.inputs")
    raw_input = _mapping(inputs.get("raw_samples_csv"), f"{label}.raw input")
    summary_input = _mapping(inputs.get("summary_json"), f"{label}.summary input")
    expected_released = baseline_manifest["released_comparable"]["per_seed"][seed]
    expected_strict = baseline_manifest["strict"]["per_seed"][seed]
    for artifact, expected_sha, artifact_label in (
        (raw_input, expected_released["raw_samples_sha256"], "raw samples"),
        (summary_input, expected_released["summary_sha256"], "summary"),
    ):
        if artifact.get("read_policy") != _BASELINE_READ_POLICY:
            raise GateValidationError(f"{label} {artifact_label} read policy changed")
        if artifact.get("sha256") != expected_sha:
            raise GateValidationError(
                f"{label} {artifact_label} digest disagrees with frozen manifest"
            )
        _integer(
            artifact.get("size_bytes"), f"{label} {artifact_label} size", minimum=1
        )
    if (
        summary_input.get("schema_version") != 2
        or summary_input.get("mutation_policy") != "read_only_never_rewritten"
    ):
        raise GateValidationError(f"{label} legacy summary policy is unexpected")

    comparison = _mapping(row.get("row_comparison"), f"{label}.row comparison")
    _required_true(comparison.get("all_match"), f"{label}.row comparison all_match")
    if (
        comparison.get("row_count") != EXPECTED_SAMPLES_PER_SEED
        or comparison.get("field_count") != EXPECTED_BASELINE_RAW_SAMPLE_FIELD_COUNT
        or comparison.get("cell_count")
        != EXPECTED_SAMPLES_PER_SEED * EXPECTED_BASELINE_RAW_SAMPLE_FIELD_COUNT
        or comparison.get("numeric_absolute_tolerance") != 1e-12
    ):
        raise GateValidationError(f"{label} row-comparison dimensions are unexpected")
    field_results = comparison.get("field_results")
    if not isinstance(field_results, list) or len(field_results) != len(row_fields):
        raise GateValidationError(f"{label} field comparisons are incomplete")
    numeric_fields = {"strict_qed", "strict_sa", "released_qed", "released_sa"}
    for expected_field, raw_result in zip(row_fields, field_results, strict=True):
        result = _mapping(raw_result, f"{label}.{expected_field} comparison")
        _exact_keys(
            result,
            {
                "compared_rows",
                "comparison",
                "field",
                "max_absolute_difference",
                "mismatch_count",
            },
            f"{label}.{expected_field} comparison",
        )
        expected_comparison = (
            "finite_numeric_absolute_tolerance_1e-12_or_exact_null"
            if expected_field in numeric_fields
            else "exact_value_and_type"
        )
        if (
            result.get("field") != expected_field
            or result.get("compared_rows") != EXPECTED_SAMPLES_PER_SEED
            or result.get("comparison") != expected_comparison
            or result.get("mismatch_count") != 0
        ):
            raise GateValidationError(f"{label}.{expected_field} comparison failed")
        expected_max = 0.0 if expected_field in numeric_fields else None
        if result.get("max_absolute_difference") != expected_max:
            raise GateValidationError(
                f"{label}.{expected_field} comparison tolerance result changed"
            )

    metrics = _mapping(row.get("metrics"), f"{label}.metrics")
    _exact_keys(metrics, {"released_comparable", "strict"}, f"{label}.metrics")
    released = _validate_attested_metric_branch(
        metrics["released_comparable"],
        expected=expected_released,
        label=f"{label}.released_comparable",
    )
    strict = _validate_attested_metric_branch(
        metrics["strict"], expected=expected_strict, label=f"{label}.strict"
    )

    failures = _mapping(row.get("failure_counts"), f"{label}.failure counts")
    _exact_keys(failures, _BASELINE_FAILURE_FIELDS, f"{label}.failure counts")
    normalized_failures = {
        name: _integer(value, f"{label}.failure_counts.{name}", minimum=0)
        for name, value in failures.items()
    }
    derived_failures = {
        "strict_decode_failed": EXPECTED_SAMPLES_PER_SEED - strict["valid_count"],
        "released_decode_failed": (EXPECTED_SAMPLES_PER_SEED - released["valid_count"]),
        "strict_duplicates": strict["valid_count"] - strict["unique_count"],
        "released_duplicates": (released["valid_count"] - released["unique_count"]),
        "released_recovered_strict_failure": (
            released["valid_count"] - strict["valid_count"]
        ),
        "raw_safe_conversion_failed": 0,
        "strict_valid_but_released_failed": 0,
    }
    for name, expected_value in derived_failures.items():
        if normalized_failures[name] != expected_value:
            raise GateValidationError(f"{label}.{name} is inconsistent with counts")

    validator_counts = _mapping(
        row.get("current_report_validator_counts"), f"{label}.validator counts"
    )
    expected_validator_counts = {
        "released_comparable": {
            name: released[name]
            for name in ("valid_count", "unique_count", "quality_count")
        },
        "strict": {
            name: strict[name]
            for name in ("valid_count", "unique_count", "quality_count")
        },
        "cross_branch": {
            name: normalized_failures[name]
            for name in (
                "raw_safe_conversion_failed",
                "released_largest_component_applied",
                "released_recovered_strict_failure",
                "strict_valid_but_released_failed",
            )
        },
    }
    if validator_counts != expected_validator_counts:
        raise GateValidationError(f"{label} current-validator counts disagree")

    if row.get("metric_inputs_sha256") != metric_inputs_sha256:
        raise GateValidationError(f"{label} metric-input digest differs")
    manifest_comparison = _mapping(
        row.get("manifest_comparison"), f"{label}.manifest comparison"
    )
    _required_true(
        manifest_comparison.get("all_counts_metrics_and_artifact_hashes_match"),
        f"{label}.manifest comparison",
    )

    source = _mapping(row.get("source_verification"), f"{label}.source")
    _required_true(
        source.get("clean_pushed_before_and_after_computation"),
        f"{label}.clean pushed source",
    )
    for phase in ("before", "after"):
        identity = _mapping(source.get(phase), f"{label}.source.{phase}")
        if identity != {
            "head": EXPECTED_BASELINE_RESCORE_SOURCE_REVISION,
            "upstream": EXPECTED_BASELINE_RESCORE_SOURCE_REVISION,
        }:
            raise GateValidationError(f"{label} {phase} source identity changed")
    expected_source_hashes = {
        name: source_files[name]["sha256"]
        for name in ("benchmark_runner", "report_validator", "rescore_runner")
    }
    if source.get("expected_file_sha256") != expected_source_hashes:
        raise GateValidationError(f"{label} source-file hashes disagree")

    environment = _mapping(row.get("environment"), f"{label}.environment")
    if (
        environment.get("device") != "cpu"
        or environment.get("cuda_visible_devices") != ""
        or environment.get("nvidia_visible_devices") != ""
        or environment.get("python_hash_seed") != str(seed)
        or environment.get("offline_environment")
        != _BASELINE_WORKER_OFFLINE_ENVIRONMENT
    ):
        raise GateValidationError(f"{label} worker isolation environment changed")
    if environment.get("python_network_guard_during_computation") != {
        "guarded_apis": _BASELINE_GUARDED_NETWORK_APIS,
        "scope_limitation": _BASELINE_NETWORK_SCOPE_LIMITATION,
    }:
        raise GateValidationError(f"{label} Python network-guard claim is unexpected")
    return {
        "metrics": {"released_comparable": released, "strict": strict},
        "failure_counts": normalized_failures,
    }


def validate_baseline_rescore_attestation(
    attestation: Mapping[str, Any], baseline_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the immutable current-code rescore against the frozen MDLM rows."""

    validate_baseline_manifest(baseline_manifest)
    if (
        attestation.get("schema_version") != 1
        or attestation.get("status") != "completed_exact_match"
    ):
        raise GateValidationError(
            "baseline rescore attestation is not completed exact-match schema 1"
        )

    inputs = _mapping(attestation.get("inputs"), "baseline rescore inputs")
    frozen_manifest = _mapping(
        inputs.get("frozen_manifest"), "baseline rescore frozen manifest"
    )
    if (
        frozen_manifest.get("schema_version") != 1
        or frozen_manifest.get("sha256") != BASELINE_SHA256
        or frozen_manifest.get("read_policy") != _BASELINE_READ_POLICY
    ):
        raise GateValidationError("baseline rescore frozen-manifest identity changed")
    historical = _mapping(
        inputs.get("historical_source_aggregate"),
        "baseline rescore historical aggregate",
    )
    baseline_source = _mapping(
        baseline_manifest.get("source_aggregate"), "baseline source aggregate"
    )
    if (
        historical.get("schema_version") != baseline_source.get("schema_version")
        or historical.get("sha256") != baseline_source.get("sha256")
        or historical.get("historical_runner_sha256")
        != baseline_source.get("runner_sha256")
        or historical.get("read_policy") != _BASELINE_READ_POLICY
    ):
        raise GateValidationError("baseline rescore historical aggregate is unbound")

    source = _mapping(attestation.get("source"), "baseline rescore source")
    if (
        source.get("revision") != EXPECTED_BASELINE_RESCORE_SOURCE_REVISION
        or source.get("upstream") != EXPECTED_BASELINE_RESCORE_SOURCE_REVISION
    ):
        raise GateValidationError("baseline rescore source revision is unexpected")
    clean_checks = _mapping(
        source.get("clean_pushed_checks"), "baseline rescore clean-source checks"
    )
    _exact_keys(
        clean_checks,
        {"before_computation", "after_computation", "immediately_before_publication"},
        "baseline rescore clean-source checks",
    )
    for name, value in clean_checks.items():
        _required_true(value, f"baseline rescore clean-source check {name}")

    raw_source_files = _mapping(source.get("files"), "baseline rescore source files")
    _exact_keys(
        raw_source_files, set(_BASELINE_SOURCE_FILES), "baseline rescore source files"
    )
    source_files: dict[str, Mapping[str, Any]] = {}
    for name, (expected_path, _module_name) in _BASELINE_SOURCE_FILES.items():
        evidence = _mapping(raw_source_files[name], f"baseline source file {name}")
        if (
            evidence.get("relative_path") != expected_path
            or evidence.get("source_revision")
            != EXPECTED_BASELINE_RESCORE_SOURCE_REVISION
        ):
            raise GateValidationError(f"baseline source file {name} identity changed")
        _sha256(evidence.get("sha256"), f"baseline source file {name} digest")
        _integer(
            evidence.get("size_bytes"), f"baseline source file {name} size", minimum=1
        )
        _required_true(
            evidence.get("stable_bytes_verified"), f"baseline source file {name} stable"
        )
        _required_true(
            evidence.get("tracked_at_source_revision"),
            f"baseline source file {name} tracked",
        )
        source_files[name] = evidence

    implementation = _mapping(
        attestation.get("implementation"), "baseline rescore implementation"
    )
    if (
        implementation.get("benchmark_schema_version")
        != EXPECTED_BASELINE_BENCHMARK_SCHEMA_VERSION
        or implementation.get("report_schema_version")
        != EXPECTED_BASELINE_REPORT_SCHEMA_VERSION
        or implementation.get("raw_sample_field_count")
        != EXPECTED_BASELINE_RAW_SAMPLE_FIELD_COUNT
    ):
        raise GateValidationError("baseline rescore implementation schemas are stale")
    metric_inputs = _mapping(
        implementation.get("metric_inputs"), "baseline rescore metric inputs"
    )
    metric_inputs_sha = _sha256(
        implementation.get("metric_inputs_sha256"),
        "baseline rescore metric-input digest",
    )
    if canonical_json_sha256(metric_inputs) != metric_inputs_sha:
        raise GateValidationError("baseline rescore metric-input self-hash differs")
    sa_policy = _mapping(
        metric_inputs.get("sa_loading_policy"), "baseline rescore SA loading policy"
    )
    if (
        metric_inputs.get("schema_version") != 1
        or sa_policy.get("network_download_allowed") is not False
        or sa_policy.get("resident_scores_loaded_from_verified_bytes") is not True
        or sa_policy.get("tdc_oracle_load_invoked") is not False
    ):
        raise GateValidationError("baseline rescore metric-input policy is unsafe")
    sa_scores = _mapping(
        metric_inputs.get("sa_fragment_scores"), "baseline rescore SA scores"
    )
    _sha256(sa_scores.get("sha256"), "baseline rescore SA-score digest")
    tdc_implementation = _mapping(
        metric_inputs.get("tdc_metric_implementation"),
        "baseline rescore TDC implementation",
    )
    if tdc_implementation.get("version") != "0.4.1":
        raise GateValidationError("baseline rescore TDC version changed")

    runtime_modules = _mapping(
        implementation.get("runtime_modules"), "baseline rescore runtime modules"
    )
    runtime_modules_sha = _sha256(
        implementation.get("runtime_modules_sha256"),
        "baseline rescore runtime-module digest",
    )
    if canonical_json_sha256(runtime_modules) != runtime_modules_sha:
        raise GateValidationError("baseline rescore runtime-module self-hash differs")
    for source_name, (
        _relative_path_value,
        module_name,
    ) in _BASELINE_SOURCE_FILES.items():
        module = _mapping(
            runtime_modules.get(module_name), f"baseline runtime module {module_name}"
        )
        if (
            module.get("module") != module_name
            or module.get("read_policy") != _BASELINE_READ_POLICY
            or module.get("sha256") != source_files[source_name]["sha256"]
        ):
            raise GateValidationError(
                f"baseline runtime module {module_name} is unbound from source"
            )

    rescore_protocol = _mapping(
        attestation.get("protocol"), "baseline rescore protocol"
    )
    row_fields = rescore_protocol.get("row_fields_compared")
    if (
        rescore_protocol.get("seeds") != list(EXPECTED_SEEDS)
        or rescore_protocol.get("one_fresh_interpreter_per_seed") is not True
        or rescore_protocol.get("python_hash_seed_equals_seed") is not True
        or rescore_protocol.get("device") != "cpu"
        or rescore_protocol.get("cuda_visible_devices") != ""
        or rescore_protocol.get("use_bracket_safe") is not False
        or not isinstance(row_fields, list)
        or len(row_fields) != EXPECTED_BASELINE_RAW_SAMPLE_FIELD_COUNT
        or len(set(row_fields)) != len(row_fields)
    ):
        raise GateValidationError("baseline rescore execution protocol changed")
    network = _mapping(
        rescore_protocol.get("network_controls"), "baseline rescore network controls"
    )
    if network != {
        "offline_environment_variables": _BASELINE_OFFLINE_ENVIRONMENT,
        "os_or_process_network_isolation": False,
        "python_runtime_guarded_apis": _BASELINE_GUARDED_NETWORK_APIS,
        "scope_limitation": _BASELINE_NETWORK_SCOPE_LIMITATION,
    }:
        raise GateValidationError("baseline rescore network-control claim is dishonest")
    environment = _mapping(
        attestation.get("environment"), "baseline rescore parent environment"
    )
    if environment.get("worker_policy") != _BASELINE_OFFLINE_ENVIRONMENT:
        raise GateValidationError("baseline rescore parent worker policy changed")

    ordered_seed_rows = _ordered_seed_rows(
        attestation.get("seed_results"), label="baseline rescore seed results"
    )
    normalized_seed_rows = [
        _validate_attested_seed(
            row,
            seed=seed,
            baseline_manifest=baseline_manifest,
            source_files=source_files,
            metric_inputs_sha256=metric_inputs_sha,
            row_fields=row_fields,
        )
        for seed, row in zip(EXPECTED_SEEDS, ordered_seed_rows, strict=True)
    ]

    aggregate = _mapping(
        attestation.get("aggregate_metrics"), "baseline rescore aggregates"
    )
    for branch_name in ("released_comparable", "strict"):
        branch = _mapping(
            aggregate.get(branch_name), f"rescore aggregate {branch_name}"
        )
        for metric in METRICS:
            metric_summary = _mapping(
                branch.get(metric), f"rescore aggregate {branch_name}.{metric}"
            )
            values = [
                float(row["metrics"][branch_name][metric])
                for row in normalized_seed_rows
            ]
            if metric_summary.get("values_by_seed") != [
                {"seed": seed, "value": value}
                for seed, value in zip(EXPECTED_SEEDS, values, strict=True)
            ]:
                raise GateValidationError(
                    f"rescore aggregate {branch_name}.{metric} seed values differ"
                )
            _close(
                metric_summary.get("mean"),
                statistics.fmean(values),
                f"rescore aggregate {branch_name}.{metric} mean",
            )
            _close(
                metric_summary.get("sample_sd"),
                statistics.stdev(values),
                f"rescore aggregate {branch_name}.{metric} sample SD",
            )
            _close(
                metric_summary.get("mean"),
                baseline_manifest[branch_name]["mean"][metric],
                f"rescore versus manifest {branch_name}.{metric} mean",
            )
            _close(
                metric_summary.get("sample_sd"),
                baseline_manifest[branch_name]["sample_sd"][metric],
                f"rescore versus manifest {branch_name}.{metric} sample SD",
            )

    recomputed_funnel = {
        "requested": len(EXPECTED_SEEDS) * EXPECTED_SAMPLES_PER_SEED,
        "strict_valid": sum(
            row["metrics"]["strict"]["valid_count"] for row in normalized_seed_rows
        ),
        "strict_unique_within_seed": sum(
            row["metrics"]["strict"]["unique_count"] for row in normalized_seed_rows
        ),
        "strict_quality": sum(
            row["metrics"]["strict"]["quality_count"] for row in normalized_seed_rows
        ),
        "released_valid": sum(
            row["metrics"]["released_comparable"]["valid_count"]
            for row in normalized_seed_rows
        ),
        "released_unique_within_seed": sum(
            row["metrics"]["released_comparable"]["unique_count"]
            for row in normalized_seed_rows
        ),
        "released_quality": sum(
            row["metrics"]["released_comparable"]["quality_count"]
            for row in normalized_seed_rows
        ),
        "released_recovered_strict_failure": sum(
            row["failure_counts"]["released_recovered_strict_failure"]
            for row in normalized_seed_rows
        ),
        "released_largest_component_applied": sum(
            row["failure_counts"]["released_largest_component_applied"]
            for row in normalized_seed_rows
        ),
    }
    if attestation.get(
        "strict_vs_repaired_funnel"
    ) != recomputed_funnel or recomputed_funnel != baseline_manifest.get(
        "strict_vs_repaired_funnel"
    ):
        raise GateValidationError("baseline rescore funnel disagrees with manifest")
    manifest_comparison = _mapping(
        attestation.get("manifest_comparison"),
        "baseline rescore manifest comparison",
    )
    if manifest_comparison != {
        "all_seed_rows_metrics_failures_hashes_and_aggregates_match": True,
        "manifest_sha256": BASELINE_SHA256,
    }:
        raise GateValidationError("baseline rescore manifest comparison is incomplete")
    if canonical_json_sha256(attestation) != BASELINE_RESCORE_CANONICAL_SHA256:
        raise GateValidationError(
            "baseline rescore attestation content is not the frozen value"
        )
    return {
        "relative_path": BASELINE_RESCORE_RELATIVE_PATH.as_posix(),
        "sha256": BASELINE_RESCORE_SHA256,
        "source_revision": EXPECTED_BASELINE_RESCORE_SOURCE_REVISION,
        "status": "completed_exact_match",
        "manifest_sha256": BASELINE_SHA256,
        "benchmark_schema_version": EXPECTED_BASELINE_BENCHMARK_SCHEMA_VERSION,
        "report_schema_version": EXPECTED_BASELINE_REPORT_SCHEMA_VERSION,
        "metric_inputs_sha256": metric_inputs_sha,
        "runtime_modules_sha256": runtime_modules_sha,
        "network_controls": {
            "os_or_process_network_isolation": False,
            "python_runtime_guarded_apis": list(_BASELINE_GUARDED_NETWORK_APIS),
            "scope_limitation": _BASELINE_NETWORK_SCOPE_LIMITATION,
        },
        "seed_count": len(normalized_seed_rows),
        "rows_compared_per_seed": EXPECTED_SAMPLES_PER_SEED,
        "fields_compared_per_row": EXPECTED_BASELINE_RAW_SAMPLE_FIELD_COUNT,
        "all_rows_and_manifest_values_exact_match": True,
    }


def wilson_score_interval(
    successes: int, trials: int, *, z: float
) -> tuple[float, float]:
    """Return the score interval used by Newcombe's independent-proportion CI."""

    successes = _integer(successes, "successes", minimum=0)
    trials = _integer(trials, "trials", minimum=1)
    z = _finite(z, "z")
    if successes > trials or z <= 0:
        raise GateValidationError("Wilson inputs are outside their valid range")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, centre - radius), min(1.0, centre + radius)


def newcombe_wilson_lower_difference(
    candidate_successes: int,
    candidate_trials: int,
    baseline_successes: int,
    baseline_trials: int,
    *,
    z: float,
) -> dict[str, Any]:
    """One-sided Newcombe method-10 lower bound for independent proportions."""

    candidate_lower, candidate_upper = wilson_score_interval(
        candidate_successes, candidate_trials, z=z
    )
    baseline_lower, baseline_upper = wilson_score_interval(
        baseline_successes, baseline_trials, z=z
    )
    candidate_rate = candidate_successes / candidate_trials
    baseline_rate = baseline_successes / baseline_trials
    difference = candidate_rate - baseline_rate
    lower = difference - math.hypot(
        candidate_rate - candidate_lower,
        baseline_upper - baseline_rate,
    )
    return {
        "method": "newcombe_wilson_method_10_independent_proportions",
        "candidate_successes": candidate_successes,
        "candidate_trials": candidate_trials,
        "baseline_successes": baseline_successes,
        "baseline_trials": baseline_trials,
        "candidate_rate": candidate_rate,
        "baseline_rate": baseline_rate,
        "difference": difference,
        "candidate_wilson_interval": [candidate_lower, candidate_upper],
        "baseline_wilson_interval": [baseline_lower, baseline_upper],
        "lower_bound": lower,
        "z": z,
    }


def welch_lower_difference(
    candidate_values: Sequence[float],
    baseline_values: Sequence[float],
    *,
    confidence: float,
) -> dict[str, Any]:
    """One-sided unpaired Welch lower bound on candidate minus baseline mean."""

    candidate = [_finite(value, "candidate seed value") for value in candidate_values]
    baseline = [_finite(value, "baseline seed value") for value in baseline_values]
    if len(candidate) < 2 or len(baseline) < 2:
        raise GateValidationError(
            "Welch intervals require at least two values per method"
        )
    confidence = _finite(confidence, "confidence")
    if not 0.5 < confidence < 1.0:
        raise GateValidationError("one-sided confidence must lie in (0.5, 1)")
    candidate_mean = statistics.fmean(candidate)
    baseline_mean = statistics.fmean(baseline)
    candidate_variance = statistics.variance(candidate)
    baseline_variance = statistics.variance(baseline)
    candidate_component = candidate_variance / len(candidate)
    baseline_component = baseline_variance / len(baseline)
    standard_error_squared = candidate_component + baseline_component
    difference = candidate_mean - baseline_mean
    if standard_error_squared == 0.0:
        raise GateValidationError(
            "Welch interval is undefined when both sample variances are zero"
        )
    denominator = candidate_component * candidate_component / (
        len(candidate) - 1
    ) + baseline_component * baseline_component / (len(baseline) - 1)
    if denominator <= 0:
        raise GateValidationError("Welch degrees of freedom are undefined")
    degrees_of_freedom = standard_error_squared**2 / denominator
    from scipy.stats import t as student_t

    critical_value = float(student_t.ppf(confidence, degrees_of_freedom))
    if not math.isfinite(critical_value):
        raise GateValidationError("Welch critical value is non-finite")
    lower_bound = difference - critical_value * math.sqrt(standard_error_squared)
    return {
        "method": "welch_t_independent_seed_level_estimates",
        "candidate_values": candidate,
        "baseline_values": baseline,
        "candidate_mean": candidate_mean,
        "baseline_mean": baseline_mean,
        "difference": difference,
        "candidate_sample_sd": math.sqrt(candidate_variance),
        "baseline_sample_sd": math.sqrt(baseline_variance),
        "standard_error": math.sqrt(standard_error_squared),
        "degrees_of_freedom": degrees_of_freedom,
        "critical_value": critical_value,
        "confidence_level_one_sided": confidence,
        "lower_bound": lower_bound,
    }


def validate_candidate_lock(
    candidate_lock: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the immutable pre-final candidate declaration."""

    _exact_keys(
        candidate_lock,
        {
            "schema_version",
            "candidate_id",
            "status",
            "locked_at_utc",
            "protocol",
            "selection",
            "training",
            "inference",
            "analysis",
            "claim_scope",
        },
        "candidate lock",
    )
    if candidate_lock.get("schema_version") != CANDIDATE_LOCK_SCHEMA_VERSION:
        raise GateValidationError(
            "candidate lock schema_version must equal "
            f"{CANDIDATE_LOCK_SCHEMA_VERSION}"
        )
    candidate_id = candidate_lock.get("candidate_id")
    if (
        not isinstance(candidate_id, str)
        or re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,95}", candidate_id) is None
    ):
        raise GateValidationError("candidate_id has invalid syntax")
    if candidate_lock.get("status") != "locked_before_final_evaluation":
        raise GateValidationError("candidate is not locked before final evaluation")
    locked_at = _timestamp(candidate_lock.get("locked_at_utc"), "locked_at_utc")

    protocol_ref = _mapping(candidate_lock.get("protocol"), "candidate lock protocol")
    _exact_keys(protocol_ref, {"id", "sha256"}, "candidate lock protocol")
    if protocol_ref.get("id") != protocol.get("protocol_id"):
        raise GateValidationError("candidate lock protocol ID disagrees")
    if protocol_ref.get("sha256") != PROTOCOL_SHA256:
        raise GateValidationError("candidate lock protocol digest disagrees")

    selection = _mapping(candidate_lock.get("selection"), "candidate selection")
    _exact_keys(
        selection,
        {
            "candidate_ledger",
            "terminal_e_exit_receipt",
            "selection_rule",
            "checkpoint_selection_rule",
            "all_pilot_attempts_disclosed",
            "selected_without_final_seed_results",
            "final_seeds_used_during_selection",
        },
        "candidate selection",
    )
    if selection.get("selection_rule") != CANDIDATE_SELECTION_RULE:
        raise GateValidationError("candidate selection rule is not the frozen enum")
    if selection.get("checkpoint_selection_rule") != CHECKPOINT_SELECTION_RULE:
        raise GateValidationError(
            "candidate checkpoint-selection rule is not the frozen enum"
        )
    _required_true(
        selection.get("all_pilot_attempts_disclosed"),
        "all_pilot_attempts_disclosed",
    )
    _required_true(
        selection.get("selected_without_final_seed_results"),
        "selected_without_final_seed_results",
    )
    if selection.get("final_seeds_used_during_selection") != []:
        raise GateValidationError("final seeds must not be used during selection")
    ledger = _artifact_reference(
        selection.get("candidate_ledger"),
        label="candidate ledger",
        suffix=".json",
        require_schema=True,
    )
    if ledger["schema_version"] != CANDIDATE_LEDGER_SCHEMA_VERSION:
        raise GateValidationError("candidate ledger schema version is unsupported")
    terminal_e_receipt = _artifact_reference(
        selection.get("terminal_e_exit_receipt"),
        label="terminal E exit receipt",
        suffix=".json",
        require_schema=True,
    )
    if terminal_e_receipt["schema_version"] != PILOT_EXIT_STATUS_SCHEMA_VERSION:
        raise GateValidationError("terminal E exit receipt schema is unsupported")
    terminal_parts = terminal_e_receipt["relative_path"].parts
    if (
        len(terminal_parts) != 4
        or terminal_parts[:2] != ("output", "udlm")
        or RUN_NAME_PATTERN.fullmatch(terminal_parts[2]) is None
        or terminal_parts[3] != "pilot_exit_status.json"
    ):
        raise GateValidationError(
            "terminal E exit receipt must be output/udlm/<run>/pilot_exit_status.json"
        )

    training = _mapping(candidate_lock.get("training"), "candidate training")
    _exact_keys(
        training,
        {
            "source_revision",
            "training_summary",
            "exit_receipt",
            "runtime_config",
            "launch_manifest",
            "resolved_training_config_sha256",
            "training_argv_sha256",
            "checkpoint",
            "startup",
            "training_seed",
            "optimizer_updates",
            "world_size",
            "data_exposure",
            "parameter_counts",
        },
        "candidate training",
    )
    source_revision = _git_revision(
        training.get("source_revision"), "candidate training source revision"
    )
    summary_ref = _artifact_reference(
        training.get("training_summary"),
        label="training summary",
        suffix=".json",
        require_schema=True,
    )
    receipt_ref = _artifact_reference(
        training.get("exit_receipt"),
        label="training exit receipt",
        suffix=".json",
        require_schema=True,
    )
    if summary_ref["schema_version"] != TRAINING_SUMMARY_SCHEMA_VERSION:
        raise GateValidationError(
            "candidate training summary schema version is unsupported"
        )
    if receipt_ref["schema_version"] != PILOT_EXIT_STATUS_SCHEMA_VERSION:
        raise GateValidationError(
            "candidate exit receipt schema version is unsupported"
        )
    runtime_ref = _artifact_reference(
        training.get("runtime_config"),
        label="training runtime config",
        suffix=".json",
        require_schema=True,
    )
    launch_manifest_ref = _artifact_reference(
        training.get("launch_manifest"),
        label="training launch manifest",
        suffix=".json",
        require_schema=True,
    )
    if runtime_ref["schema_version"] != RUNTIME_CONFIG_SCHEMA_VERSION:
        raise GateValidationError(
            "candidate runtime-config schema version is unsupported"
        )
    if launch_manifest_ref["schema_version"] != LAUNCH_MANIFEST_SCHEMA_VERSION:
        raise GateValidationError(
            "candidate launch-manifest schema version is unsupported"
        )
    expected_manifest_path = summary_ref["relative_path"].with_name(
        "launch_manifest.json"
    )
    if launch_manifest_ref["relative_path"] != expected_manifest_path:
        raise GateValidationError(
            "candidate launch manifest must be launch_manifest.json beside the summary"
        )
    if runtime_ref["relative_path"].parent != expected_manifest_path.parent:
        raise GateValidationError(
            "candidate runtime config and launch manifest must share a run directory"
        )
    if receipt_ref["relative_path"].parent != expected_manifest_path.parent:
        raise GateValidationError(
            "candidate exit receipt and launch manifest must share a run directory"
        )
    resolved_config_sha = _sha256(
        training.get("resolved_training_config_sha256"),
        "resolved training config digest",
    )
    training_argv_sha = _sha256(
        training.get("training_argv_sha256"), "training argv digest"
    )
    checkpoint = _mapping(training.get("checkpoint"), "candidate checkpoint")
    _exact_keys(
        checkpoint,
        {"relative_path", "sha256", "size_bytes", "global_step", "weights"},
        "candidate checkpoint",
    )
    checkpoint_path = _relative_path(
        checkpoint.get("relative_path"), "candidate checkpoint path", suffix=".ckpt"
    )
    checkpoint_sha = _sha256(checkpoint.get("sha256"), "candidate checkpoint digest")
    checkpoint_size = _integer(
        checkpoint.get("size_bytes"), "candidate checkpoint size", minimum=1
    )
    checkpoint_step = _integer(
        checkpoint.get("global_step"), "candidate checkpoint step", minimum=1
    )
    if checkpoint.get("weights") != "ema":
        raise GateValidationError("final inference must use candidate EMA weights")
    optimizer_updates = _integer(
        training.get("optimizer_updates"), "candidate optimizer updates", minimum=1
    )
    if checkpoint_step != optimizer_updates:
        raise GateValidationError("checkpoint step and optimizer updates disagree")
    world_size = _integer(training.get("world_size"), "training world size", minimum=1)
    if world_size not in (1, 2):
        raise GateValidationError("candidate training may use only one or two GPUs")
    training_seed = _integer(
        training.get("training_seed"), "candidate training seed", minimum=0
    )

    startup = _mapping(training.get("startup"), "candidate startup")
    _exact_keys(
        startup, {"mode", "initialization_checkpoint_sha256"}, "candidate startup"
    )
    startup_mode = startup.get("mode")
    if startup_mode not in CLAIM_SCOPE_BY_STARTUP:
        raise GateValidationError(
            "candidate startup mode must be warm_start or scratch"
        )
    initialization_sha = startup.get("initialization_checkpoint_sha256")
    if startup_mode == "warm_start":
        if initialization_sha != EXPECTED_BASELINE_CHECKPOINT_SHA256:
            raise GateValidationError("warm start must use the frozen MDLM checkpoint")
    elif initialization_sha is not None:
        raise GateValidationError("scratch startup must have null initialization hash")
    if candidate_lock.get("claim_scope") != CLAIM_SCOPE_BY_STARTUP[startup_mode]:
        raise GateValidationError("candidate claim scope disagrees with startup mode")

    analysis = _mapping(candidate_lock.get("analysis"), "candidate analysis")
    _exact_keys(
        analysis,
        {
            "gate_source_sha256",
            "report_source_sha256",
            "rescore_source_sha256",
            "rescore_dependency_sha256",
            "benchmark_launcher_source_sha256",
            "pilot_evidence_writer_source_sha256",
            "scipy_version",
        },
        "candidate analysis",
    )
    gate_source_sha = _sha256(
        analysis.get("gate_source_sha256"), "superiority gate source digest"
    )
    report_source_sha = _sha256(
        analysis.get("report_source_sha256"), "de-novo report source digest"
    )
    rescore_source_sha = _sha256(
        analysis.get("rescore_source_sha256"),
        "independent de-novo rescore source digest",
    )
    rescore_dependency_sha = _sha256(
        analysis.get("rescore_dependency_sha256"),
        "independent de-novo rescore dependency digest",
    )
    benchmark_launcher_source_sha = _sha256(
        analysis.get("benchmark_launcher_source_sha256"),
        "de-novo benchmark launcher source digest",
    )
    pilot_evidence_writer_source_sha = _sha256(
        analysis.get("pilot_evidence_writer_source_sha256"),
        "pilot evidence writer source digest",
    )
    scipy_version = analysis.get("scipy_version")
    if not isinstance(scipy_version, str) or not scipy_version.strip():
        raise GateValidationError("candidate analysis scipy_version must be nonempty")

    exposure = _mapping(training.get("data_exposure"), "candidate data exposure")
    _exact_keys(
        exposure,
        {
            "global_examples_per_optimizer_step",
            "optimizer_updates",
            "total_requested_examples",
            "stream_partition_policy",
        },
        "candidate data exposure",
    )
    global_examples = _integer(
        exposure.get("global_examples_per_optimizer_step"),
        "global examples per optimizer step",
        minimum=1,
    )
    if exposure.get("optimizer_updates") != optimizer_updates:
        raise GateValidationError("data-exposure update count disagrees")
    if exposure.get("total_requested_examples") != global_examples * optimizer_updates:
        raise GateValidationError("total requested training examples are inconsistent")
    if exposure.get("stream_partition_policy") != (
        "huggingface_split_dataset_by_node_disjoint_rank_streams"
    ):
        raise GateValidationError("candidate stream partition policy is unexpected")
    total_requested_examples = _integer(
        exposure.get("total_requested_examples"),
        "total requested training examples",
        minimum=1,
    )

    parameters = _mapping(training.get("parameter_counts"), "candidate parameters")
    additive_parameter_keys = {
        "base_model_trainable",
        "time_conditioner_trainable",
        "total_trainable",
    }
    film_parameter_keys = additive_parameter_keys | {"film_modulation_trainable"}
    observed_parameter_keys = set(parameters)
    if observed_parameter_keys not in (additive_parameter_keys, film_parameter_keys):
        raise GateValidationError(
            "candidate parameters must use exactly the additive 3-key or "
            "film_adaln 4-key schema"
        )
    base_count = _integer(
        parameters.get("base_model_trainable"), "base trainable parameters", minimum=1
    )
    adapter_count = _integer(
        parameters.get("time_conditioner_trainable"),
        "time-conditioner trainable parameters",
        minimum=1,
    )
    film_count = 0
    if observed_parameter_keys == film_parameter_keys:
        film_count = _integer(
            parameters.get("film_modulation_trainable"),
            "FiLM-modulation trainable parameters",
            minimum=1,
        )
    total_trainable = _integer(
        parameters.get("total_trainable"), "total trainable parameters", minimum=1
    )
    if total_trainable != base_count + adapter_count + film_count:
        raise GateValidationError("candidate trainable parameter counts do not add up")
    normalized_parameter_counts = {
        "base_model_trainable": base_count,
        "time_conditioner_trainable": adapter_count,
        "total_trainable": total_trainable,
    }
    if observed_parameter_keys == film_parameter_keys:
        normalized_parameter_counts["film_modulation_trainable"] = film_count

    inference = _mapping(candidate_lock.get("inference"), "candidate inference")
    _exact_keys(
        inference,
        {
            "evaluation_config_relative_path",
            "evaluation_config_sha256",
            "sampling_config",
            "sampling_sha256",
            "checkpoint_sha256",
            "weights",
            "inference_weights",
            "nfe",
            "final_seeds",
            "samples_per_seed",
            "sampler_source_sha256",
            "benchmark_runner_sha256",
            "implementation_inputs_sha256",
            "metric_inputs_sha256",
            "final_run_directories_by_seed",
        },
        "candidate inference",
    )
    evaluation_config_path = _relative_path(
        inference.get("evaluation_config_relative_path"),
        "evaluation config path",
        suffix=".yaml",
    )
    evaluation_config_sha = _sha256(
        inference.get("evaluation_config_sha256"), "evaluation config digest"
    )
    sampling_config = _mapping(inference.get("sampling_config"), "sampling config")
    sampling_sha = _sha256(inference.get("sampling_sha256"), "sampling config digest")
    if canonical_json_sha256(sampling_config) != sampling_sha:
        raise GateValidationError("candidate sampling config digest is invalid")
    if sampling_config.get("diffusion_type") != "udlm":
        raise GateValidationError("candidate sampling config must select UDLM")
    if inference.get("checkpoint_sha256") != checkpoint_sha:
        raise GateValidationError("training and inference checkpoint hashes disagree")
    if inference.get("weights") != "ema":
        raise GateValidationError("candidate inference weights must be EMA")
    try:
        locked_inference_weights = denovo_report.validate_inference_weights(
            inference.get("inference_weights"), require_ema=True
        )
    except ValueError as error:
        raise GateValidationError(
            f"candidate locked inference-weight receipt is invalid: {error}"
        ) from error
    if locked_inference_weights["ema"]["num_updates"] != optimizer_updates:
        raise GateValidationError(
            "candidate locked EMA update count must equal optimizer updates"
        )
    if inference.get("nfe") != EXPECTED_NFE:
        raise GateValidationError("candidate inference NFE must equal 128")
    if inference.get("final_seeds") != list(EXPECTED_SEEDS):
        raise GateValidationError("candidate final seeds are unexpected")
    if inference.get("samples_per_seed") != EXPECTED_SAMPLES_PER_SEED:
        raise GateValidationError("candidate final sample count is unexpected")
    if sampling_config.get("num_steps") != EXPECTED_NFE:
        raise GateValidationError("sampling num_steps must equal the registered NFE")
    sampler_source_sha = _sha256(
        inference.get("sampler_source_sha256"), "sampler source digest"
    )
    benchmark_runner_sha = _sha256(
        inference.get("benchmark_runner_sha256"), "benchmark runner digest"
    )
    implementation_inputs_sha = _sha256(
        inference.get("implementation_inputs_sha256"),
        "implementation-input map digest",
    )
    metric_inputs_sha = _sha256(
        inference.get("metric_inputs_sha256"), "metric-input map digest"
    )
    final_directories_raw = inference.get("final_run_directories_by_seed")
    if not isinstance(final_directories_raw, list):
        raise GateValidationError("final run directories must be a list")
    final_directories: dict[int, Path] = {}
    for raw_row in final_directories_raw:
        row = _mapping(raw_row, "final run directory row")
        _exact_keys(row, {"seed", "relative_path"}, "final run directory row")
        seed = row.get("seed")
        if type(seed) is not int or seed in final_directories:
            raise GateValidationError(
                "final run directory seeds must be unique integers"
            )
        final_directories[seed] = _relative_directory(
            row.get("relative_path"), f"final seed {seed} output directory"
        )
    if tuple(sorted(final_directories)) != EXPECTED_SEEDS:
        raise GateValidationError("final run directories must bind seeds 0, 1, and 2")
    if len(set(final_directories.values())) != len(EXPECTED_SEEDS):
        raise GateValidationError("final run directories must be distinct")

    return {
        "candidate_id": candidate_id,
        "locked_at": locked_at,
        "source_revision": source_revision,
        "ledger": ledger,
        "terminal_e_receipt": terminal_e_receipt,
        "selection_rule": selection["selection_rule"],
        "checkpoint_selection_rule": selection["checkpoint_selection_rule"],
        "summary": summary_ref,
        "receipt": receipt_ref,
        "runtime": runtime_ref,
        "launch_manifest": launch_manifest_ref,
        "resolved_training_config_sha256": resolved_config_sha,
        "training_argv_sha256": training_argv_sha,
        "checkpoint": {
            "relative_path": checkpoint_path,
            "sha256": checkpoint_sha,
            "size_bytes": checkpoint_size,
            "global_step": checkpoint_step,
        },
        "startup_mode": startup_mode,
        "initialization_checkpoint_sha256": initialization_sha,
        "optimizer_updates": optimizer_updates,
        "world_size": world_size,
        "training_seed": training_seed,
        "data_exposure": {
            "global_examples_per_optimizer_step": global_examples,
            "optimizer_updates": optimizer_updates,
            "total_requested_examples": total_requested_examples,
            "stream_partition_policy": exposure["stream_partition_policy"],
        },
        "parameter_counts": normalized_parameter_counts,
        "evaluation_config_relative_path": evaluation_config_path,
        "evaluation_config_sha256": evaluation_config_sha,
        "sampling_config": dict(sampling_config),
        "sampling_sha256": sampling_sha,
        "inference_weights": locked_inference_weights,
        "sampler_source_sha256": sampler_source_sha,
        "benchmark_runner_sha256": benchmark_runner_sha,
        "implementation_inputs_sha256": implementation_inputs_sha,
        "metric_inputs_sha256": metric_inputs_sha,
        "final_run_directories": final_directories,
        "gate_source_sha256": gate_source_sha,
        "report_source_sha256": report_source_sha,
        "rescore_source_sha256": rescore_source_sha,
        "rescore_dependency_sha256": rescore_dependency_sha,
        "benchmark_launcher_source_sha256": benchmark_launcher_source_sha,
        "pilot_evidence_writer_source_sha256": pilot_evidence_writer_source_sha,
        "scipy_version": scipy_version,
        "claim_scope": candidate_lock["claim_scope"],
    }


def _artifact_reference(
    value: object, *, label: str, suffix: str, require_schema: bool
) -> dict[str, Any]:
    reference = _mapping(value, label)
    expected = {"relative_path", "sha256"}
    if require_schema:
        expected.add("schema_version")
    _exact_keys(reference, expected, label)
    result = {
        "relative_path": _relative_path(
            reference.get("relative_path"), f"{label} path", suffix=suffix
        ),
        "sha256": _sha256(reference.get("sha256"), f"{label} digest"),
    }
    if require_schema:
        result["schema_version"] = _integer(
            reference.get("schema_version"), f"{label} schema version", minimum=1
        )
    return result


def _load_referenced_json(
    reference: Mapping[str, Any], *, label: str
) -> Mapping[str, Any]:
    payload = _repository_artifact_bytes(reference["relative_path"], label=label)
    observed = _sha256_bytes(payload)
    if observed != reference["sha256"]:
        raise GateValidationError(f"{label} digest disagrees with candidate lock")
    parsed = _mapping(strict_json_loads(payload, label=label), label)
    if parsed.get("schema_version") != reference["schema_version"]:
        raise GateValidationError(
            f"{label} schema version disagrees with candidate lock"
        )
    return parsed


_STABLE_SNAPSHOT_KEYS = {
    "path",
    "device",
    "inode",
    "mode",
    "link_count",
    "size_bytes",
    "mtime_ns",
    "ctime_ns",
    "sha256",
    "stable_regular_file_verified",
}
_LAUNCH_MANIFEST_KEYS = {
    "launch_manifest_schema_version",
    "created_at",
    "purpose",
    "gpu_selection_schema_version",
    "git_sha",
    "source_revision_before_final_gpu_probe",
    "run_name",
    "training_variant",
    "hydra_config_name",
    "udlm_prior_variant",
    "udlm_comparison_role",
    "matched_panel_spec",
    "matched_panel_spec_sha256",
    "matched_panel_variant_position",
    "predecessor_receipt_binding",
    "single_training_job_lock",
    "tmux_session",
    "user_requested_gpu_count",
    "gpu_selection_method",
    "gpu_inventory_scope",
    "inventory_snapshot_completed_at_utc",
    "gpu_inventory_at_selection",
    "initially_selected_gpu_states",
    "logical_cuda_devices",
    "physical_gpu_indices",
    "cuda_visible_device_uuids",
    "final_uuid_probes_completed_at_utc",
    "gpu_states_at_final_uuid_probe",
    "gpu_safety_policy",
    "training_argv",
    "training_argv_sha256",
    "resolved_training_config",
    "resolved_training_config_sha256",
    "runtime_config_path",
    "training_summary_path",
    "training_summary_schema_version",
    "pilot_exit_status_path",
    "pilot_exit_status_schema_version",
    "expected_final_checkpoint_path",
    "launch_manifest_path",
    "launch_manifest_raw_sha256_transport",
    "completion_contract",
    "log_path",
    "log_reserved_exclusively_before_manifest",
    "checkpoint",
    "checkpoint_sha256",
    "seed",
    "max_steps",
    "global_batch_size",
    "micro_batch_size_per_process",
    "accumulate_grad_batches",
    "effective_global_batch_size",
    "exclude_special_tokens",
    "dry_run",
}
_GPU_STATE_KEYS = {
    "physical_index",
    "uuid",
    "name",
    "memory_used_mib",
    "memory_total_mib",
    "utilization_percent",
    "compute_mode",
    "compute_processes",
}
_PREDECESSOR_BINDING_KEYS = {
    "schema_version",
    "state",
    "current_training_variant",
    "current_variant_position",
    "expected_predecessor_training_variant",
    "expected_predecessor_variant_position",
    "matched_panel_spec_sha256",
    "common_training_contract_sha256",
    "receipt_artifact",
    "predecessor_launch_manifest_artifact",
    "predecessor_training_summary_artifact",
    "predecessor_run_name",
    "chronology",
    "validated_before_gpu_probe",
}
_PREDECESSOR_CHRONOLOGY_KEYS = {
    "predecessor_launch_manifest_created_at_utc",
    "predecessor_training_summary_completed_at_utc",
    "predecessor_exit_receipt_recorded_at_utc",
    "strictly_ordered_timestamps_verified",
}
_RUNTIME_CONFIG_KEYS = {
    "schema_version",
    "status",
    "source_revision",
    "source",
    "training_argv",
    "observed_training_argv",
    "training_argv_sha256",
    "resolved_training_config",
    "resolved_training_config_sha256",
    "launch_manifest",
    "completion_contract",
    "python_environment",
}
_TRAINING_SUMMARY_KEYS = {
    "schema_version",
    "status",
    "completed_at_utc",
    "source_revision",
    "source",
    "resolved_training_config_sha256",
    "training_argv_sha256",
    "launch_manifest",
    "runtime_config",
    "completion_contract",
    "observed_training_state",
    "training_accounting",
    "training_health",
    "conditioning_gradient_audit",
    "screen_initialization_state_audit",
    "final_checkpoint",
    "tensor_finiteness",
    "startup",
}
_PILOT_EXIT_STATUS_KEYS = {
    "schema_version",
    "status",
    "overall_status",
    "recorded_at_utc",
    "process_exit_status",
    "expected_contract",
    "pipeline",
    "source_at_receipt",
    "launch_manifest",
    "predecessor_receipt_binding",
    "training_job_lock",
    "training_summary",
    "runtime_config",
    "final_checkpoint",
    "completion_requirements",
}
_PIPELINE_COMPONENT_KEYS = {
    "shell_exit_status",
    "succeeded",
    "possible_termination_signal",
    "shell_status_is_signal_compatible",
    "signal_provenance",
}
_MATCHED_PANEL_VARIANT_ORDER = (
    "udlm",
    "schedule_uniform",
    "udlm_categorical",
)


def _validate_successful_pipeline(value: object, *, label: str) -> None:
    pipeline = _mapping(value, label)
    _exact_keys(
        pipeline,
        {"training", "tee", "pipefail_shell_exit_status"},
        label,
    )
    pipefail_status = _integer(
        pipeline.get("pipefail_shell_exit_status"),
        f"{label} pipefail shell exit status",
    )
    if pipefail_status != 0:
        raise GateValidationError(f"{label} did not exit cleanly")
    expected_component = {
        "shell_exit_status": 0,
        "succeeded": True,
        "possible_termination_signal": None,
        "shell_status_is_signal_compatible": False,
        "signal_provenance": None,
    }
    for component_name in ("training", "tee"):
        component_label = f"{label} {component_name} component"
        component = _mapping(pipeline.get(component_name), component_label)
        _exact_keys(component, _PIPELINE_COMPONENT_KEYS, component_label)
        if dict(component) != expected_component:
            raise GateValidationError(f"{component_label} is not successful")


def _validate_receipt_source(
    value: object, *, expected_revision: str, label: str
) -> None:
    source = _mapping(value, label)
    expected = {
        "verified": True,
        "expected_revision": expected_revision,
        "head": expected_revision,
        "upstream": expected_revision,
        "output_directory_excluded_from_cleanliness_check": True,
    }
    _exact_keys(source, set(expected), label)
    if dict(source) != expected:
        raise GateValidationError(f"{label} is invalid")


def _validate_training_source(
    value: object, *, expected_revision: str, label: str
) -> None:
    source = _mapping(value, label)
    expected = {"head": expected_revision, "upstream": expected_revision}
    _exact_keys(source, set(expected), label)
    if dict(source) != expected:
        raise GateValidationError(f"{label} is invalid")


def _validate_python_environment(value: object, *, seed: int, label: str) -> None:
    environment = _mapping(value, label)
    expected = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": str(seed),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONOPTIMIZE": "0",
        "PYTHONPATH": os.pathsep.join(
            [str(REPOSITORY_ROOT / "src"), str(REPOSITORY_ROOT)]
        ),
        "PYTHONUTF8": "1",
    }
    _exact_keys(environment, set(expected), label)
    if dict(environment) != expected:
        raise GateValidationError(f"{label} is invalid")


def _validate_finiteness_record(value: object, *, label: str) -> None:
    record = _mapping(value, label)
    _exact_keys(
        record,
        {"all_finite", "floating_tensor_count", "floating_element_count"},
        label,
    )
    _required_true(record.get("all_finite"), f"{label} all-finite flag")
    tensor_count = _integer(
        record.get("floating_tensor_count"), f"{label} tensor count", minimum=1
    )
    element_count = _integer(
        record.get("floating_element_count"), f"{label} element count", minimum=1
    )
    if element_count < tensor_count:
        raise GateValidationError(f"{label} has fewer elements than tensors")


def _expected_framework_nonfinite_sentinels(
    *, expected_steps: int
) -> dict[str, object]:
    callback_key = (
        "ModelCheckpoint{'monitor': None, 'mode': 'min', "
        f"'every_n_train_steps': {expected_steps}, 'every_n_epochs': 0, "
        "'train_time_interval': None}"
    )
    return {
        "all_expected_and_only_expected_verified": True,
        "nonfinite_tensor_count": 1,
        "nonfinite_element_count": 1,
        "records": [
            {
                "tensor_path_components": [
                    "checkpoint",
                    "callbacks",
                    callback_key,
                    "kth_value",
                ],
                "framework": "lightning",
                "framework_version": "2.5.1",
                "callback": "ModelCheckpoint",
                "field": "kth_value",
                "dtype": "float32",
                "shape": [],
                "value": "+inf",
                "meaning": "unranked_min_mode_checkpoint_sentinel",
                "excluded_from_non_sentinel_finiteness": True,
            }
        ],
    }


def _validate_framework_nonfinite_sentinels(
    value: object, *, expected_steps: int, label: str
) -> None:
    record = _mapping(value, label)
    expected = _expected_framework_nonfinite_sentinels(expected_steps=expected_steps)
    if canonical_json_sha256(record) != canonical_json_sha256(expected):
        raise GateValidationError(
            f"{label} does not match the exact Lightning sentinel contract"
        )


def _validate_auxiliary_checkpoint_records(
    semantic: Mapping[str, object],
    *,
    expected_steps: int,
    resolved_training_config: object,
    label: str,
) -> int:
    resolved = _mapping(
        resolved_training_config,
        f"{label} resolved training config",
    )
    config_sha256 = canonical_json_sha256(resolved)
    trainer_config = _mapping(
        resolved.get("trainer"),
        f"{label} resolved trainer config",
    )
    accumulation = _integer(
        trainer_config.get("accumulate_grad_batches"),
        f"{label} resolved accumulation",
        minimum=1,
    )
    optim_config = _mapping(
        resolved.get("optim"),
        f"{label} resolved optimizer config",
    )
    scheduler_config = _mapping(
        optim_config.get("scheduler"),
        f"{label} resolved scheduler config",
    )
    warmup_updates = _integer(
        scheduler_config.get("warmup_updates"),
        f"{label} resolved scheduler warmup",
        minimum=0,
    )
    horizon_value = scheduler_config.get("horizon_updates")
    horizon_updates = (
        0
        if horizon_value is None
        else _integer(
            horizon_value,
            f"{label} resolved scheduler horizon",
            minimum=0,
        )
    )
    schedule_check_count = (
        max(expected_steps, warmup_updates + 1, horizon_updates + 1) + 1
    )
    python_floats = _mapping(
        semantic.get("checkpoint_python_floats"),
        f"{label} Python-float finiteness",
    )
    _exact_keys(
        python_floats,
        {"all_finite", "floating_scalar_count"},
        f"{label} Python-float finiteness",
    )
    _required_true(
        python_floats.get("all_finite"),
        f"{label} Python-float all-finite flag",
    )
    _integer(
        python_floats.get("floating_scalar_count"),
        f"{label} Python-float count",
        minimum=1,
    )

    optimizer = _mapping(
        semantic.get("optimizer_live_state_match"),
        f"{label} optimizer live-state match",
    )
    _exact_keys(
        optimizer,
        {
            "exact_serialized_live_match",
            "optimizer_count",
            "optimizer_class",
            "parameter_group_count",
            "parameter_state_count",
            "exact_resolved_config_match",
        },
        f"{label} optimizer live-state match",
    )
    _required_true(
        optimizer.get("exact_serialized_live_match"),
        f"{label} optimizer exact-match flag",
    )
    _required_true(
        optimizer.get("exact_resolved_config_match"),
        f"{label} optimizer resolved-config flag",
    )
    if (
        optimizer.get("optimizer_count") != 1
        or optimizer.get("optimizer_class") != "AdamW"
        or optimizer.get("parameter_group_count") != 1
    ):
        raise GateValidationError(f"{label} optimizer live-state match is invalid")
    parameter_state_count = _integer(
        optimizer.get("parameter_state_count"),
        f"{label} optimizer parameter-state count",
        minimum=1,
    )

    scheduler = _mapping(
        semantic.get("scheduler_live_state_match"),
        f"{label} scheduler live-state match",
    )
    if canonical_json_sha256(scheduler) != canonical_json_sha256(
        {
            "exact_serialized_live_match": True,
            "scheduler_count": 1,
            "scheduler_class": "LambdaLR",
            "interval": "step",
            "name": "lr",
            "last_epoch": expected_steps,
            "step_count": expected_steps + 1,
            "exact_model_spec_match": True,
            "exact_callable_schedule_match": True,
            "callable_schedule_index_checks": schedule_check_count,
        }
    ):
        raise GateValidationError(f"{label} scheduler live-state match is invalid")

    sampler = _mapping(
        semantic.get("sampler_live_state_match"),
        f"{label} sampler live-state match",
    )
    if canonical_json_sha256(sampler) != canonical_json_sha256(
        {
            "exact_hosted_stream_contract_match": True,
            "random_state_is_none": True,
            "live_state_dict_available": False,
            "sampler_class_module": "torch.utils.data.dataloader",
            "sampler_class_name": "_InfiniteConstantSampler",
        }
    ):
        raise GateValidationError(f"{label} sampler live-state match is invalid")

    callback_key = (
        "ModelCheckpoint{'monitor': None, 'mode': 'min', "
        f"'every_n_train_steps': {expected_steps}, 'every_n_epochs': 0, "
        "'train_time_interval': None}"
    )
    callback = _mapping(
        semantic.get("model_checkpoint_live_state_match"),
        f"{label} ModelCheckpoint live-state match",
    )
    if canonical_json_sha256(callback) != canonical_json_sha256(
        {
            "exact_serialized_live_match": True,
            "model_checkpoint_callback_count": 1,
            "state_key": callback_key,
            "configuration_matches_pilot_contract": True,
        }
    ):
        raise GateValidationError(
            f"{label} ModelCheckpoint live-state match is invalid"
        )

    hyperparameters = _mapping(
        semantic.get("checkpoint_hyperparameters_match"),
        f"{label} checkpoint hyperparameter match",
    )
    if canonical_json_sha256(hyperparameters) != canonical_json_sha256(
        {
            "hparams_name": "kwargs",
            "exact_hyperparameter_keys": True,
            "exact_checkpoint_preflight_config_match": True,
            "exact_live_model_preflight_config_match": True,
            "exact_live_hparams_preflight_config_match": True,
            "exact_checkpoint_live_model_unresolved_config_match": True,
            "exact_checkpoint_live_hparams_unresolved_config_match": True,
            "resolved_config_sha256": config_sha256,
        }
    ):
        raise GateValidationError(f"{label} checkpoint hyperparameter match is invalid")

    configured_clip = trainer_config.get("gradient_clip_val")
    configured_precision = trainer_config.get("precision")
    precision_aliases = {
        "16": "16-mixed",
        "bf16": "bf16-mixed",
        "32": "32-true",
        "64": "64-true",
        16: "16-mixed",
        32: "32-true",
        64: "64-true",
    }
    live_precision = precision_aliases.get(configured_precision, configured_precision)
    configured_clip_algorithm = trainer_config.get("gradient_clip_algorithm")
    if configured_clip_algorithm is None:
        configured_clip_algorithm = "norm"
    if (
        isinstance(configured_clip, bool)
        or not isinstance(configured_clip, (int, float))
        or not math.isfinite(configured_clip)
        or configured_clip < 0.0
        or not isinstance(live_precision, str)
        or configured_clip_algorithm not in {"norm", "value"}
    ):
        raise GateValidationError(f"{label} resolved live-Trainer config is invalid")
    trainer_match = _mapping(
        semantic.get("trainer_live_configuration_match"),
        f"{label} live Trainer configuration match",
    )
    if canonical_json_sha256(trainer_match) != canonical_json_sha256(
        {
            "exact_detect_anomaly_match": True,
            "detect_anomaly": True,
            "exact_gradient_clip_val_match": True,
            "gradient_clip_val": float(configured_clip),
            "exact_gradient_clip_algorithm_match": True,
            "gradient_clip_algorithm": configured_clip_algorithm,
            "exact_precision_match": True,
            "configured_precision": str(configured_precision),
            "live_precision": live_precision,
        }
    ):
        raise GateValidationError(
            f"{label} live Trainer configuration match is invalid"
        )

    loop_state = _mapping(
        semantic.get("checkpoint_loop_state_match"),
        f"{label} checkpoint loop-state match",
    )
    if canonical_json_sha256(loop_state) != canonical_json_sha256(
        {
            "exact_serialized_progress_match": True,
            "epoch": 0,
            "optimizer_steps": expected_steps,
            "accumulate_grad_batches": accumulation,
            "microbatches": expected_steps * accumulation,
        }
    ):
        raise GateValidationError(f"{label} checkpoint loop-state match is invalid")
    return parameter_state_count


def _validate_training_health(value: object, *, optimizer_updates: int) -> None:
    health = _mapping(value, "training health")
    _exact_keys(
        health,
        {
            "scope",
            "all_losses_finite",
            "all_observed_gradients_finite",
            "every_optimizer_step_had_a_nonzero_gradient",
            "loss_checks",
            "optimizer_step_checks",
            "gradient_tensor_observations",
            "gradient_element_observations",
        },
        "training health",
    )
    if health.get("scope") != (
        "global-rank-zero callback counters; identical fail-fast checks execute "
        "independently on every rank"
    ):
        raise GateValidationError("training health scope is unexpected")
    for field in (
        "all_losses_finite",
        "all_observed_gradients_finite",
        "every_optimizer_step_had_a_nonzero_gradient",
    ):
        _required_true(health.get(field), f"training health {field}")
    loss_checks = _integer(
        health.get("loss_checks"), "training health loss checks", minimum=1
    )
    optimizer_checks = _integer(
        health.get("optimizer_step_checks"),
        "training health optimizer-step checks",
        minimum=1,
    )
    if loss_checks < optimizer_updates or optimizer_checks != optimizer_updates:
        raise GateValidationError("training health counters disagree with updates")
    gradient_tensors = _integer(
        health.get("gradient_tensor_observations"),
        "training health gradient_tensor_observations",
        minimum=1,
    )
    gradient_elements = _integer(
        health.get("gradient_element_observations"),
        "training health gradient_element_observations",
        minimum=1,
    )
    if gradient_tensors < optimizer_checks or gradient_elements < gradient_tensors:
        raise GateValidationError(
            "training health gradient observation counts are inconsistent"
        )


def _expected_artifact_path(relative_path: Path) -> Path:
    return Path(os.path.abspath(REPOSITORY_ROOT / relative_path))


def _reviewed_training_python_executables() -> tuple[Path, Path]:
    """Return both immutable lexical interpreter locations allowed at launch."""

    return (
        REPOSITORY_ROOT / ".venv" / "bin" / "python",
        REPOSITORY_ROOT.parents[1] / ".venv" / "bin" / "python",
    )


def _project_training_python_executable() -> Path:
    """Return the one shared project interpreter used by the benchmark launcher."""

    repository_root = Path(os.path.abspath(REPOSITORY_ROOT))
    project_root = (
        repository_root.parent.parent
        if repository_root.parent.name == "run_sources"
        else repository_root
    )
    return project_root / ".venv" / "bin" / "python"


def _validate_snapshot_claim(
    value: object,
    *,
    label: str,
    expected_path: Path,
    expected_sha256: str,
    expected_size_bytes: int | None = None,
    include_selected_gpu_uuids: bool = False,
    expected_selected_gpu_uuids: Sequence[str] = (),
) -> dict[str, Any]:
    claim = _mapping(value, label)
    expected_keys = set(_STABLE_SNAPSHOT_KEYS)
    if include_selected_gpu_uuids:
        expected_keys.add("selected_gpu_uuids")
    _exact_keys(claim, expected_keys, label)
    if claim.get("path") != str(expected_path):
        raise GateValidationError(f"{label} path disagrees with candidate lock")
    if claim.get("sha256") != expected_sha256:
        raise GateValidationError(f"{label} digest disagrees with candidate lock")
    _required_true(
        claim.get("stable_regular_file_verified"),
        f"{label}.stable_regular_file_verified",
    )
    for field in ("device", "inode", "mode", "mtime_ns", "ctime_ns"):
        _integer(claim.get(field), f"{label}.{field}", minimum=0)
    if _integer(claim.get("link_count"), f"{label}.link_count", minimum=1) != 1:
        raise GateValidationError(f"{label}.link_count must equal one")
    size_bytes = _integer(claim.get("size_bytes"), f"{label}.size_bytes", minimum=1)
    if expected_size_bytes is not None and size_bytes != expected_size_bytes:
        raise GateValidationError(f"{label} size disagrees with launch manifest bytes")
    if include_selected_gpu_uuids:
        selected = _selected_gpu_uuids(
            claim.get("selected_gpu_uuids"),
            f"{label}.selected_gpu_uuids",
            expected_count=len(expected_selected_gpu_uuids),
        )
        if selected != list(expected_selected_gpu_uuids):
            raise GateValidationError(
                f"{label} selected GPU UUIDs disagree with launch manifest"
            )
    return dict(claim)


def _load_bound_predecessor_json(
    claim_value: object,
    *,
    label: str,
    basename: str,
    run_name: str,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Read one exact absolute predecessor artifact without trusting its claim."""

    claim = _mapping(claim_value, f"{label} claim")
    _exact_keys(claim, _STABLE_SNAPSHOT_KEYS, f"{label} claim")
    path_value = claim.get("path")
    if not isinstance(path_value, str) or not Path(path_value).is_absolute():
        raise GateValidationError(f"{label} path must be absolute")
    root = REPOSITORY_ROOT.resolve(strict=True)
    path = Path(path_value)
    expected_run_directory = root / "output" / "udlm" / run_name
    expected_path = expected_run_directory / basename
    if path != expected_path:
        raise GateValidationError(
            f"{label} path must equal output/udlm/{run_name}/{basename}"
        )
    try:
        resolved_parent = path.parent.resolve(strict=True)
    except OSError as error:
        raise GateValidationError(f"{label} parent is unavailable") from error
    normalized = resolved_parent / path.name
    if (
        resolved_parent != expected_run_directory
        or normalized != expected_path
        or normalized.name != basename
        or normalized.suffix != ".json"
    ):
        raise GateValidationError(f"{label} must not escape or traverse symlinks")
    payload = _stable_regular_file_bytes(normalized, label=label)
    observed_stat = normalized.stat(follow_symlinks=False)
    observed = {
        "path": str(normalized),
        "device": int(observed_stat.st_dev),
        "inode": int(observed_stat.st_ino),
        "mode": int(observed_stat.st_mode),
        "link_count": int(observed_stat.st_nlink),
        "size_bytes": int(observed_stat.st_size),
        "mtime_ns": int(observed_stat.st_mtime_ns),
        "ctime_ns": int(observed_stat.st_ctime_ns),
        "sha256": _sha256_bytes(payload),
        "stable_regular_file_verified": True,
    }
    if observed != dict(claim):
        raise GateValidationError(f"{label} no longer matches its bound snapshot")
    parsed = _mapping(strict_json_loads(payload, label=label), label)
    return parsed, observed


def _derive_predecessor_training_lock(
    *,
    run_name: str,
    manifest: Mapping[str, Any],
    manifest_snapshot: Mapping[str, Any],
    summary: Mapping[str, Any],
    summary_snapshot: Mapping[str, Any],
    receipt_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the validation inputs implied by producer-shaped predecessor evidence."""

    root = REPOSITORY_ROOT.resolve(strict=True)
    relative_run_directory = Path("output") / "udlm" / run_name
    run_directory = root / relative_run_directory
    runtime_claim = _mapping(
        summary.get("runtime_config"), "bound predecessor summary runtime config"
    )
    _exact_keys(
        runtime_claim,
        _STABLE_SNAPSHOT_KEYS | {"schema_version", "record_sha256"},
        "bound predecessor summary runtime config",
    )
    runtime_snapshot_claim = {key: runtime_claim[key] for key in _STABLE_SNAPSHOT_KEYS}
    _runtime, runtime_snapshot = _load_bound_predecessor_json(
        runtime_snapshot_claim,
        label="bound predecessor runtime config",
        basename="runtime_config.json",
        run_name=run_name,
    )
    final_checkpoint = _mapping(
        summary.get("final_checkpoint"), "bound predecessor final checkpoint"
    )
    checkpoint_path_value = final_checkpoint.get("path")
    max_steps = _integer(
        manifest.get("max_steps"), "bound predecessor max steps", minimum=1
    )
    expected_checkpoint_path = run_directory / "checkpoints" / f"{max_steps}.ckpt"
    if checkpoint_path_value != str(expected_checkpoint_path):
        raise GateValidationError(
            "bound predecessor checkpoint path must use its exact run directory"
        )
    accounting = _mapping(
        summary.get("training_accounting"),
        "bound predecessor training accounting",
    )
    parameters = _mapping(
        accounting.get("trainable_parameter_counts"),
        "bound predecessor trainable parameter counts",
    )
    semantic = _mapping(
        final_checkpoint.get("semantic_audit"),
        "bound predecessor checkpoint semantic audit",
    )
    ema_metadata = _mapping(
        semantic.get("ema_metadata"), "bound predecessor EMA metadata"
    )
    startup = _mapping(summary.get("startup"), "bound predecessor startup")
    checkpoint_sha256 = _sha256(
        final_checkpoint.get("sha256"), "bound predecessor checkpoint digest"
    )
    checkpoint_size = _integer(
        final_checkpoint.get("size_bytes"),
        "bound predecessor checkpoint size",
        minimum=1,
    )
    checkpoint_step = _integer(
        semantic.get("global_step"),
        "bound predecessor checkpoint step",
        minimum=1,
    )
    global_examples = _integer(
        accounting.get("effective_global_examples_per_optimizer_step"),
        "bound predecessor global examples",
        minimum=1,
    )
    optimizer_updates = _integer(
        accounting.get("optimizer_updates"),
        "bound predecessor optimizer updates",
        minimum=1,
    )
    derived_parameter_counts = {
        "base_model_trainable": parameters.get("base_backbone"),
        "time_conditioner_trainable": parameters.get("time_conditioner"),
        "total_trainable": parameters.get("total"),
    }
    if "film_modulation" in parameters:
        derived_parameter_counts["film_modulation_trainable"] = parameters.get(
            "film_modulation"
        )
    return {
        "summary": {
            "relative_path": relative_run_directory / "training_summary.json",
            "sha256": summary_snapshot["sha256"],
            "schema_version": summary.get("schema_version"),
        },
        "receipt": {
            "relative_path": relative_run_directory / "pilot_exit_status.json",
            "sha256": receipt_snapshot["sha256"],
            "schema_version": PILOT_EXIT_STATUS_SCHEMA_VERSION,
        },
        "runtime": {
            "relative_path": relative_run_directory / "runtime_config.json",
            "sha256": runtime_snapshot["sha256"],
            "schema_version": runtime_claim.get("schema_version"),
        },
        "launch_manifest": {
            "relative_path": relative_run_directory / "launch_manifest.json",
            "sha256": manifest_snapshot["sha256"],
            "schema_version": manifest.get("launch_manifest_schema_version"),
        },
        "resolved_training_config_sha256": manifest.get(
            "resolved_training_config_sha256"
        ),
        "training_argv_sha256": manifest.get("training_argv_sha256"),
        "checkpoint": {
            "relative_path": (
                relative_run_directory / "checkpoints" / f"{max_steps}.ckpt"
            ),
            "sha256": checkpoint_sha256,
            "size_bytes": checkpoint_size,
            "global_step": checkpoint_step,
        },
        "source_revision": manifest.get("git_sha"),
        "initialization_checkpoint_sha256": manifest.get("checkpoint_sha256"),
        "optimizer_updates": optimizer_updates,
        "world_size": manifest.get("user_requested_gpu_count"),
        "training_seed": accounting.get("training_seed"),
        "data_exposure": {
            "global_examples_per_optimizer_step": global_examples,
            "total_requested_examples": global_examples * optimizer_updates,
            "stream_partition_policy": accounting.get(
                "hosted_stream_rank_partition_policy"
            ),
        },
        "parameter_counts": derived_parameter_counts,
        "startup_mode": startup.get("mode"),
        "inference_weights": {
            "source": "ema",
            "ema_applied": True,
            "ema": dict(ema_metadata),
        },
    }


def _validate_predecessor_receipt_chain(
    manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    current_receipt_reference: Mapping[str, Any],
    recursion_depth: int = 0,
) -> dict[str, Any]:
    """Independently reconstruct the launch-bound R→S→E receipt chain."""

    if recursion_depth >= len(_MATCHED_PANEL_VARIANT_ORDER):
        raise GateValidationError("predecessor receipt chain is cyclic or too deep")
    panel = _mapping(
        manifest.get("matched_panel_spec"), "predecessor-chain matched-panel spec"
    )
    panel_sha256 = manifest.get("matched_panel_spec_sha256")
    if (
        panel.get("schema_version") != 2
        or not isinstance(panel_sha256, str)
        or canonical_json_sha256(panel) != panel_sha256
    ):
        raise GateValidationError("predecessor-chain matched-panel spec is unbound")
    common = _mapping(
        panel.get("common_training_contract"),
        "predecessor-chain common training contract",
    )
    common_sha256 = canonical_json_sha256(common)
    variant = manifest.get("training_variant")
    position = manifest.get("matched_panel_variant_position")
    if (
        variant not in _MATCHED_PANEL_VARIANT_ORDER
        or type(position) is not int
        or position != _MATCHED_PANEL_VARIANT_ORDER.index(variant)
    ):
        raise GateValidationError("predecessor-chain treatment position is invalid")
    current_receipt_path = current_receipt_reference.get("relative_path")
    current_receipt_sha256 = current_receipt_reference.get("sha256")
    if (
        not isinstance(current_receipt_path, Path)
        or not isinstance(current_receipt_sha256, str)
        or HEX_SHA256.fullmatch(current_receipt_sha256) is None
    ):
        raise GateValidationError(
            "predecessor-chain current receipt reference is invalid"
        )
    current_receipt_member = {
        "variant": variant,
        "relative_path": current_receipt_path.as_posix(),
        "sha256": current_receipt_sha256,
    }

    binding = _mapping(
        manifest.get("predecessor_receipt_binding"),
        "launch predecessor receipt binding",
    )
    _exact_keys(binding, _PREDECESSOR_BINDING_KEYS, "predecessor receipt binding")
    if (
        binding.get("schema_version") != 1
        or binding.get("current_training_variant") != variant
        or binding.get("current_variant_position") != position
        or binding.get("matched_panel_spec_sha256") != panel_sha256
        or binding.get("common_training_contract_sha256") != common_sha256
        or binding.get("validated_before_gpu_probe") is not True
    ):
        raise GateValidationError("launch predecessor receipt binding is inconsistent")
    if receipt.get("predecessor_receipt_binding") != binding:
        raise GateValidationError(
            "exit receipt does not mirror the launch predecessor binding"
        )
    current_completion = _mapping(
        receipt.get("completion_requirements"),
        "predecessor-chain receipt completion requirements",
    )
    _required_true(
        current_completion.get("predecessor_receipt_binding_unchanged_and_valid"),
        "receipt predecessor binding completion requirement",
    )

    artifact_fields = (
        "receipt_artifact",
        "predecessor_launch_manifest_artifact",
        "predecessor_training_summary_artifact",
    )
    if position == 0:
        if binding.get("state") != "explicit_genesis_no_predecessor":
            raise GateValidationError("R launch lacks explicit genesis state")
        for field in (
            "expected_predecessor_training_variant",
            "expected_predecessor_variant_position",
            "predecessor_run_name",
            "chronology",
            *artifact_fields,
        ):
            if binding.get(field) is not None:
                raise GateValidationError(
                    f"R genesis predecessor binding {field} must be null"
                )
        return {
            "chain_depth": 1,
            "variant_order_prefix": ["udlm"],
            "receipt_members": [current_receipt_member],
            "machine_enforced": True,
        }

    expected_position = position - 1
    expected_variant = _MATCHED_PANEL_VARIANT_ORDER[expected_position]
    if (
        binding.get("state") != "validated_successful_predecessor"
        or binding.get("expected_predecessor_training_variant") != expected_variant
        or binding.get("expected_predecessor_variant_position") != expected_position
    ):
        raise GateValidationError("S/E predecessor identity is not the prior arm")
    predecessor_run_name = binding.get("predecessor_run_name")
    if (
        not isinstance(predecessor_run_name, str)
        or RUN_NAME_PATTERN.fullmatch(predecessor_run_name) is None
    ):
        raise GateValidationError(
            "predecessor run name does not use the launcher-normalized syntax"
        )

    predecessor_receipt, receipt_snapshot = _load_bound_predecessor_json(
        binding.get("receipt_artifact"),
        label="bound predecessor exit receipt",
        basename="pilot_exit_status.json",
        run_name=predecessor_run_name,
    )
    predecessor_manifest, manifest_snapshot = _load_bound_predecessor_json(
        binding.get("predecessor_launch_manifest_artifact"),
        label="bound predecessor launch manifest",
        basename="launch_manifest.json",
        run_name=predecessor_run_name,
    )
    predecessor_summary, summary_snapshot = _load_bound_predecessor_json(
        binding.get("predecessor_training_summary_artifact"),
        label="bound predecessor training summary",
        basename="training_summary.json",
        run_name=predecessor_run_name,
    )
    artifact_parents = {
        Path(snapshot["path"]).parent
        for snapshot in (receipt_snapshot, manifest_snapshot, summary_snapshot)
    }
    if len(artifact_parents) != 1:
        raise GateValidationError(
            "predecessor artifacts do not share one run directory"
        )
    _exact_keys(
        predecessor_receipt,
        _PILOT_EXIT_STATUS_KEYS,
        "bound predecessor exit receipt",
    )
    _exact_keys(
        predecessor_manifest,
        _LAUNCH_MANIFEST_KEYS,
        "bound predecessor launch manifest",
    )
    _exact_keys(
        predecessor_summary,
        _TRAINING_SUMMARY_KEYS,
        "bound predecessor training summary",
    )
    if (
        predecessor_receipt.get("schema_version") != PILOT_EXIT_STATUS_SCHEMA_VERSION
        or predecessor_receipt.get("status") != "completed"
        or predecessor_receipt.get("overall_status") != "completed"
    ):
        raise GateValidationError("bound predecessor exit receipt is not successful")
    predecessor_process_status = _integer(
        predecessor_receipt.get("process_exit_status"),
        "bound predecessor process exit status",
    )
    if predecessor_process_status != 0:
        raise GateValidationError("bound predecessor exit receipt is not successful")
    predecessor_completion = _mapping(
        predecessor_receipt.get("completion_requirements"),
        "bound predecessor completion requirements",
    )
    if not predecessor_completion or any(
        value is not True for value in predecessor_completion.values()
    ):
        raise GateValidationError(
            "bound predecessor completion requirements are not all true"
        )
    _validate_successful_pipeline(
        predecessor_receipt.get("pipeline"), label="bound predecessor pipeline"
    )
    source_revision = common.get("source_revision")
    _validate_receipt_source(
        predecessor_receipt.get("source_at_receipt"),
        expected_revision=source_revision,
        label="bound predecessor source evidence",
    )

    if (
        predecessor_manifest.get("launch_manifest_schema_version")
        != LAUNCH_MANIFEST_SCHEMA_VERSION
        or predecessor_manifest.get("purpose") != "bounded UDLM training pilot"
        or predecessor_manifest.get("training_variant") != expected_variant
        or predecessor_manifest.get("matched_panel_variant_position")
        != expected_position
        or predecessor_manifest.get("matched_panel_spec") != panel
        or predecessor_manifest.get("matched_panel_spec_sha256") != panel_sha256
        or predecessor_manifest.get("run_name") != predecessor_run_name
    ):
        raise GateValidationError("bound predecessor launch manifest is inconsistent")
    if (
        predecessor_summary.get("schema_version") != TRAINING_SUMMARY_SCHEMA_VERSION
        or predecessor_summary.get("status") != "completed"
    ):
        raise GateValidationError("bound predecessor training summary is incomplete")
    predecessor_manifest_evidence = _mapping(
        predecessor_receipt.get("launch_manifest"),
        "bound predecessor receipt launch-manifest evidence",
    )
    predecessor_summary_evidence = _mapping(
        predecessor_receipt.get("training_summary"),
        "bound predecessor receipt summary evidence",
    )
    if (
        predecessor_manifest_evidence.get("artifact") != manifest_snapshot
        or predecessor_manifest_evidence.get("valid_and_launch_bound") is not True
        or predecessor_summary_evidence.get("artifact") != summary_snapshot
        or predecessor_summary_evidence.get("valid_and_launch_bound") is not True
    ):
        raise GateValidationError(
            "bound predecessor receipt snapshots are inconsistent"
        )
    summary_manifest_claim = _mapping(
        predecessor_summary.get("launch_manifest"),
        "bound predecessor summary launch-manifest evidence",
    )
    if any(
        summary_manifest_claim.get(key) != value
        for key, value in manifest_snapshot.items()
    ):
        raise GateValidationError(
            "bound predecessor summary launch-manifest snapshot is inconsistent"
        )

    chronology = _mapping(binding.get("chronology"), "predecessor chronology")
    _exact_keys(chronology, _PREDECESSOR_CHRONOLOGY_KEYS, "predecessor chronology")
    manifest_time = _timestamp(
        chronology.get("predecessor_launch_manifest_created_at_utc"),
        "predecessor manifest timestamp",
    )
    summary_time = _timestamp(
        chronology.get("predecessor_training_summary_completed_at_utc"),
        "predecessor summary timestamp",
    )
    receipt_time = _timestamp(
        chronology.get("predecessor_exit_receipt_recorded_at_utc"),
        "predecessor receipt timestamp",
    )
    _required_true(
        chronology.get("strictly_ordered_timestamps_verified"),
        "predecessor strict chronology flag",
    )
    if (
        not manifest_time
        < summary_time
        < receipt_time
        < _timestamp(manifest.get("created_at"), "current manifest timestamp")
    ):
        raise GateValidationError("R/S/E predecessor chronology is not strict")
    if (
        predecessor_manifest.get("created_at")
        != chronology.get("predecessor_launch_manifest_created_at_utc")
        or predecessor_summary.get("completed_at_utc")
        != chronology.get("predecessor_training_summary_completed_at_utc")
        or predecessor_receipt.get("recorded_at_utc")
        != chronology.get("predecessor_exit_receipt_recorded_at_utc")
    ):
        raise GateValidationError("R/S/E predecessor chronology is not artifact-bound")
    current_lock_binding = _mapping(
        manifest.get("single_training_job_lock"),
        "successor training-job lock binding",
    )
    current_lock_record = _mapping(
        current_lock_binding.get("record"),
        "successor training-job lock record",
    )
    current_lock_acquired_at = _timestamp(
        current_lock_record.get("acquired_at_utc"),
        "successor training-job lock acquisition timestamp",
    )
    current_inventory_completed_at = _timestamp(
        manifest.get("inventory_snapshot_completed_at_utc"),
        "successor GPU inventory timestamp",
    )
    if not receipt_time < current_lock_acquired_at <= current_inventory_completed_at:
        raise GateValidationError(
            "predecessor receipt must predate successor lock acquisition and GPU probe"
        )

    predecessor_lock = _derive_predecessor_training_lock(
        run_name=predecessor_run_name,
        manifest=predecessor_manifest,
        manifest_snapshot=manifest_snapshot,
        summary=predecessor_summary,
        summary_snapshot=summary_snapshot,
        receipt_snapshot=receipt_snapshot,
    )
    predecessor_evidence = validate_training_evidence(
        predecessor_lock,
        _recursion_depth=recursion_depth + 1,
    )
    prefix = predecessor_evidence["predecessor_receipt_chain"]
    if prefix["chain_depth"] != position:
        raise GateValidationError("R/S/E predecessor chain has the wrong depth")
    return {
        "chain_depth": position + 1,
        "variant_order_prefix": [*prefix["variant_order_prefix"], variant],
        "receipt_members": [*prefix["receipt_members"], current_receipt_member],
        "machine_enforced": True,
    }


def _validate_gpu_state(value: object, *, label: str) -> dict[str, Any]:
    state = _mapping(value, label)
    _exact_keys(state, _GPU_STATE_KEYS, label)
    _integer(state.get("physical_index"), f"{label}.physical_index", minimum=0)
    uuid = _selected_gpu_uuids([state.get("uuid")], f"{label}.uuid")[0]
    if not isinstance(state.get("name"), str) or not state["name"]:
        raise GateValidationError(f"{label}.name must be nonempty")
    memory_used = _integer(
        state.get("memory_used_mib"), f"{label}.memory_used_mib", minimum=0
    )
    memory_total = _integer(
        state.get("memory_total_mib"), f"{label}.memory_total_mib", minimum=1
    )
    if memory_used > memory_total:
        raise GateValidationError(f"{label} used memory exceeds total memory")
    utilization = _integer(
        state.get("utilization_percent"), f"{label}.utilization_percent", minimum=0
    )
    if utilization > 100:
        raise GateValidationError(f"{label}.utilization_percent exceeds 100")
    if not isinstance(state.get("compute_mode"), str) or not state["compute_mode"]:
        raise GateValidationError(f"{label}.compute_mode must be nonempty")
    processes = state.get("compute_processes")
    if not isinstance(processes, list) or not all(
        isinstance(process, Mapping) for process in processes
    ):
        raise GateValidationError(f"{label}.compute_processes must be an object array")
    for index, process in enumerate(processes):
        process_label = f"{label}.compute_processes[{index}]"
        _exact_keys(
            process,
            {"pid", "process_name", "used_memory_mib"},
            process_label,
        )
        _integer(process.get("pid"), f"{process_label}.pid", minimum=1)
        if (
            not isinstance(process.get("process_name"), str)
            or not process["process_name"]
        ):
            raise GateValidationError(f"{process_label}.process_name must be nonempty")
        used_memory = process.get("used_memory_mib")
        if used_memory is not None:
            _integer(used_memory, f"{process_label}.used_memory_mib", minimum=0)
    return {**dict(state), "uuid": uuid}


def _validate_launch_manifest(
    manifest: Mapping[str, Any],
    *,
    lock: Mapping[str, Any],
) -> list[str]:
    _exact_keys(manifest, _LAUNCH_MANIFEST_KEYS, "launch manifest")
    if manifest.get("launch_manifest_schema_version") != LAUNCH_MANIFEST_SCHEMA_VERSION:
        raise GateValidationError("launch manifest schema version is unsupported")
    if manifest.get("gpu_selection_schema_version") != 2:
        raise GateValidationError("launch manifest GPU-selection schema is unsupported")
    if manifest.get("purpose") != "bounded UDLM training pilot":
        raise GateValidationError("launch manifest purpose is unexpected")
    if manifest.get("git_sha") != lock["source_revision"]:
        raise GateValidationError("launch manifest source revision disagrees with lock")
    if (
        manifest.get("source_revision_before_final_gpu_probe")
        != lock["source_revision"]
    ):
        raise GateValidationError(
            "launch manifest final-probe source revision disagrees with lock"
        )
    manifest_created_at = _timestamp(
        manifest.get("created_at"), "launch manifest created_at"
    )
    inventory_completed_at = _timestamp(
        manifest.get("inventory_snapshot_completed_at_utc"),
        "launch manifest inventory_snapshot_completed_at_utc",
    )
    final_probe_completed_at = _timestamp(
        manifest.get("final_uuid_probes_completed_at_utc"),
        "launch manifest final_uuid_probes_completed_at_utc",
    )
    if not inventory_completed_at <= final_probe_completed_at <= manifest_created_at:
        raise GateValidationError("launch manifest GPU-probe chronology is invalid")
    for field in (
        "run_name",
        "training_variant",
        "hydra_config_name",
        "udlm_prior_variant",
        "udlm_comparison_role",
        "tmux_session",
    ):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise GateValidationError(f"launch manifest {field} must be nonempty")
    run_name = manifest["run_name"]
    if RUN_NAME_PATTERN.fullmatch(run_name) is None:
        raise GateValidationError("launch manifest run name has invalid syntax")
    expected_run_directory = Path("output") / "udlm" / run_name
    expected_run_artifacts = {
        "launch_manifest": expected_run_directory / "launch_manifest.json",
        "runtime": expected_run_directory / "runtime_config.json",
        "summary": expected_run_directory / "training_summary.json",
        "receipt": expected_run_directory / "pilot_exit_status.json",
        "checkpoint": (
            expected_run_directory / "checkpoints" / f"{lock['optimizer_updates']}.ckpt"
        ),
    }
    for lock_field, expected_path in expected_run_artifacts.items():
        if lock[lock_field]["relative_path"] != expected_path:
            raise GateValidationError(
                f"candidate {lock_field} path does not match launch run name"
            )
    if manifest.get("tmux_session") != (
        f"genmol_{manifest['training_variant']}_{run_name}"
    ):
        raise GateValidationError("launch manifest tmux session is not run-bound")
    if manifest.get("log_path") != str(
        _expected_artifact_path(Path("output/logs") / f"{run_name}.log")
    ):
        raise GateValidationError("launch manifest log path is not run-bound")
    world_size = lock["world_size"]
    if manifest.get("user_requested_gpu_count") != world_size:
        raise GateValidationError("launch manifest GPU count disagrees with lock")
    selected_gpu_uuids = _selected_gpu_uuids(
        manifest.get("cuda_visible_device_uuids"),
        "launch manifest selected GPU UUIDs",
        expected_count=world_size,
    )
    if manifest.get("logical_cuda_devices") != list(range(world_size)):
        raise GateValidationError("launch manifest logical CUDA devices are unexpected")
    physical_indices = manifest.get("physical_gpu_indices")
    if (
        not isinstance(physical_indices, list)
        or len(physical_indices) != world_size
        or len(set(physical_indices)) != world_size
        or any(type(index) is not int or index < 0 for index in physical_indices)
    ):
        raise GateValidationError("launch manifest physical GPU indices are invalid")
    if (
        manifest.get("gpu_selection_method") != "dynamic_idle_discovery"
        or manifest.get("gpu_inventory_scope") != "all_nvidia_gpus"
    ):
        raise GateValidationError("launch manifest GPU selection method is unexpected")

    inventory_raw = manifest.get("gpu_inventory_at_selection")
    initial_raw = manifest.get("initially_selected_gpu_states")
    final_raw = manifest.get("gpu_states_at_final_uuid_probe")
    if not isinstance(inventory_raw, list) or not inventory_raw:
        raise GateValidationError("launch manifest GPU inventory must be nonempty")
    if not isinstance(initial_raw, list) or len(initial_raw) != world_size:
        raise GateValidationError("launch manifest initial GPU selection is incomplete")
    if not isinstance(final_raw, list) or len(final_raw) != world_size:
        raise GateValidationError("launch manifest final GPU probe is incomplete")
    inventory = [
        _validate_gpu_state(row, label=f"launch manifest inventory GPU {index}")
        for index, row in enumerate(inventory_raw)
    ]
    initial = [
        _validate_gpu_state(row, label=f"launch manifest initial GPU {index}")
        for index, row in enumerate(initial_raw)
    ]
    final = [
        _validate_gpu_state(row, label=f"launch manifest final GPU {index}")
        for index, row in enumerate(final_raw)
    ]
    if [row["uuid"] for row in initial] != selected_gpu_uuids:
        raise GateValidationError("launch manifest initial selected UUIDs disagree")
    if [row["uuid"] for row in final] != selected_gpu_uuids:
        raise GateValidationError("launch manifest final selected UUIDs disagree")
    if [row["physical_index"] for row in final] != physical_indices:
        raise GateValidationError("launch manifest final physical indices disagree")
    inventory_uuids = [row["uuid"] for row in inventory]
    inventory_physical_indices = [row["physical_index"] for row in inventory]
    if (
        len(set(inventory_uuids)) != len(inventory_uuids)
        or len(set(inventory_physical_indices)) != len(inventory_physical_indices)
        or not set(selected_gpu_uuids).issubset(inventory_uuids)
    ):
        raise GateValidationError(
            "launch manifest GPU inventory identities are invalid"
        )
    inventory_by_uuid = {row["uuid"]: row for row in inventory}
    if any(row != inventory_by_uuid[row["uuid"]] for row in initial):
        raise GateValidationError(
            "launch manifest initial selection differs from its inventory snapshot"
        )

    safety = _mapping(manifest.get("gpu_safety_policy"), "launch GPU safety policy")
    _exact_keys(
        safety,
        {
            "max_utilization_percent",
            "utilization_comparison",
            "min_free_memory_mib",
            "active_compute_processes_allowed",
            "compute_mode_prohibited_allowed",
        },
        "launch GPU safety policy",
    )
    max_utilization = _integer(
        safety.get("max_utilization_percent"),
        "launch GPU maximum utilization",
        minimum=1,
    )
    min_free_memory = _integer(
        safety.get("min_free_memory_mib"), "launch GPU minimum free memory", minimum=1
    )
    if (
        safety.get("utilization_comparison") != "strictly_less_than"
        or max_utilization > MAX_SAFE_UTILIZATION_PERCENT
        or min_free_memory < MIN_SAFE_FREE_MEMORY_MIB
        or safety.get("active_compute_processes_allowed")
        is not ACTIVE_COMPUTE_PROCESSES_ALLOWED
        or safety.get("compute_mode_prohibited_allowed") is not False
    ):
        raise GateValidationError("launch GPU safety policy is unexpected")
    for index, state in enumerate((*initial, *final)):
        if (
            state["utilization_percent"] >= max_utilization
            or state["memory_total_mib"] - state["memory_used_mib"] < min_free_memory
            or state["compute_mode"].lower() == "prohibited"
            or (
                state["compute_processes"]
                and not safety["active_compute_processes_allowed"]
            )
        ):
            raise GateValidationError(
                f"launch manifest selected GPU state {index} violates safety policy"
            )

    training_argv = manifest.get("training_argv")
    if (
        not isinstance(training_argv, list)
        or len(training_argv) < 5
        or not all(isinstance(argument, str) and argument for argument in training_argv)
    ):
        raise GateValidationError(
            "launch manifest training argv must be a nonempty string array"
        )
    expected_suffix = [
        "-u",
        str(REPOSITORY_ROOT / "scripts" / "train.py"),
        "--config-name",
        manifest["hydra_config_name"],
    ]
    reviewed_interpreters = {
        str(candidate) for candidate in _reviewed_training_python_executables()
    }
    if training_argv[0] not in reviewed_interpreters or training_argv[1:5] != (
        expected_suffix
    ):
        raise GateValidationError(
            "launch manifest training argv producer prefix is unexpected"
        )
    if (
        manifest.get("training_argv_sha256") != lock["training_argv_sha256"]
        or canonical_json_sha256(training_argv[2:]) != lock["training_argv_sha256"]
    ):
        raise GateValidationError("launch manifest training argv is unbound")
    resolved_config = _mapping(
        manifest.get("resolved_training_config"), "launch resolved training config"
    )
    if (
        manifest.get("resolved_training_config_sha256")
        != lock["resolved_training_config_sha256"]
        or canonical_json_sha256(resolved_config)
        != lock["resolved_training_config_sha256"]
    ):
        raise GateValidationError("launch manifest resolved training config is unbound")
    resolved_training = _mapping(
        resolved_config.get("training"), "launch resolved training section"
    )
    resolved_udlm = _mapping(
        resolved_training.get("udlm"), "launch resolved UDLM section"
    )
    if resolved_udlm.get("prior_variant") != manifest.get(
        "udlm_prior_variant"
    ) or resolved_udlm.get("exclude_special_tokens") != manifest.get(
        "exclude_special_tokens"
    ):
        raise GateValidationError(
            "launch resolved UDLM treatment or token support disagrees with the manifest"
        )

    expected_paths = {
        "launch_manifest_path": lock["launch_manifest"]["relative_path"],
        "runtime_config_path": lock["runtime"]["relative_path"],
        "training_summary_path": lock["summary"]["relative_path"],
        "pilot_exit_status_path": lock["receipt"]["relative_path"],
        "expected_final_checkpoint_path": lock["checkpoint"]["relative_path"],
    }
    for field, relative_path in expected_paths.items():
        if manifest.get(field) != str(_expected_artifact_path(relative_path)):
            raise GateValidationError(f"launch manifest {field} disagrees with lock")
    if (
        manifest.get("training_summary_schema_version")
        != TRAINING_SUMMARY_SCHEMA_VERSION
        or manifest.get("pilot_exit_status_schema_version")
        != PILOT_EXIT_STATUS_SCHEMA_VERSION
        or manifest.get("launch_manifest_raw_sha256_transport")
        != "passed_out_of_band_to_training_and_receipt_to_avoid_self_hash"
        or manifest.get("dry_run") is not False
        or manifest.get("log_reserved_exclusively_before_manifest") is not True
    ):
        raise GateValidationError("launch manifest completion schema is unexpected")
    completion_contract = _mapping(
        manifest.get("completion_contract"), "launch completion contract"
    )
    expected_completion_contract = {
        "status_at_launch": "pending",
        "complete_only_if_valid_training_summary_exists": True,
        "complete_only_if_successful_exit_receipt_exists": True,
        "valid_training_summary_and_successful_exit_receipt_both_required": True,
        "missing_summary_after_tmux_exit_means": "incomplete",
        "absent_exit_receipt_means": "incomplete",
        "successful_exit_receipt_requires": {
            "training_exit_status": 0,
            "tee_exit_status": 0,
            "valid_launch_bound_training_summary": True,
            "exact_launch_manifest_still_matches": True,
            "clean_pushed_source_at_receipt": True,
            "predecessor_receipt_binding_unchanged_and_valid": True,
        },
        "training_job_lock_release": (
            "after_exit_receipt_publication_for_completed_or_failed_pipeline"
        ),
    }
    if dict(completion_contract) != expected_completion_contract:
        raise GateValidationError("launch manifest completion contract is unexpected")
    if manifest.get("checkpoint_sha256") != lock["initialization_checkpoint_sha256"]:
        raise GateValidationError("launch manifest initialization digest disagrees")
    if manifest.get("seed") != lock["training_seed"]:
        raise GateValidationError("launch manifest training seed disagrees with lock")
    if manifest.get("max_steps") != lock["optimizer_updates"]:
        raise GateValidationError("launch manifest max steps disagree with lock")
    micro_batch = _integer(
        manifest.get("micro_batch_size_per_process"),
        "launch manifest micro batch size",
        minimum=1,
    )
    accumulation = _integer(
        manifest.get("accumulate_grad_batches"),
        "launch manifest gradient accumulation",
        minimum=1,
    )
    global_batch = _integer(
        manifest.get("global_batch_size"), "launch manifest global batch", minimum=1
    )
    effective_batch = _integer(
        manifest.get("effective_global_batch_size"),
        "launch manifest effective global batch",
        minimum=1,
    )
    if (
        effective_batch != micro_batch * world_size * accumulation
        or effective_batch != global_batch
        or effective_batch
        != lock["data_exposure"]["global_examples_per_optimizer_step"]
    ):
        raise GateValidationError(
            "launch manifest batch arithmetic disagrees with lock"
        )
    if type(manifest.get("exclude_special_tokens")) is not bool:
        raise GateValidationError("launch manifest special-token flag must be boolean")

    matched_panel = _mapping(
        manifest.get("matched_panel_spec"), "launch matched-panel spec"
    )
    if (
        canonical_json_sha256(matched_panel)
        != manifest.get("matched_panel_spec_sha256")
        or matched_panel.get("schema_version") != 2
    ):
        raise GateValidationError("launch matched-panel specification is unbound")
    _exact_keys(
        matched_panel,
        {
            "schema_version",
            "purpose",
            "execution",
            "registered_treatments",
            "common_training_contract",
            "common_gpu_safety_policy",
        },
        "launch matched-panel spec",
    )
    if matched_panel.get("purpose") != "matched_R_S_E_UDLM_training_pilot":
        raise GateValidationError("launch matched-panel purpose is unexpected")
    execution = _mapping(
        matched_panel.get("execution"), "launch matched-panel execution"
    )
    if dict(execution) != {
        "mode": "single_job_lease_with_machine_enforced_predecessor_receipt_chain",
        "maximum_concurrent_training_jobs": 1,
        "concurrency_enforcement": "atomic_global_worktree_training_job_lock",
        "registered_variant_order": [
            "udlm",
            "schedule_uniform",
            "udlm_categorical",
        ],
        "advance_policy": (
            "launcher_validates_and_binds_exact_successful_predecessor_receipt"
        ),
        "predecessor_receipt_bound_in_each_manifest": True,
        "genesis_requires_explicit_declaration": True,
        "successor_launch_requires_exact_predecessor_receipt": True,
    }:
        raise GateValidationError("launch matched-panel execution policy is unexpected")
    panel_training = _mapping(
        matched_panel.get("common_training_contract"),
        "launch matched-panel training contract",
    )
    _exact_keys(
        panel_training,
        {
            "source_revision",
            "initialization_mode",
            "initialization_checkpoint_path",
            "initialization_checkpoint_sha256",
            "requested_gpu_count",
            "max_steps",
            "global_batch_size",
            "micro_batch_size_per_process",
            "accumulate_grad_batches",
            "effective_global_batch_size",
            "num_workers",
            "seed",
            "exclude_special_tokens",
            "empirical_uniform_mix",
            "empirical_uniform_mix_consumed_only_by",
            "empirical_uniform_mix_audit",
            "common_resolved_config_sha256",
        },
        "launch matched-panel training contract",
    )
    expected_initialization_mode = (
        "verified_mdlm_ema_warm_start"
        if lock["startup_mode"] == "warm_start"
        else "scratch"
    )
    common_config_sha256 = matched_panel_config_sha256(resolved_config)
    if (
        panel_training.get("source_revision") != lock["source_revision"]
        or panel_training.get("initialization_mode") != expected_initialization_mode
        or panel_training.get("initialization_checkpoint_path")
        != manifest.get("checkpoint")
        or panel_training.get("requested_gpu_count") != world_size
        or panel_training.get("max_steps") != lock["optimizer_updates"]
        or panel_training.get("global_batch_size") != global_batch
        or panel_training.get("micro_batch_size_per_process") != micro_batch
        or panel_training.get("accumulate_grad_batches") != accumulation
        or panel_training.get("seed") != lock["training_seed"]
        or panel_training.get("effective_global_batch_size") != effective_batch
        or panel_training.get("initialization_checkpoint_sha256")
        != lock["initialization_checkpoint_sha256"]
        or panel_training.get("exclude_special_tokens")
        != manifest.get("exclude_special_tokens")
        or panel_training.get("empirical_uniform_mix") != PILOT_EMPIRICAL_UNIFORM_MIX
        or panel_training.get("empirical_uniform_mix_consumed_only_by")
        != "empirical_frequency"
        or panel_training.get("empirical_uniform_mix_audit")
        != PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT
        or panel_training.get("common_resolved_config_sha256") != common_config_sha256
    ):
        raise GateValidationError("launch matched-panel contract disagrees with lock")
    resolved_loader = _mapping(
        resolved_config.get("loader"), "launch resolved loader config"
    )
    resolved_trainer = _mapping(
        resolved_config.get("trainer"), "launch resolved trainer config"
    )
    resolved_callback = _mapping(
        resolved_config.get("callback"), "launch resolved callback config"
    )
    resolved_training = _mapping(
        resolved_config.get("training"), "launch resolved training config"
    )
    resolved_udlm = _mapping(
        resolved_training.get("udlm"), "launch resolved UDLM config"
    )
    resolved_num_workers = _integer(
        resolved_loader.get("num_workers"),
        "launch resolved loader num_workers",
        minimum=0,
    )
    expected_checkpoint_directory = str(
        Path(manifest["expected_final_checkpoint_path"]).parent
    )
    if (
        panel_training.get("num_workers") != resolved_num_workers
        or resolved_config.get("seed") != manifest.get("seed")
        or resolved_loader.get("batch_size") != micro_batch
        or resolved_loader.get("global_batch_size") != global_batch
        or resolved_trainer.get("devices") != world_size
        or resolved_trainer.get("num_nodes") != 1
        or resolved_trainer.get("max_steps") != manifest.get("max_steps")
        or resolved_trainer.get("accumulate_grad_batches") != accumulation
        or resolved_callback.get("dirpath") != expected_checkpoint_directory
        or resolved_udlm.get("empirical_uniform_mix") != PILOT_EMPIRICAL_UNIFORM_MIX
    ):
        raise GateValidationError(
            "launch matched-panel controls disagree with the resolved config"
        )
    registered = matched_panel.get("registered_treatments")
    expected_registered = [
        {
            "training_variant": "udlm",
            "hydra_config_name": "udlm",
            "udlm_prior_variant": "release_uniform",
            "comparison_role": "faithful_release_control",
        },
        {
            "training_variant": "schedule_uniform",
            "hydra_config_name": "udlm",
            "udlm_prior_variant": "schedule_uniform",
            "comparison_role": "schedule_repair_uniform_control",
        },
        {
            "training_variant": "udlm_categorical",
            "hydra_config_name": "udlm_categorical",
            "udlm_prior_variant": "empirical_frequency",
            "comparison_role": "empirical_prior_treatment",
        },
    ]
    if registered != expected_registered:
        raise GateValidationError("launch matched-panel treatments are not canonical")
    position = manifest.get("matched_panel_variant_position")
    if (
        not isinstance(registered, list)
        or type(position) is not int
        or not 0 <= position < len(registered)
        or not isinstance(registered[position], Mapping)
        or registered[position].get("training_variant")
        != manifest.get("training_variant")
    ):
        raise GateValidationError("launch matched-panel treatment is unbound")
    selected_treatment = registered[position]
    for manifest_field, treatment_field in (
        ("hydra_config_name", "hydra_config_name"),
        ("udlm_prior_variant", "udlm_prior_variant"),
        ("udlm_comparison_role", "comparison_role"),
    ):
        if manifest.get(manifest_field) != selected_treatment.get(treatment_field):
            raise GateValidationError(
                f"launch {manifest_field} disagrees with matched-panel treatment"
            )

    panel_safety = _mapping(
        matched_panel.get("common_gpu_safety_policy"),
        "launch matched-panel GPU safety policy",
    )
    if dict(panel_safety) != {
        **dict(safety),
        "physical_gpu_identity_is_per_run_provenance": True,
    }:
        raise GateValidationError(
            "launch matched-panel GPU safety policy disagrees with the launch"
        )

    lock_binding = _mapping(
        manifest.get("single_training_job_lock"),
        "launch single-training-job lock binding",
    )
    _exact_keys(
        lock_binding,
        {
            "path",
            "sha256",
            "record",
            "acquired_before_any_gpu_probe",
            "stale_lock_policy",
            "release_owner",
        },
        "launch single-training-job lock binding",
    )
    expected_lock_path = _expected_artifact_path(
        Path("output/udlm/.single_training_job.lock")
    )
    if lock_binding.get("path") != str(expected_lock_path):
        raise GateValidationError("launch training-job lock path is unexpected")
    lock_sha256 = _sha256(lock_binding.get("sha256"), "launch training-job lock digest")
    lock_record = _mapping(
        lock_binding.get("record"), "launch training-job lock record"
    )
    _exact_keys(
        lock_record,
        {
            "schema_version",
            "status",
            "purpose",
            "source_revision",
            "run_name",
            "training_variant",
            "owner_token",
            "launcher_pid_at_acquisition",
            "acquired_at_utc",
            "owner_process_exit_does_not_make_lock_stale",
            "stale_lock_policy",
            "release_policy",
        },
        "launch training-job lock record",
    )
    if (
        lock_record.get("schema_version") != 1
        or lock_record.get("status") != "held"
        or lock_record.get("purpose")
        != "enforce_one_R_S_E_pilot_training_job_at_a_time"
        or lock_record.get("source_revision") != lock["source_revision"]
        or lock_record.get("run_name") != manifest["run_name"]
        or lock_record.get("training_variant") != manifest["training_variant"]
        or lock_record.get("owner_process_exit_does_not_make_lock_stale") is not True
        or lock_record.get("stale_lock_policy")
        != "fail_closed_and_require_manual_review"
        or lock_record.get("release_policy")
        != "exact_owner_lock_only_after_receipt_or_before_tmux_handoff_failure"
    ):
        raise GateValidationError("launch training-job lock record is unexpected")
    owner_token = lock_record.get("owner_token")
    if not isinstance(owner_token, str) or HEX_SHA256.fullmatch(owner_token) is None:
        raise GateValidationError("launch training-job lock owner token is invalid")
    _integer(
        lock_record.get("launcher_pid_at_acquisition"),
        "launch training-job lock launcher PID",
        minimum=1,
    )
    lock_acquired_at = _timestamp(
        lock_record.get("acquired_at_utc"),
        "launch training-job lock acquisition timestamp",
    )
    if lock_acquired_at > inventory_completed_at:
        raise GateValidationError(
            "launch training-job lock was acquired after GPU inventory probing"
        )
    lock_payload = (
        json.dumps(lock_record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if _sha256_bytes(lock_payload) != lock_sha256:
        raise GateValidationError("launch training-job lock record digest is unbound")
    if (
        lock_binding.get("acquired_before_any_gpu_probe") is not True
        or lock_binding.get("stale_lock_policy")
        != "fail_closed_and_require_manual_review"
        or lock_binding.get("release_owner")
        != "pilot_exit_receipt_writer_after_publication"
    ):
        raise GateValidationError("launch training-job lock binding is unexpected")
    return selected_gpu_uuids


def _load_launch_manifest(
    lock: Mapping[str, Any],
) -> tuple[Mapping[str, Any], int, list[str]]:
    reference = lock["launch_manifest"]
    payload = _repository_artifact_bytes(
        reference["relative_path"], label="training launch manifest"
    )
    if _sha256_bytes(payload) != reference["sha256"]:
        raise GateValidationError(
            "training launch manifest digest disagrees with candidate lock"
        )
    manifest = _mapping(
        strict_json_loads(payload, label="training launch manifest"),
        "training launch manifest",
    )
    if manifest.get("launch_manifest_schema_version") != reference["schema_version"]:
        raise GateValidationError(
            "training launch manifest schema version disagrees with candidate lock"
        )
    selected_gpu_uuids = _validate_launch_manifest(manifest, lock=lock)
    return manifest, len(payload), selected_gpu_uuids


def validate_training_evidence(
    lock: Mapping[str, Any], *, _recursion_depth: int = 0
) -> dict[str, Any]:
    """Join the lock to the launch-bound training summary and exit receipt."""

    summary = _load_referenced_json(lock["summary"], label="training summary")
    receipt = _load_referenced_json(lock["receipt"], label="training exit receipt")
    runtime = _load_referenced_json(lock["runtime"], label="training runtime config")
    summary_size_bytes = len(
        _repository_artifact_bytes(
            lock["summary"]["relative_path"], label="training summary"
        )
    )
    runtime_size_bytes = len(
        _repository_artifact_bytes(
            lock["runtime"]["relative_path"], label="training runtime config"
        )
    )
    _manifest, manifest_size_bytes, selected_gpu_uuids = _load_launch_manifest(lock)
    checkpoint_snapshot = _repository_artifact_snapshot(
        lock["checkpoint"]["relative_path"], label="training checkpoint"
    )
    if (
        checkpoint_snapshot["sha256"] != lock["checkpoint"]["sha256"]
        or checkpoint_snapshot["size_bytes"] != lock["checkpoint"]["size_bytes"]
    ):
        raise GateValidationError(
            "training checkpoint bytes disagree with the candidate lock"
        )
    _exact_keys(summary, _TRAINING_SUMMARY_KEYS, "training summary")
    _exact_keys(receipt, _PILOT_EXIT_STATUS_KEYS, "training exit receipt")
    _exact_keys(runtime, _RUNTIME_CONFIG_KEYS, "training runtime config")
    if summary.get("schema_version") != TRAINING_SUMMARY_SCHEMA_VERSION:
        raise GateValidationError("training summary schema version is unsupported")
    if receipt.get("schema_version") != PILOT_EXIT_STATUS_SCHEMA_VERSION:
        raise GateValidationError("training exit receipt schema version is unsupported")
    if runtime.get("schema_version") != RUNTIME_CONFIG_SCHEMA_VERSION:
        raise GateValidationError(
            "training runtime-config schema version is unsupported"
        )
    _validate_training_source(
        summary.get("source"),
        expected_revision=lock["source_revision"],
        label="training summary source",
    )
    _validate_training_source(
        runtime.get("source"),
        expected_revision=lock["source_revision"],
        label="training runtime source",
    )
    launch_time = _utc_timestamp(_manifest.get("created_at"), "launch timestamp")
    summary_time = _utc_timestamp(
        summary.get("completed_at_utc"), "training summary completion timestamp"
    )
    receipt_time = _utc_timestamp(
        receipt.get("recorded_at_utc"), "training exit receipt timestamp"
    )
    if not launch_time < summary_time < receipt_time:
        raise GateValidationError(
            "training launch, summary, and receipt chronology is not strict"
        )
    predecessor_chain = _validate_predecessor_receipt_chain(
        _manifest,
        receipt,
        current_receipt_reference=lock["receipt"],
        recursion_depth=_recursion_depth,
    )
    if summary.get("status") != "completed":
        raise GateValidationError("training summary is not completed")
    if summary.get("source_revision") != lock["source_revision"]:
        raise GateValidationError(
            "training summary source revision disagrees with lock"
        )
    if (
        summary.get("resolved_training_config_sha256")
        != lock["resolved_training_config_sha256"]
    ):
        raise GateValidationError("training summary config digest disagrees with lock")
    if summary.get("training_argv_sha256") != lock["training_argv_sha256"]:
        raise GateValidationError("training summary argv digest disagrees with lock")
    manifest_path = _expected_artifact_path(lock["launch_manifest"]["relative_path"])
    summary_manifest_claim = _validate_snapshot_claim(
        summary.get("launch_manifest"),
        label="summary launch-manifest evidence",
        expected_path=manifest_path,
        expected_sha256=lock["launch_manifest"]["sha256"],
        expected_size_bytes=manifest_size_bytes,
        include_selected_gpu_uuids=True,
        expected_selected_gpu_uuids=selected_gpu_uuids,
    )
    completion_contract = _mapping(
        summary.get("completion_contract"), "summary completion contract"
    )
    _exact_keys(
        completion_contract,
        {
            "summary_schema_version",
            "summary_path",
            "final_checkpoint_path",
            "expected_max_steps",
            "expected_world_size",
            "fail_on_nonfinite_loss",
            "backward_anomaly_detection",
        },
        "summary completion contract",
    )
    expected_completion_contract = {
        "summary_schema_version": TRAINING_SUMMARY_SCHEMA_VERSION,
        "summary_path": str(_expected_artifact_path(lock["summary"]["relative_path"])),
        "final_checkpoint_path": str(
            _expected_artifact_path(lock["checkpoint"]["relative_path"])
        ),
        "expected_max_steps": lock["optimizer_updates"],
        "expected_world_size": lock["world_size"],
        "fail_on_nonfinite_loss": True,
        "backward_anomaly_detection": True,
    }
    if dict(completion_contract) != expected_completion_contract:
        raise GateValidationError("summary completion contract disagrees with lock")
    runtime_claim = _mapping(summary.get("runtime_config"), "summary runtime config")
    _exact_keys(
        runtime_claim,
        _STABLE_SNAPSHOT_KEYS | {"schema_version", "record_sha256"},
        "summary runtime config",
    )
    summary_runtime_artifact = _validate_snapshot_claim(
        {key: runtime_claim[key] for key in _STABLE_SNAPSHOT_KEYS},
        label="summary runtime-config artifact",
        expected_path=_expected_artifact_path(lock["runtime"]["relative_path"]),
        expected_sha256=lock["runtime"]["sha256"],
        expected_size_bytes=runtime_size_bytes,
    )
    if runtime_claim.get("schema_version") != RUNTIME_CONFIG_SCHEMA_VERSION:
        raise GateValidationError("summary runtime-config schema is unsupported")
    if runtime_claim.get("record_sha256") != canonical_json_sha256(runtime):
        raise GateValidationError(
            "summary runtime-config canonical record digest is unbound"
        )
    observed = _mapping(
        summary.get("observed_training_state"), "observed training state"
    )
    _exact_keys(
        observed,
        {"global_rank", "global_step", "world_size"},
        "observed training state",
    )
    if observed.get("global_rank") != 0:
        raise GateValidationError("training summary must be published by global rank 0")
    if observed.get("global_step") != lock["optimizer_updates"]:
        raise GateValidationError("training summary step count disagrees with lock")
    if observed.get("world_size") != lock["world_size"]:
        raise GateValidationError("training summary world size disagrees with lock")
    final_checkpoint = _mapping(summary.get("final_checkpoint"), "summary checkpoint")
    _exact_keys(
        final_checkpoint,
        _STABLE_SNAPSHOT_KEYS | {"semantic_audit"},
        "summary checkpoint",
    )
    summary_checkpoint_artifact = _validate_snapshot_claim(
        {key: final_checkpoint[key] for key in _STABLE_SNAPSHOT_KEYS},
        label="summary checkpoint artifact",
        expected_path=_expected_artifact_path(lock["checkpoint"]["relative_path"]),
        expected_sha256=lock["checkpoint"]["sha256"],
        expected_size_bytes=lock["checkpoint"]["size_bytes"],
    )
    if summary_checkpoint_artifact != checkpoint_snapshot:
        raise GateValidationError(
            "summary checkpoint snapshot disagrees with the live artifact"
        )
    semantic = _mapping(
        final_checkpoint.get("semantic_audit"), "checkpoint semantic audit"
    )
    _exact_keys(
        semantic,
        {
            "deserialized",
            "global_step",
            "raw_model",
            "ema",
            "ema_metadata",
            "optimizer",
            "non_sentinel_checkpoint_tensors",
            "checkpoint_python_floats",
            "framework_nonfinite_sentinels",
            "checkpoint_hyperparameters_match",
            "checkpoint_loop_state_match",
            "optimizer_live_state_match",
            "scheduler_live_state_match",
            "sampler_live_state_match",
            "trainer_live_configuration_match",
            "model_checkpoint_live_state_match",
            "udlm_process_identity_verified",
            "live_model_match",
            "live_ema_match",
        },
        "checkpoint semantic audit",
    )
    _required_true(semantic.get("deserialized"), "checkpoint deserialization")
    _required_true(
        semantic.get("udlm_process_identity_verified"),
        "checkpoint UDLM process identity",
    )
    for field in (
        "raw_model",
        "ema",
        "optimizer",
        "non_sentinel_checkpoint_tensors",
    ):
        _validate_finiteness_record(
            semantic.get(field), label=f"checkpoint {field} finiteness"
        )
    _validate_framework_nonfinite_sentinels(
        semantic.get("framework_nonfinite_sentinels"),
        expected_steps=lock["optimizer_updates"],
        label="checkpoint framework non-finite sentinels",
    )
    optimizer_parameter_state_count = _validate_auxiliary_checkpoint_records(
        semantic,
        expected_steps=lock["optimizer_updates"],
        resolved_training_config=runtime.get("resolved_training_config"),
        label="checkpoint",
    )
    if semantic.get("global_step") != lock["checkpoint"]["global_step"]:
        raise GateValidationError("semantic checkpoint step disagrees with lock")
    serialized_ema = _mapping(semantic.get("ema"), "serialized EMA finiteness")
    _required_true(serialized_ema.get("all_finite"), "serialized EMA finiteness")
    live_ema_match = _mapping(
        semantic.get("live_ema_match"), "live EMA checkpoint match"
    )
    _exact_keys(
        live_ema_match,
        {"exact_tensor_values", "tensor_count"},
        "live EMA checkpoint match",
    )
    _required_true(
        live_ema_match.get("exact_tensor_values"), "live EMA checkpoint match"
    )
    _integer(
        live_ema_match.get("tensor_count"),
        "live EMA checkpoint tensor count",
        minimum=1,
    )
    live_model_match = _mapping(
        semantic.get("live_model_match"), "live model checkpoint match"
    )
    _exact_keys(
        live_model_match,
        {"exact_key_set", "exact_tensor_values", "tensor_count"},
        "live model checkpoint match",
    )
    _required_true(
        live_model_match.get("exact_key_set"),
        "live model checkpoint exact key set",
    )
    _required_true(
        live_model_match.get("exact_tensor_values"),
        "live model checkpoint match",
    )
    _integer(
        live_model_match.get("tensor_count"),
        "live model checkpoint tensor count",
        minimum=1,
    )
    summary_ema_metadata = _mapping(
        semantic.get("ema_metadata"), "training-summary EMA metadata"
    )
    try:
        summary_inference_weights = denovo_report.validate_inference_weights(
            {
                "source": "ema",
                "ema_applied": True,
                "ema": dict(summary_ema_metadata),
            },
            require_ema=True,
        )
    except ValueError as error:
        raise GateValidationError(
            f"training-summary EMA metadata is invalid: {error}"
        ) from error
    if summary_inference_weights != lock["inference_weights"]:
        raise GateValidationError("training-summary EMA metadata disagrees with lock")
    shadow_count = summary_inference_weights["ema"]["shadow_parameter_count"]
    if optimizer_parameter_state_count != shadow_count:
        raise GateValidationError(
            "checkpoint optimizer parameter-state count disagrees with EMA metadata"
        )
    if live_ema_match.get("tensor_count") != shadow_count:
        raise GateValidationError("live EMA tensor count disagrees with EMA metadata")
    if serialized_ema.get("floating_tensor_count") != shadow_count:
        raise GateValidationError("serialized EMA tensor count disagrees with metadata")
    startup = _mapping(summary.get("startup"), "training summary startup")
    _exact_keys(
        startup,
        {"mode", "verified_mdlm_warm_start_report"},
        "training summary startup",
    )
    if startup.get("mode") != lock["startup_mode"]:
        raise GateValidationError("training startup mode disagrees with candidate lock")
    warm_report = startup.get("verified_mdlm_warm_start_report")
    if lock["startup_mode"] == "warm_start":
        warm_report = _mapping(warm_report, "warm-start report")
        locked_parameter_counts = _mapping(
            lock.get("parameter_counts"), "locked trainable parameter counts"
        )
        expected_conditioning_variant = (
            "film_adaln"
            if "film_modulation_trainable" in locked_parameter_counts
            else "additive"
        )
        expected_warm_report_keys = {
            "source_path",
            "source_resolved_path",
            "source_sha256",
            "source_size_bytes",
            "expected_source_sha256",
            "byte_identity_verified_before_and_after_load",
            "weights",
            "parameter_tensors",
        }
        if expected_conditioning_variant == "film_adaln":
            expected_warm_report_keys.update(
                {"conditioning_variant", "conditioning_parameter_tensors"}
            )
        _exact_keys(
            warm_report,
            expected_warm_report_keys,
            "warm-start report",
        )
        if warm_report.get("source_sha256") != lock["initialization_checkpoint_sha256"]:
            raise GateValidationError("warm-start source digest disagrees with lock")
        if (
            warm_report.get("expected_source_sha256")
            != lock["initialization_checkpoint_sha256"]
        ):
            raise GateValidationError(
                "warm-start expected source digest disagrees with lock"
            )
        _required_true(
            warm_report.get("byte_identity_verified_before_and_after_load"),
            "warm-start byte identity",
        )
        if warm_report.get("weights") != "ema":
            raise GateValidationError("warm-start initialization must use MDLM EMA")
        for field in ("source_path", "source_resolved_path"):
            if not isinstance(warm_report.get(field), str) or not warm_report[field]:
                raise GateValidationError(f"warm-start {field} must be nonempty")
        _integer(
            warm_report.get("source_size_bytes"),
            "warm-start source size",
            minimum=1,
        )
        _integer(
            warm_report.get("parameter_tensors"),
            "warm-start parameter tensor count",
            minimum=1,
        )
        if expected_conditioning_variant == "film_adaln":
            if warm_report.get("conditioning_variant") != "film_adaln":
                raise GateValidationError(
                    "warm-start conditioning variant disagrees with candidate lock"
                )
            conditioning_tensor_count = _integer(
                warm_report.get("conditioning_parameter_tensors"),
                "warm-start conditioning parameter tensor count",
                minimum=1,
            )
            if conditioning_tensor_count != 28:
                raise GateValidationError(
                    "warm-start conditioning parameter tensor count disagrees with "
                    "candidate topology"
                )
    elif warm_report is not None:
        raise GateValidationError(
            "scratch summary unexpectedly contains warm-start evidence"
        )
    if (
        "conditioning_gradient_audit" not in summary
        or summary.get("conditioning_gradient_audit") is not None
    ):
        raise GateValidationError(
            "registered R/S/E candidate summary must have a null conditioning "
            "gradient audit"
        )
    if (
        "screen_initialization_state_audit" not in summary
        or summary.get("screen_initialization_state_audit") is not None
    ):
        raise GateValidationError(
            "registered R/S/E candidate summary must have a null optimization-"
            "screen initialization-state audit"
        )

    _validate_training_health(
        summary.get("training_health"), optimizer_updates=lock["optimizer_updates"]
    )
    tensor_finiteness = _mapping(
        summary.get("tensor_finiteness"), "summary tensor finiteness"
    )
    _exact_keys(tensor_finiteness, {"raw_model", "ema"}, "summary tensor finiteness")
    for field in ("raw_model", "ema"):
        _validate_finiteness_record(
            tensor_finiteness.get(field), label=f"summary {field} finiteness"
        )
        if tensor_finiteness.get(field) != semantic.get(field):
            raise GateValidationError(
                f"live and serialized {field} finiteness evidence disagree"
            )

    if (
        receipt.get("status") != "completed"
        or receipt.get("overall_status") != "completed"
    ):
        raise GateValidationError("training exit receipt is not completed")
    process_exit_status = _integer(
        receipt.get("process_exit_status"), "training exit receipt process status"
    )
    if process_exit_status != 0:
        raise GateValidationError("training exit receipt records nonzero status")
    _validate_successful_pipeline(
        receipt.get("pipeline"), label="training exit receipt pipeline"
    )
    expected = _mapping(receipt.get("expected_contract"), "receipt expected contract")
    _exact_keys(
        expected,
        {
            "training_summary_schema_version",
            "source_revision",
            "resolved_training_config_sha256",
            "training_argv_sha256",
            "launch_manifest_path",
            "launch_manifest_sha256",
            "selected_gpu_uuids",
            "training_job_lock_path",
            "training_job_lock_sha256",
            "max_steps",
            "world_size",
            "training_summary_path",
            "final_checkpoint_path",
            "initialization_checkpoint_sha256",
        },
        "receipt expected contract",
    )
    expected_values = {
        "training_summary_schema_version": TRAINING_SUMMARY_SCHEMA_VERSION,
        "source_revision": lock["source_revision"],
        "resolved_training_config_sha256": lock["resolved_training_config_sha256"],
        "training_argv_sha256": lock["training_argv_sha256"],
        "launch_manifest_path": str(manifest_path),
        "launch_manifest_sha256": lock["launch_manifest"]["sha256"],
        "selected_gpu_uuids": selected_gpu_uuids,
        "training_job_lock_path": _manifest["single_training_job_lock"]["path"],
        "training_job_lock_sha256": _manifest["single_training_job_lock"]["sha256"],
        "max_steps": lock["optimizer_updates"],
        "world_size": lock["world_size"],
        "training_summary_path": str(
            _expected_artifact_path(lock["summary"]["relative_path"])
        ),
        "final_checkpoint_path": str(
            _expected_artifact_path(lock["checkpoint"]["relative_path"])
        ),
        "initialization_checkpoint_sha256": lock["initialization_checkpoint_sha256"],
    }
    for key, value in expected_values.items():
        if expected.get(key) != value:
            raise GateValidationError(f"exit receipt {key} disagrees with lock")
    _validate_receipt_source(
        receipt.get("source_at_receipt"),
        expected_revision=lock["source_revision"],
        label="receipt source evidence",
    )
    receipt_manifest = _mapping(
        receipt.get("launch_manifest"), "receipt launch-manifest evidence"
    )
    _exact_keys(
        receipt_manifest,
        {
            "path",
            "present",
            "matches_expected_raw_sha256",
            "selected_gpu_uuids_match_expected",
            "matches_training_summary_snapshot",
            "matches_runtime_config_snapshot",
            "valid_and_launch_bound",
            "expected_selected_gpu_uuids",
            "observed_selected_gpu_uuids",
            "artifact",
            "validation_error",
        },
        "receipt launch-manifest evidence",
    )
    if receipt_manifest.get("path") != str(manifest_path):
        raise GateValidationError("receipt launch-manifest path disagrees with lock")
    for field in (
        "present",
        "matches_expected_raw_sha256",
        "selected_gpu_uuids_match_expected",
        "matches_training_summary_snapshot",
        "matches_runtime_config_snapshot",
        "valid_and_launch_bound",
    ):
        _required_true(receipt_manifest.get(field), f"receipt launch-manifest {field}")
    if receipt_manifest.get("validation_error") is not None:
        raise GateValidationError(
            "receipt launch-manifest validation recorded an error"
        )
    for field in ("expected_selected_gpu_uuids", "observed_selected_gpu_uuids"):
        selected = _selected_gpu_uuids(
            receipt_manifest.get(field),
            f"receipt launch-manifest {field}",
            expected_count=lock["world_size"],
        )
        if selected != selected_gpu_uuids:
            raise GateValidationError(
                f"receipt launch-manifest {field} disagrees with manifest"
            )
    receipt_manifest_artifact = _validate_snapshot_claim(
        receipt_manifest.get("artifact"),
        label="receipt launch-manifest artifact",
        expected_path=manifest_path,
        expected_sha256=lock["launch_manifest"]["sha256"],
        expected_size_bytes=manifest_size_bytes,
    )
    summary_manifest_artifact = dict(summary_manifest_claim)
    del summary_manifest_artifact["selected_gpu_uuids"]
    if receipt_manifest_artifact != summary_manifest_artifact:
        raise GateValidationError(
            "receipt and summary launch-manifest snapshots disagree"
        )
    receipt_lock = _mapping(
        receipt.get("training_job_lock"), "receipt training-job lock evidence"
    )
    _exact_keys(
        receipt_lock,
        {
            "path",
            "present",
            "expected_sha256",
            "matches_expected_raw_sha256",
            "matches_launch_manifest_binding",
            "valid_and_launch_bound_before_receipt_publication",
            "artifact",
            "record",
            "release_policy",
            "release_result_not_claimed_inside_pre_release_receipt",
            "validation_error",
        },
        "receipt training-job lock evidence",
    )
    manifest_lock = _mapping(
        _manifest["single_training_job_lock"], "manifest training-job lock binding"
    )
    expected_lock_path = Path(manifest_lock["path"])
    expected_lock_sha256 = manifest_lock["sha256"]
    if (
        receipt_lock.get("path") != str(expected_lock_path)
        or receipt_lock.get("expected_sha256") != expected_lock_sha256
        or receipt_lock.get("record") != manifest_lock["record"]
        or receipt_lock.get("release_policy")
        != "publish_receipt_then_unlink_only_same_stat_identity_and_sha256"
        or receipt_lock.get("validation_error") is not None
    ):
        raise GateValidationError(
            "receipt training-job lock evidence disagrees with launch manifest"
        )
    for field in (
        "present",
        "matches_expected_raw_sha256",
        "matches_launch_manifest_binding",
        "valid_and_launch_bound_before_receipt_publication",
        "release_result_not_claimed_inside_pre_release_receipt",
    ):
        _required_true(receipt_lock.get(field), f"receipt training-job lock {field}")
    lock_payload_size = len(
        (
            json.dumps(
                manifest_lock["record"], indent=2, sort_keys=True, allow_nan=False
            )
            + "\n"
        ).encode("utf-8")
    )
    _validate_snapshot_claim(
        receipt_lock.get("artifact"),
        label="receipt training-job lock artifact",
        expected_path=expected_lock_path,
        expected_sha256=expected_lock_sha256,
        expected_size_bytes=lock_payload_size,
    )
    receipt_summary = _mapping(
        receipt.get("training_summary"), "receipt summary evidence"
    )
    _exact_keys(
        receipt_summary,
        {
            "path",
            "present",
            "valid_and_launch_bound",
            "artifact",
            "validated_bindings",
            "validation_error",
        },
        "receipt summary evidence",
    )
    if receipt_summary.get("path") != str(
        _expected_artifact_path(lock["summary"]["relative_path"])
    ):
        raise GateValidationError("receipt training-summary path disagrees with lock")
    _required_true(receipt_summary.get("present"), "receipt summary presence")
    if receipt_summary.get("validation_error") is not None:
        raise GateValidationError("receipt summary evidence records an error")
    _required_true(
        receipt_summary.get("valid_and_launch_bound"),
        "launch-bound training summary",
    )
    summary_artifact = _mapping(
        receipt_summary.get("artifact"), "receipt training-summary artifact"
    )
    _validate_snapshot_claim(
        summary_artifact,
        label="receipt training-summary artifact",
        expected_path=_expected_artifact_path(lock["summary"]["relative_path"]),
        expected_sha256=lock["summary"]["sha256"],
        expected_size_bytes=summary_size_bytes,
    )
    receipt_checkpoint = _mapping(
        receipt.get("final_checkpoint"), "receipt checkpoint evidence"
    )
    _exact_keys(
        receipt_checkpoint,
        {"path", "present", "matches_training_summary_snapshot", "artifact"},
        "receipt checkpoint evidence",
    )
    if receipt_checkpoint.get("path") != str(
        _expected_artifact_path(lock["checkpoint"]["relative_path"])
    ):
        raise GateValidationError("receipt checkpoint path disagrees with lock")
    _required_true(receipt_checkpoint.get("present"), "receipt checkpoint presence")
    _required_true(
        receipt_checkpoint.get("matches_training_summary_snapshot"),
        "receipt checkpoint snapshot match",
    )
    receipt_checkpoint_artifact = _validate_snapshot_claim(
        receipt_checkpoint.get("artifact"),
        label="receipt checkpoint artifact",
        expected_path=_expected_artifact_path(lock["checkpoint"]["relative_path"]),
        expected_sha256=lock["checkpoint"]["sha256"],
        expected_size_bytes=lock["checkpoint"]["size_bytes"],
    )
    if receipt_checkpoint_artifact != summary_checkpoint_artifact:
        raise GateValidationError("receipt and summary checkpoint snapshots disagree")
    receipt_runtime = _mapping(
        receipt.get("runtime_config"), "receipt runtime evidence"
    )
    _exact_keys(
        receipt_runtime,
        {
            "path",
            "present",
            "matches_training_summary_snapshot",
            "semantic_validation_passed",
            "artifact",
        },
        "receipt runtime evidence",
    )
    if receipt_runtime.get("path") != str(
        _expected_artifact_path(lock["runtime"]["relative_path"])
    ):
        raise GateValidationError("receipt runtime path disagrees with lock")
    _required_true(receipt_runtime.get("present"), "receipt runtime presence")
    _required_true(
        receipt_runtime.get("matches_training_summary_snapshot"),
        "receipt runtime snapshot match",
    )
    _required_true(
        receipt_runtime.get("semantic_validation_passed"),
        "receipt runtime semantic validation",
    )
    runtime_artifact = _mapping(
        receipt_runtime.get("artifact"), "receipt runtime artifact"
    )
    receipt_runtime_artifact = _validate_snapshot_claim(
        runtime_artifact,
        label="receipt runtime-config artifact",
        expected_path=_expected_artifact_path(lock["runtime"]["relative_path"]),
        expected_sha256=lock["runtime"]["sha256"],
        expected_size_bytes=runtime_size_bytes,
    )
    if receipt_runtime_artifact != summary_runtime_artifact:
        raise GateValidationError(
            "receipt and summary runtime-config snapshots disagree"
        )
    if runtime.get("status") != "preflight_completed":
        raise GateValidationError(
            "runtime training config is not a completed preflight"
        )
    if runtime.get("source_revision") != lock["source_revision"]:
        raise GateValidationError("runtime source revision disagrees with lock")
    if (
        runtime.get("resolved_training_config_sha256")
        != lock["resolved_training_config_sha256"]
    ):
        raise GateValidationError("runtime resolved-config digest disagrees with lock")
    resolved_config = _mapping(
        runtime.get("resolved_training_config"), "runtime resolved training config"
    )
    if (
        canonical_json_sha256(resolved_config)
        != lock["resolved_training_config_sha256"]
    ):
        raise GateValidationError("runtime resolved training config content is unbound")
    if runtime.get("training_argv_sha256") != lock["training_argv_sha256"]:
        raise GateValidationError("runtime training argv digest disagrees with lock")
    training_config = _mapping(
        resolved_config.get("training"), "runtime training configuration"
    )
    udlm_config = _mapping(
        training_config.get("udlm"), "runtime UDLM training configuration"
    )
    conditioning_variant = udlm_config.get("conditioning_variant")
    if conditioning_variant not in {"additive", "film_adaln"}:
        raise GateValidationError(
            "runtime UDLM conditioning_variant must be additive or film_adaln"
        )
    if runtime.get("completion_contract") != expected_completion_contract:
        raise GateValidationError(
            "runtime and summary completion contracts disagree with lock"
        )
    runtime_manifest_claim = _validate_snapshot_claim(
        runtime.get("launch_manifest"),
        label="runtime launch-manifest evidence",
        expected_path=manifest_path,
        expected_sha256=lock["launch_manifest"]["sha256"],
        expected_size_bytes=manifest_size_bytes,
        include_selected_gpu_uuids=True,
        expected_selected_gpu_uuids=selected_gpu_uuids,
    )
    if runtime_manifest_claim != summary_manifest_claim:
        raise GateValidationError(
            "runtime and summary launch-manifest snapshots disagree"
        )
    training_argv = runtime.get("training_argv")
    if not isinstance(training_argv, list) or not all(
        isinstance(value, str) for value in training_argv
    ):
        raise GateValidationError("runtime training argv must be a string list")
    if canonical_json_sha256(training_argv) != lock["training_argv_sha256"]:
        raise GateValidationError("runtime training argv content is unbound")
    if training_argv != _manifest["training_argv"][2:]:
        raise GateValidationError(
            "runtime training argv disagrees with launch producer command"
        )
    if runtime.get("observed_training_argv") != training_argv:
        raise GateValidationError(
            "runtime observed training argv disagrees with its base argv"
        )
    _validate_python_environment(
        runtime.get("python_environment"),
        seed=lock["training_seed"],
        label="runtime Python environment",
    )

    accounting = _mapping(
        summary.get("training_accounting"), "summary training accounting"
    )
    _exact_keys(
        accounting,
        {
            "training_seed",
            "optimizer_updates",
            "world_size",
            "micro_batch_size_per_rank",
            "accumulate_grad_batches",
            "effective_global_examples_per_optimizer_step",
            "total_requested_example_exposures",
            "hosted_stream_rank_partition_policy",
            "trainable_parameter_counts",
        },
        "summary training accounting",
    )
    training_seed = _integer(
        accounting.get("training_seed"), "accounting training seed", minimum=0
    )
    optimizer_updates = _integer(
        accounting.get("optimizer_updates"),
        "accounting optimizer updates",
        minimum=1,
    )
    world_size = _integer(
        accounting.get("world_size"), "accounting world size", minimum=1
    )
    micro_batch = _integer(
        accounting.get("micro_batch_size_per_rank"),
        "accounting micro-batch size per rank",
        minimum=1,
    )
    accumulation = _integer(
        accounting.get("accumulate_grad_batches"),
        "accounting gradient accumulation",
        minimum=1,
    )
    global_examples = _integer(
        accounting.get("effective_global_examples_per_optimizer_step"),
        "accounting effective global examples",
        minimum=1,
    )
    requested_exposures = _integer(
        accounting.get("total_requested_example_exposures"),
        "accounting requested example exposures",
        minimum=1,
    )
    if global_examples != micro_batch * world_size * accumulation:
        raise GateValidationError(
            "summary training-accounting batch arithmetic disagrees"
        )
    if requested_exposures != global_examples * optimizer_updates:
        raise GateValidationError(
            "summary training-accounting exposure arithmetic disagrees"
        )
    if training_seed != lock["training_seed"]:
        raise GateValidationError("training-accounting seed disagrees with lock")
    if optimizer_updates != lock["optimizer_updates"]:
        raise GateValidationError("training-accounting updates disagree with lock")
    if world_size != lock["world_size"]:
        raise GateValidationError("training-accounting world size disagrees with lock")
    locked_exposure = lock["data_exposure"]
    if global_examples != locked_exposure["global_examples_per_optimizer_step"]:
        raise GateValidationError(
            "training-accounting global examples disagree with lock"
        )
    if requested_exposures != locked_exposure["total_requested_examples"]:
        raise GateValidationError(
            "training-accounting requested exposure disagrees with lock"
        )
    if (
        accounting.get("hosted_stream_rank_partition_policy")
        != locked_exposure["stream_partition_policy"]
    ):
        raise GateValidationError(
            "training-accounting stream policy disagrees with lock"
        )

    parameter_counts = _mapping(
        accounting.get("trainable_parameter_counts"),
        "summary trainable parameter counts",
    )
    expected_summary_parameter_keys = {
        "base_backbone",
        "time_conditioner",
        "total",
    }
    expected_locked_parameter_keys = {
        "base_model_trainable",
        "time_conditioner_trainable",
        "total_trainable",
    }
    if conditioning_variant == "film_adaln":
        expected_summary_parameter_keys.add("film_modulation")
        expected_locked_parameter_keys.add("film_modulation_trainable")
    _exact_keys(
        parameter_counts,
        expected_summary_parameter_keys,
        "summary trainable parameter counts",
    )
    base_count = _integer(
        parameter_counts.get("base_backbone"),
        "summary base-backbone trainable parameters",
        minimum=1,
    )
    conditioner_count = _integer(
        parameter_counts.get("time_conditioner"),
        "summary time-conditioner trainable parameters",
        minimum=1,
    )
    film_count = 0
    if conditioning_variant == "film_adaln":
        film_count = _integer(
            parameter_counts.get("film_modulation"),
            "summary FiLM-modulation trainable parameters",
            minimum=1,
        )
    total_count = _integer(
        parameter_counts.get("total"),
        "summary total trainable parameters",
        minimum=1,
    )
    if total_count != base_count + conditioner_count + film_count:
        raise GateValidationError("summary trainable parameter counts do not add up")
    locked_parameters = _mapping(
        lock.get("parameter_counts"), "locked trainable parameter counts"
    )
    _exact_keys(
        locked_parameters,
        expected_locked_parameter_keys,
        "locked trainable parameter counts",
    )
    locked_base_count = _integer(
        locked_parameters.get("base_model_trainable"),
        "locked base trainable parameters",
        minimum=1,
    )
    locked_conditioner_count = _integer(
        locked_parameters.get("time_conditioner_trainable"),
        "locked time-conditioner trainable parameters",
        minimum=1,
    )
    locked_film_count = 0
    if conditioning_variant == "film_adaln":
        locked_film_count = _integer(
            locked_parameters.get("film_modulation_trainable"),
            "locked FiLM-modulation trainable parameters",
            minimum=1,
        )
    locked_total_count = _integer(
        locked_parameters.get("total_trainable"),
        "locked total trainable parameters",
        minimum=1,
    )
    if locked_total_count != (
        locked_base_count + locked_conditioner_count + locked_film_count
    ):
        raise GateValidationError(
            "training-accounting parameter counts disagree with lock: locked counts "
            "do not add up"
        )
    if (
        base_count != locked_base_count
        or conditioner_count != locked_conditioner_count
        or film_count != locked_film_count
        or total_count != locked_total_count
    ):
        raise GateValidationError(
            "training-accounting parameter counts disagree with lock"
        )

    validated_bindings = _mapping(
        receipt_summary.get("validated_bindings"),
        "receipt validated training-summary bindings",
    )
    _exact_keys(
        validated_bindings,
        {
            "schema_version",
            "source_revision",
            "resolved_training_config_sha256",
            "training_argv_sha256",
            "launch_manifest_path",
            "launch_manifest_sha256",
            "selected_gpu_uuids",
            "observed_global_step",
            "observed_world_size",
            "training_accounting",
            "ema_metadata",
            "final_checkpoint_path",
            "final_checkpoint_sha256",
            "startup_mode",
            "conditioning_gradient_audit",
            "screen_initialization_state_audit",
        },
        "receipt validated training-summary bindings",
    )
    expected_validated_bindings = {
        "schema_version": TRAINING_SUMMARY_SCHEMA_VERSION,
        "source_revision": lock["source_revision"],
        "resolved_training_config_sha256": lock["resolved_training_config_sha256"],
        "training_argv_sha256": lock["training_argv_sha256"],
        "launch_manifest_path": str(manifest_path),
        "launch_manifest_sha256": lock["launch_manifest"]["sha256"],
        "selected_gpu_uuids": selected_gpu_uuids,
        "observed_global_step": lock["optimizer_updates"],
        "observed_world_size": lock["world_size"],
        "final_checkpoint_path": str(
            _expected_artifact_path(lock["checkpoint"]["relative_path"])
        ),
        "final_checkpoint_sha256": lock["checkpoint"]["sha256"],
        "startup_mode": lock["startup_mode"],
        "conditioning_gradient_audit": None,
        "screen_initialization_state_audit": None,
    }
    for key, expected_value in expected_validated_bindings.items():
        if validated_bindings.get(key) != expected_value:
            raise GateValidationError(
                f"receipt validated {key} disagrees with candidate lock"
            )
    receipt_accounting = _mapping(
        validated_bindings.get("training_accounting"),
        "receipt validated training accounting",
    )
    if dict(receipt_accounting) != dict(accounting):
        raise GateValidationError(
            "receipt-validated training accounting disagrees with summary"
        )
    receipt_ema_metadata = _mapping(
        validated_bindings.get("ema_metadata"),
        "receipt validated EMA metadata",
    )
    if dict(receipt_ema_metadata) != dict(summary_ema_metadata):
        raise GateValidationError(
            "receipt-validated EMA metadata disagrees with training summary"
        )

    completion_requirements = _mapping(
        receipt.get("completion_requirements"), "receipt completion requirements"
    )
    _exact_keys(
        completion_requirements,
        {
            "training_exit_zero",
            "tee_exit_zero",
            "training_summary_valid_and_launch_bound",
            "launch_manifest_matches_summary_runtime_and_launch",
            "predecessor_receipt_binding_unchanged_and_valid",
            "training_job_lock_valid_before_receipt_publication",
            "runtime_config_matches_summary_and_launch",
            "final_checkpoint_matches_training_summary",
            "clean_pushed_source_still_matches_launch",
            "all_must_hold",
        },
        "receipt completion requirements",
    )
    for field, value in completion_requirements.items():
        _required_true(value, f"receipt completion requirement {field}")

    if resolved_config.get("data") != "safe":
        raise GateValidationError("runtime accounting must use the hosted SAFE stream")
    if resolved_config.get("seed") != training_seed:
        raise GateValidationError("runtime training seed disagrees with accounting")
    trainer_config = _mapping(
        resolved_config.get("trainer"), "runtime trainer configuration"
    )
    loader_config = _mapping(
        resolved_config.get("loader"), "runtime loader configuration"
    )
    if trainer_config.get("max_steps") != optimizer_updates:
        raise GateValidationError("runtime max steps disagree with accounting")
    if trainer_config.get("num_nodes") != 1:
        raise GateValidationError("runtime hosted-stream accounting requires one node")
    if trainer_config.get("devices") != world_size:
        raise GateValidationError("runtime device count disagrees with accounting")
    if trainer_config.get("accumulate_grad_batches") != accumulation:
        raise GateValidationError("runtime accumulation disagrees with accounting")
    if loader_config.get("batch_size") != micro_batch:
        raise GateValidationError("runtime micro-batch size disagrees with accounting")
    if loader_config.get("global_batch_size") != global_examples:
        raise GateValidationError("runtime global batch size disagrees with accounting")
    _close(
        training_config.get("ema"),
        summary_inference_weights["ema"]["decay"],
        "runtime EMA decay",
    )
    return {
        "training_summary_sha256": lock["summary"]["sha256"],
        "exit_receipt_sha256": lock["receipt"]["sha256"],
        "runtime_config_sha256": lock["runtime"]["sha256"],
        "launch_manifest_sha256": lock["launch_manifest"]["sha256"],
        "selected_gpu_uuids": selected_gpu_uuids,
        "resolved_training_config_sha256": lock["resolved_training_config_sha256"],
        "training_argv_sha256": lock["training_argv_sha256"],
        "checkpoint_sha256": lock["checkpoint"]["sha256"],
        "checkpoint_global_step": lock["checkpoint"]["global_step"],
        "ema_finite_and_checkpoint_bound": True,
        "successful_exit_receipt": True,
        "training_accounting": dict(accounting),
        "ema_metadata": dict(summary_ema_metadata),
        "predecessor_receipt_chain": predecessor_chain,
        "training_variant": _manifest["training_variant"],
        "matched_panel_spec_sha256": _manifest["matched_panel_spec_sha256"],
        "launch_manifest_created_at_utc": _manifest["created_at"],
        "exit_receipt_recorded_at_utc": receipt["recorded_at_utc"],
    }


def validate_completed_matched_panel(
    lock: Mapping[str, Any],
    selected_training_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Require an immutable terminal-E receipt proving the full R→S→E panel."""

    terminal_ref = _mapping(
        lock.get("terminal_e_receipt"), "terminal E exit-receipt reference"
    )
    terminal_receipt = _load_referenced_json(
        terminal_ref, label="terminal E exit receipt"
    )
    _exact_keys(
        terminal_receipt,
        _PILOT_EXIT_STATUS_KEYS,
        "terminal E exit receipt",
    )
    relative_path = terminal_ref.get("relative_path")
    if not isinstance(relative_path, Path):
        raise GateValidationError("normalized terminal E receipt path is unavailable")
    parts = relative_path.parts
    if (
        len(parts) != 4
        or parts[:2] != ("output", "udlm")
        or RUN_NAME_PATTERN.fullmatch(parts[2]) is None
        or parts[3] != "pilot_exit_status.json"
    ):
        raise GateValidationError(
            "terminal E exit receipt must be output/udlm/<run>/pilot_exit_status.json"
        )
    run_name = parts[2]
    receipt_snapshot = _repository_artifact_snapshot(
        relative_path, label="terminal E exit receipt"
    )
    if receipt_snapshot["sha256"] != terminal_ref.get("sha256"):
        raise GateValidationError(
            "terminal E exit receipt digest disagrees with candidate lock"
        )

    receipt_manifest_evidence = _mapping(
        terminal_receipt.get("launch_manifest"),
        "terminal E receipt launch-manifest evidence",
    )
    receipt_summary_evidence = _mapping(
        terminal_receipt.get("training_summary"),
        "terminal E receipt training-summary evidence",
    )
    terminal_manifest, manifest_snapshot = _load_bound_predecessor_json(
        receipt_manifest_evidence.get("artifact"),
        label="terminal E launch manifest",
        basename="launch_manifest.json",
        run_name=run_name,
    )
    terminal_summary, summary_snapshot = _load_bound_predecessor_json(
        receipt_summary_evidence.get("artifact"),
        label="terminal E training summary",
        basename="training_summary.json",
        run_name=run_name,
    )
    terminal_lock = _derive_predecessor_training_lock(
        run_name=run_name,
        manifest=terminal_manifest,
        manifest_snapshot=manifest_snapshot,
        summary=terminal_summary,
        summary_snapshot=summary_snapshot,
        receipt_snapshot=receipt_snapshot,
    )
    if terminal_lock["receipt"] != dict(terminal_ref):
        raise GateValidationError(
            "terminal E receipt reference disagrees with its producer evidence"
        )
    terminal_evidence = validate_training_evidence(terminal_lock)
    expected_order = list(_MATCHED_PANEL_VARIANT_ORDER)
    terminal_chain = _mapping(
        terminal_evidence.get("predecessor_receipt_chain"),
        "terminal E predecessor chain",
    )
    raw_receipt_members = terminal_chain.get("receipt_members")
    if not isinstance(raw_receipt_members, list):
        raise GateValidationError("terminal E chain receipt members must be a list")
    receipt_members: list[dict[str, str]] = []
    for index, raw_member in enumerate(raw_receipt_members):
        member = _mapping(raw_member, f"terminal E chain receipt member {index}")
        _exact_keys(
            member,
            {"variant", "relative_path", "sha256"},
            f"terminal E chain receipt member {index}",
        )
        member_variant = member.get("variant")
        member_relative_path = member.get("relative_path")
        member_sha256 = member.get("sha256")
        if (
            not isinstance(member_variant, str)
            or not isinstance(member_relative_path, str)
            or not isinstance(member_sha256, str)
            or HEX_SHA256.fullmatch(member_sha256) is None
        ):
            raise GateValidationError("terminal E chain receipt member is invalid")
        receipt_members.append(dict(member))
    if (
        terminal_evidence.get("training_variant") != expected_order[-1]
        or terminal_manifest.get("matched_panel_variant_position")
        != len(expected_order) - 1
        or terminal_chain.get("chain_depth") != len(expected_order)
        or terminal_chain.get("variant_order_prefix") != expected_order
        or [member["variant"] for member in receipt_members] != expected_order
        or len({member["relative_path"] for member in receipt_members})
        != len(expected_order)
        or terminal_chain.get("machine_enforced") is not True
    ):
        raise GateValidationError(
            "terminal E receipt does not prove the complete registered R/S/E panel"
        )
    selected_variant = selected_training_evidence.get("training_variant")
    selected_receipt_ref = _mapping(
        lock.get("receipt"), "selected candidate receipt reference"
    )
    if selected_variant not in _MATCHED_PANEL_VARIANT_ORDER:
        raise GateValidationError("selected candidate training variant is invalid")
    expected_selected_member = {
        "variant": selected_variant,
        "relative_path": selected_receipt_ref["relative_path"].as_posix(),
        "sha256": selected_receipt_ref["sha256"],
    }
    selected_position = _MATCHED_PANEL_VARIANT_ORDER.index(selected_variant)
    if receipt_members[selected_position] != expected_selected_member:
        raise GateValidationError(
            "selected candidate receipt is not a member of the terminal R/S/E chain"
        )
    expected_terminal_member = {
        "variant": expected_order[-1],
        "relative_path": terminal_ref["relative_path"].as_posix(),
        "sha256": terminal_ref["sha256"],
    }
    if receipt_members[-1] != expected_terminal_member:
        raise GateValidationError(
            "terminal E receipt reference is not the chain's terminal member"
        )
    selected_panel_sha256 = selected_training_evidence.get("matched_panel_spec_sha256")
    if (
        not isinstance(selected_panel_sha256, str)
        or terminal_evidence.get("matched_panel_spec_sha256") != selected_panel_sha256
    ):
        raise GateValidationError(
            "terminal E and selected candidate do not share one matched panel"
        )
    terminal_recorded_at = _utc_timestamp(
        terminal_receipt.get("recorded_at_utc"),
        "terminal E exit receipt timestamp",
    )
    locked_at = lock.get("locked_at")
    selected_recorded_at = _utc_timestamp(
        selected_training_evidence.get("exit_receipt_recorded_at_utc"),
        "selected candidate exit receipt timestamp",
    )
    if (
        not isinstance(locked_at, datetime)
        or not selected_recorded_at < locked_at
        or not terminal_recorded_at < locked_at
    ):
        raise GateValidationError(
            "selected candidate and terminal E receipts must strictly predate "
            "the candidate lock"
        )
    return {
        "terminal_e_run_name": run_name,
        "terminal_e_exit_receipt_sha256": terminal_ref["sha256"],
        "terminal_e_exit_receipt_recorded_at_utc": terminal_receipt["recorded_at_utc"],
        "selected_exit_receipt_recorded_at_utc": selected_training_evidence[
            "exit_receipt_recorded_at_utc"
        ],
        "candidate_locked_at_utc": locked_at.isoformat(),
        "matched_panel_spec_sha256": selected_panel_sha256,
        "chain_depth": len(expected_order),
        "variant_order": expected_order,
        "receipt_members": receipt_members,
        "selected_receipt_membership_proved": True,
        "complete_registered_panel_proved": True,
    }


def _validate_rescore_worker_proof(
    value: object,
    *,
    label: str,
    expected_seed: int,
    expected_sample_count: int,
    expected_summary_path: Path,
    expected_raw_path: Path,
    expected_summary_sha256: str,
    expected_raw_sha256: str,
) -> Mapping[str, Any]:
    """Validate the complete fresh-worker result before trusting any metric."""

    result = _mapping(value, label)
    _exact_keys(
        result,
        {
            "status",
            "seed",
            "summary_sha256",
            "raw_samples_sha256",
            "row_comparison",
            "metrics",
            "failure_counts",
            "identity",
            "independent_recomputation",
            "stable_inputs",
            "worker_environment",
        },
        label,
    )
    if (
        result.get("status") != "exact_match"
        or result.get("seed") != expected_seed
        or result.get("summary_sha256") != expected_summary_sha256
        or result.get("raw_samples_sha256") != expected_raw_sha256
    ):
        raise GateValidationError(f"{label} completion/artifact identity differs")

    row_comparison = _mapping(result.get("row_comparison"), f"{label} rows")
    _exact_keys(
        row_comparison,
        {
            "all_match",
            "row_count",
            "field_count",
            "cell_count",
            "numeric_absolute_tolerance",
            "field_results",
        },
        f"{label} rows",
    )
    if (
        row_comparison.get("all_match") is not True
        or row_comparison.get("row_count") != expected_sample_count
        or row_comparison.get("field_count") != EXPECTED_BASELINE_RAW_SAMPLE_FIELD_COUNT
        or row_comparison.get("cell_count")
        != expected_sample_count * EXPECTED_BASELINE_RAW_SAMPLE_FIELD_COUNT
        or row_comparison.get("numeric_absolute_tolerance") != 1e-12
    ):
        raise GateValidationError(f"{label} row-comparison totals are incomplete")
    field_results = row_comparison.get("field_results")
    if not isinstance(field_results, list) or len(field_results) != len(
        denovo_report.RAW_SAMPLE_FIELDS
    ):
        raise GateValidationError(f"{label} field comparisons are incomplete")
    for expected_field, raw_field_result in zip(
        denovo_report.RAW_SAMPLE_FIELDS, field_results, strict=True
    ):
        field_result = _mapping(
            raw_field_result, f"{label} field comparison {expected_field}"
        )
        _exact_keys(
            field_result,
            {
                "field",
                "comparison",
                "compared_rows",
                "mismatch_count",
                "max_absolute_difference",
            },
            f"{label} field comparison {expected_field}",
        )
        maximum_difference = field_result.get("max_absolute_difference")
        if maximum_difference is not None:
            maximum_difference = _finite(
                maximum_difference,
                f"{label} field comparison {expected_field} maximum difference",
            )
        expected_comparison = (
            "finite_numeric_absolute_tolerance_1e-12_or_exact_null"
            if expected_field in NUMERIC_RAW_SAMPLE_FIELDS
            else "exact_value_and_type"
        )
        if (
            field_result.get("field") != expected_field
            or field_result.get("comparison") != expected_comparison
            or field_result.get("compared_rows") != expected_sample_count
            or field_result.get("mismatch_count") != 0
            or (
                expected_field not in NUMERIC_RAW_SAMPLE_FIELDS
                and maximum_difference is not None
            )
            or (maximum_difference is not None and maximum_difference > 1e-12)
        ):
            raise GateValidationError(
                f"{label} field comparison {expected_field} is incomplete"
            )

    recomputation = _mapping(
        result.get("independent_recomputation"), f"{label} recomputation"
    )
    expected_recomputation = {
        "raw_input_field": "raw_model_text",
        "decoder": "benchmark.decode_records",
        "metric_evaluator": "benchmark.evaluate_records",
        "qed_recomputed": True,
        "sa_recomputed": True,
        "released_diversity_recomputed": True,
        "strict_branch_recomputed": True,
        "all_21_raw_fields_compared": True,
        "numeric_absolute_tolerance": 1e-12,
    }
    _exact_keys(recomputation, set(expected_recomputation), f"{label} recomputation")
    if dict(recomputation) != expected_recomputation:
        raise GateValidationError(f"{label} independent recomputation is incomplete")

    stable_inputs = _mapping(result.get("stable_inputs"), f"{label} stable inputs")
    _exact_keys(
        stable_inputs,
        {
            "summary_json",
            "raw_samples_csv",
            "recorded_path_bindings",
            "revalidated_unchanged_after_rescore",
        },
        f"{label} stable inputs",
    )
    if stable_inputs.get("revalidated_unchanged_after_rescore") is not True:
        raise GateValidationError(f"{label} inputs were not revalidated after rescore")
    expected_artifacts = {
        "summary_json": (expected_summary_path, expected_summary_sha256),
        "raw_samples_csv": (expected_raw_path, expected_raw_sha256),
    }
    for name, (expected_path, expected_sha256) in expected_artifacts.items():
        artifact = _mapping(stable_inputs.get(name), f"{label} stable {name}")
        _exact_keys(
            artifact,
            {"path", "sha256", "size_bytes", "read_policy"},
            f"{label} stable {name}",
        )
        if (
            artifact.get("path") != str(expected_path)
            or artifact.get("sha256") != expected_sha256
            or artifact.get("read_policy")
            != "regular_file_no_symlink_stable_descriptor_bytes_retained_in_memory"
        ):
            raise GateValidationError(f"{label} stable {name} identity differs")
        _integer(artifact.get("size_bytes"), f"{label} stable {name} size", minimum=1)
    bindings = _mapping(
        stable_inputs.get("recorded_path_bindings"),
        f"{label} recorded path bindings",
    )
    _exact_keys(bindings, {"summary_json", "raw_samples_csv"}, f"{label} path bindings")
    for name, (expected_path, _expected_sha256) in expected_artifacts.items():
        binding = _mapping(bindings.get(name), f"{label} path binding {name}")
        expected_binding = {
            "recorded_path": str(expected_path),
            "supplied_resolved_path": str(expected_path),
            "exact_path_match": True,
        }
        _exact_keys(binding, set(expected_binding), f"{label} path binding {name}")
        if dict(binding) != expected_binding:
            raise GateValidationError(f"{label} path binding {name} differs")

    environment = _mapping(
        result.get("worker_environment"), f"{label} worker environment"
    )
    _exact_keys(
        environment,
        {
            "python_hash_seed",
            "device",
            "cuda_visible_devices",
            "nvidia_visible_devices",
            "offline_environment",
            "python_network_guard_during_computation",
            "executable",
            "python",
            "platform",
            "pid",
        },
        f"{label} worker environment",
    )
    expected_environment = {
        "python_hash_seed": str(expected_seed),
        "device": "cpu",
        "cuda_visible_devices": "",
        "nvidia_visible_devices": "",
    }
    if any(
        environment.get(key) != expected
        for key, expected in expected_environment.items()
    ):
        raise GateValidationError(f"{label} worker CPU/seed environment differs")
    expected_offline = {
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "WANDB_MODE": "offline",
        "WANDB_DISABLED": "true",
    }
    if environment.get("offline_environment") != expected_offline:
        raise GateValidationError(f"{label} worker offline environment differs")
    network_guard = _mapping(
        environment.get("python_network_guard_during_computation"),
        f"{label} worker network guard",
    )
    _exact_keys(
        network_guard,
        {"guarded_apis", "scope_limitation"},
        f"{label} worker network guard",
    )
    if network_guard.get("guarded_apis") != [
        "socket.create_connection",
        "socket.getaddrinfo",
        "socket.socket.connect",
        "socket.socket.connect_ex",
    ] or not isinstance(network_guard.get("scope_limitation"), str):
        raise GateValidationError(f"{label} worker network guard differs")
    for field in ("executable", "python", "platform"):
        if not isinstance(environment.get(field), str) or not environment[field]:
            raise GateValidationError(f"{label} worker {field} is missing")
    _integer(environment.get("pid"), f"{label} worker PID", minimum=1)

    identity = _mapping(result.get("identity"), f"{label} identity")
    _exact_keys(
        identity,
        {
            "seed",
            "sample_count",
            "started_at_utc",
            "completed_at_utc",
            "checkpoint",
            "config",
            "generation",
            "source",
            "artifacts",
        },
        f"{label} identity",
    )
    if (
        identity.get("seed") != expected_seed
        or identity.get("sample_count") != expected_sample_count
    ):
        raise GateValidationError(f"{label} identity seed/sample count differs")
    started_at = _timestamp(identity.get("started_at_utc"), f"{label} start")
    completed_at = _timestamp(identity.get("completed_at_utc"), f"{label} completion")
    if completed_at < started_at:
        raise GateValidationError(f"{label} completion predates its start")
    for field in ("checkpoint", "config", "generation", "source", "artifacts"):
        _mapping(identity.get(field), f"{label} identity {field}")
    _mapping(result.get("metrics"), f"{label} metrics")
    _mapping(result.get("failure_counts"), f"{label} failure counts")
    return result


def _require_completed_pilot_references_at_revision(
    evidence: Mapping[str, Any], benchmark_revision: str
) -> None:
    """Require every completed-envelope payload to be a Git blob at evaluation."""

    receipt_ref = _artifact_reference(
        evidence.get("training_exit_receipt"),
        label="committed pilot training exit receipt",
        suffix=".json",
        require_schema=True,
    )
    benchmark_artifacts = _mapping(
        evidence.get("benchmark_artifacts"), "committed pilot benchmark artifacts"
    )
    _exact_keys(
        benchmark_artifacts,
        {"summary_json", "raw_samples_csv"},
        "committed pilot benchmark artifacts",
    )
    summary_ref = _artifact_reference(
        benchmark_artifacts.get("summary_json"),
        label="committed pilot summary",
        suffix=".json",
        require_schema=True,
    )
    raw_ref = _artifact_reference(
        benchmark_artifacts.get("raw_samples_csv"),
        label="committed pilot raw samples",
        suffix=".csv",
        require_schema=False,
    )
    for label, reference in (
        ("pilot training exit receipt", receipt_ref),
        ("pilot benchmark summary", summary_ref),
        ("pilot raw samples", raw_ref),
    ):
        blob = _git_blob(benchmark_revision, reference["relative_path"])
        if _sha256_bytes(blob) != reference["sha256"]:
            raise GateValidationError(
                f"benchmark revision {label} Git blob digest differs"
            )


def _validate_completed_pilot_evidence_live(
    evidence: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate and independently re-score one live schema-7 pilot run.

    The committed evidence envelope supplies only identities.  This adapter
    validates the producer-shaped files, checks the successful training receipt,
    and then invokes a fresh seed-at-interpreter-start CPU worker so PyTDC's
    diversity reduction is deterministic for both registered pilot seeds.
    """

    from scripts.udlm import rescore_denovo_run as denovo_rescore
    from scripts.udlm import write_pilot_evidence as pilot_evidence_writer

    attempt_id = evidence["attempt_id"]
    candidate_id = evidence["candidate_id"]
    pilot_seed = evidence["pilot_seed"]
    benchmark_artifacts = _mapping(
        evidence.get("benchmark_artifacts"), "pilot benchmark artifacts"
    )
    summary_ref = _artifact_reference(
        benchmark_artifacts.get("summary_json"),
        label="pilot summary",
        suffix=".json",
        require_schema=True,
    )
    raw_ref = _artifact_reference(
        benchmark_artifacts.get("raw_samples_csv"),
        label="pilot raw samples",
        suffix=".csv",
        require_schema=False,
    )
    summary_path = _expected_artifact_path(summary_ref["relative_path"])
    raw_path = _expected_artifact_path(raw_ref["relative_path"])
    summary_payload = _repository_artifact_bytes(
        summary_ref["relative_path"], label="pilot benchmark summary"
    )
    if _sha256_bytes(summary_payload) != summary_ref["sha256"]:
        raise GateValidationError("pilot benchmark summary digest differs")
    raw_payload = _repository_artifact_bytes(
        raw_ref["relative_path"], label="pilot raw samples"
    )
    if _sha256_bytes(raw_payload) != raw_ref["sha256"]:
        raise GateValidationError("pilot raw-sample digest differs")
    summary_document = _mapping(
        strict_json_loads(summary_payload, label="pilot benchmark summary"),
        "pilot benchmark summary",
    )
    requested_samples = _integer(
        summary_document.get("num_samples"),
        "pilot benchmark requested samples",
        minimum=1,
    )
    expected_tier = (
        "final" if requested_samples == EXPECTED_SAMPLES_PER_SEED else "pilot"
    )
    try:
        structural = denovo_report.validate_run_evidence(
            summary_path.parent,
            pilot_seed,
            expected_samples=requested_samples,
            expected_tier=expected_tier,
            final_protocol_eligible=(expected_tier == "final"),
        )
    except (OSError, ValueError) as error:
        raise GateValidationError(
            f"pilot schema-7 structural validation failed: {error}"
        ) from error
    if (
        Path(structural["summary_path"]) != summary_path
        or Path(structural["raw_samples_path"]) != raw_path
        or structural["summary_sha256"] != summary_ref["sha256"]
        or structural["raw_samples_sha256"] != raw_ref["sha256"]
    ):
        raise GateValidationError(
            "pilot structural validator returned different artifact identities"
        )

    receipt_ref = _artifact_reference(
        evidence.get("training_exit_receipt"),
        label="pilot training exit receipt",
        suffix=".json",
        require_schema=True,
    )
    receipt_payload = _repository_artifact_bytes(
        receipt_ref["relative_path"], label="pilot training exit receipt"
    )
    if _sha256_bytes(receipt_payload) != receipt_ref["sha256"]:
        raise GateValidationError("pilot training exit receipt digest differs")
    receipt = _mapping(
        strict_json_loads(receipt_payload, label="pilot training exit receipt"),
        "pilot training exit receipt",
    )
    structural_checkpoint = structural["checkpoint"]
    try:
        receipt_recorded_at, training_inputs = (
            pilot_evidence_writer.validate_successful_training_receipt(
                receipt,
                receipt_path=_expected_artifact_path(receipt_ref["relative_path"]),
                structural=structural,
            )
        )
    except (OSError, ValueError) as error:
        raise GateValidationError(
            f"pilot training exit receipt validation failed: {error}"
        ) from error

    implementation_inputs = structural["implementation_inputs"]
    metric_inputs = structural["metric_inputs"]
    config = structural["config"]
    source = structural["git"]
    try:
        rescored = denovo_rescore.invoke_rescore_worker(
            summary_path=summary_path,
            raw_samples_path=raw_path,
            allowed_root=REPOSITORY_ROOT,
            expected_summary_sha256=summary_ref["sha256"],
            expected_raw_samples_sha256=raw_ref["sha256"],
            expected_seed=pilot_seed,
            expected_sample_count=requested_samples,
            expected_checkpoint_sha256=structural_checkpoint["sha256"],
            expected_config_sha256=config["sha256"],
            expected_source_revision=source["commit"],
            expected_runner_sha256=structural["runner_sha256"],
            expected_sampler_source_sha256=implementation_inputs["sampler_source"][
                "sha256"
            ],
            expected_ema_source_sha256=implementation_inputs["ema_source"]["sha256"],
            expected_implementation_inputs_sha256=canonical_json_sha256(
                implementation_inputs
            ),
            expected_metric_inputs_sha256=canonical_json_sha256(metric_inputs),
        )
    except (OSError, ValueError) as error:
        raise GateValidationError(
            f"independent pilot rescore failed: {error}"
        ) from error
    rescored = _validate_rescore_worker_proof(
        rescored,
        label="pilot independent rescore",
        expected_seed=pilot_seed,
        expected_sample_count=requested_samples,
        expected_summary_path=summary_path,
        expected_raw_path=raw_path,
        expected_summary_sha256=summary_ref["sha256"],
        expected_raw_sha256=raw_ref["sha256"],
    )
    identity = _mapping(rescored.get("identity"), "pilot rescore identity")
    generation = _mapping(identity.get("generation"), "pilot rescore generation")
    identity_config = _mapping(identity.get("config"), "pilot rescore config")
    identity_source = _mapping(identity.get("source"), "pilot rescore source")
    metrics = _mapping(rescored.get("metrics"), "pilot rescore metrics")
    released = _mapping(
        metrics.get(REGISTERED_SELECTION_METRIC_BRANCH),
        "pilot rescore released metrics",
    )
    if (
        identity_source.get("revision") != source["commit"]
        or identity_source.get("runner_sha256") != structural["runner_sha256"]
        or identity_source.get("sampler_source_sha256")
        != implementation_inputs["sampler_source"]["sha256"]
        or identity_source.get("ema_source_sha256")
        != implementation_inputs["ema_source"]["sha256"]
        or identity_source.get("implementation_inputs_sha256")
        != canonical_json_sha256(implementation_inputs)
        or identity_source.get("metric_inputs_sha256")
        != canonical_json_sha256(metric_inputs)
    ):
        raise GateValidationError("pilot independent rescore source identity differs")
    try:
        pilot_evidence_writer.revalidate_inputs(training_inputs)
    except (OSError, ValueError) as error:
        raise GateValidationError(
            f"pilot training evidence changed during independent rescore: {error}"
        ) from error

    tracking = _mapping(config.get("git_tracking"), "pilot config Git tracking")
    return {
        "attempt_id": attempt_id,
        "candidate_id": candidate_id,
        "pilot_seed": pilot_seed,
        "requested_samples": requested_samples,
        "nfe": generation["nfe"],
        "metric_branch": REGISTERED_SELECTION_METRIC_BRANCH,
        "checkpoint": {
            key: structural_checkpoint[key]
            for key in ("sha256", "size_bytes", "global_step")
        },
        "evaluation_config": {
            "relative_path": tracking["relative_path"],
            "sha256": config["sha256"],
        },
        "sampling": {
            "config": identity_config["sampling"],
            "sha256": identity_config["sampling_sha256"],
        },
        "inference_weights": generation["inference_weights"],
        "runner_sha256": identity_source["runner_sha256"],
        "sampler_source_sha256": identity_source["sampler_source_sha256"],
        "implementation_inputs_sha256": identity_source["implementation_inputs_sha256"],
        "metric_inputs_sha256": identity_source["metric_inputs_sha256"],
        "benchmark_revision": identity_source["revision"],
        "started_at_utc": identity["started_at_utc"],
        "completed_at_utc": identity["completed_at_utc"],
        "training_exit_receipt": {
            "relative_path": receipt_ref["relative_path"].as_posix(),
            "sha256": receipt_ref["sha256"],
            "schema_version": receipt_ref["schema_version"],
            "recorded_at_utc": receipt_recorded_at.isoformat(),
        },
        "summary_json_sha256": summary_ref["sha256"],
        "raw_samples_csv_sha256": raw_ref["sha256"],
        "quality": released["quality"],
        "diversity": released["diversity"],
        "independent_rescore": {
            "all_21_fields_match": True,
            "both_metric_branches_match": True,
            "failure_counts_match": True,
            "raw_model_text_redecoded": True,
        },
    }


_PILOT_RUN_VALIDATION_KEYS = {
    "attempt_id",
    "candidate_id",
    "pilot_seed",
    "requested_samples",
    "nfe",
    "metric_branch",
    "checkpoint",
    "evaluation_config",
    "sampling",
    "inference_weights",
    "runner_sha256",
    "sampler_source_sha256",
    "implementation_inputs_sha256",
    "metric_inputs_sha256",
    "benchmark_revision",
    "started_at_utc",
    "completed_at_utc",
    "training_exit_receipt",
    "summary_json_sha256",
    "raw_samples_csv_sha256",
    "quality",
    "diversity",
    "independent_rescore",
}


def _pilot_output_artifact_reference(
    value: object,
    *,
    label: str,
    suffix: str,
    attempt_id: str,
    pilot_seed: int,
    require_schema: bool,
) -> dict[str, Any]:
    reference = _artifact_reference(
        value, label=label, suffix=suffix, require_schema=require_schema
    )
    relative_path = reference["relative_path"]
    expected_seed_directory = f"seed_{pilot_seed}"
    if (
        len(relative_path.parts) < 4
        or relative_path.parts[0] != "output"
        or relative_path.parent.name != expected_seed_directory
        or relative_path.parent.parent.name != attempt_id
        or relative_path.name
        != ("summary.json" if suffix == ".json" else "raw_samples.csv")
    ):
        raise GateValidationError(
            f"{label} must live below output/.../{attempt_id}/"
            f"{expected_seed_directory}"
        )
    return reference


def _absolute_repository_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise GateValidationError(f"{label} must be an absolute repository path")
    root = Path(os.path.abspath(REPOSITORY_ROOT))
    path = Path(value)
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise GateValidationError(f"{label} must be a normalized absolute path")
    if path == root or root not in path.parents:
        raise GateValidationError(f"{label} must remain inside the repository")
    return path


def _absolute_project_checkpoint_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise GateValidationError(f"{label} must be an absolute project path")
    repository_root = Path(os.path.abspath(REPOSITORY_ROOT))
    project_root = (
        repository_root.parent.parent
        if repository_root.parent.name == "run_sources"
        else repository_root
    )
    path = Path(value)
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise GateValidationError(f"{label} must be a normalized absolute path")
    if path == project_root or project_root not in path.parents:
        raise GateValidationError(f"{label} must remain inside the containing project")
    return path


def _failure_supporting_reference(
    value: object,
    *,
    label: str,
    artifact_loader: Callable[[Path], bytes],
    expected_path: Path | None = None,
) -> dict[str, Any]:
    reference = _mapping(value, label)
    _exact_keys(reference, {"path", "sha256", "size_bytes"}, label)
    path = _absolute_repository_path(reference.get("path"), f"{label} path")
    if expected_path is not None and path != expected_path:
        raise GateValidationError(f"{label} path differs from its bound artifact")
    sha256 = _sha256(reference.get("sha256"), f"{label} digest")
    size_bytes = _integer(reference.get("size_bytes"), f"{label} size", minimum=0)
    relative_path = path.relative_to(Path(os.path.abspath(REPOSITORY_ROOT)))
    try:
        payload = artifact_loader(relative_path)
    except GateValidationError:
        raise
    except Exception as error:
        raise GateValidationError(
            f"{label} is not committed and available: {relative_path}"
        ) from error
    if not isinstance(payload, bytes):
        raise GateValidationError("pilot artifact loader must return bytes")
    if len(payload) != size_bytes or _sha256_bytes(payload) != sha256:
        raise GateValidationError(f"{label} committed bytes differ from its receipt")
    return {
        "relative_path": relative_path,
        "sha256": sha256,
        "size_bytes": size_bytes,
    }


def _validate_pilot_failure_receipt(
    receipt: Mapping[str, Any],
    *,
    receipt_path: Path,
    attempt_id: str,
    candidate_id: str,
    pilot_seed: int,
    artifact_loader: Callable[[Path], bytes],
) -> dict[str, Any]:
    """Validate one launcher-authored failed-seed receipt and its committed support."""

    label = f"pilot failure receipt {receipt_path.as_posix()}"
    _exact_keys(
        receipt,
        {
            "schema_version",
            "artifact_kind",
            "status",
            "attempt_id",
            "candidate_id",
            "pilot_seed",
            "pilot_mode",
            "requested_samples",
            "stage",
            "reason",
            "started_at_utc",
            "failed_at_utc",
            "process_exit_status",
            "checkpoint",
            "config",
            "command",
            "source_revision",
            "launcher_source",
            "log",
            "partial_artifacts",
        },
        label,
    )
    expected_bindings = {
        "schema_version": PILOT_FAILURE_RECEIPT_SCHEMA_VERSION,
        "artifact_kind": PILOT_FAILURE_RECEIPT_KIND,
        "status": "failed",
        "attempt_id": attempt_id,
        "candidate_id": candidate_id,
        "pilot_seed": pilot_seed,
    }
    for field, expected in expected_bindings.items():
        if receipt.get(field) != expected:
            raise GateValidationError(f"{label} {field} disagrees with its envelope")
    pilot_mode = receipt.get("pilot_mode")
    requested_samples = _integer(
        receipt.get("requested_samples"), f"{label} requested samples", minimum=1
    )
    if pilot_mode == "engineering":
        if pilot_seed < 1000 or requested_samples > 100:
            raise GateValidationError(
                f"{label} engineering mode requires seed >=1000 and 1..100 samples"
            )
    elif pilot_mode == "registered_selection":
        if (
            pilot_seed not in REGISTERED_SELECTION_PILOT_SEEDS
            or requested_samples != REGISTERED_SELECTION_SAMPLES_PER_SEED
        ):
            raise GateValidationError(
                f"{label} registered-selection mode has an invalid seed/sample count"
            )
    else:
        raise GateValidationError(f"{label} pilot mode is invalid")

    started_at = _timestamp(receipt.get("started_at_utc"), f"{label} start")
    failed_at = _timestamp(receipt.get("failed_at_utc"), f"{label} failure")
    if failed_at < started_at:
        raise GateValidationError(f"{label} failure predates its child start")
    stage = receipt.get("stage")
    process_exit_status = receipt.get("process_exit_status")
    if stage == "benchmark_child_process":
        if type(process_exit_status) is not int or process_exit_status == 0:
            raise GateValidationError(
                f"{label} child-process failure requires a nonzero exit status"
            )
    elif stage == "completion_validation":
        if process_exit_status is not None:
            raise GateValidationError(
                f"{label} completion-validation failure requires a null exit status"
            )
    else:
        raise GateValidationError(f"{label} stage is invalid")
    reason = receipt.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise GateValidationError(f"{label} reason must be nonempty")

    checkpoint = _mapping(receipt.get("checkpoint"), f"{label} checkpoint")
    _exact_keys(
        checkpoint,
        {"path", "sha256", "size_bytes", "global_step"},
        f"{label} checkpoint",
    )
    checkpoint_path = _absolute_project_checkpoint_path(
        checkpoint.get("path"), f"{label} checkpoint path"
    )
    checkpoint_identity = {
        "path": checkpoint_path,
        "sha256": _sha256(checkpoint.get("sha256"), f"{label} checkpoint digest"),
        "size_bytes": _integer(
            checkpoint.get("size_bytes"), f"{label} checkpoint size", minimum=1
        ),
        "global_step": _integer(
            checkpoint.get("global_step"), f"{label} checkpoint step", minimum=0
        ),
    }

    config = _mapping(receipt.get("config"), f"{label} config")
    _exact_keys(
        config,
        {"path", "sha256", "sampling", "sampling_sha256"},
        f"{label} config",
    )
    config_path = _absolute_repository_path(config.get("path"), f"{label} config path")
    config_relative_path = config_path.relative_to(
        Path(os.path.abspath(REPOSITORY_ROOT))
    )
    config_sha256 = _sha256(config.get("sha256"), f"{label} config digest")
    sampling = _mapping(config.get("sampling"), f"{label} sampling config")
    sampling_sha256 = _sha256(config.get("sampling_sha256"), f"{label} sampling digest")
    if canonical_json_sha256(sampling) != sampling_sha256:
        raise GateValidationError(f"{label} sampling digest is not canonical")
    try:
        normalized_sampling = denovo_report.validate_sampling_config(sampling)
    except ValueError as error:
        raise GateValidationError(
            f"{label} sampling config is invalid: {error}"
        ) from error
    if normalized_sampling != sampling or sampling.get("diffusion_type") != "udlm":
        raise GateValidationError(f"{label} must describe canonical UDLM sampling")
    if pilot_mode == "registered_selection" and sampling.get("num_steps") != 128:
        raise GateValidationError(
            f"{label} registered selection requires the 128-NFE sampling config"
        )

    source = _mapping(receipt.get("source_revision"), f"{label} source revision")
    _exact_keys(source, {"head", "upstream"}, f"{label} source revision")
    source_revision = _git_revision(source.get("head"), f"{label} source revision")
    if source.get("upstream") != source_revision:
        raise GateValidationError(f"{label} source was not clean and pushed")

    command = receipt.get("command")
    if (
        not isinstance(command, list)
        or len(command) != 20
        or any(not isinstance(value, str) or not value for value in command)
    ):
        raise GateValidationError(f"{label} command must contain exactly ten pairs")
    interpreter = Path(command[0])
    if interpreter != _project_training_python_executable():
        raise GateValidationError(f"{label} command interpreter is not project .venv")
    expected_command = [
        command[0],
        str(
            Path(os.path.abspath(REPOSITORY_ROOT / "scripts/exps/denovo/benchmark.py"))
        ),
        "--checkpoint",
        str(checkpoint_path),
        "--expected-checkpoint-sha256",
        checkpoint_identity["sha256"],
        "--expected-source-revision",
        source_revision,
        "--config",
        str(config_path),
        "--expected-config-sha256",
        config_sha256,
        "--num-samples",
        str(requested_samples),
        "--seed",
        str(pilot_seed),
        "--device",
        "cuda:0",
        "--output-dir",
        str(Path(os.path.abspath(REPOSITORY_ROOT / receipt_path.parent))),
    ]
    if command != expected_command:
        raise GateValidationError(f"{label} command differs from its pilot launch")

    launcher_path = Path(
        os.path.abspath(REPOSITORY_ROOT / DENOVO_LAUNCHER_RELATIVE_PATH)
    )
    launcher_source = _failure_supporting_reference(
        receipt.get("launcher_source"),
        label=f"{label} launcher source",
        artifact_loader=artifact_loader,
        expected_path=launcher_path,
    )
    log = _failure_supporting_reference(
        receipt.get("log"),
        label=f"{label} log",
        artifact_loader=artifact_loader,
    )
    expected_log_name = (
        denovo_report.benchmark_run_label(
            checkpoint_identity["global_step"],
            checkpoint_identity["sha256"],
            pilot_seed,
        )
        + ".log"
    )
    if (
        log["relative_path"].name != expected_log_name
        or log["relative_path"].parent.name != attempt_id
        or log["relative_path"].parent == receipt_path.parent.parent
    ):
        raise GateValidationError(
            f"{label} log path is not keyed by a distinct attempt root"
        )

    partials = _mapping(receipt.get("partial_artifacts"), f"{label} partial artifacts")
    _exact_keys(
        partials, {"summary_json", "raw_samples_csv"}, f"{label} partial artifacts"
    )
    normalized_partials: dict[str, dict[str, Any] | None] = {}
    for name, filename in (
        ("summary_json", "summary.json"),
        ("raw_samples_csv", "raw_samples.csv"),
    ):
        value = partials.get(name)
        if value is None:
            normalized_partials[name] = None
            continue
        normalized_partials[name] = _failure_supporting_reference(
            value,
            label=f"{label} partial {name}",
            artifact_loader=artifact_loader,
            expected_path=Path(
                os.path.abspath(REPOSITORY_ROOT / receipt_path.parent / filename)
            ),
        )

    try:
        config_blob = artifact_loader(config_relative_path)
    except Exception as error:
        raise GateValidationError(
            f"{label} config is unavailable at the ledger revision"
        ) from error
    if (
        not isinstance(config_blob, bytes)
        or _sha256_bytes(config_blob) != config_sha256
    ):
        raise GateValidationError(f"{label} config Git blob differs")
    try:
        import yaml

        source_config = yaml.safe_load(config_blob.decode("utf-8"))
        if not isinstance(source_config, Mapping):
            raise ValueError("config root is not an object")
        canonical_json_sha256(source_config)
        source_sampling = denovo_report.validate_sampling_config(source_config)
    except (TypeError, UnicodeDecodeError, ValueError, yaml.YAMLError) as error:
        raise GateValidationError(
            f"{label} committed config is invalid: {error}"
        ) from error
    if source_sampling != dict(sampling):
        raise GateValidationError(
            f"{label} sampling config differs from the committed YAML"
        )
    return {
        "attempt_id": attempt_id,
        "pilot_seed": pilot_seed,
        "pilot_mode": pilot_mode,
        "requested_samples": requested_samples,
        "started_at": started_at,
        "failed_at": failed_at,
        "source_revision": source_revision,
        "launcher_source_sha256": launcher_source["sha256"],
        "config_relative_path": config_relative_path,
        "config_sha256": config_sha256,
        "checkpoint": checkpoint_identity,
        "stage": stage,
        "reason": reason,
        "process_exit_status": process_exit_status,
        "log": log,
        "partial_artifacts": normalized_partials,
    }


def _validate_pilot_run_result(
    value: object,
    *,
    evidence: Mapping[str, Any],
    summary_ref: Mapping[str, Any],
    raw_ref: Mapping[str, Any],
    receipt_ref: Mapping[str, Any],
) -> dict[str, Any]:
    label = (
        f"independent pilot run validation for {evidence['attempt_id']} "
        f"seed {evidence['pilot_seed']}"
    )
    result = _mapping(value, label)
    _exact_keys(result, _PILOT_RUN_VALIDATION_KEYS, label)
    for field in ("attempt_id", "candidate_id", "pilot_seed"):
        if result.get(field) != evidence[field]:
            raise GateValidationError(f"{label} {field} disagrees with evidence")
    requested_samples = _integer(
        result.get("requested_samples"), f"{label} requested samples", minimum=1
    )
    nfe = _integer(result.get("nfe"), f"{label} NFE", minimum=1)
    metric_branch = result.get("metric_branch")
    if metric_branch not in {"released_comparable", "strict"}:
        raise GateValidationError(f"{label} metric branch is invalid")

    checkpoint = _mapping(result.get("checkpoint"), f"{label} checkpoint")
    _exact_keys(
        checkpoint, {"sha256", "size_bytes", "global_step"}, f"{label} checkpoint"
    )
    checkpoint_normalized = {
        "sha256": _sha256(checkpoint.get("sha256"), f"{label} checkpoint digest"),
        "size_bytes": _integer(
            checkpoint.get("size_bytes"), f"{label} checkpoint size", minimum=1
        ),
        "global_step": _integer(
            checkpoint.get("global_step"), f"{label} checkpoint step", minimum=1
        ),
    }
    evaluation_config = _mapping(
        result.get("evaluation_config"), f"{label} evaluation config"
    )
    _exact_keys(
        evaluation_config,
        {"relative_path", "sha256"},
        f"{label} evaluation config",
    )
    evaluation_config_normalized = {
        "relative_path": _relative_path(
            evaluation_config.get("relative_path"),
            f"{label} evaluation config path",
            suffix=".yaml",
        ),
        "sha256": _sha256(
            evaluation_config.get("sha256"), f"{label} evaluation config digest"
        ),
    }
    sampling = _mapping(result.get("sampling"), f"{label} sampling")
    _exact_keys(sampling, {"config", "sha256"}, f"{label} sampling")
    sampling_config = _mapping(sampling.get("config"), f"{label} sampling config")
    sampling_sha256 = _sha256(sampling.get("sha256"), f"{label} sampling digest")
    if canonical_json_sha256(sampling_config) != sampling_sha256:
        raise GateValidationError(f"{label} sampling digest is not canonical")
    try:
        inference_weights = denovo_report.validate_inference_weights(
            result.get("inference_weights"), require_ema=True
        )
    except ValueError as error:
        raise GateValidationError(
            f"{label} inference weights are invalid: {error}"
        ) from error

    receipt = _mapping(
        result.get("training_exit_receipt"), f"{label} training exit receipt"
    )
    _exact_keys(
        receipt,
        {"relative_path", "sha256", "schema_version", "recorded_at_utc"},
        f"{label} training exit receipt",
    )
    receipt_normalized = _artifact_reference(
        {key: receipt[key] for key in ("relative_path", "sha256", "schema_version")},
        label=f"{label} training exit receipt",
        suffix=".json",
        require_schema=True,
    )
    if receipt_normalized != dict(receipt_ref):
        raise GateValidationError(f"{label} training receipt reference differs")
    receipt_recorded = _timestamp(
        receipt.get("recorded_at_utc"), f"{label} training receipt timestamp"
    )
    started = _timestamp(result.get("started_at_utc"), f"{label} start")
    completed = _timestamp(result.get("completed_at_utc"), f"{label} completion")
    if not receipt_recorded < started < completed:
        raise GateValidationError(
            f"{label} must start after training receipt and complete afterwards"
        )

    if result.get("summary_json_sha256") != summary_ref["sha256"]:
        raise GateValidationError(f"{label} summary digest differs from evidence")
    if result.get("raw_samples_csv_sha256") != raw_ref["sha256"]:
        raise GateValidationError(f"{label} raw CSV digest differs from evidence")
    quality = _finite(result.get("quality"), f"{label} quality")
    diversity_raw = result.get("diversity")
    diversity = (
        None if diversity_raw is None else _finite(diversity_raw, f"{label} diversity")
    )
    if not 0.0 <= quality <= 1.0 or (
        diversity is not None and not 0.0 <= diversity <= 1.0
    ):
        raise GateValidationError(f"{label} quality/diversity must lie in [0, 1]")
    rescore = _mapping(result.get("independent_rescore"), f"{label} rescore")
    expected_rescore = {
        "all_21_fields_match": True,
        "both_metric_branches_match": True,
        "failure_counts_match": True,
        "raw_model_text_redecoded": True,
    }
    _exact_keys(rescore, set(expected_rescore), f"{label} rescore")
    if dict(rescore) != expected_rescore:
        raise GateValidationError(f"{label} independent rescore is incomplete")

    normalized = {
        "attempt_id": evidence["attempt_id"],
        "candidate_id": evidence["candidate_id"],
        "pilot_seed": evidence["pilot_seed"],
        "requested_samples": requested_samples,
        "nfe": nfe,
        "metric_branch": metric_branch,
        "checkpoint": checkpoint_normalized,
        "evaluation_config": evaluation_config_normalized,
        "sampling": {"config": dict(sampling_config), "sha256": sampling_sha256},
        "inference_weights": inference_weights,
        "runner_sha256": _sha256(result.get("runner_sha256"), f"{label} runner digest"),
        "sampler_source_sha256": _sha256(
            result.get("sampler_source_sha256"), f"{label} sampler digest"
        ),
        "implementation_inputs_sha256": _sha256(
            result.get("implementation_inputs_sha256"),
            f"{label} implementation inputs digest",
        ),
        "metric_inputs_sha256": _sha256(
            result.get("metric_inputs_sha256"), f"{label} metric inputs digest"
        ),
        "benchmark_revision": _git_revision(
            result.get("benchmark_revision"), f"{label} benchmark revision"
        ),
        "started_at": started,
        "completed_at": completed,
        "training_exit_receipt": {
            **receipt_normalized,
            "recorded_at_utc": receipt_recorded.isoformat(),
        },
        "summary_json_sha256": summary_ref["sha256"],
        "raw_samples_csv_sha256": raw_ref["sha256"],
        "quality": quality,
        "diversity": diversity,
        "independent_rescore": dict(expected_rescore),
    }
    return normalized


def validate_candidate_ledger(
    ledger: Mapping[str, Any],
    *,
    candidate_id: str,
    artifact_loader: Callable[[Path], bytes],
    pilot_run_validator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute pilot selection from committed envelopes and raw molecules.

    Schema 2 treats every seed as an independent success/failure outcome.  A
    partially failed attempt therefore retains and validates its completed
    seed, but cannot enter selection.  Successful envelopes carry no score:
    the supplied validator must structurally validate the schema-7 benchmark
    artifacts and independently re-decode and re-score ``raw_model_text``.
    """

    required = {
        "schema_version",
        "protocol_id",
        "status",
        "final_seed_results_included",
        "attempts",
        "selection",
    }
    _exact_keys(ledger, required, "candidate ledger")
    if (
        ledger.get("schema_version") != CANDIDATE_LEDGER_SCHEMA_VERSION
        or ledger.get("protocol_id") != EXPECTED_PROTOCOL_ID
    ):
        raise GateValidationError("candidate ledger identity is invalid")
    if ledger.get("status") != "closed_before_final_evaluation":
        raise GateValidationError(
            "candidate ledger is not closed before final evaluation"
        )
    if ledger.get("final_seed_results_included") is not False:
        raise GateValidationError("candidate ledger must exclude final-seed results")
    attempts = ledger.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise GateValidationError(
            "candidate ledger must disclose at least one pilot attempt"
        )

    attempt_ids: set[str] = set()
    artifact_paths: set[Path] = set()
    eligible_attempts: list[dict[str, Any]] = []
    all_completed_outcomes: list[dict[str, Any]] = []
    all_failed_outcomes: list[dict[str, Any]] = []
    artifact_count = 0
    ineligible_completed_attempt_count = 0
    undefined_selection_metric_attempt_count = 0
    failed_attempt_count = 0
    partially_failed_attempt_count = 0
    for index, raw_attempt in enumerate(attempts):
        attempt_label = f"candidate ledger attempt {index}"
        attempt = _mapping(raw_attempt, attempt_label)
        _exact_keys(
            attempt,
            {
                "attempt_id",
                "candidate_id",
                "status",
                "eligible_for_selection",
                "ineligibility_reason",
                "pilot_seeds",
                "selection_score",
                "artifact_refs",
            },
            attempt_label,
        )
        attempt_id = attempt.get("attempt_id")
        if (
            not isinstance(attempt_id, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,95}", attempt_id) is None
            or attempt_id in attempt_ids
        ):
            raise GateValidationError(
                "candidate ledger attempt IDs must be unique normalized identifiers"
            )
        attempt_ids.add(attempt_id)
        attempt_candidate_id = attempt.get("candidate_id")
        if (
            not isinstance(attempt_candidate_id, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,95}", attempt_candidate_id) is None
        ):
            raise GateValidationError(
                "candidate ledger candidate_id has invalid syntax"
            )
        if attempt.get("status") not in {"completed", "failed"}:
            raise GateValidationError(
                "pilot attempt status must be completed or failed"
            )
        if type(attempt.get("eligible_for_selection")) is not bool:
            raise GateValidationError("pilot eligibility must be boolean")

        seeds = attempt.get("pilot_seeds")
        if not isinstance(seeds, list) or not seeds:
            raise GateValidationError(
                "each candidate attempt must disclose pilot seeds"
            )
        if any(type(seed) is not int or seed < 1000 for seed in seeds):
            raise GateValidationError(
                "pilot seeds must be integers greater than or equal to 1000"
            )
        if seeds != sorted(set(seeds)):
            raise GateValidationError("pilot seeds must be unique and sorted")
        if set(seeds).intersection(EXPECTED_SEEDS):
            raise GateValidationError(
                "candidate ledger contains a forbidden final seed"
            )
        refs = attempt.get("artifact_refs")
        if not isinstance(refs, list) or len(refs) != len(seeds):
            raise GateValidationError(
                "each pilot seed must bind exactly one pilot outcome artifact"
            )

        evidence_seeds: set[int] = set()
        completed_rows: list[dict[str, Any]] = []
        failed_outcomes = 0
        for ref_index, raw_ref in enumerate(refs):
            ref_label = f"candidate attempt {attempt_id} artifact ref {ref_index}"
            ref = _mapping(raw_ref, ref_label)
            _exact_keys(
                ref,
                {
                    "artifact_kind",
                    "pilot_seed",
                    "relative_path",
                    "sha256",
                    "schema_version",
                },
                ref_label,
            )
            artifact_kind = ref.get("artifact_kind")
            if artifact_kind not in {"pilot_evaluation", "pilot_failure"}:
                raise GateValidationError(f"{ref_label} kind is invalid")
            ref_seed = ref.get("pilot_seed")
            if type(ref_seed) is not int or ref_seed not in seeds:
                raise GateValidationError(f"{ref_label} pilot seed is not declared")
            if ref_seed in evidence_seeds:
                raise GateValidationError(
                    f"candidate attempt {attempt_id} repeats pilot evidence seed"
                )
            evidence_seeds.add(ref_seed)
            relative_path = _relative_path(
                ref.get("relative_path"), f"{ref_label} path", suffix=".json"
            )
            if relative_path.parts != (
                "experiments",
                "udlm",
                "pilots",
                attempt_id,
                f"seed_{ref_seed}.json",
            ):
                raise GateValidationError(
                    "pilot evidence path must be exactly experiments/udlm/pilots/"
                    "<attempt_id>/seed_<pilot_seed>.json"
                )
            if relative_path in artifact_paths:
                raise GateValidationError(
                    "pilot evidence paths must be unique across the ledger"
                )
            artifact_paths.add(relative_path)
            expected_sha = _sha256(ref.get("sha256"), f"{ref_label} digest")
            if ref.get("schema_version") != PILOT_EVIDENCE_SCHEMA_VERSION:
                raise GateValidationError(f"{ref_label} schema version is unsupported")
            try:
                artifact_blob = artifact_loader(relative_path)
            except GateValidationError:
                raise
            except Exception as error:
                raise GateValidationError(
                    f"pilot evidence is unavailable: {relative_path}"
                ) from error
            if not isinstance(artifact_blob, bytes):
                raise GateValidationError("pilot artifact loader must return bytes")
            if _sha256_bytes(artifact_blob) != expected_sha:
                raise GateValidationError(
                    f"pilot evidence digest differs: {relative_path}"
                )
            evidence_label = f"pilot evidence {relative_path.as_posix()}"
            evidence = _mapping(
                strict_json_loads(artifact_blob, label=evidence_label), evidence_label
            )
            common_fields = {
                "schema_version",
                "artifact_kind",
                "status",
                "attempt_id",
                "candidate_id",
                "pilot_seed",
                "final_seed_results_included",
            }
            if artifact_kind == "pilot_evaluation":
                _exact_keys(
                    evidence,
                    common_fields | {"training_exit_receipt", "benchmark_artifacts"},
                    evidence_label,
                )
                expected_status = "completed"
            else:
                _exact_keys(
                    evidence,
                    common_fields | {"failure_receipt"},
                    evidence_label,
                )
                expected_status = "failed"
            bindings = {
                "schema_version": PILOT_EVIDENCE_SCHEMA_VERSION,
                "artifact_kind": artifact_kind,
                "status": expected_status,
                "attempt_id": attempt_id,
                "candidate_id": attempt_candidate_id,
                "pilot_seed": ref_seed,
                "final_seed_results_included": False,
            }
            for field, expected in bindings.items():
                if evidence.get(field) != expected:
                    raise GateValidationError(
                        f"pilot evidence {field} disagrees with its ledger attempt"
                    )

            if artifact_kind == "pilot_evaluation":
                receipt_ref = _artifact_reference(
                    evidence.get("training_exit_receipt"),
                    label=f"{evidence_label} training exit receipt",
                    suffix=".json",
                    require_schema=True,
                )
                receipt_parts = receipt_ref["relative_path"].parts
                if (
                    receipt_ref["schema_version"] != PILOT_EXIT_STATUS_SCHEMA_VERSION
                    or len(receipt_parts) != 4
                    or receipt_parts[:2] != ("output", "udlm")
                    or RUN_NAME_PATTERN.fullmatch(receipt_parts[2]) is None
                    or receipt_parts[3] != "pilot_exit_status.json"
                ):
                    raise GateValidationError(
                        f"{evidence_label} training exit receipt path/schema is invalid"
                    )
                benchmark_artifacts = _mapping(
                    evidence.get("benchmark_artifacts"),
                    f"{evidence_label} benchmark artifacts",
                )
                _exact_keys(
                    benchmark_artifacts,
                    {"summary_json", "raw_samples_csv"},
                    f"{evidence_label} benchmark artifacts",
                )
                summary_ref = _pilot_output_artifact_reference(
                    benchmark_artifacts.get("summary_json"),
                    label=f"{evidence_label} summary",
                    suffix=".json",
                    attempt_id=attempt_id,
                    pilot_seed=ref_seed,
                    require_schema=True,
                )
                if summary_ref["schema_version"] != denovo_report.RUN_SCHEMA_VERSION:
                    raise GateValidationError(
                        f"{evidence_label} benchmark summary schema is unsupported"
                    )
                raw_ref = _pilot_output_artifact_reference(
                    benchmark_artifacts.get("raw_samples_csv"),
                    label=f"{evidence_label} raw samples",
                    suffix=".csv",
                    attempt_id=attempt_id,
                    pilot_seed=ref_seed,
                    require_schema=False,
                )
                if (
                    summary_ref["relative_path"].parent
                    != raw_ref["relative_path"].parent
                ):
                    raise GateValidationError(
                        f"{evidence_label} summary and raw CSV must be siblings"
                    )
                try:
                    raw_result = pilot_run_validator(evidence)
                except GateValidationError:
                    raise
                except Exception as error:
                    raise GateValidationError(
                        f"{evidence_label} independent validation failed: {error}"
                    ) from error
                completed_row = _validate_pilot_run_result(
                    raw_result,
                    evidence=evidence,
                    summary_ref=summary_ref,
                    raw_ref=raw_ref,
                    receipt_ref=receipt_ref,
                )
                completed_rows.append(completed_row)
                all_completed_outcomes.append(
                    {
                        "attempt_id": attempt_id,
                        "pilot_seed": ref_seed,
                        "benchmark_revision": completed_row["benchmark_revision"],
                        "completed_at_utc": completed_row["completed_at"].isoformat(),
                    }
                )
            else:
                failure_ref = _artifact_reference(
                    evidence.get("failure_receipt"),
                    label=f"{evidence_label} failure receipt",
                    suffix=".json",
                    require_schema=True,
                )
                failure_path = failure_ref["relative_path"]
                if (
                    failure_ref["schema_version"]
                    != PILOT_FAILURE_RECEIPT_SCHEMA_VERSION
                    or len(failure_path.parts) < 4
                    or failure_path.parts[0] != "output"
                    or failure_path.parent.name != f"seed_{ref_seed}"
                    or failure_path.parent.parent.name != attempt_id
                    or failure_path.name != "failure_receipt.json"
                ):
                    raise GateValidationError(
                        "pilot failure receipt must be exactly output/.../"
                        "<attempt_id>/seed_<pilot_seed>/failure_receipt.json"
                    )
                try:
                    failure_blob = artifact_loader(failure_path)
                except GateValidationError:
                    raise
                except Exception as error:
                    raise GateValidationError(
                        f"pilot failure receipt is unavailable: {failure_path}"
                    ) from error
                if (
                    not isinstance(failure_blob, bytes)
                    or _sha256_bytes(failure_blob) != failure_ref["sha256"]
                ):
                    raise GateValidationError(
                        f"pilot failure receipt digest differs: {failure_path}"
                    )
                failure_document = _mapping(
                    strict_json_loads(
                        failure_blob,
                        label=f"pilot failure receipt {failure_path.as_posix()}",
                    ),
                    f"pilot failure receipt {failure_path.as_posix()}",
                )
                failure_result = _validate_pilot_failure_receipt(
                    failure_document,
                    receipt_path=failure_path,
                    attempt_id=attempt_id,
                    candidate_id=attempt_candidate_id,
                    pilot_seed=ref_seed,
                    artifact_loader=artifact_loader,
                )
                all_failed_outcomes.append(failure_result)
                failed_outcomes += 1
            artifact_count += 1

        if evidence_seeds != set(seeds):
            raise GateValidationError(
                f"candidate attempt {attempt_id} lacks evidence for a pilot seed"
            )
        derived_status = "failed" if failed_outcomes else "completed"
        if attempt.get("status") != derived_status:
            raise GateValidationError(
                "pilot attempt status disagrees with its per-seed outcomes"
            )
        if completed_rows:
            identity_fields = (
                "checkpoint",
                "evaluation_config",
                "sampling",
                "inference_weights",
                "runner_sha256",
                "sampler_source_sha256",
                "implementation_inputs_sha256",
                "metric_inputs_sha256",
                "benchmark_revision",
                "training_exit_receipt",
            )
            first_identity = {
                field: completed_rows[0][field] for field in identity_fields
            }
            for row in completed_rows[1:]:
                if {field: row[field] for field in identity_fields} != first_identity:
                    raise GateValidationError(
                        "all completed seed outcomes for one attempt must use one "
                        "checkpoint, config, implementation, and training receipt"
                    )
        else:
            first_identity = None

        if failed_outcomes:
            if (
                attempt.get("eligible_for_selection") is not False
                or attempt.get("selection_score") is not None
                or attempt.get("ineligibility_reason") != FAILED_PILOT_REASON
            ):
                raise GateValidationError(
                    "any failed seed makes the complete attempt ineligible with the "
                    "fixed failure reason and a null score"
                )
            failed_attempt_count += 1
            if completed_rows:
                partially_failed_attempt_count += 1
            continue

        registered_operating_point = tuple(
            seeds
        ) == REGISTERED_SELECTION_PILOT_SEEDS and all(
            row["requested_samples"] == REGISTERED_SELECTION_SAMPLES_PER_SEED
            and row["nfe"] == REGISTERED_SELECTION_NFE
            and row["metric_branch"] == REGISTERED_SELECTION_METRIC_BRANCH
            for row in completed_rows
        )
        undefined_selection_metric = any(
            row["diversity"] is None for row in completed_rows
        )
        derived_eligible = registered_operating_point and not undefined_selection_metric
        if attempt.get("eligible_for_selection") is not derived_eligible:
            raise GateValidationError(
                "completed pilot eligibility disagrees with the registered operating point"
            )
        if not derived_eligible:
            expected_reason = (
                UNDEFINED_SELECTION_METRIC_REASON
                if registered_operating_point and undefined_selection_metric
                else NONREGISTERED_OPERATING_POINT_REASON
            )
            if (
                attempt.get("selection_score") is not None
                or attempt.get("ineligibility_reason") != expected_reason
            ):
                raise GateValidationError(
                    "nonregistered completed pilots must have a null score and the "
                    "derived fixed ineligibility reason"
                )
            ineligible_completed_attempt_count += 1
            if undefined_selection_metric:
                undefined_selection_metric_attempt_count += 1
            continue

        if attempt.get("ineligibility_reason") is not None:
            raise GateValidationError(
                "eligible pilot attempts must have a null ineligibility reason"
            )
        score = _mapping(
            attempt.get("selection_score"),
            f"candidate ledger attempt {attempt_id} selection score",
        )
        _exact_keys(
            score,
            {"mean_released_quality", "mean_released_diversity"},
            f"candidate ledger attempt {attempt_id} selection score",
        )
        recomputed_quality = statistics.fmean(row["quality"] for row in completed_rows)
        recomputed_diversity = statistics.fmean(
            row["diversity"] for row in completed_rows
        )
        _close(
            score.get("mean_released_quality"),
            recomputed_quality,
            f"candidate attempt {attempt_id} mean released quality",
        )
        _close(
            score.get("mean_released_diversity"),
            recomputed_diversity,
            f"candidate attempt {attempt_id} mean released diversity",
        )
        eligible_attempts.append(
            {
                "attempt_id": attempt_id,
                "candidate_id": attempt_candidate_id,
                "quality": recomputed_quality,
                "diversity": recomputed_diversity,
                "identity": first_identity,
                "completed_at": max(row["completed_at"] for row in completed_rows),
            }
        )

    selection = _mapping(ledger.get("selection"), "candidate ledger selection")
    _exact_keys(
        selection,
        {
            "candidate_id",
            "selected_attempt_id",
            "rule",
            "checkpoint_selection_rule",
            "selected_without_final_seed_results",
        },
        "candidate ledger selection",
    )
    if selection.get("candidate_id") != candidate_id:
        raise GateValidationError("candidate ledger selected a different candidate")
    selected_attempt_id = selection.get("selected_attempt_id")
    if not isinstance(selected_attempt_id, str) or not selected_attempt_id:
        raise GateValidationError("selected candidate attempt ID must be a string")
    _required_true(
        selection.get("selected_without_final_seed_results"),
        "ledger selection without final seeds",
    )
    if selection.get("rule") != CANDIDATE_SELECTION_RULE:
        raise GateValidationError(
            "candidate ledger selection rule is not the frozen enum"
        )
    if selection.get("checkpoint_selection_rule") != CHECKPOINT_SELECTION_RULE:
        raise GateValidationError(
            "candidate ledger checkpoint-selection rule is not the frozen enum"
        )
    if not eligible_attempts:
        raise GateValidationError("candidate ledger has no eligible completed attempt")
    expected_selected = min(
        eligible_attempts,
        key=lambda item: (-item["quality"], -item["diversity"], item["attempt_id"]),
    )
    if selected_attempt_id != expected_selected["attempt_id"]:
        raise GateValidationError(
            "selected_attempt_id is not the deterministic pilot-score winner"
        )
    if selection.get("candidate_id") != expected_selected["candidate_id"]:
        raise GateValidationError(
            "selected candidate_id disagrees with the deterministic winner"
        )
    if expected_selected["candidate_id"] != candidate_id:
        raise GateValidationError("candidate lock does not name the pilot-score winner")
    selected_identity = expected_selected["identity"]
    if not isinstance(
        selected_identity, Mapping
    ):  # pragma: no cover - eligible invariant
        raise GateValidationError("selected attempt has no completed run identity")
    return {
        "attempt_count": len(attempts),
        "eligible_attempt_count": len(eligible_attempts),
        "ineligible_completed_attempt_count": ineligible_completed_attempt_count,
        "undefined_selection_metric_attempt_count": (
            undefined_selection_metric_attempt_count
        ),
        "failed_attempt_count": failed_attempt_count,
        "partially_failed_attempt_count": partially_failed_attempt_count,
        "committed_pilot_artifact_count": artifact_count,
        "completed_outcomes": sorted(
            all_completed_outcomes,
            key=lambda row: (row["attempt_id"], row["pilot_seed"]),
        ),
        "failed_outcomes": sorted(
            all_failed_outcomes,
            key=lambda row: (row["attempt_id"], row["pilot_seed"]),
        ),
        "selected_attempt_id": expected_selected["attempt_id"],
        "selected_candidate_id": expected_selected["candidate_id"],
        "selected_checkpoint": selected_identity["checkpoint"],
        "selected_checkpoint_sha256": selected_identity["checkpoint"]["sha256"],
        "selected_evaluation_config": selected_identity["evaluation_config"],
        "selected_sampling": selected_identity["sampling"],
        "selected_inference_weights": selected_identity["inference_weights"],
        "selected_runner_sha256": selected_identity["runner_sha256"],
        "selected_sampler_source_sha256": selected_identity["sampler_source_sha256"],
        "selected_implementation_inputs_sha256": selected_identity[
            "implementation_inputs_sha256"
        ],
        "selected_metric_inputs_sha256": selected_identity["metric_inputs_sha256"],
        "selected_benchmark_revision": selected_identity["benchmark_revision"],
        "selected_training_exit_receipt": selected_identity["training_exit_receipt"],
        "selected_pilot_completed_at_utc": expected_selected[
            "completed_at"
        ].isoformat(),
        "selected_score": {
            "mean_released_quality": expected_selected["quality"],
            "mean_released_diversity": expected_selected["diversity"],
        },
        "selection_recomputed_from_pilot_evidence": True,
        "selection_recomputed_from_raw_model_text": True,
    }


def _git_blob(revision: str, relative_path: Path) -> bytes:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(REPOSITORY_ROOT),
                "show",
                f"{revision}:{relative_path.as_posix()}",
            ],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise GateValidationError(
            f"Git revision {revision} lacks {relative_path.as_posix()}"
        ) from error
    return result.stdout


def validate_git_lock_firewall(
    *,
    benchmark_revision: str,
    candidate_lock_path: Path,
    candidate_lock_bytes: bytes,
    lock: Mapping[str, Any],
    pilot_run_validator: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Prove the benchmark revision already contained the lock and ledger."""

    if _git_blob(benchmark_revision, candidate_lock_path) != candidate_lock_bytes:
        raise GateValidationError("benchmark revision candidate-lock blob differs")
    protocol_blob = _git_blob(benchmark_revision, PROTOCOL_RELATIVE_PATH)
    if _sha256_bytes(protocol_blob) != PROTOCOL_SHA256:
        raise GateValidationError(
            "benchmark revision superiority protocol blob differs"
        )
    baseline_blob = _git_blob(benchmark_revision, BASELINE_RELATIVE_PATH)
    if _sha256_bytes(baseline_blob) != BASELINE_SHA256:
        raise GateValidationError("benchmark revision baseline manifest blob differs")
    baseline_rescore_blob = _git_blob(
        benchmark_revision, BASELINE_RESCORE_RELATIVE_PATH
    )
    if _sha256_bytes(baseline_rescore_blob) != BASELINE_RESCORE_SHA256:
        raise GateValidationError(
            "benchmark revision baseline rescore-attestation blob differs"
        )
    baseline_document = _mapping(
        strict_json_loads(baseline_blob, label="baseline manifest Git blob"),
        "baseline manifest Git blob",
    )
    baseline_rescore_document = _mapping(
        strict_json_loads(
            baseline_rescore_blob, label="baseline rescore-attestation Git blob"
        ),
        "baseline rescore-attestation Git blob",
    )
    baseline_rescore_evidence = validate_baseline_rescore_attestation(
        baseline_rescore_document, baseline_document
    )
    attested_source_files = _mapping(
        _mapping(
            baseline_rescore_document.get("source"),
            "baseline rescore Git source",
        ).get("files"),
        "baseline rescore Git source files",
    )
    for name, (relative_path, _module_name) in _BASELINE_SOURCE_FILES.items():
        source_blob = _git_blob(
            EXPECTED_BASELINE_RESCORE_SOURCE_REVISION, Path(relative_path)
        )
        if _sha256_bytes(source_blob) != attested_source_files[name]["sha256"]:
            raise GateValidationError(
                f"baseline rescore source-revision blob differs for {relative_path}"
            )
    ledger_ref = lock["ledger"]
    ledger_blob = _git_blob(benchmark_revision, ledger_ref["relative_path"])
    if _sha256_bytes(ledger_blob) != ledger_ref["sha256"]:
        raise GateValidationError("benchmark revision candidate-ledger blob differs")
    ledger = _mapping(
        strict_json_loads(ledger_blob, label="candidate ledger Git blob"),
        "candidate ledger Git blob",
    )
    if ledger.get("schema_version") != ledger_ref["schema_version"]:
        raise GateValidationError("candidate ledger schema disagrees with lock")
    live_pilot_validator = (
        _validate_completed_pilot_evidence_live
        if pilot_run_validator is None
        else pilot_run_validator
    )

    def committed_pilot_validator(evidence: Mapping[str, Any]) -> Mapping[str, Any]:
        _require_completed_pilot_references_at_revision(evidence, benchmark_revision)
        return live_pilot_validator(evidence)

    ledger_evidence = validate_candidate_ledger(
        ledger,
        candidate_id=lock["candidate_id"],
        artifact_loader=lambda relative_path: _git_blob(
            benchmark_revision, relative_path
        ),
        pilot_run_validator=committed_pilot_validator,
    )
    for outcome in ledger_evidence["completed_outcomes"]:
        if (
            not _timestamp(
                outcome["completed_at_utc"],
                f"pilot {outcome['attempt_id']} seed {outcome['pilot_seed']} completion",
            )
            < lock["locked_at"]
        ):
            raise GateValidationError(
                "every disclosed completed pilot outcome must strictly predate "
                "the candidate lock"
            )
        try:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(REPOSITORY_ROOT),
                    "merge-base",
                    "--is-ancestor",
                    outcome["benchmark_revision"],
                    benchmark_revision,
                ],
                check=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise GateValidationError(
                "a disclosed pilot benchmark revision is not an ancestor of the "
                "final benchmark revision"
            ) from error
    for outcome in ledger_evidence["failed_outcomes"]:
        if not outcome["failed_at"] < lock["locked_at"]:
            raise GateValidationError(
                "every disclosed failed pilot outcome must strictly predate "
                "the candidate lock"
            )
        checkpoint_snapshot = _project_checkpoint_snapshot(
            outcome["checkpoint"]["path"],
            label=(
                f"failed pilot {outcome['attempt_id']} seed "
                f"{outcome['pilot_seed']} checkpoint"
            ),
        )
        if (
            checkpoint_snapshot["sha256"] != outcome["checkpoint"]["sha256"]
            or checkpoint_snapshot["size_bytes"] != outcome["checkpoint"]["size_bytes"]
        ):
            raise GateValidationError(
                "failed pilot checkpoint bytes differ from its producer receipt"
            )
        if (
            outcome["launcher_source_sha256"]
            != lock["benchmark_launcher_source_sha256"]
        ):
            raise GateValidationError(
                "failed pilot launcher source differs from the candidate lock"
            )
        failure_revision = outcome["source_revision"]
        try:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(REPOSITORY_ROOT),
                    "merge-base",
                    "--is-ancestor",
                    failure_revision,
                    benchmark_revision,
                ],
                check=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise GateValidationError(
                "a disclosed failed-pilot source revision is not an ancestor of "
                "the final benchmark revision"
            ) from error
        if (
            _sha256_bytes(_git_blob(failure_revision, DENOVO_LAUNCHER_RELATIVE_PATH))
            != outcome["launcher_source_sha256"]
        ):
            raise GateValidationError(
                "failed pilot source revision does not contain the receipt launcher"
            )
        if (
            _sha256_bytes(_git_blob(failure_revision, outcome["config_relative_path"]))
            != outcome["config_sha256"]
        ):
            raise GateValidationError(
                "failed pilot source revision does not contain the receipt config"
            )
    if ledger["selection"]["rule"] != lock["selection_rule"]:
        raise GateValidationError("candidate-lock and ledger selection rules disagree")
    if (
        ledger["selection"]["checkpoint_selection_rule"]
        != lock["checkpoint_selection_rule"]
    ):
        raise GateValidationError(
            "candidate-lock and ledger checkpoint-selection rules disagree"
        )
    locked_checkpoint_identity = {
        key: lock["checkpoint"][key] for key in ("sha256", "size_bytes", "global_step")
    }
    if ledger_evidence["selected_checkpoint"] != locked_checkpoint_identity:
        raise GateValidationError(
            "candidate-lock checkpoint identity is not the pilot-ledger winner checkpoint"
        )
    if ledger_evidence["selected_evaluation_config"] != {
        "relative_path": lock["evaluation_config_relative_path"],
        "sha256": lock["evaluation_config_sha256"],
    }:
        raise GateValidationError(
            "candidate-lock evaluation config is not the pilot-ledger winner config"
        )
    if ledger_evidence["selected_sampling"] != {
        "config": lock["sampling_config"],
        "sha256": lock["sampling_sha256"],
    }:
        raise GateValidationError(
            "candidate-lock sampling config is not the pilot-ledger winner config"
        )
    selected_identity_checks = {
        "selected_inference_weights": lock["inference_weights"],
        "selected_runner_sha256": lock["benchmark_runner_sha256"],
        "selected_sampler_source_sha256": lock["sampler_source_sha256"],
        "selected_implementation_inputs_sha256": lock["implementation_inputs_sha256"],
        "selected_metric_inputs_sha256": lock["metric_inputs_sha256"],
    }
    for evidence_field, expected in selected_identity_checks.items():
        if ledger_evidence[evidence_field] != expected:
            raise GateValidationError(
                f"candidate-lock {evidence_field.removeprefix('selected_')} "
                "is not the pilot-ledger winner identity"
            )
    selected_receipt = ledger_evidence["selected_training_exit_receipt"]
    selected_receipt_ref = {
        key: selected_receipt[key]
        for key in ("relative_path", "sha256", "schema_version")
    }
    if selected_receipt_ref != lock["receipt"]:
        raise GateValidationError(
            "candidate-lock training receipt is not the pilot-ledger winner receipt"
        )
    if (
        not _timestamp(
            ledger_evidence["selected_pilot_completed_at_utc"],
            "selected pilot completion",
        )
        < lock["locked_at"]
    ):
        raise GateValidationError(
            "selected pilot evidence must strictly predate the candidate lock"
        )
    selected_pilot_revision = ledger_evidence["selected_benchmark_revision"]
    selected_revision_blobs = {
        lock["evaluation_config_relative_path"]: lock["evaluation_config_sha256"],
        Path("scripts/exps/denovo/benchmark.py"): lock["benchmark_runner_sha256"],
        Path("src/genmol/sampler.py"): lock["sampler_source_sha256"],
        Path("scripts/exps/denovo/report.py"): lock["report_source_sha256"],
        DENOVO_LAUNCHER_RELATIVE_PATH: lock["benchmark_launcher_source_sha256"],
        PILOT_EVIDENCE_WRITER_RELATIVE_PATH: lock[
            "pilot_evidence_writer_source_sha256"
        ],
        DENOVO_RESCORE_RELATIVE_PATH: lock["rescore_source_sha256"],
        DENOVO_RESCORE_DEPENDENCY_RELATIVE_PATH: lock["rescore_dependency_sha256"],
    }
    for relative_path, expected_digest in selected_revision_blobs.items():
        if (
            _sha256_bytes(_git_blob(selected_pilot_revision, relative_path))
            != expected_digest
        ):
            raise GateValidationError(
                "selected pilot revision does not contain its locked evaluation "
                f"source/config bytes: {relative_path.as_posix()}"
            )
    config_blob = _git_blob(benchmark_revision, lock["evaluation_config_relative_path"])
    if _sha256_bytes(config_blob) != lock["evaluation_config_sha256"]:
        raise GateValidationError("benchmark revision evaluation-config blob differs")
    sampler_blob = _git_blob(benchmark_revision, Path("src/genmol/sampler.py"))
    if _sha256_bytes(sampler_blob) != lock["sampler_source_sha256"]:
        raise GateValidationError("benchmark revision sampler source blob differs")
    runner_blob = _git_blob(
        benchmark_revision, Path("scripts/exps/denovo/benchmark.py")
    )
    if _sha256_bytes(runner_blob) != lock["benchmark_runner_sha256"]:
        raise GateValidationError("benchmark revision runner source blob differs")
    launcher_blob = _git_blob(benchmark_revision, DENOVO_LAUNCHER_RELATIVE_PATH)
    if _sha256_bytes(launcher_blob) != lock["benchmark_launcher_source_sha256"]:
        raise GateValidationError("benchmark revision launcher source blob differs")
    pilot_writer_blob = _git_blob(
        benchmark_revision, PILOT_EVIDENCE_WRITER_RELATIVE_PATH
    )
    if _sha256_bytes(pilot_writer_blob) != lock["pilot_evidence_writer_source_sha256"]:
        raise GateValidationError(
            "benchmark revision pilot-evidence-writer source blob differs"
        )
    gate_blob = _git_blob(benchmark_revision, Path("scripts/udlm/superiority_gate.py"))
    if _sha256_bytes(gate_blob) != lock["gate_source_sha256"]:
        raise GateValidationError("benchmark revision superiority-gate blob differs")
    report_blob = _git_blob(benchmark_revision, Path("scripts/exps/denovo/report.py"))
    if _sha256_bytes(report_blob) != lock["report_source_sha256"]:
        raise GateValidationError("benchmark revision de-novo-report blob differs")
    rescore_blob = _git_blob(benchmark_revision, DENOVO_RESCORE_RELATIVE_PATH)
    if _sha256_bytes(rescore_blob) != lock["rescore_source_sha256"]:
        raise GateValidationError(
            "benchmark revision independent-rescore source blob differs"
        )
    rescore_dependency_blob = _git_blob(
        benchmark_revision, DENOVO_RESCORE_DEPENDENCY_RELATIVE_PATH
    )
    if _sha256_bytes(rescore_dependency_blob) != lock["rescore_dependency_sha256"]:
        raise GateValidationError(
            "benchmark revision independent-rescore dependency blob differs"
        )
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(REPOSITORY_ROOT),
                "merge-base",
                "--is-ancestor",
                lock["source_revision"],
                benchmark_revision,
            ],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise GateValidationError(
            "training source revision is not an ancestor of benchmark revision"
        ) from error
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(REPOSITORY_ROOT),
                "merge-base",
                "--is-ancestor",
                EXPECTED_BASELINE_RESCORE_SOURCE_REVISION,
                benchmark_revision,
            ],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise GateValidationError(
            "baseline rescore source revision is not an ancestor of benchmark revision"
        ) from error
    return {
        "benchmark_revision": benchmark_revision,
        "candidate_lock_relative_path": candidate_lock_path.as_posix(),
        "candidate_lock_sha256": _sha256_bytes(candidate_lock_bytes),
        "candidate_lock_exact_blob_at_benchmark_revision": True,
        "protocol_exact_blob_at_benchmark_revision": True,
        "baseline_exact_blob_at_benchmark_revision": True,
        "baseline_rescore_attestation_relative_path": (
            BASELINE_RESCORE_RELATIVE_PATH.as_posix()
        ),
        "baseline_rescore_attestation_sha256": BASELINE_RESCORE_SHA256,
        "baseline_rescore_attestation_exact_blob_at_benchmark_revision": True,
        "baseline_rescore_source_revision": (EXPECTED_BASELINE_RESCORE_SOURCE_REVISION),
        "baseline_rescore_source_exact_blobs_verified": True,
        "baseline_rescore_source_revision_is_ancestor": True,
        "baseline_rescore_evidence": baseline_rescore_evidence,
        "candidate_ledger_relative_path": ledger_ref["relative_path"].as_posix(),
        "candidate_ledger_sha256": ledger_ref["sha256"],
        "candidate_ledger_exact_blob_at_benchmark_revision": True,
        "evaluation_config_exact_blob_at_benchmark_revision": True,
        "ema_sampler_source_exact_blob_at_benchmark_revision": True,
        "benchmark_runner_exact_blob_at_benchmark_revision": True,
        "benchmark_launcher_exact_blob_at_benchmark_revision": True,
        "pilot_evidence_writer_exact_blob_at_benchmark_revision": True,
        "analysis_sources_exact_blobs_at_benchmark_revision": True,
        "independent_rescore_source_exact_blob_at_benchmark_revision": True,
        "independent_rescore_dependency_exact_blob_at_benchmark_revision": True,
        "committed_pilot_artifact_count": ledger_evidence[
            "committed_pilot_artifact_count"
        ],
        "eligible_pilot_attempt_count": ledger_evidence["eligible_attempt_count"],
        "ineligible_completed_pilot_attempt_count": ledger_evidence[
            "ineligible_completed_attempt_count"
        ],
        "undefined_selection_metric_pilot_attempt_count": ledger_evidence[
            "undefined_selection_metric_attempt_count"
        ],
        "failed_pilot_attempt_count": ledger_evidence["failed_attempt_count"],
        "partially_failed_pilot_attempt_count": ledger_evidence[
            "partially_failed_attempt_count"
        ],
        "registered_selection_operating_point": {
            "generation_seeds": list(REGISTERED_SELECTION_PILOT_SEEDS),
            "requested_samples_per_seed": REGISTERED_SELECTION_SAMPLES_PER_SEED,
            "nfe": REGISTERED_SELECTION_NFE,
            "metric_branch": REGISTERED_SELECTION_METRIC_BRANCH,
        },
        "selected_attempt_id": ledger_evidence["selected_attempt_id"],
        "selected_checkpoint_sha256": ledger_evidence["selected_checkpoint_sha256"],
        "selected_training_exit_receipt": selected_receipt,
        "selected_pilot_completed_at_utc": ledger_evidence[
            "selected_pilot_completed_at_utc"
        ],
        "selected_score": ledger_evidence["selected_score"],
        "selection_recomputed_from_committed_pilot_evidence": True,
        "selection_recomputed_from_raw_model_text": True,
        "training_revision_is_ancestor": True,
    }


def _candidate_series(
    candidate_report: Mapping[str, Any], lock: Mapping[str, Any]
) -> dict[str, Any]:
    if candidate_report.get("schema_version") != denovo_report.REPORT_SCHEMA_VERSION:
        raise GateValidationError("candidate report schema is not current")
    if candidate_report.get("status") != "completed":
        raise GateValidationError("candidate report is not completed")
    required = _mapping(candidate_report.get("required_protocol"), "candidate protocol")
    expected_required = {
        "seeds": list(EXPECTED_SEEDS),
        "samples_per_seed": EXPECTED_SAMPLES_PER_SEED,
        "seed_count": 3,
        "total_requested_samples": 3_000,
    }
    if dict(required) != expected_required:
        raise GateValidationError("candidate report final seed/sample protocol differs")
    checkpoint = _mapping(
        candidate_report.get("checkpoint"), "candidate report checkpoint"
    )
    if checkpoint.get("diffusion_type") != "udlm":
        raise GateValidationError("candidate report checkpoint is not UDLM")
    for key in ("sha256", "size_bytes", "global_step"):
        if checkpoint.get(key) != lock["checkpoint"][key]:
            raise GateValidationError(
                f"candidate report checkpoint {key} differs from lock"
            )
    config = _mapping(candidate_report.get("config"), "candidate report config")
    if config.get("sha256") != lock["evaluation_config_sha256"]:
        raise GateValidationError(
            "candidate evaluation config digest differs from lock"
        )
    if config.get("sampling_sha256") != lock["sampling_sha256"]:
        raise GateValidationError("candidate sampling digest differs from lock")
    if config.get("sampling") != lock["sampling_config"]:
        raise GateValidationError("candidate sampling settings differ from lock")
    tracking = _mapping(config.get("git_tracking"), "candidate config tracking")
    if (
        tracking.get("relative_path")
        != lock["evaluation_config_relative_path"].as_posix()
    ):
        raise GateValidationError("candidate evaluation config path differs from lock")

    generation = _mapping(
        candidate_report.get("generation_protocol"), "candidate generation protocol"
    )
    if generation.get("diffusion_type") != "udlm":
        raise GateValidationError("candidate generation protocol is not UDLM")
    if (
        generation.get("nfe") != EXPECTED_NFE
        or generation.get("num_steps") != EXPECTED_NFE
    ):
        raise GateValidationError("candidate final evaluation must use exactly 128 NFE")
    expected_nfe = [{"seed": seed, "nfe": EXPECTED_NFE} for seed in EXPECTED_SEEDS]
    if generation.get("nfe_by_seed") != expected_nfe:
        raise GateValidationError("candidate NFE differs across final seeds")
    try:
        inference_weights = denovo_report.validate_inference_weights(
            generation.get("inference_weights"), require_ema=True
        )
    except ValueError as error:
        raise GateValidationError(
            f"candidate inference-weight receipt is invalid: {error}"
        ) from error
    if candidate_report.get("inference_weights") != inference_weights:
        raise GateValidationError(
            "candidate top-level inference-weight receipt disagrees with generation"
        )
    if inference_weights != lock["inference_weights"]:
        raise GateValidationError(
            "candidate runtime inference-weight receipt differs from candidate lock"
        )

    ordered = _ordered_seed_rows(
        candidate_report.get("seed_runs"), label="candidate seed runs"
    )
    values: dict[str, list[float]] = {metric: [] for metric in METRICS}
    pooled_valid = 0
    pooled_requested = 0
    revisions: set[str] = set()
    started_at: list[datetime] = []
    raw_hashes: list[str] = []
    summary_hashes: list[str] = []
    root = REPOSITORY_ROOT.resolve(strict=True)
    for expected_seed, run in zip(EXPECTED_SEEDS, ordered, strict=True):
        if run.get("seed") != expected_seed:
            raise GateValidationError("candidate seed rows are incomplete")
        if run.get("inference_weights") != inference_weights:
            raise GateValidationError(
                f"candidate seed {expected_seed} inference-weight receipt disagrees"
            )
        metrics = _mapping(
            run.get("metrics"), f"candidate seed {expected_seed} metrics"
        )
        if set(metrics) != {"released_comparable", "strict"}:
            raise GateValidationError(
                "candidate must report repaired and strict branches"
            )
        branch = _mapping(metrics["released_comparable"], "candidate released metrics")
        requested = _integer(
            branch.get("validity_denominator"),
            "candidate validity denominator",
            minimum=1,
        )
        quality_denominator = _integer(
            branch.get("quality_denominator"),
            "candidate quality denominator",
            minimum=1,
        )
        if requested != EXPECTED_SAMPLES_PER_SEED or quality_denominator != requested:
            raise GateValidationError("candidate metric denominator is not 1000")
        valid_count = _integer(
            branch.get("valid_count"), "candidate valid count", minimum=0
        )
        unique_count = _integer(
            branch.get("unique_count"), "candidate unique count", minimum=0
        )
        quality_count = _integer(
            branch.get("quality_count"), "candidate quality count", minimum=0
        )
        if not 0 <= quality_count <= unique_count <= valid_count <= requested:
            raise GateValidationError("candidate count funnel is inconsistent")
        if branch.get("uniqueness_denominator") != valid_count:
            raise GateValidationError("candidate uniqueness denominator is invalid")
        expected_values = {
            "validity": valid_count / requested,
            "uniqueness": unique_count / valid_count if valid_count else math.nan,
            "quality": quality_count / requested,
        }
        for metric, expected in expected_values.items():
            _close(
                branch.get(metric), expected, f"candidate seed {expected_seed} {metric}"
            )
        diversity = _finite(branch.get("diversity"), "candidate diversity")
        if not 0 <= diversity <= 1:
            raise GateValidationError("candidate diversity must lie in [0, 1]")
        for metric in METRICS:
            values[metric].append(float(branch[metric]))
        pooled_valid += valid_count
        pooled_requested += requested
        git = _mapping(run.get("git"), "candidate seed Git evidence")
        revisions.add(_git_revision(git.get("commit"), "candidate benchmark revision"))
        if git.get("dirty") is not False:
            raise GateValidationError("candidate benchmark source was dirty")
        started_at.append(_timestamp(run.get("started_at_utc"), "candidate run start"))
        summary_path_value = run.get("summary_path")
        if not isinstance(summary_path_value, str):
            raise GateValidationError("candidate summary path is missing")
        expected_run_directory = root / lock["final_run_directories"][expected_seed]
        if Path(summary_path_value).resolve().parent != expected_run_directory:
            raise GateValidationError(
                f"candidate seed {expected_seed} used an unlocked output directory"
            )
        raw_hashes.append(
            _sha256(run.get("raw_samples_sha256"), "candidate raw digest")
        )
        summary_hashes.append(
            _sha256(run.get("summary_sha256"), "candidate summary digest")
        )
    if len(revisions) != 1:
        raise GateValidationError("candidate final seeds used different revisions")
    if any(not lock["locked_at"] < timestamp for timestamp in started_at):
        raise GateValidationError(
            "every final candidate run must start strictly after the candidate lock"
        )
    if len(set(raw_hashes)) != 3 or len(set(summary_hashes)) != 3:
        raise GateValidationError("candidate final evidence hashes are not distinct")

    aggregate = _mapping(
        candidate_report.get("aggregate_metrics"), "candidate aggregate"
    )
    released = _mapping(
        aggregate.get("released_comparable"), "candidate released aggregate"
    )
    strict = _mapping(aggregate.get("strict"), "candidate strict aggregate")
    if set(released) != set(METRICS) or set(strict) != set(METRICS):
        raise GateValidationError("candidate aggregate metric branches are incomplete")
    for metric in METRICS:
        row = _mapping(released[metric], f"candidate aggregate {metric}")
        expected_mean = statistics.fmean(values[metric])
        expected_sd = _sample_sd(values[metric])
        _close(row.get("mean"), expected_mean, f"candidate aggregate mean {metric}")
        _close(row.get("sample_sd"), expected_sd, f"candidate aggregate SD {metric}")
        expected_values = [
            {"seed": seed, "value": value}
            for seed, value in zip(EXPECTED_SEEDS, values[metric], strict=True)
        ]
        if row.get("values_by_seed") != expected_values:
            raise GateValidationError(
                f"candidate aggregate seed values differ for {metric}"
            )
    consistency = _mapping(
        candidate_report.get("environment_consistency"),
        "candidate environment consistency",
    )
    for key in (
        "all_seed_signatures_equal",
        "all_launch_policies_equal",
        "idle_gpu_policy_verified",
        "distinct_raw_sample_csv_sha256",
    ):
        _required_true(consistency.get(key), f"candidate environment {key}")
    implementation = _mapping(
        candidate_report.get("implementation_inputs"),
        "candidate implementation inputs",
    )
    sampler_source = _mapping(
        implementation.get("sampler_source"), "candidate sampler source"
    )
    if sampler_source.get("sha256") != lock["sampler_source_sha256"]:
        raise GateValidationError(
            "candidate sampler source differs from candidate lock"
        )
    if canonical_json_sha256(implementation) != lock["implementation_inputs_sha256"]:
        raise GateValidationError("candidate implementation inputs differ from lock")
    metric_inputs = _mapping(
        candidate_report.get("metric_inputs"), "candidate metric inputs"
    )
    if canonical_json_sha256(metric_inputs) != lock["metric_inputs_sha256"]:
        raise GateValidationError("candidate metric inputs differ from candidate lock")
    runner_sha = _sha256(
        candidate_report.get("runner_sha256"), "candidate runner digest"
    )
    if runner_sha != lock["benchmark_runner_sha256"]:
        raise GateValidationError(
            "candidate benchmark runner differs from candidate lock"
        )
    return {
        "values": values,
        "means": {
            metric: statistics.fmean(series) for metric, series in values.items()
        },
        "sample_sds": {metric: _sample_sd(series) for metric, series in values.items()},
        "pooled_valid": pooled_valid,
        "pooled_requested": pooled_requested,
        "benchmark_revision": revisions.pop(),
        "started_at": started_at,
        "raw_hashes": raw_hashes,
        "summary_hashes": summary_hashes,
        "runner_sha256": runner_sha,
        "sampler_source_sha256": lock["sampler_source_sha256"],
        "inference_weights": inference_weights,
    }


def independently_rescore_candidate_runs(
    candidate_report: Mapping[str, Any],
    lock: Mapping[str, Any],
    *,
    worker_invoker: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Recompute every final seed from raw model text in isolated CPU workers."""

    if worker_invoker is None:
        from scripts.udlm.rescore_denovo_run import invoke_rescore_worker

        worker_invoker = invoke_rescore_worker
    checkpoint = _mapping(
        candidate_report.get("checkpoint"), "candidate rescore checkpoint"
    )
    config = _mapping(candidate_report.get("config"), "candidate rescore config")
    implementation_inputs = _mapping(
        candidate_report.get("implementation_inputs"),
        "candidate rescore implementation inputs",
    )
    metric_inputs = _mapping(
        candidate_report.get("metric_inputs"), "candidate rescore metric inputs"
    )
    if {
        key: checkpoint.get(key) for key in ("sha256", "size_bytes", "global_step")
    } != {
        key: lock["checkpoint"][key] for key in ("sha256", "size_bytes", "global_step")
    }:
        raise GateValidationError("candidate rescore checkpoint differs from lock")
    if (
        config.get("sha256") != lock["evaluation_config_sha256"]
        or config.get("sampling") != lock["sampling_config"]
        or config.get("sampling_sha256") != lock["sampling_sha256"]
        or canonical_json_sha256(implementation_inputs)
        != lock["implementation_inputs_sha256"]
        or canonical_json_sha256(metric_inputs) != lock["metric_inputs_sha256"]
    ):
        raise GateValidationError("candidate rescore inputs differ from lock")
    ema_source = _mapping(
        implementation_inputs.get("ema_source"),
        "candidate rescore EMA implementation input",
    )
    ema_source_sha256 = _sha256(
        ema_source.get("sha256"), "candidate rescore EMA source digest"
    )
    ordered = _ordered_seed_rows(
        candidate_report.get("seed_runs"), label="candidate rescore seed runs"
    )
    seed_results: list[dict[str, Any]] = []
    for expected_seed, run in zip(EXPECTED_SEEDS, ordered, strict=True):
        summary_path = Path(run.get("summary_path", ""))
        raw_path = Path(run.get("raw_samples_path", ""))
        if not summary_path.is_absolute() or not raw_path.is_absolute():
            raise GateValidationError(
                f"candidate seed {expected_seed} rescore paths must be absolute"
            )
        if summary_path.parent != raw_path.parent:
            raise GateValidationError(
                f"candidate seed {expected_seed} summary/raw files are not siblings"
            )
        git = _mapping(run.get("git"), f"candidate seed {expected_seed} Git evidence")
        try:
            result = worker_invoker(
                summary_path=summary_path,
                raw_samples_path=raw_path,
                allowed_root=REPOSITORY_ROOT,
                expected_summary_sha256=run["summary_sha256"],
                expected_raw_samples_sha256=run["raw_samples_sha256"],
                expected_seed=expected_seed,
                expected_sample_count=EXPECTED_SAMPLES_PER_SEED,
                expected_checkpoint_sha256=lock["checkpoint"]["sha256"],
                expected_config_sha256=lock["evaluation_config_sha256"],
                expected_source_revision=git["commit"],
                expected_runner_sha256=lock["benchmark_runner_sha256"],
                expected_sampler_source_sha256=lock["sampler_source_sha256"],
                expected_ema_source_sha256=ema_source_sha256,
                expected_implementation_inputs_sha256=lock[
                    "implementation_inputs_sha256"
                ],
                expected_metric_inputs_sha256=lock["metric_inputs_sha256"],
            )
        except (OSError, ValueError) as error:
            raise GateValidationError(
                f"candidate seed {expected_seed} independent rescore failed: {error}"
            ) from error
        result = _validate_rescore_worker_proof(
            result,
            label=f"candidate seed {expected_seed} independent rescore",
            expected_seed=expected_seed,
            expected_sample_count=EXPECTED_SAMPLES_PER_SEED,
            expected_summary_path=summary_path,
            expected_raw_path=raw_path,
            expected_summary_sha256=run["summary_sha256"],
            expected_raw_sha256=run["raw_samples_sha256"],
        )
        identity = _mapping(
            result.get("identity"), f"candidate seed {expected_seed} rescore identity"
        )
        identity_checkpoint = _mapping(
            identity.get("checkpoint"),
            f"candidate seed {expected_seed} rescore checkpoint",
        )
        identity_config = _mapping(
            identity.get("config"), f"candidate seed {expected_seed} rescore config"
        )
        identity_generation = _mapping(
            identity.get("generation"),
            f"candidate seed {expected_seed} rescore generation",
        )
        identity_source = _mapping(
            identity.get("source"), f"candidate seed {expected_seed} rescore source"
        )
        if {
            key: identity_checkpoint.get(key)
            for key in ("sha256", "size_bytes", "global_step")
        } != {
            key: lock["checkpoint"][key]
            for key in ("sha256", "size_bytes", "global_step")
        }:
            raise GateValidationError(
                f"candidate seed {expected_seed} rescored checkpoint differs from lock"
            )
        if (
            identity_config.get("sha256") != lock["evaluation_config_sha256"]
            or identity_config.get("sampling") != lock["sampling_config"]
            or identity_config.get("sampling_sha256") != lock["sampling_sha256"]
            or identity_generation.get("nfe") != EXPECTED_NFE
            or identity_generation.get("inference_weights") != lock["inference_weights"]
        ):
            raise GateValidationError(
                f"candidate seed {expected_seed} rescored inference identity differs"
            )
        if (
            identity_source.get("revision") != git["commit"]
            or identity_source.get("runner_sha256") != lock["benchmark_runner_sha256"]
            or identity_source.get("sampler_source_sha256")
            != lock["sampler_source_sha256"]
            or identity_source.get("ema_source_sha256") != ema_source_sha256
            or identity_source.get("implementation_inputs_sha256")
            != lock["implementation_inputs_sha256"]
            or identity_source.get("metric_inputs_sha256")
            != lock["metric_inputs_sha256"]
        ):
            raise GateValidationError(
                f"candidate seed {expected_seed} rescored source identity differs"
            )
        rescored_metrics = _mapping(
            result.get("metrics"), f"candidate seed {expected_seed} rescored metrics"
        )
        reported_metrics = _mapping(
            run.get("metrics"), f"candidate seed {expected_seed} reported metrics"
        )
        for branch_name in ("released_comparable", "strict"):
            rescored_branch = _mapping(
                rescored_metrics.get(branch_name),
                f"candidate seed {expected_seed} rescored {branch_name}",
            )
            reported_branch = _mapping(
                reported_metrics.get(branch_name),
                f"candidate seed {expected_seed} reported {branch_name}",
            )
            for metric in METRICS:
                label = (
                    f"candidate seed {expected_seed} rescored "
                    f"{branch_name}.{metric}"
                )
                rescored_value = rescored_branch.get(metric)
                reported_value = reported_branch.get(metric)
                if rescored_value is None or reported_value is None:
                    if rescored_value is not None or reported_value is not None:
                        raise GateValidationError(f"{label} null status differs")
                    if metric != "diversity":
                        raise GateValidationError(f"{label} may not be null")
                    continue
                _close(rescored_value, _finite(reported_value, label), label)
        if result.get("failure_counts") != run.get("failure_counts"):
            raise GateValidationError(
                f"candidate seed {expected_seed} rescored failure counts differ"
            )
        seed_results.append(
            {
                "seed": expected_seed,
                "summary_sha256": run["summary_sha256"],
                "raw_samples_sha256": run["raw_samples_sha256"],
                "worker_environment": result.get("worker_environment"),
                "all_21_fields_match": True,
                "both_metric_branches_match": True,
                "failure_counts_match": True,
            }
        )
    return {
        "status": "completed_exact_match",
        "seed_results": seed_results,
        "fresh_seed_specific_cpu_workers": True,
        "raw_model_text_redecoded": True,
        "qed_sa_and_diversity_recomputed": True,
    }


def validate_candidate_rescore_attestation(
    value: object, candidate_report: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind the public statistical decision to the preceding raw-text rescore."""

    attestation = _mapping(value, "candidate independent-rescore attestation")
    _exact_keys(
        attestation,
        {
            "status",
            "seed_results",
            "fresh_seed_specific_cpu_workers",
            "raw_model_text_redecoded",
            "qed_sa_and_diversity_recomputed",
        },
        "candidate independent-rescore attestation",
    )
    if attestation.get("status") != "completed_exact_match":
        raise GateValidationError("candidate independent rescore is not complete")
    for field in (
        "fresh_seed_specific_cpu_workers",
        "raw_model_text_redecoded",
        "qed_sa_and_diversity_recomputed",
    ):
        _required_true(attestation.get(field), f"candidate rescore {field}")
    report_rows = _ordered_seed_rows(
        candidate_report.get("seed_runs"), label="candidate report seed runs"
    )
    rescore_rows = _ordered_seed_rows(
        attestation.get("seed_results"), label="candidate rescore seed results"
    )
    normalized_rows: list[dict[str, Any]] = []
    for expected_seed, (report_row, rescore_row) in enumerate(
        zip(report_rows, rescore_rows, strict=True)
    ):
        _exact_keys(
            rescore_row,
            {
                "seed",
                "summary_sha256",
                "raw_samples_sha256",
                "worker_environment",
                "all_21_fields_match",
                "both_metric_branches_match",
                "failure_counts_match",
            },
            f"candidate rescore seed {expected_seed}",
        )
        if rescore_row.get("summary_sha256") != report_row.get(
            "summary_sha256"
        ) or rescore_row.get("raw_samples_sha256") != report_row.get(
            "raw_samples_sha256"
        ):
            raise GateValidationError(
                f"candidate rescore seed {expected_seed} artifact hashes differ"
            )
        for field in (
            "all_21_fields_match",
            "both_metric_branches_match",
            "failure_counts_match",
        ):
            _required_true(
                rescore_row.get(field),
                f"candidate rescore seed {expected_seed} {field}",
            )
        environment = _mapping(
            rescore_row.get("worker_environment"),
            f"candidate rescore seed {expected_seed} worker environment",
        )
        expected_environment = {
            "python_hash_seed": str(expected_seed),
            "device": "cpu",
            "cuda_visible_devices": "",
            "nvidia_visible_devices": "",
        }
        for field, expected in expected_environment.items():
            if environment.get(field) != expected:
                raise GateValidationError(
                    f"candidate rescore seed {expected_seed} worker environment differs"
                )
        normalized_rows.append(dict(rescore_row))
    return {**dict(attestation), "seed_results": normalized_rows}


def evaluate_candidate_report(
    candidate_report: Mapping[str, Any],
    baseline_manifest: Mapping[str, Any],
    baseline_rescore_attestation: Mapping[str, Any],
    protocol: Mapping[str, Any],
    candidate_lock: Mapping[str, Any],
    *,
    independent_candidate_rescore: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a deterministic registered decision from validated report data."""

    validate_protocol(protocol)
    baseline = validate_baseline_manifest(baseline_manifest)
    baseline_rescore = validate_baseline_rescore_attestation(
        baseline_rescore_attestation, baseline_manifest
    )
    lock = validate_candidate_lock(candidate_lock, protocol)
    candidate = _candidate_series(candidate_report, lock)
    candidate_rescore = validate_candidate_rescore_attestation(
        independent_candidate_rescore, candidate_report
    )

    point_protocol = protocol["point_estimate_gates"]
    baseline_means = baseline["means"]
    if point_protocol["validity"]["threshold"] != baseline_means["validity"]:
        raise GateValidationError("validity point threshold differs from baseline")
    if point_protocol["uniqueness"]["threshold"] != baseline_means["uniqueness"]:
        raise GateValidationError("uniqueness point threshold differs from baseline")
    if point_protocol["quality"]["threshold"] != baseline_means["quality"]:
        raise GateValidationError("quality point threshold differs from baseline")
    diversity_threshold = (
        baseline_means["diversity"] + point_protocol["diversity"]["margin"]
    )
    _close(
        point_protocol["diversity"]["threshold"],
        diversity_threshold,
        "diversity point threshold",
    )
    point_pass = {
        "validity": candidate["means"]["validity"] >= baseline_means["validity"],
        "uniqueness": candidate["means"]["uniqueness"] >= baseline_means["uniqueness"],
        "quality": candidate["means"]["quality"] > baseline_means["quality"],
        "diversity": candidate["means"]["diversity"] >= diversity_threshold,
    }

    uncertainty_protocol = protocol["uncertainty_gates"]
    confidence = uncertainty_protocol["confidence_level_one_sided"]
    validity_interval = newcombe_wilson_lower_difference(
        candidate["pooled_valid"],
        candidate["pooled_requested"],
        baseline["pooled_valid"],
        baseline["pooled_requested"],
        z=uncertainty_protocol["normal_quantile"],
    )
    intervals: dict[str, dict[str, Any]] = {"validity": validity_interval}
    for metric in ("uniqueness", "quality", "diversity"):
        intervals[metric] = welch_lower_difference(
            candidate["values"][metric],
            baseline["values"][metric],
            confidence=confidence,
        )
    interval_pass: dict[str, bool] = {}
    for metric in METRICS:
        threshold = uncertainty_protocol[metric][
            "candidate_minus_baseline_lower_bound_strictly_greater_than"
        ]
        interval_pass[metric] = intervals[metric]["lower_bound"] > threshold
        intervals[metric]["registered_threshold_strictly_greater_than"] = threshold
        intervals[metric]["passed"] = interval_pass[metric]

    metrics: dict[str, Any] = {}
    for metric in METRICS:
        metrics[metric] = {
            "candidate_mean": candidate["means"][metric],
            "candidate_sample_sd": candidate["sample_sds"][metric],
            "baseline_mean": baseline_means[metric],
            "baseline_sample_sd": baseline["sample_sds"][metric],
            "candidate_minus_baseline": (
                candidate["means"][metric] - baseline_means[metric]
            ),
            "point_gate_passed": point_pass[metric],
            "uncertainty": intervals[metric],
        }
    all_point = all(point_pass.values())
    all_intervals = all(interval_pass.values())
    passed = all_point and all_intervals
    boundary = protocol["claim_boundaries"][
        "warm_start" if lock["startup_mode"] == "warm_start" else "scratch"
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "superiority_gate_passed": passed,
        "all_point_estimate_gates_passed": all_point,
        "all_uncertainty_gates_passed": all_intervals,
        "protocol": {
            "id": EXPECTED_PROTOCOL_ID,
            "relative_path": PROTOCOL_RELATIVE_PATH.as_posix(),
            "sha256": PROTOCOL_SHA256,
        },
        "baseline": {
            "relative_path": BASELINE_RELATIVE_PATH.as_posix(),
            "sha256": BASELINE_SHA256,
            "checkpoint_sha256": EXPECTED_BASELINE_CHECKPOINT_SHA256,
            "runner_sha256": baseline_manifest["source_aggregate"]["runner_sha256"],
            "rescore_attestation": baseline_rescore,
        },
        "candidate": {
            "candidate_id": lock["candidate_id"],
            "checkpoint_sha256": lock["checkpoint"]["sha256"],
            "checkpoint_global_step": lock["checkpoint"]["global_step"],
            "benchmark_revision": candidate["benchmark_revision"],
            "runner_sha256": candidate["runner_sha256"],
            "nfe": EXPECTED_NFE,
            "inference_weights": candidate["inference_weights"],
            "sampler_source_sha256": candidate["sampler_source_sha256"],
            "implementation_inputs_sha256": lock["implementation_inputs_sha256"],
            "metric_inputs_sha256": lock["metric_inputs_sha256"],
            "raw_samples_sha256_by_seed": [
                {"seed": seed, "sha256": digest}
                for seed, digest in zip(
                    EXPECTED_SEEDS, candidate["raw_hashes"], strict=True
                )
            ],
            "summary_sha256_by_seed": [
                {"seed": seed, "sha256": digest}
                for seed, digest in zip(
                    EXPECTED_SEEDS, candidate["summary_hashes"], strict=True
                )
            ],
        },
        "independent_candidate_rescore": candidate_rescore,
        "metrics": metrics,
        "claim": {
            "scope": lock["claim_scope"],
            "statement_if_passed": boundary if passed else None,
            "method_only_claim_supported": False,
            "inference_speed_claim_supported": False,
        },
        "limitations": [
            protocol["claim_boundaries"]["uncertainty_limitation"],
            protocol["claim_boundaries"]["method_only_requirement"],
            "The audited local MDLM comparator is not an exact paper reproduction.",
            "The baseline and candidate seed labels are not treated as paired runs.",
            (
                "Committed ledgers and locked output directories make disclosed "
                "selection auditable, but cannot prove that no undisclosed pilot or "
                "deleted fresh-directory retry ever existed."
            ),
        ],
    }


def validate_analysis_runtime(lock: Mapping[str, Any]) -> dict[str, Any]:
    gate_bytes = _repository_artifact_bytes(
        Path("scripts/udlm/superiority_gate.py"), label="superiority gate source"
    )
    report_bytes = _repository_artifact_bytes(
        Path("scripts/exps/denovo/report.py"), label="de-novo report source"
    )
    rescore_bytes = _repository_artifact_bytes(
        DENOVO_RESCORE_RELATIVE_PATH,
        label="independent de-novo rescore source",
    )
    rescore_dependency_bytes = _repository_artifact_bytes(
        DENOVO_RESCORE_DEPENDENCY_RELATIVE_PATH,
        label="independent de-novo rescore dependency source",
    )
    benchmark_launcher_bytes = _repository_artifact_bytes(
        DENOVO_LAUNCHER_RELATIVE_PATH,
        label="de-novo benchmark launcher source",
    )
    pilot_evidence_writer_bytes = _repository_artifact_bytes(
        PILOT_EVIDENCE_WRITER_RELATIVE_PATH,
        label="pilot evidence writer source",
    )
    if _sha256_bytes(gate_bytes) != lock["gate_source_sha256"]:
        raise GateValidationError("runtime superiority-gate source differs from lock")
    if _sha256_bytes(report_bytes) != lock["report_source_sha256"]:
        raise GateValidationError("runtime de-novo report source differs from lock")
    if _sha256_bytes(rescore_bytes) != lock["rescore_source_sha256"]:
        raise GateValidationError(
            "runtime independent de-novo rescore source differs from lock"
        )
    if _sha256_bytes(rescore_dependency_bytes) != lock["rescore_dependency_sha256"]:
        raise GateValidationError(
            "runtime independent de-novo rescore dependency differs from lock"
        )
    if (
        _sha256_bytes(benchmark_launcher_bytes)
        != lock["benchmark_launcher_source_sha256"]
    ):
        raise GateValidationError(
            "runtime de-novo benchmark launcher source differs from lock"
        )
    if (
        _sha256_bytes(pilot_evidence_writer_bytes)
        != lock["pilot_evidence_writer_source_sha256"]
    ):
        raise GateValidationError(
            "runtime pilot evidence writer source differs from lock"
        )
    import scipy

    if scipy.__version__ != lock["scipy_version"]:
        raise GateValidationError("runtime SciPy version differs from candidate lock")
    return {
        "gate_source_sha256": lock["gate_source_sha256"],
        "report_source_sha256": lock["report_source_sha256"],
        "rescore_source_sha256": lock["rescore_source_sha256"],
        "rescore_dependency_sha256": lock["rescore_dependency_sha256"],
        "benchmark_launcher_source_sha256": lock["benchmark_launcher_source_sha256"],
        "pilot_evidence_writer_source_sha256": lock[
            "pilot_evidence_writer_source_sha256"
        ],
        "scipy_version": scipy.__version__,
        "sources_match_prelocked_bytes": True,
    }


def _atomic_write_json_exclusive(path: Path, value: object) -> None:
    root = REPOSITORY_ROOT.resolve(strict=True)
    absolute = Path(os.path.abspath(path))
    if absolute == root or root not in absolute.parents or absolute.suffix != ".json":
        raise GateValidationError("output must be an in-repository JSON file")
    try:
        resolved_parent = absolute.parent.resolve(strict=True)
    except OSError as error:
        raise GateValidationError(
            "output parent must already exist as an in-repository directory"
        ) from error
    if (
        resolved_parent != absolute.parent
        or (resolved_parent != root and root not in resolved_parent.parents)
        or not resolved_parent.is_dir()
    ):
        raise GateValidationError(
            "output parent must be a real in-repository directory without symlinks"
        )
    absolute = resolved_parent / absolute.name
    if os.path.lexists(absolute):
        raise FileExistsError(f"refusing to replace superiority decision: {absolute}")
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=absolute.parent, prefix=f".{absolute.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o644)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, absolute)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to replace superiority decision: {absolute}"
            ) from error
        temporary.unlink()
        directory_descriptor = os.open(absolute.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        required=True,
        help="Tree containing the exact completed final seed 0, 1, and 2 runs.",
    )
    parser.add_argument(
        "--candidate-lock",
        type=Path,
        required=True,
        help="Committed pre-final candidate-lock JSON, relative to the repository.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New in-repository JSON decision path; existing files are never replaced.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol, _protocol_bytes = load_pinned_json(
        PROTOCOL_RELATIVE_PATH, PROTOCOL_SHA256, label="superiority protocol"
    )
    baseline, _baseline_bytes = load_pinned_json(
        BASELINE_RELATIVE_PATH, BASELINE_SHA256, label="MDLM baseline manifest"
    )
    baseline_rescore, _baseline_rescore_bytes = load_pinned_json(
        BASELINE_RESCORE_RELATIVE_PATH,
        BASELINE_RESCORE_SHA256,
        label="MDLM baseline rescore attestation",
    )
    lock_relative = _relative_path(
        args.candidate_lock.as_posix(), "candidate lock path", suffix=".json"
    )
    lock_bytes = _repository_artifact_bytes(lock_relative, label="candidate lock")
    lock_json = _mapping(
        strict_json_loads(lock_bytes, label="candidate lock"), "candidate lock"
    )
    lock = validate_candidate_lock(lock_json, protocol)
    training_evidence = validate_training_evidence(lock)
    matched_panel_completion = validate_completed_matched_panel(lock, training_evidence)
    analysis_evidence = validate_analysis_runtime(lock)
    candidate_report = denovo_report.collect_report(args.runs_dir)
    candidate = _candidate_series(candidate_report, lock)
    independent_candidate_rescore = independently_rescore_candidate_runs(
        candidate_report, lock
    )
    clean_source = denovo_report.require_clean_pushed_source(
        candidate["benchmark_revision"]
    )
    firewall = validate_git_lock_firewall(
        benchmark_revision=candidate["benchmark_revision"],
        candidate_lock_path=lock_relative,
        candidate_lock_bytes=lock_bytes,
        lock=lock,
    )
    decision = evaluate_candidate_report(
        candidate_report,
        baseline,
        baseline_rescore,
        protocol,
        lock_json,
        independent_candidate_rescore=independent_candidate_rescore,
    )
    decision["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    decision["candidate_lock"] = firewall
    decision["training_evidence"] = training_evidence
    decision["matched_panel_completion"] = matched_panel_completion
    decision["analysis_evidence"] = analysis_evidence
    decision["analysis_evidence"]["clean_pushed_source"] = clean_source
    decision["candidate_runs_root"] = candidate_report["input_root"]
    _atomic_write_json_exclusive(args.output, decision)
    print(f"Superiority gate: {decision['status'].upper()}")
    print(f"Decision: {Path(args.output).resolve()}")
    return 0 if decision["superiority_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate and publish one schema-2 pilot-evidence envelope.

The collector is CPU-only.  It does not generate molecules or load a model.
For a completed pilot it validates one existing schema-7 de-novo run,
independently re-decodes and re-scores its raw model text in a fresh
seed-specific worker, and joins it to an exact successful schema-5 training
receipt.  For a failed pilot it validates the launcher's exact schema-1 failure
receipt and every live supporting artifact.  Both modes publish references
only, and existing output entries are never replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_SRC = REPOSITORY_ROOT / "src"
for _import_root in (REPOSITORY_SRC, REPOSITORY_ROOT):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))
PILOT_EVIDENCE_SCHEMA_VERSION = 2
BENCHMARK_SCHEMA_VERSION = 7
EXIT_RECEIPT_SCHEMA_VERSION = 5
PILOT_FAILURE_RECEIPT_SCHEMA_VERSION = 1
PILOT_FAILURE_RECEIPT_KIND = "pilot_failure"
FINAL_BENCHMARK_SAMPLES_PER_SEED = 1_000
REGISTERED_SELECTION_PILOT_SEEDS = (1_000, 1_001)
REGISTERED_SELECTION_SAMPLES_PER_SEED = 256
REGISTERED_SELECTION_NFE = 128
RAW_SAMPLE_FIELD_COUNT = 21
ATTEMPT_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,95}\Z")
CANDIDATE_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{2,95}\Z")
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
HEX_GIT_REVISION = re.compile(r"[0-9a-f]{40}\Z")


class PilotEvidenceError(ValueError):
    """Raised before publication when pilot evidence is not authoritative."""


ReportValidator = Callable[..., Mapping[str, Any]]
RescoreWorker = Callable[..., Mapping[str, Any]]
TrainingArtifactValidator = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True)
class StableArtifact:
    """Bytes and file identity retained by one race-resistant read."""

    path: Path
    payload: bytes | None
    sha256: str
    identity: tuple[int, ...]
    project_scope: bool = False

    def snapshot(self) -> dict[str, Any]:
        """Return the exact snapshot shape emitted by the schema-5 writer."""

        device, inode, mode, link_count, size, mtime_ns, ctime_ns = self.identity
        return {
            "path": str(self.path),
            "device": device,
            "inode": inode,
            "mode": mode,
            "link_count": link_count,
            "size_bytes": size,
            "mtime_ns": mtime_ns,
            "ctime_ns": ctime_ns,
            "sha256": self.sha256,
            "stable_regular_file_verified": True,
        }


_RECEIPT_FIELDS = {
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
_EXPECTED_CONTRACT_FIELDS = {
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
}
_COMPLETION_REQUIREMENT_FIELDS = {
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
}
_PIPELINE_COMPONENT_FIELDS = {
    "shell_exit_status",
    "succeeded",
    "possible_termination_signal",
    "shell_status_is_signal_compatible",
    "signal_provenance",
}
_SNAPSHOT_FIELDS = {
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
_FAILURE_RECEIPT_FIELDS = {
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
}


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PilotEvidenceError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise PilotEvidenceError(
            f"{label} fields differ: expected {sorted(expected)}, found {sorted(value)}"
        )


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PilotEvidenceError(f"{label} must be an integer >= {minimum}")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or HEX_SHA256.fullmatch(value) is None:
        raise PilotEvidenceError(f"{label} must be 64 lowercase hexadecimal digits")
    return value


def _git_revision(value: object, label: str) -> str:
    if not isinstance(value, str) or HEX_GIT_REVISION.fullmatch(value) is None:
        raise PilotEvidenceError(f"{label} must be 40 lowercase hexadecimal digits")
    return value


def _timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise PilotEvidenceError(f"{label} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise PilotEvidenceError(
            f"{label} must be an ISO-8601 UTC timestamp"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise PilotEvidenceError(f"{label} must carry an explicit UTC offset")
    return parsed


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PilotEvidenceError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise PilotEvidenceError(f"non-finite JSON constant is forbidden: {value}")


def strict_json_loads(payload: bytes, *, label: str) -> Any:
    """Decode strict UTF-8 JSON without duplicate keys or nonfinite constants."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PilotEvidenceError(f"{label} is not UTF-8") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except PilotEvidenceError:
        raise
    except json.JSONDecodeError as error:
        raise PilotEvidenceError(f"{label} is not valid JSON") from error


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _absolute_in_repository(path: Path, *, label: str) -> Path:
    root = REPOSITORY_ROOT.resolve(strict=True)
    candidate = path if path.is_absolute() else root / path
    candidate = Path(os.path.abspath(candidate))
    if candidate == root or root not in candidate.parents:
        raise PilotEvidenceError(f"{label} must remain inside the repository")
    return candidate


def _existing_directory(path: Path, *, label: str) -> Path:
    candidate = _absolute_in_repository(path, label=label)
    try:
        resolved = candidate.resolve(strict=True)
        mode = candidate.stat(follow_symlinks=False).st_mode
    except OSError as error:
        raise PilotEvidenceError(f"{label} is unavailable: {candidate}") from error
    if resolved != candidate or not stat.S_ISDIR(mode):
        raise PilotEvidenceError(f"{label} must be a real directory without symlinks")
    return candidate


def read_stable_regular_file(
    path: Path,
    *,
    label: str,
    capture_payload: bool = True,
    allow_empty: bool = False,
    project_scope: bool = False,
) -> StableArtifact:
    """Read a repository file while rejecting symlinks and replacement races."""

    candidate = (
        _absolute_project_checkpoint_path(str(path), label=label)
        if project_scope
        else _absolute_in_repository(path, label=label)
    )
    try:
        resolved = candidate.resolve(strict=True)
        path_before = candidate.stat(follow_symlinks=False)
    except OSError as error:
        raise PilotEvidenceError(f"{label} is unavailable: {candidate}") from error
    if resolved != candidate:
        raise PilotEvidenceError(f"{label} must not traverse a symlink")
    if not stat.S_ISREG(path_before.st_mode):
        raise PilotEvidenceError(f"{label} is not a regular file")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as error:
        raise PilotEvidenceError(f"cannot safely open {label}: {candidate}") from error
    try:
        before = os.fstat(descriptor)
        identity = (
            int(before.st_dev),
            int(before.st_ino),
            int(before.st_mode),
            int(before.st_nlink),
            int(before.st_size),
            int(before.st_mtime_ns),
            int(before.st_ctime_ns),
        )
        path_identity = (
            int(path_before.st_dev),
            int(path_before.st_ino),
            int(path_before.st_mode),
            int(path_before.st_nlink),
            int(path_before.st_size),
            int(path_before.st_mtime_ns),
            int(path_before.st_ctime_ns),
        )
        if not stat.S_ISREG(before.st_mode) or path_identity != identity:
            raise PilotEvidenceError(f"{label} changed before open")
        digest = hashlib.sha256()
        chunks: list[bytes] | None = [] if capture_payload else None
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    try:
        path_after = candidate.stat(follow_symlinks=False)
    except OSError as error:
        raise PilotEvidenceError(f"{label} disappeared while being read") from error
    for observed in (after, path_after):
        observed_identity = (
            int(observed.st_dev),
            int(observed.st_ino),
            int(observed.st_mode),
            int(observed.st_nlink),
            int(observed.st_size),
            int(observed.st_mtime_ns),
            int(observed.st_ctime_ns),
        )
        if observed_identity != identity:
            raise PilotEvidenceError(f"{label} changed while being read")
    if identity[4] < 1 and not allow_empty:
        raise PilotEvidenceError(f"{label} is empty")
    payload = None if chunks is None else b"".join(chunks)
    return StableArtifact(
        path=candidate,
        payload=payload,
        sha256=digest.hexdigest(),
        identity=identity,
        project_scope=project_scope,
    )


def _validate_attempt_id(value: str) -> str:
    if ATTEMPT_ID_PATTERN.fullmatch(value) is None:
        raise PilotEvidenceError("attempt-id is not a normalized 1-96 character ID")
    return value


def _validate_candidate_id(value: str) -> str:
    if CANDIDATE_ID_PATTERN.fullmatch(value) is None:
        raise PilotEvidenceError("candidate-id is not a normalized 3-96 character ID")
    return value


def _validate_run_directory(path: Path, *, attempt_id: str, pilot_seed: int) -> Path:
    run_dir = _existing_directory(path, label="pilot run directory")
    relative = run_dir.relative_to(REPOSITORY_ROOT.resolve(strict=True))
    if (
        len(relative.parts) < 3
        or relative.parts[0] != "output"
        or relative.parts[-2] != attempt_id
        or relative.parts[-1] != f"seed_{pilot_seed}"
    ):
        raise PilotEvidenceError(
            "pilot run directory must be exactly output/.../<attempt-id>/seed_<pilot-seed>"
        )
    return run_dir


def _validate_receipt_path(path: Path) -> Path:
    candidate = _absolute_in_repository(path, label="training exit receipt")
    relative = candidate.relative_to(REPOSITORY_ROOT.resolve(strict=True))
    if (
        len(relative.parts) != 4
        or relative.parts[:2] != ("output", "udlm")
        or relative.name != "pilot_exit_status.json"
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", relative.parts[2]) is None
    ):
        raise PilotEvidenceError(
            "training receipt must be output/udlm/<run-name>/pilot_exit_status.json"
        )
    return candidate


def _validate_output_path(path: Path, *, attempt_id: str, pilot_seed: int) -> Path:
    candidate = _absolute_in_repository(path, label="pilot evidence output")
    expected = (
        REPOSITORY_ROOT.resolve(strict=True)
        / "experiments"
        / "udlm"
        / "pilots"
        / attempt_id
        / f"seed_{pilot_seed}.json"
    )
    if candidate != expected:
        raise PilotEvidenceError(
            "output must be exactly experiments/udlm/pilots/<attempt-id>/"
            "seed_<pilot-seed>.json"
        )
    if os.path.lexists(candidate):
        raise FileExistsError(f"refusing to replace pilot evidence: {candidate}")
    return candidate


def _validate_pipeline(value: object) -> None:
    pipeline = _mapping(value, "training receipt pipeline")
    _exact_keys(
        pipeline,
        {"training", "tee", "pipefail_shell_exit_status"},
        "training receipt pipeline",
    )
    if pipeline["pipefail_shell_exit_status"] != 0:
        raise PilotEvidenceError("training receipt pipeline is not successful")
    expected = {
        "shell_exit_status": 0,
        "succeeded": True,
        "possible_termination_signal": None,
        "shell_status_is_signal_compatible": False,
        "signal_provenance": None,
    }
    for name in ("training", "tee"):
        component = _mapping(pipeline.get(name), f"training receipt pipeline {name}")
        _exact_keys(component, _PIPELINE_COMPONENT_FIELDS, f"pipeline {name}")
        if dict(component) != expected:
            raise PilotEvidenceError(
                f"training receipt pipeline {name} is not successful"
            )


def _validate_snapshot(value: object, *, label: str) -> Mapping[str, Any]:
    snapshot = _mapping(value, label)
    _exact_keys(snapshot, _SNAPSHOT_FIELDS, label)
    if snapshot.get("stable_regular_file_verified") is not True:
        raise PilotEvidenceError(f"{label} was not verified as a stable regular file")
    _sha256(snapshot.get("sha256"), f"{label} digest")
    _integer(snapshot.get("size_bytes"), f"{label} size", minimum=1)
    return snapshot


def _require_live_snapshot(
    value: object, artifact: StableArtifact, *, label: str
) -> Mapping[str, Any]:
    claim = _validate_snapshot(value, label=label)
    if any(claim.get(key) != observed for key, observed in artifact.snapshot().items()):
        raise PilotEvidenceError(f"{label} differs from the live stable file")
    return claim


def _production_training_artifact_validator(
    *,
    manifest: Mapping[str, Any],
    manifest_payload: bytes,
    summary: Mapping[str, Any],
    runtime: Mapping[str, Any],
    expected_contract: Mapping[str, Any],
    summary_path: Path,
    manifest_path: Path,
    checkpoint_path: Path,
) -> Mapping[str, Any]:
    """Run the schema-5 producer's semantic validators over retained bytes."""

    from scripts.udlm import write_pilot_exit_status as receipt_writer

    selected_gpu_uuids = list(expected_contract["selected_gpu_uuids"])
    parsed_manifest = receipt_writer._validate_launch_manifest_content(  # noqa: SLF001
        manifest_payload,
        expected_sha256=expected_contract["launch_manifest_sha256"],
        expected_selected_gpu_uuids=selected_gpu_uuids,
    )
    if parsed_manifest != dict(manifest):
        raise PilotEvidenceError("launch-manifest validator changed retained content")
    receipt_writer._require_exact_keys(  # noqa: SLF001
        parsed_manifest,
        receipt_writer._PILOT_LAUNCH_MANIFEST_KEYS,  # noqa: SLF001
        label="pilot launch manifest",
    )
    if parsed_manifest.get("launch_manifest_schema_version") != 2:
        raise PilotEvidenceError("pilot launch manifest is not schema 2")
    resolved_config = runtime.get("resolved_training_config")
    validated_bindings = receipt_writer.validate_training_summary(
        dict(summary),
        summary_path=summary_path,
        expected_schema_version=4,
        expected_source_revision=expected_contract["source_revision"],
        expected_config_sha256=expected_contract["resolved_training_config_sha256"],
        expected_argv_sha256=expected_contract["training_argv_sha256"],
        expected_launch_manifest_path=manifest_path,
        expected_launch_manifest_sha256=expected_contract["launch_manifest_sha256"],
        expected_selected_gpu_uuids=selected_gpu_uuids,
        expected_max_steps=expected_contract["max_steps"],
        expected_world_size=expected_contract["world_size"],
        expected_final_checkpoint_path=checkpoint_path,
        expected_initialization_checkpoint_sha256=expected_contract[
            "initialization_checkpoint_sha256"
        ],
        resolved_training_config=resolved_config,
        launch_manifest=parsed_manifest,
    )
    completion_contract = summary.get("completion_contract")
    if not isinstance(completion_contract, dict):
        raise PilotEvidenceError("training summary completion contract is missing")
    receipt_writer.validate_runtime_config(
        dict(runtime),
        expected_source_revision=expected_contract["source_revision"],
        expected_config_sha256=expected_contract["resolved_training_config_sha256"],
        expected_argv_sha256=expected_contract["training_argv_sha256"],
        expected_launch_manifest_path=manifest_path,
        expected_launch_manifest_sha256=expected_contract["launch_manifest_sha256"],
        expected_selected_gpu_uuids=selected_gpu_uuids,
        expected_completion_contract=completion_contract,
    )
    predecessor = receipt_writer._validated_predecessor_binding_at_receipt(  # noqa: SLF001
        parsed_manifest
    )
    return {
        "validated_bindings": validated_bindings,
        "predecessor_receipt_binding": predecessor,
    }


def _require_true_fields(
    value: Mapping[str, Any], fields: Sequence[str], *, label: str
) -> None:
    for field in fields:
        if value.get(field) is not True:
            raise PilotEvidenceError(f"{label}.{field} must be true")


def validate_successful_training_receipt(
    receipt: Mapping[str, Any],
    *,
    receipt_path: Path,
    structural: Mapping[str, Any],
    artifact_validator: TrainingArtifactValidator | None = None,
) -> tuple[datetime, tuple[StableArtifact, ...]]:
    """Validate the exact successful schema-5 shape and benchmark checkpoint join."""

    _exact_keys(receipt, _RECEIPT_FIELDS, "training exit receipt")
    if (
        receipt.get("schema_version") != EXIT_RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != "completed"
        or receipt.get("overall_status") != "completed"
        or receipt.get("process_exit_status") != 0
    ):
        raise PilotEvidenceError(
            "training exit receipt is not a successful schema-5 receipt"
        )
    recorded_at = _timestamp(receipt.get("recorded_at_utc"), "receipt recorded_at_utc")
    _validate_pipeline(receipt.get("pipeline"))

    checkpoint = _mapping(structural.get("checkpoint"), "benchmark checkpoint")
    checkpoint_sha = _sha256(checkpoint.get("sha256"), "benchmark checkpoint digest")
    checkpoint_size = _integer(
        checkpoint.get("size_bytes"), "benchmark checkpoint size", minimum=1
    )
    checkpoint_step = _integer(
        checkpoint.get("global_step"), "benchmark checkpoint step", minimum=1
    )
    checkpoint_path_value = checkpoint.get("path")
    if not isinstance(checkpoint_path_value, str) or not checkpoint_path_value:
        raise PilotEvidenceError("benchmark checkpoint path is missing")
    checkpoint_path = _absolute_project_checkpoint_path(
        checkpoint_path_value, label="training checkpoint"
    )

    expected = _mapping(receipt.get("expected_contract"), "receipt expected contract")
    _exact_keys(expected, _EXPECTED_CONTRACT_FIELDS, "receipt expected contract")
    if expected.get("training_summary_schema_version") != 4:
        raise PilotEvidenceError(
            "receipt expects an unsupported training-summary schema"
        )
    _git_revision(expected.get("source_revision"), "receipt source revision")
    for field in (
        "resolved_training_config_sha256",
        "training_argv_sha256",
        "launch_manifest_sha256",
        "training_job_lock_sha256",
    ):
        _sha256(expected.get(field), f"receipt {field}")
    _integer(expected.get("max_steps"), "receipt max_steps", minimum=1)
    world_size = _integer(expected.get("world_size"), "receipt world_size", minimum=1)
    if world_size not in (1, 2):
        raise PilotEvidenceError("receipt world_size must be one or two")
    initialization_sha = expected.get("initialization_checkpoint_sha256")
    if initialization_sha is not None:
        _sha256(initialization_sha, "receipt initialization checkpoint digest")
    selected = expected.get("selected_gpu_uuids")
    if (
        not isinstance(selected, list)
        or len(selected) != world_size
        or len(set(selected)) != len(selected)
        or any(
            not isinstance(item, str) or not item.startswith("GPU-")
            for item in selected
        )
    ):
        raise PilotEvidenceError("receipt selected GPU UUIDs are invalid")

    source = _mapping(receipt.get("source_at_receipt"), "receipt source evidence")
    expected_source = {
        "verified": True,
        "expected_revision": expected["source_revision"],
        "head": expected["source_revision"],
        "upstream": expected["source_revision"],
        "output_directory_excluded_from_cleanliness_check": True,
    }
    _exact_keys(source, set(expected_source), "receipt source evidence")
    if dict(source) != expected_source:
        raise PilotEvidenceError("receipt clean pushed-source evidence is invalid")

    training_run_dir = receipt_path.parent
    expected_summary_path = training_run_dir / "training_summary.json"
    expected_manifest_path = training_run_dir / "launch_manifest.json"
    expected_runtime_path = training_run_dir / "runtime_config.json"
    expected_lock_path = (
        REPOSITORY_ROOT.resolve(strict=True)
        / "output"
        / "udlm"
        / (".single_training_job.lock")
    )
    expected_path_bindings = {
        "training_summary_path": str(expected_summary_path),
        "launch_manifest_path": str(expected_manifest_path),
        "training_job_lock_path": str(expected_lock_path),
    }
    for field, path_value in expected_path_bindings.items():
        if expected.get(field) != path_value:
            raise PilotEvidenceError(
                f"receipt {field} does not use its exact producer path"
            )
    if expected.get("final_checkpoint_path") != str(checkpoint_path):
        raise PilotEvidenceError("receipt final checkpoint path differs from benchmark")

    summary_artifact = read_stable_regular_file(
        expected_summary_path, label="live training summary"
    )
    manifest_artifact = read_stable_regular_file(
        expected_manifest_path, label="live launch manifest"
    )
    runtime_artifact = read_stable_regular_file(
        expected_runtime_path, label="live runtime config"
    )
    checkpoint_artifact = read_stable_regular_file(
        checkpoint_path,
        label="live training checkpoint",
        capture_payload=False,
        project_scope=True,
    )
    if (
        summary_artifact.payload is None
        or manifest_artifact.payload is None
        or runtime_artifact.payload is None
    ):  # pragma: no cover - capture contract
        raise PilotEvidenceError("training JSON payload capture failed")
    summary_document = _mapping(
        strict_json_loads(summary_artifact.payload, label="live training summary"),
        "live training summary",
    )
    manifest_document = _mapping(
        strict_json_loads(manifest_artifact.payload, label="live launch manifest"),
        "live launch manifest",
    )
    runtime_document = _mapping(
        strict_json_loads(runtime_artifact.payload, label="live runtime config"),
        "live runtime config",
    )

    completion = _mapping(
        receipt.get("completion_requirements"), "receipt completion requirements"
    )
    _exact_keys(
        completion,
        _COMPLETION_REQUIREMENT_FIELDS,
        "receipt completion requirements",
    )
    _require_true_fields(
        completion,
        tuple(_COMPLETION_REQUIREMENT_FIELDS),
        label="receipt completion requirements",
    )

    training_summary = _mapping(
        receipt.get("training_summary"), "receipt training-summary evidence"
    )
    _exact_keys(
        training_summary,
        {
            "path",
            "present",
            "valid_and_launch_bound",
            "artifact",
            "validated_bindings",
            "validation_error",
        },
        "receipt training-summary evidence",
    )
    _require_true_fields(
        training_summary,
        ("present", "valid_and_launch_bound"),
        label="receipt training-summary evidence",
    )
    if training_summary.get("validation_error") is not None:
        raise PilotEvidenceError(
            "receipt training-summary validation recorded an error"
        )
    training_summary_snapshot = _require_live_snapshot(
        training_summary.get("artifact"),
        summary_artifact,
        label="receipt training-summary snapshot",
    )
    if training_summary.get("path") != str(
        expected_summary_path
    ) or training_summary_snapshot.get("path") != str(expected_summary_path):
        raise PilotEvidenceError("receipt training-summary path binding is invalid")
    if not isinstance(training_summary.get("validated_bindings"), Mapping):
        raise PilotEvidenceError(
            "receipt training-summary validated bindings are missing"
        )

    launch_manifest = _mapping(
        receipt.get("launch_manifest"), "receipt launch-manifest evidence"
    )
    _exact_keys(
        launch_manifest,
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
    _require_true_fields(
        launch_manifest,
        (
            "present",
            "matches_expected_raw_sha256",
            "selected_gpu_uuids_match_expected",
            "matches_training_summary_snapshot",
            "matches_runtime_config_snapshot",
            "valid_and_launch_bound",
        ),
        label="receipt launch-manifest evidence",
    )
    if launch_manifest.get("validation_error") is not None:
        raise PilotEvidenceError("receipt launch-manifest validation recorded an error")
    manifest_snapshot = _require_live_snapshot(
        launch_manifest.get("artifact"),
        manifest_artifact,
        label="receipt launch-manifest snapshot",
    )
    if (
        launch_manifest.get("path") != str(expected_manifest_path)
        or manifest_snapshot.get("path") != str(expected_manifest_path)
        or manifest_snapshot.get("sha256") != expected["launch_manifest_sha256"]
        or launch_manifest.get("expected_selected_gpu_uuids") != selected
        or launch_manifest.get("observed_selected_gpu_uuids") != selected
    ):
        raise PilotEvidenceError("receipt launch-manifest binding is invalid")

    runtime = _mapping(receipt.get("runtime_config"), "receipt runtime evidence")
    _exact_keys(
        runtime,
        {
            "path",
            "present",
            "matches_training_summary_snapshot",
            "semantic_validation_passed",
            "artifact",
        },
        "receipt runtime evidence",
    )
    _require_true_fields(
        runtime,
        ("present", "matches_training_summary_snapshot", "semantic_validation_passed"),
        label="receipt runtime evidence",
    )
    runtime_snapshot = _require_live_snapshot(
        runtime.get("artifact"),
        runtime_artifact,
        label="receipt runtime snapshot",
    )
    if runtime.get("path") != str(expected_runtime_path) or runtime_snapshot.get(
        "path"
    ) != str(expected_runtime_path):
        raise PilotEvidenceError("receipt runtime-config path binding is invalid")

    training_lock = _mapping(
        receipt.get("training_job_lock"), "receipt training-job-lock evidence"
    )
    _exact_keys(
        training_lock,
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
        "receipt training-job-lock evidence",
    )
    _require_true_fields(
        training_lock,
        (
            "present",
            "matches_expected_raw_sha256",
            "matches_launch_manifest_binding",
            "valid_and_launch_bound_before_receipt_publication",
            "release_result_not_claimed_inside_pre_release_receipt",
        ),
        label="receipt training-job-lock evidence",
    )
    if training_lock.get("validation_error") is not None:
        raise PilotEvidenceError(
            "receipt training-job-lock validation recorded an error"
        )
    lock_snapshot = _validate_snapshot(
        training_lock.get("artifact"), label="receipt training-job-lock snapshot"
    )
    if (
        training_lock.get("path") != str(expected_lock_path)
        or training_lock.get("expected_sha256") != expected["training_job_lock_sha256"]
        or lock_snapshot.get("path") != str(expected_lock_path)
        or lock_snapshot.get("sha256") != expected["training_job_lock_sha256"]
        or training_lock.get("release_policy")
        != "publish_receipt_then_unlink_only_same_stat_identity_and_sha256"
        or not isinstance(training_lock.get("record"), Mapping)
    ):
        raise PilotEvidenceError("receipt training-job-lock binding is invalid")

    checkpoint_evidence = _mapping(
        receipt.get("final_checkpoint"), "receipt final-checkpoint evidence"
    )
    _exact_keys(
        checkpoint_evidence,
        {"path", "present", "matches_training_summary_snapshot", "artifact"},
        "receipt final-checkpoint evidence",
    )
    _require_true_fields(
        checkpoint_evidence,
        ("present", "matches_training_summary_snapshot"),
        label="receipt final-checkpoint evidence",
    )
    checkpoint_snapshot = _require_live_snapshot(
        checkpoint_evidence.get("artifact"),
        checkpoint_artifact,
        label="receipt checkpoint snapshot",
    )
    if (
        checkpoint_evidence.get("path") != str(checkpoint_path)
        or checkpoint_snapshot.get("path") != str(checkpoint_path)
        or checkpoint_snapshot.get("sha256") != checkpoint_sha
        or checkpoint_snapshot.get("size_bytes") != checkpoint_size
        or expected.get("final_checkpoint_path") != str(checkpoint_path)
        or expected.get("max_steps") != checkpoint_step
    ):
        raise PilotEvidenceError(
            "benchmark checkpoint differs from the successful training receipt"
        )
    manifest_lock = _mapping(
        manifest_document.get("single_training_job_lock"),
        "launch-manifest training-job-lock binding",
    )
    lock_record = _mapping(manifest_lock.get("record"), "launch-manifest lock record")
    lock_record_bytes = (
        json.dumps(lock_record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    lock_record_sha256 = hashlib.sha256(lock_record_bytes).hexdigest()
    if (
        manifest_lock.get("path") != str(expected_lock_path)
        or manifest_lock.get("sha256") != expected["training_job_lock_sha256"]
        or training_lock.get("record") != lock_record
        or lock_record_sha256 != expected["training_job_lock_sha256"]
        or lock_snapshot.get("size_bytes") != len(lock_record_bytes)
        or lock_snapshot.get("link_count") != 1
    ):
        raise PilotEvidenceError(
            "recorded pre-release job-lock bytes differ from the launch binding"
        )

    summary_manifest_claim = _mapping(
        summary_document.get("launch_manifest"), "training-summary manifest snapshot"
    )
    summary_runtime_claim = _mapping(
        summary_document.get("runtime_config"), "training-summary runtime snapshot"
    )
    summary_checkpoint_claim = _mapping(
        summary_document.get("final_checkpoint"), "training-summary checkpoint snapshot"
    )
    runtime_manifest_claim = _mapping(
        runtime_document.get("launch_manifest"), "runtime manifest snapshot"
    )
    for label, claim, artifact in (
        ("training-summary manifest", summary_manifest_claim, manifest_artifact),
        ("training-summary runtime", summary_runtime_claim, runtime_artifact),
        ("training-summary checkpoint", summary_checkpoint_claim, checkpoint_artifact),
        ("runtime manifest", runtime_manifest_claim, manifest_artifact),
    ):
        if any(claim.get(key) != value for key, value in artifact.snapshot().items()):
            raise PilotEvidenceError(f"{label} snapshot differs from the live artifact")
    if (
        summary_runtime_claim.get("schema_version") != 2
        or summary_runtime_claim.get("record_sha256")
        != canonical_json_sha256(runtime_document)
        or summary_manifest_claim.get("selected_gpu_uuids") != selected
        or runtime_manifest_claim.get("selected_gpu_uuids") != selected
    ):
        raise PilotEvidenceError(
            "training summary/runtime cross-artifact joins are invalid"
        )

    manifest_created_at = _timestamp(
        manifest_document.get("created_at"), "training launch timestamp"
    )
    training_completed_at = _timestamp(
        summary_document.get("completed_at_utc"), "training completion timestamp"
    )
    if not manifest_created_at < training_completed_at < recorded_at:
        raise PilotEvidenceError(
            "training launch, summary, and receipt timestamps are not strictly ordered"
        )

    if artifact_validator is None:
        artifact_validator = _production_training_artifact_validator
    try:
        semantic = _mapping(
            artifact_validator(
                manifest=manifest_document,
                manifest_payload=manifest_artifact.payload,
                summary=summary_document,
                runtime=runtime_document,
                expected_contract=expected,
                summary_path=expected_summary_path,
                manifest_path=expected_manifest_path,
                checkpoint_path=checkpoint_path,
            ),
            "training artifact semantic validation",
        )
    except (OSError, ValueError) as error:
        raise PilotEvidenceError(
            f"training artifact semantic validation failed: {error}"
        ) from error
    validated_predecessor = semantic.get("predecessor_receipt_binding")
    if validated_predecessor != manifest_document.get(
        "predecessor_receipt_binding"
    ) or validated_predecessor != receipt.get("predecessor_receipt_binding"):
        raise PilotEvidenceError(
            "receipt predecessor binding differs from live recursive validation"
        )
    if semantic.get("validated_bindings") != training_summary.get("validated_bindings"):
        raise PilotEvidenceError(
            "receipt training-summary bindings differ from semantic validation"
        )
    return recorded_at, (
        summary_artifact,
        manifest_artifact,
        runtime_artifact,
        checkpoint_artifact,
    )


def _validate_structural_result(
    value: Mapping[str, Any],
    *,
    run_dir: Path,
    summary: StableArtifact,
    raw_samples: StableArtifact,
    pilot_seed: int,
) -> Mapping[str, Any]:
    structural = _mapping(value, "schema-7 report validation result")
    expected_paths = {
        "summary_path": summary.path,
        "raw_samples_path": raw_samples.path,
    }
    for field, expected in expected_paths.items():
        raw_path = structural.get(field)
        if not isinstance(raw_path, (str, os.PathLike)) or Path(raw_path) != expected:
            raise PilotEvidenceError(f"report validator returned a different {field}")
    if (
        structural.get("summary_sha256") != summary.sha256
        or structural.get("raw_samples_sha256") != raw_samples.sha256
        or structural.get("seed") != pilot_seed
        or structural.get("run_dir") != str(run_dir)
    ):
        raise PilotEvidenceError("report validator returned different run identities")
    for field in (
        "checkpoint",
        "config",
        "metrics",
        "failure_counts",
        "git",
        "implementation_inputs",
        "metric_inputs",
        "inference_weights",
    ):
        if field not in structural:
            raise PilotEvidenceError(f"report validator omitted {field}")
    _sha256(structural.get("runner_sha256"), "report runner digest")
    return structural


def _validate_rescore_result(
    value: Mapping[str, Any],
    *,
    structural: Mapping[str, Any],
    summary: StableArtifact,
    raw_samples: StableArtifact,
    pilot_seed: int,
    requested_samples: int,
) -> None:
    rescored = _mapping(value, "independent pilot rescore")
    if (
        rescored.get("status") != "exact_match"
        or rescored.get("seed") != pilot_seed
        or rescored.get("summary_sha256") != summary.sha256
        or rescored.get("raw_samples_sha256") != raw_samples.sha256
        or rescored.get("metrics") != structural["metrics"]
        or rescored.get("failure_counts") != structural["failure_counts"]
    ):
        raise PilotEvidenceError(
            "independent pilot rescore differs from report evidence"
        )
    rows = _mapping(rescored.get("row_comparison"), "rescore row comparison")
    recomputation = _mapping(
        rescored.get("independent_recomputation"), "rescore recomputation record"
    )
    if (
        rows.get("all_match") is not True
        or rows.get("field_count") != RAW_SAMPLE_FIELD_COUNT
        or recomputation.get("raw_input_field") != "raw_model_text"
        or recomputation.get("qed_recomputed") is not True
        or recomputation.get("sa_recomputed") is not True
        or recomputation.get("released_diversity_recomputed") is not True
        or recomputation.get("strict_branch_recomputed") is not True
        or recomputation.get("all_21_raw_fields_compared") is not True
    ):
        raise PilotEvidenceError("independent pilot rescore is incomplete")

    identity = _mapping(rescored.get("identity"), "rescore identity")
    expected_identity = {
        "seed": pilot_seed,
        "sample_count": requested_samples,
        "started_at_utc": structural["started_at_utc"],
        "completed_at_utc": structural["completed_at_utc"],
    }
    for field, expected in expected_identity.items():
        if identity.get(field) != expected:
            raise PilotEvidenceError(f"rescore identity {field} differs from report")
    structural_checkpoint = _mapping(structural["checkpoint"], "checkpoint")
    expected_checkpoint = {
        key: structural_checkpoint[key]
        for key in ("path", "sha256", "size_bytes", "global_step")
    }
    if identity.get("checkpoint") != expected_checkpoint:
        raise PilotEvidenceError("rescore checkpoint identity differs from report")


def _create_output_parent(path: Path) -> None:
    base = REPOSITORY_ROOT.resolve(strict=True) / "experiments" / "udlm"
    try:
        if base.resolve(strict=True) != base or not base.is_dir():
            raise PilotEvidenceError("experiments/udlm must be a real directory")
    except OSError as error:
        raise PilotEvidenceError("experiments/udlm is unavailable") from error
    for directory in (base / "pilots", path.parent):
        try:
            directory.mkdir(exist_ok=True)
        except OSError as error:
            raise PilotEvidenceError(
                f"cannot create output directory: {directory}"
            ) from error
        try:
            if directory.resolve(strict=True) != directory or not directory.is_dir():
                raise PilotEvidenceError(
                    f"pilot evidence output parent is unsafe: {directory}"
                )
        except OSError as error:
            raise PilotEvidenceError(
                f"pilot evidence output parent is unsafe: {directory}"
            ) from error


def atomic_write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    """Publish fsynced JSON atomically without replacing any directory entry."""

    _create_output_parent(path)
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to replace pilot evidence: {path}")
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o644)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to replace concurrently created pilot evidence: {path}"
            ) from error
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def revalidate_inputs(inputs: Sequence[StableArtifact]) -> None:
    """Verify retained artifact bytes and file identities have not changed."""

    for original in inputs:
        current = read_stable_regular_file(
            original.path,
            label=f"final recheck of {original.path.name}",
            capture_payload=original.payload is not None,
            allow_empty=original.identity[-3] == 0,
            project_scope=original.project_scope,
        )
        if current.sha256 != original.sha256 or current.identity != original.identity:
            raise PilotEvidenceError(
                f"input changed before publication: {original.path}"
            )


def collect_completed_pilot_evidence(
    *,
    attempt_id: str,
    candidate_id: str,
    pilot_seed: int,
    run_dir: Path,
    training_exit_receipt: Path,
    output: Path,
    report_validator: ReportValidator | None = None,
    rescore_worker: RescoreWorker | None = None,
    training_artifact_validator: TrainingArtifactValidator | None = None,
) -> dict[str, Any]:
    """Validate one completed pilot and exclusively publish its reference envelope."""

    attempt_id = _validate_attempt_id(attempt_id)
    candidate_id = _validate_candidate_id(candidate_id)
    pilot_seed = _integer(pilot_seed, "pilot seed", minimum=1_000)
    run_dir = _validate_run_directory(
        run_dir, attempt_id=attempt_id, pilot_seed=pilot_seed
    )
    output = _validate_output_path(output, attempt_id=attempt_id, pilot_seed=pilot_seed)
    receipt_path = _validate_receipt_path(training_exit_receipt)
    summary_path = run_dir / "summary.json"
    raw_samples_path = run_dir / "raw_samples.csv"
    summary_artifact = read_stable_regular_file(
        summary_path, label="schema-7 pilot summary"
    )
    raw_artifact = read_stable_regular_file(
        raw_samples_path, label="schema-7 pilot raw samples"
    )
    receipt_artifact = read_stable_regular_file(
        receipt_path, label="schema-5 training exit receipt"
    )

    if summary_artifact.payload is None or receipt_artifact.payload is None:
        raise PilotEvidenceError(
            "pilot JSON payload capture failed"
        )  # pragma: no cover
    summary_document = _mapping(
        strict_json_loads(summary_artifact.payload, label="schema-7 pilot summary"),
        "schema-7 pilot summary",
    )
    if summary_document.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise PilotEvidenceError("pilot summary is not schema 7")
    requested_samples = _integer(
        summary_document.get("num_samples"), "pilot requested samples", minimum=1
    )
    expected_tier = (
        "final" if requested_samples == FINAL_BENCHMARK_SAMPLES_PER_SEED else "pilot"
    )
    final_protocol_eligible = expected_tier == "final"

    if report_validator is None:
        from scripts.exps.denovo import report as denovo_report

        report_validator = denovo_report.validate_run_evidence
    try:
        raw_structural = report_validator(
            run_dir,
            pilot_seed,
            expected_samples=requested_samples,
            expected_tier=expected_tier,
            final_protocol_eligible=final_protocol_eligible,
        )
    except (OSError, ValueError) as error:
        raise PilotEvidenceError(
            f"schema-7 pilot validation failed: {error}"
        ) from error
    structural = _validate_structural_result(
        raw_structural,
        run_dir=run_dir,
        summary=summary_artifact,
        raw_samples=raw_artifact,
        pilot_seed=pilot_seed,
    )

    receipt = _mapping(
        strict_json_loads(receipt_artifact.payload, label="training exit receipt"),
        "training exit receipt",
    )
    receipt_time, training_inputs = validate_successful_training_receipt(
        receipt,
        receipt_path=receipt_path,
        structural=structural,
        artifact_validator=training_artifact_validator,
    )
    started_at = _timestamp(structural.get("started_at_utc"), "pilot start")
    completed_at = _timestamp(structural.get("completed_at_utc"), "pilot completion")
    if not receipt_time < started_at < completed_at:
        raise PilotEvidenceError(
            "training receipt must predate pilot start, which must predate completion"
        )

    checkpoint = _mapping(structural["checkpoint"], "benchmark checkpoint")
    config = _mapping(structural["config"], "benchmark config")
    source = _mapping(structural["git"], "benchmark Git provenance")
    implementation_inputs = _mapping(
        structural["implementation_inputs"], "implementation inputs"
    )
    metric_inputs = _mapping(structural["metric_inputs"], "metric inputs")
    sampler_input = _mapping(
        implementation_inputs.get("sampler_source"), "sampler implementation input"
    )
    ema_input = _mapping(
        implementation_inputs.get("ema_source"), "EMA implementation input"
    )

    if rescore_worker is None:
        from scripts.udlm import rescore_denovo_run

        rescore_worker = rescore_denovo_run.invoke_rescore_worker
    try:
        rescored = rescore_worker(
            summary_path=summary_path,
            raw_samples_path=raw_samples_path,
            allowed_root=REPOSITORY_ROOT.resolve(strict=True),
            expected_summary_sha256=summary_artifact.sha256,
            expected_raw_samples_sha256=raw_artifact.sha256,
            expected_seed=pilot_seed,
            expected_sample_count=requested_samples,
            expected_checkpoint_sha256=_sha256(
                checkpoint.get("sha256"), "benchmark checkpoint digest"
            ),
            expected_config_sha256=_sha256(
                config.get("sha256"), "benchmark config digest"
            ),
            expected_source_revision=_git_revision(
                source.get("commit"), "benchmark source revision"
            ),
            expected_runner_sha256=_sha256(
                structural.get("runner_sha256"), "benchmark runner digest"
            ),
            expected_sampler_source_sha256=_sha256(
                sampler_input.get("sha256"), "sampler source digest"
            ),
            expected_ema_source_sha256=_sha256(
                ema_input.get("sha256"), "EMA source digest"
            ),
            expected_implementation_inputs_sha256=canonical_json_sha256(
                implementation_inputs
            ),
            expected_metric_inputs_sha256=canonical_json_sha256(metric_inputs),
        )
    except (OSError, ValueError) as error:
        raise PilotEvidenceError(
            f"independent pilot rescore failed: {error}"
        ) from error
    _validate_rescore_result(
        rescored,
        structural=structural,
        summary=summary_artifact,
        raw_samples=raw_artifact,
        pilot_seed=pilot_seed,
        requested_samples=requested_samples,
    )

    root = REPOSITORY_ROOT.resolve(strict=True)
    envelope = {
        "schema_version": PILOT_EVIDENCE_SCHEMA_VERSION,
        "artifact_kind": "pilot_evaluation",
        "status": "completed",
        "attempt_id": attempt_id,
        "candidate_id": candidate_id,
        "pilot_seed": pilot_seed,
        "final_seed_results_included": False,
        "training_exit_receipt": {
            "relative_path": receipt_path.relative_to(root).as_posix(),
            "sha256": receipt_artifact.sha256,
            "schema_version": EXIT_RECEIPT_SCHEMA_VERSION,
        },
        "benchmark_artifacts": {
            "summary_json": {
                "relative_path": summary_path.relative_to(root).as_posix(),
                "sha256": summary_artifact.sha256,
                "schema_version": BENCHMARK_SCHEMA_VERSION,
            },
            "raw_samples_csv": {
                "relative_path": raw_samples_path.relative_to(root).as_posix(),
                "sha256": raw_artifact.sha256,
            },
        },
    }
    revalidate_inputs(
        (receipt_artifact, summary_artifact, raw_artifact, *training_inputs)
    )
    atomic_write_json_exclusive(output, envelope)
    return envelope


def _project_root() -> Path:
    repository_root = REPOSITORY_ROOT.resolve(strict=True)
    if repository_root.parent.name == "run_sources":
        return repository_root.parent.parent
    return repository_root


def _absolute_project_checkpoint_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PilotEvidenceError(f"{label} must be a normalized absolute path")
    path = Path(value)
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise PilotEvidenceError(f"{label} must be a normalized absolute path")
    project_root = _project_root()
    if path == project_root or project_root not in path.parents:
        raise PilotEvidenceError(f"{label} must remain inside the containing project")
    return path


def _absolute_repository_reference_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PilotEvidenceError(f"{label} must be a normalized absolute path")
    path = Path(value)
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise PilotEvidenceError(f"{label} must be a normalized absolute path")
    return _absolute_in_repository(path, label=label)


def _validate_failure_receipt_path(
    path: Path, *, attempt_id: str, pilot_seed: int
) -> Path:
    candidate = _absolute_in_repository(path, label="pilot failure receipt")
    relative = candidate.relative_to(REPOSITORY_ROOT.resolve(strict=True))
    if (
        len(relative.parts) < 4
        or relative.parts[0] != "output"
        or relative.parent.name != f"seed_{pilot_seed}"
        or relative.parent.parent.name != attempt_id
        or relative.name != "failure_receipt.json"
    ):
        raise PilotEvidenceError(
            "pilot failure receipt must be exactly output/.../<attempt-id>/"
            "seed_<pilot-seed>/failure_receipt.json"
        )
    return candidate


def _failure_supporting_artifact(
    value: object,
    *,
    label: str,
    expected_path: Path | None = None,
) -> StableArtifact:
    reference = _mapping(value, label)
    _exact_keys(reference, {"path", "sha256", "size_bytes"}, label)
    path = _absolute_repository_reference_path(
        reference.get("path"), label=f"{label} path"
    )
    if expected_path is not None and path != expected_path:
        raise PilotEvidenceError(f"{label} path differs from its producer path")
    expected_sha256 = _sha256(reference.get("sha256"), f"{label} digest")
    expected_size = _integer(reference.get("size_bytes"), f"{label} size", minimum=0)
    artifact = read_stable_regular_file(
        path, label=label, capture_payload=False, allow_empty=True
    )
    if artifact.sha256 != expected_sha256 or artifact.identity[-3] != expected_size:
        raise PilotEvidenceError(f"{label} bytes differ from the failure receipt")
    return artifact


def _validate_failed_pilot_config(
    value: object, *, pilot_mode: str
) -> tuple[Path, StableArtifact]:
    config = _mapping(value, "pilot failure receipt config")
    _exact_keys(
        config,
        {"path", "sha256", "sampling", "sampling_sha256"},
        "pilot failure receipt config",
    )
    config_path = _absolute_repository_reference_path(
        config.get("path"), label="pilot failure config path"
    )
    config_sha256 = _sha256(config.get("sha256"), "pilot failure config digest")
    sampling = _mapping(config.get("sampling"), "pilot failure receipt sampling config")
    sampling_sha256 = _sha256(
        config.get("sampling_sha256"), "pilot failure sampling digest"
    )
    try:
        observed_sampling_sha256 = canonical_json_sha256(sampling)
    except (TypeError, ValueError) as error:
        raise PilotEvidenceError(
            "pilot failure sampling config is not canonical JSON"
        ) from error
    if observed_sampling_sha256 != sampling_sha256:
        raise PilotEvidenceError("pilot failure sampling digest is not canonical")

    config_artifact = read_stable_regular_file(
        config_path, label="pilot failure config", capture_payload=True
    )
    if config_artifact.sha256 != config_sha256:
        raise PilotEvidenceError(
            "pilot failure config bytes differ from the failure receipt"
        )
    assert config_artifact.payload is not None
    try:
        import yaml

        source_config = yaml.safe_load(config_artifact.payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise PilotEvidenceError(
            "pilot failure config is not valid UTF-8 YAML"
        ) from error
    if not isinstance(source_config, Mapping):
        raise PilotEvidenceError("pilot failure config must contain a YAML object")
    try:
        canonical_json_sha256(source_config)
    except (TypeError, ValueError) as error:
        raise PilotEvidenceError(
            "pilot failure config must be fully JSON-serializable"
        ) from error
    try:
        from scripts.exps.denovo import benchmark as benchmark_runner

        normalized_sampling = benchmark_runner.validate_sampling_config(source_config)
    except (OSError, ValueError) as error:
        raise PilotEvidenceError(
            f"pilot failure config has invalid sampling parameters: {error}"
        ) from error
    if normalized_sampling != dict(sampling):
        raise PilotEvidenceError(
            "pilot failure sampling config differs from the referenced YAML"
        )
    if sampling.get("diffusion_type") != "udlm":
        raise PilotEvidenceError("pilot failure receipt must describe UDLM sampling")
    if (
        pilot_mode == "registered_selection"
        and sampling.get("num_steps") != REGISTERED_SELECTION_NFE
    ):
        raise PilotEvidenceError(
            "registered-selection failure receipt requires 128 NFE"
        )
    return config_path, config_artifact


def collect_failed_pilot_evidence(
    *,
    attempt_id: str,
    candidate_id: str,
    pilot_seed: int,
    failure_receipt: Path,
    output: Path,
) -> dict[str, Any]:
    """Validate one failed launcher run and publish its reference-only envelope."""

    attempt_id = _validate_attempt_id(attempt_id)
    candidate_id = _validate_candidate_id(candidate_id)
    pilot_seed = _integer(pilot_seed, "pilot seed", minimum=1_000)
    output = _validate_output_path(output, attempt_id=attempt_id, pilot_seed=pilot_seed)
    receipt_path = _validate_failure_receipt_path(
        failure_receipt, attempt_id=attempt_id, pilot_seed=pilot_seed
    )
    receipt_artifact = read_stable_regular_file(
        receipt_path, label="schema-1 pilot failure receipt", capture_payload=True
    )
    assert receipt_artifact.payload is not None
    receipt = _mapping(
        strict_json_loads(
            receipt_artifact.payload, label="schema-1 pilot failure receipt"
        ),
        "schema-1 pilot failure receipt",
    )
    _exact_keys(receipt, _FAILURE_RECEIPT_FIELDS, "pilot failure receipt")
    exact_bindings = {
        "schema_version": PILOT_FAILURE_RECEIPT_SCHEMA_VERSION,
        "artifact_kind": PILOT_FAILURE_RECEIPT_KIND,
        "status": "failed",
        "attempt_id": attempt_id,
        "candidate_id": candidate_id,
        "pilot_seed": pilot_seed,
    }
    for field, expected in exact_bindings.items():
        if receipt.get(field) != expected:
            raise PilotEvidenceError(
                f"pilot failure receipt {field} differs from its envelope"
            )

    pilot_mode = receipt.get("pilot_mode")
    requested_samples = _integer(
        receipt.get("requested_samples"), "pilot failure requested samples", minimum=1
    )
    if pilot_mode == "engineering":
        if requested_samples > 100:
            raise PilotEvidenceError(
                "engineering failure receipt requires 1..100 samples"
            )
    elif pilot_mode == "registered_selection":
        if (
            pilot_seed not in REGISTERED_SELECTION_PILOT_SEEDS
            or requested_samples != REGISTERED_SELECTION_SAMPLES_PER_SEED
        ):
            raise PilotEvidenceError(
                "registered-selection failure receipt has an invalid seed/sample count"
            )
    else:
        raise PilotEvidenceError("pilot failure receipt mode is invalid")

    started_at = _timestamp(
        receipt.get("started_at_utc"), "pilot failure receipt start"
    )
    failed_at = _timestamp(
        receipt.get("failed_at_utc"), "pilot failure receipt failure"
    )
    if failed_at < started_at:
        raise PilotEvidenceError("pilot failure receipt failure predates its start")
    stage = receipt.get("stage")
    process_exit_status = receipt.get("process_exit_status")
    if stage == "benchmark_child_process":
        if type(process_exit_status) is not int or process_exit_status == 0:
            raise PilotEvidenceError(
                "child-process failure receipt requires a nonzero exit status"
            )
    elif stage == "completion_validation":
        if process_exit_status is not None:
            raise PilotEvidenceError(
                "completion-validation failure receipt requires a null exit status"
            )
    else:
        raise PilotEvidenceError("pilot failure receipt stage is invalid")
    reason = receipt.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise PilotEvidenceError("pilot failure receipt reason must be nonempty")

    checkpoint = _mapping(receipt.get("checkpoint"), "pilot failure checkpoint")
    _exact_keys(
        checkpoint,
        {"path", "sha256", "size_bytes", "global_step"},
        "pilot failure checkpoint",
    )
    checkpoint_path = _absolute_project_checkpoint_path(
        checkpoint.get("path"), label="pilot failure checkpoint path"
    )
    checkpoint_sha256 = _sha256(
        checkpoint.get("sha256"), "pilot failure checkpoint digest"
    )
    checkpoint_size = _integer(
        checkpoint.get("size_bytes"), "pilot failure checkpoint size", minimum=1
    )
    checkpoint_step = _integer(
        checkpoint.get("global_step"), "pilot failure checkpoint step", minimum=0
    )
    checkpoint_artifact = read_stable_regular_file(
        checkpoint_path,
        label="pilot failure checkpoint",
        capture_payload=False,
        project_scope=True,
    )
    if (
        checkpoint_artifact.sha256 != checkpoint_sha256
        or checkpoint_artifact.identity[-3] != checkpoint_size
    ):
        raise PilotEvidenceError(
            "pilot failure checkpoint bytes differ from the failure receipt"
        )

    config_path, config_artifact = _validate_failed_pilot_config(
        receipt.get("config"), pilot_mode=pilot_mode
    )
    config = _mapping(receipt["config"], "pilot failure receipt config")
    config_sha256 = _sha256(config.get("sha256"), "pilot failure config digest")

    source_revision = _mapping(
        receipt.get("source_revision"), "pilot failure source revision"
    )
    _exact_keys(source_revision, {"head", "upstream"}, "pilot failure source revision")
    revision = _git_revision(
        source_revision.get("head"), "pilot failure source revision"
    )
    if source_revision.get("upstream") != revision:
        raise PilotEvidenceError("pilot failure source was not clean and pushed")

    command = receipt.get("command")
    if (
        not isinstance(command, list)
        or len(command) != 20
        or any(not isinstance(item, str) or not item for item in command)
    ):
        raise PilotEvidenceError(
            "pilot failure command must contain exactly ten argument pairs"
        )
    expected_command = [
        str(_project_root() / ".venv/bin/python"),
        str(REPOSITORY_ROOT.resolve(strict=True) / "scripts/exps/denovo/benchmark.py"),
        "--checkpoint",
        str(checkpoint_path),
        "--expected-checkpoint-sha256",
        checkpoint_sha256,
        "--expected-source-revision",
        revision,
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
        str(receipt_path.parent),
    ]
    if command != expected_command:
        raise PilotEvidenceError(
            "pilot failure command differs from its exact producer launch"
        )

    repository_root = REPOSITORY_ROOT.resolve(strict=True)
    launcher_artifact = _failure_supporting_artifact(
        receipt.get("launcher_source"),
        label="pilot failure launcher source",
        expected_path=repository_root / "scripts/exps/denovo/launch_benchmark.py",
    )
    log_artifact = _failure_supporting_artifact(
        receipt.get("log"), label="pilot failure log"
    )
    expected_log_name = (
        f"denovo_step{checkpoint_step}_{checkpoint_sha256[:12]}_seed{pilot_seed}.log"
    )
    if (
        log_artifact.path.name != expected_log_name
        or log_artifact.path.parent.name != attempt_id
        or log_artifact.path.parent == receipt_path.parent.parent
    ):
        raise PilotEvidenceError(
            "pilot failure log path is not keyed by a distinct attempt root"
        )

    partials = _mapping(
        receipt.get("partial_artifacts"), "pilot failure partial artifacts"
    )
    _exact_keys(
        partials,
        {"summary_json", "raw_samples_csv"},
        "pilot failure partial artifacts",
    )
    support_artifacts = [
        receipt_artifact,
        checkpoint_artifact,
        config_artifact,
        launcher_artifact,
        log_artifact,
    ]
    for name, filename in (
        ("summary_json", "summary.json"),
        ("raw_samples_csv", "raw_samples.csv"),
    ):
        reference = partials.get(name)
        if reference is None:
            continue
        support_artifacts.append(
            _failure_supporting_artifact(
                reference,
                label=f"pilot failure partial {name}",
                expected_path=receipt_path.parent / filename,
            )
        )

    envelope = {
        "schema_version": PILOT_EVIDENCE_SCHEMA_VERSION,
        "artifact_kind": "pilot_failure",
        "status": "failed",
        "attempt_id": attempt_id,
        "candidate_id": candidate_id,
        "pilot_seed": pilot_seed,
        "final_seed_results_included": False,
        "failure_receipt": {
            "relative_path": receipt_path.relative_to(repository_root).as_posix(),
            "sha256": receipt_artifact.sha256,
            "schema_version": PILOT_FAILURE_RECEIPT_SCHEMA_VERSION,
        },
    }
    revalidate_inputs(tuple(support_artifacts))
    atomic_write_json_exclusive(output, envelope)
    return envelope


def _canonical_seed(value: str) -> int:
    if re.fullmatch(r"0|[1-9][0-9]*", value) is None:
        raise argparse.ArgumentTypeError("pilot seed must be a canonical integer")
    parsed = int(value)
    if parsed < 1_000:
        raise argparse.ArgumentTypeError("pilot seed must be at least 1000")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outcome",
        required=True,
        choices=("completed", "failed"),
        help="Select completed-run validation or launcher-failure validation.",
    )
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--pilot-seed", required=True, type=_canonical_seed)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--training-exit-receipt", type=Path)
    parser.add_argument("--failure-receipt", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    common = {
        "attempt_id": args.attempt_id,
        "candidate_id": args.candidate_id,
        "pilot_seed": args.pilot_seed,
        "output": args.output,
    }
    if args.outcome == "completed":
        if args.run_dir is None or args.training_exit_receipt is None:
            parser.error(
                "--outcome completed requires --run-dir and " "--training-exit-receipt"
            )
        if args.failure_receipt is not None:
            parser.error("--failure-receipt is valid only with --outcome failed")
        from scripts.exps.denovo import report as denovo_report
        from scripts.udlm import rescore_denovo_run

        collect_completed_pilot_evidence(
            **common,
            run_dir=args.run_dir,
            training_exit_receipt=args.training_exit_receipt,
            report_validator=denovo_report.validate_run_evidence,
            rescore_worker=rescore_denovo_run.invoke_rescore_worker,
        )
    else:
        if args.failure_receipt is None:
            parser.error("--outcome failed requires --failure-receipt")
        if args.run_dir is not None or args.training_exit_receipt is not None:
            parser.error(
                "--run-dir and --training-exit-receipt are valid only with "
                "--outcome completed"
            )
        collect_failed_pilot_evidence(
            **common,
            failure_receipt=args.failure_receipt,
        )
    print(f"Pilot evidence: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

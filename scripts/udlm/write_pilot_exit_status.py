"""Publish the durable exit receipt for one detached UDLM pilot.

This helper runs after the ``train.py | tee`` pipeline.  It deliberately uses
only the Python standard library so that a partially broken training import
stack cannot prevent post-pipeline accounting.  A receipt is published once,
with a stable hash and strict validation of ``training_summary.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXIT_STATUS_SCHEMA_VERSION = 1
INCOMPLETE_EXIT_STATUS = 97


def _canonical_integer(value: str, *, label: str, minimum: int, maximum: int) -> int:
    if not re.fullmatch(r"0|[1-9][0-9]*", value):
        raise argparse.ArgumentTypeError(f"{label} must be a canonical integer")
    parsed = int(value)
    if not minimum <= parsed <= maximum:
        raise argparse.ArgumentTypeError(
            f"{label} must be between {minimum} and {maximum}"
        )
    return parsed


def _shell_status(value: str) -> int:
    return _canonical_integer(
        value,
        label="pipeline shell exit status",
        minimum=0,
        maximum=255,
    )


def _positive_integer(value: str) -> int:
    return _canonical_integer(
        value,
        label="positive integer",
        minimum=1,
        maximum=2**63 - 1,
    )


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number: {value}")
    return parsed


def strict_json_loads(payload: bytes) -> object:
    """Decode UTF-8 JSON while rejecting duplicate keys and all non-finite values."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("training summary is not valid UTF-8") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_float,
        )
    except json.JSONDecodeError as error:
        raise ValueError("training summary is not valid JSON") from error


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact_path(value: Path, *, suffix: str, label: str) -> Path:
    root = REPOSITORY_ROOT.resolve(strict=True)
    absolute = Path(os.path.abspath(os.fspath(value)))
    path = absolute.parent.resolve(strict=False) / absolute.name
    if path == root or root not in path.parents or path.suffix != suffix:
        raise ValueError(f"{label} must be an in-repository {suffix} file")
    return path


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def stable_file_snapshot(
    path: Path, *, capture_bytes: bool = False
) -> tuple[dict[str, object], bytes | None]:
    """Read and hash one regular file while rejecting replacement or mutation."""

    before_path = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before_path.st_mode):
        raise ValueError(f"pilot artifact is not a regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    payload = bytearray() if capture_bytes else None
    try:
        before_descriptor = os.fstat(descriptor)
        if not stat.S_ISREG(before_descriptor.st_mode) or _stat_identity(
            before_descriptor
        ) != _stat_identity(before_path):
            raise ValueError(f"pilot artifact changed before open: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if payload is not None:
                payload.extend(chunk)
        after_descriptor = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = path.stat(follow_symlinks=False)
    identities = {
        _stat_identity(value)
        for value in (
            before_path,
            before_descriptor,
            after_descriptor,
            after_path,
        )
    }
    if len(identities) != 1:
        raise ValueError(f"pilot artifact changed while being hashed: {path}")
    return (
        {
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
        },
        None if payload is None else bytes(payload),
    )


def _git_output(*arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def verify_clean_pushed_source(expected_revision: str) -> dict[str, object]:
    """Bind the receipt to the same clean pushed source, ignoring run output."""

    if not re.fullmatch(r"[0-9a-f]{40}", expected_revision):
        raise ValueError("expected source revision must be a full commit hash")
    status = subprocess.run(
        [
            "git",
            "-C",
            str(REPOSITORY_ROOT),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "-z",
            "--",
            ".",
            ":(exclude)output",
            ":(exclude)output/**",
        ],
        check=True,
        capture_output=True,
    )
    if status.stdout:
        raise RuntimeError("pilot source worktree is dirty outside output/")
    head = _git_output("rev-parse", "HEAD")
    upstream = _git_output("rev-parse", "@{upstream}")
    if head != expected_revision or upstream != expected_revision:
        raise RuntimeError(
            "pilot receipt source disagrees with the clean pushed launch revision"
        )
    return {
        "verified": True,
        "expected_revision": expected_revision,
        "head": head,
        "upstream": upstream,
        "output_directory_excluded_from_cleanliness_check": True,
    }


def _exact_integer(value: object, expected: int, *, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError(f"{label} does not equal the launch-pinned value")


def _exact_string(value: object, expected: str, *, label: str) -> None:
    if not isinstance(value, str) or value != expected:
        raise ValueError(f"{label} does not equal the launch-pinned value")


def _required_mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _required_true(mapping: dict[str, object], key: str, *, label: str) -> None:
    if mapping.get(key) is not True:
        raise ValueError(f"{label} must be true")


def _positive_integer_field(mapping: dict[str, object], key: str, *, label: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _sha256_field(mapping: dict[str, object], key: str, *, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal digits")
    return value


def _validate_snapshot_claim(
    value: object,
    *,
    expected_path: Path,
    label: str,
) -> dict[str, object]:
    snapshot = _required_mapping(value, label=label)
    _exact_string(snapshot.get("path"), str(expected_path), label=f"{label} path")
    _required_true(
        snapshot,
        "stable_regular_file_verified",
        label=f"{label} stable regular-file verification",
    )
    integer_minimums = {
        "device": 0,
        "inode": 1,
        "mode": 1,
        "link_count": 1,
        "size_bytes": 1,
        "mtime_ns": 0,
        "ctime_ns": 0,
    }
    for key, minimum in integer_minimums.items():
        observed = snapshot.get(key)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, int)
            or observed < minimum
        ):
            raise ValueError(f"{label} {key} is invalid")
    if not stat.S_ISREG(snapshot["mode"]):
        raise ValueError(f"{label} mode is not a regular file")
    _sha256_field(snapshot, "sha256", label=f"{label} SHA-256")
    return snapshot


def _require_snapshot_matches_claim(
    current: dict[str, object],
    path: Path,
    claim: object,
    *,
    label: str,
) -> dict[str, object]:
    claimed = _validate_snapshot_claim(claim, expected_path=path, label=label)
    fields = (
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
    )
    if any(claimed.get(field) != current[field] for field in fields):
        raise ValueError(f"{label} no longer matches its stable summary snapshot")
    return current


def _validate_finiteness_record(value: object, *, label: str) -> None:
    record = _required_mapping(value, label=label)
    _required_true(record, "all_finite", label=f"{label} all-finite flag")
    tensor_count = _positive_integer_field(
        record,
        "floating_tensor_count",
        label=f"{label} floating tensor count",
    )
    element_count = _positive_integer_field(
        record,
        "floating_element_count",
        label=f"{label} floating element count",
    )
    if element_count < tensor_count:
        raise ValueError(f"{label} has fewer elements than tensors")


def validate_runtime_config(
    runtime: object,
    *,
    expected_source_revision: str,
    expected_config_sha256: str,
    expected_argv_sha256: str,
    expected_completion_contract: dict[str, object],
) -> None:
    record = _required_mapping(runtime, label="runtime config record")
    _exact_integer(record.get("schema_version"), 1, label="runtime config schema")
    _exact_string(
        record.get("status"),
        "preflight_completed",
        label="runtime config status",
    )
    _exact_string(
        record.get("source_revision"),
        expected_source_revision,
        label="runtime config source revision",
    )
    source = _required_mapping(record.get("source"), label="runtime config source")
    for key in ("head", "upstream"):
        _exact_string(
            source.get(key),
            expected_source_revision,
            label=f"runtime config source {key}",
        )
    _exact_string(
        record.get("resolved_training_config_sha256"),
        expected_config_sha256,
        label="runtime config resolved-config digest",
    )
    resolved_config = record.get("resolved_training_config")
    if not isinstance(resolved_config, dict):
        raise ValueError("runtime resolved training config must be an object")
    _exact_string(
        canonical_json_sha256(resolved_config),
        expected_config_sha256,
        label="runtime resolved training config content digest",
    )
    _exact_string(
        record.get("training_argv_sha256"),
        expected_argv_sha256,
        label="runtime config training argv digest",
    )
    training_argv = record.get("training_argv")
    if (
        not isinstance(training_argv, list)
        or not training_argv
        or not all(isinstance(value, str) for value in training_argv)
    ):
        raise ValueError("runtime training argv must be a string array")
    _exact_string(
        canonical_json_sha256(training_argv),
        expected_argv_sha256,
        label="runtime training argv content digest",
    )
    if record.get("observed_training_argv") != training_argv:
        raise ValueError("runtime observed argv disagrees with its base training argv")
    if record.get("completion_contract") != expected_completion_contract:
        raise ValueError("runtime completion contract disagrees with training summary")
    python_environment = record.get("python_environment")
    if not isinstance(python_environment, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in python_environment.items()
    ):
        raise ValueError("runtime Python environment must be a string mapping")


def validate_training_summary(
    summary: object,
    *,
    summary_path: Path,
    expected_schema_version: int,
    expected_source_revision: str,
    expected_config_sha256: str,
    expected_argv_sha256: str,
    expected_max_steps: int,
    expected_world_size: int,
    expected_final_checkpoint_path: Path,
    expected_initialization_checkpoint_sha256: str | None,
) -> dict[str, object]:
    """Validate the completion fields that bind the summary to this launch."""

    if not isinstance(summary, dict):
        raise ValueError("training summary root must be a JSON object")
    _exact_integer(
        summary.get("schema_version"),
        expected_schema_version,
        label="training summary schema version",
    )
    _exact_string(summary.get("status"), "completed", label="training summary status")
    completed_at = summary.get("completed_at_utc")
    if not isinstance(completed_at, str):
        raise ValueError("training completion timestamp must be a string")
    try:
        parsed_completed_at = datetime.fromisoformat(completed_at)
    except ValueError as error:
        raise ValueError("training completion timestamp is not ISO-8601") from error
    if parsed_completed_at.tzinfo is None:
        raise ValueError("training completion timestamp must include a timezone")
    _exact_string(
        summary.get("source_revision"),
        expected_source_revision,
        label="training summary source revision",
    )
    _exact_string(
        summary.get("resolved_training_config_sha256"),
        expected_config_sha256,
        label="training summary config digest",
    )
    _exact_string(
        summary.get("training_argv_sha256"),
        expected_argv_sha256,
        label="training summary argv digest",
    )

    source = summary.get("source")
    if not isinstance(source, dict):
        raise ValueError("training summary source binding must be an object")
    _exact_string(
        source.get("head"),
        expected_source_revision,
        label="training summary source HEAD",
    )
    _exact_string(
        source.get("upstream"),
        expected_source_revision,
        label="training summary source upstream",
    )

    completion = summary.get("completion_contract")
    if not isinstance(completion, dict):
        raise ValueError("training summary completion contract must be an object")
    _exact_integer(
        completion.get("summary_schema_version"),
        expected_schema_version,
        label="completion summary schema version",
    )
    _exact_integer(
        completion.get("expected_max_steps"),
        expected_max_steps,
        label="completion max steps",
    )
    _exact_integer(
        completion.get("expected_world_size"),
        expected_world_size,
        label="completion world size",
    )
    _exact_string(
        completion.get("summary_path"),
        str(summary_path),
        label="completion summary path",
    )
    _exact_string(
        completion.get("final_checkpoint_path"),
        str(expected_final_checkpoint_path),
        label="completion final checkpoint path",
    )
    _required_true(
        completion,
        "fail_on_nonfinite_loss",
        label="completion fail-on-nonfinite-loss flag",
    )
    _required_true(
        completion,
        "backward_anomaly_detection",
        label="completion backward-anomaly flag",
    )

    runtime_path = summary_path.with_name("runtime_config.json")
    runtime_config = _validate_snapshot_claim(
        summary.get("runtime_config"),
        expected_path=runtime_path,
        label="runtime config evidence",
    )
    _exact_integer(
        runtime_config.get("schema_version"),
        1,
        label="runtime config schema version",
    )
    _sha256_field(
        runtime_config,
        "record_sha256",
        label="runtime config canonical record digest",
    )

    observed = summary.get("observed_training_state")
    if not isinstance(observed, dict):
        raise ValueError("observed training state must be an object")
    _exact_integer(observed.get("global_rank"), 0, label="observed global rank")
    _exact_integer(
        observed.get("global_step"),
        expected_max_steps,
        label="observed global step",
    )
    _exact_integer(
        observed.get("world_size"),
        expected_world_size,
        label="observed world size",
    )

    health = _required_mapping(
        summary.get("training_health"), label="training health evidence"
    )
    if not isinstance(health.get("scope"), str) or not health["scope"]:
        raise ValueError("training health scope must be a nonempty string")
    for key, label in (
        ("all_losses_finite", "all-losses-finite flag"),
        ("all_observed_gradients_finite", "all-gradients-finite flag"),
        (
            "every_optimizer_step_had_a_nonzero_gradient",
            "nonzero-gradient-per-step flag",
        ),
    ):
        _required_true(health, key, label=label)
    loss_checks = _positive_integer_field(
        health, "loss_checks", label="training loss-check count"
    )
    optimizer_step_checks = _positive_integer_field(
        health,
        "optimizer_step_checks",
        label="optimizer-step check count",
    )
    gradient_tensors = _positive_integer_field(
        health,
        "gradient_tensor_observations",
        label="gradient tensor observation count",
    )
    gradient_elements = _positive_integer_field(
        health,
        "gradient_element_observations",
        label="gradient element observation count",
    )
    if loss_checks < expected_max_steps:
        raise ValueError("training loss-check count is below completed steps")
    if optimizer_step_checks != expected_max_steps:
        raise ValueError("optimizer-step check count disagrees with completed steps")
    if gradient_tensors < optimizer_step_checks or gradient_elements < gradient_tensors:
        raise ValueError("training gradient observation counts are inconsistent")

    final_checkpoint = _validate_snapshot_claim(
        summary.get("final_checkpoint"),
        expected_path=expected_final_checkpoint_path,
        label="final checkpoint evidence",
    )
    semantic = _required_mapping(
        final_checkpoint.get("semantic_audit"),
        label="final checkpoint semantic audit",
    )
    _required_true(
        semantic,
        "deserialized",
        label="checkpoint deserialization flag",
    )
    _exact_integer(
        semantic.get("global_step"),
        expected_max_steps,
        label="checkpoint global step",
    )
    for key, label in (
        ("raw_model", "serialized checkpoint raw model"),
        ("ema", "serialized checkpoint EMA"),
        ("optimizer", "serialized checkpoint optimizer"),
        ("all_checkpoint_tensors", "all serialized checkpoint tensors"),
    ):
        _validate_finiteness_record(semantic.get(key), label=label)
    _required_true(
        semantic,
        "udlm_process_identity_verified",
        label="checkpoint UDLM process identity flag",
    )
    live_model = _required_mapping(
        semantic.get("live_model_match"), label="checkpoint/live-model match"
    )
    _required_true(live_model, "exact_key_set", label="live-model exact-key flag")
    _required_true(
        live_model,
        "exact_tensor_values",
        label="live-model exact-value flag",
    )
    _positive_integer_field(
        live_model, "tensor_count", label="live-model match tensor count"
    )
    live_ema = _required_mapping(
        semantic.get("live_ema_match"), label="checkpoint/live-EMA match"
    )
    _required_true(
        live_ema,
        "exact_tensor_values",
        label="live-EMA exact-value flag",
    )
    _positive_integer_field(live_ema, "tensor_count", label="live-EMA tensor count")

    finiteness = _required_mapping(
        summary.get("tensor_finiteness"), label="live tensor finiteness evidence"
    )
    for key, label in (
        ("raw_model", "live raw model"),
        ("ema", "live EMA"),
    ):
        _validate_finiteness_record(finiteness.get(key), label=label)
    if finiteness["raw_model"] != semantic["raw_model"]:
        raise ValueError("live and serialized raw-model finiteness evidence disagree")
    if finiteness["ema"] != semantic["ema"]:
        raise ValueError("live and serialized EMA finiteness evidence disagree")

    startup = _required_mapping(summary.get("startup"), label="startup evidence")
    startup_mode = startup.get("mode")
    expected_startup_mode = (
        "warm_start"
        if expected_initialization_checkpoint_sha256 is not None
        else "scratch"
    )
    _exact_string(startup_mode, expected_startup_mode, label="startup mode")
    warm_start = startup.get("verified_mdlm_warm_start_report")
    if startup_mode == "warm_start":
        report = _required_mapping(warm_start, label="MDLM warm-start report")
        source_sha256 = _sha256_field(
            report, "source_sha256", label="warm-start source digest"
        )
        _exact_string(
            source_sha256,
            expected_initialization_checkpoint_sha256,
            label="warm-start launch-pinned source digest",
        )
        _exact_string(
            report.get("expected_source_sha256"),
            source_sha256,
            label="warm-start expected source digest",
        )
        _required_true(
            report,
            "byte_identity_verified_before_and_after_load",
            label="warm-start byte identity flag",
        )
        _exact_string(report.get("weights"), "ema", label="warm-start weights")
        _positive_integer_field(
            report, "source_size_bytes", label="warm-start source size"
        )
        _positive_integer_field(
            report, "parameter_tensors", label="warm-start parameter tensor count"
        )
        for key in ("source_path", "source_resolved_path"):
            if not isinstance(report.get(key), str) or not report[key]:
                raise ValueError(f"warm-start {key} is invalid")
    elif warm_start is not None:
        raise ValueError("non-warm-start summary contains an MDLM warm-start report")

    return {
        "schema_version": expected_schema_version,
        "source_revision": expected_source_revision,
        "resolved_training_config_sha256": expected_config_sha256,
        "training_argv_sha256": expected_argv_sha256,
        "observed_global_step": expected_max_steps,
        "observed_world_size": expected_world_size,
        "final_checkpoint_path": str(expected_final_checkpoint_path),
        "final_checkpoint_sha256": final_checkpoint["sha256"],
        "startup_mode": startup_mode,
    }


def _pipeline_component(exit_status: int) -> dict[str, object]:
    possible_signal = exit_status - 128 if 129 <= exit_status <= 192 else None
    return {
        "shell_exit_status": exit_status,
        "succeeded": exit_status == 0,
        "possible_termination_signal": possible_signal,
        "shell_status_is_signal_compatible": possible_signal is not None,
        "signal_provenance": "ambiguous_exit_or_signal" if possible_signal else None,
    }


def _atomic_write_json_exclusive(path: Path, value: object) -> None:
    """Publish one fsynced JSON receipt without replacing an existing path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to replace pilot exit receipt: {path}")
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o644)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to replace pilot exit receipt: {path}"
            ) from error
        temporary.unlink()
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_exit_receipt(args: argparse.Namespace) -> tuple[dict[str, object], int]:
    summary_path = _artifact_path(
        args.training_summary_path,
        suffix=".json",
        label="training summary path",
    )
    receipt_path = _artifact_path(
        args.receipt_path,
        suffix=".json",
        label="exit receipt path",
    )
    final_checkpoint_path = _artifact_path(
        args.expected_final_checkpoint_path,
        suffix=".ckpt",
        label="expected final checkpoint path",
    )
    if len({summary_path, receipt_path, final_checkpoint_path}) != 3:
        raise ValueError(
            "pilot summary, receipt, and checkpoint paths must be distinct"
        )
    if os.path.lexists(receipt_path):
        raise FileExistsError(f"refusing to replace pilot exit receipt: {receipt_path}")
    runtime_config_path = summary_path.with_name("runtime_config.json")

    summary_evidence: dict[str, object] = {
        "path": str(summary_path),
        "present": os.path.lexists(summary_path),
        "valid_and_launch_bound": False,
        "artifact": None,
        "validated_bindings": None,
        "validation_error": None,
    }
    runtime_evidence: dict[str, object] = {
        "path": str(runtime_config_path),
        "present": os.path.lexists(runtime_config_path),
        "matches_training_summary_snapshot": False,
        "semantic_validation_passed": False,
        "artifact": None,
    }
    checkpoint_evidence: dict[str, object] = {
        "path": str(final_checkpoint_path),
        "present": os.path.lexists(final_checkpoint_path),
        "matches_training_summary_snapshot": False,
        "artifact": None,
    }
    try:
        snapshot, payload = stable_file_snapshot(summary_path, capture_bytes=True)
        summary_evidence["artifact"] = snapshot
        if not payload:
            raise ValueError("training summary is empty")
        parsed = strict_json_loads(payload)
        bindings = validate_training_summary(
            parsed,
            summary_path=summary_path,
            expected_schema_version=args.expected_summary_schema_version,
            expected_source_revision=args.expected_source_revision,
            expected_config_sha256=args.expected_config_sha256,
            expected_argv_sha256=args.expected_argv_sha256,
            expected_max_steps=args.expected_max_steps,
            expected_world_size=args.expected_world_size,
            expected_final_checkpoint_path=final_checkpoint_path,
            expected_initialization_checkpoint_sha256=(
                args.expected_initialization_checkpoint_sha256
            ),
        )
        if not isinstance(parsed, dict):
            raise ValueError("training summary root must be a JSON object")
        current_runtime, runtime_payload = stable_file_snapshot(
            runtime_config_path, capture_bytes=True
        )
        runtime_evidence["artifact"] = current_runtime
        _require_snapshot_matches_claim(
            current_runtime,
            runtime_config_path,
            parsed.get("runtime_config"),
            label="runtime config evidence",
        )
        runtime_evidence["matches_training_summary_snapshot"] = True
        if not runtime_payload:
            raise ValueError("runtime config is empty")
        parsed_runtime = strict_json_loads(runtime_payload)
        expected_runtime_record_sha256 = parsed["runtime_config"]["record_sha256"]
        _exact_string(
            canonical_json_sha256(parsed_runtime),
            expected_runtime_record_sha256,
            label="runtime config canonical record digest",
        )
        validate_runtime_config(
            parsed_runtime,
            expected_source_revision=args.expected_source_revision,
            expected_config_sha256=args.expected_config_sha256,
            expected_argv_sha256=args.expected_argv_sha256,
            expected_completion_contract=parsed["completion_contract"],
        )
        runtime_evidence["semantic_validation_passed"] = True
        current_checkpoint, _unused = stable_file_snapshot(
            final_checkpoint_path, capture_bytes=False
        )
        checkpoint_evidence["artifact"] = current_checkpoint
        _require_snapshot_matches_claim(
            current_checkpoint,
            final_checkpoint_path,
            parsed.get("final_checkpoint"),
            label="final checkpoint evidence",
        )
        checkpoint_evidence["matches_training_summary_snapshot"] = True
        summary_evidence["validated_bindings"] = bindings
        summary_evidence["valid_and_launch_bound"] = True
    except (OSError, ValueError) as error:
        summary_evidence["validation_error"] = f"{type(error).__name__}: {error}"

    try:
        source = verify_clean_pushed_source(args.expected_source_revision)
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
        source = {
            "verified": False,
            "expected_revision": args.expected_source_revision,
            "error": f"{type(error).__name__}: {error}",
            "output_directory_excluded_from_cleanliness_check": True,
        }

    training = _pipeline_component(args.training_exit_status)
    tee = _pipeline_component(args.tee_exit_status)
    pipeline_status = (
        args.tee_exit_status if args.tee_exit_status != 0 else args.training_exit_status
    )
    summary_valid = summary_evidence["valid_and_launch_bound"] is True
    source_valid = source["verified"] is True
    completed = (
        args.training_exit_status == 0
        and args.tee_exit_status == 0
        and summary_valid
        and source_valid
    )
    if completed:
        process_exit_status = 0
    elif not summary_valid or not source_valid:
        process_exit_status = INCOMPLETE_EXIT_STATUS
    else:
        process_exit_status = pipeline_status
    overall_status = "completed" if completed else "failed"
    receipt = {
        "schema_version": EXIT_STATUS_SCHEMA_VERSION,
        "status": overall_status,
        "overall_status": overall_status,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "process_exit_status": process_exit_status,
        "expected_contract": {
            "training_summary_schema_version": args.expected_summary_schema_version,
            "source_revision": args.expected_source_revision,
            "resolved_training_config_sha256": args.expected_config_sha256,
            "training_argv_sha256": args.expected_argv_sha256,
            "max_steps": args.expected_max_steps,
            "world_size": args.expected_world_size,
            "training_summary_path": str(summary_path),
            "final_checkpoint_path": str(final_checkpoint_path),
            "initialization_checkpoint_sha256": (
                args.expected_initialization_checkpoint_sha256
            ),
        },
        "pipeline": {
            "training": training,
            "tee": tee,
            "pipefail_shell_exit_status": pipeline_status,
        },
        "source_at_receipt": source,
        "training_summary": summary_evidence,
        "runtime_config": runtime_evidence,
        "final_checkpoint": checkpoint_evidence,
        "completion_requirements": {
            "training_exit_zero": args.training_exit_status == 0,
            "tee_exit_zero": args.tee_exit_status == 0,
            "training_summary_valid_and_launch_bound": summary_valid,
            "runtime_config_matches_summary_and_launch": (
                runtime_evidence["matches_training_summary_snapshot"] is True
                and runtime_evidence["semantic_validation_passed"] is True
            ),
            "final_checkpoint_matches_training_summary": (
                checkpoint_evidence["matches_training_summary_snapshot"] is True
            ),
            "clean_pushed_source_still_matches_launch": source_valid,
            "all_must_hold": True,
        },
    }
    return receipt, process_exit_status


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-exit-status", type=_shell_status, required=True)
    parser.add_argument("--tee-exit-status", type=_shell_status, required=True)
    parser.add_argument("--training-summary-path", type=Path, required=True)
    parser.add_argument("--receipt-path", type=Path, required=True)
    parser.add_argument(
        "--expected-summary-schema-version", type=_positive_integer, required=True
    )
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-argv-sha256", required=True)
    parser.add_argument("--expected-max-steps", type=_positive_integer, required=True)
    parser.add_argument("--expected-world-size", type=_positive_integer, required=True)
    parser.add_argument("--expected-final-checkpoint-path", type=Path, required=True)
    parser.add_argument("--expected-initialization-checkpoint-sha256")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.expected_summary_schema_version != 1:
        raise ValueError("unsupported training summary schema version")
    if args.expected_world_size not in (1, 2):
        raise ValueError("expected world size must be 1 or 2")
    for label, digest in (
        ("expected config digest", args.expected_config_sha256),
        ("expected argv digest", args.expected_argv_sha256),
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"{label} must be 64 lowercase hexadecimal digits")
    if args.expected_initialization_checkpoint_sha256 is not None and not re.fullmatch(
        r"[0-9a-f]{64}", args.expected_initialization_checkpoint_sha256
    ):
        raise ValueError(
            "expected initialization checkpoint digest must be 64 lowercase "
            "hexadecimal digits"
        )
    receipt, process_exit_status = build_exit_receipt(args)
    receipt_path = _artifact_path(
        args.receipt_path,
        suffix=".json",
        label="exit receipt path",
    )
    _atomic_write_json_exclusive(receipt_path, receipt)
    return process_exit_status


if __name__ == "__main__":
    raise SystemExit(main())

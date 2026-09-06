"""Publish the durable exit receipt for one detached UDLM pilot.

This helper runs after the ``train.py | tee`` pipeline.  It deliberately uses
only the Python standard library so that a partially broken training import
stack cannot prevent post-pipeline accounting.  A receipt is published once,
with strict validation of ``training_summary.json`` and its launch evidence.
Only after publication does the helper release the exact unchanged global
training-job lock owned by the launch.
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
RUNTIME_CONFIG_SCHEMA_VERSION = 2
TRAINING_SUMMARY_SCHEMA_VERSION = 4
EXIT_STATUS_SCHEMA_VERSION = 4
INCOMPLETE_EXIT_STATUS = 97
HOSTED_STREAM_RANK_PARTITION_POLICY = (
    "huggingface_split_dataset_by_node_disjoint_rank_streams"
)


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


def strict_json_loads(payload: bytes, *, label: str = "training summary") -> object:
    """Decode UTF-8 JSON while rejecting duplicate keys and all non-finite values."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not valid UTF-8") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_float,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error


def _selected_gpu_uuids_json(value: str) -> list[str]:
    try:
        parsed = strict_json_loads(
            value.encode("utf-8"), label="selected GPU UUID contract"
        )
    except (UnicodeEncodeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if (
        not isinstance(parsed, list)
        or not parsed
        or any(
            not isinstance(uuid, str)
            or not uuid.startswith("GPU-")
            or len(uuid) <= len("GPU-")
            or "," in uuid
            for uuid in parsed
        )
        or len(set(parsed)) != len(parsed)
    ):
        raise argparse.ArgumentTypeError(
            "selected GPU UUIDs must be a nonempty JSON array of unique NVIDIA "
            "GPU UUID strings"
        )
    return parsed


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


def _training_job_lock_path(value: Path) -> Path:
    path = Path(os.path.abspath(os.fspath(value)))
    expected = (
        REPOSITORY_ROOT.resolve(strict=True)
        / "output"
        / "udlm"
        / (".single_training_job.lock")
    )
    if path != expected:
        raise ValueError(
            "training-job lock path must be the repository-wide reviewed pilot lock"
        )
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


def release_exact_training_job_lock(
    path: Path,
    *,
    expected_sha256: str,
    expected_snapshot: object,
) -> None:
    """Unlink only the unchanged lock certified before receipt publication."""

    path = _training_job_lock_path(path)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError(
            "expected training-job lock digest must be 64 lowercase hexadecimal digits"
        )
    current, _payload = stable_file_snapshot(path, capture_bytes=False)
    _require_snapshot_matches_claim(
        current,
        path,
        expected_snapshot,
        label="training-job lock evidence",
    )
    _exact_string(
        current.get("sha256"),
        expected_sha256,
        label="training-job lock raw SHA-256",
    )
    immediately_before_unlink = path.stat(follow_symlinks=False)
    if _stat_identity(immediately_before_unlink) != tuple(
        current[key]
        for key in (
            "device",
            "inode",
            "mode",
            "link_count",
            "size_bytes",
            "mtime_ns",
            "ctime_ns",
        )
    ):
        raise RuntimeError("training-job lock changed before exact release")
    os.unlink(path)
    directory_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


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


def _nonnegative_integer_field(
    mapping: dict[str, object], key: str, *, label: str
) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _require_exact_keys(
    mapping: dict[str, object], expected: set[str], *, label: str
) -> None:
    observed = set(mapping)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ValueError(
            f"{label} keys are invalid: missing={missing}, unexpected={unexpected}"
        )


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


def _validate_launch_manifest_content(
    payload: bytes,
    *,
    expected_sha256: str,
    expected_selected_gpu_uuids: list[str],
) -> dict[str, object]:
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    _exact_string(
        observed_sha256,
        expected_sha256,
        label="launch manifest raw SHA-256",
    )
    manifest = strict_json_loads(payload, label="launch manifest")
    manifest = _required_mapping(manifest, label="launch manifest")
    if manifest.get("cuda_visible_device_uuids") != expected_selected_gpu_uuids:
        raise ValueError(
            "launch manifest selected GPU UUIDs do not equal the launch-pinned value"
        )
    _exact_integer(
        manifest.get("user_requested_gpu_count"),
        len(expected_selected_gpu_uuids),
        label="launch manifest selected GPU count",
    )
    return manifest


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


def _conditioning_configuration(
    resolved_training_config: object,
) -> tuple[str, bool, int]:
    resolved = _required_mapping(
        resolved_training_config, label="resolved training configuration"
    )
    training = _required_mapping(
        resolved.get("training"), label="resolved training configuration.training"
    )
    udlm_value = training.get("udlm", {})
    udlm = _required_mapping(udlm_value, label="resolved UDLM configuration")
    variant = udlm.get("conditioning_variant", "additive")
    if variant not in {"additive", "film_adaln"}:
        raise ValueError("resolved conditioning variant is unsupported")
    reseed = training.get("reseed_after_model_initialization", False)
    if type(reseed) is not bool:
        raise ValueError("resolved post-initialization reseed flag must be boolean")
    seed = _nonnegative_integer_field(resolved, "seed", label="resolved training seed")
    if variant == "film_adaln" and reseed is not True:
        raise ValueError("FiLM screen requires post-initialization reseeding")
    return variant, reseed, seed


def _validate_gradient_contract(
    value: object, expected_sha256: object
) -> dict[str, object]:
    contract = _required_mapping(value, label="conditioning gradient contract")
    _require_exact_keys(
        contract,
        {
            "schema_version",
            "observation_point",
            "optimizer_checks",
            "first_positive_lr_optimizer_step",
            "timestep_mlp_required_optimizer_check",
            "groups",
        },
        label="conditioning gradient contract",
    )
    _exact_integer(
        contract.get("schema_version"), 1, label="conditioning gradient schema"
    )
    _exact_string(
        contract.get("observation_point"),
        "on_before_optimizer_step_global_rank_zero_after_gradient_accumulation",
        label="conditioning gradient observation point",
    )
    if contract.get("optimizer_checks") != [1, 2, 3]:
        raise ValueError("conditioning gradient optimizer checks are invalid")
    _exact_integer(
        contract.get("first_positive_lr_optimizer_step"),
        2,
        label="conditioning gradient first positive-LR step",
    )
    _exact_integer(
        contract.get("timestep_mlp_required_optimizer_check"),
        3,
        label="conditioning gradient timestep backward index",
    )
    groups = contract.get("groups")
    if not isinstance(groups, list) or len(groups) != 2:
        raise ValueError("conditioning gradient contract must have two groups")
    expected_groups = (
        ("film_modulation", "film"),
        ("timestep_mlp", "timestep_mlp"),
    )
    names: set[str] = set()
    for group, (group_id, kind) in zip(groups, expected_groups, strict=True):
        group = _required_mapping(group, label="conditioning gradient group")
        _require_exact_keys(
            group,
            {"group_id", "kind", "parameters"},
            label="conditioning gradient group",
        )
        _exact_string(group.get("group_id"), group_id, label="gradient group ID")
        _exact_string(group.get("kind"), kind, label="gradient group kind")
        parameters = group.get("parameters")
        if not isinstance(parameters, list) or not parameters:
            raise ValueError("conditioning gradient group must have parameters")
        for parameter in parameters:
            parameter = _required_mapping(
                parameter, label="conditioning parameter manifest"
            )
            _require_exact_keys(
                parameter,
                {"name", "shape"},
                label="conditioning parameter manifest",
            )
            name = parameter.get("name")
            shape = parameter.get("shape")
            if (
                not isinstance(name, str)
                or not name
                or name in names
                or not isinstance(shape, list)
                or not shape
                or any(
                    type(dimension) is not int or dimension <= 0 for dimension in shape
                )
            ):
                raise ValueError("conditioning parameter manifest is invalid")
            names.add(name)
    digest = canonical_json_sha256(contract)
    _exact_string(
        expected_sha256,
        digest,
        label="registered conditioning gradient contract digest",
    )
    return contract


def _validate_gradient_group_report(
    value: object,
    *,
    contract_group: dict[str, object],
    require_nonzero: bool,
) -> dict[str, object]:
    report = _required_mapping(value, label="conditioning gradient group report")
    _require_exact_keys(
        report,
        {
            "group_id",
            "ordered_parameter_manifest_sha256",
            "parameter_count",
            "gradient_element_count",
            "all_gradients_present",
            "all_gradients_finite",
            "all_parameter_gradients_nonzero",
        },
        label="conditioning gradient group report",
    )
    _exact_string(
        report.get("group_id"),
        contract_group["group_id"],
        label="conditioning gradient report group ID",
    )
    _exact_string(
        report.get("ordered_parameter_manifest_sha256"),
        canonical_json_sha256(contract_group["parameters"]),
        label="conditioning gradient parameter-manifest digest",
    )
    _exact_integer(
        report.get("parameter_count"),
        len(contract_group["parameters"]),
        label="conditioning gradient parameter count",
    )
    expected_elements = sum(
        math.prod(parameter["shape"]) for parameter in contract_group["parameters"]
    )
    _exact_integer(
        report.get("gradient_element_count"),
        expected_elements,
        label="conditioning gradient element count",
    )
    _required_true(report, "all_gradients_present", label="gradient-present flag")
    _required_true(report, "all_gradients_finite", label="gradient-finite flag")
    nonzero = report.get("all_parameter_gradients_nonzero")
    if type(nonzero) is not bool:
        raise ValueError("conditioning gradient nonzero flag must be boolean")
    if require_nonzero and nonzero is not True:
        raise ValueError("required conditioning gradients are zero")
    return report


def _validate_conditioning_gradient_audit(
    value: object,
    *,
    conditioning_variant: str,
    launch_manifest: object,
) -> dict[str, object] | None:
    if conditioning_variant != "film_adaln":
        if value is not None:
            raise ValueError("non-FiLM summary must have a null gradient audit")
        return None
    report = _required_mapping(value, label="conditioning gradient audit")
    _require_exact_keys(
        report,
        {
            "schema_version",
            "status",
            "observation_point",
            "registered_contract_sha256",
            "first_positive_lr_optimizer_step",
            "timestep_mlp_required_optimizer_check",
            "optimizer_checks",
        },
        label="conditioning gradient audit",
    )
    manifest = _required_mapping(launch_manifest, label="launch manifest")
    screen = _required_mapping(
        manifest.get("optimization_screen"), label="optimization-screen manifest"
    )
    contract = _validate_gradient_contract(
        screen.get("conditioning_gradient_contract"),
        screen.get("conditioning_gradient_contract_sha256"),
    )
    _exact_integer(report.get("schema_version"), 1, label="gradient audit schema")
    _exact_string(report.get("status"), "completed", label="gradient audit status")
    _exact_string(
        report.get("observation_point"),
        contract["observation_point"],
        label="gradient audit observation point",
    )
    _exact_string(
        report.get("registered_contract_sha256"),
        canonical_json_sha256(contract),
        label="gradient audit registered contract digest",
    )
    _exact_integer(
        report.get("first_positive_lr_optimizer_step"),
        2,
        label="gradient audit first positive-LR step",
    )
    _exact_integer(
        report.get("timestep_mlp_required_optimizer_check"),
        3,
        label="gradient audit timestep backward index",
    )
    checks = report.get("optimizer_checks")
    if not isinstance(checks, list) or len(checks) != 3:
        raise ValueError("conditioning gradient audit must have three checks")
    contract_groups = {group["group_id"]: group for group in contract["groups"]}
    for index, check in enumerate(checks, start=1):
        check = _required_mapping(check, label="conditioning gradient check")
        _require_exact_keys(
            check,
            {
                "optimizer_gradient_observation_index",
                "optimizer_step_index",
                "learning_rate_before_step",
                "film_groups",
                "timestep_mlp_groups",
            },
            label="conditioning gradient check",
        )
        _exact_integer(
            check.get("optimizer_gradient_observation_index"),
            index,
            label="optimizer gradient observation index",
        )
        _exact_integer(
            check.get("optimizer_step_index"), index, label="optimizer-step index"
        )
        learning_rate = check.get("learning_rate_before_step")
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or not math.isfinite(float(learning_rate))
            or float(learning_rate) < 0.0
            or (index == 1 and float(learning_rate) != 0.0)
            or (index in (2, 3) and float(learning_rate) <= 0.0)
        ):
            raise ValueError(
                "conditioning gradient learning-rate transition is invalid"
            )
        for key, group_id, required_index in (
            ("film_groups", "film_modulation", 1),
            ("timestep_mlp_groups", "timestep_mlp", 3),
        ):
            groups = check.get(key)
            if not isinstance(groups, list) or len(groups) != 1:
                raise ValueError("conditioning gradient check group list is invalid")
            _validate_gradient_group_report(
                groups[0],
                contract_group=contract_groups[group_id],
                require_nonzero=index == required_index,
            )
    return report


def _validate_screen_initialization_state_audit(
    value: object,
    *,
    conditioning_variant: str,
    training_seed: int,
    expected_checkpoint_sha256: str | None,
    expected_config_sha256: str,
    launch_manifest: object,
) -> dict[str, object] | None:
    """Validate the exact post-warm-start state bound to a screen attempt."""

    manifest = _required_mapping(launch_manifest, label="launch manifest")
    screen = manifest.get("optimization_screen")
    if screen is None:
        if value is not None:
            raise ValueError(
                "non-screen summary must have a null initialization-state audit"
            )
        return None
    _required_mapping(screen, label="optimization-screen manifest")
    if expected_checkpoint_sha256 is None:
        raise ValueError("optimization screen requires a warm-start checkpoint")
    audit = _required_mapping(
        value, label="optimization-screen initialization-state audit"
    )
    _require_exact_keys(
        audit,
        {
            "schema_version",
            "phase",
            "source_checkpoint_sha256",
            "resolved_training_config_sha256",
            "training_seed",
            "conditioning_variant",
            "common_backbone_tensor_count",
            "common_backbone_state_sha256",
            "full_initial_tensor_count",
            "full_initial_state_sha256",
        },
        label="optimization-screen initialization-state audit",
    )
    _exact_integer(audit.get("schema_version"), 1, label="state-audit schema")
    _exact_string(
        audit.get("phase"),
        "after_verified_mdlm_ema_warm_start_before_training_rng_reseed_and_optimizer_creation",
        label="state-audit phase",
    )
    _exact_string(
        audit.get("source_checkpoint_sha256"),
        expected_checkpoint_sha256,
        label="state-audit warm-start checkpoint digest",
    )
    _exact_string(
        audit.get("resolved_training_config_sha256"),
        expected_config_sha256,
        label="state-audit resolved-config digest",
    )
    _exact_integer(
        audit.get("training_seed"), training_seed, label="state-audit training seed"
    )
    _exact_string(
        audit.get("conditioning_variant"),
        conditioning_variant,
        label="state-audit conditioning variant",
    )
    common_count = _positive_integer_field(
        audit,
        "common_backbone_tensor_count",
        label="state-audit common-backbone tensor count",
    )
    full_count = _positive_integer_field(
        audit,
        "full_initial_tensor_count",
        label="state-audit full tensor count",
    )
    if full_count < common_count:
        raise ValueError("state-audit full tensor count is smaller than common state")
    _sha256_field(
        audit,
        "common_backbone_state_sha256",
        label="state-audit common-backbone digest",
    )
    _sha256_field(
        audit,
        "full_initial_state_sha256",
        label="state-audit full-state digest",
    )
    return audit


def validate_runtime_config(
    runtime: object,
    *,
    expected_source_revision: str,
    expected_config_sha256: str,
    expected_argv_sha256: str,
    expected_launch_manifest_path: Path,
    expected_launch_manifest_sha256: str,
    expected_selected_gpu_uuids: list[str],
    expected_completion_contract: dict[str, object],
) -> None:
    record = _required_mapping(runtime, label="runtime config record")
    _exact_integer(
        record.get("schema_version"),
        RUNTIME_CONFIG_SCHEMA_VERSION,
        label="runtime config schema",
    )
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
    launch_manifest_claim = _validate_snapshot_claim(
        record.get("launch_manifest"),
        expected_path=expected_launch_manifest_path,
        label="runtime launch manifest evidence",
    )
    _exact_string(
        launch_manifest_claim.get("sha256"),
        expected_launch_manifest_sha256,
        label="runtime launch manifest raw SHA-256",
    )
    if launch_manifest_claim.get("selected_gpu_uuids") != expected_selected_gpu_uuids:
        raise ValueError(
            "runtime launch manifest selected GPU UUIDs do not equal the "
            "launch-pinned value"
        )
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
    expected_launch_manifest_path: Path,
    expected_launch_manifest_sha256: str,
    expected_selected_gpu_uuids: list[str],
    expected_max_steps: int,
    expected_world_size: int,
    expected_final_checkpoint_path: Path,
    expected_initialization_checkpoint_sha256: str | None,
    resolved_training_config: object,
    launch_manifest: object,
) -> dict[str, object]:
    """Validate the completion fields that bind the summary to this launch."""

    if expected_schema_version != TRAINING_SUMMARY_SCHEMA_VERSION:
        raise ValueError(
            "unsupported training summary schema version "
            f"{expected_schema_version!r}; expected {TRAINING_SUMMARY_SCHEMA_VERSION}"
        )
    if not isinstance(summary, dict):
        raise ValueError("training summary root must be a JSON object")
    _require_exact_keys(
        summary,
        {
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
        },
        label="training summary",
    )
    conditioning_variant, reseed_after_initialization, configured_seed = (
        _conditioning_configuration(resolved_training_config)
    )
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

    summary_manifest_claim = _validate_snapshot_claim(
        summary.get("launch_manifest"),
        expected_path=expected_launch_manifest_path,
        label="training summary launch manifest evidence",
    )
    _exact_string(
        summary_manifest_claim.get("sha256"),
        expected_launch_manifest_sha256,
        label="training summary launch manifest raw SHA-256",
    )
    if summary_manifest_claim.get("selected_gpu_uuids") != expected_selected_gpu_uuids:
        raise ValueError(
            "training summary selected GPU UUIDs do not equal the launch-pinned value"
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
        RUNTIME_CONFIG_SCHEMA_VERSION,
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

    accounting = _required_mapping(
        summary.get("training_accounting"), label="training accounting"
    )
    _require_exact_keys(
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
        label="training accounting",
    )
    training_seed = _nonnegative_integer_field(
        accounting, "training_seed", label="accounting training seed"
    )
    optimizer_updates = _positive_integer_field(
        accounting, "optimizer_updates", label="accounting optimizer updates"
    )
    accounting_world_size = _positive_integer_field(
        accounting, "world_size", label="accounting world size"
    )
    micro_batch_size = _positive_integer_field(
        accounting,
        "micro_batch_size_per_rank",
        label="accounting micro-batch size per rank",
    )
    accumulation = _positive_integer_field(
        accounting,
        "accumulate_grad_batches",
        label="accounting gradient accumulation",
    )
    effective_global_examples = _positive_integer_field(
        accounting,
        "effective_global_examples_per_optimizer_step",
        label="accounting effective global examples per optimizer step",
    )
    total_requested_exposures = _positive_integer_field(
        accounting,
        "total_requested_example_exposures",
        label="accounting total requested example exposures",
    )
    _exact_integer(
        optimizer_updates,
        expected_max_steps,
        label="accounting optimizer updates",
    )
    _exact_integer(
        accounting_world_size,
        expected_world_size,
        label="accounting world size",
    )
    if effective_global_examples != (
        micro_batch_size * accounting_world_size * accumulation
    ):
        raise ValueError(
            "accounting effective global examples do not equal micro-batch per "
            "rank times world size times accumulation"
        )
    if total_requested_exposures != effective_global_examples * optimizer_updates:
        raise ValueError(
            "accounting total requested example exposures do not equal effective "
            "global examples times optimizer updates"
        )
    _exact_string(
        accounting.get("hosted_stream_rank_partition_policy"),
        HOSTED_STREAM_RANK_PARTITION_POLICY,
        label="accounting hosted-stream rank partition policy",
    )

    parameter_counts = _required_mapping(
        accounting.get("trainable_parameter_counts"),
        label="accounting trainable parameter counts",
    )
    expected_parameter_count_keys = {"base_backbone", "time_conditioner", "total"}
    if conditioning_variant == "film_adaln":
        expected_parameter_count_keys.add("film_modulation")
    _require_exact_keys(
        parameter_counts,
        expected_parameter_count_keys,
        label="accounting trainable parameter counts",
    )
    base_backbone = _positive_integer_field(
        parameter_counts,
        "base_backbone",
        label="accounting base-backbone trainable parameter count",
    )
    time_conditioner = _positive_integer_field(
        parameter_counts,
        "time_conditioner",
        label="accounting time-conditioner trainable parameter count",
    )
    film_modulation = 0
    if conditioning_variant == "film_adaln":
        film_modulation = _positive_integer_field(
            parameter_counts,
            "film_modulation",
            label="accounting FiLM trainable parameter count",
        )
    total_trainable = _positive_integer_field(
        parameter_counts,
        "total",
        label="accounting total trainable parameter count",
    )
    if total_trainable != base_backbone + time_conditioner + film_modulation:
        raise ValueError("accounting trainable parameter counts do not add up")

    resolved_config = _required_mapping(
        resolved_training_config, label="resolved training config for accounting"
    )
    if training_seed != configured_seed:
        raise ValueError("accounting training seed disagrees with resolved config")
    if resolved_config.get("data") != "safe":
        raise ValueError("accounting requires the hosted SAFE training stream")
    trainer_config = _required_mapping(
        resolved_config.get("trainer"), label="resolved trainer config"
    )
    loader_config = _required_mapping(
        resolved_config.get("loader"), label="resolved loader config"
    )
    configured_updates = _positive_integer_field(
        trainer_config, "max_steps", label="configured optimizer updates"
    )
    configured_devices = _positive_integer_field(
        trainer_config, "devices", label="configured device count"
    )
    configured_nodes = _positive_integer_field(
        trainer_config, "num_nodes", label="configured node count"
    )
    configured_accumulation = _positive_integer_field(
        trainer_config,
        "accumulate_grad_batches",
        label="configured gradient accumulation",
    )
    configured_micro_batch = _positive_integer_field(
        loader_config,
        "batch_size",
        label="configured micro-batch size per rank",
    )
    configured_global_batch = _positive_integer_field(
        loader_config,
        "global_batch_size",
        label="configured global batch size",
    )
    if configured_updates != optimizer_updates:
        raise ValueError("accounting optimizer updates disagree with resolved config")
    if configured_nodes != 1:
        raise ValueError("accounting hosted-stream policy requires one configured node")
    if configured_devices * configured_nodes != accounting_world_size:
        raise ValueError("accounting world size disagrees with resolved config")
    if configured_micro_batch != micro_batch_size:
        raise ValueError("accounting micro-batch size disagrees with resolved config")
    if configured_accumulation != accumulation:
        raise ValueError("accounting accumulation disagrees with resolved config")
    if configured_global_batch != effective_global_examples:
        raise ValueError(
            "accounting effective global examples disagree with resolved global batch"
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
    if optimizer_step_checks != optimizer_updates:
        raise ValueError("optimizer-step checks disagree with training accounting")
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
    live_ema_tensor_count = _positive_integer_field(
        live_ema, "tensor_count", label="live-EMA tensor count"
    )

    ema_metadata = _required_mapping(
        semantic.get("ema_metadata"), label="checkpoint EMA metadata"
    )
    _require_exact_keys(
        ema_metadata,
        {"shadow_parameter_count", "decay", "num_updates"},
        label="checkpoint EMA metadata",
    )
    shadow_parameter_count = _positive_integer_field(
        ema_metadata,
        "shadow_parameter_count",
        label="checkpoint EMA shadow-parameter count",
    )
    if shadow_parameter_count != live_ema_tensor_count:
        raise ValueError("checkpoint EMA metadata count disagrees with live EMA")
    serialized_ema = _required_mapping(
        semantic.get("ema"), label="serialized checkpoint EMA"
    )
    if serialized_ema.get("floating_tensor_count") != shadow_parameter_count:
        raise ValueError("checkpoint EMA metadata count disagrees with serialized EMA")
    decay = ema_metadata.get("decay")
    if isinstance(decay, bool) or not isinstance(decay, (int, float)):
        raise ValueError("checkpoint EMA decay must be a real number")
    decay = float(decay)
    if not math.isfinite(decay) or not 0.0 < decay < 1.0:
        raise ValueError("checkpoint EMA decay must be finite and in (0, 1)")
    _exact_integer(
        ema_metadata.get("num_updates"),
        expected_max_steps,
        label="checkpoint EMA update count",
    )
    training_config = _required_mapping(
        resolved_config.get("training"), label="resolved training configuration"
    )
    configured_decay = training_config.get("ema")
    if (
        isinstance(configured_decay, bool)
        or not isinstance(configured_decay, (int, float))
        or not math.isfinite(float(configured_decay))
        or float(configured_decay) != decay
    ):
        raise ValueError("checkpoint EMA decay disagrees with resolved config")

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
    expected_startup_keys = {"mode", "verified_mdlm_warm_start_report"}
    if reseed_after_initialization:
        expected_startup_keys.add("training_rng_policy")
    _require_exact_keys(startup, expected_startup_keys, label="startup evidence")
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

    if reseed_after_initialization:
        if startup_mode != "warm_start":
            raise ValueError("post-initialization reseeding requires MDLM warm-start")
        rng_policy = _required_mapping(
            startup.get("training_rng_policy"), label="training RNG policy"
        )
        expected_rng_policy = {
            "policy": "reseed_all_training_rng_streams_after_model_and_warm_start",
            "seed": configured_seed,
            "purpose": "isolate_training_randomness_from_architecture_constructor_draws",
            "applied_before_dataloader_and_trainer_construction": True,
        }
        if rng_policy != expected_rng_policy:
            raise ValueError("training RNG policy disagrees with the resolved config")

    conditioning_gradient_audit = _validate_conditioning_gradient_audit(
        summary.get("conditioning_gradient_audit"),
        conditioning_variant=conditioning_variant,
        launch_manifest=launch_manifest,
    )
    screen_initialization_state_audit = _validate_screen_initialization_state_audit(
        summary.get("screen_initialization_state_audit"),
        conditioning_variant=conditioning_variant,
        training_seed=configured_seed,
        expected_checkpoint_sha256=(expected_initialization_checkpoint_sha256),
        expected_config_sha256=expected_config_sha256,
        launch_manifest=launch_manifest,
    )
    if conditioning_variant == "film_adaln":
        screen = _required_mapping(
            _required_mapping(launch_manifest, label="launch manifest").get(
                "optimization_screen"
            ),
            label="optimization-screen manifest",
        )
        gradient_contract = _validate_gradient_contract(
            screen.get("conditioning_gradient_contract"),
            screen.get("conditioning_gradient_contract_sha256"),
        )
        contract_counts = {
            group["group_id"]: sum(
                math.prod(parameter["shape"]) for parameter in group["parameters"]
            )
            for group in gradient_contract["groups"]
        }
        if (
            film_modulation != contract_counts["film_modulation"]
            or time_conditioner != contract_counts["timestep_mlp"]
        ):
            raise ValueError(
                "conditioning contract parameter sizes disagree with training "
                "accounting"
            )

    return {
        "schema_version": expected_schema_version,
        "source_revision": expected_source_revision,
        "resolved_training_config_sha256": expected_config_sha256,
        "training_argv_sha256": expected_argv_sha256,
        "launch_manifest_path": str(expected_launch_manifest_path),
        "launch_manifest_sha256": expected_launch_manifest_sha256,
        "selected_gpu_uuids": list(expected_selected_gpu_uuids),
        "observed_global_step": expected_max_steps,
        "observed_world_size": expected_world_size,
        "training_accounting": dict(accounting),
        "ema_metadata": dict(ema_metadata),
        "final_checkpoint_path": str(expected_final_checkpoint_path),
        "final_checkpoint_sha256": final_checkpoint["sha256"],
        "startup_mode": startup_mode,
        "conditioning_gradient_audit": conditioning_gradient_audit,
        "screen_initialization_state_audit": (screen_initialization_state_audit),
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
    launch_manifest_path = _artifact_path(
        args.expected_launch_manifest_path,
        suffix=".json",
        label="expected launch manifest path",
    )
    training_job_lock_path = _training_job_lock_path(args.training_job_lock_path)
    if launch_manifest_path != summary_path.with_name("launch_manifest.json"):
        raise ValueError(
            "launch manifest must be launch_manifest.json beside the training summary"
        )
    if (
        len({summary_path, receipt_path, final_checkpoint_path, launch_manifest_path})
        != 4
    ):
        raise ValueError(
            "pilot manifest, summary, receipt, and checkpoint paths must be distinct"
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
    manifest_evidence: dict[str, object] = {
        "path": str(launch_manifest_path),
        "present": os.path.lexists(launch_manifest_path),
        "matches_expected_raw_sha256": False,
        "selected_gpu_uuids_match_expected": False,
        "matches_training_summary_snapshot": False,
        "matches_runtime_config_snapshot": False,
        "valid_and_launch_bound": False,
        "expected_selected_gpu_uuids": list(args.expected_selected_gpu_uuids),
        "observed_selected_gpu_uuids": None,
        "artifact": None,
        "validation_error": None,
    }
    lock_evidence: dict[str, object] = {
        "path": str(training_job_lock_path),
        "present": os.path.lexists(training_job_lock_path),
        "expected_sha256": args.expected_training_job_lock_sha256,
        "matches_expected_raw_sha256": False,
        "matches_launch_manifest_binding": False,
        "valid_and_launch_bound_before_receipt_publication": False,
        "artifact": None,
        "record": None,
        "release_policy": (
            "publish_receipt_then_unlink_only_same_stat_identity_and_sha256"
        ),
        "release_result_not_claimed_inside_pre_release_receipt": True,
        "validation_error": None,
    }
    current_manifest = None
    parsed_manifest = None
    try:
        current_manifest, manifest_payload = stable_file_snapshot(
            launch_manifest_path, capture_bytes=True
        )
        manifest_evidence["artifact"] = current_manifest
        if not manifest_payload:
            raise ValueError("launch manifest is empty")
        parsed_manifest = _validate_launch_manifest_content(
            manifest_payload,
            expected_sha256=args.expected_launch_manifest_sha256,
            expected_selected_gpu_uuids=args.expected_selected_gpu_uuids,
        )
        manifest_evidence["observed_selected_gpu_uuids"] = list(
            parsed_manifest["cuda_visible_device_uuids"]
        )
        manifest_evidence["matches_expected_raw_sha256"] = True
        manifest_evidence["selected_gpu_uuids_match_expected"] = True
    except (OSError, ValueError) as error:
        manifest_evidence["validation_error"] = f"{type(error).__name__}: {error}"
    try:
        lock_snapshot, lock_payload = stable_file_snapshot(
            training_job_lock_path, capture_bytes=True
        )
        lock_evidence["artifact"] = lock_snapshot
        if not lock_payload:
            raise ValueError("training-job lock is empty")
        _exact_string(
            lock_snapshot.get("sha256"),
            args.expected_training_job_lock_sha256,
            label="training-job lock raw SHA-256",
        )
        lock_evidence["matches_expected_raw_sha256"] = True
        lock_record = _required_mapping(
            strict_json_loads(lock_payload, label="training-job lock"),
            label="training-job lock record",
        )
        _exact_string(
            lock_record.get("status"), "held", label="training-job lock status"
        )
        if parsed_manifest is None:
            raise ValueError(
                "training-job lock cannot be bound because the launch manifest is invalid"
            )
        manifest_lock = _required_mapping(
            parsed_manifest.get("single_training_job_lock"),
            label="launch manifest training-job lock binding",
        )
        _exact_string(
            manifest_lock.get("path"),
            str(training_job_lock_path),
            label="launch manifest training-job lock path",
        )
        _exact_string(
            manifest_lock.get("sha256"),
            args.expected_training_job_lock_sha256,
            label="launch manifest training-job lock raw SHA-256",
        )
        if manifest_lock.get("record") != lock_record:
            raise ValueError(
                "launch manifest training-job lock record disagrees with lock bytes"
            )
        lock_evidence["record"] = lock_record
        lock_evidence["matches_launch_manifest_binding"] = True
        lock_evidence["valid_and_launch_bound_before_receipt_publication"] = True
    except (OSError, ValueError) as error:
        lock_evidence["validation_error"] = f"{type(error).__name__}: {error}"
    try:
        if (
            current_manifest is None
            or manifest_evidence["validation_error"] is not None
        ):
            raise ValueError(
                "launch manifest validation failed: "
                f"{manifest_evidence['validation_error']}"
            )
        snapshot, payload = stable_file_snapshot(summary_path, capture_bytes=True)
        summary_evidence["artifact"] = snapshot
        if not payload:
            raise ValueError("training summary is empty")
        parsed = strict_json_loads(payload)
        if not isinstance(parsed, dict):
            raise ValueError("training summary root must be a JSON object")
        _require_snapshot_matches_claim(
            current_manifest,
            launch_manifest_path,
            parsed.get("launch_manifest"),
            label="training summary launch manifest evidence",
        )
        manifest_evidence["matches_training_summary_snapshot"] = True
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
        parsed_runtime = strict_json_loads(runtime_payload, label="runtime config")
        runtime_mapping = _required_mapping(
            parsed_runtime, label="runtime config record"
        )
        _require_snapshot_matches_claim(
            current_manifest,
            launch_manifest_path,
            runtime_mapping.get("launch_manifest"),
            label="runtime launch manifest evidence",
        )
        manifest_evidence["matches_runtime_config_snapshot"] = True
        bindings = validate_training_summary(
            parsed,
            summary_path=summary_path,
            expected_schema_version=args.expected_summary_schema_version,
            expected_source_revision=args.expected_source_revision,
            expected_config_sha256=args.expected_config_sha256,
            expected_argv_sha256=args.expected_argv_sha256,
            expected_launch_manifest_path=launch_manifest_path,
            expected_launch_manifest_sha256=(args.expected_launch_manifest_sha256),
            expected_selected_gpu_uuids=args.expected_selected_gpu_uuids,
            expected_max_steps=args.expected_max_steps,
            expected_world_size=args.expected_world_size,
            expected_final_checkpoint_path=final_checkpoint_path,
            expected_initialization_checkpoint_sha256=(
                args.expected_initialization_checkpoint_sha256
            ),
            resolved_training_config=runtime_mapping.get("resolved_training_config"),
            launch_manifest=parsed_manifest,
        )
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
            expected_launch_manifest_path=launch_manifest_path,
            expected_launch_manifest_sha256=(args.expected_launch_manifest_sha256),
            expected_selected_gpu_uuids=args.expected_selected_gpu_uuids,
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
        final_manifest, final_manifest_payload = stable_file_snapshot(
            launch_manifest_path, capture_bytes=True
        )
        if final_manifest != current_manifest:
            raise ValueError("launch manifest changed during receipt validation")
        _validate_launch_manifest_content(
            final_manifest_payload,
            expected_sha256=args.expected_launch_manifest_sha256,
            expected_selected_gpu_uuids=args.expected_selected_gpu_uuids,
        )
        manifest_evidence["valid_and_launch_bound"] = True
        summary_evidence["validated_bindings"] = bindings
        summary_evidence["valid_and_launch_bound"] = True
    except (OSError, ValueError) as error:
        validation_error = f"{type(error).__name__}: {error}"
        summary_evidence["validation_error"] = validation_error
        if (
            manifest_evidence["valid_and_launch_bound"] is not True
            and manifest_evidence["validation_error"] is None
            and "launch manifest" in str(error)
        ):
            manifest_evidence["validation_error"] = validation_error

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
    manifest_valid = manifest_evidence["valid_and_launch_bound"] is True
    lock_valid = (
        lock_evidence["valid_and_launch_bound_before_receipt_publication"] is True
    )
    source_valid = source["verified"] is True
    completed = (
        args.training_exit_status == 0
        and args.tee_exit_status == 0
        and summary_valid
        and manifest_valid
        and lock_valid
        and source_valid
    )
    if completed:
        process_exit_status = 0
    elif not summary_valid or not manifest_valid or not lock_valid or not source_valid:
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
            "launch_manifest_path": str(launch_manifest_path),
            "launch_manifest_sha256": args.expected_launch_manifest_sha256,
            "selected_gpu_uuids": list(args.expected_selected_gpu_uuids),
            "training_job_lock_path": str(training_job_lock_path),
            "training_job_lock_sha256": args.expected_training_job_lock_sha256,
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
        "launch_manifest": manifest_evidence,
        "training_job_lock": lock_evidence,
        "training_summary": summary_evidence,
        "runtime_config": runtime_evidence,
        "final_checkpoint": checkpoint_evidence,
        "completion_requirements": {
            "training_exit_zero": args.training_exit_status == 0,
            "tee_exit_zero": args.tee_exit_status == 0,
            "training_summary_valid_and_launch_bound": summary_valid,
            "launch_manifest_matches_summary_runtime_and_launch": manifest_valid,
            "training_job_lock_valid_before_receipt_publication": lock_valid,
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
    parser.add_argument("--expected-launch-manifest-path", type=Path, required=True)
    parser.add_argument("--expected-launch-manifest-sha256", required=True)
    parser.add_argument(
        "--expected-selected-gpu-uuids-json",
        dest="expected_selected_gpu_uuids",
        type=_selected_gpu_uuids_json,
        required=True,
    )
    parser.add_argument("--training-job-lock-path", type=Path, required=True)
    parser.add_argument("--expected-training-job-lock-sha256", required=True)
    parser.add_argument("--expected-max-steps", type=_positive_integer, required=True)
    parser.add_argument("--expected-world-size", type=_positive_integer, required=True)
    parser.add_argument("--expected-final-checkpoint-path", type=Path, required=True)
    parser.add_argument("--expected-initialization-checkpoint-sha256")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.expected_summary_schema_version != TRAINING_SUMMARY_SCHEMA_VERSION:
        raise ValueError(
            "unsupported training summary schema version "
            f"{args.expected_summary_schema_version!r}; expected "
            f"{TRAINING_SUMMARY_SCHEMA_VERSION}"
        )
    if args.expected_world_size not in (1, 2):
        raise ValueError("expected world size must be 1 or 2")
    if len(args.expected_selected_gpu_uuids) != args.expected_world_size:
        raise ValueError(
            "expected selected GPU UUID count must equal expected world size"
        )
    for label, digest in (
        ("expected config digest", args.expected_config_sha256),
        ("expected argv digest", args.expected_argv_sha256),
        ("expected launch manifest digest", args.expected_launch_manifest_sha256),
        (
            "expected training-job lock digest",
            args.expected_training_job_lock_sha256,
        ),
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
    release_exact_training_job_lock(
        args.training_job_lock_path,
        expected_sha256=args.expected_training_job_lock_sha256,
        expected_snapshot=receipt["training_job_lock"]["artifact"],
    )
    return process_exit_status


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate and select the prospective UDLM optimization screens.

This module is deliberately standard-library-only.  It does not import the
training stack, inspect accelerators, or execute models.  A frozen registry
defines the two small engineering screens; immutable evidence binds the bytes
produced by each registered run.  Missing, failed, malformed, or unmatched
evidence produces an explicit ``incomplete`` decision with no winner.

The registry itself is intentionally not shipped by this module.  Once the
screen implementations and exact resolved configurations exist, an operator
must freeze the registry before either scheduler arm is trained and pass both
its raw and canonical SHA-256 digests to this verifier.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
import re
import stat
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPOSITORY_ROOT.parents[1]
REGISTRY_SCHEMA_VERSION = 2
EVIDENCE_SCHEMA_VERSION = 1
SELECTION_SCHEMA_VERSION = 1
DENOISING_REPORT_SCHEMA_VERSION = 1
EVALUATOR_REPORT_SCHEMA_VERSION = 4
GRADIENT_AUDIT_SCHEMA_VERSION = 1
GRADIENT_CONTRACT_SCHEMA_VERSION = 1
INITIALIZATION_AUDIT_SCHEMA_VERSION = 1
EXPECTED_GRADIENT_CONTRACT_PATH = (
    "experiments/udlm/protocols/film_gradient_contract_v1.json"
)
EXPECTED_GRADIENT_CONTRACT_SHA256 = (
    "b2a666a23351eb0882a179f7ae5d09fafd2188fee924313cdf60ee94888e7ac5"
)
EXPECTED_GRADIENT_CONTRACT_CANONICAL_SHA256 = (
    "ff45961276df75f445221fd1aa4629262d21fdb852bd9b226ad56fe2559315d5"
)
EXPECTED_INITIALIZATION_FIXTURE_PATH = (
    "experiments/udlm/protocols/conditioning_init_fixture_v1.json"
)
EXPECTED_INITIALIZATION_FIXTURE_SHA256 = (
    "a7069082bd9a5a7d76d7d52b324345f4a46ba2dea386b87411b1fcb8112cc989"
)
EXPECTED_INITIALIZATION_FIXTURE_CANONICAL_SHA256 = (
    "96ed170d2e3db2c68101dd07d603ff44db7f77a95c3b9578d6c378d1e1f85d1b"
)
EXPECTED_INITIALIZATION_FIXTURE_SIZE_BYTES = 643
EXPECTED_REGISTRY_ID = "genmol_udlm_scheduler_conditioning_screen_v2"
EXPECTED_REGISTRY_STATUS = "frozen_before_any_screen_training"
EXPECTED_CLAIM_SCOPE = "engineering_selection_only_not_superiority_or_causal_evidence"
EXPECTED_PANEL_PATH = "experiments/udlm/validation_panel/first_256.json"
EXPECTED_PANEL_SHA256 = (
    "e2493da4f3cb3217b48c78dc2901dc7524d7d90dc3959cffa87a1b6f8a9a7658"
)
EXPECTED_PANEL_TOKEN_IDS_SHA256 = (
    "a641f0335f0c155040fdce6dcb131af3dd50be364a29e54c7517968e6f3e2be2"
)
EXPECTED_FREQUENCY_PATH = "experiments/udlm/token_frequency/train_first_10000.json"
EXPECTED_FREQUENCY_SHA256 = (
    "088c78e75611f3cc42c4011e1da6f65a377e673b9cba07a28b126b0fc62f06ed"
)
EXPECTED_FREQUENCY_ORDERED_TEXT_SHA256 = (
    "53aee8e5592fc96159788e86519abbbcc9f1ab7c6348a1cb59a939bd57051d8f"
)
EXPECTED_PRIOR_FLOOR_AUDIT_PATH = (
    "experiments/udlm/prior_geometry/" "floor_selection_train_rows_10001_30000.json"
)
EXPECTED_PRIOR_FLOOR_AUDIT_SHA256 = (
    "02908dafaf589ca9a49e560aa1eab470a18d6bfe616b781164784c489f54a9f1"
)
EXPECTED_PRIOR_FLOOR_AUDIT_CANONICAL_SHA256 = (
    "2435a36af83e88a1bb1d602e840bb6ae1e2a97963a48abf68606a37e6320694d"
)
EXPECTED_PRIOR_FLOOR_AUDIT_SIZE_BYTES = 73_953
EXPECTED_PRIOR_FLOOR_AUDIT_SOURCE_PATH = "scripts/udlm/audit_empirical_prior_floor.py"
EXPECTED_PRIOR_FLOOR_AUDIT_SOURCE_SHA256 = (
    "305db0bdb9195ef0bf0ff8da6c1fc31c42e562c513bd9bccaed34bcb2023ccc8"
)
EXPECTED_PRIOR_FLOOR_AUDIT_SOURCE_SIZE_BYTES = 26_989
EXPECTED_PRIOR_FLOOR_AUDIT_SOURCE_REVISION = "6424b323084358ea050ba22d7e13ef8d45962496"
EXPECTED_EMPIRICAL_UNIFORM_MIX = Decimal("0.0002")
EXPECTED_PANEL_ROWS = 256
EXPECTED_PANEL_CONTENT_TOKENS = 13_627
EXPECTED_TIME_BINS = (Decimal("0.1"), Decimal("0.5"), Decimal("0.9"))
EXPECTED_TRAINING_SEED = 17
EXPECTED_CORRUPTION_SEED = 17
FINAL_GENERATION_SEEDS = (0, 1, 2)
EXPECTED_STAGE_ORDER = ("scheduler", "conditioning")
EXPECTED_HEALTH_VARIANT_ORDER = (
    "udlm",
    "schedule_uniform",
    "udlm_categorical",
)
EXPECTED_HEALTH_VARIANT_SLUGS = ("r", "s", "e")
EXPECTED_HEALTH_ELIGIBILITY = {
    "generation": False,
    "ranking": False,
    "superiority": False,
    "candidate_lock": False,
    "screen_authorization": True,
}
EXPECTED_HEALTH_CLAIM_SCOPE = "training_health_and_provenance_only"
EXPECTED_HEALTH_STATUS = "validated"
EXPECTED_HEALTH_SCHEMA_VERSION = 1
EXPECTED_HEALTH_RECEIPT_SCHEMA_VERSION = 5
EXPECTED_HEALTH_CHECKPOINT_STEP = 10
EXPECTED_SCREEN_CONFIG_DIRECTORY_TEMPLATE = (
    "experiments/udlm/protocols/optimization_screen_configs_gpu{gpu_count}"
)
EXPECTED_SCREEN_CONFIG_FILENAMES = (
    "scheduler_e_l0.json",
    "scheduler_e_l1.json",
    "conditioning_e_a0__e_l0.json",
    "conditioning_e_a0__e_l1.json",
    "conditioning_e_a1__e_l0.json",
    "conditioning_e_a1__e_l1.json",
)
EXPECTED_ARM_ORDER = {
    "scheduler": ("E-L0", "E-L1"),
    "conditioning": ("E-A0", "E-A1"),
}
EXPECTED_UPDATES = {"scheduler": 100, "conditioning": 500}
EXPECTED_ARTIFACT_SCHEMA_VERSIONS = {
    "launch_manifest": 2,
    "runtime_config": 2,
    "training_summary": 5,
    "exit_receipt": 5,
    "denoising_report": DENOISING_REPORT_SCHEMA_VERSION,
}
_STABLE_ARTIFACT_SNAPSHOT_KEYS = {
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
_TRAINING_ACCOUNTING_KEYS = {
    "training_seed",
    "optimizer_updates",
    "world_size",
    "micro_batch_size_per_rank",
    "accumulate_grad_batches",
    "effective_global_examples_per_optimizer_step",
    "total_requested_example_exposures",
    "hosted_stream_rank_partition_policy",
    "trainable_parameter_counts",
}
_TRAINING_HEALTH_KEYS = {
    "scope",
    "all_losses_finite",
    "all_observed_gradients_finite",
    "every_optimizer_step_had_a_nonzero_gradient",
    "loss_checks",
    "optimizer_step_checks",
    "gradient_tensor_observations",
    "gradient_element_observations",
}
_EXIT_RECEIPT_KEYS = {
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
_EXIT_COMPLETION_KEYS = {
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
_HOSTED_STREAM_RANK_PARTITION_POLICY = (
    "huggingface_split_dataset_by_node_disjoint_rank_streams"
)
_TRAINING_HEALTH_SCOPE = (
    "global-rank-zero callback counters; identical fail-fast checks "
    "execute independently on every rank"
)
_CONTROLLED_PYTHON_ENVIRONMENT = {
    "PYTHONNOUSERSITE": "1",
    "PYTHONOPTIMIZE": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
}
MAX_SAFE_UTILIZATION_PERCENT = 10
MIN_SAFE_FREE_MEMORY_MIB = 30_000
ACTIVE_COMPUTE_PROCESSES_ALLOWED = True
INCOMPLETE_EXIT_STATUS = 97
EXPECTED_OBSERVATION_POINT = (
    "on_before_optimizer_step_global_rank_zero_after_gradient_accumulation"
)
EVALUATOR_SOURCE_PATHS = (
    "scripts/udlm/evaluate_denoising_panel.py",
    "scripts/udlm/materialize_validation_panel.py",
    "scripts/udlm/launch_train_pilot.py",
    "scripts/train.py",
    "configs/base.yaml",
    "configs/udlm.yaml",
    "configs/udlm_categorical.yaml",
    "scripts/udlm/token_frequency_audit.py",
    "src/genmol/model.py",
    "src/genmol/diffusion.py",
    "src/genmol/backbone.py",
    "src/genmol/utils/ema.py",
    "src/genmol/utils/utils_data.py",
)
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
HEX_GIT_REVISION = re.compile(r"[0-9a-f]{40}\Z")
NORMALIZED_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")


@dataclass(frozen=True)
class BlobSnapshot:
    """Hash/size evidence for a large blob that was streamed, not retained."""

    size_bytes: int
    sha256: str


BlobLoader = Callable[[str, PurePosixPath], bytes | BlobSnapshot]
GitBlobLoader = Callable[[str, PurePosixPath], bytes]
GitAncestorChecker = Callable[[str, str], bool]
GitSoleParentChecker = Callable[[str, str], bool]
GitTreePathsLoader = Callable[[str, PurePosixPath], frozenset[str]]
GitPushedChecker = Callable[[str], bool]
GitDiffChecker = Callable[[str, str, frozenset[str]], bool]
HealthGateValidator = Callable[..., Mapping[str, Any]]


class ScreenValidationError(ValueError):
    """Raised when a registry or evidence object violates its frozen contract."""


class EvidenceIncomplete(ScreenValidationError):
    """Raised when registered evidence is absent or a run did not complete."""


@dataclass(frozen=True)
class ValidatedRegistry:
    """A structurally and byte-semantically validated prospective registry."""

    data: Mapping[str, Any]
    relative_path: PurePosixPath
    raw_sha256: str
    raw_size_bytes: int
    canonical_sha256: str
    git_blob_loader: GitBlobLoader
    git_ancestor_checker: GitAncestorChecker
    git_sole_parent_checker: GitSoleParentChecker
    git_tree_paths_loader: GitTreePathsLoader
    git_pushed_checker: GitPushedChecker
    git_diff_checker: GitDiffChecker

    @property
    def reference(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path.as_posix(),
            "sha256": self.raw_sha256,
            "canonical_sha256": self.canonical_sha256,
            "schema_version": REGISTRY_SCHEMA_VERSION,
        }


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ScreenValidationError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ScreenValidationError(f"non-finite JSON constant: {value}")


def _finite_json_decimal(value: str) -> Decimal:
    parsed = Decimal(value)
    if (
        not parsed.is_finite()
        or (parsed and not -324 <= parsed.adjusted() <= 308)
        or len(parsed.as_tuple().digits) > 32
    ):
        raise ScreenValidationError(f"non-finite JSON number: {value}")
    return parsed


def _bounded_json_integer(value: str) -> int:
    digits = value.removeprefix("-")
    if len(digits) > 128:
        raise ScreenValidationError("JSON integer has more than 128 digits")
    return int(value)


def strict_json_loads(payload: bytes, *, label: str) -> object:
    """Decode UTF-8 JSON while rejecting duplicate keys and nonfinite numbers."""

    try:
        source = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ScreenValidationError(f"{label} is not UTF-8") from error
    try:
        return json.loads(
            source,
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_decimal,
            parse_int=_bounded_json_integer,
        )
    except (json.JSONDecodeError, OverflowError, ValueError) as error:
        if isinstance(error, ScreenValidationError):
            raise
        raise ScreenValidationError(f"{label} is not valid JSON") from error


def canonical_json_sha256(value: object) -> str:
    def json_compatible(item: object) -> object:
        if isinstance(item, Decimal):
            return float(item)
        if isinstance(item, Mapping):
            return {key: json_compatible(child) for key, child in item.items()}
        if isinstance(item, list):
            return [json_compatible(child) for child in item]
        return item

    encoded = json.dumps(
        json_compatible(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _exact_json_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return left.keys() == right.keys() and all(
            _exact_json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _exact_json_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return bool(left == right)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScreenValidationError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ScreenValidationError(
            f"{label} fields are invalid: missing={missing}, extra={extra}"
        )


def _boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise ScreenValidationError(f"{label} must be a boolean")
    return value


def _integer(
    value: object,
    label: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise ScreenValidationError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ScreenValidationError(f"{label} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ScreenValidationError(f"{label} must be at most {maximum}")
    return value


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or NORMALIZED_ID.fullmatch(value) is None:
        raise ScreenValidationError(f"{label} is not a normalized identifier")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or HEX_SHA256.fullmatch(value) is None:
        raise ScreenValidationError(f"{label} must be a lowercase SHA-256")
    return value


def _git_revision(value: object, label: str) -> str:
    if not isinstance(value, str) or HEX_GIT_REVISION.fullmatch(value) is None:
        raise ScreenValidationError(f"{label} must be a full lowercase Git revision")
    return value


def _decimal(value: object, label: str, *, nonnegative: bool = False) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ScreenValidationError(f"{label} must be a JSON number")
    try:
        result = Decimal(str(value))
    except InvalidOperation as error:  # pragma: no cover - guarded by JSON parsing
        raise ScreenValidationError(f"{label} is not a finite decimal") from error
    if not result.is_finite() or (nonnegative and result < 0):
        raise ScreenValidationError(f"{label} is outside its finite numeric domain")
    return result


def _decimal_text(value: Decimal) -> str:
    return str(value.normalize()) if value else "0"


def _relative_path(
    value: object, label: str, *, suffix: str | None = None
) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ScreenValidationError(f"{label} must be a nonempty POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ScreenValidationError(f"{label} must be normalized and relative")
    if suffix is not None and path.suffix != suffix:
        raise ScreenValidationError(f"{label} must end in {suffix}")
    return path


_BLOB_REF_KEYS = {"root", "relative_path", "sha256", "size_bytes"}
_JSON_REF_KEYS = _BLOB_REF_KEYS | {"schema_version", "canonical_sha256"}
_CONFIG_REF_KEYS = _BLOB_REF_KEYS | {"canonical_sha256"}


def _blob_ref(
    value: object,
    label: str,
    *,
    required_root: str | None = None,
    suffix: str | None = None,
) -> dict[str, Any]:
    ref = _mapping(value, label)
    _exact_keys(ref, _BLOB_REF_KEYS, label)
    root = ref.get("root")
    if root not in {"repository", "project"}:
        raise ScreenValidationError(f"{label}.root is invalid")
    if required_root is not None and root != required_root:
        raise ScreenValidationError(f"{label}.root must equal {required_root}")
    return {
        "root": root,
        "relative_path": _relative_path(
            ref.get("relative_path"), f"{label}.relative_path", suffix=suffix
        ).as_posix(),
        "sha256": _sha256(ref.get("sha256"), f"{label}.sha256"),
        "size_bytes": _integer(ref.get("size_bytes"), f"{label}.size_bytes", minimum=1),
    }


def _json_ref(
    value: object,
    label: str,
    *,
    required_root: str | None = None,
) -> dict[str, Any]:
    ref = _mapping(value, label)
    _exact_keys(ref, _JSON_REF_KEYS, label)
    base = _blob_ref(
        {key: ref[key] for key in _BLOB_REF_KEYS},
        label,
        required_root=required_root,
        suffix=".json",
    )
    return {
        **base,
        "schema_version": _integer(
            ref.get("schema_version"), f"{label}.schema_version", minimum=1
        ),
        "canonical_sha256": _sha256(
            ref.get("canonical_sha256"), f"{label}.canonical_sha256"
        ),
    }


def _config_ref(value: object, label: str) -> dict[str, Any]:
    ref = _mapping(value, label)
    _exact_keys(ref, _CONFIG_REF_KEYS, label)
    base = _blob_ref(
        {key: ref[key] for key in _BLOB_REF_KEYS},
        label,
        required_root="repository",
        suffix=".json",
    )
    return {
        **base,
        "canonical_sha256": _sha256(
            ref.get("canonical_sha256"), f"{label}.canonical_sha256"
        ),
    }


def _expected_screen_config_paths(gpu_count: int) -> tuple[str, ...]:
    """Return the exact selected-world-size R0 config family."""

    if gpu_count not in {1, 2}:  # pragma: no cover - caller validates first
        raise ScreenValidationError("screen GPU count must equal 1 or 2")
    directory = EXPECTED_SCREEN_CONFIG_DIRECTORY_TEMPLATE.format(gpu_count=gpu_count)
    return tuple(
        sorted(
            f"{directory}/{filename}" for filename in EXPECTED_SCREEN_CONFIG_FILENAMES
        )
    )


def _health_run_name(
    *, gpu_count: int, variant_slug: str, health_source_revision: str
) -> str:
    return f"health-w{gpu_count}-{variant_slug}-{health_source_revision}"


def _health_terminal_receipt_ref(value: object) -> dict[str, Any]:
    label = "health-gate terminal receipt"
    ref = _mapping(value, label)
    _exact_keys(
        ref,
        _JSON_REF_KEYS | {"training_variant", "position"},
        label,
    )
    json_ref = _json_ref(
        {key: ref[key] for key in _JSON_REF_KEYS},
        label,
        required_root="repository",
    )
    if (
        ref.get("training_variant") != EXPECTED_HEALTH_VARIANT_ORDER[-1]
        or _integer(ref.get("position"), f"{label}.position") != 2
    ):
        raise ScreenValidationError("health-gate terminal receipt must identify E")
    return {
        **json_ref,
        "training_variant": EXPECTED_HEALTH_VARIANT_ORDER[-1],
        "position": 2,
    }


def _load_bound_blob(
    ref: Mapping[str, Any], *, loader: BlobLoader, label: str
) -> bytes | None:
    try:
        payload = loader(ref["root"], PurePosixPath(ref["relative_path"]))
    except ScreenValidationError:
        raise
    except Exception as error:
        raise ScreenValidationError(f"{label} is unavailable") from error
    if isinstance(payload, BlobSnapshot):
        size_bytes = payload.size_bytes
        digest = payload.sha256
        retained = None
    elif isinstance(payload, bytes):
        size_bytes = len(payload)
        digest = hashlib.sha256(payload).hexdigest()
        retained = payload
    else:
        raise ScreenValidationError(f"{label} loader returned an invalid value")
    if size_bytes != ref["size_bytes"]:
        raise ScreenValidationError(f"{label} size differs from its binding")
    if digest != ref["sha256"]:
        raise ScreenValidationError(f"{label} digest differs from its binding")
    return retained


def _validate_prior_floor_audit(
    source_by_path: Mapping[str, Mapping[str, Any]], *, loader: BlobLoader
) -> None:
    """Require the immutable training-only audit that selected the pilot floor."""

    source_ref = source_by_path.get(EXPECTED_PRIOR_FLOOR_AUDIT_SOURCE_PATH)
    artifact_ref = source_by_path.get(EXPECTED_PRIOR_FLOOR_AUDIT_PATH)
    if source_ref is None or artifact_ref is None:
        raise ScreenValidationError(
            "registry source map omits the empirical-prior floor audit"
        )
    if (
        source_ref["sha256"] != EXPECTED_PRIOR_FLOOR_AUDIT_SOURCE_SHA256
        or source_ref["size_bytes"] != EXPECTED_PRIOR_FLOOR_AUDIT_SOURCE_SIZE_BYTES
    ):
        raise ScreenValidationError("prior-floor audit source identity is invalid")
    if (
        artifact_ref["sha256"] != EXPECTED_PRIOR_FLOOR_AUDIT_SHA256
        or artifact_ref["size_bytes"] != EXPECTED_PRIOR_FLOOR_AUDIT_SIZE_BYTES
    ):
        raise ScreenValidationError("prior-floor audit artifact identity is invalid")
    payload = _load_bound_blob(
        artifact_ref, loader=loader, label="registry prior-floor audit artifact"
    )
    if payload is None:
        raise ScreenValidationError(
            "prior-floor audit artifact bytes were not retained"
        )
    audit = _mapping(
        strict_json_loads(payload, label="registry prior-floor audit artifact"),
        "registry prior-floor audit artifact",
    )
    if canonical_json_sha256(audit) != EXPECTED_PRIOR_FLOOR_AUDIT_CANONICAL_SHA256:
        raise ScreenValidationError("prior-floor audit canonical identity is invalid")
    git = _mapping(audit.get("git"), "prior-floor audit git provenance")
    recommendation = _mapping(
        audit.get("recommendation"), "prior-floor audit recommendation"
    )
    data_use = _mapping(audit.get("data_use"), "prior-floor audit data use")
    inputs = _mapping(audit.get("inputs"), "prior-floor audit inputs")
    source_files = _mapping(
        inputs.get("source_files"), "prior-floor audit source files"
    )
    recorded_source = _mapping(
        source_files.get(EXPECTED_PRIOR_FLOOR_AUDIT_SOURCE_PATH),
        "prior-floor audit recorded source",
    )
    if (
        audit.get("schema_version") != 1
        or git.get("commit") != EXPECTED_PRIOR_FLOOR_AUDIT_SOURCE_REVISION
        or git.get("upstream") != EXPECTED_PRIOR_FLOOR_AUDIT_SOURCE_REVISION
        or git.get("dirty") is not False
        or recorded_source.get("sha256") != EXPECTED_PRIOR_FLOOR_AUDIT_SOURCE_SHA256
        or recorded_source.get("git_blob_verified") is not True
        or recommendation.get("status")
        != "training_only_retrospective_engineering_recommendation"
        or _decimal(
            recommendation.get("recommended_uniform_mixture_weight"),
            "prior-floor audit recommended mixture weight",
        )
        != EXPECTED_EMPIRICAL_UNIFORM_MIX
        or _decimal(
            recommendation.get("candidate_uniform_mixture_weight"),
            "prior-floor audit candidate mixture weight",
        )
        != EXPECTED_EMPIRICAL_UNIFORM_MIX
        or recommendation.get(
            "candidate_nll_strictly_better_than_current_on_both_blocks"
        )
        is not True
        or recommendation.get("both_block_optima_within_0_0001_to_0_0003") is not True
        or data_use.get("split") != "training"
        or data_use.get("final_generation_seeds_or_metrics_used") is not False
        or data_use.get("formal_preregistration_before_data_access") is not False
    ):
        raise ScreenValidationError(
            "prior-floor audit provenance or training-only recommendation is invalid"
        )


def _load_json_ref(
    ref: Mapping[str, Any],
    *,
    loader: BlobLoader,
    label: str,
    schema_field: str = "schema_version",
) -> Mapping[str, Any]:
    payload = _load_bound_blob(ref, loader=loader, label=label)
    if payload is None:
        raise ScreenValidationError(f"{label} bytes were not retained for JSON parsing")
    parsed = _mapping(strict_json_loads(payload, label=label), label)
    if _integer(parsed.get(schema_field), f"{label} schema") != ref["schema_version"]:
        raise ScreenValidationError(f"{label} schema differs from its binding")
    if canonical_json_sha256(parsed) != ref["canonical_sha256"]:
        raise ScreenValidationError(
            f"{label} canonical digest differs from its binding"
        )
    return parsed


def _load_config_ref(
    ref: Mapping[str, Any], *, loader: BlobLoader, label: str
) -> Mapping[str, Any]:
    payload = _load_bound_blob(ref, loader=loader, label=label)
    if payload is None:
        raise ScreenValidationError(f"{label} bytes were not retained for JSON parsing")
    parsed = _mapping(strict_json_loads(payload, label=label), label)
    if canonical_json_sha256(parsed) != ref["canonical_sha256"]:
        raise ScreenValidationError(
            f"{label} canonical digest differs from its binding"
        )
    return parsed


def _verify_git_blob(
    ref: Mapping[str, Any],
    *,
    revision: str,
    git_blob_loader: GitBlobLoader,
    label: str,
) -> None:
    if ref["root"] != "repository":
        raise ScreenValidationError(f"{label} is not a repository Git blob")
    try:
        payload = git_blob_loader(revision, PurePosixPath(ref["relative_path"]))
    except ScreenValidationError:
        raise
    except Exception as error:
        raise ScreenValidationError(f"{label} Git blob is unavailable") from error
    if not isinstance(payload, bytes):
        raise ScreenValidationError(f"{label} Git loader must return bytes")
    if (
        len(payload) != ref["size_bytes"]
        or hashlib.sha256(payload).hexdigest() != ref["sha256"]
    ):
        raise ScreenValidationError(f"{label} Git blob differs from its binding")


def _validate_firewall(value: object) -> None:
    firewall = _mapping(value, "registry firewall")
    _exact_keys(
        firewall,
        {
            "final_generation_seeds",
            "final_generation_seeds_forbidden",
            "generation_metrics_allowed",
            "health_gate_evidence_eligible",
            "superiority_evidence_eligible",
            "unregistered_attempts_allowed",
            "failed_or_missing_evidence_policy",
        },
        "registry firewall",
    )
    if firewall.get("final_generation_seeds") != list(FINAL_GENERATION_SEEDS):
        raise ScreenValidationError(
            "registry final-generation seed firewall is invalid"
        )
    required_false = (
        "generation_metrics_allowed",
        "health_gate_evidence_eligible",
        "superiority_evidence_eligible",
        "unregistered_attempts_allowed",
    )
    if _boolean(
        firewall.get("final_generation_seeds_forbidden"),
        "registry final seed prohibition",
    ) is not True or any(
        _boolean(firewall.get(field), f"registry firewall {field}") is not False
        for field in required_false
    ):
        raise ScreenValidationError("registry selection firewall is not fail-closed")
    if firewall.get("failed_or_missing_evidence_policy") != "incomplete_no_winner":
        raise ScreenValidationError("registry evidence-failure policy is invalid")


def _validate_gradient_contract(value: object) -> tuple[Mapping[str, Any], str]:
    contract = _mapping(value, "conditioning gradient contract")
    _exact_keys(
        contract,
        {
            "schema_version",
            "observation_point",
            "optimizer_checks",
            "first_positive_lr_optimizer_step",
            "timestep_mlp_required_optimizer_check",
            "groups",
        },
        "conditioning gradient contract",
    )
    if (
        _integer(contract.get("schema_version"), "gradient contract schema")
        != GRADIENT_CONTRACT_SCHEMA_VERSION
    ):
        raise ScreenValidationError(
            "conditioning gradient contract schema is unsupported"
        )
    if contract.get("observation_point") != EXPECTED_OBSERVATION_POINT:
        raise ScreenValidationError(
            "conditioning gradient observation point is invalid"
        )
    if contract.get("optimizer_checks") != [1, 2, 3]:
        raise ScreenValidationError("conditioning gradient checks must equal [1, 2, 3]")
    if contract.get("first_positive_lr_optimizer_step") != 2:
        raise ScreenValidationError("first positive-LR optimizer step must equal 2")
    if contract.get("timestep_mlp_required_optimizer_check") != 3:
        raise ScreenValidationError(
            "timestep-MLP optimizer-check requirement must equal 3"
        )
    groups = contract.get("groups")
    if not isinstance(groups, list) or len(groups) != 2:
        raise ScreenValidationError("conditioning gradient contract needs two groups")
    expected = (("film_modulation", "film"), ("timestep_mlp", "timestep_mlp"))
    all_names: set[str] = set()
    for index, (raw_group, (expected_id, expected_kind)) in enumerate(
        zip(groups, expected, strict=True)
    ):
        label = f"conditioning gradient group {index}"
        group = _mapping(raw_group, label)
        _exact_keys(group, {"group_id", "kind", "parameters"}, label)
        if group.get("group_id") != expected_id or group.get("kind") != expected_kind:
            raise ScreenValidationError(f"{label} identity is invalid")
        parameters = group.get("parameters")
        if not isinstance(parameters, list) or not parameters:
            raise ScreenValidationError(f"{label} must register parameters")
        for parameter_index, raw_parameter in enumerate(parameters):
            parameter_label = f"{label} parameter {parameter_index}"
            parameter = _mapping(raw_parameter, parameter_label)
            _exact_keys(parameter, {"name", "shape"}, parameter_label)
            name = parameter.get("name")
            if not isinstance(name, str) or not name or name in all_names:
                raise ScreenValidationError(
                    "conditioning gradient parameter names must be nonempty and unique"
                )
            all_names.add(name)
            shape = parameter.get("shape")
            if (
                not isinstance(shape, list)
                or not shape
                or any(
                    type(dimension) is not int or dimension <= 0 for dimension in shape
                )
            ):
                raise ScreenValidationError(f"{parameter_label}.shape is invalid")
    film_parameters = groups[0]["parameters"]
    timestep_parameters = groups[1]["parameters"]
    if (
        len(film_parameters) != 24
        or sum(math.prod(parameter["shape"]) for parameter in film_parameters)
        != 14_174_208
        or len(timestep_parameters) != 4
        or sum(math.prod(parameter["shape"]) for parameter in timestep_parameters)
        != 787_968
    ):
        raise ScreenValidationError(
            "conditioning gradient topology is not BERT-base FiLM"
        )
    return contract, canonical_json_sha256(contract)


def _validate_scheduler_spec(value: object, *, arm_id: str) -> None:
    scheduler = _mapping(value, f"{arm_id} scheduler")
    _exact_keys(
        scheduler,
        {
            "kind",
            "warmup_updates",
            "horizon_updates",
            "peak_learning_rate",
            "minimum_learning_rate",
        },
        f"{arm_id} scheduler",
    )
    peak = _decimal(scheduler.get("peak_learning_rate"), f"{arm_id} peak LR")
    if peak != Decimal("0.0003"):
        raise ScreenValidationError(f"{arm_id} peak learning rate must equal 3e-4")
    if arm_id == "E-L0":
        expected = {
            "kind": "constant_with_warmup",
            "warmup_updates": 2500,
            "horizon_updates": None,
            "minimum_learning_rate": None,
        }
    else:
        expected = {
            "kind": "cosine_with_minimum",
            "warmup_updates": 50,
            "horizon_updates": 1000,
            "minimum_learning_rate": Decimal("0.000003"),
        }
    for field in ("kind", "warmup_updates", "horizon_updates"):
        if scheduler.get(field) != expected[field]:
            raise ScreenValidationError(f"{arm_id} scheduler {field} is invalid")
    if expected["minimum_learning_rate"] is None:
        if scheduler.get("minimum_learning_rate") is not None:
            raise ScreenValidationError("E-L0 minimum learning rate must be null")
    elif (
        _decimal(scheduler.get("minimum_learning_rate"), f"{arm_id} minimum LR")
        != expected["minimum_learning_rate"]
    ):
        raise ScreenValidationError("E-L1 minimum learning rate must equal 3e-6")


def _validate_conditioner_spec(value: object, *, arm_id: str) -> None:
    conditioner = _mapping(value, f"{arm_id} conditioner")
    _exact_keys(
        conditioner,
        {"kind", "post_timestep_mlp_silu", "zero_initialized_per_layer_film"},
        f"{arm_id} conditioner",
    )
    candidate = arm_id == "E-A1"
    expected_kind = "film_adaln" if candidate else "additive"
    if conditioner.get("kind") != expected_kind:
        raise ScreenValidationError(f"{arm_id} conditioner kind is invalid")
    for field in ("post_timestep_mlp_silu", "zero_initialized_per_layer_film"):
        if (
            _boolean(conditioner.get(field), f"{arm_id} conditioner {field}")
            is not candidate
        ):
            raise ScreenValidationError(f"{arm_id} conditioner {field} is invalid")


def _nested_mapping(
    value: Mapping[str, Any], key: str, label: str
) -> Mapping[str, Any]:
    return _mapping(value.get(key), f"{label}.{key}")


def _validate_resolved_config_semantics(
    config: Mapping[str, Any],
    *,
    stage_id: str,
    arm_id: str,
    scheduler_arm_id: str | None,
    output_directory: str,
    gpu_count: int,
    initialization_checkpoint: Mapping[str, Any],
    global_batch_size: int,
    micro_batch_size: int,
    accumulation: int,
) -> None:
    label = f"{arm_id} resolved config"
    training = _nested_mapping(config, "training", label)
    udlm = _nested_mapping(training, "udlm", f"{label}.training")
    trainer = _nested_mapping(config, "trainer", label)
    loader = _nested_mapping(config, "loader", label)
    optim = _nested_mapping(config, "optim", label)
    scheduler = _nested_mapping(optim, "scheduler", f"{label}.optim")
    callback = _nested_mapping(config, "callback", label)
    if (
        config.get("seed") != EXPECTED_TRAINING_SEED
        or training.get("diffusion") != "udlm"
        or udlm.get("prior_variant") != "empirical_frequency"
        or _decimal(
            udlm.get("empirical_uniform_mix"),
            f"{label} empirical uniform mixture weight",
        )
        != EXPECTED_EMPIRICAL_UNIFORM_MIX
        or trainer.get("accelerator") != "cuda"
        or trainer.get("num_nodes") != 1
        or trainer.get("devices") != gpu_count
        or trainer.get("max_steps") != EXPECTED_UPDATES[stage_id]
        or loader.get("global_batch_size") != global_batch_size
        or loader.get("batch_size") != micro_batch_size
        or trainer.get("accumulate_grad_batches") != accumulation
        or training.get("init_from_mdlm_checkpoint_sha256")
        != initialization_checkpoint["sha256"]
        or training.get("init_from_mdlm_ema") is not True
        or training.get("reseed_after_model_initialization")
        is not (stage_id == "conditioning")
    ):
        raise ScreenValidationError(f"{label} common training semantics are unmatched")
    checkpoint_path = training.get("init_from_mdlm_checkpoint")
    if not isinstance(checkpoint_path, str) or not checkpoint_path.replace(
        "\\", "/"
    ).endswith("/" + initialization_checkpoint["relative_path"]):
        raise ScreenValidationError(f"{label} warm-start path is unmatched")
    for forbidden in ("ckpt_path", "resume_from_checkpoint"):
        if config.get(forbidden) not in (None, False) or trainer.get(forbidden) not in (
            None,
            False,
        ):
            raise ScreenValidationError(f"{label} attempts to resume training")
    expected_conditioner = "film_adaln" if arm_id == "E-A1" else "additive"
    expected_zero_init = arm_id != "E-A1"
    if (
        udlm.get("conditioning_variant") != expected_conditioner
        or udlm.get("zero_init_conditioning") is not expected_zero_init
    ):
        raise ScreenValidationError(f"{label} conditioner semantics are unmatched")
    selected_scheduler = arm_id if stage_id == "scheduler" else scheduler_arm_id
    if selected_scheduler == "E-L0":
        expected_scheduler = {
            "name": "constant_with_linear_warmup",
            "warmup_updates": 2500,
            "horizon_updates": None,
            "decay_floor_lr": None,
        }
    elif selected_scheduler == "E-L1":
        expected_scheduler = {
            "name": "half_cosine_with_linear_warmup_and_floor",
            "warmup_updates": 50,
            "horizon_updates": 1000,
            "decay_floor_lr": Decimal("0.000003"),
        }
    else:  # pragma: no cover - caller validates the contingency first
        raise ScreenValidationError(f"{label} scheduler contingency is invalid")
    for field, expected in expected_scheduler.items():
        observed = scheduler.get(field)
        if isinstance(expected, Decimal):
            observed = _decimal(observed, f"{label} scheduler {field}")
        if observed != expected:
            raise ScreenValidationError(f"{label} scheduler {field} is unmatched")
    if _decimal(optim.get("lr"), f"{label} peak learning rate") != Decimal("0.0003"):
        raise ScreenValidationError(f"{label} peak learning rate is unmatched")
    callback_dir = callback.get("dirpath")
    expected_callback_suffix = f"/{output_directory}/checkpoints"
    if not isinstance(callback_dir, str) or not callback_dir.replace(
        "\\", "/"
    ).endswith(expected_callback_suffix):
        raise ScreenValidationError(f"{label} checkpoint directory is unmatched")


def _validate_selection_rule(
    value: object,
    *,
    stage_id: str,
    gradient_contract_sha256: str | None,
) -> None:
    rule = _mapping(value, f"{stage_id} selection rule")
    common = {
        "rule_id",
        "pooled_candidate_max_percent_of_control",
        "per_bin_candidate_max_percent_of_control",
        "complete_failure_fallback_arm",
        "incomplete_evidence_winner",
    }
    if stage_id == "scheduler":
        expected_keys = common | {"minimum_strictly_better_bins"}
        expected_id = (
            "l1_if_pooled_loss_le_98pct_l0_and_two_bins_strictly_better_"
            "and_each_bin_le_102pct"
        )
        expected_fallback = "E-L0"
    else:
        expected_keys = common | {
            "pooled_clean_token_accuracy_nondecreasing",
            "exact_initialization_equality_required",
            "gradient_contract_sha256",
        }
        expected_id = (
            "a1_if_exact_init_and_pooled_loss_le_98pct_a0_and_each_bin_le_"
            "102pct_and_pooled_accuracy_nondecreasing_and_registered_gradients_pass"
        )
        expected_fallback = "E-A0"
    _exact_keys(rule, expected_keys, f"{stage_id} selection rule")
    if (
        rule.get("rule_id") != expected_id
        or rule.get("pooled_candidate_max_percent_of_control") != 98
        or rule.get("per_bin_candidate_max_percent_of_control") != 102
        or rule.get("complete_failure_fallback_arm") != expected_fallback
        or rule.get("incomplete_evidence_winner") is not None
    ):
        raise ScreenValidationError(f"{stage_id} selection rule is invalid")
    if stage_id == "scheduler":
        if rule.get("minimum_strictly_better_bins") != 2:
            raise ScreenValidationError("scheduler needs two strictly better bins")
    elif (
        _boolean(
            rule.get("pooled_clean_token_accuracy_nondecreasing"),
            "conditioning accuracy rule",
        )
        is not True
        or _boolean(
            rule.get("exact_initialization_equality_required"),
            "conditioning initialization rule",
        )
        is not True
        or rule.get("gradient_contract_sha256") != gradient_contract_sha256
    ):
        raise ScreenValidationError("conditioning selection prerequisites are invalid")


def _validate_registry_stage(
    value: object,
    *,
    index: int,
    loader: BlobLoader,
    git_blob_loader: GitBlobLoader,
    source_revision: str,
    gpu_count: int,
    initialization_checkpoint: Mapping[str, Any],
    global_batch_size: int,
    micro_batch_size: int,
    accumulation: int,
) -> dict[str, Any]:
    stage_id = EXPECTED_STAGE_ORDER[index]
    stage = _mapping(value, f"registry stage {index}")
    _exact_keys(
        stage,
        {
            "stage_id",
            "order_index",
            "training_seed",
            "optimizer_updates",
            "arm_order",
            "arms",
            "selection_rule",
            "gradient_contract",
            "initialization_fixture",
        },
        f"registry stage {index}",
    )
    if (
        stage.get("stage_id") != stage_id
        or stage.get("order_index") != index
        or stage.get("training_seed") != EXPECTED_TRAINING_SEED
        or stage.get("optimizer_updates") != EXPECTED_UPDATES[stage_id]
        or stage.get("arm_order") != list(EXPECTED_ARM_ORDER[stage_id])
    ):
        raise ScreenValidationError(f"registry {stage_id} stage identity is invalid")
    gradient_contract: Mapping[str, Any] | None
    gradient_contract_sha256: str | None
    if stage_id == "scheduler":
        if stage.get("gradient_contract") is not None:
            raise ScreenValidationError("scheduler gradient contract must be null")
        if stage.get("initialization_fixture") is not None:
            raise ScreenValidationError("scheduler initialization fixture must be null")
        gradient_contract = None
        gradient_contract_sha256 = None
        gradient_contract_artifact = None
        initialization_fixture = None
    else:
        gradient_contract_artifact = _json_ref(
            stage.get("gradient_contract"),
            "conditioning gradient contract artifact",
            required_root="repository",
        )
        if (
            gradient_contract_artifact["relative_path"]
            != EXPECTED_GRADIENT_CONTRACT_PATH
            or gradient_contract_artifact["sha256"] != EXPECTED_GRADIENT_CONTRACT_SHA256
            or gradient_contract_artifact["canonical_sha256"]
            != EXPECTED_GRADIENT_CONTRACT_CANONICAL_SHA256
            or gradient_contract_artifact["schema_version"]
            != GRADIENT_CONTRACT_SCHEMA_VERSION
        ):
            raise ScreenValidationError("conditioning gradient contract is not pinned")
        gradient_contract_document = _load_json_ref(
            gradient_contract_artifact,
            loader=loader,
            label="conditioning gradient contract artifact",
        )
        gradient_contract, gradient_contract_sha256 = _validate_gradient_contract(
            gradient_contract_document
        )
        _verify_git_blob(
            gradient_contract_artifact,
            revision=source_revision,
            git_blob_loader=git_blob_loader,
            label="conditioning gradient contract artifact",
        )
        initialization_fixture = _json_ref(
            stage.get("initialization_fixture"),
            "conditioning initialization fixture",
            required_root="repository",
        )
        if (
            initialization_fixture["relative_path"]
            != EXPECTED_INITIALIZATION_FIXTURE_PATH
            or initialization_fixture["sha256"]
            != EXPECTED_INITIALIZATION_FIXTURE_SHA256
            or initialization_fixture["canonical_sha256"]
            != EXPECTED_INITIALIZATION_FIXTURE_CANONICAL_SHA256
            or initialization_fixture["size_bytes"]
            != EXPECTED_INITIALIZATION_FIXTURE_SIZE_BYTES
            or initialization_fixture["schema_version"] != 1
        ):
            raise ScreenValidationError(
                "conditioning initialization fixture is not exactly pinned"
            )
        fixture = _load_json_ref(
            initialization_fixture,
            loader=loader,
            label="conditioning initialization fixture",
        )
        _exact_keys(
            fixture,
            {
                "schema_version",
                "purpose",
                "checkpoint_sha256",
                "probe_phase",
                "input_ids",
                "attention_mask",
                "noise_tensor",
                "timestep_tensor",
            },
            "conditioning initialization fixture",
        )
        if (
            _integer(fixture.get("schema_version"), "initialization fixture schema")
            != 1
            or fixture.get("purpose")
            != "pre_optimizer_exact_additive_vs_film_warm_start_logit_identity"
            or fixture.get("checkpoint_sha256") != initialization_checkpoint["sha256"]
            or fixture.get("probe_phase")
            != "after_mdlm_ema_load_before_training_rng_reseed_and_optimizer_creation"
        ):
            raise ScreenValidationError(
                "conditioning initialization fixture is invalid"
            )
        _sha256(fixture.get("checkpoint_sha256"), "conditioning fixture checkpoint")
        input_ids = fixture.get("input_ids")
        attention_mask = fixture.get("attention_mask")
        noise_tensor = fixture.get("noise_tensor")
        timestep_tensor = fixture.get("timestep_tensor")
        if (
            not isinstance(input_ids, list)
            or not input_ids
            or any(
                not isinstance(row, list)
                or not row
                or any(type(item) is not int or item < 0 for item in row)
                for row in input_ids
            )
            or not isinstance(attention_mask, list)
            or len(attention_mask) != len(input_ids)
            or any(
                not isinstance(row, list)
                or len(row) != len(input_ids[index])
                or any(type(item) is not int or item not in {0, 1} for item in row)
                for index, row in enumerate(attention_mask)
            )
            or not isinstance(noise_tensor, list)
            or len(noise_tensor) != len(input_ids)
            or not isinstance(timestep_tensor, list)
            or len(timestep_tensor) != len(input_ids)
        ):
            raise ScreenValidationError(
                "conditioning initialization fixture tensors are invalid"
            )
        for label, values in (
            ("noise", noise_tensor),
            ("timestep", timestep_tensor),
        ):
            if any(
                _decimal(value, f"conditioning fixture {label}") < 0 for value in values
            ):
                raise ScreenValidationError(
                    "conditioning initialization fixture values are invalid"
                )
        _verify_git_blob(
            initialization_fixture,
            revision=source_revision,
            git_blob_loader=git_blob_loader,
            label="conditioning initialization fixture",
        )
    _validate_selection_rule(
        stage.get("selection_rule"),
        stage_id=stage_id,
        gradient_contract_sha256=gradient_contract_sha256,
    )
    arms = stage.get("arms")
    if not isinstance(arms, list) or len(arms) != 2:
        raise ScreenValidationError(f"registry {stage_id} must contain two arms")
    seen_attempts: set[str] = set()
    seen_outputs: set[str] = set()
    normalized_arms = []
    for arm_index, (raw_arm, expected_arm_id) in enumerate(
        zip(arms, EXPECTED_ARM_ORDER[stage_id], strict=True)
    ):
        label = f"registry {stage_id} arm {arm_index}"
        arm = _mapping(raw_arm, label)
        _exact_keys(
            arm,
            {
                "arm_id",
                "attempt_id",
                "role",
                "scheduler",
                "conditioner",
                "resolved_configs",
            },
            label,
        )
        if arm.get("arm_id") != expected_arm_id:
            raise ScreenValidationError(f"{label} is out of registered order")
        attempt_id = _identifier(arm.get("attempt_id"), f"{label}.attempt_id")
        if attempt_id in seen_attempts:
            raise ScreenValidationError("registry attempt IDs must be unique")
        seen_attempts.add(attempt_id)
        expected_role = "control" if arm_index == 0 else "candidate"
        if arm.get("role") != expected_role:
            raise ScreenValidationError(f"{label}.role is invalid")
        if stage_id == "scheduler":
            _validate_scheduler_spec(arm.get("scheduler"), arm_id=expected_arm_id)
            additive_id = "E-A0"
        else:
            if arm.get("scheduler") != "selected_scheduler_arm":
                raise ScreenValidationError(
                    "conditioning arm scheduler must be selected_scheduler_arm"
                )
            additive_id = expected_arm_id
        _validate_conditioner_spec(arm.get("conditioner"), arm_id=additive_id)
        configs = arm.get("resolved_configs")
        expected_scheduler_ids: tuple[str | None, ...] = (
            (None,) if stage_id == "scheduler" else EXPECTED_ARM_ORDER["scheduler"]
        )
        if not isinstance(configs, list) or len(configs) != len(expected_scheduler_ids):
            raise ScreenValidationError(f"{label} has the wrong contingent configs")
        normalized_configs = []
        for config_index, (raw_config, scheduler_arm_id) in enumerate(
            zip(configs, expected_scheduler_ids, strict=True)
        ):
            config_label = f"{label} config {config_index}"
            config_entry = _mapping(raw_config, config_label)
            _exact_keys(
                config_entry,
                {"scheduler_arm_id", "output_directory", "config"},
                config_label,
            )
            if config_entry.get("scheduler_arm_id") != scheduler_arm_id:
                raise ScreenValidationError(f"{config_label} contingency is invalid")
            output_directory = _relative_path(
                config_entry.get("output_directory"),
                f"{config_label}.output_directory",
            ).as_posix()
            if not output_directory.startswith("output/udlm/screens/"):
                raise ScreenValidationError(
                    "screen output directories must live under output/udlm/screens"
                )
            if output_directory in seen_outputs:
                raise ScreenValidationError("screen output directories must be unique")
            seen_outputs.add(output_directory)
            config_ref = _config_ref(
                config_entry.get("config"), f"{config_label}.config"
            )
            parsed_config = _load_config_ref(
                config_ref, loader=loader, label=f"{config_label}.config"
            )
            _validate_resolved_config_semantics(
                parsed_config,
                stage_id=stage_id,
                arm_id=expected_arm_id,
                scheduler_arm_id=scheduler_arm_id,
                output_directory=output_directory,
                gpu_count=gpu_count,
                initialization_checkpoint=initialization_checkpoint,
                global_batch_size=global_batch_size,
                micro_batch_size=micro_batch_size,
                accumulation=accumulation,
            )
            _verify_git_blob(
                config_ref,
                revision=source_revision,
                git_blob_loader=git_blob_loader,
                label=f"{config_label}.config",
            )
            normalized_configs.append(
                {
                    "scheduler_arm_id": scheduler_arm_id,
                    "output_directory": output_directory,
                    "config": config_ref,
                    "parsed_config": parsed_config,
                }
            )
        normalized_arms.append(
            {
                **dict(arm),
                "attempt_id": attempt_id,
                "resolved_configs": normalized_configs,
            }
        )
    return {
        **dict(stage),
        "arms": normalized_arms,
        "gradient_contract": gradient_contract,
        "gradient_contract_artifact": gradient_contract_artifact,
        "gradient_contract_sha256": gradient_contract_sha256,
        "initialization_fixture": initialization_fixture,
    }


def _validate_prerequisite_health_gate(
    value: object,
    *,
    loader: BlobLoader,
    registry_source_revision: str,
    gpu_count: int,
    stages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the frozen health evidence and its declared H-to-R0 boundary."""

    gate = _mapping(value, "registry prerequisite health gate")
    _exact_keys(
        gate,
        {"evidence", "source_transition"},
        "registry prerequisite health gate",
    )
    evidence = _mapping(gate.get("evidence"), "registry health-gate evidence")
    _exact_keys(
        evidence,
        {
            "schema_version",
            "status",
            "claim_scope",
            "health_source_revision",
            "gpu_count",
            "matched_panel_spec_sha256",
            "terminal_receipt",
            "receipt_members",
            "checkpoint_members",
            "eligibility",
        },
        "registry health-gate evidence",
    )
    if (
        _integer(evidence.get("schema_version"), "health-gate evidence schema")
        != EXPECTED_HEALTH_SCHEMA_VERSION
        or evidence.get("status") != EXPECTED_HEALTH_STATUS
        or evidence.get("claim_scope") != EXPECTED_HEALTH_CLAIM_SCOPE
    ):
        raise ScreenValidationError("registry health-gate identity is invalid")
    health_source_revision = _git_revision(
        evidence.get("health_source_revision"), "health-gate source revision"
    )
    health_gpu_count = _integer(evidence.get("gpu_count"), "health-gate GPU count")
    if health_gpu_count != gpu_count:
        raise ScreenValidationError(
            "health-gate GPU count differs from screen common training"
        )
    matched_panel_sha256 = _sha256(
        evidence.get("matched_panel_spec_sha256"),
        "health-gate matched-panel digest",
    )

    terminal_receipt = _health_terminal_receipt_ref(evidence.get("terminal_receipt"))
    _load_json_ref(
        terminal_receipt,
        loader=loader,
        label="health-gate terminal receipt",
    )

    raw_receipts = evidence.get("receipt_members")
    if not isinstance(raw_receipts, list) or len(raw_receipts) != 3:
        raise ScreenValidationError(
            "health-gate receipt membership must contain exactly R, S, E"
        )
    receipt_members: list[dict[str, Any]] = []
    for position, (raw_member, variant, variant_slug) in enumerate(
        zip(
            raw_receipts,
            EXPECTED_HEALTH_VARIANT_ORDER,
            EXPECTED_HEALTH_VARIANT_SLUGS,
            strict=True,
        )
    ):
        label = f"health-gate receipt member {position}"
        member = _mapping(raw_member, label)
        _exact_keys(
            member,
            {
                "position",
                "training_variant",
                "run_name",
                "relative_path",
                "sha256",
                "schema_version",
                "recorded_at_utc",
            },
            label,
        )
        run_name = _health_run_name(
            gpu_count=gpu_count,
            variant_slug=variant_slug,
            health_source_revision=health_source_revision,
        )
        expected_path = f"output/udlm/{run_name}/pilot_exit_status.json"
        recorded_at_utc = member.get("recorded_at_utc")
        if not isinstance(recorded_at_utc, str) or not recorded_at_utc.strip():
            raise ScreenValidationError(f"{label}.recorded_at_utc must be nonempty")
        if (
            _integer(member.get("position"), f"{label}.position") != position
            or member.get("training_variant") != variant
            or member.get("run_name") != run_name
            or _relative_path(
                member.get("relative_path"),
                f"{label}.relative_path",
                suffix=".json",
            ).as_posix()
            != expected_path
            or _integer(member.get("schema_version"), f"{label}.schema_version")
            != EXPECTED_HEALTH_RECEIPT_SCHEMA_VERSION
        ):
            raise ScreenValidationError(
                f"{label} is not the deterministic {variant} receipt"
            )
        receipt_members.append(
            {
                "position": position,
                "training_variant": variant,
                "run_name": run_name,
                "relative_path": expected_path,
                "sha256": _sha256(member.get("sha256"), f"{label}.sha256"),
                "schema_version": EXPECTED_HEALTH_RECEIPT_SCHEMA_VERSION,
                "recorded_at_utc": recorded_at_utc,
            }
        )

    raw_checkpoints = evidence.get("checkpoint_members")
    if not isinstance(raw_checkpoints, list) or len(raw_checkpoints) != 3:
        raise ScreenValidationError(
            "health-gate checkpoint membership must contain exactly R, S, E"
        )
    checkpoint_members: list[dict[str, Any]] = []
    for position, (raw_member, variant, variant_slug) in enumerate(
        zip(
            raw_checkpoints,
            EXPECTED_HEALTH_VARIANT_ORDER,
            EXPECTED_HEALTH_VARIANT_SLUGS,
            strict=True,
        )
    ):
        label = f"health-gate checkpoint member {position}"
        member = _mapping(raw_member, label)
        _exact_keys(
            member,
            {
                "position",
                "training_variant",
                "run_name",
                "relative_path",
                "sha256",
                "size_bytes",
                "global_step",
            },
            label,
        )
        run_name = _health_run_name(
            gpu_count=gpu_count,
            variant_slug=variant_slug,
            health_source_revision=health_source_revision,
        )
        expected_path = (
            f"output/udlm/{run_name}/checkpoints/"
            f"{EXPECTED_HEALTH_CHECKPOINT_STEP}.ckpt"
        )
        if (
            _integer(member.get("position"), f"{label}.position") != position
            or member.get("training_variant") != variant
            or member.get("run_name") != run_name
            or _relative_path(
                member.get("relative_path"),
                f"{label}.relative_path",
                suffix=".ckpt",
            ).as_posix()
            != expected_path
            or _integer(member.get("global_step"), f"{label}.global_step")
            != EXPECTED_HEALTH_CHECKPOINT_STEP
        ):
            raise ScreenValidationError(
                f"{label} is not the deterministic step-10 {variant} checkpoint"
            )
        checkpoint_members.append(
            {
                "position": position,
                "training_variant": variant,
                "run_name": run_name,
                "relative_path": expected_path,
                "sha256": _sha256(member.get("sha256"), f"{label}.sha256"),
                "size_bytes": _integer(
                    member.get("size_bytes"), f"{label}.size_bytes", minimum=1
                ),
                "global_step": EXPECTED_HEALTH_CHECKPOINT_STEP,
            }
        )

    if (
        terminal_receipt["relative_path"] != receipt_members[-1]["relative_path"]
        or terminal_receipt["sha256"] != receipt_members[-1]["sha256"]
        or terminal_receipt["schema_version"] != receipt_members[-1]["schema_version"]
    ):
        raise ScreenValidationError(
            "health-gate terminal receipt differs from the E receipt member"
        )
    eligibility = _mapping(
        evidence.get("eligibility"), "registry health-gate eligibility"
    )
    _exact_keys(
        eligibility,
        set(EXPECTED_HEALTH_ELIGIBILITY),
        "registry health-gate eligibility",
    )
    for key, expected in EXPECTED_HEALTH_ELIGIBILITY.items():
        if (
            _boolean(eligibility.get(key), f"health-gate eligibility {key}")
            is not expected
        ):
            raise ScreenValidationError("registry health-gate eligibility is invalid")

    transition = _mapping(
        gate.get("source_transition"), "registry health source transition"
    )
    _exact_keys(
        transition,
        {
            "health_source_revision",
            "registry_source_revision",
            "allowed_config_paths",
            "health_source_is_registry_source_parent",
            "exact_config_only_transition_verified",
            "opposite_gpu_config_family_absent",
        },
        "registry health source transition",
    )
    expected_config_paths = _expected_screen_config_paths(gpu_count)
    allowed_config_paths = transition.get("allowed_config_paths")
    if not isinstance(allowed_config_paths, list) or allowed_config_paths != list(
        expected_config_paths
    ):
        raise ScreenValidationError(
            "health source transition must name the exact selected-W six configs"
        )
    registered_config_paths = [
        config["config"]["relative_path"]
        for stage in stages
        for arm in stage["arms"]
        for config in arm["resolved_configs"]
    ]
    if len(registered_config_paths) != len(expected_config_paths) or set(
        registered_config_paths
    ) != set(expected_config_paths):
        raise ScreenValidationError(
            "registered configs are not the exact selected-W six-file family"
        )
    if (
        _git_revision(
            transition.get("health_source_revision"),
            "health transition source revision",
        )
        != health_source_revision
        or _git_revision(
            transition.get("registry_source_revision"),
            "health transition registry revision",
        )
        != registry_source_revision
        or any(
            _boolean(transition.get(field), f"health transition {field}") is not True
            for field in (
                "health_source_is_registry_source_parent",
                "exact_config_only_transition_verified",
                "opposite_gpu_config_family_absent",
            )
        )
    ):
        raise ScreenValidationError("registry health source transition is invalid")

    normalized_evidence = {
        "schema_version": EXPECTED_HEALTH_SCHEMA_VERSION,
        "status": EXPECTED_HEALTH_STATUS,
        "claim_scope": EXPECTED_HEALTH_CLAIM_SCOPE,
        "health_source_revision": health_source_revision,
        "gpu_count": gpu_count,
        "matched_panel_spec_sha256": matched_panel_sha256,
        "terminal_receipt": terminal_receipt,
        "receipt_members": receipt_members,
        "checkpoint_members": checkpoint_members,
        "eligibility": dict(EXPECTED_HEALTH_ELIGIBILITY),
    }
    return {
        "evidence": normalized_evidence,
        "source_transition": {
            "health_source_revision": health_source_revision,
            "registry_source_revision": registry_source_revision,
            "allowed_config_paths": list(expected_config_paths),
            "health_source_is_registry_source_parent": True,
            "exact_config_only_transition_verified": True,
            "opposite_gpu_config_family_absent": True,
        },
    }


def validate_registry(
    registry: Mapping[str, Any],
    *,
    loader: BlobLoader,
    git_blob_loader: GitBlobLoader,
) -> Mapping[str, Any]:
    """Validate the exact prospective contract and every bound immutable input."""

    _exact_keys(
        registry,
        {
            "schema_version",
            "registry_id",
            "status",
            "claim_scope",
            "firewall",
            "source",
            "prerequisite_health_gate",
            "common_training",
            "panel",
            "stages",
        },
        "optimization-screen registry",
    )
    if (
        _integer(registry.get("schema_version"), "registry schema")
        != REGISTRY_SCHEMA_VERSION
        or registry.get("registry_id") != EXPECTED_REGISTRY_ID
        or registry.get("status") != EXPECTED_REGISTRY_STATUS
        or registry.get("claim_scope") != EXPECTED_CLAIM_SCOPE
    ):
        raise ScreenValidationError("optimization-screen registry identity is invalid")
    _validate_firewall(registry.get("firewall"))

    source = _mapping(registry.get("source"), "registry source")
    _exact_keys(source, {"revision", "clean", "pushed", "blobs"}, "registry source")
    source_revision = _git_revision(source.get("revision"), "registry source revision")
    if (
        _boolean(source.get("clean"), "registry source clean") is not True
        or _boolean(source.get("pushed"), "registry source pushed") is not True
    ):
        raise ScreenValidationError("registry source must be clean and pushed")
    source_blobs = source.get("blobs")
    if not isinstance(source_blobs, list) or not source_blobs:
        raise ScreenValidationError("registry source must bind implementation blobs")
    source_paths: set[str] = set()
    normalized_source_blobs = []
    for index, raw_ref in enumerate(source_blobs):
        label = f"registry source blob {index}"
        ref = _blob_ref(raw_ref, label, required_root="repository")
        if ref["relative_path"] in source_paths:
            raise ScreenValidationError("registry source blob paths must be unique")
        source_paths.add(ref["relative_path"])
        _load_bound_blob(ref, loader=loader, label=label)
        _verify_git_blob(
            ref,
            revision=source_revision,
            git_blob_loader=git_blob_loader,
            label=label,
        )
        normalized_source_blobs.append(ref)
    required_source_paths = {
        "scripts/train.py",
        "scripts/udlm/audit_conditioning_initialization.py",
        EXPECTED_PRIOR_FLOOR_AUDIT_SOURCE_PATH,
        EXPECTED_PRIOR_FLOOR_AUDIT_PATH,
        "scripts/udlm/collect_optimization_screen_evidence.py",
        "scripts/udlm/evaluate_denoising_panel.py",
        "scripts/udlm/launch_health_panel.py",
        "scripts/udlm/launch_optimization_screen.py",
        "scripts/udlm/launch_train_pilot.py",
        "scripts/udlm/validate_health_panel.py",
        "scripts/udlm/verify_optimization_screen.py",
        "scripts/udlm/write_pilot_evidence.py",
        "scripts/udlm/write_pilot_exit_status.py",
        "src/genmol/model.py",
        "src/genmol/backbone.py",
        "src/genmol/diffusion.py",
    } | set(EVALUATOR_SOURCE_PATHS)
    if not required_source_paths.issubset(source_paths):
        raise ScreenValidationError(
            "registry source map omits required implementation blobs"
        )
    source_by_path = {ref["relative_path"]: ref for ref in normalized_source_blobs}
    _validate_prior_floor_audit(source_by_path, loader=loader)

    common = _mapping(registry.get("common_training"), "registry common training")
    _exact_keys(
        common,
        {
            "training_seed",
            "gpu_count",
            "prior_variant",
            "global_batch_size",
            "micro_batch_size_per_process",
            "accumulate_grad_batches",
            "effective_global_batch_size",
            "initialization",
            "artifact_schema_versions",
        },
        "registry common training",
    )
    if common.get("training_seed") != EXPECTED_TRAINING_SEED:
        raise ScreenValidationError("screen training seed must equal 17")
    gpu_count = _integer(common.get("gpu_count"), "screen GPU count")
    if gpu_count not in {1, 2}:
        raise ScreenValidationError("screen GPU count must equal 1 or 2")
    if common.get("prior_variant") != "empirical_frequency":
        raise ScreenValidationError("optimization screens must use E only")
    global_batch = _integer(
        common.get("global_batch_size"), "screen global batch", minimum=1
    )
    micro_batch = _integer(
        common.get("micro_batch_size_per_process"),
        "screen micro batch",
        minimum=1,
    )
    accumulation = _integer(
        common.get("accumulate_grad_batches"), "screen accumulation", minimum=1
    )
    effective_batch = _integer(
        common.get("effective_global_batch_size"),
        "screen effective global batch",
        minimum=1,
    )
    if (
        effective_batch != micro_batch * gpu_count * accumulation
        or global_batch != effective_batch
    ):
        raise ScreenValidationError("screen batch arithmetic is inconsistent")
    versions = _mapping(
        common.get("artifact_schema_versions"), "screen artifact schema versions"
    )
    _exact_keys(
        versions,
        set(EXPECTED_ARTIFACT_SCHEMA_VERSIONS),
        "screen artifact schema versions",
    )
    if dict(versions) != EXPECTED_ARTIFACT_SCHEMA_VERSIONS:
        raise ScreenValidationError("screen artifact schema versions are unsupported")
    initialization = _mapping(common.get("initialization"), "screen initialization")
    _exact_keys(
        initialization,
        {
            "mode",
            "checkpoint",
            "weights",
            "optimizer_reset",
            "scheduler_reset",
            "global_step_reset",
            "ema_reset",
        },
        "screen initialization",
    )
    if (
        initialization.get("mode") != "fresh_independent_mdlm_ema_warm_start_each_arm"
        or initialization.get("weights") != "ema"
        or any(
            _boolean(initialization.get(field), f"screen initialization {field}")
            is not True
            for field in (
                "optimizer_reset",
                "scheduler_reset",
                "global_step_reset",
                "ema_reset",
            )
        )
    ):
        raise ScreenValidationError("screen warm-start/reset contract is invalid")
    checkpoint_ref = _blob_ref(
        initialization.get("checkpoint"),
        "screen initialization checkpoint",
        required_root="project",
        suffix=".ckpt",
    )
    _load_bound_blob(
        checkpoint_ref, loader=loader, label="screen initialization checkpoint"
    )

    panel = _mapping(registry.get("panel"), "registry panel")
    _exact_keys(
        panel,
        {
            "artifact",
            "ordered_token_ids_sha256",
            "rows",
            "content_tokens_per_time_bin",
            "time_bins",
            "corruption_seed",
            "weights",
            "device",
            "batch_size",
            "frequency_artifact",
            "frequency_ordered_text_sha256",
            "evaluator_report_schema_version",
            "evaluator_source",
        },
        "registry panel",
    )
    panel_ref = _json_ref(
        panel.get("artifact"), "registry panel artifact", required_root="repository"
    )
    if (
        panel_ref["relative_path"] != EXPECTED_PANEL_PATH
        or panel_ref["sha256"] != EXPECTED_PANEL_SHA256
        or panel_ref["schema_version"] != 1
        or panel.get("ordered_token_ids_sha256") != EXPECTED_PANEL_TOKEN_IDS_SHA256
        or panel.get("rows") != EXPECTED_PANEL_ROWS
        or panel.get("content_tokens_per_time_bin") != EXPECTED_PANEL_CONTENT_TOKENS
        or panel.get("corruption_seed") != EXPECTED_CORRUPTION_SEED
        or panel.get("weights") != "ema"
        or panel.get("device") != "cpu"
        or panel.get("frequency_ordered_text_sha256")
        != EXPECTED_FREQUENCY_ORDERED_TEXT_SHA256
        or panel.get("evaluator_report_schema_version")
        != EVALUATOR_REPORT_SCHEMA_VERSION
    ):
        raise ScreenValidationError("registry fixed-panel contract is invalid")
    batch_size = _integer(
        panel.get("batch_size"), "registry panel batch size", minimum=1
    )
    del batch_size
    observed_times = panel.get("time_bins")
    if (
        not isinstance(observed_times, list)
        or tuple(
            _decimal(value, f"registry panel time {index}")
            for index, value in enumerate(observed_times)
        )
        != EXPECTED_TIME_BINS
    ):
        raise ScreenValidationError("registry panel time bins are invalid")
    parsed_panel = _load_json_ref(
        panel_ref, loader=loader, label="registry panel artifact"
    )
    rows = parsed_panel.get("rows")
    if (
        parsed_panel.get("sample_count") != EXPECTED_PANEL_ROWS
        or parsed_panel.get("ordered_token_ids_sha256")
        != EXPECTED_PANEL_TOKEN_IDS_SHA256
        or not isinstance(rows, list)
        or len(rows) != EXPECTED_PANEL_ROWS
        or any(not isinstance(row, Mapping) for row in rows)
        or sum(
            _integer(row.get("content_length"), "panel row content length", minimum=1)
            for row in rows
        )
        != EXPECTED_PANEL_CONTENT_TOKENS
    ):
        raise ScreenValidationError("bound panel content is invalid")
    _verify_git_blob(
        panel_ref,
        revision=source_revision,
        git_blob_loader=git_blob_loader,
        label="registry panel artifact",
    )
    frequency_ref = _json_ref(
        panel.get("frequency_artifact"),
        "registry frequency artifact",
        required_root="repository",
    )
    if (
        frequency_ref["relative_path"] != EXPECTED_FREQUENCY_PATH
        or frequency_ref["sha256"] != EXPECTED_FREQUENCY_SHA256
        or frequency_ref["schema_version"] != 1
    ):
        raise ScreenValidationError("registry frequency-artifact contract is invalid")
    _load_json_ref(frequency_ref, loader=loader, label="registry frequency artifact")
    _verify_git_blob(
        frequency_ref,
        revision=source_revision,
        git_blob_loader=git_blob_loader,
        label="registry frequency artifact",
    )
    evaluator_ref = _blob_ref(
        panel.get("evaluator_source"),
        "registry panel evaluator source",
        required_root="repository",
        suffix=".py",
    )
    if evaluator_ref["relative_path"] != "scripts/udlm/evaluate_denoising_panel.py":
        raise ScreenValidationError("registry evaluator source path is invalid")
    _load_bound_blob(
        evaluator_ref, loader=loader, label="registry panel evaluator source"
    )
    _verify_git_blob(
        evaluator_ref,
        revision=source_revision,
        git_blob_loader=git_blob_loader,
        label="registry panel evaluator source",
    )
    if source_by_path.get(evaluator_ref["relative_path"]) != evaluator_ref:
        raise ScreenValidationError(
            "panel evaluator is not identical to its source binding"
        )

    stages = registry.get("stages")
    if not isinstance(stages, list) or len(stages) != 2:
        raise ScreenValidationError("registry must contain exactly two ordered stages")
    normalized_stages = [
        _validate_registry_stage(
            stage,
            index=index,
            loader=loader,
            git_blob_loader=git_blob_loader,
            source_revision=source_revision,
            gpu_count=gpu_count,
            initialization_checkpoint=checkpoint_ref,
            global_batch_size=global_batch,
            micro_batch_size=micro_batch,
            accumulation=accumulation,
        )
        for index, stage in enumerate(stages)
    ]
    all_attempts = [
        arm["attempt_id"] for stage in normalized_stages for arm in stage["arms"]
    ]
    all_outputs = [
        config["output_directory"]
        for stage in normalized_stages
        for arm in stage["arms"]
        for config in arm["resolved_configs"]
    ]
    if len(all_attempts) != len(set(all_attempts)):
        raise ScreenValidationError("registry attempt IDs must be globally unique")
    if len(all_outputs) != len(set(all_outputs)):
        raise ScreenValidationError(
            "registry output directories must be globally unique"
        )
    _validate_matched_config_pairs(normalized_stages)
    prerequisite_health_gate = _validate_prerequisite_health_gate(
        registry.get("prerequisite_health_gate"),
        loader=loader,
        registry_source_revision=source_revision,
        gpu_count=gpu_count,
        stages=normalized_stages,
    )
    return {
        **dict(registry),
        "source": {**dict(source), "blobs": normalized_source_blobs},
        "prerequisite_health_gate": prerequisite_health_gate,
        "common_training": {
            **dict(common),
            "initialization": {**dict(initialization), "checkpoint": checkpoint_ref},
        },
        "panel": {
            **dict(panel),
            "artifact": panel_ref,
            "frequency_artifact": frequency_ref,
            "evaluator_source": evaluator_ref,
        },
        "stages": normalized_stages,
    }


def _without_config_paths(
    config: Mapping[str, Any], paths: Sequence[tuple[str, ...]]
) -> Mapping[str, Any]:
    normalized = copy.deepcopy(dict(config))
    for path in paths:
        parent: object = normalized
        for key in path[:-1]:
            if not isinstance(parent, dict):
                break
            parent = parent.get(key)
        if isinstance(parent, dict):
            parent.pop(path[-1], None)
    return normalized


def _validate_matched_config_pairs(stages: Sequence[Mapping[str, Any]]) -> None:
    scheduler_stage, conditioning_stage = stages
    scheduler_allowed = (
        ("optim", "scheduler"),
        ("callback", "dirpath"),
    )
    scheduler_configs = [arm["resolved_configs"][0] for arm in scheduler_stage["arms"]]
    if not _exact_json_equal(
        _without_config_paths(scheduler_configs[0]["parsed_config"], scheduler_allowed),
        _without_config_paths(scheduler_configs[1]["parsed_config"], scheduler_allowed),
    ):
        raise ScreenValidationError(
            "scheduler configs differ outside the registered bundle"
        )
    conditioning_allowed = (
        ("training", "udlm", "conditioning_variant"),
        ("training", "udlm", "zero_init_conditioning"),
        ("callback", "dirpath"),
    )
    for scheduler_index in range(2):
        reference = conditioning_stage["arms"][0]["resolved_configs"][scheduler_index]
        candidate = conditioning_stage["arms"][1]["resolved_configs"][scheduler_index]
        if not _exact_json_equal(
            _without_config_paths(reference["parsed_config"], conditioning_allowed),
            _without_config_paths(candidate["parsed_config"], conditioning_allowed),
        ):
            raise ScreenValidationError(
                "conditioning configs differ outside the registered conditioner"
            )


def _default_health_gate_validator(
    terminal_receipt_path: Path,
    *,
    expected_gpu_count: int,
    expected_source_revision: str,
) -> Mapping[str, Any]:
    """Import the producer validator only when a registry is actually loaded."""

    import sys

    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    health_module = importlib.import_module("scripts.udlm.validate_health_panel")
    return health_module.validate_health_panel(
        terminal_receipt_path,
        expected_gpu_count=expected_gpu_count,
        expected_source_revision=expected_source_revision,
    )


def _load_git_tree_paths(
    *,
    revision: str,
    directory: PurePosixPath,
    git_tree_paths_loader: GitTreePathsLoader,
    label: str,
) -> frozenset[str]:
    """Load and type-check the complete committed blob set below a directory."""

    try:
        paths = git_tree_paths_loader(revision, directory)
    except ScreenValidationError:
        raise
    except Exception as error:
        raise ScreenValidationError(f"cannot enumerate {label}") from error
    if not isinstance(paths, frozenset) or any(
        not isinstance(path, str) or not path for path in paths
    ):
        raise ScreenValidationError(f"{label} Git tree loader returned invalid paths")
    prefix = directory.as_posix() + "/"
    if any(not path.startswith(prefix) for path in paths):
        raise ScreenValidationError(f"{label} Git tree loader escaped its directory")
    return paths


def _revalidate_health_gate_boundary(
    registry: Mapping[str, Any],
    *,
    git_ancestor_checker: GitAncestorChecker,
    git_sole_parent_checker: GitSoleParentChecker,
    git_tree_paths_loader: GitTreePathsLoader,
    git_diff_checker: GitDiffChecker,
    health_gate_validator: HealthGateValidator,
) -> None:
    """Replay the H-to-R0 Git boundary and the live health validator."""

    gate = registry["prerequisite_health_gate"]
    evidence = gate["evidence"]
    transition = gate["source_transition"]
    health_revision = evidence["health_source_revision"]
    registry_revision = registry["source"]["revision"]
    allowed_paths = frozenset(transition["allowed_config_paths"])
    if health_revision == registry_revision:
        raise ScreenValidationError(
            "health source and registry source must be distinct revisions"
        )
    try:
        health_is_sole_parent = git_sole_parent_checker(
            registry_revision, health_revision
        )
    except Exception as error:
        raise ScreenValidationError("health-to-R0 sole-parent check failed") from error
    if not health_is_sole_parent:
        raise ScreenValidationError(
            "health source is not the sole immediate parent of R0"
        )
    try:
        health_is_ancestor = git_ancestor_checker(health_revision, registry_revision)
    except Exception as error:
        raise ScreenValidationError("health-to-R0 ancestry check failed") from error
    if not health_is_ancestor:
        raise ScreenValidationError("health source is not an ancestor of R0")
    try:
        config_only_transition = git_diff_checker(
            health_revision, registry_revision, allowed_paths
        )
    except Exception as error:
        raise ScreenValidationError("health-to-R0 diff check failed") from error
    if not config_only_transition:
        raise ScreenValidationError(
            "health-to-R0 diff is not limited to the selected-W six configs"
        )
    selected_directory = PurePosixPath(
        EXPECTED_SCREEN_CONFIG_DIRECTORY_TEMPLATE.format(
            gpu_count=evidence["gpu_count"]
        )
    )
    selected_at_health = _load_git_tree_paths(
        revision=health_revision,
        directory=selected_directory,
        git_tree_paths_loader=git_tree_paths_loader,
        label="selected-W config family at health source",
    )
    selected_at_r0 = _load_git_tree_paths(
        revision=registry_revision,
        directory=selected_directory,
        git_tree_paths_loader=git_tree_paths_loader,
        label="selected-W config family at R0",
    )
    if selected_at_health:
        raise ScreenValidationError(
            "selected-W config family unexpectedly exists at health source"
        )
    if selected_at_r0 != allowed_paths:
        raise ScreenValidationError(
            "selected-W config family at R0 is not the exact six-file set"
        )
    other_gpu_count = 2 if evidence["gpu_count"] == 1 else 1
    other_directory = PurePosixPath(
        EXPECTED_SCREEN_CONFIG_DIRECTORY_TEMPLATE.format(gpu_count=other_gpu_count)
    )
    for revision, revision_label in (
        (health_revision, "health source"),
        (registry_revision, "R0"),
    ):
        if _load_git_tree_paths(
            revision=revision,
            directory=other_directory,
            git_tree_paths_loader=git_tree_paths_loader,
            label=f"opposite-W config family at {revision_label}",
        ):
            raise ScreenValidationError(
                f"opposite-W config family unexpectedly exists at {revision_label}"
            )

    terminal_ref = evidence["terminal_receipt"]
    terminal_path = _root_path(
        terminal_ref["root"], PurePosixPath(terminal_ref["relative_path"])
    )
    try:
        live_evidence = health_gate_validator(
            terminal_path,
            expected_gpu_count=evidence["gpu_count"],
            expected_source_revision=health_revision,
        )
    except Exception as error:
        raise ScreenValidationError("live health-gate validation failed") from error
    if not isinstance(live_evidence, Mapping) or not _exact_json_equal(
        dict(live_evidence), evidence
    ):
        raise ScreenValidationError(
            "live health-gate evidence differs from the frozen registry"
        )


def load_validated_registry(
    payload: bytes,
    *,
    relative_path: str,
    expected_raw_sha256: str,
    expected_canonical_sha256: str,
    loader: BlobLoader,
    git_blob_loader: GitBlobLoader,
    git_ancestor_checker: GitAncestorChecker,
    git_sole_parent_checker: GitSoleParentChecker,
    git_tree_paths_loader: GitTreePathsLoader,
    git_pushed_checker: GitPushedChecker,
    git_diff_checker: GitDiffChecker,
    health_gate_validator: HealthGateValidator | None = None,
) -> ValidatedRegistry:
    raw_sha = hashlib.sha256(payload).hexdigest()
    if raw_sha != _sha256(expected_raw_sha256, "expected registry raw digest"):
        raise ScreenValidationError("registry raw digest differs from its frozen pin")
    parsed = _mapping(
        strict_json_loads(payload, label="optimization-screen registry"),
        "optimization-screen registry",
    )
    canonical_sha = canonical_json_sha256(parsed)
    if canonical_sha != _sha256(
        expected_canonical_sha256, "expected registry canonical digest"
    ):
        raise ScreenValidationError(
            "registry canonical digest differs from its frozen pin"
        )
    normalized = validate_registry(
        parsed, loader=loader, git_blob_loader=git_blob_loader
    )
    if not git_pushed_checker(normalized["source"]["revision"]):
        raise ScreenValidationError("registry implementation revision is not pushed")
    _revalidate_health_gate_boundary(
        normalized,
        git_ancestor_checker=git_ancestor_checker,
        git_sole_parent_checker=git_sole_parent_checker,
        git_tree_paths_loader=git_tree_paths_loader,
        git_diff_checker=git_diff_checker,
        health_gate_validator=(
            _default_health_gate_validator
            if health_gate_validator is None
            else health_gate_validator
        ),
    )
    return ValidatedRegistry(
        data=normalized,
        relative_path=_relative_path(relative_path, "registry path", suffix=".json"),
        raw_sha256=raw_sha,
        raw_size_bytes=len(payload),
        canonical_sha256=canonical_sha,
        git_blob_loader=git_blob_loader,
        git_ancestor_checker=git_ancestor_checker,
        git_sole_parent_checker=git_sole_parent_checker,
        git_tree_paths_loader=git_tree_paths_loader,
        git_pushed_checker=git_pushed_checker,
        git_diff_checker=git_diff_checker,
    )


def _registry_reference(value: object, registry: ValidatedRegistry) -> None:
    reference = _mapping(value, "evidence registry reference")
    _exact_keys(reference, set(registry.reference), "evidence registry reference")
    if dict(reference) != registry.reference:
        raise ScreenValidationError("evidence does not bind the frozen registry")


def _stage(registry: ValidatedRegistry, stage_id: str) -> Mapping[str, Any]:
    if stage_id not in EXPECTED_STAGE_ORDER:
        raise ScreenValidationError("stage ID is unsupported")
    return registry.data["stages"][EXPECTED_STAGE_ORDER.index(stage_id)]


def _arm(stage: Mapping[str, Any], arm_id: str) -> Mapping[str, Any]:
    for arm in stage["arms"]:
        if arm["arm_id"] == arm_id:
            return arm
    raise ScreenValidationError(f"arm {arm_id!r} is not registered")


def _registered_config(
    arm: Mapping[str, Any], *, scheduler_arm_id: str | None
) -> Mapping[str, Any]:
    for entry in arm["resolved_configs"]:
        if entry["scheduler_arm_id"] == scheduler_arm_id:
            return entry
    raise ScreenValidationError("attempt does not have a registered contingent config")


def _screen_binding(
    registry: ValidatedRegistry,
    *,
    stage_id: str,
    arm: Mapping[str, Any],
    scheduler_dependency: Mapping[str, Any] | None,
) -> dict[str, Any]:
    launcher_ref = next(
        ref
        for ref in registry.data["source"]["blobs"]
        if ref["relative_path"] == "scripts/udlm/launch_optimization_screen.py"
    )
    gradient_contract = registry.data["stages"][1]["gradient_contract"]
    gradient_contract_sha = registry.data["stages"][1]["gradient_contract_sha256"]
    return {
        "registry_relative_path": registry.relative_path.as_posix(),
        "registry_sha256": registry.raw_sha256,
        "registry_canonical_sha256": registry.canonical_sha256,
        "stage_id": stage_id,
        "arm_id": arm["arm_id"],
        "attempt_id": arm["attempt_id"],
        "launcher_source_sha256": launcher_ref["sha256"],
        "scheduler_authorization": (
            dict(scheduler_dependency) if stage_id == "conditioning" else None
        ),
        "conditioning_gradient_contract": (
            gradient_contract if arm["arm_id"] == "E-A1" else None
        ),
        "conditioning_gradient_contract_sha256": (
            gradient_contract_sha if arm["arm_id"] == "E-A1" else None
        ),
    }


def _artifact_set(value: object, label: str) -> dict[str, Mapping[str, Any]]:
    artifacts = _mapping(value, label)
    _exact_keys(
        artifacts,
        {"launch_manifest", "runtime_config", "training_summary", "exit_receipt"},
        label,
    )
    return {
        name: _json_ref(ref, f"{label}.{name}", required_root="repository")
        for name, ref in artifacts.items()
    }


def _require_owned_artifacts(
    refs: Sequence[Mapping[str, Any]], *, output_directory: str, label: str
) -> None:
    paths = [ref["relative_path"] for ref in refs]
    prefix = output_directory.rstrip("/") + "/"
    if any(not path.startswith(prefix) for path in paths):
        raise ScreenValidationError(f"{label} escapes its registered output directory")
    if len(paths) != len(set(paths)):
        raise ScreenValidationError(f"{label} reuses one path for distinct artifacts")


def _selected_uuids(value: object, label: str, expected_count: int) -> list[str]:
    if not isinstance(value, list) or len(value) != expected_count:
        raise ScreenValidationError(
            f"{label} must contain exactly {expected_count} UUIDs"
        )
    if any(
        not isinstance(item, str) or not item.startswith("GPU-") or "," in item
        for item in value
    ):
        raise ScreenValidationError(f"{label} contains an invalid UUID")
    if len(set(value)) != len(value):
        raise ScreenValidationError(f"{label} contains duplicate UUIDs")
    return value


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
_GPU_PROCESS_KEYS = {"pid", "process_name", "used_memory_mib"}
_GPU_SAFETY_POLICY = {
    "max_utilization_percent": MAX_SAFE_UTILIZATION_PERCENT,
    "utilization_comparison": "strictly_less_than",
    "min_free_memory_mib": MIN_SAFE_FREE_MEMORY_MIB,
    "active_compute_processes_allowed": ACTIVE_COMPUTE_PROCESSES_ALLOWED,
    "compute_mode_prohibited_allowed": False,
}


def _utc_timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ScreenValidationError(f"{label} must be a nonempty UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ScreenValidationError(f"{label} is not valid ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ScreenValidationError(f"{label} must use UTC")
    return parsed


def _validate_gpu_state(value: object, label: str) -> dict[str, Any]:
    state = _mapping(value, label)
    _exact_keys(state, _GPU_STATE_KEYS, label)
    physical_index = _integer(
        state.get("physical_index"), f"{label}.physical_index", minimum=0
    )
    uuid = _selected_uuids([state.get("uuid")], f"{label}.uuid", 1)[0]
    name = state.get("name")
    if not isinstance(name, str) or not name:
        raise ScreenValidationError(f"{label}.name must be nonempty")
    gpu_memory_used = _integer(
        state.get("memory_used_mib"), f"{label}.memory_used_mib", minimum=0
    )
    memory_total = _integer(
        state.get("memory_total_mib"), f"{label}.memory_total_mib", minimum=1
    )
    if gpu_memory_used > memory_total:
        raise ScreenValidationError(f"{label} used memory exceeds total memory")
    utilization = _integer(
        state.get("utilization_percent"),
        f"{label}.utilization_percent",
        minimum=0,
        maximum=100,
    )
    compute_mode = state.get("compute_mode")
    if not isinstance(compute_mode, str) or not compute_mode.strip():
        raise ScreenValidationError(f"{label}.compute_mode must be nonempty")
    processes = state.get("compute_processes")
    if not isinstance(processes, list):
        raise ScreenValidationError(f"{label}.compute_processes must be an array")
    normalized_processes: list[dict[str, Any]] = []
    for index, value in enumerate(processes):
        process_label = f"{label}.compute_processes[{index}]"
        process = _mapping(value, process_label)
        _exact_keys(process, _GPU_PROCESS_KEYS, process_label)
        pid = _integer(process.get("pid"), f"{process_label}.pid", minimum=1)
        process_name = process.get("process_name")
        if not isinstance(process_name, str) or not process_name:
            raise ScreenValidationError(
                f"{process_label}.process_name must be nonempty"
            )
        process_memory_used = process.get("used_memory_mib")
        if process_memory_used is not None:
            process_memory_used = _integer(
                process_memory_used,
                f"{process_label}.used_memory_mib",
                minimum=0,
            )
        normalized_processes.append(
            {
                "pid": pid,
                "process_name": process_name,
                "used_memory_mib": process_memory_used,
            }
        )
    return {
        "physical_index": physical_index,
        "uuid": uuid,
        "name": name,
        "memory_used_mib": gpu_memory_used,
        "memory_total_mib": memory_total,
        "utilization_percent": utilization,
        "compute_mode": compute_mode,
        "compute_processes": normalized_processes,
    }


def _validate_launch_gpu_evidence(
    manifest: Mapping[str, Any], *, selected_uuids: list[str]
) -> None:
    expected_count = len(selected_uuids)
    if manifest.get("gpu_selection_schema_version") != 2:
        raise ScreenValidationError("launch manifest GPU-selection schema is invalid")
    if (
        manifest.get("gpu_selection_method") != "dynamic_idle_discovery"
        or manifest.get("gpu_inventory_scope") != "all_nvidia_gpus"
    ):
        raise ScreenValidationError("launch manifest GPU selection method is invalid")

    safety = _mapping(manifest.get("gpu_safety_policy"), "launch GPU safety policy")
    _exact_keys(safety, set(_GPU_SAFETY_POLICY), "launch GPU safety policy")
    max_utilization = _integer(
        safety.get("max_utilization_percent"),
        "launch GPU maximum utilization",
        minimum=1,
        maximum=MAX_SAFE_UTILIZATION_PERCENT,
    )
    min_free_memory = _integer(
        safety.get("min_free_memory_mib"),
        "launch GPU minimum free memory",
        minimum=MIN_SAFE_FREE_MEMORY_MIB,
    )
    if (
        safety.get("utilization_comparison") != "strictly_less_than"
        or safety.get("active_compute_processes_allowed")
        is not ACTIVE_COMPUTE_PROCESSES_ALLOWED
        or safety.get("compute_mode_prohibited_allowed") is not False
    ):
        raise ScreenValidationError("launch GPU safety policy is not exact")

    inventory_time = _utc_timestamp(
        manifest.get("inventory_snapshot_completed_at_utc"),
        "launch inventory completion timestamp",
    )
    final_probe_time = _utc_timestamp(
        manifest.get("final_uuid_probes_completed_at_utc"),
        "launch final-probe completion timestamp",
    )
    manifest_time = _utc_timestamp(
        manifest.get("created_at"), "launch manifest creation timestamp"
    )
    if not inventory_time <= final_probe_time <= manifest_time:
        raise ScreenValidationError(
            "launch GPU-probe/manifest timestamps are out of order"
        )

    inventory_raw = manifest.get("gpu_inventory_at_selection")
    initial_raw = manifest.get("initially_selected_gpu_states")
    final_raw = manifest.get("gpu_states_at_final_uuid_probe")
    if not isinstance(inventory_raw, list) or not inventory_raw:
        raise ScreenValidationError("launch GPU inventory must be nonempty")
    if not isinstance(initial_raw, list) or len(initial_raw) != expected_count:
        raise ScreenValidationError("launch initial GPU selection is incomplete")
    if not isinstance(final_raw, list) or len(final_raw) != expected_count:
        raise ScreenValidationError("launch final GPU probe is incomplete")
    inventory = [
        _validate_gpu_state(value, f"launch inventory GPU {index}")
        for index, value in enumerate(inventory_raw)
    ]
    initial = [
        _validate_gpu_state(value, f"launch initial GPU {index}")
        for index, value in enumerate(initial_raw)
    ]
    final = [
        _validate_gpu_state(value, f"launch final GPU {index}")
        for index, value in enumerate(final_raw)
    ]
    inventory_uuids = [state["uuid"] for state in inventory]
    inventory_indices = [state["physical_index"] for state in inventory]
    if len(set(inventory_uuids)) != len(inventory_uuids) or len(
        set(inventory_indices)
    ) != len(inventory_indices):
        raise ScreenValidationError("launch GPU inventory identities are not unique")
    inventory_by_uuid = {state["uuid"]: state for state in inventory}
    if any(uuid not in inventory_by_uuid for uuid in selected_uuids):
        raise ScreenValidationError("launch selected UUID is absent from inventory")
    if [state["uuid"] for state in initial] != selected_uuids:
        raise ScreenValidationError("launch initial GPU UUID order is unmatched")
    if [state["uuid"] for state in final] != selected_uuids:
        raise ScreenValidationError("launch final GPU UUID order is unmatched")
    if initial != [inventory_by_uuid[uuid] for uuid in selected_uuids]:
        raise ScreenValidationError("launch initial GPU state differs from inventory")

    final_indices = [state["physical_index"] for state in final]
    if len(set(final_indices)) != len(final_indices):
        raise ScreenValidationError("launch final GPU indices are not unique")
    if manifest.get("physical_gpu_indices") != final_indices:
        raise ScreenValidationError("launch physical GPU mapping is unmatched")
    if manifest.get("logical_cuda_devices") != list(range(expected_count)):
        raise ScreenValidationError("launch logical GPU mapping is unmatched")

    for index, state in enumerate((*initial, *final)):
        if (
            state["utilization_percent"] >= max_utilization
            or state["memory_total_mib"] - state["memory_used_mib"] < min_free_memory
            or state["compute_mode"].strip().lower() == "prohibited"
            or (state["compute_processes"] and not ACTIVE_COMPUTE_PROCESSES_ALLOWED)
        ):
            raise ScreenValidationError(
                f"launch selected GPU state {index} violates the safety policy"
            )


def _validate_launch_manifest(
    manifest: Mapping[str, Any],
    *,
    registry: ValidatedRegistry,
    stage_id: str,
    arm: Mapping[str, Any],
    config_entry: Mapping[str, Any],
    source_revision: str,
    scheduler_dependency: Mapping[str, Any] | None,
) -> list[str]:
    if (
        _integer(
            manifest.get("launch_manifest_schema_version"), "launch manifest schema"
        )
        != EXPECTED_ARTIFACT_SCHEMA_VERSIONS["launch_manifest"]
    ):
        raise ScreenValidationError("launch manifest schema is unsupported")
    if manifest.get("purpose") != "registered UDLM optimization screen":
        raise ScreenValidationError(
            "launch manifest purpose is not a registered screen"
        )
    if manifest.get("git_sha") != source_revision:
        raise ScreenValidationError("launch manifest source revision is unmatched")
    if manifest.get("optimization_screen") != _screen_binding(
        registry,
        stage_id=stage_id,
        arm=arm,
        scheduler_dependency=scheduler_dependency,
    ):
        raise ScreenValidationError("launch manifest screen binding is unmatched")
    if manifest.get("seed") != EXPECTED_TRAINING_SEED:
        raise ScreenValidationError("launch manifest seed is unmatched")
    if manifest.get("max_steps") != EXPECTED_UPDATES[stage_id]:
        raise ScreenValidationError("launch manifest update count is unmatched")
    if manifest.get("udlm_prior_variant") != "empirical_frequency":
        raise ScreenValidationError("launch manifest does not describe the E prior")
    if (
        manifest.get("resolved_training_config_sha256")
        != config_entry["config"]["canonical_sha256"]
    ):
        raise ScreenValidationError("launch manifest resolved config is unmatched")
    expected_gpu_count = registry.data["common_training"]["gpu_count"]
    if manifest.get("user_requested_gpu_count") != expected_gpu_count:
        raise ScreenValidationError("launch manifest GPU count is unmatched")
    selected_uuids = _selected_uuids(
        manifest.get("cuda_visible_device_uuids"),
        "launch manifest selected GPU UUIDs",
        expected_gpu_count,
    )
    if manifest.get("source_revision_before_final_gpu_probe") != source_revision:
        raise ScreenValidationError("launch final-probe source revision is unmatched")
    _validate_launch_gpu_evidence(manifest, selected_uuids=selected_uuids)
    return selected_uuids


def _absolute_artifact_path(
    value: object,
    *,
    label: str,
    expected_ref: Mapping[str, Any] | None = None,
    suffix: str | None = None,
) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or "\\" in value
        or value.startswith("//")
    ):
        raise ScreenValidationError(f"{label} must be a nonempty absolute POSIX path")
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts[1:])
    ):
        raise ScreenValidationError(f"{label} must be normalized and absolute")
    if suffix is not None and path.suffix != suffix:
        raise ScreenValidationError(f"{label} must end in {suffix}")
    if expected_ref is not None:
        relative = expected_ref.get("relative_path")
        if not isinstance(relative, str) or not value.endswith(f"/{relative}"):
            raise ScreenValidationError(f"{label} is not bound to its artifact ref")
    return value


def _validate_source_record(
    value: object, *, label: str, source_revision: str
) -> dict[str, Any]:
    source = _mapping(value, label)
    _exact_keys(source, {"head", "upstream"}, label)
    if any(source.get(key) != source_revision for key in ("head", "upstream")):
        raise ScreenValidationError(f"{label} is not bound to the run revision")
    return dict(source)


def _validate_stable_snapshot(
    value: object,
    *,
    label: str,
    expected_path: str,
    expected_ref: Mapping[str, Any] | None = None,
    extra_keys: set[str] | None = None,
) -> dict[str, Any]:
    snapshot = _mapping(value, label)
    extras = set() if extra_keys is None else extra_keys
    _exact_keys(snapshot, _STABLE_ARTIFACT_SNAPSHOT_KEYS | extras, label)
    if snapshot.get("path") != expected_path:
        raise ScreenValidationError(f"{label} path is unmatched")
    numeric_fields = (
        ("device", 0),
        ("inode", 1),
        ("mode", 1),
        ("link_count", 1),
        ("size_bytes", 1),
        ("mtime_ns", 0),
        ("ctime_ns", 0),
    )
    for key, minimum in numeric_fields:
        _integer(snapshot.get(key), f"{label}.{key}", minimum=minimum)
    if not stat.S_ISREG(snapshot["mode"]) or snapshot["link_count"] != 1:
        raise ScreenValidationError(f"{label} is not a single-link regular file")
    _sha256(snapshot.get("sha256"), f"{label}.sha256")
    if (
        _boolean(
            snapshot.get("stable_regular_file_verified"),
            f"{label}.stable_regular_file_verified",
        )
        is not True
    ):
        raise ScreenValidationError(f"{label} is not a verified stable file")
    if expected_ref is not None and (
        snapshot.get("sha256") != expected_ref.get("sha256")
        or snapshot.get("size_bytes") != expected_ref.get("size_bytes")
    ):
        raise ScreenValidationError(f"{label} content identity is unmatched")
    return dict(snapshot)


def _snapshot_base(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in _STABLE_ARTIFACT_SNAPSHOT_KEYS}


def _validate_completion_contract(
    value: object,
    *,
    label: str,
    manifest: Mapping[str, Any],
    expected_steps: int,
    expected_world_size: int,
) -> dict[str, Any]:
    completion = _mapping(value, label)
    _exact_keys(
        completion,
        {
            "summary_schema_version",
            "summary_path",
            "final_checkpoint_path",
            "expected_max_steps",
            "expected_world_size",
            "fail_on_nonfinite_loss",
            "backward_anomaly_detection",
        },
        label,
    )
    expected_values = {
        "summary_schema_version": EXPECTED_ARTIFACT_SCHEMA_VERSIONS["training_summary"],
        "summary_path": manifest.get("training_summary_path"),
        "final_checkpoint_path": manifest.get("expected_final_checkpoint_path"),
        "expected_max_steps": expected_steps,
        "expected_world_size": expected_world_size,
        "fail_on_nonfinite_loss": True,
        "backward_anomaly_detection": True,
    }
    if any(
        not _exact_json_equal(completion.get(key), expected)
        for key, expected in expected_values.items()
    ):
        raise ScreenValidationError(f"{label} is unmatched")
    return dict(completion)


def _validate_launch_completion_contract(value: object) -> None:
    label = "launch completion contract"
    completion = _mapping(value, label)
    _exact_keys(
        completion,
        {
            "status_at_launch",
            "valid_training_summary_and_successful_exit_receipt_both_required",
            "complete_only_if_valid_training_summary_exists",
            "complete_only_if_successful_exit_receipt_exists",
            "absent_exit_receipt_means",
            "missing_summary_after_tmux_exit_means",
            "successful_exit_receipt_requires",
            "training_job_lock_release",
        },
        label,
    )
    requirements = _mapping(
        completion.get("successful_exit_receipt_requires"),
        f"{label}.successful_exit_receipt_requires",
    )
    expected_requirements = {
        "training_exit_status": 0,
        "tee_exit_status": 0,
        "valid_launch_bound_training_summary": True,
        "exact_launch_manifest_still_matches": True,
        "clean_pushed_source_at_receipt": True,
    }
    if not _exact_json_equal(requirements, expected_requirements):
        raise ScreenValidationError(f"{label} requirements are unmatched")
    expected = {
        "status_at_launch": "pending",
        "valid_training_summary_and_successful_exit_receipt_both_required": True,
        "complete_only_if_valid_training_summary_exists": True,
        "complete_only_if_successful_exit_receipt_exists": True,
        "absent_exit_receipt_means": "incomplete",
        "missing_summary_after_tmux_exit_means": "incomplete",
        "successful_exit_receipt_requires": expected_requirements,
        "training_job_lock_release": (
            "after_exit_receipt_publication_for_completed_or_failed_pipeline"
        ),
    }
    if not _exact_json_equal(completion, expected):
        raise ScreenValidationError(f"{label} is unmatched")


def _validate_training_job_lock_binding(
    value: object,
    *,
    manifest: Mapping[str, Any],
    source_revision: str,
) -> dict[str, Any]:
    label = "launch training-job lock binding"
    binding = _mapping(value, label)
    _exact_keys(
        binding,
        {
            "path",
            "sha256",
            "record",
            "acquired_before_any_gpu_probe",
            "release_owner",
            "stale_lock_policy",
        },
        label,
    )
    lock_path = _absolute_artifact_path(
        binding.get("path"), label=f"{label}.path", suffix=".lock"
    )
    if not lock_path.endswith("/output/udlm/.single_training_job.lock"):
        raise ScreenValidationError(f"{label} path is outside the global lease")
    lock_sha256 = _sha256(binding.get("sha256"), f"{label}.sha256")
    record = _mapping(binding.get("record"), f"{label}.record")
    _exact_keys(
        record,
        {
            "schema_version",
            "status",
            "owner_token",
            "acquired_at_utc",
            "launcher_pid_at_acquisition",
            "owner_process_exit_does_not_make_lock_stale",
            "source_revision",
            "run_name",
            "training_variant",
            "purpose",
            "stale_lock_policy",
            "release_policy",
        },
        f"{label}.record",
    )
    acquired_at = _utc_timestamp(
        record.get("acquired_at_utc"), f"{label}.record.acquired_at_utc"
    )
    inventory_at = _utc_timestamp(
        manifest.get("inventory_snapshot_completed_at_utc"),
        "launch inventory timestamp",
    )
    expected_record = {
        "schema_version": 1,
        "status": "held",
        "owner_token": record.get("owner_token"),
        "acquired_at_utc": record.get("acquired_at_utc"),
        "launcher_pid_at_acquisition": record.get("launcher_pid_at_acquisition"),
        "owner_process_exit_does_not_make_lock_stale": True,
        "source_revision": source_revision,
        "run_name": manifest.get("run_name"),
        "training_variant": "udlm_categorical",
        "purpose": "enforce_one_registered_optimization_screen_job_at_a_time",
        "stale_lock_policy": "fail_closed_and_require_manual_review",
        "release_policy": (
            "exact_owner_lock_only_after_receipt_or_before_tmux_handoff_failure"
        ),
    }
    _sha256(record.get("owner_token"), f"{label}.record.owner_token")
    _integer(
        record.get("launcher_pid_at_acquisition"),
        f"{label}.record.launcher_pid_at_acquisition",
        minimum=1,
    )
    if not _exact_json_equal(record, expected_record) or acquired_at > inventory_at:
        raise ScreenValidationError(f"{label} record is unmatched")
    record_payload = (
        json.dumps(dict(record), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if hashlib.sha256(record_payload).hexdigest() != lock_sha256:
        raise ScreenValidationError(f"{label} digest is not bound to its record")
    if (
        _boolean(
            binding.get("acquired_before_any_gpu_probe"),
            f"{label}.acquired_before_any_gpu_probe",
        )
        is not True
        or binding.get("release_owner") != "pilot_exit_receipt_writer_after_publication"
        or binding.get("stale_lock_policy") != "fail_closed_and_require_manual_review"
    ):
        raise ScreenValidationError(f"{label} policy is unmatched")
    return {**dict(binding), "record": dict(record)}


def _validate_manifest_training_bindings(
    manifest: Mapping[str, Any],
    *,
    refs: Mapping[str, Mapping[str, Any]],
    checkpoint: Mapping[str, Any],
    arm: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    resolved_config_sha256: str,
    source_revision: str,
) -> dict[str, Any]:
    if (
        manifest.get("run_name") != arm.get("attempt_id")
        or manifest.get("training_variant") != "udlm_categorical"
        or manifest.get("hydra_config_name") != "udlm_categorical"
        or not _exact_json_equal(
            manifest.get("resolved_training_config"), resolved_config
        )
        or manifest.get("resolved_training_config_sha256") != resolved_config_sha256
    ):
        raise ScreenValidationError("launch manifest training binding is unmatched")
    if canonical_json_sha256(resolved_config) != resolved_config_sha256:
        raise ScreenValidationError("resolved config content digest is unmatched")
    argv = manifest.get("training_argv")
    if (
        not isinstance(argv, list)
        or len(argv) < 3
        or any(not isinstance(item, str) or not item for item in argv)
        or argv[1] != "-u"
    ):
        raise ScreenValidationError("launch manifest training argv is invalid")
    _absolute_artifact_path(argv[0], label="launch Python interpreter")
    _absolute_artifact_path(argv[2], label="launch training entrypoint", suffix=".py")
    if not argv[2].endswith("/scripts/train.py"):
        raise ScreenValidationError("launch training entrypoint is unmatched")
    argv_sha256 = _sha256(
        manifest.get("training_argv_sha256"), "launch training argv digest"
    )
    if canonical_json_sha256(argv[2:]) != argv_sha256:
        raise ScreenValidationError("launch training argv digest is unmatched")
    path_bindings = (
        ("launch_manifest_path", refs["launch_manifest"], ".json"),
        ("runtime_config_path", refs["runtime_config"], ".json"),
        ("training_summary_path", refs["training_summary"], ".json"),
        ("pilot_exit_status_path", refs["exit_receipt"], ".json"),
        ("expected_final_checkpoint_path", checkpoint, ".ckpt"),
    )
    for field, ref, suffix in path_bindings:
        _absolute_artifact_path(
            manifest.get(field),
            label=f"launch manifest {field}",
            expected_ref=ref,
            suffix=suffix,
        )
    repository_prefix = manifest["launch_manifest_path"].removesuffix(
        f"/{refs['launch_manifest']['relative_path']}"
    )
    if str(PurePosixPath(argv[2]).parents[1]) != repository_prefix:
        raise ScreenValidationError(
            "launch training entrypoint and artifact repository roots differ"
        )
    if (
        manifest.get("training_summary_schema_version")
        != EXPECTED_ARTIFACT_SCHEMA_VERSIONS["training_summary"]
        or manifest.get("pilot_exit_status_schema_version")
        != EXPECTED_ARTIFACT_SCHEMA_VERSIONS["exit_receipt"]
    ):
        raise ScreenValidationError("launch completion schemas are unmatched")
    _validate_launch_completion_contract(manifest.get("completion_contract"))
    return _validate_training_job_lock_binding(
        manifest.get("single_training_job_lock"),
        manifest=manifest,
        source_revision=source_revision,
    )


def _validate_runtime_record(
    runtime: Mapping[str, Any],
    *,
    registry: ValidatedRegistry,
    manifest: Mapping[str, Any],
    manifest_ref: Mapping[str, Any],
    selected_uuids: Sequence[str],
    resolved_config: Mapping[str, Any],
    resolved_config_sha256: str,
    expected_steps: int,
    source_revision: str,
) -> dict[str, Any]:
    _exact_keys(runtime, _RUNTIME_CONFIG_KEYS, "runtime config")
    if (
        _integer(runtime.get("schema_version"), "runtime config schema")
        != EXPECTED_ARTIFACT_SCHEMA_VERSIONS["runtime_config"]
    ):
        raise ScreenValidationError("runtime-config schema is unsupported")
    if runtime.get("status") != "preflight_completed":
        raise ScreenValidationError("runtime preflight did not complete")
    if runtime.get("source_revision") != source_revision:
        raise ScreenValidationError("runtime source revision is unmatched")
    source = _validate_source_record(
        runtime.get("source"), label="runtime source", source_revision=source_revision
    )
    manifest_argv = manifest.get("training_argv")
    runtime_argv = runtime.get("training_argv")
    if (
        not isinstance(manifest_argv, list)
        or not isinstance(runtime_argv, list)
        or runtime_argv != manifest_argv[2:]
        or runtime.get("observed_training_argv") != runtime_argv
        or any(not isinstance(item, str) or not item for item in runtime_argv)
        or canonical_json_sha256(runtime_argv) != manifest.get("training_argv_sha256")
        or runtime.get("training_argv_sha256") != manifest.get("training_argv_sha256")
    ):
        raise ScreenValidationError("runtime training argv is unmatched")
    if runtime.get(
        "resolved_training_config_sha256"
    ) != resolved_config_sha256 or not _exact_json_equal(
        runtime.get("resolved_training_config"), resolved_config
    ):
        raise ScreenValidationError("runtime resolved config is unmatched")
    runtime_manifest = _validate_stable_snapshot(
        runtime.get("launch_manifest"),
        label="runtime launch-manifest snapshot",
        expected_path=manifest["launch_manifest_path"],
        expected_ref=manifest_ref,
        extra_keys={"selected_gpu_uuids"},
    )
    if runtime_manifest.get("selected_gpu_uuids") != list(selected_uuids):
        raise ScreenValidationError("runtime manifest/GPU binding is unmatched")
    completion = _validate_completion_contract(
        runtime.get("completion_contract"),
        label="runtime completion contract",
        manifest=manifest,
        expected_steps=expected_steps,
        expected_world_size=registry.data["common_training"]["gpu_count"],
    )
    python_environment = _mapping(
        runtime.get("python_environment"), "runtime Python environment"
    )
    if not runtime_argv:
        raise ScreenValidationError("runtime training argv is empty")
    training_entrypoint = PurePosixPath(runtime_argv[0])
    repository_root = training_entrypoint.parents[1]
    expected_python_environment = {
        **_CONTROLLED_PYTHON_ENVIRONMENT,
        "PYTHONHASHSEED": str(EXPECTED_TRAINING_SEED),
        "PYTHONPATH": os.pathsep.join(
            (str(repository_root / "src"), str(repository_root))
        ),
    }
    if not _exact_json_equal(python_environment, expected_python_environment):
        raise ScreenValidationError("runtime Python environment is unmatched")
    return {
        "source": source,
        "training_argv": list(runtime_argv),
        "launch_manifest": runtime_manifest,
        "completion_contract": completion,
    }


def _checkpoint_ref(value: object, label: str) -> dict[str, Any]:
    checkpoint = _mapping(value, label)
    _exact_keys(checkpoint, _BLOB_REF_KEYS | {"global_step"}, label)
    ref = _blob_ref(
        {key: checkpoint[key] for key in _BLOB_REF_KEYS},
        label,
        required_root="repository",
        suffix=".ckpt",
    )
    return {
        **ref,
        "global_step": _integer(
            checkpoint.get("global_step"), f"{label}.global_step", minimum=1
        ),
    }


def _validate_initialization(
    value: object,
    *,
    registry: ValidatedRegistry,
    arm_id: str,
    resolved_config_sha256: str,
) -> dict[str, Any]:
    label = f"{arm_id} initialization"
    initialization = _mapping(value, label)
    _exact_keys(
        initialization,
        {
            "mode",
            "source_checkpoint_sha256",
            "weights",
            "optimizer_reset",
            "scheduler_reset",
            "global_step_reset",
            "ema_reset",
            "state_audit",
        },
        label,
    )
    expected = registry.data["common_training"]["initialization"]
    if (
        initialization.get("mode") != expected["mode"]
        or initialization.get("source_checkpoint_sha256")
        != expected["checkpoint"]["sha256"]
        or initialization.get("weights") != "ema"
    ):
        raise ScreenValidationError(f"{label} warm-start binding is unmatched")
    for field in (
        "optimizer_reset",
        "scheduler_reset",
        "global_step_reset",
        "ema_reset",
    ):
        if _boolean(initialization.get(field), f"{label}.{field}") is not True:
            raise ScreenValidationError(f"{label} did not reset every state")
    state_audit = _mapping(initialization.get("state_audit"), f"{label} state audit")
    _exact_keys(
        state_audit,
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
        f"{label} state audit",
    )
    if (
        _integer(state_audit.get("schema_version"), f"{label} state-audit schema") != 1
        or state_audit.get("phase")
        != "after_verified_mdlm_ema_warm_start_before_training_rng_reseed_and_optimizer_creation"
        or state_audit.get("source_checkpoint_sha256")
        != expected["checkpoint"]["sha256"]
        or state_audit.get("resolved_training_config_sha256") != resolved_config_sha256
        or state_audit.get("training_seed") != EXPECTED_TRAINING_SEED
        or state_audit.get("conditioning_variant")
        != ("film_adaln" if arm_id == "E-A1" else "additive")
    ):
        raise ScreenValidationError(f"{label} state audit is unmatched")
    for field in ("common_backbone_tensor_count", "full_initial_tensor_count"):
        _integer(state_audit.get(field), f"{label} {field}", minimum=1)
    for field in ("common_backbone_state_sha256", "full_initial_state_sha256"):
        _sha256(state_audit.get(field), f"{label} {field}")
    return {**dict(initialization), "state_audit": dict(state_audit)}


def _validate_checkpoint_finiteness_record(value: object, *, label: str) -> int:
    record = _mapping(value, label)
    _exact_keys(
        record,
        {"all_finite", "floating_tensor_count", "floating_element_count"},
        label,
    )
    if _boolean(record.get("all_finite"), f"{label}.all_finite") is not True:
        raise ScreenValidationError(f"{label} is not finite")
    tensor_count = _integer(
        record.get("floating_tensor_count"),
        f"{label}.floating_tensor_count",
        minimum=1,
    )
    element_count = _integer(
        record.get("floating_element_count"),
        f"{label}.floating_element_count",
        minimum=1,
    )
    if element_count < tensor_count:
        raise ScreenValidationError(f"{label} element count is invalid")
    return tensor_count


def _expected_framework_nonfinite_sentinels(*, expected_steps: int) -> dict[str, Any]:
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


def _validate_checkpoint_semantic_audit(
    value: object,
    *,
    expected_steps: int,
    resolved_config: Mapping[str, Any],
    resolved_config_sha256: str,
) -> dict[str, Any]:
    label = "training-summary checkpoint semantic audit"
    semantic = _mapping(value, label)
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
        label,
    )
    if (
        _boolean(semantic.get("deserialized"), f"{label}.deserialized") is not True
        or _boolean(
            semantic.get("udlm_process_identity_verified"),
            f"{label}.udlm_process_identity_verified",
        )
        is not True
        or _integer(semantic.get("global_step"), f"{label}.global_step")
        != expected_steps
    ):
        raise ScreenValidationError(f"{label} completion identity is invalid")
    finiteness_records = {
        key: dict(_mapping(semantic.get(key), f"{label}.{key}"))
        for key in (
            "raw_model",
            "ema",
            "optimizer",
            "non_sentinel_checkpoint_tensors",
        )
    }
    finiteness_counts = {
        key: _validate_checkpoint_finiteness_record(record, label=f"{label}.{key}")
        for key, record in finiteness_records.items()
    }
    covered_tensor_count = sum(
        finiteness_counts[key] for key in ("raw_model", "ema", "optimizer")
    )
    covered_element_count = sum(
        _integer(
            finiteness_records[key].get("floating_element_count"),
            f"{label}.{key}.floating_element_count",
            minimum=1,
        )
        for key in ("raw_model", "ema", "optimizer")
    )
    if (
        finiteness_counts["non_sentinel_checkpoint_tensors"] < covered_tensor_count
        or _integer(
            finiteness_records["non_sentinel_checkpoint_tensors"].get(
                "floating_element_count"
            ),
            f"{label}.non_sentinel_checkpoint_tensors.floating_element_count",
            minimum=1,
        )
        < covered_element_count
    ):
        raise ScreenValidationError(
            f"{label} aggregate tensor finiteness undercounts protected state"
        )
    sentinel = _mapping(
        semantic.get("framework_nonfinite_sentinels"),
        f"{label}.framework_nonfinite_sentinels",
    )
    if canonical_json_sha256(sentinel) != canonical_json_sha256(
        _expected_framework_nonfinite_sentinels(expected_steps=expected_steps)
    ):
        raise ScreenValidationError(f"{label} framework sentinel is invalid")

    python_floats = _mapping(
        semantic.get("checkpoint_python_floats"),
        f"{label}.checkpoint_python_floats",
    )
    _exact_keys(
        python_floats,
        {"all_finite", "floating_scalar_count"},
        f"{label}.checkpoint_python_floats",
    )
    if (
        _boolean(
            python_floats.get("all_finite"),
            f"{label}.checkpoint_python_floats.all_finite",
        )
        is not True
    ):
        raise ScreenValidationError(f"{label} Python floats are not finite")
    _integer(
        python_floats.get("floating_scalar_count"),
        f"{label}.checkpoint_python_floats.floating_scalar_count",
        minimum=1,
    )

    trainer_config = _mapping(resolved_config.get("trainer"), "resolved trainer")
    accumulation = _integer(
        trainer_config.get("accumulate_grad_batches"),
        "resolved gradient accumulation",
        minimum=1,
    )
    optim_config = _mapping(resolved_config.get("optim"), "resolved optimizer")
    scheduler_config = _mapping(
        optim_config.get("scheduler"), "resolved optimizer scheduler"
    )
    warmup_updates = _integer(
        scheduler_config.get("warmup_updates"),
        "resolved scheduler warmup",
        minimum=0,
    )
    horizon = scheduler_config.get("horizon_updates")
    horizon_updates = (
        0
        if horizon is None
        else _integer(horizon, "resolved scheduler horizon", minimum=0)
    )
    schedule_checks = (
        max(
            expected_steps,
            warmup_updates + 1,
            horizon_updates + 1,
        )
        + 1
    )

    optimizer = _mapping(
        semantic.get("optimizer_live_state_match"),
        f"{label}.optimizer_live_state_match",
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
        f"{label}.optimizer_live_state_match",
    )
    if not _exact_json_equal(
        optimizer,
        {
            "exact_serialized_live_match": True,
            "optimizer_count": 1,
            "optimizer_class": "AdamW",
            "parameter_group_count": 1,
            "parameter_state_count": optimizer.get("parameter_state_count"),
            "exact_resolved_config_match": True,
        },
    ):
        raise ScreenValidationError(f"{label} optimizer live-state match is invalid")
    optimizer_parameter_count = _integer(
        optimizer.get("parameter_state_count"),
        f"{label}.optimizer parameter count",
        minimum=1,
    )
    scheduler = _mapping(
        semantic.get("scheduler_live_state_match"),
        f"{label}.scheduler_live_state_match",
    )
    if not _exact_json_equal(
        scheduler,
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
            "callable_schedule_index_checks": schedule_checks,
        },
    ):
        raise ScreenValidationError(f"{label} scheduler live-state match is invalid")
    sampler = _mapping(
        semantic.get("sampler_live_state_match"),
        f"{label}.sampler_live_state_match",
    )
    if not _exact_json_equal(
        sampler,
        {
            "exact_hosted_stream_contract_match": True,
            "random_state_is_none": True,
            "live_state_dict_available": False,
            "sampler_class_module": "torch.utils.data.dataloader",
            "sampler_class_name": "_InfiniteConstantSampler",
        },
    ):
        raise ScreenValidationError(f"{label} sampler live-state match is invalid")
    callback_key = (
        "ModelCheckpoint{'monitor': None, 'mode': 'min', "
        f"'every_n_train_steps': {expected_steps}, 'every_n_epochs': 0, "
        "'train_time_interval': None}"
    )
    callback = _mapping(
        semantic.get("model_checkpoint_live_state_match"),
        f"{label}.model_checkpoint_live_state_match",
    )
    if not _exact_json_equal(
        callback,
        {
            "exact_serialized_live_match": True,
            "model_checkpoint_callback_count": 1,
            "state_key": callback_key,
            "configuration_matches_pilot_contract": True,
        },
    ):
        raise ScreenValidationError(f"{label} callback live-state match is invalid")
    hyperparameters = _mapping(
        semantic.get("checkpoint_hyperparameters_match"),
        f"{label}.checkpoint_hyperparameters_match",
    )
    if not _exact_json_equal(
        hyperparameters,
        {
            "hparams_name": "kwargs",
            "exact_hyperparameter_keys": True,
            "exact_checkpoint_preflight_config_match": True,
            "exact_live_model_preflight_config_match": True,
            "exact_live_hparams_preflight_config_match": True,
            "exact_checkpoint_live_model_unresolved_config_match": True,
            "exact_checkpoint_live_hparams_unresolved_config_match": True,
            "resolved_config_sha256": resolved_config_sha256,
        },
    ):
        raise ScreenValidationError(f"{label} hyperparameter match is invalid")
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
    configured_clip_decimal = _decimal(
        configured_clip,
        f"{label} resolved gradient clip",
        nonnegative=True,
    )
    if not isinstance(live_precision, str) or configured_clip_algorithm not in {
        "norm",
        "value",
    }:
        raise ScreenValidationError(f"{label} resolved live-Trainer config is invalid")
    trainer_match = _mapping(
        semantic.get("trainer_live_configuration_match"),
        f"{label}.trainer_live_configuration_match",
    )
    if not _exact_json_equal(
        trainer_match,
        {
            "exact_detect_anomaly_match": True,
            "detect_anomaly": True,
            "exact_gradient_clip_val_match": True,
            "gradient_clip_val": configured_clip_decimal,
            "exact_gradient_clip_algorithm_match": True,
            "gradient_clip_algorithm": configured_clip_algorithm,
            "exact_precision_match": True,
            "configured_precision": str(configured_precision),
            "live_precision": live_precision,
        },
    ):
        raise ScreenValidationError(f"{label} live Trainer match is invalid")
    loop_state = _mapping(
        semantic.get("checkpoint_loop_state_match"),
        f"{label}.checkpoint_loop_state_match",
    )
    if not _exact_json_equal(
        loop_state,
        {
            "exact_serialized_progress_match": True,
            "epoch": 0,
            "optimizer_steps": expected_steps,
            "accumulate_grad_batches": accumulation,
            "microbatches": expected_steps * accumulation,
        },
    ):
        raise ScreenValidationError(f"{label} loop-state match is invalid")

    ema_metadata = _mapping(semantic.get("ema_metadata"), f"{label}.ema_metadata")
    _exact_keys(
        ema_metadata,
        {"shadow_parameter_count", "decay", "num_updates"},
        f"{label}.ema_metadata",
    )
    shadow_count = _integer(
        ema_metadata.get("shadow_parameter_count"),
        f"{label}.ema_metadata.shadow_parameter_count",
        minimum=1,
    )
    ema_decay = _decimal(
        ema_metadata.get("decay"),
        f"{label}.ema_metadata.decay",
    )
    training_config = _mapping(resolved_config.get("training"), "resolved training")
    configured_ema_decay = _decimal(
        training_config.get("ema"), f"{label} resolved EMA decay"
    )
    if (
        shadow_count != finiteness_counts["ema"]
        or shadow_count != optimizer_parameter_count
        or _integer(
            ema_metadata.get("num_updates"),
            f"{label}.ema_metadata.num_updates",
        )
        != expected_steps
        or not Decimal("0") < ema_decay < Decimal("1")
        or ema_decay != configured_ema_decay
    ):
        raise ScreenValidationError(f"{label} EMA metadata is invalid")
    live_matches: dict[str, dict[str, Any]] = {}
    for key, required_keys in (
        ("live_model_match", {"exact_key_set", "exact_tensor_values", "tensor_count"}),
        ("live_ema_match", {"exact_tensor_values", "tensor_count"}),
    ):
        match = _mapping(semantic.get(key), f"{label}.{key}")
        live_matches[key] = dict(match)
        _exact_keys(match, required_keys, f"{label}.{key}")
        for flag in required_keys - {"tensor_count"}:
            if _boolean(match.get(flag), f"{label}.{key}.{flag}") is not True:
                raise ScreenValidationError(f"{label}.{key} is invalid")
        count = _integer(
            match.get("tensor_count"), f"{label}.{key}.tensor_count", minimum=1
        )
        if key == "live_ema_match" and count != shadow_count:
            raise ScreenValidationError(f"{label} live EMA count is invalid")
        if key == "live_model_match" and count < finiteness_counts["raw_model"]:
            raise ScreenValidationError(f"{label} live model count is invalid")
    return {
        "raw_model": finiteness_records["raw_model"],
        "ema": finiteness_records["ema"],
        "ema_metadata": dict(ema_metadata),
        "optimizer": finiteness_records["optimizer"],
        "non_sentinel_checkpoint_tensors": finiteness_records[
            "non_sentinel_checkpoint_tensors"
        ],
        "live_model_match": live_matches["live_model_match"],
        "live_ema_match": live_matches["live_ema_match"],
    }


def _validate_training_accounting(
    value: object,
    *,
    registry: ValidatedRegistry,
    arm_id: str,
    resolved_config: Mapping[str, Any],
    expected_steps: int,
) -> dict[str, Any]:
    label = "summary training accounting"
    accounting = _mapping(value, label)
    _exact_keys(accounting, _TRAINING_ACCOUNTING_KEYS, label)
    loader_config = _mapping(resolved_config.get("loader"), "resolved loader")
    trainer_config = _mapping(resolved_config.get("trainer"), "resolved trainer")
    world_size = registry.data["common_training"]["gpu_count"]
    micro_batch = _integer(
        loader_config.get("batch_size"), "resolved micro batch size", minimum=1
    )
    accumulation = _integer(
        trainer_config.get("accumulate_grad_batches"),
        "resolved gradient accumulation",
        minimum=1,
    )
    effective_batch = micro_batch * world_size * accumulation
    if (
        _integer(
            loader_config.get("global_batch_size"),
            "resolved global batch size",
            minimum=1,
        )
        != effective_batch
    ):
        raise ScreenValidationError("resolved batch accounting is inconsistent")
    expected = {
        "training_seed": EXPECTED_TRAINING_SEED,
        "optimizer_updates": expected_steps,
        "world_size": world_size,
        "micro_batch_size_per_rank": micro_batch,
        "accumulate_grad_batches": accumulation,
        "effective_global_examples_per_optimizer_step": effective_batch,
        "total_requested_example_exposures": effective_batch * expected_steps,
    }
    for key, expected_value in expected.items():
        observed = _integer(accounting.get(key), f"{label}.{key}", minimum=1)
        if observed != expected_value:
            raise ScreenValidationError(f"{label}.{key} is unmatched")
    if (
        accounting.get("hosted_stream_rank_partition_policy")
        != _HOSTED_STREAM_RANK_PARTITION_POLICY
    ):
        raise ScreenValidationError(f"{label} rank partition is unmatched")
    parameter_counts = _mapping(
        accounting.get("trainable_parameter_counts"),
        f"{label}.trainable_parameter_counts",
    )
    expected_parameter_keys = {"base_backbone", "time_conditioner", "total"}
    if arm_id == "E-A1":
        expected_parameter_keys.add("film_modulation")
    _exact_keys(
        parameter_counts,
        expected_parameter_keys,
        f"{label}.trainable_parameter_counts",
    )
    component_keys = expected_parameter_keys - {"total"}
    components = {
        key: _integer(
            parameter_counts.get(key),
            f"{label}.trainable_parameter_counts.{key}",
            minimum=1,
        )
        for key in component_keys
    }
    total = _integer(
        parameter_counts.get("total"),
        f"{label}.trainable_parameter_counts.total",
        minimum=1,
    )
    gradient_groups = {
        group["group_id"]: group
        for group in registry.data["stages"][1]["gradient_contract"]["groups"]
    }
    expected_time_conditioner = sum(
        math.prod(parameter["shape"])
        for parameter in gradient_groups["timestep_mlp"]["parameters"]
    )
    expected_film_modulation = sum(
        math.prod(parameter["shape"])
        for parameter in gradient_groups["film_modulation"]["parameters"]
    )
    if (
        total != sum(components.values())
        or components["time_conditioner"] != expected_time_conditioner
        or (
            arm_id == "E-A1"
            and components.get("film_modulation") != expected_film_modulation
        )
    ):
        raise ScreenValidationError(f"{label} parameter counts are inconsistent")
    return {**dict(accounting), "trainable_parameter_counts": dict(parameter_counts)}


def _validate_training_health(value: object, *, expected_steps: int) -> None:
    label = "summary training health"
    health = _mapping(value, label)
    _exact_keys(health, _TRAINING_HEALTH_KEYS, label)
    if health.get("scope") != _TRAINING_HEALTH_SCOPE:
        raise ScreenValidationError(f"{label} scope is unmatched")
    for key in (
        "all_losses_finite",
        "all_observed_gradients_finite",
        "every_optimizer_step_had_a_nonzero_gradient",
    ):
        if _boolean(health.get(key), f"{label}.{key}") is not True:
            raise ScreenValidationError(f"{label}.{key} is false")
    loss_checks = _integer(health.get("loss_checks"), f"{label}.loss_checks", minimum=1)
    optimizer_checks = _integer(
        health.get("optimizer_step_checks"),
        f"{label}.optimizer_step_checks",
        minimum=1,
    )
    gradient_tensors = _integer(
        health.get("gradient_tensor_observations"),
        f"{label}.gradient_tensor_observations",
        minimum=1,
    )
    gradient_elements = _integer(
        health.get("gradient_element_observations"),
        f"{label}.gradient_element_observations",
        minimum=1,
    )
    if (
        loss_checks < expected_steps
        or optimizer_checks != expected_steps
        or gradient_tensors < optimizer_checks
        or gradient_elements < gradient_tensors
    ):
        raise ScreenValidationError(f"{label} counters are inconsistent")


def _validate_summary_startup(
    value: object,
    *,
    registry: ValidatedRegistry,
    stage_id: str,
    arm_id: str,
    resolved_config: Mapping[str, Any],
) -> dict[str, Any]:
    label = "summary startup"
    startup = _mapping(value, label)
    expected_keys = {"mode", "verified_mdlm_warm_start_report"}
    if stage_id == "conditioning":
        expected_keys.add("training_rng_policy")
    _exact_keys(startup, expected_keys, label)
    if startup.get("mode") != "warm_start":
        raise ScreenValidationError("screen summary is not a warm start")
    expected_rng_policy = {
        "policy": "reseed_all_training_rng_streams_after_model_and_warm_start",
        "seed": EXPECTED_TRAINING_SEED,
        "purpose": "isolate_training_randomness_from_architecture_constructor_draws",
        "applied_before_dataloader_and_trainer_construction": True,
    }
    if stage_id == "conditioning" and not _exact_json_equal(
        startup.get("training_rng_policy"), expected_rng_policy
    ):
        raise ScreenValidationError("screen summary training RNG policy is unmatched")
    warm_start = _mapping(
        startup.get("verified_mdlm_warm_start_report"),
        "summary warm-start report",
    )
    expected_warm_start_keys = {
        "source_path",
        "source_resolved_path",
        "source_sha256",
        "source_size_bytes",
        "expected_source_sha256",
        "byte_identity_verified_before_and_after_load",
        "weights",
        "parameter_tensors",
    }
    if arm_id == "E-A1":
        expected_warm_start_keys.update(
            {"conditioning_variant", "conditioning_parameter_tensors"}
        )
    _exact_keys(warm_start, expected_warm_start_keys, "summary warm-start report")
    checkpoint = registry.data["common_training"]["initialization"]["checkpoint"]
    training = _mapping(resolved_config.get("training"), "resolved training")
    configured_path = training.get("init_from_mdlm_checkpoint")
    if warm_start.get("source_path") != configured_path:
        raise ScreenValidationError("summary warm-start source path is unmatched")
    _absolute_artifact_path(
        warm_start.get("source_resolved_path"),
        label="summary resolved warm-start path",
        expected_ref=checkpoint,
        suffix=".ckpt",
    )
    expected_values = {
        "source_sha256": checkpoint["sha256"],
        "source_size_bytes": checkpoint["size_bytes"],
        "expected_source_sha256": checkpoint["sha256"],
        "byte_identity_verified_before_and_after_load": True,
        "weights": "ema",
    }
    if arm_id == "E-A1":
        expected_values["conditioning_variant"] = "film_adaln"
    if any(
        not _exact_json_equal(warm_start.get(key), expected)
        for key, expected in expected_values.items()
    ):
        raise ScreenValidationError("training summary warm start is unmatched")
    _integer(
        warm_start.get("parameter_tensors"),
        "summary warm-start parameter_tensors",
        minimum=1,
    )
    if arm_id == "E-A1":
        registered_conditioning_tensors = sum(
            len(group["parameters"])
            for group in registry.data["stages"][1]["gradient_contract"]["groups"]
        )
        if (
            _integer(
                warm_start.get("conditioning_parameter_tensors"),
                "summary warm-start conditioning_parameter_tensors",
                minimum=1,
            )
            != registered_conditioning_tensors
        ):
            raise ScreenValidationError(
                "summary warm-start conditioning tensor count is unmatched"
            )
    return dict(startup)


def _validate_training_summary(
    summary: Mapping[str, Any],
    *,
    registry: ValidatedRegistry,
    stage_id: str,
    arm: Mapping[str, Any],
    config_entry: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_ref: Mapping[str, Any],
    runtime: Mapping[str, Any],
    runtime_ref: Mapping[str, Any],
    runtime_validation: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    gradient_audit: object,
    initialization: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    source_revision: str,
) -> dict[str, Any]:
    _exact_keys(summary, _TRAINING_SUMMARY_KEYS, "training summary")
    if (
        _integer(summary.get("schema_version"), "training summary schema")
        != EXPECTED_ARTIFACT_SCHEMA_VERSIONS["training_summary"]
        or summary.get("status") != "completed"
    ):
        raise ScreenValidationError(
            "training summary is not a completed schema-5 record"
        )
    completed_at = _utc_timestamp(
        summary.get("completed_at_utc"), "training-summary completion timestamp"
    )
    if completed_at <= _utc_timestamp(
        manifest.get("created_at"), "launch-manifest creation timestamp"
    ):
        raise ScreenValidationError("training summary predates the launch manifest")
    if summary.get("source_revision") != source_revision:
        raise ScreenValidationError("training-summary source revision is unmatched")
    source = _validate_source_record(
        summary.get("source"),
        label="training-summary source",
        source_revision=source_revision,
    )
    resolved_config_sha256 = config_entry["config"]["canonical_sha256"]
    if summary.get(
        "resolved_training_config_sha256"
    ) != resolved_config_sha256 or summary.get("training_argv_sha256") != manifest.get(
        "training_argv_sha256"
    ):
        raise ScreenValidationError("training-summary config/argv is unmatched")
    summary_manifest = _validate_stable_snapshot(
        summary.get("launch_manifest"),
        label="summary launch-manifest snapshot",
        expected_path=manifest["launch_manifest_path"],
        expected_ref=manifest_ref,
        extra_keys={"selected_gpu_uuids"},
    )
    if summary_manifest.get("selected_gpu_uuids") != list(
        runtime_validation["launch_manifest"]["selected_gpu_uuids"]
    ) or not _exact_json_equal(summary_manifest, runtime_validation["launch_manifest"]):
        raise ScreenValidationError("training summary is not launch-manifest-bound")
    summary_runtime = _validate_stable_snapshot(
        summary.get("runtime_config"),
        label="summary runtime-config snapshot",
        expected_path=manifest["runtime_config_path"],
        expected_ref=runtime_ref,
        extra_keys={"schema_version", "record_sha256"},
    )
    if _integer(
        summary_runtime.get("schema_version"),
        "summary runtime-config schema",
    ) != EXPECTED_ARTIFACT_SCHEMA_VERSIONS["runtime_config"] or summary_runtime.get(
        "record_sha256"
    ) != canonical_json_sha256(runtime):
        raise ScreenValidationError("summary runtime-config binding is unmatched")
    expected_steps = EXPECTED_UPDATES[stage_id]
    world_size = registry.data["common_training"]["gpu_count"]
    completion = _validate_completion_contract(
        summary.get("completion_contract"),
        label="summary completion contract",
        manifest=manifest,
        expected_steps=expected_steps,
        expected_world_size=world_size,
    )
    if not _exact_json_equal(completion, runtime_validation["completion_contract"]):
        raise ScreenValidationError("summary/runtime completion contracts differ")
    training = _mapping(resolved_config.get("training"), "resolved training")
    trainer = _mapping(resolved_config.get("trainer"), "resolved trainer")
    if (
        training.get("pilot_fail_on_nonfinite_loss") is not True
        or trainer.get("detect_anomaly") is not True
    ):
        raise ScreenValidationError("resolved config disables completion safeguards")
    state = _mapping(summary.get("observed_training_state"), "summary training state")
    _exact_keys(
        state, {"global_rank", "global_step", "world_size"}, "summary training state"
    )
    expected_state = {
        "global_rank": 0,
        "global_step": expected_steps,
        "world_size": world_size,
    }
    if not _exact_json_equal(state, expected_state):
        raise ScreenValidationError("training summary has unmatched completion state")
    accounting = _validate_training_accounting(
        summary.get("training_accounting"),
        registry=registry,
        arm_id=arm["arm_id"],
        resolved_config=resolved_config,
        expected_steps=expected_steps,
    )
    _validate_training_health(
        summary.get("training_health"), expected_steps=expected_steps
    )
    startup = _validate_summary_startup(
        summary.get("startup"),
        registry=registry,
        stage_id=stage_id,
        arm_id=arm["arm_id"],
        resolved_config=resolved_config,
    )
    final_checkpoint = _validate_stable_snapshot(
        summary.get("final_checkpoint"),
        label="summary final-checkpoint snapshot",
        expected_path=manifest["expected_final_checkpoint_path"],
        expected_ref=checkpoint,
        extra_keys={"semantic_audit"},
    )
    semantic = _validate_checkpoint_semantic_audit(
        final_checkpoint.get("semantic_audit"),
        expected_steps=expected_steps,
        resolved_config=resolved_config,
        resolved_config_sha256=resolved_config_sha256,
    )
    tensor_finiteness = _mapping(
        summary.get("tensor_finiteness"), "summary live tensor finiteness"
    )
    _exact_keys(
        tensor_finiteness,
        {"raw_model", "ema"},
        "summary live tensor finiteness",
    )
    for key in ("raw_model", "ema"):
        _validate_checkpoint_finiteness_record(
            tensor_finiteness.get(key),
            label=f"summary live tensor finiteness.{key}",
        )
        if not _exact_json_equal(tensor_finiteness.get(key), semantic[key]):
            raise ScreenValidationError(
                "summary live/checkpoint tensor finiteness differs"
            )
    if not _exact_json_equal(
        summary.get("conditioning_gradient_audit"), gradient_audit
    ):
        raise ScreenValidationError("summary gradient audit differs from evidence")
    expected_audit = gradient_audit if arm["arm_id"] == "E-A1" else None
    if not _exact_json_equal(
        summary.get("conditioning_gradient_audit"), expected_audit
    ):
        raise ScreenValidationError("gradient audit is present on the wrong arm")
    if not _exact_json_equal(
        summary.get("screen_initialization_state_audit"), initialization["state_audit"]
    ):
        raise ScreenValidationError(
            "summary initialization audit differs from evidence"
        )
    return {
        "completed_at": completed_at,
        "source": source,
        "launch_manifest": summary_manifest,
        "runtime_config": summary_runtime,
        "completion_contract": completion,
        "training_accounting": accounting,
        "ema_metadata": semantic["ema_metadata"],
        "final_checkpoint": final_checkpoint,
        "startup_mode": startup["mode"],
    }


def _validate_exit_receipt(
    receipt: Mapping[str, Any],
    *,
    registry: ValidatedRegistry,
    stage_id: str,
    config_entry: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_ref: Mapping[str, Any],
    runtime_ref: Mapping[str, Any],
    summary_ref: Mapping[str, Any],
    summary_validation: Mapping[str, Any],
    lock_binding: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    selected_uuids: Sequence[str],
    gradient_audit: object,
    initialization: Mapping[str, Any],
    source_revision: str,
) -> None:
    _exact_keys(receipt, _EXIT_RECEIPT_KEYS, "exit receipt")
    if (
        _integer(receipt.get("schema_version"), "exit receipt schema")
        != EXPECTED_ARTIFACT_SCHEMA_VERSIONS["exit_receipt"]
        or receipt.get("status") != "completed"
        or receipt.get("overall_status") != "completed"
        or _integer(receipt.get("process_exit_status"), "exit process status") != 0
    ):
        raise ScreenValidationError("exit receipt does not certify completion")
    receipt_time = _utc_timestamp(
        receipt.get("recorded_at_utc"), "exit receipt timestamp"
    )
    if receipt_time <= summary_validation["completed_at"]:
        raise ScreenValidationError("exit receipt predates the training summary")
    if receipt.get("predecessor_receipt_binding") is not None:
        raise ScreenValidationError(
            "optimization-screen receipt must not claim an R/S/E predecessor"
        )
    contract = _mapping(receipt.get("expected_contract"), "receipt expected contract")
    expected_contract = {
        "training_summary_schema_version": EXPECTED_ARTIFACT_SCHEMA_VERSIONS[
            "training_summary"
        ],
        "source_revision": source_revision,
        "resolved_training_config_sha256": config_entry["config"]["canonical_sha256"],
        "training_argv_sha256": manifest.get("training_argv_sha256"),
        "launch_manifest_path": manifest.get("launch_manifest_path"),
        "launch_manifest_sha256": manifest_ref["sha256"],
        "selected_gpu_uuids": list(selected_uuids),
        "training_job_lock_path": lock_binding["path"],
        "training_job_lock_sha256": lock_binding["sha256"],
        "max_steps": EXPECTED_UPDATES[stage_id],
        "world_size": registry.data["common_training"]["gpu_count"],
        "training_summary_path": manifest.get("training_summary_path"),
        "final_checkpoint_path": manifest.get("expected_final_checkpoint_path"),
        "initialization_checkpoint_sha256": registry.data["common_training"][
            "initialization"
        ]["checkpoint"]["sha256"],
    }
    if not _exact_json_equal(contract, expected_contract):
        raise ScreenValidationError("receipt expected contract is unmatched")
    pipeline = _mapping(receipt.get("pipeline"), "receipt pipeline")
    _exact_keys(
        pipeline,
        {"training", "tee", "pipefail_shell_exit_status"},
        "receipt pipeline",
    )
    expected_pipeline_component = {
        "possible_termination_signal": None,
        "shell_exit_status": 0,
        "shell_status_is_signal_compatible": False,
        "signal_provenance": None,
        "succeeded": True,
    }
    if (
        not _exact_json_equal(pipeline.get("training"), expected_pipeline_component)
        or not _exact_json_equal(pipeline.get("tee"), expected_pipeline_component)
        or not _exact_json_equal(pipeline.get("pipefail_shell_exit_status"), 0)
    ):
        raise ScreenValidationError("receipt pipeline does not certify clean exit")
    source_at_receipt = _mapping(
        receipt.get("source_at_receipt"), "receipt source evidence"
    )
    expected_source = {
        "verified": True,
        "expected_revision": source_revision,
        "head": source_revision,
        "upstream": source_revision,
        "output_directory_excluded_from_cleanliness_check": True,
    }
    if not _exact_json_equal(source_at_receipt, expected_source):
        raise ScreenValidationError("receipt source evidence is unmatched")

    launch_evidence = _mapping(
        receipt.get("launch_manifest"), "receipt launch-manifest evidence"
    )
    _exact_keys(
        launch_evidence,
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
    launch_snapshot = _validate_stable_snapshot(
        launch_evidence.get("artifact"),
        label="receipt launch-manifest snapshot",
        expected_path=manifest["launch_manifest_path"],
        expected_ref=manifest_ref,
    )
    expected_launch_evidence = {
        "path": manifest["launch_manifest_path"],
        "present": True,
        "matches_expected_raw_sha256": True,
        "selected_gpu_uuids_match_expected": True,
        "matches_training_summary_snapshot": True,
        "matches_runtime_config_snapshot": True,
        "valid_and_launch_bound": True,
        "expected_selected_gpu_uuids": list(selected_uuids),
        "observed_selected_gpu_uuids": list(selected_uuids),
        "artifact": launch_snapshot,
        "validation_error": None,
    }
    if not _exact_json_equal(
        launch_evidence, expected_launch_evidence
    ) or not _exact_json_equal(
        launch_snapshot,
        _snapshot_base(summary_validation["launch_manifest"]),
    ):
        raise ScreenValidationError("receipt launch-manifest evidence is unmatched")

    lock_evidence = _mapping(
        receipt.get("training_job_lock"), "receipt training-job lock evidence"
    )
    _exact_keys(
        lock_evidence,
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
    lock_payload = (
        json.dumps(
            dict(lock_binding["record"]), indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")
    lock_snapshot = _validate_stable_snapshot(
        lock_evidence.get("artifact"),
        label="receipt training-job lock snapshot",
        expected_path=lock_binding["path"],
        expected_ref={
            "sha256": lock_binding["sha256"],
            "size_bytes": len(lock_payload),
        },
    )
    expected_lock_evidence = {
        "path": lock_binding["path"],
        "present": True,
        "expected_sha256": lock_binding["sha256"],
        "matches_expected_raw_sha256": True,
        "matches_launch_manifest_binding": True,
        "valid_and_launch_bound_before_receipt_publication": True,
        "artifact": lock_snapshot,
        "record": lock_binding["record"],
        "release_policy": (
            "publish_receipt_then_unlink_only_same_stat_identity_and_sha256"
        ),
        "release_result_not_claimed_inside_pre_release_receipt": True,
        "validation_error": None,
    }
    if not _exact_json_equal(lock_evidence, expected_lock_evidence):
        raise ScreenValidationError("receipt training-job lock evidence is unmatched")

    completion = _mapping(
        receipt.get("completion_requirements"),
        "receipt completion requirements",
    )
    _exact_keys(completion, _EXIT_COMPLETION_KEYS, "receipt completion requirements")
    if any(
        _boolean(value, f"receipt completion requirement {key}") is not True
        for key, value in completion.items()
    ):
        raise ScreenValidationError("receipt completion requirements are not all true")

    summary_evidence = _mapping(
        receipt.get("training_summary"), "receipt training-summary evidence"
    )
    _exact_keys(
        summary_evidence,
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
    summary_snapshot = _validate_stable_snapshot(
        summary_evidence.get("artifact"),
        label="receipt training-summary snapshot",
        expected_path=manifest["training_summary_path"],
        expected_ref=summary_ref,
    )
    bindings = _mapping(
        summary_evidence.get("validated_bindings"), "receipt validated bindings"
    )
    expected_bindings = {
        "schema_version": EXPECTED_ARTIFACT_SCHEMA_VERSIONS["training_summary"],
        "source_revision": source_revision,
        "resolved_training_config_sha256": config_entry["config"]["canonical_sha256"],
        "training_argv_sha256": manifest["training_argv_sha256"],
        "launch_manifest_path": manifest["launch_manifest_path"],
        "launch_manifest_sha256": manifest_ref["sha256"],
        "selected_gpu_uuids": list(selected_uuids),
        "observed_global_step": EXPECTED_UPDATES[stage_id],
        "observed_world_size": registry.data["common_training"]["gpu_count"],
        "training_accounting": summary_validation["training_accounting"],
        "ema_metadata": summary_validation["ema_metadata"],
        "final_checkpoint_path": manifest["expected_final_checkpoint_path"],
        "final_checkpoint_sha256": checkpoint["sha256"],
        "startup_mode": summary_validation["startup_mode"],
        "conditioning_gradient_audit": gradient_audit,
        "screen_initialization_state_audit": initialization["state_audit"],
    }
    if not _exact_json_equal(bindings, expected_bindings):
        raise ScreenValidationError("receipt validated bindings are unmatched")
    expected_summary_evidence = {
        "path": manifest["training_summary_path"],
        "present": True,
        "valid_and_launch_bound": True,
        "artifact": summary_snapshot,
        "validated_bindings": bindings,
        "validation_error": None,
    }
    if not _exact_json_equal(summary_evidence, expected_summary_evidence):
        raise ScreenValidationError("receipt does not validate the bound summary")

    checkpoint_evidence = _mapping(
        receipt.get("final_checkpoint"), "receipt checkpoint evidence"
    )
    _exact_keys(
        checkpoint_evidence,
        {"path", "present", "matches_training_summary_snapshot", "artifact"},
        "receipt checkpoint evidence",
    )
    checkpoint_snapshot = _validate_stable_snapshot(
        checkpoint_evidence.get("artifact"),
        label="receipt final-checkpoint snapshot",
        expected_path=manifest["expected_final_checkpoint_path"],
        expected_ref=checkpoint,
    )
    expected_checkpoint_evidence = {
        "path": manifest["expected_final_checkpoint_path"],
        "present": True,
        "matches_training_summary_snapshot": True,
        "artifact": checkpoint_snapshot,
    }
    if not _exact_json_equal(
        checkpoint_evidence, expected_checkpoint_evidence
    ) or not _exact_json_equal(
        checkpoint_snapshot,
        _snapshot_base(summary_validation["final_checkpoint"]),
    ):
        raise ScreenValidationError("receipt does not validate the final checkpoint")

    runtime_evidence = _mapping(receipt.get("runtime_config"), "receipt runtime config")
    _exact_keys(
        runtime_evidence,
        {
            "path",
            "present",
            "matches_training_summary_snapshot",
            "semantic_validation_passed",
            "artifact",
        },
        "receipt runtime-config evidence",
    )
    runtime_snapshot = _validate_stable_snapshot(
        runtime_evidence.get("artifact"),
        label="receipt runtime-config snapshot",
        expected_path=manifest["runtime_config_path"],
        expected_ref=runtime_ref,
    )
    expected_runtime_evidence = {
        "path": manifest["runtime_config_path"],
        "present": True,
        "matches_training_summary_snapshot": True,
        "semantic_validation_passed": True,
        "artifact": runtime_snapshot,
    }
    if not _exact_json_equal(
        runtime_evidence, expected_runtime_evidence
    ) or not _exact_json_equal(
        runtime_snapshot,
        _snapshot_base(summary_validation["runtime_config"]),
    ):
        raise ScreenValidationError("receipt runtime_config is not valid")


def _validate_training_artifacts(
    refs: Mapping[str, Mapping[str, Any]],
    *,
    loader: BlobLoader,
    registry: ValidatedRegistry,
    stage_id: str,
    arm: Mapping[str, Any],
    config_entry: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    gradient_audit: object,
    initialization: Mapping[str, Any],
    source_revision: str,
    scheduler_dependency: Mapping[str, Any] | None,
) -> None:
    documents = {
        name: _load_json_ref(
            ref,
            loader=loader,
            label=f"{arm['arm_id']} {name}",
            schema_field=(
                "launch_manifest_schema_version"
                if name == "launch_manifest"
                else "schema_version"
            ),
        )
        for name, ref in refs.items()
    }
    for name, expected in EXPECTED_ARTIFACT_SCHEMA_VERSIONS.items():
        if name != "denoising_report" and refs[name]["schema_version"] != expected:
            raise ScreenValidationError(f"{name} reference schema is unsupported")
    selected_uuids = _validate_launch_manifest(
        documents["launch_manifest"],
        registry=registry,
        stage_id=stage_id,
        arm=arm,
        config_entry=config_entry,
        source_revision=source_revision,
        scheduler_dependency=scheduler_dependency,
    )
    resolved_config = _load_config_ref(
        config_entry["config"],
        loader=loader,
        label=f"{arm['arm_id']} resolved training config",
    )
    manifest = documents["launch_manifest"]
    lock_binding = _validate_manifest_training_bindings(
        manifest,
        refs=refs,
        checkpoint=checkpoint,
        arm=arm,
        resolved_config=resolved_config,
        resolved_config_sha256=config_entry["config"]["canonical_sha256"],
        source_revision=source_revision,
    )
    runtime_validation = _validate_runtime_record(
        documents["runtime_config"],
        registry=registry,
        manifest=manifest,
        manifest_ref=refs["launch_manifest"],
        selected_uuids=selected_uuids,
        resolved_config=resolved_config,
        resolved_config_sha256=config_entry["config"]["canonical_sha256"],
        expected_steps=EXPECTED_UPDATES[stage_id],
        source_revision=source_revision,
    )
    summary_validation = _validate_training_summary(
        documents["training_summary"],
        registry=registry,
        stage_id=stage_id,
        arm=arm,
        config_entry=config_entry,
        manifest=manifest,
        manifest_ref=refs["launch_manifest"],
        runtime=documents["runtime_config"],
        runtime_ref=refs["runtime_config"],
        runtime_validation=runtime_validation,
        checkpoint=checkpoint,
        gradient_audit=gradient_audit,
        initialization=initialization,
        resolved_config=resolved_config,
        source_revision=source_revision,
    )
    _validate_exit_receipt(
        documents["exit_receipt"],
        registry=registry,
        stage_id=stage_id,
        config_entry=config_entry,
        manifest=manifest,
        manifest_ref=refs["launch_manifest"],
        runtime_ref=refs["runtime_config"],
        summary_ref=refs["training_summary"],
        summary_validation=summary_validation,
        lock_binding=lock_binding,
        checkpoint=checkpoint,
        selected_uuids=selected_uuids,
        gradient_audit=gradient_audit,
        initialization=initialization,
        source_revision=source_revision,
    )


def _validate_gradient_audit(
    value: object,
    *,
    contract: Mapping[str, Any],
    contract_sha256: str,
    scheduler_arm_id: str,
) -> bool:
    audit = _mapping(value, "conditioning gradient audit")
    _exact_keys(
        audit,
        {
            "schema_version",
            "status",
            "observation_point",
            "registered_contract_sha256",
            "first_positive_lr_optimizer_step",
            "timestep_mlp_required_optimizer_check",
            "optimizer_checks",
        },
        "conditioning gradient audit",
    )
    if (
        _integer(audit.get("schema_version"), "gradient audit schema")
        != GRADIENT_AUDIT_SCHEMA_VERSION
        or audit.get("status") != "completed"
        or audit.get("observation_point") != EXPECTED_OBSERVATION_POINT
        or audit.get("registered_contract_sha256") != contract_sha256
        or audit.get("first_positive_lr_optimizer_step") != 2
        or audit.get("timestep_mlp_required_optimizer_check") != 3
    ):
        raise ScreenValidationError("conditioning gradient audit header is unmatched")
    checks = audit.get("optimizer_checks")
    if not isinstance(checks, list) or len(checks) != 3:
        raise ScreenValidationError("conditioning gradient audit needs three checks")
    contract_groups = {group["group_id"]: group for group in contract["groups"]}
    all_gates_pass = True
    for index, raw_check in enumerate(checks, start=1):
        label = f"conditioning gradient optimizer check {index}"
        check = _mapping(raw_check, label)
        _exact_keys(
            check,
            {
                "optimizer_gradient_observation_index",
                "optimizer_step_index",
                "learning_rate_before_step",
                "film_groups",
                "timestep_mlp_groups",
            },
            label,
        )
        if (
            check.get("optimizer_gradient_observation_index") != index
            or check.get("optimizer_step_index") != index
        ):
            raise ScreenValidationError(f"{label} index is unmatched")
        learning_rate = _decimal(
            check.get("learning_rate_before_step"),
            f"{label} learning rate",
            nonnegative=True,
        )
        if scheduler_arm_id not in {"E-L0", "E-L1"}:
            raise ScreenValidationError("gradient audit lacks a selected scheduler")
        warmup_updates = 2500 if scheduler_arm_id == "E-L0" else 50
        expected_learning_rate = Decimal(str(0.0003 * ((index - 1) / warmup_updates)))
        if learning_rate != expected_learning_rate:
            raise ScreenValidationError(f"{label} learning-rate timing is unmatched")
        observed_ids: set[str] = set()
        for collection_name, expected_kind in (
            ("film_groups", "film"),
            ("timestep_mlp_groups", "timestep_mlp"),
        ):
            reports = check.get(collection_name)
            expected_groups = [
                group for group in contract["groups"] if group["kind"] == expected_kind
            ]
            if not isinstance(reports, list) or len(reports) != len(expected_groups):
                raise ScreenValidationError(f"{label}.{collection_name} is unmatched")
            for raw_report, contract_group in zip(
                reports, expected_groups, strict=True
            ):
                group_label = f"{label} {contract_group['group_id']}"
                report = _mapping(raw_report, group_label)
                _exact_keys(
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
                    group_label,
                )
                group_id = report.get("group_id")
                if group_id in observed_ids or group_id not in contract_groups:
                    raise ScreenValidationError(f"{group_label} identity is invalid")
                observed_ids.add(group_id)
                expected_elements = sum(
                    math.prod(parameter["shape"])
                    for parameter in contract_group["parameters"]
                )
                if (
                    group_id != contract_group["group_id"]
                    or report.get("ordered_parameter_manifest_sha256")
                    != canonical_json_sha256(contract_group["parameters"])
                    or report.get("parameter_count")
                    != len(contract_group["parameters"])
                    or report.get("gradient_element_count") != expected_elements
                ):
                    raise ScreenValidationError(f"{group_label} manifest is unmatched")
                present = _boolean(
                    report.get("all_gradients_present"), f"{group_label} present"
                )
                finite = _boolean(
                    report.get("all_gradients_finite"), f"{group_label} finite"
                )
                nonzero = _boolean(
                    report.get("all_parameter_gradients_nonzero"),
                    f"{group_label} nonzero",
                )
                all_gates_pass &= present and finite
                if expected_kind == "film" and index == 1:
                    all_gates_pass &= nonzero
                if expected_kind == "timestep_mlp" and index == 3:
                    all_gates_pass &= nonzero
        if observed_ids != set(contract_groups):
            raise ScreenValidationError(f"{label} omits a registered group")
    return all_gates_pass


def _validate_initialization_audit(
    value: object,
    *,
    loader: BlobLoader,
    registry: ValidatedRegistry,
    reference_config_entry: Mapping[str, Any],
    candidate_config_entry: Mapping[str, Any],
    source_revision: str,
) -> bool:
    audit = _mapping(value, "conditioning initialization audit")
    _exact_keys(
        audit,
        {
            "schema_version",
            "reference_arm_id",
            "candidate_arm_id",
            "fixture",
            "source_revision",
            "checkpoint_sha256",
            "reference_config_canonical_sha256",
            "candidate_config_canonical_sha256",
            "probe_phase",
            "input_ids_sha256",
            "attention_mask_sha256",
            "noise_tensor_sha256",
            "timestep_tensor_sha256",
            "logits_dtype",
            "logits_shape",
            "reference_logits",
            "candidate_logits",
            "producer_source",
            "exact_equal",
        },
        "conditioning initialization audit",
    )
    if (
        _integer(audit.get("schema_version"), "initialization audit schema")
        != INITIALIZATION_AUDIT_SCHEMA_VERSION
        or audit.get("reference_arm_id") != "E-A0"
        or audit.get("candidate_arm_id") != "E-A1"
        or audit.get("source_revision") != source_revision
        or audit.get("checkpoint_sha256")
        != registry.data["common_training"]["initialization"]["checkpoint"]["sha256"]
        or audit.get("reference_config_canonical_sha256")
        != reference_config_entry["config"]["canonical_sha256"]
        or audit.get("candidate_config_canonical_sha256")
        != candidate_config_entry["config"]["canonical_sha256"]
        or audit.get("probe_phase")
        != "after_mdlm_ema_load_before_training_rng_reseed_and_optimizer_creation"
    ):
        raise ScreenValidationError("conditioning initialization audit is unmatched")
    fixture_ref = _json_ref(audit.get("fixture"), "conditioning initialization fixture")
    if fixture_ref != registry.data["stages"][1]["initialization_fixture"]:
        raise ScreenValidationError(
            "conditioning initialization fixture is unregistered"
        )
    fixture = _load_json_ref(
        fixture_ref, loader=loader, label="conditioning initialization fixture"
    )
    if fixture_ref["schema_version"] != 1:
        raise ScreenValidationError(
            "conditioning initialization fixture schema is invalid"
        )
    for field in (
        "input_ids_sha256",
        "attention_mask_sha256",
        "noise_tensor_sha256",
        "timestep_tensor_sha256",
    ):
        digest = _sha256(audit.get(field), f"conditioning initialization {field}")
        fixture_field = field.removesuffix("_sha256")
        if canonical_json_sha256(fixture.get(fixture_field)) != digest:
            raise ScreenValidationError(
                "conditioning initialization fixture is unmatched"
            )
    if audit.get("logits_dtype") != "float32-little-endian-c-order":
        raise ScreenValidationError(
            "conditioning initialization logits dtype is invalid"
        )
    shape = audit.get("logits_shape")
    if (
        not isinstance(shape, list)
        or not shape
        or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
    ):
        raise ScreenValidationError(
            "conditioning initialization logits shape is invalid"
        )
    producer_ref = _blob_ref(
        audit.get("producer_source"),
        "conditioning initialization audit producer",
        required_root="repository",
        suffix=".py",
    )
    producer_contract_ref = next(
        ref
        for ref in registry.data["source"]["blobs"]
        if ref["relative_path"] == "scripts/udlm/audit_conditioning_initialization.py"
    )
    if producer_ref != producer_contract_ref:
        raise ScreenValidationError(
            "conditioning initialization producer is unregistered"
        )
    _verify_git_blob(
        producer_ref,
        revision=source_revision,
        git_blob_loader=registry.git_blob_loader,
        label="conditioning initialization producer",
    )
    reference_ref = _blob_ref(
        audit.get("reference_logits"),
        "reference initialization logits",
        required_root="repository",
    )
    candidate_ref = _blob_ref(
        audit.get("candidate_logits"),
        "candidate initialization logits",
        required_root="repository",
    )
    if (
        "E-A0" not in PurePosixPath(reference_ref["relative_path"]).name
        or "E-A1" not in PurePosixPath(candidate_ref["relative_path"]).name
        or reference_ref["relative_path"] == candidate_ref["relative_path"]
    ):
        raise ScreenValidationError(
            "initialization logit artifact paths are not arm-specific"
        )
    _require_owned_artifacts(
        [reference_ref],
        output_directory=reference_config_entry["output_directory"],
        label="reference initialization logits",
    )
    _require_owned_artifacts(
        [candidate_ref],
        output_directory=candidate_config_entry["output_directory"],
        label="candidate initialization logits",
    )
    expected_size_bytes = math.prod(shape) * 4
    if (
        reference_ref["size_bytes"] != expected_size_bytes
        or candidate_ref["size_bytes"] != expected_size_bytes
    ):
        raise ScreenValidationError("initialization logit byte sizes are unmatched")
    reference_payload = _load_bound_blob(
        reference_ref, loader=loader, label="reference initialization logits"
    )
    candidate_payload = _load_bound_blob(
        candidate_ref, loader=loader, label="candidate initialization logits"
    )
    if reference_payload is None or candidate_payload is None:
        raise ScreenValidationError("initialization logits must be retained byte blobs")
    exact_equal = _boolean(audit.get("exact_equal"), "initialization equality")
    return (
        exact_equal
        and reference_ref["sha256"] == candidate_ref["sha256"]
        and reference_payload == candidate_payload
    )


def _validate_evaluator_report(
    report: Mapping[str, Any],
    *,
    registry: ValidatedRegistry,
    config_entry: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    source_revision: str,
    arm_id: str,
) -> list[dict[str, Any]]:
    if (
        _integer(report.get("schema_version"), "evaluator report schema")
        != EVALUATOR_REPORT_SCHEMA_VERSION
    ):
        raise ScreenValidationError("evaluator report must use current schema 4")
    conditioning = _mapping(report.get("conditioning"), "evaluator conditioning")
    validation = _mapping(
        conditioning.get("validation"), "evaluator conditioning validation"
    )
    if any(
        validation.get(field) is not True
        for field in (
            "before_strict_state_load",
            "strict_state_load",
            "after_strict_state_load",
            "runtime_identity_unchanged",
        )
    ):
        raise ScreenValidationError("evaluator conditioning load validation failed")
    if arm_id != "E-A1":
        expected_additive = {
            "runtime_conditioning_variant": "additive",
            "checkpoint_metadata_required": False,
            "checkpoint_metadata_present": False,
            "checkpoint_metadata_validation": "correctly_absent_for_additive",
            "runtime_metadata": None,
            "runtime_metadata_canonical_sha256": None,
            "checkpoint_metadata": None,
            "checkpoint_metadata_canonical_sha256": None,
        }
        if any(
            not _exact_json_equal(conditioning.get(key), value)
            for key, value in expected_additive.items()
        ):
            raise ScreenValidationError(
                "additive evaluator conditioning identity is invalid"
            )
    else:
        runtime_metadata = _mapping(
            conditioning.get("runtime_metadata"), "runtime FiLM metadata"
        )
        checkpoint_metadata = _mapping(
            conditioning.get("checkpoint_metadata"), "checkpoint FiLM metadata"
        )
        runtime_digest = canonical_json_sha256(runtime_metadata)
        if (
            conditioning.get("runtime_conditioning_variant") != "film_adaln"
            or conditioning.get("checkpoint_metadata_required") is not True
            or conditioning.get("checkpoint_metadata_present") is not True
            or conditioning.get("checkpoint_metadata_validation")
            != "required_record_matches_runtime_exactly"
            or not _exact_json_equal(runtime_metadata, checkpoint_metadata)
            or conditioning.get("runtime_metadata_canonical_sha256") != runtime_digest
            or conditioning.get("checkpoint_metadata_canonical_sha256")
            != runtime_digest
            or _integer(runtime_metadata.get("schema_version"), "FiLM metadata schema")
            != 1
            or runtime_metadata.get("variant") != "film_adaln"
            or _integer(runtime_metadata.get("hidden_size"), "FiLM hidden size") != 768
            or _integer(runtime_metadata.get("layer_count"), "FiLM layer count") != 12
        ):
            raise ScreenValidationError(
                "FiLM evaluator conditioning identity is invalid"
            )
        manifest = runtime_metadata.get("conditioning_parameter_manifest")
        contract_parameters = [
            {
                "name": parameter["name"].removeprefix("backbone."),
                "shape": parameter["shape"],
            }
            for group in registry.data["stages"][1]["gradient_contract"]["groups"]
            for parameter in group["parameters"]
        ]
        if not isinstance(manifest, list) or any(
            not isinstance(item, Mapping)
            or set(item) != {"name", "shape"}
            or not isinstance(item.get("name"), str)
            or not isinstance(item.get("shape"), list)
            for item in manifest
        ):
            raise ScreenValidationError("FiLM evaluator manifest is malformed")
        if len(manifest) != 28 or canonical_json_sha256(
            sorted(manifest, key=lambda item: item.get("name", ""))
        ) != canonical_json_sha256(
            sorted(contract_parameters, key=lambda item: item["name"])
        ):
            raise ScreenValidationError("FiLM evaluator manifest differs from registry")
    source = _mapping(report.get("source"), "evaluator source provenance")
    if (
        source.get("git_commit") != source_revision
        or source.get("git_dirty") is not False
        or source.get("git_worktree_state") != "clean"
        or _mapping(source.get("postcheck"), "evaluator source postcheck").get("status")
        != "unchanged_after_checkpoint_load_and_evaluation"
    ):
        raise ScreenValidationError("evaluator source provenance is unmatched")
    source_hashes = _mapping(source.get("files_sha256"), "evaluator source hashes")
    registry_hashes = {
        ref["relative_path"]: ref["sha256"] for ref in registry.data["source"]["blobs"]
    }
    if set(source_hashes) != set(EVALUATOR_SOURCE_PATHS):
        raise ScreenValidationError("evaluator source-hash map is not schema-exact")
    for path in EVALUATOR_SOURCE_PATHS:
        if source_hashes.get(path) != registry_hashes[path]:
            raise ScreenValidationError("evaluator implementation hash is unmatched")
    checkpoint_report = _mapping(report.get("checkpoint"), "evaluator checkpoint")
    if (
        checkpoint_report.get("sha256") != checkpoint["sha256"]
        or checkpoint_report.get("size_bytes") != checkpoint["size_bytes"]
        or checkpoint_report.get("global_step") != checkpoint["global_step"]
        or checkpoint_report.get("config_sha256")
        != config_entry["config"]["canonical_sha256"]
        or checkpoint_report.get("weights_evaluated") != "ema"
        or checkpoint_report.get("diffusion_type") != "udlm"
    ):
        raise ScreenValidationError("evaluator checkpoint binding is unmatched")
    checkpoint_conditioning = checkpoint_report.get("udlm_conditioning_metadata")
    checkpoint_conditioning_sha = checkpoint_report.get(
        "udlm_conditioning_metadata_sha256"
    )
    if arm_id == "E-A1":
        if (
            checkpoint_report.get("udlm_conditioning_metadata_declared") is not True
            or not _exact_json_equal(
                checkpoint_conditioning, conditioning.get("checkpoint_metadata")
            )
            or checkpoint_conditioning_sha
            != conditioning.get("checkpoint_metadata_canonical_sha256")
        ):
            raise ScreenValidationError(
                "evaluator checkpoint FiLM metadata is unmatched"
            )
    elif (
        checkpoint_report.get("udlm_conditioning_metadata_declared") is not False
        or checkpoint_conditioning is not None
        or checkpoint_conditioning_sha is not None
    ):
        raise ScreenValidationError("additive checkpoint declares FiLM metadata")
    artifacts = _mapping(report.get("artifacts"), "evaluator artifact provenance")
    panel = _mapping(artifacts.get("panel"), "evaluator panel provenance")
    frequency = _mapping(
        artifacts.get("training_frequency"), "evaluator frequency provenance"
    )
    if (
        panel.get("sha256") != registry.data["panel"]["artifact"]["sha256"]
        or panel.get("ordered_token_ids_sha256") != EXPECTED_PANEL_TOKEN_IDS_SHA256
        or frequency.get("sha256")
        != registry.data["panel"]["frequency_artifact"]["sha256"]
    ):
        raise ScreenValidationError("evaluator input artifacts are unmatched")
    evaluation = _mapping(report.get("evaluation"), "evaluator evaluation")
    process = _mapping(evaluation.get("process"), "evaluator process")
    if process.get("prior_variant") != "empirical_frequency":
        raise ScreenValidationError("evaluator did not use the registered E process")
    weight_application = _mapping(
        checkpoint_report.get("weight_application"), "evaluator weight application"
    )
    if weight_application.get("source") != "checkpoint.ema.shadow_params":
        raise ScreenValidationError("evaluator did not apply EMA weights")
    if not _exact_json_equal(
        weight_application.get("udlm_conditioning_identity"), conditioning
    ):
        raise ScreenValidationError("EMA conditioning identity is not cross-bound")
    observed_times = evaluation.get("time_bins")
    if (
        not isinstance(observed_times, list)
        or tuple(
            _decimal(item, f"evaluator time {index}")
            for index, item in enumerate(observed_times)
        )
        != EXPECTED_TIME_BINS
    ):
        raise ScreenValidationError("evaluator time grid is unmatched")
    if (
        evaluation.get("rows_evaluated") != EXPECTED_PANEL_ROWS
        or evaluation.get("content_tokens_per_time_bin")
        != EXPECTED_PANEL_CONTENT_TOKENS
        or evaluation.get("seed") != EXPECTED_CORRUPTION_SEED
        or evaluation.get("device") != "cpu"
        or evaluation.get("batch_size") != registry.data["panel"]["batch_size"]
    ):
        raise ScreenValidationError("evaluator execution contract is unmatched")
    corruption_grid_sha = _sha256(
        evaluation.get("corruption_grid_sha256"), "evaluator corruption grid"
    )
    bins = evaluation.get("metrics_by_time")
    if not isinstance(bins, list) or len(bins) != len(EXPECTED_TIME_BINS):
        raise ScreenValidationError("evaluator metric bins are unmatched")
    normalized = []
    grid_items = []
    for index, (raw_bin, expected_time) in enumerate(
        zip(bins, EXPECTED_TIME_BINS, strict=True)
    ):
        label = f"evaluator metric bin {index}"
        bin_record = _mapping(raw_bin, label)
        if _decimal(bin_record.get("time"), f"{label}.time") != expected_time:
            raise ScreenValidationError(f"{label} time is unmatched")
        seeds_sha = _sha256(
            bin_record.get("row_corruption_seeds_sha256"), f"{label} row seeds"
        )
        corruptions_sha = _sha256(
            bin_record.get("corrupted_token_ids_sha256"), f"{label} corruptions"
        )
        overall = _mapping(
            _mapping(bin_record.get("metrics"), f"{label} metrics").get("overall"),
            f"{label} overall metrics",
        )
        denominator = _integer(
            overall.get("denominator_tokens"), f"{label} denominator", minimum=1
        )
        if denominator != EXPECTED_PANEL_CONTENT_TOKENS:
            raise ScreenValidationError(f"{label} denominator is unmatched")
        loss_sum = _decimal(
            overall.get("production_loss_sum"), f"{label} loss sum", nonnegative=True
        )
        correct = _integer(
            overall.get("clean_token_top1_correct"),
            f"{label} correct count",
            minimum=0,
            maximum=denominator,
        )
        normalized.append(
            {
                "time": expected_time,
                "loss_sum": loss_sum,
                "denominator": denominator,
                "correct": correct,
                "row_corruption_seeds_sha256": seeds_sha,
                "corrupted_token_ids_sha256": corruptions_sha,
                "corruption_grid_sha256": corruption_grid_sha,
            }
        )
        grid_items.append(
            {
                "time": float(expected_time),
                "row_corruption_seeds_sha256": seeds_sha,
                "corrupted_token_ids_sha256": corruptions_sha,
            }
        )
    if canonical_json_sha256(grid_items) != corruption_grid_sha:
        raise ScreenValidationError("evaluator corruption-grid digest is invalid")
    return normalized


def _validate_denoising_wrapper(
    value: object,
    *,
    loader: BlobLoader,
    registry: ValidatedRegistry,
    stage_id: str,
    arm: Mapping[str, Any],
    config_entry: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    source_revision: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    wrapper = _mapping(value, f"{arm['arm_id']} denoising report wrapper")
    _exact_keys(
        wrapper,
        {
            "schema_version",
            "artifact_kind",
            "registry_sha256",
            "registry_canonical_sha256",
            "stage_id",
            "arm_id",
            "attempt_id",
            "source_revision",
            "resolved_config_canonical_sha256",
            "checkpoint_sha256",
            "producer_source",
            "evaluator_report",
            "generation_metrics_included",
            "final_generation_seeds_included",
        },
        f"{arm['arm_id']} denoising report wrapper",
    )
    expected_values = {
        "schema_version": DENOISING_REPORT_SCHEMA_VERSION,
        "artifact_kind": "optimization_screen_denoising_evaluation_binding",
        "registry_sha256": registry.raw_sha256,
        "registry_canonical_sha256": registry.canonical_sha256,
        "stage_id": stage_id,
        "arm_id": arm["arm_id"],
        "attempt_id": arm["attempt_id"],
        "source_revision": source_revision,
        "resolved_config_canonical_sha256": config_entry["config"]["canonical_sha256"],
        "checkpoint_sha256": checkpoint["sha256"],
        "generation_metrics_included": False,
        "final_generation_seeds_included": [],
    }
    if any(
        not _exact_json_equal(wrapper.get(key), expected)
        for key, expected in expected_values.items()
    ):
        raise ScreenValidationError("denoising wrapper binding is unmatched")
    collector_ref = next(
        ref
        for ref in registry.data["source"]["blobs"]
        if ref["relative_path"]
        == "scripts/udlm/collect_optimization_screen_evidence.py"
    )
    if (
        _blob_ref(
            wrapper.get("producer_source"),
            "denoising wrapper producer",
            required_root="repository",
            suffix=".py",
        )
        != collector_ref
    ):
        raise ScreenValidationError("denoising wrapper producer is unregistered")
    _verify_git_blob(
        collector_ref,
        revision=source_revision,
        git_blob_loader=registry.git_blob_loader,
        label="denoising wrapper producer",
    )
    evaluator_ref = _json_ref(
        wrapper.get("evaluator_report"),
        f"{arm['arm_id']} evaluator report",
        required_root="repository",
    )
    if evaluator_ref["schema_version"] != EVALUATOR_REPORT_SCHEMA_VERSION:
        raise ScreenValidationError("evaluator report reference must use schema 4")
    evaluator = _load_json_ref(
        evaluator_ref, loader=loader, label=f"{arm['arm_id']} evaluator report"
    )
    bins = _validate_evaluator_report(
        evaluator,
        registry=registry,
        config_entry=config_entry,
        checkpoint=checkpoint,
        source_revision=source_revision,
        arm_id=arm["arm_id"],
    )
    return evaluator_ref, bins


def _evidence_reference(
    *, payload: bytes, relative_path: str, parsed: Mapping[str, Any] | None
) -> dict[str, Any]:
    return {
        "relative_path": _relative_path(
            relative_path, "evidence path", suffix=".json"
        ).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "canonical_sha256": (None if parsed is None else canonical_json_sha256(parsed)),
        "schema_version": (None if parsed is None else parsed.get("schema_version")),
    }


def _validate_attempt(
    value: object,
    *,
    loader: BlobLoader,
    registry: ValidatedRegistry,
    stage_id: str,
    arm: Mapping[str, Any],
    scheduler_arm_id: str | None,
    source_revision: str,
    scheduler_dependency: Mapping[str, Any] | None,
) -> dict[str, Any]:
    label = f"{arm['arm_id']} attempt evidence"
    attempt = _mapping(value, label)
    _exact_keys(
        attempt,
        {
            "attempt_id",
            "arm_id",
            "status",
            "failure_reason",
            "training_seed",
            "optimizer_updates",
            "gpu_count",
            "source_revision",
            "output_directory",
            "resolved_config",
            "artifacts",
            "checkpoint",
            "initialization",
            "denoising_report",
            "conditioning_gradient_audit",
        },
        label,
    )
    config_entry = _registered_config(arm, scheduler_arm_id=scheduler_arm_id)
    expected_identity = {
        "attempt_id": arm["attempt_id"],
        "arm_id": arm["arm_id"],
        "training_seed": EXPECTED_TRAINING_SEED,
        "optimizer_updates": EXPECTED_UPDATES[stage_id],
        "gpu_count": registry.data["common_training"]["gpu_count"],
        "source_revision": source_revision,
        "output_directory": config_entry["output_directory"],
        "resolved_config": config_entry["config"],
    }
    if any(
        not _exact_json_equal(attempt.get(key), expected)
        for key, expected in expected_identity.items()
    ):
        raise ScreenValidationError(f"{label} identity is unmatched")
    status_value = attempt.get("status")
    if status_value == "failed":
        if (
            not isinstance(attempt.get("failure_reason"), str)
            or not attempt["failure_reason"]
        ):
            raise ScreenValidationError(f"{label} lacks a failure reason")
        if any(
            attempt.get(field) is not None
            for field in (
                "artifacts",
                "checkpoint",
                "initialization",
                "denoising_report",
                "conditioning_gradient_audit",
            )
        ):
            raise ScreenValidationError(f"{label} failed record contains outputs")
        raise EvidenceIncomplete(f"registered attempt {arm['attempt_id']} failed")
    if status_value != "completed" or attempt.get("failure_reason") is not None:
        raise ScreenValidationError(f"{label} status is invalid")
    checkpoint = _checkpoint_ref(attempt.get("checkpoint"), f"{label}.checkpoint")
    if checkpoint["global_step"] != EXPECTED_UPDATES[stage_id]:
        raise ScreenValidationError(f"{label} checkpoint step is unmatched")
    _load_bound_blob(checkpoint, loader=loader, label=f"{label}.checkpoint")
    initialization = _validate_initialization(
        attempt.get("initialization"),
        registry=registry,
        arm_id=arm["arm_id"],
        resolved_config_sha256=config_entry["config"]["canonical_sha256"],
    )
    gradient_audit = attempt.get("conditioning_gradient_audit")
    gradient_gate: bool | None
    if arm["arm_id"] == "E-A1":
        stage = _stage(registry, "conditioning")
        gradient_gate = _validate_gradient_audit(
            gradient_audit,
            contract=stage["gradient_contract"],
            contract_sha256=stage["gradient_contract_sha256"],
            scheduler_arm_id=scheduler_arm_id,
        )
    else:
        if gradient_audit is not None:
            raise ScreenValidationError(f"{label} has an inapplicable gradient audit")
        gradient_gate = None
    artifacts = _artifact_set(attempt.get("artifacts"), f"{label}.artifacts")
    _require_owned_artifacts(
        [*artifacts.values(), checkpoint],
        output_directory=config_entry["output_directory"],
        label=f"{label} training artifacts",
    )
    _validate_training_artifacts(
        artifacts,
        loader=loader,
        registry=registry,
        stage_id=stage_id,
        arm=arm,
        config_entry=config_entry,
        checkpoint=checkpoint,
        gradient_audit=gradient_audit,
        initialization=initialization,
        source_revision=source_revision,
        scheduler_dependency=scheduler_dependency,
    )
    denoising_ref = _json_ref(
        attempt.get("denoising_report"),
        f"{label}.denoising_report",
        required_root="repository",
    )
    if denoising_ref["schema_version"] != DENOISING_REPORT_SCHEMA_VERSION:
        raise ScreenValidationError("denoising wrapper schema is unsupported")
    wrapper = _load_json_ref(
        denoising_ref, loader=loader, label=f"{label}.denoising_report"
    )
    evaluator_ref, bins = _validate_denoising_wrapper(
        wrapper,
        loader=loader,
        registry=registry,
        stage_id=stage_id,
        arm=arm,
        config_entry=config_entry,
        checkpoint=checkpoint,
        source_revision=source_revision,
    )
    _require_owned_artifacts(
        [denoising_ref, evaluator_ref],
        output_directory=config_entry["output_directory"],
        label=f"{label} evaluation artifacts",
    )
    return {
        "arm_id": arm["arm_id"],
        "source_revision": source_revision,
        "config_entry": config_entry,
        "checkpoint": checkpoint,
        "initialization": initialization,
        "gradient_gate": gradient_gate,
        "denoising_ref": denoising_ref,
        "evaluator_ref": evaluator_ref,
        "bins": bins,
    }


def _require_matched_corruptions(attempts: Sequence[Mapping[str, Any]]) -> None:
    reference = attempts[0]["bins"]
    for candidate in attempts[1:]:
        for control_bin, candidate_bin in zip(
            reference, candidate["bins"], strict=True
        ):
            for field in (
                "time",
                "denominator",
                "row_corruption_seeds_sha256",
                "corrupted_token_ids_sha256",
                "corruption_grid_sha256",
            ):
                if candidate_bin[field] != control_bin[field]:
                    raise ScreenValidationError(
                        "screen arms were not evaluated on identical corruptions"
                    )


def _json_dependency_ref(value: object, label: str) -> dict[str, Any]:
    return _json_ref(value, label, required_root="repository")


def _verify_revision_contract(registry: ValidatedRegistry, revision: str) -> None:
    refs: list[Mapping[str, Any]] = list(registry.data["source"]["blobs"])
    refs.extend(
        [
            registry.data["panel"]["artifact"],
            registry.data["panel"]["frequency_artifact"],
        ]
    )
    conditioning_fixture = registry.data["stages"][1]["initialization_fixture"]
    if conditioning_fixture is not None:
        refs.append(conditioning_fixture)
    gradient_contract = registry.data["stages"][1]["gradient_contract_artifact"]
    if gradient_contract is not None:
        refs.append(gradient_contract)
    refs.extend(
        config["config"]
        for stage in registry.data["stages"]
        for arm in stage["arms"]
        for config in arm["resolved_configs"]
    )
    refs.append(
        {
            "root": "repository",
            "relative_path": registry.relative_path.as_posix(),
            "sha256": registry.raw_sha256,
            "size_bytes": registry.raw_size_bytes,
        }
    )
    for index, ref in enumerate(refs):
        _verify_git_blob(
            ref,
            revision=revision,
            git_blob_loader=registry.git_blob_loader,
            label=f"authorization revision blob {index}",
        )


def _load_declared_scheduler_dependency(
    value: object,
    *,
    loader: BlobLoader,
    registry: ValidatedRegistry,
) -> tuple[str, str, dict[str, Any]]:
    dependency = _mapping(value, "conditioning scheduler dependency")
    _exact_keys(
        dependency,
        {"authorization_revision", "scheduler_evidence", "scheduler_selection"},
        "conditioning scheduler dependency",
    )
    evidence_ref = _json_dependency_ref(
        dependency.get("scheduler_evidence"), "scheduler evidence dependency"
    )
    selection_ref = _json_dependency_ref(
        dependency.get("scheduler_selection"), "scheduler selection dependency"
    )
    authorization_revision = _git_revision(
        dependency.get("authorization_revision"),
        "conditioning authorization revision",
    )
    registry_revision = registry.data["source"]["revision"]
    if authorization_revision == registry_revision or not registry.git_ancestor_checker(
        registry_revision, authorization_revision
    ):
        raise ScreenValidationError(
            "registry revision is not an ancestor of conditioning authorization"
        )
    if (
        evidence_ref["schema_version"] != EVIDENCE_SCHEMA_VERSION
        or selection_ref["schema_version"] != SELECTION_SCHEMA_VERSION
    ):
        raise ScreenValidationError("scheduler dependency schemas are unsupported")
    for ref, label in (
        (evidence_ref, "scheduler evidence"),
        (selection_ref, "scheduler selection"),
    ):
        try:
            registry.git_blob_loader(
                registry_revision, PurePosixPath(ref["relative_path"])
            )
        except Exception:
            pass
        else:
            raise ScreenValidationError(
                f"{label} was already present at the pre-screen source revision"
            )
    _verify_git_blob(
        evidence_ref,
        revision=authorization_revision,
        git_blob_loader=registry.git_blob_loader,
        label="committed scheduler evidence dependency",
    )
    _verify_git_blob(
        selection_ref,
        revision=authorization_revision,
        git_blob_loader=registry.git_blob_loader,
        label="committed scheduler selection dependency",
    )
    scheduler_payload = _load_bound_blob(
        evidence_ref, loader=loader, label="scheduler evidence dependency"
    )
    if scheduler_payload is None:  # pragma: no cover - JSON refs retain bytes
        raise ScreenValidationError("scheduler evidence bytes are unavailable")
    recomputed = evaluate_evidence_bytes(
        scheduler_payload,
        evidence_relative_path=evidence_ref["relative_path"],
        stage_id="scheduler",
        registry=registry,
        loader=loader,
    )
    if recomputed["status"] != "completed" or recomputed["selected_arm_id"] not in {
        "E-L0",
        "E-L1",
    }:
        raise EvidenceIncomplete("scheduler selection is not complete")
    declared_selection = _load_json_ref(
        selection_ref, loader=loader, label="scheduler selection dependency"
    )
    if dict(declared_selection) != recomputed:
        raise ScreenValidationError("declared scheduler selection is not reproducible")
    _verify_revision_contract(registry, authorization_revision)
    return (
        recomputed["selected_arm_id"],
        authorization_revision,
        {
            "authorization_revision": authorization_revision,
            "scheduler_evidence": evidence_ref,
            "scheduler_selection": selection_ref,
            "selected_scheduler_arm_id": recomputed["selected_arm_id"],
        },
    )


def _validate_evidence_document(
    evidence: Mapping[str, Any],
    *,
    loader: BlobLoader,
    registry: ValidatedRegistry,
    stage_id: str,
) -> tuple[
    list[dict[str, Any]],
    Mapping[str, Any] | None,
    str | None,
    Mapping[str, Any] | None,
]:
    _exact_keys(
        evidence,
        {
            "schema_version",
            "registry",
            "stage_id",
            "run_source_revision",
            "producer_source",
            "status",
            "generation_metrics_included",
            "final_generation_seeds_included",
            "scheduler_dependency",
            "initialization_audit",
            "attempts",
        },
        "optimization-screen evidence",
    )
    if (
        _integer(evidence.get("schema_version"), "screen evidence schema")
        != EVIDENCE_SCHEMA_VERSION
        or evidence.get("stage_id") != stage_id
        or evidence.get("status") != "closed_after_registered_attempts"
        or evidence.get("generation_metrics_included") is not False
        or evidence.get("final_generation_seeds_included") != []
    ):
        raise ScreenValidationError("optimization-screen evidence header is invalid")
    _registry_reference(evidence.get("registry"), registry)
    run_source_revision = _git_revision(
        evidence.get("run_source_revision"), "screen run source revision"
    )
    registry_source_revision = registry.data["source"]["revision"]
    if (
        run_source_revision == registry_source_revision
        or not registry.git_ancestor_checker(
            registry_source_revision, run_source_revision
        )
        or not registry.git_pushed_checker(run_source_revision)
    ):
        raise ScreenValidationError("screen run revision is not a pushed descendant")
    collector_ref = next(
        ref
        for ref in registry.data["source"]["blobs"]
        if ref["relative_path"]
        == "scripts/udlm/collect_optimization_screen_evidence.py"
    )
    if (
        _blob_ref(
            evidence.get("producer_source"),
            "screen evidence producer",
            required_root="repository",
            suffix=".py",
        )
        != collector_ref
    ):
        raise ScreenValidationError("screen evidence producer is unregistered")
    _verify_git_blob(
        collector_ref,
        revision=run_source_revision,
        git_blob_loader=registry.git_blob_loader,
        label="screen evidence producer",
    )
    selected_scheduler: str | None
    dependency_record: Mapping[str, Any] | None
    initialization_audit: Mapping[str, Any] | None
    run_source_revision: str
    if stage_id == "scheduler":
        if evidence.get("scheduler_dependency") is not None:
            raise ScreenValidationError("scheduler evidence cannot have a dependency")
        if evidence.get("initialization_audit") is not None:
            raise ScreenValidationError("scheduler evidence cannot have an init audit")
        selected_scheduler = None
        dependency_record = None
        initialization_audit = None
        try:
            registry.git_blob_loader(registry_source_revision, registry.relative_path)
        except Exception:
            pass
        else:
            raise ScreenValidationError(
                "registry path already existed at the pre-registry source revision"
            )
        allowed_changes = frozenset({registry.relative_path.as_posix()})
        if not registry.git_diff_checker(
            registry_source_revision, run_source_revision, allowed_changes
        ):
            raise ScreenValidationError(
                "scheduler run revision changed unregistered source bytes"
            )
        _verify_revision_contract(registry, run_source_revision)
    else:
        (
            selected_scheduler,
            run_source_revision,
            dependency_record,
        ) = _load_declared_scheduler_dependency(
            evidence.get("scheduler_dependency"), loader=loader, registry=registry
        )
        if run_source_revision != evidence.get("run_source_revision"):
            raise ScreenValidationError(
                "conditioning run does not use its authorization revision"
            )
        allowed_changes = frozenset(
            {
                registry.relative_path.as_posix(),
                dependency_record["scheduler_evidence"]["relative_path"],
                dependency_record["scheduler_selection"]["relative_path"],
            }
        )
        if not registry.git_diff_checker(
            registry_source_revision, run_source_revision, allowed_changes
        ):
            raise ScreenValidationError(
                "conditioning authorization changed unregistered source bytes"
            )
        _verify_revision_contract(registry, run_source_revision)
        initialization_audit = _mapping(
            evidence.get("initialization_audit"), "conditioning initialization audit"
        )
    raw_attempts = evidence.get("attempts")
    stage = _stage(registry, stage_id)
    if not isinstance(raw_attempts, list) or len(raw_attempts) != 2:
        raise EvidenceIncomplete("both registered attempt records are required")
    attempts = [
        _validate_attempt(
            raw_attempt,
            loader=loader,
            registry=registry,
            stage_id=stage_id,
            arm=arm,
            scheduler_arm_id=selected_scheduler,
            source_revision=run_source_revision,
            scheduler_dependency=dependency_record,
        )
        for raw_attempt, arm in zip(raw_attempts, stage["arms"], strict=True)
    ]
    _require_matched_corruptions(attempts)
    return attempts, dependency_record, selected_scheduler, initialization_audit


def _loss_ratio_le(
    candidate_sum: Fraction,
    candidate_denominator: int,
    control_sum: Fraction,
    control_denominator: int,
    percent: int,
) -> bool:
    return (
        candidate_sum * control_denominator * 100
        <= control_sum * candidate_denominator * percent
    )


def _loss_strictly_better(
    candidate_sum: Fraction,
    candidate_denominator: int,
    control_sum: Fraction,
    control_denominator: int,
) -> bool:
    return candidate_sum * control_denominator < control_sum * candidate_denominator


def _metric_summary(attempts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for attempt in attempts:
        pooled_sum = sum(
            (Fraction(item["loss_sum"]) for item in attempt["bins"]), Fraction(0)
        )
        pooled_denominator = sum(item["denominator"] for item in attempt["bins"])
        pooled_correct = sum(item["correct"] for item in attempt["bins"])
        result[attempt["arm_id"]] = {
            "production_loss_sum_fraction": (
                f"{pooled_sum.numerator}/{pooled_sum.denominator}"
            ),
            "denominator_tokens": pooled_denominator,
            "clean_token_top1_correct": pooled_correct,
            "by_time": [
                {
                    "time": _decimal_text(item["time"]),
                    "production_loss_sum": _decimal_text(item["loss_sum"]),
                    "denominator_tokens": item["denominator"],
                    "clean_token_top1_correct": item["correct"],
                }
                for item in attempt["bins"]
            ],
        }
    return result


def _scheduler_gates(attempts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    control, candidate = attempts
    control_sum = sum(
        (Fraction(item["loss_sum"]) for item in control["bins"]), Fraction(0)
    )
    candidate_sum = sum(
        (Fraction(item["loss_sum"]) for item in candidate["bins"]), Fraction(0)
    )
    control_denominator = sum(item["denominator"] for item in control["bins"])
    candidate_denominator = sum(item["denominator"] for item in candidate["bins"])
    if control_sum <= 0:
        raise ScreenValidationError("scheduler control loss must be positive")
    pooled = _loss_ratio_le(
        candidate_sum,
        candidate_denominator,
        control_sum,
        control_denominator,
        98,
    )
    strict_bins = sum(
        _loss_strictly_better(
            Fraction(candidate_bin["loss_sum"]),
            candidate_bin["denominator"],
            Fraction(control_bin["loss_sum"]),
            control_bin["denominator"],
        )
        for control_bin, candidate_bin in zip(
            control["bins"], candidate["bins"], strict=True
        )
    )
    every_bin = all(
        _loss_ratio_le(
            Fraction(candidate_bin["loss_sum"]),
            candidate_bin["denominator"],
            Fraction(control_bin["loss_sum"]),
            control_bin["denominator"],
            102,
        )
        for control_bin, candidate_bin in zip(
            control["bins"], candidate["bins"], strict=True
        )
    )
    return {
        "pooled_candidate_le_98_percent_control": pooled,
        "strictly_better_bin_count": strict_bins,
        "at_least_two_strictly_better_bins": strict_bins >= 2,
        "every_bin_candidate_le_102_percent_control": every_bin,
        "all_required_gates_pass": pooled and strict_bins >= 2 and every_bin,
    }


def _conditioning_gates(
    attempts: Sequence[Mapping[str, Any]], *, initialization_equal: bool
) -> dict[str, Any]:
    control, candidate = attempts
    control_sum = sum(
        (Fraction(item["loss_sum"]) for item in control["bins"]), Fraction(0)
    )
    candidate_sum = sum(
        (Fraction(item["loss_sum"]) for item in candidate["bins"]), Fraction(0)
    )
    control_denominator = sum(item["denominator"] for item in control["bins"])
    candidate_denominator = sum(item["denominator"] for item in candidate["bins"])
    if control_sum <= 0:
        raise ScreenValidationError("conditioning control loss must be positive")
    pooled = _loss_ratio_le(
        candidate_sum,
        candidate_denominator,
        control_sum,
        control_denominator,
        98,
    )
    every_bin = all(
        _loss_ratio_le(
            Fraction(candidate_bin["loss_sum"]),
            candidate_bin["denominator"],
            Fraction(control_bin["loss_sum"]),
            control_bin["denominator"],
            102,
        )
        for control_bin, candidate_bin in zip(
            control["bins"], candidate["bins"], strict=True
        )
    )
    control_correct = sum(item["correct"] for item in control["bins"])
    candidate_correct = sum(item["correct"] for item in candidate["bins"])
    accuracy = (
        candidate_correct * control_denominator
        >= control_correct * candidate_denominator
    )
    gradients = candidate["gradient_gate"] is True
    return {
        "exact_initialization_equality": initialization_equal,
        "pooled_candidate_le_98_percent_control": pooled,
        "every_bin_candidate_le_102_percent_control": every_bin,
        "pooled_clean_token_accuracy_nondecreasing": accuracy,
        "registered_gradient_contract_pass": gradients,
        "all_required_gates_pass": (
            initialization_equal and pooled and every_bin and accuracy and gradients
        ),
    }


def _incomplete_decision(
    *,
    registry: ValidatedRegistry,
    evidence_reference: Mapping[str, Any],
    stage_id: str,
    reason: str,
    run_source_revision: object = None,
) -> dict[str, Any]:
    return {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "registry": registry.reference,
        "evidence": dict(evidence_reference),
        "stage_id": stage_id,
        "run_source_revision": run_source_revision,
        "status": "incomplete",
        "selected_arm_id": None,
        "complete_threshold_fallback_used": False,
        "selection_rule_id": _stage(registry, stage_id)["selection_rule"]["rule_id"],
        "dependency": None,
        "metrics": None,
        "gates": None,
        "reason_codes": [reason],
        "final_generation_seeds_used": [],
        "generation_metrics_used": False,
    }


def evaluate_evidence_bytes(
    payload: bytes,
    *,
    evidence_relative_path: str,
    stage_id: str,
    registry: ValidatedRegistry,
    loader: BlobLoader,
) -> dict[str, Any]:
    """Recompute a deterministic selection, or return incomplete with no winner."""

    if stage_id not in EXPECTED_STAGE_ORDER:
        raise ScreenValidationError("stage ID is unsupported")
    parsed: Mapping[str, Any] | None = None
    evidence_reference = _evidence_reference(
        payload=payload, relative_path=evidence_relative_path, parsed=None
    )
    try:
        parsed = _mapping(
            strict_json_loads(payload, label=f"{stage_id} screen evidence"),
            f"{stage_id} screen evidence",
        )
        evidence_reference = _evidence_reference(
            payload=payload,
            relative_path=evidence_relative_path,
            parsed=parsed,
        )
        attempts, dependency, selected_scheduler, initialization_audit = (
            _validate_evidence_document(
                parsed,
                loader=loader,
                registry=registry,
                stage_id=stage_id,
            )
        )
        full_states = [
            attempt["initialization"]["state_audit"]["full_initial_state_sha256"]
            for attempt in attempts
        ]
        backbone_states = [
            attempt["initialization"]["state_audit"]["common_backbone_state_sha256"]
            for attempt in attempts
        ]
        if stage_id == "scheduler":
            if len(set(full_states)) != 1 or len(set(backbone_states)) != 1:
                raise ScreenValidationError(
                    "scheduler arms do not share the exact common initialization"
                )
            gates = _scheduler_gates(attempts)
            candidate_arm = "E-L1"
            fallback_arm = "E-L0"
        else:
            if len(set(backbone_states)) != 1:
                raise ScreenValidationError(
                    "conditioning arms do not share the warm-started backbone"
                )
            reference_entry = attempts[0]["config_entry"]
            candidate_entry = attempts[1]["config_entry"]
            initialization_equal = _validate_initialization_audit(
                initialization_audit,
                loader=loader,
                registry=registry,
                reference_config_entry=reference_entry,
                candidate_config_entry=candidate_entry,
                source_revision=attempts[0]["source_revision"],
            )
            del selected_scheduler
            gates = _conditioning_gates(
                attempts, initialization_equal=initialization_equal
            )
            candidate_arm = "E-A1"
            fallback_arm = "E-A0"
        selected = candidate_arm if gates["all_required_gates_pass"] else fallback_arm
        return {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "registry": registry.reference,
            "evidence": evidence_reference,
            "stage_id": stage_id,
            "run_source_revision": parsed["run_source_revision"],
            "status": "completed",
            "selected_arm_id": selected,
            "complete_threshold_fallback_used": selected == fallback_arm,
            "selection_rule_id": _stage(registry, stage_id)["selection_rule"][
                "rule_id"
            ],
            "dependency": dependency,
            "metrics": _metric_summary(attempts),
            "gates": gates,
            "reason_codes": (
                ["candidate_satisfied_all_registered_thresholds"]
                if selected == candidate_arm
                else ["complete_evidence_retained_registered_control"]
            ),
            "final_generation_seeds_used": [],
            "generation_metrics_used": False,
        }
    except (
        EvidenceIncomplete,
        ScreenValidationError,
        OSError,
        KeyError,
        TypeError,
        AttributeError,
        ArithmeticError,
    ) as error:
        return _incomplete_decision(
            registry=registry,
            evidence_reference=evidence_reference,
            stage_id=stage_id,
            reason=(
                "registered_evidence_incomplete"
                if isinstance(error, EvidenceIncomplete)
                else "evidence_invalid_or_unmatched"
            ),
            run_source_revision=(
                None if parsed is None else parsed.get("run_source_revision")
            ),
        )


def _root_path(root: str, relative_path: PurePosixPath) -> Path:
    if root == "repository":
        base = REPOSITORY_ROOT
    elif root == "project":
        base = PROJECT_ROOT
    else:  # pragma: no cover - references validate this before loading
        raise ScreenValidationError("unknown artifact root")
    candidate = base.joinpath(*relative_path.parts)
    current = base
    for part in relative_path.parts[:-1]:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            break
        if stat.S_ISLNK(mode):
            raise ScreenValidationError(
                f"artifact path has a symlink ancestor: {candidate}"
            )
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError:
        return candidate
    if not resolved.is_relative_to(base.resolve()):
        raise ScreenValidationError(
            f"artifact path escapes its declared root: {candidate}"
        )
    return candidate


def _read_stable_file(path: Path, *, retain: bool) -> bytes | BlobSnapshot:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ScreenValidationError(
                f"artifact is not an exclusive regular file: {path}"
            )
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        size_bytes = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size_bytes += len(chunk)
            if retain:
                chunks.append(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after or size_bytes != before.st_size:
            raise ScreenValidationError(f"artifact changed while being hashed: {path}")
        observed_digest = digest.hexdigest()
        if retain:
            return b"".join(chunks)
        return BlobSnapshot(size_bytes=size_bytes, sha256=observed_digest)
    finally:
        os.close(descriptor)


def local_blob_loader(root: str, relative_path: PurePosixPath) -> bytes | BlobSnapshot:
    """Load small evidence bytes and stream large checkpoints without deserializing."""

    path = _root_path(root, relative_path)
    return _read_stable_file(path, retain=path.suffix != ".ckpt")


def git_blob_loader(revision: str, relative_path: PurePosixPath) -> bytes:
    object_name = f"{revision}:{relative_path.as_posix()}"
    type_result = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "cat-file", "-t", object_name],
        check=True,
        capture_output=True,
    )
    if type_result.stdout.strip() != b"blob":
        raise ScreenValidationError(f"Git object is not a blob: {object_name}")
    return subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "cat-file", "blob", object_name],
        check=True,
        capture_output=True,
    ).stdout


def git_ancestor_checker(ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(REPOSITORY_ROOT),
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode not in {0, 1}:
        raise ScreenValidationError("Git ancestry check failed")
    return result.returncode == 0


def git_sole_parent_checker(revision: str, expected_parent: str) -> bool:
    """Return whether ``revision`` has exactly ``expected_parent`` as its parent."""

    fields = (
        subprocess.run(
            [
                "git",
                "-C",
                str(REPOSITORY_ROOT),
                "rev-list",
                "--parents",
                "-n",
                "1",
                revision,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .split()
    )
    return len(fields) == 2 and fields[0] == revision and fields[1] == expected_parent


def git_tree_paths_loader(revision: str, directory: PurePosixPath) -> frozenset[str]:
    """Return every committed blob path recursively below ``directory``."""

    payload = subprocess.run(
        [
            "git",
            "-C",
            str(REPOSITORY_ROOT),
            "ls-tree",
            "-r",
            "--name-only",
            "-z",
            revision,
            "--",
            directory.as_posix(),
        ],
        check=True,
        capture_output=True,
    ).stdout
    if not payload:
        return frozenset()
    if not payload.endswith(b"\0"):
        raise ScreenValidationError("Git tree path listing is truncated")
    raw_paths = payload[:-1].split(b"\0")
    paths = tuple(os.fsdecode(path) for path in raw_paths)
    if any(not path for path in paths) or len(paths) != len(set(paths)):
        raise ScreenValidationError("Git tree path listing is malformed")
    return frozenset(paths)


def git_pushed_checker(revision: str) -> bool:
    upstream = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "@{upstream}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return git_ancestor_checker(revision, upstream)


def git_diff_checker(
    ancestor: str, descendant: str, allowed_paths: frozenset[str]
) -> bool:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(REPOSITORY_ROOT),
            "diff",
            "--name-only",
            "--no-renames",
            ancestor,
            descendant,
            "--",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    changed_paths = {line for line in result.stdout.splitlines() if line}
    return changed_paths <= set(allowed_paths)


def _atomic_write_json_exclusive(path: Path, value: object) -> None:
    if path.suffix != ".json" or not path.is_absolute():
        raise ScreenValidationError("selection output must be an absolute JSON path")
    normalized = Path(os.path.abspath(os.fspath(path)))
    if path != normalized or not path.is_relative_to(REPOSITORY_ROOT):
        raise ScreenValidationError("selection output must be inside the repository")
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to replace selection output: {path}")
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to replace selection output: {path}"
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


def _stable_input_bytes(path: Path) -> bytes:
    result = _read_stable_file(path, retain=True)
    if not isinstance(result, bytes):  # pragma: no cover - retain=True contract
        raise ScreenValidationError(f"input bytes were not retained: {path}")
    return result


def _repository_relative(path: Path, label: str) -> str:
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(REPOSITORY_ROOT):
        raise ScreenValidationError(f"{label} must be inside the repository")
    return resolved.relative_to(REPOSITORY_ROOT).as_posix()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--expected-registry-sha256", required=True)
    parser.add_argument("--expected-registry-canonical-sha256", required=True)
    parser.add_argument("--stage", choices=EXPECTED_STAGE_ORDER, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    registry_path = args.registry.resolve(strict=True)
    registry_relative = _repository_relative(registry_path, "registry")
    registry = load_validated_registry(
        _stable_input_bytes(registry_path),
        relative_path=registry_relative,
        expected_raw_sha256=args.expected_registry_sha256,
        expected_canonical_sha256=args.expected_registry_canonical_sha256,
        loader=local_blob_loader,
        git_blob_loader=git_blob_loader,
        git_ancestor_checker=git_ancestor_checker,
        git_sole_parent_checker=git_sole_parent_checker,
        git_tree_paths_loader=git_tree_paths_loader,
        git_pushed_checker=git_pushed_checker,
        git_diff_checker=git_diff_checker,
    )
    evidence_path = args.evidence.resolve(strict=False)
    evidence_relative = _repository_relative(evidence_path, "evidence")
    try:
        evidence_payload = _stable_input_bytes(evidence_path)
    except FileNotFoundError:
        evidence_payload = b""
    decision = evaluate_evidence_bytes(
        evidence_payload,
        evidence_relative_path=evidence_relative,
        stage_id=args.stage,
        registry=registry,
        loader=local_blob_loader,
    )
    _atomic_write_json_exclusive(args.output.resolve(strict=False), decision)
    print(json.dumps(decision, indent=2, sort_keys=True, allow_nan=False))
    return 0 if decision["status"] == "completed" else INCOMPLETE_EXIT_STATUS


if __name__ == "__main__":
    raise SystemExit(main())

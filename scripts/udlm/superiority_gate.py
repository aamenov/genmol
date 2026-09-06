"""Evaluate the frozen UDLM-over-local-GenMol de-novo superiority gate.

The production CLI starts from the three raw benchmark run directories, lets
the existing de-novo reporter revalidate every row, verifies that the exact
candidate lock was already committed at the benchmark revision, and writes one
no-clobber decision record.  It is CPU-only and never loads a model checkpoint.
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
PROTOCOL_RELATIVE_PATH = Path(
    "experiments/udlm/protocols/de_novo_superiority_v1.json"
)
PROTOCOL_SHA256 = (
    "91d3e8964b04850e4b3fdf563703d490f34b9587b2bfe9e872fd7374fed33ff0"
)
PROTOCOL_CANONICAL_SHA256 = (
    "d18f5c40125caef426d107823d97322953058969214adb62f6dfe9b09cdd588f"
)
BASELINE_RELATIVE_PATH = Path("experiments/udlm/baselines/mdlm_50000.json")
BASELINE_SHA256 = (
    "6da46fc615dedbcca436da087a2c1e9145f5d110036e0c15bb431ded3c2e5539"
)
EXPECTED_PROTOCOL_ID = "genmol_udlm_de_novo_superiority_v1"
EXPECTED_SEEDS = (0, 1, 2)
EXPECTED_SAMPLES_PER_SEED = 1_000
EXPECTED_NFE = 128
EXPECTED_BASELINE_CHECKPOINT_SHA256 = (
    "8d00aa47b02f64bf39ff6b0b2e786f213587366fc2c3d29712a00f3f84108dd6"
)
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
HEX_GIT_REVISION = re.compile(r"[0-9a-f]{40}\Z")
METRICS = ("validity", "uniqueness", "quality", "diversity")
CANDIDATE_LEDGER_SCHEMA_VERSION = 1
PILOT_EVIDENCE_SCHEMA_VERSION = 1
TRAINING_SUMMARY_SCHEMA_VERSION = 2
PILOT_EXIT_STATUS_SCHEMA_VERSION = 2
REGISTERED_SELECTION_PILOT_SEEDS = (1000, 1001)
REGISTERED_SELECTION_SAMPLES_PER_SEED = 256
REGISTERED_SELECTION_NFE = 128
REGISTERED_SELECTION_METRIC_BRANCH = "released_comparable"
NONREGISTERED_OPERATING_POINT_REASON = (
    "engineering_or_nonregistered_operating_point"
)
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
        raise GateValidationError(
            f"{label} must be 64 lowercase hexadecimal digits"
        )
    return value


def _git_revision(value: object, label: str) -> str:
    if not isinstance(value, str) or HEX_GIT_REVISION.fullmatch(value) is None:
        raise GateValidationError(
            f"{label} must be 40 lowercase hexadecimal digits"
        )
    return value


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
        if not stat.S_ISREG(before_fd.st_mode) or (
            before_path.st_dev,
            before_path.st_ino,
            before_path.st_mode,
            before_path.st_size,
            before_path.st_mtime_ns,
            before_path.st_ctime_ns,
        ) != identity:
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


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if canonical_json_sha256(protocol) != PROTOCOL_CANONICAL_SHA256:
        raise GateValidationError("superiority protocol content is not the frozen value")
    if protocol.get("schema_version") != 1:
        raise GateValidationError("protocol schema_version must equal 1")
    if protocol.get("protocol_id") != EXPECTED_PROTOCOL_ID:
        raise GateValidationError("unexpected superiority protocol ID")
    if protocol.get("status") != "frozen_before_gpu_pilots":
        raise GateValidationError("superiority protocol is not frozen")
    baseline = _mapping(protocol.get("baseline"), "protocol.baseline")
    if baseline.get("manifest_relative_path") != BASELINE_RELATIVE_PATH.as_posix():
        raise GateValidationError("protocol baseline path is unexpected")
    if baseline.get("manifest_sha256") != BASELINE_SHA256:
        raise GateValidationError("protocol baseline digest is unexpected")
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
        "total_requested_samples": len(EXPECTED_SEEDS)
        * EXPECTED_SAMPLES_PER_SEED,
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
        raise GateValidationError(
            "protocol checkpoint-selection rule is unexpected"
        )
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
    if (
        firewall.get("eligible_metric_branch")
        != REGISTERED_SELECTION_METRIC_BRANCH
    ):
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
        raise GateValidationError(f"{label} seeds must be exactly {list(EXPECTED_SEEDS)}")
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
    ordered = _ordered_seed_rows(
        branch.get("per_seed"), label="baseline per-seed rows"
    )
    values: dict[str, list[float]] = {metric: [] for metric in METRICS}
    pooled_valid = 0
    pooled_requested = 0
    for expected_seed, row in zip(EXPECTED_SEEDS, ordered, strict=True):
        if row.get("seed") != expected_seed:
            raise GateValidationError("baseline seed rows are incomplete")
        requested = _integer(row.get("requested"), "baseline requested", minimum=1)
        if requested != EXPECTED_SAMPLES_PER_SEED:
            raise GateValidationError("baseline per-seed request count is unexpected")
        valid_count = _integer(row.get("valid_count"), "baseline valid count", minimum=0)
        unique_count = _integer(row.get("unique_count"), "baseline unique count", minimum=0)
        quality_count = _integer(row.get("quality_count"), "baseline quality count", minimum=0)
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
        "means": {metric: statistics.fmean(series) for metric, series in values.items()},
        "sample_sds": {metric: _sample_sd(series) for metric, series in values.items()},
        "pooled_valid": pooled_valid,
        "pooled_requested": pooled_requested,
        "checkpoint": dict(checkpoint),
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
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
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
        raise GateValidationError("Welch intervals require at least two values per method")
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
    denominator = (
        candidate_component * candidate_component / (len(candidate) - 1)
        + baseline_component * baseline_component / (len(baseline) - 1)
    )
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
    if candidate_lock.get("schema_version") != 1:
        raise GateValidationError("candidate lock schema_version must equal 1")
    candidate_id = candidate_lock.get("candidate_id")
    if not isinstance(candidate_id, str) or re.fullmatch(
        r"[a-z0-9][a-z0-9._-]{2,95}", candidate_id
    ) is None:
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

    training = _mapping(candidate_lock.get("training"), "candidate training")
    _exact_keys(
        training,
        {
            "source_revision",
            "training_summary",
            "exit_receipt",
            "runtime_config",
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
    _exact_keys(startup, {"mode", "initialization_checkpoint_sha256"}, "candidate startup")
    startup_mode = startup.get("mode")
    if startup_mode not in CLAIM_SCOPE_BY_STARTUP:
        raise GateValidationError("candidate startup mode must be warm_start or scratch")
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
        {"gate_source_sha256", "report_source_sha256", "scipy_version"},
        "candidate analysis",
    )
    gate_source_sha = _sha256(
        analysis.get("gate_source_sha256"), "superiority gate source digest"
    )
    report_source_sha = _sha256(
        analysis.get("report_source_sha256"), "de-novo report source digest"
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
    _exact_keys(
        parameters,
        {"base_model_trainable", "time_conditioner_trainable", "total_trainable"},
        "candidate parameters",
    )
    base_count = _integer(
        parameters.get("base_model_trainable"), "base trainable parameters", minimum=1
    )
    adapter_count = _integer(
        parameters.get("time_conditioner_trainable"),
        "time-conditioner trainable parameters",
        minimum=1,
    )
    total_trainable = _integer(
        parameters.get("total_trainable"), "total trainable parameters", minimum=1
    )
    if total_trainable != base_count + adapter_count:
        raise GateValidationError("candidate trainable parameter counts do not add up")

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
            raise GateValidationError("final run directory seeds must be unique integers")
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
        "selection_rule": selection["selection_rule"],
        "checkpoint_selection_rule": selection["checkpoint_selection_rule"],
        "summary": summary_ref,
        "receipt": receipt_ref,
        "runtime": runtime_ref,
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
        "parameter_counts": {
            "base_model_trainable": base_count,
            "time_conditioner_trainable": adapter_count,
            "total_trainable": total_trainable,
        },
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


def _load_referenced_json(reference: Mapping[str, Any], *, label: str) -> Mapping[str, Any]:
    payload = _repository_artifact_bytes(reference["relative_path"], label=label)
    observed = _sha256_bytes(payload)
    if observed != reference["sha256"]:
        raise GateValidationError(f"{label} digest disagrees with candidate lock")
    parsed = _mapping(strict_json_loads(payload, label=label), label)
    if parsed.get("schema_version") != reference["schema_version"]:
        raise GateValidationError(f"{label} schema version disagrees with candidate lock")
    return parsed


def validate_training_evidence(lock: Mapping[str, Any]) -> dict[str, Any]:
    """Join the lock to the launch-bound training summary and exit receipt."""

    summary = _load_referenced_json(lock["summary"], label="training summary")
    receipt = _load_referenced_json(lock["receipt"], label="training exit receipt")
    runtime = _load_referenced_json(lock["runtime"], label="training runtime config")
    if summary.get("schema_version") != TRAINING_SUMMARY_SCHEMA_VERSION:
        raise GateValidationError("training summary schema version is unsupported")
    if receipt.get("schema_version") != PILOT_EXIT_STATUS_SCHEMA_VERSION:
        raise GateValidationError("training exit receipt schema version is unsupported")
    if summary.get("status") != "completed":
        raise GateValidationError("training summary is not completed")
    if summary.get("source_revision") != lock["source_revision"]:
        raise GateValidationError("training summary source revision disagrees with lock")
    if (
        summary.get("resolved_training_config_sha256")
        != lock["resolved_training_config_sha256"]
    ):
        raise GateValidationError("training summary config digest disagrees with lock")
    if summary.get("training_argv_sha256") != lock["training_argv_sha256"]:
        raise GateValidationError("training summary argv digest disagrees with lock")
    runtime_claim = _mapping(summary.get("runtime_config"), "summary runtime config")
    if runtime_claim.get("sha256") != lock["runtime"]["sha256"]:
        raise GateValidationError("summary runtime-config digest disagrees with lock")
    observed = _mapping(summary.get("observed_training_state"), "observed training state")
    if observed.get("global_step") != lock["optimizer_updates"]:
        raise GateValidationError("training summary step count disagrees with lock")
    if observed.get("world_size") != lock["world_size"]:
        raise GateValidationError("training summary world size disagrees with lock")
    final_checkpoint = _mapping(summary.get("final_checkpoint"), "summary checkpoint")
    for key in ("sha256", "size_bytes"):
        if final_checkpoint.get(key) != lock["checkpoint"][key]:
            raise GateValidationError(f"summary checkpoint {key} disagrees with lock")
    semantic = _mapping(final_checkpoint.get("semantic_audit"), "checkpoint semantic audit")
    if semantic.get("global_step") != lock["checkpoint"]["global_step"]:
        raise GateValidationError("semantic checkpoint step disagrees with lock")
    for path, label in (
        (("ema", "all_finite"), "serialized EMA finiteness"),
        (("live_ema_match", "exact_tensor_values"), "live EMA checkpoint match"),
        (("live_model_match", "exact_tensor_values"), "live model checkpoint match"),
    ):
        container = _mapping(semantic.get(path[0]), label)
        _required_true(container.get(path[1]), label)
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
    live_ema_match = _mapping(
        semantic.get("live_ema_match"), "live EMA checkpoint match"
    )
    serialized_ema = _mapping(semantic.get("ema"), "serialized EMA finiteness")
    shadow_count = summary_inference_weights["ema"]["shadow_parameter_count"]
    if live_ema_match.get("tensor_count") != shadow_count:
        raise GateValidationError("live EMA tensor count disagrees with EMA metadata")
    if serialized_ema.get("floating_tensor_count") != shadow_count:
        raise GateValidationError("serialized EMA tensor count disagrees with metadata")
    startup = _mapping(summary.get("startup"), "training summary startup")
    if startup.get("mode") != lock["startup_mode"]:
        raise GateValidationError("training startup mode disagrees with candidate lock")
    warm_report = startup.get("verified_mdlm_warm_start_report")
    if lock["startup_mode"] == "warm_start":
        warm_report = _mapping(warm_report, "warm-start report")
        if warm_report.get("source_sha256") != lock["initialization_checkpoint_sha256"]:
            raise GateValidationError("warm-start source digest disagrees with lock")
        if warm_report.get("weights") != "ema":
            raise GateValidationError("warm-start initialization must use MDLM EMA")
    elif warm_report is not None:
        raise GateValidationError("scratch summary unexpectedly contains warm-start evidence")

    if receipt.get("status") != "completed" or receipt.get("overall_status") != "completed":
        raise GateValidationError("training exit receipt is not completed")
    if receipt.get("process_exit_status") != 0:
        raise GateValidationError("training exit receipt records nonzero status")
    expected = _mapping(receipt.get("expected_contract"), "receipt expected contract")
    expected_values = {
        "source_revision": lock["source_revision"],
        "resolved_training_config_sha256": lock[
            "resolved_training_config_sha256"
        ],
        "training_argv_sha256": lock["training_argv_sha256"],
        "max_steps": lock["optimizer_updates"],
        "world_size": lock["world_size"],
        "initialization_checkpoint_sha256": lock[
            "initialization_checkpoint_sha256"
        ],
    }
    for key, value in expected_values.items():
        if expected.get(key) != value:
            raise GateValidationError(f"exit receipt {key} disagrees with lock")
    source = _mapping(receipt.get("source_at_receipt"), "receipt source evidence")
    _required_true(source.get("verified"), "clean pushed source at receipt")
    receipt_summary = _mapping(receipt.get("training_summary"), "receipt summary evidence")
    _required_true(
        receipt_summary.get("valid_and_launch_bound"),
        "launch-bound training summary",
    )
    summary_artifact = _mapping(
        receipt_summary.get("artifact"), "receipt training-summary artifact"
    )
    if summary_artifact.get("sha256") != lock["summary"]["sha256"]:
        raise GateValidationError("receipt training-summary digest disagrees with lock")
    receipt_checkpoint = _mapping(
        receipt.get("final_checkpoint"), "receipt checkpoint evidence"
    )
    _required_true(
        receipt_checkpoint.get("matches_training_summary_snapshot"),
        "receipt checkpoint snapshot match",
    )
    checkpoint_artifact = _mapping(
        receipt_checkpoint.get("artifact"), "receipt checkpoint artifact"
    )
    for key in ("sha256", "size_bytes"):
        if checkpoint_artifact.get(key) != lock["checkpoint"][key]:
            raise GateValidationError(f"receipt checkpoint {key} disagrees with lock")
    receipt_runtime = _mapping(receipt.get("runtime_config"), "receipt runtime evidence")
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
    if runtime_artifact.get("sha256") != lock["runtime"]["sha256"]:
        raise GateValidationError("receipt runtime-config digest disagrees with lock")
    if runtime.get("status") != "preflight_completed":
        raise GateValidationError("runtime training config is not a completed preflight")
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
    if canonical_json_sha256(resolved_config) != lock["resolved_training_config_sha256"]:
        raise GateValidationError("runtime resolved training config content is unbound")
    if runtime.get("training_argv_sha256") != lock["training_argv_sha256"]:
        raise GateValidationError("runtime training argv digest disagrees with lock")
    training_argv = runtime.get("training_argv")
    if not isinstance(training_argv, list) or not all(
        isinstance(value, str) for value in training_argv
    ):
        raise GateValidationError("runtime training argv must be a string list")
    if canonical_json_sha256(training_argv) != lock["training_argv_sha256"]:
        raise GateValidationError("runtime training argv content is unbound")

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
        raise GateValidationError("summary training-accounting batch arithmetic disagrees")
    if requested_exposures != global_examples * optimizer_updates:
        raise GateValidationError("summary training-accounting exposure arithmetic disagrees")
    if training_seed != lock["training_seed"]:
        raise GateValidationError("training-accounting seed disagrees with lock")
    if optimizer_updates != lock["optimizer_updates"]:
        raise GateValidationError("training-accounting updates disagree with lock")
    if world_size != lock["world_size"]:
        raise GateValidationError("training-accounting world size disagrees with lock")
    locked_exposure = lock["data_exposure"]
    if global_examples != locked_exposure["global_examples_per_optimizer_step"]:
        raise GateValidationError("training-accounting global examples disagree with lock")
    if requested_exposures != locked_exposure["total_requested_examples"]:
        raise GateValidationError("training-accounting requested exposure disagrees with lock")
    if (
        accounting.get("hosted_stream_rank_partition_policy")
        != locked_exposure["stream_partition_policy"]
    ):
        raise GateValidationError("training-accounting stream policy disagrees with lock")

    parameter_counts = _mapping(
        accounting.get("trainable_parameter_counts"),
        "summary trainable parameter counts",
    )
    _exact_keys(
        parameter_counts,
        {"base_backbone", "time_conditioner", "total"},
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
    total_count = _integer(
        parameter_counts.get("total"),
        "summary total trainable parameters",
        minimum=1,
    )
    if total_count != base_count + conditioner_count:
        raise GateValidationError("summary trainable parameter counts do not add up")
    locked_parameters = lock["parameter_counts"]
    if (
        base_count != locked_parameters["base_model_trainable"]
        or conditioner_count != locked_parameters["time_conditioner_trainable"]
        or total_count != locked_parameters["total_trainable"]
    ):
        raise GateValidationError("training-accounting parameter counts disagree with lock")

    validated_bindings = _mapping(
        receipt_summary.get("validated_bindings"),
        "receipt validated training-summary bindings",
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
    training_config = _mapping(
        resolved_config.get("training"), "runtime training configuration"
    )
    _close(
        training_config.get("ema"),
        summary_inference_weights["ema"]["decay"],
        "runtime EMA decay",
    )
    return {
        "training_summary_sha256": lock["summary"]["sha256"],
        "exit_receipt_sha256": lock["receipt"]["sha256"],
        "runtime_config_sha256": lock["runtime"]["sha256"],
        "resolved_training_config_sha256": lock[
            "resolved_training_config_sha256"
        ],
        "training_argv_sha256": lock["training_argv_sha256"],
        "checkpoint_sha256": lock["checkpoint"]["sha256"],
        "checkpoint_global_step": lock["checkpoint"]["global_step"],
        "ema_finite_and_checkpoint_bound": True,
        "successful_exit_receipt": True,
        "training_accounting": dict(accounting),
        "ema_metadata": dict(summary_ema_metadata),
    }


def validate_candidate_ledger(
    ledger: Mapping[str, Any],
    *,
    candidate_id: str,
    artifact_loader: Callable[[Path], bytes],
) -> dict[str, Any]:
    """Validate pilot evidence and deterministically recompute the selection.

    The ledger is intentionally a small index, not a second source of pilot
    truth.  Every score used for selection is recomputed from immutable pilot
    evidence bytes supplied by ``artifact_loader`` (Git blobs in production).
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
        raise GateValidationError("candidate ledger is not closed before final evaluation")
    if ledger.get("final_seed_results_included") is not False:
        raise GateValidationError("candidate ledger must exclude final-seed results")
    attempts = ledger.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise GateValidationError("candidate ledger must disclose at least one pilot attempt")
    attempt_ids: set[str] = set()
    artifact_paths: set[Path] = set()
    eligible_attempts: list[dict[str, Any]] = []
    artifact_count = 0
    ineligible_completed_attempt_count = 0
    failed_attempt_count = 0
    for index, raw_attempt in enumerate(attempts):
        attempt = _mapping(raw_attempt, f"candidate ledger attempt {index}")
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
            f"candidate ledger attempt {index}",
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
        if not isinstance(attempt_candidate_id, str) or re.fullmatch(
            r"[a-z0-9][a-z0-9._-]{2,95}", attempt_candidate_id
        ) is None:
            raise GateValidationError("candidate ledger candidate_id has invalid syntax")
        status = attempt.get("status")
        eligible = attempt.get("eligible_for_selection")
        if status == "completed":
            if type(eligible) is not bool:
                raise GateValidationError(
                    "completed pilot eligibility must be a boolean"
                )
            expected_artifact_kind = "pilot_evaluation"
            if eligible:
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
                declared_quality = _finite(
                    score.get("mean_released_quality"),
                    f"candidate ledger attempt {attempt_id} mean quality",
                )
                declared_diversity = _finite(
                    score.get("mean_released_diversity"),
                    f"candidate ledger attempt {attempt_id} mean diversity",
                )
                if not 0.0 <= declared_quality <= 1.0:
                    raise GateValidationError("pilot mean quality must be in [0, 1]")
                if not 0.0 <= declared_diversity <= 1.0:
                    raise GateValidationError("pilot mean diversity must be in [0, 1]")
            else:
                if attempt.get("selection_score") is not None or attempt.get(
                    "ineligibility_reason"
                ) != NONREGISTERED_OPERATING_POINT_REASON:
                    raise GateValidationError(
                        "nonregistered completed pilots must have a null score and "
                        "the fixed ineligibility reason"
                    )
                declared_quality = None
                declared_diversity = None
        elif status == "failed":
            if (
                eligible is not False
                or attempt.get("selection_score") is not None
                or attempt.get("ineligibility_reason") != FAILED_PILOT_REASON
            ):
                raise GateValidationError(
                    "failed pilot attempts must be ineligible with a null score "
                    "and the fixed failure reason"
                )
            expected_artifact_kind = "pilot_failure"
            declared_quality = None
            declared_diversity = None
            failed_attempt_count += 1
        else:
            raise GateValidationError(
                "pilot attempt status must be completed or failed"
            )
        seeds = attempt.get("pilot_seeds")
        if not isinstance(seeds, list) or not seeds:
            raise GateValidationError("each candidate attempt must disclose pilot seeds")
        if any(type(seed) is not int or seed < 1000 for seed in seeds):
            raise GateValidationError("pilot seeds must be integers greater than or equal to 1000")
        if seeds != sorted(set(seeds)):
            raise GateValidationError("pilot seeds must be unique and sorted")
        if set(seeds).intersection(EXPECTED_SEEDS):
            raise GateValidationError("candidate ledger contains a forbidden final seed")
        refs = attempt.get("artifact_refs")
        if not isinstance(refs, list) or len(refs) != len(seeds):
            raise GateValidationError(
                "each pilot seed must bind exactly one pilot evidence artifact"
            )
        evidence_seeds: set[int] = set()
        qualities: list[float] = []
        diversities: list[float] = []
        checkpoint_shas: set[str] = set()
        observed_operating_points: list[dict[str, Any]] = []
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
            if ref.get("artifact_kind") != expected_artifact_kind:
                raise GateValidationError(
                    f"{ref_label} kind disagrees with attempt status"
                )
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
            if relative_path.parts[:3] != ("experiments", "udlm", "pilots"):
                raise GateValidationError(
                    "pilot evidence must live under experiments/udlm/pilots"
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
            evidence = _mapping(
                strict_json_loads(
                    artifact_blob,
                    label=f"pilot evidence {relative_path.as_posix()}",
                ),
                f"pilot evidence {relative_path.as_posix()}",
            )
            common_evidence_fields = {
                "schema_version",
                "artifact_kind",
                "status",
                "attempt_id",
                "candidate_id",
                "pilot_seed",
                "final_seed_results_included",
            }
            if status == "completed":
                _exact_keys(
                    evidence,
                    common_evidence_fields
                    | {"checkpoint_sha256", "evaluation"},
                    f"pilot evidence {relative_path.as_posix()}",
                )
            else:
                _exact_keys(
                    evidence,
                    common_evidence_fields | {"failure"},
                    f"pilot evidence {relative_path.as_posix()}",
                )
            expected_status = "completed" if status == "completed" else "failed"
            bindings = {
                "schema_version": PILOT_EVIDENCE_SCHEMA_VERSION,
                "artifact_kind": expected_artifact_kind,
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
            if status == "completed":
                checkpoint_shas.add(
                    _sha256(
                        evidence.get("checkpoint_sha256"),
                        f"pilot evidence {relative_path} checkpoint digest",
                    )
                )
                evaluation = _mapping(
                    evidence.get("evaluation"),
                    f"pilot evidence {relative_path} evaluation",
                )
                _exact_keys(
                    evaluation,
                    {
                        "metric_branch",
                        "requested_samples",
                        "nfe",
                        "quality",
                        "diversity",
                    },
                    f"pilot evidence {relative_path} evaluation",
                )
                metric_branch = evaluation.get("metric_branch")
                if metric_branch not in {"released_comparable", "strict"}:
                    raise GateValidationError("pilot metric branch is invalid")
                requested_samples = _integer(
                    evaluation.get("requested_samples"),
                    f"pilot evidence {relative_path} requested samples",
                    minimum=1,
                )
                nfe = _integer(
                    evaluation.get("nfe"),
                    f"pilot evidence {relative_path} NFE",
                    minimum=1,
                )
                quality = _finite(
                    evaluation.get("quality"),
                    f"pilot evidence {relative_path} quality",
                )
                diversity = _finite(
                    evaluation.get("diversity"),
                    f"pilot evidence {relative_path} diversity",
                )
                if not 0.0 <= quality <= 1.0:
                    raise GateValidationError("pilot quality must be in [0, 1]")
                if not 0.0 <= diversity <= 1.0:
                    raise GateValidationError("pilot diversity must be in [0, 1]")
                qualities.append(quality)
                diversities.append(diversity)
                observed_operating_points.append(
                    {
                        "pilot_seed": ref_seed,
                        "requested_samples": requested_samples,
                        "nfe": nfe,
                        "metric_branch": metric_branch,
                    }
                )
            else:
                failure = _mapping(
                    evidence.get("failure"),
                    f"pilot evidence {relative_path} failure",
                )
                _exact_keys(
                    failure,
                    {"stage", "reason"},
                    f"pilot evidence {relative_path} failure",
                )
                if failure.get("stage") not in {
                    "training",
                    "sampling",
                    "evaluation",
                    "infrastructure",
                }:
                    raise GateValidationError("pilot failure stage is invalid")
                if (
                    not isinstance(failure.get("reason"), str)
                    or not failure["reason"].strip()
                ):
                    raise GateValidationError("pilot failure reason must be nonempty")
            artifact_count += 1
        if evidence_seeds != set(seeds):
            raise GateValidationError(
                f"candidate attempt {attempt_id} lacks evidence for a pilot seed"
            )
        if status == "completed":
            if len(checkpoint_shas) != 1:
                raise GateValidationError(
                    "all seed evidence for one attempt must use one checkpoint"
                )
            derived_eligible = tuple(seeds) == REGISTERED_SELECTION_PILOT_SEEDS and all(
                point["requested_samples"]
                == REGISTERED_SELECTION_SAMPLES_PER_SEED
                and point["nfe"] == REGISTERED_SELECTION_NFE
                and point["metric_branch"]
                == REGISTERED_SELECTION_METRIC_BRANCH
                for point in observed_operating_points
            )
            if eligible is not derived_eligible:
                raise GateValidationError(
                    "completed pilot eligibility disagrees with the frozen "
                    "registered operating point"
                )
            recomputed_quality = statistics.fmean(qualities)
            recomputed_diversity = statistics.fmean(diversities)
            if derived_eligible:
                _close(
                    declared_quality,
                    recomputed_quality,
                    f"candidate attempt {attempt_id} mean released quality",
                )
                _close(
                    declared_diversity,
                    recomputed_diversity,
                    f"candidate attempt {attempt_id} mean released diversity",
                )
                eligible_attempts.append(
                    {
                        "attempt_id": attempt_id,
                        "candidate_id": attempt_candidate_id,
                        "quality": recomputed_quality,
                        "diversity": recomputed_diversity,
                    }
                )
            else:
                ineligible_completed_attempt_count += 1
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
        raise GateValidationError("candidate ledger selection rule is not the frozen enum")
    if selection.get("checkpoint_selection_rule") != CHECKPOINT_SELECTION_RULE:
        raise GateValidationError(
            "candidate ledger checkpoint-selection rule is not the frozen enum"
        )
    if not eligible_attempts:
        raise GateValidationError("candidate ledger has no eligible completed attempt")
    expected_selected = min(
        eligible_attempts,
        key=lambda item: (
            -item["quality"],
            -item["diversity"],
            item["attempt_id"],
        ),
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
    return {
        "attempt_count": len(attempts),
        "eligible_attempt_count": len(eligible_attempts),
        "ineligible_completed_attempt_count": ineligible_completed_attempt_count,
        "failed_attempt_count": failed_attempt_count,
        "committed_pilot_artifact_count": artifact_count,
        "selected_attempt_id": expected_selected["attempt_id"],
        "selected_candidate_id": expected_selected["candidate_id"],
        "selected_score": {
            "mean_released_quality": expected_selected["quality"],
            "mean_released_diversity": expected_selected["diversity"],
        },
        "selection_recomputed_from_pilot_evidence": True,
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
) -> dict[str, Any]:
    """Prove the benchmark revision already contained the lock and ledger."""

    if _git_blob(benchmark_revision, candidate_lock_path) != candidate_lock_bytes:
        raise GateValidationError("benchmark revision candidate-lock blob differs")
    protocol_blob = _git_blob(benchmark_revision, PROTOCOL_RELATIVE_PATH)
    if _sha256_bytes(protocol_blob) != PROTOCOL_SHA256:
        raise GateValidationError("benchmark revision superiority protocol blob differs")
    baseline_blob = _git_blob(benchmark_revision, BASELINE_RELATIVE_PATH)
    if _sha256_bytes(baseline_blob) != BASELINE_SHA256:
        raise GateValidationError("benchmark revision baseline manifest blob differs")
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
    ledger_evidence = validate_candidate_ledger(
        ledger,
        candidate_id=lock["candidate_id"],
        artifact_loader=lambda relative_path: _git_blob(
            benchmark_revision, relative_path
        ),
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
    config_blob = _git_blob(
        benchmark_revision, lock["evaluation_config_relative_path"]
    )
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
    gate_blob = _git_blob(
        benchmark_revision, Path("scripts/udlm/superiority_gate.py")
    )
    if _sha256_bytes(gate_blob) != lock["gate_source_sha256"]:
        raise GateValidationError("benchmark revision superiority-gate blob differs")
    report_blob = _git_blob(
        benchmark_revision, Path("scripts/exps/denovo/report.py")
    )
    if _sha256_bytes(report_blob) != lock["report_source_sha256"]:
        raise GateValidationError("benchmark revision de-novo-report blob differs")
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
    return {
        "benchmark_revision": benchmark_revision,
        "candidate_lock_relative_path": candidate_lock_path.as_posix(),
        "candidate_lock_sha256": _sha256_bytes(candidate_lock_bytes),
        "candidate_lock_exact_blob_at_benchmark_revision": True,
        "protocol_exact_blob_at_benchmark_revision": True,
        "baseline_exact_blob_at_benchmark_revision": True,
        "candidate_ledger_relative_path": ledger_ref["relative_path"].as_posix(),
        "candidate_ledger_sha256": ledger_ref["sha256"],
        "candidate_ledger_exact_blob_at_benchmark_revision": True,
        "evaluation_config_exact_blob_at_benchmark_revision": True,
        "ema_sampler_source_exact_blob_at_benchmark_revision": True,
        "benchmark_runner_exact_blob_at_benchmark_revision": True,
        "analysis_sources_exact_blobs_at_benchmark_revision": True,
        "committed_pilot_artifact_count": ledger_evidence[
            "committed_pilot_artifact_count"
        ],
        "eligible_pilot_attempt_count": ledger_evidence[
            "eligible_attempt_count"
        ],
        "ineligible_completed_pilot_attempt_count": ledger_evidence[
            "ineligible_completed_attempt_count"
        ],
        "failed_pilot_attempt_count": ledger_evidence["failed_attempt_count"],
        "registered_selection_operating_point": {
            "generation_seeds": list(REGISTERED_SELECTION_PILOT_SEEDS),
            "requested_samples_per_seed": REGISTERED_SELECTION_SAMPLES_PER_SEED,
            "nfe": REGISTERED_SELECTION_NFE,
            "metric_branch": REGISTERED_SELECTION_METRIC_BRANCH,
        },
        "selected_attempt_id": ledger_evidence["selected_attempt_id"],
        "selected_score": ledger_evidence["selected_score"],
        "selection_recomputed_from_committed_pilot_evidence": True,
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
    checkpoint = _mapping(candidate_report.get("checkpoint"), "candidate report checkpoint")
    if checkpoint.get("diffusion_type") != "udlm":
        raise GateValidationError("candidate report checkpoint is not UDLM")
    for key in ("sha256", "size_bytes", "global_step"):
        if checkpoint.get(key) != lock["checkpoint"][key]:
            raise GateValidationError(f"candidate report checkpoint {key} differs from lock")
    config = _mapping(candidate_report.get("config"), "candidate report config")
    if config.get("sha256") != lock["evaluation_config_sha256"]:
        raise GateValidationError("candidate evaluation config digest differs from lock")
    if config.get("sampling_sha256") != lock["sampling_sha256"]:
        raise GateValidationError("candidate sampling digest differs from lock")
    if config.get("sampling") != lock["sampling_config"]:
        raise GateValidationError("candidate sampling settings differ from lock")
    tracking = _mapping(config.get("git_tracking"), "candidate config tracking")
    if tracking.get("relative_path") != lock["evaluation_config_relative_path"].as_posix():
        raise GateValidationError("candidate evaluation config path differs from lock")

    generation = _mapping(
        candidate_report.get("generation_protocol"), "candidate generation protocol"
    )
    if generation.get("diffusion_type") != "udlm":
        raise GateValidationError("candidate generation protocol is not UDLM")
    if generation.get("nfe") != EXPECTED_NFE or generation.get("num_steps") != EXPECTED_NFE:
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
        metrics = _mapping(run.get("metrics"), f"candidate seed {expected_seed} metrics")
        if set(metrics) != {"released_comparable", "strict"}:
            raise GateValidationError("candidate must report repaired and strict branches")
        branch = _mapping(metrics["released_comparable"], "candidate released metrics")
        requested = _integer(
            branch.get("validity_denominator"), "candidate validity denominator", minimum=1
        )
        quality_denominator = _integer(
            branch.get("quality_denominator"), "candidate quality denominator", minimum=1
        )
        if requested != EXPECTED_SAMPLES_PER_SEED or quality_denominator != requested:
            raise GateValidationError("candidate metric denominator is not 1000")
        valid_count = _integer(branch.get("valid_count"), "candidate valid count", minimum=0)
        unique_count = _integer(branch.get("unique_count"), "candidate unique count", minimum=0)
        quality_count = _integer(branch.get("quality_count"), "candidate quality count", minimum=0)
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
            _close(branch.get(metric), expected, f"candidate seed {expected_seed} {metric}")
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
        raw_hashes.append(_sha256(run.get("raw_samples_sha256"), "candidate raw digest"))
        summary_hashes.append(_sha256(run.get("summary_sha256"), "candidate summary digest"))
    if len(revisions) != 1:
        raise GateValidationError("candidate final seeds used different revisions")
    if any(timestamp < lock["locked_at"] for timestamp in started_at):
        raise GateValidationError("a final candidate run predates the candidate lock")
    if len(set(raw_hashes)) != 3 or len(set(summary_hashes)) != 3:
        raise GateValidationError("candidate final evidence hashes are not distinct")

    aggregate = _mapping(candidate_report.get("aggregate_metrics"), "candidate aggregate")
    released = _mapping(aggregate.get("released_comparable"), "candidate released aggregate")
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
            raise GateValidationError(f"candidate aggregate seed values differ for {metric}")
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
        raise GateValidationError("candidate sampler source differs from candidate lock")
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
        raise GateValidationError("candidate benchmark runner differs from candidate lock")
    return {
        "values": values,
        "means": {metric: statistics.fmean(series) for metric, series in values.items()},
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


def evaluate_candidate_report(
    candidate_report: Mapping[str, Any],
    baseline_manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
    candidate_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a deterministic registered decision from validated report data."""

    validate_protocol(protocol)
    baseline = validate_baseline_manifest(baseline_manifest)
    lock = validate_candidate_lock(candidate_lock, protocol)
    candidate = _candidate_series(candidate_report, lock)

    point_protocol = protocol["point_estimate_gates"]
    baseline_means = baseline["means"]
    if point_protocol["validity"]["threshold"] != baseline_means["validity"]:
        raise GateValidationError("validity point threshold differs from baseline")
    if point_protocol["uniqueness"]["threshold"] != baseline_means["uniqueness"]:
        raise GateValidationError("uniqueness point threshold differs from baseline")
    if point_protocol["quality"]["threshold"] != baseline_means["quality"]:
        raise GateValidationError("quality point threshold differs from baseline")
    diversity_threshold = baseline_means["diversity"] + point_protocol["diversity"]["margin"]
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
                for seed, digest in zip(EXPECTED_SEEDS, candidate["raw_hashes"], strict=True)
            ],
            "summary_sha256_by_seed": [
                {"seed": seed, "sha256": digest}
                for seed, digest in zip(EXPECTED_SEEDS, candidate["summary_hashes"], strict=True)
            ],
        },
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
    if _sha256_bytes(gate_bytes) != lock["gate_source_sha256"]:
        raise GateValidationError("runtime superiority-gate source differs from lock")
    if _sha256_bytes(report_bytes) != lock["report_source_sha256"]:
        raise GateValidationError("runtime de-novo report source differs from lock")
    import scipy

    if scipy.__version__ != lock["scipy_version"]:
        raise GateValidationError("runtime SciPy version differs from candidate lock")
    return {
        "gate_source_sha256": lock["gate_source_sha256"],
        "report_source_sha256": lock["report_source_sha256"],
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
    lock_relative = _relative_path(
        args.candidate_lock.as_posix(), "candidate lock path", suffix=".json"
    )
    lock_bytes = _repository_artifact_bytes(lock_relative, label="candidate lock")
    lock_json = _mapping(
        strict_json_loads(lock_bytes, label="candidate lock"), "candidate lock"
    )
    lock = validate_candidate_lock(lock_json, protocol)
    training_evidence = validate_training_evidence(lock)
    analysis_evidence = validate_analysis_runtime(lock)
    candidate_report = denovo_report.collect_report(args.runs_dir)
    candidate = _candidate_series(candidate_report, lock)
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
        candidate_report, baseline, protocol, lock_json
    )
    decision["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    decision["candidate_lock"] = firewall
    decision["training_evidence"] = training_evidence
    decision["analysis_evidence"] = analysis_evidence
    decision["analysis_evidence"]["clean_pushed_source"] = clean_source
    decision["candidate_runs_root"] = candidate_report["input_root"]
    _atomic_write_json_exclusive(args.output, decision)
    print(f"Superiority gate: {decision['status'].upper()}")
    print(f"Decision: {Path(args.output).resolve()}")
    return 0 if decision["superiority_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

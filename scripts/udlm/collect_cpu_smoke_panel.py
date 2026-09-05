"""Freeze a matched three-prior CPU smoke panel without making ranking claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
EXPECTED_VARIANTS = (
    "release_uniform",
    "schedule_uniform",
    "empirical_frequency",
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _read_regular_file(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"input is not a regular file: {path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if identity_before != identity_after:
            raise RuntimeError(f"input changed while being read: {path}")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise RuntimeError(f"input size changed while being read: {path}")
        return payload
    finally:
        os.close(descriptor)


def _git_provenance() -> dict[str, object]:
    def command(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(ROOT_DIR), *arguments],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    status = command("status", "--porcelain=v1", "--untracked-files=normal")
    commit = command("rev-parse", "HEAD")
    upstream = command("rev-parse", "@{upstream}")
    if status:
        raise RuntimeError("worktree must be clean before freezing a smoke panel")
    if commit != upstream:
        raise RuntimeError("HEAD must equal its upstream before freezing a smoke panel")
    return {"commit": commit, "upstream": upstream, "dirty": False}


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be finite")
    return converted


def _validate_diagnostic_pair(record: dict[str, Any], variant: str) -> None:
    before = record.get("fixed_diagnostics_before")
    after = record.get("fixed_diagnostics_after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError(f"{variant}: missing fixed diagnostics")
    if set(before) != {"0.1", "0.5", "0.9"} or set(after) != set(before):
        raise ValueError(f"{variant}: unexpected fixed diagnostic grid")
    for time_value in before:
        if not isinstance(before[time_value], dict) or not isinstance(
            after[time_value], dict
        ):
            raise ValueError(f"{variant}: malformed t={time_value} diagnostic")
        _finite_number(before[time_value].get("loss"), f"{variant} before loss")
        _finite_number(after[time_value].get("loss"), f"{variant} after loss")
        before_digest = before[time_value].get("corrupted_token_ids_sha256")
        after_digest = after[time_value].get("corrupted_token_ids_sha256")
        if before_digest != after_digest or not isinstance(before_digest, str):
            raise ValueError(f"{variant}: corruption changed at t={time_value}")
    if float(after["0.5"]["loss"]) >= float(before["0.5"]["loss"]):
        raise ValueError(f"{variant}: t=0.5 loss did not improve")


def _validate_result(record: dict[str, Any], expected_variant: str) -> None:
    if record.get("schema_version") != 2:
        raise ValueError(f"{expected_variant}: unsupported smoke schema")
    if record.get("purpose") != (
        "bounded CPU integration smoke; not benchmark or superiority evidence"
    ):
        raise ValueError(f"{expected_variant}: invalid claim scope")
    if record.get("prior_variant") != expected_variant:
        raise ValueError(f"{expected_variant}: prior variant mismatch")
    if record.get("device") != "cpu":
        raise ValueError(f"{expected_variant}: smoke device must be CPU")
    git = record.get("git")
    if not isinstance(git, dict) or git.get("dirty") is not False:
        raise ValueError(f"{expected_variant}: source tree was not clean")
    if not git.get("commit") or git.get("commit") != git.get("upstream"):
        raise ValueError(f"{expected_variant}: source commit was not pushed")
    metadata = record.get("prior_metadata")
    if not isinstance(metadata, dict) or metadata.get("variant") != expected_variant:
        raise ValueError(f"{expected_variant}: prior metadata mismatch")
    if expected_variant == "release_uniform":
        expected_role = "faithful_release_control"
        expected_process = "released_continuous_uniform"
        expected_schedule = "released_ideal_loss_residual_forward"
    elif expected_variant == "schedule_uniform":
        expected_role = "schedule_repair_uniform_control"
        expected_process = "rank_one_continuous_categorical"
        expected_schedule = "schedule_consistent_residual_forward_and_loss"
    else:
        expected_role = "empirical_prior_treatment"
        expected_process = "rank_one_continuous_categorical"
        expected_schedule = "schedule_consistent_residual_forward_and_loss"
    expected_identity = {
        "comparison_role": expected_role,
        "process_family": expected_process,
        "schedule_variant": expected_schedule,
    }
    for field, expected in expected_identity.items():
        if metadata.get(field) != expected:
            raise ValueError(f"{expected_variant}: prior metadata {field} mismatch")
    if expected_variant == "empirical_frequency":
        if metadata.get("uniform_mixture_weight") != 0.01:
            raise ValueError("empirical_frequency: unexpected mixture weight")
        if metadata.get("frequency_artifact_sha256") != (
            "088c78e75611f3cc42c4011e1da6f65a377e673b9cba07a28b126b0fc62f06ed"
        ):
            raise ValueError("empirical_frequency: frequency artifact mismatch")
    elif metadata.get("uniform_mixture_weight") is not None:
        raise ValueError(f"{expected_variant}: unexpected empirical mixture")
    if record.get("all_losses_finite") is not True:
        raise ValueError(f"{expected_variant}: non-finite loss was reported")
    if record.get("all_gradient_norms_finite") is not True:
        raise ValueError(f"{expected_variant}: non-finite gradient was reported")
    first_mean = _finite_number(
        record.get("loss_first_five_mean"), f"{expected_variant} first loss mean"
    )
    last_mean = _finite_number(
        record.get("loss_last_five_mean"), f"{expected_variant} last loss mean"
    )
    if last_mean >= first_mean:
        raise ValueError(f"{expected_variant}: loss window did not decrease")
    requested = record.get("sample_count_requested")
    decoded = record.get("no_repair_decodable_samples")
    generated = record.get("generated_smiles")
    if (
        type(requested) is not int
        or requested <= 0
        or type(decoded) is not int
        or not 0 < decoded <= requested
        or not isinstance(generated, list)
        or decoded != len(generated)
    ):
        raise ValueError(f"{expected_variant}: invalid no-repair decode evidence")
    _validate_diagnostic_pair(record, expected_variant)


def build_panel(
    artifacts: dict[str, tuple[Path, bytes, dict[str, Any]]],
    *,
    git: dict[str, object],
) -> dict[str, object]:
    if tuple(sorted(artifacts)) != tuple(sorted(EXPECTED_VARIANTS)):
        raise ValueError("panel requires exactly one artifact for each prior variant")
    for variant in EXPECTED_VARIANTS:
        _validate_result(artifacts[variant][2], variant)

    records = [artifacts[variant][2] for variant in EXPECTED_VARIANTS]
    shared_fields = (
        "seed",
        "steps",
        "sampling_steps",
        "sample_count_requested",
        "exclude_special_tokens",
        "toy_smiles",
        "batch_shape",
        "clean_input_ids_sha256",
        "attention_mask_sha256",
    )
    for field in shared_fields:
        values = [record.get(field) for record in records]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"smoke panel is not matched on {field}")
    artifact_commits = {record["git"]["commit"] for record in records}
    if artifact_commits != {git["commit"]}:
        raise ValueError("smoke artifacts do not match the current pushed commit")

    return {
        "schema_version": 1,
        "purpose": "matched three-prior CPU integration panel",
        "claim_scope": (
            "All rows are tiny same-seed optimization and reverse-chain checks. "
            "They cannot rank priors, estimate molecular quality, or support a "
            "UDLM-versus-GenMol superiority claim."
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git,
        "matched_fields": {
            field: records[0][field] for field in shared_fields
        },
        "variants": {
            variant: {
                "source_path": str(artifacts[variant][0].relative_to(ROOT_DIR)),
                "source_artifact_sha256": _sha256(artifacts[variant][1]),
                "result": artifacts[variant][2],
            }
            for variant in EXPECTED_VARIANTS
        },
    }


def _write_json_exclusive(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.input) != len(EXPECTED_VARIANTS):
        raise ValueError("provide exactly three --input artifacts")
    output = args.output.resolve()
    if output != ROOT_DIR and ROOT_DIR not in output.parents:
        raise ValueError("output must remain inside the repository")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite panel artifact: {output}")

    artifacts: dict[str, tuple[Path, bytes, dict[str, Any]]] = {}
    for raw_path in args.input:
        path = raw_path.resolve()
        if path != ROOT_DIR and ROOT_DIR not in path.parents:
            raise ValueError("every input must remain inside the repository")
        payload = _read_regular_file(path)
        record = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
        if not isinstance(record, dict):
            raise ValueError(f"smoke artifact must be a JSON object: {path}")
        variant = record.get("prior_variant")
        if variant not in EXPECTED_VARIANTS:
            raise ValueError(f"unexpected prior variant in {path}: {variant!r}")
        if variant in artifacts:
            raise ValueError(f"duplicate prior variant: {variant}")
        artifacts[variant] = (path, payload, record)

    git = _git_provenance()
    panel = build_panel(artifacts, git=git)
    if _git_provenance() != git:
        raise RuntimeError("Git state changed while collecting the smoke panel")
    _write_json_exclusive(output, panel)
    print(json.dumps(panel, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

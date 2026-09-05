"""Freeze a matched three-prior CPU smoke panel without making ranking claims."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT_DIR = Path(__file__).resolve().parents[2]
EXPECTED_VARIANTS = (
    "release_uniform",
    "schedule_uniform",
    "empirical_frequency",
)
FREQUENCY_RELATIVE_PATH = Path(
    "experiments/udlm/token_frequency/train_first_10000.json"
)
FREQUENCY_SHA256 = "088c78e75611f3cc42c4011e1da6f65a377e673b9cba07a28b126b0fc62f06ed"
TOKENIZER_SHA256 = "0db5f4dbdc7e8ff759e98483759611a426e187ee7f3f0a91edc8800abe7bf140"
SOURCE_PATHS = {
    "smoke_runner": Path("scripts/udlm/cpu_smoke.py"),
    "model": Path("src/genmol/model.py"),
    "diffusion": Path("src/genmol/diffusion.py"),
    "sampler": Path("src/genmol/sampler.py"),
}
PRIOR_METADATA_FIELDS = {
    "schema_version",
    "variant",
    "comparison_role",
    "process_family",
    "schedule_variant",
    "objective_scope",
    "prior_source",
    "full_vocab_size",
    "active_vocab_size",
    "excluded_token_ids",
    "sampling_eps",
    "noise_eps",
    "antithetic_sampling",
    "active_token_ids_sha256",
    "stationary_probs_sha256",
    "uniform_mixture_weight",
    "frequency_artifact_path",
    "frequency_artifact_sha256",
    "frequency_artifact_schema_version",
    "frequency_example_count",
    "frequency_content_token_count",
    "frequency_active_token_count",
    "frequency_dataset_repo_id",
    "frequency_dataset_revision",
    "frequency_dataset_split",
    "frequency_dataset_selection",
    "frequency_ordered_text_sha256",
    "frequency_implementation_git_sha",
    "tokenizer_repo_id",
    "tokenizer_revision",
    "tokenizer_json_sha256",
}
FREQUENCY_METADATA_FIELDS = {
    "uniform_mixture_weight",
    "frequency_artifact_path",
    "frequency_artifact_sha256",
    "frequency_artifact_schema_version",
    "frequency_example_count",
    "frequency_content_token_count",
    "frequency_active_token_count",
    "frequency_dataset_repo_id",
    "frequency_dataset_revision",
    "frequency_dataset_split",
    "frequency_dataset_selection",
    "frequency_ordered_text_sha256",
    "frequency_implementation_git_sha",
}
VARIANT_IDENTITIES = {
    "release_uniform": {
        "comparison_role": "faithful_release_control",
        "process_family": "released_continuous_uniform",
        "schedule_variant": "released_ideal_loss_residual_forward",
        "objective_scope": "released_model_dependent_ct_integrand",
        "prior_source": "uniform",
    },
    "schedule_uniform": {
        "comparison_role": "schedule_repair_uniform_control",
        "process_family": "rank_one_continuous_categorical",
        "schedule_variant": "schedule_consistent_residual_forward_and_loss",
        "objective_scope": (
            "model_dependent_ct_integrand_without_parameter_independent_endpoint_kl"
        ),
        "prior_source": "uniform",
    },
    "empirical_frequency": {
        "comparison_role": "empirical_prior_treatment",
        "process_family": "rank_one_continuous_categorical",
        "schedule_variant": "schedule_consistent_residual_forward_and_loss",
        "objective_scope": (
            "model_dependent_ct_integrand_without_parameter_independent_endpoint_kl"
        ),
        "prior_source": "pinned_frequency_artifact_uniform_mixture",
    },
}


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


def _canonical_sequence_sha256(values: list[int] | list[float]) -> str:
    canonical = [value.hex() if isinstance(value, float) else value for value in values]
    payload = json.dumps(canonical, separators=(",", ":")).encode("ascii")
    return _sha256(payload)


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _current_source_inputs(variant: str) -> dict[str, dict[str, object]]:
    paths = dict(SOURCE_PATHS)
    if variant == "empirical_frequency":
        paths["frequency_artifact"] = FREQUENCY_RELATIVE_PATH
    records = {}
    for name, relative_path in paths.items():
        payload = _read_regular_file(ROOT_DIR / relative_path)
        records[name] = {
            "path": relative_path.as_posix(),
            "sha256": _sha256(payload),
            "size_bytes": len(payload),
        }
    return records


def _load_frequency_artifact() -> tuple[dict[str, Any], list[int]]:
    payload = _read_regular_file(ROOT_DIR / FREQUENCY_RELATIVE_PATH)
    if _sha256(payload) != FREQUENCY_SHA256:
        raise ValueError("pinned frequency artifact bytes changed")
    artifact = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(artifact, dict):
        raise ValueError("frequency artifact must be a JSON object")
    counts = artifact.get("counts_by_token_id")
    if (
        artifact.get("schema_version") != 1
        or artifact.get("example_count") != 10_000
        or artifact.get("content_token_count") != 517_090
        or not isinstance(counts, list)
        or len(counts) != 1880
        or any(type(count) is not int or count < 0 for count in counts)
        or sum(counts) != 517_090
    ):
        raise ValueError("frequency artifact counts or schema changed")
    return artifact, counts


def _expected_prior_geometry(
    variant: str,
    *,
    excluded_token_ids: list[int],
    mixture_weight: float,
) -> tuple[list[int], list[float], dict[str, Any] | None, int | None]:
    active_ids = [
        token_id for token_id in range(1880) if token_id not in excluded_token_ids
    ]
    if variant == "empirical_frequency":
        artifact, counts = _load_frequency_artifact()
        active_count = sum(counts[token_id] for token_id in active_ids)
        if active_count <= 0:
            raise ValueError("frequency artifact has no active token observations")
        # Preserve model.py's two-step Python-float evaluation exactly. Folding
        # the division into the mixture expression changes some entries by one
        # ULP and therefore produces a different canonical prior digest.
        empirical_probabilities = [
            counts[token_id] / active_count for token_id in active_ids
        ]
        probabilities = [
            (1.0 - mixture_weight) * empirical + mixture_weight / len(active_ids)
            for empirical in empirical_probabilities
        ]
    else:
        artifact = None
        active_count = None
        probabilities = [1.0 / len(active_ids)] * len(active_ids)
    canonical = torch.tensor(probabilities, dtype=torch.float64)
    canonical /= canonical.sum()
    return (
        active_ids,
        [float(value) for value in canonical.tolist()],
        artifact,
        active_count,
    )


def _validate_effective_config(record: dict[str, Any], variant: str) -> dict[str, Any]:
    config = record.get("effective_config")
    if not isinstance(config, dict):
        raise ValueError(f"{variant}: effective_config must be an object")
    training = config.get("training")
    model = config.get("model")
    if not isinstance(training, dict) or not isinstance(model, dict):
        raise ValueError(f"{variant}: malformed effective model/training config")
    udlm = training.get("udlm")
    if not isinstance(udlm, dict):
        raise ValueError(f"{variant}: missing effective UDLM config")
    expected = {
        "diffusion": "udlm",
        "antithetic_sampling": True,
        "sampling_eps": 1e-3,
        "global_mean_loss": True,
    }
    if any(training.get(field) != value for field, value in expected.items()):
        raise ValueError(f"{variant}: effective training config changed")
    if model.get("vocab_size") != 1880 or model.get("pad_token_id") != 3:
        raise ValueError(f"{variant}: effective tokenizer/model geometry changed")
    if (
        udlm.get("prior_variant") != variant
        or udlm.get("empirical_uniform_mix") != 0.01
        or udlm.get("exclude_special_tokens") != record.get("exclude_special_tokens")
        or udlm.get("noise_eps") != 1e-3
        or udlm.get("sampling_steps") != record.get("sampling_steps")
    ):
        raise ValueError(f"{variant}: result and effective UDLM config disagree")
    if config.get("seed") != 1 or record.get("seed") != 1:
        raise ValueError(f"{variant}: unexpected smoke seed")
    return udlm


def _validate_prior_metadata(record: dict[str, Any], variant: str) -> None:
    metadata = record.get("prior_metadata")
    if not isinstance(metadata, dict) or set(metadata) != PRIOR_METADATA_FIELDS:
        raise ValueError(f"{variant}: prior metadata schema mismatch")
    if metadata.get("schema_version") != 1 or metadata.get("variant") != variant:
        raise ValueError(f"{variant}: prior metadata identity mismatch")
    for field, expected in VARIANT_IDENTITIES[variant].items():
        if metadata.get(field) != expected:
            raise ValueError(f"{variant}: prior metadata {field} mismatch")
    udlm = _validate_effective_config(record, variant)
    excluded = [0, 1, 2, 3, 4] if record["exclude_special_tokens"] else []
    if metadata.get("excluded_token_ids") != excluded:
        raise ValueError(f"{variant}: excluded-token metadata mismatch")
    mixture = float(udlm["empirical_uniform_mix"])
    active_ids, probabilities, artifact, active_count = _expected_prior_geometry(
        variant,
        excluded_token_ids=excluded,
        mixture_weight=mixture,
    )
    scalar_expectations = {
        "full_vocab_size": 1880,
        "active_vocab_size": len(active_ids),
        "sampling_eps": 1e-3,
        "noise_eps": 1e-3,
        "antithetic_sampling": True,
        "active_token_ids_sha256": _canonical_sequence_sha256(active_ids),
        "stationary_probs_sha256": _canonical_sequence_sha256(probabilities),
        "tokenizer_repo_id": "datamol-io/safe-gpt",
        "tokenizer_revision": "3d5fa0988383e898d5ac5db7cd52bf715bc37061",
        "tokenizer_json_sha256": TOKENIZER_SHA256,
    }
    for field, expected in scalar_expectations.items():
        if metadata.get(field) != expected:
            raise ValueError(f"{variant}: prior metadata {field} mismatch")
    _require_sha256(metadata["active_token_ids_sha256"], "active token digest")
    _require_sha256(metadata["stationary_probs_sha256"], "stationary prior digest")

    if variant != "empirical_frequency":
        if any(metadata.get(field) is not None for field in FREQUENCY_METADATA_FIELDS):
            raise ValueError(f"{variant}: unexpected empirical-prior metadata")
        return
    assert artifact is not None and active_count is not None
    dataset = artifact["dataset"]
    frequency_expected = {
        "uniform_mixture_weight": mixture,
        "frequency_artifact_path": FREQUENCY_RELATIVE_PATH.as_posix(),
        "frequency_artifact_sha256": FREQUENCY_SHA256,
        "frequency_artifact_schema_version": artifact["schema_version"],
        "frequency_example_count": artifact["example_count"],
        "frequency_content_token_count": artifact["content_token_count"],
        "frequency_active_token_count": active_count,
        "frequency_dataset_repo_id": dataset["repo_id"],
        "frequency_dataset_revision": dataset["revision"],
        "frequency_dataset_split": dataset["split"],
        "frequency_dataset_selection": dataset["selection"],
        "frequency_ordered_text_sha256": dataset["ordered_safe_text_sha256"],
        "frequency_implementation_git_sha": artifact["git_sha"],
    }
    for field, expected in frequency_expected.items():
        if metadata.get(field) != expected:
            raise ValueError(f"{variant}: empirical metadata {field} mismatch")


def _validate_source_inputs(record: dict[str, Any], variant: str) -> None:
    source_inputs = record.get("source_inputs")
    expected = _current_source_inputs(variant)
    if source_inputs != expected:
        raise ValueError(f"{variant}: source input hashes do not match current files")


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
        if before_digest != after_digest:
            raise ValueError(f"{variant}: corruption changed at t={time_value}")
        _require_sha256(before_digest, f"{variant} t={time_value} corruption digest")
    if float(after["0.5"]["loss"]) >= float(before["0.5"]["loss"]):
        raise ValueError(f"{variant}: t=0.5 loss did not improve")


def _validate_result(
    record: dict[str, Any],
    expected_variant: str,
    *,
    strict_provenance: bool = True,
) -> None:
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
    if git.get("status_porcelain") not in (None, []):
        raise ValueError(f"{expected_variant}: recorded Git status was not empty")
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
    if strict_provenance:
        _validate_prior_metadata(record, expected_variant)
        _validate_source_inputs(record, expected_variant)
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
        or any(not isinstance(value, str) or not value for value in generated)
        or decoded != len(generated)
    ):
        raise ValueError(f"{expected_variant}: invalid no-repair decode evidence")
    if (
        type(record.get("steps")) is not int
        or record["steps"] <= 0
        or type(record.get("sampling_steps")) is not int
        or record["sampling_steps"] <= 0
        or record.get("empirical_uniform_mix_requested") != 0.01
    ):
        raise ValueError(f"{expected_variant}: invalid bounded-run configuration")
    _finite_number(record.get("runtime_seconds"), f"{expected_variant} runtime")
    _validate_diagnostic_pair(record, expected_variant)


def build_panel(
    artifacts: dict[str, tuple[Path, bytes, dict[str, Any]]],
    *,
    git: dict[str, object],
    strict_provenance: bool = True,
) -> dict[str, object]:
    if tuple(sorted(artifacts)) != tuple(sorted(EXPECTED_VARIANTS)):
        raise ValueError("panel requires exactly one artifact for each prior variant")
    for variant in EXPECTED_VARIANTS:
        _validate_result(
            artifacts[variant][2],
            variant,
            strict_provenance=strict_provenance,
        )

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
    _require_sha256(records[0]["clean_input_ids_sha256"], "clean input digest")
    _require_sha256(records[0]["attention_mask_sha256"], "attention-mask digest")
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
        "matched_fields": {field: records[0][field] for field in shared_fields},
        "variants": {
            variant: {
                "source_path": str(artifacts[variant][0].relative_to(ROOT_DIR)),
                "source_artifact_sha256": _sha256(artifacts[variant][1]),
                "source_artifact_base64": base64.b64encode(
                    artifacts[variant][1]
                ).decode("ascii"),
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


def _bind_output_path(path: Path) -> Path:
    """Resolve the parent while retaining and rejecting any existing leaf."""

    lexical = Path(os.path.abspath(os.fspath(path)))
    if not lexical.name:
        raise ValueError("output must name a JSON file")
    bound = lexical.parent.resolve() / lexical.name
    if bound == ROOT_DIR or ROOT_DIR not in bound.parents:
        raise ValueError("output must remain inside the repository")
    if os.path.lexists(bound):
        raise FileExistsError(f"refusing to overwrite panel artifact: {bound}")
    return bound


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.input) != len(EXPECTED_VARIANTS):
        raise ValueError("provide exactly three --input artifacts")
    output = _bind_output_path(args.output)

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

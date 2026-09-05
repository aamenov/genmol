"""Evaluate a UDLM checkpoint on the frozen held-out denoising panel.

This is a diagnostic, not a molecule-generation benchmark.  It measures the
method-specific model-dependent loss and clean-token reconstruction for one
deterministic corruption per row at each fixed time.  For categorical UDLM it
also reports the parameter-independent endpoint-prior KL separately.  The grid
is not an integrated or unbiased NELBO estimate.  It does not decode molecules
or support a claim that one generator beats another.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPOSITORY_ROOT.parents[1]
for _import_root in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    while str(_import_root) in sys.path:
        sys.path.remove(str(_import_root))
    sys.path.insert(0, str(_import_root))

from scripts.udlm.materialize_validation_panel import validate_panel  # noqa: E402
from genmol.diffusion import (  # noqa: E402
    ContinuousCategoricalDiffusion,
    ContinuousUniformDiffusion,
)


DEFAULT_PANEL = REPOSITORY_ROOT / "experiments/udlm/validation_panel/first_256.json"
DEFAULT_FREQUENCIES = (
    REPOSITORY_ROOT / "experiments/udlm/token_frequency/train_first_10000.json"
)
FROZEN_PANEL_SHA256 = "e2493da4f3cb3217b48c78dc2901dc7524d7d90dc3959cffa87a1b6f8a9a7658"
FROZEN_FREQUENCY_SHA256 = (
    "088c78e75611f3cc42c4011e1da6f65a377e673b9cba07a28b126b0fc62f06ed"
)
DEFAULT_TIME_BINS = (0.1, 0.3, 0.5, 0.7, 0.9)
SCHEMA_VERSION = 3

UDLM_PRIOR_CHECKPOINT_KEY = "udlm_prior_metadata"
EMPIRICAL_FREQUENCY_RELATIVE_PATH = Path(
    "experiments/udlm/token_frequency/train_first_10000.json"
)
EMPIRICAL_FREQUENCY_ORDERED_TEXT_SHA256 = (
    "53aee8e5592fc96159788e86519abbbcc9f1ab7c6348a1cb59a939bd57051d8f"
)
UDLM_PRIOR_VARIANT_IDENTITIES = {
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
UDLM_PRIOR_METADATA_FIELDS = frozenset(
    {
        "schema_version",
        "variant",
        *next(iter(UDLM_PRIOR_VARIANT_IDENTITIES.values())),
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
)

SOURCE_INPUTS = (
    Path("scripts/udlm/evaluate_denoising_panel.py"),
    Path("scripts/udlm/materialize_validation_panel.py"),
    Path("scripts/udlm/launch_train_pilot.py"),
    Path("scripts/train.py"),
    Path("configs/base.yaml"),
    Path("configs/udlm.yaml"),
    Path("configs/udlm_categorical.yaml"),
    Path("scripts/udlm/token_frequency_audit.py"),
    Path("src/genmol/model.py"),
    Path("src/genmol/diffusion.py"),
    Path("src/genmol/backbone.py"),
    Path("src/genmol/utils/ema.py"),
    Path("src/genmol/utils/utils_data.py"),
)

FREQUENCY_BUCKETS = (
    ("unseen", 0, 0),
    ("count_1_9", 1, 9),
    ("count_10_99", 10, 99),
    ("count_100_999", 100, 999),
    ("count_ge_1000", 1_000, None),
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path, expected_sha256: str | None) -> tuple[dict, str]:
    """Parse exactly the bytes whose digest is returned.

    Reading once avoids the former hash-then-reread race in which validation
    could apply to bytes different from the bytes named by provenance.
    """

    if not path.is_file():
        raise FileNotFoundError(path)
    payload = path.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"frozen artifact hash mismatch for {path}: "
            f"{actual_sha256} != {expected_sha256}"
        )
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact root must be an object: {path}")
    return value, actual_sha256


def validate_frequency_artifact(frequencies: dict, panel: dict) -> None:
    """Validate counts and tokenizer identity against the held-out panel."""

    if frequencies.get("schema_version") != 1:
        raise ValueError("frequency artifact must use schema_version 1")
    counts = frequencies.get("counts_by_token_id")
    vocab_size = panel.get("tokenizer", {}).get("base_vocab_size")
    if type(vocab_size) is not int or vocab_size < 2:
        raise ValueError("panel tokenizer has an invalid base vocabulary size")
    if not isinstance(counts, list) or len(counts) != vocab_size:
        raise ValueError("frequency counts do not match the panel vocabulary")
    if any(type(count) is not int or count < 0 for count in counts):
        raise ValueError("frequency counts must be non-negative integers")
    if sum(counts) != frequencies.get("content_token_count"):
        raise ValueError("frequency content_token_count does not match the counts")
    if sum(count > 0 for count in counts) != frequencies.get("observed_token_types"):
        raise ValueError("frequency observed_token_types does not match the counts")

    panel_tokenizer = panel["tokenizer"]
    frequency_tokenizer = frequencies.get("tokenizer", {})
    identity_fields = (
        "repo_id",
        "revision",
        "tokenizer_json_sha256",
        "base_vocab_size",
        "special_token_ids",
    )
    for field in identity_fields:
        if frequency_tokenizer.get(field) != panel_tokenizer.get(field):
            raise ValueError(f"panel/frequency tokenizer mismatch for {field}")

    panel_dataset = panel.get("dataset", {})
    frequency_dataset = frequencies.get("dataset", {})
    for field in ("repo_id", "revision"):
        if frequency_dataset.get(field) != panel_dataset.get(field):
            raise ValueError(f"panel/frequency dataset mismatch for {field}")
    if frequency_dataset.get("split") != "train":
        raise ValueError("frequency artifact must be computed from the training split")
    if panel_dataset.get("split") != "validation":
        raise ValueError("denoising panel must come from the validation split")


def validate_evaluation_inputs(panel: dict, frequencies: dict) -> None:
    """Apply structural and cross-artifact validation before model execution."""

    validate_panel(panel)
    if panel.get("schema_version") != 1:
        raise ValueError("panel must use schema_version 1")
    tokenizer = panel.get("tokenizer", {})
    vocab_size = tokenizer.get("base_vocab_size")
    special_ids = tokenizer.get("special_token_ids")
    if type(vocab_size) is not int or vocab_size < 2:
        raise ValueError("panel tokenizer has an invalid base vocabulary size")
    if (
        not isinstance(special_ids, list)
        or not special_ids
        or any(type(token_id) is not int for token_id in special_ids)
        or special_ids != sorted(set(special_ids))
        or not all(0 <= token_id < vocab_size for token_id in special_ids)
    ):
        raise ValueError("panel tokenizer has invalid special token IDs")
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if tokenizer.get(name) not in special_ids:
            raise ValueError(f"panel {name} must belong to special_token_ids")
    for row_index, row in enumerate(panel["rows"]):
        if any(token_id >= vocab_size for token_id in row["input_ids"]):
            raise ValueError(f"out-of-vocabulary ID at panel row {row_index}")
        safe_hash = row.get("safe_sha256")
        if (
            not isinstance(safe_hash, str)
            or len(safe_hash) != 64
            or any(character not in "0123456789abcdef" for character in safe_hash)
        ):
            raise ValueError(f"invalid SAFE hash at panel row {row_index}")
    validate_frequency_artifact(frequencies, panel)


def load_frozen_artifacts(
    panel_path: Path = DEFAULT_PANEL,
    frequency_path: Path = DEFAULT_FREQUENCIES,
    *,
    panel_sha256: str | None = FROZEN_PANEL_SHA256,
    frequency_sha256: str | None = FROZEN_FREQUENCY_SHA256,
) -> tuple[dict, dict, dict]:
    """Load, hash, and validate both immutable diagnostic inputs."""

    panel_path = panel_path.resolve()
    frequency_path = frequency_path.resolve()
    panel, actual_panel_sha256 = _load_json(panel_path, panel_sha256)
    frequencies, actual_frequency_sha256 = _load_json(frequency_path, frequency_sha256)
    validate_evaluation_inputs(panel, frequencies)
    provenance = {
        "panel": {
            "path": str(panel_path),
            "sha256": actual_panel_sha256,
            "ordered_token_ids_sha256": panel["ordered_token_ids_sha256"],
            "sample_count_available": panel["sample_count"],
            "materializer_git_sha": panel.get("git_sha"),
        },
        "training_frequency": {
            "path": str(frequency_path),
            "sha256": actual_frequency_sha256,
            "example_count": frequencies.get("example_count"),
            "content_token_count": frequencies["content_token_count"],
            "audit_git_sha": frequencies.get("git_sha"),
        },
    }
    return panel, frequencies, provenance


def validate_time_bins(time_bins: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in time_bins)
    if not values:
        raise ValueError("at least one time bin is required")
    if any(not math.isfinite(value) or not 0 < value < 1 for value in values):
        raise ValueError("each time bin must be finite and lie strictly in (0, 1)")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError("time bins must be unique and strictly increasing")
    return values


def corruption_seed(base_seed: int, time_value: float, source_index: int) -> int:
    """Derive a stable per-row seed independent of evaluator batch size."""

    if not 0 <= base_seed < 2**63:
        raise ValueError("seed must lie in [0, 2**63)")
    payload = (
        f"genmol-udlm-fixed-panel-v1|{base_seed}|"
        f"{float(time_value).hex()}|{source_index}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


def frequency_bucket_masks(training_counts: torch.Tensor) -> dict[str, torch.Tensor]:
    masks = {}
    for name, lower, upper in FREQUENCY_BUCKETS:
        mask = training_counts >= lower
        if upper is not None:
            mask &= training_counts <= upper
        masks[name] = mask
    return masks


def _empty_accumulator() -> dict[str, float | int]:
    return {
        "denominator_tokens": 0,
        "production_loss_sum": 0.0,
        "clean_token_nll_sum": 0.0,
        "clean_token_top1_correct": 0,
    }


def _update_accumulator(
    accumulator: dict[str, float | int],
    mask: torch.Tensor,
    production_loss: torch.Tensor,
    clean_nll: torch.Tensor,
    clean_top1_correct: torch.Tensor,
) -> None:
    count = int(mask.sum().item())
    if count == 0:
        return
    accumulator["denominator_tokens"] += count
    accumulator["production_loss_sum"] += float(
        production_loss[mask].double().sum().item()
    )
    accumulator["clean_token_nll_sum"] += float(clean_nll[mask].double().sum().item())
    accumulator["clean_token_top1_correct"] += int(
        clean_top1_correct[mask].sum().item()
    )


def _finalize_accumulator(accumulator: dict[str, float | int]) -> dict[str, Any]:
    count = int(accumulator["denominator_tokens"])
    result = {
        "denominator_tokens": count,
        "production_loss_sum": float(accumulator["production_loss_sum"]),
        "clean_token_nll_sum": float(accumulator["clean_token_nll_sum"]),
        "clean_token_top1_correct": int(accumulator["clean_token_top1_correct"]),
    }
    if count:
        result.update(
            {
                "production_loss_mean": result["production_loss_sum"] / count,
                "clean_token_nll_mean": result["clean_token_nll_sum"] / count,
                "clean_token_top1_accuracy": (
                    result["clean_token_top1_correct"] / count
                ),
            }
        )
    else:
        result.update(
            {
                "production_loss_mean": None,
                "clean_token_nll_mean": None,
                "clean_token_top1_accuracy": None,
            }
        )
    return result


def _new_metric_groups() -> dict[str, Any]:
    return {
        "overall": _empty_accumulator(),
        "by_observed_corruption": {
            "observed_changed": _empty_accumulator(),
            "observed_unchanged": _empty_accumulator(),
        },
        "by_training_frequency": {
            name: _empty_accumulator() for name, _lower, _upper in FREQUENCY_BUCKETS
        },
    }


def _finalize_metric_groups(groups: dict[str, Any]) -> dict[str, Any]:
    return {
        "overall": _finalize_accumulator(groups["overall"]),
        "by_observed_corruption": {
            key: _finalize_accumulator(value)
            for key, value in groups["by_observed_corruption"].items()
        },
        "by_training_frequency": {
            key: _finalize_accumulator(value)
            for key, value in groups["by_training_frequency"].items()
        },
    }


def _merge_accumulator(
    destination: dict[str, float | int], source: dict[str, float | int]
) -> None:
    for key in destination:
        destination[key] += source[key]


def _merge_metric_groups(destination: dict[str, Any], source: dict[str, Any]) -> None:
    _merge_accumulator(destination["overall"], source["overall"])
    for group_name in ("by_observed_corruption", "by_training_frequency"):
        for key in destination[group_name]:
            _merge_accumulator(destination[group_name][key], source[group_name][key])


def _empty_endpoint_accumulator() -> dict[str, float | int]:
    return {"denominator_tokens": 0, "endpoint_prior_kl_sum": 0.0}


def _update_endpoint_accumulator(
    accumulator: dict[str, float | int],
    mask: torch.Tensor,
    endpoint_prior_kl: torch.Tensor,
) -> None:
    count = int(mask.sum().item())
    if count == 0:
        return
    accumulator["denominator_tokens"] += count
    accumulator["endpoint_prior_kl_sum"] += float(
        endpoint_prior_kl[mask].double().sum().item()
    )


def _finalize_endpoint_accumulator(
    accumulator: Mapping[str, float | int],
) -> dict[str, float | int | None]:
    count = int(accumulator["denominator_tokens"])
    total = float(accumulator["endpoint_prior_kl_sum"])
    return {
        "denominator_tokens": count,
        "endpoint_prior_kl_sum": total,
        "endpoint_prior_kl_mean": None if count == 0 else total / count,
    }


def _new_endpoint_groups() -> dict[str, Any]:
    return {
        "overall": _empty_endpoint_accumulator(),
        "by_training_frequency": {
            name: _empty_endpoint_accumulator()
            for name, _lower, _upper in FREQUENCY_BUCKETS
        },
    }


def _finalize_endpoint_groups(groups: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "overall": _finalize_endpoint_accumulator(groups["overall"]),
        "by_training_frequency": {
            name: _finalize_endpoint_accumulator(accumulator)
            for name, accumulator in groups["by_training_frequency"].items()
        },
    }


def _pad_rows(rows: Sequence[torch.Tensor], pad_token_id: int) -> torch.Tensor:
    max_length = max(int(row.numel()) for row in rows)
    result = torch.full(
        (len(rows), max_length),
        pad_token_id,
        dtype=torch.long,
        device=rows[0].device,
    )
    for index, row in enumerate(rows):
        result[index, : row.numel()] = row
    return result


def _generator_for(device: torch.device, seed: int) -> torch.Generator:
    generator_device = "cpu" if device.type == "cpu" else str(device)
    return torch.Generator(device=generator_device).manual_seed(seed)


def _ordered_token_rows_sha256(rows: Sequence[torch.Tensor]) -> str:
    """Hash variable-length token rows without padding ambiguity."""

    digest = hashlib.sha256()
    for row in rows:
        value = json.dumps(
            [int(token_id) for token_id in row.detach().cpu().tolist()],
            separators=(",", ":"),
        ).encode("ascii")
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


def _canonical_numeric_sequence_sha256(values: Sequence[int | float]) -> str:
    """Match model.py's platform-independent ordered numeric-sequence hash."""

    canonical = [value.hex() if isinstance(value, float) else value for value in values]
    return hashlib.sha256(
        json.dumps(canonical, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _exact_data_equal(observed: Any, expected: Any) -> bool:
    """Compare provenance records without Python's bool/int coercion."""

    if type(observed) is not type(expected):
        return False
    if isinstance(expected, Mapping):
        return set(observed) == set(expected) and all(
            _exact_data_equal(observed[key], expected[key]) for key in expected
        )
    if isinstance(expected, (list, tuple)):
        return len(observed) == len(expected) and all(
            _exact_data_equal(left, right)
            for left, right in zip(observed, expected, strict=True)
        )
    return bool(observed == expected)


def _runtime_prior_metadata(model: Any) -> dict[str, Any] | None:
    raw_metadata = getattr(model, "udlm_prior_metadata", None)
    if raw_metadata is None:
        return None
    if hasattr(raw_metadata, "to_dict"):
        raw_metadata = raw_metadata.to_dict()
    if not isinstance(raw_metadata, Mapping):
        raise ValueError(
            "model.udlm_prior_metadata must be an immutable mapping record"
        )
    metadata = dict(raw_metadata)
    if set(metadata) != UDLM_PRIOR_METADATA_FIELDS:
        missing = sorted(UDLM_PRIOR_METADATA_FIELDS - set(metadata))
        extra = sorted(set(metadata) - UDLM_PRIOR_METADATA_FIELDS)
        raise ValueError(
            "UDLM prior metadata fields do not match schema version 1; "
            f"missing={missing}, extra={extra}"
        )
    if type(metadata["schema_version"]) is not int or metadata["schema_version"] != 1:
        raise ValueError("UDLM prior metadata must use schema_version 1")
    return metadata


def _configured_prior_variant(model: Any) -> str | None:
    config = getattr(model, "config", None)
    try:
        value = config.training.udlm.prior_variant
    except (AttributeError, KeyError, TypeError):
        return None
    return str(value).lower()


def _validate_sha256(value: Any, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"UDLM prior metadata {field} is not a lowercase SHA-256")


def _validate_empirical_artifact_identity(
    metadata: Mapping[str, Any],
    frequencies: Mapping[str, Any],
    artifact_provenance: Mapping[str, Any] | None,
    active_token_ids: Sequence[int],
    stationary_probs: torch.Tensor,
) -> None:
    if artifact_provenance is None:
        raise ValueError(
            "empirical_frequency evaluation requires byte-level artifact provenance"
        )
    frequency_provenance = artifact_provenance.get("training_frequency")
    if not isinstance(frequency_provenance, Mapping):
        raise ValueError("artifact provenance has no training_frequency record")
    expected = {
        "frequency_artifact_path": EMPIRICAL_FREQUENCY_RELATIVE_PATH.as_posix(),
        "frequency_artifact_sha256": FROZEN_FREQUENCY_SHA256,
        "frequency_artifact_schema_version": frequencies["schema_version"],
        "frequency_example_count": frequencies["example_count"],
        "frequency_content_token_count": frequencies["content_token_count"],
        "frequency_active_token_count": sum(
            frequencies["counts_by_token_id"][token_id] for token_id in active_token_ids
        ),
        "frequency_dataset_repo_id": frequencies["dataset"]["repo_id"],
        "frequency_dataset_revision": frequencies["dataset"]["revision"],
        "frequency_dataset_split": frequencies["dataset"]["split"],
        "frequency_dataset_selection": frequencies["dataset"].get("selection"),
        "frequency_ordered_text_sha256": frequencies["dataset"].get(
            "ordered_safe_text_sha256"
        ),
        "frequency_implementation_git_sha": frequencies.get("git_sha"),
    }
    for field, expected_value in expected.items():
        if not _exact_data_equal(metadata[field], expected_value):
            raise ValueError(
                f"empirical prior metadata {field} disagrees with the pinned artifact"
            )
    if frequency_provenance.get("sha256") != FROZEN_FREQUENCY_SHA256:
        raise ValueError(
            "empirical prior metadata is not bound to the evaluated frequency bytes"
        )
    if expected["frequency_ordered_text_sha256"] != (
        EMPIRICAL_FREQUENCY_ORDERED_TEXT_SHA256
    ):
        raise ValueError("frequency artifact ordered training-text identity is invalid")

    mixture = metadata["uniform_mixture_weight"]
    if isinstance(mixture, bool) or not isinstance(mixture, (int, float)):
        raise ValueError("empirical prior uniform_mixture_weight must be real")
    mixture = float(mixture)
    if not math.isfinite(mixture) or not 0.0 < mixture < 1.0:
        raise ValueError("empirical prior uniform_mixture_weight must lie in (0, 1)")
    active_count = expected["frequency_active_token_count"]
    if type(active_count) is not int or active_count <= 0:
        raise ValueError("frequency artifact has no count mass on the active alphabet")
    support_size = len(active_token_ids)
    expected_probs = torch.tensor(
        [
            (1.0 - mixture)
            * (frequencies["counts_by_token_id"][token_id] / active_count)
            + mixture / support_size
            for token_id in active_token_ids
        ],
        dtype=torch.float64,
    )
    expected_probs /= expected_probs.sum()
    if not torch.equal(stationary_probs.detach().cpu(), expected_probs):
        raise ValueError(
            "categorical stationary_probs disagree with the pinned smoothed "
            "empirical prior"
        )


def _validate_udlm_backend(
    model: Any,
    *,
    panel: Mapping[str, Any],
    frequencies: Mapping[str, Any],
    checkpoint_prior_metadata: Mapping[str, Any] | None,
    artifact_provenance: Mapping[str, Any] | None,
) -> tuple[ContinuousUniformDiffusion, dict[str, Any]]:
    """Validate exact process type, buffers, prior law, and immutable identity."""

    if str(getattr(model, "diffusion_type", "")).lower() != "udlm":
        raise ValueError("denoising panel evaluation requires a UDLM model")
    if not hasattr(model, "mdlm"):
        raise ValueError("UDLM model has no diffusion process at model.mdlm")
    process = model.mdlm
    if type(process) not in {
        ContinuousUniformDiffusion,
        ContinuousCategoricalDiffusion,
    }:
        observed = f"{type(process).__module__}.{type(process).__qualname__}"
        raise ValueError(
            "evaluator requires an exact supported UDLM backend, not a subclass; "
            f"observed {observed}"
        )

    active_tensor = process.diffusion_token_ids.detach().cpu()
    mapping_tensor = process.token_to_diffusion_index.detach().cpu()
    if active_tensor.dtype != torch.long or active_tensor.ndim != 1:
        raise ValueError(
            "UDLM diffusion_token_ids must be a one-dimensional long buffer"
        )
    active_token_ids = [int(value) for value in active_tensor.tolist()]
    if active_token_ids != sorted(set(active_token_ids)):
        raise ValueError("UDLM diffusion_token_ids must be unique and strictly ordered")
    num_classes = int(process.num_classes)
    if not active_token_ids or not all(
        0 <= token_id < num_classes for token_id in active_token_ids
    ):
        raise ValueError("UDLM diffusion_token_ids lie outside the model vocabulary")
    if int(process.diffusion_vocab_size) != len(active_token_ids):
        raise ValueError(
            "UDLM diffusion_vocab_size disagrees with its active-token buffer"
        )
    expected_mapping = torch.full((num_classes,), -1, dtype=torch.long)
    expected_mapping[active_tensor] = torch.arange(len(active_token_ids))
    if mapping_tensor.dtype != torch.long or not torch.equal(
        mapping_tensor, expected_mapping
    ):
        raise ValueError("UDLM compact-token mapping is inconsistent")
    active_set = set(active_token_ids)
    excluded_token_ids = [
        token_id for token_id in range(num_classes) if token_id not in active_set
    ]

    categorical = type(process) is ContinuousCategoricalDiffusion
    if categorical:
        stationary_probs = process.stationary_probs.detach().cpu()
        if (
            stationary_probs.dtype != torch.float64
            or stationary_probs.shape != (len(active_token_ids),)
            or not torch.isfinite(stationary_probs).all()
            or torch.any(stationary_probs <= 0)
            or not torch.isclose(
                stationary_probs.sum(),
                torch.tensor(1.0, dtype=torch.float64),
                rtol=1e-12,
                atol=1e-12,
            )
        ):
            raise ValueError(
                "categorical stationary_probs must be a normalized float64 "
                "full-support buffer"
            )
    else:
        stationary_probs = torch.full(
            (len(active_token_ids),),
            1.0 / len(active_token_ids),
            dtype=torch.float64,
        )
        stationary_probs /= stationary_probs.sum()

    metadata = _runtime_prior_metadata(model)
    if metadata is None:
        if categorical:
            raise ValueError(
                "categorical UDLM model is missing immutable prior metadata"
            )
        if checkpoint_prior_metadata is not None:
            raise ValueError(
                "release_uniform checkpoint must not declare categorical prior metadata"
            )
        variant = "release_uniform"
        metadata_status = "legacy_or_injected_release_uniform_without_runtime_record"
    else:
        variant = metadata.get("variant")
        if variant not in UDLM_PRIOR_VARIANT_IDENTITIES:
            raise ValueError(f"unsupported UDLM prior variant {variant!r}")
        expected_type = (
            ContinuousUniformDiffusion
            if variant == "release_uniform"
            else ContinuousCategoricalDiffusion
        )
        if type(process) is not expected_type:
            raise ValueError("UDLM prior variant disagrees with the exact process type")
        identity = UDLM_PRIOR_VARIANT_IDENTITIES[variant]
        for field, expected_value in identity.items():
            if not _exact_data_equal(metadata[field], expected_value):
                raise ValueError(
                    f"UDLM prior metadata {field} is invalid for variant {variant}"
                )
        configured_variant = _configured_prior_variant(model)
        if configured_variant is not None and configured_variant != variant:
            raise ValueError(
                "checkpoint config and runtime UDLM prior variants disagree"
            )
        exact_fields = {
            "full_vocab_size": num_classes,
            "active_vocab_size": len(active_token_ids),
            "excluded_token_ids": excluded_token_ids,
            "sampling_eps": float(process.sampling_eps),
            "noise_eps": float(process.noise_eps),
            "antithetic_sampling": bool(process.antithetic_sampling),
            "active_token_ids_sha256": _canonical_numeric_sequence_sha256(
                active_token_ids
            ),
            "stationary_probs_sha256": _canonical_numeric_sequence_sha256(
                [float(value) for value in stationary_probs.tolist()]
            ),
        }
        for field, expected_value in exact_fields.items():
            if not _exact_data_equal(metadata[field], expected_value):
                raise ValueError(
                    f"UDLM prior metadata {field} disagrees with the live process"
                )
        _validate_sha256(metadata["active_token_ids_sha256"], "active_token_ids_sha256")
        _validate_sha256(metadata["stationary_probs_sha256"], "stationary_probs_sha256")

        tokenizer = panel["tokenizer"]
        for field, expected_value in {
            "tokenizer_repo_id": tokenizer["repo_id"],
            "tokenizer_revision": tokenizer["revision"],
            "tokenizer_json_sha256": tokenizer["tokenizer_json_sha256"],
        }.items():
            if not _exact_data_equal(metadata[field], expected_value):
                raise ValueError(
                    f"UDLM prior metadata {field} disagrees with the panel tokenizer"
                )

        if categorical:
            if not isinstance(checkpoint_prior_metadata, Mapping):
                raise ValueError(
                    "categorical UDLM checkpoint is missing immutable prior metadata"
                )
            if not _exact_data_equal(dict(checkpoint_prior_metadata), metadata):
                raise ValueError(
                    "categorical checkpoint prior metadata disagrees with the runtime "
                    "process"
                )
            metadata_status = "required_checkpoint_record_matches_runtime_exactly"
        else:
            if checkpoint_prior_metadata is not None:
                raise ValueError(
                    "release_uniform checkpoint must not declare categorical prior metadata"
                )
            metadata_status = "release_checkpoint_record_correctly_absent"

        frequency_fields = {
            field
            for field in UDLM_PRIOR_METADATA_FIELDS
            if field.startswith("frequency_")
        }
        if variant == "empirical_frequency":
            _validate_empirical_artifact_identity(
                metadata,
                frequencies,
                artifact_provenance,
                active_token_ids,
                stationary_probs,
            )
        else:
            if metadata["uniform_mixture_weight"] is not None or any(
                metadata[field] is not None for field in frequency_fields
            ):
                raise ValueError(
                    f"{variant} metadata must not claim an empirical frequency artifact"
                )
            uniform_probs = torch.full_like(
                stationary_probs, 1.0 / len(active_token_ids)
            )
            uniform_probs /= uniform_probs.sum()
            if not torch.equal(stationary_probs, uniform_probs):
                raise ValueError(f"{variant} stationary prior must be exactly uniform")

    return process, {
        "variant": variant,
        "comparison_role": UDLM_PRIOR_VARIANT_IDENTITIES[variant]["comparison_role"],
        "checkpoint_metadata_validation": metadata_status,
        "runtime_metadata": metadata,
        "runtime_metadata_sha256": (
            None if metadata is None else _canonical_sha256(metadata)
        ),
        "active_token_ids": active_token_ids,
        "excluded_token_ids": excluded_token_ids,
        "stationary_probs": stationary_probs,
    }


def _process_metadata(
    process: Any,
    time_bins: Sequence[float],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    device = process.diffusion_token_ids.device
    time_tensor = torch.tensor(time_bins, device=device, dtype=torch.float32)
    with torch.no_grad():
        alpha = process.alpha(time_tensor).detach().cpu().double().tolist()
        sigma = process.sigma(time_tensor).detach().cpu().double().tolist()
    diffusion_token_ids = [
        int(value) for value in process.diffusion_token_ids.detach().cpu().tolist()
    ]
    variant = str(identity["variant"])
    variant_identity = UDLM_PRIOR_VARIANT_IDENTITIES[variant]
    categorical = type(process) is ContinuousCategoricalDiffusion
    stationary_probs = identity["stationary_probs"]
    probability_values = [float(value) for value in stationary_probs.tolist()]
    prior_identity = {
        "family": (
            "categorical_stationary_over_diffusion_token_ids"
            if variant == "empirical_frequency"
            else "uniform_over_diffusion_token_ids"
        ),
        "support_size": len(diffusion_token_ids),
        "probability_per_supported_token": (
            1.0 / len(diffusion_token_ids) if variant != "empirical_frequency" else None
        ),
        "minimum_probability": min(probability_values),
        "maximum_probability": max(probability_values),
        "entropy_nats": -sum(
            probability * math.log(probability) for probability in probability_values
        ),
        "ordered_token_ids_sha256": _canonical_numeric_sequence_sha256(
            diffusion_token_ids
        ),
        "ordered_probabilities_sha256": _canonical_numeric_sequence_sha256(
            probability_values
        ),
    }
    prior_identity["identity_sha256"] = _canonical_sha256(prior_identity)
    return {
        "backend": f"{type(process).__module__}.{type(process).__qualname__}",
        "backend_exact_type_required": True,
        "prior_variant": variant,
        **variant_identity,
        "comparison_design": (
            "schedule_uniform is the matched schedule/process-family control for "
            "empirical_frequency. Fixed-grid differences are descriptive and do "
            "not by themselves identify a causal prior effect."
        ),
        "valid_single_factor_prior_control_for": (
            "empirical_frequency" if variant == "schedule_uniform" else None
        ),
        "loss_values_cross_backend_comparable": False,
        "training_frequency_artifact_usage": (
            ["stationary_prior_construction", "metric_stratification"]
            if variant == "empirical_frequency"
            else ["metric_stratification_only"]
        ),
        "checkpoint_metadata_validation": identity["checkpoint_metadata_validation"],
        "runtime_prior_metadata": identity["runtime_metadata"],
        "runtime_prior_metadata_sha256": identity["runtime_metadata_sha256"],
        "prior": prior_identity,
        "num_classes": int(process.num_classes),
        "diffusion_vocab_size": int(process.diffusion_vocab_size),
        "excluded_token_ids": identity["excluded_token_ids"],
        "sampling_eps": float(process.sampling_eps),
        "noise_eps": float(process.noise_eps),
        "antithetic_sampling": bool(process.antithetic_sampling),
        "corruption_schedule": {
            "definition": "alpha_corrupt(t) = 1 - (1 - noise_eps) * t",
            "time_bin_alpha": alpha,
            "time_bin_sigma": sigma,
        },
        "production_loss_schedule": {
            "definition": (
                "ContinuousCategoricalDiffusion.loss_per_token uses the exact "
                "rank-one categorical model-dependent integrand and "
                "beta(t)=(1-noise_eps)/alpha_corrupt(t)"
                if categorical
                else "ContinuousUniformDiffusion.loss_per_token uses the released "
                "idealized alpha_loss(t)=1-t and coefficient 1/(N*(1-t))"
            ),
            "qualification": (
                "The parameter-independent endpoint-prior KL is excluded here and "
                "reported separately; fixed-grid values are not a NELBO."
                if categorical
                else "This intentionally differs from the residual-noise corruption "
                "endpoint when noise_eps > 0."
            ),
            "time_bin_beta": (
                process.beta(time_tensor).detach().cpu().double().tolist()
                if categorical
                else None
            ),
        },
    }


def evaluate_denoising_panel(
    model: Any,
    panel: dict,
    frequencies: dict,
    *,
    time_bins: Sequence[float] = DEFAULT_TIME_BINS,
    seed: int = 1,
    batch_size: int = 4,
    max_rows: int | None = None,
    device: str | torch.device = "cpu",
    checkpoint_prior_metadata: Mapping[str, Any] | None = None,
    artifact_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run deterministic denoising diagnostics on an injected UDLM model.

    Keeping the model injectable lets focused tests and CPU smoke runs use a
    tiny denoiser without loading the 1.4 GB production checkpoint.
    """

    validate_evaluation_inputs(panel, frequencies)
    time_bins = validate_time_bins(time_bins)
    if not 0 <= seed < 2**63:
        raise ValueError("seed must lie in [0, 2**63)")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    available_rows = len(panel["rows"])
    if max_rows is None:
        max_rows = available_rows
    if not 1 <= max_rows <= available_rows:
        raise ValueError(f"max_rows must lie in [1, {available_rows}]")
    process, process_identity = _validate_udlm_backend(
        model,
        panel=panel,
        frequencies=frequencies,
        checkpoint_prior_metadata=checkpoint_prior_metadata,
        artifact_provenance=artifact_provenance,
    )
    initial_process_identity_sha256 = _canonical_sha256(
        {
            "variant": process_identity["variant"],
            "active_token_ids": process_identity["active_token_ids"],
            "excluded_token_ids": process_identity["excluded_token_ids"],
            "stationary_probs_sha256": _canonical_numeric_sequence_sha256(
                [
                    float(value)
                    for value in process_identity["stationary_probs"].tolist()
                ]
            ),
            "runtime_metadata_sha256": process_identity["runtime_metadata_sha256"],
        }
    )
    parsed_device = torch.device(device)
    if parsed_device.type != "cpu":
        raise ValueError(
            "deterministic denoising-panel evaluation is CPU-only; GPU inference "
            "would weaken the recorded numerical reproducibility contract"
        )
    model.to(parsed_device)
    process.to_device(parsed_device)
    model.eval()

    vocab_size = int(panel["tokenizer"]["base_vocab_size"])
    if int(process.num_classes) != vocab_size:
        raise ValueError(
            f"model/panel vocabulary mismatch: {process.num_classes} != {vocab_size}"
        )
    model_tokenizer = getattr(model, "tokenizer", None)
    if model_tokenizer is not None:
        tokenizer_checks = {
            "vocab_size": (
                int(model_tokenizer.vocab_size),
                vocab_size,
            ),
            "special_token_ids": (
                sorted(int(value) for value in model_tokenizer.all_special_ids),
                panel["tokenizer"]["special_token_ids"],
            ),
            "bos_token_id": (
                int(model_tokenizer.bos_token_id),
                panel["tokenizer"]["bos_token_id"],
            ),
            "eos_token_id": (
                int(model_tokenizer.eos_token_id),
                panel["tokenizer"]["eos_token_id"],
            ),
            "pad_token_id": (
                int(model_tokenizer.pad_token_id),
                panel["tokenizer"]["pad_token_id"],
            ),
        }
        for field, (observed, expected) in tokenizer_checks.items():
            if observed != expected:
                raise ValueError(
                    f"model/panel tokenizer mismatch for {field}: "
                    f"{observed} != {expected}"
                )
    max_position_embeddings = getattr(
        getattr(getattr(model, "config", None), "model", None),
        "max_position_embeddings",
        None,
    )
    selected_rows = panel["rows"][:max_rows]
    if max_position_embeddings is not None and any(
        len(row["input_ids"]) > int(max_position_embeddings) for row in selected_rows
    ):
        raise ValueError("panel contains a row longer than the model position limit")

    special_ids = set(int(value) for value in panel["tokenizer"]["special_token_ids"])
    pad_token_id = int(panel["tokenizer"]["pad_token_id"])
    training_counts = torch.tensor(
        frequencies["counts_by_token_id"], dtype=torch.long, device=parsed_device
    )
    pooled_groups = _new_metric_groups()
    endpoint_groups = (
        _new_endpoint_groups()
        if type(process) is ContinuousCategoricalDiffusion
        else None
    )
    bin_results = []

    with torch.inference_mode():
        for time_index, time_value in enumerate(time_bins):
            bin_groups = _new_metric_groups()
            row_seeds = [
                corruption_seed(seed, time_value, int(row["source_index"]))
                for row in selected_rows
            ]
            # Corruption is always materialized on CPU, one row at a time.
            # Consequently its exact token grid is independent of evaluator
            # device and batch partition (subject only to the pinned process
            # implementation and PyTorch CPU RNG behavior).
            process.to_device(torch.device("cpu"))
            noisy_rows_cpu = []
            for row, row_seed in zip(selected_rows, row_seeds):
                clean_cpu = torch.tensor(row["input_ids"], dtype=torch.long)
                mutable_cpu = torch.tensor(
                    [token_id not in special_ids for token_id in row["input_ids"]],
                    dtype=torch.bool,
                )
                row_time_cpu = torch.tensor([time_value], dtype=torch.float32)
                noisy_rows_cpu.append(
                    process.forward_process(
                        clean_cpu.unsqueeze(0),
                        row_time_cpu,
                        mutable_mask=mutable_cpu.unsqueeze(0),
                        generator=_generator_for(torch.device("cpu"), row_seed),
                    ).squeeze(0)
                )
            corrupted_token_ids_sha256 = _ordered_token_rows_sha256(noisy_rows_cpu)
            process.to_device(parsed_device)

            for batch_start in range(0, max_rows, batch_size):
                batch_rows = selected_rows[batch_start : batch_start + batch_size]
                clean_rows = []
                noisy_rows = []
                for row_offset, row in enumerate(batch_rows):
                    clean = torch.tensor(
                        row["input_ids"], dtype=torch.long, device=parsed_device
                    )
                    clean_rows.append(clean)
                    noisy_rows.append(
                        noisy_rows_cpu[batch_start + row_offset].to(parsed_device)
                    )

                x0 = _pad_rows(clean_rows, pad_token_id)
                xt = _pad_rows(noisy_rows, pad_token_id)
                attention_mask = x0 != pad_token_id
                content_mask = attention_mask.clone()
                for special_id in special_ids:
                    content_mask &= x0 != special_id
                t = torch.full(
                    (x0.shape[0],),
                    time_value,
                    dtype=torch.float32,
                    device=parsed_device,
                )
                logits = model(xt, attention_mask, t=t)
                if logits.shape != (*x0.shape, vocab_size):
                    raise RuntimeError(
                        "model returned logits with shape "
                        f"{tuple(logits.shape)}; expected {(*x0.shape, vocab_size)}"
                    )

                production_loss = process.loss_per_token(
                    logits, x0, xt, t, mask=content_mask
                )
                clean_log_probs = process.clean_log_probs(logits)
                clean_compact = process.token_to_diffusion_index[x0]
                if torch.any(content_mask & (clean_compact < 0)):
                    invalid = torch.unique(
                        x0[content_mask & (clean_compact < 0)]
                    ).tolist()
                    raise RuntimeError(
                        f"content tokens are outside the diffusion alphabet: {invalid}"
                    )
                clean_nll = -torch.gather(
                    clean_log_probs,
                    -1,
                    clean_compact.clamp_min(0).unsqueeze(-1),
                ).squeeze(-1)
                predicted_compact = clean_log_probs.argmax(dim=-1)
                predicted_ids = process.diffusion_token_ids[predicted_compact]
                top1_correct = predicted_ids == x0
                if not (
                    torch.isfinite(production_loss[content_mask]).all()
                    and torch.isfinite(clean_nll[content_mask]).all()
                ):
                    raise RuntimeError("non-finite denoising metric on a content token")

                observed_changed = content_mask & (xt != x0)
                observed_unchanged = content_mask & (xt == x0)
                original_training_counts = training_counts[x0]
                bucket_masks = frequency_bucket_masks(original_training_counts)

                if endpoint_groups is not None and time_index == 0:
                    endpoint_prior_kl = process.endpoint_prior_kl(x0, mask=content_mask)
                    if endpoint_prior_kl.shape != x0.shape:
                        raise RuntimeError(
                            "endpoint_prior_kl returned an invalid tensor shape"
                        )
                    if (
                        not torch.isfinite(endpoint_prior_kl[content_mask]).all()
                        or torch.any(endpoint_prior_kl[content_mask] < -1e-12)
                        or torch.any(endpoint_prior_kl[~content_mask] != 0)
                    ):
                        raise RuntimeError(
                            "invalid categorical endpoint-prior KL on panel tokens"
                        )
                    _update_endpoint_accumulator(
                        endpoint_groups["overall"],
                        content_mask,
                        endpoint_prior_kl,
                    )
                    for name, bucket_mask in bucket_masks.items():
                        _update_endpoint_accumulator(
                            endpoint_groups["by_training_frequency"][name],
                            content_mask & bucket_mask,
                            endpoint_prior_kl,
                        )

                _update_accumulator(
                    bin_groups["overall"],
                    content_mask,
                    production_loss,
                    clean_nll,
                    top1_correct,
                )
                for name, mask in (
                    ("observed_changed", observed_changed),
                    ("observed_unchanged", observed_unchanged),
                ):
                    _update_accumulator(
                        bin_groups["by_observed_corruption"][name],
                        mask,
                        production_loss,
                        clean_nll,
                        top1_correct,
                    )
                for name, bucket_mask in bucket_masks.items():
                    _update_accumulator(
                        bin_groups["by_training_frequency"][name],
                        content_mask & bucket_mask,
                        production_loss,
                        clean_nll,
                        top1_correct,
                    )

            changed_count = int(
                bin_groups["by_observed_corruption"]["observed_changed"][
                    "denominator_tokens"
                ]
            )
            total_count = int(bin_groups["overall"]["denominator_tokens"])
            _merge_metric_groups(pooled_groups, bin_groups)
            bin_results.append(
                {
                    "time": time_value,
                    "row_corruption_seeds_sha256": _canonical_sha256(row_seeds),
                    "corrupted_token_ids_sha256": corrupted_token_ids_sha256,
                    "observed_changed_rate": changed_count / total_count,
                    "metrics": _finalize_metric_groups(bin_groups),
                }
            )

    expected_content_tokens = sum(int(row["content_length"]) for row in selected_rows)
    for result in bin_results:
        observed = result["metrics"]["overall"]["denominator_tokens"]
        if observed != expected_content_tokens:
            raise RuntimeError(
                f"content denominator invariant failed: {observed} != "
                f"{expected_content_tokens}"
            )
    if endpoint_groups is not None and (
        endpoint_groups["overall"]["denominator_tokens"] != expected_content_tokens
    ):
        raise RuntimeError("endpoint-prior KL denominator invariant failed")

    _final_process, final_process_identity = _validate_udlm_backend(
        model,
        panel=panel,
        frequencies=frequencies,
        checkpoint_prior_metadata=checkpoint_prior_metadata,
        artifact_provenance=artifact_provenance,
    )
    final_process_identity_sha256 = _canonical_sha256(
        {
            "variant": final_process_identity["variant"],
            "active_token_ids": final_process_identity["active_token_ids"],
            "excluded_token_ids": final_process_identity["excluded_token_ids"],
            "stationary_probs_sha256": _canonical_numeric_sequence_sha256(
                [
                    float(value)
                    for value in final_process_identity["stationary_probs"].tolist()
                ]
            ),
            "runtime_metadata_sha256": final_process_identity[
                "runtime_metadata_sha256"
            ],
        }
    )
    if final_process_identity_sha256 != initial_process_identity_sha256:
        raise RuntimeError("UDLM process identity changed during panel evaluation")

    endpoint_result = {
        "applicable": endpoint_groups is not None,
        "mathematically_defined_for_residual_forward": True,
        "reported_by_evaluator": endpoint_groups is not None,
        "parameter_independent": True,
        "included_in_production_loss": False,
        "definition": (
            "KL(q(z_1 | x0) || stationary_prior) induced by residual "
            "alpha_corrupt(1)=noise_eps, evaluated once per clean content token"
            if endpoint_groups is not None
            else (
                "mathematically nonzero for the residual-clean released forward "
                "kernel, but not exposed or included by the faithful released "
                "ContinuousUniformDiffusion objective"
            )
        ),
        "qualification": (
            "Adding this endpoint value to fixed-grid instantaneous losses still "
            "does not produce a NELBO because no continuous-time integral is "
            "estimated."
        ),
        "metrics": (
            _finalize_endpoint_groups(endpoint_groups)
            if endpoint_groups is not None
            else None
        ),
    }

    return {
        "rows_evaluated": max_rows,
        "row_selection": f"first {max_rows} rows of the frozen ordered panel",
        "content_tokens_per_time_bin": expected_content_tokens,
        "time_bins": list(time_bins),
        "seed": seed,
        "batch_size": batch_size,
        "device": str(parsed_device),
        "estimator_scope": (
            "one deterministic corruption per selected row and fixed time; "
            "descriptive grid, not an integrated or unbiased NELBO estimate; "
            "cross-backend differences are not causal estimates"
        ),
        "seed_protocol": {
            "definition": (
                "SHA256('genmol-udlm-fixed-panel-v1|base_seed|"
                "float.hex(time)|source_index')[:8] modulo 2**63"
            ),
            "per_row": True,
            "corruption_device": "cpu",
            "corruption_batch_partition_invariant": True,
            "qualification": (
                "Exact corruptions are committed by corrupted_token_ids_sha256. "
                "Model outputs and floating-point metric sums are not claimed to "
                "be bit-exact across inference batch sizes, kernels, or versions."
            ),
        },
        "corruption_grid_sha256": _canonical_sha256(
            [
                {
                    "time": result["time"],
                    "row_corruption_seeds_sha256": result[
                        "row_corruption_seeds_sha256"
                    ],
                    "corrupted_token_ids_sha256": result["corrupted_token_ids_sha256"],
                }
                for result in bin_results
            ]
        ),
        "process_identity_sha256": initial_process_identity_sha256,
        "process_identity_postcheck": "unchanged_after_evaluation",
        "process": _process_metadata(process, time_bins, process_identity),
        "endpoint_prior_kl": endpoint_result,
        "metrics_by_time": bin_results,
        "pooled_token_time_metrics": _finalize_metric_groups(pooled_groups),
    }


def _stat_identity(stat_result: os.stat_result) -> dict[str, int]:
    return {
        "device": int(stat_result.st_dev),
        "inode": int(stat_result.st_ino),
        "size_bytes": int(stat_result.st_size),
        "mtime_ns": int(stat_result.st_mtime_ns),
    }


def _stable_file_snapshot(path: Path) -> dict[str, Any]:
    """Hash a file and reject replacement or mutation during that hash."""

    before = _stat_identity(path.stat())
    digest = sha256_file(path)
    after = _stat_identity(path.stat())
    if after != before:
        raise RuntimeError(f"file changed while it was being hashed: {path}")
    return {**after, "sha256": digest}


def load_verified_checkpoint(path: Path) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Load a checkpoint once, guarded by identical pre/post byte snapshots."""

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    before = _stable_file_snapshot(path)
    # Deliberately do not mmap: tensors must be a resident snapshot rather than
    # lazy views whose backing file could mutate after the post-load hash.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    after = _stable_file_snapshot(path)
    if after != before:
        raise RuntimeError(f"checkpoint changed while it was being loaded: {path}")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint root must be a mapping")
    return checkpoint, before


def _plain_config(config: Any) -> Any:
    """Convert an OmegaConf or ordinary checkpoint config to canonical JSON data."""

    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(config):
            config = OmegaConf.to_container(config, resolve=False, enum_to_str=True)
    except ImportError:  # pragma: no cover - GenMol itself requires OmegaConf
        pass
    if isinstance(config, Mapping):
        return {str(key): _plain_config(value) for key, value in config.items()}
    if isinstance(config, Sequence) and not isinstance(config, (str, bytes, bytearray)):
        return [_plain_config(value) for value in config]
    if isinstance(config, Path):
        return str(config)
    if config is None or isinstance(config, (str, int, float, bool)):
        return config
    raise ValueError(
        "checkpoint config contains a value that cannot be canonically recorded: "
        f"{type(config).__module__}.{type(config).__qualname__}"
    )


def _checkpoint_metadata_from_payload(
    path: Path,
    checkpoint: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    hyper_parameters = checkpoint.get("hyper_parameters", {})
    if not isinstance(hyper_parameters, Mapping):
        raise ValueError("checkpoint hyper_parameters must be a mapping")
    config = _plain_config(hyper_parameters.get("config", {}))
    if not isinstance(config, dict):
        raise ValueError("checkpoint hyper_parameters.config must be a mapping")
    training = config.get("training", {})
    if not isinstance(training, dict):
        raise ValueError("checkpoint training config must be a mapping")
    diffusion_type = str(training.get("diffusion", "mdlm")).lower()
    initialization_checkpoint = training.get("init_from_mdlm_checkpoint")
    if initialization_checkpoint is not None and not isinstance(
        initialization_checkpoint, str
    ):
        initialization_checkpoint = str(initialization_checkpoint)
    checkpoint_prior_metadata = checkpoint.get(UDLM_PRIOR_CHECKPOINT_KEY)
    if checkpoint_prior_metadata is not None and not isinstance(
        checkpoint_prior_metadata, Mapping
    ):
        raise ValueError("checkpoint udlm_prior_metadata must be a mapping")
    plain_prior_metadata = (
        None
        if checkpoint_prior_metadata is None
        else _plain_config(checkpoint_prior_metadata)
    )
    return {
        "path": str(path.resolve()),
        "sha256": snapshot["sha256"],
        "size_bytes": int(snapshot["size_bytes"]),
        "mtime_utc": datetime.fromtimestamp(
            int(snapshot["mtime_ns"]) / 1e9, timezone.utc
        ).isoformat(),
        "global_step": (
            None
            if checkpoint.get("global_step") is None
            else int(checkpoint["global_step"])
        ),
        "epoch": None if checkpoint.get("epoch") is None else int(checkpoint["epoch"]),
        "diffusion_type": diffusion_type,
        "config_sha256": _canonical_sha256(config),
        "udlm_prior_metadata_declared": plain_prior_metadata is not None,
        "udlm_prior_metadata_sha256": (
            None
            if plain_prior_metadata is None
            else _canonical_sha256(plain_prior_metadata)
        ),
        "udlm_prior_metadata": plain_prior_metadata,
        "training_initialization_declaration": {
            "init_from_mdlm_checkpoint": initialization_checkpoint,
            "init_from_mdlm_ema": bool(training.get("init_from_mdlm_ema", True)),
            "qualification": (
                "This is the checkpoint's saved configuration declaration. It does "
                "not by itself distinguish scratch, one-time warm start, or a later "
                "resume, and is not proof that the named source bytes were used."
            ),
        },
    }


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    """Load once and describe the exact checkpoint bytes (public helper)."""

    checkpoint, snapshot = load_verified_checkpoint(path)
    return _checkpoint_metadata_from_payload(path, checkpoint, snapshot)


def _git_output(arguments: Sequence[str]) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def source_provenance(
    *,
    capture_phase: str = "unspecified",
    validate_loaded_modules: bool = True,
) -> dict[str, Any]:
    hashes = {}
    for relative_path in SOURCE_INPUTS:
        absolute_path = REPOSITORY_ROOT / relative_path
        if not absolute_path.is_file():
            raise FileNotFoundError(absolute_path)
        hashes[str(relative_path)] = _stable_file_snapshot(absolute_path)["sha256"]
    loaded_modules = {
        "evaluator": Path(__file__).resolve(),
        "diffusion": Path(inspect.getfile(ContinuousUniformDiffusion)).resolve(),
        "validation_panel_materializer": Path(
            inspect.getfile(validate_panel)
        ).resolve(),
    }
    expected_loaded_modules = {
        "evaluator": (REPOSITORY_ROOT / SOURCE_INPUTS[0]).resolve(),
        "diffusion": (REPOSITORY_ROOT / "src/genmol/diffusion.py").resolve(),
        "validation_panel_materializer": (
            REPOSITORY_ROOT / "scripts/udlm/materialize_validation_panel.py"
        ).resolve(),
    }
    if validate_loaded_modules and loaded_modules != expected_loaded_modules:
        raise RuntimeError(
            "loaded evaluator modules do not originate from the isolated worktree"
        )
    status = _git_output(["status", "--porcelain=v1", "--untracked-files=normal"])
    if status is None:
        worktree_state = "unknown"
        git_dirty: bool | None = None
    elif status:
        worktree_state = "dirty"
        git_dirty = True
    else:
        worktree_state = "clean"
        git_dirty = False
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "capture_phase": capture_phase,
        "repository_root": str(REPOSITORY_ROOT),
        "git_commit": _git_output(["rev-parse", "HEAD"]),
        "git_branch": _git_output(["branch", "--show-current"]),
        "git_worktree_state": worktree_state,
        "git_dirty": git_dirty,
        "git_status_sha256": None
        if status is None
        else hashlib.sha256(status.encode()).hexdigest(),
        "loaded_module_paths": {
            name: str(path) for name, path in loaded_modules.items()
        },
        "files_sha256": hashes,
    }


def verify_source_provenance(source: Mapping[str, Any]) -> None:
    """Reject a report if an implementation input changed during evaluation."""

    expected = source.get("files_sha256")
    if not isinstance(expected, Mapping):
        raise ValueError("source provenance has no files_sha256 mapping")
    observed = {}
    for relative_path in SOURCE_INPUTS:
        absolute_path = REPOSITORY_ROOT / relative_path
        if not absolute_path.is_file():
            raise RuntimeError(
                f"source input disappeared during evaluation: {absolute_path}"
            )
        observed[str(relative_path)] = _stable_file_snapshot(absolute_path)["sha256"]
    if observed != dict(expected):
        changed = sorted(
            path
            for path in set(observed) | set(expected)
            if observed.get(path) != expected.get(path)
        )
        raise RuntimeError(
            "implementation inputs changed during evaluation: " + ", ".join(changed)
        )


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_provenance() -> dict[str, Any]:
    import genmol

    module_path = Path(inspect.getfile(genmol)).resolve()
    if not module_path.is_relative_to(REPOSITORY_ROOT):
        raise RuntimeError(
            f"genmol resolved outside the isolated worktree: {module_path}"
        )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_build_cuda": torch.version.cuda,
        "packages": {
            name: _package_version(name)
            for name in (
                "lightning",
                "transformers",
                "tokenizers",
                "omegaconf",
                "hydra-core",
                "bionemo-moco",
            )
        },
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "evaluation_device_policy": (
            "CLI is CPU-only until a dynamic idle-GPU launcher records a fresh "
            "physical-to-logical mapping."
        ),
        "genmol_module": str(module_path),
    }


def _backbone_parameter_manifest(
    model: Any, checkpoint_state: Mapping[str, Any]
) -> list[tuple[str, torch.nn.Parameter, torch.Tensor]]:
    """Resolve every trainable backbone parameter by its exact state key."""

    manifest = []
    for name, parameter in model.backbone.named_parameters():
        if not parameter.requires_grad:
            continue
        checkpoint_name = f"backbone.{name}"
        raw_tensor = checkpoint_state.get(checkpoint_name)
        if not isinstance(raw_tensor, torch.Tensor):
            raise ValueError(
                f"checkpoint has no tensor for trainable parameter {checkpoint_name}"
            )
        if raw_tensor.shape != parameter.shape:
            raise ValueError(
                f"raw checkpoint shape mismatch for {checkpoint_name}: "
                f"{tuple(raw_tensor.shape)} != {tuple(parameter.shape)}"
            )
        if raw_tensor.dtype != parameter.dtype:
            raise ValueError(
                f"raw checkpoint dtype mismatch for {checkpoint_name}: "
                f"{raw_tensor.dtype} != {parameter.dtype}"
            )
        manifest.append((checkpoint_name, parameter, raw_tensor))
    if not manifest:
        raise ValueError("model backbone has no trainable parameters")
    return manifest


def apply_ema_weights(
    model: Any,
    checkpoint_state: Mapping[str, Any],
    ema_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the complete positional EMA mapping, then copy it atomically.

    The released EMA format contains an ordered tensor list rather than names.
    We therefore first bind every list position to the exact ordered backbone
    parameter name, and validate all names, counts, shapes, and dtypes before
    changing any parameter.  A malformed tail can never produce hybrid weights.
    """

    manifest = _backbone_parameter_manifest(model, checkpoint_state)
    shadows = ema_state.get("shadow_params")
    if not isinstance(shadows, Sequence):
        raise ValueError("checkpoint EMA shadow_params must be a sequence")
    if len(shadows) != len(manifest):
        raise ValueError(
            "EMA parameter count does not match the trainable backbone: "
            f"{len(shadows)} != {len(manifest)}"
        )

    errors = []
    for index, (shadow, (name, parameter, raw_tensor)) in enumerate(
        zip(shadows, manifest, strict=True)
    ):
        if not isinstance(shadow, torch.Tensor):
            errors.append(f"EMA[{index}] for {name} is not a tensor")
            continue
        if shadow.shape != parameter.shape or shadow.shape != raw_tensor.shape:
            errors.append(
                f"EMA[{index}] shape mismatch for {name}: "
                f"{tuple(shadow.shape)} != {tuple(parameter.shape)}"
            )
        if shadow.dtype != parameter.dtype or shadow.dtype != raw_tensor.dtype:
            errors.append(
                f"EMA[{index}] dtype mismatch for {name}: "
                f"{shadow.dtype} != {parameter.dtype}"
            )
    if errors:
        raise ValueError("invalid EMA mapping: " + "; ".join(errors))

    with torch.no_grad():
        for shadow, (_name, parameter, _raw_tensor) in zip(
            shadows, manifest, strict=True
        ):
            parameter.copy_(shadow)

    names = [name for name, _parameter, _raw_tensor in manifest]
    return {
        "source": "checkpoint.ema.shadow_params",
        "mapping": "ordered trainable model.backbone.named_parameters",
        "parameter_tensors": len(manifest),
        "ordered_parameter_names_sha256": _canonical_sha256(names),
        "validation": "exact count, state name, shape, and dtype before any copy",
    }


def load_checkpoint_model(
    checkpoint: Mapping[str, Any], weights: str
) -> tuple[Any, dict[str, Any]]:
    """Construct GenMol from the already verified, once-loaded checkpoint."""

    from genmol.model import (
        EMPIRICAL_FREQUENCY_RELATIVE_PATH as MODEL_FREQUENCY_RELATIVE_PATH,
        EMPIRICAL_FREQUENCY_SHA256 as MODEL_FREQUENCY_SHA256,
        UDLM_PRIOR_CHECKPOINT_KEY as MODEL_PRIOR_CHECKPOINT_KEY,
        UDLM_PRIOR_VARIANT_IDENTITIES as MODEL_PRIOR_VARIANT_IDENTITIES,
        GenMol,
    )

    if (
        MODEL_PRIOR_CHECKPOINT_KEY != UDLM_PRIOR_CHECKPOINT_KEY
        or MODEL_FREQUENCY_RELATIVE_PATH != EMPIRICAL_FREQUENCY_RELATIVE_PATH
        or MODEL_FREQUENCY_SHA256 != FROZEN_FREQUENCY_SHA256
        or MODEL_PRIOR_VARIANT_IDENTITIES != UDLM_PRIOR_VARIANT_IDENTITIES
    ):
        raise RuntimeError(
            "evaluator and GenMol UDLM prior contracts disagree; refusing to label "
            "the checkpoint"
        )
    model_source_path = Path(inspect.getfile(GenMol)).resolve()
    if model_source_path != (REPOSITORY_ROOT / "src/genmol/model.py").resolve():
        raise RuntimeError("GenMol model resolved outside the isolated worktree")

    hyper_parameters = checkpoint.get("hyper_parameters")
    if not isinstance(hyper_parameters, Mapping) or "config" not in hyper_parameters:
        raise ValueError("checkpoint has no hyper_parameters.config")
    checkpoint_state = checkpoint.get("state_dict")
    if not isinstance(checkpoint_state, Mapping):
        raise ValueError("checkpoint state_dict must be a mapping")

    model = GenMol(hyper_parameters["config"])
    # Manual construction bypasses Lightning's on_load_checkpoint hook. Invoke
    # the same immutable categorical-prior validation before accepting tensors.
    model._validate_runtime_udlm_prior_identity()
    model._validate_udlm_prior_checkpoint(checkpoint)
    runtime_metadata = getattr(model, "udlm_prior_metadata", None)
    runtime_record = None if runtime_metadata is None else runtime_metadata.to_dict()
    checkpoint_record = checkpoint.get(UDLM_PRIOR_CHECKPOINT_KEY)
    if type(model.mdlm) is ContinuousCategoricalDiffusion and (
        not isinstance(checkpoint_record, Mapping)
        or not _exact_data_equal(dict(checkpoint_record), runtime_record)
    ):
        raise ValueError("categorical UDLM checkpoint prior metadata is not type-exact")
    model.load_state_dict(checkpoint_state, strict=True)
    model._validate_runtime_udlm_prior_identity()
    manifest = _backbone_parameter_manifest(model, checkpoint_state)
    names = [name for name, _parameter, _raw_tensor in manifest]
    ema_enabled = model.ema is not None
    # GenMol's constructor initializes a fresh shadow copy. Evaluation reads
    # the checkpoint's separately validated shadows directly, so retaining the
    # fresh copy would waste one complete backbone of memory.
    model.ema = None
    if weights == "ema":
        ema_state = checkpoint.get("ema")
        if not ema_enabled or not isinstance(ema_state, Mapping):
            raise ValueError(
                "EMA weights requested, but the checkpoint has no enabled EMA state"
            )
        weight_provenance = apply_ema_weights(model, checkpoint_state, ema_state)
    elif weights == "raw":
        weight_provenance = {
            "source": "checkpoint.state_dict backbone.*",
            "parameter_tensors": len(manifest),
            "ordered_parameter_names_sha256": _canonical_sha256(names),
            "validation": "strict model state load plus exact backbone name/shape/dtype",
        }
    else:
        raise ValueError("weights must be 'ema' or 'raw'")
    runtime_prior_metadata = getattr(model, "udlm_prior_metadata", None)
    runtime_prior_record = (
        None if runtime_prior_metadata is None else runtime_prior_metadata.to_dict()
    )
    categorical = type(model.mdlm) is ContinuousCategoricalDiffusion
    weight_provenance["udlm_process_identity"] = {
        "exact_backend": f"{type(model.mdlm).__module__}.{type(model.mdlm).__qualname__}",
        "checkpoint_prior_metadata_required": categorical,
        "checkpoint_prior_metadata_present": (UDLM_PRIOR_CHECKPOINT_KEY in checkpoint),
        "runtime_prior_metadata_sha256": (
            None
            if runtime_prior_record is None
            else _canonical_sha256(runtime_prior_record)
        ),
        "validation": (
            "GenMol categorical checkpoint metadata, process buffers, configured "
            "pinned artifact, and live process identity were checked before and "
            "after strict state loading"
            if categorical
            else "release_uniform categorical metadata/state absence and live "
            "process identity were checked before and after strict state loading"
        ),
        "model_source_path": str(model_source_path),
    }
    model.backbone.eval()
    return model, weight_provenance


def _validate_device(device: str) -> None:
    parsed = torch.device(device)
    if parsed.type != "cpu":
        raise ValueError(
            "the standalone denoising evaluator is CPU-only until a dynamic "
            "idle-GPU launcher with recorded device mapping is implemented"
        )


def _validate_output_path(
    path: Path, *, force: bool, protected_paths: Sequence[Path] = ()
) -> Path:
    path = path.resolve()
    if not path.is_relative_to(REPOSITORY_ROOT):
        raise ValueError(f"output must be inside {REPOSITORY_ROOT}")
    if path in {protected.resolve() for protected in protected_paths}:
        raise ValueError(f"output cannot overwrite an evaluation input: {path}")
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {path}; pass --force")
    return path


def _protected_evaluation_paths(
    checkpoint_path: Path,
    panel_path: Path,
    frequency_path: Path,
) -> tuple[Path, ...]:
    """Bind CLI inputs, canonical artifacts, and implementation sources."""

    candidates = (
        checkpoint_path,
        panel_path,
        frequency_path,
        DEFAULT_PANEL,
        DEFAULT_FREQUENCIES,
        *(REPOSITORY_ROOT / path for path in SOURCE_INPUTS),
    )
    return tuple(dict.fromkeys(path.resolve() for path in candidates))


def _atomic_write_json(
    path: Path,
    payload: dict,
    force: bool,
    *,
    protected_paths: Sequence[Path] = (),
    path_is_prevalidated_and_resolved: bool = False,
) -> None:
    if path_is_prevalidated_and_resolved:
        # Preserve the exact target that was protected before checkpoint work;
        # do not resolve the user's spelling a second time after a long run.
        normalized_path = Path(os.path.abspath(os.fspath(path)))
        if (
            not path.is_absolute()
            or path != normalized_path
            or not path.is_relative_to(REPOSITORY_ROOT.resolve())
        ):
            raise ValueError(
                "prevalidated output path is not an absolute worktree path"
            )
        if path in set(protected_paths):
            raise ValueError(f"output cannot overwrite an evaluation input: {path}")
        if path.exists() and not force:
            raise FileExistsError(f"refusing to overwrite {path}; pass --force")
    else:
        path = _validate_output_path(path, force=force, protected_paths=protected_paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if force:
            os.replace(temporary, path)
        else:
            # A hard link gives no-clobber publication atomically.  Both paths
            # are in the same directory/filesystem; the private temp name was
            # itself created with O_CREAT|O_EXCL by mkstemp.
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise FileExistsError(
                    f"refusing to overwrite {path}; pass --force"
                ) from error
            temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument(
        "--training-frequencies", type=Path, default=DEFAULT_FREQUENCIES
    )
    parser.add_argument("--times", type=float, nargs="+", default=DEFAULT_TIME_BINS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=16,
        help="Evaluate a small 16-row prefix by default; pass 256 for the full panel.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu",),
        default="cpu",
        help=(
            "CPU only. GPU use requires a future launcher that dynamically selects "
            "and records genuinely idle physical devices."
        ),
    )
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    checkpoint_path = args.checkpoint.resolve()
    panel_path = args.panel.resolve()
    frequency_path = args.training_frequencies.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not checkpoint_path.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"checkpoint must be inside {PROJECT_ROOT}")
    protected_paths = _protected_evaluation_paths(
        checkpoint_path,
        panel_path,
        frequency_path,
    )
    output_path = _validate_output_path(
        args.output,
        force=args.force,
        protected_paths=protected_paths,
    )
    _validate_device(args.device)
    # Bind the already imported implementation before any potentially slow
    # artifact or checkpoint work, then re-hash the same set before publication.
    source_info = source_provenance(
        capture_phase="before_artifact_and_checkpoint_loading"
    )
    runtime_info = runtime_provenance()
    panel, frequencies, artifact_provenance = load_frozen_artifacts(
        panel_path, frequency_path
    )
    checkpoint, checkpoint_snapshot = load_verified_checkpoint(checkpoint_path)
    checkpoint_info = _checkpoint_metadata_from_payload(
        checkpoint_path, checkpoint, checkpoint_snapshot
    )
    if checkpoint_info["diffusion_type"] != "udlm":
        raise ValueError("checkpoint metadata does not declare training.diffusion=udlm")
    model, weight_provenance = load_checkpoint_model(checkpoint, args.weights)
    evaluation = evaluate_denoising_panel(
        model,
        panel,
        frequencies,
        time_bins=args.times,
        seed=args.seed,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
        device=args.device,
        checkpoint_prior_metadata=checkpoint.get(UDLM_PRIOR_CHECKPOINT_KEY),
        artifact_provenance=artifact_provenance,
    )
    del checkpoint
    verify_source_provenance(source_info)
    source_info["postcheck"] = {
        "status": "unchanged_after_checkpoint_load_and_evaluation",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    variant = evaluation["process"]["prior_variant"]
    result = {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            f"fixed held-out {variant} UDLM one-corruption denoising diagnostic; "
            "not molecule-generation benchmark evidence"
        ),
        "claim_scope": (
            "These token-level diagnostics do not measure generated-molecule quality "
            "and cannot establish that UDLM beats GenMol. Cross-backend or "
            "cross-checkpoint differences are descriptive, not causal estimates."
        ),
        "started_at_utc": started_at.isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": {
            **checkpoint_info,
            "weights_evaluated": args.weights,
            "weight_application": weight_provenance,
        },
        "artifacts": artifact_provenance,
        "source": source_info,
        "runtime": runtime_info,
        "definitions": {
            "estimator_scope": (
                "One deterministic corruption realization for every selected "
                "panel-row/time-bin pair. The fixed time grid is descriptive: it "
                "is neither an integrated nor an unbiased estimate of NELBO, and "
                "rows/time bins are not independent stochastic replicates."
            ),
            "content_token": (
                "A non-padding panel token whose clean token ID is absent from the "
                "panel tokenizer's complete special_token_ids list."
            ),
            "production_loss": (
                "The exact selected process's instantaneous loss_per_token value at "
                "the named time and realized corruption, summed in float64 and "
                "divided by the reported content-token denominator. For categorical "
                "UDLM this is only the model-dependent integrand and excludes the "
                "separately reported endpoint-prior KL. It is not an integrated "
                "NELBO estimate."
            ),
            "clean_token_nll": (
                "Negative log probability of the original clean token under "
                "the selected process's clean_log_probs(logits)."
            ),
            "clean_token_top1": (
                "Argmax over clean-token probabilities equals the original clean "
                "token ID."
            ),
            "observed_changed": (
                "The realized noisy token ID differs from its clean token ID. "
                "A latent prior refresh that redraws the same ID is therefore counted "
                "as observed_unchanged; this rate is token/prior-dependent and is not "
                "the latent refresh probability."
            ),
            "observed_unchanged": "The realized noisy token ID equals its clean token ID.",
            "training_frequency_bucket": (
                "Bucketed by the original clean token's integer count in the first "
                "10,000 revision-pinned training rows: 0, 1-9, 10-99, 100-999, "
                "or at least 1,000."
            ),
            "pooled_token_time_metrics": (
                "Micro-average over all reported content-token/time-bin observations; "
                "it is not an average of already-rounded bin means or an integral "
                "over time."
            ),
            "endpoint_prior_kl": (
                "For categorical UDLM only, the parameter-independent terminal "
                "KL(q(z_1|x0) || stationary_prior), micro-averaged once per clean "
                "content token and excluded from production_loss. Adding it to a "
                "fixed-time grid still does not yield a NELBO."
            ),
        },
        "evaluation": evaluation,
    }
    verify_source_provenance(source_info)
    _atomic_write_json(
        output_path,
        result,
        args.force,
        protected_paths=protected_paths,
        path_is_prevalidated_and_resolved=True,
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "output_sha256": sha256_file(output_path),
                "rows_evaluated": evaluation["rows_evaluated"],
                "time_bins": evaluation["time_bins"],
                "purpose": result["purpose"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

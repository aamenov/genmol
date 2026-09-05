"""Evaluate a UDLM checkpoint on the frozen held-out denoising panel.

This is a diagnostic, not a molecule-generation benchmark.  It measures the
same per-token objective used by the production uniform-UDLM implementation and
clean-token reconstruction for one deterministic corruption per row at each
fixed time.  The grid is not an integrated or unbiased NELBO estimate.  It does
not decode molecules or support a claim that one generator beats another.
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
from genmol.diffusion import ContinuousUniformDiffusion  # noqa: E402


DEFAULT_PANEL = REPOSITORY_ROOT / "experiments/udlm/validation_panel/first_256.json"
DEFAULT_FREQUENCIES = (
    REPOSITORY_ROOT / "experiments/udlm/token_frequency/train_first_10000.json"
)
FROZEN_PANEL_SHA256 = "e2493da4f3cb3217b48c78dc2901dc7524d7d90dc3959cffa87a1b6f8a9a7658"
FROZEN_FREQUENCY_SHA256 = (
    "088c78e75611f3cc42c4011e1da6f65a377e673b9cba07a28b126b0fc62f06ed"
)
DEFAULT_TIME_BINS = (0.1, 0.3, 0.5, 0.7, 0.9)
SCHEMA_VERSION = 2

SOURCE_INPUTS = (
    Path("scripts/udlm/evaluate_denoising_panel.py"),
    Path("scripts/udlm/materialize_validation_panel.py"),
    Path("scripts/udlm/launch_train_pilot.py"),
    Path("scripts/train.py"),
    Path("configs/base.yaml"),
    Path("configs/udlm.yaml"),
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


def _validate_uniform_udlm_backend(model: Any) -> ContinuousUniformDiffusion:
    """Accept only the exact uniform process this diagnostic defines.

    A categorical UDLM is also called ``udlm`` at the model level, but its
    transition kernel and loss are different.  Silently labelling it as the
    uniform control would invalidate the diagnostic.
    """

    if str(getattr(model, "diffusion_type", "")).lower() != "udlm":
        raise ValueError("denoising panel evaluation requires a UDLM model")
    if not hasattr(model, "mdlm"):
        raise ValueError("UDLM model has no diffusion process at model.mdlm")
    process = model.mdlm
    if type(process) is not ContinuousUniformDiffusion:
        observed = f"{type(process).__module__}.{type(process).__qualname__}"
        expected = (
            f"{ContinuousUniformDiffusion.__module__}."
            f"{ContinuousUniformDiffusion.__qualname__}"
        )
        raise ValueError(
            "this evaluator is defined only for the exact uniform UDLM backend; "
            f"observed {observed}, expected {expected}"
        )
    return process


def _process_metadata(process: Any, time_bins: Sequence[float]) -> dict[str, Any]:
    device = process.diffusion_token_ids.device
    time_tensor = torch.tensor(time_bins, device=device, dtype=torch.float32)
    with torch.no_grad():
        alpha = process.alpha(time_tensor).detach().cpu().double().tolist()
        sigma = process.sigma(time_tensor).detach().cpu().double().tolist()
    diffusion_token_ids = [
        int(value) for value in process.diffusion_token_ids.detach().cpu().tolist()
    ]
    allowed = set(diffusion_token_ids)
    excluded = [
        token_id
        for token_id in range(int(process.num_classes))
        if token_id not in allowed
    ]
    prior_identity = {
        "family": "uniform_over_diffusion_token_ids",
        "support_size": len(diffusion_token_ids),
        "probability_per_supported_token": 1.0 / len(diffusion_token_ids),
        "ordered_token_ids_sha256": _canonical_sha256(diffusion_token_ids),
    }
    prior_identity["identity_sha256"] = _canonical_sha256(prior_identity)
    return {
        "backend": f"{type(process).__module__}.{type(process).__qualname__}",
        "backend_exact_type_required": True,
        "prior": prior_identity,
        "num_classes": int(process.num_classes),
        "diffusion_vocab_size": int(process.diffusion_vocab_size),
        "excluded_token_ids": excluded,
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
                "ContinuousUniformDiffusion.loss_per_token uses the released "
                "idealized alpha_loss(t)=1-t and coefficient 1/(N*(1-t))"
            ),
            "qualification": (
                "This intentionally differs from the residual-noise corruption "
                "endpoint when noise_eps > 0."
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
    process = _validate_uniform_udlm_backend(model)
    parsed_device = torch.device(device)
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
    bin_results = []

    with torch.inference_mode():
        for time_value in time_bins:
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
            "descriptive grid, not an integrated or unbiased NELBO estimate"
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
                    "corrupted_token_ids_sha256": result[
                        "corrupted_token_ids_sha256"
                    ],
                }
                for result in bin_results
            ]
        ),
        "process": _process_metadata(process, time_bins),
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


def source_provenance() -> dict[str, Any]:
    hashes = {}
    for relative_path in SOURCE_INPUTS:
        absolute_path = REPOSITORY_ROOT / relative_path
        if not absolute_path.is_file():
            raise FileNotFoundError(absolute_path)
        hashes[str(relative_path)] = sha256_file(absolute_path)
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
        "repository_root": str(REPOSITORY_ROOT),
        "git_commit": _git_output(["rev-parse", "HEAD"]),
        "git_branch": _git_output(["branch", "--show-current"]),
        "git_worktree_state": worktree_state,
        "git_dirty": git_dirty,
        "git_status_sha256": None
        if status is None
        else hashlib.sha256(status.encode()).hexdigest(),
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
            raise RuntimeError(f"source input disappeared during evaluation: {absolute_path}")
        observed[str(relative_path)] = sha256_file(absolute_path)
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

    from genmol.model import GenMol

    hyper_parameters = checkpoint.get("hyper_parameters")
    if not isinstance(hyper_parameters, Mapping) or "config" not in hyper_parameters:
        raise ValueError("checkpoint has no hyper_parameters.config")
    checkpoint_state = checkpoint.get("state_dict")
    if not isinstance(checkpoint_state, Mapping):
        raise ValueError("checkpoint state_dict must be a mapping")

    model = GenMol(hyper_parameters["config"])
    model.load_state_dict(checkpoint_state, strict=True)
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


def _atomic_write_json(path: Path, payload: dict, force: bool) -> None:
    path = _validate_output_path(path, force=force)
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
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not checkpoint_path.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"checkpoint must be inside {PROJECT_ROOT}")
    _validate_output_path(
        args.output,
        force=args.force,
        protected_paths=(
            checkpoint_path,
            args.panel,
            args.training_frequencies,
            *(REPOSITORY_ROOT / path for path in SOURCE_INPUTS),
        ),
    )
    _validate_device(args.device)
    panel, frequencies, artifact_provenance = load_frozen_artifacts(
        args.panel, args.training_frequencies
    )
    checkpoint, checkpoint_snapshot = load_verified_checkpoint(checkpoint_path)
    checkpoint_info = _checkpoint_metadata_from_payload(
        checkpoint_path, checkpoint, checkpoint_snapshot
    )
    if checkpoint_info["diffusion_type"] != "udlm":
        raise ValueError("checkpoint metadata does not declare training.diffusion=udlm")
    source_info = source_provenance()
    runtime_info = runtime_provenance()
    model, weight_provenance = load_checkpoint_model(checkpoint, args.weights)
    del checkpoint
    evaluation = evaluate_denoising_panel(
        model,
        panel,
        frequencies,
        time_bins=args.times,
        seed=args.seed,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
        device=args.device,
    )
    verify_source_provenance(source_info)
    result = {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "fixed held-out uniform-UDLM one-corruption denoising diagnostic; "
            "not benchmark evidence"
        ),
        "claim_scope": (
            "These token-level diagnostics do not measure generated-molecule quality "
            "and cannot establish that UDLM beats GenMol."
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
                "The instantaneous ContinuousUniformDiffusion.loss_per_token value "
                "at the named time and realized corruption, summed in float64 and "
                "divided by the reported content-token denominator. It is not an "
                "integrated NELBO estimate."
            ),
            "clean_token_nll": (
                "Negative log probability of the original clean token under "
                "ContinuousUniformDiffusion.clean_log_probs(logits)."
            ),
            "clean_token_top1": (
                "Argmax over clean-token probabilities equals the original clean "
                "token ID."
            ),
            "observed_changed": (
                "The realized noisy token ID differs from its clean token ID. "
                "A latent uniform replacement that redraws the same ID is therefore "
                "counted as observed_unchanged."
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
        },
        "evaluation": evaluation,
    }
    _atomic_write_json(args.output, result, args.force)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "output_sha256": sha256_file(args.output.resolve()),
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

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import hashlib
import itertools
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import lightning as L
import torch
from transformers import BertForMaskedLM, get_constant_schedule_with_warmup
from transformers.models.bert.configuration_bert import BertConfig
from bionemo.moco.interpolants import MDLM
from bionemo.moco.distributions.time import UniformTimeDistribution
from genmol.utils.utils_moco import AntitheticUniformTimeDistribution
from bionemo.moco.schedules.noise.continuous_noise_transforms import LogLinearExpNoiseTransform
from bionemo.moco.distributions.prior import DiscreteMaskedPrior

from genmol.backbone import (
    ADDITIVE_CONDITIONING,
    FILM_ADALN_CONDITIONING,
    TimeConditionedBertForMaskedLM,
    is_conditioning_parameter_name,
)
from genmol.diffusion import (
    ContinuousCategoricalDiffusion,
    ContinuousUniformDiffusion,
)
from genmol.utils.checkpoint_io import verified_checkpoint_file
from genmol.utils.ema import ExponentialMovingAverage
from genmol.utils.utils_data import (
    SAFE_GPT_DATASET_REVISION,
    SAFE_GPT_REPO_ID,
    SAFE_GPT_TOKENIZER_REVISION,
    SAFE_GPT_TOKENIZER_SHA256,
    get_tokenizer,
)
from genmol.utils.utils_save import clean_checkpoint, fast_forward_info


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EMPIRICAL_FREQUENCY_RELATIVE_PATH = Path(
    "experiments/udlm/token_frequency/train_first_10000.json"
)
EMPIRICAL_FREQUENCY_PATH = REPOSITORY_ROOT / EMPIRICAL_FREQUENCY_RELATIVE_PATH
EMPIRICAL_FREQUENCY_SHA256 = (
    "088c78e75611f3cc42c4011e1da6f65a377e673b9cba07a28b126b0fc62f06ed"
)
EMPIRICAL_FREQUENCY_ORDERED_TEXT_SHA256 = (
    "53aee8e5592fc96159788e86519abbbcc9f1ab7c6348a1cb59a939bd57051d8f"
)
UDLM_PRIOR_CHECKPOINT_KEY = "udlm_prior_metadata"
UDLM_CONDITIONING_CHECKPOINT_KEY = "udlm_conditioning_metadata"
UDLM_CONDITIONING_METADATA_SCHEMA_VERSION = 1
OFFICIAL_UDLM_REFERENCE_REVISION = "edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661"
UDLM_PRIOR_VARIANTS = frozenset(
    {"release_uniform", "schedule_uniform", "empirical_frequency"}
)
CONSTANT_WITH_LINEAR_WARMUP = "constant_with_linear_warmup"
HALF_COSINE_WITH_LINEAR_WARMUP_AND_FLOOR = (
    "half_cosine_with_linear_warmup_and_floor"
)
OPTIMIZER_SCHEDULER_SPEC_SCHEMA_VERSION = 1
_OPTIMIZER_SCHEDULER_FIELDS = frozenset(
    {"name", "warmup_updates", "horizon_updates", "decay_floor_lr"}
)
_MISSING = object()
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


@dataclass(frozen=True)
class OptimizerSchedulerSpec:
    """Immutable optimizer-step schedule identity for execution evidence."""

    schema_version: int
    name: str
    peak_lr: float
    warmup_updates: int
    horizon_updates: int | None
    decay_floor_lr: float | None
    step_unit: str
    horizon_includes_warmup: bool
    post_horizon_policy: str

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != OPTIMIZER_SCHEDULER_SPEC_SCHEMA_VERSION
        ):
            raise ValueError("optimizer scheduler spec schema version is invalid")
        _finite_positive_real(self.peak_lr, "optimizer scheduler spec peak_lr")
        _nonnegative_update_count(
            self.warmup_updates, "optimizer scheduler spec warmup_updates"
        )
        if self.step_unit != "optimizer_update":
            raise ValueError("optimizer scheduler spec step_unit is invalid")
        if self.horizon_includes_warmup is not True:
            raise ValueError(
                "optimizer scheduler spec must count warmup inside its horizon"
            )
        if self.name == CONSTANT_WITH_LINEAR_WARMUP:
            if self.horizon_updates is not None or self.decay_floor_lr is not None:
                raise ValueError("constant optimizer scheduler spec is inconsistent")
            if self.post_horizon_policy != "constant_at_peak":
                raise ValueError("constant optimizer scheduler policy is invalid")
            return
        if self.name != HALF_COSINE_WITH_LINEAR_WARMUP_AND_FLOOR:
            raise ValueError("optimizer scheduler spec name is not registered")
        horizon_updates = _nonnegative_update_count(
            self.horizon_updates, "optimizer scheduler spec horizon_updates"
        )
        if horizon_updates <= self.warmup_updates:
            raise ValueError(
                "optimizer scheduler spec horizon must exceed its warmup"
            )
        if isinstance(self.decay_floor_lr, bool) or not isinstance(
            self.decay_floor_lr, Real
        ):
            raise ValueError("optimizer scheduler spec decay floor is invalid")
        decay_floor_lr = float(self.decay_floor_lr)
        if (
            not math.isfinite(decay_floor_lr)
            or decay_floor_lr < 0.0
            or decay_floor_lr >= float(self.peak_lr)
        ):
            raise ValueError("optimizer scheduler spec decay floor is invalid")
        if self.post_horizon_policy != "clamped_at_decay_floor":
            raise ValueError("cosine optimizer scheduler policy is invalid")


@dataclass(frozen=True)
class UDLMConditioningMetadata:
    """Immutable identity for the non-legacy BERT conditioning topology."""

    schema_version: int
    variant: str
    architecture: str
    timestep_input: str
    timestep_outer_activation: str
    timestep_mlp_output_initialization: str
    per_layer_modulation: str
    per_layer_modulation_initialization: str
    hidden_size: int
    layer_count: int
    conditioning_parameter_manifest: tuple[tuple[str, tuple[int, ...]], ...]
    official_udlm_reference_revision: str
    official_architecture_difference: str

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible checkpoint record."""

        record = asdict(self)
        record["conditioning_parameter_manifest"] = [
            {"name": name, "shape": list(shape)}
            for name, shape in self.conditioning_parameter_manifest
        ]
        return record


def _finite_positive_real(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite positive real number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{label} must be a finite positive real number")
    return parsed


def _nonnegative_update_count(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _exact_nested_data_equal(observed: object, expected: object) -> bool:
    """Compare JSON-like records without Python's bool/int coercions."""

    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        return observed.keys() == expected.keys() and all(
            _exact_nested_data_equal(observed[key], expected[key])
            for key in expected
        )
    if isinstance(expected, list):
        return len(observed) == len(expected) and all(
            _exact_nested_data_equal(left, right)
            for left, right in zip(observed, expected, strict=True)
        )
    return observed == expected


def optimizer_scheduler_spec(optim_config: Mapping[str, Any]) -> OptimizerSchedulerSpec:
    """Resolve one strict schedule, retaining the released config fallback.

    Checkpoints created before scheduler configuration was exposed have no
    ``optim.scheduler`` block.  That one absence maps exactly to the released
    constant schedule with 2,500 linear-warmup optimizer updates.  Once the
    block is present, every field is required so a typo cannot silently change
    an experiment.
    """

    if not isinstance(optim_config, Mapping):
        raise ValueError("optim must be a mapping")
    peak_lr = _finite_positive_real(optim_config.get("lr"), "optim.lr")
    scheduler_config = optim_config.get("scheduler", _MISSING)
    if scheduler_config is _MISSING:
        return OptimizerSchedulerSpec(
            schema_version=OPTIMIZER_SCHEDULER_SPEC_SCHEMA_VERSION,
            name=CONSTANT_WITH_LINEAR_WARMUP,
            peak_lr=peak_lr,
            warmup_updates=2500,
            horizon_updates=None,
            decay_floor_lr=None,
            step_unit="optimizer_update",
            horizon_includes_warmup=True,
            post_horizon_policy="constant_at_peak",
        )
    if not isinstance(scheduler_config, Mapping):
        raise ValueError("optim.scheduler must be a mapping when present")
    actual_fields = frozenset(scheduler_config)
    if actual_fields != _OPTIMIZER_SCHEDULER_FIELDS:
        missing = sorted(_OPTIMIZER_SCHEDULER_FIELDS - actual_fields)
        unexpected = sorted(actual_fields - _OPTIMIZER_SCHEDULER_FIELDS)
        raise ValueError(
            "optim.scheduler must contain exactly the registered fields: "
            f"missing={missing}, unexpected={unexpected}"
        )

    name = scheduler_config["name"]
    if not isinstance(name, str) or name not in {
        CONSTANT_WITH_LINEAR_WARMUP,
        HALF_COSINE_WITH_LINEAR_WARMUP_AND_FLOOR,
    }:
        raise ValueError("optim.scheduler.name is not registered")
    warmup_updates = _nonnegative_update_count(
        scheduler_config["warmup_updates"], "optim.scheduler.warmup_updates"
    )
    horizon_updates = scheduler_config["horizon_updates"]
    decay_floor_lr = scheduler_config["decay_floor_lr"]

    if name == CONSTANT_WITH_LINEAR_WARMUP:
        if horizon_updates is not None or decay_floor_lr is not None:
            raise ValueError(
                "constant schedule requires null horizon_updates and decay_floor_lr"
            )
        post_horizon_policy = "constant_at_peak"
    else:
        horizon_updates = _nonnegative_update_count(
            horizon_updates, "optim.scheduler.horizon_updates"
        )
        if horizon_updates <= warmup_updates:
            raise ValueError(
                "cosine horizon_updates must be greater than warmup_updates"
            )
        if isinstance(decay_floor_lr, bool) or not isinstance(decay_floor_lr, Real):
            raise ValueError(
                "optim.scheduler.decay_floor_lr must be a finite real number"
            )
        decay_floor_lr = float(decay_floor_lr)
        if (
            not math.isfinite(decay_floor_lr)
            or decay_floor_lr < 0.0
            or decay_floor_lr >= peak_lr
        ):
            raise ValueError(
                "optim.scheduler.decay_floor_lr must be finite and lie in "
                "[0, optim.lr)"
            )
        post_horizon_policy = "clamped_at_decay_floor"

    return OptimizerSchedulerSpec(
        schema_version=OPTIMIZER_SCHEDULER_SPEC_SCHEMA_VERSION,
        name=name,
        peak_lr=peak_lr,
        warmup_updates=warmup_updates,
        horizon_updates=horizon_updates,
        decay_floor_lr=decay_floor_lr,
        step_unit="optimizer_update",
        horizon_includes_warmup=True,
        post_horizon_policy=post_horizon_policy,
    )


def optimizer_scheduler_multiplier(
    spec: OptimizerSchedulerSpec, scheduler_index: int
) -> float:
    """Return the LR multiplier at one nonnegative ``LambdaLR`` index."""

    if not isinstance(spec, OptimizerSchedulerSpec):
        raise TypeError("spec must be an OptimizerSchedulerSpec")
    scheduler_index = _nonnegative_update_count(
        scheduler_index, "scheduler_index"
    )
    if scheduler_index < spec.warmup_updates:
        return scheduler_index / spec.warmup_updates
    if spec.name == CONSTANT_WITH_LINEAR_WARMUP:
        return 1.0
    if spec.name != HALF_COSINE_WITH_LINEAR_WARMUP_AND_FLOOR:
        raise ValueError("optimizer scheduler spec name is not registered")
    if spec.horizon_updates is None or spec.decay_floor_lr is None:
        raise ValueError("cosine optimizer scheduler spec is incomplete")
    progress = min(
        max(
            (scheduler_index - spec.warmup_updates)
            / (spec.horizon_updates - spec.warmup_updates),
            0.0,
        ),
        1.0,
    )
    floor_multiplier = spec.decay_floor_lr / spec.peak_lr
    half_cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return floor_multiplier + (1.0 - floor_multiplier) * half_cosine


def build_optimizer_scheduler(
    optimizer: torch.optim.Optimizer,
    spec: OptimizerSchedulerSpec,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Construct the scheduler identified by a validated immutable spec."""

    if not isinstance(spec, OptimizerSchedulerSpec):
        raise TypeError("spec must be an OptimizerSchedulerSpec")
    if spec.name == CONSTANT_WITH_LINEAR_WARMUP:
        # Keep the exact released Transformers implementation and state shape.
        return get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=spec.warmup_updates,
        )
    if spec.name == HALF_COSINE_WITH_LINEAR_WARMUP_AND_FLOOR:
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda index: optimizer_scheduler_multiplier(spec, index),
        )
    raise ValueError("optimizer scheduler spec name is not registered")


def _build_udlm_conditioning_metadata(
    backbone: TimeConditionedBertForMaskedLM,
) -> UDLMConditioningMetadata | None:
    """Describe and structurally validate a configured UDLM conditioner."""

    variant = getattr(backbone, "conditioning_variant", None)
    if variant == ADDITIVE_CONDITIONING:
        return None
    if variant != FILM_ADALN_CONDITIONING:
        raise RuntimeError("UDLM backbone has an unknown conditioning variant")

    hidden_size = int(backbone.config.hidden_size)
    layer_count = int(backbone.config.num_hidden_layers)
    manifest = tuple(
        (name, tuple(parameter.shape))
        for name, parameter in backbone.named_parameters()
        if is_conditioning_parameter_name(name)
    )
    expected_names_and_shapes = []
    for index in range(layer_count):
        prefix = f"bert.encoder.layer.{index}.film_modulation"
        expected_names_and_shapes.extend(
            [
                (f"{prefix}.weight", (2 * hidden_size, hidden_size)),
                (f"{prefix}.bias", (2 * hidden_size,)),
            ]
        )
    time_parameters = tuple(
        (name, tuple(parameter.shape))
        for name, parameter in backbone.named_parameters()
        if name.startswith("time_conditioner.")
    )
    if len(time_parameters) != 4:
        raise RuntimeError("FiLM UDLM must have exactly four timestep-MLP tensors")
    expected = set(expected_names_and_shapes) | set(time_parameters)
    if len(manifest) != len(expected) or set(manifest) != expected:
        raise RuntimeError(
            "FiLM UDLM conditioning parameter names or shapes are inconsistent"
        )

    return UDLMConditioningMetadata(
        schema_version=UDLM_CONDITIONING_METADATA_SCHEMA_VERSION,
        variant=FILM_ADALN_CONDITIONING,
        architecture="bert_post_block_film",
        timestep_input="continuous_noise_level_sigma",
        timestep_outer_activation="silu",
        timestep_mlp_output_initialization="pytorch_default_nonzero",
        per_layer_modulation="linear_hidden_to_shift_and_scale",
        per_layer_modulation_initialization="exact_zero_weight_and_bias",
        hidden_size=hidden_size,
        layer_count=layer_count,
        conditioning_parameter_manifest=manifest,
        official_udlm_reference_revision=OFFICIAL_UDLM_REFERENCE_REVISION,
        official_architecture_difference=(
            "minimal_post_block_BERT_FiLM_not_official_DiT_six_vector_adaLN_gating"
        ),
    )


@dataclass(frozen=True)
class UDLMPriorMetadata:
    """Immutable identity for one configured UDLM corruption process.

    The two categorical variants deliberately share ``process_family`` and
    ``schedule_variant``.  This makes the uniform categorical variant the
    schedule-repair control for the empirical-prior treatment.
    """

    schema_version: int
    variant: str
    comparison_role: str
    process_family: str
    schedule_variant: str
    objective_scope: str
    prior_source: str
    full_vocab_size: int
    active_vocab_size: int
    excluded_token_ids: tuple[int, ...]
    sampling_eps: float
    noise_eps: float
    antithetic_sampling: bool
    active_token_ids_sha256: str
    stationary_probs_sha256: str
    uniform_mixture_weight: float | None
    frequency_artifact_path: str | None
    frequency_artifact_sha256: str | None
    frequency_artifact_schema_version: int | None
    frequency_example_count: int | None
    frequency_content_token_count: int | None
    frequency_active_token_count: int | None
    frequency_dataset_repo_id: str | None
    frequency_dataset_revision: str | None
    frequency_dataset_split: str | None
    frequency_dataset_selection: str | None
    frequency_ordered_text_sha256: str | None
    frequency_implementation_git_sha: str | None
    tokenizer_repo_id: str
    tokenizer_revision: str
    tokenizer_json_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible record for later provenance."""

        record = asdict(self)
        record["excluded_token_ids"] = list(self.excluded_token_ids)
        return record


def _canonical_sequence_sha256(values: list[int] | list[float]) -> str:
    """Hash an ordered numeric sequence without platform-dependent bytes."""

    canonical = [value.hex() if isinstance(value, float) else value for value in values]
    encoded = json.dumps(canonical, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _integer_field(value: object, name: str, *, positive: bool = False) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer, not {type(value).__name__}")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _parse_frequency_artifact(
    artifact_path: Path,
    *,
    expected_sha256: str,
    model_vocab_size: int,
    tokenizer,
) -> tuple[dict[str, Any], list[int], str]:
    """Verify and parse exactly one pinned frequency-artifact byte snapshot."""

    try:
        artifact_bytes = artifact_path.read_bytes()
    except OSError as error:
        raise ValueError(
            f"cannot read empirical frequency artifact: {artifact_path}"
        ) from error
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    if artifact_sha256 != expected_sha256:
        raise ValueError(
            "empirical frequency artifact SHA-256 mismatch: "
            f"{artifact_sha256} != {expected_sha256}"
        )

    def reject_nonfinite_json(value: str):
        raise ValueError(f"frequency artifact contains non-finite JSON value {value}")

    try:
        artifact = json.loads(artifact_bytes, parse_constant=reject_nonfinite_json)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("empirical frequency artifact is not valid JSON") from error
    if not isinstance(artifact, dict):
        raise ValueError("empirical frequency artifact root must be an object")
    schema_version = _integer_field(
        artifact.get("schema_version"), "frequency artifact schema_version"
    )
    if schema_version != 1:
        raise ValueError(
            f"unsupported frequency artifact schema_version {schema_version}; expected 1"
        )
    if artifact.get("purpose") != (
        "CPU-only token-frequency diagnostic; not benchmark evidence"
    ):
        raise ValueError("frequency artifact purpose does not match its pinned schema")

    dataset_record = artifact.get("dataset")
    if not isinstance(dataset_record, dict):
        raise ValueError("frequency artifact dataset must be an object")
    expected_dataset_fields = {
        "repo_id": SAFE_GPT_REPO_ID,
        "revision": SAFE_GPT_DATASET_REVISION,
        "split": "train",
        "selection": "first 10000 streaming rows",
        "ordered_safe_text_sha256": EMPIRICAL_FREQUENCY_ORDERED_TEXT_SHA256,
    }
    for field, expected in expected_dataset_fields.items():
        if dataset_record.get(field) != expected:
            raise ValueError(
                f"frequency artifact dataset.{field} does not match the pinned "
                f"10,000-row training prefix"
            )
    max_sequence_length = _integer_field(
        artifact.get("max_sequence_length"),
        "frequency artifact max_sequence_length",
        positive=True,
    )
    if max_sequence_length != 256:
        raise ValueError("frequency artifact max_sequence_length must be 256")

    tokenizer_record = artifact.get("tokenizer")
    if not isinstance(tokenizer_record, dict):
        raise ValueError("frequency artifact tokenizer must be an object")
    artifact_vocab_size = _integer_field(
        tokenizer_record.get("base_vocab_size"),
        "frequency artifact tokenizer.base_vocab_size",
        positive=True,
    )
    tokenizer_vocab_size = _integer_field(
        getattr(tokenizer, "vocab_size", None),
        "runtime tokenizer.vocab_size",
        positive=True,
    )
    if (
        artifact_vocab_size != model_vocab_size
        or tokenizer_vocab_size != model_vocab_size
    ):
        raise ValueError(
            "frequency artifact, runtime tokenizer, and model vocab sizes must agree: "
            f"artifact={artifact_vocab_size}, tokenizer={tokenizer_vocab_size}, "
            f"model={model_vocab_size}"
        )
    artifact_tokenizer_hash = tokenizer_record.get("tokenizer_json_sha256")
    if artifact_tokenizer_hash != SAFE_GPT_TOKENIZER_SHA256:
        raise ValueError(
            "frequency artifact tokenizer SHA-256 mismatch: "
            f"{artifact_tokenizer_hash!r} != {SAFE_GPT_TOKENIZER_SHA256!r}"
        )
    if tokenizer_record.get("repo_id") != SAFE_GPT_REPO_ID:
        raise ValueError(
            "frequency artifact tokenizer repository is not the pinned SAFE tokenizer"
        )
    if tokenizer_record.get("revision") != SAFE_GPT_TOKENIZER_REVISION:
        raise ValueError("frequency artifact tokenizer revision is not pinned")

    artifact_special_ids = tokenizer_record.get("special_token_ids")
    if not isinstance(artifact_special_ids, list) or any(
        type(token_id) is not int for token_id in artifact_special_ids
    ):
        raise ValueError(
            "frequency artifact tokenizer.special_token_ids must be integers"
        )
    if artifact_special_ids != sorted(set(artifact_special_ids)) or any(
        not 0 <= token_id < model_vocab_size for token_id in artifact_special_ids
    ):
        raise ValueError(
            "frequency artifact tokenizer.special_token_ids must be unique, sorted, "
            "and inside the model vocabulary"
        )
    runtime_special_ids = sorted(set(int(value) for value in tokenizer.all_special_ids))
    if artifact_special_ids != runtime_special_ids:
        raise ValueError(
            "frequency artifact special token IDs do not match the runtime tokenizer: "
            f"artifact={artifact_special_ids}, runtime={runtime_special_ids}"
        )

    counts = artifact.get("counts_by_token_id")
    if not isinstance(counts, list) or len(counts) != model_vocab_size:
        received = "not a list" if not isinstance(counts, list) else len(counts)
        raise ValueError(
            "frequency artifact counts_by_token_id must have one entry per model token: "
            f"expected {model_vocab_size}, received {received}"
        )
    for token_id, count in enumerate(counts):
        if type(count) is not int or count < 0:
            raise ValueError(
                "frequency artifact counts must be non-negative integers; "
                f"token {token_id} has {count!r}"
            )
    content_token_count = _integer_field(
        artifact.get("content_token_count"),
        "frequency artifact content_token_count",
        positive=True,
    )
    if sum(counts) != content_token_count:
        raise ValueError(
            "frequency artifact count sum does not match content_token_count: "
            f"{sum(counts)} != {content_token_count}"
        )
    example_count = _integer_field(
        artifact.get("example_count"),
        "frequency artifact example_count",
        positive=True,
    )
    if example_count != 10_000:
        raise ValueError("frequency artifact example_count must be exactly 10000")
    nonzero_special_ids = [
        token_id for token_id in artifact_special_ids if counts[token_id] != 0
    ]
    if nonzero_special_ids:
        raise ValueError(
            "frequency artifact content counts include declared special tokens: "
            f"{nonzero_special_ids}"
        )
    implementation_git_sha = artifact.get("git_sha")
    if not isinstance(implementation_git_sha, str) or (
        len(implementation_git_sha) != 40
        or any(character not in "0123456789abcdef" for character in implementation_git_sha)
    ):
        raise ValueError("frequency artifact git_sha must be a full lowercase commit hash")
    return artifact, counts, artifact_sha256


def _validate_uniform_mixture_weight(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("training.udlm.empirical_uniform_mix must be a real number")
    weight = float(value)
    if not math.isfinite(weight) or not 0.0 < weight < 1.0:
        raise ValueError(
            "training.udlm.empirical_uniform_mix must lie in (0, 1): positive "
            "uniform mass gives full support and positive empirical mass keeps "
            "this a genuine empirical-prior treatment"
        )
    return weight


def _build_udlm_process(
    *,
    variant: str,
    model_vocab_size: int,
    excluded_token_ids: tuple[int, ...],
    sampling_eps: float,
    noise_eps: float,
    antithetic_sampling: bool,
    empirical_uniform_mix: object,
    tokenizer,
) -> tuple[ContinuousUniformDiffusion, UDLMPriorMetadata]:
    """Construct one explicit UDLM prior variant and its immutable identity."""

    if variant not in UDLM_PRIOR_VARIANTS:
        allowed = ", ".join(sorted(UDLM_PRIOR_VARIANTS))
        raise ValueError(f"training.udlm.prior_variant must be one of: {allowed}")
    excluded = tuple(sorted(set(int(token_id) for token_id in excluded_token_ids)))
    active_token_ids = [
        token_id for token_id in range(model_vocab_size) if token_id not in excluded
    ]
    active_size = len(active_token_ids)
    if active_size < 2:
        raise ValueError("the UDLM diffusion alphabet must contain at least 2 tokens")
    uniform_probs = [1.0 / active_size] * active_size

    artifact = None
    artifact_sha256 = None
    active_frequency_count = None
    mixture_weight = None
    if variant == "release_uniform":
        process = ContinuousUniformDiffusion(
            num_classes=model_vocab_size,
            excluded_token_ids=excluded,
            sampling_eps=sampling_eps,
            noise_eps=noise_eps,
            antithetic_sampling=antithetic_sampling,
        )
        stationary_probs = uniform_probs
    else:
        if variant == "schedule_uniform":
            stationary_probs = uniform_probs
        else:
            mixture_weight = _validate_uniform_mixture_weight(empirical_uniform_mix)
            artifact, counts, artifact_sha256 = _parse_frequency_artifact(
                EMPIRICAL_FREQUENCY_PATH,
                expected_sha256=EMPIRICAL_FREQUENCY_SHA256,
                model_vocab_size=model_vocab_size,
                tokenizer=tokenizer,
            )
            active_counts = [counts[token_id] for token_id in active_token_ids]
            active_frequency_count = sum(active_counts)
            if active_frequency_count <= 0:
                raise ValueError(
                    "frequency artifact has no observations in the active diffusion alphabet"
                )
            empirical_probs = [
                count / active_frequency_count for count in active_counts
            ]
            stationary_probs = [
                (1.0 - mixture_weight) * empirical + mixture_weight / active_size
                for empirical in empirical_probs
            ]
            if not all(
                math.isfinite(probability) and probability > 0.0
                for probability in stationary_probs
            ):
                raise ValueError(
                    "smoothed empirical prior must have finite full support"
                )
            if not math.isclose(
                sum(stationary_probs), 1.0, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(
                    "smoothed empirical prior failed active-alphabet normalization"
                )

        process = ContinuousCategoricalDiffusion(
            num_classes=model_vocab_size,
            stationary_probs=stationary_probs,
            excluded_token_ids=excluded,
            sampling_eps=sampling_eps,
            noise_eps=noise_eps,
            antithetic_sampling=antithetic_sampling,
        )

    variant_identity = UDLM_PRIOR_VARIANT_IDENTITIES[variant]

    process_active_ids = [int(value) for value in process.diffusion_token_ids.tolist()]
    if process_active_ids != active_token_ids:
        raise RuntimeError("UDLM process changed the declared compact token ordering")
    if isinstance(process, ContinuousCategoricalDiffusion):
        stored_probs = [float(value) for value in process.stationary_probs.tolist()]
    else:
        # Canonicalize the implicit released uniform law exactly as the
        # categorical constructor would store it. Its probability identity
        # must match schedule_uniform even though the process implementation
        # and loss schedule intentionally differ.
        canonical_probs = torch.tensor(stationary_probs, dtype=torch.float64)
        canonical_probs /= canonical_probs.sum()
        stored_probs = [float(value) for value in canonical_probs.tolist()]
    stationary_probs_sha256 = _canonical_sequence_sha256(stored_probs)
    if variant == "empirical_frequency":
        uniform_tensor = torch.tensor(uniform_probs, dtype=torch.float64)
        uniform_tensor /= uniform_tensor.sum()
        uniform_sha256 = _canonical_sequence_sha256(
            [float(value) for value in uniform_tensor.tolist()]
        )
        if stationary_probs_sha256 == uniform_sha256:
            raise ValueError(
                "smoothed empirical prior is exactly uniform on the active alphabet; "
                "it would not be a distinct prior treatment"
            )
    metadata = UDLMPriorMetadata(
        schema_version=1,
        variant=variant,
        comparison_role=variant_identity["comparison_role"],
        process_family=variant_identity["process_family"],
        schedule_variant=variant_identity["schedule_variant"],
        objective_scope=variant_identity["objective_scope"],
        prior_source=variant_identity["prior_source"],
        full_vocab_size=model_vocab_size,
        active_vocab_size=active_size,
        excluded_token_ids=excluded,
        sampling_eps=sampling_eps,
        noise_eps=noise_eps,
        antithetic_sampling=antithetic_sampling,
        active_token_ids_sha256=_canonical_sequence_sha256(active_token_ids),
        stationary_probs_sha256=stationary_probs_sha256,
        uniform_mixture_weight=mixture_weight,
        frequency_artifact_path=(
            EMPIRICAL_FREQUENCY_RELATIVE_PATH.as_posix()
            if artifact is not None
            else None
        ),
        frequency_artifact_sha256=artifact_sha256,
        frequency_artifact_schema_version=(
            artifact["schema_version"] if artifact is not None else None
        ),
        frequency_example_count=(
            artifact["example_count"] if artifact is not None else None
        ),
        frequency_content_token_count=(
            artifact["content_token_count"] if artifact is not None else None
        ),
        frequency_active_token_count=active_frequency_count,
        frequency_dataset_repo_id=(
            artifact["dataset"]["repo_id"] if artifact is not None else None
        ),
        frequency_dataset_revision=(
            artifact["dataset"]["revision"] if artifact is not None else None
        ),
        frequency_dataset_split=(
            artifact["dataset"]["split"] if artifact is not None else None
        ),
        frequency_dataset_selection=(
            artifact["dataset"]["selection"] if artifact is not None else None
        ),
        frequency_ordered_text_sha256=(
            artifact["dataset"]["ordered_safe_text_sha256"]
            if artifact is not None
            else None
        ),
        frequency_implementation_git_sha=(
            artifact["git_sha"] if artifact is not None else None
        ),
        tokenizer_repo_id=SAFE_GPT_REPO_ID,
        tokenizer_revision=SAFE_GPT_TOKENIZER_REVISION,
        tokenizer_json_sha256=SAFE_GPT_TOKENIZER_SHA256,
    )
    return process, metadata

class GenMol(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        optim_config = self.config.get("optim")
        self._optimizer_scheduler_spec = (
            optimizer_scheduler_spec(optim_config)
            if optim_config is not None
            else None
        )
        self._udlm_prior_metadata: UDLMPriorMetadata | None = None
        self._udlm_conditioning_metadata: UDLMConditioningMetadata | None = None
        # set up tokenizer
        self.tokenizer = get_tokenizer()
        self.mask_index = self.tokenizer.mask_token_id
        self.bos_index = self.tokenizer.bos_token_id
        self.eos_index = self.tokenizer.eos_token_id
        self.pad_index = self.tokenizer.pad_token_id

        # Checkpoints released before UDLM have no diffusion selector.  The
        # explicit MDLM fallback keeps their architecture and state keys exact.
        self.diffusion_type = str(self.config.training.get('diffusion', 'mdlm')).lower()
        if self.diffusion_type not in {'mdlm', 'udlm'}:
            raise ValueError("training.diffusion must be either 'mdlm' or 'udlm'")

        backbone_config = BertConfig.from_dict(dict(self.config.model))
        if self.diffusion_type == 'udlm':
            udlm_config = self.config.training.get('udlm', {})
            zero_init_conditioning = udlm_config.get(
                'zero_init_conditioning', True
            )
            if type(zero_init_conditioning) is not bool:
                raise ValueError(
                    "training.udlm.zero_init_conditioning must be a boolean"
                )
            self.backbone = TimeConditionedBertForMaskedLM(
                backbone_config,
                time_embedding_size=int(udlm_config.get('time_embedding_size', 256)),
                zero_init_conditioning=zero_init_conditioning,
                conditioning_variant=str(
                    udlm_config.get('conditioning_variant', ADDITIVE_CONDITIONING)
                ),
            )
            self._udlm_conditioning_metadata = _build_udlm_conditioning_metadata(
                self.backbone
            )
        else:
            self.backbone = BertForMaskedLM(backbone_config)

        # Keep ``mdlm`` as the process attribute for downstream compatibility.
        # For a UDLM checkpoint it points to the uniform process instead.
        if self.diffusion_type == 'mdlm':
            if self.config.training.antithetic_sampling:
                time_distribution = AntitheticUniformTimeDistribution(
                    sampling_eps=self.config.training.sampling_eps
                )
            else:
                time_distribution = UniformTimeDistribution()
            prior = DiscreteMaskedPrior(
                num_classes=self.config.model.vocab_size,
                mask_dim=self.mask_index,
            )
            noise_schedule = LogLinearExpNoiseTransform()
            self.mdlm = MDLM(
                time_distribution=time_distribution,
                prior_distribution=prior,
                noise_schedule=noise_schedule,
            )
        else:
            udlm_config = self.config.training.get('udlm', {})
            excluded_token_ids = ()
            if udlm_config.get('exclude_special_tokens', False):
                excluded_token_ids = tuple(self.tokenizer.all_special_ids)
            prior_variant = str(
                udlm_config.get('prior_variant', 'release_uniform')
            ).lower()
            self.mdlm, self._udlm_prior_metadata = _build_udlm_process(
                variant=prior_variant,
                model_vocab_size=int(self.config.model.vocab_size),
                excluded_token_ids=excluded_token_ids,
                sampling_eps=float(self.config.training.sampling_eps),
                noise_eps=float(udlm_config.get('noise_eps', 1e-3)),
                antithetic_sampling=bool(self.config.training.antithetic_sampling),
                empirical_uniform_mix=udlm_config.get(
                    'empirical_uniform_mix', None
                ),
                tokenizer=self.tokenizer,
            )
        # set up ema
        if self.config.training.ema > 0:
            self.ema = ExponentialMovingAverage(self.backbone.parameters(), decay=self.config.training.ema)
        else:
            self.ema = None

    @property
    def udlm_prior_metadata(self) -> UDLMPriorMetadata | None:
        """Read-only process identity; its returned record is itself frozen."""

        return self._udlm_prior_metadata

    @property
    def optimizer_scheduler_spec(self) -> OptimizerSchedulerSpec | None:
        """Return the immutable schedule identity, if optimization was configured."""

        return self._optimizer_scheduler_spec

    @property
    def udlm_conditioning_metadata(self) -> UDLMConditioningMetadata | None:
        """Return the frozen identity for a non-legacy UDLM conditioner."""

        return self._udlm_conditioning_metadata

    def _validate_runtime_udlm_conditioning_identity(self) -> None:
        """Ensure config, live modules, and immutable conditioner identity agree."""

        metadata = self.udlm_conditioning_metadata
        if self.diffusion_type != "udlm":
            if metadata is not None:
                raise RuntimeError(
                    "UDLM conditioning metadata is attached to a non-UDLM model"
                )
            return
        observed = _build_udlm_conditioning_metadata(self.backbone)
        if observed != metadata:
            raise RuntimeError(
                "live UDLM conditioning topology disagrees with its metadata"
            )

    def _validate_runtime_udlm_prior_identity(self) -> None:
        """Ensure the immutable record still identifies the live process."""

        metadata = self.udlm_prior_metadata
        if metadata is None:
            if self.diffusion_type == "udlm":
                raise RuntimeError("UDLM model is missing immutable prior metadata")
            return
        if self.diffusion_type != "udlm":
            raise RuntimeError("UDLM prior metadata is attached to a non-UDLM model")
        expected_class = (
            ContinuousUniformDiffusion
            if metadata.variant == "release_uniform"
            else ContinuousCategoricalDiffusion
        )
        if (
            metadata.variant not in UDLM_PRIOR_VARIANTS
            or type(self.mdlm) is not expected_class
        ):
            raise RuntimeError(
                "live UDLM process class disagrees with its declared prior variant"
            )
        expected_identity = UDLM_PRIOR_VARIANT_IDENTITIES[metadata.variant]
        if any(
            getattr(metadata, field) != value
            for field, value in expected_identity.items()
        ):
            raise RuntimeError("UDLM prior metadata contains an invalid variant identity")
        if (
            self.mdlm.num_classes != metadata.full_vocab_size
            or self.mdlm.sampling_eps != metadata.sampling_eps
            or self.mdlm.noise_eps != metadata.noise_eps
            or self.mdlm.antithetic_sampling != metadata.antithetic_sampling
        ):
            raise RuntimeError(
                "live UDLM vocabulary, schedule, or time sampler disagrees with prior metadata"
            )
        active_token_ids = [int(value) for value in self.mdlm.diffusion_token_ids.tolist()]
        active_token_set = set(active_token_ids)
        excluded_token_ids = tuple(
            token_id
            for token_id in range(self.mdlm.num_classes)
            if token_id not in active_token_set
        )
        if len(active_token_ids) != metadata.active_vocab_size or (
            _canonical_sequence_sha256(active_token_ids)
            != metadata.active_token_ids_sha256
        ) or excluded_token_ids != metadata.excluded_token_ids:
            raise RuntimeError("live UDLM active alphabet disagrees with prior metadata")
        expected_mapping = torch.full(
            (self.mdlm.num_classes,),
            -1,
            dtype=torch.long,
            device=self.mdlm.token_to_diffusion_index.device,
        )
        expected_mapping[self.mdlm.diffusion_token_ids] = torch.arange(
            len(active_token_ids),
            dtype=torch.long,
            device=expected_mapping.device,
        )
        if not torch.equal(self.mdlm.token_to_diffusion_index, expected_mapping):
            raise RuntimeError("live UDLM compact-token mapping is inconsistent")
        if type(self.mdlm) is ContinuousCategoricalDiffusion:
            probabilities = [
                float(value) for value in self.mdlm.stationary_probs.tolist()
            ]
        else:
            probabilities_tensor = torch.full(
                (len(active_token_ids),),
                1.0 / len(active_token_ids),
                dtype=torch.float64,
            )
            probabilities_tensor /= probabilities_tensor.sum()
            probabilities = [float(value) for value in probabilities_tensor.tolist()]
        if (
            _canonical_sequence_sha256(probabilities)
            != metadata.stationary_probs_sha256
        ):
            raise RuntimeError("live UDLM stationary prior disagrees with prior metadata")

    def _validate_udlm_prior_state_dict(
        self, state_dict: Mapping[str, object]
    ) -> None:
        """Reject checkpoint state that would contradict configured provenance."""

        metadata = self.udlm_prior_metadata
        prior_key = "mdlm.stationary_probs"
        if metadata is None:
            return
        expected_buffers = {
            "mdlm.diffusion_token_ids": self.mdlm.diffusion_token_ids,
            "mdlm.token_to_diffusion_index": self.mdlm.token_to_diffusion_index,
        }
        for key, expected in expected_buffers.items():
            checkpoint_value = state_dict.get(key)
            if not isinstance(checkpoint_value, torch.Tensor):
                raise ValueError(f"checkpoint is missing tensor process buffer {key}")
            if (
                checkpoint_value.device.type == "meta"
                or checkpoint_value.shape != expected.shape
                or checkpoint_value.dtype != expected.dtype
                or not torch.equal(checkpoint_value.detach().cpu(), expected.detach().cpu())
            ):
                raise ValueError(
                    f"checkpoint process buffer {key} disagrees with the configured "
                    "UDLM active alphabet"
                )
        categorical = type(self.mdlm) is ContinuousCategoricalDiffusion
        if not categorical:
            if prior_key in state_dict:
                raise ValueError(
                    "checkpoint contains a categorical stationary prior but the "
                    f"configured UDLM variant is {metadata.variant!r}"
                )
            return
        if prior_key not in state_dict:
            raise ValueError(
                "categorical UDLM checkpoint is missing mdlm.stationary_probs"
            )
        checkpoint_prior = state_dict[prior_key]
        if not isinstance(checkpoint_prior, torch.Tensor):
            raise ValueError("checkpoint mdlm.stationary_probs must be a tensor")
        expected_prior = self.mdlm.stationary_probs.detach().cpu()
        if (
            checkpoint_prior.device.type == "meta"
            or checkpoint_prior.shape != expected_prior.shape
            or checkpoint_prior.dtype != expected_prior.dtype
            or not torch.equal(checkpoint_prior.detach().cpu(), expected_prior)
        ):
            raise ValueError(
                "checkpoint stationary prior disagrees with the configured, "
                "verified UDLM prior"
            )

    def _validate_udlm_prior_checkpoint(self, checkpoint: Mapping[str, object]) -> None:
        """Validate categorical checkpoint metadata before tensors are loaded."""

        metadata = self.udlm_prior_metadata
        checkpoint_declares_metadata = UDLM_PRIOR_CHECKPOINT_KEY in checkpoint
        checkpoint_metadata = checkpoint.get(UDLM_PRIOR_CHECKPOINT_KEY)
        if metadata is None:
            if checkpoint_declares_metadata:
                raise ValueError("non-UDLM checkpoint unexpectedly declares UDLM prior metadata")
            return
        categorical = type(self.mdlm) is ContinuousCategoricalDiffusion
        if not categorical:
            if checkpoint_declares_metadata:
                raise ValueError(
                    "release_uniform checkpoint must not declare categorical prior metadata"
                )
            return
        if not isinstance(checkpoint_metadata, Mapping):
            raise ValueError(
                "categorical UDLM checkpoint is missing immutable prior metadata"
            )
        expected_metadata = metadata.to_dict()
        if not _exact_nested_data_equal(dict(checkpoint_metadata), expected_metadata):
            raise ValueError(
                "categorical UDLM checkpoint prior metadata does not match the "
                "configured process and pinned artifact"
            )
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise ValueError("categorical UDLM checkpoint state_dict must be a mapping")
        self._validate_udlm_prior_state_dict(state_dict)

    def _validate_udlm_conditioning_state_dict(
        self, state_dict: Mapping[str, object]
    ) -> None:
        """Reject cross-topology state loads even when callers request non-strict."""

        expected = {
            name for name in self.state_dict() if is_conditioning_parameter_name(name)
        }
        observed = {
            name for name in state_dict if is_conditioning_parameter_name(name)
        }
        if observed != expected:
            missing = sorted(expected - observed)
            unexpected = sorted(observed - expected)
            raise ValueError(
                "checkpoint conditioning state disagrees with the configured "
                f"topology: missing={missing}, unexpected={unexpected}"
            )

    def _validate_udlm_conditioning_checkpoint(
        self, checkpoint: Mapping[str, object]
    ) -> None:
        """Validate A1 topology metadata before Lightning loads its tensors."""

        self._validate_runtime_udlm_conditioning_identity()
        checkpoint_declares_metadata = (
            UDLM_CONDITIONING_CHECKPOINT_KEY in checkpoint
        )
        checkpoint_metadata = checkpoint.get(UDLM_CONDITIONING_CHECKPOINT_KEY)
        metadata = self.udlm_conditioning_metadata
        if metadata is None:
            if checkpoint_declares_metadata:
                raise ValueError(
                    "legacy/additive checkpoint unexpectedly declares FiLM metadata"
                )
        else:
            if not isinstance(checkpoint_metadata, Mapping):
                raise ValueError(
                    "FiLM checkpoint is missing immutable conditioning metadata"
                )
            if not _exact_nested_data_equal(
                dict(checkpoint_metadata), metadata.to_dict()
            ):
                raise ValueError(
                    "checkpoint conditioning metadata does not match the "
                    "configured topology"
                )
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise ValueError("checkpoint state_dict must be a mapping")
        self._validate_udlm_conditioning_state_dict(state_dict)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load weights only when their process prior matches this model."""

        self._validate_runtime_udlm_prior_identity()
        self._validate_runtime_udlm_conditioning_identity()
        self._validate_udlm_prior_state_dict(state_dict)
        self._validate_udlm_conditioning_state_dict(state_dict)
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._validate_runtime_udlm_prior_identity()
        self._validate_runtime_udlm_conditioning_identity()
        return result

    def initialize_from_mdlm_checkpoint(
        self,
        checkpoint_path,
        use_ema=True,
        expected_sha256=None,
    ):
        """Warm-start UDLM's BERT only, resetting all training state.

        This is intentionally not a Lightning resume: the optimizer, learning
        rate schedule, global step, UDLM process, time conditioner, and EMA are
        all new.  When available, the source MDLM EMA weights are preferred
        because they are the weights used by the released GenMol sampler.
        """

        if self.diffusion_type != 'udlm':
            raise ValueError('MDLM backbone initialization is only valid for UDLM')
        with verified_checkpoint_file(
            checkpoint_path,
            expected_sha256=expected_sha256,
        ) as (checkpoint_file, source_identity):
            checkpoint = torch.load(
                checkpoint_file,
                map_location='cpu',
                weights_only=False,
            )
        source_state = checkpoint.get('state_dict', checkpoint)
        backbone_state = {
            key.removeprefix('backbone.'): value
            for key, value in source_state.items()
            if key.startswith('backbone.')
        }
        if not backbone_state:
            raise ValueError('checkpoint contains no backbone.* tensors')

        load_result = self.backbone.load_state_dict(backbone_state, strict=False)
        expected_missing = {
            key
            for key in self.backbone.state_dict()
            if is_conditioning_parameter_name(key)
        }
        if set(load_result.missing_keys) != expected_missing or load_result.unexpected_keys:
            raise ValueError(
                'MDLM backbone is incompatible with this UDLM architecture: '
                f'missing={load_result.missing_keys}, '
                f'unexpected={load_result.unexpected_keys}'
            )

        weights = 'raw'
        base_parameters = [
            (name, parameter)
            for name, parameter in self.backbone.named_parameters()
            if not is_conditioning_parameter_name(name)
        ]
        conditioning_parameters = [
            (name, parameter)
            for name, parameter in self.backbone.named_parameters()
            if is_conditioning_parameter_name(name)
        ]
        if use_ema:
            ema_state = checkpoint.get('ema')
            shadow_parameters = None if ema_state is None else ema_state.get('shadow_params')
            if shadow_parameters is None:
                raise ValueError('requested MDLM EMA initialization, but checkpoint has no EMA')
            if len(shadow_parameters) != len(base_parameters):
                raise ValueError(
                    'MDLM EMA parameter count does not match the BERT backbone: '
                    f'{len(shadow_parameters)} != {len(base_parameters)}'
                )
            for (name, parameter), shadow in zip(
                base_parameters, shadow_parameters, strict=True
            ):
                if (
                    not isinstance(shadow, torch.Tensor)
                    or shadow.shape != parameter.shape
                    or shadow.dtype != parameter.dtype
                ):
                    raise ValueError(
                        "MDLM EMA tensor is incompatible with target backbone "
                        f"parameter {name}"
                    )
            with torch.no_grad():
                for (_name, parameter), shadow in zip(
                    base_parameters, shadow_parameters, strict=True
                ):
                    parameter.copy_(shadow)
            weights = 'ema'

        if self.ema:
            self.ema = ExponentialMovingAverage(
                self.backbone.parameters(),
                decay=self.config.training.ema,
            )
        report = {
            'source_path': str(checkpoint_path),
            'source_resolved_path': source_identity.resolved_path,
            'source_sha256': source_identity.sha256,
            'source_size_bytes': source_identity.size_bytes,
            'expected_source_sha256': expected_sha256,
            'byte_identity_verified_before_and_after_load': True,
            'weights': weights,
            'parameter_tensors': len(base_parameters),
        }
        if self.udlm_conditioning_metadata is not None:
            report['conditioning_variant'] = self.backbone.conditioning_variant
            report['conditioning_parameter_tensors'] = len(
                conditioning_parameters
            )
        return report

    def on_load_checkpoint(self, checkpoint):
        self._validate_runtime_udlm_prior_identity()
        self._validate_udlm_prior_checkpoint(checkpoint)
        self._validate_udlm_conditioning_checkpoint(checkpoint)
        if self.ema:
            self.ema.load_state_dict(checkpoint['ema'])
        self.fast_forward_epochs, self.fast_forward_batches = fast_forward_info(checkpoint)
        
    def on_save_checkpoint(self, checkpoint):
        self._validate_runtime_udlm_prior_identity()
        self._validate_runtime_udlm_conditioning_identity()
        if type(self.mdlm) is ContinuousCategoricalDiffusion:
            checkpoint[UDLM_PRIOR_CHECKPOINT_KEY] = (
                self.udlm_prior_metadata.to_dict()
            )
        if self.udlm_conditioning_metadata is not None:
            checkpoint[UDLM_CONDITIONING_CHECKPOINT_KEY] = (
                self.udlm_conditioning_metadata.to_dict()
            )
        if self.ema:
            checkpoint['ema'] = self.ema.state_dict()
        clean_checkpoint(checkpoint, self.trainer.accumulate_grad_batches)
        if 'sampler' not in checkpoint.keys():
            checkpoint['sampler'] = {}
        if hasattr(self.trainer.train_dataloader.sampler, 'state_dict'):
            sampler_state_dict = self.trainer.train_dataloader.sampler.state_dict()
            checkpoint['sampler']['random_state'] = sampler_state_dict.get('random_state', None)
        else:
            checkpoint['sampler']['random_state'] = None

    def configure_optimizers(self):
        scheduler_spec = self.optimizer_scheduler_spec
        if scheduler_spec is None:
            raise RuntimeError(
                "cannot configure optimization without an optim configuration"
            )
        optimizer = torch.optim.AdamW(
            self.backbone.parameters(),
            lr=scheduler_spec.peak_lr,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay)

        scheduler = build_optimizer_scheduler(optimizer, scheduler_spec)
        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step',
            'name': 'lr'}
        return [optimizer], [scheduler_dict]

    def on_train_start(self):
        self.backbone.train()
        self.mdlm.to_device(self.device)
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)
        
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update(itertools.chain(self.backbone.parameters()))
        
    def forward(self, x, attention_mask=None, t=None):
        with torch.amp.autocast('cuda', dtype=torch.float32):
            if self.diffusion_type == 'udlm':
                if t is None:
                    raise ValueError("UDLM forward passes require one time value per sequence")
                noise_level = self.mdlm.sigma(t.to(device=x.device, dtype=torch.float32))
                return self.backbone(
                    x,
                    attention_mask,
                    noise_level=noise_level,
                )['logits']
            return self.backbone(x, attention_mask)['logits']

    def diffusion_token_mask(self, input_ids, attention_mask):
        """Select molecular content positions while preserving sequence framing."""

        token_mask = attention_mask.to(dtype=torch.bool)
        for token_id in (self.pad_index, self.bos_index, self.eos_index):
            if token_id is not None:
                token_mask &= input_ids != token_id
        return token_mask
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        # sample time
        t = self.mdlm.sample_time(input_ids.shape[0], device=input_ids.device)
        if self.diffusion_type == 'udlm':
            loss_mask = self.diffusion_token_mask(input_ids, attention_mask)
            xt = self.mdlm.forward_process(input_ids, t, mutable_mask=loss_mask)
            logits = self(xt, attention_mask, t=t)
        else:
            loss_mask = attention_mask
            # Forward process to add absorbing mask tokens.
            xt = self.mdlm.forward_process(input_ids, t)
            with torch.amp.autocast('cuda', dtype=torch.float32):
                logits = self.backbone(xt, attention_mask)["logits"]
        # compute loss
        if self.config.training.global_mean_loss:
            loss = self.mdlm.loss(
                logits,
                input_ids,
                xt,
                t,
                mask=loss_mask,
                global_mean=True,
            )
        else:
            loss = self.mdlm.loss(
                logits,
                input_ids,
                xt,
                t,
                mask=loss_mask,
            ).mean()
        self.log(name='train_loss',
                 value=loss.item(),
                 on_step=True,
                 on_epoch=False,
                 prog_bar=True,
                 sync_dist=True)
        return loss

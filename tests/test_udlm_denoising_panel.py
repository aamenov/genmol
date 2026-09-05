import copy
import hashlib
import json
import math
from pathlib import Path

import pytest
import torch
from torch import nn

from genmol.diffusion import (
    ContinuousCategoricalDiffusion,
    ContinuousUniformDiffusion,
)
from scripts.udlm import evaluate_denoising_panel as evaluator
from scripts.udlm.evaluate_denoising_panel import (
    DEFAULT_FREQUENCIES,
    DEFAULT_PANEL,
    FROZEN_FREQUENCY_SHA256,
    FROZEN_PANEL_SHA256,
    SOURCE_INPUTS,
    _atomic_write_json,
    _canonical_numeric_sequence_sha256,
    _load_json,
    _validate_device,
    _validate_output_path,
    apply_ema_weights,
    checkpoint_metadata,
    corruption_seed,
    evaluate_denoising_panel,
    load_verified_checkpoint,
    load_checkpoint_model,
    load_frozen_artifacts,
    runtime_provenance,
    sha256_file,
    source_provenance,
    validate_evaluation_inputs,
    validate_time_bins,
    verify_source_provenance,
)


def _ordered_token_digest(rows):
    digest = hashlib.sha256()
    for row in rows:
        value = json.dumps(row["input_ids"], separators=(",", ":")).encode()
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


def _toy_artifacts():
    rows = [
        {
            "source_index": 0,
            "input_ids": [1, 4, 5, 6, 7, 2],
            "content_length": 4,
            "safe_sha256": "0" * 64,
        },
        {
            "source_index": 3,
            "input_ids": [1, 7, 6, 5, 4, 7, 6, 2],
            "content_length": 6,
            "safe_sha256": "1" * 64,
        },
        {
            "source_index": 9,
            "input_ids": [1, 5, 5, 4, 6, 7, 7, 4, 2],
            "content_length": 7,
            "safe_sha256": "2" * 64,
        },
    ]
    tokenizer = {
        "repo_id": "toy-safe",
        "revision": "tokenizer-revision",
        "tokenizer_json_sha256": "a" * 64,
        "base_vocab_size": 8,
        "special_token_ids": [0, 1, 2, 3],
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 3,
    }
    panel = {
        "schema_version": 1,
        "purpose": "test panel",
        "sample_count": len(rows),
        "max_sequence_length": 256,
        "dataset": {
            "repo_id": "toy-safe",
            "revision": "dataset-revision",
            "split": "validation",
        },
        "tokenizer": tokenizer,
        "ordered_token_ids_sha256": _ordered_token_digest(rows),
        "rows": rows,
    }
    counts = [0, 0, 0, 0, 0, 5, 50, 5_000]
    frequencies = {
        "schema_version": 1,
        "purpose": "test frequencies",
        "example_count": 100,
        "content_token_count": sum(counts),
        "observed_token_types": sum(count > 0 for count in counts),
        "counts_by_token_id": counts,
        "dataset": {
            "repo_id": "toy-safe",
            "revision": "dataset-revision",
            "split": "train",
        },
        "tokenizer": copy.deepcopy(tokenizer),
    }
    return panel, frequencies


class _ToyUDLM(nn.Module):
    """A tiny injected denoiser whose top-1 prediction is the noisy token."""

    diffusion_type = "udlm"

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.mdlm = ContinuousUniformDiffusion(
            8,
            excluded_token_ids=(0, 1, 2, 3),
            sampling_eps=1e-3,
            noise_eps=0.1,
            antithetic_sampling=True,
        )

    def forward(self, input_ids, attention_mask=None, t=None):
        del attention_mask
        assert t is not None
        token_ids = torch.arange(8, device=input_ids.device)
        return -4.0 * (token_ids - input_ids.unsqueeze(-1)).abs() + self.anchor


def _categorical_metadata(
    process,
    panel,
    *,
    variant="schedule_uniform",
    frequencies=None,
    artifact_sha256=None,
    mixture=None,
):
    active_ids = [int(value) for value in process.diffusion_token_ids.tolist()]
    probabilities = [float(value) for value in process.stationary_probs.tolist()]
    identity = evaluator.UDLM_PRIOR_VARIANT_IDENTITIES[variant]
    frequency_fields = {
        "frequency_artifact_path": None,
        "frequency_artifact_sha256": None,
        "frequency_artifact_schema_version": None,
        "frequency_example_count": None,
        "frequency_content_token_count": None,
        "frequency_active_token_count": None,
        "frequency_dataset_repo_id": None,
        "frequency_dataset_revision": None,
        "frequency_dataset_split": None,
        "frequency_dataset_selection": None,
        "frequency_ordered_text_sha256": None,
        "frequency_implementation_git_sha": None,
    }
    if variant == "empirical_frequency":
        assert frequencies is not None
        frequency_fields = {
            "frequency_artifact_path": (
                evaluator.EMPIRICAL_FREQUENCY_RELATIVE_PATH.as_posix()
            ),
            "frequency_artifact_sha256": artifact_sha256,
            "frequency_artifact_schema_version": frequencies["schema_version"],
            "frequency_example_count": frequencies["example_count"],
            "frequency_content_token_count": frequencies["content_token_count"],
            "frequency_active_token_count": sum(
                frequencies["counts_by_token_id"][token_id] for token_id in active_ids
            ),
            "frequency_dataset_repo_id": frequencies["dataset"]["repo_id"],
            "frequency_dataset_revision": frequencies["dataset"]["revision"],
            "frequency_dataset_split": frequencies["dataset"]["split"],
            "frequency_dataset_selection": frequencies["dataset"]["selection"],
            "frequency_ordered_text_sha256": frequencies["dataset"][
                "ordered_safe_text_sha256"
            ],
            "frequency_implementation_git_sha": frequencies["git_sha"],
        }
    return {
        "schema_version": 1,
        "variant": variant,
        **identity,
        "full_vocab_size": process.num_classes,
        "active_vocab_size": process.diffusion_vocab_size,
        "excluded_token_ids": [0, 1, 2, 3],
        "sampling_eps": process.sampling_eps,
        "noise_eps": process.noise_eps,
        "antithetic_sampling": process.antithetic_sampling,
        "active_token_ids_sha256": _canonical_numeric_sequence_sha256(active_ids),
        "stationary_probs_sha256": _canonical_numeric_sequence_sha256(probabilities),
        "uniform_mixture_weight": mixture,
        **frequency_fields,
        "tokenizer_repo_id": panel["tokenizer"]["repo_id"],
        "tokenizer_revision": panel["tokenizer"]["revision"],
        "tokenizer_json_sha256": panel["tokenizer"]["tokenizer_json_sha256"],
    }


class _ToyCategoricalUDLM(_ToyUDLM):
    def __init__(
        self,
        panel,
        *,
        probabilities=(0.25, 0.25, 0.25, 0.25),
        variant="schedule_uniform",
        frequencies=None,
        artifact_sha256=None,
        mixture=None,
    ):
        super().__init__()
        self.mdlm = ContinuousCategoricalDiffusion(
            8,
            probabilities,
            excluded_token_ids=(0, 1, 2, 3),
            sampling_eps=1e-3,
            noise_eps=0.1,
            antithetic_sampling=True,
        )
        self.udlm_prior_metadata = _categorical_metadata(
            self.mdlm,
            panel,
            variant=variant,
            frequencies=frequencies,
            artifact_sha256=artifact_sha256,
            mixture=mixture,
        )


def test_committed_frozen_artifacts_match_pinned_hashes_and_validate():
    panel, frequencies, provenance = load_frozen_artifacts()

    assert sha256_file(DEFAULT_PANEL) == FROZEN_PANEL_SHA256
    assert sha256_file(DEFAULT_FREQUENCIES) == FROZEN_FREQUENCY_SHA256
    assert panel["sample_count"] == 256
    assert frequencies["example_count"] == 10_000
    assert provenance["panel"]["sha256"] == FROZEN_PANEL_SHA256


def test_input_validation_rejects_cross_artifact_tokenizer_mismatch():
    panel, frequencies = _toy_artifacts()
    frequencies["tokenizer"]["revision"] = "different"

    with pytest.raises(ValueError, match="tokenizer mismatch for revision"):
        validate_evaluation_inputs(panel, frequencies)


def test_input_validation_rejects_bad_row_hash_and_inconsistent_counts():
    panel, frequencies = _toy_artifacts()
    bad_panel = copy.deepcopy(panel)
    bad_panel["rows"][0]["safe_sha256"] = "not-a-sha"
    with pytest.raises(ValueError, match="invalid SAFE hash"):
        validate_evaluation_inputs(bad_panel, frequencies)

    bad_frequencies = copy.deepcopy(frequencies)
    bad_frequencies["content_token_count"] += 1
    with pytest.raises(ValueError, match="content_token_count"):
        validate_evaluation_inputs(panel, bad_frequencies)


def test_time_and_seed_validation_is_strict_and_seed_is_stable():
    assert validate_time_bins((0.1, 0.5, 0.9)) == (0.1, 0.5, 0.9)
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_time_bins((0.5, 0.5))
    with pytest.raises(ValueError, match="strictly in"):
        validate_time_bins((0.0, 0.5))

    assert corruption_seed(7, 0.5, 3) == corruption_seed(7, 0.5, 3)
    assert corruption_seed(7, 0.5, 3) != corruption_seed(7, 0.5, 4)


def test_toy_evaluation_is_deterministic_and_preserves_metric_denominators():
    panel, frequencies = _toy_artifacts()
    first = evaluate_denoising_panel(
        _ToyUDLM(),
        panel,
        frequencies,
        time_bins=(0.35, 0.8),
        seed=123,
        batch_size=1,
        device="cpu",
    )
    second = evaluate_denoising_panel(
        _ToyUDLM(),
        panel,
        frequencies,
        time_bins=(0.35, 0.8),
        seed=123,
        batch_size=3,
        device="cpu",
    )

    assert first["content_tokens_per_time_bin"] == 17
    assert first["pooled_token_time_metrics"]["overall"][
        "production_loss_mean"
    ] == pytest.approx(
        second["pooled_token_time_metrics"]["overall"]["production_loss_mean"],
        rel=1e-6,
        abs=1e-8,
    )
    for first_bin, second_bin in zip(
        first["metrics_by_time"], second["metrics_by_time"]
    ):
        assert (
            first_bin["row_corruption_seeds_sha256"]
            == second_bin["row_corruption_seeds_sha256"]
        )
        assert (
            first_bin["corrupted_token_ids_sha256"]
            == second_bin["corrupted_token_ids_sha256"]
        )
        # The corruption grid is exact across batch partitions.  Numerical
        # equality of real neural-network outputs is intentionally not part of
        # the contract, so metric comparisons use a tolerance.
        assert first_bin["metrics"]["overall"]["production_loss_mean"] == pytest.approx(
            second_bin["metrics"]["overall"]["production_loss_mean"],
            rel=1e-6,
            abs=1e-8,
        )

        overall = first_bin["metrics"]["overall"]
        changed = first_bin["metrics"]["by_observed_corruption"]["observed_changed"]
        unchanged = first_bin["metrics"]["by_observed_corruption"]["observed_unchanged"]
        assert overall["denominator_tokens"] == 17
        assert changed["denominator_tokens"] > 0
        assert unchanged["denominator_tokens"] > 0
        assert changed["denominator_tokens"] + unchanged["denominator_tokens"] == 17
        assert changed["clean_token_top1_accuracy"] == 0.0
        assert unchanged["clean_token_top1_accuracy"] == 1.0
        assert overall["production_loss_mean"] >= 0

        frequency_groups = first_bin["metrics"]["by_training_frequency"]
        assert (
            sum(group["denominator_tokens"] for group in frequency_groups.values())
            == 17
        )
        assert frequency_groups["unseen"]["denominator_tokens"] > 0
        assert frequency_groups["count_ge_1000"]["denominator_tokens"] > 0

    assert first["pooled_token_time_metrics"]["overall"]["denominator_tokens"] == 2 * 17
    assert first["corruption_grid_sha256"] == second["corruption_grid_sha256"]
    assert first["seed_protocol"]["corruption_device"] == "cpu"
    assert "not claimed" in first["seed_protocol"]["qualification"]
    assert "not an integrated" in first["estimator_scope"]


def test_evaluation_can_use_a_small_prefix_without_redefining_the_panel():
    panel, frequencies = _toy_artifacts()
    result = evaluate_denoising_panel(
        _ToyUDLM(),
        panel,
        frequencies,
        time_bins=(0.5,),
        seed=5,
        batch_size=2,
        max_rows=1,
    )

    assert result["rows_evaluated"] == 1
    assert result["row_selection"] == "first 1 rows of the frozen ordered panel"
    assert result["content_tokens_per_time_bin"] == 4


def test_schedule_uniform_categorical_endpoint_is_separate_and_time_invariant():
    panel, frequencies = _toy_artifacts()
    model = _ToyCategoricalUDLM(panel)
    metadata = copy.deepcopy(model.udlm_prior_metadata)
    first = evaluate_denoising_panel(
        model,
        panel,
        frequencies,
        time_bins=(0.35,),
        seed=11,
        batch_size=1,
        checkpoint_prior_metadata=metadata,
    )
    second_model = _ToyCategoricalUDLM(panel)
    second = evaluate_denoising_panel(
        second_model,
        panel,
        frequencies,
        time_bins=(0.2, 0.5, 0.8),
        seed=999,
        batch_size=3,
        checkpoint_prior_metadata=copy.deepcopy(second_model.udlm_prior_metadata),
    )
    partition_model = _ToyCategoricalUDLM(panel)
    different_partition = evaluate_denoising_panel(
        partition_model,
        panel,
        frequencies,
        time_bins=(0.35,),
        seed=11,
        batch_size=3,
        checkpoint_prior_metadata=copy.deepcopy(partition_model.udlm_prior_metadata),
    )

    process = first["process"]
    assert process["backend"].endswith(".ContinuousCategoricalDiffusion")
    assert process["prior_variant"] == "schedule_uniform"
    assert process["comparison_role"] == "schedule_repair_uniform_control"
    assert process["valid_single_factor_prior_control_for"] == ("empirical_frequency")
    assert process["training_frequency_artifact_usage"] == [
        "metric_stratification_only"
    ]
    assert process["production_loss_schedule"]["time_bin_beta"] is not None
    assert "not a NELBO" in process["production_loss_schedule"]["qualification"]

    endpoint = first["endpoint_prior_kl"]
    other_endpoint = second["endpoint_prior_kl"]
    assert endpoint["applicable"] is True
    assert endpoint["mathematically_defined_for_residual_forward"] is True
    assert endpoint["reported_by_evaluator"] is True
    assert endpoint["parameter_independent"] is True
    assert endpoint["included_in_production_loss"] is False
    assert "does not produce a NELBO" in endpoint["qualification"]
    assert endpoint["metrics"] == other_endpoint["metrics"]
    assert endpoint["metrics"]["overall"]["denominator_tokens"] == 17

    pi_x = 0.25
    residual = 0.1
    q_x = residual + (1.0 - residual) * pi_x
    expected_per_token = q_x * math.log(q_x / pi_x) + (
        (1.0 - residual) * (1.0 - pi_x) * math.log1p(-residual)
    )
    assert endpoint["metrics"]["overall"]["endpoint_prior_kl_sum"] == (
        pytest.approx(17 * expected_per_token, rel=1e-12, abs=1e-12)
    )
    assert (
        "endpoint_prior_kl_sum" not in first["metrics_by_time"][0]["metrics"]["overall"]
    )
    assert first["process_identity_postcheck"] == "unchanged_after_evaluation"
    assert (
        first["corruption_grid_sha256"] == different_partition["corruption_grid_sha256"]
    )
    assert first["pooled_token_time_metrics"]["overall"][
        "production_loss_mean"
    ] == pytest.approx(
        different_partition["pooled_token_time_metrics"]["overall"][
            "production_loss_mean"
        ],
        rel=1e-6,
        abs=1e-8,
    )
    release = evaluate_denoising_panel(
        _ToyUDLM(),
        panel,
        frequencies,
        time_bins=(0.35,),
        max_rows=1,
    )
    assert (
        release["process"]["prior"]["ordered_probabilities_sha256"]
        == (process["prior"]["ordered_probabilities_sha256"])
    )
    assert release["process"]["backend"] != process["backend"]
    assert release["endpoint_prior_kl"]["applicable"] is False
    assert (
        release["endpoint_prior_kl"]["mathematically_defined_for_residual_forward"]
        is True
    )
    assert release["endpoint_prior_kl"]["reported_by_evaluator"] is False
    assert (
        release["process"]["production_loss_schedule"]["definition"]
        != (process["production_loss_schedule"]["definition"])
    )


def test_empirical_treatment_reconstructs_exact_ordered_prior_and_artifact(monkeypatch):
    panel, frequencies = _toy_artifacts()
    frequencies["dataset"].update(
        {
            "selection": "first 100 toy rows",
            "ordered_safe_text_sha256": "c" * 64,
        }
    )
    frequencies["git_sha"] = "d" * 40
    artifact_sha256 = "b" * 64
    monkeypatch.setattr(evaluator, "FROZEN_FREQUENCY_SHA256", artifact_sha256)
    monkeypatch.setattr(evaluator, "EMPIRICAL_FREQUENCY_ORDERED_TEXT_SHA256", "c" * 64)
    mixture = 0.1
    active_counts = frequencies["counts_by_token_id"][4:]
    active_total = sum(active_counts)
    probabilities = [
        (1.0 - mixture) * (count / active_total) + mixture / len(active_counts)
        for count in active_counts
    ]
    model = _ToyCategoricalUDLM(
        panel,
        probabilities=probabilities,
        variant="empirical_frequency",
        frequencies=frequencies,
        artifact_sha256=artifact_sha256,
        mixture=mixture,
    )
    provenance = {"training_frequency": {"sha256": artifact_sha256}}
    result = evaluate_denoising_panel(
        model,
        panel,
        frequencies,
        time_bins=(0.5,),
        max_rows=1,
        checkpoint_prior_metadata=copy.deepcopy(model.udlm_prior_metadata),
        artifact_provenance=provenance,
    )

    process = result["process"]
    assert process["prior_variant"] == "empirical_frequency"
    assert process["comparison_role"] == "empirical_prior_treatment"
    assert process["valid_single_factor_prior_control_for"] is None
    assert process["prior"]["family"] == (
        "categorical_stationary_over_diffusion_token_ids"
    )
    assert process["training_frequency_artifact_usage"] == [
        "stationary_prior_construction",
        "metric_stratification",
    ]

    permuted = copy.deepcopy(frequencies)
    permuted["counts_by_token_id"][5], permuted["counts_by_token_id"][6] = (
        permuted["counts_by_token_id"][6],
        permuted["counts_by_token_id"][5],
    )
    with pytest.raises(ValueError, match="stationary_probs disagree"):
        evaluate_denoising_panel(
            model,
            panel,
            permuted,
            time_bins=(0.5,),
            max_rows=1,
            checkpoint_prior_metadata=model.udlm_prior_metadata,
            artifact_provenance=provenance,
        )
    with pytest.raises(ValueError, match="not bound"):
        evaluate_denoising_panel(
            model,
            panel,
            frequencies,
            time_bins=(0.5,),
            max_rows=1,
            checkpoint_prior_metadata=model.udlm_prior_metadata,
            artifact_provenance={"training_frequency": {"sha256": "e" * 64}},
        )


def test_checkpoint_metadata_records_exact_bytes_and_udlm_identity(tmp_path: Path):
    checkpoint_path = tmp_path / "tiny.ckpt"
    torch.save(
        {
            "global_step": 12,
            "epoch": 2,
            "hyper_parameters": {
                "config": {
                    "seed": 7,
                    "training": {
                        "diffusion": "udlm",
                        "init_from_mdlm_checkpoint": "source.ckpt",
                        "init_from_mdlm_ema": False,
                    },
                }
            },
        },
        checkpoint_path,
    )

    metadata = checkpoint_metadata(checkpoint_path)

    assert metadata["sha256"] == sha256_file(checkpoint_path)
    assert metadata["size_bytes"] == checkpoint_path.stat().st_size
    assert metadata["global_step"] == 12
    assert metadata["epoch"] == 2
    assert metadata["diffusion_type"] == "udlm"
    assert len(metadata["config_sha256"]) == 64
    assert metadata["training_initialization_declaration"] == {
        "init_from_mdlm_checkpoint": "source.ckpt",
        "init_from_mdlm_ema": False,
        "qualification": metadata["training_initialization_declaration"][
            "qualification"
        ],
    }


def test_json_loader_parses_the_same_bytes_it_hashes(tmp_path: Path, monkeypatch):
    artifact = tmp_path / "artifact.json"
    artifact.write_text('{"value": 1}')
    expected = hashlib.sha256(artifact.read_bytes()).hexdigest()

    def forbidden_reread(_self):
        raise AssertionError("JSON must not be reread after hashing")

    monkeypatch.setattr(Path, "read_text", forbidden_reread)
    value, observed = _load_json(artifact, expected)

    assert value == {"value": 1}
    assert observed == expected


def test_exact_uniform_backend_and_prior_identity_are_enforced():
    panel, frequencies = _toy_artifacts()
    result = evaluate_denoising_panel(
        _ToyUDLM(),
        panel,
        frequencies,
        time_bins=(0.5,),
        max_rows=1,
    )
    process = result["process"]
    assert process["backend"].endswith(".ContinuousUniformDiffusion")
    assert process["backend_exact_type_required"] is True
    assert process["prior"]["family"] == "uniform_over_diffusion_token_ids"
    assert process["prior"]["support_size"] == 4
    assert len(process["prior"]["identity_sha256"]) == 64

    class UnsupportedUniformSubclass(ContinuousUniformDiffusion):
        pass

    model = _ToyUDLM()
    model.mdlm = UnsupportedUniformSubclass(
        8,
        excluded_token_ids=(0, 1, 2, 3),
        sampling_eps=1e-3,
        noise_eps=0.1,
    )
    with pytest.raises(ValueError, match="exact supported UDLM backend"):
        evaluate_denoising_panel(
            model,
            panel,
            frequencies,
            time_bins=(0.5,),
            max_rows=1,
        )


def test_categorical_backend_metadata_and_mapping_are_adversarially_checked():
    panel, frequencies = _toy_artifacts()

    class UnsupportedCategoricalSubclass(ContinuousCategoricalDiffusion):
        pass

    subclass_model = _ToyCategoricalUDLM(panel)
    subclass_model.mdlm = UnsupportedCategoricalSubclass(
        8,
        (0.25, 0.25, 0.25, 0.25),
        excluded_token_ids=(0, 1, 2, 3),
        sampling_eps=1e-3,
        noise_eps=0.1,
    )
    with pytest.raises(ValueError, match="exact supported UDLM backend"):
        evaluate_denoising_panel(
            subclass_model,
            panel,
            frequencies,
            time_bins=(0.5,),
            max_rows=1,
            checkpoint_prior_metadata=subclass_model.udlm_prior_metadata,
        )

    malformed_metadata = _ToyCategoricalUDLM(panel)
    malformed_metadata.udlm_prior_metadata["unexpected"] = True
    with pytest.raises(ValueError, match="extra=.*unexpected"):
        evaluate_denoising_panel(
            malformed_metadata,
            panel,
            frequencies,
            time_bins=(0.5,),
            max_rows=1,
            checkpoint_prior_metadata=malformed_metadata.udlm_prior_metadata,
        )

    malformed_mapping = _ToyCategoricalUDLM(panel)
    malformed_mapping.mdlm.token_to_diffusion_index[4] = 2
    with pytest.raises(ValueError, match="compact-token mapping"):
        evaluate_denoising_panel(
            malformed_mapping,
            panel,
            frequencies,
            time_bins=(0.5,),
            max_rows=1,
            checkpoint_prior_metadata=malformed_mapping.udlm_prior_metadata,
        )

    malformed_prior = _ToyCategoricalUDLM(panel)
    malformed_prior.mdlm.stationary_probs[0] += 0.01
    malformed_prior.mdlm.stationary_probs[1] -= 0.01
    with pytest.raises(ValueError, match="stationary_probs_sha256"):
        evaluate_denoising_panel(
            malformed_prior,
            panel,
            frequencies,
            time_bins=(0.5,),
            max_rows=1,
            checkpoint_prior_metadata=malformed_prior.udlm_prior_metadata,
        )

    wrong_dtype = _ToyCategoricalUDLM(panel)
    wrong_dtype.mdlm.stationary_probs = wrong_dtype.mdlm.stationary_probs.float()
    with pytest.raises(ValueError, match="normalized float64"):
        evaluate_denoising_panel(
            wrong_dtype,
            panel,
            frequencies,
            time_bins=(0.5,),
            max_rows=1,
            checkpoint_prior_metadata=wrong_dtype.udlm_prior_metadata,
        )

    class MutatesPriorDuringForward(_ToyCategoricalUDLM):
        def forward(self, input_ids, attention_mask=None, t=None):
            logits = super().forward(input_ids, attention_mask=attention_mask, t=t)
            if self.mdlm.stationary_probs[0] == 0.25:
                self.mdlm.stationary_probs[0] += 0.01
                self.mdlm.stationary_probs[1] -= 0.01
            return logits

    mutating = MutatesPriorDuringForward(panel)
    with pytest.raises(ValueError, match="stationary_probs_sha256"):
        evaluate_denoising_panel(
            mutating,
            panel,
            frequencies,
            time_bins=(0.5,),
            max_rows=1,
            checkpoint_prior_metadata=mutating.udlm_prior_metadata,
        )


class _TinyTokenizer:
    vocab_size = 8
    all_special_ids = [0, 1, 2, 3]
    mask_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 3


def _tiny_real_udlm_config(prior_variant):
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "model": {
                "attention_probs_dropout_prob": 0.0,
                "classifier_dropout": None,
                "hidden_act": "gelu",
                "hidden_dropout_prob": 0.0,
                "hidden_size": 8,
                "initializer_range": 0.02,
                "intermediate_size": 16,
                "layer_norm_eps": 1e-12,
                "max_position_embeddings": 16,
                "model_type": "bert",
                "num_attention_heads": 2,
                "num_hidden_layers": 1,
                "pad_token_id": 3,
                "position_embedding_type": "absolute",
                "torch_dtype": "float32",
                "type_vocab_size": 2,
                "use_cache": True,
                "vocab_size": 8,
            },
            "training": {
                "diffusion": "udlm",
                "ema": 0.0,
                "antithetic_sampling": True,
                "sampling_eps": 1e-3,
                "udlm": {
                    "exclude_special_tokens": True,
                    "prior_variant": prior_variant,
                    "empirical_uniform_mix": 0.01,
                    "noise_eps": 0.1,
                    "time_embedding_size": 8,
                    "zero_init_conditioning": True,
                },
            },
        }
    )


def _model_compatible_toy_artifacts(model_module):
    panel, frequencies = _toy_artifacts()
    identity = {
        "repo_id": model_module.SAFE_GPT_REPO_ID,
        "revision": model_module.SAFE_GPT_TOKENIZER_REVISION,
        "tokenizer_json_sha256": model_module.SAFE_GPT_TOKENIZER_SHA256,
    }
    panel["tokenizer"].update(identity)
    frequencies["tokenizer"].update(identity)
    return panel, frequencies


def test_tiny_real_categorical_checkpoint_cpu_smoke_and_metadata_attacks(
    tmp_path: Path, monkeypatch
):
    from genmol import model as model_module

    monkeypatch.setattr(model_module, "get_tokenizer", lambda: _TinyTokenizer())
    config = _tiny_real_udlm_config("schedule_uniform")
    source_model = model_module.GenMol(config)
    checkpoint = {
        "global_step": 7,
        "epoch": 0,
        "hyper_parameters": {"config": config},
        "state_dict": {
            key: value.detach().clone()
            for key, value in source_model.state_dict().items()
        },
        evaluator.UDLM_PRIOR_CHECKPOINT_KEY: (
            source_model.udlm_prior_metadata.to_dict()
        ),
    }
    checkpoint_path = tmp_path / "tiny-categorical.ckpt"
    torch.save(checkpoint, checkpoint_path)
    loaded_checkpoint, snapshot = load_verified_checkpoint(checkpoint_path)
    assert snapshot["sha256"] == sha256_file(checkpoint_path)
    loaded_model, weight_provenance = load_checkpoint_model(loaded_checkpoint, "raw")
    assert (
        weight_provenance["udlm_process_identity"]["checkpoint_prior_metadata_required"]
        is True
    )

    panel, frequencies = _model_compatible_toy_artifacts(model_module)
    result = evaluate_denoising_panel(
        loaded_model,
        panel,
        frequencies,
        time_bins=(0.5,),
        batch_size=2,
        checkpoint_prior_metadata=loaded_checkpoint[
            evaluator.UDLM_PRIOR_CHECKPOINT_KEY
        ],
    )
    assert result["process"]["prior_variant"] == "schedule_uniform"
    assert result["endpoint_prior_kl"]["metrics"]["overall"]["denominator_tokens"] == 17
    assert math.isfinite(
        result["pooled_token_time_metrics"]["overall"]["production_loss_mean"]
    )

    attacks = []
    missing = copy.deepcopy(checkpoint)
    missing.pop(evaluator.UDLM_PRIOR_CHECKPOINT_KEY)
    attacks.append(missing)
    for field, value in (
        ("schema_version", True),
        ("variant", "empirical_frequency"),
        ("schedule_variant", "forged"),
        ("stationary_probs_sha256", "f" * 64),
        ("frequency_artifact_sha256", "f" * 64),
        ("noise_eps", 0.2),
    ):
        tampered = copy.deepcopy(checkpoint)
        tampered[evaluator.UDLM_PRIOR_CHECKPOINT_KEY][field] = value
        attacks.append(tampered)
    extra = copy.deepcopy(checkpoint)
    extra[evaluator.UDLM_PRIOR_CHECKPOINT_KEY]["unexpected"] = True
    attacks.append(extra)
    for tampered in attacks:
        with pytest.raises(ValueError, match="categorical UDLM checkpoint"):
            load_checkpoint_model(tampered, "raw")

    process_attacks = []
    wrong_prior = copy.deepcopy(checkpoint)
    wrong_prior["state_dict"]["mdlm.stationary_probs"][0] += 0.01
    wrong_prior["state_dict"]["mdlm.stationary_probs"][1] -= 0.01
    process_attacks.append(wrong_prior)
    wrong_mapping = copy.deepcopy(checkpoint)
    wrong_mapping["state_dict"]["mdlm.token_to_diffusion_index"][4] = 2
    process_attacks.append(wrong_mapping)
    wrong_ids = copy.deepcopy(checkpoint)
    wrong_ids["state_dict"]["mdlm.diffusion_token_ids"][[0, 1]] = wrong_ids[
        "state_dict"
    ]["mdlm.diffusion_token_ids"][[1, 0]]
    process_attacks.append(wrong_ids)
    wrong_prior_dtype = copy.deepcopy(checkpoint)
    wrong_prior_dtype["state_dict"]["mdlm.stationary_probs"] = wrong_prior_dtype[
        "state_dict"
    ]["mdlm.stationary_probs"].float()
    process_attacks.append(wrong_prior_dtype)
    for tampered in process_attacks:
        with pytest.raises(ValueError, match="checkpoint"):
            load_checkpoint_model(tampered, "raw")

    release_config = _tiny_real_udlm_config("release_uniform")
    release_model = model_module.GenMol(release_config)
    release_checkpoint = {
        "hyper_parameters": {"config": release_config},
        "state_dict": {
            key: value.detach().clone()
            for key, value in release_model.state_dict().items()
        },
        evaluator.UDLM_PRIOR_CHECKPOINT_KEY: checkpoint[
            evaluator.UDLM_PRIOR_CHECKPOINT_KEY
        ],
    }
    with pytest.raises(ValueError, match="release_uniform checkpoint"):
        load_checkpoint_model(release_checkpoint, "raw")


class _TinyBackboneHolder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1))


def _tiny_weight_inputs():
    model = _TinyBackboneHolder()
    state = {
        f"backbone.{name}": parameter.detach().clone()
        for name, parameter in model.backbone.named_parameters()
    }
    shadows = [
        torch.full_like(parameter, index + 1)
        for index, parameter in enumerate(model.backbone.parameters())
    ]
    return model, state, shadows


def test_ema_mapping_validates_every_name_shape_and_dtype_before_copy():
    model, state, shadows = _tiny_weight_inputs()
    provenance = apply_ema_weights(model, state, {"shadow_params": shadows})
    assert provenance["parameter_tensors"] == 4
    assert len(provenance["ordered_parameter_names_sha256"]) == 64
    for parameter, expected in zip(model.backbone.parameters(), shadows, strict=True):
        assert torch.equal(parameter, expected)

    for malformed in (
        shadows[:-1],
        [*shadows[:-1], torch.zeros(2, dtype=shadows[-1].dtype)],
        [*shadows[:-1], shadows[-1].double()],
    ):
        candidate, candidate_state, _candidate_shadows = _tiny_weight_inputs()
        before = [parameter.detach().clone() for parameter in candidate.parameters()]
        with pytest.raises(ValueError, match="EMA"):
            apply_ema_weights(candidate, candidate_state, {"shadow_params": malformed})
        assert all(
            torch.equal(parameter, original)
            for parameter, original in zip(candidate.parameters(), before, strict=True)
        )

    candidate, candidate_state, candidate_shadows = _tiny_weight_inputs()
    candidate_state.pop("backbone.0.weight")
    with pytest.raises(ValueError, match="backbone.0.weight"):
        apply_ema_weights(
            candidate, candidate_state, {"shadow_params": candidate_shadows}
        )


def test_verified_checkpoint_loads_once_and_rejects_a_changed_snapshot(
    tmp_path: Path, monkeypatch
):
    checkpoint_path = tmp_path / "checkpoint.ckpt"
    checkpoint_path.write_bytes(b"stable-placeholder")
    load_calls = []

    def fake_load(path, **_kwargs):
        load_calls.append(path)
        return {"state_dict": {}}

    monkeypatch.setattr(torch, "load", fake_load)
    checkpoint, snapshot = load_verified_checkpoint(checkpoint_path)
    assert checkpoint == {"state_dict": {}}
    assert snapshot["sha256"] == hashlib.sha256(b"stable-placeholder").hexdigest()
    assert len(load_calls) == 1

    hashes = iter(("a" * 64, "b" * 64))
    monkeypatch.setattr(evaluator, "sha256_file", lambda _path: next(hashes))
    load_calls.clear()
    with pytest.raises(RuntimeError, match="changed while it was being loaded"):
        load_verified_checkpoint(checkpoint_path)
    assert len(load_calls) == 1


def test_cli_device_validation_is_explicitly_cpu_only():
    _validate_device("cpu")
    with pytest.raises(ValueError, match="CPU-only"):
        _validate_device("cuda:0")
    with pytest.raises(ValueError, match="CPU-only"):
        _validate_device("mps")
    panel, frequencies = _toy_artifacts()
    with pytest.raises(ValueError, match="CPU-only"):
        evaluate_denoising_panel(
            _ToyUDLM(),
            panel,
            frequencies,
            time_bins=(0.5,),
            max_rows=1,
            device="cuda:0",
        )


def test_source_provenance_distinguishes_unknown_git_and_is_rechecked(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / "implementation.py"
    source.write_text("before\n")
    monkeypatch.setattr(evaluator, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(evaluator, "SOURCE_INPUTS", (Path("implementation.py"),))
    monkeypatch.setattr(evaluator, "_git_output", lambda _arguments: None)

    provenance = source_provenance(validate_loaded_modules=False)
    assert provenance["git_worktree_state"] == "unknown"
    assert provenance["git_dirty"] is None
    verify_source_provenance(provenance)

    source.write_text("after\n")
    with pytest.raises(RuntimeError, match="implementation inputs changed"):
        verify_source_provenance(provenance)


def test_runtime_and_source_inputs_cover_execution_and_training_dependencies():
    assert Path("src/genmol/utils/ema.py") in SOURCE_INPUTS
    assert Path("scripts/udlm/launch_train_pilot.py") in SOURCE_INPUTS
    assert Path("scripts/train.py") in SOURCE_INPUTS
    assert Path("configs/base.yaml") in SOURCE_INPUTS
    assert Path("configs/udlm.yaml") in SOURCE_INPUTS
    assert Path("configs/udlm_categorical.yaml") in SOURCE_INPUTS
    assert Path("scripts/udlm/token_frequency_audit.py") in SOURCE_INPUTS
    runtime = runtime_provenance()
    assert set(runtime["packages"]) == {
        "lightning",
        "transformers",
        "tokenizers",
        "omegaconf",
        "hydra-core",
        "bionemo-moco",
    }
    assert "CPU-only" in runtime["evaluation_device_policy"]


def test_atomic_json_uses_unique_private_temps_and_never_clobbers(
    tmp_path: Path, monkeypatch
):
    # The production guard confines reports to the worktree. Isolate this unit
    # test to its pytest-owned directory while retaining all write semantics.
    monkeypatch.setattr(evaluator, "REPOSITORY_ROOT", tmp_path)
    created = []
    real_mkstemp = evaluator.tempfile.mkstemp

    def recording_mkstemp(**kwargs):
        descriptor, name = real_mkstemp(**kwargs)
        created.append(name)
        return descriptor, name

    monkeypatch.setattr(evaluator.tempfile, "mkstemp", recording_mkstemp)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _atomic_write_json(first, {"value": 1}, force=False)
    _atomic_write_json(second, {"value": 2}, force=False)
    assert len(created) == len(set(created)) == 2
    assert json.loads(first.read_text()) == {"value": 1}
    assert all(not Path(name).exists() for name in created)

    with pytest.raises(FileExistsError, match="pass --force"):
        _atomic_write_json(first, {"value": 3}, force=False)
    assert json.loads(first.read_text()) == {"value": 1}
    _atomic_write_json(first, {"value": 3}, force=True)
    assert json.loads(first.read_text()) == {"value": 3}


def test_bound_output_path_is_retained_and_still_protects_inputs(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(evaluator, "REPOSITORY_ROOT", tmp_path)
    original_directory = tmp_path / "original"
    redirected_directory = tmp_path / "redirected"
    original_directory.mkdir()
    redirected_directory.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(original_directory, target_is_directory=True)
    bound_output = _validate_output_path(alias / "report.json", force=False)
    assert bound_output == original_directory / "report.json"

    alias.unlink()
    alias.symlink_to(redirected_directory, target_is_directory=True)
    _atomic_write_json(
        bound_output,
        {"value": 1},
        force=False,
        protected_paths=(),
        path_is_prevalidated_and_resolved=True,
    )
    assert json.loads((original_directory / "report.json").read_text()) == {"value": 1}
    assert not (redirected_directory / "report.json").exists()

    protected = (original_directory / "input.json").resolve()
    protected.write_text("input")
    with pytest.raises(ValueError, match="evaluation input"):
        _atomic_write_json(
            protected,
            {"value": 2},
            force=True,
            protected_paths=(protected,),
            path_is_prevalidated_and_resolved=True,
        )
    assert protected.read_text() == "input"


def test_canonical_artifacts_are_always_protected_from_force_output(
    tmp_path: Path, monkeypatch
):
    canonical_panel = tmp_path / "canonical-panel.json"
    canonical_frequency = tmp_path / "canonical-frequency.json"
    alternate_panel = tmp_path / "alternate-panel.json"
    alternate_frequency = tmp_path / "alternate-frequency.json"
    checkpoint = tmp_path / "model.ckpt"
    for path in (
        canonical_panel,
        canonical_frequency,
        alternate_panel,
        alternate_frequency,
        checkpoint,
    ):
        path.write_text("input")
    monkeypatch.setattr(evaluator, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(evaluator, "DEFAULT_PANEL", canonical_panel)
    monkeypatch.setattr(evaluator, "DEFAULT_FREQUENCIES", canonical_frequency)
    monkeypatch.setattr(evaluator, "SOURCE_INPUTS", ())

    protected = evaluator._protected_evaluation_paths(
        checkpoint, alternate_panel, alternate_frequency
    )

    assert canonical_panel.resolve() in protected
    assert canonical_frequency.resolve() in protected
    with pytest.raises(ValueError, match="evaluation input"):
        evaluator._validate_output_path(
            canonical_frequency,
            force=True,
            protected_paths=protected,
        )

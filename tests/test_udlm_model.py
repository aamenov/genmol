import copy
import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from transformers import BertForMaskedLM

import genmol.model as model_module
from genmol.backbone import TimeConditionedBertForMaskedLM
from genmol.diffusion import (
    ContinuousCategoricalDiffusion,
    ContinuousUniformDiffusion,
)


class _Tokenizer:
    vocab_size = 11
    mask_token_id = 4
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 3
    all_special_ids = [0, 1, 2, 3, 4]


def _config(
    *,
    diffusion=None,
    exclude_special=False,
    prior_variant=None,
    empirical_uniform_mix=0.01,
):
    udlm = {
        "exclude_special_tokens": exclude_special,
        "noise_eps": 1e-3,
        "time_embedding_size": 8,
        "zero_init_conditioning": True,
    }
    if prior_variant is not None:
        udlm["prior_variant"] = prior_variant
    if empirical_uniform_mix is not None:
        udlm["empirical_uniform_mix"] = empirical_uniform_mix
    training = {
        "ema": 0.0,
        "antithetic_sampling": True,
        "sampling_eps": 1e-3,
        "global_mean_loss": True,
        "use_bracket_safe": False,
        "udlm": udlm,
    }
    if diffusion is not None:
        training["diffusion"] = diffusion
    return OmegaConf.create(
        {
            "model": {
                "vocab_size": 11,
                "hidden_size": 24,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "intermediate_size": 48,
                "max_position_embeddings": 16,
                "pad_token_id": 3,
                "type_vocab_size": 2,
            },
            "training": training,
            "optim": {
                "lr": 3e-4,
                "beta1": 0.9,
                "beta2": 0.999,
                "eps": 1e-8,
                "weight_decay": 0.0,
            },
        }
    )


@pytest.fixture(autouse=True)
def _fake_tokenizer(monkeypatch):
    monkeypatch.setattr(model_module, "get_tokenizer", lambda: _Tokenizer())


def _frequency_payload(counts=None):
    if counts is None:
        counts = [0, 0, 0, 0, 0, 10, 0, 30, 20, 0, 40]
    return {
        "schema_version": 1,
        "purpose": "CPU-only token-frequency diagnostic; not benchmark evidence",
        "dataset": {
            "repo_id": model_module.SAFE_GPT_REPO_ID,
            "revision": model_module.SAFE_GPT_DATASET_REVISION,
            "split": "train",
            "selection": "first 10000 streaming rows",
            "ordered_safe_text_sha256": (
                model_module.EMPIRICAL_FREQUENCY_ORDERED_TEXT_SHA256
            ),
        },
        "tokenizer": {
            "repo_id": model_module.SAFE_GPT_REPO_ID,
            "revision": model_module.SAFE_GPT_TOKENIZER_REVISION,
            "tokenizer_json_sha256": model_module.SAFE_GPT_TOKENIZER_SHA256,
            "base_vocab_size": 11,
            "special_token_ids": [0, 1, 2, 3, 4],
        },
        "example_count": 10_000,
        "content_token_count": sum(counts),
        "counts_by_token_id": counts,
        "max_sequence_length": 256,
        "git_sha": "1" * 40,
    }


def _install_frequency_artifact(monkeypatch, tmp_path, payload):
    artifact_path = tmp_path / "frequencies.json"
    artifact_bytes = (json.dumps(payload, sort_keys=True) + "\n").encode()
    artifact_path.write_bytes(artifact_bytes)
    monkeypatch.setattr(model_module, "EMPIRICAL_FREQUENCY_PATH", artifact_path)
    monkeypatch.setattr(
        model_module,
        "EMPIRICAL_FREQUENCY_SHA256",
        hashlib.sha256(artifact_bytes).hexdigest(),
    )
    return artifact_path


def _run_checkpoint_save_hook(monkeypatch, model):
    monkeypatch.setattr(model_module, "clean_checkpoint", lambda *_args: None)
    model._trainer = SimpleNamespace(
        accumulate_grad_batches=1,
        train_dataloader=SimpleNamespace(sampler=object()),
    )
    checkpoint = {
        "state_dict": {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }
    }
    model.on_save_checkpoint(checkpoint)
    return checkpoint


def test_missing_diffusion_setting_is_strict_legacy_mdlm_default():
    model = model_module.GenMol(_config())

    assert model.diffusion_type == "mdlm"
    assert isinstance(model.backbone, BertForMaskedLM)
    assert not isinstance(model.backbone, TimeConditionedBertForMaskedLM)
    assert not any("time_conditioner" in key for key in model.state_dict())


def test_udlm_missing_prior_selector_is_release_uniform():
    model = model_module.GenMol(_config(diffusion="udlm"))

    assert model.diffusion_type == "udlm"
    assert isinstance(model.backbone, TimeConditionedBertForMaskedLM)
    assert type(model.mdlm) is ContinuousUniformDiffusion
    assert model.udlm_prior_metadata.variant == "release_uniform"
    assert model.udlm_prior_metadata.comparison_role == "faithful_release_control"
    assert model.udlm_prior_metadata.schedule_variant == (
        "released_ideal_loss_residual_forward"
    )
    assert model.udlm_prior_metadata.frequency_artifact_sha256 is None
    assert "mdlm.stationary_probs" not in model.state_dict()
    assert isinstance(model.mdlm, ContinuousUniformDiffusion)
    assert "backbone.time_conditioner.mlp.2.weight" in model.state_dict()


def test_explicit_release_uniform_has_strict_old_udlm_state_keys():
    old_config_model = model_module.GenMol(_config(diffusion="udlm"))
    explicit_model = model_module.GenMol(
        _config(diffusion="udlm", prior_variant="release_uniform")
    )

    assert old_config_model.state_dict().keys() == explicit_model.state_dict().keys()
    result = explicit_model.load_state_dict(old_config_model.state_dict(), strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []


def test_schedule_uniform_uses_categorical_process_with_exact_uniform_prior():
    released = model_module.GenMol(
        _config(diffusion="udlm", prior_variant="release_uniform")
    )
    model = model_module.GenMol(
        _config(diffusion="udlm", prior_variant="schedule_uniform")
    )
    expected = torch.full((11,), 1.0 / 11, dtype=torch.float64)
    expected /= expected.sum()

    assert type(model.mdlm) is ContinuousCategoricalDiffusion
    assert torch.equal(model.mdlm.stationary_probs, expected)
    assert model.udlm_prior_metadata.variant == "schedule_uniform"
    assert model.udlm_prior_metadata.comparison_role == (
        "schedule_repair_uniform_control"
    )
    assert model.udlm_prior_metadata.process_family == (
        "rank_one_continuous_categorical"
    )
    assert (
        model.udlm_prior_metadata.stationary_probs_sha256
        == released.udlm_prior_metadata.stationary_probs_sha256
    )
    assert "mdlm.stationary_probs" in model.state_dict()


def test_empirical_prior_exactly_mixes_active_frequencies_with_uniform(
    monkeypatch, tmp_path
):
    counts = [0, 0, 0, 0, 0, 10, 0, 30, 20, 0, 40]
    artifact_path = _install_frequency_artifact(
        monkeypatch, tmp_path, _frequency_payload(counts)
    )
    model = model_module.GenMol(
        _config(
            diffusion="udlm",
            prior_variant="empirical_frequency",
            empirical_uniform_mix=0.2,
        )
    )
    empirical_weight = 1.0 - 0.2
    expected = torch.tensor(
        [empirical_weight * (count / 100) + 0.2 / 11 for count in counts],
        dtype=torch.float64,
    )
    expected /= expected.sum()
    metadata = model.udlm_prior_metadata

    assert type(model.mdlm) is ContinuousCategoricalDiffusion
    assert torch.equal(model.mdlm.stationary_probs, expected)
    assert torch.all(model.mdlm.stationary_probs > 0)
    assert metadata.variant == "empirical_frequency"
    assert metadata.comparison_role == "empirical_prior_treatment"
    assert metadata.process_family == "rank_one_continuous_categorical"
    assert metadata.schedule_variant == (
        "schedule_consistent_residual_forward_and_loss"
    )
    assert metadata.objective_scope == (
        "model_dependent_ct_integrand_without_parameter_independent_endpoint_kl"
    )
    assert metadata.uniform_mixture_weight == 0.2
    assert metadata.frequency_artifact_sha256 == hashlib.sha256(
        artifact_path.read_bytes()
    ).hexdigest()
    assert metadata.frequency_content_token_count == 100
    assert metadata.frequency_active_token_count == 100
    assert metadata.frequency_dataset_revision == model_module.SAFE_GPT_DATASET_REVISION
    assert metadata.frequency_dataset_selection == "first 10000 streaming rows"
    assert metadata.frequency_ordered_text_sha256 == (
        model_module.EMPIRICAL_FREQUENCY_ORDERED_TEXT_SHA256
    )


def test_empirical_exclusion_uses_exact_compact_diffusion_token_order(
    monkeypatch, tmp_path
):
    counts = [0, 0, 0, 0, 0, 10, 0, 30, 20, 0, 40]
    _install_frequency_artifact(monkeypatch, tmp_path, _frequency_payload(counts))
    model = model_module.GenMol(
        _config(
            diffusion="udlm",
            exclude_special=True,
            prior_variant="empirical_frequency",
            empirical_uniform_mix=0.2,
        )
    )
    active_ids = [5, 6, 7, 8, 9, 10]
    empirical_weight = 1.0 - 0.2
    expected = torch.tensor(
        [
            empirical_weight * (counts[token_id] / 100) + 0.2 / 6
            for token_id in active_ids
        ],
        dtype=torch.float64,
    )
    expected /= expected.sum()

    assert model.mdlm.diffusion_token_ids.tolist() == active_ids
    assert torch.equal(model.mdlm.stationary_probs, expected)
    assert model.udlm_prior_metadata.excluded_token_ids == (0, 1, 2, 3, 4)
    assert model.udlm_prior_metadata.active_vocab_size == 6


def test_categorical_uniform_control_and_empirical_treatment_share_process(
    monkeypatch, tmp_path
):
    _install_frequency_artifact(monkeypatch, tmp_path, _frequency_payload())
    control = model_module.GenMol(
        _config(diffusion="udlm", prior_variant="schedule_uniform")
    )
    treatment = model_module.GenMol(
        _config(diffusion="udlm", prior_variant="empirical_frequency")
    )

    assert type(control.mdlm) is type(treatment.mdlm)
    assert (
        control.udlm_prior_metadata.process_family
        == treatment.udlm_prior_metadata.process_family
    )
    assert (
        control.udlm_prior_metadata.schedule_variant
        == treatment.udlm_prior_metadata.schedule_variant
    )
    assert control.state_dict().keys() == treatment.state_dict().keys()
    assert not torch.equal(
        control.mdlm.stationary_probs, treatment.mdlm.stationary_probs
    )
    assert (
        control.udlm_prior_metadata.stationary_probs_sha256
        != treatment.udlm_prior_metadata.stationary_probs_sha256
    )


def test_empirical_variant_rejects_an_exactly_uniform_frequency_artifact(
    monkeypatch, tmp_path
):
    counts = [10] * 11
    payload = _frequency_payload(counts)
    payload["tokenizer"]["special_token_ids"] = []
    _install_frequency_artifact(monkeypatch, tmp_path, payload)

    class NoSpecialTokenizer(_Tokenizer):
        all_special_ids = []

    monkeypatch.setattr(model_module, "get_tokenizer", lambda: NoSpecialTokenizer())
    with pytest.raises(ValueError, match="exactly uniform"):
        model_module.GenMol(
            _config(diffusion="udlm", prior_variant="empirical_frequency")
        )


def test_prior_metadata_is_frozen_and_dict_export_is_detached():
    model = model_module.GenMol(
        _config(diffusion="udlm", prior_variant="schedule_uniform")
    )

    with pytest.raises(FrozenInstanceError):
        model.udlm_prior_metadata.variant = "empirical_frequency"
    with pytest.raises(AttributeError):
        model.udlm_prior_metadata = None
    exported = model.udlm_prior_metadata.to_dict()
    exported["excluded_token_ids"].append(10)
    assert model.udlm_prior_metadata.excluded_token_ids == ()


def test_categorical_checkpoint_persists_and_validates_exact_prior_metadata(
    monkeypatch,
):
    source = model_module.GenMol(
        _config(diffusion="udlm", prior_variant="schedule_uniform")
    )
    checkpoint = _run_checkpoint_save_hook(monkeypatch, source)
    saved_metadata = checkpoint[model_module.UDLM_PRIOR_CHECKPOINT_KEY]

    assert saved_metadata == source.udlm_prior_metadata.to_dict()
    json.dumps(saved_metadata)

    restored = model_module.GenMol(
        _config(diffusion="udlm", prior_variant="schedule_uniform")
    )
    monkeypatch.setattr(model_module, "fast_forward_info", lambda _checkpoint: (0, 0))
    restored.on_load_checkpoint(checkpoint)
    result = restored.load_state_dict(checkpoint["state_dict"], strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []

    wrong_metadata = copy.deepcopy(checkpoint)
    wrong_metadata[model_module.UDLM_PRIOR_CHECKPOINT_KEY]["noise_eps"] = 0.02
    with pytest.raises(ValueError, match="metadata does not match"):
        restored.on_load_checkpoint(wrong_metadata)

    missing_metadata = copy.deepcopy(checkpoint)
    del missing_metadata[model_module.UDLM_PRIOR_CHECKPOINT_KEY]
    with pytest.raises(ValueError, match="missing immutable prior metadata"):
        restored.on_load_checkpoint(missing_metadata)


def test_release_uniform_checkpoint_hook_keeps_legacy_top_level_schema(monkeypatch):
    model = model_module.GenMol(
        _config(diffusion="udlm", prior_variant="release_uniform")
    )
    checkpoint = _run_checkpoint_save_hook(monkeypatch, model)

    assert model_module.UDLM_PRIOR_CHECKPOINT_KEY not in checkpoint
    monkeypatch.setattr(model_module, "fast_forward_info", lambda _checkpoint: (0, 0))
    model.on_load_checkpoint(checkpoint)


def test_categorical_state_cannot_overwrite_a_different_configured_prior(
    monkeypatch, tmp_path
):
    _install_frequency_artifact(monkeypatch, tmp_path, _frequency_payload())
    uniform = model_module.GenMol(
        _config(diffusion="udlm", prior_variant="schedule_uniform")
    )
    empirical = model_module.GenMol(
        _config(diffusion="udlm", prior_variant="empirical_frequency")
    )

    with pytest.raises(ValueError, match="stationary prior disagrees"):
        uniform.load_state_dict(empirical.state_dict(), strict=True)


def test_runtime_prior_mutation_is_detected_before_checkpoint_save(monkeypatch):
    model = model_module.GenMol(
        _config(diffusion="udlm", prior_variant="schedule_uniform")
    )
    with torch.no_grad():
        model.mdlm.stationary_probs[0] += 0.01

    with pytest.raises(RuntimeError, match="stationary prior disagrees"):
        _run_checkpoint_save_hook(monkeypatch, model)


@pytest.mark.parametrize(
    ("attribute", "changed_value"),
    [
        ("num_classes", 12),
        ("sampling_eps", 0.02),
        ("noise_eps", 0.02),
        ("antithetic_sampling", False),
    ],
)
def test_runtime_schedule_and_sampler_mutation_is_detected(
    monkeypatch, attribute, changed_value
):
    model = model_module.GenMol(
        _config(diffusion="udlm", prior_variant="schedule_uniform")
    )
    setattr(model.mdlm, attribute, changed_value)

    with pytest.raises(RuntimeError, match="vocabulary, schedule, or time sampler"):
        _run_checkpoint_save_hook(monkeypatch, model)


def test_runtime_process_class_must_exactly_match_declared_variant(monkeypatch):
    model = model_module.GenMol(
        _config(diffusion="udlm", prior_variant="schedule_uniform")
    )
    model.mdlm = ContinuousUniformDiffusion(num_classes=11)

    with pytest.raises(RuntimeError, match="process class"):
        _run_checkpoint_save_hook(monkeypatch, model)


def test_runtime_compact_mapping_mutation_is_detected(monkeypatch):
    model = model_module.GenMol(
        _config(diffusion="udlm", prior_variant="schedule_uniform")
    )
    model.mdlm.token_to_diffusion_index[5] = 0

    with pytest.raises(RuntimeError, match="compact-token mapping"):
        _run_checkpoint_save_hook(monkeypatch, model)


def test_checkpoint_active_alphabet_cannot_override_configured_exclusion():
    full_alphabet = model_module.GenMol(
        _config(diffusion="udlm", prior_variant="schedule_uniform")
    )
    excluded_specials = model_module.GenMol(
        _config(
            diffusion="udlm",
            prior_variant="schedule_uniform",
            exclude_special=True,
        )
    )

    with pytest.raises(ValueError, match="process buffer"):
        full_alphabet.load_state_dict(excluded_specials.state_dict(), strict=True)


def test_unknown_prior_variant_is_rejected():
    with pytest.raises(ValueError, match="prior_variant"):
        model_module.GenMol(_config(diffusion="udlm", prior_variant="mystery"))


@pytest.mark.parametrize(
    "weight",
    [None, True, False, 0.0, -0.1, 1.0, 1.01, float("nan"), float("inf")],
)
def test_empirical_uniform_mix_requires_finite_strictly_positive_probability(
    weight,
):
    with pytest.raises(ValueError, match="empirical_uniform_mix"):
        model_module.GenMol(
            _config(
                diffusion="udlm",
                prior_variant="empirical_frequency",
                empirical_uniform_mix=weight,
            )
        )


def test_frequency_artifact_tampering_is_rejected_before_parsing(
    monkeypatch, tmp_path
):
    artifact_path = _install_frequency_artifact(
        monkeypatch, tmp_path, _frequency_payload()
    )
    artifact_path.write_bytes(artifact_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        model_module.GenMol(
            _config(diffusion="udlm", prior_variant="empirical_frequency")
        )


@pytest.mark.parametrize(
    "case",
    [
        "schema_bool",
        "schema_unknown",
        "purpose",
        "dataset_not_object",
        "dataset_revision",
        "dataset_selection",
        "dataset_digest",
        "max_length",
        "tokenizer_not_object",
        "vocab_bool",
        "vocab_mismatch",
        "tokenizer_hash",
        "tokenizer_revision",
        "special_ids_unsorted",
        "counts_short",
        "count_bool",
        "count_negative",
        "count_sum",
        "special_count",
        "example_bool",
        "example_count",
        "git_sha",
    ],
)
def test_frequency_artifact_schema_and_counts_are_strict(
    monkeypatch, tmp_path, case
):
    payload = _frequency_payload()
    if case == "schema_bool":
        payload["schema_version"] = True
    elif case == "schema_unknown":
        payload["schema_version"] = 2
    elif case == "purpose":
        payload["purpose"] = "benchmark evidence"
    elif case == "dataset_not_object":
        payload["dataset"] = []
    elif case == "dataset_revision":
        payload["dataset"]["revision"] = "unpinned"
    elif case == "dataset_selection":
        payload["dataset"]["selection"] = "random 10000 rows"
    elif case == "dataset_digest":
        payload["dataset"]["ordered_safe_text_sha256"] = "0" * 64
    elif case == "max_length":
        payload["max_sequence_length"] = 255
    elif case == "tokenizer_not_object":
        payload["tokenizer"] = []
    elif case == "vocab_bool":
        payload["tokenizer"]["base_vocab_size"] = True
    elif case == "vocab_mismatch":
        payload["tokenizer"]["base_vocab_size"] = 12
    elif case == "tokenizer_hash":
        payload["tokenizer"]["tokenizer_json_sha256"] = "0" * 64
    elif case == "tokenizer_revision":
        payload["tokenizer"]["revision"] = "unpinned"
    elif case == "special_ids_unsorted":
        payload["tokenizer"]["special_token_ids"] = [1, 0, 2, 3, 4]
    elif case == "counts_short":
        payload["counts_by_token_id"] = payload["counts_by_token_id"][:-1]
    elif case == "count_bool":
        payload["counts_by_token_id"][5] = True
    elif case == "count_negative":
        payload["counts_by_token_id"][5] = -1
    elif case == "count_sum":
        payload["content_token_count"] += 1
    elif case == "special_count":
        payload["counts_by_token_id"][0] = 1
        payload["counts_by_token_id"][5] -= 1
    elif case == "example_bool":
        payload["example_count"] = True
    elif case == "example_count":
        payload["example_count"] = 9_999
    elif case == "git_sha":
        payload["git_sha"] = "not-a-full-commit"
    _install_frequency_artifact(monkeypatch, tmp_path, payload)

    with pytest.raises(ValueError):
        model_module.GenMol(
            _config(diffusion="udlm", prior_variant="empirical_frequency")
        )


def test_frequency_artifact_runtime_tokenizer_vocab_is_validated(
    monkeypatch, tmp_path
):
    _install_frequency_artifact(monkeypatch, tmp_path, _frequency_payload())

    class WrongVocabTokenizer(_Tokenizer):
        vocab_size = 12

    monkeypatch.setattr(model_module, "get_tokenizer", lambda: WrongVocabTokenizer())
    with pytest.raises(ValueError, match="vocab sizes must agree"):
        model_module.GenMol(
            _config(diffusion="udlm", prior_variant="empirical_frequency")
        )


def test_committed_frequency_artifact_matches_the_pinned_identity():
    artifact_path = (
        Path(__file__).resolve().parents[1]
        / model_module.EMPIRICAL_FREQUENCY_RELATIVE_PATH
    )
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    assert digest == (
        model_module.EMPIRICAL_FREQUENCY_SHA256
    )

    class PinnedTokenizer:
        vocab_size = 1880
        all_special_ids = [0, 1, 2, 3, 4]

    artifact, counts, parsed_digest = model_module._parse_frequency_artifact(
        artifact_path,
        expected_sha256=model_module.EMPIRICAL_FREQUENCY_SHA256,
        model_vocab_size=1880,
        tokenizer=PinnedTokenizer(),
    )
    assert artifact["schema_version"] == 1
    assert len(counts) == 1880
    assert sum(counts) == artifact["content_token_count"] == 517090
    assert parsed_digest == digest


def test_hydra_configs_keep_release_default_and_label_empirical_variant():
    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        base = compose(config_name="base")
        release = compose(config_name="udlm")
        empirical = compose(config_name="udlm_categorical")

    assert base.training.diffusion == "mdlm"
    assert base.training.udlm.prior_variant == "release_uniform"
    assert release.training.diffusion == "udlm"
    assert release.training.udlm.prior_variant == "release_uniform"
    assert empirical.training.diffusion == "udlm"
    assert empirical.training.udlm.prior_variant == "empirical_frequency"
    assert empirical.training.udlm.empirical_uniform_mix == 0.01


def test_schedule_consistent_categorical_process_runs_training_contract():
    torch.manual_seed(21)
    model = model_module.GenMol(
        _config(diffusion="udlm", prior_variant="schedule_uniform")
    )
    model.log = lambda *args, **kwargs: None
    batch = {
        "input_ids": torch.tensor(
            [[1, 5, 6, 7, 2, 3], [1, 8, 9, 10, 2, 3]], dtype=torch.long
        ),
        "attention_mask": torch.tensor(
            [[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0]], dtype=torch.long
        ),
    }

    loss = model.training_step(batch, 0)
    loss.backward()

    assert torch.isfinite(loss)
    assert "mdlm.stationary_probs" in model.state_dict()
    gradient = model.backbone.time_conditioner.mlp[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()


def test_udlm_training_step_is_finite_and_trains_time_conditioner():
    torch.manual_seed(9)
    model = model_module.GenMol(_config(diffusion="udlm"))
    model.log = lambda *args, **kwargs: None
    batch = {
        "input_ids": torch.tensor(
            [[1, 5, 6, 7, 2, 3], [1, 8, 9, 10, 2, 3]], dtype=torch.long
        ),
        "attention_mask": torch.tensor(
            [[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0]], dtype=torch.long
        ),
    }

    loss = model.training_step(batch, 0)
    loss.backward()

    assert torch.isfinite(loss)
    gradient = model.backbone.time_conditioner.mlp[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_udlm_forward_requires_time():
    model = model_module.GenMol(_config(diffusion="udlm"))

    with pytest.raises(ValueError, match="time"):
        model(torch.tensor([[1, 5, 2]]), torch.ones(1, 3, dtype=torch.long))


def test_molecular_diffusion_mask_clamps_bos_eos_and_padding():
    model = model_module.GenMol(_config(diffusion="udlm"))
    input_ids = torch.tensor([[1, 5, 6, 2, 3]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 0]])

    assert torch.equal(
        model.diffusion_token_mask(input_ids, attention_mask),
        torch.tensor([[False, True, True, False, False]]),
    )


def test_special_token_exclusion_is_explicit_ablation():
    faithful = model_module.GenMol(_config(diffusion="udlm", exclude_special=False))
    adapted = model_module.GenMol(_config(diffusion="udlm", exclude_special=True))

    assert faithful.mdlm.diffusion_vocab_size == 11
    assert adapted.mdlm.diffusion_vocab_size == 6


def test_legacy_state_dict_roundtrip_remains_strict():
    first = model_module.GenMol(_config())
    second = model_module.GenMol(_config())

    result = second.load_state_dict(first.state_dict(), strict=True)

    assert result.missing_keys == []
    assert result.unexpected_keys == []


def test_udlm_warm_start_uses_mdlm_ema_and_resets_new_ema(tmp_path):
    source_config = _config()
    source_config.training.ema = 0.9
    source = model_module.GenMol(source_config)
    with torch.no_grad():
        for index, shadow in enumerate(source.ema.shadow_params):
            shadow.fill_(0.001 * (index + 1))
    checkpoint_path = tmp_path / "mdlm.ckpt"
    torch.save(
        {"state_dict": source.state_dict(), "ema": source.ema.state_dict()},
        checkpoint_path,
    )
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    target_config = _config(diffusion="udlm")
    target_config.training.ema = 0.9
    target = model_module.GenMol(target_config)
    report = target.initialize_from_mdlm_checkpoint(
        checkpoint_path,
        use_ema=True,
        expected_sha256=checkpoint_sha256,
    )
    base_parameters = [
        parameter
        for name, parameter in target.backbone.named_parameters()
        if not name.startswith("time_conditioner.")
    ]

    assert report["weights"] == "ema"
    assert report["parameter_tensors"] == len(source.ema.shadow_params)
    assert report["source_path"] == str(checkpoint_path)
    assert report["source_resolved_path"] == str(checkpoint_path.resolve())
    assert report["source_sha256"] == checkpoint_sha256
    assert report["source_size_bytes"] == checkpoint_path.stat().st_size
    assert report["expected_source_sha256"] == checkpoint_sha256
    assert report["byte_identity_verified_before_and_after_load"] is True
    assert torch.allclose(base_parameters[0], source.ema.shadow_params[0])
    assert len(target.ema.shadow_params) == len(list(target.backbone.parameters()))
    assert torch.allclose(target.ema.shadow_params[0], base_parameters[0])
    assert torch.count_nonzero(target.backbone.time_conditioner.mlp[-1].weight) == 0


def test_udlm_warm_start_wrong_digest_does_not_mutate_parameters(tmp_path):
    source_config = _config()
    source_config.training.ema = 0.9
    source = model_module.GenMol(source_config)
    checkpoint_path = tmp_path / "mdlm.ckpt"
    torch.save(
        {"state_dict": source.state_dict(), "ema": source.ema.state_dict()},
        checkpoint_path,
    )

    target_config = _config(diffusion="udlm")
    target_config.training.ema = 0.9
    target = model_module.GenMol(target_config)
    state_before = {
        name: value.detach().clone()
        for name, value in target.state_dict().items()
    }
    ema_object = target.ema
    ema_before = [parameter.clone() for parameter in target.ema.shadow_params]
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    wrong_sha256 = (
        ("0" if checkpoint_sha256[0] != "0" else "1") + checkpoint_sha256[1:]
    )

    with pytest.raises(RuntimeError, match="launch-pinned digest"):
        target.initialize_from_mdlm_checkpoint(
            checkpoint_path,
            use_ema=True,
            expected_sha256=wrong_sha256,
        )

    assert state_before.keys() == target.state_dict().keys()
    for name, value in target.state_dict().items():
        assert torch.equal(value, state_before[name])
    assert target.ema is ema_object
    for parameter, expected in zip(target.ema.shadow_params, ema_before, strict=True):
        assert torch.equal(parameter, expected)

import pytest
import torch
from omegaconf import OmegaConf
from transformers import BertForMaskedLM

import genmol.model as model_module
from genmol.backbone import TimeConditionedBertForMaskedLM
from genmol.diffusion import ContinuousUniformDiffusion


class _Tokenizer:
    vocab_size = 11
    mask_token_id = 4
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 3
    all_special_ids = [0, 1, 2, 3, 4]


def _config(*, diffusion=None, exclude_special=False):
    training = {
        "ema": 0.0,
        "antithetic_sampling": True,
        "sampling_eps": 1e-3,
        "global_mean_loss": True,
        "use_bracket_safe": False,
        "udlm": {
            "exclude_special_tokens": exclude_special,
            "noise_eps": 1e-3,
            "time_embedding_size": 8,
            "zero_init_conditioning": True,
        },
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


def test_missing_diffusion_setting_is_strict_legacy_mdlm_default():
    model = model_module.GenMol(_config())

    assert model.diffusion_type == "mdlm"
    assert isinstance(model.backbone, BertForMaskedLM)
    assert not isinstance(model.backbone, TimeConditionedBertForMaskedLM)
    assert not any("time_conditioner" in key for key in model.state_dict())


def test_udlm_selects_time_conditioned_backbone_and_uniform_process():
    model = model_module.GenMol(_config(diffusion="udlm"))

    assert model.diffusion_type == "udlm"
    assert isinstance(model.backbone, TimeConditionedBertForMaskedLM)
    assert isinstance(model.mdlm, ContinuousUniformDiffusion)
    assert "backbone.time_conditioner.mlp.2.weight" in model.state_dict()


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

    target_config = _config(diffusion="udlm")
    target_config.training.ema = 0.9
    target = model_module.GenMol(target_config)
    report = target.initialize_from_mdlm_checkpoint(checkpoint_path, use_ema=True)
    base_parameters = [
        parameter
        for name, parameter in target.backbone.named_parameters()
        if not name.startswith("time_conditioner.")
    ]

    assert report["weights"] == "ema"
    assert report["parameter_tensors"] == len(source.ema.shadow_params)
    assert torch.allclose(base_parameters[0], source.ema.shadow_params[0])
    assert len(target.ema.shadow_params) == len(list(target.backbone.parameters()))
    assert torch.allclose(target.ema.shadow_params[0], base_parameters[0])
    assert torch.count_nonzero(target.backbone.time_conditioner.mlp[-1].weight) == 0

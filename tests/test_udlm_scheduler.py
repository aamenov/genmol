import math
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from transformers import get_constant_schedule_with_warmup
from transformers.models.bert.configuration_bert import BertConfig

from genmol.backbone import (
    FILM_ADALN_CONDITIONING,
    TimeConditionedBertForMaskedLM,
)
from genmol.model import (
    CONSTANT_WITH_LINEAR_WARMUP,
    GenMol,
    HALF_COSINE_WITH_LINEAR_WARMUP_AND_FLOOR,
    build_optimizer_scheduler,
    optimizer_scheduler_multiplier,
    optimizer_scheduler_spec,
)


def _optim_config(scheduler=...):
    config = {"lr": 3e-4}
    if scheduler is not ...:
        config["scheduler"] = scheduler
    return config


def _l0_config():
    return {
        "name": CONSTANT_WITH_LINEAR_WARMUP,
        "warmup_updates": 2500,
        "horizon_updates": None,
        "decay_floor_lr": None,
    }


def _l1_config():
    return {
        "name": HALF_COSINE_WITH_LINEAR_WARMUP_AND_FLOOR,
        "warmup_updates": 50,
        "horizon_updates": 1000,
        "decay_floor_lr": 3e-6,
    }


def _scheduler_trace(builder, updates):
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=3e-4)
    scheduler = builder(optimizer)
    used = []
    post_update = []
    for _ in range(updates):
        parameter.grad = torch.ones_like(parameter)
        used.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        post_update.append(optimizer.param_groups[0]["lr"])
    return scheduler, used, post_update


def test_absent_scheduler_block_is_exact_released_transformers_fallback():
    spec = optimizer_scheduler_spec(_optim_config())
    assert asdict(spec) == {
        "schema_version": 1,
        "name": CONSTANT_WITH_LINEAR_WARMUP,
        "peak_lr": 3e-4,
        "warmup_updates": 2500,
        "horizon_updates": None,
        "decay_floor_lr": None,
        "step_unit": "optimizer_update",
        "horizon_includes_warmup": True,
        "post_horizon_policy": "constant_at_peak",
    }

    actual, actual_used, actual_post = _scheduler_trace(
        lambda optimizer: build_optimizer_scheduler(optimizer, spec), 2510
    )
    expected, expected_used, expected_post = _scheduler_trace(
        lambda optimizer: get_constant_schedule_with_warmup(optimizer, 2500), 2510
    )
    assert type(actual) is type(expected)
    assert actual.state_dict() == expected.state_dict()
    assert actual_used == expected_used
    assert actual_post == expected_post
    assert actual_post[9] == pytest.approx(1.2e-6)
    assert actual_post[2499] == pytest.approx(3e-4)
    assert actual_post[-1] == pytest.approx(3e-4)


def test_explicit_l0_matches_legacy_fallback_and_spec_is_frozen():
    fallback = optimizer_scheduler_spec(_optim_config())
    explicit = optimizer_scheduler_spec(_optim_config(_l0_config()))
    assert explicit == fallback
    with pytest.raises(FrozenInstanceError):
        explicit.warmup_updates = 50
    with pytest.raises(ValueError, match="step_unit"):
        replace(explicit, step_unit="microbatch")
    for wrong_schema_version in (True, 1.0):
        with pytest.raises(ValueError, match="schema version"):
            replace(explicit, schema_version=wrong_schema_version)


def test_l1_half_cosine_has_exact_milestones_and_clamped_floor():
    spec = optimizer_scheduler_spec(_optim_config(_l1_config()))
    expected = {
        0: 0.0,
        1: 6e-6,
        10: 6e-5,
        50: 3e-4,
        100: 2.979746535553042e-4,
        999: 3.000811986100421e-6,
        1000: 3e-6,
        1001: 3e-6,
        5000: 3e-6,
    }
    for index, learning_rate in expected.items():
        observed = spec.peak_lr * optimizer_scheduler_multiplier(spec, index)
        assert observed == pytest.approx(learning_rate, rel=1e-13, abs=1e-15)

    warmup = [
        optimizer_scheduler_multiplier(spec, index)
        for index in range(spec.warmup_updates + 1)
    ]
    decay = [
        optimizer_scheduler_multiplier(spec, index)
        for index in range(spec.warmup_updates, spec.horizon_updates + 100)
    ]
    assert all(left <= right for left, right in zip(warmup, warmup[1:]))
    assert all(left >= right for left, right in zip(decay, decay[1:]))
    assert all(math.isfinite(value) for value in (*warmup, *decay))
    floor_multiplier = spec.decay_floor_lr / spec.peak_lr
    assert all(value >= floor_multiplier for value in decay)


def test_l1_scheduler_uses_index_zero_for_first_optimizer_update():
    spec = optimizer_scheduler_spec(_optim_config(_l1_config()))
    scheduler, used, post = _scheduler_trace(
        lambda optimizer: build_optimizer_scheduler(optimizer, spec), 100
    )
    assert used[0] == 0.0
    assert used[-1] == pytest.approx(
        spec.peak_lr * optimizer_scheduler_multiplier(spec, 99)
    )
    assert post[-1] == pytest.approx(
        spec.peak_lr * optimizer_scheduler_multiplier(spec, 100)
    )
    assert scheduler.last_epoch == 100


def test_film_timestep_gradient_staging_respects_zero_lr_first_update():
    torch.manual_seed(59)
    model = TimeConditionedBertForMaskedLM(
        BertConfig(
            vocab_size=13,
            hidden_size=24,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=48,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            max_position_embeddings=16,
        ),
        time_embedding_size=8,
        zero_init_conditioning=False,
        conditioning_variant=FILM_ADALN_CONDITIONING,
    )
    model.eval()
    spec = optimizer_scheduler_spec(_optim_config(_l1_config()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.peak_lr)
    scheduler = build_optimizer_scheduler(optimizer, spec)
    input_ids = torch.tensor([[1, 5, 8, 2], [1, 6, 7, 2]])
    attention_mask = torch.ones_like(input_ids)
    noise = torch.tensor([0.1, 3.0])
    film_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if ".film_modulation." in name
    ]
    time_parameters = list(model.time_conditioner.parameters())

    def backward():
        optimizer.zero_grad(set_to_none=True)
        model(
            input_ids,
            attention_mask,
            noise_level=noise,
        ).logits.square().mean().backward()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in (*film_parameters, *time_parameters)
        )
        return any(
            torch.count_nonzero(parameter.grad) > 0
            for parameter in time_parameters
        )

    assert optimizer.param_groups[0]["lr"] == 0.0
    assert backward() is False
    assert all(torch.count_nonzero(parameter.grad) > 0 for parameter in film_parameters)
    optimizer.step()
    assert all(torch.count_nonzero(parameter) == 0 for parameter in film_parameters)
    scheduler.step()

    assert optimizer.param_groups[0]["lr"] > 0.0
    assert backward() is False
    optimizer.step()
    assert any(torch.count_nonzero(parameter) > 0 for parameter in film_parameters)
    scheduler.step()

    assert backward() is True


def test_genmol_configure_optimizers_uses_immutable_schedule_spec():
    spec = optimizer_scheduler_spec(_optim_config(_l1_config()))
    model_like = SimpleNamespace(
        optimizer_scheduler_spec=spec,
        backbone=torch.nn.Linear(3, 2),
        config=SimpleNamespace(
            optim=SimpleNamespace(
                beta1=0.9,
                beta2=0.999,
                eps=1e-8,
                weight_decay=0.0,
            )
        ),
    )

    optimizers, scheduler_configs = GenMol.configure_optimizers(model_like)

    assert len(optimizers) == len(scheduler_configs) == 1
    optimizer = optimizers[0]
    scheduler_config = scheduler_configs[0]
    assert optimizer.defaults["lr"] == pytest.approx(spec.peak_lr)
    assert optimizer.param_groups[0]["initial_lr"] == pytest.approx(spec.peak_lr)
    assert optimizer.param_groups[0]["lr"] == 0.0
    assert scheduler_config["interval"] == "step"
    assert scheduler_config["name"] == "lr"
    assert scheduler_config["scheduler"].last_epoch == 0


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"lr": True}, "optim.lr"),
        ({"lr": float("nan")}, "optim.lr"),
        (_optim_config(None), "must be a mapping"),
        (
            _optim_config({**_l0_config(), "unexpected": 1}),
            "exactly the registered fields",
        ),
        (
            _optim_config({key: value for key, value in _l0_config().items() if key != "name"}),
            "exactly the registered fields",
        ),
        (_optim_config({**_l0_config(), "name": "cosine"}), "not registered"),
        (_optim_config({**_l0_config(), "warmup_updates": True}), "nonnegative integer"),
        (_optim_config({**_l0_config(), "warmup_updates": -1}), "nonnegative integer"),
        (_optim_config({**_l0_config(), "horizon_updates": 1000}), "requires null"),
        (
            _optim_config({**_l1_config(), "horizon_updates": True}),
            "nonnegative integer",
        ),
        (
            _optim_config({**_l1_config(), "horizon_updates": 50}),
            "greater than warmup",
        ),
        (
            _optim_config({**_l1_config(), "decay_floor_lr": True}),
            "finite real number",
        ),
        (
            _optim_config({**_l1_config(), "decay_floor_lr": float("inf")}),
            "lie in",
        ),
        (
            _optim_config({**_l1_config(), "decay_floor_lr": -1e-6}),
            "lie in",
        ),
        (
            _optim_config({**_l1_config(), "decay_floor_lr": 3e-4}),
            "lie in",
        ),
    ],
)
def test_scheduler_config_validation_fails_closed(config, message):
    with pytest.raises(ValueError, match=message):
        optimizer_scheduler_spec(config)


def test_scheduler_index_validation_rejects_bool_negative_and_wrong_spec():
    spec = optimizer_scheduler_spec(_optim_config(_l1_config()))
    for value in (True, -1, 1.5):
        with pytest.raises(ValueError, match="scheduler_index"):
            optimizer_scheduler_multiplier(spec, value)
    with pytest.raises(TypeError, match="OptimizerSchedulerSpec"):
        optimizer_scheduler_multiplier({}, 0)
    with pytest.raises(TypeError, match="OptimizerSchedulerSpec"):
        build_optimizer_scheduler(object(), {})


def test_hydra_configs_resolve_exact_prospective_e_schedule_bundles():
    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        base = compose(config_name="base")
        e_l0 = compose(config_name="udlm_e_l0")
        e_l1 = compose(config_name="udlm_e_l1")
        a0_l0 = compose(config_name="udlm_e_a0_l0")
        a0_l1 = compose(config_name="udlm_e_a0_l1")
        a1_l0 = compose(config_name="udlm_e_a1_l0")
        a1_l1 = compose(config_name="udlm_e_a1_l1")

    assert dict(base.optim.scheduler) == _l0_config()
    assert e_l0.training.diffusion == "udlm"
    assert e_l0.training.udlm.prior_variant == "empirical_frequency"
    assert e_l0.training.reseed_after_model_initialization is False
    assert dict(e_l0.optim.scheduler) == _l0_config()
    assert e_l1.training.diffusion == "udlm"
    assert e_l1.training.udlm.prior_variant == "empirical_frequency"
    assert e_l1.training.reseed_after_model_initialization is False
    assert e_l1.optim.lr == pytest.approx(3e-4)
    assert dict(e_l1.optim.scheduler) == _l1_config()
    for config, scheduler in (
        (a0_l0, _l0_config()),
        (a0_l1, _l1_config()),
        (a1_l0, _l0_config()),
        (a1_l1, _l1_config()),
    ):
        assert config.training.diffusion == "udlm"
        assert config.training.udlm.prior_variant == "empirical_frequency"
        assert config.training.reseed_after_model_initialization is True
        assert dict(config.optim.scheduler) == scheduler
    assert a0_l0.training.udlm.conditioning_variant == "additive"
    assert a0_l1.training.udlm.conditioning_variant == "additive"
    assert a0_l0.training.udlm.zero_init_conditioning is True
    assert a0_l1.training.udlm.zero_init_conditioning is True
    assert a1_l0.training.udlm.conditioning_variant == "film_adaln"
    assert a1_l1.training.udlm.conditioning_variant == "film_adaln"
    assert a1_l0.training.udlm.zero_init_conditioning is False
    assert a1_l1.training.udlm.zero_init_conditioning is False

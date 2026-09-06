import pytest
import torch
from transformers import BertForMaskedLM
from transformers.models.bert.configuration_bert import BertConfig

from genmol.backbone import (
    ADDITIVE_CONDITIONING,
    FILM_ADALN_CONDITIONING,
    TimeConditionedBertForMaskedLM,
    TimestepEmbedder,
    is_conditioning_parameter_name,
)


def _tiny_config():
    return BertConfig(
        vocab_size=13,
        hidden_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=48,
        max_position_embeddings=16,
        pad_token_id=3,
    )


def test_sinusoidal_embedding_shape_and_fractional_time_support():
    values = torch.tensor([0.0, 0.25, 3.5])
    embedding = TimestepEmbedder.sinusoidal_embedding(values, 7)

    assert embedding.shape == (3, 7)
    assert torch.isfinite(embedding).all()
    assert not torch.equal(embedding[0], embedding[1])


def test_time_conditioner_is_checkpoint_nested_under_backbone():
    model = TimeConditionedBertForMaskedLM(_tiny_config())
    state_keys = set(model.state_dict())

    assert "bert.embeddings.word_embeddings.weight" in state_keys
    assert "time_conditioner.mlp.2.weight" in state_keys


def test_default_additive_variant_preserves_exact_state_and_behavior():
    torch.manual_seed(31)
    default = TimeConditionedBertForMaskedLM(_tiny_config())
    torch.manual_seed(31)
    explicit = TimeConditionedBertForMaskedLM(
        _tiny_config(), conditioning_variant=ADDITIVE_CONDITIONING
    )

    assert default.state_dict().keys() == explicit.state_dict().keys()
    assert not any("film_modulation" in key for key in default.state_dict())
    for name, value in default.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[name])

    input_ids = torch.tensor([[1, 5, 8, 2], [1, 6, 7, 2]])
    attention_mask = torch.ones_like(input_ids)
    noise = torch.tensor([0.1, 3.0])
    default.eval()
    explicit.eval()
    with torch.no_grad():
        default_logits = default(
            input_ids, attention_mask, noise_level=noise
        ).logits
        explicit_logits = explicit(
            input_ids, attention_mask, noise_level=noise
        ).logits
    assert torch.equal(default_logits, explicit_logits)


def test_zero_initialized_conditioner_receives_gradient_then_changes_output():
    torch.manual_seed(2)
    model = TimeConditionedBertForMaskedLM(_tiny_config())
    model.eval()
    input_ids = torch.tensor([[1, 5, 8, 2], [1, 6, 7, 2]])
    attention_mask = torch.ones_like(input_ids)
    noise_a = torch.tensor([0.1, 0.1])
    noise_b = torch.tensor([3.0, 3.0])

    initial_a = model(input_ids, attention_mask, noise_level=noise_a).logits
    initial_b = model(input_ids, attention_mask, noise_level=noise_b).logits
    assert torch.allclose(initial_a, initial_b)

    loss = model(input_ids, attention_mask, noise_level=noise_b).logits.square().mean()
    loss.backward()
    output_projection = model.time_conditioner.mlp[-1]
    assert output_projection.weight.grad is not None
    assert torch.count_nonzero(output_projection.weight.grad) > 0

    with torch.no_grad():
        output_projection.weight.add_(-0.1 * output_projection.weight.grad)
        output_projection.bias.add_(-0.1 * output_projection.bias.grad)
    updated_a = model(input_ids, attention_mask, noise_level=noise_a).logits
    updated_b = model(input_ids, attention_mask, noise_level=noise_b).logits
    assert not torch.allclose(updated_a, updated_b)


def _film_model():
    return TimeConditionedBertForMaskedLM(
        _tiny_config(),
        time_embedding_size=8,
        zero_init_conditioning=False,
        conditioning_variant=FILM_ADALN_CONDITIONING,
    )


def _film_modules(model):
    return [layer.film_modulation for layer in model.bert.encoder.layer]


def test_film_variant_is_an_exact_warm_start_identity_with_expected_shapes():
    torch.manual_seed(41)
    base = BertForMaskedLM(_tiny_config())
    model = _film_model()
    load_result = model.load_state_dict(base.state_dict(), strict=False)
    expected_missing = {
        name
        for name in model.state_dict()
        if is_conditioning_parameter_name(name)
    }

    assert set(load_result.missing_keys) == expected_missing
    assert load_result.unexpected_keys == []
    assert len(expected_missing) == 8
    assert {
        name
        for name, _ in model.named_parameters()
        if is_conditioning_parameter_name(name)
    } == expected_missing
    roundtrip = _film_model().load_state_dict(model.state_dict(), strict=True)
    assert roundtrip.missing_keys == []
    assert roundtrip.unexpected_keys == []

    for modulation in _film_modules(model):
        assert modulation.weight.shape == (48, 24)
        assert modulation.bias.shape == (48,)
        assert torch.count_nonzero(modulation.weight) == 0
        assert torch.count_nonzero(modulation.bias) == 0
    assert torch.count_nonzero(model.time_conditioner.mlp[-1].weight) > 0

    input_ids = torch.tensor([[1, 5, 8, 2], [1, 6, 7, 2]])
    attention_mask = torch.ones_like(input_ids)
    noise_a = torch.tensor([0.1, 0.2])
    noise_b = torch.tensor([2.0, 3.0])
    base.eval()
    model.eval()
    with torch.no_grad():
        base_output = base(
            input_ids,
            attention_mask,
            output_hidden_states=True,
            output_attentions=True,
        )
        output_a = model(
            input_ids,
            attention_mask,
            noise_level=noise_a,
            output_hidden_states=True,
            output_attentions=True,
        )
        output_b = model(
            input_ids,
            attention_mask,
            noise_level=noise_b,
            output_hidden_states=True,
            output_attentions=True,
        )

    assert torch.equal(output_a.logits, base_output.logits)
    assert torch.equal(output_b.logits, base_output.logits)
    assert len(output_a.hidden_states) == len(base_output.hidden_states) == 3
    assert len(output_a.attentions) == len(base_output.attentions) == 2
    for actual, expected in zip(
        output_a.hidden_states, base_output.hidden_states, strict=True
    ):
        assert torch.equal(actual, expected)
    for actual, expected in zip(
        output_a.attentions, base_output.attentions, strict=True
    ):
        assert torch.equal(actual, expected)

    word_embeddings = base.bert.embeddings.word_embeddings(input_ids)
    with torch.no_grad():
        base_from_embeddings = base(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=word_embeddings,
        ).logits
        film_from_embeddings = model(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=word_embeddings,
            noise_level=noise_a,
        ).logits
    assert torch.equal(film_from_embeddings, base_from_embeddings)


def test_film_gradient_staging_learns_modulation_then_timestep_mlp():
    torch.manual_seed(47)
    model = _film_model()
    model.eval()
    input_ids = torch.tensor([[1, 5, 8, 2], [1, 6, 7, 2]])
    attention_mask = torch.ones_like(input_ids)
    noise_a = torch.tensor([0.1, 0.2])
    noise_b = torch.tensor([2.0, 3.0])

    first_loss = model(
        input_ids, attention_mask, noise_level=noise_b
    ).logits.square().mean()
    first_loss.backward()
    for modulation in _film_modules(model):
        for parameter in modulation.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(modulation.weight.grad) > 0
        assert torch.count_nonzero(modulation.bias.grad) > 0
    for parameter in model.time_conditioner.parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) == 0

    with torch.no_grad():
        for modulation in _film_modules(model):
            for parameter in modulation.parameters():
                parameter.add_(-0.1 * parameter.grad)
    model.zero_grad(set_to_none=True)

    updated_a = model(input_ids, attention_mask, noise_level=noise_a).logits
    updated_b = model(input_ids, attention_mask, noise_level=noise_b).logits
    assert not torch.allclose(updated_a, updated_b)
    updated_b.square().mean().backward()
    time_gradients = [
        parameter.grad for parameter in model.time_conditioner.parameters()
    ]
    assert all(gradient is not None for gradient in time_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in time_gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in time_gradients)


def test_film_rejects_dead_double_zero_initialization_and_unknown_variant():
    with pytest.raises(ValueError, match="kills all conditioning"):
        TimeConditionedBertForMaskedLM(
            _tiny_config(),
            conditioning_variant=FILM_ADALN_CONDITIONING,
            zero_init_conditioning=True,
        )
    with pytest.raises(ValueError, match="conditioning_variant"):
        TimeConditionedBertForMaskedLM(
            _tiny_config(), conditioning_variant="not-a-conditioning-variant"
        )


def test_film_tuple_outputs_and_gradient_checkpointing_are_executable():
    torch.manual_seed(53)
    model = _film_model()
    model.gradient_checkpointing_enable()
    model.train()
    input_ids = torch.tensor([[1, 5, 8, 2], [1, 6, 7, 2]])
    attention_mask = torch.ones_like(input_ids)
    noise = torch.tensor([0.1, 3.0])

    output = model(
        input_ids,
        attention_mask,
        noise_level=noise,
        output_hidden_states=True,
        output_attentions=True,
        return_dict=False,
    )
    assert isinstance(output, tuple)
    assert output[0].shape == (2, 4, 13)
    assert len(output[1]) == 3
    assert len(output[2]) == 2
    output[0].square().mean().backward()
    assert all(
        modulation.weight.grad is not None
        and torch.isfinite(modulation.weight.grad).all()
        for modulation in _film_modules(model)
    )


def test_missing_noise_level_is_rejected():
    model = TimeConditionedBertForMaskedLM(_tiny_config())
    input_ids = torch.tensor([[1, 5, 2]])

    try:
        model(input_ids)
    except ValueError as error:
        assert "noise_level" in str(error)
    else:
        raise AssertionError("time-conditioned model accepted a missing noise level")

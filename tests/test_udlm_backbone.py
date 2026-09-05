import torch
from transformers.models.bert.configuration_bert import BertConfig

from genmol.backbone import TimeConditionedBertForMaskedLM, TimestepEmbedder


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


def test_missing_noise_level_is_rejected():
    model = TimeConditionedBertForMaskedLM(_tiny_config())
    input_ids = torch.tensor([[1, 5, 2]])

    try:
        model(input_ids)
    except ValueError as error:
        assert "noise_level" in str(error)
    else:
        raise AssertionError("time-conditioned model accepted a missing noise level")

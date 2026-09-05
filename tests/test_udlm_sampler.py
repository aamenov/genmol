import pytest
import torch
from omegaconf import OmegaConf

import genmol.sampler as sampler_module


class _Tokenizer:
    def batch_decode(self, values, skip_special_tokens=True):
        assert skip_special_tokens
        return ["decoded" for _ in values]


class _UDLMModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.mask_index = 4
        self.bos_index = 1
        self.eos_index = 2
        self.tokenizer = _Tokenizer()
        self.config = OmegaConf.create(
            {
                "training": {
                    "udlm": {"sampling_steps": 3, "inference_eps": 1e-5},
                    "use_bracket_safe": False,
                }
            }
        )
        self.calls = []

    def __call__(self, x, attention_mask, t=None):
        self.calls.append((x.clone(), attention_mask.clone(), t.clone()))
        return torch.zeros(*x.shape, 9)


class _UDLMProcess:
    def __init__(self):
        self.steps = []

    def sample_prior(self, shape, device=None):
        return torch.full(shape, 6, dtype=torch.long, device=device)

    def step(
        self,
        logits,
        x,
        t,
        s,
        *,
        mutable_mask,
        temperature,
    ):
        self.steps.append(
            (x.clone(), t.clone(), s.clone(), mutable_mask.clone(), temperature)
        )
        return torch.where(mutable_mask, torch.full_like(x, 7), x)


class _MDLMModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.mask_index = 4
        self.bos_index = 1
        self.eos_index = 2
        self.tokenizer = _Tokenizer()
        self.config = OmegaConf.create({"training": {"use_bracket_safe": False}})
        self.calls = []

    def __call__(self, x, attention_mask=None):
        self.calls.append((x.clone(), attention_mask.clone()))
        return torch.zeros(*x.shape, 9)


class _MDLMProcess:
    def __init__(self):
        self.steps = []

    def get_num_steps_confidence(self, x):
        return 3

    def step_confidence(
        self, logits, x, step, num_steps, softmax_temp, randomness
    ):
        self.steps.append((step, num_steps, softmax_temp, randomness))
        return x


def _sampler(model, process, diffusion_type):
    sampler = sampler_module.Sampler.__new__(sampler_module.Sampler)
    sampler.model = model
    sampler.pad_index = 3
    sampler.mdlm = process
    sampler.diffusion_type = diffusion_type
    return sampler


@pytest.fixture(autouse=True)
def _identity_decoder(monkeypatch):
    monkeypatch.setattr(
        sampler_module, "safe_to_smiles", lambda value, fix=True: value
    )


def test_udlm_sampling_uses_fixed_time_grid_and_clamps_context():
    model = _UDLMModel()
    process = _UDLMProcess()
    sampler = _sampler(model, process, "udlm")
    x = torch.tensor([[1, 4, 4, 2, 3]])

    samples = sampler.generate(x, softmax_temp=0.7, num_steps=3)

    assert samples == ["decoded"]
    assert len(model.calls) == 3
    assert len(process.steps) == 3
    expected_editable = torch.tensor([[False, True, True, False, False]])
    for model_input, attention_mask, _ in model.calls:
        assert model_input[0, 0] == 1
        assert model_input[0, 3] == 2
        assert model_input[0, 4] == 3
        assert torch.equal(attention_mask, torch.tensor([[True, True, True, True, False]]))
    for _, _, _, editable, temperature in process.steps:
        assert torch.equal(editable, expected_editable)
        assert temperature == pytest.approx(0.7)
    assert process.steps[0][1].item() == 1.0
    assert process.steps[-1][2].item() == pytest.approx(1e-5)


@pytest.mark.parametrize("diffusion_type", ["mdlm", "udlm"])
def test_sampler_can_return_raw_token_ids_before_chemical_decoding(diffusion_type):
    if diffusion_type == "udlm":
        model, process = _UDLMModel(), _UDLMProcess()
    else:
        model, process = _MDLMModel(), _MDLMProcess()
    sampler = _sampler(model, process, diffusion_type)
    inputs = torch.tensor([[1, 4, 4, 2]])

    token_ids = sampler.generate(inputs, return_token_ids=True)

    assert isinstance(token_ids, torch.Tensor)
    assert token_ids.shape == inputs.shape
    if diffusion_type == "udlm":
        assert torch.equal(token_ids, torch.tensor([[1, 7, 7, 2]]))
    else:
        assert torch.equal(token_ids, inputs)


def test_udlm_rejects_mdlm_specific_molecular_context_guidance():
    sampler = _sampler(_UDLMModel(), _UDLMProcess(), "udlm")

    with pytest.raises(ValueError, match="posterior-space"):
        sampler.generate(torch.tensor([[1, 4, 2]]), gamma=0.5, w=2)


def test_mdlm_confidence_loop_contract_is_preserved():
    model = _MDLMModel()
    process = _MDLMProcess()
    sampler = _sampler(model, process, "mdlm")

    samples = sampler.generate(
        torch.tensor([[1, 4, 4, 2]]),
        softmax_temp=0.5,
        randomness=0.25,
    )

    assert samples == ["decoded"]
    assert len(model.calls) == 3
    assert process.steps == [
        (0, 3, 0.5, 0.25),
        (1, 3, 0.5, 0.25),
        (2, 3, 0.5, 0.25),
    ]

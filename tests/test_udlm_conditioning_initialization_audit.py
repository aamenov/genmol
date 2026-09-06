from __future__ import annotations

import copy
import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from scripts.udlm import audit_conditioning_initialization as audit


FIXTURE_PATH = (
    Path(__file__).parents[1]
    / "experiments/udlm/protocols/conditioning_init_fixture_v1.json"
)
FIXTURE_RAW_SHA256 = "a7069082bd9a5a7d76d7d52b324345f4a46ba2dea386b87411b1fcb8112cc989"
FIXTURE_CANONICAL_SHA256 = (
    "96ed170d2e3db2c68101dd07d603ff44db7f77a95c3b9578d6c378d1e1f85d1b"
)


def _config(arm_id: str) -> dict:
    return {
        "seed": 17,
        "model": {"vocab_size": 11},
        "training": {
            "diffusion": "udlm",
            "init_from_mdlm_ema": True,
            "init_from_mdlm_checkpoint_sha256": "c" * 64,
            "reseed_after_model_initialization": True,
            "udlm": {
                "prior_variant": "empirical_frequency",
                "conditioning_variant": (
                    "additive" if arm_id == "E-A0" else "film_adaln"
                ),
                "zero_init_conditioning": arm_id == "E-A0",
            },
        },
    }


def _json_ref(path: str, value: object, *, schema_version: int = 1) -> dict:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    return {
        "root": "repository",
        "relative_path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "schema_version": schema_version,
        "canonical_sha256": audit.canonical_json_sha256(value),
    }


def _config_ref(path: str, value: object) -> dict:
    result = _json_ref(path, value)
    result.pop("schema_version")
    return result


def _arm(arm_id: str, *, output_directory: str) -> audit.ArmContract:
    config = _config(arm_id)
    return audit.ArmContract(
        arm_id=arm_id,
        output_directory=output_directory,
        config_ref=_config_ref(f"configs/{arm_id}.json", config),
        config=config,
    )


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


def _contract(tmp_root: Path) -> audit.AuditContract:
    fixture = _fixture()
    fixture_ref = _json_ref(
        "experiments/udlm/protocols/conditioning_init_fixture_v1.json", fixture
    )
    return audit.AuditContract(
        registry=object(),
        source_revision="d" * 40,
        checkpoint_ref={
            "root": "project",
            "relative_path": "outputs/paper/checkpoints/50000.ckpt",
            "sha256": "c" * 64,
            "size_bytes": 123,
        },
        checkpoint_path=tmp_root.parent / "outputs/paper/checkpoints/50000.ckpt",
        fixture_ref=fixture_ref,
        fixture=fixture,
        producer_ref={
            "root": "repository",
            "relative_path": "scripts/udlm/audit_conditioning_initialization.py",
            "sha256": "e" * 64,
            "size_bytes": 456,
        },
        reference=_arm(
            "E-A0", output_directory="output/udlm/screens/conditioning/E-A0"
        ),
        candidate=_arm(
            "E-A1", output_directory="output/udlm/screens/conditioning/E-A1"
        ),
    )


def test_frozen_literal_fixture_matches_exact_schema_and_digests():
    payload = FIXTURE_PATH.read_bytes()
    fixture = audit.strict_json_loads(payload, label="fixture")

    assert hashlib.sha256(payload).hexdigest() == FIXTURE_RAW_SHA256
    assert audit.canonical_json_sha256(fixture) == FIXTURE_CANONICAL_SHA256
    assert (
        audit.EXPECTED_FIXTURE_PATH
        == FIXTURE_PATH.relative_to(Path(__file__).parents[1]).as_posix()
    )
    assert audit.EXPECTED_FIXTURE_SIZE_BYTES == len(payload)
    assert audit.EXPECTED_FIXTURE_SHA256 == FIXTURE_RAW_SHA256
    assert audit.EXPECTED_FIXTURE_CANONICAL_SHA256 == FIXTURE_CANONICAL_SHA256
    assert set(fixture) == {
        "schema_version",
        "purpose",
        "checkpoint_sha256",
        "probe_phase",
        "input_ids",
        "attention_mask",
        "noise_tensor",
        "timestep_tensor",
    }
    assert fixture["checkpoint_sha256"] == (
        "8d00aa47b02f64bf39ff6b0b2e786f213587366fc2c3d29712a00f3f84108dd6"
    )


def test_frozen_noise_literals_are_exact_float32_sigma_of_timesteps():
    fixture = _fixture()
    timestep = torch.tensor(fixture["timestep_tensor"], dtype=torch.float32)
    observed = -torch.log1p(-(1.0 - 1e-3) * timestep)
    expected = torch.tensor(fixture["noise_tensor"], dtype=torch.float32)

    assert torch.equal(observed, expected)


@pytest.mark.parametrize(
    "payload,match",
    [
        (b'{"x": 1, "x": 2}', "duplicate JSON key"),
        (b'{"x": NaN}', "non-finite JSON constant"),
        (b"\xff", "not UTF-8"),
    ],
)
def test_strict_json_rejects_ambiguous_or_nonfinite_input(payload, match):
    with pytest.raises(audit.InitializationAuditError, match=match):
        audit.strict_json_loads(payload, label="test")


def test_fixture_validation_rejects_ragged_or_boolean_numeric_tensors():
    fixture = _fixture()
    audit._validate_fixture(fixture, fixture["checkpoint_sha256"])

    ragged = copy.deepcopy(fixture)
    ragged["input_ids"][1].pop()
    with pytest.raises(audit.InitializationAuditError, match="rectangular"):
        audit._validate_fixture(ragged, fixture["checkpoint_sha256"])

    boolean_noise = copy.deepcopy(fixture)
    boolean_noise["noise_tensor"][0] = True
    with pytest.raises(audit.InitializationAuditError, match="noise_tensor"):
        audit._validate_fixture(boolean_noise, fixture["checkpoint_sha256"])


def test_config_refs_have_no_schema_field_and_are_loaded_exactly(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(audit, "PROJECT_ROOT", tmp_path.parent)
    config = _config("E-A0")
    path = tmp_path / "configs/E-A0.json"
    path.parent.mkdir()
    payload = (json.dumps(config, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(payload)
    ref = {
        "root": "repository",
        "relative_path": "configs/E-A0.json",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "canonical_sha256": audit.canonical_json_sha256(config),
    }

    normalized, observed = audit._load_exact_json_ref(
        ref, "resolved config", config_ref=True
    )

    assert normalized == ref
    assert observed == config


class _FakeDiffusion:
    @staticmethod
    def sigma(timestep):
        return -torch.log1p(-(1.0 - 1e-3) * timestep)


class _FakeBackbone(nn.Module):
    def __init__(self, variant: str):
        super().__init__()
        self.conditioning_variant = variant
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, input_ids, attention_mask, *, noise_level):
        del attention_mask, noise_level
        values = input_ids.to(torch.float32).unsqueeze(-1)
        logits = torch.cat((values, values + 0.5, values - 0.25), dim=-1)
        return SimpleNamespace(logits=logits)


class _FakeModel:
    def __init__(self, config):
        self.diffusion_type = config["training"]["diffusion"]
        variant = config["training"]["udlm"]["conditioning_variant"]
        self.backbone = _FakeBackbone(variant)
        self.mdlm = _FakeDiffusion()

    def initialize_from_mdlm_checkpoint(
        self, checkpoint_path, *, use_ema, expected_sha256
    ):
        assert checkpoint_path.name == "50000.ckpt"
        assert use_ema is True
        return {
            "source_sha256": expected_sha256,
            "expected_source_sha256": expected_sha256,
            "weights": "ema",
            "byte_identity_verified_before_and_after_load": True,
        }


def _fake_stack():
    return SimpleNamespace(
        torch=torch,
        numpy=np,
        OmegaConf=SimpleNamespace(create=lambda value: value),
        GenMol=_FakeModel,
        seed_everything=lambda seed, workers: seed if workers else None,
    )


@pytest.mark.parametrize("arm_id", audit.ARM_IDS)
def test_registered_arm_probe_is_cpu_float32_and_uses_exact_sigma(arm_id, tmp_path):
    payload, shape = audit.probe_registered_arm(
        _arm(arm_id, output_directory=f"output/udlm/screens/{arm_id}"),
        fixture=_fixture(),
        checkpoint_path=tmp_path / "50000.ckpt",
        checkpoint_sha256="c" * 64,
        stack=_fake_stack(),
    )

    assert shape == [2, 4, 3]
    assert len(payload) == 2 * 4 * 3 * 4
    assert struct.unpack("<f", payload[:4]) == (1.0,)


def test_serialize_logits_is_explicit_little_endian_c_order():
    logits = torch.tensor([[[1.0, -2.5], [3.25, 0.0]]], dtype=torch.float32)

    payload, shape = audit._serialize_logits(logits, _fake_stack())

    assert shape == [1, 2, 2]
    assert payload == struct.pack("<4f", 1.0, -2.5, 3.25, 0.0)


def test_run_audit_publishes_exact_verifier_object_and_is_no_clobber(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(audit, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(audit, "PROJECT_ROOT", tmp_path.parent)
    contract = _contract(tmp_path)
    calls = []

    def fake_probe(arm, **kwargs):
        calls.append((arm.arm_id, kwargs["checkpoint_sha256"]))
        return struct.pack("<2f", 1.25, -0.5), [1, 1, 2]

    monkeypatch.setattr(audit, "probe_registered_arm", fake_probe)
    output = tmp_path / "output/udlm/screens/conditioning/init_audit.json"

    report = audit.run_audit(contract, output_path=output, stack=object())

    assert calls == [("E-A0", "c" * 64), ("E-A1", "c" * 64)]
    assert set(report) == {
        "schema_version",
        "reference_arm_id",
        "candidate_arm_id",
        "fixture",
        "source_revision",
        "checkpoint_sha256",
        "reference_config_canonical_sha256",
        "candidate_config_canonical_sha256",
        "probe_phase",
        "input_ids_sha256",
        "attention_mask_sha256",
        "noise_tensor_sha256",
        "timestep_tensor_sha256",
        "logits_dtype",
        "logits_shape",
        "reference_logits",
        "candidate_logits",
        "producer_source",
        "exact_equal",
    }
    assert report["exact_equal"] is True
    assert report["logits_dtype"] == "float32-little-endian-c-order"
    assert report["logits_shape"] == [1, 1, 2]
    assert "E-A0" in report["reference_logits"]["relative_path"]
    assert "E-A1" in report["candidate_logits"]["relative_path"]
    assert report["reference_logits"]["sha256"] == report["candidate_logits"]["sha256"]
    assert json.loads(output.read_text()) == report

    with pytest.raises(FileExistsError, match="refusing to replace"):
        audit.run_audit(contract, output_path=output, stack=object())
    assert calls == [("E-A0", "c" * 64), ("E-A1", "c" * 64)]


def test_run_audit_records_inequality_without_claiming_exact_identity(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(audit, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(audit, "PROJECT_ROOT", tmp_path.parent)
    contract = _contract(tmp_path)

    def unequal_probe(arm, **_kwargs):
        value = 1.0 if arm.arm_id == "E-A0" else 2.0
        return struct.pack("<f", value), [1]

    monkeypatch.setattr(audit, "probe_registered_arm", unequal_probe)

    report = audit.run_audit(
        contract,
        output_path=tmp_path / "output/udlm/screens/conditioning/unequal.json",
        stack=object(),
    )

    assert report["exact_equal"] is False
    assert report["reference_logits"]["sha256"] != report["candidate_logits"]["sha256"]

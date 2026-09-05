import pytest

from scripts.udlm.cpu_smoke import (
    _bind_output_path,
    _config,
    _validate_git_provenance,
    _validate_smoke_gate,
    _write_json_exclusive,
)


def _diagnostics(before_loss=6.0, after_loss=2.0):
    return (
        {"0.5": {"loss": before_loss}},
        {"0.5": {"loss": after_loss}},
    )


@pytest.mark.parametrize(
    "variant",
    ["release_uniform", "schedule_uniform", "empirical_frequency"],
)
def test_smoke_config_exposes_only_explicit_prior_identity(variant):
    config = _config(
        exclude_special_tokens=False,
        sampling_steps=16,
        prior_variant=variant,
        empirical_uniform_mix=0.01,
    )
    assert config.training.udlm.prior_variant == variant
    assert config.training.udlm.empirical_uniform_mix == 0.01


def test_smoke_gate_requires_the_same_fixed_corruption_before_and_after():
    before = {
        "0.5": {
            "loss": 6.0,
            "corrupted_token_ids_sha256": "a" * 64,
        }
    }
    after = {
        "0.5": {
            "loss": 2.0,
            "corrupted_token_ids_sha256": "b" * 64,
        }
    }
    with pytest.raises(RuntimeError, match="corruption changed"):
        _validate_smoke_gate([6.0, 5.0, 2.0, 1.0], before, after, ["CCO"])


def test_smoke_gate_accepts_falling_losses_and_a_strict_decode():
    before, after = _diagnostics()
    _validate_smoke_gate([6.0, 5.0, 2.0, 1.0], before, after, ["CCO"])


@pytest.mark.parametrize(
    ("losses", "before_after", "generated", "message"),
    [
        ([1.0, 2.0], _diagnostics(), ["CCO"], "window mean"),
        ([2.0, 1.0], _diagnostics(1.0, 2.0), ["CCO"], "diagnostic loss"),
        ([2.0, 1.0], _diagnostics(), [], "no strictly decodable"),
    ],
)
def test_smoke_gate_rejects_failed_evidence(losses, before_after, generated, message):
    before, after = before_after
    with pytest.raises(RuntimeError, match=message):
        _validate_smoke_gate(losses, before, after, generated)


def test_smoke_requires_clean_pushed_commit():
    _validate_git_provenance({"commit": "a" * 40, "upstream": "a" * 40, "dirty": False})
    with pytest.raises(RuntimeError, match="upstream"):
        _validate_git_provenance(
            {"commit": "a" * 40, "upstream": "b" * 40, "dirty": False}
        )
    with pytest.raises(RuntimeError, match="clean"):
        _validate_git_provenance(
            {"commit": "a" * 40, "upstream": "a" * 40, "dirty": True}
        )


def test_smoke_writer_is_no_clobber(tmp_path):
    path = tmp_path / "result.json"
    _write_json_exclusive(path, {"value": 1})
    with pytest.raises(FileExistsError):
        _write_json_exclusive(path, {"value": 2})
    assert path.read_text(encoding="utf-8") == '{\n  "value": 1\n}\n'


def test_smoke_output_binding_rejects_a_dangling_leaf_symlink(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.udlm.cpu_smoke.ROOT_DIR", tmp_path)
    dangling = tmp_path / "result.json"
    dangling.symlink_to(tmp_path / "missing-target.json")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _bind_output_path(dangling)
    assert dangling.is_symlink()
    assert not (tmp_path / "missing-target.json").exists()

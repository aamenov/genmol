import copy
import json
from pathlib import Path

import pytest

from scripts.udlm.collect_cpu_smoke_panel import (
    EXPECTED_VARIANTS,
    _bind_output_path,
    _canonical_sequence_sha256,
    _current_source_inputs,
    _expected_prior_geometry,
    _reject_duplicate_keys,
    _validate_prior_metadata,
    _validate_source_inputs,
    _validate_result,
    build_panel,
)


COMMIT = "a" * 40


def _record(variant):
    if variant == "release_uniform":
        role = "faithful_release_control"
        process = "released_continuous_uniform"
        schedule = "released_ideal_loss_residual_forward"
        mixture = None
    elif variant == "schedule_uniform":
        role = "schedule_repair_uniform_control"
        process = "rank_one_continuous_categorical"
        schedule = "schedule_consistent_residual_forward_and_loss"
        mixture = None
    else:
        role = "empirical_prior_treatment"
        process = "rank_one_continuous_categorical"
        schedule = "schedule_consistent_residual_forward_and_loss"
        mixture = 0.01
    diagnostic_digests = {"0.1": "1" * 64, "0.5": "5" * 64, "0.9": "9" * 64}
    diagnostics_before = {
        time_value: {
            "loss": 3.0,
            "corrupted_token_ids_sha256": diagnostic_digests[time_value],
        }
        for time_value in ("0.1", "0.5", "0.9")
    }
    diagnostics_after = copy.deepcopy(diagnostics_before)
    for row in diagnostics_after.values():
        row["loss"] = 1.0
    return {
        "schema_version": 2,
        "purpose": "bounded CPU integration smoke; not benchmark or superiority evidence",
        "prior_variant": variant,
        "device": "cpu",
        "git": {"commit": COMMIT, "upstream": COMMIT, "dirty": False},
        "prior_metadata": {
            "variant": variant,
            "comparison_role": role,
            "process_family": process,
            "schedule_variant": schedule,
            "uniform_mixture_weight": mixture,
            "frequency_artifact_sha256": (
                "088c78e75611f3cc42c4011e1da6f65a377e673b9cba07a28b126b0fc62f06ed"
                if mixture is not None
                else None
            ),
        },
        "all_losses_finite": True,
        "all_gradient_norms_finite": True,
        "loss_first_five_mean": 4.0,
        "loss_last_five_mean": 1.0,
        "sample_count_requested": 2,
        "empirical_uniform_mix_requested": 0.01,
        "no_repair_decodable_samples": 1,
        "generated_smiles": ["CCO"],
        "fixed_diagnostics_before": diagnostics_before,
        "fixed_diagnostics_after": diagnostics_after,
        "seed": 1,
        "steps": 20,
        "sampling_steps": 8,
        "exclude_special_tokens": True,
        "toy_smiles": ["CCO"],
        "batch_shape": [1, 3],
        "clean_input_ids_sha256": "b" * 64,
        "attention_mask_sha256": "c" * 64,
        "runtime_seconds": 1.0,
    }


def _artifacts(root=Path.cwd()):
    return {
        variant: (
            root / "output" / f"{variant}.json",
            json.dumps(_record(variant)).encode(),
            _record(variant),
        )
        for variant in EXPECTED_VARIANTS
    }


def test_build_panel_accepts_only_a_matched_three_variant_gate(monkeypatch, tmp_path):
    monkeypatch.setattr("scripts.udlm.collect_cpu_smoke_panel.ROOT_DIR", tmp_path)
    artifacts = _artifacts(tmp_path)
    panel = build_panel(
        artifacts,
        git={"commit": COMMIT, "upstream": COMMIT, "dirty": False},
        strict_provenance=False,
    )
    assert tuple(panel["variants"]) == EXPECTED_VARIANTS
    assert "cannot rank priors" in panel["claim_scope"]
    assert panel["matched_fields"]["steps"] == 20
    assert panel["variants"]["release_uniform"]["source_artifact_base64"]


def test_panel_rejects_unmatched_inputs(monkeypatch, tmp_path):
    monkeypatch.setattr("scripts.udlm.collect_cpu_smoke_panel.ROOT_DIR", tmp_path)
    artifacts = _artifacts(tmp_path)
    artifacts["schedule_uniform"][2]["seed"] = 2
    with pytest.raises(ValueError, match="not matched on seed"):
        build_panel(
            artifacts,
            git={"commit": COMMIT, "upstream": COMMIT, "dirty": False},
            strict_provenance=False,
        )


def test_result_rejects_failed_or_mislabeled_gate():
    record = _record("empirical_frequency")
    record["prior_metadata"]["frequency_artifact_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frequency artifact mismatch"):
        _validate_result(record, "empirical_frequency", strict_provenance=False)


def test_duplicate_json_keys_are_rejected():
    with pytest.raises(ValueError, match="duplicate JSON key"):
        json.loads('{"a": 1, "a": 2}', object_pairs_hook=_reject_duplicate_keys)


def test_source_input_validation_binds_current_committed_files():
    source_inputs = _current_source_inputs("release_uniform")
    _validate_source_inputs({"source_inputs": source_inputs}, "release_uniform")
    changed = copy.deepcopy(source_inputs)
    changed["model"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="do not match current files"):
        _validate_source_inputs({"source_inputs": changed}, "release_uniform")


def test_strict_empirical_geometry_matches_real_genmol_metadata(monkeypatch):
    import genmol.model as model_module
    from omegaconf import OmegaConf

    from scripts.udlm.cpu_smoke import _config

    class _Tokenizer:
        vocab_size = 1880
        mask_token_id = 4
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 3
        all_special_ids = [0, 1, 2, 3, 4]

    monkeypatch.setattr(model_module, "get_tokenizer", lambda: _Tokenizer())
    config = _config(
        exclude_special_tokens=False,
        sampling_steps=32,
        prior_variant="empirical_frequency",
        empirical_uniform_mix=0.01,
    )
    model = model_module.GenMol(config)
    metadata = model.udlm_prior_metadata.to_dict()
    active_ids, probabilities, _artifact, active_count = _expected_prior_geometry(
        "empirical_frequency",
        excluded_token_ids=[],
        mixture_weight=0.01,
    )

    assert active_count == 517_090
    assert metadata["active_token_ids_sha256"] == _canonical_sequence_sha256(active_ids)
    assert metadata["stationary_probs_sha256"] == _canonical_sequence_sha256(
        probabilities
    )
    record = {
        "prior_metadata": metadata,
        "effective_config": OmegaConf.to_container(config, resolve=True),
        "exclude_special_tokens": False,
        "sampling_steps": 32,
        "seed": 1,
    }
    _validate_prior_metadata(record, "empirical_frequency")


def test_panel_output_binding_rejects_a_dangling_leaf_symlink(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.udlm.collect_cpu_smoke_panel.ROOT_DIR", tmp_path)
    dangling = tmp_path / "panel.json"
    dangling.symlink_to(tmp_path / "missing-panel.json")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _bind_output_path(dangling)
    assert dangling.is_symlink()
    assert not (tmp_path / "missing-panel.json").exists()

import copy
import json
from pathlib import Path

import pytest

from scripts.udlm.collect_cpu_smoke_panel import (
    EXPECTED_VARIANTS,
    _reject_duplicate_keys,
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
    diagnostics_before = {
        time_value: {
            "loss": 3.0,
            "corrupted_token_ids_sha256": time_value * 64,
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
    monkeypatch.setattr(
        "scripts.udlm.collect_cpu_smoke_panel.ROOT_DIR", tmp_path
    )
    artifacts = _artifacts(tmp_path)
    panel = build_panel(
        artifacts,
        git={"commit": COMMIT, "upstream": COMMIT, "dirty": False},
    )
    assert tuple(panel["variants"]) == EXPECTED_VARIANTS
    assert "cannot rank priors" in panel["claim_scope"]
    assert panel["matched_fields"]["steps"] == 20


def test_panel_rejects_unmatched_inputs(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "scripts.udlm.collect_cpu_smoke_panel.ROOT_DIR", tmp_path
    )
    artifacts = _artifacts(tmp_path)
    artifacts["schedule_uniform"][2]["seed"] = 2
    with pytest.raises(ValueError, match="not matched on seed"):
        build_panel(
            artifacts,
            git={"commit": COMMIT, "upstream": COMMIT, "dirty": False},
        )


def test_result_rejects_failed_or_mislabeled_gate():
    record = _record("empirical_frequency")
    record["prior_metadata"]["frequency_artifact_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frequency artifact mismatch"):
        _validate_result(record, "empirical_frequency")


def test_duplicate_json_keys_are_rejected():
    with pytest.raises(ValueError, match="duplicate JSON key"):
        json.loads('{"a": 1, "a": 2}', object_pairs_hook=_reject_duplicate_keys)

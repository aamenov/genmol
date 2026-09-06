import math
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.udlm import audit_empirical_prior_floor as audit


FROZEN_AUDIT_PATH = (
    audit.REPOSITORY_ROOT
    / "experiments/udlm/prior_geometry/floor_selection_train_rows_10001_30000.json"
)
FROZEN_AUDIT_SHA256 = "02908dafaf589ca9a49e560aa1eab470a18d6bfe616b781164784c489f54a9f1"


def _grid():
    return (
        0.0001,
        audit.CANDIDATE_UNIFORM_MIXTURE_WEIGHT,
        0.0003,
        0.001,
        audit.CURRENT_UNIFORM_MIXTURE_WEIGHT,
    )


def test_block_analysis_counts_unseen_tokens_and_matches_direct_nll():
    training = [90, 10, 0]
    heldout = [80, 15, 5]

    result = audit.analyze_block(training, heldout, _grid())

    assert result["heldout_content_tokens"] == 100
    assert result["heldout_observed_token_types"] == 3
    assert result["training_unseen_token_types"] == 1
    assert result["heldout_new_token_types"] == 1
    assert result["heldout_tokens_from_training_unseen_types"] == 5
    assert result["heldout_training_unseen_token_fraction"] == 0.05
    row = next(
        row
        for row in result["mixture_grid"]
        if row["uniform_mixture_weight"] == audit.CANDIDATE_UNIFORM_MIXTURE_WEIGHT
    )
    probabilities = [
        (1 - audit.CANDIDATE_UNIFORM_MIXTURE_WEIGHT) * 0.9
        + audit.CANDIDATE_UNIFORM_MIXTURE_WEIGHT / 3,
        (1 - audit.CANDIDATE_UNIFORM_MIXTURE_WEIGHT) * 0.1
        + audit.CANDIDATE_UNIFORM_MIXTURE_WEIGHT / 3,
        audit.CANDIDATE_UNIFORM_MIXTURE_WEIGHT / 3,
    ]
    expected = -sum(
        count * math.log(probability)
        for count, probability in zip(heldout, probabilities, strict=True)
    ) / sum(heldout)
    assert row["heldout_content_token_nll_nats"] == pytest.approx(expected)
    assert 0.0 < result["continuous_maximum_likelihood_uniform_mixture_weight"] < 1.0


def test_cumulative_block_counts_is_exact_and_rejects_decreases():
    assert audit.cumulative_block_counts([5, 2, 0], [8, 2, 4]) == (3, 0, 4)
    with pytest.raises(ValueError, match="contain"):
        audit.cumulative_block_counts([5, 2, 0], [4, 3, 1])
    with pytest.raises(ValueError, match="vocabulary"):
        audit.cumulative_block_counts([5, 2], [6, 2, 1])


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        ([0, 0], "counts"),
        ([1, -1], "counts"),
        ([1, True], "counts"),
        ([1], "counts"),
    ],
)
def test_count_validation_is_type_exact(counts, message):
    with pytest.raises(ValueError, match=message):
        audit.analyze_block(counts, [1] * len(counts), _grid())


def test_recommendation_requires_both_stable_optima_and_improvements():
    block = {
        "continuous_maximum_likelihood_uniform_mixture_weight": 0.00018,
        "candidate_minus_current_nll_nats": -0.008,
        "heldout_tokens_from_training_unseen_types": 10,
    }
    result = audit.recommend_candidate_weight(block, dict(block))
    assert result["recommended_uniform_mixture_weight"] == 0.0002
    assert result["candidate_nll_strictly_better_than_current_on_both_blocks"]

    worse = dict(block, candidate_minus_current_nll_nats=0.0)
    retained = audit.recommend_candidate_weight(block, worse)
    assert retained["recommended_uniform_mixture_weight"] == 0.01

    unstable = dict(block, continuous_maximum_likelihood_uniform_mixture_weight=0.00031)
    retained = audit.recommend_candidate_weight(block, unstable)
    assert retained["recommended_uniform_mixture_weight"] == 0.01


def test_real_base_frequency_artifact_has_the_frozen_identity():
    payload = audit._stable_file_bytes(audit.BASE_FREQUENCY_PATH)
    assert audit._sha256(payload) == audit.BASE_FREQUENCY_SHA256
    value = audit._validate_base_frequency_artifact(
        audit.strict_json_loads(payload, label="test frequency")
    )
    assert value["example_count"] == 10_000
    assert value["content_token_count"] == 517_090
    assert value["observed_token_types"] == 184


def test_frozen_training_only_audit_records_exact_replay_and_recommendation():
    payload = audit._stable_file_bytes(FROZEN_AUDIT_PATH)
    assert audit._sha256(payload) == FROZEN_AUDIT_SHA256
    value = audit.strict_json_loads(payload, label="frozen prior-floor audit")

    assert value["git"]["commit"] == "6424b323084358ea050ba22d7e13ef8d45962496"
    assert value["data_use"]["formal_preregistration_before_data_access"] is False
    assert value["data_use"]["final_generation_seeds_or_metrics_used"] is False
    base = audit.strict_json_loads(
        audit._stable_file_bytes(audit.BASE_FREQUENCY_PATH),
        label="base frequency artifact",
    )
    assert (
        value["stream_checkpoints"]["10000"]["counts_by_token_id"]
        == base["counts_by_token_id"]
    )
    assert value["recommendation"]["recommended_uniform_mixture_weight"] == 0.0002
    assert all(
        delta < 0.0
        for delta in value["recommendation"]["candidate_minus_current_nll_by_block"]
    )


def test_build_audit_rejects_stream_prefix_drift_and_preserves_scope():
    base_counts = [90, 10, 0]
    base = {
        "counts_by_token_id": base_counts,
        "content_token_count": 100,
        "dataset": {"ordered_safe_text_sha256": "a" * 64},
    }
    checkpoints = {
        10_000: {
            "counts_by_token_id": list(base_counts),
            "content_token_count": 100,
            "ordered_safe_text_sha256": "a" * 64,
        },
        20_000: {
            "counts_by_token_id": [180, 25, 5],
            "content_token_count": 210,
            "ordered_safe_text_sha256": "b" * 64,
        },
        30_000: {
            "counts_by_token_id": [270, 40, 10],
            "content_token_count": 320,
            "ordered_safe_text_sha256": "c" * 64,
        },
    }
    # The synthetic mixture optimum is outside the production recommendation
    # guardrail, but the audit remains a valid non-ranking record.
    result = audit.build_audit(
        commit="d" * 40,
        source_records={"script.py": {"sha256": "e" * 64}},
        base_frequency=base,
        checkpoints=checkpoints,
    )
    assert result["schema_version"] == 1
    assert "does not train" in result["claim_scope"]
    assert result["data_use"]["formal_preregistration_before_data_access"] is False
    assert result["data_use"]["final_generation_seeds_or_metrics_used"] is False

    checkpoints[10_000]["counts_by_token_id"][0] += 1
    with pytest.raises(audit.PriorFloorAuditError, match="does not reproduce"):
        audit.build_audit(
            commit="d" * 40,
            source_records={},
            base_frequency=base,
            checkpoints=checkpoints,
        )


def test_strict_json_rejects_duplicate_and_nonfinite_values():
    with pytest.raises(ValueError, match="duplicate"):
        audit.strict_json_loads(b'{"a":1,"a":2}', label="fixture")
    with pytest.raises(ValueError, match="non-finite"):
        audit.strict_json_loads(b'{"a":NaN}', label="fixture")


def test_help_works_without_site_packages_or_gpu_access():
    result = subprocess.run(
        [sys.executable, "-S", str(audit.__file__), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--output" in result.stdout


def test_source_contains_no_gpu_probe_or_training_entrypoint():
    source = Path(audit.__file__).read_text()
    assert "nvidia-smi" not in source
    assert "torch.cuda" not in source
    assert "scripts/train.py" not in source

import math

import pytest

from scripts.udlm.token_frequency_audit import (
    MODEL_VOCAB_SIZE,
    summarize_counts,
    types_for_coverage,
    uniform_hit_probability,
)


def test_uniform_sequence_hit_probability_is_stable_and_exact():
    actual = uniform_hit_probability(200, 49, 1_880)
    expected = 1 - (1 - 200 / 1_880) ** 49
    assert actual == pytest.approx(expected)
    assert actual > 0.99


def test_coverage_counts_types_in_descending_frequency_order():
    assert types_for_coverage([70, 20, 10, 0], 0.7) == 1
    assert types_for_coverage([70, 20, 10, 0], 0.9) == 2
    assert types_for_coverage([70, 20, 10, 0], 1.0) == 3


def test_summary_separates_empirical_tail_from_uniform_tail_risk():
    counts = [0] * MODEL_VOCAB_SIZE
    counts[5] = 70
    counts[6] = 29
    counts[1_800] = 1
    tokens = [f"token-{index}" for index in range(MODEL_VOCAB_SIZE)]

    result = summarize_counts(counts, [2, 3, 3], tokens, [0, 1, 2, 3, 4])

    assert result["content_token_count"] == 100
    assert result["observed_token_types"] == 3
    assert result["id_tail_1680_1879"]["empirical_mass"] == pytest.approx(0.01)
    assert result["id_tail_1680_1879"]["uniform_per_token_mass"] == pytest.approx(
        200 / 1_880
    )
    assert result["tokenizer_control_symbols"]["category_count"] == 5
    assert math.isfinite(result["empirical_entropy_nats"])

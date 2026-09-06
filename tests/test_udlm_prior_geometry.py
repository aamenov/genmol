import hashlib
import json
import math

import pytest

from scripts.udlm import audit_prior_geometry as audit


TOKENIZER_SHA256 = "0db5f4dbdc7e8ff759e98483759611a426e187ee7f3f0a91edc8800abe7bf140"
FREQUENCY_SHA256 = "088c78e75611f3cc42c4011e1da6f65a377e673b9cba07a28b126b0fc62f06ed"
PANEL_SHA256 = "e2493da4f3cb3217b48c78dc2901dc7524d7d90dc3959cffa87a1b6f8a9a7658"
ORDERED_FIXTURE_SHA256 = (
    "77e86b14c9cbd69b41c746218f44cd336a9c6942618d8d8dfb1103949d29318a"
)
ACTIVE_IDS = (1, 2, 3, 4)
CONFIGURED_PROBABILITIES = (0.5965, 0.2995, 0.1015, 0.0025)


def _frequency_fixture():
    return {
        "schema_version": 1,
        "purpose": "CPU-only token-frequency diagnostic; not benchmark evidence",
        "example_count": 3,
        "content_token_count": 10,
        "observed_token_types": 3,
        "unobserved_token_types": 2,
        "counts_by_token_id": [0, 6, 3, 1, 0],
        "dataset": {"repo_id": "dataset", "revision": "rev", "split": "train"},
        "tokenizer": {
            "repo_id": "dataset",
            "revision": "tokenizer-rev",
            "tokenizer_json_sha256": TOKENIZER_SHA256,
            "base_vocab_size": 5,
            "special_token_ids": [0],
        },
    }


def _panel_fixture():
    rows = [
        {"source_index": 0, "input_ids": [0, 1, 2], "content_length": 2},
        {"source_index": 1, "input_ids": [0, 1, 4], "content_length": 2},
    ]
    return {
        "schema_version": 1,
        "purpose": "fixed held-out denoising panel; not generative benchmark evidence",
        "sample_count": 2,
        "ordered_token_ids_sha256": ORDERED_FIXTURE_SHA256,
        "rows": rows,
        "dataset": {
            "repo_id": "dataset",
            "revision": "rev",
            "split": "validation",
        },
        "tokenizer": {
            "repo_id": "dataset",
            "revision": "tokenizer-rev",
            "tokenizer_json_sha256": TOKENIZER_SHA256,
            "base_vocab_size": 5,
            "special_token_ids": [0],
        },
    }


def _analyze(mixtures=(0.01, 0.1), probabilities=CONFIGURED_PROBABILITIES):
    return audit.analyze_prior_geometry(
        _frequency_fixture(),
        _panel_fixture(),
        mixtures,
        active_token_ids=ACTIVE_IDS,
        configured_probabilities=probabilities,
    )


def test_geometry_matches_hand_computed_active_alphabet_math():
    result = _analyze()

    assert result["training"] == {
        "examples": 3,
        "full_vocabulary_size": 5,
        "full_content_tokens": 10,
        "active_vocabulary_size": 4,
        "active_content_tokens": 10,
        "active_observed_token_types": 3,
        "active_unobserved_token_types": 1,
        "excluded_token_ids": [0],
    }
    assert result["validation"] == {
        "examples": 2,
        "content_tokens": 4,
        "observed_token_types": 3,
        "tokens_unseen_in_training_prefix": 1,
    }

    configured = result["mixture_grid"][0]
    expected_nll = -(2 * math.log(0.5965) + math.log(0.2995) + math.log(0.0025)) / 4
    expected_entropy = -sum(
        probability * math.log(probability) for probability in CONFIGURED_PROBABILITIES
    )
    assert configured["uniform_mixture_weight"] == 0.01
    assert configured["validation_content_token_nll_nats"] == pytest.approx(
        expected_nll, abs=1e-15
    )
    assert configured["validation_content_token_perplexity"] == pytest.approx(
        math.exp(expected_nll), abs=1e-15
    )
    assert configured["stationary_entropy_nats"] == pytest.approx(
        expected_entropy, abs=1e-15
    )
    assert configured["stationary_effective_vocabulary"] == pytest.approx(
        math.exp(expected_entropy), abs=1e-15
    )
    assert configured["training_unseen_token_mass"] == pytest.approx(0.0025)
    assert configured["maximum_token_probability"] == pytest.approx(0.5965)
    assert configured["top_10_token_mass"] == pytest.approx(1.0)
    assert configured["minimum_token_probability"] == pytest.approx(0.0025)

    uniform = result["uniform_baseline"]
    assert uniform == {
        "validation_content_token_nll_nats": pytest.approx(math.log(4)),
        "validation_content_token_perplexity": 4.0,
        "stationary_entropy_nats": pytest.approx(math.log(4)),
        "stationary_effective_vocabulary": 4.0,
        "training_unseen_token_mass": 0.25,
        "maximum_token_probability": 0.25,
        "top_10_token_mass": 1.0,
        "minimum_token_probability": 0.25,
    }
    assert (
        result["mixture_grid"][0]["training_unseen_token_mass"]
        < result["mixture_grid"][1]["training_unseen_token_mass"]
    )
    assert result["configured_process_agreement"]["status"] == (
        "live_process_matches_audited_formula"
    )
    assert (
        "contains 1 token observations unseen"
        in result["grid_minimum_validation_nll"]["qualification"]
    )


@pytest.mark.parametrize(
    "mixtures",
    [
        (),
        (0.0,),
        (1.0,),
        (-0.1,),
        (1.1,),
        (float("nan"),),
        (float("inf"),),
        (True,),
        ("0.1",),
        (None,),
        (0.1, 0.01),
        (0.1, 0.1),
    ],
)
def test_mixture_grid_rejects_invalid_or_ambiguous_values(mixtures):
    with pytest.raises(ValueError, match="mixture|at least"):
        audit.analyze_prior_geometry(
            _frequency_fixture(),
            _panel_fixture(),
            mixtures,
            active_token_ids=ACTIVE_IDS,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frequency, panel: frequency.update(schema_version=2), "schema"),
        (
            lambda frequency, panel: frequency["dataset"].update(revision="other"),
            "dataset identity",
        ),
        (
            lambda frequency, panel: panel["tokenizer"].update(revision="other"),
            "tokenizer mismatch",
        ),
        (
            lambda frequency, panel: frequency.update(counts_by_token_id=[0, 6, 3, 1]),
            "training token counts",
        ),
        (
            lambda frequency, panel: frequency.update(content_token_count=11),
            "content-token total",
        ),
        (
            lambda frequency, panel: frequency.update(observed_token_types=4),
            "observed-type count",
        ),
        (
            lambda frequency, panel: frequency.update(unobserved_token_types=1),
            "unobserved-type count",
        ),
        (
            lambda frequency, panel: frequency.update(
                counts_by_token_id=[1, 5, 3, 1, 0],
                observed_token_types=4,
                unobserved_token_types=1,
            ),
            "special token",
        ),
        (
            lambda frequency, panel: frequency["counts_by_token_id"].__setitem__(
                1, True
            ),
            "training token counts",
        ),
        (
            lambda frequency, panel: panel.update(sample_count=3),
            "validation rows",
        ),
        (
            lambda frequency, panel: panel["rows"][0].update(content_length=1),
            "content length",
        ),
        (
            lambda frequency, panel: panel["rows"][1].update(source_index=0),
            "source indices",
        ),
        (
            lambda frequency, panel: panel["rows"][0]["input_ids"].__setitem__(1, 5),
            "invalid token IDs",
        ),
        (
            lambda frequency, panel: panel["tokenizer"].update(special_token_ids=[5]),
            "special-token IDs|tokenizer mismatch",
        ),
    ],
)
def test_input_validation_rejects_semantic_drift(mutation, message):
    frequencies = _frequency_fixture()
    panel = _panel_fixture()
    mutation(frequencies, panel)
    with pytest.raises(ValueError, match=message):
        audit.validate_inputs(frequencies, panel)


def test_input_validation_rejects_token_order_drift():
    panel = _panel_fixture()
    panel["rows"][0]["input_ids"][1] = 2
    with pytest.raises(ValueError, match="ordered-token digest"):
        audit.validate_inputs(_frequency_fixture(), panel)


@pytest.mark.parametrize(
    "active_ids",
    [(), (1,), (1, 1, 2), (2, 1), (1, 2, 5), (True, 2)],
)
def test_geometry_rejects_invalid_active_alphabet(active_ids):
    with pytest.raises(ValueError, match="active"):
        audit.analyze_prior_geometry(
            _frequency_fixture(),
            _panel_fixture(),
            (0.01,),
            active_token_ids=active_ids,
        )


def test_geometry_rejects_live_process_probability_disagreement():
    with pytest.raises(RuntimeError, match="live configured prior"):
        _analyze(probabilities=(0.5964, 0.2996, 0.1015, 0.0025))


def test_real_frozen_inputs_reproduce_registered_process_and_geometry():
    frequencies, frequency_bytes = audit._load_frozen_json(
        audit.FREQUENCY_PATH, FREQUENCY_SHA256
    )
    panel, panel_bytes = audit._load_frozen_json(
        audit.VALIDATION_PANEL_PATH, PANEL_SHA256
    )
    assert hashlib.sha256(frequency_bytes).hexdigest() == FREQUENCY_SHA256
    assert hashlib.sha256(panel_bytes).hexdigest() == PANEL_SHA256
    contract, active_ids, probabilities = audit.load_empirical_process_contract(
        frequencies
    )
    result = audit.analyze_prior_geometry(
        frequencies,
        panel,
        (0.001, 0.01, 0.05),
        active_token_ids=active_ids,
        configured_probabilities=probabilities,
    )

    assert contract["exact_process_backend"] == (
        "genmol.diffusion.ContinuousCategoricalDiffusion"
    )
    assert contract["resolved_relevant_config_sha256"] == (
        "c030d4948365b1be7bc3b8e67a12c733dd3dfc6e32b1305ff92b567976a3708e"
    )
    assert contract["prior_metadata"]["stationary_probs_sha256"] == (
        "51aa38acaf5cf4d5642c30dbdf14246e9540d4711917265cd1961e0df1902c97"
    )
    assert contract["prior_metadata_sha256"] == (
        "21cc62825086fa381e7918c6feb81ee6b4f8bcabd0857e26b3d32fc701540bbe"
    )
    assert active_ids == tuple(range(1880))
    assert result["training"] == {
        "examples": 10_000,
        "full_vocabulary_size": 1_880,
        "full_content_tokens": 517_090,
        "active_vocabulary_size": 1_880,
        "active_content_tokens": 517_090,
        "active_observed_token_types": 184,
        "active_unobserved_token_types": 1_696,
        "excluded_token_ids": [],
    }
    assert result["validation"] == {
        "examples": 256,
        "content_tokens": 13_627,
        "observed_token_types": 64,
        "tokens_unseen_in_training_prefix": 0,
    }
    configured = result["configured_0_01"]
    assert configured["validation_content_token_nll_nats"] == pytest.approx(
        2.759559111037053
    )
    assert configured["validation_content_token_perplexity"] == pytest.approx(
        15.792878507273613
    )
    assert configured["training_unseen_token_mass"] == pytest.approx(
        0.009021276595744681
    )
    assert configured["stationary_effective_vocabulary"] == pytest.approx(
        17.47562729522436
    )
    assert (
        result["configured_process_agreement"]["maximum_absolute_formula_difference"]
        <= 2e-15
    )
    assert "no tokens unseen" in result["grid_minimum_validation_nll"]["qualification"]


def test_build_audit_preserves_provenance_and_nonranking_scope():
    process_contract = {"contract": "pinned"}
    inputs = {"script.py": {"sha256": "b" * 64}}
    result = audit.build_audit(
        _frequency_fixture(),
        _panel_fixture(),
        (0.01, 0.1),
        git_commit="a" * 40,
        inputs=inputs,
        process_contract=process_contract,
        active_token_ids=ACTIVE_IDS,
        configured_probabilities=CONFIGURED_PROBABILITIES,
    )

    assert result["schema_version"] == 1
    assert result["claim_scope"] == (
        "This held-out unigram/concentration analysis performs no model training "
        "or molecular generation. It cannot rank generators, estimate chemical "
        "quality, or establish that UDLM beats GenMol."
    )
    assert result["git"] == {
        "commit": "a" * 40,
        "upstream": "a" * 40,
        "dirty": False,
    }
    assert result["inputs"]["source_files"] == inputs
    assert result["configured_process_contract"] == process_contract
    assert "active alphabet A" in result["definitions"]["stationary_prior"]


def test_length_prefixed_digest_known_vector():
    assert audit._length_prefixed_digest([b"[0,1,2]", b"[0,1,4]"]) == (
        ORDERED_FIXTURE_SHA256
    )


def test_output_binding_rejects_existing_outside_and_symlink_escape(
    tmp_path, monkeypatch
):
    root = tmp_path / "repo"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    monkeypatch.setattr(audit, "REPOSITORY_ROOT", root)
    existing = root / "existing.json"
    existing.write_text("occupied")
    (root / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(FileExistsError, match="overwrite"):
        audit._bind_output(existing)
    with pytest.raises(ValueError, match="inside"):
        audit._bind_output(outside / "result.json")
    with pytest.raises(ValueError, match="inside"):
        audit._bind_output(root / ".." / "outside" / "result.json")
    with pytest.raises(ValueError, match="inside"):
        audit._bind_output(root / "escape" / "result.json")


def test_writer_publishes_once_without_leaving_temporary_files(tmp_path):
    output = tmp_path / "nested" / "audit.json"
    audit._write_exclusive(output, {"answer": 42})

    assert json.loads(output.read_text()) == {"answer": 42}
    assert output.stat().st_mode & 0o777 == 0o644
    assert list(output.parent.glob(f".{output.name}.*.tmp")) == []
    with pytest.raises(FileExistsError, match="overwrite"):
        audit._write_exclusive(output, {"answer": 43})
    assert json.loads(output.read_text()) == {"answer": 42}


def test_frozen_json_rejects_duplicate_keys_and_nonfinite_constants(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"x": 1, "x": 2}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        audit._load_frozen_json(
            duplicate,
            "3d007c3ece9a36d5f10f8a65c4e1e99eeec84902b260fd2fce21a3f49072cf97",
        )

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"x": NaN}')
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        audit._load_frozen_json(
            nonfinite,
            "baab068bbf85ddffee705062ebcac22b5b8e58b7634a5fc48ac941d3968349c2",
        )

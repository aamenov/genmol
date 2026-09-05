"""CPU-only tests for the PMO ablation runner."""

from __future__ import annotations

import random
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import numpy as np

from genmol.utils.utils_chem import cut, cut_all
from scripts.exps.pmo.run_ablation import (
    ORACLES,
    PAPER_GAMMA,
    SAFE_EXPERIMENT_ID,
    VARIANT_SETTINGS,
    CachedOracle,
    _attribution_metadata,
    _attach_fragments,
    _derived_seed,
    _git_output,
    _molecule_size_bounds,
    _repair_event_tail,
    _resolved_config,
    _transition_observation_id,
    _validate_args,
    _vocabulary_has_sufficient_statistics,
    run,
)


class CountingEvaluator:
    def __init__(self):
        self.calls = []

    def __call__(self, smiles):
        self.calls.append(smiles)
        return len(smiles) / 100


class CachedOracleTests(unittest.TestCase):
    def test_cache_and_budget_use_unique_canonical_molecules(self):
        evaluator = CountingEvaluator()
        oracle = CachedOracle(evaluator, budget=2)

        first = oracle.score("C(C)O")
        duplicate = oracle.score("CCO")
        second = oracle.score("CCN")
        exhausted = oracle.score("CCC")

        self.assertTrue(first.charged)
        self.assertFalse(duplicate.charged)
        self.assertEqual(duplicate.reason, "cache_hit")
        self.assertTrue(second.charged)
        self.assertEqual(exhausted.reason, "budget_exhausted")
        self.assertEqual(oracle.calls, 2)
        self.assertEqual(len(evaluator.calls), 2)
        self.assertEqual(oracle.scores_in_call_order(), [0.03, 0.03])

    def test_oracle_state_roundtrip_checks_contiguous_indices(self):
        oracle = CachedOracle(CountingEvaluator(), budget=3)
        oracle.score("CCO")
        oracle.score("CCN")
        state = oracle.state_dict()

        restored = CachedOracle(CountingEvaluator(), budget=3)
        restored.load_state_dict(state)
        self.assertEqual(restored.buffer, oracle.buffer)

        state["buffer"]["CCC"] = [0.2, 4]
        with self.assertRaisesRegex(ValueError, "contiguous"):
            restored.load_state_dict(state)


class ConfigurationTests(unittest.TestCase):
    def test_git_output_preserves_porcelain_status_prefix(self):
        completed = mock.Mock(stdout=" M first.py\n?? second.py\n")
        with mock.patch(
            "scripts.exps.pmo.run_ablation.subprocess.run",
            return_value=completed,
        ):
            self.assertEqual(_git_output("status"), " M first.py\n?? second.py")

    def test_paper_gamma_covers_every_cli_oracle(self):
        self.assertEqual(set(PAPER_GAMMA), set(ORACLES))
        self.assertTrue(all(0 <= value <= 1 for value in PAPER_GAMMA.values()))

    def test_task_specific_size_bounds_match_release(self):
        self.assertEqual(_molecule_size_bounds("qed", 20, 40), (10, 30))
        self.assertEqual(_molecule_size_bounds("jnk3", 20, 40), (30, 80))
        self.assertEqual(_molecule_size_bounds("drd2", 20, 40), (20, 40))

    def test_rng_stream_seeds_are_stable_and_distinct(self):
        self.assertEqual(_derived_seed(7, "a"), _derived_seed(7, "a"))
        self.assertNotEqual(_derived_seed(7, "a"), _derived_seed(7, "b"))
        self.assertNotEqual(_derived_seed(7, "a"), _derived_seed(8, "a"))

    def test_experiment_identifier_cannot_escape_output_root(self):
        self.assertIsNotNone(SAFE_EXPERIMENT_ID.fullmatch("pilot-50k_v1.2"))
        self.assertIsNone(SAFE_EXPERIMENT_ID.fullmatch("../outside"))
        self.assertIsNone(SAFE_EXPERIMENT_ID.fullmatch("/absolute"))

    def test_enriched_vocabulary_detection_requires_both_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            enriched = root / "enriched.csv"
            enriched.write_text("frag,score,count,score_sum\nf,0.5,2,1.0\n")
            legacy = root / "legacy.csv"
            legacy.write_text("frag,score,size\nf,0.5,2\n")
            self.assertTrue(_vocabulary_has_sufficient_statistics(enriched))
            self.assertFalse(_vocabulary_has_sufficient_statistics(legacy))

    def test_resume_tail_repair_archives_uncheckpointed_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            first = b'{"event":1}\n'
            tail = b'{"event":2}\n{"partial":'
            path.write_bytes(first + tail)
            orphaned = _repair_event_tail(path, expected_events=1)
            self.assertIsNotNone(orphaned)
            self.assertEqual(path.read_bytes(), first)
            assert orphaned is not None
            self.assertEqual(orphaned.read_bytes(), tail)

    def test_delta_and_matched_control_resolve_the_same_noncredit_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.ckpt"
            model.write_bytes(b"model")
            vocab = root / "vocab.csv"
            vocab.write_text("frag,score\n[1*]CC,0.5\n[1*]CO,0.4\n")
            expected = {
                "warmup_update_policy": "frozen",
                "observation_identity": "unique_canonical_parent_child_transition",
                "parent_domain_policy": "parent_and_child_within_configured_atom_bounds",
                "credit_fragment_policy": "deterministic_cut_all_child_minus_parent",
                "statistical_duplicate_policy": (
                    "one update per unique canonical parent-child transition"
                ),
            }
            for variant in ("delta", "running_mean_delta_control"):
                args = RunnerIntegrationTests()._args(
                    root,
                    model,
                    vocab,
                    variant=variant,
                )
                config = _resolved_config(args)
                for key, value in expected.items():
                    self.assertEqual(config[key], value)

        first = _transition_observation_id("CCO", "CCOC")
        self.assertEqual(first, _transition_observation_id("CCO", "CCOC"))
        self.assertNotEqual(first, _transition_observation_id("CCN", "CCOC"))


class ChemistryTests(unittest.TestCase):
    def test_injected_cut_rng_is_reproducible_and_isolated(self):
        smiles = "CCOC(=O)NCC"
        first = cut(smiles, rng=random.Random(123))
        second = cut(smiles, rng=random.Random(123))
        self.assertEqual(first, second)

        random.seed(99)
        expected_next = random.random()
        random.seed(99)
        cut(smiles, rng=random.Random(123))
        self.assertEqual(random.random(), expected_next)

    def test_exhaustive_cut_is_deterministic(self):
        smiles = "CCOC(=O)NCC"
        self.assertEqual(cut_all(smiles), cut_all(smiles))
        self.assertGreater(len(cut_all(smiles)), 0)

    def test_attachment_uses_explicit_numpy_rng(self):
        first = _attach_fragments("[1*]CC", "[1*]N", np.random.default_rng(9))
        second = _attach_fragments("[1*]CC", "[1*]N", np.random.default_rng(9))
        self.assertEqual(first, second)
        self.assertIsNotNone(first)

    def test_delta_attribution_uses_deterministic_set_difference(self):
        fragments = {
            "parent": {"shared", "removed"},
            "child": {"shared", "added-a", "added-b"},
        }
        with mock.patch(
            "scripts.exps.pmo.run_ablation.cut_all",
            side_effect=lambda smiles: fragments[smiles],
        ):
            metadata = _attribution_metadata(
                "parent", "child", delta_attribution="novel_vs_parent"
            )

        self.assertEqual(metadata["parent_all_fragments"], ["removed", "shared"])
        self.assertEqual(
            metadata["credited_fragments"], ["added-a", "added-b"]
        )
        self.assertEqual(
            metadata["mapping_counts"],
            {"parent_all": 2, "child_all": 3, "shared": 1, "credited": 2},
        )
        self.assertAlmostEqual(metadata["mapping_coverage"], 2 / 3)


class FakeModel:
    def __init__(self):
        self.device = "cpu"

    def to(self, device):
        self.device = device
        return self


class FakeMDLM:
    def to_device(self, device):
        self.device = device


class FakeSampler:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = FakeModel()
        self.mdlm = FakeMDLM()

    def mask_modification(self, smiles, **kwargs):
        return smiles + "C"


class ConstantChildSampler(FakeSampler):
    def mask_modification(self, smiles, **kwargs):
        return "CCOC"


class RunnerIntegrationTests(unittest.TestCase):
    def _args(self, root, model_path, vocab_path, **overrides):
        values = {
            "oracle": "qed",
            "variant": "released",
            "model_path": model_path,
            "vocab_path": vocab_path,
            "device": "cpu",
            "seed": 0,
            "max_oracle_calls": 5,
            "reporting_frequency": 1,
            "checkpoint_every": 2,
            "max_iterations": 20,
            "population_size": 3,
            "warmup": 100,
            "legacy_warmup_off_by_one": True,
            "gamma": 0.0,
            "softmax_temp": 1.2,
            "randomness": 2.0,
            "guidance_scale": 2.0,
            "min_mol_size": 1,
            "max_mol_size": 100,
            "legacy_seed_count": 1,
            "prior_mean": None,
            "prior_mean_source": None,
            "delta_attribution": "novel_vs_parent",
            "experiment_id": "unit_test",
            "scientific_status": "unit-test synthetic run; not a scientific result",
            "output_root": root,
            "resume": False,
            "durable_events": True,
        }
        values.update(overrides)
        return Namespace(**values)

    def _files(self, directory):
        root = Path(directory)
        model = root / "model.ckpt"
        model.write_bytes(b"fake model")
        vocab = root / "vocab.csv"
        vocab.write_text(
            "frag,score,size\n"
            "[1*]CC,0.9,2\n"
            "[1*]CN,0.8,2\n"
            "[1*]CO,0.7,2\n"
        )
        return model, vocab

    def test_resolved_config_records_event_durability(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, vocab = self._files(root)
            args = self._args(
                root,
                model_path=model,
                vocab_path=vocab,
                durable_events=True,
            )
            self.assertIs(_resolved_config(args)["durable_events"], True)
            args.durable_events = False
            self.assertIs(_resolved_config(args)["durable_events"], False)

    def test_bayesian_variant_metadata_controls_strength_and_prior_validation(self):
        expected_strengths = {
            "shrink1": 1.0,
            "shrink3": 3.0,
            "shrink10": 10.0,
            "shrink30": 30.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, vocab = self._files(root)
            for variant, strength in expected_strengths.items():
                with self.subTest(variant=variant):
                    args = self._args(
                        root,
                        model,
                        vocab,
                        variant=variant,
                        prior_mean=0.5,
                        prior_mean_source="frozen unit-test prior",
                    )
                    _validate_args(args)
                    config = _resolved_config(args)
                    self.assertEqual(VARIANT_SETTINGS[variant]["mode"], "bayes")
                    self.assertEqual(
                        VARIANT_SETTINGS[variant]["prior_strength"], strength
                    )
                    self.assertEqual(config["prior_strength"], strength)
                    self.assertEqual(config["prior_mean"], 0.5)

            missing_prior = self._args(
                root,
                model,
                vocab,
                variant="shrink1",
                prior_mean=None,
                prior_mean_source=None,
            )
            with self.assertRaisesRegex(ValueError, "shrink1 requires"):
                _validate_args(missing_prior)

            non_bayesian = self._args(
                root,
                model,
                vocab,
                variant="running_mean",
                prior_mean=0.5,
                prior_mean_source="not allowed",
            )
            with self.assertRaisesRegex(ValueError, "Bayesian variants"):
                _validate_args(non_bayesian)

    def test_end_to_end_mocked_run_reaches_exact_budget_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, vocab = self._files(root)
            parents = iter(["CCO", "CCN", "CCC", "CCOC", "CCNC"])
            with (
                mock.patch("scripts.exps.pmo.run_ablation.Sampler", FakeSampler),
                mock.patch("scripts.exps.pmo.run_ablation.TDCOracle", return_value=lambda smi: len(smi) / 10),
                mock.patch("scripts.exps.pmo.run_ablation._attach_fragments", side_effect=lambda *args: next(parents)),
                mock.patch("scripts.exps.pmo.run_ablation._molecule_size_bounds", return_value=(1, 100)),
            ):
                run_dir = run(self._args(root / "output", model, vocab))

            summary = __import__("json").loads((run_dir / "summary.json").read_text())
            self.assertEqual(summary["schema_version"], 2)
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["scores"]["all_charged_molecules"]["oracle_calls"], 5)
            self.assertTrue(summary["checkpoint_consistent"])
            self.assertTrue((run_dir / "state" / "latest.pkl").exists())
            self.assertEqual(len((run_dir / "events.jsonl").read_text().splitlines()), 5)

    def test_parent_control_stops_after_final_parent_without_overshoot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, vocab = self._files(root)
            parents = iter(["CCO", "CCN"])
            args = self._args(
                root / "output",
                model,
                vocab,
                variant="running_mean_parent_control",
                max_oracle_calls=3,
                warmup=0,
                legacy_warmup_off_by_one=False,
            )
            with (
                mock.patch("scripts.exps.pmo.run_ablation.Sampler", FakeSampler),
                mock.patch("scripts.exps.pmo.run_ablation.TDCOracle", return_value=lambda smi: len(smi) / 10),
                mock.patch("scripts.exps.pmo.run_ablation._attach_fragments", side_effect=lambda *args: next(parents)),
                mock.patch("scripts.exps.pmo.run_ablation._molecule_size_bounds", return_value=(1, 100)),
            ):
                run_dir = run(args)

            summary = __import__("json").loads((run_dir / "summary.json").read_text())
            self.assertEqual(summary["scores"]["all_charged_molecules"]["oracle_calls"], 3)
            total_axis = summary["scores"]["charged_children_total_call_axis"]
            child_axis = summary["scores"]["charged_children_child_count_axis"]
            self.assertEqual(total_axis["oracle_calls"], 3)
            self.assertEqual(total_axis["score_count"], 1)
            self.assertEqual(child_axis["score_count"], 1)
            self.assertEqual(child_axis["child_count_horizon"], 1)
            events = [__import__("json").loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
            self.assertEqual(events[-1]["population_update"]["reason"], "budget_after_parent")
            self.assertIn("elapsed_seconds", events[-1])
            self.assertIn("population_cutoff_after", events[-1])

    def test_corrected_delta_and_control_match_transition_but_not_credit_value(self):
        captured = {}
        for variant in ("delta", "running_mean_delta_control"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model, vocab = self._files(root)
                args = self._args(
                    root / "output",
                    model,
                    vocab,
                    variant=variant,
                    experiment_id=f"unit_{variant}",
                    max_oracle_calls=2,
                    warmup=0,
                    legacy_warmup_off_by_one=False,
                )
                parents = iter(["C", "CCO"])
                with (
                    mock.patch("scripts.exps.pmo.run_ablation.Sampler", FakeSampler),
                    mock.patch(
                        "scripts.exps.pmo.run_ablation.TDCOracle",
                        return_value=lambda smiles: len(smiles) / 10,
                    ),
                    mock.patch(
                        "scripts.exps.pmo.run_ablation._attach_fragments",
                        side_effect=lambda *unused: next(parents),
                    ),
                    mock.patch(
                        "scripts.exps.pmo.run_ablation._molecule_size_bounds",
                        return_value=(3, 4),
                    ),
                ):
                    run_dir = run(args)

                event = json.loads((run_dir / "events.jsonl").read_text().strip())
                self.assertEqual(event["proposal_attempts"], 2)
                self.assertEqual(event["parent_smiles"], "CCO")
                self.assertEqual(event["child_smiles"], "CCOC")
                self.assertEqual(event["parent_atom_count"], 3)
                self.assertEqual(event["child_atom_count"], 4)
                self.assertTrue(event["attribution"]["applicable"])
                self.assertEqual(
                    event["population_update"]["observed_fragments"],
                    event["attribution"]["credited_fragments"],
                )
                expected_credit = 0.1 if variant == "delta" else 0.4
                for stats in event["fragment_statistics_after"].values():
                    self.assertEqual(stats["count"], 1)
                    self.assertAlmostEqual(stats["total"], expected_credit)
                captured[variant] = event

        delta_event = captured["delta"]
        control_event = captured["running_mean_delta_control"]
        for key in (
            "selected_fragments",
            "parent_smiles",
            "child_smiles",
            "parent_atom_count",
            "child_atom_count",
            "proposal_attempts",
            "remask_enabled",
            "attribution",
        ):
            self.assertEqual(delta_event[key], control_event[key])

    def test_delta_matched_control_freezes_vocabulary_during_warmup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, vocab = self._files(root)
            args = self._args(
                root / "output",
                model,
                vocab,
                variant="running_mean_delta_control",
                max_oracle_calls=1,
                warmup=10,
                legacy_warmup_off_by_one=False,
            )
            with (
                mock.patch("scripts.exps.pmo.run_ablation.Sampler", FakeSampler),
                mock.patch(
                    "scripts.exps.pmo.run_ablation.TDCOracle",
                    return_value=lambda smiles: len(smiles) / 10,
                ),
                mock.patch(
                    "scripts.exps.pmo.run_ablation._attach_fragments",
                    return_value="CCO",
                ),
                mock.patch(
                    "scripts.exps.pmo.run_ablation._molecule_size_bounds",
                    return_value=(3, 4),
                ),
            ):
                run_dir = run(args)

            event = json.loads((run_dir / "events.jsonl").read_text().strip())
            self.assertFalse(event["remask_enabled"])
            self.assertIsNone(event["parent_oracle"])
            self.assertEqual(event["population_update"]["reason"], "frozen_warmup")
            self.assertEqual(event["attribution"]["reason"], "warmup_frozen")
            self.assertFalse(event["attribution"]["applicable"])

    def test_delta_deduplicates_transitions_not_repeated_children(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, vocab = self._files(root)
            args = self._args(
                root / "output",
                model,
                vocab,
                variant="delta",
                max_oracle_calls=4,
                warmup=0,
                legacy_warmup_off_by_one=False,
            )
            parents = iter(["CCO", "CCN", "COC"])
            with (
                mock.patch(
                    "scripts.exps.pmo.run_ablation.Sampler", ConstantChildSampler
                ),
                mock.patch(
                    "scripts.exps.pmo.run_ablation.TDCOracle",
                    return_value=lambda smiles: len(smiles) / 10,
                ),
                mock.patch(
                    "scripts.exps.pmo.run_ablation._attach_fragments",
                    side_effect=lambda *unused: next(parents),
                ),
                mock.patch(
                    "scripts.exps.pmo.run_ablation._molecule_size_bounds",
                    return_value=(3, 4),
                ),
            ):
                run_dir = run(args)

            events = [
                json.loads(line)
                for line in (run_dir / "events.jsonl").read_text().splitlines()
            ]
            self.assertEqual(events[0]["child_smiles"], events[1]["child_smiles"])
            self.assertNotEqual(events[0]["parent_smiles"], events[1]["parent_smiles"])
            self.assertEqual(events[0]["population_update"]["reason"], "updated")
            self.assertEqual(events[1]["population_update"]["reason"], "updated")
            shared_credited = set(events[0]["attribution"]["credited_fragments"]) & set(
                events[1]["attribution"]["credited_fragments"]
            )
            self.assertTrue(shared_credited)
            for fragment in shared_credited:
                self.assertEqual(
                    events[1]["fragment_statistics_after"][fragment]["count"], 2
                )


if __name__ == "__main__":
    unittest.main()

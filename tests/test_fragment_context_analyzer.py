"""Production-faithful tests for fragment-context evidence analysis."""

from __future__ import annotations

import fcntl
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable
from unittest import mock

from scripts.exps.pmo import analyze_fragment_contexts as analyzer
from scripts.exps.pmo.main.genmol.experiment_io import (
    append_event,
    load_checkpoint,
    save_checkpoint,
    sha256_config,
    sha256_file,
    summarize_indexed_scores,
    summarize_scores,
    write_manifest,
)


class FragmentContextAnalyzerTests(unittest.TestCase):
    def test_git_output_preserves_porcelain_status_prefix(self):
        completed = mock.Mock(stdout=" M first.py\n?? second.py\n")
        with mock.patch.object(analyzer.subprocess, "run", return_value=completed):
            self.assertEqual(
                analyzer._git_output("status"),
                " M first.py\n?? second.py",
            )

    def _fixture(
        self,
        directory: str | Path,
        *,
        summary_schema: int = 2,
        enriched_vocabulary: bool = False,
        enriched_seed_a_score: str = "0.5",
    ) -> dict[str, Path]:
        root = Path(directory)
        run_dir = root / "qed" / "running_mean_parent_control" / "seed_7"
        run_dir.mkdir(parents=True)
        model = root / "model.ckpt"
        vocabulary = root / "vocab.csv"
        seed_a_score = (
            float(enriched_seed_a_score) if enriched_vocabulary else 0.5
        )
        model.write_bytes(b"model checkpoint")
        vocabulary.write_text(
            (
                "frag,score,count,score_sum\n"
                f"seed-A,{enriched_seed_a_score},2,1.0\n"
                "seed-B,0.4,2,0.8\n"
                if enriched_vocabulary
                else "frag,score\nseed-A,0.5\nseed-B,0.4\n"
            ),
            encoding="utf-8",
        )

        config = {
            "experiment_id": "context_validation",
            "scientific_status": "production-faithful unit-test fixture",
            "oracle": "qed",
            "variant": "running_mean_parent_control",
            "policy_mode": "mean",
            "parent_control": True,
            "seed": 7,
            "max_oracle_calls": 3,
            "reporting_frequency": 1,
            "checkpoint_every": 1,
            "max_iterations": 20,
            "population_size": 2,
            "warmup": 0,
            "legacy_warmup_off_by_one": False,
            "min_support": 1,
            "prior_strength": 0.0,
            "prior_mean": None,
            "legacy_seed_count": None if enriched_vocabulary else 2,
            "model_path": str(model.resolve()),
            "vocab_path": str(vocabulary.resolve()),
            "population_sampling_order": "canonical fragment string before uniform sampling",
        }
        config_hash = sha256_config(config)
        git = {
            "commit": "c" * 40,
            "branch": "codex/test",
            "dirty": False,
            "status": [],
            "tracked_diff_sha256": "d" * 64,
        }
        manifest = {
            "schema_version": 1,
            "run_id": "context_validation:qed:running_mean_parent_control:seed7",
            "task": "qed",
            "variant": "running_mean_parent_control",
            "seed": 7,
            "status": "completed",
            "error": None,
            "oracle_budget": 3,
            "oracle_calls": 3,
            "elapsed_seconds": 1.3,
            "config": config,
            "config_sha256": config_hash,
            "model": {
                "path": str(model.resolve()),
                "size_bytes": model.stat().st_size,
                "sha256": sha256_file(model),
            },
            "extra": {
                "git": git,
                "vocabulary": {
                    "path": str(vocabulary.resolve()),
                    "sha256": sha256_file(vocabulary),
                },
            },
            "runtime": {"executable": "python", "hostname": "test"},
            "events_path": str((run_dir / "events.jsonl").resolve()),
            "summary_path": str((run_dir / "summary.json").resolve()),
            "checkpoint_path": str((run_dir / "state" / "latest.pkl").resolve()),
        }

        event0 = {
            "event_index": 0,
            "iteration": 0,
            "selected_fragments": ["seed-A", "seed-B"],
            "parent_smiles": "CC",
            "child_smiles": "CC",
            "atom_count": 2,
            "proposal_attempts": 1,
            "remask_enabled": True,
            "parent_oracle": self._oracle_row("CC", 0.9, 1, charged=True),
            "child_oracle": self._oracle_row("CC", 0.9, 1, charged=False),
            "population_update": {
                "updated": True,
                "reason": "updated",
                "observed_fragments": ["novel-X", "seed-A"],
                "admitted": ["novel-X"],
                "displaced": ["seed-B"],
            },
            "fragment_statistics_after": {
                "novel-X": self._record("novel-X", 0.9, 1, None, None, 1, 1),
                "seed-A": self._record(
                    "seed-A", 1.9, 3, seed_a_score, 0, 1, 1
                ),
            },
            "population_cutoff_after": 1.9 / 3,
            "population_size_after": 2,
            "oracle_calls": 1,
            "top_1": 0.9,
            "top_10": 0.9,
            "top_100": 0.9,
            "elapsed_seconds": 0.1,
        }
        event1 = {
            "event_index": 1,
            "iteration": 1,
            "selected_fragments": ["novel-X", "seed-A"],
            "parent_smiles": "CC",
            "child_smiles": "CCC",
            "atom_count": 3,
            "proposal_attempts": 1,
            "remask_enabled": True,
            "parent_oracle": self._oracle_row("CC", 0.9, 1, charged=False),
            "child_oracle": self._oracle_row("CCC", 0.3, 2, charged=True),
            "population_update": {
                "updated": True,
                "reason": "updated",
                "observed_fragments": ["novel-X", "novel-Y", "seed-A"],
                "admitted": [],
                "displaced": [],
            },
            "fragment_statistics_after": {
                "novel-X": self._record("novel-X", 1.2, 2, None, None, 1, 2),
                "novel-Y": self._record("novel-Y", 0.3, 1, None, None, 2, 2),
                "seed-A": self._record(
                    "seed-A", 2.2, 4, seed_a_score, 0, 1, 2
                ),
            },
            "population_cutoff_after": 2.2 / 4,
            "population_size_after": 2,
            "oracle_calls": 2,
            "top_1": 0.9,
            "top_10": 0.6,
            "top_100": 0.6,
            "elapsed_seconds": 0.2,
        }
        event2 = {
            "event_index": 2,
            "iteration": 2,
            "selected_fragments": ["novel-X", "seed-A"],
            "parent_smiles": "CCCC",
            "child_smiles": "CCCC",
            "atom_count": 4,
            "proposal_attempts": 1,
            "remask_enabled": True,
            "parent_oracle": self._oracle_row("CCCC", 0.6, 3, charged=True),
            "child_oracle": None,
            "population_update": {
                "updated": False,
                "reason": "budget_after_parent",
            },
            "fragment_statistics_after": {},
            "population_cutoff_after": 2.2 / 4,
            "population_size_after": 2,
            "oracle_calls": 3,
            "top_1": 0.9,
            "top_10": 0.6,
            "top_100": 0.6,
            "elapsed_seconds": 0.3,
        }
        events_path = run_dir / "events.jsonl"
        for event in (event0, event1, event2):
            append_event(events_path, event)

        final_records = [
            self._record("novel-X", 1.2, 2, None, None, 1, 2),
            self._record("novel-Y", 0.3, 1, None, None, 2, 2),
            self._record("seed-A", 2.2, 4, seed_a_score, 0, 1, 2),
            self._record("seed-B", 0.8, 2, 0.4, 1, 0, 0),
        ]
        population_state = {
            "version": 1,
            "config": {
                "capacity": 2,
                "mode": "mean",
                "min_support": 1,
                "prior_strength": 0.0,
                "prior_mean": None,
                "legacy_seed_count": None if enriched_vocabulary else 2,
                "deduplicate_observations": True,
                "delta_missing_parent": "skip",
            },
            "event_index": 2,
            "seen_observation_ids": ["CC", "CCC"],
            "released_population": [],
            "records": final_records,
            "rng_state": None,
        }
        checkpoint_path = run_dir / "state" / "latest.pkl"
        save_checkpoint(
            checkpoint_path,
            {
                "next_iteration": 3,
                "event_count": 3,
                "elapsed_seconds": 1.2,
                "next_checkpoint_call": 4,
                "population": population_state,
                "oracle": {
                    "budget": 3,
                    "buffer": {
                        "CC": [0.9, 1],
                        "CCC": [0.3, 2],
                        "CCCC": [0.6, 3],
                    },
                },
                "rng": {},
            },
            metadata={
                "config_sha256": config_hash,
                "model_sha256": manifest["model"]["sha256"],
                "vocabulary_sha256": manifest["extra"]["vocabulary"]["sha256"],
                "git_commit": git["commit"],
                "tracked_diff_sha256": git["tracked_diff_sha256"],
            },
        )

        all_scores = summarize_scores([0.9, 0.3, 0.6], reporting_frequency=1, budget=3)
        child_indexed = [(2, 0.3)]
        scores: dict[str, Any] = {
            "all_charged_molecules": all_scores,
            "interpretation": "production-faithful unit test",
        }
        if summary_schema == 1:
            scores["charged_children_only"] = summarize_scores(
                [0.3], reporting_frequency=1, budget=3
            )
        else:
            scores["charged_children_total_call_axis"] = summarize_indexed_scores(
                child_indexed,
                observed_oracle_calls=3,
                reporting_frequency=1,
                budget=3,
            )
            child_count = summarize_scores([0.3], reporting_frequency=1, budget=1)
            child_count["axis"] = "charged_child_count"
            child_count["score_count"] = child_count.pop("oracle_calls")
            child_count["child_count_horizon"] = child_count.pop("oracle_budget")
            scores["charged_children_child_count_axis"] = child_count
        summary = {
            "schema_version": summary_schema,
            "run_id": manifest["run_id"],
            "status": "completed",
            "error": None,
            "iterations_completed": 3,
            "events": 3,
            "elapsed_seconds": 1.3,
            "scores": scores,
            "population": {
                "size": 2,
                "active_rows": [[0.6, "novel-X"], [0.55, "seed-A"]],
            },
            "config_sha256": config_hash,
            "model_sha256": manifest["model"]["sha256"],
            "checkpoint_consistent": True,
            "recoverable_oracle_calls": 3,
            "recoverable_events": 3,
        }
        write_manifest(run_dir / "manifest.json", manifest)
        write_manifest(run_dir / "summary.json", summary)
        (run_dir / ".run.lock").touch()
        return {
            "run_dir": run_dir,
            "events": events_path,
            "manifest": run_dir / "manifest.json",
            "summary": run_dir / "summary.json",
            "checkpoint": checkpoint_path,
            "model": model,
            "vocabulary": vocabulary,
        }

    @staticmethod
    def _oracle_row(
        smiles: str,
        score: float,
        call_index: int,
        *,
        charged: bool,
    ) -> dict[str, Any]:
        return {
            "raw_smiles": smiles,
            "canonical_smiles": smiles,
            "valid": True,
            "score": score,
            "charged": charged,
            "call_index": call_index,
            "reason": "scored" if charged else "cache_hit",
        }

    @staticmethod
    def _record(
        fragment: str,
        total: float,
        count: int,
        seed_score: float | None,
        seed_order: int | None,
        first_seen: int,
        last_seen: int,
    ) -> dict[str, Any]:
        return {
            "fragment": fragment,
            "total": total,
            "count": count,
            "seed_score": seed_score,
            "seed_order": seed_order,
            "first_seen": first_seen,
            "last_seen": last_seen,
        }

    @staticmethod
    def _mutate_json(path: Path, mutation: Callable[[dict[str, Any]], None]) -> None:
        value = json.loads(path.read_text(encoding="utf-8"))
        mutation(value)
        path.write_text(
            json.dumps(value, allow_nan=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _mutate_checkpoint(
        path: Path,
        mutation: Callable[[dict[str, Any], dict[str, Any]], None],
    ) -> None:
        state, metadata = load_checkpoint(path, with_metadata=True)
        mutation(state, metadata)
        save_checkpoint(path, state, metadata=metadata)

    def test_replays_parent_cache_child_and_splits_seed_novel_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(directory)
            report = analyzer.analyze_fragment_contexts(paths["events"], top_n=2)

            self.assertEqual(report["validation"]["global_oracle_replay"], "passed")
            self.assertEqual(report["validation"]["statistical_population_replay"], "passed")
            self.assertEqual(report["counts"]["charged_oracle_calls"], 3)
            self.assertEqual(report["counts"]["charged_parent_oracle_calls"], 2)
            self.assertEqual(report["counts"]["charged_child_oracle_calls"], 1)
            self.assertEqual(report["counts"]["cached_parent_oracle_lookups"], 1)
            self.assertEqual(report["counts"]["cached_child_oracle_lookups"], 1)
            self.assertEqual(report["counts"]["unique_child_observations"], 2)
            self.assertEqual(report["counts"]["fragment_observations"], 5)
            self.assertEqual(report["counts"]["seed_fragment_observations"], 2)
            self.assertEqual(report["counts"]["novel_fragment_observations"], 3)

            classification = report["fragment_classification"]
            self.assertEqual(classification["checkpoint_seed_fragments"], 2)
            self.assertEqual(classification["checkpoint_novel_fragments"], 2)
            groups = report["fragment_groups"]
            self.assertEqual(
                groups["all"]["support_distribution"]["histogram"],
                {"1": 1, "2": 2},
            )
            self.assertEqual(
                groups["seed"]["support_distribution"]["histogram"],
                {"2": 1},
            )
            self.assertEqual(
                groups["novel"]["support_distribution"]["histogram"],
                {"1": 1, "2": 1},
            )
            self.assertEqual(
                groups["all"]["registry_dynamic_support"]["histogram"],
                {"0": 1, "1": 1, "2": 2},
            )
            self.assertEqual(
                groups["seed"]["registry_dynamic_support"]["histogram"],
                {"0": 1, "2": 1},
            )
            self.assertEqual(
                groups["seed"]["registry_dynamic_support"]["denominator_fragments"],
                2,
            )
            self.assertEqual(
                report["fragment_classification"][
                    "seed_fragments_with_zero_dynamic_support"
                ],
                ["seed-B"],
            )
            self.assertEqual(groups["seed"]["repeated_fragment_metrics"]["fragments"], 1)
            self.assertEqual(groups["novel"]["repeated_fragment_metrics"]["fragments"], 1)
            self.assertAlmostEqual(
                groups["novel"]["repeated_fragment_metrics"]["max_first_overshoot"],
                0.3,
            )
            self.assertEqual(
                [row["fragment"] for row in report["top_first_score_overestimates"]],
                ["novel-X", "seed-A"],
            )
            self.assertEqual(set(report["source"]), {"manifest", "summary", "events", "checkpoint"})
            self.assertEqual(
                report["analyzer"]["implementation"]["sha256"],
                sha256_file(analyzer.__file__),
            )
            helper = report["analyzer"]["imported_helpers"]["experiment_io"]
            self.assertEqual(
                helper["sha256"], sha256_file(analyzer.experiment_io.__file__)
            )
            self.assertEqual(
                set(report["analyzer"]["git"]),
                {"commit", "branch", "dirty", "status", "tracked_diff_sha256"},
            )
            self.assertEqual(
                len(report["analyzer"]["git"]["tracked_diff_sha256"]), 64
            )
            self.assertEqual(
                report["analyzer"]["runtime"]["executable"],
                __import__("sys").executable,
            )
            self.assertIn("rdkit", report["analyzer"]["runtime"]["packages"])
            evidence = report["validation"]["event_fragment_evidence"]
            self.assertFalse(evidence["chemical_decomposition_recomputed"])
            self.assertFalse(evidence["fragmentation_rng_replayed"])
            self.assertFalse(evidence["selection_rng_replayed"])
            caveats = " ".join(report["caveats"]).lower()
            self.assertIn("sampled decomposition", caveats)
            self.assertIn("oracle-call efficiency", caveats)

    def test_accepts_valid_legacy_summary_but_recomputes_its_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(directory, summary_schema=1)
            report = analyzer.analyze_fragment_contexts(paths["events"])
            self.assertEqual(report["validation"]["summary_schema_version"], 1)

            self._mutate_json(
                paths["summary"],
                lambda value: value["scores"]["charged_children_only"].__setitem__(
                    "top_1", 0.99
                ),
            )
            with self.assertRaisesRegex(analyzer.AnalysisError, "summary child metrics"):
                analyzer.analyze_fragment_contexts(paths["events"])

    def test_rejects_oracle_population_and_identity_tampering(self) -> None:
        cases: list[
            tuple[
                str,
                Callable[[dict[str, Path]], None],
                str,
            ]
        ] = [
            (
                "cached child score",
                lambda paths: self._mutate_event(
                    paths["events"], 0, ("child_oracle", "score"), 0.8
                ),
                "child_oracle.score",
            ),
            (
                "cumulative oracle calls",
                lambda paths: self._mutate_event(
                    paths["events"], 1, ("oracle_calls",), 1
                ),
                "oracle_calls",
            ),
            (
                "fragment total",
                lambda paths: self._mutate_event(
                    paths["events"],
                    1,
                    ("fragment_statistics_after", "novel-X", "total"),
                    1.4,
                ),
                "statistics for 'novel-X'.total",
            ),
            (
                "inactive selected fragment",
                lambda paths: self._mutate_event(
                    paths["events"],
                    1,
                    ("selected_fragments",),
                    ["novel-X", "BOGUS"],
                ),
                "selected_fragments contains inactive",
            ),
            (
                "duplicate selected fragment",
                lambda paths: self._mutate_event(
                    paths["events"],
                    1,
                    ("selected_fragments",),
                    ["novel-X", "novel-X"],
                ),
                "two distinct fragments",
            ),
            (
                "seen observations",
                lambda paths: self._mutate_checkpoint(
                    paths["checkpoint"],
                    lambda state, metadata: state["population"].__setitem__(
                        "seen_observation_ids", ["CC"]
                    ),
                ),
                "seen_observation_ids",
            ),
            (
                "population event index",
                lambda paths: self._mutate_checkpoint(
                    paths["checkpoint"],
                    lambda state, metadata: state["population"].__setitem__(
                        "event_index", 1
                    ),
                ),
                "population event_index",
            ),
            (
                "checkpoint git identity",
                lambda paths: self._mutate_checkpoint(
                    paths["checkpoint"],
                    lambda state, metadata: metadata.__setitem__(
                        "git_commit", "wrong"
                    ),
                ),
                "checkpoint.metadata.git_commit",
            ),
            (
                "summary config identity",
                lambda paths: self._mutate_json(
                    paths["summary"],
                    lambda value: value.__setitem__("config_sha256", "0" * 64),
                ),
                "summary.config_sha256",
            ),
        ]
        for name, tamper, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                paths = self._fixture(directory)
                tamper(paths)
                with self.assertRaisesRegex(analyzer.AnalysisError, message):
                    analyzer.analyze_fragment_contexts(paths["events"])

    def test_rejects_events_after_parent_or_child_exhausts_budget(self) -> None:
        for final_via_child in (False, True):
            with (
                self.subTest(final_via_child=final_via_child),
                tempfile.TemporaryDirectory() as directory,
            ):
                paths = self._fixture(directory)
                self._append_post_budget_event(paths, final_via_child=final_via_child)
                with self.assertRaisesRegex(
                    analyzer.AnalysisError,
                    "exhausts the oracle budget before the final event",
                ):
                    analyzer.analyze_fragment_contexts(paths["events"])

    def test_enriched_seed_state_is_anchored_to_vocabulary_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(directory, enriched_vocabulary=True)
            report = analyzer.analyze_fragment_contexts(paths["events"])
            self.assertEqual(
                report["referenced_artifacts"]["vocabulary"]["rows_loaded"], 2
            )

            self._mutate_checkpoint(
                paths["checkpoint"],
                lambda state, metadata: state["population"]["records"][3].update(
                    {"count": 4, "total": 1.6}
                ),
            )
            with self.assertRaisesRegex(
                analyzer.AnalysisError, "final fragment 'seed-B'"
            ):
                analyzer.analyze_fragment_contexts(paths["events"])

    def test_vocabulary_mean_tolerance_exactly_matches_population_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(
                directory,
                enriched_vocabulary=True,
                enriched_seed_a_score="0.500000004",
            )
            analyzer.analyze_fragment_contexts(paths["events"])

        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(
                directory,
                enriched_vocabulary=True,
                enriched_seed_a_score="0.500000006",
            )
            with self.assertRaisesRegex(
                analyzer.AnalysisError,
                "inconsistent score/count/score_sum",
            ):
                analyzer.analyze_fragment_contexts(paths["events"])

    def test_schema_two_cannot_claim_legacy_terminal_field_omission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(directory, summary_schema=2)
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            manifest["config"].pop("population_sampling_order")
            updated_hash = sha256_config(manifest["config"])
            manifest["config_sha256"] = updated_hash
            paths["manifest"].write_text(
                json.dumps(manifest, allow_nan=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._mutate_json(
                paths["summary"],
                lambda summary: summary.__setitem__("config_sha256", updated_hash),
            )
            self._mutate_checkpoint(
                paths["checkpoint"],
                lambda state, metadata: metadata.__setitem__(
                    "config_sha256", updated_hash
                ),
            )
            events = [
                json.loads(line)
                for line in paths["events"].read_text(encoding="utf-8").splitlines()
            ]
            events[-1].pop("elapsed_seconds")
            paths["events"].write_text(
                "".join(
                    json.dumps(event, allow_nan=False, sort_keys=True) + "\n"
                    for event in events
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                analyzer.AnalysisError, "elapsed_seconds must be a finite number"
            ):
                analyzer.analyze_fragment_contexts(paths["events"])

    def _append_post_budget_event(
        self,
        paths: dict[str, Path],
        *,
        final_via_child: bool,
    ) -> None:
        events = [
            json.loads(line)
            for line in paths["events"].read_text(encoding="utf-8").splitlines()
        ]
        if final_via_child:
            events[2]["parent_smiles"] = "CC"
            events[2]["parent_oracle"] = self._oracle_row(
                "CC", 0.9, 1, charged=False
            )
            events[2]["child_oracle"] = self._oracle_row(
                "CCCC", 0.6, 3, charged=True
            )
            events[2]["population_update"] = {
                "updated": False,
                "reason": "no_fragments",
                "observed_fragments": [],
                "admitted": [],
                "displaced": [],
            }
        events.append(
            {
                "event_index": 3,
                "iteration": 3,
                "selected_fragments": ["novel-X", "seed-A"],
                "parent_smiles": "CC",
                "child_smiles": "CCC",
                "atom_count": 3,
                "proposal_attempts": 1,
                "remask_enabled": True,
                "parent_oracle": self._oracle_row("CC", 0.9, 1, charged=False),
                "child_oracle": self._oracle_row("CCC", 0.3, 2, charged=False),
                "population_update": {
                    "updated": False,
                    "reason": "duplicate_observation",
                    "observed_fragments": [],
                    "admitted": [],
                    "displaced": [],
                },
                "fragment_statistics_after": {},
                "population_cutoff_after": 0.55,
                "population_size_after": 2,
                "oracle_calls": 3,
                "top_1": 0.9,
                "top_10": 0.6,
                "top_100": 0.6,
                "elapsed_seconds": 0.4,
            }
        )
        paths["events"].write_text(
            "".join(
                json.dumps(event, allow_nan=False, sort_keys=True) + "\n"
                for event in events
            ),
            encoding="utf-8",
        )
        self._mutate_json(
            paths["summary"],
            lambda summary: summary.update(
                {"events": 4, "iterations_completed": 4, "recoverable_events": 4}
            ),
        )
        self._mutate_checkpoint(
            paths["checkpoint"],
            lambda state, metadata: state.update(
                {"event_count": 4, "next_iteration": 4}
            ),
        )

    @staticmethod
    def _mutate_event(
        events_path: Path,
        event_index: int,
        key_path: tuple[str, ...],
        replacement: Any,
    ) -> None:
        events = [
            json.loads(line)
            for line in events_path.read_text(encoding="utf-8").splitlines()
        ]
        target = events[event_index]
        for key in key_path[:-1]:
            target = target[key]
        target[key_path[-1]] = replacement
        events_path.write_text(
            "".join(
                json.dumps(event, allow_nan=False, sort_keys=True) + "\n"
                for event in events
            ),
            encoding="utf-8",
        )

    def test_source_and_destination_locks_and_no_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(directory)
            output = Path(directory) / "analysis" / "contexts.json"
            destination = analyzer.write_analysis(paths["events"], output, top_n=1)
            self.assertEqual(destination, output.resolve())
            before = output.read_bytes()
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["analyzer"]["arguments"]["output"], str(output.resolve()))
            self.assertFalse(report["analyzer"]["arguments"]["overwrite"])
            with self.assertRaises(FileExistsError):
                analyzer.write_analysis(paths["events"], output)
            self.assertEqual(output.read_bytes(), before)
            self.assertEqual(list(output.parent.glob(".*.tmp")), [])

            run_handle = (paths["run_dir"] / ".run.lock").open("rb")
            try:
                fcntl.flock(run_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(analyzer.AnalysisError, "run directory is active"):
                    analyzer.analyze_fragment_contexts(paths["events"])
            finally:
                fcntl.flock(run_handle.fileno(), fcntl.LOCK_UN)
                run_handle.close()

            second_output = Path(directory) / "analysis" / "second.json"
            destination_lock = second_output.with_name(f".{second_output.name}.lock")
            lock_handle = destination_lock.open("a+b")
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(
                    analyzer.AnalysisError, "analysis destination is active"
                ):
                    analyzer.write_analysis(paths["events"], second_output)
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                lock_handle.close()

    def test_output_cannot_alias_sources_code_helpers_model_or_vocabulary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(directory)
            protected = {
                "events": paths["events"],
                "manifest": paths["manifest"],
                "summary": paths["summary"],
                "checkpoint": paths["checkpoint"],
                "run_lock": paths["run_dir"] / ".run.lock",
                "analyzer": Path(analyzer.__file__),
                "helper": Path(analyzer.experiment_io.__file__),
                "model": paths["model"],
                "vocabulary": paths["vocabulary"],
            }
            for label, path in protected.items():
                with self.subTest(label=label):
                    before = path.read_bytes()
                    with self.assertRaisesRegex(
                        analyzer.AnalysisError, "output path aliases protected"
                    ):
                        analyzer.write_analysis(
                            paths["events"],
                            path,
                            overwrite=True,
                        )
                    self.assertEqual(path.read_bytes(), before)

            symbolic_alias = Path(directory) / "vocabulary-alias.json"
            symbolic_alias.symlink_to(paths["vocabulary"])
            with self.assertRaisesRegex(
                analyzer.AnalysisError, "aliases protected vocabulary"
            ):
                analyzer.write_analysis(
                    paths["events"], symbolic_alias, overwrite=True
                )
            hard_alias = Path(directory) / "model-hardlink.json"
            hard_alias.hardlink_to(paths["model"])
            with self.assertRaisesRegex(
                analyzer.AnalysisError, "aliases protected model"
            ):
                analyzer.write_analysis(paths["events"], hard_alias, overwrite=True)

    def test_requires_completed_schema_one_manifest_and_all_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(directory)
            self._mutate_json(
                paths["manifest"],
                lambda value: value.__setitem__("status", "running"),
            )
            with self.assertRaisesRegex(analyzer.AnalysisError, "manifest.status"):
                analyzer.analyze_fragment_contexts(paths["events"])

        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(directory)
            paths["summary"].unlink()
            with self.assertRaisesRegex(analyzer.AnalysisError, "missing required source"):
                analyzer.analyze_fragment_contexts(paths["events"])


if __name__ == "__main__":
    unittest.main()

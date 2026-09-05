"""CPU-only tests for reproducible PMO result collection."""

from __future__ import annotations

import csv
import fcntl
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.exps.pmo import collect_ablation_results as collector
from scripts.exps.pmo import launch_ablation as launcher
from scripts.exps.pmo.main.genmol.experiment_io import (
    append_event,
    build_manifest,
    load_checkpoint,
    save_checkpoint,
    sha256_file,
    summarize_indexed_scores,
    summarize_scores,
    write_manifest,
)


class AblationCollectorTests(unittest.TestCase):
    def _collect_run(self, run_dir: Path, **kwargs):
        return getattr(collector, "collect_run")(
            run_dir, trust_local_checkpoint=True, **kwargs
        )

    def _collect_results(self, experiment_root: Path, output_dir: Path, **kwargs):
        matrix_path = kwargs.pop("matrix_path", None)
        if matrix_path is None:
            candidate = Path(experiment_root).parent / "matrix.yaml"
            matrix_path = candidate if candidate.exists() else self._write_matrix(candidate.parent)
        return getattr(collector, "collect_results")(
            experiment_root,
            output_dir,
            matrix_path=matrix_path,
            trust_local_checkpoints=True,
            **kwargs,
        )

    def _completed_run(
        self,
        directory: str | Path,
        *,
        summary_schema: int = 1,
        resume_count: int = 0,
        seed: int = 0,
        experiment_id: str = "test_experiment",
        durable_events: bool | None = True,
        extra_config: dict | None = None,
    ) -> tuple[Path, Path]:
        workspace = Path(directory)
        experiment_root = workspace / experiment_id
        run_dir = experiment_root / "qed" / "delta" / f"seed_{seed}"
        run_dir.mkdir(parents=True)
        model_path = workspace / "model.ckpt"
        model_path.write_bytes(b"synthetic weights")
        vocabulary_path = workspace / "qed.csv"
        vocabulary_path.write_text(
            "frag,score,size\n[1*]CC,0.5,2\n",
            encoding="utf-8",
        )
        config = {
            "experiment_id": experiment_id,
            "scientific_status": "synthetic unit test; not a scientific result",
            "oracle": "qed",
            "variant": "delta",
            "policy_mode": "delta",
            "parent_control": True,
            "model_path": str(model_path.resolve()),
            "vocab_path": str(
                Path(collector.__file__).parent / "vocab" / "qed.csv"
            ),
            "device": "cuda:0",
            "seed": seed,
            "max_oracle_calls": 3,
            "reporting_frequency": 1,
            "checkpoint_every": 1,
            "max_iterations": 10,
            "population_size": 100,
            "warmup": 2,
            "legacy_warmup_off_by_one": True,
            "gamma": 0.0,
            "softmax_temp": 1.2,
            "randomness": 2.0,
            "guidance_scale": 2.0,
            "min_mol_size": 10,
            "max_mol_size": 30,
            "min_support": 1,
            "prior_strength": 0.0,
            "prior_mean": None,
            "prior_mean_source": None,
            "legacy_seed_count": 1,
            "delta_attribution": "novel_vs_parent",
            "statistical_duplicate_policy": "one update per unique canonical child",
            "released_duplicate_policy": "repeat cached-child decomposition, matching release",
        }
        if durable_events is not None:
            config["durable_events"] = durable_events
        config.update(extra_config or {})
        vocabulary_path = Path(config["vocab_path"])
        git = {
            "commit": "a" * 40,
            "branch": "codex/test",
            "dirty": False,
            "status": [],
            "tracked_diff_sha256": "b" * 64,
        }
        vocabulary_hash = sha256_file(vocabulary_path)
        manifest = build_manifest(
            run_id=f"{experiment_id}:qed:delta:seed{seed}",
            model_path=model_path,
            config=config,
            task="qed",
            variant="delta",
            seed=seed,
            oracle_budget=3,
            created_at="2026-09-05T00:00:00Z",
            extra={
                "git": git,
                "vocabulary": {
                    "path": str(vocabulary_path.resolve()),
                    "sha256": vocabulary_hash,
                },
                "runtime": {
                    "cuda_visible_devices": "7",
                    "gpu": {"logical_index": 0, "name": "synthetic GPU"},
                    "torch": "test",
                    "rdkit": "test",
                },
            },
        )
        manifest.update(
            {
                "status": "completed",
                "error": None,
                "oracle_calls": 3,
                "resume_count": resume_count,
                "elapsed_seconds": 1.25,
            }
        )
        write_manifest(run_dir / "manifest.json", manifest)

        outcomes = {
            1: {
                "raw_smiles": "CC",
                "canonical_smiles": "CC",
                "valid": True,
                "score": 0.2,
                "charged": True,
                "call_index": 1,
                "reason": "scored",
            },
            2: {
                "raw_smiles": "CCC",
                "canonical_smiles": "CCC",
                "valid": True,
                "score": 0.9,
                "charged": True,
                "call_index": 2,
                "reason": "scored",
            },
            3: {
                "raw_smiles": "CCCC",
                "canonical_smiles": "CCCC",
                "valid": True,
                "score": 0.7,
                "charged": True,
                "call_index": 3,
                "reason": "scored",
            },
        }
        events = [
            {
                "event_index": 0,
                "iteration": 0,
                "oracle_calls": 1,
                "parent_oracle": None,
                "child_oracle": outcomes[1],
                "top_1": 0.2,
                "top_10": 0.2,
                "top_100": 0.2,
                "elapsed_seconds": 0.2,
            },
            {
                "event_index": 1,
                "iteration": 1,
                "oracle_calls": 3,
                "parent_oracle": outcomes[2],
                "child_oracle": outcomes[3],
                "top_1": 0.9,
                "top_10": 0.6,
                "top_100": 0.6,
                "elapsed_seconds": 1.0,
            },
        ]
        for event in events:
            append_event(run_dir / "events.jsonl", event)

        all_scores = summarize_scores(
            [0.2, 0.9, 0.7],
            reporting_frequency=1,
            budget=3,
        )
        child_indexed = [(1, 0.2), (3, 0.7)]
        scores = {"all_charged_molecules": all_scores, "interpretation": "unit test"}
        if summary_schema == 1:
            scores["charged_children_only"] = summarize_scores(
                [0.2, 0.7],
                reporting_frequency=1,
                budget=3,
            )
        else:
            scores["charged_children_total_call_axis"] = summarize_indexed_scores(
                child_indexed,
                observed_oracle_calls=3,
                reporting_frequency=1,
                budget=3,
            )
            child_count = summarize_scores(
                [0.2, 0.7],
                reporting_frequency=1,
                budget=2,
            )
            child_count["axis"] = "charged_child_count"
            child_count["score_count"] = child_count.pop("oracle_calls")
            child_count["child_count_horizon"] = child_count.pop("oracle_budget")
            scores["charged_children_child_count_axis"] = child_count
        summary = {
            "schema_version": summary_schema,
            "run_id": manifest["run_id"],
            "status": "completed",
            "error": None,
            "iterations_completed": 2,
            "events": 2,
            "elapsed_seconds": 1.25,
            "scores": scores,
            "config_sha256": manifest["config_sha256"],
            "model_sha256": manifest["model"]["sha256"],
            "checkpoint_consistent": True,
            "recoverable_oracle_calls": 3,
            "recoverable_events": 2,
        }
        write_manifest(run_dir / "summary.json", summary)
        save_checkpoint(
            run_dir / "state" / "latest.pkl",
            {
                "event_count": 2,
                "next_iteration": 2,
                "elapsed_seconds": 1.1,
                "oracle": {
                    "budget": 3,
                    "buffer": {
                        "CC": [0.2, 1],
                        "CCC": [0.9, 2],
                        "CCCC": [0.7, 3],
                    },
                },
            },
            metadata={
                "config_sha256": manifest["config_sha256"],
                "model_sha256": manifest["model"]["sha256"],
                "vocabulary_sha256": vocabulary_hash,
                "git_commit": git["commit"],
                "tracked_diff_sha256": git["tracked_diff_sha256"],
            },
        )
        (run_dir / ".run.lock").touch()
        return experiment_root, run_dir

    def _write_matrix(
        self,
        directory: str | Path,
        *,
        seeds: list[int] | None = None,
        filename: str = "matrix.yaml",
    ) -> Path:
        root = Path(directory).resolve()
        matrix = {
            "schema_version": 1,
            "experiment_id": "test_experiment",
            "scientific_status": "synthetic unit test; not a scientific result",
            "model_path": str(root / "model.ckpt"),
            "tasks": [{"oracle": "qed", "gamma": 0.0}],
            "variants": ["delta"],
            "seeds": seeds or [0],
            "common": {
                "output_root": str(root),
                "max_oracle_calls": 3,
                "reporting_frequency": 1,
                "checkpoint_every": 1,
                "max_iterations": 10,
                "population_size": 100,
                "warmup": 2,
                "legacy_warmup_off_by_one": True,
                "softmax_temp": 1.2,
                "randomness": 2.0,
                "guidance_scale": 2.0,
                "legacy_seed_count": 1,
                "delta_attribution": "novel_vs_parent",
                "durable_events": True,
            },
        }
        path = root / filename
        path.write_text(json.dumps(matrix), encoding="utf-8")
        return path

    def _launch_record(
        self,
        run_dir: Path,
        *,
        time_unix: float,
        gpu_uuid: str,
        resume: bool = False,
        model_path: str | None = None,
        gpu_index: int = 7,
    ) -> dict:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        config = manifest["config"]
        if config.get("matrix_path"):
            matrix = launcher._load_matrix(Path(config["matrix_path"]))
            job = launcher._jobs(matrix)[0]
            command = launcher._command(
                Path(config["matrix_path"]), config["matrix_sha256"], matrix, job
            )
            command = [item for item in command if item != "--resume"]
        else:
            command = [
                manifest["runtime"]["executable"],
                str(Path(collector.__file__).with_name("run_ablation.py").resolve()),
                "--oracle",
                config["oracle"],
                "--variant",
                config["variant"],
                "--seed",
                str(config["seed"]),
                "--model-path",
                model_path or config["model_path"],
                "--vocab-path",
                config["vocab_path"],
                "--device",
                config["device"],
                "--experiment-id",
                config["experiment_id"],
                "--scientific-status",
                config["scientific_status"],
                "--output-root",
                str(run_dir.parents[3]),
                "--max-oracle-calls",
                str(config["max_oracle_calls"]),
                "--reporting-frequency",
                str(config["reporting_frequency"]),
                "--checkpoint-every",
                str(config["checkpoint_every"]),
                "--max-iterations",
                str(config["max_iterations"]),
                "--population-size",
                str(config["population_size"]),
                "--warmup",
                str(config["warmup"]),
                "--softmax-temp",
                str(config["softmax_temp"]),
                "--randomness",
                str(config["randomness"]),
                "--guidance-scale",
                str(config["guidance_scale"]),
                "--legacy-seed-count",
                str(config["legacy_seed_count"]),
                "--delta-attribution",
                config["delta_attribution"],
                "--gamma",
                str(config["gamma"]),
                "--legacy-warmup-off-by-one",
                "--durable-events",
            ]
        if config.get("durable_events", True) is False:
            command.remove("--durable-events")
        if resume:
            command.append("--resume")
        return {
            "event": "launch",
            "job": f"qed__delta__seed{config['seed']}",
            "command": command,
            "physical_gpu": {
                "index": gpu_index,
                "uuid": gpu_uuid,
                "memory_total_mib": 48_000,
                "memory_used_mib": 1_000,
                "utilization_percent": 0,
            },
            "compute_processes": [],
            "utilization_threshold": 10,
            "min_free_memory_mib": 20_000,
            "sharing_authorized": False,
            "sharing_actual": False,
            "wall_time_comparable": True,
            "time_unix": time_unix,
        }

    def _launch_log(self, directory: str | Path, run_dir: Path) -> Path:
        logs_dir = Path(directory) / "logs"
        logs_dir.mkdir()
        first = self._launch_record(
            run_dir,
            time_unix=1234.5,
            gpu_uuid="GPU-first",
        )
        second = self._launch_record(
            run_dir,
            time_unix=1250.0,
            gpu_uuid="GPU-second",
            resume=True,
            gpu_index=8,
        )
        log_path = logs_dir / "pmo_test.log"
        log_path.write_text(
            json.dumps(first) + "\nprocess output\n" + json.dumps(second) + "\n",
            encoding="utf-8",
        )
        return logs_dir

    def test_collects_recomputed_metrics_full_config_hashes_and_launch_history(self):
        with tempfile.TemporaryDirectory() as directory:
            matrix_path = self._write_matrix(directory)
            extra = {
                "population_sampling_order": "canonical fragment string before uniform sampling",
                "future_nested_config": {"alpha": [1, 2, 3]},
                "matrix_path": str(matrix_path.resolve()),
                "matrix_sha256": sha256_file(matrix_path),
                "vocab_path": str(
                    Path(collector.__file__).parent / "vocab" / "qed.csv"
                ),
            }
            experiment_root, run_dir = self._completed_run(
                directory,
                summary_schema=2,
                resume_count=1,
                extra_config=extra,
            )
            logs_dir = self._launch_log(directory, run_dir)
            output_dir = Path(directory) / "collection"
            results_path, collection_path = self._collect_results(
                experiment_root,
                output_dir,
                logs_dir=logs_dir,
            )

            self.assertRegex(results_path.name, r"^results\.[0-9a-f]{64}\.csv$")
            self.assertFalse((output_dir / "results.csv").exists())
            with results_path.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["child_views"], "total|count")
            self.assertEqual(row["launch_attempt_count"], "2")
            self.assertEqual(json.loads(row["launch_gpu_uuids"]), ["GPU-first", "GPU-second"])
            manifest = json.loads(collection_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["results_csv"]["sha256"], sha256_file(results_path))
            collected = manifest["runs"][0]
            source_manifest = json.loads((run_dir / "manifest.json").read_text())
            self.assertEqual(collected["config"], source_manifest["config"])
            self.assertEqual(collected["config"]["future_nested_config"], {"alpha": [1, 2, 3]})
            self.assertEqual(collected["launch"]["attempt_count"], 2)
            self.assertTrue(collected["launch"]["policy_complete"])
            self.assertTrue(collected["launch"]["matrix_provenance_complete"])
            self.assertTrue(collected["launch"]["provenance_complete"])
            self.assertTrue(collected["launch"]["gpu_migrated"])
            self.assertFalse(collected["launch"]["any_gpu_sharing"])
            self.assertTrue(collected["launch"]["wall_time_comparable"])
            self.assertEqual(
                [row["line_number"] for row in collected["launch"]["attempts"]],
                [1, 3],
            )
            self.assertEqual(
                collected["metrics"]["charged_children_total_call_axis"]["auc_top_1"],
                summarize_indexed_scores(
                    [(1, 0.2), (3, 0.7)],
                    observed_oracle_calls=3,
                    reporting_frequency=1,
                    budget=3,
                )["auc_top_1"],
            )

    def test_rejects_metric_and_event_checkpoint_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            _, run_dir = self._completed_run(directory)
            summary_path = run_dir / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["scores"]["all_charged_molecules"]["top_1"] = 0.1
            write_manifest(summary_path, summary, overwrite=True)
            with self.assertRaisesRegex(collector.CollectionError, "all-charged metrics"):
                self._collect_run(run_dir)

        with tempfile.TemporaryDirectory() as directory:
            _, run_dir = self._completed_run(directory)
            events = [
                json.loads(line)
                for line in (run_dir / "events.jsonl").read_text().splitlines()
            ]
            events[0]["child_oracle"]["score"] = 0.3
            (run_dir / "events.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(collector.CollectionError, "child_oracle checkpoint"):
                self._collect_run(run_dir)

    def test_replay_rejects_future_cache_and_malformed_null_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            _, run_dir = self._completed_run(directory)
            events_path = run_dir / "events.jsonl"
            events = [json.loads(line) for line in events_path.read_text().splitlines()]
            events[0]["parent_oracle"] = {
                "raw_smiles": "CCCC",
                "canonical_smiles": "CCCC",
                "valid": True,
                "score": 0.7,
                "charged": False,
                "call_index": 3,
                "reason": "cache_hit",
            }
            events_path.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(collector.CollectionError, "future call 3"):
                self._collect_run(run_dir)

        with tempfile.TemporaryDirectory() as directory:
            _, run_dir = self._completed_run(directory)
            events_path = run_dir / "events.jsonl"
            events = [json.loads(line) for line in events_path.read_text().splitlines()]
            events[0]["parent_oracle"] = {
                "raw_smiles": None,
                "canonical_smiles": None,
                "valid": False,
                "score": None,
                "charged": False,
                "call_index": None,
                "reason": "cache_hit",
            }
            events_path.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(collector.CollectionError, "raw_smiles"):
                self._collect_run(run_dir)

    def test_elapsed_reconciliation_and_schema_timing_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            _, run_dir = self._completed_run(directory)
            events_path = run_dir / "events.jsonl"
            events = [json.loads(line) for line in events_path.read_text().splitlines()]
            events[1]["elapsed_seconds"] = 0.1
            events_path.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(collector.CollectionError, "not chronological"):
                self._collect_run(run_dir)

        with tempfile.TemporaryDirectory() as directory:
            _, run_dir = self._completed_run(directory)
            checkpoint_path = run_dir / "state" / "latest.pkl"
            state, metadata = load_checkpoint(checkpoint_path, with_metadata=True)
            state["elapsed_seconds"] = 2.0
            save_checkpoint(checkpoint_path, state, metadata=metadata)
            with self.assertRaisesRegex(collector.CollectionError, "exceeds summary"):
                self._collect_run(run_dir)

        with tempfile.TemporaryDirectory() as directory:
            _, run_dir = self._completed_run(directory)
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["elapsed_seconds"] = 2.0
            write_manifest(manifest_path, manifest, overwrite=True)
            with self.assertRaisesRegex(collector.CollectionError, "manifest/summary"):
                self._collect_run(run_dir)

        with tempfile.TemporaryDirectory() as directory:
            _, run_dir = self._completed_run(directory, summary_schema=1)
            events_path = run_dir / "events.jsonl"
            events = [json.loads(line) for line in events_path.read_text().splitlines()]
            del events[-1]["elapsed_seconds"]
            events_path.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            collected = self._collect_run(run_dir)
            self.assertFalse(collected.row["timing_provenance_complete"])

        with tempfile.TemporaryDirectory() as directory:
            _, run_dir = self._completed_run(directory, summary_schema=2)
            events_path = run_dir / "events.jsonl"
            events = [json.loads(line) for line in events_path.read_text().splitlines()]
            del events[-1]["elapsed_seconds"]
            events_path.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(collector.CollectionError, "lacks elapsed"):
                self._collect_run(run_dir)

    def test_durable_event_config_is_validated_with_legacy_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            _, run_dir = self._completed_run(directory)
            record = self._launch_record(
                run_dir, time_unix=1.0, gpu_uuid="GPU-durable-mismatch"
            )
            record["command"].remove("--durable-events")
            logs_dir = Path(directory) / "logs"
            logs_dir.mkdir()
            (logs_dir / "mismatch.log").write_text(json.dumps(record) + "\n")
            with self.assertRaisesRegex(collector.CollectionError, "durable_events"):
                self._collect_run(
                    run_dir,
                    launch_records=collector.load_launch_records(logs_dir),
                )

        with tempfile.TemporaryDirectory() as directory:
            _, run_dir = self._completed_run(directory, durable_events=None)
            record = self._launch_record(
                run_dir, time_unix=1.0, gpu_uuid="GPU-legacy-durable"
            )
            logs_dir = Path(directory) / "logs"
            logs_dir.mkdir()
            (logs_dir / "legacy.log").write_text(json.dumps(record) + "\n")
            collected = self._collect_run(
                run_dir, launch_records=collector.load_launch_records(logs_dir)
            )
            self.assertFalse(collected.row["durable_events_provenance_complete"])
            self.assertFalse(collected.row["launch_provenance_complete"])

        with tempfile.TemporaryDirectory() as directory:
            _, run_dir = self._completed_run(directory, durable_events=False)
            collected = self._collect_run(run_dir)
            self.assertTrue(collected.row["durable_events_provenance_complete"])
            self.assertIs(collected.row["durable_events"], False)

    def test_launch_reconstruction_preserves_every_bayesian_strength(self):
        expected_strengths = {
            "shrink1": 1.0,
            "shrink3": 3.0,
            "shrink10": 10.0,
            "shrink30": 30.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix_path = self._write_matrix(root)
            matrix = launcher._load_matrix(matrix_path)
            matrix["variants"] = list(expected_strengths)
            matrix["tasks"][0].update(
                prior_mean=0.42,
                prior_mean_source="frozen unit-test prior",
            )
            matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
            matrix_hash = sha256_file(matrix_path)

            for job in launcher._jobs(matrix):
                with self.subTest(variant=job.variant):
                    command = launcher._command(
                        matrix_path, matrix_hash, matrix, job
                    )
                    parsed = collector._parse_launch_command(command)
                    resolved = collector._resolved_launch_config(parsed)
                    self.assertEqual(resolved["policy_mode"], "bayes")
                    self.assertEqual(
                        resolved["prior_strength"], expected_strengths[job.variant]
                    )
                    self.assertEqual(resolved["prior_mean"], 0.42)
                    self.assertEqual(
                        resolved["prior_mean_source"], "frozen unit-test prior"
                    )

            missing_prior_command = launcher._command(
                matrix_path,
                matrix_hash,
                matrix,
                launcher._jobs(matrix)[0],
            )
            for flag in ("--prior-mean", "--prior-mean-source"):
                index = missing_prior_command.index(flag)
                del missing_prior_command[index : index + 2]
            with self.assertRaisesRegex(collector.CollectionError, "requires"):
                collector._resolved_launch_config(
                    collector._parse_launch_command(missing_prior_command)
                )

    def test_resolves_and_validates_corrected_delta_protocol_metadata(self):
        parsed = {
            "--oracle": "qed",
            "--variant": "running_mean_delta_control",
            "--model-path": "/tmp/model.ckpt",
            "--experiment-id": "delta_control_test",
            "--scientific-status": "unit test",
            "--output-root": "/tmp/output",
            "--delta-attribution": "novel_vs_parent",
        }
        config = collector._resolved_launch_config(parsed)
        self.assertEqual(config["policy_mode"], "delta_control")
        self.assertTrue(config["parent_control"])
        self.assertEqual(
            config["seed_initialization_policy"],
            "neutral_zero_with_seed_score_tiebreak",
        )
        self.assertEqual(config["credit_value_policy"], "absolute_child_score")
        self.assertEqual(config["warmup_update_policy"], "frozen")
        self.assertEqual(
            config["observation_identity"],
            "unique_canonical_parent_child_transition",
        )
        self.assertEqual(
            config["credit_fragment_policy"],
            "deterministic_cut_all_child_minus_parent",
        )
        delta_parsed = dict(parsed, **{"--variant": "delta"})
        delta_config = collector._resolved_launch_config(delta_parsed)
        self.assertEqual(
            delta_config["credit_value_policy"],
            "child_score_minus_parent_score",
        )
        config["min_mol_size"] = 3
        config["max_mol_size"] = 4

        parent_fragments = sorted(collector.cut_all("CCO"))
        child_fragments = sorted(collector.cut_all("CCOC"))
        credited = sorted(set(child_fragments) - set(parent_fragments))
        attribution = {
            "applicable": True,
            "reason": "deterministic_mapping",
            "attribution_mode": "novel_vs_parent",
            "parent_all_fragments": parent_fragments,
            "child_all_fragments": child_fragments,
            "credited_fragments": credited,
            "mapping_counts": {
                "parent_all": len(parent_fragments),
                "child_all": len(child_fragments),
                "shared": len(set(parent_fragments) & set(child_fragments)),
                "credited": len(credited),
            },
            "mapping_covered": bool(credited),
            "mapping_coverage": len(credited) / len(child_fragments),
        }
        event = {
            "remask_enabled": True,
            "parent_atom_count": 3,
            "child_atom_count": 4,
            "atom_count": 4,
            "parent_smiles": "CCO",
            "child_smiles": "CCOC",
            "parent_oracle": {"canonical_smiles": "CCO"},
            "child_oracle": {"canonical_smiles": "CCOC"},
            "attribution": attribution,
        }
        collector._validate_corrected_delta_event(
            event,
            config=config,
            event_index=0,
            update_reason="updated",
        )
        seen_transitions = set()
        collector._validate_corrected_delta_event(
            event,
            config=config,
            event_index=0,
            update_reason="updated",
            seen_transitions=seen_transitions,
        )
        collector._validate_corrected_delta_event(
            event,
            config=config,
            event_index=1,
            update_reason="duplicate_observation",
            seen_transitions=seen_transitions,
        )
        with self.assertRaisesRegex(collector.CollectionError, "transition update reason"):
            collector._validate_corrected_delta_event(
                event,
                config=config,
                event_index=2,
                update_reason="updated",
                seen_transitions=seen_transitions,
            )
        attribution["credited_fragments"] = []
        with self.assertRaisesRegex(collector.CollectionError, "credited fragments"):
            collector._validate_corrected_delta_event(
                event,
                config=config,
                event_index=0,
                update_reason="updated",
            )

    def test_v2_rejects_missing_strict_estimator_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            _, run_dir = self._completed_run(
                directory,
                experiment_id="fragment_vocab_qed_50k_delta_1k_v2",
                extra_config={
                    "warmup_update_policy": "frozen",
                    "observation_identity": (
                        "unique_canonical_parent_child_transition"
                    ),
                    "parent_domain_policy": (
                        "parent_and_child_within_configured_atom_bounds"
                    ),
                    "credit_fragment_policy": (
                        "deterministic_cut_all_child_minus_parent"
                    ),
                    "credit_value_policy": "child_score_minus_parent_score",
                    "statistical_duplicate_policy": (
                        "one update per unique canonical parent-child transition"
                    ),
                },
            )
            with self.assertRaisesRegex(
                collector.CollectionError,
                "missing strict matched-estimator metadata",
            ):
                self._collect_run(run_dir)

    def test_matrix_plan_resolves_only_the_declared_output_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean_clone = root / "clean-clone"
            clean_clone.mkdir()
            physical_output = root / "physical-output"
            (physical_output / "pmo_ablation" / "test_experiment").mkdir(
                parents=True
            )
            (clean_clone / "output").symlink_to(
                physical_output, target_is_directory=True
            )
            matrix_path = self._write_matrix(clean_clone)
            matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
            matrix["common"]["output_root"] = "output/pmo_ablation"
            matrix_path.write_text(json.dumps(matrix), encoding="utf-8")

            experiment_root = physical_output / "pmo_ablation" / "test_experiment"
            with mock.patch.object(launcher, "REPOSITORY_ROOT", clean_clone):
                plan = collector._load_matrix_plan(matrix_path, experiment_root)

            self.assertEqual(set(plan.jobs), {("qed", "delta", 0)})
            with self.assertRaisesRegex(collector.CollectionError, "symlinks"):
                collector.collect_results(
                    clean_clone
                    / "output"
                    / "pmo_ablation"
                    / "test_experiment",
                    root / "archive",
                    matrix_path=matrix_path,
                    trust_local_checkpoints=True,
                )

    def test_schema_gates_explicit_pair_and_nonempty_null_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            _, run_dir = self._completed_run(directory, summary_schema=2)
            summary_path = run_dir / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            del summary["scores"]["charged_children_child_count_axis"]
            write_manifest(summary_path, summary, overwrite=True)
            with self.assertRaisesRegex(collector.CollectionError, "schema-2 child metric keys"):
                self._collect_run(run_dir)

        with tempfile.TemporaryDirectory() as directory:
            _, run_dir = self._completed_run(directory, summary_schema=2)
            summary_path = run_dir / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["scores"]["charged_children_child_count_axis"]["top_1"] = None
            write_manifest(summary_path, summary, overwrite=True)
            with self.assertRaisesRegex(collector.CollectionError, "child-count metrics"):
                self._collect_run(run_dir)

    def test_rejects_stale_launch_but_accepts_retry_history(self):
        with tempfile.TemporaryDirectory() as directory:
            _, run_dir = self._completed_run(directory, resume_count=1)
            logs_dir = self._launch_log(directory, run_dir)
            records = collector.load_launch_records(logs_dir)
            collected = self._collect_run(run_dir, launch_records=records)
            self.assertEqual(len(collected.launch_attempts), 2)
            self.assertTrue(collected.row["launch_history_complete"])

        with tempfile.TemporaryDirectory() as directory:
            _, run_dir = self._completed_run(directory)
            logs_dir = Path(directory) / "logs"
            logs_dir.mkdir()
            record = self._launch_record(
                run_dir,
                time_unix=1.0,
                gpu_uuid="GPU-stale",
                model_path=str(Path(directory) / "stale.ckpt"),
            )
            (logs_dir / "stale.log").write_text(json.dumps(record) + "\n")
            with self.assertRaisesRegex(collector.CollectionError, "model_path"):
                self._collect_run(
                    run_dir,
                    launch_records=collector.load_launch_records(logs_dir),
                )

        with tempfile.TemporaryDirectory() as directory:
            matrix_path = self._write_matrix(directory)
            _, run_dir = self._completed_run(
                directory,
                extra_config={
                    "matrix_path": str(matrix_path.resolve()),
                    "matrix_sha256": sha256_file(matrix_path),
                    "vocab_path": str(
                        Path(collector.__file__).parent / "vocab" / "qed.csv"
                    ),
                },
            )
            record = self._launch_record(
                run_dir, time_unix=1.0, gpu_uuid="GPU-matrix"
            )
            matrix_path.write_text("version: changed\n", encoding="utf-8")
            logs_dir = Path(directory) / "logs"
            logs_dir.mkdir()
            (logs_dir / "matrix.log").write_text(json.dumps(record) + "\n")
            with self.assertRaisesRegex(collector.CollectionError, "matrix SHA256"):
                self._collect_run(
                    run_dir,
                    launch_records=collector.load_launch_records(logs_dir),
                )

        with tempfile.TemporaryDirectory() as directory:
            _, run_dir = self._completed_run(directory)
            logs_dir = Path(directory) / "logs"
            logs_dir.mkdir()
            record = self._launch_record(
                run_dir, time_unix=1.0, gpu_uuid="GPU-unknown-flag"
            )
            record["command"].extend(["--future-unvalidated-flag", "value"])
            (logs_dir / "unknown.log").write_text(json.dumps(record) + "\n")
            with self.assertRaisesRegex(collector.CollectionError, "unknown argument"):
                self._collect_run(
                    run_dir,
                    launch_records=collector.load_launch_records(logs_dir),
                )

    def test_launch_gpu_policy_validation_and_legacy_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            _, run_dir = self._completed_run(directory)
            record = self._launch_record(
                run_dir, time_unix=1.0, gpu_uuid="GPU-legacy"
            )
            for field in collector.GPU_POLICY_FIELDS:
                del record[field]
            record["shared_gpu_authorized"] = True
            record["compute_processes"] = [
                {
                    "gpu_uuid": "GPU-legacy",
                    "pid": 42,
                    "process_name": "older-worker",
                    "used_memory_mib": 100,
                }
            ]
            logs_dir = Path(directory) / "logs"
            logs_dir.mkdir()
            (logs_dir / "legacy.log").write_text(json.dumps(record) + "\n")
            collected = self._collect_run(
                run_dir, launch_records=collector.load_launch_records(logs_dir)
            )
            self.assertFalse(collected.row["launch_policy_complete"])
            self.assertFalse(collected.row["launch_provenance_complete"])
            self.assertIsNone(collected.row["launch_wall_time_comparable"])

        cases = {
            "utilization": lambda row: row.update(utilization_threshold=0),
            "free memory": lambda row: row.update(min_free_memory_mib=47_001),
            "sharing_actual": lambda row: row.update(sharing_actual=True),
            "without authorization": lambda row: row.update(
                compute_processes=[
                    {
                        "gpu_uuid": "GPU-policy",
                        "pid": 42,
                        "process_name": "other-worker",
                        "used_memory_mib": 100,
                    }
                ],
                sharing_actual=True,
                wall_time_comparable=False,
            ),
            "wall_time_comparable": lambda row: row.update(wall_time_comparable=False),
        }
        for message, mutate in cases.items():
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                _, run_dir = self._completed_run(directory)
                record = self._launch_record(
                    run_dir, time_unix=1.0, gpu_uuid="GPU-policy"
                )
                mutate(record)
                logs_dir = Path(directory) / "logs"
                logs_dir.mkdir()
                (logs_dir / "policy.log").write_text(json.dumps(record) + "\n")
                with self.assertRaisesRegex(collector.CollectionError, message):
                    self._collect_run(
                        run_dir,
                        launch_records=collector.load_launch_records(logs_dir),
                    )

        with tempfile.TemporaryDirectory() as directory:
            _, run_dir = self._completed_run(directory)
            record = self._launch_record(
                run_dir, time_unix=1.0, gpu_uuid="GPU-wrong-index", gpu_index=6
            )
            logs_dir = Path(directory) / "logs"
            logs_dir.mkdir()
            (logs_dir / "wrong-index.log").write_text(json.dumps(record) + "\n")
            with self.assertRaisesRegex(collector.CollectionError, "CUDA_VISIBLE_DEVICES"):
                self._collect_run(
                    run_dir,
                    launch_records=collector.load_launch_records(logs_dir),
                )

    def test_partial_history_and_pre_manifest_retries_do_not_fake_initial_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            _, run_dir = self._completed_run(directory, resume_count=1)
            record = self._launch_record(
                run_dir,
                time_unix=2.0,
                gpu_uuid="GPU-resume-only",
                gpu_index=8,
                resume=True,
            )
            logs_dir = Path(directory) / "logs"
            logs_dir.mkdir()
            (logs_dir / "partial.log").write_text(json.dumps(record) + "\n")
            collected = self._collect_run(
                run_dir, launch_records=collector.load_launch_records(logs_dir)
            )
            self.assertFalse(collected.row["launch_history_complete"])
            self.assertIsNone(collected.row["launch_gpu_migrated"])

        with tempfile.TemporaryDirectory() as directory:
            _, run_dir = self._completed_run(directory)
            failed = self._launch_record(
                run_dir, time_unix=1.0, gpu_uuid="GPU-failed", gpu_index=6
            )
            initial = self._launch_record(
                run_dir, time_unix=2.0, gpu_uuid="GPU-initial", gpu_index=7
            )
            logs_dir = Path(directory) / "logs"
            logs_dir.mkdir()
            (logs_dir / "retry.log").write_text(
                json.dumps(failed) + "\n" + json.dumps(initial) + "\n"
            )
            collected = self._collect_run(
                run_dir, launch_records=collector.load_launch_records(logs_dir)
            )
            self.assertTrue(collected.row["launch_history_complete"])
            self.assertFalse(collected.row["launch_gpu_migrated"])

    def test_publication_lock_no_clobber_and_atomic_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            payload = b"a,b\n1,2\n"

            def concurrent_publish(_source, target):
                Path(target).write_bytes(payload)
                raise FileExistsError(target)

            with mock.patch.object(
                collector.os, "link", side_effect=concurrent_publish
            ):
                path, _, created = collector._publish_immutable_csv(
                    destination, payload
                )
            self.assertFalse(created)
            self.assertEqual(path.read_bytes(), payload)

        with tempfile.TemporaryDirectory() as directory:
            experiment_root, _ = self._completed_run(directory)
            output_dir = Path(directory) / "collection"
            with mock.patch.object(
                collector, "write_manifest", side_effect=OSError("injected")
            ):
                with self.assertRaises(OSError):
                    self._collect_results(experiment_root, output_dir)
            self.assertFalse((output_dir / "collection_manifest.json").exists())
            self.assertEqual(list(output_dir.glob("results.*.csv")), [])

        with tempfile.TemporaryDirectory() as directory:
            experiment_root, _ = self._completed_run(directory)
            output_dir = Path(directory) / "collection"
            with mock.patch.object(
                collector, "_manifest_run", side_effect=RuntimeError("injected")
            ):
                with self.assertRaises(RuntimeError):
                    self._collect_results(experiment_root, output_dir)
            self.assertFalse((output_dir / "collection_manifest.json").exists())
            self.assertEqual(list(output_dir.glob("results.*.csv")), [])

        with tempfile.TemporaryDirectory() as directory:
            experiment_root, _ = self._completed_run(directory)
            output_dir = Path(directory) / "collection"
            results_path, manifest_path = self._collect_results(experiment_root, output_dir)
            old_results = results_path.read_bytes()
            old_manifest = manifest_path.read_bytes()
            with self.assertRaises(FileExistsError):
                self._collect_results(experiment_root, output_dir)
            self.assertEqual(results_path.read_bytes(), old_results)
            self.assertEqual(manifest_path.read_bytes(), old_manifest)
            with mock.patch.object(collector, "write_manifest", side_effect=OSError("injected")):
                with self.assertRaises(OSError):
                    self._collect_results(experiment_root, output_dir, overwrite=True)
            self.assertEqual(results_path.read_bytes(), old_results)
            self.assertEqual(manifest_path.read_bytes(), old_manifest)

        with tempfile.TemporaryDirectory() as directory:
            experiment_root, run_dir = self._completed_run(directory)
            output_dir = Path(directory) / "collection"
            output_dir.mkdir()
            with collector.FileLock(output_dir / ".collection.lock", fcntl.LOCK_EX):
                with self.assertRaisesRegex(collector.CollectionError, "another collector"):
                    self._collect_results(experiment_root, output_dir)
            self.assertFalse((output_dir / "collection_manifest.json").exists())
            with collector.FileLock(run_dir / ".run.lock", fcntl.LOCK_EX):
                with self.assertRaisesRegex(collector.CollectionError, "run directory is active"):
                    self._collect_results(experiment_root, output_dir)
            self.assertFalse((output_dir / "collection_manifest.json").exists())

    def test_existing_legacy_result_and_incomplete_handling(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment_root, _ = self._completed_run(directory)
            output_dir = Path(directory) / "collection"
            output_dir.mkdir()
            legacy = output_dir / "results.csv"
            legacy.write_bytes(b"previous result\n")
            with self.assertRaises(FileExistsError):
                self._collect_results(experiment_root, output_dir)
            self.assertEqual(legacy.read_bytes(), b"previous result\n")

        with tempfile.TemporaryDirectory() as directory:
            experiment_root, _ = self._completed_run(directory)
            incomplete = experiment_root / "qed" / "delta" / "seed_1"
            incomplete.mkdir()
            (incomplete / ".run.lock").touch()
            (incomplete / "manifest.json").write_text(json.dumps({"status": "running"}))
            (incomplete / "summary.json").write_text(json.dumps({"status": "running"}))
            active = experiment_root / "qed" / "delta" / "seed_2"
            active.mkdir()
            (active / "manifest.json").write_text(json.dumps({"status": "running"}))
            (active / "summary.json").write_text(json.dumps({"status": "running"}))
            with collector.FileLock(active / ".run.lock", fcntl.LOCK_EX):
                _, manifest_path = self._collect_results(
                    experiment_root,
                    Path(directory) / "collection",
                    matrix_path=self._write_matrix(directory, seeds=[0, 1, 2]),
                    skip_incomplete=True,
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["skipped"],
                [
                    {"reason": "incomplete", "run_dir": "qed/delta/seed_1"},
                    {"reason": "active", "run_dir": "qed/delta/seed_2"},
                ],
            )

    def test_matrix_enumeration_prevents_partial_or_unexpected_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment_root, _ = self._completed_run(directory, seed=0)
            matrix_path = self._write_matrix(directory, seeds=[0, 1])
            with self.assertRaisesRegex(collector.CollectionError, "missing matrix jobs"):
                self._collect_results(
                    experiment_root,
                    Path(directory) / "default-fails",
                    matrix_path=matrix_path,
                )

            _, pointer = self._collect_results(
                experiment_root,
                Path(directory) / "partial",
                matrix_path=matrix_path,
                skip_incomplete=True,
            )
            manifest = json.loads(pointer.read_text(encoding="utf-8"))
            self.assertFalse(manifest["collection_complete"])
            self.assertEqual(manifest["matrix"]["expected_job_count"], 2)
            self.assertEqual(
                manifest["missing_jobs"],
                [{"oracle": "qed", "variant": "delta", "seed": 1, "reason": "missing"}],
            )

            self._completed_run(directory, seed=1)
            _, complete_pointer = self._collect_results(
                experiment_root,
                Path(directory) / "complete",
                matrix_path=matrix_path,
            )
            complete = json.loads(complete_pointer.read_text(encoding="utf-8"))
            self.assertTrue(complete["collection_complete"])
            self.assertEqual(complete["run_count"], 2)
            self.assertEqual(complete["missing_jobs"], [])

            self._completed_run(directory, seed=2)
            with self.assertRaisesRegex(collector.CollectionError, "absent from matrix"):
                self._collect_results(
                    experiment_root,
                    Path(directory) / "unexpected",
                    matrix_path=matrix_path,
                    skip_incomplete=True,
                )

    def test_collection_rejects_mixed_recorded_matrices(self):
        with tempfile.TemporaryDirectory() as directory:
            matrix_a = self._write_matrix(
                directory, seeds=[0, 1], filename="matrix-a.yaml"
            )
            matrix_b = self._write_matrix(
                directory, seeds=[0, 1], filename="matrix-b.yaml"
            )
            experiment_root, _ = self._completed_run(
                directory,
                seed=0,
                extra_config={
                    "matrix_path": str(matrix_a.resolve()),
                    "matrix_sha256": sha256_file(matrix_a),
                },
            )
            self._completed_run(
                directory,
                seed=1,
                extra_config={
                    "matrix_path": str(matrix_b.resolve()),
                    "matrix_sha256": sha256_file(matrix_b),
                },
            )
            with self.assertRaisesRegex(collector.CollectionError, "matrix path"):
                self._collect_results(
                    experiment_root,
                    Path(directory) / "mixed",
                    matrix_path=matrix_a,
                )

    def test_checkpoint_trust_and_symlink_containment_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment_root, run_dir = self._completed_run(directory)
            matrix_path = self._write_matrix(directory)
            with self.assertRaisesRegex(collector.CollectionError, "explicit trust"):
                collector.collect_run(run_dir)
            linked_root = Path(directory) / "linked-experiment"
            linked_root.symlink_to(experiment_root, target_is_directory=True)
            with self.assertRaisesRegex(collector.CollectionError, "symlink"):
                collector.collect_results(
                    linked_root,
                    Path(directory) / "linked-output",
                    matrix_path=matrix_path,
                    trust_local_checkpoints=True,
                )

        with tempfile.TemporaryDirectory() as directory:
            _, run_dir = self._completed_run(directory)
            checkpoint = run_dir / "state" / "latest.pkl"
            escaped = Path(directory) / "escaped.pkl"
            escaped.write_bytes(checkpoint.read_bytes())
            checkpoint.unlink()
            checkpoint.symlink_to(escaped)
            with mock.patch.object(collector, "load_checkpoint") as loader:
                with self.assertRaisesRegex(collector.CollectionError, "symlink"):
                    self._collect_run(run_dir)
                loader.assert_not_called()

    def test_rejects_event_count_and_path_identity_mismatches(self):
        with tempfile.TemporaryDirectory() as directory:
            _, run_dir = self._completed_run(directory)
            append_event(
                run_dir / "events.jsonl",
                {
                    "event_index": 2,
                    "iteration": 2,
                    "oracle_calls": 3,
                    "parent_oracle": None,
                    "child_oracle": {
                        "raw_smiles": "CC",
                        "canonical_smiles": "CC",
                        "valid": True,
                        "score": 0.2,
                        "charged": False,
                        "call_index": 1,
                        "reason": "cache_hit",
                    },
                    "population_update": {
                        "updated": False,
                        "reason": "duplicate_observation",
                    },
                    "elapsed_seconds": 1.05,
                },
            )
            with self.assertRaisesRegex(collector.CollectionError, "event log count"):
                self._collect_run(run_dir)

        with tempfile.TemporaryDirectory() as directory:
            _, run_dir = self._completed_run(directory)
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["task"] = "jnk3"
            write_manifest(manifest_path, manifest, overwrite=True)
            with self.assertRaisesRegex(collector.CollectionError, "manifest task"):
                self._collect_run(run_dir)


if __name__ == "__main__":
    unittest.main()

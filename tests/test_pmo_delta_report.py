"""CPU-only tests for the fixed QED delta-y report."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from pypdf import PdfReader

from scripts.exps.pmo import report_delta_ablation as report
from scripts.exps.pmo.main.genmol.experiment_io import sha256_config, sha256_file


CSV_FIELDS = (
    "experiment_id",
    "run_id",
    "oracle",
    "variant",
    "seed",
    "summary_schema_version",
    "status",
    "oracle_budget",
    "all_oracle_calls",
    "charged_child_count",
    "elapsed_seconds",
    "reporting_frequency",
    "config_sha256",
    "model_path",
    "model_sha256",
    "vocabulary_sha256",
    "source_summary_sha256",
    "source_events_sha256",
    "all_auc_top_1",
    "all_auc_top_10",
    "all_auc_top_100",
    "all_top_1",
    "all_top_10",
    "all_top_100",
    "child_total_auc_top_1",
    "child_total_auc_top_10",
    "child_total_auc_top_100",
    "child_total_top_1",
    "child_total_top_10",
    "child_total_top_100",
)


class DeltaReportTests(unittest.TestCase):
    def _oracle_outcome(self, score: float, call: int, smiles: str) -> dict[str, object]:
        return {
            "raw_smiles": smiles,
            "canonical_smiles": smiles,
            "valid": True,
            "score": score,
            "charged": True,
            "call_index": call,
            "reason": "scored",
        }

    def _empty_attribution(self, reason: str) -> dict[str, object]:
        return {
            "applicable": False,
            "reason": reason,
            "attribution_mode": None,
            "parent_all_fragments": [],
            "child_all_fragments": [],
            "credited_fragments": [],
            "mapping_counts": {
                "parent_all": 0,
                "child_all": 0,
                "shared": 0,
                "credited": 0,
            },
            "mapping_covered": False,
            "mapping_coverage": 0.0,
        }

    def _attribution(self, index: int) -> dict[str, object]:
        return {
            "applicable": True,
            "reason": "deterministic_mapping",
            "attribution_mode": "novel_vs_parent",
            "parent_all_fragments": [f"parent_fragment_{index}"],
            "child_all_fragments": [f"child_fragment_{index}"],
            "credited_fragments": [f"child_fragment_{index}"],
            "mapping_counts": {
                "parent_all": 1,
                "child_all": 1,
                "shared": 0,
                "credited": 1,
            },
            "mapping_covered": True,
            "mapping_coverage": 1.0,
        }

    def _events(self, variant: str) -> list[dict[str, object]]:
        target = variant in report.TARGET_VARIANTS
        events: list[dict[str, object]] = []
        calls = 0
        delta_totals: dict[str, float] = {}
        delta_counts: dict[str, int] = {}
        index = 0
        while calls < 1000:
            remask = index > 100
            parent = None
            child = None
            attribution = None
            update: dict[str, object] = {"updated": False, "reason": "unused"}
            statistics: dict[str, object] = {}
            if not target:
                calls += 1
                child = self._oracle_outcome(0.6, calls, f"child_{variant}_{index}")
                update = {"updated": False, "reason": "score_below_cutoff"}
            elif not remask:
                calls += 1
                child = self._oracle_outcome(0.5, calls, f"warmup_child_{variant}_{index}")
                update = {"updated": False, "reason": "frozen_warmup"}
                attribution = self._empty_attribution("warmup_frozen")
            else:
                calls += 1
                parent = self._oracle_outcome(0.4, calls, f"parent_{variant}_{index}")
                if calls == 1000:
                    update = {"updated": False, "reason": "budget_after_parent"}
                    attribution = self._empty_attribution("budget_after_parent")
                else:
                    calls += 1
                    child = self._oracle_outcome(0.5, calls, f"child_{variant}_{index}")
                    attribution = self._attribution(index)
                    fragment = f"child_fragment_{index}"
                    update = {
                        "updated": True,
                        "reason": "updated",
                        "observed_fragments": [fragment],
                        "admitted": [],
                        "displaced": [],
                    }
                    if variant == "delta":
                        delta_totals[fragment] = delta_totals.get(fragment, 0.0) + 0.1
                        delta_counts[fragment] = delta_counts.get(fragment, 0) + 1
                        statistics[fragment] = {
                            "fragment": fragment,
                            "total": delta_totals[fragment],
                            "count": delta_counts[fragment],
                            "seed_score": None,
                            "seed_order": None,
                            "first_seen": index - 100,
                            "last_seen": index - 100,
                        }
            events.append(
                {
                    "event_index": index,
                    "iteration": index,
                    "remask_enabled": remask,
                    "parent_oracle": parent,
                    "child_oracle": child,
                    "population_update": update,
                    "fragment_statistics_after": statistics,
                    "attribution": attribution,
                    "oracle_calls": calls,
                }
            )
            index += 1
        return events

    def _config(
        self,
        root: Path,
        matrix_path: Path,
        matrix_sha: str,
        variant: str,
        seed: int,
    ) -> dict[str, object]:
        mode, parent_control = report.EXPECTED_POLICY[variant]
        target = variant in report.TARGET_VARIANTS
        return {
            "experiment_id": report.EXPERIMENT_ID,
            "scientific_status": report.SCIENTIFIC_STATUS,
            "matrix_path": str(matrix_path.resolve()),
            "matrix_sha256": matrix_sha,
            "oracle": "qed",
            "variant": variant,
            "policy_mode": mode,
            "parent_control": parent_control,
            "model_path": str(root / "outputs/paper_v1/checkpoints/50000.ckpt"),
            "vocab_path": str(root / "qed.csv"),
            "device": "cuda:0",
            "seed": seed,
            "max_oracle_calls": 1000,
            "reporting_frequency": 100,
            "checkpoint_every": 100,
            "max_iterations": 10000,
            "population_size": 100,
            "warmup": 100,
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
            "population_sampling_order": "canonical fragment string before uniform sampling",
            "statistical_duplicate_policy": (
                "one update per unique canonical parent-child transition"
                if target
                else "one update per unique canonical child"
            ),
            "released_duplicate_policy": "repeat cached-child decomposition, matching release",
            "durable_events": True,
            "warmup_update_policy": "frozen" if target else "standard",
            "observation_identity": (
                "unique_canonical_parent_child_transition"
                if target
                else "canonical_child_occurrence"
                if variant == "released"
                else "unique_canonical_child"
            ),
            "parent_domain_policy": (
                "parent_and_child_within_configured_atom_bounds"
                if target
                else "child_within_configured_atom_bounds"
            ),
            "credit_fragment_policy": (
                "deterministic_cut_all_child_minus_parent"
                if target
                else "sampled_three_cut_child"
            ),
        }

    def _bundle(self, root: Path) -> tuple[Path, dict[str, object]]:
        experiment_root = root / "raw" / report.EXPERIMENT_ID
        archive = root / "archive"
        experiment_root.mkdir(parents=True)
        archive.mkdir()
        matrix_source = (
            Path(report.__file__).resolve().parents[3]
            / "experiments/fragment_vocabulary/configs/qed_50k_delta_1k_v1.yaml"
        )
        matrix_path = root / matrix_source.name
        matrix_path.write_bytes(matrix_source.read_bytes())
        matrix_sha = sha256_file(matrix_path)
        runs: list[dict[str, object]] = []
        rows: list[dict[str, object]] = []
        expected_jobs: list[dict[str, object]] = []

        for variant_index, variant in enumerate(report.EXPECTED_VARIANTS):
            for seed in report.EXPECTED_SEEDS:
                run_id = f"{report.EXPERIMENT_ID}:qed:{variant}:seed{seed}"
                expected_jobs.append({"oracle": "qed", "variant": variant, "seed": seed})
                run_dir = experiment_root / "qed" / variant / f"seed_{seed}"
                (run_dir / "state").mkdir(parents=True)
                events = self._events(variant)
                events_path = run_dir / "events.jsonl"
                events_path.write_text(
                    "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
                    encoding="utf-8",
                )
                child_count = sum(
                    bool(event["child_oracle"] and event["child_oracle"]["charged"])
                    for event in events
                )
                offset = 0.005 * variant_index + 0.001 * seed
                all_values = {
                    "auc_top_1": 0.90 + offset,
                    "auc_top_10": 0.88 + offset,
                    "auc_top_100": 0.82 + offset,
                    "top_1": 0.96 + offset,
                    "top_10": 0.94 + offset,
                    "top_100": 0.89 + offset,
                }
                child_values = {key: value - 0.002 for key, value in all_values.items()}
                metrics = {
                    "all_charged_molecules": {
                        "source_key": "all_charged_molecules",
                        "semantics": "all calls",
                        "axis": None,
                        "score_count": 1000,
                        "axis_budget": 1000,
                        "reporting_frequency": 100,
                        **all_values,
                    },
                    "charged_children_total_call_axis": {
                        "source_key": "charged_children_total_call_axis",
                        "semantics": "children on total axis",
                        "axis": "total_unique_oracle_calls",
                        "score_count": child_count,
                        "axis_budget": 1000,
                        "reporting_frequency": 100,
                        **child_values,
                    },
                }
                summary = {
                    "schema_version": 2,
                    "run_id": run_id,
                    "status": "completed",
                    "checkpoint_consistent": True,
                    "scores": {
                        name: {
                            "oracle_calls": 1000,
                            "oracle_budget": 1000,
                            "reporting_frequency": 100,
                            **{key: value for key, value in group.items() if key in report.SCALAR_NAMES},
                        }
                        for name, group in metrics.items()
                    },
                }
                summary_path = run_dir / "summary.json"
                summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
                manifest_path = run_dir / "manifest.json"
                manifest_path.write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
                checkpoint_path = run_dir / "state/latest.pkl"
                checkpoint_path.write_bytes(b"synthetic checkpoint")
                config = self._config(root, matrix_path, matrix_sha, variant, seed)
                config_sha = sha256_config(config)
                sources = {
                    "manifest": {"path": str(manifest_path.relative_to(experiment_root)), "sha256": sha256_file(manifest_path)},
                    "summary": {"path": str(summary_path.relative_to(experiment_root)), "sha256": sha256_file(summary_path)},
                    "events": {"path": str(events_path.relative_to(experiment_root)), "sha256": sha256_file(events_path)},
                    "checkpoint": {"path": str(checkpoint_path.relative_to(experiment_root)), "sha256": sha256_file(checkpoint_path)},
                }
                provenance = {
                    "config_sha256": config_sha,
                    "model_path": config["model_path"],
                    "model_sha256": report.EXPECTED_MODEL_SHA256,
                    "vocabulary_path": config["vocab_path"],
                    "vocabulary_sha256": report.EXPECTED_VOCABULARY_SHA256,
                    "git_commit": "1" * 40,
                    "git_dirty": False,
                    "tracked_diff_sha256": hashlib.sha256(b"").hexdigest(),
                    "resume_count": 0,
                    "durable_events_provenance_complete": True,
                    "timing_provenance_complete": True,
                }
                gpu_index = seed + 3
                launch = {
                    "history_complete": True,
                    "policy_complete": True,
                    "matrix_provenance_complete": True,
                    "provenance_complete": True,
                    "durable_events_config_complete": True,
                    "gpu_migrated": False,
                    "any_gpu_sharing": True,
                    "attempts": [
                        {
                            "record": {
                                "utilization_threshold": 10,
                                "sharing_actual": True,
                                "sharing_authorized": True,
                                "physical_gpu": {
                                    "index": gpu_index,
                                    "uuid": f"GPU-synthetic-{gpu_index}",
                                    "utilization_percent": 5,
                                    "memory_total_mib": 49140,
                                    "memory_used_mib": 2000,
                                },
                            }
                        }
                    ],
                }
                runs.append(
                    {
                        "identity": {
                            "experiment_id": report.EXPERIMENT_ID,
                            "oracle": "qed",
                            "variant": variant,
                            "seed": seed,
                            "run_id": run_id,
                        },
                        "status": "completed",
                        "summary_schema_version": 2,
                        "oracle_budget": 1000,
                        "charged_child_count": child_count,
                        "elapsed_seconds": 12.0 + seed,
                        "events": len(events),
                        "iterations_completed": len(events),
                        "config": config,
                        "metrics": metrics,
                        "provenance": provenance,
                        "launch": launch,
                        "sources": sources,
                    }
                )
                row: dict[str, object] = {
                    "experiment_id": report.EXPERIMENT_ID,
                    "run_id": run_id,
                    "oracle": "qed",
                    "variant": variant,
                    "seed": seed,
                    "summary_schema_version": 2,
                    "status": "completed",
                    "oracle_budget": 1000,
                    "all_oracle_calls": 1000,
                    "charged_child_count": child_count,
                    "elapsed_seconds": 12.0 + seed,
                    "reporting_frequency": 100,
                    "config_sha256": config_sha,
                    "model_path": config["model_path"],
                    "model_sha256": report.EXPECTED_MODEL_SHA256,
                    "vocabulary_sha256": report.EXPECTED_VOCABULARY_SHA256,
                    "source_summary_sha256": sources["summary"]["sha256"],
                    "source_events_sha256": sources["events"]["sha256"],
                }
                row.update({f"all_{key}": value for key, value in all_values.items()})
                row.update({f"child_total_{key}": value for key, value in child_values.items()})
                rows.append(row)

        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        csv_payload = stream.getvalue().encode("utf-8")
        csv_sha = hashlib.sha256(csv_payload).hexdigest()
        csv_path = archive / f"results.{csv_sha}.csv"
        csv_path.write_bytes(csv_payload)
        collection: dict[str, object] = {
            "schema_version": 3,
            "created_at": "2026-09-06T00:00:00Z",
            "collection_complete": True,
            "experiment_id": report.EXPERIMENT_ID,
            "experiment_root": str(experiment_root.resolve()),
            "missing_jobs": [],
            "skipped": [],
            "run_count": 12,
            "matrix": {
                "schema_version": 1,
                "path": str(matrix_path.resolve()),
                "sha256": matrix_sha,
                "expected_job_count": 12,
                "expected_jobs": expected_jobs,
                "recorded_by_all_collected_runs": True,
            },
            "results_csv": {
                "path": csv_path.name,
                "sha256": csv_sha,
                "immutable": True,
                "columns": list(CSV_FIELDS),
            },
            "runs": runs,
        }
        collection_path = archive / "collection_manifest.json"
        collection_path.write_text(json.dumps(collection, sort_keys=True), encoding="utf-8")
        return collection_path, collection

    def test_writes_deterministic_valid_pdf_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collection_path, _ = self._bundle(root)
            first = report.write_report(collection_path)
            second = report.write_report(collection_path)
            self.assertEqual(first, second)
            self.assertEqual(first["page_count"], 4)
            pdf_path = Path(first["pdf_path"])
            self.assertEqual(sha256_file(pdf_path), first["pdf_sha256"])
            reader = PdfReader(pdf_path)
            self.assertEqual(len(reader.pages), 4)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            self.assertIn("Delta-y and attribution diagnostics", text)
            manifest = json.loads(Path(first["report_path"]).read_text())
            self.assertEqual(manifest["design"]["run_count"], 12)
            paired = manifest["paired_delta_vs_matched_control"]["metrics"]
            self.assertAlmostEqual(
                paired["all_charged_molecules"]["auc_top_10"]["mean"],
                0.005,
            )
            diagnostics = manifest["aggregates"]["delta"]["diagnostics_by_seed"]["0"]
            self.assertEqual(diagnostics["charged_parent_calls"], 450)
            self.assertEqual(diagnostics["charged_child_calls"], 550)
            self.assertEqual(diagnostics["delta_y_unique_transitions"]["positive"], 449)

    def test_rejects_wrong_checkpoint_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collection_path, collection = self._bundle(root)
            collection["runs"][0]["provenance"]["model_sha256"] = "0" * 64
            collection_path.write_text(json.dumps(collection, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(report.ReportError, "final checkpoint SHA-256"):
                report.load_report_data(collection_path)

    def test_rejects_tampered_attribution_even_with_updated_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collection_path, collection = self._bundle(root)
            target = next(
                run
                for run in collection["runs"]
                if run["identity"]["variant"] == "delta" and run["identity"]["seed"] == 0
            )
            events_path = Path(collection["experiment_root"]) / target["sources"]["events"]["path"]
            events = [json.loads(line) for line in events_path.read_text().splitlines()]
            events[101]["attribution"]["mapping_coverage"] = 0.5
            events_path.write_text(
                "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
                encoding="utf-8",
            )
            new_hash = sha256_file(events_path)
            target["sources"]["events"]["sha256"] = new_hash
            rows_path = collection_path.parent / collection["results_csv"]["path"]
            with rows_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                if row["run_id"] == target["identity"]["run_id"]:
                    row["source_events_sha256"] = new_hash
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            payload = stream.getvalue().encode("utf-8")
            csv_sha = hashlib.sha256(payload).hexdigest()
            new_csv = collection_path.parent / f"results.{csv_sha}.csv"
            new_csv.write_bytes(payload)
            collection["results_csv"].update({"path": new_csv.name, "sha256": csv_sha})
            collection_path.write_text(json.dumps(collection, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(report.ReportError, "mapping_coverage"):
                report.load_report_data(collection_path)


if __name__ == "__main__":
    unittest.main()

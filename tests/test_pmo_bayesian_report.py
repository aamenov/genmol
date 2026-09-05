"""Focused CPU-only tests for the fixed Bayesian-ablation PDF report."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from pypdf import PdfReader

from scripts.exps.pmo import report_bayesian_ablation as report
from scripts.exps.pmo.main.genmol.experiment_io import (
    sha256_config,
    sha256_file,
    trajectory_auc,
)


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
    "all_auc_top_1",
    "all_auc_top_10",
    "all_auc_top_100",
    "all_top_1",
    "all_top_10",
    "all_top_100",
    "prior_mean",
    "prior_strength",
    "prior_mean_source",
    "model_path",
    "model_sha256",
    "vocabulary_path",
    "vocabulary_sha256",
    "config_sha256",
    "source_summary_sha256",
)


class BayesianReportTests(unittest.TestCase):
    def _write_csv(
        self,
        archive: Path,
        rows: list[dict[str, object]],
        collection: dict,
    ) -> Path:
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        payload = stream.getvalue().encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        path = archive / f"results.{digest}.csv"
        path.write_bytes(payload)
        collection["results_csv"] = {
            "path": path.name,
            "sha256": digest,
            "immutable": True,
            "columns": list(CSV_FIELDS),
        }
        return path

    def _bundle(self, root: Path) -> tuple[Path, dict, list[dict[str, object]]]:
        experiment_id = "fragment_vocab_qed_50k_bayes_1k_v1"
        experiment_root = root / "raw" / experiment_id
        archive = root / "archive"
        experiment_root.mkdir(parents=True)
        archive.mkdir()
        matrix_path = root / "qed_50k_bayes_1k_v1.yaml"
        matrix_path.write_text("schema_version: 1\nexperiment_id: test\n", encoding="utf-8")
        matrix_hash = sha256_file(matrix_path)
        model_path = root / "outputs" / "paper_v1" / "checkpoints" / "50000.ckpt"
        vocabulary_path = root / "qed.csv"
        model_hash = report.EXPECTED_MODEL_SHA256
        vocabulary_hash = "2" * 64
        prior_source = "fixed neutral proxy; source molecule scores unavailable"
        expected_jobs = []
        runs = []
        csv_rows: list[dict[str, object]] = []
        variant_offset = {
            variant: index * 0.008
            for index, variant in enumerate(report.EXPECTED_VARIANTS)
        }

        for variant in report.EXPECTED_VARIANTS:
            for seed in report.EXPECTED_SEEDS:
                run_id = f"{experiment_id}:qed:{variant}:seed{seed}"
                expected_jobs.append(
                    {"oracle": "qed", "variant": variant, "seed": seed}
                )
                final_top_10 = 0.72 + variant_offset[variant] + seed * 0.002
                trajectory = [
                    {"oracle_calls": 0, "top_k_mean": 0.0},
                    *[
                        {
                            "oracle_calls": call,
                            "top_k_mean": final_top_10,
                        }
                        for call in range(100, 1001, 100)
                    ],
                ]
                auc_top_10 = trajectory_auc(trajectory, normalize_by=1000)
                final_top_1 = final_top_10 + 0.02
                final_top_100 = final_top_10 - 0.04
                values = {
                    "all_auc_top_1": 0.95 * final_top_1,
                    "all_auc_top_10": auc_top_10,
                    "all_auc_top_100": 0.95 * final_top_100,
                    "all_top_1": final_top_1,
                    "all_top_10": final_top_10,
                    "all_top_100": final_top_100,
                }
                config = {
                    "experiment_id": experiment_id,
                    "scientific_status": (
                        "Exploratory QED-only 1,000-call Bayesian sweep using the "
                        "local 50k checkpoint; not paper-comparable."
                    ),
                    "matrix_path": str(matrix_path.resolve()),
                    "matrix_sha256": matrix_hash,
                    "oracle": "qed",
                    "variant": variant,
                    "policy_mode": report.EXPECTED_POLICY_MODE[variant],
                    "parent_control": False,
                    "model_path": str(model_path.resolve()),
                    "vocab_path": str(vocabulary_path.resolve()),
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
                    "prior_strength": report.EXPECTED_PRIOR_STRENGTH[variant],
                    "prior_mean": 0.5 if variant.startswith("shrink") else None,
                    "prior_mean_source": (
                        prior_source if variant.startswith("shrink") else None
                    ),
                    "legacy_seed_count": 1,
                    "delta_attribution": "novel_vs_parent",
                    "population_sampling_order": (
                        "canonical fragment string before uniform sampling"
                    ),
                    "statistical_duplicate_policy": (
                        "one update per unique canonical child"
                    ),
                    "released_duplicate_policy": (
                        "repeat cached-child decomposition, matching release"
                    ),
                    "durable_events": True,
                }
                summary = {
                    "schema_version": 2,
                    "run_id": run_id,
                    "status": "completed",
                    "scores": {
                        "all_charged_molecules": {
                            "oracle_calls": 1000,
                            "oracle_budget": 1000,
                            "reporting_frequency": 100,
                            "top_1": final_top_1,
                            "top_10": final_top_10,
                            "top_100": final_top_100,
                            "auc_top_1": values["all_auc_top_1"],
                            "auc_top_10": values["all_auc_top_10"],
                            "auc_top_100": values["all_auc_top_100"],
                            "trajectory_top_10": trajectory,
                        }
                    },
                }
                summary_path = (
                    experiment_root / "qed" / variant / f"seed_{seed}" / "summary.json"
                )
                summary_path.parent.mkdir(parents=True)
                summary_path.write_text(
                    json.dumps(summary, allow_nan=False, sort_keys=True), encoding="utf-8"
                )
                summary_hash = sha256_file(summary_path)
                compact = {
                    "source_key": "all_charged_molecules",
                    "semantics": "all charged molecules on the total oracle-call axis",
                    "axis": None,
                    "score_count": 1000,
                    "axis_budget": 1000,
                    "reporting_frequency": 100,
                    "top_1": final_top_1,
                    "top_10": final_top_10,
                    "top_100": final_top_100,
                    "auc_top_1": values["all_auc_top_1"],
                    "auc_top_10": values["all_auc_top_10"],
                    "auc_top_100": values["all_auc_top_100"],
                }
                config_hash = sha256_config(config)
                provenance = {
                    "config_sha256": config_hash,
                    "model_path": str(model_path.resolve()),
                    "model_sha256": model_hash,
                    "vocabulary_path": str(vocabulary_path.resolve()),
                    "vocabulary_sha256": vocabulary_hash,
                    "scientific_status": config["scientific_status"],
                    "git_commit": "3" * 40,
                    "git_branch": "codex/test",
                    "git_dirty": False,
                    "tracked_diff_sha256": "4" * 64,
                    "cuda_visible_devices": "3",
                    "resume_count": 0,
                    "durable_events_provenance_complete": True,
                    "timing_provenance_complete": True,
                }
                runs.append(
                    {
                        "identity": {
                            "experiment_id": experiment_id,
                            "run_id": run_id,
                            "oracle": "qed",
                            "variant": variant,
                            "seed": seed,
                        },
                        "run_dir": f"qed/{variant}/seed_{seed}",
                        "summary_schema_version": 2,
                        "status": "completed",
                        "oracle_budget": 1000,
                        "events": 1000,
                        "iterations_completed": 1000,
                        "elapsed_seconds": 10.0,
                        "charged_child_count": 1000,
                        "metrics": {"all_charged_molecules": compact},
                        "config": config,
                        "provenance": provenance,
                        "sources": {
                            "summary": {
                                "path": str(summary_path.relative_to(experiment_root)),
                                "sha256": summary_hash,
                            }
                        },
                        "launch": {
                            "history_complete": True,
                            "policy_complete": True,
                            "matrix_provenance_complete": True,
                            "provenance_complete": True,
                            "durable_events_config_complete": True,
                            "any_gpu_sharing": False,
                            "wall_time_comparable": True,
                            "attempt_count": 1,
                            "attempts": [
                                {
                                    "record": {
                                        "physical_gpu": {
                                            "index": 3,
                                            "uuid": "GPU-synthetic-3",
                                            "memory_total_mib": 49140,
                                            "memory_used_mib": 1000,
                                            "utilization_percent": 5,
                                        },
                                        "utilization_threshold": 10,
                                        "min_free_memory_mib": 30000,
                                        "sharing_actual": False,
                                        "time_unix": 1000.0,
                                    }
                                }
                            ],
                        },
                    }
                )
                csv_rows.append(
                    {
                        "experiment_id": experiment_id,
                        "run_id": run_id,
                        "oracle": "qed",
                        "variant": variant,
                        "seed": seed,
                        "summary_schema_version": 2,
                        "status": "completed",
                        "oracle_budget": 1000,
                        "all_oracle_calls": 1000,
                        "charged_child_count": 1000,
                        "elapsed_seconds": 10.0,
                        "reporting_frequency": 100,
                        **values,
                        "prior_mean": 0.5 if variant.startswith("shrink") else "",
                        "prior_strength": report.EXPECTED_PRIOR_STRENGTH[variant],
                        "prior_mean_source": (
                            prior_source if variant.startswith("shrink") else ""
                        ),
                        "model_path": str(model_path.resolve()),
                        "model_sha256": model_hash,
                        "vocabulary_path": str(vocabulary_path.resolve()),
                        "vocabulary_sha256": vocabulary_hash,
                        "config_sha256": config_hash,
                        "source_summary_sha256": summary_hash,
                    }
                )

        collection = {
            "schema_version": 3,
            "created_at": "2026-09-05T00:00:00Z",
            "experiment_root": str(experiment_root.resolve()),
            "experiment_id": experiment_id,
            "run_count": 18,
            "collection_complete": True,
            "missing_jobs": [],
            "skipped": [],
            "matrix": {
                "path": str(matrix_path.resolve()),
                "sha256": matrix_hash,
                "schema_version": 1,
                "expected_job_count": 18,
                "expected_jobs": expected_jobs,
                "recorded_by_all_collected_runs": True,
            },
            "collector": {"path": "collector.py", "sha256": "5" * 64},
            "runs": runs,
        }
        self._write_csv(archive, csv_rows, collection)
        collection_path = archive / "collection_manifest.json"
        collection_path.write_text(
            json.dumps(collection, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return collection_path, collection, csv_rows

    @staticmethod
    def _write_collection(path: Path, collection: dict) -> None:
        path.write_text(
            json.dumps(collection, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _rewrite_csv(
        self,
        collection_path: Path,
        collection: dict,
        rows: list[dict[str, object]],
    ) -> None:
        self._write_csv(collection_path.parent, rows, collection)
        self._write_collection(collection_path, collection)

    def test_valid_bundle_renders_deterministically_and_publishes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collection_path, _, _ = self._bundle(root)
            data = report.load_report_data(collection_path)

            self.assertEqual(len(data.runs), 18)
            expected_mean = sum(
                0.72 + 2 * 0.008 + seed * 0.002
                for seed in report.EXPECTED_SEEDS
            ) / 3
            self.assertAlmostEqual(
                data.aggregates["shrink1"]["metrics"]["all_top_10"]["mean"],
                expected_mean,
            )
            first = report.render_report_pdf(data)
            second = report.render_report_pdf(data)
            self.assertEqual(first, second)
            audit = report.validate_pdf(first)
            self.assertGreaterEqual(audit["page_count"], 5)

            output = root / "published"
            pdf_path, manifest_path = report.write_report(collection_path, output)
            self.assertEqual(pdf_path.read_bytes(), first)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["pdf"]["sha256"], sha256_file(pdf_path))
            self.assertEqual(manifest["design"]["run_count"], 18)
            reader = PdfReader(pdf_path)
            extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
            self.assertIn("not a paper-scale PMO reproduction", extracted)
            self.assertIn("GenMol Table 13", extracted)
            self.assertIn("0.942 +/- 0.000", extracted)
            self.assertIn("10,000-oracle-call budget", extracted)
            self.assertIn("Top-100 AUC", extracted)
            self.assertIn("Final top-100", extracted)
            self.assertIn("n/a", extracted)
            self.assertIn("legacy seed count", extracted)
            self.assertIn("Physical GPU mapping", extracted)
            self.assertIn("controller span", extracted)
            for variant in report.EXPECTED_VARIANTS:
                self.assertIn(variant, extracted)
            with self.assertRaises(FileExistsError):
                report.write_report(collection_path, output)
            replacement, replacement_manifest = report.write_report(
                collection_path, output, overwrite=True
            )
            self.assertEqual(replacement, pdf_path)
            self.assertEqual(replacement_manifest, manifest_path)

    def test_rejects_incomplete_or_substituted_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collection_path, collection, _ = self._bundle(root)
            collection["collection_complete"] = False
            self._write_collection(collection_path, collection)
            with self.assertRaisesRegex(report.ReportError, "completeness"):
                report.load_report_data(collection_path)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collection_path, collection, _ = self._bundle(root)
            collection["matrix"]["expected_jobs"][0]["variant"] = "support3"
            self._write_collection(collection_path, collection)
            with self.assertRaisesRegex(report.ReportError, "matrix job set"):
                report.load_report_data(collection_path)

    def test_rejects_csv_and_summary_hash_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collection_path, collection, _ = self._bundle(root)
            csv_path = collection_path.parent / collection["results_csv"]["path"]
            csv_path.write_bytes(csv_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(report.ReportError, "CSV SHA-256"):
                report.load_report_data(collection_path)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collection_path, collection, _ = self._bundle(root)
            summary_ref = collection["runs"][0]["sources"]["summary"]
            summary_path = Path(collection["experiment_root"]) / summary_ref["path"]
            summary_path.write_bytes(summary_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(report.ReportError, "summary SHA-256"):
                report.load_report_data(collection_path)

    def test_rejects_a_different_model_hash_even_if_layers_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collection_path, collection, rows = self._bundle(root)
            wrong_hash = "1" * 64
            for run, row in zip(collection["runs"], rows):
                run["provenance"]["model_sha256"] = wrong_hash
                row["model_sha256"] = wrong_hash
            self._rewrite_csv(collection_path, collection, rows)
            with self.assertRaisesRegex(report.ReportError, "50k model hash"):
                report.load_report_data(collection_path)

    def test_rejects_semantically_corrupt_trajectory_and_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collection_path, collection, rows = self._bundle(root)
            run = collection["runs"][0]
            summary_path = Path(collection["experiment_root"]) / run["sources"]["summary"]["path"]
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["scores"]["all_charged_molecules"]["trajectory_top_10"][4][
                "top_k_mean"
            ] += 0.01
            summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
            updated_hash = sha256_file(summary_path)
            run["sources"]["summary"]["sha256"] = updated_hash
            rows[0]["source_summary_sha256"] = updated_hash
            self._rewrite_csv(collection_path, collection, rows)
            with self.assertRaisesRegex(report.ReportError, "not monotone|trajectory AUC"):
                report.load_report_data(collection_path)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collection_path, collection, rows = self._bundle(root)
            run = collection["runs"][0]
            summary_path = Path(collection["experiment_root"]) / run["sources"]["summary"]["path"]
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["scores"]["all_charged_molecules"]["trajectory_top_10"][-1][
                "top_k_mean"
            ] += 0.01
            summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
            updated_hash = sha256_file(summary_path)
            run["sources"]["summary"]["sha256"] = updated_hash
            rows[0]["source_summary_sha256"] = updated_hash
            self._rewrite_csv(collection_path, collection, rows)
            with self.assertRaisesRegex(report.ReportError, "trajectory endpoint"):
                report.load_report_data(collection_path)

    def test_rejects_summary_symlink_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collection_path, collection, _ = self._bundle(root)
            run = collection["runs"][0]
            summary_path = Path(collection["experiment_root"]) / run["sources"]["summary"]["path"]
            outside = root / "outside-summary.json"
            outside.write_bytes(summary_path.read_bytes())
            summary_path.unlink()
            summary_path.symlink_to(outside)
            with self.assertRaisesRegex(report.ReportError, "symlink"):
                report.load_report_data(collection_path)


if __name__ == "__main__":
    unittest.main()

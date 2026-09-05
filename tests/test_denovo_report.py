"""CPU-only tests for the three-seed de novo benchmark report."""

from __future__ import annotations

import csv
import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.exps.denovo import benchmark, report


class DenovoReportTests(unittest.TestCase):
    def _workspace(self) -> tempfile.TemporaryDirectory[str]:
        return tempfile.TemporaryDirectory(dir=report.REPOSITORY_ROOT)

    @staticmethod
    def _branch_metrics(
        *, valid: int, unique: int, quality: int, diversity: float, definition: str
    ) -> dict:
        return {
            "validity": valid / 1_000,
            "valid_count": valid,
            "validity_denominator": 1_000,
            "uniqueness": unique / valid if valid else None,
            "unique_count": unique,
            "uniqueness_denominator": valid,
            "diversity": diversity if unique else None,
            "diversity_input_count": unique,
            "diversity_undefined_reason": None if unique else "no_unique_valid_molecules",
            "quality": quality / 1_000,
            "quality_count": quality,
            "quality_denominator": 1_000,
            "quality_thresholds": {
                "qed_min_inclusive": 0.6,
                "sa_max_inclusive": 4.0,
            },
            "definition": definition,
        }

    @staticmethod
    def _records(
        seed: int,
        *,
        strict_valid: int,
        strict_unique: int,
        strict_quality: int,
        released_valid: int,
        released_unique: int,
        released_quality: int,
        component_count: int,
    ) -> list[dict]:
        records = []
        for index in range(1_000):
            record = {field: None for field in report.RAW_SAMPLE_FIELDS}
            record.update(
                {
                    "sample_index": index,
                    "raw_model_text": f"raw-{seed}-{index}",
                    "raw_safe": f"safe-{seed}-{index}",
                    "strict_is_first_unique": False,
                    "strict_quality_counted": False,
                    "released_is_first_unique": False,
                    "released_quality_counted": False,
                    "released_was_recovered": False,
                    "released_largest_component_applied": False,
                }
            )

            if index < strict_valid:
                unique_index = index if index < strict_unique else 0
                strict_smiles = f"strict_{seed}_{unique_index}"
                strict_pass = unique_index < strict_quality
                record.update(
                    {
                        "strict_smiles": strict_smiles,
                        "strict_qed": 0.7 if strict_pass else 0.5,
                        "strict_sa": 3.0,
                        "strict_is_first_unique": index < strict_unique,
                        "strict_quality_pass": strict_pass,
                        "strict_quality_counted": index < strict_quality,
                    }
                )
            else:
                record["strict_decode_error"] = "decode_returned_none"

            if index < released_valid:
                unique_index = index if index < released_unique else 0
                released_smiles = f"released_{seed}_{unique_index}"
                released_pass = unique_index < released_quality
                repaired_smiles = released_smiles
                if index < component_count:
                    repaired_smiles = f"C.{released_smiles}"
                record.update(
                    {
                        "released_repaired_smiles": repaired_smiles,
                        "released_smiles": released_smiles,
                        "released_qed": 0.7 if released_pass else 0.5,
                        "released_sa": 3.0,
                        "released_is_first_unique": index < released_unique,
                        "released_quality_pass": released_pass,
                        "released_quality_counted": index < released_quality,
                        "released_was_recovered": index >= strict_valid,
                        "released_largest_component_applied": index < component_count,
                    }
                )
            else:
                record["released_decode_error"] = "decode_returned_none"
            records.append(record)
        return records

    def _completed_run(
        self,
        root: Path,
        seed: int,
        *,
        strict_valid: int,
        strict_unique: int,
        strict_quality: int,
        strict_diversity: float,
        released_valid: int,
        released_unique: int,
        released_quality: int,
        released_diversity: float,
        component_count: int = 4,
        effective_extra: dict | None = None,
    ) -> Path:
        run_dir = root / f"seed_{seed}"
        run_dir.mkdir(parents=True)
        records = self._records(
            seed,
            strict_valid=strict_valid,
            strict_unique=strict_unique,
            strict_quality=strict_quality,
            released_valid=released_valid,
            released_unique=released_unique,
            released_quality=released_quality,
            component_count=component_count,
        )
        raw_path = run_dir / "raw_samples.csv"
        with raw_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=report.RAW_SAMPLE_FIELDS)
            writer.writeheader()
            writer.writerows(records)

        sampling = dict(report.PAPER_V1_SAMPLING_CONFIG)
        source = {"model_path": "model.ckpt", "num_samples": 1_000, **sampling}
        effective = {
            **source,
            "model_path": str(
                report.REPOSITORY_ROOT / "outputs/paper_v1/checkpoints/50000.ckpt"
            ),
            "num_samples": 1_000,
            "device": "cuda:0",
            **(effective_extra or {}),
        }
        strict_metrics = self._branch_metrics(
            valid=strict_valid,
            unique=strict_unique,
            quality=strict_quality,
            diversity=strict_diversity,
            definition="strict unit-test path",
        )
        released_metrics = self._branch_metrics(
            valid=released_valid,
            unique=released_unique,
            quality=released_quality,
            diversity=released_diversity,
            definition="released-comparable unit-test path",
        )
        failure_counts = {
            "raw_safe_conversion_failed": 0,
            "strict_decode_failed": 1_000 - strict_valid,
            "released_decode_failed": 1_000 - released_valid,
            "released_recovered_strict_failure": released_valid - strict_valid,
            "strict_valid_but_released_failed": 0,
            "released_largest_component_applied": component_count,
            "strict_duplicates": strict_valid - strict_unique,
            "released_duplicates": released_valid - released_unique,
        }
        implementation_inputs = {
            "sampler_source": {
                "path": str(report.REPOSITORY_ROOT / "src/genmol/sampler.py"),
                "sha256": "1" * 64,
                "size_bytes": 123,
            },
            "chemistry_utils_source": {
                "path": str(report.REPOSITORY_ROOT / "src/genmol/utils/utils_chem.py"),
                "sha256": "2" * 64,
                "size_bytes": 456,
            },
            "model_source": {
                "path": str(report.REPOSITORY_ROOT / "src/genmol/model.py"),
                "sha256": "6" * 64,
                "size_bytes": 234,
            },
            "diffusion_source": {
                "path": str(report.REPOSITORY_ROOT / "src/genmol/diffusion.py"),
                "sha256": "b" * 64,
                "size_bytes": 222,
            },
            "backbone_source": {
                "path": str(report.REPOSITORY_ROOT / "src/genmol/backbone.py"),
                "sha256": "c" * 64,
                "size_bytes": 333,
            },
            "data_utils_source": {
                "path": str(report.REPOSITORY_ROOT / "src/genmol/utils/utils_data.py"),
                "sha256": "7" * 64,
                "size_bytes": 345,
            },
            "bracket_safe_converter_source": {
                "path": str(
                    report.REPOSITORY_ROOT
                    / "src/genmol/utils/bracket_safe_converter.py"
                ),
                "sha256": "8" * 64,
                "size_bytes": 567,
            },
            "length_distribution": {
                "path": str(report.REPOSITORY_ROOT / "data/len.pk"),
                "sha256": report.TRAINING_CONTEXT["data_and_tokenizer"]["length_file"][
                    "sha256"
                ],
                "size_bytes": 789,
                "count": 249_455,
                "minimum": 10,
                "median": 49.0,
                "maximum": 87,
            },
        }
        run_command = [
            ".venv/bin/python",
            "scripts/exps/denovo/benchmark.py",
            "--checkpoint",
            str(report.REPOSITORY_ROOT / "outputs/paper_v1/checkpoints/50000.ckpt"),
            "--config",
            str(report.REPOSITORY_ROOT / "scripts/exps/denovo/hparams.yaml"),
            "--num-samples",
            "1000",
            "--seed",
            str(seed),
            "--device",
            "cuda:0",
            "--output-dir",
            str(run_dir),
        ]
        physical_index = seed % 2 + 1
        gpu_uuid = f"GPU-synthetic-{physical_index}"
        selection = {
            "event": "launch",
            "timestamp_utc": "2026-09-05T00:00:00+00:00",
            "source_revision": {"head": "4" * 40, "upstream": "4" * 40},
            "physical_gpu": {
                "index": physical_index,
                "uuid": gpu_uuid,
                "name": "NVIDIA RTX A6000",
                "memory_used_mib": 8 + seed,
                "memory_total_mib": 49_140,
                "utilization_percent": seed,
                "compute_mode": "Default",
                "compute_processes": [],
            },
            "policy": {
                "max_utilization_percent": 10,
                "utilization_comparison": "strictly_less_than",
                "min_free_memory_mib": 40_000,
                "active_compute_processes_allowed": False,
                "selected_physical_indices": [1, 2],
            },
            "command": run_command,
        }
        summary = {
            "schema_version": report.RUN_SCHEMA_VERSION,
            "status": "completed",
            "seed": seed,
            "num_samples": 1_000,
            "run": {
                "seed": seed,
                "requested_sample_count": 1_000,
                "evaluation_tier": "final",
                "final_protocol_eligible": True,
                "started_at_utc": "2026-09-05T00:00:00+00:00",
                "completed_at_utc": "2026-09-05T00:01:00+00:00",
                "one_seed_per_invocation": True,
                "single_generation_batch": True,
                "generation_protocol": dict(report.EXPECTED_GENERATION_PROTOCOL),
                "command": run_command,
                "seed_configuration": {
                    "seed": seed,
                    "seed_applied_immediately_before_generation": True,
                    "python_random": True,
                    "numpy": True,
                    "torch_cpu": True,
                    "torch_cuda_all": True,
                    "python_hash_seed": str(seed),
                },
            },
            "checkpoint": {
                "path": str(
                    report.REPOSITORY_ROOT / "outputs/paper_v1/checkpoints/50000.ckpt"
                ),
                "sha256": report.EXPECTED_CHECKPOINT_SHA256,
                "size_bytes": report.EXPECTED_CHECKPOINT_SIZE_BYTES,
                "mtime_utc": "2026-09-05T00:00:00+00:00",
                "global_step": 50_000,
                "epoch": 0,
                "diffusion_type": "mdlm",
                "udlm_inference_eps": None,
                "udlm_exclude_special_tokens": None,
            },
            "config": {
                "path": str(report.REPOSITORY_ROOT / "scripts/exps/denovo/hparams.yaml"),
                "sha256": "3" * 64,
                "sampling_sha256": report._sha256_json(sampling),
                "effective_sha256": report._sha256_json(effective),
                "source": source,
                "sampling": sampling,
                "effective": effective,
            },
            "metrics": {
                "released_comparable": released_metrics,
                "strict": strict_metrics,
            },
            "failure_counts": failure_counts,
            "runtime_seconds": {
                "model_load_and_device_move": 1.0,
                "model_sampling_and_tokenizer": 19.5 + seed,
                "released_postprocessing": 0.5,
                "generation": 20.0 + seed,
                "decode_and_metrics": 3.0,
                "total_before_summary_write": 24.0 + seed,
            },
            "environment": {
                "python": "3.10.0 synthetic",
                "platform": "Linux-synthetic",
                "executable": ".venv/bin/python",
                "working_directory": str(report.REPOSITORY_ROOT),
                "versions": {
                    "torch": "2.6.0",
                    "lightning": "2.5.1",
                    "transformers": "4.52.4",
                    "numpy": "1.26.4",
                    "pandas": "2.1.0",
                    "pyyaml": "6.0.2",
                    "safe": "0.1.14",
                    "rdkit": "2023.9.6",
                    "tdc": "0.4.1",
                    "bionemo_moco": "0.0.2.1",
                },
                "requested_device": "cuda:0",
                "resolved_model_device": "cuda:0",
                "torch_cuda_available": True,
                "torch_cuda_version": "12.4",
                "cudnn_version": 90100,
                "cuda_device": {
                    "logical_index": 0,
                    "name": "NVIDIA RTX A6000",
                    "total_memory_bytes": 51_527_139_328,
                    "compute_capability": [8, 6],
                },
                "launch_environment": {
                    "CUDA_VISIBLE_DEVICES": gpu_uuid,
                    "GENMOL_BENCHMARK_GPU_PHYSICAL_INDEX": str(physical_index),
                    "GENMOL_BENCHMARK_GPU_UUID": gpu_uuid,
                    "GENMOL_BENCHMARK_GPU_SELECTION_SNAPSHOT": json.dumps(
                        selection, separators=(",", ":"), sort_keys=True
                    ),
                    "GENMOL_BENCHMARK_RUN_LABEL": report.benchmark_run_label(
                        report.EXPECTED_GLOBAL_STEP,
                        report.EXPECTED_CHECKPOINT_SHA256,
                        seed,
                    ),
                },
            },
            "git": {
                "commit": "4" * 40,
                "branch": "test",
                "dirty": True,
                "runner_sha256": "5" * 64,
            },
            "implementation_inputs": implementation_inputs,
            "tokenizer": {
                "requested_identifier": "datamol-io/safe-gpt",
                "class": "transformers.PreTrainedTokenizerFast",
                "name_or_path": None,
                "declared_revision": None,
                "resolved_commit_hash": None,
                "base_vocab_size": 1_880,
                "effective_size": 1_882,
                "vocabulary_sha256": "9" * 64,
                "added_vocabulary_sha256": "a" * 64,
                "backend_json_sha256": None,
                "backend_serialization_error": (
                    "Exception: Custom PreTokenizer cannot be serialized"
                ),
                "special_token_ids": {"pad": 3, "bos": 1, "eos": 2, "mask": 4},
            },
            "artifacts": {
                "raw_samples_csv": {
                    "path": str(raw_path),
                    "sha256": report._sha256_file(raw_path),
                    "row_count": 1_000,
                    "fields": list(report.RAW_SAMPLE_FIELDS),
                },
                "summary_json": {"path": str(run_dir / "summary.json")},
            },
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return run_dir

    def _three_runs(self, root: Path) -> None:
        for seed in (0, 1, 2):
            self._completed_run(
                root,
                seed,
                strict_valid=900 + 10 * seed,
                strict_unique=890 + 10 * seed,
                strict_quality=700 + 10 * seed,
                strict_diversity=0.79 + 0.01 * seed,
                released_valid=1_000 - seed,
                released_unique=997 - seed,
                released_quality=(840, 850, 830)[seed],
                released_diversity=(0.817, 0.819, 0.818)[seed],
            )

    def _three_udlm_runs(self, root: Path) -> None:
        self._three_runs(root)
        checkpoint_path = report.REPOSITORY_ROOT / "output/udlm/checkpoints/100.ckpt"
        checkpoint_sha = "d" * 64
        sampling = {
            "diffusion_type": "udlm",
            "softmax_temp": 1.0,
            "randomness": 0.0,
            "min_add_len": 40,
            "num_steps": 32,
            "inference_eps": 1e-5,
            "exclude_special_tokens": False,
        }
        for seed in report.EXPECTED_SEEDS:
            summary_path = root / f"seed_{seed}" / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["checkpoint"].update(
                {
                    "path": str(checkpoint_path),
                    "sha256": checkpoint_sha,
                    "global_step": 100,
                    "diffusion_type": "udlm",
                    "udlm_inference_eps": 1e-5,
                    "udlm_exclude_special_tokens": False,
                }
            )
            summary["config"]["path"] = str(
                report.REPOSITORY_ROOT / "scripts/exps/denovo/hparams_udlm.yaml"
            )
            summary["config"]["sampling"] = sampling
            summary["config"]["sampling_sha256"] = report._sha256_json(sampling)
            summary["config"]["source"].update(sampling)
            summary["config"]["source"]["model_path"] = "model.ckpt"
            summary["config"]["effective"].update(sampling)
            summary["config"]["effective"]["model_path"] = str(checkpoint_path)
            summary["config"]["effective_sha256"] = report._sha256_json(
                summary["config"]["effective"]
            )
            protocol = summary["run"]["generation_protocol"]
            protocol.update(
                {
                    "diffusion_type": "udlm",
                    "nfe": 32,
                    "num_steps": 32,
                    "num_steps_source": "explicit UDLM reverse-transition count",
                    "inference_eps": 1e-5,
                    "temperature": 1.0,
                    "randomness": 0.0,
                    "randomness_used_by_sampler": False,
                    "exclude_special_tokens": False,
                }
            )
            run_label = report.benchmark_run_label(100, checkpoint_sha, seed)
            summary["environment"]["launch_environment"][
                "GENMOL_BENCHMARK_RUN_LABEL"
            ] = run_label
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    def test_collect_recounts_rows_and_uses_sample_sd(self):
        with self._workspace() as directory:
            runs = Path(directory) / "nested" / "runs"
            self._three_runs(runs)
            payload = report.collect_report(runs)

            self.assertEqual(payload["required_protocol"]["total_requested_samples"], 3_000)
            released = payload["aggregate_metrics"]["released_comparable"]
            self.assertAlmostEqual(released["validity"]["mean"], 0.999)
            self.assertAlmostEqual(released["validity"]["sample_sd"], 0.001)
            self.assertAlmostEqual(released["quality"]["mean"], 0.84)
            self.assertAlmostEqual(released["quality"]["sample_sd"], 0.01)
            funnel = payload["strict_vs_repaired_funnel"]["sum_across_seeds"]
            self.assertEqual(funnel["requested"], 3_000)
            self.assertEqual(funnel["strict_valid"], 2_730)
            self.assertEqual(funnel["released_recovered_strict_failure"], 267)
            self.assertIn("ddof=1", payload["metric_definitions"]["aggregation"])

    def test_collect_supports_udlm_and_records_effective_nfe(self):
        with self._workspace() as directory:
            runs = Path(directory) / "runs"
            self._three_udlm_runs(runs)
            payload = report.collect_report(runs)

            self.assertEqual(payload["checkpoint"]["diffusion_type"], "udlm")
            self.assertEqual(payload["generation_protocol"]["nfe"], 32)
            self.assertFalse(
                payload["generation_protocol"]["randomness_used_by_sampler"]
            )
            self.assertEqual(
                payload["generation_protocol"]["nfe_by_seed"],
                [
                    {"seed": 0, "nfe": 32},
                    {"seed": 1, "nfe": 32},
                    {"seed": 2, "nfe": 32},
                ],
            )
            outputs = report.write_report_bundle(
                payload,
                output_dir=Path(directory) / "udlm-aggregate",
                pdf_path=Path(directory) / "udlm-report.pdf",
            )
            self.assertTrue(outputs["pdf"].is_file())
            self.assertGreaterEqual(
                report.validate_pdf(
                    outputs["pdf"], expected_checkpoint_sha256="d" * 64
                )["page_count"],
                3,
            )

            summary_path = runs / "seed_1" / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["run"]["generation_protocol"][
                "randomness_used_by_sampler"
            ] = True
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(
                report.ReportValidationError,
                "UDLM ignores randomness",
            ):
                report.collect_report(runs)

    def test_full_bundle_writes_machine_outputs_and_valid_pdf(self):
        with self._workspace() as directory:
            workspace = Path(directory)
            runs = workspace / "runs"
            output = workspace / "aggregate"
            pdf_path = workspace / "pdf" / "benchmark.pdf"
            self._three_runs(runs)
            payload = report.collect_report(runs)
            outputs = report.write_report_bundle(
                payload, output_dir=output, pdf_path=pdf_path
            )

            self.assertEqual(set(outputs), {"json", "csv", "pdf"})
            aggregate = json.loads(outputs["json"].read_text(encoding="utf-8"))
            self.assertEqual(aggregate["status"], "completed")
            self.assertEqual(
                aggregate["checkpoint"]["sha256"], report.EXPECTED_CHECKPOINT_SHA256
            )
            with outputs["csv"].open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 36)
            self.assertEqual(
                len([row for row in rows if row["row_type"] == "aggregate_metric"]),
                9,
            )
            validation = report.validate_pdf(outputs["pdf"])
            self.assertGreaterEqual(validation["page_count"], 3)

    def test_missing_seed_is_rejected(self):
        with self._workspace() as directory:
            runs = Path(directory) / "runs"
            for seed in (0, 1):
                self._completed_run(
                    runs,
                    seed,
                    strict_valid=900,
                    strict_unique=890,
                    strict_quality=700,
                    strict_diversity=0.8,
                    released_valid=1_000,
                    released_unique=997,
                    released_quality=840,
                    released_diversity=0.818,
                )
            with self.assertRaisesRegex(report.ReportValidationError, "exactly 3"):
                report.collect_report(runs)

    def test_config_mismatch_is_rejected_even_with_valid_fingerprint(self):
        with self._workspace() as directory:
            runs = Path(directory) / "runs"
            self._three_runs(runs)
            summary_path = runs / "seed_2" / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["config"]["source"]["run_note"] = "different"
            summary["config"]["effective"]["run_note"] = "different"
            summary["config"]["effective_sha256"] = report._sha256_json(
                summary["config"]["effective"]
            )
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(report.ReportValidationError, "effective_sha256 differs"):
                report.collect_report(runs)

    def test_summary_count_tampering_is_rejected(self):
        with self._workspace() as directory:
            runs = Path(directory) / "runs"
            self._three_runs(runs)
            summary_path = runs / "seed_1" / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["metrics"]["strict"]["quality_count"] += 1
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(report.ReportValidationError, "disagrees with raw rows"):
                report.collect_report(runs)

    def test_uniform_self_consistent_nonpaper_sampling_is_rejected(self):
        with self._workspace() as directory:
            runs = Path(directory) / "runs"
            self._three_runs(runs)
            wrong = {
                **report.PAPER_V1_SAMPLING_CONFIG,
                "softmax_temp": 1.0,
                "randomness": 1.0,
                "min_add_len": 10,
            }
            for seed in (0, 1, 2):
                summary_path = runs / f"seed_{seed}" / "summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary["config"]["sampling"] = dict(wrong)
                summary["config"]["sampling_sha256"] = report._sha256_json(wrong)
                summary["config"]["source"].update(wrong)
                summary["config"]["effective"].update(wrong)
                summary["config"]["effective_sha256"] = report._sha256_json(
                    summary["config"]["effective"]
                )
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(report.ReportValidationError, "exact GenMol V1"):
                report.collect_report(runs)

    def test_duplicate_ordered_raw_outputs_are_rejected(self):
        with self._workspace() as directory:
            runs = Path(directory) / "runs"
            self._three_runs(runs)
            source_summary_path = runs / "seed_0" / "summary.json"
            target_summary_path = runs / "seed_1" / "summary.json"
            source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
            target_summary = json.loads(target_summary_path.read_text(encoding="utf-8"))
            source_raw = (runs / "seed_0" / "raw_samples.csv").read_bytes()
            target_raw_path = runs / "seed_1" / "raw_samples.csv"
            target_raw_path.write_bytes(source_raw)
            target_summary["metrics"] = copy.deepcopy(source_summary["metrics"])
            target_summary["failure_counts"] = copy.deepcopy(
                source_summary["failure_counts"]
            )
            target_summary["artifacts"]["raw_samples_csv"]["sha256"] = (
                report._sha256_file(target_raw_path)
            )
            target_summary_path.write_text(json.dumps(target_summary), encoding="utf-8")
            with self.assertRaisesRegex(report.ReportValidationError, "identical ordered"):
                report.collect_report(runs)

    def test_invalid_seed_provenance_is_rejected(self):
        with self._workspace() as directory:
            runs = Path(directory) / "runs"
            self._three_runs(runs)
            summary_path = runs / "seed_2" / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["run"]["seed_configuration"]["python_hash_seed"] = "0"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(report.ReportValidationError, "python_hash_seed"):
                report.collect_report(runs)

    def test_launcher_run_label_contract_is_required(self):
        with self._workspace() as directory:
            runs = Path(directory) / "runs"
            self._three_runs(runs)
            summary_path = runs / "seed_1" / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            launch = summary["environment"]["launch_environment"]
            expected = benchmark.benchmark_run_label(
                report.EXPECTED_GLOBAL_STEP,
                report.EXPECTED_CHECKPOINT_SHA256,
                1,
            )

            self.assertEqual(
                expected,
                "denovo_step50000_8d00aa47b02f_seed1",
            )
            self.assertEqual(launch["GENMOL_BENCHMARK_RUN_LABEL"], expected)
            report.collect_report(runs)

            launch["GENMOL_BENCHMARK_RUN_LABEL"] = "denovo_50000_seed1"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(
                report.ReportValidationError,
                "unexpected benchmark run label",
            ):
                report.collect_report(runs)

    def test_unpushed_source_revision_is_rejected(self):
        with self._workspace() as directory:
            runs = Path(directory) / "runs"
            self._three_runs(runs)
            summary_path = runs / "seed_0" / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            launch = summary["environment"]["launch_environment"]
            snapshot = json.loads(
                launch["GENMOL_BENCHMARK_GPU_SELECTION_SNAPSHOT"]
            )
            snapshot["source_revision"]["upstream"] = "f" * 40
            launch["GENMOL_BENCHMARK_GPU_SELECTION_SNAPSHOT"] = json.dumps(snapshot)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(
                report.ReportValidationError,
                "not the recorded pushed commit",
            ):
                report.collect_report(runs)

    def test_dependency_version_mismatch_is_rejected(self):
        with self._workspace() as directory:
            runs = Path(directory) / "runs"
            self._three_runs(runs)
            summary_path = runs / "seed_1" / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["environment"]["versions"]["torch"] = "different"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(report.ReportValidationError, "metadata differs"):
                report.collect_report(runs)

    def test_active_gpu_process_is_rejected_even_below_utilization_threshold(self):
        with self._workspace() as directory:
            runs = Path(directory) / "runs"
            self._three_runs(runs)
            summary_path = runs / "seed_0" / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            launch = summary["environment"]["launch_environment"]
            snapshot = json.loads(
                launch["GENMOL_BENCHMARK_GPU_SELECTION_SNAPSHOT"]
            )
            snapshot["physical_gpu"]["compute_processes"] = [
                {"pid": 123, "process_name": "other", "used_memory_mib": 4}
            ]
            snapshot["physical_gpu"]["utilization_percent"] = 14
            launch["GENMOL_BENCHMARK_GPU_SELECTION_SNAPSHOT"] = json.dumps(snapshot)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(
                report.ReportValidationError,
                "active compute processes",
            ):
                report.collect_report(runs)

    def test_gpu_utilization_equal_to_exclusive_threshold_is_rejected(self):
        with self._workspace() as directory:
            runs = Path(directory) / "runs"
            self._three_runs(runs)
            summary_path = runs / "seed_0" / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            launch = summary["environment"]["launch_environment"]
            snapshot = json.loads(
                launch["GENMOL_BENCHMARK_GPU_SELECTION_SNAPSHOT"]
            )
            snapshot["physical_gpu"]["utilization_percent"] = 10
            launch["GENMOL_BENCHMARK_GPU_SELECTION_SNAPSHOT"] = json.dumps(snapshot)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(
                report.ReportValidationError,
                "strict utilization threshold",
            ):
                report.collect_report(runs)

    def test_gpu_outside_user_selected_physical_ids_is_rejected(self):
        with self._workspace() as directory:
            runs = Path(directory) / "runs"
            self._three_runs(runs)
            summary_path = runs / "seed_0" / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            launch = summary["environment"]["launch_environment"]
            snapshot = json.loads(
                launch["GENMOL_BENCHMARK_GPU_SELECTION_SNAPSHOT"]
            )
            snapshot["policy"]["selected_physical_indices"] = [2, 3]
            launch["GENMOL_BENCHMARK_GPU_SELECTION_SNAPSHOT"] = json.dumps(snapshot)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(
                report.ReportValidationError,
                "not explicitly chosen by the user",
            ):
                report.collect_report(runs)

    def test_tokenizer_fingerprint_mismatch_is_rejected(self):
        with self._workspace() as directory:
            runs = Path(directory) / "runs"
            self._three_runs(runs)
            summary_path = runs / "seed_1" / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["tokenizer"]["vocabulary_sha256"] = "c" * 64
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(report.ReportValidationError, "tokenizer metadata"):
                report.collect_report(runs)

    def test_generation_timing_must_equal_audited_subcomponents(self):
        with self._workspace() as directory:
            runs = Path(directory) / "runs"
            self._three_runs(runs)
            summary_path = runs / "seed_0" / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["runtime_seconds"]["generation"] += 1.0
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(report.ReportValidationError, "generation runtime"):
                report.collect_report(runs)


if __name__ == "__main__":
    unittest.main()

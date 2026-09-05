import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from scripts.exps.pmo import launch_ablation as launcher


class LauncherTests(unittest.TestCase):
    @staticmethod
    def _matrix(root: Path) -> dict:
        return {
            "schema_version": 1,
            "experiment_id": "test_matrix",
            "scientific_status": "unit-test fixture",
            "model_path": str((root / "model.ckpt").resolve()),
            "tasks": [{"oracle": "qed", "gamma": 0.0}],
            "variants": ["released"],
            "seeds": [3],
            "common": {
                "output_root": str((root / "runs").resolve()),
                "max_oracle_calls": 7,
            },
        }

    def test_matrix_schema_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.yaml"
            matrix = self._matrix(Path(directory))
            matrix["schema_version"] = 2
            path.write_text(yaml.safe_dump(matrix))
            with self.assertRaisesRegex(ValueError, "schema_version"):
                launcher._load_matrix(path)

    def test_gpu_total_limit_defaults_to_four_and_must_be_positive(self):
        args = launcher._parse_args(
            ["--matrix", "matrix.yaml", "--gpu-indices", "3", "4"]
        )

        self.assertEqual(args.max_total_active_gpus, 4)
        launcher._validate_gpu_request([3, 4], 2, args.max_total_active_gpus)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            launcher._validate_gpu_request([3], 0, 0)

    def test_gpu_total_limit_allows_explicit_five(self):
        args = launcher._parse_args(
            [
                "--matrix",
                "matrix.yaml",
                "--gpu-indices",
                "3",
                "4",
                "--reserved-active-gpus",
                "3",
                "--max-total-active-gpus",
                "5",
            ]
        )

        self.assertEqual(args.max_total_active_gpus, 5)
        launcher._validate_gpu_request(
            args.gpu_indices,
            args.reserved_active_gpus,
            args.max_total_active_gpus,
        )

    def test_gpu_total_limit_rejects_excess_total(self):
        with self.assertRaisesRegex(ValueError, r"max-total-active-gpus \(4\)"):
            launcher._validate_gpu_request([3, 4], 3, 4)

    def test_command_records_exact_matrix_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix_path = (root / "matrix.yaml").resolve()
            matrix = self._matrix(root)
            matrix_path.write_text(yaml.safe_dump(matrix))
            digest = "a" * 64
            job = launcher._jobs(matrix)[0]

            command = launcher._command(matrix_path, digest, matrix, job)

            self.assertEqual(command[command.index("--matrix-path") + 1], str(matrix_path))
            self.assertEqual(command[command.index("--matrix-sha256") + 1], digest)

    def test_command_accepts_running_mean_delta_control(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix_path = (root / "matrix.yaml").resolve()
            matrix = self._matrix(root)
            matrix["variants"] = ["running_mean_delta_control"]
            matrix_path.write_text(yaml.safe_dump(matrix))
            job = launcher._jobs(matrix)[0]

            command = launcher._command(matrix_path, "a" * 64, matrix, job)

            self.assertEqual(
                command[command.index("--variant") + 1],
                "running_mean_delta_control",
            )
            self.assertEqual(
                launcher.VARIANT_SETTINGS["running_mean_delta_control"]["mode"],
                "delta_control",
            )

    def test_command_forwards_prior_for_every_bayesian_strength(self):
        expected_strengths = {
            "shrink1": 1.0,
            "shrink3": 3.0,
            "shrink10": 10.0,
            "shrink30": 30.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix_path = (root / "matrix.yaml").resolve()
            matrix = self._matrix(root)
            matrix["tasks"][0].update(
                prior_mean=0.42,
                prior_mean_source="frozen unit-test prior",
            )
            matrix["variants"] = list(expected_strengths)
            matrix_path.write_text(yaml.safe_dump(matrix))

            for job in launcher._jobs(matrix):
                with self.subTest(variant=job.variant):
                    command = launcher._command(matrix_path, "a" * 64, matrix, job)
                    self.assertEqual(
                        launcher.VARIANT_SETTINGS[job.variant]["prior_strength"],
                        expected_strengths[job.variant],
                    )
                    self.assertEqual(
                        command[command.index("--prior-mean") + 1], "0.42"
                    )
                    self.assertEqual(
                        command[command.index("--prior-mean-source") + 1],
                        "frozen unit-test prior",
                    )

    def test_each_bayesian_variant_requires_finite_sourced_prior(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix_path = (root / "matrix.yaml").resolve()
            for variant in ("shrink1", "shrink3", "shrink10", "shrink30"):
                with self.subTest(variant=variant):
                    matrix = self._matrix(root)
                    matrix["variants"] = [variant]
                    matrix["tasks"][0].update(
                        prior_mean=float("nan"),
                        prior_mean_source="",
                    )
                    job = launcher._jobs(matrix)[0]
                    with self.assertRaisesRegex(ValueError, f"{variant} task"):
                        launcher._command(matrix_path, "a" * 64, matrix, job)

    def test_completed_requires_matching_matrix_and_consistent_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix_path = (root / "matrix.yaml").resolve()
            matrix = self._matrix(root)
            matrix["experiment_id"] = "fragment_vocab_qed_50k_delta_1k_v2"
            matrix_path.write_text(yaml.safe_dump(matrix))
            matrix_hash = launcher.hashlib.sha256(matrix_path.read_bytes()).hexdigest()
            job = launcher._jobs(matrix)[0]
            run_dir = launcher._run_dir(matrix, job)
            (run_dir / "state").mkdir(parents=True)
            (run_dir / "state" / "latest.pkl").write_bytes(b"checkpoint")
            config = {
                "experiment_id": matrix["experiment_id"],
                "scientific_status": matrix["scientific_status"],
                "oracle": job.oracle,
                "variant": job.variant,
                "seed": job.seed,
                "policy_mode": "released",
                "seed_initialization_policy": "released_absolute_rows",
                "credit_value_policy": "absolute_child_score",
                "model_path": matrix["model_path"],
                "matrix_path": str(matrix_path),
                "matrix_sha256": matrix_hash,
            }
            run_id = f"{matrix['experiment_id']}:qed:released:seed3"
            config_hash = launcher._config_sha256(config)
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "completed",
                        "run_id": run_id,
                        "task": "qed",
                        "variant": "released",
                        "seed": 3,
                        "oracle_budget": 7,
                        "oracle_calls": 7,
                        "config": config,
                        "config_sha256": config_hash,
                    }
                )
            )
            summary = {
                "schema_version": 2,
                "status": "completed",
                "run_id": run_id,
                "checkpoint_consistent": True,
                "recoverable_oracle_calls": 7,
                "config_sha256": config_hash,
                "scores": {"all_charged_molecules": {"oracle_calls": 7}},
            }
            summary_path = run_dir / "summary.json"
            summary_path.write_text(json.dumps(summary))

            self.assertTrue(
                launcher._completed(matrix_path, matrix_hash, matrix, job)
            )
            self.assertFalse(
                launcher._completed(matrix_path, "b" * 64, matrix, job)
            )
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            config.pop("seed_initialization_policy")
            config_hash = launcher._config_sha256(config)
            manifest["config"] = config
            manifest["config_sha256"] = config_hash
            summary["config_sha256"] = config_hash
            manifest_path.write_text(json.dumps(manifest))
            summary_path.write_text(json.dumps(summary))
            self.assertFalse(
                launcher._completed(matrix_path, matrix_hash, matrix, job)
            )
            config["seed_initialization_policy"] = "released_absolute_rows"
            config_hash = launcher._config_sha256(config)
            manifest["config"] = config
            manifest["config_sha256"] = config_hash
            summary["config_sha256"] = config_hash
            manifest_path.write_text(json.dumps(manifest))
            summary["checkpoint_consistent"] = False
            summary_path.write_text(json.dumps(summary))
            self.assertFalse(
                launcher._completed(matrix_path, matrix_hash, matrix, job)
            )

    def test_gpu_gate_is_strict_and_labels_shared_snapshot(self):
        state = launcher.GPUState(
            index=4,
            uuid="GPU-test",
            memory_total_mib=49_000,
            memory_used_mib=4_000,
            utilization_percent=9,
        )
        process = {
            "gpu_uuid": "GPU-test",
            "pid": 123,
            "process_name": "other",
            "used_memory_mib": 4_000,
        }
        with mock.patch.object(launcher, "_gpu_states", return_value={4: state}), mock.patch.object(
            launcher,
            "_compute_process_snapshot",
            return_value=[process],
        ):
            self.assertIsNone(
                launcher._eligible_gpu_snapshot(
                    4,
                    utilization_threshold=10,
                    min_free_memory_mib=20_000,
                    allow_shared_low_utilization=False,
                )
            )
            eligible = launcher._eligible_gpu_snapshot(
                4,
                utilization_threshold=10,
                min_free_memory_mib=20_000,
                allow_shared_low_utilization=True,
            )
            self.assertEqual(eligible, (state, [process]))

        busy = launcher.GPUState(4, "GPU-test", 49_000, 4_000, 10)
        with mock.patch.object(launcher, "_gpu_states", return_value={4: busy}), mock.patch.object(
            launcher,
            "_compute_process_snapshot",
            return_value=[],
        ):
            self.assertIsNone(
                launcher._eligible_gpu_snapshot(
                    4,
                    utilization_threshold=10,
                    min_free_memory_mib=20_000,
                    allow_shared_low_utilization=True,
                )
            )


if __name__ == "__main__":
    unittest.main()

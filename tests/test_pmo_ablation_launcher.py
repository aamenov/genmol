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

    def test_completed_requires_matching_matrix_and_consistent_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix_path = (root / "matrix.yaml").resolve()
            matrix = self._matrix(root)
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
                "model_path": matrix["model_path"],
                "matrix_path": str(matrix_path),
                "matrix_sha256": matrix_hash,
            }
            run_id = "test_matrix:qed:released:seed3"
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

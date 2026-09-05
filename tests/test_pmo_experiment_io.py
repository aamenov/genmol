import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.exps.pmo.main.genmol import experiment_io


class HashAndManifestTests(unittest.TestCase):
    def test_file_and_config_hashes_are_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.ckpt"
            model_path.write_bytes(b"small fake checkpoint")

            expected = hashlib.sha256(b"small fake checkpoint").hexdigest()
            self.assertEqual(experiment_io.sha256_file(model_path, chunk_size=3), expected)
            self.assertEqual(
                experiment_io.sha256_config({"seed": 7, "nested": {"a": 1, "b": 2}}),
                experiment_io.sha256_config({"nested": {"b": 2, "a": 1}, "seed": 7}),
            )
            self.assertNotEqual(
                experiment_io.sha256_config({"seed": 7}),
                experiment_io.sha256_config({"seed": 8}),
            )

    def test_manifest_contains_model_and_config_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.ckpt"
            model_path.write_bytes(b"weights")
            config = {"population_size": 100, "seeds": [0, 1, 2]}
            manifest = experiment_io.build_manifest(
                run_id="qed-running-mean-seed0",
                model_path=model_path,
                config=config,
                task="qed",
                variant="running_mean",
                seed=0,
                oracle_budget=10_000,
                created_at="2026-09-05T00:00:00Z",
                extra={"gpu_uuids": ["GPU-test"]},
            )

            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["model"]["sha256"], experiment_io.sha256_file(model_path))
            self.assertEqual(manifest["config_sha256"], experiment_io.sha256_config(config))
            self.assertEqual(manifest["task"], "qed")
            self.assertEqual(manifest["seed"], 0)

            destination = root / "run" / "manifest.json"
            experiment_io.write_manifest(destination, manifest)
            self.assertEqual(json.loads(destination.read_text()), manifest)
            with self.assertRaises(FileExistsError):
                experiment_io.write_manifest(destination, manifest)


class EventLogTests(unittest.TestCase):
    def test_events_are_appended_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events" / "events.jsonl"
            log = experiment_io.JsonlEventLog(path, durable=True)
            log.append({"oracle_call": 1, "smiles": "CCO"})
            log.append({"oracle_call": 2, "smiles": "CCN", "score": 0.4})

            self.assertEqual(
                list(log.events()),
                [
                    {"oracle_call": 1, "smiles": "CCO"},
                    {"oracle_call": 2, "score": 0.4, "smiles": "CCN"},
                ],
            )
            self.assertEqual(len(path.read_text().splitlines()), 2)

    def test_bad_event_does_not_change_existing_log(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            experiment_io.append_event(path, {"oracle_call": 1})
            before = path.read_bytes()
            with self.assertRaises(TypeError):
                experiment_io.append_event(path, {"bad": object()})
            self.assertEqual(path.read_bytes(), before)

    def test_truncated_final_line_can_be_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_bytes(b'{"oracle_call":1}\n{"oracle_call":')
            with self.assertRaises(ValueError):
                list(experiment_io.iter_events(path))
            self.assertEqual(
                list(experiment_io.iter_events(path, tolerate_truncated_last_line=True)),
                [{"oracle_call": 1}],
            )


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_round_trips_rng_and_oracle_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state" / "latest.pkl"
            random.seed(123)
            state = {
                "iteration": 91,
                "python_rng_state": random.getstate(),
                "oracle_buffer": {"CCO": [0.5, 1], "CCN": [0.7, 2]},
                "population": [(0.7, "[1*]CN")],
            }
            experiment_io.save_checkpoint(path, state, metadata={"config_sha256": "abc"})

            loaded, metadata = experiment_io.load_checkpoint(path, with_metadata=True)
            self.assertEqual(loaded, state)
            self.assertEqual(metadata, {"config_sha256": "abc"})

    def test_failed_atomic_replace_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pkl"
            experiment_io.save_checkpoint(path, {"iteration": 1})

            with mock.patch.object(experiment_io.os, "replace", side_effect=OSError("stop")):
                with self.assertRaises(OSError):
                    experiment_io.save_checkpoint(path, {"iteration": 2})

            self.assertEqual(experiment_io.load_checkpoint(path), {"iteration": 1})
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])


class MetricTests(unittest.TestCase):
    def test_released_constant_curve_auc_convention(self):
        scores = [0.5] * 100
        self.assertAlmostEqual(
            experiment_io.top_k_auc(scores, k=10, budget=10_000),
            0.4975,
        )

    def test_trajectory_keeps_partial_interval_and_pads(self):
        scores = [0.1, 0.9, 0.2, 0.8]
        trajectory = experiment_io.top_k_trajectory(
            scores,
            k=2,
            reporting_frequency=2,
            budget=6,
            pad_to_budget=True,
        )
        self.assertEqual(
            trajectory,
            [
                {"oracle_calls": 0, "top_k_mean": 0.0},
                {"oracle_calls": 2, "top_k_mean": 0.5},
                {"oracle_calls": 4, "top_k_mean": 0.8500000000000001},
                {"oracle_calls": 6, "top_k_mean": 0.8500000000000001},
            ],
        )
        self.assertAlmostEqual(
            experiment_io.trajectory_auc(trajectory, normalize_by=6),
            3.55 / 6,
        )

    def test_summary_is_json_ready_and_budget_limited(self):
        summary = experiment_io.summarize_scores(
            [0.1, 0.4, 0.2, 0.9],
            ks=(1, 2),
            reporting_frequency=2,
            budget=3,
        )
        self.assertEqual(summary["oracle_calls"], 3)
        self.assertEqual(summary["top_1"], 0.4)
        self.assertEqual(summary["top_2"], 0.30000000000000004)
        json.dumps(summary, allow_nan=False)

    def test_indexed_child_trajectory_uses_global_call_positions(self):
        rows = [(2, 0.5), (4, 0.9)]
        trajectory = experiment_io.top_k_trajectory_at_calls(
            rows,
            k=1,
            reporting_frequency=2,
            observed_oracle_calls=4,
            budget=4,
        )
        self.assertEqual(
            trajectory,
            [
                {"oracle_calls": 0, "top_k_mean": 0.0},
                {"oracle_calls": 2, "top_k_mean": 0.5},
                {"oracle_calls": 4, "top_k_mean": 0.9},
            ],
        )
        summary = experiment_io.summarize_indexed_scores(
            rows,
            ks=(1,),
            reporting_frequency=2,
            observed_oracle_calls=4,
            budget=4,
        )
        self.assertEqual(summary["score_count"], 2)
        self.assertEqual(summary["oracle_calls"], 4)
        self.assertAlmostEqual(summary["auc_top_1"], 0.475)

    def test_indexed_scores_reject_invalid_global_positions(self):
        with self.assertRaises(ValueError):
            experiment_io.top_k_trajectory_at_calls(
                [(2, 0.4), (1, 0.5)],
                observed_oracle_calls=2,
                budget=2,
            )
        with self.assertRaises(ValueError):
            experiment_io.top_k_trajectory_at_calls(
                [(3, 0.4)],
                observed_oracle_calls=2,
                budget=3,
            )

    def test_invalid_metric_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            experiment_io.top_k_mean([], 10)
        with self.assertRaises(ValueError):
            experiment_io.top_k_auc([0.5], budget=0)
        with self.assertRaises(ValueError):
            experiment_io.top_k_trajectory([float("nan")])


if __name__ == "__main__":
    unittest.main()

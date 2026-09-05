import json
import tempfile
import unittest
from pathlib import Path

from scripts.exps.pmo.main.genmol.experiment_io import sha256_config, sha256_file
from scripts.exps.pmo.simulate_fragment_updates import (
    SimulationConfig,
    _one_replicate,
    run_simulation,
)


class FragmentUpdateSimulationTests(unittest.TestCase):
    def test_records_zero_baseline_and_exact_statistical_endpoint(self):
        config = SimulationConfig(
            fragments=6,
            capacity=2,
            observations_per_fragment=3,
            replicates=1,
            context_stds=(0.0,),
            checkpoint_every=2,
            seed=17,
        )
        rows = _one_replicate(config=config, context_std=0.0, replicate=0)

        for variant in ("released", "running_mean", "support3", "shrink10"):
            variant_rows = [row for row in rows if row["variant"] == variant]
            self.assertEqual(variant_rows[0]["observations"], 0)
            self.assertEqual(variant_rows[-1]["observations"], 16)

        for variant in ("running_mean", "support3", "shrink10"):
            final = [row for row in rows if row["variant"] == variant][-1]
            self.assertAlmostEqual(final["regret"], 0.0)
            self.assertAlmostEqual(final["top_v_jaccard"], 1.0)

    def test_atomic_publication_is_hashed_and_refuses_overwrite(self):
        config = SimulationConfig(
            fragments=6,
            capacity=2,
            observations_per_fragment=3,
            replicates=1,
            context_stds=(0.2,),
            checkpoint_every=2,
            seed=19,
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "result"
            summary, published = run_simulation(config, destination)

            self.assertEqual(published, destination.resolve())
            self.assertIn("context_scale", summary)
            self.assertIn("achieved_pooled_within_fragment_sample_sd", summary)
            manifest = json.loads((destination / "manifest.json").read_text())
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["config_sha256"], sha256_config(manifest["config"]))
            for output in manifest["outputs"].values():
                path = destination / output["path"]
                self.assertEqual(output["sha256"], sha256_file(path))
                self.assertEqual(output["size_bytes"], path.stat().st_size)

            before = {
                path.name: sha256_file(path)
                for path in destination.iterdir()
                if path.is_file()
            }
            with self.assertRaises(FileExistsError):
                run_simulation(config, destination)
            self.assertEqual(
                before,
                {
                    path.name: sha256_file(path)
                    for path in destination.iterdir()
                    if path.is_file()
                },
            )


if __name__ == "__main__":
    unittest.main()

"""Focused stdlib tests for GenMol PMO fragment population policies."""

from __future__ import annotations

import copy
import csv
import math
import random
import tempfile
import unittest
from pathlib import Path

from scripts.exps.pmo.main.genmol.fragment_population import (
    FragmentObservation,
    FragmentPopulation,
    FragmentSeed,
)


class MappingFragmenter:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def __call__(self, smiles):
        self.calls.append(smiles)
        return self.values.get(smiles, ())


class ReleasedPopulationTests(unittest.TestCase):
    def test_released_gate_first_score_and_evicted_reentry(self):
        fragmenter = MappingFragmenter(
            {
                "low": ("unused",),
                "high": ("B", "C", "C"),
                "reentry": ("B",),
            }
        )
        population = FragmentPopulation(
            [FragmentSeed("A", 0.9), FragmentSeed("B", 0.5)],
            capacity=2,
            mode="released",
            fragmenter=fragmenter,
            rng=random.Random(7),
        )

        result = population.observe(FragmentObservation("1", "low", 0.4))
        self.assertEqual(result.reason, "score_below_cutoff")
        self.assertEqual(fragmenter.calls, [])

        result = population.observe(FragmentObservation("2", "high", 0.8))
        self.assertEqual(population.active_rows(), [(0.9, "A"), (0.8, "C")])
        self.assertEqual(result.admitted, ("C",))
        self.assertEqual(result.displaced, ("B",))

        population.observe(FragmentObservation("3", "reentry", 0.85))
        self.assertEqual(population.active_rows(), [(0.9, "A"), (0.85, "B")])

    def test_released_ties_match_reverse_tuple_sort(self):
        fragmenter = MappingFragmenter({"x": ("C", "D")})
        population = FragmentPopulation(
            [FragmentSeed("A", 0.9), FragmentSeed("B", 0.5)],
            capacity=2,
            mode="released",
            fragmenter=fragmenter,
        )
        population.observe(FragmentObservation("1", "x", 0.8))
        self.assertEqual(population.active_rows(), [(0.9, "A"), (0.8, "D")])

    def test_injected_rng_controls_uniform_sampling(self):
        expected_rng = random.Random(19)
        expected = expected_rng.sample(["A", "B", "C"], 2)
        population = FragmentPopulation(
            [FragmentSeed("A", 0.9), FragmentSeed("B", 0.8), FragmentSeed("C", 0.7)],
            capacity=3,
            fragmenter=MappingFragmenter({}),
            rng=random.Random(19),
        )
        self.assertEqual(population.sample(2), expected)

    def test_sampling_canonicalizes_cross_policy_tie_order(self):
        # The released constructor preserves CSV order, while statistical
        # ranking normalizes equal-score rows to the released boundary rule.
        seeds = [FragmentSeed("A", 0.8), FragmentSeed("B", 0.8), FragmentSeed("C", 0.7)]
        released = FragmentPopulation(
            seeds,
            capacity=3,
            mode="released",
            fragmenter=MappingFragmenter({}),
            rng=random.Random(23),
        )
        statistical = FragmentPopulation(
            seeds,
            capacity=3,
            mode="mean",
            fragmenter=MappingFragmenter({}),
            rng=random.Random(23),
            legacy_seed_count=1,
        )

        self.assertNotEqual(released.active_fragments, statistical.active_fragments)
        self.assertEqual(released.sample(2), statistical.sample(2))


class StatisticalPopulationTests(unittest.TestCase):
    def test_direct_construction_limits_seed_registry_to_capacity(self):
        population = FragmentPopulation(
            [
                FragmentSeed("first", 0.9),
                FragmentSeed("second", 0.8),
                FragmentSeed("hidden", 1.0),
            ],
            capacity=2,
            mode="mean",
            fragmenter=MappingFragmenter({}),
            legacy_seed_count=1,
        )

        self.assertEqual(set(population.active_fragments), {"first", "second"})
        self.assertIsNone(population.get_stats("hidden"))

    def test_running_mean_updates_every_observation_and_deduplicates(self):
        fragmenter = MappingFragmenter(
            {"one": ("f", "f"), "two": ("f",), "three": ("f",)}
        )
        population = FragmentPopulation(
            [], capacity=2, mode="mean", fragmenter=fragmenter
        )
        population.observe(FragmentObservation("1", "one", 0.95))
        population.observe(FragmentObservation("2", "two", 0.10))
        population.observe(FragmentObservation("3", "three", 0.20))
        duplicate = population.observe(FragmentObservation("3", "three", 1.0))

        stats = population.get_stats("f")
        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertEqual(stats.count, 3)
        self.assertAlmostEqual(stats.total / stats.count, 1.25 / 3)
        self.assertEqual(duplicate.reason, "duplicate_observation")
        self.assertEqual(population.active_rows(), [(stats.total / stats.count, "f")])

    def test_running_mean_can_credit_an_explicit_matched_fragment_set(self):
        fragmenter = MappingFragmenter({"child": ("unmatched",)})
        population = FragmentPopulation(
            [], capacity=2, mode="mean", fragmenter=fragmenter
        )

        result = population.observe(
            FragmentObservation(
                "parent-to-child",
                "child",
                0.75,
                credit_fragments=frozenset({"changed"}),
            )
        )

        self.assertEqual(result.observed_fragments, ("changed",))
        self.assertEqual(fragmenter.calls, [])
        self.assertEqual(population.get_stats("changed").total, 0.75)
        self.assertIsNone(population.get_stats("unmatched"))

    def test_bayesian_shrinkage_arithmetic(self):
        fragmenter = MappingFragmenter({"x": ("f",), "y": ("f",)})
        population = FragmentPopulation(
            [],
            capacity=1,
            mode="bayes",
            fragmenter=fragmenter,
            prior_strength=2,
            prior_mean=0.25,
        )
        population.observe(FragmentObservation("1", "x", 0.4))
        population.observe(FragmentObservation("2", "y", 0.6))
        self.assertEqual(population.active_rows(), [(0.375, "f")])

    def test_min_support_delays_new_admission_but_grandfathers_seed(self):
        fragmenter = MappingFragmenter({"one": ("new",), "two": ("new",)})
        population = FragmentPopulation(
            [FragmentSeed("seed", 0.4, count=1)],
            capacity=2,
            mode="mean",
            min_support=2,
            fragmenter=fragmenter,
        )
        first = population.observe(FragmentObservation("1", "one", 0.9))
        self.assertEqual(population.active_fragments, ["seed"])
        self.assertEqual(first.admitted, ())

        second = population.observe(FragmentObservation("2", "two", 0.7))
        self.assertEqual(population.active_rows(), [(0.8, "new"), (0.4, "seed")])
        self.assertEqual(second.admitted, ("new",))

    def test_delta_credits_only_requested_fragments_and_keeps_negative_values(self):
        population = FragmentPopulation(
            [FragmentSeed("seed", 0.8)],
            capacity=2,
            mode="delta",
            fragmenter=MappingFragmenter({}),
        )
        population.observe(
            FragmentObservation("1", "child", 0.75, 0.5, frozenset({"changed"}))
        )
        population.observe(
            FragmentObservation("2", "child2", 0.25, 0.5, frozenset({"changed"}))
        )

        stats = population.get_stats("changed")
        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertEqual(stats.count, 2)
        self.assertAlmostEqual(stats.total, 0.0)
        # Both ranks are neutral, so seed quality is the delta-mode tie-break.
        self.assertEqual(population.active_fragments, ["seed", "changed"])
        self.assertIsNone(population.get_stats("uncredited"))

    def test_delta_missing_parent_can_skip_or_raise(self):
        observation = FragmentObservation("1", "child", 0.8)
        skip = FragmentPopulation(
            [], capacity=1, mode="delta", fragmenter=MappingFragmenter({"child": ("f",)})
        )
        self.assertEqual(skip.observe(observation).reason, "missing_parent")

        strict = FragmentPopulation(
            [],
            capacity=1,
            mode="delta",
            fragmenter=MappingFragmenter({"child": ("f",)}),
            delta_missing_parent="raise",
        )
        with self.assertRaisesRegex(ValueError, "parent_score"):
            strict.observe(observation)

    def test_statistical_ties_match_released_fragment_descending_rule(self):
        fragmenter = MappingFragmenter({"x": ("z", "a")})
        population = FragmentPopulation(
            [], capacity=2, mode="mean", fragmenter=fragmenter
        )
        population.observe(FragmentObservation("1", "x", 0.5))
        self.assertEqual(population.active_fragments, ["z", "a"])

    def test_cross_policy_boundary_ties_retain_the_same_fragments(self):
        fragmenter = MappingFragmenter({"child": ("a", "b", "c")})
        seeds = [FragmentSeed("seed-1", 0.2), FragmentSeed("seed-2", 0.1)]
        released = FragmentPopulation(
            seeds,
            capacity=2,
            mode="released",
            fragmenter=fragmenter,
        )
        statistical = FragmentPopulation(
            seeds,
            capacity=2,
            mode="mean",
            fragmenter=fragmenter,
            legacy_seed_count=1,
        )
        observation = FragmentObservation("child", "child", 0.9)

        released.observe(observation)
        statistical.observe(observation)

        self.assertEqual(set(released.active_fragments), {"b", "c"})
        self.assertEqual(set(statistical.active_fragments), {"b", "c"})


class PersistenceAndLegacyTests(unittest.TestCase):
    def test_versioned_state_roundtrip_restores_rng_and_statistics(self):
        fragmenter = MappingFragmenter({"x": ("f",)})
        population = FragmentPopulation(
            [FragmentSeed("seed", 0.2, count=2)],
            capacity=2,
            mode="mean",
            fragmenter=fragmenter,
            rng=random.Random(31),
        )
        population.observe(FragmentObservation("1", "x", 0.7))
        population.sample(1)
        state = population.state_dict()
        self.assertEqual(state["version"], FragmentPopulation.STATE_VERSION)

        restored = FragmentPopulation.from_state_dict(
            copy.deepcopy(state), fragmenter=fragmenter, rng=random.Random(999)
        )
        self.assertEqual(restored.active_rows(), population.active_rows())
        self.assertEqual(restored.get_stats("f"), population.get_stats("f"))
        self.assertEqual(restored.sample(2), population.sample(2))

    def test_unknown_state_version_is_rejected(self):
        population = FragmentPopulation(
            [FragmentSeed("seed", 0.5)],
            capacity=1,
            fragmenter=MappingFragmenter({}),
        )
        state = population.state_dict()
        state["version"] = 999
        with self.assertRaisesRegex(ValueError, "state version"):
            population.load_state_dict(state)

    def test_legacy_csv_requires_explicit_statistical_pseudocount(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vocab.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["frag", "score", "size"])
                writer.writeheader()
                writer.writerow({"frag": "f", "score": "0.75", "size": "3"})

            released = FragmentPopulation.from_csv(
                path,
                capacity=1,
                mode="released",
                fragmenter=MappingFragmenter({}),
            )
            self.assertEqual(released.active_rows(), [(0.75, "f")])

            with self.assertRaisesRegex(ValueError, "legacy_seed_count"):
                FragmentPopulation.from_csv(
                    path,
                    capacity=1,
                    mode="mean",
                    fragmenter=MappingFragmenter({}),
                )

            approximate = FragmentPopulation.from_csv(
                path,
                capacity=1,
                mode="mean",
                fragmenter=MappingFragmenter({}),
                legacy_seed_count=4,
            )
            stats = approximate.get_stats("f")
            self.assertIsNotNone(stats)
            assert stats is not None
            self.assertEqual(stats.count, 4)
            self.assertEqual(stats.total, 3.0)
            self.assertEqual(approximate.active_rows(), [(0.75, "f")])

    def test_enriched_csv_loads_score_sum_without_pseudocount(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vocab.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=["frag", "score", "count", "score_sum", "size"]
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "frag": "f",
                        "score": "0.5",
                        "count": "3",
                        "score_sum": "1.5",
                        "size": "3",
                    }
                )

            population = FragmentPopulation.from_csv(
                path,
                capacity=1,
                mode="mean",
                fragmenter=MappingFragmenter({}),
            )
            stats = population.get_stats("f")
            self.assertIsNotNone(stats)
            assert stats is not None
            self.assertEqual(stats.count, 3)
            self.assertEqual(stats.total, 1.5)
            self.assertEqual(population.active_rows(), [(0.5, "f")])

    def test_csv_loading_limits_statistical_registry_to_initial_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vocab.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=["frag", "score", "count", "score_sum"]
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"frag": "first", "score": "0.9", "count": "2", "score_sum": "1.8"},
                        {"frag": "second", "score": "0.8", "count": "2", "score_sum": "1.6"},
                        {"frag": "hidden", "score": "1.0", "count": "2", "score_sum": "2.0"},
                    ]
                )

            population = FragmentPopulation.from_csv(
                path,
                capacity=2,
                mode="mean",
                fragmenter=MappingFragmenter({}),
            )

            self.assertEqual(population.active_fragments, ["first", "second"])
            self.assertIsNone(population.get_stats("hidden"))

    def test_enriched_csv_rejects_inconsistent_sufficient_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vocab.csv"
            path.write_text("frag,score,count,score_sum\nf,0.8,2,1.0\n")
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                FragmentPopulation.from_csv(
                    path,
                    capacity=1,
                    mode="mean",
                    fragmenter=MappingFragmenter({}),
                )

    def test_nonfinite_scores_are_rejected(self):
        population = FragmentPopulation(
            [], capacity=1, mode="mean", fragmenter=MappingFragmenter({"x": ("f",)})
        )
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                population.observe(FragmentObservation(str(value), "x", value))


if __name__ == "__main__":
    unittest.main()

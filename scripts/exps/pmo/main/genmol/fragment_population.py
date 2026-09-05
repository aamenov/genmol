"""Fragment scoring and active-population policies for GenMol PMO.

The released optimizer stores only ``(score, fragment)`` pairs and assigns a
new fragment the score of the first sufficiently good molecule in which it is
observed.  This module preserves that behavior under ``mode="released"`` and
also provides evidence-retaining estimators for controlled ablations.

Chemistry is intentionally outside this module.  A fragmenter is injected so
that population mechanics are deterministic and independently testable.  In
released mode fragmentation is lazy: a molecule below the current cutoff does
not consume fragmenter randomness, matching the released implementation.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, FrozenSet, Iterable, Mapping, Optional, Protocol, Sequence


Fragmenter = Callable[[str], Iterable[str]]


class SamplingRNG(Protocol):
    """Small portion of ``random.Random`` used by the population."""

    def sample(self, population: Sequence[str], k: int) -> list[str]:
        """Sample ``k`` distinct members from ``population``."""


@dataclass(frozen=True)
class FragmentSeed:
    """One initial vocabulary row.

    ``count`` and ``total`` are sufficient statistics for statistical modes.
    Historical GenMol CSVs contain only ``score``.  Such rows require an
    explicit ``legacy_seed_count`` in running-mean and Bayesian modes; the
    implied total is then ``score * legacy_seed_count``.
    """

    fragment: str
    score: float
    count: Optional[int] = None
    total: Optional[float] = None


@dataclass(frozen=True)
class FragmentObservation:
    """One scored child molecule presented to the vocabulary.

    Delta mode uses ``child_score - parent_score``.  If
    ``credit_fragments`` is supplied, only those fragments receive the delta;
    otherwise all fragments returned by the injected child fragmenter do.
    """

    observation_id: str
    child_smiles: str
    child_score: float
    parent_score: Optional[float] = None
    credit_fragments: Optional[FrozenSet[str]] = None


@dataclass
class FragmentStats:
    """Sufficient statistics retained by non-released policies."""

    fragment: str
    total: float = 0.0
    count: int = 0
    seed_score: Optional[float] = None
    seed_order: Optional[int] = None
    first_seen: int = 0
    last_seen: int = 0


@dataclass(frozen=True)
class UpdateResult:
    """Compact, inspectable outcome of a population observation."""

    updated: bool
    reason: str
    observed_fragments: tuple[str, ...] = ()
    admitted: tuple[str, ...] = ()
    displaced: tuple[str, ...] = ()


class FragmentPopulation:
    """Maintain the fragment registry and top-V sampling population.

    Args:
        seeds: Initial vocabulary rows.  Input order is preserved by released
            mode and is retained as the bootstrap order for delta mode.
        capacity: Maximum number of active fragments.
        mode: ``released``, ``mean``, ``bayes``, or ``delta``.
        fragmenter: Callable mapping a SMILES string to fragment strings.
        rng: Object supporting ``sample``.  The default is Python's global
            ``random`` module, preserving the released optimizer's seeding.
        min_support: Number of observations required before a newly discovered
            statistical fragment can enter the active population.  Initial
            seed rows are grandfathered so sampling remains possible.
        prior_strength: Bayesian pseudo-count lambda.  Used only in ``bayes``.
        prior_mean: Frozen Bayesian prior mean mu.  Required in ``bayes``.
        legacy_seed_count: Explicit pseudo-count for historical seed rows that
            lack counts.  Required for those rows in ``mean``/``bayes``.
        deduplicate_observations: Whether an observation ID can update at most
            once.  Defaults to false for exact released behavior and true for
            statistical modes.
        delta_missing_parent: ``skip`` or ``raise`` when delta mode receives no
            parent score.
    """

    STATE_VERSION = 1
    MODES = frozenset({"released", "mean", "bayes", "delta"})
    MISSING_PARENT_POLICIES = frozenset({"skip", "raise"})

    def __init__(
        self,
        seeds: Iterable[FragmentSeed],
        *,
        capacity: int,
        mode: str = "released",
        fragmenter: Fragmenter,
        rng: Optional[SamplingRNG] = None,
        min_support: int = 1,
        prior_strength: float = 0.0,
        prior_mean: Optional[float] = None,
        legacy_seed_count: Optional[int] = None,
        deduplicate_observations: Optional[bool] = None,
        delta_missing_parent: str = "skip",
    ) -> None:
        self._validate_configuration(
            capacity=capacity,
            mode=mode,
            min_support=min_support,
            prior_strength=prior_strength,
            prior_mean=prior_mean,
            legacy_seed_count=legacy_seed_count,
            delta_missing_parent=delta_missing_parent,
        )
        self.capacity = capacity
        self.mode = mode
        self.fragmenter = fragmenter
        self.rng: SamplingRNG = rng if rng is not None else random
        self.min_support = min_support
        self.prior_strength = float(prior_strength)
        self.prior_mean = None if prior_mean is None else float(prior_mean)
        self.legacy_seed_count = legacy_seed_count
        self.deduplicate_observations = (
            mode != "released" if deduplicate_observations is None else deduplicate_observations
        )
        self.delta_missing_parent = delta_missing_parent

        self._event_index = 0
        self._seen_observation_ids: set[str] = set()
        self._released_population: list[tuple[float, str]] = []
        self._records: dict[str, FragmentStats] = {}
        self._active_cache: Optional[list[FragmentStats]] = None

        seed_rows = list(seeds)
        self._initialize(seed_rows)

    @staticmethod
    def _validate_configuration(
        *,
        capacity: int,
        mode: str,
        min_support: int,
        prior_strength: float,
        prior_mean: Optional[float],
        legacy_seed_count: Optional[int],
        delta_missing_parent: str,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        if mode not in FragmentPopulation.MODES:
            raise ValueError(f"unsupported mode {mode!r}; expected one of {sorted(FragmentPopulation.MODES)}")
        if min_support < 1:
            raise ValueError("min_support must be at least 1")
        if legacy_seed_count is not None and legacy_seed_count < 1:
            raise ValueError("legacy_seed_count must be positive when provided")
        if delta_missing_parent not in FragmentPopulation.MISSING_PARENT_POLICIES:
            raise ValueError("delta_missing_parent must be 'skip' or 'raise'")
        if not math.isfinite(prior_strength) or prior_strength < 0:
            raise ValueError("prior_strength must be finite and nonnegative")
        if mode == "bayes":
            if prior_strength <= 0:
                raise ValueError("bayes mode requires prior_strength > 0")
            if prior_mean is None or not math.isfinite(prior_mean):
                raise ValueError("bayes mode requires a finite prior_mean")
        elif prior_strength != 0.0 or prior_mean is not None:
            raise ValueError("prior_strength and prior_mean are only valid in bayes mode")
        if mode == "released" and min_support != 1:
            raise ValueError("min_support is not defined for released mode")

    @staticmethod
    def _validate_score(score: float, label: str = "score") -> float:
        value = float(score)
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite")
        return value

    @staticmethod
    def _unique_fragments(fragments: Iterable[str]) -> tuple[str, ...]:
        clean = {fragment for fragment in fragments if isinstance(fragment, str) and fragment}
        return tuple(sorted(clean))

    def _initialize(self, seeds: list[FragmentSeed]) -> None:
        seen: set[str] = set()
        for seed in seeds:
            if not seed.fragment:
                raise ValueError("seed fragment cannot be empty")
            if seed.fragment in seen:
                raise ValueError(f"duplicate seed fragment {seed.fragment!r}")
            seen.add(seed.fragment)
            self._validate_score(seed.score, "seed score")

        if self.mode == "released":
            self._released_population = [
                (float(seed.score), seed.fragment) for seed in seeds[: self.capacity]
            ]
            return

        # Every policy starts from exactly the caller-provided first V rows.
        # Retaining later seed rows as a hidden registry would let statistical
        # arms backfill from fragments unavailable to the released arm.
        for order, seed in enumerate(seeds[: self.capacity]):
            if self.mode == "delta":
                # Offline absolute scores bootstrap deterministic ties but are
                # not mixed into the delta estimand.
                total = 0.0
                count = 0
            else:
                count = seed.count
                if count is None:
                    if self.legacy_seed_count is None:
                        raise ValueError(
                            "statistical modes require seed counts; pass legacy_seed_count "
                            "explicitly to treat a legacy score as pseudo-observations"
                        )
                    count = self.legacy_seed_count
                if count < 1:
                    raise ValueError("seed count must be positive")
                total = float(seed.total) if seed.total is not None else float(seed.score) * count
                self._validate_score(total, "seed total")

            self._records[seed.fragment] = FragmentStats(
                fragment=seed.fragment,
                total=total,
                count=count,
                seed_score=float(seed.score),
                seed_order=order,
            )

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        capacity: int,
        mode: str = "released",
        fragmenter: Fragmenter,
        rng: Optional[SamplingRNG] = None,
        min_support: int = 1,
        prior_strength: float = 0.0,
        prior_mean: Optional[float] = None,
        legacy_seed_count: Optional[int] = None,
        deduplicate_observations: Optional[bool] = None,
        delta_missing_parent: str = "skip",
    ) -> "FragmentPopulation":
        """Load released or enriched vocabulary CSV rows.

        Supported columns are ``frag``, ``score``, optional ``count``, and
        optional ``score_sum`` (with ``sum`` and ``total`` accepted as legacy
        aliases).  Missing counts are never guessed.  Only the first
        ``capacity`` rows are loaded so every policy starts from the same
        released top-V population rather than giving statistical policies a
        hidden reservoir of inactive vocabulary rows.
        """

        seeds: list[FragmentSeed] = []
        with Path(path).open(newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"frag", "score"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError("vocabulary CSV must contain 'frag' and 'score' columns")
            for row_index, row in enumerate(reader):
                if row_index >= capacity:
                    break
                raw_count = row.get("count")
                count = int(raw_count) if raw_count not in (None, "") else None
                raw_total = row.get("score_sum")
                if raw_total in (None, ""):
                    raw_total = row.get("sum")
                if raw_total in (None, ""):
                    raw_total = row.get("total")
                total = float(raw_total) if raw_total not in (None, "") else None
                score = float(row["score"])
                if total is not None and count is None:
                    raise ValueError("vocabulary score_sum requires a matching count")
                if count is not None and total is not None:
                    if count <= 0:
                        raise ValueError("vocabulary count must be positive")
                    if not math.isclose(score, total / count, rel_tol=1e-8, abs_tol=1e-10):
                        raise ValueError(
                            f"inconsistent score/count/score_sum for fragment {row['frag']!r}"
                        )
                seeds.append(
                    FragmentSeed(
                        fragment=row["frag"],
                        score=score,
                        count=count,
                        total=total,
                    )
                )
        return cls(
            seeds,
            capacity=capacity,
            mode=mode,
            fragmenter=fragmenter,
            rng=rng,
            min_support=min_support,
            prior_strength=prior_strength,
            prior_mean=prior_mean,
            legacy_seed_count=legacy_seed_count,
            deduplicate_observations=deduplicate_observations,
            delta_missing_parent=delta_missing_parent,
        )

    def _rank(self, record: FragmentStats) -> float:
        if self.mode == "mean":
            return record.total / record.count
        if self.mode == "bayes":
            assert self.prior_mean is not None
            return (record.total + self.prior_strength * self.prior_mean) / (
                record.count + self.prior_strength
            )
        if self.mode == "delta":
            return record.total / record.count if record.count else 0.0
        raise RuntimeError("released populations do not rank FragmentStats")

    def _statistical_sort_key(self, record: FragmentStats) -> tuple[Any, ...]:
        rank = self._rank(record)
        if self.mode == "delta":
            # Absolute seed quality is only a tie-break for neutral delta
            # estimates; it is never added to the delta value.
            seed_score = record.seed_score if record.seed_score is not None else -math.inf
            return (-rank, -seed_score)
        return (-rank,)

    def _active_statistical_records(self) -> list[FragmentStats]:
        if self._active_cache is not None:
            return self._active_cache
        eligible = [
            record
            for record in self._records.values()
            if record.seed_score is not None or record.count >= self.min_support
        ]
        # Released GenMol sorts ``(score, fragment)`` tuples in reverse order.
        # A stable primary-score sort after this descending fragment sort gives
        # every policy the same boundary tie-break without changing its score.
        eligible.sort(key=lambda record: record.fragment, reverse=True)
        self._active_cache = sorted(eligible, key=self._statistical_sort_key)[: self.capacity]
        return self._active_cache

    def active_rows(self) -> list[tuple[float, str]]:
        """Return active ``(ranking score, fragment)`` rows in ranking order."""

        if self.mode == "released":
            return list(self._released_population)
        return [(self._rank(record), record.fragment) for record in self._active_statistical_records()]

    @property
    def active_fragments(self) -> list[str]:
        """Return the unique fragments currently available for sampling."""

        return [fragment for _, fragment in self.active_rows()]

    def get_stats(self, fragment: str) -> Optional[FragmentStats]:
        """Return a defensive copy of retained statistical state."""

        record = self._records.get(fragment)
        return None if record is None else replace(record)

    def sample(self, k: int = 2) -> list[str]:
        """Uniformly sample distinct active fragments using the injected RNG.

        Canonicalizing the population order before mapping RNG indices to
        fragments keeps paired ablation arms on the same proposal stream for
        as long as their active fragment *sets* agree.  It does not alter the
        uniform sampling distribution.
        """

        if k < 0:
            raise ValueError("k must be nonnegative")
        fragments = sorted(self.active_fragments)
        if k > len(fragments):
            raise ValueError(f"cannot sample {k} fragments from active population of size {len(fragments)}")
        return list(self.rng.sample(fragments, k))

    def observe(self, observation: FragmentObservation) -> UpdateResult:
        """Update population state from one molecule observation."""

        child_score = self._validate_score(observation.child_score, "child score")
        if self.deduplicate_observations and observation.observation_id in self._seen_observation_ids:
            return UpdateResult(False, "duplicate_observation")

        if self.mode == "released":
            return self._observe_released(observation, child_score)
        return self._observe_statistical(observation, child_score)

    def _observe_released(
        self, observation: FragmentObservation, child_score: float
    ) -> UpdateResult:
        if not self._released_population:
            raise RuntimeError("released mode requires at least one active seed")
        if child_score <= self._released_population[-1][0]:
            return UpdateResult(False, "score_below_cutoff")

        fragments = self._unique_fragments(self.fragmenter(observation.child_smiles))
        if self.deduplicate_observations:
            self._seen_observation_ids.add(observation.observation_id)
        if not fragments:
            return UpdateResult(False, "no_fragments")

        before = set(self.active_fragments)
        active_before_update = set(before)
        self._released_population.extend(
            (child_score, fragment)
            for fragment in fragments
            if fragment not in active_before_update
        )
        self._released_population.sort(reverse=True)
        self._released_population = self._released_population[: self.capacity]
        after = set(self.active_fragments)
        admitted = tuple(sorted(after - before))
        displaced = tuple(sorted(before - after))
        changed = before != after
        return UpdateResult(changed, "updated" if changed else "no_new_fragments", fragments, admitted, displaced)

    def _observe_statistical(
        self, observation: FragmentObservation, child_score: float
    ) -> UpdateResult:
        credit = child_score
        if self.mode == "delta":
            if observation.parent_score is None:
                if self.delta_missing_parent == "raise":
                    raise ValueError("delta mode requires parent_score")
                return UpdateResult(False, "missing_parent")
            parent_score = self._validate_score(observation.parent_score, "parent score")
            credit = self._validate_score(child_score - parent_score, "delta credit")

        source_fragments: Iterable[str]
        if self.mode == "delta" and observation.credit_fragments is not None:
            source_fragments = observation.credit_fragments
        else:
            source_fragments = self.fragmenter(observation.child_smiles)
        fragments = self._unique_fragments(source_fragments)

        # Mark an empty but successfully processed observation as seen so a
        # stochastic fragmenter cannot manufacture support from cached repeats.
        if self.deduplicate_observations:
            self._seen_observation_ids.add(observation.observation_id)
        if not fragments:
            return UpdateResult(False, "no_fragments")

        before = set(self.active_fragments)
        self._event_index += 1
        for fragment in fragments:
            record = self._records.get(fragment)
            if record is None:
                record = FragmentStats(
                    fragment=fragment,
                    first_seen=self._event_index,
                )
                self._records[fragment] = record
            if record.first_seen == 0:
                record.first_seen = self._event_index
            record.last_seen = self._event_index
            record.total += credit
            record.count += 1

        self._active_cache = None
        after = set(self.active_fragments)
        admitted = tuple(sorted(after - before))
        displaced = tuple(sorted(before - after))
        return UpdateResult(True, "updated", fragments, admitted, displaced)

    def state_dict(self) -> dict[str, Any]:
        """Return a versioned, pickle/YAML-friendly state snapshot."""

        getstate = getattr(self.rng, "getstate", None)
        rng_state = getstate() if callable(getstate) else None
        return {
            "version": self.STATE_VERSION,
            "config": {
                "capacity": self.capacity,
                "mode": self.mode,
                "min_support": self.min_support,
                "prior_strength": self.prior_strength,
                "prior_mean": self.prior_mean,
                "legacy_seed_count": self.legacy_seed_count,
                "deduplicate_observations": self.deduplicate_observations,
                "delta_missing_parent": self.delta_missing_parent,
            },
            "event_index": self._event_index,
            "seen_observation_ids": sorted(self._seen_observation_ids),
            "released_population": [
                {"score": score, "fragment": fragment}
                for score, fragment in self._released_population
            ],
            "records": [asdict(self._records[key]) for key in sorted(self._records)],
            "rng_state": rng_state,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Replace policy state while preserving injected dependencies."""

        version = state.get("version")
        if version != self.STATE_VERSION:
            raise ValueError(f"unsupported fragment-population state version {version!r}")
        config = state.get("config")
        if not isinstance(config, Mapping):
            raise ValueError("state is missing a configuration mapping")

        self._validate_configuration(
            capacity=int(config["capacity"]),
            mode=str(config["mode"]),
            min_support=int(config["min_support"]),
            prior_strength=float(config["prior_strength"]),
            prior_mean=config.get("prior_mean"),
            legacy_seed_count=config.get("legacy_seed_count"),
            delta_missing_parent=str(config["delta_missing_parent"]),
        )
        self.capacity = int(config["capacity"])
        self.mode = str(config["mode"])
        self.min_support = int(config["min_support"])
        self.prior_strength = float(config["prior_strength"])
        raw_prior_mean = config.get("prior_mean")
        self.prior_mean = None if raw_prior_mean is None else float(raw_prior_mean)
        raw_legacy_count = config.get("legacy_seed_count")
        self.legacy_seed_count = None if raw_legacy_count is None else int(raw_legacy_count)
        self.deduplicate_observations = bool(config["deduplicate_observations"])
        self.delta_missing_parent = str(config["delta_missing_parent"])
        self._event_index = int(state.get("event_index", 0))
        self._seen_observation_ids = set(state.get("seen_observation_ids", []))
        self._released_population = [
            (float(row["score"]), str(row["fragment"]))
            for row in state.get("released_population", [])
        ]
        self._records = {}
        self._active_cache = None
        for row in state.get("records", []):
            record = FragmentStats(
                fragment=str(row["fragment"]),
                total=float(row["total"]),
                count=int(row["count"]),
                seed_score=None if row.get("seed_score") is None else float(row["seed_score"]),
                seed_order=None if row.get("seed_order") is None else int(row["seed_order"]),
                first_seen=int(row.get("first_seen", 0)),
                last_seen=int(row.get("last_seen", 0)),
            )
            if record.fragment in self._records:
                raise ValueError(f"duplicate fragment {record.fragment!r} in state")
            self._records[record.fragment] = record

        rng_state = state.get("rng_state")
        setstate = getattr(self.rng, "setstate", None)
        if rng_state is not None and callable(setstate):
            setstate(rng_state)

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
        *,
        fragmenter: Fragmenter,
        rng: Optional[SamplingRNG] = None,
    ) -> "FragmentPopulation":
        """Construct a population from serialized state and live dependencies."""

        config = state.get("config")
        if not isinstance(config, Mapping):
            raise ValueError("state is missing a configuration mapping")
        population = cls(
            [],
            capacity=int(config["capacity"]),
            mode=str(config["mode"]),
            fragmenter=fragmenter,
            rng=rng,
            min_support=int(config["min_support"]),
            prior_strength=float(config["prior_strength"]),
            prior_mean=config.get("prior_mean"),
            legacy_seed_count=config.get("legacy_seed_count"),
            deduplicate_observations=bool(config["deduplicate_observations"]),
            delta_missing_parent=str(config["delta_missing_parent"]),
        )
        population.load_state_dict(state)
        return population

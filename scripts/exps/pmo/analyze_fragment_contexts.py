"""Validate and describe fragment-context evidence from one completed PMO run.

The checkpoint and append-only event stream are treated as the primary
evidence.  The analyzer replays global oracle accounting and statistical
fragment updates before computing descriptive context-variation statistics.
Checkpoint files use pickle and must only be loaded from trusted local runs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as datetime_module
import fcntl
import hashlib
import importlib.metadata
import math
import os
import platform
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.exps.pmo.main.genmol import experiment_io  # noqa: E402
from scripts.exps.pmo.main.genmol.experiment_io import (  # noqa: E402
    iter_events,
    load_checkpoint,
    sha256_config,
    sha256_file,
    summarize_indexed_scores,
    summarize_scores,
    write_manifest,
)


ANALYSIS_SCHEMA_VERSION = 2
SUPPORTED_POLICY_MODES = frozenset({"mean", "bayes"})
POPULATION_STATE_VERSION = 1
SHA256_LENGTH = 64


class AnalysisError(ValueError):
    """The supplied artifacts do not form a trustworthy completed run."""


class FileLock:
    """Non-blocking advisory lock used for source and destination ownership."""

    def __init__(self, path: Path, operation: int, *, create: bool) -> None:
        self.path = path
        if create:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = path.open("a+b")
        else:
            if not path.is_file():
                raise AnalysisError(f"required run lock is missing: {path}")
            self._handle = path.open("rb")
        try:
            fcntl.flock(self._handle.fileno(), operation | fcntl.LOCK_NB)
        except BaseException:
            self._handle.close()
            raise

    def close(self) -> None:
        if self._handle.closed:
            return
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()

    def __enter__(self) -> "FileLock":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


@dataclass(frozen=True)
class FragmentScoreObservation:
    """One whole-molecule score associated with one sampled fragment."""

    score: float
    event_index: int
    call_index: int
    canonical_smiles: str


@dataclass(frozen=True)
class OracleOutcome:
    """Validated event-log representation of one oracle lookup."""

    canonical_smiles: str
    score: float
    call_index: int
    charged: bool
    reason: str


@dataclass
class FragmentRecord:
    """Replayable form of the persisted fragment sufficient statistics."""

    fragment: str
    total: float
    count: int
    seed_score: Optional[float]
    seed_order: Optional[int]
    first_seen: int
    last_seen: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Defaults to manifest.json beside --events",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Defaults to summary.json beside --events",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Defaults to state/latest.pkl beside --events",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _utc_timestamp() -> str:
    now = datetime_module.datetime.now(datetime_module.timezone.utc)
    return now.isoformat(timespec="seconds").replace("+00:00", "Z")


def _git_output(*arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=REPOSITORY_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise AnalysisError(f"cannot resolve analyzer Git identity: {error}") from error
    # Preserve both status columns on the first porcelain-status entry.
    return completed.stdout.rstrip("\r\n")


def _current_git_metadata() -> dict[str, Any]:
    status = _git_output("status", "--porcelain=v1")
    try:
        tracked_diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise AnalysisError(f"cannot hash analyzer tracked Git diff: {error}") from error
    return {
        "commit": _git_output("rev-parse", "HEAD"),
        "branch": _git_output("branch", "--show-current"),
        "dirty": bool(status),
        "status": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
    }


def _package_version(distribution: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _analyzer_runtime_metadata() -> dict[str, Any]:
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": {
            "numpy": _package_version("numpy"),
            "rdkit": _package_version("rdkit"),
            "torch": _package_version("torch"),
            "pytdc": _package_version("pytdc"),
        },
    }


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AnalysisError(f"{label} must be an object")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AnalysisError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise AnalysisError(f"{label} must be a finite number")
    return result


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise AnalysisError(f"{label} must be a nonempty string")
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise AnalysisError(f"{label} must be boolean")
    return value


def _digest(value: Any, label: str) -> str:
    result = _string(value, label)
    if len(result) != SHA256_LENGTH or any(character not in "0123456789abcdef" for character in result):
        raise AnalysisError(f"{label} must be a lowercase SHA-256 digest")
    return result


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AnalysisError(f"{label} is {actual!r}, expected {expected!r}")


def _number_equal(actual: Any, expected: float, label: str) -> None:
    value = _number(actual, label)
    if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise AnalysisError(f"{label} is {value!r}, expected {expected!r}")


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        import json

        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                AnalysisError(f"{label} contains non-finite JSON constant {token}")
            ),
        )
    except (OSError, UnicodeError, ValueError) as error:
        if isinstance(error, AnalysisError):
            raise
        raise AnalysisError(f"invalid {label} at {path}: {error}") from error
    return _mapping(value, label)


def _declared_path(value: Any, label: str, manifest_dir: Path) -> Path:
    path = Path(_string(value, label)).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve()


def _source_entry(path: Path, digest: str) -> dict[str, Any]:
    return {"path": str(path), "sha256": digest, "size_bytes": path.stat().st_size}


def _resolve_paths(
    events_path: str | Path,
    manifest_path: str | Path | None,
    summary_path: str | Path | None,
    checkpoint_path: str | Path | None,
) -> dict[str, Path]:
    events = Path(events_path).expanduser().resolve()
    run_dir = events.parent
    paths = {
        "events": events,
        "manifest": (
            run_dir / "manifest.json"
            if manifest_path is None
            else Path(manifest_path).expanduser().resolve()
        ),
        "summary": (
            run_dir / "summary.json"
            if summary_path is None
            else Path(summary_path).expanduser().resolve()
        ),
        "checkpoint": (
            run_dir / "state" / "latest.pkl"
            if checkpoint_path is None
            else Path(checkpoint_path).expanduser().resolve()
        ),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise AnalysisError(f"run is missing required source files: {missing}")
    return paths


def _output_protected_paths(
    paths: Mapping[str, Path],
    manifest: Mapping[str, Any],
) -> dict[str, Path]:
    manifest_dir = paths["manifest"].parent
    model = _mapping(manifest.get("model"), "manifest.model")
    extra = _mapping(manifest.get("extra"), "manifest.extra")
    vocabulary = _mapping(extra.get("vocabulary"), "manifest.extra.vocabulary")
    helper_file = getattr(experiment_io, "__file__", None)
    if helper_file is None:
        raise AnalysisError("imported experiment_io helper has no filesystem identity")
    protected = dict(paths)
    protected.update(
        {
            "run_lock": paths["events"].parent / ".run.lock",
            "analyzer_implementation": Path(__file__).resolve(),
            "experiment_io_helper": Path(helper_file).resolve(),
            "model": _declared_path(
                model.get("path"), "manifest.model.path", manifest_dir
            ),
            "vocabulary": _declared_path(
                vocabulary.get("path"),
                "manifest.extra.vocabulary.path",
                manifest_dir,
            ),
        }
    )
    return {name: path.resolve() for name, path in protected.items()}


def _paths_alias(first: Path, second: Path) -> bool:
    if first.resolve() == second.resolve():
        return True
    if not first.exists() or not second.exists():
        return False
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def _validate_manifest(
    manifest: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    _equal(manifest.get("schema_version"), 1, "manifest.schema_version")
    _equal(manifest.get("status"), "completed", "manifest.status")
    _equal(manifest.get("error"), None, "manifest.error")
    config = dict(_mapping(manifest.get("config"), "manifest.config"))
    config_hash = _digest(manifest.get("config_sha256"), "manifest.config_sha256")
    _equal(sha256_config(config), config_hash, "manifest.config_sha256")

    experiment_id = _string(config.get("experiment_id"), "manifest.config.experiment_id")
    task = _string(manifest.get("task"), "manifest.task")
    variant = _string(manifest.get("variant"), "manifest.variant")
    seed = _integer(manifest.get("seed"), "manifest.seed")
    _equal(config.get("oracle"), task, "manifest.config.oracle")
    _equal(config.get("variant"), variant, "manifest.config.variant")
    _equal(config.get("seed"), seed, "manifest.config.seed")
    _equal(
        manifest.get("run_id"),
        f"{experiment_id}:{task}:{variant}:seed{seed}",
        "manifest.run_id",
    )
    _equal(paths["events"].parent.name, f"seed_{seed}", "run seed directory")
    _equal(paths["events"].parent.parent.name, variant, "run variant directory")
    _equal(paths["events"].parent.parent.parent.name, task, "run task directory")

    mode = config.get("policy_mode")
    if mode not in SUPPORTED_POLICY_MODES:
        raise AnalysisError(
            "context analysis requires a mean or bayes statistical arm; "
            f"got policy_mode={mode!r}"
        )
    budget = _integer(manifest.get("oracle_budget"), "manifest.oracle_budget", minimum=1)
    _equal(config.get("max_oracle_calls"), budget, "manifest.config.max_oracle_calls")
    _equal(manifest.get("oracle_calls"), budget, "manifest.oracle_calls")

    for key in ("events", "summary", "checkpoint"):
        _equal(
            _declared_path(
                manifest.get(f"{key}_path"),
                f"manifest.{key}_path",
                paths["manifest"].parent,
            ),
            paths[key],
            f"manifest.{key}_path",
        )

    model = _mapping(manifest.get("model"), "manifest.model")
    model_path = _string(model.get("path"), "manifest.model.path")
    model_hash = _digest(model.get("sha256"), "manifest.model.sha256")
    _integer(model.get("size_bytes"), "manifest.model.size_bytes", minimum=1)
    _equal(config.get("model_path"), model_path, "manifest.config.model_path")

    extra = _mapping(manifest.get("extra"), "manifest.extra")
    vocabulary = _mapping(extra.get("vocabulary"), "manifest.extra.vocabulary")
    vocabulary_path = _string(
        vocabulary.get("path"), "manifest.extra.vocabulary.path"
    )
    vocabulary_hash = _digest(
        vocabulary.get("sha256"), "manifest.extra.vocabulary.sha256"
    )
    _equal(config.get("vocab_path"), vocabulary_path, "manifest.config.vocab_path")
    git = _mapping(extra.get("git"), "manifest.extra.git")
    git_commit = _string(git.get("commit"), "manifest.extra.git.commit")
    tracked_diff = _digest(
        git.get("tracked_diff_sha256"),
        "manifest.extra.git.tracked_diff_sha256",
    )
    _string(git.get("branch"), "manifest.extra.git.branch")
    _boolean(git.get("dirty"), "manifest.extra.git.dirty")

    identity = {
        "experiment_id": experiment_id,
        "run_id": manifest["run_id"],
        "task": task,
        "variant": variant,
        "seed": seed,
        "status": "completed",
        "oracle_budget": budget,
        "oracle_calls": budget,
        "config_sha256": config_hash,
        "model_path": model_path,
        "model_sha256": model_hash,
        "vocabulary_path": vocabulary_path,
        "vocabulary_sha256": vocabulary_hash,
        "git_commit": git_commit,
        "git_branch": git["branch"],
        "git_dirty": git["dirty"],
        "tracked_diff_sha256": tracked_diff,
    }
    return config, identity


def _validate_summary(
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[int, int]:
    schema = summary.get("schema_version")
    if schema not in {1, 2}:
        raise AnalysisError(f"unsupported summary.schema_version {schema!r}")
    _equal(summary.get("run_id"), manifest.get("run_id"), "summary.run_id")
    _equal(summary.get("status"), "completed", "summary.status")
    _equal(summary.get("error"), None, "summary.error")
    _equal(summary.get("checkpoint_consistent"), True, "summary.checkpoint_consistent")
    _equal(
        summary.get("config_sha256"),
        manifest.get("config_sha256"),
        "summary.config_sha256",
    )
    _equal(
        summary.get("model_sha256"),
        _mapping(manifest.get("model"), "manifest.model").get("sha256"),
        "summary.model_sha256",
    )
    events = _integer(summary.get("events"), "summary.events", minimum=1)
    iterations = _integer(
        summary.get("iterations_completed"),
        "summary.iterations_completed",
        minimum=1,
    )
    _equal(iterations, events, "summary iterations versus events")
    budget = int(manifest["oracle_budget"])
    _equal(summary.get("recoverable_events"), events, "summary.recoverable_events")
    _equal(
        summary.get("recoverable_oracle_calls"),
        budget,
        "summary.recoverable_oracle_calls",
    )
    elapsed = _number(summary.get("elapsed_seconds"), "summary.elapsed_seconds")
    _number_equal(manifest.get("elapsed_seconds"), elapsed, "manifest.elapsed_seconds")
    _integer(config.get("reporting_frequency"), "config.reporting_frequency", minimum=1)
    return schema, events


def _load_and_validate_checkpoint(
    path: Path,
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    event_count: int,
) -> tuple[Mapping[str, Any], Mapping[int, tuple[str, float]]]:
    try:
        state, metadata = load_checkpoint(path, with_metadata=True)
    except Exception as error:
        raise AnalysisError(f"invalid trusted checkpoint at {path}: {error}") from error
    state = _mapping(state, "checkpoint.state")
    metadata = _mapping(metadata, "checkpoint.metadata")
    _equal(state.get("event_count"), event_count, "checkpoint.state.event_count")
    _equal(
        state.get("next_iteration"),
        summary.get("iterations_completed"),
        "checkpoint.state.next_iteration",
    )
    checkpoint_elapsed = _number(
        state.get("elapsed_seconds"), "checkpoint.state.elapsed_seconds"
    )
    if checkpoint_elapsed > float(summary["elapsed_seconds"]) + 1e-9:
        raise AnalysisError("checkpoint elapsed time exceeds summary elapsed time")

    extra = _mapping(manifest.get("extra"), "manifest.extra")
    git = _mapping(extra.get("git"), "manifest.extra.git")
    vocabulary = _mapping(extra.get("vocabulary"), "manifest.extra.vocabulary")
    expected_metadata = {
        "config_sha256": manifest.get("config_sha256"),
        "model_sha256": _mapping(manifest.get("model"), "manifest.model").get("sha256"),
        "vocabulary_sha256": vocabulary.get("sha256"),
        "git_commit": git.get("commit"),
        "tracked_diff_sha256": git.get("tracked_diff_sha256"),
    }
    for key, expected in expected_metadata.items():
        _equal(metadata.get(key), expected, f"checkpoint.metadata.{key}")

    budget = int(manifest["oracle_budget"])
    oracle = _mapping(state.get("oracle"), "checkpoint.state.oracle")
    _equal(oracle.get("budget"), budget, "checkpoint oracle budget")
    buffer = _mapping(oracle.get("buffer"), "checkpoint.state.oracle.buffer")
    _equal(len(buffer), budget, "checkpoint oracle buffer size")
    records: dict[int, tuple[str, float]] = {}
    for canonical, raw_row in buffer.items():
        canonical = _string(canonical, "checkpoint oracle canonical SMILES")
        if not isinstance(raw_row, (list, tuple)) or len(raw_row) != 2:
            raise AnalysisError("checkpoint oracle entries must be [score, call_index]")
        score = _number(raw_row[0], f"checkpoint oracle score for {canonical}")
        call_index = _integer(
            raw_row[1], f"checkpoint oracle call index for {canonical}", minimum=1
        )
        if call_index in records:
            raise AnalysisError(f"duplicate checkpoint oracle call index {call_index}")
        records[call_index] = (canonical, score)
    _equal(
        sorted(records),
        list(range(1, budget + 1)),
        "checkpoint oracle call indices",
    )
    return state, records


def _fragments(value: Any, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise AnalysisError(f"{label} must be a list")
    result = tuple(_string(item, f"{label}[{index}]") for index, item in enumerate(value))
    if tuple(sorted(set(result))) != result:
        raise AnalysisError(f"{label} must contain sorted unique fragments")
    return result


def _outcome(
    raw: Any,
    label: str,
    checkpoint: Mapping[int, tuple[str, float]],
    cumulative_calls: int,
) -> tuple[Optional[OracleOutcome], int]:
    if raw is None:
        return None, cumulative_calls
    row = _mapping(raw, label)
    charged = _boolean(row.get("charged"), f"{label}.charged")
    reason = _string(row.get("reason"), f"{label}.reason")
    _equal(row.get("valid"), True, f"{label}.valid")
    _string(row.get("raw_smiles"), f"{label}.raw_smiles")
    canonical = _string(row.get("canonical_smiles"), f"{label}.canonical_smiles")
    score = _number(row.get("score"), f"{label}.score")
    call_index = _integer(row.get("call_index"), f"{label}.call_index", minimum=1)
    if charged:
        _equal(reason, "scored", f"{label}.reason")
        _equal(call_index, cumulative_calls + 1, f"{label}.call_index")
        cumulative_calls += 1
    else:
        _equal(reason, "cache_hit", f"{label}.reason")
        if call_index > cumulative_calls:
            raise AnalysisError(f"{label} cache hit precedes its charged oracle call")
    expected = checkpoint.get(call_index)
    if expected is None:
        raise AnalysisError(f"{label} references missing checkpoint call {call_index}")
    _equal(canonical, expected[0], f"{label}.canonical_smiles")
    _number_equal(score, expected[1], f"{label}.score")
    return OracleOutcome(canonical, score, call_index, charged, reason), cumulative_calls


def _top_means(
    checkpoint: Mapping[int, tuple[str, float]],
    calls: int,
) -> dict[str, float]:
    scores = sorted((checkpoint[index][1] for index in range(1, calls + 1)), reverse=True)
    return {
        f"top_{k}": float(statistics.fmean(scores[:k]))
        for k in (1, 10, 100)
    }


def _replay_oracle_events(
    events_path: Path,
    *,
    checkpoint: Mapping[int, tuple[str, float]],
    config: Mapping[str, Any],
    expected_events: int,
    budget: int,
    summary_schema: int,
) -> tuple[list[dict[str, Any]], list[tuple[int, float]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    child_indexed_scores: list[tuple[int, float]] = []
    cumulative_calls = 0
    previous_elapsed = -math.inf
    counters = {
        "events": 0,
        "charged_oracle_calls": 0,
        "charged_parent_oracle_calls": 0,
        "charged_child_oracle_calls": 0,
        "cached_parent_oracle_lookups": 0,
        "cached_child_oracle_lookups": 0,
        "events_without_child": 0,
        "legacy_terminal_event_omissions": 0,
    }
    parent_control = _boolean(config.get("parent_control"), "config.parent_control")
    warmup = _integer(config.get("warmup"), "config.warmup")
    legacy_boundary = _boolean(
        config.get("legacy_warmup_off_by_one"),
        "config.legacy_warmup_off_by_one",
    )

    for expected_index, raw_event in enumerate(iter_events(events_path)):
        if cumulative_calls >= budget:
            raise AnalysisError(
                f"event[{expected_index}] occurs after the oracle budget was exhausted"
            )
        event = dict(_mapping(raw_event, f"event[{expected_index}]"))
        _equal(event.get("event_index"), expected_index, f"event[{expected_index}].event_index")
        _equal(event.get("iteration"), expected_index, f"event[{expected_index}].iteration")
        expected_remask = expected_index > warmup if legacy_boundary else expected_index >= warmup
        _equal(event.get("remask_enabled"), expected_remask, f"event[{expected_index}].remask_enabled")

        parent, cumulative_calls = _outcome(
            event.get("parent_oracle"),
            f"event[{expected_index}].parent_oracle",
            checkpoint,
            cumulative_calls,
        )
        parent_exhausted_budget = (
            parent is not None and parent.charged and cumulative_calls == budget
        )
        if parent_control and expected_remask:
            if parent is None:
                raise AnalysisError(f"event[{expected_index}] is missing its parent control")
        elif parent is not None:
            raise AnalysisError(f"event[{expected_index}] has an unexpected parent oracle row")
        if parent is not None:
            _equal(
                _string(event.get("parent_smiles"), f"event[{expected_index}].parent_smiles"),
                _mapping(event["parent_oracle"], "parent oracle").get("raw_smiles"),
                f"event[{expected_index}] parent raw SMILES",
            )
            counters[
                "charged_parent_oracle_calls"
                if parent.charged
                else "cached_parent_oracle_lookups"
            ] += 1

        if parent_exhausted_budget and event.get("child_oracle") is not None:
            raise AnalysisError(
                f"event[{expected_index}] has a child after its parent exhausted the budget"
            )
        child, cumulative_calls = _outcome(
            event.get("child_oracle"),
            f"event[{expected_index}].child_oracle",
            checkpoint,
            cumulative_calls,
        )
        update = _mapping(event.get("population_update"), f"event[{expected_index}].population_update")
        if child is None:
            counters["events_without_child"] += 1
            _equal(update.get("reason"), "budget_after_parent", f"event[{expected_index}] update reason")
            if parent is None or not parent.charged or cumulative_calls != budget:
                raise AnalysisError(
                    f"event[{expected_index}] child may be absent only after the final charged parent"
                )
        else:
            _equal(
                _string(event.get("child_smiles"), f"event[{expected_index}].child_smiles"),
                _mapping(event["child_oracle"], "child oracle").get("raw_smiles"),
                f"event[{expected_index}] child raw SMILES",
            )
            if child.charged:
                counters["charged_child_oracle_calls"] += 1
                child_indexed_scores.append((child.call_index, child.score))
            else:
                counters["cached_child_oracle_lookups"] += 1

        _equal(
            event.get("oracle_calls"),
            cumulative_calls,
            f"event[{expected_index}].oracle_calls",
        )
        for key, expected in _top_means(checkpoint, cumulative_calls).items():
            _number_equal(event.get(key), expected, f"event[{expected_index}].{key}")
        raw_elapsed = event.get("elapsed_seconds")
        if (
            raw_elapsed is None
            and child is None
            and summary_schema == 1
            and "population_sampling_order" not in config
        ):
            # Schema-1 runs written before the terminal-event repair omitted
            # timing and population fields when the final parent consumed the
            # budget. The checkpoint still makes the scientific state
            # replayable; expose the omission rather than inventing a value.
            counters["legacy_terminal_event_omissions"] += 1
        else:
            elapsed = _number(
                raw_elapsed, f"event[{expected_index}].elapsed_seconds"
            )
            if elapsed < previous_elapsed:
                raise AnalysisError(f"event[{expected_index}].elapsed_seconds decreased")
            previous_elapsed = elapsed
        counters["events"] += 1
        rows.append({"event": event, "parent": parent, "child": child})
        if cumulative_calls == budget and expected_index != expected_events - 1:
            raise AnalysisError(
                f"event[{expected_index}] exhausts the oracle budget before the final event"
            )

    _equal(len(rows), expected_events, "event log row count")
    _equal(cumulative_calls, budget, "charged event oracle count")
    counters["charged_oracle_calls"] = cumulative_calls
    return rows, child_indexed_scores, counters


def _parse_record(value: Any, label: str) -> FragmentRecord:
    row = _mapping(value, label)
    expected_keys = {
        "fragment",
        "total",
        "count",
        "seed_score",
        "seed_order",
        "first_seen",
        "last_seen",
    }
    _equal(set(row), expected_keys, f"{label} fields")
    raw_seed_score = row.get("seed_score")
    seed_score = None if raw_seed_score is None else _number(raw_seed_score, f"{label}.seed_score")
    raw_seed_order = row.get("seed_order")
    seed_order = (
        None
        if raw_seed_order is None
        else _integer(raw_seed_order, f"{label}.seed_order")
    )
    return FragmentRecord(
        fragment=_string(row.get("fragment"), f"{label}.fragment"),
        total=_number(row.get("total"), f"{label}.total"),
        count=_integer(row.get("count"), f"{label}.count"),
        seed_score=seed_score,
        seed_order=seed_order,
        first_seen=_integer(row.get("first_seen"), f"{label}.first_seen"),
        last_seen=_integer(row.get("last_seen"), f"{label}.last_seen"),
    )


def _record_equal(actual: FragmentRecord, expected: FragmentRecord, label: str) -> None:
    _equal(actual.fragment, expected.fragment, f"{label}.fragment")
    _number_equal(actual.total, expected.total, f"{label}.total")
    _equal(actual.count, expected.count, f"{label}.count")
    if expected.seed_score is None:
        _equal(actual.seed_score, None, f"{label}.seed_score")
    else:
        _number_equal(actual.seed_score, expected.seed_score, f"{label}.seed_score")
    _equal(actual.seed_order, expected.seed_order, f"{label}.seed_order")
    _equal(actual.first_seen, expected.first_seen, f"{label}.first_seen")
    _equal(actual.last_seen, expected.last_seen, f"{label}.last_seen")


def _rank(record: FragmentRecord, config: Mapping[str, Any]) -> float:
    if record.count < 1:
        raise AnalysisError(f"ranked fragment {record.fragment!r} has zero support")
    if config["mode"] == "mean":
        return record.total / record.count
    prior_mean = _number(config.get("prior_mean"), "checkpoint population prior_mean")
    prior_strength = _number(
        config.get("prior_strength"), "checkpoint population prior_strength"
    )
    return (record.total + prior_strength * prior_mean) / (
        record.count + prior_strength
    )


def _active_rows(
    records: Mapping[str, FragmentRecord],
    population_config: Mapping[str, Any],
    run_config: Mapping[str, Any],
) -> list[tuple[float, str]]:
    minimum = int(population_config["min_support"])
    eligible = [
        record
        for record in records.values()
        if record.seed_score is not None or record.count >= minimum
    ]
    if "population_sampling_order" in run_config:
        eligible.sort(key=lambda record: record.fragment, reverse=True)
        eligible.sort(key=lambda record: -_rank(record, population_config))
    else:
        eligible.sort(
            key=lambda record: (-_rank(record, population_config), record.fragment)
        )
    return [
        (_rank(record, population_config), record.fragment)
        for record in eligible[: int(population_config["capacity"])]
    ]


def _rows_equal(actual: Any, expected: Sequence[tuple[float, str]], label: str) -> None:
    if not isinstance(actual, list) or len(actual) != len(expected):
        raise AnalysisError(f"{label} has the wrong number of rows")
    for index, (raw, wanted) in enumerate(zip(actual, expected)):
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise AnalysisError(f"{label}[{index}] must be [score, fragment]")
        _number_equal(raw[0], wanted[0], f"{label}[{index}].score")
        _equal(raw[1], wanted[1], f"{label}[{index}].fragment")


def _validate_population_config(
    population: Mapping[str, Any],
    run_config: Mapping[str, Any],
) -> Mapping[str, Any]:
    _equal(population.get("version"), POPULATION_STATE_VERSION, "population state version")
    config = _mapping(population.get("config"), "checkpoint population config")
    pairs = {
        "capacity": "population_size",
        "mode": "policy_mode",
        "min_support": "min_support",
        "prior_strength": "prior_strength",
        "prior_mean": "prior_mean",
        "legacy_seed_count": "legacy_seed_count",
    }
    for population_key, run_key in pairs.items():
        _equal(
            config.get(population_key),
            run_config.get(run_key),
            f"checkpoint population config {population_key}",
        )
    _equal(
        config.get("deduplicate_observations"),
        True,
        "checkpoint population deduplicate_observations",
    )
    _equal(config.get("delta_missing_parent"), "skip", "checkpoint population delta_missing_parent")
    _equal(population.get("released_population"), [], "checkpoint released population")
    return config


def _load_seed_vocabulary(
    manifest: Mapping[str, Any],
    population_config: Mapping[str, Any],
) -> tuple[dict[str, FragmentRecord], dict[str, Any]]:
    """Load the exact first-V seed state used by FragmentPopulation.from_csv."""

    extra = _mapping(manifest.get("extra"), "manifest.extra")
    declared = _mapping(extra.get("vocabulary"), "manifest.extra.vocabulary")
    path = Path(_string(declared.get("path"), "manifest vocabulary path")).expanduser().resolve()
    if not path.is_file():
        raise AnalysisError(f"declared vocabulary file is missing: {path}")
    actual_hash = sha256_file(path)
    _equal(actual_hash, declared.get("sha256"), "declared vocabulary SHA-256")
    capacity = int(population_config["capacity"])
    legacy_count = population_config.get("legacy_seed_count")
    seeds: dict[str, FragmentRecord] = {}
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            if not {"frag", "score"}.issubset(fields):
                raise AnalysisError("vocabulary must contain frag and score columns")
            for order, row in enumerate(reader):
                if order >= capacity:
                    break
                fragment = _string(row.get("frag"), f"vocabulary row {order} frag")
                if fragment in seeds:
                    raise AnalysisError(f"duplicate seed fragment {fragment!r} in vocabulary")
                score = _number(
                    float(row.get("score")), f"vocabulary row {order} score"
                )
                raw_count = row.get("count")
                count = (
                    None
                    if raw_count in (None, "")
                    else _integer(int(raw_count), f"vocabulary row {order} count", minimum=1)
                )
                raw_total = row.get("score_sum")
                if raw_total in (None, ""):
                    raw_total = row.get("sum")
                if raw_total in (None, ""):
                    raw_total = row.get("total")
                total = (
                    None
                    if raw_total in (None, "")
                    else _number(float(raw_total), f"vocabulary row {order} score_sum")
                )
                if total is not None and count is None:
                    raise AnalysisError(
                        f"vocabulary row {order} has score_sum without count"
                    )
                if count is None:
                    if legacy_count is None:
                        raise AnalysisError(
                            f"vocabulary row {order} lacks count and no legacy pseudo-count is configured"
                        )
                    count = _integer(
                        legacy_count,
                        "checkpoint population legacy_seed_count",
                        minimum=1,
                    )
                if total is None:
                    total = score * count
                # Match FragmentPopulation.from_csv exactly so evidence that
                # production accepts is neither rejected nor broadened here.
                if not math.isclose(
                    score,
                    total / count,
                    rel_tol=1e-8,
                    abs_tol=1e-10,
                ):
                    raise AnalysisError(
                        f"vocabulary row {order} has inconsistent "
                        "score/count/score_sum"
                    )
                seeds[fragment] = FragmentRecord(
                    fragment=fragment,
                    total=total,
                    count=count,
                    seed_score=score,
                    seed_order=order,
                    first_seen=0,
                    last_seen=0,
                )
    except (OSError, UnicodeError, ValueError, csv.Error, TypeError) as error:
        if isinstance(error, AnalysisError):
            raise
        raise AnalysisError(f"invalid vocabulary at {path}: {error}") from error
    if not 2 <= len(seeds) <= capacity:
        raise AnalysisError("vocabulary seed count is outside [2, capacity]")
    _equal(sha256_file(path), actual_hash, "vocabulary SHA-256 after loading")
    return seeds, {
        "path": str(path),
        "sha256": actual_hash,
        "size_bytes": path.stat().st_size,
        "rows_loaded": len(seeds),
    }


def _selected_fragments(value: Any, label: str) -> tuple[str, str]:
    if not isinstance(value, list) or len(value) != 2:
        raise AnalysisError(f"{label} must contain exactly two fragments")
    first = _string(value[0], f"{label}[0]")
    second = _string(value[1], f"{label}[1]")
    if first == second:
        raise AnalysisError(f"{label} must contain two distinct fragments")
    return first, second


def _validate_population_replay(
    replay_rows: Sequence[Mapping[str, Any]],
    *,
    population_state: Mapping[str, Any],
    population_config: Mapping[str, Any],
    seed_vocabulary: Mapping[str, FragmentRecord],
    run_config: Mapping[str, Any],
    summary: Mapping[str, Any],
    summary_schema: int,
) -> tuple[
    dict[str, list[FragmentScoreObservation]],
    dict[str, FragmentRecord],
    dict[str, int],
]:
    raw_records = population_state.get("records")
    if not isinstance(raw_records, list):
        raise AnalysisError("checkpoint population records must be a list")
    final_records: dict[str, FragmentRecord] = {}
    for index, raw_record in enumerate(raw_records):
        record = _parse_record(raw_record, f"checkpoint population records[{index}]")
        if record.fragment in final_records:
            raise AnalysisError(f"duplicate checkpoint fragment {record.fragment!r}")
        final_records[record.fragment] = record

    seed_records = {
        record.fragment: record
        for record in final_records.values()
        if record.seed_score is not None
    }
    _equal(
        set(seed_records),
        set(seed_vocabulary),
        "checkpoint versus vocabulary seed fragments",
    )
    for record in final_records.values():
        if (record.seed_score is None) != (record.seed_order is None):
            raise AnalysisError(
                f"fragment {record.fragment!r} has inconsistent seed_score/seed_order"
            )
    for fragment, expected_seed in seed_vocabulary.items():
        final = seed_records[fragment]
        if final.seed_score is None:
            raise AnalysisError(f"seed fragment {fragment!r} lost seed_score")
        _number_equal(
            final.seed_score,
            expected_seed.seed_score,
            f"checkpoint seed fragment {fragment!r} seed_score",
        )
        _equal(
            final.seed_order,
            expected_seed.seed_order,
            f"checkpoint seed fragment {fragment!r} seed_order",
        )

    running = {
        fragment: replace(record) for fragment, record in seed_vocabulary.items()
    }
    seen: set[str] = set()
    population_event_index = 0
    histories: dict[str, list[FragmentScoreObservation]] = defaultdict(list)
    counters = {
        "unique_child_observations": 0,
        "duplicate_child_observations": 0,
        "population_updates": 0,
        "updates_with_fragments": 0,
        "updates_without_fragments": 0,
        "fragment_observations": 0,
        "seed_fragment_observations": 0,
        "novel_fragment_observations": 0,
        "budget_after_parent_events": 0,
    }

    for replay in replay_rows:
        event = _mapping(replay.get("event"), "replay event")
        event_index = int(event["event_index"])
        child = replay.get("child")
        update = _mapping(
            event.get("population_update"),
            f"event[{event_index}].population_update",
        )
        updated = _boolean(update.get("updated"), f"event[{event_index}] population updated")
        reason = _string(update.get("reason"), f"event[{event_index}] population reason")
        fragments = _fragments(
            update.get("observed_fragments", []),
            f"event[{event_index}].population_update.observed_fragments",
        )
        admitted = _fragments(
            update.get("admitted", []),
            f"event[{event_index}].population_update.admitted",
        )
        displaced = _fragments(
            update.get("displaced", []),
            f"event[{event_index}].population_update.displaced",
        )
        raw_statistics = event.get("fragment_statistics_after")
        if raw_statistics is None and child is None and summary_schema == 1:
            statistics_after = {}
        else:
            statistics_after = _mapping(
                raw_statistics,
                f"event[{event_index}].fragment_statistics_after",
            )
        _equal(
            set(statistics_after),
            set(fragments),
            f"event[{event_index}] fragment statistics keys",
        )
        before = _active_rows(running, population_config, run_config)
        selected = _selected_fragments(
            event.get("selected_fragments"),
            f"event[{event_index}].selected_fragments",
        )
        inactive = set(selected) - {fragment for _, fragment in before}
        if inactive:
            raise AnalysisError(
                f"event[{event_index}].selected_fragments contains inactive fragments: "
                f"{sorted(inactive)!r}"
            )

        if child is None:
            _equal(reason, "budget_after_parent", f"event[{event_index}] update reason")
            _equal(updated, False, f"event[{event_index}] update flag")
            _equal(fragments, (), f"event[{event_index}] observed fragments")
            counters["budget_after_parent_events"] += 1
        else:
            if not isinstance(child, OracleOutcome):
                raise AnalysisError(f"event[{event_index}] has malformed replay child")
            duplicate = child.canonical_smiles in seen
            if duplicate:
                _equal(reason, "duplicate_observation", f"event[{event_index}] update reason")
                _equal(updated, False, f"event[{event_index}] update flag")
                _equal(fragments, (), f"event[{event_index}] duplicate fragments")
                counters["duplicate_child_observations"] += 1
            else:
                seen.add(child.canonical_smiles)
                counters["unique_child_observations"] += 1
                if fragments:
                    _equal(reason, "updated", f"event[{event_index}] update reason")
                    _equal(updated, True, f"event[{event_index}] update flag")
                    population_event_index += 1
                    counters["population_updates"] += 1
                    counters["updates_with_fragments"] += 1
                    for fragment in fragments:
                        final = final_records.get(fragment)
                        if final is None:
                            raise AnalysisError(
                                f"event[{event_index}] fragment {fragment!r} is absent from checkpoint"
                            )
                        record = running.get(fragment)
                        if record is None:
                            record = FragmentRecord(
                                fragment,
                                0.0,
                                0,
                                final.seed_score,
                                final.seed_order,
                                0,
                                0,
                            )
                            running[fragment] = record
                        record.total += child.score
                        record.count += 1
                        if record.first_seen == 0:
                            record.first_seen = population_event_index
                        record.last_seen = population_event_index
                        snapshot = _parse_record(
                            statistics_after[fragment],
                            f"event[{event_index}].fragment_statistics_after[{fragment!r}]",
                        )
                        _record_equal(snapshot, record, f"event[{event_index}] statistics for {fragment!r}")
                        histories[fragment].append(
                            FragmentScoreObservation(
                                child.score,
                                event_index,
                                child.call_index,
                                child.canonical_smiles,
                            )
                        )
                        counters["fragment_observations"] += 1
                        counters[
                            "seed_fragment_observations"
                            if final.seed_score is not None
                            else "novel_fragment_observations"
                        ] += 1
                else:
                    _equal(reason, "no_fragments", f"event[{event_index}] update reason")
                    _equal(updated, False, f"event[{event_index}] update flag")
                    counters["updates_without_fragments"] += 1

        if not fragments:
            _equal(statistics_after, {}, f"event[{event_index}] fragment statistics")
        after = _active_rows(running, population_config, run_config)
        expected_admitted = tuple(sorted({fragment for _, fragment in after} - {fragment for _, fragment in before}))
        expected_displaced = tuple(sorted({fragment for _, fragment in before} - {fragment for _, fragment in after}))
        _equal(admitted, expected_admitted, f"event[{event_index}] admitted fragments")
        _equal(displaced, expected_displaced, f"event[{event_index}] displaced fragments")

        missing_legacy_terminal_fields = (
            summary_schema == 1
            and child is None
            and "population_size_after" not in event
            and "population_cutoff_after" not in event
        )
        if not missing_legacy_terminal_fields:
            _equal(
                event.get("population_size_after"),
                len(after),
                f"event[{event_index}].population_size_after",
            )
            if not after:
                raise AnalysisError(f"event[{event_index}] has an empty active population")
            _number_equal(
                event.get("population_cutoff_after"),
                after[-1][0],
                f"event[{event_index}].population_cutoff_after",
            )

    _equal(
        population_state.get("event_index"),
        population_event_index,
        "checkpoint population event_index",
    )
    raw_seen = population_state.get("seen_observation_ids")
    if not isinstance(raw_seen, list):
        raise AnalysisError("checkpoint seen_observation_ids must be a list")
    _equal(raw_seen, sorted(seen), "checkpoint seen_observation_ids")
    _equal(set(running), set(final_records), "replayed checkpoint fragment registry")
    for fragment, final in final_records.items():
        _record_equal(running[fragment], final, f"final fragment {fragment!r}")

    final_active = _active_rows(running, population_config, run_config)
    summary_population = _mapping(summary.get("population"), "summary.population")
    _equal(summary_population.get("size"), len(final_active), "summary population size")
    _rows_equal(
        summary_population.get("active_rows"),
        final_active,
        "summary population active_rows",
    )
    return dict(histories), final_records, counters


def _child_count_summary(
    scores: Sequence[float],
    *,
    reporting_frequency: int,
) -> dict[str, Any]:
    if scores:
        result = summarize_scores(
            scores,
            reporting_frequency=reporting_frequency,
            budget=len(scores),
        )
        result["axis"] = "charged_child_count"
        result["score_count"] = result.pop("oracle_calls")
        result["child_count_horizon"] = result.pop("oracle_budget")
        return result
    result: dict[str, Any] = {
        "axis": "charged_child_count",
        "score_count": 0,
        "child_count_horizon": 0,
        "reporting_frequency": reporting_frequency,
    }
    for k in (1, 10, 100):
        label = f"top_{k}"
        result[label] = None
        result[f"auc_{label}"] = None
        result[f"trajectory_{label}"] = [{"oracle_calls": 0, "top_k_mean": 0.0}]
    return result


def _validate_summary_scores(
    summary: Mapping[str, Any],
    *,
    summary_schema: int,
    checkpoint: Mapping[int, tuple[str, float]],
    child_indexed_scores: Sequence[tuple[int, float]],
    budget: int,
    reporting_frequency: int,
) -> None:
    groups = _mapping(summary.get("scores"), "summary.scores")
    all_scores = [checkpoint[index][1] for index in range(1, budget + 1)]
    expected_all = summarize_scores(
        all_scores,
        reporting_frequency=reporting_frequency,
        budget=budget,
    )
    _equal(
        groups.get("all_charged_molecules"),
        expected_all,
        "summary all-charged metrics",
    )
    recognized = {key for key in groups if str(key).startswith("charged_children")}
    child_scores = [score for _, score in child_indexed_scores]
    if summary_schema == 1:
        _equal(recognized, {"charged_children_only"}, "summary schema-1 child views")
        expected = summarize_scores(
            child_scores,
            reporting_frequency=reporting_frequency,
            budget=budget,
        )
        _equal(groups.get("charged_children_only"), expected, "summary child metrics")
    else:
        expected_keys = {
            "charged_children_total_call_axis",
            "charged_children_child_count_axis",
        }
        _equal(recognized, expected_keys, "summary schema-2 child views")
        expected_total = summarize_indexed_scores(
            child_indexed_scores,
            observed_oracle_calls=budget,
            reporting_frequency=reporting_frequency,
            budget=budget,
        )
        expected_count = _child_count_summary(
            child_scores,
            reporting_frequency=reporting_frequency,
        )
        _equal(
            groups.get("charged_children_total_call_axis"),
            expected_total,
            "summary total-call child metrics",
        )
        _equal(
            groups.get("charged_children_child_count_axis"),
            expected_count,
            "summary child-count metrics",
        )


def _sample_sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise AnalysisError("sample SD requires at least two values")
    return float(statistics.stdev(values))


def _support_values_summary(
    supports: Sequence[int],
    *,
    denominator: str,
) -> dict[str, Any]:
    histogram = {
        str(support): supports.count(support)
        for support in sorted(set(supports))
    }
    fragments = len(supports)
    positive = sum(support > 0 for support in supports)
    zeros = fragments - positive
    singletons = sum(support == 1 for support in supports)
    repeated = sum(support >= 2 for support in supports)
    return {
        "denominator": denominator,
        "denominator_fragments": fragments,
        "fragments": fragments,
        "fragments_with_dynamic_support": positive,
        "zero_dynamic_support_fragments": zeros,
        "fragment_observations": sum(supports),
        "minimum": min(supports) if supports else None,
        "median": float(statistics.median(supports)) if supports else None,
        "mean": float(statistics.mean(supports)) if supports else None,
        "maximum": max(supports) if supports else None,
        "singletons": singletons,
        "singleton_fraction": singletons / fragments if fragments else None,
        "repeated_fragments": repeated,
        "repeated_fragment_fraction": repeated / fragments if fragments else None,
        "histogram": histogram,
    }


def _support_summary(
    histories: Mapping[str, Sequence[FragmentScoreObservation]],
) -> dict[str, Any]:
    return _support_values_summary(
        [len(rows) for rows in histories.values()],
        denominator="fragments with at least one logged dynamic observation",
    )


def _registry_support_summary(
    registry_fragments: Sequence[str],
    histories: Mapping[str, Sequence[FragmentScoreObservation]],
) -> dict[str, Any]:
    return _support_values_summary(
        [len(histories.get(fragment, ())) for fragment in registry_fragments],
        denominator="all checkpoint registry fragments in this fragment class",
    )


def _repeated_fragment_metrics(
    histories: Mapping[str, Sequence[FragmentScoreObservation]],
    *,
    top_n: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    within_sum_squares = 0.0
    within_degrees_freedom = 0
    for fragment, observations in histories.items():
        if len(observations) < 2:
            continue
        scores = [observation.score for observation in observations]
        final_mean = float(statistics.mean(scores))
        sample_sd = _sample_sd(scores)
        difference = scores[0] - final_mean
        within_sum_squares += sum((score - final_mean) ** 2 for score in scores)
        within_degrees_freedom += len(scores) - 1
        rows.append(
            {
                "fragment": fragment,
                "support": len(scores),
                "first_score": scores[0],
                "final_mean": final_mean,
                "first_minus_final_mean": difference,
                "absolute_first_minus_final_mean": abs(difference),
                "within_fragment_sample_sd": sample_sd,
                "first_event_index": observations[0].event_index,
                "first_call_index": observations[0].call_index,
                "last_event_index": observations[-1].event_index,
                "last_call_index": observations[-1].call_index,
            }
        )
    if not rows:
        return {
            "fragments": 0,
            "mean_within_fragment_sample_sd": None,
            "pooled_within_fragment_sample_sd": None,
            "mean_absolute_first_minus_final_mean": None,
            "signed_optimism_mean": None,
            "first_above_final_mean_fraction": None,
            "max_first_overshoot": None,
        }, []
    differences = [row["first_minus_final_mean"] for row in rows]
    metrics = {
        "fragments": len(rows),
        "mean_within_fragment_sample_sd": float(
            statistics.mean(row["within_fragment_sample_sd"] for row in rows)
        ),
        "pooled_within_fragment_sample_sd": math.sqrt(
            within_sum_squares / within_degrees_freedom
        ),
        "mean_absolute_first_minus_final_mean": float(
            statistics.mean(abs(value) for value in differences)
        ),
        "signed_optimism_mean": float(statistics.mean(differences)),
        "first_above_final_mean_fraction": sum(value > 0 for value in differences)
        / len(differences),
        "max_first_overshoot": max(0.0, max(differences)),
    }
    top = sorted(
        (row for row in rows if row["first_minus_final_mean"] > 0),
        key=lambda row: (-row["first_minus_final_mean"], row["fragment"]),
    )[:top_n]
    return metrics, top


def _group_report(
    histories: Mapping[str, Sequence[FragmentScoreObservation]],
    *,
    registry_fragments: Sequence[str],
    top_n: int,
) -> dict[str, Any]:
    repeated, top = _repeated_fragment_metrics(histories, top_n=top_n)
    dynamic_support = _support_summary(histories)
    return {
        "support_distribution": dynamic_support,
        "dynamic_observation_support": dynamic_support,
        "registry_dynamic_support": _registry_support_summary(
            registry_fragments, histories
        ),
        "repeated_fragment_metrics": repeated,
        "top_first_score_overestimates": top,
    }


def _analyze_locked(
    paths: Mapping[str, Path],
    *,
    top_n: int,
) -> dict[str, Any]:
    implementation_path = Path(__file__).resolve()
    implementation_hash = sha256_file(implementation_path)
    raw_helper_path = getattr(experiment_io, "__file__", None)
    if raw_helper_path is None:
        raise AnalysisError("imported experiment_io helper has no filesystem identity")
    helper_path = Path(raw_helper_path).resolve()
    helper_hash = sha256_file(helper_path)
    analyzer_git = _current_git_metadata()
    analyzer_runtime = _analyzer_runtime_metadata()
    hashes_before = {name: sha256_file(path) for name, path in paths.items()}
    sizes_before = {name: path.stat().st_size for name, path in paths.items()}
    manifest = _load_json(paths["manifest"], "manifest")
    summary = _load_json(paths["summary"], "summary")
    config, identity = _validate_manifest(manifest, paths)
    summary_schema, event_count = _validate_summary(summary, manifest, config)
    state, checkpoint = _load_and_validate_checkpoint(
        paths["checkpoint"],
        manifest,
        summary,
        event_count,
    )
    replay_rows, child_indexed_scores, oracle_counts = _replay_oracle_events(
        paths["events"],
        checkpoint=checkpoint,
        config=config,
        expected_events=event_count,
        budget=int(manifest["oracle_budget"]),
        summary_schema=summary_schema,
    )
    population_state = _mapping(state.get("population"), "checkpoint.state.population")
    population_config = _validate_population_config(population_state, config)
    seed_vocabulary, vocabulary_artifact = _load_seed_vocabulary(
        manifest, population_config
    )
    histories, final_records, population_counts = _validate_population_replay(
        replay_rows,
        population_state=population_state,
        population_config=population_config,
        seed_vocabulary=seed_vocabulary,
        run_config=config,
        summary=summary,
        summary_schema=summary_schema,
    )
    _validate_summary_scores(
        summary,
        summary_schema=summary_schema,
        checkpoint=checkpoint,
        child_indexed_scores=child_indexed_scores,
        budget=int(manifest["oracle_budget"]),
        reporting_frequency=int(config["reporting_frequency"]),
    )
    hashes_after = {name: sha256_file(path) for name, path in paths.items()}
    _equal(hashes_after, hashes_before, "source hashes after analysis")
    _equal(
        {name: path.stat().st_size for name, path in paths.items()},
        sizes_before,
        "source sizes after analysis",
    )
    _equal(
        sha256_file(implementation_path),
        implementation_hash,
        "analyzer implementation hash after analysis",
    )
    _equal(
        sha256_file(helper_path),
        helper_hash,
        "experiment_io helper hash after analysis",
    )
    _equal(
        _current_git_metadata(),
        analyzer_git,
        "analyzer Git identity after analysis",
    )

    seed_fragments = {
        fragment for fragment, record in final_records.items() if record.seed_score is not None
    }
    seed_histories = {
        fragment: rows for fragment, rows in histories.items() if fragment in seed_fragments
    }
    novel_histories = {
        fragment: rows for fragment, rows in histories.items() if fragment not in seed_fragments
    }
    all_registry_fragments = sorted(final_records)
    seed_registry_fragments = sorted(seed_fragments)
    novel_registry_fragments = sorted(set(final_records) - seed_fragments)
    zero_support_seed_fragments = sorted(seed_fragments - set(seed_histories))
    groups = {
        "all": _group_report(
            histories,
            registry_fragments=all_registry_fragments,
            top_n=top_n,
        ),
        "seed": _group_report(
            seed_histories,
            registry_fragments=seed_registry_fragments,
            top_n=top_n,
        ),
        "novel": _group_report(
            novel_histories,
            registry_fragments=novel_registry_fragments,
            top_n=top_n,
        ),
    }
    all_group = groups["all"]
    sources = {
        name: {
            "path": str(paths[name]),
            "sha256": hashes_before[name],
            "size_bytes": sizes_before[name],
        }
        for name in ("manifest", "summary", "events", "checkpoint")
    }
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "created_at": _utc_timestamp(),
        "analysis": "validated descriptive fragment context-variance diagnostic",
        "analysis_scope": "adaptive on-policy single-run association; not causal",
        "causal_claim": False,
        "oracle_call_efficiency_claim": False,
        "run_identity": identity,
        "config": config,
        "source": sources,
        "referenced_artifacts": {
            "model": {
                "path": identity["model_path"],
                "sha256": identity["model_sha256"],
                "validation": "identity reconciled across config, manifest, summary, and checkpoint metadata",
            },
            "vocabulary": vocabulary_artifact,
        },
        "analyzer": {
            "implementation": _source_entry(implementation_path, implementation_hash),
            "imported_helpers": {
                "experiment_io": _source_entry(helper_path, helper_hash),
            },
            "git": analyzer_git,
            "runtime": analyzer_runtime,
            "arguments": {
                "events": str(paths["events"]),
                "manifest": str(paths["manifest"]),
                "summary": str(paths["summary"]),
                "checkpoint": str(paths["checkpoint"]),
                "top_n": top_n,
            },
        },
        "validation": {
            "manifest_schema_version": 1,
            "summary_schema_version": summary_schema,
            "checkpoint_population_state_version": POPULATION_STATE_VERSION,
            "global_oracle_replay": "passed",
            "statistical_population_replay": "passed",
            "summary_metric_recomputation": "passed",
            "event_fragment_evidence": {
                "observed_fragments": (
                    "structurally reconciled with logged fragment_statistics_after "
                    "and checkpoint population state"
                ),
                "selected_fragments": (
                    "structurally checked as two distinct members of the replayed "
                    "pre-update active population"
                ),
                "chemical_decomposition_recomputed": False,
                "fragmentation_rng_replayed": False,
                "selection_rng_replayed": False,
                "claim_boundary": (
                    "The analyzer validates internal record consistency; it does not "
                    "independently prove that molecular fragmentation or random selection "
                    "produced the logged fragment strings."
                ),
            },
        },
        "counts": {**oracle_counts, **population_counts},
        "fragment_classification": {
            "checkpoint_registry_fragments": len(final_records),
            "checkpoint_seed_fragments": len(seed_fragments),
            "checkpoint_novel_fragments": len(final_records) - len(seed_fragments),
            "classification_rule": "checkpoint population record seed_score is non-null",
            "seed_fragments_with_zero_dynamic_support": zero_support_seed_fragments,
            "seed_fragments_with_zero_dynamic_support_count": len(
                zero_support_seed_fragments
            ),
        },
        "fragment_groups": groups,
        "support_distribution": all_group["support_distribution"],
        "repeated_fragment_metrics": all_group["repeated_fragment_metrics"],
        "top_first_score_overestimates": all_group["top_first_score_overestimates"],
        "observation_semantics": (
            "One whole-child score per unique fragment emitted by the run's sampled cut "
            "decomposition for each unique canonical child accepted by the statistical "
            "population update. Seed/novel labels come from checkpoint seed_score."
        ),
        "metric_definitions": {
            "within_fragment_sample_sd": (
                "Sample SD of whole-molecule scores across logged sampled-cut contexts for "
                "one fragment; defined only for dynamic support >= 2."
            ),
            "pooled_within_fragment_sample_sd": (
                "sqrt(sum_f sum_i (y_fi - mean_f)^2 / sum_f (n_f - 1))."
            ),
            "first_minus_final_mean": (
                "First logged whole-molecule score minus that fragment's mean over all "
                "logged dynamic observations in this run, including the first."
            ),
            "seed_fragment": "A checkpoint population record with non-null seed_score.",
            "novel_fragment": "A checkpoint population record with null seed_score.",
        },
        "caveats": [
            (
                "Sampled decomposition caveat: the run uses one stochastic cut decomposition "
                "per accepted child. These observations do not enumerate every fragment in "
                "the molecule and are not Eq. 5's set of all molecules containing a fragment."
            ),
            (
                "Replay-boundary caveat: observed_fragments and selected_fragments are "
                "structurally validated against logged statistics and active state, but the "
                "chemical cut operation and fragmentation/selection RNG streams are not "
                "independently recomputed."
            ),
            (
                "Efficiency caveat: context variance and first-score overshoot in one stream "
                "cannot establish fewer optimization steps or oracle calls. Oracle-call "
                "efficiency requires a controlled, fixed-budget, multi-seed ablation."
            ),
            (
                "Adaptive/on-policy caveat: the arm changes which fragments and contexts are "
                "generated, so observations are not iid or directly exchangeable across arms."
            ),
            (
                "Non-causal caveat: each value is a whole-molecule score associated with every "
                "logged fragment; co-fragments and molecular context confound attribution."
            ),
            (
                "Same-stream target caveat: the final mean includes the first observation and "
                "is not a held-out estimate."
            ),
            (
                "Support caveat: singleton fragments are excluded from repeated-context "
                "metrics; all, seed, and novel support distributions must accompany them."
            ),
            (
                "Initialization caveat: seed/novel classification is validated from the "
                "checkpoint, but offline seed-vocabulary observations are not added to the "
                "dynamic context histories."
            ),
            (
                "Legacy-record caveat: legacy_terminal_event_omissions reports completed "
                "schema-1 parent-control streams whose final parent-only event predates the "
                "timing/population-field repair; no missing value is imputed."
            ),
        ],
    }


def analyze_fragment_contexts(
    events_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    top_n: int = 20,
) -> dict[str, Any]:
    """Validate a completed run under its shared lock and return a report."""

    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise AnalysisError("top_n must be a positive integer")
    paths = _resolve_paths(events_path, manifest_path, summary_path, checkpoint_path)
    lock_path = paths["events"].parent / ".run.lock"
    try:
        lock = FileLock(lock_path, fcntl.LOCK_SH, create=False)
    except BlockingIOError as error:
        raise AnalysisError(f"run directory is active: {paths['events'].parent}") from error
    with lock:
        return _analyze_locked(paths, top_n=top_n)


def write_analysis(
    events_path: str | Path,
    output_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    top_n: int = 20,
    overwrite: bool = False,
) -> Path:
    """Validate sources and atomically publish one concurrency-safe JSON report."""

    paths = _resolve_paths(events_path, manifest_path, summary_path, checkpoint_path)
    output = Path(output_path).expanduser().resolve()
    run_lock_path = paths["events"].parent / ".run.lock"
    try:
        source_lock = FileLock(run_lock_path, fcntl.LOCK_SH, create=False)
    except BlockingIOError as error:
        raise AnalysisError(f"run directory is active: {paths['events'].parent}") from error
    with source_lock:
        manifest = _load_json(paths["manifest"], "manifest")
        protected_paths = _output_protected_paths(paths, manifest)
        for label, protected in protected_paths.items():
            if _paths_alias(output, protected):
                raise AnalysisError(
                    f"output path aliases protected {label} path: {protected}"
                )
    destination_lock_path = output.with_name(f".{output.name}.lock")
    try:
        destination_lock = FileLock(destination_lock_path, fcntl.LOCK_EX, create=True)
    except BlockingIOError as error:
        raise AnalysisError(f"analysis destination is active: {output}") from error
    with destination_lock:
        if output.exists() and not overwrite:
            raise FileExistsError(output)
        report = analyze_fragment_contexts(
            paths["events"],
            manifest_path=paths["manifest"],
            summary_path=paths["summary"],
            checkpoint_path=paths["checkpoint"],
            top_n=top_n,
        )
        report["analyzer"]["arguments"]["output"] = str(output)
        report["analyzer"]["arguments"]["overwrite"] = overwrite
        return write_manifest(output, report, overwrite=overwrite)


def main() -> None:
    args = _parse_args()
    destination = write_analysis(
        args.events,
        args.output,
        manifest_path=args.manifest,
        summary_path=args.summary,
        checkpoint_path=args.checkpoint,
        top_n=args.top_n,
        overwrite=args.overwrite,
    )
    print(f"Results: {destination}", flush=True)


if __name__ == "__main__":
    main()

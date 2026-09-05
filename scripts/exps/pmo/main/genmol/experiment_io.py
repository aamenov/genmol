"""Small, dependency-free I/O helpers for reproducible PMO experiments.

The experiment runner is intentionally kept separate from these utilities.  This
module only owns durable records: a reproducibility manifest, an append-only
event stream, restart state, and metrics derived from scores in oracle-call
order.  Checkpoints use pickle and must therefore only be loaded from trusted
local experiment directories.
"""

from __future__ import annotations

import dataclasses
import datetime as datetime_module
import hashlib
import json
import math
import os
import pickle
import platform
import socket
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1


def _json_compatible(value: Any) -> Any:
    """Return a deterministic, JSON-compatible representation of ``value``."""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)

    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("NaN and infinite values are not valid experiment metadata")
        return value
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("experiment metadata mappings must use string keys")
            result[key] = _json_compatible(item)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_json_compatible(item) for item in value]
        return sorted(items, key=lambda item: _canonical_json_bytes(item))
    raise TypeError(f"unsupported experiment metadata value: {type(value).__name__}")


def _canonical_json_bytes(value: Any) -> bytes:
    normalized = _json_compatible(value)
    return json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading a potentially large model into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_config(config: Any) -> str:
    """Hash resolved configuration values independently of mapping key order."""

    return hashlib.sha256(_canonical_json_bytes(config)).hexdigest()


def _utc_timestamp() -> str:
    now = datetime_module.datetime.now(datetime_module.timezone.utc)
    return now.isoformat(timespec="seconds").replace("+00:00", "Z")


def build_manifest(
    *,
    run_id: str,
    model_path: str | os.PathLike[str],
    config: Any,
    task: str | None = None,
    variant: str | None = None,
    seed: int | None = None,
    oracle_budget: int | None = None,
    extra: Mapping[str, Any] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a JSON-ready manifest containing model and config identities."""

    if not run_id:
        raise ValueError("run_id must be non-empty")
    if oracle_budget is not None and oracle_budget <= 0:
        raise ValueError("oracle_budget must be positive")

    resolved_model_path = Path(model_path).expanduser().resolve()
    model_stat = resolved_model_path.stat()
    resolved_config = _json_compatible(config)
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": created_at or _utc_timestamp(),
        "model": {
            "path": str(resolved_model_path),
            "size_bytes": model_stat.st_size,
            "sha256": sha256_file(resolved_model_path),
        },
        "config": resolved_config,
        "config_sha256": sha256_config(resolved_config),
        "runtime": {
            "hostname": socket.gethostname(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "executable": sys.executable,
        },
    }
    optional_values = {
        "task": task,
        "variant": variant,
        "seed": seed,
        "oracle_budget": oracle_budget,
    }
    manifest.update({key: value for key, value in optional_values.items() if value is not None})
    if extra is not None:
        manifest["extra"] = _json_compatible(extra)
    return manifest


def _fsync_directory(directory: Path) -> None:
    """Best-effort persistence of a rename on platforms supporting directory fsync."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return path


def write_manifest(
    path: str | os.PathLike[str],
    manifest: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write a manifest, refusing accidental replacement by default."""

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    payload = json.dumps(
        _json_compatible(manifest),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    return _atomic_write_bytes(destination, payload)


def append_event(
    path: str | os.PathLike[str],
    event: Mapping[str, Any],
    *,
    durable: bool = False,
) -> int:
    """Append one canonical JSON object and return the resulting line length.

    Serializing before opening the file ensures an unsupported event cannot
    leave a partial line.  ``O_APPEND`` prevents this helper from rewriting an
    existing event stream.
    """

    if not isinstance(event, Mapping):
        raise TypeError("event must be a mapping")
    payload = _canonical_json_bytes(event) + b"\n"
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("failed to append event")
            view = view[written:]
        if durable:
            os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return len(payload)


def iter_events(
    path: str | os.PathLike[str],
    *,
    tolerate_truncated_last_line: bool = False,
) -> Iterator[dict[str, Any]]:
    """Yield JSONL events, optionally ignoring an interrupted final append."""

    with Path(path).open("rb") as handle:
        line_number = 0
        while True:
            raw_line = handle.readline()
            if not raw_line:
                return
            line_number += 1
            if not raw_line.endswith(b"\n"):
                if tolerate_truncated_last_line and not handle.read(1):
                    return
                raise ValueError(f"event log line {line_number} is truncated")
            try:
                event = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid JSON on event log line {line_number}") from error
            if not isinstance(event, dict):
                raise ValueError(f"event log line {line_number} is not a JSON object")
            yield event


class JsonlEventLog:
    """Convenience wrapper around an append-only JSONL path."""

    def __init__(self, path: str | os.PathLike[str], *, durable: bool = False):
        self.path = Path(path)
        self.durable = durable

    def append(self, event: Mapping[str, Any]) -> int:
        return append_event(self.path, event, durable=self.durable)

    def events(self, *, tolerate_truncated_last_line: bool = False) -> Iterator[dict[str, Any]]:
        return iter_events(
            self.path,
            tolerate_truncated_last_line=tolerate_truncated_last_line,
        )


def save_checkpoint(
    path: str | os.PathLike[str],
    state: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically pickle restart state, including RNG states and oracle buffers."""

    envelope = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "metadata": dict(metadata or {}),
        "state": state,
    }
    payload = pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)
    return _atomic_write_bytes(Path(path), payload)


def load_checkpoint(
    path: str | os.PathLike[str],
    *,
    with_metadata: bool = False,
) -> Any:
    """Load trusted local restart state written by :func:`save_checkpoint`."""

    with Path(path).open("rb") as handle:
        envelope = pickle.load(handle)
    if not isinstance(envelope, dict) or envelope.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported experiment checkpoint schema")
    if "state" not in envelope or "metadata" not in envelope:
        raise ValueError("malformed experiment checkpoint")
    if with_metadata:
        return envelope["state"], envelope["metadata"]
    return envelope["state"]


def _validated_scores(scores: Sequence[float], budget: int | None = None) -> list[float]:
    if budget is not None and budget <= 0:
        raise ValueError("budget must be positive")
    selected = scores if budget is None else scores[:budget]
    result = [float(score) for score in selected]
    if any(not math.isfinite(score) for score in result):
        raise ValueError("scores must be finite")
    return result


def top_k_mean(scores: Sequence[float], k: int) -> float:
    """Mean of the best available ``k`` scores."""

    if k <= 0:
        raise ValueError("k must be positive")
    values = _validated_scores(scores)
    if not values:
        raise ValueError("at least one score is required")
    return sum(sorted(values, reverse=True)[:k]) / min(k, len(values))


def top_k_trajectory(
    scores: Sequence[float],
    *,
    k: int = 10,
    reporting_frequency: int = 100,
    budget: int | None = None,
    pad_to_budget: bool = False,
) -> list[dict[str, int | float]]:
    """Return top-k means versus unique oracle calls in arrival order.

    The origin is fixed at zero, matching the released PMO AUC convention.  A
    final partial interval is retained, and an early-stopped curve can be held
    constant through ``budget``.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    if reporting_frequency <= 0:
        raise ValueError("reporting_frequency must be positive")
    values = _validated_scores(scores, budget)
    points: list[dict[str, int | float]] = [{"oracle_calls": 0, "top_k_mean": 0.0}]

    for call_count in range(reporting_frequency, len(values) + 1, reporting_frequency):
        points.append(
            {
                "oracle_calls": call_count,
                "top_k_mean": top_k_mean(values[:call_count], k),
            }
        )
    if values and points[-1]["oracle_calls"] != len(values):
        points.append(
            {
                "oracle_calls": len(values),
                "top_k_mean": top_k_mean(values, k),
            }
        )
    if pad_to_budget:
        if budget is None:
            raise ValueError("budget is required when pad_to_budget is true")
        if points[-1]["oracle_calls"] < budget:
            points.append(
                {
                    "oracle_calls": budget,
                    "top_k_mean": points[-1]["top_k_mean"],
                }
            )
    return points


def trajectory_auc(
    trajectory: Sequence[Mapping[str, int | float]],
    *,
    normalize_by: int | float | None = None,
) -> float:
    """Integrate a trajectory with the trapezoidal rule."""

    if not trajectory:
        raise ValueError("trajectory must contain at least one point")
    area = 0.0
    previous_x = float(trajectory[0]["oracle_calls"])
    previous_y = float(trajectory[0]["top_k_mean"])
    if not math.isfinite(previous_x) or not math.isfinite(previous_y):
        raise ValueError("trajectory values must be finite")
    for point in trajectory[1:]:
        x_value = float(point["oracle_calls"])
        y_value = float(point["top_k_mean"])
        if not math.isfinite(x_value) or not math.isfinite(y_value):
            raise ValueError("trajectory values must be finite")
        if x_value <= previous_x:
            raise ValueError("trajectory oracle_calls must be strictly increasing")
        area += (x_value - previous_x) * (previous_y + y_value) / 2.0
        previous_x, previous_y = x_value, y_value
    if normalize_by is not None:
        if normalize_by <= 0:
            raise ValueError("normalize_by must be positive")
        area /= normalize_by
    return area


def top_k_auc(
    scores: Sequence[float],
    *,
    k: int = 10,
    reporting_frequency: int = 100,
    budget: int = 10_000,
    pad_to_budget: bool = True,
) -> float:
    """Compute normalized PMO top-k AUC under a fixed unique-call budget."""

    trajectory = top_k_trajectory(
        scores,
        k=k,
        reporting_frequency=reporting_frequency,
        budget=budget,
        pad_to_budget=pad_to_budget,
    )
    return trajectory_auc(trajectory, normalize_by=budget)


def top_k_trajectory_at_calls(
    indexed_scores: Sequence[tuple[int, float]],
    *,
    k: int = 10,
    reporting_frequency: int = 100,
    observed_oracle_calls: int,
    budget: int = 10_000,
) -> list[dict[str, int | float]]:
    """Return a top-k curve on the *global* unique-oracle-call axis.

    ``indexed_scores`` can be a subset of charged calls, such as children from
    an arm that also charges parent molecules.  Retaining each global call
    index prevents those children from being compressed into fictitious early
    calls when computing sample-efficiency AUC.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    if reporting_frequency <= 0:
        raise ValueError("reporting_frequency must be positive")
    if budget <= 0:
        raise ValueError("budget must be positive")
    if not 0 <= observed_oracle_calls <= budget:
        raise ValueError("observed_oracle_calls must lie in [0, budget]")

    validated: list[tuple[int, float]] = []
    previous_call = 0
    for raw_call, raw_score in indexed_scores:
        call = int(raw_call)
        score = float(raw_score)
        if call != raw_call or call <= previous_call:
            raise ValueError("indexed score call indices must be strictly increasing integers")
        if call > observed_oracle_calls:
            raise ValueError("indexed score exceeds observed_oracle_calls")
        if not math.isfinite(score):
            raise ValueError("scores must be finite")
        validated.append((call, score))
        previous_call = call

    checkpoints = list(range(reporting_frequency, observed_oracle_calls + 1, reporting_frequency))
    if observed_oracle_calls and (not checkpoints or checkpoints[-1] != observed_oracle_calls):
        checkpoints.append(observed_oracle_calls)
    if observed_oracle_calls < budget:
        checkpoints.append(budget)

    points: list[dict[str, int | float]] = [{"oracle_calls": 0, "top_k_mean": 0.0}]
    seen: list[float] = []
    score_index = 0
    for checkpoint in checkpoints:
        while score_index < len(validated) and validated[score_index][0] <= checkpoint:
            seen.append(validated[score_index][1])
            score_index += 1
        points.append(
            {
                "oracle_calls": checkpoint,
                "top_k_mean": 0.0 if not seen else top_k_mean(seen, k),
            }
        )
    return points


def summarize_indexed_scores(
    indexed_scores: Sequence[tuple[int, float]],
    *,
    observed_oracle_calls: int,
    ks: Sequence[int] = (1, 10, 100),
    reporting_frequency: int = 100,
    budget: int = 10_000,
) -> dict[str, Any]:
    """Summarize a score subset on the global unique-call budget axis."""

    rows = list(indexed_scores)
    # Validate once even when ``ks`` is empty and obtain JSON-ready values.
    validation_trajectory = top_k_trajectory_at_calls(
        rows,
        k=1,
        reporting_frequency=reporting_frequency,
        observed_oracle_calls=observed_oracle_calls,
        budget=budget,
    )
    del validation_trajectory
    values = [float(score) for _, score in rows]
    summary: dict[str, Any] = {
        "axis": "total_unique_oracle_calls",
        "score_count": len(values),
        "oracle_calls": observed_oracle_calls,
        "oracle_budget": budget,
        "reporting_frequency": reporting_frequency,
    }
    for k in ks:
        if k <= 0:
            raise ValueError("all k values must be positive")
        label = f"top_{k}"
        trajectory = top_k_trajectory_at_calls(
            rows,
            k=k,
            reporting_frequency=reporting_frequency,
            observed_oracle_calls=observed_oracle_calls,
            budget=budget,
        )
        summary[label] = None if not values else top_k_mean(values, k)
        summary[f"auc_{label}"] = trajectory_auc(trajectory, normalize_by=budget)
        summary[f"trajectory_{label}"] = trajectory
    return summary


def summarize_scores(
    scores: Sequence[float],
    *,
    ks: Sequence[int] = (1, 10, 100),
    reporting_frequency: int = 100,
    budget: int = 10_000,
) -> dict[str, Any]:
    """Build JSON-ready final-score, trajectory, and AUC summaries."""

    values = _validated_scores(scores, budget)
    summary: dict[str, Any] = {
        "oracle_calls": len(values),
        "oracle_budget": budget,
        "reporting_frequency": reporting_frequency,
    }
    for k in ks:
        if k <= 0:
            raise ValueError("all k values must be positive")
        label = f"top_{k}"
        trajectory = top_k_trajectory(
            values,
            k=k,
            reporting_frequency=reporting_frequency,
            budget=budget,
            pad_to_budget=True,
        )
        summary[label] = None if not values else top_k_mean(values, k)
        summary[f"auc_{label}"] = trajectory_auc(trajectory, normalize_by=budget)
        summary[f"trajectory_{label}"] = trajectory
    return summary


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "JsonlEventLog",
    "MANIFEST_SCHEMA_VERSION",
    "append_event",
    "build_manifest",
    "iter_events",
    "load_checkpoint",
    "save_checkpoint",
    "sha256_config",
    "sha256_file",
    "summarize_indexed_scores",
    "summarize_scores",
    "top_k_auc",
    "top_k_mean",
    "top_k_trajectory",
    "top_k_trajectory_at_calls",
    "trajectory_auc",
    "write_manifest",
]

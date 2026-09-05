"""Launch reproducible de novo benchmark runs on explicitly selected GPUs.

The controller is intended to run inside ``tmux``.  The caller supplies both
the GPU count and the exact physical device indices.  A selected device is
eligible when its utilization is strictly below the requested threshold, it
has enough free memory, its compute mode is not prohibited, and it has no
active compute process.  Every device is rechecked
immediately before a child starts and is then mapped into the child by UUID.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
REPOSITORY_SRC = REPOSITORY_ROOT / "src"
for import_root in (REPOSITORY_ROOT, REPOSITORY_SRC):
    while str(import_root) in sys.path:
        sys.path.remove(str(import_root))
    sys.path.insert(0, str(import_root))

from scripts.exps.denovo import benchmark as benchmark_runner  # noqa: E402


class CompletionArtifactError(RuntimeError):
    """Raised when a seed directory cannot safely be skipped or relaunched."""


@dataclass(frozen=True)
class GPUState:
    index: int
    uuid: str
    name: str
    memory_used_mib: int
    memory_total_mib: int
    utilization_percent: int
    compute_mode: str
    compute_processes: tuple[dict[str, Any], ...]

    def rejection_reasons(
        self,
        *,
        max_utilization_percent: int,
        min_free_memory_mib: int,
    ) -> list[str]:
        reasons = []
        free_memory_mib = self.memory_total_mib - self.memory_used_mib
        if free_memory_mib < min_free_memory_mib:
            reasons.append(
                f"{free_memory_mib} MiB free is below the required "
                f"{min_free_memory_mib} MiB"
            )
        if self.utilization_percent >= max_utilization_percent:
            reasons.append(
                f"{self.utilization_percent}% utilization is not strictly below "
                f"{max_utilization_percent}%"
            )
        if self.compute_mode.lower() == "prohibited":
            reasons.append("compute mode is prohibited")
        if self.compute_processes:
            reasons.append(
                f"{len(self.compute_processes)} active compute process(es) detected"
            )
        return reasons

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "uuid": self.uuid,
            "name": self.name,
            "memory_used_mib": self.memory_used_mib,
            "memory_total_mib": self.memory_total_mib,
            "utilization_percent": self.utilization_percent,
            "compute_mode": self.compute_mode,
            "compute_processes": list(self.compute_processes),
        }


@dataclass
class RunningJob:
    seed: int
    gpu: GPUState
    process: subprocess.Popen
    log_handle: Any
    log_path: Path


@dataclass(frozen=True)
class ExpectedRunIdentity:
    """Inputs that must match before an existing seed is considered complete."""

    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_global_step: int
    checkpoint_size_bytes: int
    checkpoint_diffusion_type: str
    checkpoint_udlm_inference_eps: float | None
    checkpoint_udlm_exclude_special_tokens: bool | None
    config_path: Path
    source_config: Mapping[str, Any]
    source_config_sha256: str
    sampling_config: Mapping[str, Any]
    sampling_config_sha256: str
    effective_config: Mapping[str, Any]
    effective_config_sha256: str
    benchmark_runner_sha256: str
    implementation_inputs: Mapping[str, Any]
    num_samples: int
    device: str = "cuda:0"


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=1_000)
    parser.add_argument(
        "--pilot",
        action="store_true",
        help=(
            "Allow a deliberately small run of at most 100 samples. Without this "
            "flag, the final protocol requires exactly 1000 samples per seed."
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--gpu-count",
        type=int,
        required=True,
        help="Number of benchmark GPUs requested by the user.",
    )
    parser.add_argument(
        "--gpu-indices",
        type=int,
        nargs="+",
        required=True,
        metavar="PHYSICAL_INDEX",
        help="Exact physical GPU indices authorized by the user.",
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-utilization-percent", type=int, default=10)
    parser.add_argument("--min-free-memory-mib", type=int, default=30_000)
    parser.add_argument("--log-root", type=Path, default=Path("output/logs"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _validate_gpu_request(
    gpu_count: int,
    gpu_indices: Sequence[int],
) -> tuple[int, ...]:
    """Validate and freeze the caller's explicit physical-device selection."""

    if not 1 <= gpu_count <= 2:
        raise ValueError("gpu-count must be 1 or 2")
    selected = tuple(gpu_indices)
    if any(index < 0 for index in selected):
        raise ValueError("gpu-indices must be non-negative physical device IDs")
    unique_count = len(set(selected))
    if unique_count != len(selected):
        raise ValueError("gpu-indices must contain unique physical device IDs")
    if unique_count != gpu_count:
        raise ValueError(
            "the unique --gpu-indices count must equal --gpu-count "
            f"({unique_count} != {gpu_count})"
        )
    return selected


def _validate_sample_tier(num_samples: int, *, pilot: bool) -> str:
    """Prevent a pilot from being mistaken for the final benchmark."""

    if num_samples <= 0:
        raise ValueError("num-samples must be positive")
    if pilot:
        if num_samples > 100:
            raise ValueError("pilot runs are capped at 100 samples per seed")
        return "pilot"
    if num_samples != 1_000:
        raise ValueError(
            "final benchmark runs require exactly 1000 samples per seed; "
            "pass --pilot for a run of at most 100"
        )
    return "final"


def _selection_policy(
    *,
    selected_physical_indices: Sequence[int],
    max_utilization_percent: int,
    min_free_memory_mib: int,
) -> dict[str, Any]:
    """Return the policy schema embedded in controller and child provenance."""

    return {
        "max_utilization_percent": max_utilization_percent,
        "utilization_comparison": "strictly_less_than",
        "min_free_memory_mib": min_free_memory_mib,
        "active_compute_processes_allowed": False,
        "selected_physical_indices": list(selected_physical_indices),
    }


def _resolve_in_repo(path: Path) -> Path:
    resolved = path if path.is_absolute() else REPOSITORY_ROOT / path
    resolved = resolved.resolve()
    if resolved != REPOSITORY_ROOT and REPOSITORY_ROOT not in resolved.parents:
        raise ValueError(f"path escapes repository root: {resolved}")
    return resolved


def _resolve_checkpoint(path: Path) -> Path:
    """Allow shared checkpoints in the containing project, never outside it."""

    resolved = path if path.is_absolute() else REPOSITORY_ROOT / path
    resolved = resolved.resolve()
    project_root = (
        REPOSITORY_ROOT.parent.parent
        if REPOSITORY_ROOT.parent.name == "run_sources"
        else REPOSITORY_ROOT
    )
    if resolved != project_root and project_root not in resolved.parents:
        raise ValueError(f"checkpoint escapes project root: {resolved}")
    return resolved


def _run_nvidia_smi(gpu_index: int, query: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu_index),
            query,
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def _physical_gpu_indices() -> list[int]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    indices = [int(line.strip()) for line in completed.stdout.splitlines() if line.strip()]
    if not indices or len(indices) != len(set(indices)):
        raise RuntimeError("nvidia-smi returned no GPUs or duplicate physical indices")
    return indices


def _probe_gpu(gpu_index: int) -> GPUState:
    status = _run_nvidia_smi(
        gpu_index,
        "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,compute_mode",
    )
    if status.returncode or status.stderr.strip():
        detail = status.stderr.strip() or status.stdout.strip()
        raise RuntimeError(f"GPU {gpu_index} status query failed: {detail}")
    rows = [
        [field.strip() for field in row]
        for row in csv.reader(io.StringIO(status.stdout), skipinitialspace=True)
        if any(field.strip() for field in row)
    ]
    if len(rows) != 1 or len(rows[0]) != 7:
        raise RuntimeError(f"GPU {gpu_index} status query returned an unexpected row")
    (
        index_text,
        uuid,
        name,
        memory_used_text,
        memory_total_text,
        utilization_text,
        compute_mode,
    ) = rows[0]
    if int(index_text) != gpu_index or not uuid.startswith("GPU-"):
        raise RuntimeError(f"GPU {gpu_index} status query returned inconsistent identity")
    memory_used_mib = int(memory_used_text)
    memory_total_mib = int(memory_total_text)
    utilization_percent = int(utilization_text)
    if (
        memory_used_mib < 0
        or memory_total_mib <= 0
        or memory_used_mib > memory_total_mib
        or not 0 <= utilization_percent <= 100
    ):
        raise RuntimeError(f"GPU {gpu_index} status query returned invalid telemetry")

    processes = _run_nvidia_smi(
        gpu_index,
        "--query-compute-apps=pid,process_name,used_memory",
    )
    if processes.returncode or processes.stderr.strip():
        detail = processes.stderr.strip() or processes.stdout.strip()
        raise RuntimeError(f"GPU {gpu_index} process query failed: {detail}")
    process_rows = []
    for row in csv.reader(io.StringIO(processes.stdout), skipinitialspace=True):
        fields = [field.strip() for field in row]
        if not any(fields) or fields[0].lower().startswith("no running"):
            continue
        if len(fields) != 3 or not fields[0].isdigit():
            raise RuntimeError(f"GPU {gpu_index} process query returned an unexpected row")
        process_rows.append(
            {
                "pid": int(fields[0]),
                "process_name": fields[1],
                "used_memory_mib": int(fields[2]),
            }
        )
    return GPUState(
        index=gpu_index,
        uuid=uuid,
        name=name,
        memory_used_mib=memory_used_mib,
        memory_total_mib=memory_total_mib,
        utilization_percent=utilization_percent,
        compute_mode=compute_mode,
        compute_processes=tuple(process_rows),
    )


def _snapshot(selected_physical_indices: Sequence[int]) -> list[GPUState]:
    """Probe only the exact physical devices authorized by the caller."""

    available_indices = set(_physical_gpu_indices())
    missing_indices = sorted(set(selected_physical_indices) - available_indices)
    if missing_indices:
        raise RuntimeError(
            "Requested physical GPU indices are unavailable: "
            f"{missing_indices}; available indices: {sorted(available_indices)}"
        )
    states = []
    for index in selected_physical_indices:
        try:
            states.append(_probe_gpu(index))
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
            print(f"GPU {index} rejected because its state is unverifiable: {error}", flush=True)
    return states


def _eligible(
    state: GPUState,
    *,
    max_utilization_percent: int,
    min_free_memory_mib: int,
) -> bool:
    return not state.rejection_reasons(
        max_utilization_percent=max_utilization_percent,
        min_free_memory_mib=min_free_memory_mib,
    )


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_text(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _require_clean_pushed_source() -> dict[str, str]:
    """Require immutable, upstream-backed code while allowing output artifacts."""

    status = _git_text("status", "--porcelain=v1", "--untracked-files=normal")
    disallowed = []
    for line in status.splitlines():
        path_text = line[3:]
        if " -> " in path_text:
            path_text = path_text.split(" -> ", 1)[1]
        if path_text != "output" and not path_text.startswith("output/"):
            disallowed.append(line)
    if disallowed:
        raise RuntimeError(
            "benchmark source worktree is dirty outside output/: "
            + "; ".join(disallowed)
        )

    head = _git_text("rev-parse", "HEAD")
    try:
        upstream = _git_text("rev-parse", "@{upstream}")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "benchmark branch has no upstream; commit and push it before launch"
        ) from exc
    if head != upstream:
        raise RuntimeError(
            f"benchmark HEAD {head} is not the pushed upstream commit {upstream}"
        )
    return {"head": head, "upstream": upstream}


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_expected_run_identity(
    checkpoint: Path,
    config: Path,
    num_samples: int,
) -> ExpectedRunIdentity:
    """Resolve and fingerprint the exact inputs passed to every child run."""

    checkpoint_info = benchmark_runner.checkpoint_metadata(checkpoint)
    source_config = benchmark_runner.load_yaml_config(config)
    sampling_config = benchmark_runner.validate_sampling_config(source_config)
    checkpoint_diffusion_type = str(
        checkpoint_info.get("diffusion_type", "mdlm")
    ).lower()
    if sampling_config["diffusion_type"] != checkpoint_diffusion_type:
        raise ValueError(
            "config diffusion_type does not match checkpoint metadata: "
            f"{sampling_config['diffusion_type']!r} != {checkpoint_diffusion_type!r}"
        )
    checkpoint_udlm_inference_eps = checkpoint_info.get("udlm_inference_eps")
    checkpoint_udlm_exclude_special_tokens = checkpoint_info.get(
        "udlm_exclude_special_tokens"
    )
    if checkpoint_diffusion_type == "udlm":
        if checkpoint_udlm_inference_eps is None or not math.isclose(
            float(checkpoint_udlm_inference_eps),
            float(sampling_config["inference_eps"]),
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "config inference_eps does not match checkpoint metadata: "
                f"{sampling_config['inference_eps']} != "
                f"{checkpoint_udlm_inference_eps}"
            )
        if (
            checkpoint_udlm_exclude_special_tokens
            is not sampling_config["exclude_special_tokens"]
        ):
            raise ValueError(
                "config exclude_special_tokens does not match checkpoint metadata"
            )
    implementation_inputs = benchmark_runner.implementation_input_provenance()
    benchmark_runner_path = Path(benchmark_runner.__file__).resolve()
    effective_config = dict(source_config)
    effective_config.update(
        {
            "model_path": str(checkpoint),
            "num_samples": num_samples,
            "device": "cuda:0",
        }
    )
    return ExpectedRunIdentity(
        checkpoint_path=checkpoint,
        checkpoint_sha256=str(checkpoint_info["sha256"]),
        checkpoint_global_step=int(checkpoint_info["global_step"]),
        checkpoint_size_bytes=int(checkpoint_info["size_bytes"]),
        checkpoint_diffusion_type=checkpoint_diffusion_type,
        checkpoint_udlm_inference_eps=(
            float(checkpoint_udlm_inference_eps)
            if checkpoint_udlm_inference_eps is not None
            else None
        ),
        checkpoint_udlm_exclude_special_tokens=(
            bool(checkpoint_udlm_exclude_special_tokens)
            if checkpoint_udlm_exclude_special_tokens is not None
            else None
        ),
        config_path=config,
        source_config=source_config,
        source_config_sha256=_sha256_file(config),
        sampling_config=sampling_config,
        sampling_config_sha256=_canonical_json_sha256(sampling_config),
        effective_config=effective_config,
        effective_config_sha256=_canonical_json_sha256(effective_config),
        benchmark_runner_sha256=_sha256_file(benchmark_runner_path),
        implementation_inputs=implementation_inputs,
        num_samples=num_samples,
    )


def _recorded_path(value: Any) -> Optional[Path]:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _validate_raw_csv(
    samples_path: Path,
    *,
    expected_row_count: int,
    expected_fields: tuple[str, ...],
) -> tuple[str, int, tuple[str, ...], list[str]]:
    """Return the CSV digest and shape while retaining every schema error."""

    errors: list[str] = []
    actual_fields: tuple[str, ...] = ()
    actual_row_count = 0
    try:
        with samples_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            try:
                actual_fields = tuple(next(reader))
            except StopIteration:
                errors.append("raw_samples.csv is empty and has no header")
                return _sha256_file(samples_path), 0, actual_fields, errors

            sample_index_column = (
                actual_fields.index("sample_index")
                if "sample_index" in actual_fields
                else None
            )
            for expected_index, row in enumerate(reader):
                actual_row_count += 1
                if len(row) != len(actual_fields):
                    errors.append(
                        "raw_samples.csv row "
                        f"{actual_row_count} has {len(row)} columns; expected "
                        f"{len(actual_fields)}"
                    )
                    continue
                if sample_index_column is not None:
                    try:
                        observed_index = int(row[sample_index_column])
                    except ValueError:
                        errors.append(
                            f"raw_samples.csv row {actual_row_count} has a non-integer "
                            "sample_index"
                        )
                    else:
                        if observed_index != expected_index:
                            errors.append(
                                f"raw_samples.csv row {actual_row_count} has sample_index "
                                f"{observed_index}; expected {expected_index}"
                            )
    except (OSError, csv.Error, UnicodeError) as error:
        errors.append(f"raw_samples.csv could not be parsed: {error}")

    if actual_fields != expected_fields:
        errors.append(
            "raw_samples.csv header/schema differs from benchmark RAW_SAMPLE_FIELDS"
        )
    if actual_row_count != expected_row_count:
        errors.append(
            f"raw_samples.csv has {actual_row_count} data rows; expected "
            f"{expected_row_count}"
        )
    try:
        digest = _sha256_file(samples_path)
    except OSError as error:
        errors.append(f"raw_samples.csv could not be hashed: {error}")
        digest = ""
    return digest, actual_row_count, actual_fields, errors


def _completed(
    output_root: Path,
    seed: int,
    expected: ExpectedRunIdentity,
) -> bool:
    """Return true only for complete artifacts matching every requested input.

    No artifacts means the seed is safe to launch.  Any partial, unreadable, or
    mismatched state is an error because the child runner intentionally refuses
    to overwrite it; treating that state as merely pending would waste a GPU and
    fail only after model startup.
    """

    run_dir = output_root / f"seed_{seed}"
    summary_path = run_dir / "summary.json"
    samples_path = run_dir / "raw_samples.csv"
    lock_path = run_dir / benchmark_runner.LOCK_FILENAME
    summary_exists = summary_path.exists()
    samples_exist = samples_path.exists()

    if lock_path.exists():
        raise CompletionArtifactError(
            f"Seed {seed} output is locked by {lock_path}. Verify whether another "
            "benchmark is running before removing a stale lock or choosing a new "
            "output root."
        )
    if not summary_exists and not samples_exist:
        return False
    if summary_exists != samples_exist:
        present = summary_path if summary_exists else samples_path
        missing = samples_path if summary_exists else summary_path
        raise CompletionArtifactError(
            f"Seed {seed} has partial benchmark artifacts: {present} exists but "
            f"{missing} is missing. The summary is the completion marker and the "
            "launcher will not write into this directory; inspect/recover the run "
            "or choose a fresh output root."
        )
    if not summary_path.is_file() or not samples_path.is_file():
        raise CompletionArtifactError(
            f"Seed {seed} artifact paths must be regular files: {summary_path}, "
            f"{samples_path}"
        )

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CompletionArtifactError(
            f"Seed {seed} summary cannot be read as JSON: {summary_path}: {error}"
        ) from error
    if not isinstance(summary, Mapping):
        raise CompletionArtifactError(
            f"Seed {seed} summary root must be a JSON object: {summary_path}"
        )

    errors: list[str] = []

    def expect(label: str, actual: Any, wanted: Any) -> None:
        if actual != wanted:
            errors.append(f"{label}={actual!r}; expected {wanted!r}")

    expect("schema_version", summary.get("schema_version"), benchmark_runner.SCHEMA_VERSION)
    expect("status", summary.get("status"), "completed")
    expect("seed", summary.get("seed"), seed)
    expect("num_samples", summary.get("num_samples"), expected.num_samples)

    run = _mapping(summary.get("run"))
    expect("run.seed", run.get("seed"), seed)
    expect(
        "run.requested_sample_count",
        run.get("requested_sample_count"),
        expected.num_samples,
    )
    expect(
        "run.evaluation_tier",
        run.get("evaluation_tier"),
        "final" if expected.num_samples == 1_000 else "pilot",
    )
    expect(
        "run.final_protocol_eligible",
        run.get("final_protocol_eligible"),
        expected.num_samples == 1_000,
    )
    protocol = _mapping(run.get("generation_protocol"))
    sampling = expected.sampling_config
    for key, wanted in {
        "diffusion_type": sampling["diffusion_type"],
        "num_steps": sampling["num_steps"],
        "inference_eps": sampling["inference_eps"],
        "exclude_special_tokens": sampling["exclude_special_tokens"],
        "temperature": sampling["softmax_temp"],
        "randomness": sampling["randomness"],
        "randomness_used_by_sampler": sampling["diffusion_type"] == "mdlm",
    }.items():
        expect(f"run.generation_protocol.{key}", protocol.get(key), wanted)
    nfe = protocol.get("nfe")
    if isinstance(nfe, bool) or not isinstance(nfe, int) or nfe <= 0:
        errors.append("run.generation_protocol.nfe must be a positive integer")
    elif sampling["diffusion_type"] == "udlm" and nfe != sampling["num_steps"]:
        errors.append("run.generation_protocol.nfe must equal UDLM num_steps")

    checkpoint = _mapping(summary.get("checkpoint"))
    expect(
        "checkpoint.path",
        _recorded_path(checkpoint.get("path")),
        expected.checkpoint_path,
    )
    expect(
        "checkpoint.sha256",
        checkpoint.get("sha256"),
        expected.checkpoint_sha256,
    )
    expect(
        "checkpoint.global_step",
        checkpoint.get("global_step"),
        expected.checkpoint_global_step,
    )
    expect(
        "checkpoint.size_bytes",
        checkpoint.get("size_bytes"),
        expected.checkpoint_size_bytes,
    )
    expect(
        "checkpoint.diffusion_type",
        checkpoint.get("diffusion_type"),
        expected.checkpoint_diffusion_type,
    )
    expect(
        "checkpoint.udlm_inference_eps",
        checkpoint.get("udlm_inference_eps"),
        expected.checkpoint_udlm_inference_eps,
    )
    expect(
        "checkpoint.udlm_exclude_special_tokens",
        checkpoint.get("udlm_exclude_special_tokens"),
        expected.checkpoint_udlm_exclude_special_tokens,
    )

    config = _mapping(summary.get("config"))
    expect("config.path", _recorded_path(config.get("path")), expected.config_path)
    expect("config.sha256", config.get("sha256"), expected.source_config_sha256)
    expect("config.source", config.get("source"), expected.source_config)
    expect("config.sampling", config.get("sampling"), expected.sampling_config)
    expect(
        "config.sampling_sha256",
        config.get("sampling_sha256"),
        expected.sampling_config_sha256,
    )
    expect("config.effective", config.get("effective"), expected.effective_config)
    expect(
        "config.effective_sha256",
        config.get("effective_sha256"),
        expected.effective_config_sha256,
    )

    git = _mapping(summary.get("git"))
    expect(
        "git.runner_sha256",
        git.get("runner_sha256"),
        expected.benchmark_runner_sha256,
    )
    expect(
        "implementation_inputs",
        summary.get("implementation_inputs"),
        expected.implementation_inputs,
    )

    artifacts = _mapping(summary.get("artifacts"))
    raw_artifact = _mapping(artifacts.get("raw_samples_csv"))
    summary_artifact = _mapping(artifacts.get("summary_json"))
    expected_fields = tuple(benchmark_runner.RAW_SAMPLE_FIELDS)
    expect(
        "artifacts.raw_samples_csv.path",
        _recorded_path(raw_artifact.get("path")),
        samples_path.resolve(),
    )
    expect(
        "artifacts.raw_samples_csv.row_count",
        raw_artifact.get("row_count"),
        expected.num_samples,
    )
    expect(
        "artifacts.raw_samples_csv.fields",
        raw_artifact.get("fields"),
        list(expected_fields),
    )
    expect(
        "artifacts.summary_json.path",
        _recorded_path(summary_artifact.get("path")),
        summary_path.resolve(),
    )

    actual_sha256, actual_rows, actual_fields, csv_errors = _validate_raw_csv(
        samples_path,
        expected_row_count=expected.num_samples,
        expected_fields=expected_fields,
    )
    errors.extend(csv_errors)
    expect("raw_samples.csv actual row_count", actual_rows, expected.num_samples)
    expect("raw_samples.csv actual fields", actual_fields, expected_fields)
    expect("artifacts.raw_samples_csv.sha256", raw_artifact.get("sha256"), actual_sha256)

    if errors:
        detail = "\n  - ".join(errors)
        raise CompletionArtifactError(
            f"Seed {seed} has benchmark artifacts that do not match the requested "
            f"run or fail integrity checks:\n  - {detail}\nRefusing to skip or "
            "relaunch into the existing directory. Inspect the artifacts or choose "
            "a fresh output root."
        )
    return True


def _command(
    *,
    checkpoint: Path,
    config: Path,
    num_samples: int,
    seed: int,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts/exps/denovo/benchmark.py"),
        "--checkpoint",
        str(checkpoint),
        "--config",
        str(config),
        "--num-samples",
        str(num_samples),
        "--seed",
        str(seed),
        "--device",
        "cuda:0",
        "--output-dir",
        str(output_dir),
    ]


def _snapshot_signature(states: list[GPUState]) -> str:
    return json.dumps([state.as_dict() for state in states], sort_keys=True)


def _recheck_gpu_for_launch(
    candidate: GPUState,
    *,
    max_utilization_percent: int,
    min_free_memory_mib: int,
) -> tuple[Optional[GPUState], list[str]]:
    """Probe once more and reject a changed, out-of-policy, or unknown device."""

    try:
        rechecked = _probe_gpu(candidate.index)
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
        return None, [f"final GPU probe failed: {error}"]
    reasons = rechecked.rejection_reasons(
        max_utilization_percent=max_utilization_percent,
        min_free_memory_mib=min_free_memory_mib,
    )
    if rechecked.uuid != candidate.uuid:
        reasons.append(
            f"physical index identity changed from {candidate.uuid} to {rechecked.uuid}"
        )
    if reasons:
        return None, reasons
    return rechecked, []


def _child_environment(
    *,
    seed: int,
    gpu: GPUState,
    selection: Mapping[str, Any],
    run_label: str,
) -> dict[str, str]:
    """Map the verified physical GPU UUID into the child environment."""

    environment = os.environ.copy()
    required_python_paths = [str(REPOSITORY_ROOT / "src"), str(REPOSITORY_ROOT)]
    inherited_python_paths = [
        entry
        for entry in environment.get("PYTHONPATH", "").split(os.pathsep)
        if entry and entry not in required_python_paths
    ]
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": gpu.uuid,
            "PYTHONHASHSEED": str(seed),
            "HF_HOME": str(REPOSITORY_ROOT / ".cache/huggingface"),
            "TORCH_HOME": str(REPOSITORY_ROOT / ".cache/torch"),
            "PIP_CACHE_DIR": str(REPOSITORY_ROOT / ".cache/pip"),
            "TOKENIZERS_PARALLELISM": "false",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONPATH": os.pathsep.join(
                [*required_python_paths, *inherited_python_paths]
            ),
            "GENMOL_BENCHMARK_GPU_PHYSICAL_INDEX": str(gpu.index),
            "GENMOL_BENCHMARK_GPU_UUID": gpu.uuid,
            "GENMOL_BENCHMARK_GPU_SELECTION_SNAPSHOT": json.dumps(
                selection, separators=(",", ":"), sort_keys=True
            ),
            "GENMOL_BENCHMARK_RUN_LABEL": run_label,
        }
    )
    return environment


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    if Path.cwd().resolve() != REPOSITORY_ROOT:
        raise RuntimeError(f"run from repository root: {REPOSITORY_ROOT}")
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError(
            "controller inherited CUDA_VISIBLE_DEVICES; start it from a shell without "
            "a pre-existing GPU visibility mask"
        )
    source_revision = _require_clean_pushed_source()
    evaluation_tier = _validate_sample_tier(args.num_samples, pilot=args.pilot)
    selected_gpu_indices = _validate_gpu_request(args.gpu_count, args.gpu_indices)
    if args.poll_seconds <= 0:
        raise ValueError("poll-seconds must be positive")
    if args.min_free_memory_mib < 30_000:
        raise ValueError("min-free-memory-mib must be at least 30000")
    if not 1 <= args.max_utilization_percent <= 10:
        raise ValueError("max-utilization-percent must lie in [1, 10]")
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("seeds must be unique")
    selection_policy = _selection_policy(
        selected_physical_indices=selected_gpu_indices,
        max_utilization_percent=args.max_utilization_percent,
        min_free_memory_mib=args.min_free_memory_mib,
    )

    checkpoint = _resolve_checkpoint(args.checkpoint)
    config = _resolve_in_repo(args.config)
    output_root = _resolve_in_repo(args.output_root)
    log_root = _resolve_in_repo(args.log_root)
    if not checkpoint.is_file() or not config.is_file():
        raise FileNotFoundError("checkpoint and config must both exist")
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    expected = _build_expected_run_identity(checkpoint, config, args.num_samples)
    pending = []
    completed_at_start = []
    for seed in args.seeds:
        if _completed(output_root, seed, expected):
            completed_at_start.append(seed)
        else:
            pending.append(seed)
    print(
        json.dumps(
            {
                "event": "controller_start",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": expected.checkpoint_sha256,
                "checkpoint_global_step": expected.checkpoint_global_step,
                "config": str(config),
                "config_sha256": expected.source_config_sha256,
                "sampling_config": expected.sampling_config,
                "sampling_config_sha256": expected.sampling_config_sha256,
                "num_samples": args.num_samples,
                "evaluation_tier": evaluation_tier,
                "seeds": args.seeds,
                "completed_seeds": completed_at_start,
                "pending_seeds": pending,
                "user_requested_gpu_count": args.gpu_count,
                "user_selected_physical_indices": list(selected_gpu_indices),
                "selection_policy": selection_policy,
                "source_revision": source_revision,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not pending:
        print(
            "All requested de novo benchmark seeds already have matching, "
            "integrity-checked artifacts.",
            flush=True,
        )
        return
    if args.dry_run:
        print(
            json.dumps(
                [state.as_dict() for state in _snapshot(selected_gpu_indices)],
                sort_keys=True,
            )
        )
        for seed in pending:
            print(
                "DRY RUN",
                _command(
                    checkpoint=checkpoint,
                    config=config,
                    num_samples=args.num_samples,
                    seed=seed,
                    output_dir=output_root / f"seed_{seed}",
                ),
            )
        return

    running: dict[int, RunningJob] = {}
    failure_seen = False
    failure_details: list[str] = []
    last_signature: Optional[str] = None
    unchanged_polls = 0

    while pending or running:
        for gpu_index, job in list(running.items()):
            return_code = job.process.poll()
            if return_code is None:
                continue
            job.log_handle.close()
            del running[gpu_index]
            print(
                f"FINISHED seed={job.seed} physical_gpu={gpu_index} "
                f"exit={return_code} log={job.log_path}",
                flush=True,
            )
            if return_code != 0:
                failure_seen = True
                failure_details.append(
                    f"seed {job.seed} exited with status {return_code}; see {job.log_path}"
                )
                continue
            try:
                child_completed = _completed(output_root, job.seed, expected)
            except CompletionArtifactError as error:
                child_completed = False
                failure_details.append(f"seed {job.seed}: {error}")
            if not child_completed:
                failure_details.append(
                    f"seed {job.seed} exited successfully but produced neither completion "
                    "artifact"
                )
            failure_seen = failure_seen or not child_completed

        if failure_seen:
            if not running:
                details = "\n  - ".join(failure_details)
                raise RuntimeError(
                    "A benchmark child failed completion validation; no further seeds "
                    f"were launched:\n  - {details}"
                )
            time.sleep(args.poll_seconds)
            continue

        states = _snapshot(selected_gpu_indices)
        signature = _snapshot_signature(states)
        unchanged_polls = unchanged_polls + 1 if signature == last_signature else 0
        last_signature = signature
        if unchanged_polls == 0 or unchanged_polls % 20 == 0:
            eligible_indices = [
                state.index
                for state in states
                if _eligible(
                    state,
                    max_utilization_percent=args.max_utilization_percent,
                    min_free_memory_mib=args.min_free_memory_mib,
                )
            ]
            print(
                json.dumps(
                    {
                        "event": "gpu_poll",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "eligible_physical_indices": eligible_indices,
                        "pending_seeds": pending,
                        "running_seeds": [job.seed for job in running.values()],
                        "gpu_states": [state.as_dict() for state in states],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        free_slots = args.gpu_count - len(running)
        candidates = sorted(
            (
                state
                for state in states
                if state.index not in running
                and _eligible(
                    state,
                    max_utilization_percent=args.max_utilization_percent,
                    min_free_memory_mib=args.min_free_memory_mib,
                )
            ),
            key=lambda state: (
                -(state.memory_total_mib - state.memory_used_mib),
                state.utilization_percent,
                state.index,
            ),
        )[: max(free_slots, 0)]

        for candidate in candidates:
            if not pending:
                break
            # The second probe is the final point-in-time guard before launch.
            rechecked, rejection_reasons = _recheck_gpu_for_launch(
                candidate,
                max_utilization_percent=args.max_utilization_percent,
                min_free_memory_mib=args.min_free_memory_mib,
            )
            if rechecked is None:
                print(
                    json.dumps(
                        {
                            "event": "gpu_final_probe_rejected",
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "physical_index": candidate.index,
                            "initial_uuid": candidate.uuid,
                            "reasons": rejection_reasons,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                continue

            seed = pending[0]
            # Recheck the output immediately before spending GPU resources.  This
            # also catches artifacts created by another controller after startup.
            if _completed(output_root, seed, expected):
                pending.pop(0)
                print(
                    f"SKIPPED seed={seed}; matching artifacts appeared while waiting",
                    flush=True,
                )
                continue

            pending.pop(0)
            output_dir = output_root / f"seed_{seed}"
            command = _command(
                checkpoint=checkpoint,
                config=config,
                num_samples=args.num_samples,
                seed=seed,
                output_dir=output_dir,
            )
            run_label = benchmark_runner.benchmark_run_label(
                expected.checkpoint_global_step,
                expected.checkpoint_sha256,
                seed,
            )
            log_path = log_root / f"{run_label}.log"
            log_handle = log_path.open("a", encoding="utf-8", buffering=1)
            selection = {
                "event": "launch",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "source_revision": source_revision,
                "physical_gpu": rechecked.as_dict(),
                "policy": selection_policy,
                "command": command,
            }
            log_handle.write(json.dumps(selection, sort_keys=True) + "\n")
            environment = _child_environment(
                seed=seed,
                gpu=rechecked,
                selection=selection,
                run_label=run_label,
            )
            process = subprocess.Popen(
                command,
                cwd=REPOSITORY_ROOT,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            running[rechecked.index] = RunningJob(
                seed=seed,
                gpu=rechecked,
                process=process,
                log_handle=log_handle,
                log_path=log_path,
            )
            print(
                f"LAUNCHED seed={seed} pid={process.pid} "
                f"physical_gpu={rechecked.index} uuid={rechecked.uuid} log={log_path}",
                flush=True,
            )

        if pending or running:
            time.sleep(args.poll_seconds)

    for seed in args.seeds:
        if not _completed(output_root, seed, expected):
            raise RuntimeError(
                f"Seed {seed} is missing completion artifacts after the controller "
                "finished"
            )
    print("All de novo benchmark seeds completed.", flush=True)


if __name__ == "__main__":
    main()

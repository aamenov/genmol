"""Launch reproducible de novo benchmark runs on dynamically selected GPUs.

The controller is intended to run inside ``tmux``. The caller supplies only a
GPU count of one or two. Before each child starts, the controller inventories
every NVIDIA GPU and dynamically chooses an idle device under the configured
utilization, memory, compute-mode, and process guards. It then re-probes that
exact UUID and maps it into the child as logical ``cuda:0``. The controller
never interrupts or reuses a device with an active compute process.
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
from typing import Any, Mapping, Optional


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
    checkpoint_udlm_prior_variant: str | None
    checkpoint_udlm_prior_metadata: Mapping[str, Any] | None
    checkpoint_udlm_prior_metadata_sha256: str | None
    config_path: Path
    source_config: Mapping[str, Any]
    source_config_sha256: str
    sampling_config: Mapping[str, Any]
    sampling_config_sha256: str
    effective_config: Mapping[str, Any]
    effective_config_sha256: str
    benchmark_runner_sha256: str
    implementation_inputs: Mapping[str, Any]
    metric_inputs: Mapping[str, Any]
    num_samples: int
    source_revision: str | None = None
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
        help=(
            "Maximum concurrent benchmark GPUs, selected dynamically from the "
            "full NVIDIA inventory; must be 1 or 2."
        ),
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-utilization-percent", type=int, default=10)
    parser.add_argument("--min-free-memory-mib", type=int, default=30_000)
    parser.add_argument("--log-root", type=Path, default=Path("output/logs"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _validate_gpu_count(gpu_count: int) -> int:
    """Validate the user-selected concurrency without accepting physical IDs."""

    if type(gpu_count) is not int or not 1 <= gpu_count <= 2:
        raise ValueError("gpu-count must be 1 or 2")
    return gpu_count


def _validate_sample_tier(num_samples: int, *, pilot: bool) -> str:
    """Prevent a pilot from being mistaken for the final benchmark."""

    if type(num_samples) is not int:
        raise ValueError("num-samples must be an integer")
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
    requested_gpu_count: int,
    max_utilization_percent: int,
    min_free_memory_mib: int,
) -> dict[str, Any]:
    """Return the policy schema embedded in controller and child provenance."""

    requested_gpu_count = _validate_gpu_count(requested_gpu_count)
    return {
        "selection_method": "dynamic_idle_discovery",
        "inventory_scope": "all_nvidia_gpus",
        "requested_gpu_count": requested_gpu_count,
        "max_utilization_percent": max_utilization_percent,
        "utilization_comparison": "strictly_less_than",
        "min_free_memory_mib": min_free_memory_mib,
        "active_compute_processes_allowed": False,
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
    project_root = _project_root()
    if resolved != project_root and project_root not in resolved.parents:
        raise ValueError(f"checkpoint escapes project root: {resolved}")
    return resolved


def _project_root() -> Path:
    return (
        REPOSITORY_ROOT.parent.parent
        if REPOSITORY_ROOT.parent.name == "run_sources"
        else REPOSITORY_ROOT
    )


def _project_venv_python() -> Path:
    return _project_root() / ".venv/bin/python"


def _require_project_virtual_environment() -> Path:
    """Require the shared project ``.venv`` for controller and child execution."""

    expected_python = _project_venv_python()
    if not expected_python.is_file():
        raise RuntimeError(f"project virtual-environment Python is missing: {expected_python}")
    if Path(sys.executable).resolve() != expected_python.resolve():
        raise RuntimeError(
            "benchmark launcher must run with the project virtual environment: "
            f"{expected_python}"
        )
    return expected_python


def _run_nvidia_smi(
    gpu_identifier: int | str,
    query: str,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu_identifier),
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
    if (
        not indices
        or any(index < 0 for index in indices)
        or len(indices) != len(set(indices))
    ):
        raise RuntimeError(
            "nvidia-smi returned no GPUs or invalid/duplicate physical indices"
        )
    return sorted(indices)


def _probe_gpu(gpu_identifier: int | str) -> GPUState:
    """Probe one physical GPU by enumerated index or exact NVIDIA UUID."""

    if isinstance(gpu_identifier, bool) or not isinstance(gpu_identifier, (int, str)):
        raise TypeError("GPU identifier must be a physical index or NVIDIA UUID")
    expected_index = gpu_identifier if isinstance(gpu_identifier, int) else None
    expected_uuid = gpu_identifier if isinstance(gpu_identifier, str) else None
    if expected_index is not None and expected_index < 0:
        raise ValueError("physical GPU index must be non-negative")
    if expected_uuid is not None and not expected_uuid.startswith("GPU-"):
        raise ValueError(f"invalid NVIDIA GPU UUID: {expected_uuid!r}")
    context = f"GPU {gpu_identifier}"
    status = _run_nvidia_smi(
        gpu_identifier,
        "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,compute_mode",
    )
    if status.returncode or status.stderr.strip():
        detail = status.stderr.strip() or status.stdout.strip()
        raise RuntimeError(f"{context} status query failed: {detail}")
    rows = [
        [field.strip() for field in row]
        for row in csv.reader(io.StringIO(status.stdout), skipinitialspace=True)
        if any(field.strip() for field in row)
    ]
    if len(rows) != 1 or len(rows[0]) != 7:
        raise RuntimeError(f"{context} status query returned an unexpected row")
    (
        index_text,
        uuid,
        name,
        memory_used_text,
        memory_total_text,
        utilization_text,
        compute_mode,
    ) = rows[0]
    try:
        physical_index = int(index_text)
    except ValueError as error:
        raise RuntimeError(
            f"{context} status query returned a non-integer physical index"
        ) from error
    if (
        physical_index < 0
        or (expected_index is not None and physical_index != expected_index)
        or (expected_uuid is not None and uuid != expected_uuid)
        or not uuid.startswith("GPU-")
        or not name
        or not compute_mode
    ):
        raise RuntimeError(f"{context} status query returned inconsistent identity")
    try:
        memory_used_mib = int(memory_used_text)
        memory_total_mib = int(memory_total_text)
        utilization_percent = int(utilization_text)
    except ValueError as error:
        raise RuntimeError(
            f"{context} status query returned non-integer telemetry"
        ) from error
    if (
        memory_used_mib < 0
        or memory_total_mib <= 0
        or memory_used_mib > memory_total_mib
        or not 0 <= utilization_percent <= 100
    ):
        raise RuntimeError(f"{context} status query returned invalid telemetry")

    processes = _run_nvidia_smi(
        gpu_identifier,
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
    )
    if processes.returncode or processes.stderr.strip():
        detail = processes.stderr.strip() or processes.stdout.strip()
        raise RuntimeError(f"{context} process query failed: {detail}")
    raw_process_rows = [
        [field.strip() for field in row]
        for row in csv.reader(io.StringIO(processes.stdout), skipinitialspace=True)
        if any(field.strip() for field in row)
    ]
    no_process_markers = [
        fields
        for fields in raw_process_rows
        if fields[0].lower().startswith("no running")
    ]
    if no_process_markers and (
        len(raw_process_rows) != 1 or len(no_process_markers[0]) != 1
    ):
        raise RuntimeError(
            f"{context} process query returned ambiguous no-process telemetry"
        )

    process_rows = []
    seen_process_pids: set[int] = set()
    for fields in raw_process_rows:
        if fields[0].lower().startswith("no running"):
            continue
        if len(fields) != 4 or not fields[1].isdigit():
            raise RuntimeError(f"{context} process query returned an unexpected row")
        process_uuid = fields[0]
        process_pid = int(fields[1])
        process_name = fields[2]
        try:
            process_memory_mib = int(fields[3])
        except ValueError as error:
            raise RuntimeError(
                f"{context} process query returned non-integer memory telemetry"
            ) from error
        if (
            process_uuid != uuid
            or process_pid <= 0
            or not process_name
            or process_memory_mib < 0
            or process_pid in seen_process_pids
        ):
            raise RuntimeError(f"{context} process query returned invalid telemetry")
        seen_process_pids.add(process_pid)
        process_rows.append(
            {
                "pid": process_pid,
                "process_name": process_name,
                "used_memory_mib": process_memory_mib,
            }
        )
    return GPUState(
        index=physical_index,
        uuid=uuid,
        name=name,
        memory_used_mib=memory_used_mib,
        memory_total_mib=memory_total_mib,
        utilization_percent=utilization_percent,
        compute_mode=compute_mode,
        compute_processes=tuple(process_rows),
    )


def _snapshot() -> list[GPUState]:
    """Enumerate and inspect the complete NVIDIA GPU inventory."""

    available_indices = _physical_gpu_indices()
    states = []
    errors = {}
    for index in available_indices:
        try:
            states.append(_probe_gpu(index))
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
            errors[index] = str(error)
    if errors:
        raise RuntimeError(
            "complete NVIDIA GPU inventory is unverifiable; refusing dynamic "
            f"selection: {errors}"
        )
    if len({state.uuid for state in states}) != len(states):
        raise RuntimeError(
            "complete NVIDIA GPU inventory contains duplicate UUIDs; refusing "
            "dynamic selection"
        )
    return states


def _require_tmux_for_execution() -> None:
    """Refuse a disconnect-fragile controller for every non-dry GPU run."""

    if not os.environ.get("TMUX"):
        raise RuntimeError(
            "benchmark execution must run inside tmux; dry-run and completed-artifact "
            "validation remain available outside tmux"
        )


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
    *,
    source_revision: str | None = None,
) -> ExpectedRunIdentity:
    """Resolve and fingerprint the exact inputs passed to every child run."""

    # Quality depends on this ignored binary artifact. Verify it before the
    # checkpoint inspection and well before any GPU selection/model startup.
    metric_inputs = benchmark_runner.metric_input_provenance()
    checkpoint_info = benchmark_runner.checkpoint_metadata(checkpoint)
    source_config_sha256 = _sha256_file(config)
    source_config = benchmark_runner.load_yaml_config(config)
    if _sha256_file(config) != source_config_sha256:
        raise RuntimeError(f"Inference config changed while being inspected: {config}")
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
    checkpoint_udlm_prior_variant = checkpoint_info.get(
        "udlm_prior_variant",
        "release_uniform" if checkpoint_diffusion_type == "udlm" else None,
    )
    checkpoint_udlm_prior_metadata = checkpoint_info.get("udlm_prior_metadata")
    checkpoint_udlm_prior_metadata_sha256 = checkpoint_info.get(
        "udlm_prior_metadata_sha256"
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
        if checkpoint_udlm_prior_variant != sampling_config["prior_variant"]:
            raise ValueError(
                "config prior_variant does not match checkpoint metadata: "
                f"{sampling_config['prior_variant']!r} != "
                f"{checkpoint_udlm_prior_variant!r}"
            )
        if (
            checkpoint_udlm_prior_metadata_sha256
            != sampling_config["prior_metadata_sha256"]
        ):
            raise ValueError(
                "config prior_metadata_sha256 does not match checkpoint metadata"
            )
        if checkpoint_udlm_prior_variant in (
            benchmark_runner.UDLM_CATEGORICAL_PRIOR_VARIANTS
        ) and not isinstance(checkpoint_udlm_prior_metadata, Mapping):
            raise ValueError(
                "categorical checkpoint metadata inspection did not return the full "
                "immutable prior record"
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
        checkpoint_udlm_prior_variant=(
            str(checkpoint_udlm_prior_variant)
            if checkpoint_udlm_prior_variant is not None
            else None
        ),
        checkpoint_udlm_prior_metadata=(
            dict(checkpoint_udlm_prior_metadata)
            if isinstance(checkpoint_udlm_prior_metadata, Mapping)
            else None
        ),
        checkpoint_udlm_prior_metadata_sha256=(
            str(checkpoint_udlm_prior_metadata_sha256)
            if checkpoint_udlm_prior_metadata_sha256 is not None
            else None
        ),
        config_path=config,
        source_config=source_config,
        source_config_sha256=source_config_sha256,
        sampling_config=sampling_config,
        sampling_config_sha256=_canonical_json_sha256(sampling_config),
        effective_config=effective_config,
        effective_config_sha256=_canonical_json_sha256(effective_config),
        benchmark_runner_sha256=_sha256_file(benchmark_runner_path),
        implementation_inputs=implementation_inputs,
        metric_inputs=metric_inputs,
        num_samples=num_samples,
        source_revision=source_revision,
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

    expected_top_level = {
        "schema_version",
        "status",
        "seed",
        "num_samples",
        "run",
        "checkpoint",
        "config",
        "metrics",
        "failure_counts",
        "runtime_seconds",
        "environment",
        "git",
        "implementation_inputs",
        "metric_inputs",
        "tokenizer",
        "artifacts",
    }
    if set(summary) != expected_top_level:
        errors.append(
            "summary top-level fields differ from the current child contract: "
            f"found {sorted(summary)}, expected {sorted(expected_top_level)}"
        )

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
    expect("run.one_seed_per_invocation", run.get("one_seed_per_invocation"), True)
    expect("run.single_generation_batch", run.get("single_generation_batch"), True)
    expected_seed_configuration = {
        "seed": seed,
        "seed_applied_immediately_before_generation": True,
        "python_random": True,
        "numpy": True,
        "torch_cpu": True,
        "torch_cuda_all": True,
        "python_hash_seed": str(seed),
    }
    expect(
        "run.seed_configuration",
        run.get("seed_configuration"),
        expected_seed_configuration,
    )
    for timestamp_field in ("started_at_utc", "completed_at_utc"):
        timestamp = run.get(timestamp_field)
        try:
            parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        except ValueError:
            errors.append(f"run.{timestamp_field} is not an ISO-8601 timestamp")
        else:
            if parsed.tzinfo is None:
                errors.append(f"run.{timestamp_field} lacks a timezone")

    if expected.source_revision is not None:
        expected_command = _command(
            checkpoint=expected.checkpoint_path,
            expected_checkpoint_sha256=expected.checkpoint_sha256,
            expected_source_revision=expected.source_revision,
            config=expected.config_path,
            expected_config_sha256=expected.source_config_sha256,
            num_samples=expected.num_samples,
            seed=seed,
            output_dir=run_dir.resolve(),
        )
        expect("run.command", run.get("command"), expected_command)
    protocol = _mapping(run.get("generation_protocol"))
    sampling = expected.sampling_config
    for key, wanted in {
        "diffusion_type": sampling["diffusion_type"],
        "num_steps": sampling["num_steps"],
        "inference_eps": sampling["inference_eps"],
        "exclude_special_tokens": sampling["exclude_special_tokens"],
        "prior_variant": sampling["prior_variant"],
        "prior_metadata_sha256": sampling["prior_metadata_sha256"],
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
        "checkpoint.byte_identity_verified_before_and_after_load",
        checkpoint.get("byte_identity_verified_before_and_after_load"),
        True,
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
    expect(
        "checkpoint.udlm_prior_variant",
        checkpoint.get("udlm_prior_variant"),
        expected.checkpoint_udlm_prior_variant,
    )
    expect(
        "checkpoint.udlm_prior_metadata",
        checkpoint.get("udlm_prior_metadata"),
        expected.checkpoint_udlm_prior_metadata,
    )
    expect(
        "checkpoint.udlm_prior_metadata_sha256",
        checkpoint.get("udlm_prior_metadata_sha256"),
        expected.checkpoint_udlm_prior_metadata_sha256,
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
    if expected.source_revision is not None:
        expect("git.commit", git.get("commit"), expected.source_revision)
        expect("git.upstream", git.get("upstream"), expected.source_revision)
        expect(
            "git.expected_source_revision",
            git.get("expected_source_revision"),
            expected.source_revision,
        )
        expect("git.dirty", git.get("dirty"), False)
        expect(
            "git.clean_pushed_source_verified_before_and_after_run",
            git.get("clean_pushed_source_verified_before_and_after_run"),
            True,
        )
    expect(
        "implementation_inputs",
        summary.get("implementation_inputs"),
        expected.implementation_inputs,
    )
    expect(
        "metric_inputs",
        summary.get("metric_inputs"),
        expected.metric_inputs,
    )

    metrics = _mapping(summary.get("metrics"))
    if set(metrics) != {"released_comparable", "strict"} or any(
        not isinstance(metrics.get(branch), Mapping)
        or not metrics.get(branch)
        for branch in ("released_comparable", "strict")
    ):
        errors.append("metrics must contain non-empty released_comparable and strict branches")
    expected_failure_fields = {
        "raw_safe_conversion_failed",
        "strict_decode_failed",
        "released_decode_failed",
        "released_recovered_strict_failure",
        "strict_valid_but_released_failed",
        "released_largest_component_applied",
        "strict_duplicates",
        "released_duplicates",
    }
    failure_counts = _mapping(summary.get("failure_counts"))
    if set(failure_counts) != expected_failure_fields or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in failure_counts.values()
    ):
        errors.append("failure_counts does not match the current nonnegative count schema")
    expected_runtime_fields = {
        "model_load_and_device_move",
        "model_sampling_and_tokenizer",
        "released_postprocessing",
        "generation",
        "decode_and_metrics",
        "total_before_summary_write",
    }
    runtime = _mapping(summary.get("runtime_seconds"))
    if set(runtime) != expected_runtime_fields or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
        for value in runtime.values()
    ):
        errors.append("runtime_seconds does not match the current finite timing schema")
    environment = _mapping(summary.get("environment"))
    expect("environment.requested_device", environment.get("requested_device"), "cuda:0")
    expect(
        "environment.resolved_model_device",
        environment.get("resolved_model_device"),
        "cuda:0",
    )
    expect("environment.torch_cuda_available", environment.get("torch_cuda_available"), True)
    if not _mapping(summary.get("tokenizer")):
        errors.append("tokenizer provenance must be a non-empty mapping")
    if expected.source_revision is not None:
        launch_environment = _mapping(environment.get("launch_environment"))
        snapshot_text = launch_environment.get(
            "GENMOL_BENCHMARK_GPU_SELECTION_SNAPSHOT"
        )
        try:
            snapshot = json.loads(snapshot_text)
        except (TypeError, json.JSONDecodeError):
            errors.append("launch environment lacks a valid GPU selection snapshot")
        else:
            snapshot_mapping = _mapping(snapshot)
            if not snapshot_mapping:
                errors.append("launch GPU selection snapshot must be a JSON object")
            expect(
                "launch snapshot.command",
                snapshot_mapping.get("command"),
                expected_command,
            )
            expect(
                "launch snapshot.source_revision",
                snapshot_mapping.get("source_revision"),
                {
                    "head": expected.source_revision,
                    "upstream": expected.source_revision,
                },
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
    if expected.num_samples == 1_000:
        # The final tier must be consumable by the actual report validator, not
        # merely resemble a child summary structurally.  This shares the strict
        # raw-row arithmetic, metric, tokenizer, CUDA, and provenance contract
        # without imposing the final-report-only 1,000-row rules on pilot runs.
        from scripts.exps.denovo import report as benchmark_report

        try:
            benchmark_report._validate_summary_and_rows(run_dir, seed)
        except benchmark_report.ReportValidationError as error:
            raise CompletionArtifactError(
                f"Seed {seed} artifacts fail final report validation: {error}. "
                "Refusing to skip or relaunch into the existing directory."
            ) from error
    return True


def _command(
    *,
    checkpoint: Path,
    expected_checkpoint_sha256: str,
    expected_source_revision: str,
    config: Path,
    expected_config_sha256: str,
    num_samples: int,
    seed: int,
    output_dir: Path,
) -> list[str]:
    return [
        str(_project_venv_python()),
        str(REPOSITORY_ROOT / "scripts/exps/denovo/benchmark.py"),
        "--checkpoint",
        str(checkpoint),
        "--expected-checkpoint-sha256",
        expected_checkpoint_sha256,
        "--expected-source-revision",
        expected_source_revision,
        "--config",
        str(config),
        "--expected-config-sha256",
        expected_config_sha256,
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
    """Probe the exact chosen UUID and reject changed or occupied devices."""

    try:
        rechecked = _probe_gpu(candidate.uuid)
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
        return None, [f"final GPU probe failed: {error}"]
    reasons = rechecked.rejection_reasons(
        max_utilization_percent=max_utilization_percent,
        min_free_memory_mib=min_free_memory_mib,
    )
    if rechecked.uuid != candidate.uuid:
        reasons.append(
            f"UUID identity changed from {candidate.uuid} to {rechecked.uuid}"
        )
    if rechecked.index != candidate.index:
        reasons.append(
            f"physical index identity changed from {candidate.index} to {rechecked.index}"
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
    _require_project_virtual_environment()
    source_revision = _require_clean_pushed_source()
    evaluation_tier = _validate_sample_tier(args.num_samples, pilot=args.pilot)
    gpu_count = _validate_gpu_count(args.gpu_count)
    if args.poll_seconds <= 0:
        raise ValueError("poll-seconds must be positive")
    if args.min_free_memory_mib < 30_000:
        raise ValueError("min-free-memory-mib must be at least 30000")
    if not 1 <= args.max_utilization_percent <= 10:
        raise ValueError("max-utilization-percent must lie in [1, 10]")
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("seeds must be unique")
    selection_policy = _selection_policy(
        requested_gpu_count=gpu_count,
        max_utilization_percent=args.max_utilization_percent,
        min_free_memory_mib=args.min_free_memory_mib,
    )

    checkpoint = _resolve_checkpoint(args.checkpoint)
    config = _resolve_in_repo(args.config)
    output_root = _resolve_in_repo(args.output_root)
    log_root = _resolve_in_repo(args.log_root)
    if not checkpoint.is_file() or not config.is_file():
        raise FileNotFoundError("checkpoint and config must both exist")

    expected = _build_expected_run_identity(
        checkpoint,
        config,
        args.num_samples,
        source_revision=source_revision["head"],
    )
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
                "checkpoint_udlm_prior_variant": (
                    expected.checkpoint_udlm_prior_variant
                ),
                "checkpoint_udlm_prior_metadata": (
                    expected.checkpoint_udlm_prior_metadata
                ),
                "checkpoint_udlm_prior_metadata_sha256": (
                    expected.checkpoint_udlm_prior_metadata_sha256
                ),
                "config": str(config),
                "config_sha256": expected.source_config_sha256,
                "sampling_config": expected.sampling_config,
                "sampling_config_sha256": expected.sampling_config_sha256,
                "num_samples": args.num_samples,
                "evaluation_tier": evaluation_tier,
                "seeds": args.seeds,
                "completed_seeds": completed_at_start,
                "pending_seeds": pending,
                "user_requested_gpu_count": gpu_count,
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
        states = _snapshot()
        eligible = sorted(
            (
                state
                for state in states
                if _eligible(
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
        )
        requested_now = min(gpu_count, len(pending))
        if len(eligible) < requested_now:
            raise RuntimeError(
                f"dry-run requested {requested_now} concurrent GPU(s), but only "
                f"{len(eligible)} are genuinely idle"
            )
        print(
            json.dumps(
                {
                    "event": "dry_run_gpu_inventory",
                    "gpu_states": [state.as_dict() for state in states],
                    "dynamically_selected_gpu_states": [
                        state.as_dict() for state in eligible[:gpu_count]
                    ],
                },
                sort_keys=True,
            )
        )
        for seed in pending:
            print(
                "DRY RUN",
                _command(
                    checkpoint=checkpoint,
                    expected_checkpoint_sha256=expected.checkpoint_sha256,
                    expected_source_revision=source_revision["head"],
                    config=config,
                    expected_config_sha256=expected.source_config_sha256,
                    num_samples=args.num_samples,
                    seed=seed,
                    output_dir=output_root / f"seed_{seed}",
                ),
            )
        return

    _require_tmux_for_execution()
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

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

        # Every launch opportunity starts from a fresh enumeration of the full
        # NVIDIA inventory. When multiple slots are free, the inventory is
        # queried again after each child launch rather than reusing stale data.
        states = _snapshot()
        inventory_snapshot_completed_at_utc = datetime.now(timezone.utc).isoformat()
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

        free_slots = gpu_count - len(running)
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
        )[:1] if free_slots > 0 else []

        launched_job = False
        for candidate in candidates:
            if not pending:
                break
            seed = pending[0]
            output_dir = output_root / f"seed_{seed}"
            command = _command(
                checkpoint=checkpoint,
                expected_checkpoint_sha256=expected.checkpoint_sha256,
                expected_source_revision=source_revision["head"],
                config=config,
                expected_config_sha256=expected.source_config_sha256,
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
            # Catch artifacts created by another controller before the final GPU
            # probe so that the exact-UUID check remains as close to launch as
            # possible.
            if _completed(output_root, seed, expected):
                pending.pop(0)
                print(
                    f"SKIPPED seed={seed}; matching artifacts appeared while waiting",
                    flush=True,
                )
                continue

            # This exact-UUID query is the final substantive safety guard before
            # the child is constructed and launched.
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
            final_uuid_probe_completed_at_utc = datetime.now(timezone.utc).isoformat()

            pending.pop(0)
            log_handle = log_path.open("a", encoding="utf-8", buffering=1)
            selection = {
                "event": "launch",
                "gpu_selection_schema_version": 2,
                "timestamp_utc": final_uuid_probe_completed_at_utc,
                "inventory_snapshot_completed_at_utc": (
                    inventory_snapshot_completed_at_utc
                ),
                "final_uuid_probe_completed_at_utc": (
                    final_uuid_probe_completed_at_utc
                ),
                "source_revision": source_revision,
                "gpu_inventory_at_selection": [
                    state.as_dict() for state in states
                ],
                "running_physical_indices_at_selection": sorted(running),
                "physical_gpu_at_final_uuid_probe": rechecked.as_dict(),
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
            launched_job = True
            print(
                f"LAUNCHED seed={seed} pid={process.pid} "
                f"physical_gpu={rechecked.index} uuid={rechecked.uuid} log={log_path}",
                flush=True,
            )

            # Do not choose another GPU from the pre-launch inventory snapshot.
            # The next outer iteration re-enumerates all devices before the next
            # child and provides its own exact-UUID final probe.
            break

        if launched_job and pending and len(running) < gpu_count:
            continue
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

"""Launch a versioned PMO ablation matrix on explicitly selected GPUs.

Run this controller inside ``tmux``.  It checks utilization and free memory
immediately before every child launch, maps one physical GPU to each process,
and enforces an explicit project-wide active-GPU limit (four by default).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MATRIX_SCHEMA_VERSION = 1
VARIANT_SETTINGS: dict[str, dict[str, Any]] = {
    "released": {"mode": "released", "prior_strength": 0.0},
    "running_mean": {"mode": "mean", "prior_strength": 0.0},
    "support3": {"mode": "mean", "prior_strength": 0.0},
    "shrink1": {"mode": "bayes", "prior_strength": 1.0},
    "shrink3": {"mode": "bayes", "prior_strength": 3.0},
    "shrink10": {"mode": "bayes", "prior_strength": 10.0},
    "shrink30": {"mode": "bayes", "prior_strength": 30.0},
    "delta": {"mode": "delta", "prior_strength": 0.0},
    "running_mean_parent_control": {"mode": "mean", "prior_strength": 0.0},
    "running_mean_delta_control": {"mode": "mean", "prior_strength": 0.0},
}


@dataclass(frozen=True)
class GPUState:
    index: int
    uuid: str
    memory_total_mib: int
    memory_used_mib: int
    utilization_percent: int

    @property
    def memory_free_mib(self) -> int:
        return self.memory_total_mib - self.memory_used_mib


@dataclass(frozen=True)
class Job:
    oracle: str
    variant: str
    seed: int
    task_config: dict[str, Any]

    @property
    def label(self) -> str:
        return f"{self.oracle}__{self.variant}__seed{self.seed}"


@dataclass
class RunningJob:
    job: Job
    gpu: GPUState
    process: subprocess.Popen
    log_handle: Any
    log_path: Path


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--gpu-indices", type=int, nargs="+", required=True)
    parser.add_argument("--reserved-active-gpus", type=int, default=0)
    parser.add_argument(
        "--max-total-active-gpus",
        type=int,
        default=4,
        help=(
            "Maximum selected plus reserved active project GPUs (default: 4)."
        ),
    )
    parser.add_argument("--utilization-threshold", type=int, default=10)
    parser.add_argument("--min-free-memory-mib", type=int, default=20_000)
    parser.add_argument(
        "--allow-shared-low-utilization",
        action="store_true",
        help=(
            "Allow a selected GPU with existing compute processes only when it remains "
            "below the utilization threshold and has enough free memory."
        ),
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _validate_gpu_request(
    gpu_indices: list[int],
    reserved_active_gpus: int,
    max_total_active_gpus: int,
) -> None:
    unique_gpu_indices = set(gpu_indices)
    if len(unique_gpu_indices) != len(gpu_indices):
        raise ValueError("GPU indices must be unique")
    if reserved_active_gpus < 0:
        raise ValueError("reserved-active-gpus cannot be negative")
    if max_total_active_gpus <= 0:
        raise ValueError("max-total-active-gpus must be positive")
    if len(unique_gpu_indices) + reserved_active_gpus > max_total_active_gpus:
        raise ValueError(
            "requested plus reserved active GPUs exceeds max-total-active-gpus "
            f"({max_total_active_gpus})"
        )


def _gpu_states() -> dict[int, GPUState]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE).stdout
    states = {}
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        state = GPUState(
            index=int(fields[0]),
            uuid=fields[1],
            memory_total_mib=int(fields[2]),
            memory_used_mib=int(fields[3]),
            utilization_percent=int(fields[4]),
        )
        states[state.index] = state
    return states


def _compute_process_snapshot() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE)
    rows = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4:
            rows.append(
                {
                    "gpu_uuid": fields[0],
                    "pid": int(fields[1]),
                    "process_name": fields[2],
                    "used_memory_mib": int(fields[3]),
                }
            )
    return rows


def _eligible_gpu_snapshot(
    gpu_index: int,
    *,
    utilization_threshold: int,
    min_free_memory_mib: int,
    allow_shared_low_utilization: bool,
) -> Optional[tuple[GPUState, list[dict[str, Any]]]]:
    """Return the final pre-launch device/process snapshot when eligible."""

    # Query processes first so utilization and memory are the last device
    # measurements before launch. NVIDIA's CLI does not expose both tables in
    # one query, so the two timestamps cannot be perfectly atomic.
    process_snapshot = _compute_process_snapshot()
    state = _gpu_states().get(gpu_index)
    if state is None:
        raise ValueError(f"GPU index {gpu_index} does not exist")
    if state.utilization_percent >= utilization_threshold:
        return None
    if state.memory_free_mib < min_free_memory_mib:
        return None
    gpu_processes = [
        row for row in process_snapshot if row["gpu_uuid"] == state.uuid
    ]
    if gpu_processes and not allow_shared_low_utilization:
        return None
    return state, gpu_processes


def _load_matrix(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        matrix = yaml.safe_load(handle)
    if not isinstance(matrix, dict):
        raise ValueError("matrix YAML must contain a mapping")
    if matrix.get("schema_version") != MATRIX_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported matrix schema_version {matrix.get('schema_version')!r}; "
            f"expected {MATRIX_SCHEMA_VERSION}"
        )
    required = {
        "experiment_id",
        "scientific_status",
        "model_path",
        "tasks",
        "variants",
        "seeds",
        "common",
    }
    missing = required - set(matrix)
    if missing:
        raise ValueError(f"matrix is missing keys: {sorted(missing)}")
    if not str(matrix["scientific_status"]).strip():
        raise ValueError("matrix scientific_status must be nonempty")
    return matrix


def _config_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _jobs(matrix: dict[str, Any]) -> list[Job]:
    tasks = []
    for row in matrix["tasks"]:
        if isinstance(row, str):
            tasks.append({"oracle": row})
        elif isinstance(row, dict) and "oracle" in row:
            tasks.append(dict(row))
        else:
            raise ValueError("each task must be an oracle name or mapping with 'oracle'")
    return [
        Job(str(task["oracle"]), str(variant), int(seed), task)
        for task, variant, seed in itertools.product(tasks, matrix["variants"], matrix["seeds"])
    ]


def _run_dir(matrix: dict[str, Any], job: Job) -> Path:
    output_root = Path(matrix["common"].get("output_root", "output/pmo_ablation"))
    if not output_root.is_absolute():
        output_root = REPOSITORY_ROOT / output_root
    return output_root / str(matrix["experiment_id"]) / job.oracle / job.variant / f"seed_{job.seed}"


def _completed(
    matrix_path: Path,
    matrix_sha256: str,
    matrix: dict[str, Any],
    job: Job,
) -> bool:
    """Return true only for a terminal run tied to this exact matrix file."""

    run_dir = _run_dir(matrix, job)
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "summary.json"
    checkpoint_path = run_dir / "state" / "latest.pkl"
    if not all(path.is_file() for path in (manifest_path, summary_path, checkpoint_path)):
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
        summary = json.loads(summary_path.read_text())
        config = manifest["config"]
        expected_budget = int(matrix["common"].get("max_oracle_calls", 10_000))
        expected_model = Path(matrix["model_path"])
        if not expected_model.is_absolute():
            expected_model = REPOSITORY_ROOT / expected_model
        expected_run_id = (
            f"{matrix['experiment_id']}:{job.oracle}:{job.variant}:seed{job.seed}"
        )
        return bool(
            manifest.get("schema_version") == 1
            and summary.get("schema_version") in {1, 2}
            and manifest.get("status") == "completed"
            and summary.get("status") == "completed"
            and summary.get("checkpoint_consistent") is True
            and manifest.get("run_id") == expected_run_id
            and summary.get("run_id") == expected_run_id
            and manifest.get("task") == job.oracle
            and manifest.get("variant") == job.variant
            and manifest.get("seed") == job.seed
            and manifest.get("oracle_budget") == expected_budget
            and manifest.get("oracle_calls") == expected_budget
            and summary.get("recoverable_oracle_calls") == expected_budget
            and manifest.get("config_sha256") == _config_sha256(config)
            and summary.get("config_sha256") == manifest.get("config_sha256")
            and config.get("experiment_id") == matrix["experiment_id"]
            and config.get("scientific_status") == matrix["scientific_status"]
            and config.get("oracle") == job.oracle
            and config.get("variant") == job.variant
            and config.get("seed") == job.seed
            and config.get("model_path") == str(expected_model.resolve())
            and summary.get("scores", {})
            .get("all_charged_molecules", {})
            .get("oracle_calls")
            == expected_budget
            and config.get("matrix_path") == str(matrix_path)
            and config.get("matrix_sha256") == matrix_sha256
        )
    except (AttributeError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _command(
    matrix_path: Path,
    matrix_sha256: str,
    matrix: dict[str, Any],
    job: Job,
) -> list[str]:
    common = dict(matrix["common"])
    model_path = Path(matrix["model_path"])
    if not model_path.is_absolute():
        model_path = REPOSITORY_ROOT / model_path
    output_root = Path(common.pop("output_root", "output/pmo_ablation"))
    if not output_root.is_absolute():
        output_root = REPOSITORY_ROOT / output_root

    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "exps" / "pmo" / "run_ablation.py"),
        "--oracle",
        job.oracle,
        "--variant",
        job.variant,
        "--seed",
        str(job.seed),
        "--model-path",
        str(model_path.resolve()),
        "--device",
        "cuda:0",
        "--experiment-id",
        str(matrix["experiment_id"]),
        "--scientific-status",
        str(matrix["scientific_status"]),
        "--matrix-path",
        str(matrix_path),
        "--matrix-sha256",
        matrix_sha256,
        "--output-root",
        str(output_root.resolve()),
    ]
    flag_map = {
        "max_oracle_calls": "--max-oracle-calls",
        "reporting_frequency": "--reporting-frequency",
        "checkpoint_every": "--checkpoint-every",
        "max_iterations": "--max-iterations",
        "population_size": "--population-size",
        "warmup": "--warmup",
        "softmax_temp": "--softmax-temp",
        "randomness": "--randomness",
        "guidance_scale": "--guidance-scale",
        "min_mol_size": "--min-mol-size",
        "max_mol_size": "--max-mol-size",
        "legacy_seed_count": "--legacy-seed-count",
        "delta_attribution": "--delta-attribution",
    }
    for key, flag in flag_map.items():
        if key in common and common[key] is not None:
            command.extend([flag, str(common.pop(key))])
    for boolean_key, flag in {
        "legacy_warmup_off_by_one": "--legacy-warmup-off-by-one",
        "durable_events": "--durable-events",
    }.items():
        if common.pop(boolean_key, False):
            command.append(flag)
    if common:
        raise ValueError(f"unsupported common matrix keys: {sorted(common)}")

    if "gamma" in job.task_config:
        command.extend(["--gamma", str(job.task_config["gamma"])])
    settings = VARIANT_SETTINGS.get(job.variant)
    if settings is None:
        raise ValueError(f"unsupported ablation variant {job.variant!r}")
    if settings["mode"] == "bayes":
        prior_mean = job.task_config.get("prior_mean")
        prior_source = job.task_config.get("prior_mean_source")
        try:
            finite_prior_mean = not isinstance(prior_mean, bool) and math.isfinite(
                float(prior_mean)
            )
        except (TypeError, ValueError):
            finite_prior_mean = False
        if not finite_prior_mean or not str(prior_source or "").strip():
            raise ValueError(
                f"{job.variant} task {job.oracle} requires a finite prior_mean "
                "and nonempty prior_mean_source"
            )
        command.extend(["--prior-mean", str(prior_mean)])
        command.extend(["--prior-mean-source", str(prior_source)])
    run_dir = _run_dir(matrix, job)
    if (run_dir / "manifest.json").exists() and (run_dir / "state" / "latest.pkl").exists():
        command.append("--resume")
    return command


def main() -> None:
    args = _parse_args()
    _validate_gpu_request(
        args.gpu_indices,
        args.reserved_active_gpus,
        args.max_total_active_gpus,
    )
    if not 1 <= args.utilization_threshold <= 100:
        raise ValueError("utilization threshold must lie in [1, 100]")
    if args.poll_seconds < 1:
        raise ValueError("poll-seconds must be positive")

    matrix_path = args.matrix.expanduser().resolve()
    matrix = _load_matrix(matrix_path)
    matrix_sha256 = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    pending = [
        job
        for job in _jobs(matrix)
        if not _completed(matrix_path, matrix_sha256, matrix, job)
    ]
    print(
        json.dumps(
            {
                "controller": "GenMol fragment-vocabulary ablation",
                "matrix": str(matrix_path),
                "matrix_sha256": matrix_sha256,
                "experiment_id": matrix["experiment_id"],
                "pending_jobs": len(pending),
                "gpu_indices": args.gpu_indices,
                "reserved_active_gpus": args.reserved_active_gpus,
                "max_total_active_gpus": args.max_total_active_gpus,
                "utilization_threshold": args.utilization_threshold,
                "min_free_memory_mib": args.min_free_memory_mib,
                "allow_shared_low_utilization": args.allow_shared_low_utilization,
                "initial_gpu_state": [state.__dict__ for state in _gpu_states().values()],
                "initial_compute_processes": _compute_process_snapshot(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.dry_run:
        for job in pending:
            print(
                "DRY RUN",
                job.label,
                _command(matrix_path, matrix_sha256, matrix, job),
                flush=True,
            )
        return

    running: dict[int, RunningJob] = {}
    failure_seen = False
    log_root = REPOSITORY_ROOT / "output" / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    while pending or running:
        for gpu_index, running_job in list(running.items()):
            return_code = running_job.process.poll()
            if return_code is None:
                continue
            running_job.log_handle.close()
            del running[gpu_index]
            print(
                f"FINISHED {running_job.job.label} GPU {gpu_index} exit={return_code} "
                f"log={running_job.log_path}",
                flush=True,
            )
            failure_seen = failure_seen or return_code != 0

        if failure_seen and matrix.get("stop_on_failure", True):
            if not running:
                raise RuntimeError("an ablation job failed; no further jobs were launched")
            time.sleep(args.poll_seconds)
            continue

        free_worker_indices = [index for index in args.gpu_indices if index not in running]
        for gpu_index in free_worker_indices:
            if not pending:
                break
            snapshot = _eligible_gpu_snapshot(
                gpu_index,
                utilization_threshold=args.utilization_threshold,
                min_free_memory_mib=args.min_free_memory_mib,
                allow_shared_low_utilization=args.allow_shared_low_utilization,
            )
            if snapshot is None:
                continue
            state, gpu_processes = snapshot

            if hashlib.sha256(matrix_path.read_bytes()).hexdigest() != matrix_sha256:
                raise RuntimeError("matrix file changed while the controller was running")

            job = pending.pop(0)
            command = _command(matrix_path, matrix_sha256, matrix, job)
            log_path = log_root / f"pmo_{matrix['experiment_id']}__{job.label}.log"
            log_handle = log_path.open("a", buffering=1)
            snapshot = {
                "event": "launch",
                "job": job.label,
                "physical_gpu": state.__dict__,
                "compute_processes": gpu_processes,
                "utilization_threshold": args.utilization_threshold,
                "min_free_memory_mib": args.min_free_memory_mib,
                "sharing_authorized": args.allow_shared_low_utilization,
                "sharing_actual": bool(gpu_processes),
                "wall_time_comparable": not bool(gpu_processes),
                "command": command,
                "time_unix": time.time(),
            }
            log_handle.write(json.dumps(snapshot, sort_keys=True) + "\n")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
            environment["PYTHONHASHSEED"] = str(job.seed)
            process = subprocess.Popen(
                command,
                cwd=REPOSITORY_ROOT,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            running[gpu_index] = RunningJob(job, state, process, log_handle, log_path)
            print(
                f"LAUNCHED {job.label} pid={process.pid} physical_gpu={gpu_index} "
                f"uuid={state.uuid} log={log_path}",
                flush=True,
            )

        if pending or running:
            time.sleep(args.poll_seconds)

    print("All matrix jobs completed.", flush=True)


if __name__ == "__main__":
    main()

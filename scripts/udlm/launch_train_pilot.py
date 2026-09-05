"""Launch a bounded UDLM pilot on explicitly authorized idle GPUs.

The launcher refuses more than two devices, active compute processes, an
unpushed implementation commit, or an existing tmux session. It probes the
selected physical devices immediately before launch and exposes their UUIDs as
the child's logical CUDA devices.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shlex
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPOSITORY_ROOT.parents[1]
RUN_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
MAX_SAFE_UTILIZATION_PERCENT = 10
MIN_SAFE_FREE_MEMORY_MIB = 30_000


@dataclass(frozen=True)
class GPUState:
    physical_index: int
    uuid: str
    name: str
    memory_used_mib: int
    memory_total_mib: int
    utilization_percent: int
    compute_processes: tuple[dict[str, object], ...]

    @property
    def free_memory_mib(self) -> int:
        return self.memory_total_mib - self.memory_used_mib

    def rejection_reasons(
        self,
        *,
        max_utilization_percent: int,
        min_free_memory_mib: int,
    ) -> list[str]:
        reasons = []
        if self.utilization_percent >= max_utilization_percent:
            reasons.append(
                f"utilization {self.utilization_percent}% is not below "
                f"{max_utilization_percent}%"
            )
        if self.free_memory_mib < min_free_memory_mib:
            reasons.append(
                f"free memory {self.free_memory_mib} MiB is below "
                f"{min_free_memory_mib} MiB"
            )
        if self.compute_processes:
            reasons.append(f"{len(self.compute_processes)} active compute process(es)")
        return reasons


def validate_gpu_request(gpu_count: int, gpu_indices: Sequence[int]) -> tuple[int, ...]:
    selected = tuple(int(index) for index in gpu_indices)
    if gpu_count not in (1, 2):
        raise ValueError("gpu-count must be 1 or 2")
    if len(selected) != gpu_count:
        raise ValueError("gpu-indices count must equal gpu-count")
    if len(set(selected)) != len(selected):
        raise ValueError("gpu-indices must be unique")
    if any(index < 0 for index in selected):
        raise ValueError("gpu-indices must be non-negative physical IDs")
    return selected


def exact_accumulation_steps(
    global_batch_size: int,
    micro_batch_size: int,
    world_size: int,
) -> int:
    """Return exact accumulation, refusing a silently inflated global batch."""
    if min(global_batch_size, micro_batch_size, world_size) <= 0:
        raise ValueError("batch sizes and world size must be positive")
    samples_per_micro_step = micro_batch_size * world_size
    quotient, remainder = divmod(global_batch_size, samples_per_micro_step)
    if quotient < 1 or remainder:
        raise ValueError(
            "global-batch-size must be an exact positive multiple of "
            "gpu-count * micro-batch-size"
        )
    return quotient


def validate_safety_thresholds(
    max_utilization_percent: int,
    min_free_memory_mib: int,
) -> None:
    if not 1 <= max_utilization_percent <= MAX_SAFE_UTILIZATION_PERCENT:
        raise ValueError(
            "max-utilization-percent must be between 1 and "
            f"{MAX_SAFE_UTILIZATION_PERCENT}"
        )
    if min_free_memory_mib < MIN_SAFE_FREE_MEMORY_MIB:
        raise ValueError(
            "min-free-memory-mib cannot be below "
            f"{MIN_SAFE_FREE_MEMORY_MIB}"
        )


def _run(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, text=True, capture_output=True, check=True)


def probe_gpus(selected_indices: Sequence[int]) -> list[GPUState]:
    status = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = {}
    for row in csv.reader(io.StringIO(status.stdout), skipinitialspace=True):
        fields = [field.strip() for field in row]
        if len(fields) != 6:
            raise RuntimeError(f"unexpected nvidia-smi GPU row: {fields}")
        index = int(fields[0])
        rows[index] = {
            "uuid": fields[1],
            "name": fields[2],
            "memory_used_mib": int(fields[3]),
            "memory_total_mib": int(fields[4]),
            "utilization_percent": int(fields[5]),
        }
    missing = sorted(set(selected_indices) - set(rows))
    if missing:
        raise RuntimeError(f"physical GPUs not reported by nvidia-smi: {missing}")

    processes_by_uuid: dict[str, list[dict[str, object]]] = {}
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    # NVIDIA returns exit 0 and a human-readable "No running processes" line
    # on some driver versions; other versions return an empty table.
    if processes.returncode != 0 and "No running" not in processes.stderr:
        raise RuntimeError(processes.stderr.strip() or "compute process query failed")
    for row in csv.reader(io.StringIO(processes.stdout), skipinitialspace=True):
        fields = [field.strip() for field in row]
        if not fields or not any(fields) or fields[0].lower().startswith("no running"):
            continue
        if len(fields) != 4:
            raise RuntimeError(f"unexpected nvidia-smi process row: {fields}")
        processes_by_uuid.setdefault(fields[0], []).append(
            {
                "pid": int(fields[1]),
                "process_name": fields[2],
                "used_memory_mib": int(fields[3]),
            }
        )

    states = []
    for index in selected_indices:
        row = rows[index]
        states.append(
            GPUState(
                physical_index=index,
                uuid=str(row["uuid"]),
                name=str(row["name"]),
                memory_used_mib=int(row["memory_used_mib"]),
                memory_total_mib=int(row["memory_total_mib"]),
                utilization_percent=int(row["utilization_percent"]),
                compute_processes=tuple(processes_by_uuid.get(str(row["uuid"]), [])),
            )
        )
    return states


def _python_executable() -> Path:
    candidates = (
        REPOSITORY_ROOT / ".venv" / "bin" / "python",
        PROJECT_ROOT / ".venv" / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("project .venv/bin/python was not found")


def _git_output(*arguments: str) -> str:
    return _run(["git", "-C", str(REPOSITORY_ROOT), *arguments]).stdout.strip()


def require_pushed_commit() -> str:
    if subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "diff", "--quiet"], check=False
    ).returncode:
        raise RuntimeError("tracked working-tree changes must be committed before launch")
    if subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "diff", "--cached", "--quiet"],
        check=False,
    ).returncode:
        raise RuntimeError("staged changes must be committed before launch")
    status = _run(
        [
            "git",
            "-C",
            str(REPOSITORY_ROOT),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "-z",
        ]
    ).stdout
    untracked_source = []
    for entry in status.split("\0"):
        if not entry or entry[:2] != "??":
            continue
        relative_path = entry[3:]
        if relative_path == "output" or relative_path.startswith("output/"):
            continue
        untracked_source.append(relative_path)
    if untracked_source:
        raise RuntimeError(
            "untracked non-output files must be committed before launch: "
            + ", ".join(sorted(untracked_source))
        )
    head = _git_output("rev-parse", "HEAD")
    upstream = _git_output("rev-parse", "@{upstream}")
    if head != upstream:
        raise RuntimeError(f"HEAD {head} is not pushed to upstream {upstream}")
    return head


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_training_command(
    *,
    gpu_count: int,
    run_dir: Path,
    max_steps: int,
    global_batch_size: int,
    micro_batch_size: int,
    num_workers: int,
    seed: int,
    checkpoint: Path | None,
    exclude_special_tokens: bool,
) -> list[str]:
    command = [
        str(_python_executable()),
        "-u",
        str(REPOSITORY_ROOT / "scripts" / "train.py"),
        "--config-name",
        "udlm",
        f"seed={seed}",
        f"trainer.devices={gpu_count}",
        f"trainer.max_steps={max_steps}",
        f"loader.global_batch_size={global_batch_size}",
        f"loader.batch_size={micro_batch_size}",
        f"loader.num_workers={num_workers}",
        f"callback.every_n_train_steps={max_steps}",
        f"callback.dirpath={run_dir / 'checkpoints'}",
        f"hydra.run.dir={run_dir / 'hydra'}",
        f"training.udlm.exclude_special_tokens={str(exclude_special_tokens).lower()}",
    ]
    if checkpoint is not None:
        command.append(f"training.init_from_mdlm_checkpoint={checkpoint}")
        command.append("training.init_from_mdlm_ema=true")
    return command


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--gpu-count", type=int, required=True)
    parser.add_argument("--gpu-indices", type=int, nargs="+", required=True)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "paper_v1" / "checkpoints" / "50000.ckpt",
    )
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--exclude-special-tokens", action="store_true")
    parser.add_argument(
        "--max-utilization-percent",
        type=int,
        default=MAX_SAFE_UTILIZATION_PERCENT,
        help="May make the shared-server guard stricter, never looser than 10%%.",
    )
    parser.add_argument(
        "--min-free-memory-mib",
        type=int,
        default=MIN_SAFE_FREE_MEMORY_MIB,
        help="May make the shared-server guard stricter, never below 30000 MiB.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    if not RUN_NAME_PATTERN.fullmatch(args.run_name):
        raise ValueError("run-name must contain only letters, digits, '.', '_', or '-'")
    selected_indices = validate_gpu_request(args.gpu_count, args.gpu_indices)
    if not 1 <= args.max_steps <= 1_000:
        raise ValueError("pilot max-steps must be in [1, 1000]")
    if min(args.global_batch_size, args.micro_batch_size) <= 0:
        raise ValueError("batch sizes must be positive")
    accumulation_steps = exact_accumulation_steps(
        args.global_batch_size,
        args.micro_batch_size,
        args.gpu_count,
    )
    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative")
    validate_safety_thresholds(
        args.max_utilization_percent,
        args.min_free_memory_mib,
    )
    checkpoint = None if args.scratch else args.checkpoint.resolve()
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if checkpoint is not None and not checkpoint.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"checkpoint must be inside project root {PROJECT_ROOT}")

    git_sha = require_pushed_commit()
    run_dir = REPOSITORY_ROOT / "output" / "udlm" / args.run_name
    log_path = REPOSITORY_ROOT / "output" / "logs" / f"{args.run_name}.log"
    manifest_path = run_dir / "launch_manifest.json"
    if run_dir.exists() or log_path.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing pilot: {run_dir} or {log_path}"
        )
    command = build_training_command(
        gpu_count=args.gpu_count,
        run_dir=run_dir,
        max_steps=args.max_steps,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        checkpoint=checkpoint,
        exclude_special_tokens=args.exclude_special_tokens,
    )

    session_name = f"genmol_udlm_{args.run_name}"
    if subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0:
        raise RuntimeError(f"tmux session already exists: {session_name}")

    checkpoint_sha256 = None if checkpoint is None else sha256_file(checkpoint)

    # This is deliberately the final substantive check before recording and
    # launching. A cooperative nvidia-smi check cannot provide an atomic lease.
    gpu_states = probe_gpus(selected_indices)
    rejected = {
        state.physical_index: state.rejection_reasons(
            max_utilization_percent=args.max_utilization_percent,
            min_free_memory_mib=args.min_free_memory_mib,
        )
        for state in gpu_states
    }
    rejected = {index: reasons for index, reasons in rejected.items() if reasons}
    if rejected:
        raise RuntimeError(f"selected GPU(s) are not genuinely idle: {rejected}")

    visible_uuids = ",".join(state.uuid for state in gpu_states)
    environment_command = [
        "env",
        f"CUDA_VISIBLE_DEVICES={visible_uuids}",
        f"PYTHONPATH={REPOSITORY_ROOT}:{REPOSITORY_ROOT / 'src'}",
        *command,
    ]
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "bounded UDLM training pilot",
        "git_sha": git_sha,
        "run_name": args.run_name,
        "tmux_session": session_name,
        "physical_gpu_indices": list(selected_indices),
        "logical_cuda_devices": list(range(args.gpu_count)),
        "cuda_visible_device_uuids": [state.uuid for state in gpu_states],
        "gpu_states_at_launch": [asdict(state) for state in gpu_states],
        "command": environment_command,
        "log_path": str(log_path),
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "global_batch_size": args.global_batch_size,
        "micro_batch_size_per_process": args.micro_batch_size,
        "accumulate_grad_batches": accumulation_steps,
        "effective_global_batch_size": (
            args.micro_batch_size * args.gpu_count * accumulation_steps
        ),
        "exclude_special_tokens": args.exclude_special_tokens,
        "dry_run": args.dry_run,
    }
    run_dir.mkdir(parents=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if args.dry_run:
        return

    shell_command = (
        "set -o pipefail; "
        + shlex.join(environment_command)
        + " 2>&1 | tee -a "
        + shlex.quote(str(log_path))
    )
    subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            session_name,
            "-c",
            str(REPOSITORY_ROOT),
            "bash",
            "-lc",
            shell_command,
        ],
        check=True,
    )
    print(f"launched tmux session {session_name}; log: {log_path}")


if __name__ == "__main__":
    main()

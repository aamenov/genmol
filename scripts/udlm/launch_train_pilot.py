"""Launch a bounded UDLM pilot on dynamically selected idle GPUs.

The caller chooses only the number of GPUs (one or two). Immediately before
launch, the controller inventories every NVIDIA GPU, selects genuinely idle
devices, re-probes those exact UUIDs, and exposes the UUIDs as the child's
logical CUDA devices. It never interrupts or reuses a device with an active
compute process.
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
    compute_mode: str
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
        if self.compute_mode.lower() == "prohibited":
            reasons.append("compute mode is prohibited")
        if self.compute_processes:
            reasons.append(f"{len(self.compute_processes)} active compute process(es)")
        return reasons


def validate_gpu_count(gpu_count: int) -> int:
    """Validate the user-selected device count without accepting physical IDs."""

    if type(gpu_count) is not int or gpu_count not in (1, 2):
        raise ValueError("gpu-count must be 1 or 2")
    return gpu_count


def exact_accumulation_steps(
    global_batch_size: int,
    micro_batch_size: int,
    world_size: int,
) -> int:
    """Return exact accumulation, refusing a silently inflated global batch."""
    if any(
        type(value) is not int
        for value in (global_batch_size, micro_batch_size, world_size)
    ):
        raise ValueError("batch sizes and world size must be integers")
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


def _probe_gpus(device_uuid: str | None = None) -> list[GPUState]:
    """Query GPU telemetry and compute processes, optionally for one UUID."""

    prefix = ["nvidia-smi"]
    if device_uuid is not None:
        if not device_uuid.startswith("GPU-"):
            raise ValueError(f"invalid NVIDIA GPU UUID: {device_uuid!r}")
        prefix.extend(["-i", device_uuid])
    status = _run(
        [
            *prefix,
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,compute_mode",
            "--format=csv,noheader,nounits",
        ]
    )
    if status.stderr.strip():
        raise RuntimeError(f"nvidia-smi GPU query returned stderr: {status.stderr.strip()}")
    rows: dict[int, dict[str, object]] = {}
    seen_uuids: set[str] = set()
    for row in csv.reader(io.StringIO(status.stdout), skipinitialspace=True):
        fields = [field.strip() for field in row]
        if not any(fields):
            continue
        if len(fields) != 7:
            raise RuntimeError(f"unexpected nvidia-smi GPU row: {fields}")
        try:
            index = int(fields[0])
            memory_used_mib = int(fields[3])
            memory_total_mib = int(fields[4])
            utilization_percent = int(fields[5])
        except ValueError as error:
            raise RuntimeError(
                f"nvidia-smi returned non-integer GPU telemetry: {fields}"
            ) from error
        uuid = fields[1]
        if index in rows or uuid in seen_uuids:
            raise RuntimeError("nvidia-smi returned duplicate GPU identities")
        if (
            index < 0
            or not uuid.startswith("GPU-")
            or memory_used_mib < 0
            or memory_total_mib <= 0
            or memory_used_mib > memory_total_mib
            or not 0 <= utilization_percent <= 100
            or not fields[2]
            or not fields[6]
        ):
            raise RuntimeError(f"nvidia-smi returned invalid GPU telemetry: {fields}")
        rows[index] = {
            "uuid": uuid,
            "name": fields[2],
            "memory_used_mib": memory_used_mib,
            "memory_total_mib": memory_total_mib,
            "utilization_percent": utilization_percent,
            "compute_mode": fields[6],
        }
        seen_uuids.add(uuid)
    if not rows:
        raise RuntimeError("nvidia-smi returned no NVIDIA GPUs")

    processes_by_uuid: dict[str, list[dict[str, object]]] = {}
    processes = subprocess.run(
        [
            *prefix,
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    # NVIDIA returns exit 0 and a human-readable "No running processes" line
    # on some driver versions; other versions return an empty table.
    if processes.returncode != 0 or processes.stderr.strip():
        raise RuntimeError(processes.stderr.strip() or "compute process query failed")
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
        raise RuntimeError("nvidia-smi returned ambiguous no-process telemetry")
    seen_processes: set[tuple[str, int]] = set()
    for fields in raw_process_rows:
        if fields[0].lower().startswith("no running"):
            continue
        if len(fields) != 4:
            raise RuntimeError(f"unexpected nvidia-smi process row: {fields}")
        process_uuid = fields[0]
        process_name = fields[2]
        try:
            process_pid = int(fields[1])
            process_memory_mib = int(fields[3])
        except ValueError as error:
            raise RuntimeError(
                f"nvidia-smi returned non-integer process telemetry: {fields}"
            ) from error
        process_identity = (process_uuid, process_pid)
        if (
            process_uuid not in seen_uuids
            or process_pid <= 0
            or not process_name
            or process_memory_mib < 0
            or process_identity in seen_processes
        ):
            raise RuntimeError(f"invalid nvidia-smi process row: {fields}")
        seen_processes.add(process_identity)
        processes_by_uuid.setdefault(process_uuid, []).append(
            {
                "pid": process_pid,
                "process_name": process_name,
                "used_memory_mib": process_memory_mib,
            }
        )

    states = []
    for index in sorted(rows):
        row = rows[index]
        states.append(
            GPUState(
                physical_index=index,
                uuid=str(row["uuid"]),
                name=str(row["name"]),
                memory_used_mib=int(row["memory_used_mib"]),
                memory_total_mib=int(row["memory_total_mib"]),
                utilization_percent=int(row["utilization_percent"]),
                compute_mode=str(row["compute_mode"]),
                compute_processes=tuple(processes_by_uuid.get(str(row["uuid"]), [])),
            )
        )
    if device_uuid is not None:
        if len(states) != 1 or states[0].uuid != device_uuid:
            identities = [(state.physical_index, state.uuid) for state in states]
            raise RuntimeError(
                f"UUID-specific probe for {device_uuid} returned {identities}"
            )
    return states


def probe_all_gpus() -> list[GPUState]:
    """Enumerate and fully inspect every NVIDIA GPU on the host."""

    return _probe_gpus()


def probe_gpu_uuid(device_uuid: str) -> GPUState:
    """Re-probe one exact UUID immediately before exposing it to a child."""

    return _probe_gpus(device_uuid)[0]


def select_idle_gpus(
    states: list[GPUState],
    *,
    gpu_count: int,
    max_utilization_percent: int,
    min_free_memory_mib: int,
) -> tuple[GPUState, ...]:
    """Choose the requested number of best genuinely idle devices."""

    validate_gpu_count(gpu_count)
    eligible = sorted(
        (
            state
            for state in states
            if not state.rejection_reasons(
                max_utilization_percent=max_utilization_percent,
                min_free_memory_mib=min_free_memory_mib,
            )
        ),
        key=lambda state: (
            -state.free_memory_mib,
            state.utilization_percent,
            state.physical_index,
        ),
    )
    if len(eligible) < gpu_count:
        inventory = {
            state.physical_index: state.rejection_reasons(
                max_utilization_percent=max_utilization_percent,
                min_free_memory_mib=min_free_memory_mib,
            )
            for state in states
        }
        raise RuntimeError(
            f"requested {gpu_count} GPU(s), but only {len(eligible)} are genuinely "
            f"idle; full inventory rejection reasons: {inventory}"
        )
    return tuple(eligible[:gpu_count])


def reprobe_selected_gpus(
    selected: tuple[GPUState, ...],
    *,
    max_utilization_percent: int,
    min_free_memory_mib: int,
) -> tuple[GPUState, ...]:
    """Verify exact selected UUIDs one last time and refuse identity changes."""

    if (
        len({state.uuid for state in selected}) != len(selected)
        or len({state.physical_index for state in selected}) != len(selected)
    ):
        raise RuntimeError("selected GPU identities must be unique before final probes")
    rechecked_states = []
    rejected: dict[str, list[str]] = {}
    for initial in selected:
        try:
            current = probe_gpu_uuid(initial.uuid)
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
            rejected[initial.uuid] = [f"final UUID probe failed: {error}"]
            continue
        reasons = current.rejection_reasons(
            max_utilization_percent=max_utilization_percent,
            min_free_memory_mib=min_free_memory_mib,
        )
        if current.uuid != initial.uuid:
            reasons.append(
                f"UUID identity changed from {initial.uuid} to {current.uuid}"
            )
        if current.physical_index != initial.physical_index:
            reasons.append(
                "physical index identity changed from "
                f"{initial.physical_index} to {current.physical_index}"
            )
        if reasons:
            rejected[initial.uuid] = reasons
        else:
            rechecked_states.append(current)
    if rejected:
        raise RuntimeError(f"selected GPU UUID(s) failed final idle probe: {rejected}")
    if len(rechecked_states) != len(selected):
        raise RuntimeError("final UUID probes did not return every selected GPU")
    return tuple(rechecked_states)


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
    gpu_count = validate_gpu_count(gpu_count)
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


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--gpu-count",
        type=int,
        required=True,
        help=(
            "Number of GPUs to select dynamically from the full NVIDIA inventory; "
            "must be 1 or 2."
        ),
    )
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
    return parser.parse_args(argv)


def main():
    args = _parse_args()
    if not RUN_NAME_PATTERN.fullmatch(args.run_name):
        raise ValueError("run-name must contain only letters, digits, '.', '_', or '-'")
    gpu_count = validate_gpu_count(args.gpu_count)
    if not 1 <= args.max_steps <= 1_000:
        raise ValueError("pilot max-steps must be in [1, 1000]")
    if min(args.global_batch_size, args.micro_batch_size) <= 0:
        raise ValueError("batch sizes must be positive")
    accumulation_steps = exact_accumulation_steps(
        args.global_batch_size,
        args.micro_batch_size,
        gpu_count,
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
        gpu_count=gpu_count,
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

    # Inventory every NVIDIA GPU only after all potentially expensive source and
    # checkpoint checks. Select from that point-in-time snapshot, then address
    # each chosen GPU by UUID for the final pre-launch safety probe. A cooperative
    # nvidia-smi check cannot provide an atomic lease, so any failed re-probe
    # aborts instead of using or interrupting a newly occupied device.
    gpu_inventory = probe_all_gpus()
    inventory_snapshot_completed_at_utc = datetime.now(timezone.utc).isoformat()
    initially_selected = select_idle_gpus(
        gpu_inventory,
        gpu_count=gpu_count,
        max_utilization_percent=args.max_utilization_percent,
        min_free_memory_mib=args.min_free_memory_mib,
    )
    gpu_states = reprobe_selected_gpus(
        initially_selected,
        max_utilization_percent=args.max_utilization_percent,
        min_free_memory_mib=args.min_free_memory_mib,
    )
    final_uuid_probes_completed_at_utc = datetime.now(timezone.utc).isoformat()

    visible_uuids = ",".join(state.uuid for state in gpu_states)
    environment_command = [
        "env",
        "CUDA_DEVICE_ORDER=PCI_BUS_ID",
        f"CUDA_VISIBLE_DEVICES={visible_uuids}",
        f"PYTHONPATH={REPOSITORY_ROOT / 'src'}:{REPOSITORY_ROOT}",
        *command,
    ]
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "bounded UDLM training pilot",
        "gpu_selection_schema_version": 2,
        "git_sha": git_sha,
        "run_name": args.run_name,
        "tmux_session": session_name,
        "user_requested_gpu_count": gpu_count,
        "gpu_selection_method": "dynamic_idle_discovery",
        "gpu_inventory_scope": "all_nvidia_gpus",
        "inventory_snapshot_completed_at_utc": inventory_snapshot_completed_at_utc,
        "gpu_inventory_at_selection": [
            asdict(state) for state in gpu_inventory
        ],
        "initially_selected_gpu_states": [
            asdict(state) for state in initially_selected
        ],
        "physical_gpu_indices": [state.physical_index for state in gpu_states],
        "logical_cuda_devices": list(range(gpu_count)),
        "cuda_visible_device_uuids": [state.uuid for state in gpu_states],
        "final_uuid_probes_completed_at_utc": final_uuid_probes_completed_at_utc,
        "gpu_states_at_final_uuid_probe": [asdict(state) for state in gpu_states],
        "gpu_safety_policy": {
            "max_utilization_percent": args.max_utilization_percent,
            "utilization_comparison": "strictly_less_than",
            "min_free_memory_mib": args.min_free_memory_mib,
            "active_compute_processes_allowed": False,
            "compute_mode_prohibited_allowed": False,
        },
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
            args.micro_batch_size * gpu_count * accumulation_steps
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

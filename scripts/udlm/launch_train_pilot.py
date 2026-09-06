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
import os
import re
import secrets
import shlex
import stat
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPOSITORY_ROOT.parents[1]
RUN_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
MAX_SAFE_UTILIZATION_PERCENT = 10
MIN_SAFE_FREE_MEMORY_MIB = 30_000
TRAINING_SUMMARY_SCHEMA_VERSION = 4
PILOT_EXIT_STATUS_SCHEMA_VERSION = 4
LAUNCH_MANIFEST_SCHEMA_VERSION = 1
MATCHED_PANEL_SCHEMA_VERSION = 1
TRAINING_JOB_LOCK_SCHEMA_VERSION = 1
TRAINING_JOB_LOCK_PURPOSES = frozenset(
    {
        "enforce_one_R_S_E_pilot_training_job_at_a_time",
        "enforce_one_registered_optimization_screen_job_at_a_time",
    }
)
MAX_TRAINING_SEED = 2**32 - 1
TRAINING_VARIANTS = {
    "udlm": {
        "config_name": "udlm",
        "prior_variant": "release_uniform",
        "comparison_role": "faithful_release_control",
        "fixed_overrides": (),
    },
    "schedule_uniform": {
        "config_name": "udlm",
        "prior_variant": "schedule_uniform",
        "comparison_role": "schedule_repair_uniform_control",
        "fixed_overrides": ("training.udlm.prior_variant=schedule_uniform",),
    },
    "udlm_categorical": {
        "config_name": "udlm_categorical",
        "prior_variant": "empirical_frequency",
        "comparison_role": "empirical_prior_treatment",
        "fixed_overrides": (),
    },
}
MATCHED_PANEL_VARIANT_ORDER = tuple(TRAINING_VARIANTS)
PILOT_ENVIRONMENT_PREFIX = "GENMOL_TRAIN_"
CONTROLLED_PYTHON_ENVIRONMENT = {
    "PYTHONNOUSERSITE": "1",
    "PYTHONOPTIMIZE": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
}
KNOWN_PYTHON_ENVIRONMENT_KEYS = {
    "PYTHONHOME",
    "PYTHONSTARTUP",
    "PYTHONPLATLIBDIR",
    "PYTHONUSERBASE",
    "PYTHONPYCACHEPREFIX",
    "PYTHONWARNINGS",
    "PYTHONBREAKPOINT",
    "PYTHONDEBUG",
    "PYTHONINSPECT",
    "PYTHONUNBUFFERED",
    "PYTHONVERBOSE",
    "PYTHONCASEOK",
    "PYTHONFAULTHANDLER",
    "PYTHONTRACEMALLOC",
    "PYTHONPROFILEIMPORTTIME",
    "PYTHONASYNCIODEBUG",
    "PYTHONMALLOC",
    "PYTHONCOERCECLOCALE",
    "PYTHONWARNDEFAULTENCODING",
    "PYTHONNODEBUGRANGES",
    "PYTHONINTMAXSTRDIGITS",
    "PYTHONSAFEPATH",
}
DISTRIBUTED_ENVIRONMENT_KEYS = {
    "GROUP_RANK",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "NODE_RANK",
    "RANK",
    "WORLD_SIZE",
}


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


def validate_training_variant(training_variant: str) -> str:
    """Allow only the three reviewed Hydra configurations."""

    if training_variant not in TRAINING_VARIANTS:
        allowed = ", ".join(TRAINING_VARIANTS)
        raise ValueError(f"training-variant must be one of: {allowed}")
    return training_variant


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
            "min-free-memory-mib cannot be below " f"{MIN_SAFE_FREE_MEMORY_MIB}"
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
        raise RuntimeError(
            f"nvidia-smi GPU query returned stderr: {status.stderr.strip()}"
        )
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

    if len({state.uuid for state in selected}) != len(selected) or len(
        {state.physical_index for state in selected}
    ) != len(selected):
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
        raise RuntimeError(
            "tracked working-tree changes must be committed before launch"
        )
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
    """Hash one stable regular-file descriptor and retain its path binding."""

    resolved = path.resolve(strict=True)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"checkpoint is not a regular file: {resolved}")
        state = _stable_stat_identity(before)
        digest = hashlib.sha256()
        offset = 0
        while True:
            chunk = os.pread(descriptor, 8 * 1024 * 1024, offset)
            if not chunk:
                break
            digest.update(chunk)
            offset += len(chunk)
        after = os.fstat(descriptor)
        path_state = os.stat(resolved, follow_symlinks=False)
        observed_states = [
            _stable_stat_identity(observed) for observed in (after, path_state)
        ]
        if any(observed != state for observed in observed_states):
            raise RuntimeError(f"checkpoint changed while it was hashed: {resolved}")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _stable_stat_identity(value: os.stat_result) -> tuple[int, ...]:
    """Return mutation-sensitive identity fields while deliberately ignoring atime."""

    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def matched_panel_config_sha256(resolved_config: dict[str, object]) -> str:
    """Hash the resolved config after masking only the registered treatment.

    Output-directory differences are also masked because each variant must write
    to its own run directory. Any other resolved-config difference changes the
    digest and therefore prevents the runs from claiming one matched panel.
    """

    try:
        normalized = json.loads(
            json.dumps(resolved_config, allow_nan=False, ensure_ascii=False)
        )
        prior_variant = normalized["training"]["udlm"]["prior_variant"]
        callback_dirpath = normalized["callback"]["dirpath"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("resolved config lacks the matched-panel fields") from error
    registered_priors = {
        variant["prior_variant"] for variant in TRAINING_VARIANTS.values()
    }
    if prior_variant not in registered_priors:
        raise ValueError("resolved config has an unregistered UDLM treatment")
    if not isinstance(callback_dirpath, str) or not callback_dirpath:
        raise ValueError("resolved config callback.dirpath must be a nonempty string")
    normalized["training"]["udlm"]["prior_variant"] = "<REGISTERED_TREATMENT>"
    normalized["callback"]["dirpath"] = "<VARIANT_RUN_DIR>/checkpoints"
    return canonical_json_sha256(normalized)


def build_matched_panel_spec(
    *,
    source_revision: str,
    checkpoint: Path | None,
    checkpoint_sha256: str | None,
    gpu_count: int,
    max_steps: int,
    global_batch_size: int,
    micro_batch_size: int,
    num_workers: int,
    seed: int,
    exclude_special_tokens: bool,
    max_utilization_percent: int,
    min_free_memory_mib: int,
    common_resolved_config_sha256: str,
) -> tuple[dict[str, object], str]:
    """Return the canonical common contract for the sequential R/S/E pilot.

    The three variants must be launched one at a time, in the registered order,
    and the next variant may start only after the prior run has a validated,
    successful exit receipt. The digest intentionally excludes run names,
    timestamps, and physical GPU identities, which are per-run provenance.
    """

    gpu_count = validate_gpu_count(gpu_count)
    validate_safety_thresholds(max_utilization_percent, min_free_memory_mib)
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ValueError("source_revision must be 40 lowercase hexadecimal digits")
    if not re.fullmatch(r"[0-9a-f]{64}", common_resolved_config_sha256):
        raise ValueError("common resolved-config digest must be SHA-256")
    if checkpoint is None:
        if checkpoint_sha256 is not None:
            raise ValueError("checkpoint digest requires a checkpoint")
        checkpoint_path = None
        initialization_mode = "scratch"
    else:
        if not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256 or ""):
            raise ValueError("matched warm start requires a checkpoint SHA-256")
        checkpoint_path = str(checkpoint)
        initialization_mode = "verified_mdlm_ema_warm_start"
    if (
        type(max_steps) is not int
        or not 1 <= max_steps <= 1_000
        or type(global_batch_size) is not int
        or type(micro_batch_size) is not int
        or type(num_workers) is not int
        or num_workers < 0
        or type(seed) is not int
        or not 0 <= seed <= MAX_TRAINING_SEED
        or type(exclude_special_tokens) is not bool
    ):
        raise ValueError("matched-panel training controls are invalid")
    accumulation_steps = exact_accumulation_steps(
        global_batch_size, micro_batch_size, gpu_count
    )
    spec = {
        "schema_version": MATCHED_PANEL_SCHEMA_VERSION,
        "purpose": "matched_R_S_E_UDLM_training_pilot",
        "execution": {
            "mode": "single_job_lease_with_registered_order_policy",
            "maximum_concurrent_training_jobs": 1,
            "concurrency_enforcement": "atomic_global_worktree_training_job_lock",
            "registered_variant_order": list(MATCHED_PANEL_VARIANT_ORDER),
            "advance_policy": "operator_validates_successful_predecessor_receipt",
            "predecessor_receipt_bound_in_each_manifest": False,
        },
        "registered_treatments": [
            {
                "training_variant": name,
                "hydra_config_name": definition["config_name"],
                "udlm_prior_variant": definition["prior_variant"],
                "comparison_role": definition["comparison_role"],
            }
            for name, definition in TRAINING_VARIANTS.items()
        ],
        "common_training_contract": {
            "source_revision": source_revision,
            "initialization_mode": initialization_mode,
            "initialization_checkpoint_path": checkpoint_path,
            "initialization_checkpoint_sha256": checkpoint_sha256,
            "requested_gpu_count": gpu_count,
            "max_steps": max_steps,
            "global_batch_size": global_batch_size,
            "micro_batch_size_per_process": micro_batch_size,
            "accumulate_grad_batches": accumulation_steps,
            "effective_global_batch_size": (
                micro_batch_size * gpu_count * accumulation_steps
            ),
            "num_workers": num_workers,
            "seed": seed,
            "exclude_special_tokens": exclude_special_tokens,
            "common_resolved_config_sha256": common_resolved_config_sha256,
        },
        "common_gpu_safety_policy": {
            "max_utilization_percent": max_utilization_percent,
            "utilization_comparison": "strictly_less_than",
            "min_free_memory_mib": min_free_memory_mib,
            "active_compute_processes_allowed": False,
            "compute_mode_prohibited_allowed": False,
            "physical_gpu_identity_is_per_run_provenance": True,
        },
    }
    return spec, canonical_json_sha256(spec)


def _atomic_publish_bytes_exclusive(path: Path, payload: bytes, *, label: str) -> str:
    """Publish complete bytes once using a same-directory hard-link commit."""

    if not isinstance(payload, bytes) or not payload:
        raise ValueError(f"{label} payload must be nonempty bytes")
    path = Path(os.path.abspath(os.fspath(path)))
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to replace {label}: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o644)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to replace {label}: {path}") from error
        temporary.unlink()
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(payload).hexdigest()


def training_job_lock_path() -> Path:
    """Return the one host-worktree lock shared by all reviewed pilot variants."""

    return REPOSITORY_ROOT / "output" / "udlm" / ".single_training_job.lock"


def acquire_training_job_lock(
    *,
    source_revision: str,
    run_name: str,
    training_variant: str,
    purpose: str = "enforce_one_R_S_E_pilot_training_job_at_a_time",
) -> tuple[Path, dict[str, object], str]:
    """Atomically claim the sole pilot training slot; never recover stale locks."""

    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ValueError("source_revision must be 40 lowercase hexadecimal digits")
    if not RUN_NAME_PATTERN.fullmatch(run_name):
        raise ValueError("run_name is invalid")
    validate_training_variant(training_variant)
    if purpose not in TRAINING_JOB_LOCK_PURPOSES:
        raise ValueError("training-job lock purpose is not reviewed")
    lock_path = training_job_lock_path()
    lock_record = {
        "schema_version": TRAINING_JOB_LOCK_SCHEMA_VERSION,
        "status": "held",
        "purpose": purpose,
        "source_revision": source_revision,
        "run_name": run_name,
        "training_variant": training_variant,
        "owner_token": secrets.token_hex(32),
        "launcher_pid_at_acquisition": os.getpid(),
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
        "owner_process_exit_does_not_make_lock_stale": True,
        "stale_lock_policy": "fail_closed_and_require_manual_review",
        "release_policy": (
            "exact_owner_lock_only_after_receipt_or_before_tmux_handoff_failure"
        ),
    }
    payload = (
        json.dumps(lock_record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    try:
        digest = _atomic_publish_bytes_exclusive(
            lock_path,
            payload,
            label="single pilot training-job lock",
        )
    except FileExistsError as error:
        raise RuntimeError(
            "another or stale pilot training-job lock exists; fail closed and "
            f"review it manually before any launch: {lock_path}"
        ) from error
    return lock_path, lock_record, digest


def release_exact_training_job_lock(path: Path, *, expected_sha256: str) -> None:
    """Remove only the unchanged regular lock owned by this launch attempt."""

    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("expected lock digest must be 64 lowercase hexadecimal digits")
    path = Path(os.path.abspath(os.fspath(path)))
    before_path = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before_path.st_mode):
        raise RuntimeError("training-job lock is not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    try:
        before_descriptor = os.fstat(descriptor)
        if _stable_stat_identity(before_descriptor) != _stable_stat_identity(
            before_path
        ):
            raise RuntimeError("training-job lock changed before exact release")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after_descriptor = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = path.stat(follow_symlinks=False)
    stable_identity = _stable_stat_identity(before_path)
    if (
        _stable_stat_identity(after_descriptor) != stable_identity
        or _stable_stat_identity(after_path) != stable_identity
    ):
        raise RuntimeError("training-job lock changed during exact release")
    if digest.hexdigest() != expected_sha256:
        raise RuntimeError(
            "refusing to release a training-job lock owned by another run"
        )
    os.unlink(path)
    directory_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def reserve_log_path(path: Path) -> None:
    """Reserve a new regular log file without following or replacing a path."""

    path = Path(os.path.abspath(os.fspath(path)))
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to replace pilot log: {path}") from error
    try:
        state = os.fstat(descriptor)
        if not stat.S_ISREG(state.st_mode) or state.st_nlink != 1:
            raise RuntimeError("reserved pilot log is not a single-link regular file")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def validate_pilot_exit_receipt_path(path: Path) -> Path:
    """Require one new JSON receipt path inside this source worktree."""

    repository_root = REPOSITORY_ROOT.resolve(strict=True)
    absolute = Path(os.path.abspath(os.fspath(path)))
    resolved = absolute.parent.resolve(strict=False) / absolute.name
    if (
        resolved == repository_root
        or repository_root not in resolved.parents
        or resolved.suffix != ".json"
    ):
        raise ValueError("pilot exit receipt must be an in-repository .json file")
    if os.path.lexists(resolved):
        raise FileExistsError(f"refusing to replace pilot exit receipt: {resolved}")
    return resolved


def training_argv_sha256(command: list[str]) -> str:
    """Fingerprint exactly the argv observed by scripts/train.py."""

    expected_script = str(REPOSITORY_ROOT / "scripts" / "train.py")
    if len(command) < 5 or command[2] != expected_script:
        raise ValueError("training command does not invoke the reviewed train.py")
    return canonical_json_sha256(command[2:])


def compose_resolved_training_config(
    *,
    config_name: str,
    overrides: list[str],
    gpu_count: int,
) -> tuple[dict[str, object], str]:
    """Compose the exact Hydra task config before any GPU is exposed."""

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    validate_gpu_count(gpu_count)
    if config_name not in {
        str(variant["config_name"]) for variant in TRAINING_VARIANTS.values()
    }:
        raise ValueError(f"unreviewed Hydra config name: {config_name}")
    resolvers = {
        "cwd": lambda: str(REPOSITORY_ROOT),
        "device_count": lambda: gpu_count,
        "eval": lambda expression: eval(expression, {"__builtins__": {}}, {}),
        "div_up": lambda x, y: (x + y - 1) // y,
    }
    for name, resolver in resolvers.items():
        if OmegaConf.has_resolver(name):
            OmegaConf.clear_resolver(name)
        OmegaConf.register_new_resolver(name, resolver)
    with initialize_config_dir(
        version_base=None,
        config_dir=str(REPOSITORY_ROOT / "configs"),
    ):
        config = compose(
            config_name=config_name,
            overrides=overrides,
            return_hydra_config=False,
        )
    resolved = OmegaConf.to_container(config, resolve=True, enum_to_str=True)
    if not isinstance(resolved, dict):
        raise RuntimeError("resolved Hydra task config must be a mapping")
    return resolved, canonical_json_sha256(resolved)


def build_child_environment_command(
    *,
    command: list[str],
    source_revision: str,
    resolved_config_sha256: str,
    runtime_config_path: Path,
    training_summary_path: Path,
    final_checkpoint_path: Path,
    launch_manifest_path: Path,
    launch_manifest_sha256: str,
    expected_max_steps: int,
    expected_world_size: int,
    visible_uuids: str,
    seed: int,
) -> tuple[list[str], dict[str, str]]:
    """Sanitize Python controls and bind the child to source, argv, and config."""

    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ValueError("source_revision must be 40 lowercase hexadecimal digits")
    if not re.fullmatch(r"[0-9a-f]{64}", resolved_config_sha256):
        raise ValueError(
            "resolved_config_sha256 must be 64 lowercase hexadecimal digits"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", launch_manifest_sha256):
        raise ValueError(
            "launch_manifest_sha256 must be 64 lowercase hexadecimal digits"
        )
    if not visible_uuids or any(
        not value.startswith("GPU-") for value in visible_uuids.split(",")
    ):
        raise ValueError("visible_uuids must contain NVIDIA GPU UUIDs")
    artifact_paths = {
        "runtime config record": (runtime_config_path.resolve(), ".json"),
        "training summary": (training_summary_path.resolve(), ".json"),
        "final checkpoint": (final_checkpoint_path.resolve(), ".ckpt"),
        "launch manifest": (launch_manifest_path.resolve(), ".json"),
    }
    for label, (path, suffix) in artifact_paths.items():
        if (
            path == REPOSITORY_ROOT
            or REPOSITORY_ROOT not in path.parents
            or path.suffix != suffix
        ):
            raise ValueError(f"{label} must be an in-repository {suffix} file")
    runtime_config_path = artifact_paths["runtime config record"][0]
    training_summary_path = artifact_paths["training summary"][0]
    final_checkpoint_path = artifact_paths["final checkpoint"][0]
    launch_manifest_path = artifact_paths["launch manifest"][0]
    if (
        len(
            {
                runtime_config_path,
                training_summary_path,
                final_checkpoint_path,
                launch_manifest_path,
            }
        )
        != 4
    ):
        raise ValueError("pilot completion artifact paths must be distinct")
    if (
        type(expected_max_steps) is not int
        or expected_max_steps <= 0
        or type(expected_world_size) is not int
        or expected_world_size not in (1, 2)
    ):
        raise ValueError("pilot expected steps/world size are invalid")
    controlled_python = {
        **CONTROLLED_PYTHON_ENVIRONMENT,
        "PYTHONPATH": os.pathsep.join(
            [str(REPOSITORY_ROOT / "src"), str(REPOSITORY_ROOT)]
        ),
        "PYTHONHASHSEED": str(seed),
    }
    pilot_environment = {
        "GENMOL_TRAIN_EXPECTED_SOURCE_REVISION": source_revision,
        "GENMOL_TRAIN_EXPECTED_CONFIG_SHA256": resolved_config_sha256,
        "GENMOL_TRAIN_EXPECTED_ARGV_SHA256": training_argv_sha256(command),
        "GENMOL_TRAIN_RUNTIME_CONFIG_PATH": str(runtime_config_path),
        "GENMOL_TRAIN_SUMMARY_PATH": str(training_summary_path),
        "GENMOL_TRAIN_EXPECTED_SUMMARY_SCHEMA_VERSION": str(
            TRAINING_SUMMARY_SCHEMA_VERSION
        ),
        "GENMOL_TRAIN_EXPECTED_FINAL_CHECKPOINT_PATH": str(final_checkpoint_path),
        "GENMOL_TRAIN_LAUNCH_MANIFEST_PATH": str(launch_manifest_path),
        "GENMOL_TRAIN_EXPECTED_LAUNCH_MANIFEST_SHA256": launch_manifest_sha256,
        "GENMOL_TRAIN_SELECTED_GPU_UUIDS_JSON": json.dumps(
            visible_uuids.split(","), separators=(",", ":")
        ),
        "GENMOL_TRAIN_EXPECTED_MAX_STEPS": str(expected_max_steps),
        "GENMOL_TRAIN_EXPECTED_WORLD_SIZE": str(expected_world_size),
    }
    assigned_environment = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": visible_uuids,
        **controlled_python,
        **pilot_environment,
    }
    inherited_python_keys = {key for key in os.environ if key.startswith("PYTHON")}
    inherited_pilot_keys = {
        key for key in os.environ if key.startswith(PILOT_ENVIRONMENT_PREFIX)
    }
    unset_environment_keys = sorted(
        KNOWN_PYTHON_ENVIRONMENT_KEYS
        | inherited_python_keys
        | inherited_pilot_keys
        | set(controlled_python)
        | DISTRIBUTED_ENVIRONMENT_KEYS
    )
    environment_command = ["env"]
    for key in unset_environment_keys:
        environment_command.extend(["-u", key])
    environment_command.extend(
        f"{key}={value}" for key, value in assigned_environment.items()
    )
    environment_command.extend(command)
    return environment_command, assigned_environment


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
    checkpoint_sha256: str | None = None,
    exclude_special_tokens: bool,
    training_variant: str = "udlm",
) -> list[str]:
    gpu_count = validate_gpu_count(gpu_count)
    training_variant = validate_training_variant(training_variant)
    if checkpoint is None and checkpoint_sha256 is not None:
        raise ValueError("checkpoint_sha256 requires a checkpoint")
    if checkpoint_sha256 is not None and (
        not isinstance(checkpoint_sha256, str)
        or len(checkpoint_sha256) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint_sha256)
    ):
        raise ValueError("checkpoint_sha256 must be 64 lowercase hexadecimal digits")
    variant = TRAINING_VARIANTS[training_variant]
    command = [
        str(_python_executable()),
        "-u",
        str(REPOSITORY_ROOT / "scripts" / "train.py"),
        "--config-name",
        str(variant["config_name"]),
        f"seed={seed}",
        f"trainer.devices={gpu_count}",
        f"trainer.max_steps={max_steps}",
        "trainer.detect_anomaly=true",
        f"loader.global_batch_size={global_batch_size}",
        f"loader.batch_size={micro_batch_size}",
        f"loader.num_workers={num_workers}",
        f"callback.every_n_train_steps={max_steps}",
        f"callback.dirpath={run_dir / 'checkpoints'}",
        f"hydra.run.dir={run_dir / 'hydra'}",
        "training.pilot_fail_on_nonfinite_loss=true",
        f"training.udlm.exclude_special_tokens={str(exclude_special_tokens).lower()}",
        *variant["fixed_overrides"],
    ]
    if checkpoint is not None:
        command.append(f"training.init_from_mdlm_checkpoint={checkpoint}")
        if checkpoint_sha256 is not None:
            command.append(
                "training.init_from_mdlm_checkpoint_sha256=" + checkpoint_sha256
            )
        command.append("training.init_from_mdlm_ema=true")
    return command


def build_tmux_shell_command(
    environment_command: list[str],
    *,
    log_path: Path,
    training_summary_path: Path,
    exit_receipt_path: Path,
    expected_source_revision: str,
    expected_config_sha256: str,
    expected_argv_sha256: str,
    expected_summary_schema_version: int,
    expected_max_steps: int,
    expected_world_size: int,
    expected_final_checkpoint_path: Path,
    expected_launch_manifest_path: Path,
    expected_launch_manifest_sha256: str,
    expected_selected_gpu_uuids_json: str,
    expected_training_job_lock_path: Path,
    expected_training_job_lock_sha256: str,
    expected_initialization_checkpoint_sha256: str | None = None,
) -> str:
    """Capture both pipeline statuses and publish the detached-run exit receipt."""

    receipt_path = validate_pilot_exit_receipt_path(exit_receipt_path)
    if not re.fullmatch(r"[0-9a-f]{40}", expected_source_revision):
        raise ValueError("expected source revision must be a full commit hash")
    for label, digest in (
        ("expected config digest", expected_config_sha256),
        ("expected argv digest", expected_argv_sha256),
        ("expected launch-manifest digest", expected_launch_manifest_sha256),
        ("expected training-job lock digest", expected_training_job_lock_sha256),
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"{label} must be 64 lowercase hexadecimal digits")
    if expected_initialization_checkpoint_sha256 is not None and not re.fullmatch(
        r"[0-9a-f]{64}", expected_initialization_checkpoint_sha256
    ):
        raise ValueError(
            "expected initialization checkpoint digest must be 64 lowercase "
            "hexadecimal digits"
        )
    if expected_summary_schema_version != TRAINING_SUMMARY_SCHEMA_VERSION:
        raise ValueError("unexpected training summary schema version")
    if (
        type(expected_max_steps) is not int
        or expected_max_steps <= 0
        or type(expected_world_size) is not int
        or expected_world_size not in (1, 2)
    ):
        raise ValueError("pilot expected steps/world size are invalid")
    try:
        selected_gpu_uuids = json.loads(expected_selected_gpu_uuids_json)
    except json.JSONDecodeError as error:
        raise ValueError("selected GPU UUIDs must be canonical JSON") from error
    if (
        not isinstance(selected_gpu_uuids, list)
        or len(selected_gpu_uuids) != expected_world_size
        or len(set(selected_gpu_uuids)) != len(selected_gpu_uuids)
        or any(
            not isinstance(value, str) or not value.startswith("GPU-")
            for value in selected_gpu_uuids
        )
        or json.dumps(selected_gpu_uuids, separators=(",", ":"))
        != expected_selected_gpu_uuids_json
    ):
        raise ValueError("selected GPU UUIDs must be a unique canonical JSON list")

    receipt_python_environment = {
        **CONTROLLED_PYTHON_ENVIRONMENT,
        "PYTHONPATH": os.pathsep.join(
            [str(REPOSITORY_ROOT / "src"), str(REPOSITORY_ROOT)]
        ),
        "PYTHONHASHSEED": "0",
    }
    receipt_command_parts = ["env"]
    receipt_python_keys = sorted(
        KNOWN_PYTHON_ENVIRONMENT_KEYS
        | {key for key in os.environ if key.startswith("PYTHON")}
        | set(receipt_python_environment)
    )
    for key in receipt_python_keys:
        receipt_command_parts.extend(["-u", key])
    receipt_command_parts.extend(
        f"{key}={value}" for key, value in receipt_python_environment.items()
    )
    receipt_command_parts.extend(
        [
            str(_python_executable()),
            "-u",
            str(REPOSITORY_ROOT / "scripts" / "udlm" / "write_pilot_exit_status.py"),
            "--training-exit-status",
            "GENMOL_PIPELINE_TRAINING_STATUS",
            "--tee-exit-status",
            "GENMOL_PIPELINE_TEE_STATUS",
            "--training-summary-path",
            str(training_summary_path),
            "--receipt-path",
            str(receipt_path),
            "--expected-summary-schema-version",
            str(expected_summary_schema_version),
            "--expected-source-revision",
            expected_source_revision,
            "--expected-config-sha256",
            expected_config_sha256,
            "--expected-argv-sha256",
            expected_argv_sha256,
            "--expected-max-steps",
            str(expected_max_steps),
            "--expected-world-size",
            str(expected_world_size),
            "--expected-final-checkpoint-path",
            str(expected_final_checkpoint_path),
            "--expected-launch-manifest-path",
            str(expected_launch_manifest_path),
            "--expected-launch-manifest-sha256",
            expected_launch_manifest_sha256,
            "--expected-selected-gpu-uuids-json",
            expected_selected_gpu_uuids_json,
            "--training-job-lock-path",
            str(expected_training_job_lock_path),
            "--expected-training-job-lock-sha256",
            expected_training_job_lock_sha256,
        ]
    )
    if expected_initialization_checkpoint_sha256 is not None:
        receipt_command_parts.extend(
            [
                "--expected-initialization-checkpoint-sha256",
                expected_initialization_checkpoint_sha256,
            ]
        )
    shell_status_arguments = {
        "GENMOL_PIPELINE_TRAINING_STATUS": '"$training_status"',
        "GENMOL_PIPELINE_TEE_STATUS": '"$tee_status"',
    }
    receipt_command = " ".join(
        shell_status_arguments.get(part, shlex.quote(part))
        for part in receipt_command_parts
    )
    return (
        "set +e; set -o pipefail; "
        + shlex.join(environment_command)
        + " 2>&1 | tee -a "
        + shlex.quote(str(log_path))
        + '; pipeline_status=("${PIPESTATUS[@]}"); '
        + 'training_status="${pipeline_status[0]}"; '
        + 'tee_status="${pipeline_status[1]}"; '
        + receipt_command
        + '; receipt_writer_status=$?; exit "$receipt_writer_status"'
    )


def _launch_locked_pilot(
    *,
    args: argparse.Namespace,
    git_sha: str,
    checkpoint: Path | None,
    checkpoint_sha256: str | None,
    command: list[str],
    resolved_config: dict[str, object],
    resolved_config_sha256: str,
    argv_sha256: str,
    matched_panel_spec: dict[str, object],
    matched_panel_spec_sha256: str,
    accumulation_steps: int,
    session_name: str,
    lock_path: Path,
    lock_record: dict[str, object],
    lock_sha256: str,
) -> tuple[bytes, str, Path]:
    """Probe, publish, and hand one lock-owning pilot to detached tmux."""

    gpu_count = validate_gpu_count(args.gpu_count)
    training_variant = validate_training_variant(args.training_variant)
    variant = TRAINING_VARIANTS[training_variant]
    run_dir = REPOSITORY_ROOT / "output" / "udlm" / args.run_name
    log_path = REPOSITORY_ROOT / "output" / "logs" / f"{args.run_name}.log"
    manifest_path = run_dir / "launch_manifest.json"

    # This phase begins only after the global training-job lock is held, so an
    # overlapping or stale owner fails before either invocation can probe GPUs.
    gpu_inventory = probe_all_gpus()
    inventory_snapshot_completed_at_utc = datetime.now(timezone.utc).isoformat()
    initially_selected = select_idle_gpus(
        gpu_inventory,
        gpu_count=gpu_count,
        max_utilization_percent=args.max_utilization_percent,
        min_free_memory_mib=args.min_free_memory_mib,
    )
    runtime_config_path = run_dir / "runtime_config.json"
    training_summary_path = run_dir / "training_summary.json"
    exit_receipt_path = validate_pilot_exit_receipt_path(
        run_dir / "pilot_exit_status.json"
    )
    final_checkpoint_path = run_dir / "checkpoints" / f"{args.max_steps}.ckpt"
    source_revision_before_final_gpu_probe = require_pushed_commit()
    if source_revision_before_final_gpu_probe != git_sha:
        raise RuntimeError("source revision changed before the pilot's final GPU probe")
    gpu_states = reprobe_selected_gpus(
        initially_selected,
        max_utilization_percent=args.max_utilization_percent,
        min_free_memory_mib=args.min_free_memory_mib,
    )
    final_uuid_probes_completed_at_utc = datetime.now(timezone.utc).isoformat()
    selected_gpu_uuids = [state.uuid for state in gpu_states]
    selected_gpu_uuids_json = json.dumps(selected_gpu_uuids, separators=(",", ":"))
    visible_uuids = ",".join(selected_gpu_uuids)

    # Reserve every externally visible destination before publishing the
    # immutable launch certificate. Retained partial reservations make a failed
    # launch non-reusable rather than silently overwritable.
    run_dir.mkdir(parents=True)
    reserve_log_path(log_path)
    manifest = {
        "launch_manifest_schema_version": LAUNCH_MANIFEST_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "bounded UDLM training pilot",
        "gpu_selection_schema_version": 2,
        "git_sha": git_sha,
        "source_revision_before_final_gpu_probe": (
            source_revision_before_final_gpu_probe
        ),
        "run_name": args.run_name,
        "training_variant": training_variant,
        "hydra_config_name": variant["config_name"],
        "udlm_prior_variant": variant["prior_variant"],
        "udlm_comparison_role": variant["comparison_role"],
        "matched_panel_spec": matched_panel_spec,
        "matched_panel_spec_sha256": matched_panel_spec_sha256,
        "matched_panel_variant_position": MATCHED_PANEL_VARIANT_ORDER.index(
            training_variant
        ),
        "single_training_job_lock": {
            "path": str(lock_path),
            "sha256": lock_sha256,
            "record": lock_record,
            "acquired_before_any_gpu_probe": True,
            "stale_lock_policy": "fail_closed_and_require_manual_review",
            "release_owner": "pilot_exit_receipt_writer_after_publication",
        },
        "tmux_session": session_name,
        "user_requested_gpu_count": gpu_count,
        "gpu_selection_method": "dynamic_idle_discovery",
        "gpu_inventory_scope": "all_nvidia_gpus",
        "inventory_snapshot_completed_at_utc": inventory_snapshot_completed_at_utc,
        "gpu_inventory_at_selection": [asdict(state) for state in gpu_inventory],
        "initially_selected_gpu_states": [
            asdict(state) for state in initially_selected
        ],
        "logical_cuda_devices": list(range(gpu_count)),
        "physical_gpu_indices": [state.physical_index for state in gpu_states],
        "cuda_visible_device_uuids": selected_gpu_uuids,
        "final_uuid_probes_completed_at_utc": final_uuid_probes_completed_at_utc,
        "gpu_states_at_final_uuid_probe": [asdict(state) for state in gpu_states],
        "gpu_safety_policy": {
            "max_utilization_percent": args.max_utilization_percent,
            "utilization_comparison": "strictly_less_than",
            "min_free_memory_mib": args.min_free_memory_mib,
            "active_compute_processes_allowed": False,
            "compute_mode_prohibited_allowed": False,
        },
        "training_argv": command,
        "training_argv_sha256": argv_sha256,
        "resolved_training_config": resolved_config,
        "resolved_training_config_sha256": resolved_config_sha256,
        "runtime_config_path": str(runtime_config_path),
        "training_summary_path": str(training_summary_path),
        "training_summary_schema_version": TRAINING_SUMMARY_SCHEMA_VERSION,
        "pilot_exit_status_path": str(exit_receipt_path),
        "pilot_exit_status_schema_version": PILOT_EXIT_STATUS_SCHEMA_VERSION,
        "expected_final_checkpoint_path": str(final_checkpoint_path),
        "launch_manifest_path": str(manifest_path),
        "launch_manifest_raw_sha256_transport": (
            "passed_out_of_band_to_training_and_receipt_to_avoid_self_hash"
        ),
        "completion_contract": {
            "status_at_launch": "pending",
            "complete_only_if_valid_training_summary_exists": True,
            "complete_only_if_successful_exit_receipt_exists": True,
            "valid_training_summary_and_successful_exit_receipt_both_required": True,
            "missing_summary_after_tmux_exit_means": "incomplete",
            "absent_exit_receipt_means": "incomplete",
            "successful_exit_receipt_requires": {
                "training_exit_status": 0,
                "tee_exit_status": 0,
                "valid_launch_bound_training_summary": True,
                "exact_launch_manifest_still_matches": True,
                "clean_pushed_source_at_receipt": True,
            },
            "training_job_lock_release": (
                "after_exit_receipt_publication_for_completed_or_failed_pipeline"
            ),
        },
        "log_path": str(log_path),
        "log_reserved_exclusively_before_manifest": True,
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
        "dry_run": False,
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    manifest_sha256 = _atomic_publish_bytes_exclusive(
        manifest_path,
        manifest_bytes,
        label="pilot launch manifest",
    )
    environment_command, _child_environment = build_child_environment_command(
        command=command,
        source_revision=git_sha,
        resolved_config_sha256=resolved_config_sha256,
        runtime_config_path=runtime_config_path,
        training_summary_path=training_summary_path,
        final_checkpoint_path=final_checkpoint_path,
        launch_manifest_path=manifest_path,
        launch_manifest_sha256=manifest_sha256,
        expected_max_steps=args.max_steps,
        expected_world_size=gpu_count,
        visible_uuids=visible_uuids,
        seed=args.seed,
    )
    shell_command = build_tmux_shell_command(
        environment_command,
        log_path=log_path,
        training_summary_path=training_summary_path,
        exit_receipt_path=exit_receipt_path,
        expected_source_revision=git_sha,
        expected_config_sha256=resolved_config_sha256,
        expected_argv_sha256=argv_sha256,
        expected_summary_schema_version=TRAINING_SUMMARY_SCHEMA_VERSION,
        expected_max_steps=args.max_steps,
        expected_world_size=gpu_count,
        expected_final_checkpoint_path=final_checkpoint_path,
        expected_launch_manifest_path=manifest_path,
        expected_launch_manifest_sha256=manifest_sha256,
        expected_selected_gpu_uuids_json=selected_gpu_uuids_json,
        expected_training_job_lock_path=lock_path,
        expected_training_job_lock_sha256=lock_sha256,
        expected_initialization_checkpoint_sha256=checkpoint_sha256,
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
    return manifest_bytes, manifest_sha256, log_path


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--training-variant",
        choices=tuple(TRAINING_VARIANTS),
        default="udlm",
        help=(
            "Reviewed pilot only: faithful udlm, matched schedule_uniform control, "
            "or empirical-prior udlm_categorical."
        ),
    )
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


def tmux_session_exists(session_name: str) -> bool:
    """Return exact tmux-session presence; reject an indeterminate query."""

    result = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"could not determine whether tmux session exists: {session_name}"
        )
    return result.returncode == 0


def main():
    args = _parse_args()
    if not RUN_NAME_PATTERN.fullmatch(args.run_name):
        raise ValueError("run-name must contain only letters, digits, '.', '_', or '-'")
    gpu_count = validate_gpu_count(args.gpu_count)
    training_variant = validate_training_variant(args.training_variant)
    variant = TRAINING_VARIANTS[training_variant]
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
    if os.path.lexists(run_dir) or os.path.lexists(log_path):
        raise FileExistsError(
            f"refusing to overwrite an existing pilot: {run_dir} or {log_path}"
        )
    checkpoint_sha256 = None if checkpoint is None else sha256_file(checkpoint)
    command = build_training_command(
        gpu_count=gpu_count,
        run_dir=run_dir,
        max_steps=args.max_steps,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        exclude_special_tokens=args.exclude_special_tokens,
        training_variant=training_variant,
    )
    resolved_config, resolved_config_sha256 = compose_resolved_training_config(
        config_name=str(variant["config_name"]),
        overrides=command[5:],
        gpu_count=gpu_count,
    )
    try:
        resolved_prior_variant = resolved_config["training"]["udlm"]["prior_variant"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            "resolved training config lacks its UDLM treatment"
        ) from error
    if resolved_prior_variant != variant["prior_variant"]:
        raise RuntimeError(
            "resolved UDLM treatment disagrees with the registered training variant"
        )
    argv_sha256 = training_argv_sha256(command)
    common_resolved_config_sha256 = matched_panel_config_sha256(resolved_config)
    matched_panel_spec, matched_panel_spec_sha256 = build_matched_panel_spec(
        source_revision=git_sha,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        gpu_count=gpu_count,
        max_steps=args.max_steps,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        exclude_special_tokens=args.exclude_special_tokens,
        max_utilization_percent=args.max_utilization_percent,
        min_free_memory_mib=args.min_free_memory_mib,
        common_resolved_config_sha256=common_resolved_config_sha256,
    )

    # A dry run is a CPU-only preview. In particular, it neither calls
    # nvidia-smi nor creates/reserves any output, log, manifest, or tmux object.
    if args.dry_run:
        preview = {
            "schema_version": 1,
            "status": "dry_run_preflight_completed_no_launch",
            "project_launch_artifact_mutation_performed": False,
            "gpu_probe_performed": False,
            "tmux_operation_performed": False,
            "source_revision": git_sha,
            "run_name": args.run_name,
            "training_variant": training_variant,
            "predicted_launch_manifest_path": str(manifest_path),
            "predicted_log_path": str(log_path),
            "training_argv": command,
            "training_argv_sha256": argv_sha256,
            "resolved_training_config": resolved_config,
            "resolved_training_config_sha256": resolved_config_sha256,
            "matched_panel_spec": matched_panel_spec,
            "matched_panel_spec_sha256": matched_panel_spec_sha256,
        }
        print(json.dumps(preview, indent=2, sort_keys=True))
        return

    session_name = f"genmol_{training_variant}_{args.run_name}"
    if tmux_session_exists(session_name):
        raise RuntimeError(f"tmux session already exists: {session_name}")

    lock_path, lock_record, lock_sha256 = acquire_training_job_lock(
        source_revision=git_sha,
        run_name=args.run_name,
        training_variant=training_variant,
    )
    try:
        manifest_bytes, manifest_sha256, launched_log_path = _launch_locked_pilot(
            args=args,
            git_sha=git_sha,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            command=command,
            resolved_config=resolved_config,
            resolved_config_sha256=resolved_config_sha256,
            argv_sha256=argv_sha256,
            matched_panel_spec=matched_panel_spec,
            matched_panel_spec_sha256=matched_panel_spec_sha256,
            accumulation_steps=accumulation_steps,
            session_name=session_name,
            lock_path=lock_path,
            lock_record=lock_record,
            lock_sha256=lock_sha256,
        )
    except BaseException as launch_error:
        try:
            handoff_may_have_succeeded = tmux_session_exists(session_name)
        except BaseException as verification_error:
            raise RuntimeError(
                "pilot launch failed with an indeterminate tmux handoff; the exact "
                "training-job lock was retained fail-closed for manual review: "
                f"{type(verification_error).__name__}: {verification_error}"
            ) from launch_error
        if handoff_may_have_succeeded:
            raise RuntimeError(
                "pilot launch raised after tmux may have accepted the detached job; "
                "the exact training-job lock was retained fail-closed"
            ) from launch_error
        try:
            release_exact_training_job_lock(
                lock_path,
                expected_sha256=lock_sha256,
            )
        except Exception as release_error:
            raise RuntimeError(
                "pilot launch failed before tmux handoff and its exact training-job "
                f"lock could not be released: {type(release_error).__name__}: "
                f"{release_error}"
            ) from launch_error
        raise
    print(manifest_bytes.decode("utf-8"), end="")
    print(f"launch manifest SHA-256: {manifest_sha256}")
    print(f"launched tmux session {session_name}; log: {launched_log_path}")


if __name__ == "__main__":
    main()

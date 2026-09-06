"""Launch a bounded UDLM pilot on dynamically selected idle GPUs.

The caller chooses the number of GPUs (one or two) and explicitly supplies the
matched-panel predecessor state: genesis for R, R's successful receipt for S,
or S's successful receipt for E. Immediately before launch, the controller
inventories every NVIDIA GPU, selects genuinely idle devices, re-probes those
exact UUIDs, and exposes the UUIDs as the child's logical CUDA devices. It
records active compute processes without rejecting a device solely for their
presence. It never interrupts or kills an existing process.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import secrets
import shlex
import stat
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPOSITORY_ROOT.parents[1]
RUN_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
MAX_SAFE_UTILIZATION_PERCENT = 10
MIN_SAFE_FREE_MEMORY_MIB = 30_000
# User-authorized shared-server policy: a GPU below the strict utilization
# threshold remains eligible when process telemetry is nonempty. Processes are
# still captured in both selection and final-probe evidence.
ACTIVE_COMPUTE_PROCESSES_ALLOWED = True
TRAINING_SUMMARY_SCHEMA_VERSION = 5
PILOT_EXIT_STATUS_SCHEMA_VERSION = 5
LAUNCH_MANIFEST_SCHEMA_VERSION = 2
MATCHED_PANEL_SCHEMA_VERSION = 2
PREDECESSOR_RECEIPT_BINDING_SCHEMA_VERSION = 1
TRAINING_JOB_LOCK_SCHEMA_VERSION = 1
TRAINING_JOB_LOCK_PURPOSES = frozenset(
    {
        "enforce_one_R_S_E_pilot_training_job_at_a_time",
        "enforce_one_registered_optimization_screen_job_at_a_time",
    }
)
MAX_TRAINING_SEED = 2**32 - 1
# Pilot-only hyperparameter selected by the immutable CPU training-block audit.
# The 0.01 defaults in base.yaml and udlm_categorical.yaml intentionally remain
# unchanged for historical/manual replay.  Applying this override to every
# matched R/S/E launch keeps the non-treatment config common; release_uniform
# and schedule_uniform do not consume the empirical-prior mixture weight.
PILOT_EMPIRICAL_UNIFORM_MIX = 0.0002
PILOT_EMPIRICAL_UNIFORM_MIX_OVERRIDE = (
    f"training.udlm.empirical_uniform_mix={PILOT_EMPIRICAL_UNIFORM_MIX}"
)
PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_PATH = (
    "experiments/udlm/prior_geometry/" "floor_selection_train_rows_10001_30000.json"
)
PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_SHA256 = (
    "02908dafaf589ca9a49e560aa1eab470a18d6bfe616b781164784c489f54a9f1"
)
PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_CANONICAL_SHA256 = (
    "2435a36af83e88a1bb1d602e840bb6ae1e2a97963a48abf68606a37e6320694d"
)
PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_SOURCE_REVISION = (
    "6424b323084358ea050ba22d7e13ef8d45962496"
)
PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_SCOPE = (
    "retrospective_training_only_engineering_selection"
)
TRAINING_VARIANTS = {
    "udlm": {
        "config_name": "udlm",
        "prior_variant": "release_uniform",
        "comparison_role": "faithful_release_control",
        "fixed_overrides": (PILOT_EMPIRICAL_UNIFORM_MIX_OVERRIDE,),
    },
    "schedule_uniform": {
        "config_name": "udlm",
        "prior_variant": "schedule_uniform",
        "comparison_role": "schedule_repair_uniform_control",
        "fixed_overrides": (
            "training.udlm.prior_variant=schedule_uniform",
            PILOT_EMPIRICAL_UNIFORM_MIX_OVERRIDE,
        ),
    },
    "udlm_categorical": {
        "config_name": "udlm_categorical",
        "prior_variant": "empirical_frequency",
        "comparison_role": "empirical_prior_treatment",
        "fixed_overrides": (PILOT_EMPIRICAL_UNIFORM_MIX_OVERRIDE,),
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
        if self.compute_processes and not ACTIVE_COMPUTE_PROCESSES_ALLOWED:
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

    The three variants must be launched one at a time, in the registered order.
    The launcher requires an explicit genesis declaration for R and validates
    and binds the immediately prior run's exact successful receipt for S/E.
    The digest intentionally excludes run names, timestamps, and physical GPU
    identities, which are per-run provenance.
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
            "mode": (
                "single_job_lease_with_machine_enforced_predecessor_receipt_chain"
            ),
            "maximum_concurrent_training_jobs": 1,
            "concurrency_enforcement": "atomic_global_worktree_training_job_lock",
            "registered_variant_order": list(MATCHED_PANEL_VARIANT_ORDER),
            "advance_policy": (
                "launcher_validates_and_binds_exact_successful_predecessor_receipt"
            ),
            "predecessor_receipt_bound_in_each_manifest": True,
            "genesis_requires_explicit_declaration": True,
            "successor_launch_requires_exact_predecessor_receipt": True,
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
            "empirical_uniform_mix": PILOT_EMPIRICAL_UNIFORM_MIX,
            "empirical_uniform_mix_consumed_only_by": "empirical_frequency",
            "empirical_uniform_mix_audit": {
                "relative_path": PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_PATH,
                "sha256": PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_SHA256,
                "source_revision": (PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_SOURCE_REVISION),
                "scope": PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_SCOPE,
            },
            "common_resolved_config_sha256": common_resolved_config_sha256,
        },
        "common_gpu_safety_policy": {
            "max_utilization_percent": max_utilization_percent,
            "utilization_comparison": "strictly_less_than",
            "min_free_memory_mib": min_free_memory_mib,
            "active_compute_processes_allowed": ACTIVE_COMPUTE_PROCESSES_ALLOWED,
            "compute_mode_prohibited_allowed": False,
            "physical_gpu_identity_is_per_run_provenance": True,
        },
    }
    return spec, canonical_json_sha256(spec)


_STABLE_ARTIFACT_SNAPSHOT_KEYS = frozenset(
    {
        "path",
        "device",
        "inode",
        "mode",
        "link_count",
        "size_bytes",
        "mtime_ns",
        "ctime_ns",
        "sha256",
        "stable_regular_file_verified",
    }
)
_PREDECESSOR_RECEIPT_BINDING_KEYS = frozenset(
    {
        "schema_version",
        "state",
        "current_training_variant",
        "current_variant_position",
        "expected_predecessor_training_variant",
        "expected_predecessor_variant_position",
        "matched_panel_spec_sha256",
        "common_training_contract_sha256",
        "receipt_artifact",
        "predecessor_launch_manifest_artifact",
        "predecessor_training_summary_artifact",
        "predecessor_run_name",
        "chronology",
        "validated_before_gpu_probe",
    }
)
_PREDECESSOR_CHRONOLOGY_KEYS = frozenset(
    {
        "predecessor_launch_manifest_created_at_utc",
        "predecessor_training_summary_completed_at_utc",
        "predecessor_exit_receipt_recorded_at_utc",
        "strictly_ordered_timestamps_verified",
    }
)
_PILOT_EXIT_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "overall_status",
        "recorded_at_utc",
        "process_exit_status",
        "expected_contract",
        "pipeline",
        "source_at_receipt",
        "launch_manifest",
        "predecessor_receipt_binding",
        "training_job_lock",
        "training_summary",
        "runtime_config",
        "final_checkpoint",
        "completion_requirements",
    }
)
_PILOT_EXIT_COMPLETION_KEYS = frozenset(
    {
        "training_exit_zero",
        "tee_exit_zero",
        "training_summary_valid_and_launch_bound",
        "launch_manifest_matches_summary_runtime_and_launch",
        "predecessor_receipt_binding_unchanged_and_valid",
        "training_job_lock_valid_before_receipt_publication",
        "runtime_config_matches_summary_and_launch",
        "final_checkpoint_matches_training_summary",
        "clean_pushed_source_still_matches_launch",
        "all_must_hold",
    }
)
_PILOT_LAUNCH_MANIFEST_KEYS = frozenset(
    {
        "launch_manifest_schema_version",
        "created_at",
        "purpose",
        "gpu_selection_schema_version",
        "git_sha",
        "source_revision_before_final_gpu_probe",
        "run_name",
        "training_variant",
        "hydra_config_name",
        "udlm_prior_variant",
        "udlm_comparison_role",
        "matched_panel_spec",
        "matched_panel_spec_sha256",
        "matched_panel_variant_position",
        "predecessor_receipt_binding",
        "single_training_job_lock",
        "tmux_session",
        "user_requested_gpu_count",
        "gpu_selection_method",
        "gpu_inventory_scope",
        "inventory_snapshot_completed_at_utc",
        "gpu_inventory_at_selection",
        "initially_selected_gpu_states",
        "logical_cuda_devices",
        "physical_gpu_indices",
        "cuda_visible_device_uuids",
        "final_uuid_probes_completed_at_utc",
        "gpu_states_at_final_uuid_probe",
        "gpu_safety_policy",
        "training_argv",
        "training_argv_sha256",
        "resolved_training_config",
        "resolved_training_config_sha256",
        "runtime_config_path",
        "training_summary_path",
        "training_summary_schema_version",
        "pilot_exit_status_path",
        "pilot_exit_status_schema_version",
        "expected_final_checkpoint_path",
        "launch_manifest_path",
        "launch_manifest_raw_sha256_transport",
        "completion_contract",
        "log_path",
        "log_reserved_exclusively_before_manifest",
        "checkpoint",
        "checkpoint_sha256",
        "seed",
        "max_steps",
        "global_batch_size",
        "micro_batch_size_per_process",
        "accumulate_grad_batches",
        "effective_global_batch_size",
        "exclude_special_tokens",
        "dry_run",
    }
)
_PILOT_TRAINING_SUMMARY_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "completed_at_utc",
        "source_revision",
        "source",
        "resolved_training_config_sha256",
        "training_argv_sha256",
        "launch_manifest",
        "runtime_config",
        "completion_contract",
        "observed_training_state",
        "training_accounting",
        "training_health",
        "conditioning_gradient_audit",
        "screen_initialization_state_audit",
        "final_checkpoint",
        "tensor_finiteness",
        "startup",
    }
)
_PILOT_RUNTIME_CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "source_revision",
        "source",
        "training_argv",
        "observed_training_argv",
        "training_argv_sha256",
        "resolved_training_config",
        "resolved_training_config_sha256",
        "launch_manifest",
        "completion_contract",
        "python_environment",
    }
)
_PILOT_TRAINING_ACCOUNTING_KEYS = frozenset(
    {
        "training_seed",
        "optimizer_updates",
        "world_size",
        "micro_batch_size_per_rank",
        "accumulate_grad_batches",
        "effective_global_examples_per_optimizer_step",
        "total_requested_example_exposures",
        "hosted_stream_rank_partition_policy",
        "trainable_parameter_counts",
    }
)
_HOSTED_STREAM_RANK_PARTITION_POLICY = (
    "huggingface_split_dataset_by_node_disjoint_rank_streams"
)


def _strict_json_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for key, value in pairs:
        if key in parsed:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        parsed[key] = value
    return parsed


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number: {value}")
    return parsed


def strict_json_loads(payload: bytes, *, label: str) -> object:
    """Decode UTF-8 JSON while rejecting duplicate keys and non-finite values."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not valid UTF-8") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_json_pairs,
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_float,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error


def _required_mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_keys(
    value: dict[str, object], required: set[str] | frozenset[str], *, label: str
) -> None:
    missing = set(required) - set(value)
    if missing:
        raise ValueError(f"{label} is missing required keys: {sorted(missing)}")


def _require_exact_keys(
    value: dict[str, object], expected: set[str] | frozenset[str], *, label: str
) -> None:
    observed = set(value)
    if observed != set(expected):
        raise ValueError(
            f"{label} keys differ; missing={sorted(set(expected) - observed)}, "
            f"extra={sorted(observed - set(expected))}"
        )


def _exact_integer(value: object, expected: int, *, label: str) -> None:
    if type(value) is not int or value != expected:
        raise ValueError(f"{label} must equal {expected}")


def _exact_string(value: object, expected: str, *, label: str) -> None:
    if not isinstance(value, str) or value != expected:
        raise ValueError(f"{label} must equal {expected!r}")


def _required_true(value: object, *, label: str) -> None:
    if value is not True:
        raise ValueError(f"{label} must be true")


def _positive_integer(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal digits")
    return value


def _utc_timestamp(value: object, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{label} is not ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{label} must carry an explicit UTC offset")
    return parsed


def _validate_finiteness_record(value: object, *, label: str) -> dict[str, object]:
    record = _required_mapping(value, label=label)
    _require_exact_keys(
        record,
        {"all_finite", "floating_tensor_count", "floating_element_count"},
        label=label,
    )
    _required_true(record.get("all_finite"), label=f"{label} all-finite flag")
    tensors = _positive_integer(
        record.get("floating_tensor_count"), label=f"{label} tensor count"
    )
    elements = _positive_integer(
        record.get("floating_element_count"), label=f"{label} element count"
    )
    if elements < tensors:
        raise ValueError(f"{label} has fewer elements than tensors")
    return record


def _expected_framework_nonfinite_sentinels(
    *, expected_steps: int
) -> dict[str, object]:
    callback_key = (
        "ModelCheckpoint{'monitor': None, 'mode': 'min', "
        f"'every_n_train_steps': {expected_steps}, 'every_n_epochs': 0, "
        "'train_time_interval': None}"
    )
    return {
        "all_expected_and_only_expected_verified": True,
        "nonfinite_tensor_count": 1,
        "nonfinite_element_count": 1,
        "records": [
            {
                "tensor_path_components": [
                    "checkpoint",
                    "callbacks",
                    callback_key,
                    "kth_value",
                ],
                "framework": "lightning",
                "framework_version": "2.5.1",
                "callback": "ModelCheckpoint",
                "field": "kth_value",
                "dtype": "float32",
                "shape": [],
                "value": "+inf",
                "meaning": "unranked_min_mode_checkpoint_sentinel",
                "excluded_from_non_sentinel_finiteness": True,
            }
        ],
    }


def _validate_framework_nonfinite_sentinels(
    value: object, *, expected_steps: int, label: str
) -> dict[str, object]:
    record = _required_mapping(value, label=label)
    expected = _expected_framework_nonfinite_sentinels(expected_steps=expected_steps)
    if canonical_json_sha256(record) != canonical_json_sha256(expected):
        raise ValueError(
            f"{label} does not match the exact Lightning sentinel contract"
        )
    return record


def _validate_auxiliary_checkpoint_records(
    semantic: Mapping[str, object],
    *,
    expected_steps: int,
    resolved_training_config: object,
    label: str,
) -> int:
    resolved = _required_mapping(
        resolved_training_config,
        label=f"{label} resolved training config",
    )
    config_sha256 = canonical_json_sha256(resolved)
    trainer_config = _required_mapping(
        resolved.get("trainer"),
        label=f"{label} resolved trainer config",
    )
    accumulation = _positive_integer(
        trainer_config.get("accumulate_grad_batches"),
        label=f"{label} resolved accumulation",
    )
    optim_config = _required_mapping(
        resolved.get("optim"),
        label=f"{label} resolved optimizer config",
    )
    scheduler_config = _required_mapping(
        optim_config.get("scheduler"),
        label=f"{label} resolved scheduler config",
    )
    warmup_updates = scheduler_config.get("warmup_updates")
    horizon_updates = scheduler_config.get("horizon_updates")
    if type(warmup_updates) is not int or warmup_updates < 0:
        raise ValueError(f"{label} resolved scheduler warmup is invalid")
    if horizon_updates is not None and (
        type(horizon_updates) is not int or horizon_updates < 0
    ):
        raise ValueError(f"{label} resolved scheduler horizon is invalid")
    schedule_check_count = (
        max(expected_steps, warmup_updates + 1, (horizon_updates or 0) + 1) + 1
    )
    python_floats = _required_mapping(
        semantic.get("checkpoint_python_floats"),
        label=f"{label} Python-float finiteness",
    )
    _require_exact_keys(
        python_floats,
        {"all_finite", "floating_scalar_count"},
        label=f"{label} Python-float finiteness",
    )
    _required_true(
        python_floats.get("all_finite"),
        label=f"{label} Python-float all-finite flag",
    )
    _positive_integer(
        python_floats.get("floating_scalar_count"),
        label=f"{label} Python-float count",
    )

    optimizer = _required_mapping(
        semantic.get("optimizer_live_state_match"),
        label=f"{label} optimizer live-state match",
    )
    _require_exact_keys(
        optimizer,
        {
            "exact_serialized_live_match",
            "optimizer_count",
            "optimizer_class",
            "parameter_group_count",
            "parameter_state_count",
            "exact_resolved_config_match",
        },
        label=f"{label} optimizer live-state match",
    )
    _required_true(
        optimizer.get("exact_serialized_live_match"),
        label=f"{label} optimizer exact-match flag",
    )
    _required_true(
        optimizer.get("exact_resolved_config_match"),
        label=f"{label} optimizer resolved-config flag",
    )
    _exact_integer(
        optimizer.get("optimizer_count"), 1, label=f"{label} optimizer count"
    )
    _exact_string(
        optimizer.get("optimizer_class"),
        "AdamW",
        label=f"{label} optimizer class",
    )
    _exact_integer(
        optimizer.get("parameter_group_count"),
        1,
        label=f"{label} optimizer parameter-group count",
    )
    parameter_state_count = _positive_integer(
        optimizer.get("parameter_state_count"),
        label=f"{label} optimizer parameter-state count",
    )

    scheduler = _required_mapping(
        semantic.get("scheduler_live_state_match"),
        label=f"{label} scheduler live-state match",
    )
    expected_scheduler = {
        "exact_serialized_live_match": True,
        "scheduler_count": 1,
        "scheduler_class": "LambdaLR",
        "interval": "step",
        "name": "lr",
        "last_epoch": expected_steps,
        "step_count": expected_steps + 1,
        "exact_model_spec_match": True,
        "exact_callable_schedule_match": True,
        "callable_schedule_index_checks": schedule_check_count,
    }
    if canonical_json_sha256(scheduler) != canonical_json_sha256(expected_scheduler):
        raise ValueError(f"{label} scheduler live-state match is invalid")

    sampler = _required_mapping(
        semantic.get("sampler_live_state_match"),
        label=f"{label} sampler live-state match",
    )
    expected_sampler = {
        "exact_hosted_stream_contract_match": True,
        "random_state_is_none": True,
        "live_state_dict_available": False,
        "sampler_class_module": "torch.utils.data.dataloader",
        "sampler_class_name": "_InfiniteConstantSampler",
    }
    if canonical_json_sha256(sampler) != canonical_json_sha256(expected_sampler):
        raise ValueError(f"{label} sampler live-state match is invalid")

    callback_key = (
        "ModelCheckpoint{'monitor': None, 'mode': 'min', "
        f"'every_n_train_steps': {expected_steps}, 'every_n_epochs': 0, "
        "'train_time_interval': None}"
    )
    callback = _required_mapping(
        semantic.get("model_checkpoint_live_state_match"),
        label=f"{label} ModelCheckpoint live-state match",
    )
    expected_callback = {
        "exact_serialized_live_match": True,
        "model_checkpoint_callback_count": 1,
        "state_key": callback_key,
        "configuration_matches_pilot_contract": True,
    }
    if canonical_json_sha256(callback) != canonical_json_sha256(expected_callback):
        raise ValueError(f"{label} ModelCheckpoint live-state match is invalid")

    hyperparameters = _required_mapping(
        semantic.get("checkpoint_hyperparameters_match"),
        label=f"{label} checkpoint hyperparameter match",
    )
    expected_hyperparameters = {
        "hparams_name": "kwargs",
        "exact_hyperparameter_keys": True,
        "exact_checkpoint_preflight_config_match": True,
        "exact_live_model_preflight_config_match": True,
        "exact_live_hparams_preflight_config_match": True,
        "exact_checkpoint_live_model_unresolved_config_match": True,
        "exact_checkpoint_live_hparams_unresolved_config_match": True,
        "resolved_config_sha256": config_sha256,
    }
    if canonical_json_sha256(hyperparameters) != canonical_json_sha256(
        expected_hyperparameters
    ):
        raise ValueError(f"{label} checkpoint hyperparameter match is invalid")

    configured_clip = trainer_config.get("gradient_clip_val")
    configured_precision = trainer_config.get("precision")
    precision_aliases = {
        "16": "16-mixed",
        "bf16": "bf16-mixed",
        "32": "32-true",
        "64": "64-true",
        16: "16-mixed",
        32: "32-true",
        64: "64-true",
    }
    live_precision = precision_aliases.get(configured_precision, configured_precision)
    configured_clip_algorithm = trainer_config.get("gradient_clip_algorithm")
    if configured_clip_algorithm is None:
        configured_clip_algorithm = "norm"
    if (
        isinstance(configured_clip, bool)
        or not isinstance(configured_clip, (int, float))
        or not math.isfinite(configured_clip)
        or configured_clip < 0.0
        or not isinstance(live_precision, str)
        or configured_clip_algorithm not in {"norm", "value"}
    ):
        raise ValueError(f"{label} resolved live-Trainer config is invalid")
    trainer_match = _required_mapping(
        semantic.get("trainer_live_configuration_match"),
        label=f"{label} live Trainer configuration match",
    )
    expected_trainer_match = {
        "exact_detect_anomaly_match": True,
        "detect_anomaly": True,
        "exact_gradient_clip_val_match": True,
        "gradient_clip_val": float(configured_clip),
        "exact_gradient_clip_algorithm_match": True,
        "gradient_clip_algorithm": configured_clip_algorithm,
        "exact_precision_match": True,
        "configured_precision": str(configured_precision),
        "live_precision": live_precision,
    }
    if canonical_json_sha256(trainer_match) != canonical_json_sha256(
        expected_trainer_match
    ):
        raise ValueError(f"{label} live Trainer configuration match is invalid")

    loop_state = _required_mapping(
        semantic.get("checkpoint_loop_state_match"),
        label=f"{label} checkpoint loop-state match",
    )
    expected_loop_state = {
        "exact_serialized_progress_match": True,
        "epoch": 0,
        "optimizer_steps": expected_steps,
        "accumulate_grad_batches": accumulation,
        "microbatches": expected_steps * accumulation,
    }
    if canonical_json_sha256(loop_state) != canonical_json_sha256(expected_loop_state):
        raise ValueError(f"{label} checkpoint loop-state match is invalid")
    return parameter_state_count


def _validate_gpu_state_record(value: object, *, label: str) -> dict[str, object]:
    record = _required_mapping(value, label=label)
    _require_exact_keys(
        record,
        {
            "physical_index",
            "uuid",
            "name",
            "memory_used_mib",
            "memory_total_mib",
            "utilization_percent",
            "compute_mode",
            "compute_processes",
        },
        label=label,
    )
    for key in (
        "physical_index",
        "memory_used_mib",
        "memory_total_mib",
        "utilization_percent",
    ):
        observed = record.get(key)
        if type(observed) is not int or observed < 0:
            raise ValueError(f"{label} {key} must be a nonnegative integer")
    if record["memory_used_mib"] > record["memory_total_mib"]:
        raise ValueError(f"{label} used memory exceeds total memory")
    if record["utilization_percent"] > 100:
        raise ValueError(f"{label} utilization exceeds 100 percent")
    uuid = record.get("uuid")
    if not isinstance(uuid, str) or not uuid.startswith("GPU-"):
        raise ValueError(f"{label} UUID is invalid")
    for key in ("name", "compute_mode"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError(f"{label} {key} is invalid")
    processes = record.get("compute_processes")
    if not isinstance(processes, list):
        raise ValueError(f"{label} compute processes must be an array")
    for process in processes:
        process = _required_mapping(process, label=f"{label} compute process")
        _require_exact_keys(
            process,
            {"pid", "process_name", "used_memory_mib"},
            label=f"{label} compute process",
        )
        _positive_integer(process.get("pid"), label=f"{label} process PID")
        if (
            not isinstance(process.get("process_name"), str)
            or not process["process_name"]
        ):
            raise ValueError(f"{label} process name is invalid")
        used = process.get("used_memory_mib")
        if used is not None and (type(used) is not int or used < 0):
            raise ValueError(f"{label} process memory is invalid")
    return record


def _repository_artifact_path(
    value: Path, *, suffix: str, label: str, basename: str | None = None
) -> Path:
    """Normalize one existing artifact without accepting an outside symlink."""

    repository_root = REPOSITORY_ROOT.resolve(strict=True)
    absolute = Path(os.path.abspath(os.fspath(value)))
    parent = absolute.parent.resolve(strict=True)
    normalized = parent / absolute.name
    if (
        normalized == repository_root
        or repository_root not in normalized.parents
        or normalized.suffix != suffix
        or (basename is not None and normalized.name != basename)
    ):
        raise ValueError(f"{label} must be the reviewed in-repository {suffix} file")
    return normalized


def validate_pilot_output_parents(
    *, run_dir: Path, log_path: Path, create_missing: bool
) -> None:
    """Reject symlinked output ancestors and optionally create canonical parents."""

    if type(create_missing) is not bool:
        raise ValueError("create_missing must be boolean")
    lexical_root = Path(os.path.abspath(os.fspath(REPOSITORY_ROOT)))
    resolved_root = lexical_root.resolve(strict=True)
    expected_run_parent = lexical_root / "output" / "udlm"
    expected_log_parent = lexical_root / "output" / "logs"
    absolute_run_dir = Path(os.path.abspath(os.fspath(run_dir)))
    absolute_log_path = Path(os.path.abspath(os.fspath(log_path)))
    if (
        absolute_run_dir.parent != expected_run_parent
        or not RUN_NAME_PATTERN.fullmatch(absolute_run_dir.name)
        or absolute_log_path != expected_log_parent / f"{absolute_run_dir.name}.log"
    ):
        raise ValueError("pilot outputs must use the canonical repository layout")

    for relative_parent in (Path("output/udlm"), Path("output/logs")):
        current = lexical_root
        for index, part in enumerate(relative_parent.parts, start=1):
            current /= part
            try:
                state = current.stat(follow_symlinks=False)
            except FileNotFoundError:
                if not create_missing:
                    break
                try:
                    current.mkdir(mode=0o755)
                except FileExistsError:
                    pass
                state = current.stat(follow_symlinks=False)
            if not stat.S_ISDIR(state.st_mode) or stat.S_ISLNK(state.st_mode):
                raise ValueError(
                    f"pilot output ancestor must be a real directory: {current}"
                )
            expected_resolved = resolved_root.joinpath(*relative_parent.parts[:index])
            if current.resolve(strict=True) != expected_resolved:
                raise ValueError(
                    f"pilot output ancestor escapes its canonical path: {current}"
                )


def stable_repository_artifact_snapshot(
    path: Path,
    *,
    suffix: str,
    label: str,
    capture_bytes: bool = True,
) -> tuple[dict[str, object], bytes]:
    """Hash one stable artifact, optionally retaining its exact bytes."""

    if type(capture_bytes) is not bool:
        raise ValueError("capture_bytes must be boolean")
    normalized = _repository_artifact_path(path, suffix=suffix, label=label)
    before_path = normalized.stat(follow_symlinks=False)
    if not stat.S_ISREG(before_path.st_mode) or before_path.st_nlink != 1:
        raise ValueError(f"{label} must be a single-link regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(normalized, flags)
    digest = hashlib.sha256()
    payload = bytearray() if capture_bytes else None
    try:
        before_descriptor = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before_descriptor.st_mode)
            or before_descriptor.st_nlink != 1
            or _stable_stat_identity(before_descriptor)
            != _stable_stat_identity(before_path)
        ):
            raise ValueError(f"{label} changed before it was opened")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if payload is not None:
                payload.extend(chunk)
        after_descriptor = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = normalized.stat(follow_symlinks=False)
    identity = _stable_stat_identity(before_path)
    if any(
        _stable_stat_identity(observed) != identity
        for observed in (before_descriptor, after_descriptor, after_path)
    ):
        raise ValueError(f"{label} changed while its exact bytes were read")
    if after_path.st_size == 0:
        raise ValueError(f"{label} must not be empty")
    return (
        {
            "path": str(normalized),
            "device": int(after_path.st_dev),
            "inode": int(after_path.st_ino),
            "mode": int(after_path.st_mode),
            "link_count": int(after_path.st_nlink),
            "size_bytes": int(after_path.st_size),
            "mtime_ns": int(after_path.st_mtime_ns),
            "ctime_ns": int(after_path.st_ctime_ns),
            "sha256": digest.hexdigest(),
            "stable_regular_file_verified": True,
        },
        bytes(payload) if payload is not None else b"",
    )


def _pilot_empirical_uniform_mix_audit_binding() -> dict[str, object]:
    return {
        "relative_path": PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_PATH,
        "sha256": PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_SHA256,
        "source_revision": PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_SOURCE_REVISION,
        "scope": PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_SCOPE,
    }


def verify_pilot_empirical_uniform_mix_audit() -> dict[str, object]:
    """Verify the immutable training-only floor audit from its live exact bytes."""

    audit_path = REPOSITORY_ROOT / PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_PATH
    snapshot, payload = stable_repository_artifact_snapshot(
        audit_path,
        suffix=".json",
        label="pilot empirical-uniform-mix audit",
    )
    _exact_string(
        snapshot.get("sha256"),
        PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_SHA256,
        label="pilot empirical-uniform-mix audit raw SHA-256",
    )
    audit = _required_mapping(
        strict_json_loads(payload, label="pilot empirical-uniform-mix audit"),
        label="pilot empirical-uniform-mix audit",
    )
    _exact_string(
        canonical_json_sha256(audit),
        PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_CANONICAL_SHA256,
        label="pilot empirical-uniform-mix audit canonical SHA-256",
    )
    _exact_integer(
        audit.get("schema_version"),
        1,
        label="pilot empirical-uniform-mix audit schema",
    )
    git = _required_mapping(
        audit.get("git"), label="pilot empirical-uniform-mix audit git provenance"
    )
    for key in ("commit", "upstream"):
        _exact_string(
            git.get(key),
            PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_SOURCE_REVISION,
            label=f"pilot empirical-uniform-mix audit git {key}",
        )
    if git.get("dirty") is not False:
        raise ValueError("pilot empirical-uniform-mix audit source must be clean")

    recommendation = _required_mapping(
        audit.get("recommendation"),
        label="pilot empirical-uniform-mix audit recommendation",
    )
    _exact_string(
        recommendation.get("status"),
        "training_only_retrospective_engineering_recommendation",
        label="pilot empirical-uniform-mix audit recommendation status",
    )
    for key in (
        "candidate_uniform_mixture_weight",
        "recommended_uniform_mixture_weight",
    ):
        value = recommendation.get(key)
        if type(value) not in (int, float) or value != PILOT_EMPIRICAL_UNIFORM_MIX:
            raise ValueError(
                f"pilot empirical-uniform-mix audit {key} must equal "
                f"{PILOT_EMPIRICAL_UNIFORM_MIX}"
            )
    for key in (
        "candidate_nll_strictly_better_than_current_on_both_blocks",
        "both_block_optima_within_0_0001_to_0_0003",
    ):
        _required_true(
            recommendation.get(key),
            label=f"pilot empirical-uniform-mix audit {key}",
        )

    data_use = _required_mapping(
        audit.get("data_use"), label="pilot empirical-uniform-mix audit data use"
    )
    _exact_string(
        data_use.get("split"),
        "training",
        label="pilot empirical-uniform-mix audit data split",
    )
    for key in (
        "final_generation_seeds_or_metrics_used",
        "formal_preregistration_before_data_access",
    ):
        if data_use.get(key) is not False:
            raise ValueError(
                "pilot empirical-uniform-mix audit does not have the exact "
                f"{PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_SCOPE} scope"
            )
    return _pilot_empirical_uniform_mix_audit_binding()


def _snapshot_payload_matching_claim(
    claim: object, *, suffix: str, label: str
) -> tuple[dict[str, object], bytes]:
    snapshot_claim = _required_mapping(claim, label=f"{label} claim")
    _require_exact_keys(
        snapshot_claim,
        _STABLE_ARTIFACT_SNAPSHOT_KEYS,
        label=f"{label} claim",
    )
    _required_true(
        snapshot_claim.get("stable_regular_file_verified"),
        label=f"{label} stable regular-file flag",
    )
    _sha256(snapshot_claim.get("sha256"), label=f"{label} SHA-256")
    path_value = snapshot_claim.get("path")
    if not isinstance(path_value, str):
        raise ValueError(f"{label} path must be a string")
    current, payload = stable_repository_artifact_snapshot(
        Path(path_value), suffix=suffix, label=label
    )
    if current != snapshot_claim:
        raise ValueError(f"{label} no longer matches its exact bound snapshot")
    return current, payload


def _require_extended_snapshot_matches(
    snapshot: dict[str, object],
    claim: object,
    *,
    extra_keys: set[str] | frozenset[str],
    label: str,
) -> dict[str, object]:
    extended = _required_mapping(claim, label=label)
    _require_exact_keys(
        extended,
        _STABLE_ARTIFACT_SNAPSHOT_KEYS | frozenset(extra_keys),
        label=label,
    )
    observed_snapshot = {
        key: extended.get(key) for key in _STABLE_ARTIFACT_SNAPSHOT_KEYS
    }
    if observed_snapshot != snapshot:
        raise ValueError(f"{label} does not match the live exact artifact snapshot")
    return extended


def _validate_stable_snapshot_claim(
    value: object, *, expected_path: Path, label: str
) -> dict[str, object]:
    claim = _required_mapping(value, label=label)
    _require_exact_keys(claim, _STABLE_ARTIFACT_SNAPSHOT_KEYS, label=label)
    _exact_string(claim.get("path"), str(expected_path), label=f"{label} artifact path")
    _required_true(
        claim.get("stable_regular_file_verified"),
        label=f"{label} regular-file verification",
    )
    for key, minimum in (
        ("device", 0),
        ("inode", 1),
        ("mode", 1),
        ("link_count", 1),
        ("size_bytes", 1),
        ("mtime_ns", 0),
        ("ctime_ns", 0),
    ):
        observed = claim.get(key)
        if type(observed) is not int or observed < minimum:
            raise ValueError(f"{label} {key} is invalid")
    if not stat.S_ISREG(claim["mode"]):
        raise ValueError(f"{label} mode is not a regular file")
    _sha256(claim.get("sha256"), label=f"{label} SHA-256")
    return claim


def _matched_panel_contract(
    matched_panel_spec: dict[str, object], matched_panel_spec_sha256: str
) -> tuple[dict[str, object], str]:
    _sha256(matched_panel_spec_sha256, label="matched-panel specification digest")
    if canonical_json_sha256(matched_panel_spec) != matched_panel_spec_sha256:
        raise ValueError(
            "matched-panel specification content disagrees with its digest"
        )
    _exact_integer(
        matched_panel_spec.get("schema_version"),
        MATCHED_PANEL_SCHEMA_VERSION,
        label="matched-panel schema version",
    )
    execution = _required_mapping(
        matched_panel_spec.get("execution"), label="matched-panel execution contract"
    )
    if execution.get("registered_variant_order") != list(MATCHED_PANEL_VARIANT_ORDER):
        raise ValueError("matched-panel variant order is not canonical")
    common = _required_mapping(
        matched_panel_spec.get("common_training_contract"),
        label="matched-panel common training contract",
    )
    return common, canonical_json_sha256(common)


def _validate_predecessor_binding_shape(
    value: object,
    *,
    current_training_variant: str,
    matched_panel_spec_sha256: str,
    common_training_contract_sha256: str,
    revalidate_bound_artifacts: bool,
) -> dict[str, object]:
    binding = _required_mapping(value, label="predecessor receipt binding")
    _require_exact_keys(
        binding,
        _PREDECESSOR_RECEIPT_BINDING_KEYS,
        label="predecessor receipt binding",
    )
    position = MATCHED_PANEL_VARIANT_ORDER.index(
        validate_training_variant(current_training_variant)
    )
    _exact_integer(
        binding.get("schema_version"),
        PREDECESSOR_RECEIPT_BINDING_SCHEMA_VERSION,
        label="predecessor receipt-binding schema",
    )
    _exact_string(
        binding.get("current_training_variant"),
        current_training_variant,
        label="predecessor binding current variant",
    )
    _exact_integer(
        binding.get("current_variant_position"),
        position,
        label="predecessor binding current position",
    )
    _exact_string(
        binding.get("matched_panel_spec_sha256"),
        matched_panel_spec_sha256,
        label="predecessor binding matched-panel digest",
    )
    _exact_string(
        binding.get("common_training_contract_sha256"),
        common_training_contract_sha256,
        label="predecessor binding common-contract digest",
    )
    _required_true(
        binding.get("validated_before_gpu_probe"),
        label="predecessor binding pre-GPU validation flag",
    )
    if position == 0:
        _exact_string(
            binding.get("state"),
            "explicit_genesis_no_predecessor",
            label="genesis predecessor state",
        )
        for key in (
            "expected_predecessor_training_variant",
            "expected_predecessor_variant_position",
            "receipt_artifact",
            "predecessor_launch_manifest_artifact",
            "predecessor_training_summary_artifact",
            "predecessor_run_name",
            "chronology",
        ):
            if binding.get(key) is not None:
                raise ValueError(f"genesis predecessor binding {key} must be null")
        return binding

    expected_variant = MATCHED_PANEL_VARIANT_ORDER[position - 1]
    _exact_string(
        binding.get("state"),
        "validated_successful_predecessor",
        label="successor predecessor state",
    )
    _exact_string(
        binding.get("expected_predecessor_training_variant"),
        expected_variant,
        label="expected predecessor variant",
    )
    _exact_integer(
        binding.get("expected_predecessor_variant_position"),
        position - 1,
        label="expected predecessor position",
    )
    run_name = binding.get("predecessor_run_name")
    if not isinstance(run_name, str) or not RUN_NAME_PATTERN.fullmatch(run_name):
        raise ValueError("predecessor run name is invalid")
    for key in (
        "receipt_artifact",
        "predecessor_launch_manifest_artifact",
        "predecessor_training_summary_artifact",
    ):
        claim = _required_mapping(binding.get(key), label=key.replace("_", " "))
        _require_exact_keys(
            claim,
            _STABLE_ARTIFACT_SNAPSHOT_KEYS,
            label=key.replace("_", " "),
        )
    chronology = _required_mapping(
        binding.get("chronology"), label="predecessor chronology"
    )
    _require_exact_keys(
        chronology,
        _PREDECESSOR_CHRONOLOGY_KEYS,
        label="predecessor chronology",
    )
    manifest_time = _utc_timestamp(
        chronology.get("predecessor_launch_manifest_created_at_utc"),
        label="predecessor manifest creation timestamp",
    )
    summary_time = _utc_timestamp(
        chronology.get("predecessor_training_summary_completed_at_utc"),
        label="predecessor training completion timestamp",
    )
    receipt_time = _utc_timestamp(
        chronology.get("predecessor_exit_receipt_recorded_at_utc"),
        label="predecessor exit receipt timestamp",
    )
    _required_true(
        chronology.get("strictly_ordered_timestamps_verified"),
        label="predecessor chronology verification flag",
    )
    if not manifest_time < summary_time < receipt_time:
        raise ValueError(
            "predecessor manifest, training completion, and receipt timestamps "
            "must be strictly ordered"
        )
    if revalidate_bound_artifacts:
        _snapshot_payload_matching_claim(
            binding["receipt_artifact"],
            suffix=".json",
            label="transitively bound predecessor receipt",
        )
        _snapshot_payload_matching_claim(
            binding["predecessor_launch_manifest_artifact"],
            suffix=".json",
            label="transitively bound predecessor launch manifest",
        )
        _snapshot_payload_matching_claim(
            binding["predecessor_training_summary_artifact"],
            suffix=".json",
            label="transitively bound predecessor training summary",
        )
    return binding


def _require_predecessor_receipt_before_manifest(
    binding: dict[str, object], *, current_manifest_created_at_utc: str
) -> None:
    """Require a successor manifest to postdate its predecessor receipt."""

    manifest_time = _utc_timestamp(
        current_manifest_created_at_utc,
        label="current manifest creation timestamp",
    )
    if binding.get("state") == "explicit_genesis_no_predecessor":
        return
    chronology = _required_mapping(
        binding.get("chronology"), label="predecessor chronology"
    )
    receipt_time = _utc_timestamp(
        chronology.get("predecessor_exit_receipt_recorded_at_utc"),
        label="predecessor exit receipt timestamp",
    )
    if not receipt_time < manifest_time:
        raise ValueError(
            "predecessor exit receipt recorded_at must be strictly earlier than "
            "the exact current manifest created_at"
        )


def _require_predecessor_receipt_before_lock(
    binding: dict[str, object], *, current_lock_acquired_at_utc: str
) -> None:
    """Require every successor receipt to predate this launch's GPU lock."""

    lock_time = _utc_timestamp(
        current_lock_acquired_at_utc,
        label="current training-job lock acquisition timestamp",
    )
    if binding.get("state") == "explicit_genesis_no_predecessor":
        return
    chronology = _required_mapping(
        binding.get("chronology"), label="predecessor chronology"
    )
    receipt_time = _utc_timestamp(
        chronology.get("predecessor_exit_receipt_recorded_at_utc"),
        label="predecessor exit receipt timestamp",
    )
    if not receipt_time < lock_time:
        raise ValueError(
            "predecessor exit receipt recorded_at must be strictly earlier than "
            "the current training-job lock acquisition"
        )


def _explicit_genesis_binding(
    *,
    training_variant: str,
    matched_panel_spec_sha256: str,
    common_training_contract_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": PREDECESSOR_RECEIPT_BINDING_SCHEMA_VERSION,
        "state": "explicit_genesis_no_predecessor",
        "current_training_variant": training_variant,
        "current_variant_position": 0,
        "expected_predecessor_training_variant": None,
        "expected_predecessor_variant_position": None,
        "matched_panel_spec_sha256": matched_panel_spec_sha256,
        "common_training_contract_sha256": common_training_contract_sha256,
        "receipt_artifact": None,
        "predecessor_launch_manifest_artifact": None,
        "predecessor_training_summary_artifact": None,
        "predecessor_run_name": None,
        "chronology": None,
        "validated_before_gpu_probe": True,
    }


def build_predecessor_receipt_binding(
    *,
    training_variant: str,
    explicit_genesis: bool,
    predecessor_receipt_path: Path | None,
    matched_panel_spec: dict[str, object],
    matched_panel_spec_sha256: str,
) -> dict[str, object]:
    """Validate and normalize the exact predecessor required for this R/S/E arm."""

    training_variant = validate_training_variant(training_variant)
    position = MATCHED_PANEL_VARIANT_ORDER.index(training_variant)
    common, common_sha256 = _matched_panel_contract(
        matched_panel_spec, matched_panel_spec_sha256
    )
    if type(explicit_genesis) is not bool:
        raise ValueError("explicit genesis declaration must be boolean")
    if position == 0:
        if not explicit_genesis or predecessor_receipt_path is not None:
            raise ValueError(
                "R/udlm requires --genesis and forbids a predecessor receipt"
            )
        binding = _explicit_genesis_binding(
            training_variant=training_variant,
            matched_panel_spec_sha256=matched_panel_spec_sha256,
            common_training_contract_sha256=common_sha256,
        )
        return _validate_predecessor_binding_shape(
            binding,
            current_training_variant=training_variant,
            matched_panel_spec_sha256=matched_panel_spec_sha256,
            common_training_contract_sha256=common_sha256,
            revalidate_bound_artifacts=False,
        )
    if explicit_genesis or predecessor_receipt_path is None:
        raise ValueError(
            f"{training_variant} requires the immediately preceding successful "
            "--predecessor-receipt and cannot declare genesis"
        )

    receipt_path = _repository_artifact_path(
        predecessor_receipt_path,
        suffix=".json",
        basename="pilot_exit_status.json",
        label="predecessor exit receipt",
    )
    expected_runs_root = REPOSITORY_ROOT.resolve(strict=True) / "output" / "udlm"
    if (
        receipt_path.parent.parent != expected_runs_root
        or not RUN_NAME_PATTERN.fullmatch(receipt_path.parent.name)
    ):
        raise ValueError(
            "predecessor exit receipt must belong to one direct output/udlm run"
        )
    receipt_snapshot, receipt_payload = stable_repository_artifact_snapshot(
        receipt_path,
        suffix=".json",
        label="predecessor exit receipt",
    )
    receipt = _required_mapping(
        strict_json_loads(receipt_payload, label="predecessor exit receipt"),
        label="predecessor exit receipt",
    )
    _require_exact_keys(
        receipt, _PILOT_EXIT_RECEIPT_KEYS, label="predecessor exit receipt"
    )
    _exact_integer(
        receipt.get("schema_version"),
        PILOT_EXIT_STATUS_SCHEMA_VERSION,
        label="predecessor exit receipt schema",
    )
    for key in ("status", "overall_status"):
        _exact_string(receipt.get(key), "completed", label=f"predecessor receipt {key}")
    _exact_integer(
        receipt.get("process_exit_status"),
        0,
        label="predecessor receipt process exit status",
    )
    receipt_time = _utc_timestamp(
        receipt.get("recorded_at_utc"), label="predecessor receipt timestamp"
    )

    pipeline = _required_mapping(
        receipt.get("pipeline"), label="predecessor receipt pipeline"
    )
    _require_exact_keys(
        pipeline,
        {"training", "tee", "pipefail_shell_exit_status"},
        label="predecessor receipt pipeline",
    )
    _exact_integer(
        pipeline.get("pipefail_shell_exit_status"),
        0,
        label="predecessor pipeline exit status",
    )
    for component_name in ("training", "tee"):
        component = _required_mapping(
            pipeline.get(component_name),
            label=f"predecessor {component_name} pipeline component",
        )
        _require_exact_keys(
            component,
            {
                "shell_exit_status",
                "succeeded",
                "possible_termination_signal",
                "shell_status_is_signal_compatible",
                "signal_provenance",
            },
            label=f"predecessor {component_name} pipeline component",
        )
        _exact_integer(
            component.get("shell_exit_status"),
            0,
            label=f"predecessor {component_name} shell exit status",
        )
        _required_true(
            component.get("succeeded"),
            label=f"predecessor {component_name} success flag",
        )
        if (
            component.get("possible_termination_signal") is not None
            or component.get("shell_status_is_signal_compatible") is not False
            or component.get("signal_provenance") is not None
        ):
            raise ValueError(
                f"predecessor {component_name} zero exit has signal evidence"
            )

    completion = _required_mapping(
        receipt.get("completion_requirements"),
        label="predecessor completion requirements",
    )
    _require_exact_keys(
        completion,
        _PILOT_EXIT_COMPLETION_KEYS,
        label="predecessor completion requirements",
    )
    _required_true(
        completion.get("predecessor_receipt_binding_unchanged_and_valid"),
        label="predecessor receipt-binding completion requirement",
    )
    _required_true(
        completion.get("all_must_hold"),
        label="predecessor all-must-hold completion requirement",
    )
    if any(value is not True for value in completion.values()):
        raise ValueError("every predecessor completion requirement must be true")

    source_revision = common.get("source_revision")
    if not isinstance(source_revision, str) or not re.fullmatch(
        r"[0-9a-f]{40}", source_revision
    ):
        raise ValueError("matched-panel source revision is invalid")
    source = _required_mapping(
        receipt.get("source_at_receipt"), label="predecessor receipt source"
    )
    _require_exact_keys(
        source,
        {
            "verified",
            "expected_revision",
            "head",
            "upstream",
            "output_directory_excluded_from_cleanliness_check",
        },
        label="predecessor receipt source",
    )
    _required_true(source.get("verified"), label="predecessor source verification")
    _required_true(
        source.get("output_directory_excluded_from_cleanliness_check"),
        label="predecessor source output-directory exclusion",
    )
    for key in ("expected_revision", "head", "upstream"):
        _exact_string(
            source.get(key), source_revision, label=f"predecessor source {key}"
        )

    expected = _required_mapping(
        receipt.get("expected_contract"), label="predecessor expected contract"
    )
    _require_exact_keys(
        expected,
        {
            "training_summary_schema_version",
            "source_revision",
            "resolved_training_config_sha256",
            "training_argv_sha256",
            "launch_manifest_path",
            "launch_manifest_sha256",
            "selected_gpu_uuids",
            "training_job_lock_path",
            "training_job_lock_sha256",
            "max_steps",
            "world_size",
            "training_summary_path",
            "final_checkpoint_path",
            "initialization_checkpoint_sha256",
        },
        label="predecessor expected contract",
    )
    _exact_integer(
        expected.get("training_summary_schema_version"),
        TRAINING_SUMMARY_SCHEMA_VERSION,
        label="predecessor expected summary schema",
    )
    _exact_string(
        expected.get("source_revision"),
        source_revision,
        label="predecessor expected source revision",
    )
    _exact_integer(
        expected.get("max_steps"),
        common["max_steps"],
        label="predecessor expected optimizer steps",
    )
    _exact_integer(
        expected.get("world_size"),
        common["requested_gpu_count"],
        label="predecessor expected world size",
    )
    if expected.get("initialization_checkpoint_sha256") != common.get(
        "initialization_checkpoint_sha256"
    ):
        raise ValueError("predecessor initialization checkpoint digest is unmatched")

    expected_manifest_path_value = expected.get("launch_manifest_path")
    if not isinstance(expected_manifest_path_value, str):
        raise ValueError("predecessor expected launch-manifest path must be a string")
    manifest_path = _repository_artifact_path(
        Path(expected_manifest_path_value),
        suffix=".json",
        basename="launch_manifest.json",
        label="predecessor launch manifest",
    )
    if manifest_path.parent != receipt_path.parent:
        raise ValueError(
            "predecessor receipt and launch manifest must belong to the same run"
        )
    manifest_snapshot, manifest_payload = stable_repository_artifact_snapshot(
        manifest_path,
        suffix=".json",
        label="predecessor launch manifest",
    )
    expected_manifest_sha256 = _sha256(
        expected.get("launch_manifest_sha256"),
        label="predecessor expected launch-manifest digest",
    )
    _exact_string(
        manifest_snapshot["sha256"],
        expected_manifest_sha256,
        label="predecessor launch-manifest raw digest",
    )
    manifest_evidence = _required_mapping(
        receipt.get("launch_manifest"), label="predecessor manifest evidence"
    )
    _require_exact_keys(
        manifest_evidence,
        {
            "path",
            "present",
            "matches_expected_raw_sha256",
            "selected_gpu_uuids_match_expected",
            "matches_training_summary_snapshot",
            "matches_runtime_config_snapshot",
            "valid_and_launch_bound",
            "expected_selected_gpu_uuids",
            "observed_selected_gpu_uuids",
            "artifact",
            "validation_error",
        },
        label="predecessor manifest evidence",
    )
    for key in (
        "matches_expected_raw_sha256",
        "selected_gpu_uuids_match_expected",
        "matches_training_summary_snapshot",
        "matches_runtime_config_snapshot",
        "valid_and_launch_bound",
    ):
        _required_true(
            manifest_evidence.get(key),
            label=f"predecessor manifest evidence {key}",
        )
    if manifest_evidence.get("validation_error") is not None:
        raise ValueError("predecessor manifest evidence contains a validation error")
    _exact_string(
        manifest_evidence.get("path"),
        str(manifest_path),
        label="predecessor manifest evidence path",
    )
    _required_true(
        manifest_evidence.get("present"),
        label="predecessor manifest evidence presence",
    )
    if manifest_evidence.get("artifact") != manifest_snapshot:
        raise ValueError(
            "predecessor launch manifest no longer matches the receipt snapshot"
        )

    predecessor_manifest = _required_mapping(
        strict_json_loads(manifest_payload, label="predecessor launch manifest"),
        label="predecessor launch manifest",
    )
    _require_exact_keys(
        predecessor_manifest,
        _PILOT_LAUNCH_MANIFEST_KEYS,
        label="predecessor launch manifest",
    )
    expected_variant = MATCHED_PANEL_VARIANT_ORDER[position - 1]
    expected_definition = TRAINING_VARIANTS[expected_variant]
    _exact_integer(
        predecessor_manifest.get("launch_manifest_schema_version"),
        LAUNCH_MANIFEST_SCHEMA_VERSION,
        label="predecessor launch-manifest schema",
    )
    _exact_string(
        predecessor_manifest.get("purpose"),
        "bounded UDLM training pilot",
        label="predecessor launch-manifest purpose",
    )
    for key in ("git_sha", "source_revision_before_final_gpu_probe"):
        _exact_string(
            predecessor_manifest.get(key),
            source_revision,
            label=f"predecessor manifest {key}",
        )
    _exact_string(
        predecessor_manifest.get("training_variant"),
        expected_variant,
        label="predecessor training variant",
    )
    _exact_integer(
        predecessor_manifest.get("matched_panel_variant_position"),
        position - 1,
        label="predecessor variant position",
    )
    for manifest_key, definition_key in (
        ("hydra_config_name", "config_name"),
        ("udlm_prior_variant", "prior_variant"),
        ("udlm_comparison_role", "comparison_role"),
    ):
        _exact_string(
            predecessor_manifest.get(manifest_key),
            str(expected_definition[definition_key]),
            label=f"predecessor manifest {manifest_key}",
        )
    if predecessor_manifest.get("matched_panel_spec") != matched_panel_spec:
        raise ValueError(
            "predecessor launch uses a different matched-panel specification"
        )
    _exact_string(
        predecessor_manifest.get("matched_panel_spec_sha256"),
        matched_panel_spec_sha256,
        label="predecessor matched-panel digest",
    )
    _exact_string(
        predecessor_manifest.get("pilot_exit_status_path"),
        str(receipt_path),
        label="predecessor receipt path binding",
    )
    _exact_integer(
        predecessor_manifest.get("pilot_exit_status_schema_version"),
        PILOT_EXIT_STATUS_SCHEMA_VERSION,
        label="predecessor receipt schema binding",
    )
    for manifest_key, expected_value in (
        ("launch_manifest_path", str(manifest_path)),
        ("training_summary_path", expected.get("training_summary_path")),
        ("expected_final_checkpoint_path", expected.get("final_checkpoint_path")),
    ):
        _exact_string(
            predecessor_manifest.get(manifest_key),
            expected_value,
            label=f"predecessor manifest {manifest_key}",
        )
    _exact_integer(
        predecessor_manifest.get("training_summary_schema_version"),
        TRAINING_SUMMARY_SCHEMA_VERSION,
        label="predecessor manifest summary schema",
    )
    if predecessor_manifest.get("dry_run") is not False:
        raise ValueError("predecessor launch manifest must describe a real launch")
    direct_common_fields = {
        "checkpoint": "initialization_checkpoint_path",
        "checkpoint_sha256": "initialization_checkpoint_sha256",
        "user_requested_gpu_count": "requested_gpu_count",
        "max_steps": "max_steps",
        "seed": "seed",
        "global_batch_size": "global_batch_size",
        "micro_batch_size_per_process": "micro_batch_size_per_process",
        "accumulate_grad_batches": "accumulate_grad_batches",
        "effective_global_batch_size": "effective_global_batch_size",
        "exclude_special_tokens": "exclude_special_tokens",
    }
    for manifest_key, common_key in direct_common_fields.items():
        if predecessor_manifest.get(manifest_key) != common.get(common_key):
            raise ValueError(
                f"predecessor manifest {manifest_key} disagrees with the matched "
                "common contract"
            )
    resolved_config = _required_mapping(
        predecessor_manifest.get("resolved_training_config"),
        label="predecessor resolved training config",
    )
    resolved_config_sha256 = _sha256(
        predecessor_manifest.get("resolved_training_config_sha256"),
        label="predecessor resolved-config digest",
    )
    _exact_string(
        canonical_json_sha256(resolved_config),
        resolved_config_sha256,
        label="predecessor resolved-config content digest",
    )
    _exact_string(
        matched_panel_config_sha256(resolved_config),
        str(common["common_resolved_config_sha256"]),
        label="predecessor common resolved-config digest",
    )
    resolved_training = _required_mapping(
        resolved_config.get("training"), label="predecessor resolved training"
    )
    resolved_udlm = _required_mapping(
        resolved_training.get("udlm"), label="predecessor resolved UDLM config"
    )
    _exact_string(
        resolved_udlm.get("prior_variant"),
        str(expected_definition["prior_variant"]),
        label="predecessor resolved UDLM treatment",
    )
    resolved_trainer = _required_mapping(
        resolved_config.get("trainer"), label="predecessor resolved trainer config"
    )
    resolved_loader = _required_mapping(
        resolved_config.get("loader"), label="predecessor resolved loader config"
    )
    resolved_callback = _required_mapping(
        resolved_config.get("callback"), label="predecessor resolved callback config"
    )
    resolved_expectations = (
        (resolved_config, "seed", common["seed"]),
        (resolved_trainer, "devices", common["requested_gpu_count"]),
        (resolved_trainer, "num_nodes", 1),
        (resolved_trainer, "max_steps", common["max_steps"]),
        (
            resolved_trainer,
            "accumulate_grad_batches",
            common["accumulate_grad_batches"],
        ),
        (resolved_loader, "global_batch_size", common["global_batch_size"]),
        (resolved_loader, "batch_size", common["micro_batch_size_per_process"]),
        (resolved_loader, "num_workers", common["num_workers"]),
        (
            resolved_callback,
            "dirpath",
            str(Path(str(expected["final_checkpoint_path"])).parent),
        ),
    )
    for resolved_section, key, expected_value in resolved_expectations:
        if resolved_section.get(key) != expected_value:
            raise ValueError(f"predecessor resolved config {key} is unmatched")
    _exact_string(
        expected.get("resolved_training_config_sha256"),
        resolved_config_sha256,
        label="predecessor receipt resolved-config digest",
    )
    training_argv = predecessor_manifest.get("training_argv")
    if not isinstance(training_argv, list) or not all(
        isinstance(value, str) and value for value in training_argv
    ):
        raise ValueError("predecessor training argv must be a nonempty string array")
    expected_training_argv_suffix = [
        "-u",
        str(REPOSITORY_ROOT / "scripts" / "train.py"),
        "--config-name",
        str(expected_definition["config_name"]),
    ]
    reviewed_python_executables = {
        str(REPOSITORY_ROOT / ".venv" / "bin" / "python"),
        str(PROJECT_ROOT / ".venv" / "bin" / "python"),
    }
    if (
        len(training_argv) < 1 + len(expected_training_argv_suffix)
        or training_argv[0] not in reviewed_python_executables
        or training_argv[1 : 1 + len(expected_training_argv_suffix)]
        != expected_training_argv_suffix
    ):
        raise ValueError("predecessor training argv prefix is unmatched")
    argv_sha256 = _sha256(
        predecessor_manifest.get("training_argv_sha256"),
        label="predecessor training argv digest",
    )
    _exact_string(
        canonical_json_sha256(training_argv[2:]),
        argv_sha256,
        label="predecessor training argv content digest",
    )
    _exact_string(
        expected.get("training_argv_sha256"),
        argv_sha256,
        label="predecessor receipt training argv digest",
    )
    selected_uuids = predecessor_manifest.get("cuda_visible_device_uuids")
    if (
        not isinstance(selected_uuids, list)
        or len(selected_uuids) != common["requested_gpu_count"]
        or len(set(selected_uuids)) != len(selected_uuids)
        or any(
            not isinstance(value, str) or not value.startswith("GPU-")
            for value in selected_uuids
        )
        or expected.get("selected_gpu_uuids") != selected_uuids
    ):
        raise ValueError("predecessor selected GPU UUID contract is invalid")
    if (
        manifest_evidence.get("expected_selected_gpu_uuids") != selected_uuids
        or manifest_evidence.get("observed_selected_gpu_uuids") != selected_uuids
    ):
        raise ValueError("predecessor manifest evidence GPU UUIDs are unmatched")

    _exact_integer(
        predecessor_manifest.get("gpu_selection_schema_version"),
        2,
        label="predecessor GPU selection schema",
    )
    _exact_string(
        predecessor_manifest.get("gpu_selection_method"),
        "dynamic_idle_discovery",
        label="predecessor GPU selection method",
    )
    _exact_string(
        predecessor_manifest.get("gpu_inventory_scope"),
        "all_nvidia_gpus",
        label="predecessor GPU inventory scope",
    )
    inventory_time = _utc_timestamp(
        predecessor_manifest.get("inventory_snapshot_completed_at_utc"),
        label="predecessor GPU inventory timestamp",
    )
    final_probe_time = _utc_timestamp(
        predecessor_manifest.get("final_uuid_probes_completed_at_utc"),
        label="predecessor final GPU probe timestamp",
    )
    manifest_created_time = _utc_timestamp(
        predecessor_manifest.get("created_at"),
        label="predecessor manifest creation timestamp",
    )
    if not inventory_time <= final_probe_time <= manifest_created_time:
        raise ValueError("predecessor GPU and manifest timestamps are out of order")
    inventory = predecessor_manifest.get("gpu_inventory_at_selection")
    initially_selected = predecessor_manifest.get("initially_selected_gpu_states")
    final_states = predecessor_manifest.get("gpu_states_at_final_uuid_probe")
    if (
        not isinstance(inventory, list)
        or not inventory
        or not isinstance(initially_selected, list)
        or not isinstance(final_states, list)
        or len(initially_selected) != len(selected_uuids)
        or len(final_states) != len(selected_uuids)
    ):
        raise ValueError("predecessor GPU state arrays are invalid")
    inventory_records = [
        _validate_gpu_state_record(state, label=f"predecessor inventory GPU {index}")
        for index, state in enumerate(inventory)
    ]
    inventory_uuids = [state["uuid"] for state in inventory_records]
    inventory_physical_indices = [
        state["physical_index"] for state in inventory_records
    ]
    if len(set(inventory_uuids)) != len(inventory_uuids):
        raise ValueError("predecessor GPU inventory UUIDs must be unique")
    if len(set(inventory_physical_indices)) != len(inventory_physical_indices):
        raise ValueError("predecessor GPU inventory physical indices must be unique")
    inventory_by_uuid = {state["uuid"]: state for state in inventory_records}
    if any(uuid not in inventory_by_uuid for uuid in selected_uuids):
        raise ValueError("predecessor selected GPU UUIDs are absent from inventory")
    validated_state_arrays: dict[str, list[dict[str, object]]] = {}
    for label, states in (
        ("initial selection", initially_selected),
        ("final probe", final_states),
    ):
        validated_states = [
            _validate_gpu_state_record(state, label=f"predecessor {label} GPU {index}")
            for index, state in enumerate(states)
        ]
        validated_state_arrays[label] = validated_states
        if [state["uuid"] for state in validated_states] != selected_uuids:
            raise ValueError(f"predecessor {label} UUID order is unmatched")
    initially_selected_records = validated_state_arrays["initial selection"]
    final_state_records = validated_state_arrays["final probe"]
    if initially_selected_records != [
        inventory_by_uuid[uuid] for uuid in selected_uuids
    ]:
        raise ValueError(
            "predecessor initially selected GPUs differ from inventory rows"
        )
    final_physical_indices = [state["physical_index"] for state in final_state_records]
    if len(set(final_physical_indices)) != len(final_physical_indices):
        raise ValueError("predecessor final GPU physical indices must be unique")
    if (
        predecessor_manifest.get("logical_cuda_devices")
        != list(range(len(selected_uuids)))
        or predecessor_manifest.get("physical_gpu_indices") != final_physical_indices
    ):
        raise ValueError("predecessor logical/physical GPU mapping is invalid")

    safety = _required_mapping(
        predecessor_manifest.get("gpu_safety_policy"),
        label="predecessor GPU safety policy",
    )
    _require_exact_keys(
        safety,
        {
            "max_utilization_percent",
            "utilization_comparison",
            "min_free_memory_mib",
            "active_compute_processes_allowed",
            "compute_mode_prohibited_allowed",
        },
        label="predecessor GPU safety policy",
    )
    expected_safety = matched_panel_spec.get("common_gpu_safety_policy")
    expected_safety = _required_mapping(
        expected_safety, label="matched-panel GPU safety policy"
    )
    for key in safety:
        if safety[key] != expected_safety.get(key):
            raise ValueError(f"predecessor GPU safety policy {key} is unmatched")
    if (
        safety["active_compute_processes_allowed"]
        is not ACTIVE_COMPUTE_PROCESSES_ALLOWED
    ):
        raise ValueError(
            "predecessor active-compute-process policy is not the reviewed policy"
        )
    for state in (*initially_selected, *final_states):
        if (
            state["utilization_percent"] >= safety["max_utilization_percent"]
            or state["memory_total_mib"] - state["memory_used_mib"]
            < safety["min_free_memory_mib"]
            or (
                state["compute_processes"]
                and not safety["active_compute_processes_allowed"]
            )
            or str(state["compute_mode"]).strip().lower() == "prohibited"
        ):
            raise ValueError("predecessor selected GPU state violates safety policy")

    completion_contract = _required_mapping(
        predecessor_manifest.get("completion_contract"),
        label="predecessor manifest completion contract",
    )
    _require_exact_keys(
        completion_contract,
        {
            "status_at_launch",
            "complete_only_if_valid_training_summary_exists",
            "complete_only_if_successful_exit_receipt_exists",
            "valid_training_summary_and_successful_exit_receipt_both_required",
            "missing_summary_after_tmux_exit_means",
            "absent_exit_receipt_means",
            "successful_exit_receipt_requires",
            "training_job_lock_release",
        },
        label="predecessor manifest completion contract",
    )
    for key in (
        "complete_only_if_valid_training_summary_exists",
        "complete_only_if_successful_exit_receipt_exists",
        "valid_training_summary_and_successful_exit_receipt_both_required",
    ):
        _required_true(
            completion_contract.get(key),
            label=f"predecessor manifest completion contract {key}",
        )
    _exact_string(
        completion_contract.get("status_at_launch"),
        "pending",
        label="predecessor manifest status at launch",
    )
    for key in ("missing_summary_after_tmux_exit_means", "absent_exit_receipt_means"):
        _exact_string(
            completion_contract.get(key),
            "incomplete",
            label=f"predecessor manifest completion contract {key}",
        )
    receipt_requirements = _required_mapping(
        completion_contract.get("successful_exit_receipt_requires"),
        label="predecessor manifest successful-receipt requirements",
    )
    _require_exact_keys(
        receipt_requirements,
        {
            "training_exit_status",
            "tee_exit_status",
            "valid_launch_bound_training_summary",
            "exact_launch_manifest_still_matches",
            "clean_pushed_source_at_receipt",
            "predecessor_receipt_binding_unchanged_and_valid",
        },
        label="predecessor manifest successful-receipt requirements",
    )
    for key, value in receipt_requirements.items():
        if key in {"training_exit_status", "tee_exit_status"}:
            _exact_integer(
                value, 0, label=f"predecessor manifest receipt requirement {key}"
            )
        else:
            _required_true(
                value, label=f"predecessor manifest receipt requirement {key}"
            )
    _exact_string(
        completion_contract.get("training_job_lock_release"),
        "after_exit_receipt_publication_for_completed_or_failed_pipeline",
        label="predecessor manifest lock release policy",
    )

    lock_binding = _required_mapping(
        predecessor_manifest.get("single_training_job_lock"),
        label="predecessor manifest training-job lock",
    )
    _require_exact_keys(
        lock_binding,
        {
            "path",
            "sha256",
            "record",
            "acquired_before_any_gpu_probe",
            "stale_lock_policy",
            "release_owner",
        },
        label="predecessor manifest training-job lock",
    )
    lock_path = REPOSITORY_ROOT.resolve(strict=True) / (
        "output/udlm/.single_training_job.lock"
    )
    _exact_string(
        expected.get("training_job_lock_path"),
        str(lock_path),
        label="predecessor expected training-job lock path",
    )
    lock_sha256 = _sha256(
        expected.get("training_job_lock_sha256"),
        label="predecessor expected training-job lock digest",
    )
    for key, value in (("path", str(lock_path)), ("sha256", lock_sha256)):
        _exact_string(
            lock_binding.get(key),
            value,
            label=f"predecessor manifest training-job lock {key}",
        )
    _required_true(
        lock_binding.get("acquired_before_any_gpu_probe"),
        label="predecessor lock acquisition-before-GPU flag",
    )
    _exact_string(
        lock_binding.get("stale_lock_policy"),
        "fail_closed_and_require_manual_review",
        label="predecessor manifest stale-lock policy",
    )
    _exact_string(
        lock_binding.get("release_owner"),
        "pilot_exit_receipt_writer_after_publication",
        label="predecessor manifest lock release owner",
    )
    lock_record = _required_mapping(
        lock_binding.get("record"), label="predecessor training-job lock record"
    )
    _require_exact_keys(
        lock_record,
        {
            "schema_version",
            "status",
            "purpose",
            "source_revision",
            "run_name",
            "training_variant",
            "owner_token",
            "launcher_pid_at_acquisition",
            "acquired_at_utc",
            "owner_process_exit_does_not_make_lock_stale",
            "stale_lock_policy",
            "release_policy",
        },
        label="predecessor training-job lock record",
    )
    _exact_integer(
        lock_record.get("schema_version"),
        TRAINING_JOB_LOCK_SCHEMA_VERSION,
        label="predecessor training-job lock schema",
    )
    for key, value in (
        ("status", "held"),
        ("purpose", "enforce_one_R_S_E_pilot_training_job_at_a_time"),
        ("source_revision", source_revision),
        ("training_variant", expected_variant),
        ("stale_lock_policy", "fail_closed_and_require_manual_review"),
        (
            "release_policy",
            "exact_owner_lock_only_after_receipt_or_before_tmux_handoff_failure",
        ),
    ):
        _exact_string(
            lock_record.get(key), value, label=f"predecessor lock record {key}"
        )
    if lock_record.get("run_name") != predecessor_manifest.get("run_name"):
        raise ValueError("predecessor lock record run name is unmatched")
    lock_acquired_time = _utc_timestamp(
        lock_record.get("acquired_at_utc"),
        label="predecessor training-job lock acquisition timestamp",
    )
    if lock_acquired_time > inventory_time:
        raise ValueError(
            "predecessor training-job lock was acquired after GPU inventory"
        )
    _positive_integer(
        lock_record.get("launcher_pid_at_acquisition"),
        label="predecessor lock launcher PID",
    )
    owner_token = lock_record.get("owner_token")
    if not isinstance(owner_token, str) or not re.fullmatch(
        r"[0-9a-f]{64}", owner_token
    ):
        raise ValueError("predecessor lock owner token is invalid")
    _required_true(
        lock_record.get("owner_process_exit_does_not_make_lock_stale"),
        label="predecessor lock process-exit policy",
    )
    lock_record_payload = (
        json.dumps(lock_record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _exact_string(
        hashlib.sha256(lock_record_payload).hexdigest(),
        lock_sha256,
        label="predecessor training-job lock record digest",
    )

    prior_binding = _validate_predecessor_binding_shape(
        predecessor_manifest.get("predecessor_receipt_binding"),
        current_training_variant=expected_variant,
        matched_panel_spec_sha256=matched_panel_spec_sha256,
        common_training_contract_sha256=common_sha256,
        revalidate_bound_artifacts=True,
    )
    _require_predecessor_receipt_before_lock(
        prior_binding,
        current_lock_acquired_at_utc=str(lock_record.get("acquired_at_utc")),
    )
    _require_predecessor_receipt_before_manifest(
        prior_binding,
        current_manifest_created_at_utc=str(predecessor_manifest.get("created_at")),
    )
    if position - 1 > 0:
        recursively_validated = build_predecessor_receipt_binding(
            training_variant=expected_variant,
            explicit_genesis=False,
            predecessor_receipt_path=Path(
                str(prior_binding["receipt_artifact"]["path"])
            ),
            matched_panel_spec=matched_panel_spec,
            matched_panel_spec_sha256=matched_panel_spec_sha256,
        )
        if (
            canonical_json_sha256(recursively_validated)
            != canonical_json_sha256(prior_binding)
            or recursively_validated != prior_binding
        ):
            raise ValueError(
                "predecessor launch manifest carries a noncanonical transitive chain"
            )
    if receipt.get("predecessor_receipt_binding") != prior_binding:
        raise ValueError(
            "predecessor receipt does not attest its exact launch predecessor binding"
        )

    summary_path_value = expected.get("training_summary_path")
    if not isinstance(summary_path_value, str):
        raise ValueError("predecessor training-summary path must be a string")
    summary_path = _repository_artifact_path(
        Path(summary_path_value),
        suffix=".json",
        basename="training_summary.json",
        label="predecessor training summary",
    )
    if summary_path.parent != receipt_path.parent:
        raise ValueError(
            "predecessor receipt and training summary must belong to the same run"
        )
    summary_snapshot, summary_payload = stable_repository_artifact_snapshot(
        summary_path,
        suffix=".json",
        label="predecessor training summary",
    )
    summary_evidence = _required_mapping(
        receipt.get("training_summary"), label="predecessor summary evidence"
    )
    _require_exact_keys(
        summary_evidence,
        {
            "path",
            "present",
            "valid_and_launch_bound",
            "artifact",
            "validated_bindings",
            "validation_error",
        },
        label="predecessor summary evidence",
    )
    _exact_string(
        summary_evidence.get("path"),
        str(summary_path),
        label="predecessor summary evidence path",
    )
    _required_true(
        summary_evidence.get("present"),
        label="predecessor summary evidence presence",
    )
    _required_true(
        summary_evidence.get("valid_and_launch_bound"),
        label="predecessor summary validity",
    )
    if summary_evidence.get("validation_error") is not None:
        raise ValueError("predecessor summary evidence contains a validation error")
    if summary_evidence.get("artifact") != summary_snapshot:
        raise ValueError(
            "predecessor training summary no longer matches the receipt snapshot"
        )
    summary = _required_mapping(
        strict_json_loads(summary_payload, label="predecessor training summary"),
        label="predecessor training summary",
    )
    _require_exact_keys(
        summary,
        _PILOT_TRAINING_SUMMARY_KEYS,
        label="predecessor training summary",
    )
    _exact_integer(
        summary.get("schema_version"),
        TRAINING_SUMMARY_SCHEMA_VERSION,
        label="predecessor training-summary schema",
    )
    _exact_string(
        summary.get("status"), "completed", label="predecessor summary status"
    )
    summary_time = _utc_timestamp(
        summary.get("completed_at_utc"), label="predecessor completion timestamp"
    )
    for key, expected_value in (
        ("source_revision", source_revision),
        ("resolved_training_config_sha256", resolved_config_sha256),
        ("training_argv_sha256", argv_sha256),
    ):
        if summary.get(key) != expected_value:
            raise ValueError(f"predecessor summary {key} is unmatched")
    summary_source = _required_mapping(
        summary.get("source"), label="predecessor summary source"
    )
    _require_exact_keys(
        summary_source, {"head", "upstream"}, label="predecessor summary source"
    )
    for key in ("head", "upstream"):
        _exact_string(
            summary_source.get(key),
            source_revision,
            label=f"predecessor summary source {key}",
        )
    summary_manifest = _require_extended_snapshot_matches(
        manifest_snapshot,
        summary.get("launch_manifest"),
        extra_keys={"selected_gpu_uuids"},
        label="predecessor summary launch-manifest evidence",
    )
    if summary_manifest.get("selected_gpu_uuids") != selected_uuids:
        raise ValueError("predecessor summary selected GPU UUIDs are unmatched")

    summary_completion = _required_mapping(
        summary.get("completion_contract"),
        label="predecessor summary completion contract",
    )
    _require_exact_keys(
        summary_completion,
        {
            "summary_schema_version",
            "summary_path",
            "final_checkpoint_path",
            "expected_max_steps",
            "expected_world_size",
            "fail_on_nonfinite_loss",
            "backward_anomaly_detection",
        },
        label="predecessor summary completion contract",
    )
    for key, expected_value in (
        ("summary_schema_version", TRAINING_SUMMARY_SCHEMA_VERSION),
        ("expected_max_steps", common["max_steps"]),
        ("expected_world_size", common["requested_gpu_count"]),
    ):
        _exact_integer(
            summary_completion.get(key),
            expected_value,
            label=f"predecessor summary completion {key}",
        )
    for key, expected_value in (
        ("summary_path", str(summary_path)),
        ("final_checkpoint_path", expected.get("final_checkpoint_path")),
    ):
        _exact_string(
            summary_completion.get(key),
            expected_value,
            label=f"predecessor summary completion {key}",
        )
    for key in ("fail_on_nonfinite_loss", "backward_anomaly_detection"):
        _required_true(
            summary_completion.get(key),
            label=f"predecessor summary completion {key}",
        )

    runtime_path_value = predecessor_manifest.get("runtime_config_path")
    if not isinstance(runtime_path_value, str):
        raise ValueError("predecessor runtime-config path must be a string")
    runtime_path = _repository_artifact_path(
        Path(runtime_path_value),
        suffix=".json",
        basename="runtime_config.json",
        label="predecessor runtime config",
    )
    if runtime_path.parent != receipt_path.parent:
        raise ValueError("predecessor runtime config belongs to a different run")
    runtime_snapshot, runtime_payload = stable_repository_artifact_snapshot(
        runtime_path,
        suffix=".json",
        label="predecessor runtime config",
    )
    runtime_evidence = _required_mapping(
        receipt.get("runtime_config"), label="predecessor runtime-config evidence"
    )
    _require_exact_keys(
        runtime_evidence,
        {
            "path",
            "present",
            "matches_training_summary_snapshot",
            "semantic_validation_passed",
            "artifact",
        },
        label="predecessor runtime-config evidence",
    )
    _exact_string(
        runtime_evidence.get("path"),
        str(runtime_path),
        label="predecessor runtime-config evidence path",
    )
    for key in (
        "present",
        "matches_training_summary_snapshot",
        "semantic_validation_passed",
    ):
        _required_true(
            runtime_evidence.get(key),
            label=f"predecessor runtime-config evidence {key}",
        )
    if runtime_evidence.get("artifact") != runtime_snapshot:
        raise ValueError("predecessor runtime config no longer matches its receipt")
    summary_runtime = _require_extended_snapshot_matches(
        runtime_snapshot,
        summary.get("runtime_config"),
        extra_keys={"schema_version", "record_sha256"},
        label="predecessor summary runtime-config evidence",
    )
    _exact_integer(
        summary_runtime.get("schema_version"),
        2,
        label="predecessor runtime-config schema",
    )
    runtime = _required_mapping(
        strict_json_loads(runtime_payload, label="predecessor runtime config"),
        label="predecessor runtime config",
    )
    _require_exact_keys(
        runtime, _PILOT_RUNTIME_CONFIG_KEYS, label="predecessor runtime config"
    )
    _exact_string(
        canonical_json_sha256(runtime),
        _sha256(
            summary_runtime.get("record_sha256"),
            label="predecessor runtime-config canonical digest",
        ),
        label="predecessor runtime-config canonical content digest",
    )
    _exact_integer(
        runtime.get("schema_version"), 2, label="predecessor runtime-config schema"
    )
    _exact_string(
        runtime.get("status"),
        "preflight_completed",
        label="predecessor runtime-config status",
    )
    for key, expected_value in (
        ("source_revision", source_revision),
        ("resolved_training_config_sha256", resolved_config_sha256),
        ("training_argv_sha256", argv_sha256),
    ):
        _exact_string(
            runtime.get(key), expected_value, label=f"predecessor runtime {key}"
        )
    if runtime.get("source") != summary_source:
        raise ValueError("predecessor runtime and summary source bindings differ")
    runtime_argv = runtime.get("training_argv")
    if (
        runtime_argv != training_argv[2:]
        or runtime.get("observed_training_argv") != runtime_argv
        or canonical_json_sha256(runtime_argv) != argv_sha256
    ):
        raise ValueError("predecessor runtime training argv is unmatched")
    if (
        runtime.get("resolved_training_config") != resolved_config
        or runtime.get("completion_contract") != summary_completion
    ):
        raise ValueError(
            "predecessor runtime config or completion binding is unmatched"
        )
    runtime_manifest = _require_extended_snapshot_matches(
        manifest_snapshot,
        runtime.get("launch_manifest"),
        extra_keys={"selected_gpu_uuids"},
        label="predecessor runtime launch-manifest evidence",
    )
    if runtime_manifest != summary_manifest:
        raise ValueError("predecessor runtime and summary manifest evidence differ")
    python_environment = _required_mapping(
        runtime.get("python_environment"),
        label="predecessor runtime Python environment",
    )
    expected_python_environment = {
        **CONTROLLED_PYTHON_ENVIRONMENT,
        "PYTHONHASHSEED": str(common["seed"]),
        "PYTHONPATH": os.pathsep.join(
            (str(REPOSITORY_ROOT / "src"), str(REPOSITORY_ROOT))
        ),
    }
    if python_environment != expected_python_environment:
        raise ValueError("predecessor runtime Python environment is unmatched")

    final_checkpoint_value = expected.get("final_checkpoint_path")
    if not isinstance(final_checkpoint_value, str):
        raise ValueError("predecessor final-checkpoint path must be a string")
    final_checkpoint_path = _repository_artifact_path(
        Path(final_checkpoint_value),
        suffix=".ckpt",
        basename=f"{common['max_steps']}.ckpt",
        label="predecessor final checkpoint",
    )
    if final_checkpoint_path.parent.parent != receipt_path.parent:
        raise ValueError("predecessor final checkpoint belongs to a different run")
    checkpoint_snapshot, _checkpoint_payload = stable_repository_artifact_snapshot(
        final_checkpoint_path,
        suffix=".ckpt",
        label="predecessor final checkpoint",
        capture_bytes=False,
    )
    checkpoint_evidence = _required_mapping(
        receipt.get("final_checkpoint"),
        label="predecessor final-checkpoint evidence",
    )
    _require_exact_keys(
        checkpoint_evidence,
        {"path", "present", "matches_training_summary_snapshot", "artifact"},
        label="predecessor final-checkpoint evidence",
    )
    _exact_string(
        checkpoint_evidence.get("path"),
        str(final_checkpoint_path),
        label="predecessor final-checkpoint evidence path",
    )
    for key in ("present", "matches_training_summary_snapshot"):
        _required_true(
            checkpoint_evidence.get(key),
            label=f"predecessor final-checkpoint evidence {key}",
        )
    if checkpoint_evidence.get("artifact") != checkpoint_snapshot:
        raise ValueError("predecessor final checkpoint no longer matches its receipt")
    summary_checkpoint = _require_extended_snapshot_matches(
        checkpoint_snapshot,
        summary.get("final_checkpoint"),
        extra_keys={"semantic_audit"},
        label="predecessor summary final-checkpoint evidence",
    )
    semantic_checkpoint = _required_mapping(
        summary_checkpoint.get("semantic_audit"),
        label="predecessor checkpoint semantic audit",
    )
    _require_exact_keys(
        semantic_checkpoint,
        {
            "deserialized",
            "global_step",
            "raw_model",
            "ema",
            "ema_metadata",
            "optimizer",
            "non_sentinel_checkpoint_tensors",
            "checkpoint_python_floats",
            "framework_nonfinite_sentinels",
            "checkpoint_hyperparameters_match",
            "checkpoint_loop_state_match",
            "optimizer_live_state_match",
            "scheduler_live_state_match",
            "sampler_live_state_match",
            "trainer_live_configuration_match",
            "model_checkpoint_live_state_match",
            "udlm_process_identity_verified",
            "live_model_match",
            "live_ema_match",
        },
        label="predecessor checkpoint semantic audit",
    )
    _required_true(
        semantic_checkpoint.get("deserialized"),
        label="predecessor checkpoint deserialization",
    )
    _required_true(
        semantic_checkpoint.get("udlm_process_identity_verified"),
        label="predecessor checkpoint UDLM identity",
    )
    _exact_integer(
        semantic_checkpoint.get("global_step"),
        common["max_steps"],
        label="predecessor checkpoint global step",
    )
    checkpoint_finiteness = {}
    for key in (
        "raw_model",
        "ema",
        "optimizer",
        "non_sentinel_checkpoint_tensors",
    ):
        checkpoint_finiteness[key] = _validate_finiteness_record(
            semantic_checkpoint.get(key),
            label=f"predecessor checkpoint {key}",
        )
    _validate_framework_nonfinite_sentinels(
        semantic_checkpoint.get("framework_nonfinite_sentinels"),
        expected_steps=common["max_steps"],
        label="predecessor checkpoint framework non-finite sentinels",
    )
    optimizer_parameter_state_count = _validate_auxiliary_checkpoint_records(
        semantic_checkpoint,
        expected_steps=common["max_steps"],
        resolved_training_config=resolved_config,
        label="predecessor checkpoint",
    )
    ema_metadata = _required_mapping(
        semantic_checkpoint.get("ema_metadata"),
        label="predecessor checkpoint EMA metadata",
    )
    _require_exact_keys(
        ema_metadata,
        {"shadow_parameter_count", "decay", "num_updates"},
        label="predecessor checkpoint EMA metadata",
    )
    shadow_parameter_count = _positive_integer(
        ema_metadata.get("shadow_parameter_count"),
        label="predecessor checkpoint EMA shadow count",
    )
    if optimizer_parameter_state_count != shadow_parameter_count:
        raise ValueError(
            "predecessor optimizer parameter-state count disagrees with EMA shadow count"
        )
    _exact_integer(
        ema_metadata.get("num_updates"),
        common["max_steps"],
        label="predecessor checkpoint EMA update count",
    )
    decay = ema_metadata.get("decay")
    if (
        type(decay) not in (int, float)
        or not math.isfinite(decay)
        or not 0.0 < decay < 1.0
    ):
        raise ValueError("predecessor checkpoint EMA decay is invalid")
    configured_decay = resolved_training.get("ema")
    if (
        type(configured_decay) not in (int, float)
        or not math.isfinite(configured_decay)
        or float(configured_decay) != float(decay)
    ):
        raise ValueError(
            "predecessor checkpoint EMA decay disagrees with resolved config"
        )
    live_matches = {}
    for key, required_keys in (
        ("live_model_match", {"exact_key_set", "exact_tensor_values", "tensor_count"}),
        ("live_ema_match", {"exact_tensor_values", "tensor_count"}),
    ):
        live_match = _required_mapping(
            semantic_checkpoint.get(key), label=f"predecessor checkpoint {key}"
        )
        _require_exact_keys(
            live_match, required_keys, label=f"predecessor checkpoint {key}"
        )
        live_matches[key] = live_match
        for flag in required_keys - {"tensor_count"}:
            _required_true(
                live_match.get(flag), label=f"predecessor checkpoint {key} {flag}"
            )
        _positive_integer(
            live_match.get("tensor_count"),
            label=f"predecessor checkpoint {key} tensor count",
        )
    _exact_integer(
        live_matches["live_ema_match"].get("tensor_count"),
        shadow_parameter_count,
        label="predecessor live EMA tensor count",
    )
    _exact_integer(
        checkpoint_finiteness["ema"].get("floating_tensor_count"),
        shadow_parameter_count,
        label="predecessor serialized EMA tensor count",
    )

    observed_state = _required_mapping(
        summary.get("observed_training_state"),
        label="predecessor observed training state",
    )
    _require_exact_keys(
        observed_state,
        {"global_rank", "global_step", "world_size"},
        label="predecessor observed training state",
    )
    for key, expected_value in (
        ("global_rank", 0),
        ("global_step", common["max_steps"]),
        ("world_size", common["requested_gpu_count"]),
    ):
        _exact_integer(
            observed_state.get(key),
            expected_value,
            label=f"predecessor observed training {key}",
        )

    summary_accounting = _required_mapping(
        summary.get("training_accounting"),
        label="predecessor summary training accounting",
    )
    _require_exact_keys(
        summary_accounting,
        _PILOT_TRAINING_ACCOUNTING_KEYS,
        label="predecessor summary training accounting",
    )
    summary_accounting_expectations = {
        "training_seed": common["seed"],
        "optimizer_updates": common["max_steps"],
        "world_size": common["requested_gpu_count"],
        "micro_batch_size_per_rank": common["micro_batch_size_per_process"],
        "accumulate_grad_batches": common["accumulate_grad_batches"],
        "effective_global_examples_per_optimizer_step": common[
            "effective_global_batch_size"
        ],
        "total_requested_example_exposures": common["effective_global_batch_size"]
        * common["max_steps"],
    }
    for key, expected_value in summary_accounting_expectations.items():
        _exact_integer(
            summary_accounting.get(key),
            expected_value,
            label=f"predecessor summary accounting {key}",
        )
    _exact_string(
        summary_accounting.get("hosted_stream_rank_partition_policy"),
        _HOSTED_STREAM_RANK_PARTITION_POLICY,
        label="predecessor hosted-stream partition policy",
    )
    parameter_counts = _required_mapping(
        summary_accounting.get("trainable_parameter_counts"),
        label="predecessor trainable parameter counts",
    )
    conditioning_variant = resolved_udlm.get("conditioning_variant")
    if conditioning_variant == "additive":
        expected_parameter_keys = {
            "base_backbone",
            "time_conditioner",
            "total",
        }
    elif conditioning_variant == "film_adaln":
        expected_parameter_keys = {
            "base_backbone",
            "time_conditioner",
            "film_modulation",
            "total",
        }
    else:
        raise ValueError(
            "predecessor resolved UDLM conditioning_variant must be additive or "
            "film_adaln"
        )
    _require_exact_keys(
        parameter_counts,
        expected_parameter_keys,
        label="predecessor trainable parameter counts",
    )
    base_count = _positive_integer(
        parameter_counts.get("base_backbone"),
        label="predecessor base-backbone parameter count",
    )
    time_count = _positive_integer(
        parameter_counts.get("time_conditioner"),
        label="predecessor time-conditioner parameter count",
    )
    film_count = 0
    if conditioning_variant == "film_adaln":
        film_count = _positive_integer(
            parameter_counts.get("film_modulation"),
            label="predecessor FiLM-modulation parameter count",
        )
    _exact_integer(
        parameter_counts.get("total"),
        base_count + time_count + film_count,
        label="predecessor total trainable parameter count",
    )

    health = _required_mapping(
        summary.get("training_health"), label="predecessor training health"
    )
    _require_exact_keys(
        health,
        {
            "scope",
            "all_losses_finite",
            "all_observed_gradients_finite",
            "every_optimizer_step_had_a_nonzero_gradient",
            "loss_checks",
            "optimizer_step_checks",
            "gradient_tensor_observations",
            "gradient_element_observations",
        },
        label="predecessor training health",
    )
    _exact_string(
        health.get("scope"),
        (
            "global-rank-zero callback counters; identical fail-fast checks "
            "execute independently on every rank"
        ),
        label="predecessor training-health scope",
    )
    for key in (
        "all_losses_finite",
        "all_observed_gradients_finite",
        "every_optimizer_step_had_a_nonzero_gradient",
    ):
        _required_true(health.get(key), label=f"predecessor training health {key}")
    loss_checks = _positive_integer(
        health.get("loss_checks"), label="predecessor loss-check count"
    )
    optimizer_checks = _positive_integer(
        health.get("optimizer_step_checks"),
        label="predecessor optimizer-check count",
    )
    gradient_tensors = _positive_integer(
        health.get("gradient_tensor_observations"),
        label="predecessor gradient-tensor count",
    )
    gradient_elements = _positive_integer(
        health.get("gradient_element_observations"),
        label="predecessor gradient-element count",
    )
    if (
        loss_checks < common["max_steps"]
        or optimizer_checks != common["max_steps"]
        or gradient_tensors < optimizer_checks
        or gradient_elements < gradient_tensors
    ):
        raise ValueError("predecessor training-health counters are inconsistent")
    if (
        summary.get("conditioning_gradient_audit") is not None
        or summary.get("screen_initialization_state_audit") is not None
    ):
        raise ValueError("matched R/S/E predecessor contains screen-only audits")
    tensor_finiteness = _required_mapping(
        summary.get("tensor_finiteness"),
        label="predecessor live tensor finiteness",
    )
    _require_exact_keys(
        tensor_finiteness,
        {"raw_model", "ema"},
        label="predecessor live tensor finiteness",
    )
    for key in ("raw_model", "ema"):
        live_finiteness = _validate_finiteness_record(
            tensor_finiteness.get(key),
            label=f"predecessor live {key} finiteness",
        )
        if live_finiteness != semantic_checkpoint[key]:
            raise ValueError("predecessor live/checkpoint finiteness differs")
    startup = _required_mapping(summary.get("startup"), label="predecessor startup")
    _require_exact_keys(
        startup,
        {"mode", "verified_mdlm_warm_start_report"},
        label="predecessor startup",
    )
    expected_startup = (
        "scratch"
        if common.get("initialization_checkpoint_sha256") is None
        else "warm_start"
    )
    _exact_string(
        startup.get("mode"), expected_startup, label="predecessor startup mode"
    )
    if expected_startup == "scratch":
        if startup.get("verified_mdlm_warm_start_report") is not None:
            raise ValueError("scratch predecessor contains a warm-start report")
    else:
        warm_start = _required_mapping(
            startup.get("verified_mdlm_warm_start_report"),
            label="predecessor warm-start report",
        )
        expected_warm_start_keys = {
            "source_path",
            "source_resolved_path",
            "source_sha256",
            "source_size_bytes",
            "expected_source_sha256",
            "byte_identity_verified_before_and_after_load",
            "weights",
            "parameter_tensors",
        }
        if conditioning_variant == "film_adaln":
            expected_warm_start_keys.update(
                {"conditioning_variant", "conditioning_parameter_tensors"}
            )
        _require_exact_keys(
            warm_start,
            expected_warm_start_keys,
            label="predecessor warm-start report",
        )
        for key in ("source_sha256", "expected_source_sha256"):
            _exact_string(
                warm_start.get(key),
                common["initialization_checkpoint_sha256"],
                label=f"predecessor warm-start {key}",
            )
        _required_true(
            warm_start.get("byte_identity_verified_before_and_after_load"),
            label="predecessor warm-start byte identity",
        )
        expected_source_path = common["initialization_checkpoint_path"]
        for key in ("source_path", "source_resolved_path"):
            _exact_string(
                warm_start.get(key),
                expected_source_path,
                label=f"predecessor warm-start {key}",
            )
        _exact_string(
            warm_start.get("weights"),
            "ema",
            label="predecessor warm-start weights",
        )
        _positive_integer(
            warm_start.get("source_size_bytes"),
            label="predecessor warm-start source size",
        )
        _positive_integer(
            warm_start.get("parameter_tensors"),
            label="predecessor warm-start parameter tensor count",
        )
        if conditioning_variant == "film_adaln":
            _exact_string(
                warm_start.get("conditioning_variant"),
                "film_adaln",
                label="predecessor warm-start conditioning variant",
            )
            _exact_integer(
                warm_start.get("conditioning_parameter_tensors"),
                28,
                label="predecessor warm-start conditioning parameter tensor count",
            )
    bindings = _required_mapping(
        summary_evidence.get("validated_bindings"),
        label="predecessor validated summary bindings",
    )
    _require_exact_keys(
        bindings,
        {
            "schema_version",
            "source_revision",
            "resolved_training_config_sha256",
            "training_argv_sha256",
            "launch_manifest_path",
            "launch_manifest_sha256",
            "selected_gpu_uuids",
            "observed_global_step",
            "observed_world_size",
            "training_accounting",
            "ema_metadata",
            "final_checkpoint_path",
            "final_checkpoint_sha256",
            "startup_mode",
            "conditioning_gradient_audit",
            "screen_initialization_state_audit",
        },
        label="predecessor validated summary bindings",
    )
    binding_expectations = {
        "schema_version": TRAINING_SUMMARY_SCHEMA_VERSION,
        "source_revision": source_revision,
        "resolved_training_config_sha256": resolved_config_sha256,
        "training_argv_sha256": argv_sha256,
        "launch_manifest_path": str(manifest_path),
        "launch_manifest_sha256": expected_manifest_sha256,
        "selected_gpu_uuids": selected_uuids,
        "observed_global_step": common["max_steps"],
        "observed_world_size": common["requested_gpu_count"],
        "final_checkpoint_path": expected.get("final_checkpoint_path"),
        "final_checkpoint_sha256": checkpoint_snapshot["sha256"],
        "startup_mode": expected_startup,
        "conditioning_gradient_audit": None,
        "screen_initialization_state_audit": None,
    }
    for key, expected_value in binding_expectations.items():
        if bindings.get(key) != expected_value:
            raise ValueError(
                f"predecessor validated summary binding {key} is unmatched"
            )
    accounting = _required_mapping(
        bindings.get("training_accounting"),
        label="predecessor validated training accounting",
    )
    accounting_expectations = {
        "training_seed": common["seed"],
        "optimizer_updates": common["max_steps"],
        "world_size": common["requested_gpu_count"],
        "micro_batch_size_per_rank": common["micro_batch_size_per_process"],
        "accumulate_grad_batches": common["accumulate_grad_batches"],
        "effective_global_examples_per_optimizer_step": common[
            "effective_global_batch_size"
        ],
    }
    for key, expected_value in accounting_expectations.items():
        if accounting.get(key) != expected_value:
            raise ValueError(f"predecessor accounting {key} is unmatched")
    if accounting != summary_accounting:
        raise ValueError(
            "predecessor receipt accounting differs from the training summary"
        )
    if bindings.get("ema_metadata") != ema_metadata:
        raise ValueError("predecessor receipt EMA metadata differs from checkpoint")

    lock_evidence = _required_mapping(
        receipt.get("training_job_lock"),
        label="predecessor training-job lock evidence",
    )
    _require_exact_keys(
        lock_evidence,
        {
            "path",
            "present",
            "expected_sha256",
            "matches_expected_raw_sha256",
            "matches_launch_manifest_binding",
            "valid_and_launch_bound_before_receipt_publication",
            "artifact",
            "record",
            "release_policy",
            "release_result_not_claimed_inside_pre_release_receipt",
            "validation_error",
        },
        label="predecessor training-job lock evidence",
    )
    _exact_string(
        lock_evidence.get("path"),
        str(lock_path),
        label="predecessor training-job lock evidence path",
    )
    _exact_string(
        lock_evidence.get("expected_sha256"),
        lock_sha256,
        label="predecessor training-job lock expected digest",
    )
    _required_true(
        lock_evidence.get("present"),
        label="predecessor training-job lock evidence presence",
    )
    _required_true(
        lock_evidence.get("release_result_not_claimed_inside_pre_release_receipt"),
        label="predecessor training-job lock pre-release receipt flag",
    )
    lock_snapshot = _validate_stable_snapshot_claim(
        lock_evidence.get("artifact"),
        expected_path=lock_path,
        label="predecessor training-job lock snapshot",
    )
    _exact_string(
        lock_snapshot.get("sha256"),
        lock_sha256,
        label="predecessor training-job lock snapshot digest",
    )
    _exact_integer(
        lock_snapshot.get("link_count"),
        1,
        label="predecessor training-job lock snapshot link count",
    )
    _exact_integer(
        lock_snapshot.get("size_bytes"),
        len(lock_record_payload),
        label="predecessor training-job lock snapshot size",
    )
    if lock_evidence.get("record") != lock_record:
        raise ValueError("predecessor receipt and manifest lock records differ")

    for evidence_key, flags in (
        (
            "runtime_config",
            ("matches_training_summary_snapshot", "semantic_validation_passed"),
        ),
        ("final_checkpoint", ("matches_training_summary_snapshot",)),
        (
            "training_job_lock",
            (
                "matches_expected_raw_sha256",
                "matches_launch_manifest_binding",
                "valid_and_launch_bound_before_receipt_publication",
            ),
        ),
    ):
        evidence = _required_mapping(
            receipt.get(evidence_key), label=f"predecessor {evidence_key} evidence"
        )
        for flag in flags:
            _required_true(
                evidence.get(flag), label=f"predecessor {evidence_key} {flag}"
            )
        if evidence.get("validation_error") is not None:
            raise ValueError(
                f"predecessor {evidence_key} evidence contains a validation error"
            )

    manifest_time = _utc_timestamp(
        predecessor_manifest.get("created_at"),
        label="predecessor manifest creation timestamp",
    )
    if not manifest_time < summary_time < receipt_time:
        raise ValueError(
            "predecessor manifest, training completion, and receipt timestamps "
            "must be strictly ordered"
        )
    run_name = predecessor_manifest.get("run_name")
    if not isinstance(run_name, str) or not RUN_NAME_PATTERN.fullmatch(run_name):
        raise ValueError("predecessor manifest run name is invalid")
    if run_name != receipt_path.parent.name:
        raise ValueError("predecessor run name disagrees with its artifact directory")
    _exact_string(
        predecessor_manifest.get("tmux_session"),
        f"genmol_{expected_variant}_{run_name}",
        label="predecessor tmux session",
    )
    _exact_string(
        predecessor_manifest.get("launch_manifest_raw_sha256_transport"),
        "passed_out_of_band_to_training_and_receipt_to_avoid_self_hash",
        label="predecessor manifest digest transport",
    )
    _required_true(
        predecessor_manifest.get("log_reserved_exclusively_before_manifest"),
        label="predecessor log reservation flag",
    )
    _exact_string(
        predecessor_manifest.get("log_path"),
        str(REPOSITORY_ROOT.resolve(strict=True) / f"output/logs/{run_name}.log"),
        label="predecessor log path",
    )
    _exact_string(
        lock_evidence.get("release_policy"),
        "publish_receipt_then_unlink_only_same_stat_identity_and_sha256",
        label="predecessor receipt lock release policy",
    )
    binding = {
        "schema_version": PREDECESSOR_RECEIPT_BINDING_SCHEMA_VERSION,
        "state": "validated_successful_predecessor",
        "current_training_variant": training_variant,
        "current_variant_position": position,
        "expected_predecessor_training_variant": expected_variant,
        "expected_predecessor_variant_position": position - 1,
        "matched_panel_spec_sha256": matched_panel_spec_sha256,
        "common_training_contract_sha256": common_sha256,
        "receipt_artifact": receipt_snapshot,
        "predecessor_launch_manifest_artifact": manifest_snapshot,
        "predecessor_training_summary_artifact": summary_snapshot,
        "predecessor_run_name": run_name,
        "chronology": {
            "predecessor_launch_manifest_created_at_utc": predecessor_manifest[
                "created_at"
            ],
            "predecessor_training_summary_completed_at_utc": summary[
                "completed_at_utc"
            ],
            "predecessor_exit_receipt_recorded_at_utc": receipt["recorded_at_utc"],
            "strictly_ordered_timestamps_verified": True,
        },
        "validated_before_gpu_probe": True,
    }
    return _validate_predecessor_binding_shape(
        binding,
        current_training_variant=training_variant,
        matched_panel_spec_sha256=matched_panel_spec_sha256,
        common_training_contract_sha256=common_sha256,
        revalidate_bound_artifacts=True,
    )


def revalidate_predecessor_receipt_binding(
    binding: dict[str, object],
    *,
    training_variant: str,
    matched_panel_spec: dict[str, object],
    matched_panel_spec_sha256: str,
    current_manifest_created_at_utc: str | None = None,
) -> None:
    """Rebuild the unchanged full chain immediately around GPU probing."""

    _common, common_sha256 = _matched_panel_contract(
        matched_panel_spec, matched_panel_spec_sha256
    )
    validated = _validate_predecessor_binding_shape(
        binding,
        current_training_variant=training_variant,
        matched_panel_spec_sha256=matched_panel_spec_sha256,
        common_training_contract_sha256=common_sha256,
        revalidate_bound_artifacts=True,
    )
    position = MATCHED_PANEL_VARIANT_ORDER.index(
        validate_training_variant(training_variant)
    )
    if position > 0:
        receipt = _required_mapping(
            validated.get("receipt_artifact"),
            label="predecessor receipt artifact",
        )
        rebuilt = build_predecessor_receipt_binding(
            training_variant=training_variant,
            explicit_genesis=False,
            predecessor_receipt_path=Path(str(receipt.get("path"))),
            matched_panel_spec=matched_panel_spec,
            matched_panel_spec_sha256=matched_panel_spec_sha256,
        )
        if (
            canonical_json_sha256(rebuilt) != canonical_json_sha256(validated)
            or rebuilt != validated
        ):
            raise ValueError(
                "rebuilt predecessor chain differs from its exact canonical binding"
            )
    if current_manifest_created_at_utc is not None:
        _require_predecessor_receipt_before_manifest(
            validated,
            current_manifest_created_at_utc=current_manifest_created_at_utc,
        )


def _snapshot_stat_identity_matching_claim(
    claim: object, *, suffix: str, label: str
) -> Path:
    """Cheaply require an artifact to retain every bound lstat identity field."""

    snapshot = _required_mapping(claim, label=f"{label} claim")
    path_value = snapshot.get("path")
    if not isinstance(path_value, str):
        raise ValueError(f"{label} path must be a string")
    normalized = _repository_artifact_path(Path(path_value), suffix=suffix, label=label)
    _validate_stable_snapshot_claim(
        snapshot,
        expected_path=normalized,
        label=f"{label} snapshot",
    )
    state = normalized.stat(follow_symlinks=False)
    claimed_identity = (
        snapshot["device"],
        snapshot["inode"],
        snapshot["mode"],
        snapshot["link_count"],
        snapshot["size_bytes"],
        snapshot["mtime_ns"],
        snapshot["ctime_ns"],
    )
    if _stable_stat_identity(state) != claimed_identity:
        raise ValueError(f"{label} no longer matches its bound stat identity")
    return normalized


def revalidate_predecessor_chain_artifact_identities(
    binding: dict[str, object], *, current_manifest_created_at_utc: str
) -> None:
    """Cheaply close normal mutation races after full transitive validation."""

    current_binding = _required_mapping(binding, label="predecessor receipt binding")
    successor_manifest_created_at_utc = current_manifest_created_at_utc
    visited_receipts: set[str] = set()
    while current_binding.get("state") != "explicit_genesis_no_predecessor":
        if current_binding.get("state") != "validated_successful_predecessor":
            raise ValueError("predecessor chain contains an invalid binding state")
        _require_predecessor_receipt_before_manifest(
            current_binding,
            current_manifest_created_at_utc=successor_manifest_created_at_utc,
        )
        receipt_claim = current_binding.get("receipt_artifact")
        receipt_snapshot, receipt_payload = _snapshot_payload_matching_claim(
            receipt_claim,
            suffix=".json",
            label="post-probe predecessor receipt",
        )
        receipt_path = str(receipt_snapshot["path"])
        if receipt_path in visited_receipts:
            raise ValueError("predecessor chain contains an artifact cycle")
        visited_receipts.add(receipt_path)
        manifest_snapshot, manifest_payload = _snapshot_payload_matching_claim(
            current_binding.get("predecessor_launch_manifest_artifact"),
            suffix=".json",
            label="post-probe predecessor launch manifest",
        )
        _snapshot_stat_identity_matching_claim(
            current_binding.get("predecessor_training_summary_artifact"),
            suffix=".json",
            label="post-probe predecessor training summary",
        )
        receipt = _required_mapping(
            strict_json_loads(receipt_payload, label="post-probe predecessor receipt"),
            label="post-probe predecessor receipt",
        )
        for evidence_key, suffix in (
            ("runtime_config", ".json"),
            ("final_checkpoint", ".ckpt"),
        ):
            evidence = _required_mapping(
                receipt.get(evidence_key),
                label=f"post-probe predecessor {evidence_key} evidence",
            )
            _snapshot_stat_identity_matching_claim(
                evidence.get("artifact"),
                suffix=suffix,
                label=f"post-probe predecessor {evidence_key}",
            )
        manifest = _required_mapping(
            strict_json_loads(
                manifest_payload, label="post-probe predecessor launch manifest"
            ),
            label="post-probe predecessor launch manifest",
        )
        if manifest_snapshot["path"] == receipt_path:
            raise ValueError("predecessor receipt and manifest paths must differ")
        current_binding = _required_mapping(
            manifest.get("predecessor_receipt_binding"),
            label="nested predecessor receipt binding",
        )
        successor_manifest_created_at_utc = str(manifest.get("created_at"))


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
    predecessor_receipt_binding: dict[str, object],
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
    validate_pilot_output_parents(
        run_dir=run_dir,
        log_path=log_path,
        create_missing=False,
    )
    _require_predecessor_receipt_before_lock(
        predecessor_receipt_binding,
        current_lock_acquired_at_utc=str(lock_record.get("acquired_at_utc")),
    )

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
    # The initial binding was fully validated before any GPU query. Rebuild and
    # hash the transitive chain again under the global lock after selection, but
    # before the genuinely final UUID telemetry is captured.
    revalidate_predecessor_receipt_binding(
        predecessor_receipt_binding,
        training_variant=training_variant,
        matched_panel_spec=matched_panel_spec,
        matched_panel_spec_sha256=matched_panel_spec_sha256,
    )
    gpu_states = reprobe_selected_gpus(
        initially_selected,
        max_utilization_percent=args.max_utilization_percent,
        min_free_memory_mib=args.min_free_memory_mib,
    )
    final_uuid_probes_completed_at_utc = datetime.now(timezone.utc).isoformat()
    manifest_created_at_utc = datetime.now(timezone.utc).isoformat()
    revalidate_predecessor_chain_artifact_identities(
        predecessor_receipt_binding,
        current_manifest_created_at_utc=manifest_created_at_utc,
    )
    selected_gpu_uuids = [state.uuid for state in gpu_states]
    selected_gpu_uuids_json = json.dumps(selected_gpu_uuids, separators=(",", ":"))
    visible_uuids = ",".join(selected_gpu_uuids)

    # Reserve every externally visible destination before publishing the
    # immutable launch certificate. Retained partial reservations make a failed
    # launch non-reusable rather than silently overwritable.
    validate_pilot_output_parents(
        run_dir=run_dir,
        log_path=log_path,
        create_missing=False,
    )
    run_dir.mkdir(parents=True)
    reserve_log_path(log_path)
    manifest = {
        "launch_manifest_schema_version": LAUNCH_MANIFEST_SCHEMA_VERSION,
        "created_at": manifest_created_at_utc,
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
        "predecessor_receipt_binding": predecessor_receipt_binding,
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
            "active_compute_processes_allowed": ACTIVE_COMPUTE_PROCESSES_ALLOWED,
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
                "predecessor_receipt_binding_unchanged_and_valid": True,
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
    predecessor = parser.add_mutually_exclusive_group(required=True)
    predecessor.add_argument(
        "--genesis",
        action="store_true",
        help="Explicitly declare the R/udlm arm has no predecessor.",
    )
    predecessor.add_argument(
        "--predecessor-receipt",
        type=Path,
        help=(
            "Exact successful receipt from the immediately preceding R/S arm; "
            "required for S and E."
        ),
    )
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


def main(argv: list[str] | None = None):
    """Run one bounded pilot, optionally from an explicit wrapper-supplied argv."""

    args = _parse_args() if argv is None else _parse_args(argv)
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

    run_dir = REPOSITORY_ROOT / "output" / "udlm" / args.run_name
    log_path = REPOSITORY_ROOT / "output" / "logs" / f"{args.run_name}.log"
    manifest_path = run_dir / "launch_manifest.json"
    validate_pilot_output_parents(
        run_dir=run_dir,
        log_path=log_path,
        create_missing=False,
    )
    verified_floor_audit_binding = verify_pilot_empirical_uniform_mix_audit()
    git_sha = require_pushed_commit()
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
    common_training_contract = _required_mapping(
        matched_panel_spec.get("common_training_contract"),
        label="matched-panel common training contract",
    )
    if (
        common_training_contract.get("empirical_uniform_mix_audit")
        != verified_floor_audit_binding
        or common_training_contract.get("empirical_uniform_mix")
        != PILOT_EMPIRICAL_UNIFORM_MIX
    ):
        raise RuntimeError(
            "matched-panel floor selection does not equal the live verified audit"
        )
    predecessor_receipt_binding = build_predecessor_receipt_binding(
        training_variant=training_variant,
        explicit_genesis=args.genesis,
        predecessor_receipt_path=args.predecessor_receipt,
        matched_panel_spec=matched_panel_spec,
        matched_panel_spec_sha256=matched_panel_spec_sha256,
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
            "predecessor_receipt_binding": predecessor_receipt_binding,
        }
        print(json.dumps(preview, indent=2, sort_keys=True))
        return

    validate_pilot_output_parents(
        run_dir=run_dir,
        log_path=log_path,
        create_missing=True,
    )
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
            predecessor_receipt_binding=predecessor_receipt_binding,
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

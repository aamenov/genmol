"""Validate the exact health-only R -> S -> E UDLM training panel.

This module is deliberately CPU-only.  It consumes completed producer
artifacts, delegates their full receipt semantics (including the recursive
predecessor chain) to :mod:`write_pilot_evidence`, and then narrows the
validated panel to the registered ten-update health contract.  A successful
result authorizes only the optimization screen; it is not generation,
ranking, superiority, or candidate-lock evidence.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.udlm import launch_train_pilot
from scripts.udlm import write_pilot_evidence as pilot_evidence_writer


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPOSITORY_ROOT.parents[1]
HEALTH_PANEL_EVIDENCE_SCHEMA_VERSION = 1
HEALTH_PANEL_MAX_STEPS = 10
HEALTH_PANEL_GLOBAL_BATCH_SIZE = 16
HEALTH_PANEL_MICRO_BATCH_SIZE = 2
HEALTH_PANEL_NUM_WORKERS = 1
HEALTH_PANEL_SEED = 1
HEALTH_PANEL_MAX_UTILIZATION_PERCENT = 10
HEALTH_PANEL_MIN_FREE_MEMORY_MIB = 30_000
HEALTH_PANEL_ACTIVE_COMPUTE_PROCESSES_ALLOWED = True
EXPECTED_MDLM_CHECKPOINT_PATH = (
    PROJECT_ROOT / "outputs" / "paper_v1" / "checkpoints" / "50000.ckpt"
)
EXPECTED_MDLM_CHECKPOINT_SHA256 = (
    "8d00aa47b02f64bf39ff6b0b2e786f213587366fc2c3d29712a00f3f84108dd6"
)
EXPECTED_MDLM_CHECKPOINT_SIZE_BYTES = 1_396_998_679
_GIT_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class HealthPanelValidationError(ValueError):
    """Raised when retained artifacts do not prove the exact health panel."""


def health_run_name(gpu_count: int, training_variant: str, source_revision: str) -> str:
    """Return the deterministic run name for one registered health arm."""

    try:
        gpu_count = launch_train_pilot.validate_gpu_count(gpu_count)
        training_variant = launch_train_pilot.validate_training_variant(
            training_variant
        )
    except ValueError as error:
        raise HealthPanelValidationError(str(error)) from error
    source_revision = _git_revision(
        source_revision, label="health run-name source revision"
    )
    arm = {
        "udlm": "r",
        "schedule_uniform": "s",
        "udlm_categorical": "e",
    }[training_variant]
    return f"health-w{gpu_count}-{arm}-{source_revision}"


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HealthPanelValidationError(f"{label} must be an object")
    return value


def _exact(value: object, expected: object, *, label: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise HealthPanelValidationError(
            f"{label} must equal {expected!r}; observed {value!r}"
        )


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise HealthPanelValidationError(f"{label} must be a lowercase SHA-256")
    return value


def _git_revision(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _GIT_REVISION_PATTERN.fullmatch(value) is None:
        raise HealthPanelValidationError(
            f"{label} must be a 40-character lowercase Git revision"
        )
    return value


def _utc_timestamp(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise HealthPanelValidationError(f"{label} must be a nonempty UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise HealthPanelValidationError(f"{label} is not valid ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise HealthPanelValidationError(f"{label} must use UTC")
    return parsed


def _artifact_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise HealthPanelValidationError(f"{label} must be an absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise HealthPanelValidationError(f"{label} must be an absolute path")
    return Path(os.path.abspath(path))


def _validated_receipt_path(path: Path) -> Path:
    candidate = path if path.is_absolute() else REPOSITORY_ROOT / path
    candidate = Path(os.path.abspath(candidate))
    root = REPOSITORY_ROOT.resolve(strict=True)
    expected_parent = root / "output" / "udlm"
    try:
        relative = candidate.relative_to(expected_parent)
    except ValueError as error:
        raise HealthPanelValidationError(
            "health receipt must be inside output/udlm"
        ) from error
    if (
        len(relative.parts) != 2
        or relative.name != "pilot_exit_status.json"
        or launch_train_pilot.RUN_NAME_PATTERN.fullmatch(relative.parts[0]) is None
    ):
        raise HealthPanelValidationError(
            "health receipt must be exactly "
            "output/udlm/<run-name>/pilot_exit_status.json"
        )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise HealthPanelValidationError(
            f"health receipt is unavailable: {candidate}"
        ) from error
    if resolved != candidate:
        raise HealthPanelValidationError(
            "health receipt and its parents must not traverse symlinks"
        )
    return candidate


def _read_json(path: Path, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        artifact = pilot_evidence_writer.read_stable_regular_file(path, label=label)
        if artifact.payload is None:  # pragma: no cover - capture contract
            raise HealthPanelValidationError(f"{label} payload capture failed")
        value = pilot_evidence_writer.strict_json_loads(artifact.payload, label=label)
    except (OSError, ValueError) as error:
        if isinstance(error, HealthPanelValidationError):
            raise
        raise HealthPanelValidationError(f"cannot validate {label}: {error}") from error
    return dict(_mapping(value, label=label)), artifact.snapshot()


def _relative_path(path: Path, *, label: str) -> str:
    try:
        return str(path.relative_to(REPOSITORY_ROOT.resolve(strict=True)))
    except ValueError as error:
        raise HealthPanelValidationError(
            f"{label} must remain inside the repository"
        ) from error


def _expected_panel(
    *, source_revision: str, gpu_count: int, common_config_sha256: str
) -> tuple[dict[str, object], str]:
    try:
        verified_audit = launch_train_pilot.verify_pilot_empirical_uniform_mix_audit()
        panel, panel_sha256 = launch_train_pilot.build_matched_panel_spec(
            source_revision=source_revision,
            checkpoint=EXPECTED_MDLM_CHECKPOINT_PATH,
            checkpoint_sha256=EXPECTED_MDLM_CHECKPOINT_SHA256,
            gpu_count=gpu_count,
            max_steps=HEALTH_PANEL_MAX_STEPS,
            global_batch_size=HEALTH_PANEL_GLOBAL_BATCH_SIZE,
            micro_batch_size=HEALTH_PANEL_MICRO_BATCH_SIZE,
            num_workers=HEALTH_PANEL_NUM_WORKERS,
            seed=HEALTH_PANEL_SEED,
            exclude_special_tokens=False,
            max_utilization_percent=HEALTH_PANEL_MAX_UTILIZATION_PERCENT,
            min_free_memory_mib=HEALTH_PANEL_MIN_FREE_MEMORY_MIB,
            common_resolved_config_sha256=common_config_sha256,
        )
    except (OSError, ValueError) as error:
        raise HealthPanelValidationError(
            f"registered health contract cannot be reconstructed: {error}"
        ) from error
    common = _mapping(
        panel.get("common_training_contract"), label="expected common contract"
    )
    if common.get("empirical_uniform_mix_audit") != verified_audit:
        raise HealthPanelValidationError(
            "registered empirical-mixture audit binding is inconsistent"
        )
    return panel, panel_sha256


def _reconstructed_member_contracts(
    *, source_revision: str, gpu_count: int
) -> tuple[tuple[dict[str, Any], ...], str]:
    """Rebuild the only commands and resolved configs eligible for health.

    The common-config digest must originate from this independent reconstruction,
    never from a digest declared by a retained receipt or manifest.  Both helpers
    used here are CPU-only and execute before any launcher GPU probe.
    """

    members: list[dict[str, Any]] = []
    common_config_sha256: str | None = None
    for variant in launch_train_pilot.MATCHED_PANEL_VARIANT_ORDER:
        treatment = launch_train_pilot.TRAINING_VARIANTS[variant]
        run_dir = (
            REPOSITORY_ROOT
            / "output"
            / "udlm"
            / health_run_name(gpu_count, variant, source_revision)
        )
        try:
            command = launch_train_pilot.build_training_command(
                gpu_count=gpu_count,
                run_dir=run_dir,
                max_steps=HEALTH_PANEL_MAX_STEPS,
                global_batch_size=HEALTH_PANEL_GLOBAL_BATCH_SIZE,
                micro_batch_size=HEALTH_PANEL_MICRO_BATCH_SIZE,
                num_workers=HEALTH_PANEL_NUM_WORKERS,
                seed=HEALTH_PANEL_SEED,
                checkpoint=EXPECTED_MDLM_CHECKPOINT_PATH,
                checkpoint_sha256=EXPECTED_MDLM_CHECKPOINT_SHA256,
                exclude_special_tokens=False,
                training_variant=variant,
            )
            resolved_config, resolved_config_sha256 = (
                launch_train_pilot.compose_resolved_training_config(
                    config_name=str(treatment["config_name"]),
                    overrides=command[5:],
                    gpu_count=gpu_count,
                )
            )
            training_argv_sha256 = launch_train_pilot.training_argv_sha256(command)
            member_common_sha256 = launch_train_pilot.matched_panel_config_sha256(
                resolved_config
            )
        except Exception as error:
            raise HealthPanelValidationError(
                f"cannot reconstruct the exact {variant} health configuration: "
                f"{error}"
            ) from error
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(value, str) for value in command)
        ):
            raise HealthPanelValidationError(
                f"reconstructed {variant} training argv is invalid"
            )
        if not isinstance(resolved_config, dict):
            raise HealthPanelValidationError(
                f"reconstructed {variant} resolved config is not an object"
            )
        resolved_config_sha256 = _sha256(
            resolved_config_sha256,
            label=f"reconstructed {variant} resolved-config digest",
        )
        _exact(
            launch_train_pilot.canonical_json_sha256(resolved_config),
            resolved_config_sha256,
            label=f"reconstructed {variant} resolved-config content digest",
        )
        training_argv_sha256 = _sha256(
            training_argv_sha256,
            label=f"reconstructed {variant} training-argv digest",
        )
        member_common_sha256 = _sha256(
            member_common_sha256,
            label=f"reconstructed {variant} common resolved-config digest",
        )
        if common_config_sha256 is None:
            common_config_sha256 = member_common_sha256
        else:
            _exact(
                member_common_sha256,
                common_config_sha256,
                label=f"reconstructed {variant} common resolved-config digest",
            )
        members.append(
            {
                "training_variant": variant,
                "training_argv": command,
                "training_argv_sha256": training_argv_sha256,
                "resolved_training_config": resolved_config,
                "resolved_training_config_sha256": resolved_config_sha256,
                "common_resolved_config_sha256": member_common_sha256,
            }
        )
    if common_config_sha256 is None:  # pragma: no cover - fixed nonempty order
        raise HealthPanelValidationError("health variant order is empty")
    return tuple(members), common_config_sha256


def _structural_checkpoint(receipt: Mapping[str, Any]) -> dict[str, object]:
    expected = _mapping(receipt.get("expected_contract"), label="receipt contract")
    checkpoint_evidence = _mapping(
        receipt.get("final_checkpoint"), label="receipt checkpoint evidence"
    )
    snapshot = _mapping(
        checkpoint_evidence.get("artifact"), label="receipt checkpoint snapshot"
    )
    return {
        "checkpoint": {
            "path": expected.get("final_checkpoint_path"),
            "sha256": snapshot.get("sha256"),
            "size_bytes": snapshot.get("size_bytes"),
            "global_step": expected.get("max_steps"),
        }
    }


_GPU_STATE_KEYS = {
    "physical_index",
    "uuid",
    "name",
    "memory_used_mib",
    "memory_total_mib",
    "utilization_percent",
    "compute_mode",
    "compute_processes",
}
_GPU_PROCESS_KEYS = {"pid", "process_name", "used_memory_mib"}


def _gpu_state(value: object, *, label: str) -> dict[str, Any]:
    state = _mapping(value, label=label)
    if set(state) != _GPU_STATE_KEYS:
        raise HealthPanelValidationError(f"{label} fields are not exact")
    integer_fields = (
        "physical_index",
        "memory_used_mib",
        "memory_total_mib",
        "utilization_percent",
    )
    if any(type(state.get(field)) is not int for field in integer_fields):
        raise HealthPanelValidationError(f"{label} integer fields are invalid")
    if (
        state["physical_index"] < 0
        or state["memory_used_mib"] < 0
        or state["memory_total_mib"] <= 0
        or state["memory_used_mib"] > state["memory_total_mib"]
        or not 0 <= state["utilization_percent"] <= 100
    ):
        raise HealthPanelValidationError(f"{label} numeric fields are invalid")
    uuid = state.get("uuid")
    if not isinstance(uuid, str) or not uuid.startswith("GPU-") or "," in uuid:
        raise HealthPanelValidationError(f"{label} UUID is invalid")
    for field in ("name", "compute_mode"):
        if not isinstance(state.get(field), str) or not state[field].strip():
            raise HealthPanelValidationError(f"{label} {field} is invalid")
    processes = state.get("compute_processes")
    if not isinstance(processes, list):
        raise HealthPanelValidationError(f"{label} process evidence is not an array")
    normalized_processes: list[dict[str, Any]] = []
    for index, value in enumerate(processes):
        process_label = f"{label} process {index}"
        process = _mapping(value, label=process_label)
        if set(process) != _GPU_PROCESS_KEYS:
            raise HealthPanelValidationError(f"{process_label} fields are not exact")
        pid = process.get("pid")
        process_name = process.get("process_name")
        used_memory = process.get("used_memory_mib")
        if type(pid) is not int or pid <= 0:
            raise HealthPanelValidationError(f"{process_label} PID is invalid")
        if not isinstance(process_name, str) or not process_name:
            raise HealthPanelValidationError(f"{process_label} name is invalid")
        if used_memory is not None and (
            type(used_memory) is not int or used_memory < 0
        ):
            raise HealthPanelValidationError(f"{process_label} memory is invalid")
        normalized_processes.append(dict(process))
    return {**dict(state), "compute_processes": normalized_processes}


def _require_member_gpu_evidence(
    manifest: Mapping[str, Any], *, variant: str, gpu_count: int
) -> None:
    for key, expected in (
        ("gpu_selection_schema_version", 2),
        ("gpu_selection_method", "dynamic_idle_discovery"),
        ("gpu_inventory_scope", "all_nvidia_gpus"),
    ):
        _exact(manifest.get(key), expected, label=f"{variant} manifest {key}")
    inventory_time = _utc_timestamp(
        manifest.get("inventory_snapshot_completed_at_utc"),
        label=f"{variant} inventory completion timestamp",
    )
    final_probe_time = _utc_timestamp(
        manifest.get("final_uuid_probes_completed_at_utc"),
        label=f"{variant} final-probe completion timestamp",
    )
    manifest_time = _utc_timestamp(
        manifest.get("created_at"), label=f"{variant} manifest creation timestamp"
    )
    if not inventory_time <= final_probe_time <= manifest_time:
        raise HealthPanelValidationError(
            f"{variant} GPU-probe/manifest timestamps are out of order"
        )
    selected = manifest.get("cuda_visible_device_uuids")
    if (
        not isinstance(selected, list)
        or len(selected) != gpu_count
        or any(
            not isinstance(uuid, str) or not uuid.startswith("GPU-") or "," in uuid
            for uuid in selected
        )
        or len(set(selected)) != gpu_count
    ):
        raise HealthPanelValidationError(
            f"{variant} selected GPU UUID contract is invalid"
        )
    inventory_raw = manifest.get("gpu_inventory_at_selection")
    initial_raw = manifest.get("initially_selected_gpu_states")
    final_raw = manifest.get("gpu_states_at_final_uuid_probe")
    if not isinstance(inventory_raw, list) or not inventory_raw:
        raise HealthPanelValidationError(f"{variant} GPU inventory is invalid")
    if not isinstance(initial_raw, list) or len(initial_raw) != gpu_count:
        raise HealthPanelValidationError(
            f"{variant} initial GPU selection is incomplete"
        )
    if not isinstance(final_raw, list) or len(final_raw) != gpu_count:
        raise HealthPanelValidationError(f"{variant} final GPU probe is incomplete")
    inventory = [
        _gpu_state(value, label=f"{variant} inventory GPU {index}")
        for index, value in enumerate(inventory_raw)
    ]
    initial = [
        _gpu_state(value, label=f"{variant} initial GPU {index}")
        for index, value in enumerate(initial_raw)
    ]
    final = [
        _gpu_state(value, label=f"{variant} final GPU {index}")
        for index, value in enumerate(final_raw)
    ]
    inventory_uuids = [state["uuid"] for state in inventory]
    inventory_indices = [state["physical_index"] for state in inventory]
    if len(set(inventory_uuids)) != len(inventory_uuids) or len(
        set(inventory_indices)
    ) != len(inventory_indices):
        raise HealthPanelValidationError(
            f"{variant} GPU inventory identities are not unique"
        )
    inventory_by_uuid = {state["uuid"]: state for state in inventory}
    if any(uuid not in inventory_by_uuid for uuid in selected):
        raise HealthPanelValidationError(
            f"{variant} selected UUID is absent from inventory"
        )
    if [state["uuid"] for state in initial] != selected:
        raise HealthPanelValidationError(f"{variant} initial GPU UUID order is invalid")
    if [state["uuid"] for state in final] != selected:
        raise HealthPanelValidationError(f"{variant} final GPU UUID order is invalid")
    if initial != [inventory_by_uuid[uuid] for uuid in selected]:
        raise HealthPanelValidationError(
            f"{variant} initial GPU selection differs from inventory"
        )
    final_indices = [state["physical_index"] for state in final]
    if (
        len(set(final_indices)) != gpu_count
        or manifest.get("physical_gpu_indices") != final_indices
    ):
        raise HealthPanelValidationError(f"{variant} physical GPU mapping is invalid")
    if manifest.get("logical_cuda_devices") != list(range(gpu_count)):
        raise HealthPanelValidationError(f"{variant} logical GPU mapping is invalid")
    for state in (*initial, *final):
        if (
            state["utilization_percent"] >= HEALTH_PANEL_MAX_UTILIZATION_PERCENT
            or state["memory_total_mib"] - state["memory_used_mib"]
            < HEALTH_PANEL_MIN_FREE_MEMORY_MIB
            or state["compute_mode"].strip().lower() == "prohibited"
        ):
            raise HealthPanelValidationError(
                f"{variant} selected GPU violates the health safety policy"
            )


def _require_member_contract(
    *,
    receipt_path: Path,
    receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    panel: Mapping[str, Any],
    panel_sha256: str,
    source_revision: str,
    gpu_count: int,
    position: int,
    reconstructed: Mapping[str, Any],
) -> None:
    variant = launch_train_pilot.MATCHED_PANEL_VARIANT_ORDER[position]
    treatment = launch_train_pilot.TRAINING_VARIANTS[variant]
    run_dir = receipt_path.parent
    expected_run_name = health_run_name(gpu_count, variant, source_revision)
    expected_manifest_path = run_dir / "launch_manifest.json"
    expected_summary_path = run_dir / "training_summary.json"
    expected_checkpoint_path = (
        run_dir / "checkpoints" / f"{HEALTH_PANEL_MAX_STEPS}.ckpt"
    )
    accumulation_steps = launch_train_pilot.exact_accumulation_steps(
        HEALTH_PANEL_GLOBAL_BATCH_SIZE,
        HEALTH_PANEL_MICRO_BATCH_SIZE,
        gpu_count,
    )
    _exact(
        reconstructed.get("training_variant"),
        variant,
        label=f"reconstructed health member {position} variant",
    )

    expected = _mapping(receipt.get("expected_contract"), label="receipt contract")
    for key, value in (
        ("source_revision", source_revision),
        ("max_steps", HEALTH_PANEL_MAX_STEPS),
        ("world_size", gpu_count),
        ("launch_manifest_path", str(expected_manifest_path)),
        ("training_summary_path", str(expected_summary_path)),
        ("final_checkpoint_path", str(expected_checkpoint_path)),
        ("initialization_checkpoint_sha256", EXPECTED_MDLM_CHECKPOINT_SHA256),
    ):
        _exact(expected.get(key), value, label=f"{variant} receipt {key}")

    for key, value in (
        ("git_sha", source_revision),
        ("source_revision_before_final_gpu_probe", source_revision),
        ("run_name", expected_run_name),
        ("training_variant", variant),
        ("hydra_config_name", treatment["config_name"]),
        ("udlm_prior_variant", treatment["prior_variant"]),
        ("udlm_comparison_role", treatment["comparison_role"]),
        ("matched_panel_variant_position", position),
        ("matched_panel_spec_sha256", panel_sha256),
        ("user_requested_gpu_count", gpu_count),
        ("checkpoint", str(EXPECTED_MDLM_CHECKPOINT_PATH)),
        ("checkpoint_sha256", EXPECTED_MDLM_CHECKPOINT_SHA256),
        ("seed", HEALTH_PANEL_SEED),
        ("max_steps", HEALTH_PANEL_MAX_STEPS),
        ("global_batch_size", HEALTH_PANEL_GLOBAL_BATCH_SIZE),
        ("micro_batch_size_per_process", HEALTH_PANEL_MICRO_BATCH_SIZE),
        (
            "accumulate_grad_batches",
            accumulation_steps,
        ),
        ("effective_global_batch_size", HEALTH_PANEL_GLOBAL_BATCH_SIZE),
        ("exclude_special_tokens", False),
        ("dry_run", False),
        ("launch_manifest_path", str(expected_manifest_path)),
        ("training_summary_path", str(expected_summary_path)),
        ("pilot_exit_status_path", str(receipt_path)),
        ("expected_final_checkpoint_path", str(expected_checkpoint_path)),
    ):
        _exact(manifest.get(key), value, label=f"{variant} manifest {key}")
    if manifest.get("matched_panel_spec") != panel:
        raise HealthPanelValidationError(
            f"{variant} manifest does not retain the exact health panel"
        )
    _exact(
        run_dir.name,
        expected_run_name,
        label=f"{variant} run-directory name",
    )
    _exact(
        manifest.get("training_argv"),
        reconstructed.get("training_argv"),
        label=f"{variant} exact reconstructed training argv",
    )
    _exact(
        manifest.get("training_argv_sha256"),
        reconstructed.get("training_argv_sha256"),
        label=f"{variant} exact reconstructed training-argv digest",
    )

    manifest_safety = _mapping(
        manifest.get("gpu_safety_policy"), label=f"{variant} GPU safety policy"
    )
    expected_manifest_safety = {
        "max_utilization_percent": HEALTH_PANEL_MAX_UTILIZATION_PERCENT,
        "utilization_comparison": "strictly_less_than",
        "min_free_memory_mib": HEALTH_PANEL_MIN_FREE_MEMORY_MIB,
        "active_compute_processes_allowed": (
            HEALTH_PANEL_ACTIVE_COMPUTE_PROCESSES_ALLOWED
        ),
        "compute_mode_prohibited_allowed": False,
    }
    if dict(manifest_safety) != expected_manifest_safety:
        raise HealthPanelValidationError(
            f"{variant} manifest GPU safety policy is not exact"
        )
    _require_member_gpu_evidence(manifest, variant=variant, gpu_count=gpu_count)

    resolved = _mapping(
        manifest.get("resolved_training_config"),
        label=f"{variant} resolved training config",
    )
    trainer = _mapping(resolved.get("trainer"), label=f"{variant} trainer config")
    loader = _mapping(resolved.get("loader"), label=f"{variant} loader config")
    model = _mapping(resolved.get("model"), label=f"{variant} model config")
    training = _mapping(resolved.get("training"), label=f"{variant} training config")
    udlm = _mapping(training.get("udlm"), label=f"{variant} UDLM config")
    resolved_expectations = (
        (resolved, "seed", HEALTH_PANEL_SEED),
        (trainer, "devices", gpu_count),
        (trainer, "num_nodes", 1),
        (trainer, "max_steps", HEALTH_PANEL_MAX_STEPS),
        (
            trainer,
            "accumulate_grad_batches",
            accumulation_steps,
        ),
        (loader, "global_batch_size", HEALTH_PANEL_GLOBAL_BATCH_SIZE),
        (loader, "batch_size", HEALTH_PANEL_MICRO_BATCH_SIZE),
        (loader, "num_workers", HEALTH_PANEL_NUM_WORKERS),
        (model, "vocab_size", 1880),
        (udlm, "prior_variant", treatment["prior_variant"]),
        (udlm, "exclude_special_tokens", False),
        (
            udlm,
            "empirical_uniform_mix",
            launch_train_pilot.PILOT_EMPIRICAL_UNIFORM_MIX,
        ),
    )
    for section, key, value in resolved_expectations:
        _exact(section.get(key), value, label=f"{variant} resolved config {key}")
    _exact(
        manifest.get("resolved_training_config_sha256"),
        reconstructed.get("resolved_training_config_sha256"),
        label=f"{variant} exact reconstructed resolved-config digest",
    )
    _exact(
        dict(resolved),
        reconstructed.get("resolved_training_config"),
        label=f"{variant} exact reconstructed resolved training config",
    )
    common = _mapping(
        panel.get("common_training_contract"), label="health common contract"
    )
    try:
        common_config_sha256 = launch_train_pilot.matched_panel_config_sha256(
            dict(resolved)
        )
    except (KeyError, TypeError, ValueError) as error:
        raise HealthPanelValidationError(
            f"{variant} resolved config cannot be normalized: {error}"
        ) from error
    _exact(
        common_config_sha256,
        common.get("common_resolved_config_sha256"),
        label=f"{variant} common resolved-config digest",
    )
    _exact(
        common_config_sha256,
        reconstructed.get("common_resolved_config_sha256"),
        label=f"{variant} reconstructed common resolved-config digest",
    )

    startup = _mapping(summary.get("startup"), label=f"{variant} startup")
    _exact(startup.get("mode"), "warm_start", label=f"{variant} startup mode")
    warm_start = _mapping(
        startup.get("verified_mdlm_warm_start_report"),
        label=f"{variant} MDLM warm-start report",
    )
    for key, value in (
        ("source_path", str(EXPECTED_MDLM_CHECKPOINT_PATH)),
        ("source_resolved_path", str(EXPECTED_MDLM_CHECKPOINT_PATH)),
        ("source_sha256", EXPECTED_MDLM_CHECKPOINT_SHA256),
        ("expected_source_sha256", EXPECTED_MDLM_CHECKPOINT_SHA256),
        ("weights", "ema"),
        ("byte_identity_verified_before_and_after_load", True),
        ("source_size_bytes", EXPECTED_MDLM_CHECKPOINT_SIZE_BYTES),
    ):
        _exact(warm_start.get(key), value, label=f"{variant} warm start {key}")

    binding = _mapping(
        receipt.get("predecessor_receipt_binding"),
        label=f"{variant} predecessor binding",
    )
    _exact(
        binding.get("current_training_variant"),
        variant,
        label=f"{variant} predecessor current variant",
    )
    _exact(
        binding.get("current_variant_position"),
        position,
        label=f"{variant} predecessor current position",
    )
    if manifest.get("predecessor_receipt_binding") != binding:
        raise HealthPanelValidationError(
            f"{variant} manifest and receipt predecessor bindings differ"
        )
    if position == 0:
        _exact(
            binding.get("state"),
            "explicit_genesis_no_predecessor",
            label="R predecessor state",
        )
        if binding.get("receipt_artifact") is not None:
            raise HealthPanelValidationError("R genesis must not bind a receipt")
    else:
        _exact(
            binding.get("state"),
            "validated_successful_predecessor",
            label=f"{variant} predecessor state",
        )
        _exact(
            binding.get("expected_predecessor_training_variant"),
            launch_train_pilot.MATCHED_PANEL_VARIANT_ORDER[position - 1],
            label=f"{variant} expected predecessor variant",
        )
        _exact(
            binding.get("expected_predecessor_variant_position"),
            position - 1,
            label=f"{variant} expected predecessor position",
        )


def validate_health_panel(
    terminal_receipt_path: Path,
    *,
    expected_gpu_count: int | None = None,
    expected_source_revision: str | None = None,
) -> dict[str, object]:
    """Validate and normalize one exact ten-update R/S/E health panel.

    ``terminal_receipt_path`` must name E's successful schema-5 receipt at the
    producer-defined path.  Optional expectations cross-bind a caller's chosen
    world size and source revision.  The function performs no GPU queries.
    """

    receipt_path = _validated_receipt_path(Path(terminal_receipt_path))
    terminal_receipt, terminal_snapshot = _read_json(
        receipt_path, label="terminal E health receipt"
    )
    try:
        pilot_evidence_writer.validate_successful_training_receipt(
            terminal_receipt,
            receipt_path=receipt_path,
            structural=_structural_checkpoint(terminal_receipt),
        )
    except (OSError, ValueError) as error:
        raise HealthPanelValidationError(
            f"terminal E receipt or recursive R/S/E chain is invalid: {error}"
        ) from error
    retained_terminal_receipt, retained_terminal_snapshot = _read_json(
        receipt_path, label="retained terminal E health receipt"
    )
    if (
        retained_terminal_receipt != terminal_receipt
        or retained_terminal_snapshot != terminal_snapshot
    ):
        raise HealthPanelValidationError(
            "terminal E receipt changed during recursive validation"
        )

    terminal_expected = _mapping(
        terminal_receipt.get("expected_contract"), label="terminal receipt contract"
    )
    source_revision = _git_revision(
        terminal_expected.get("source_revision"), label="health source revision"
    )
    gpu_count = terminal_expected.get("world_size")
    try:
        gpu_count = launch_train_pilot.validate_gpu_count(gpu_count)
    except ValueError as error:
        raise HealthPanelValidationError(str(error)) from error
    if expected_gpu_count is not None:
        try:
            expected_gpu_count = launch_train_pilot.validate_gpu_count(
                expected_gpu_count
            )
        except ValueError as error:
            raise HealthPanelValidationError(str(error)) from error
        _exact(gpu_count, expected_gpu_count, label="health GPU count")
    if expected_source_revision is not None:
        expected_source_revision = _git_revision(
            expected_source_revision, label="expected health source revision"
        )
        _exact(
            source_revision,
            expected_source_revision,
            label="health source revision",
        )

    reconstructed_members, common_config_sha256 = _reconstructed_member_contracts(
        source_revision=source_revision,
        gpu_count=gpu_count,
    )

    terminal_manifest_path = receipt_path.parent / "launch_manifest.json"
    terminal_manifest, _terminal_manifest_snapshot = _read_json(
        terminal_manifest_path, label="terminal E launch manifest"
    )
    panel = dict(
        _mapping(
            terminal_manifest.get("matched_panel_spec"),
            label="terminal matched-panel specification",
        )
    )
    panel_sha256 = _sha256(
        terminal_manifest.get("matched_panel_spec_sha256"),
        label="terminal matched-panel digest",
    )
    if launch_train_pilot.canonical_json_sha256(panel) != panel_sha256:
        raise HealthPanelValidationError(
            "terminal matched-panel content differs from its digest"
        )
    expected_panel, expected_panel_sha256 = _expected_panel(
        source_revision=source_revision,
        gpu_count=gpu_count,
        common_config_sha256=common_config_sha256,
    )
    if panel != expected_panel or panel_sha256 != expected_panel_sha256:
        raise HealthPanelValidationError(
            "matched-panel specification is not the exact registered "
            "ten-update health contract"
        )

    descending_members: list[
        tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]
    ] = []
    current_path = receipt_path
    current_receipt = terminal_receipt
    current_receipt_snapshot = terminal_snapshot
    visited_paths: set[Path] = set()
    for position in range(2, -1, -1):
        if current_path in visited_paths:
            raise HealthPanelValidationError(
                "health predecessor chain contains a cycle"
            )
        visited_paths.add(current_path)
        current_path = _validated_receipt_path(current_path)
        receipt = current_receipt
        receipt_snapshot = current_receipt_snapshot
        manifest_path = current_path.parent / "launch_manifest.json"
        summary_path = current_path.parent / "training_summary.json"
        manifest, manifest_snapshot = _read_json(
            manifest_path, label=f"health manifest at position {position}"
        )
        summary, _summary_snapshot = _read_json(
            summary_path, label=f"health summary at position {position}"
        )
        _require_member_contract(
            receipt_path=current_path,
            receipt=receipt,
            manifest=manifest,
            summary=summary,
            panel=panel,
            panel_sha256=panel_sha256,
            source_revision=source_revision,
            gpu_count=gpu_count,
            position=position,
            reconstructed=reconstructed_members[position],
        )
        descending_members.append(
            (current_path, receipt, receipt_snapshot, manifest, manifest_snapshot)
        )
        if position == 0:
            continue
        binding = _mapping(
            receipt.get("predecessor_receipt_binding"),
            label="health predecessor binding",
        )
        claim = _mapping(
            binding.get("receipt_artifact"),
            label="health predecessor receipt artifact",
        )
        predecessor_path = _artifact_path(
            claim.get("path"), label="health predecessor receipt path"
        )
        predecessor_receipt, predecessor_snapshot = _read_json(
            predecessor_path, label="bound health predecessor receipt"
        )
        if dict(claim) != predecessor_snapshot:
            raise HealthPanelValidationError(
                "health predecessor receipt differs from its bound stable snapshot"
            )
        current_path = predecessor_path
        current_receipt = predecessor_receipt
        current_receipt_snapshot = predecessor_snapshot

    members = list(reversed(descending_members))
    receipt_members: list[dict[str, object]] = []
    checkpoint_members: list[dict[str, object]] = []
    for position, (path, receipt, snapshot, manifest, _manifest_snapshot) in enumerate(
        members
    ):
        variant = launch_train_pilot.MATCHED_PANEL_VARIANT_ORDER[position]
        checkpoint = _mapping(
            receipt.get("final_checkpoint"),
            label=f"{variant} final-checkpoint evidence",
        )
        checkpoint_snapshot = _mapping(
            checkpoint.get("artifact"), label=f"{variant} checkpoint snapshot"
        )
        checkpoint_path = _artifact_path(
            checkpoint_snapshot.get("path"), label=f"{variant} checkpoint path"
        )
        expected_checkpoint_path = (
            path.parent / "checkpoints" / f"{HEALTH_PANEL_MAX_STEPS}.ckpt"
        )
        _exact(
            checkpoint_path,
            expected_checkpoint_path,
            label=f"{variant} checkpoint path",
        )
        checkpoint_sha256 = _sha256(
            checkpoint_snapshot.get("sha256"), label=f"{variant} checkpoint digest"
        )
        checkpoint_size = checkpoint_snapshot.get("size_bytes")
        if type(checkpoint_size) is not int or checkpoint_size <= 0:
            raise HealthPanelValidationError(
                f"{variant} checkpoint size must be a positive integer"
            )
        receipt_members.append(
            {
                "position": position,
                "training_variant": variant,
                "run_name": path.parent.name,
                "relative_path": _relative_path(path, label=f"{variant} receipt"),
                "sha256": _sha256(
                    snapshot.get("sha256"), label=f"{variant} receipt digest"
                ),
                "schema_version": receipt.get("schema_version"),
                "recorded_at_utc": receipt.get("recorded_at_utc"),
            }
        )
        checkpoint_members.append(
            {
                "position": position,
                "training_variant": variant,
                "run_name": manifest.get("run_name"),
                "relative_path": _relative_path(
                    checkpoint_path, label=f"{variant} checkpoint"
                ),
                "sha256": checkpoint_sha256,
                "size_bytes": checkpoint_size,
                "global_step": HEALTH_PANEL_MAX_STEPS,
            }
        )

    return {
        "schema_version": HEALTH_PANEL_EVIDENCE_SCHEMA_VERSION,
        "status": "validated",
        "claim_scope": "training_health_and_provenance_only",
        "health_source_revision": source_revision,
        "gpu_count": gpu_count,
        "matched_panel_spec_sha256": panel_sha256,
        "terminal_receipt": {
            "root": "repository",
            "relative_path": _relative_path(
                receipt_path, label="terminal E health receipt"
            ),
            "sha256": terminal_snapshot["sha256"],
            "size_bytes": terminal_snapshot["size_bytes"],
            "schema_version": terminal_receipt.get("schema_version"),
            "canonical_sha256": pilot_evidence_writer.canonical_json_sha256(
                terminal_receipt
            ),
            "training_variant": "udlm_categorical",
            "position": 2,
        },
        "receipt_members": receipt_members,
        "checkpoint_members": checkpoint_members,
        "eligibility": {
            "generation": False,
            "ranking": False,
            "superiority": False,
            "candidate_lock": False,
            "screen_authorization": True,
        },
    }

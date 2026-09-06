"""Launch one arm from the frozen UDLM optimization-screen registry.

The registry, rather than command-line overrides, fixes the GPU count, seed,
training budget, resolved configuration, MDLM-EMA warm start, and output
directory.  ``--dry-run`` performs the complete CPU/Git preflight but never
queries NVIDIA devices, reserves output, acquires the training lock, or touches
tmux.  A real launch reuses the hardened pilot launch primitives to select and
re-probe genuinely idle GPU UUIDs immediately before detached execution.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.udlm import launch_train_pilot as pilot  # noqa: E402
from scripts.udlm import verify_optimization_screen as verifier  # noqa: E402


SCREEN_TRAINING_VARIANT = "udlm_categorical"
SCREEN_PURPOSE = "registered UDLM optimization screen"
SCREEN_LOCK_PURPOSE = "enforce_one_registered_optimization_screen_job_at_a_time"
SCREEN_LOG_PREFIX = "optimization_screen_"
_ARM_TO_STAGE = {
    "E-L0": "scheduler",
    "E-L1": "scheduler",
    "E-A0": "conditioning",
    "E-A1": "conditioning",
}


@dataclass(frozen=True)
class ScreenLaunchPlan:
    """All immutable, CPU-resolved inputs for one registered arm."""

    registry: verifier.ValidatedRegistry
    stage_id: str
    arm: Mapping[str, Any]
    config_entry: Mapping[str, Any]
    scheduler_dependency: Mapping[str, Any] | None
    source_revision: str
    gpu_count: int
    checkpoint: Path
    checkpoint_sha256: str
    run_dir: Path
    log_path: Path
    session_name: str
    command: list[str]
    resolved_config: dict[str, object]
    resolved_config_sha256: str
    argv_sha256: str

    @property
    def arm_id(self) -> str:
        return str(self.arm["arm_id"])

    @property
    def attempt_id(self) -> str:
        return str(self.arm["attempt_id"])

    @property
    def max_steps(self) -> int:
        return verifier.EXPECTED_UPDATES[self.stage_id]


def _json_compatible(value: object) -> object:
    """Convert verifier Decimal leaves to the JSON types Hydra will produce."""

    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_json_compatible(child) for child in value]
    return value


def _hydra_literal(value: object) -> str:
    if isinstance(value, Decimal):
        return str(value)
    if value is None:
        return "null"
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) in {int, float}:
        return json.dumps(value, allow_nan=False)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return json.dumps(
            _json_compatible(value), separators=(",", ":"), allow_nan=False
        )
    raise ValueError(f"unsupported resolved-config leaf type: {type(value).__name__}")


def _resolved_config_leaf_overrides(
    value: Mapping[str, Any], *, prefix: tuple[str, ...] = ()
) -> list[str]:
    """Encode every resolved leaf, avoiding any unregistered implicit default."""

    overrides: list[str] = []
    for key in sorted(value):
        if not isinstance(key, str) or not key or "." in key:
            raise ValueError("resolved configuration contains an invalid key")
        child = value[key]
        path = (*prefix, key)
        if isinstance(child, Mapping):
            if not child:
                raise ValueError("empty resolved-config mappings are unsupported")
            overrides.extend(_resolved_config_leaf_overrides(child, prefix=path))
        else:
            overrides.append(f"{'.'.join(path)}={_hydra_literal(child)}")
    return overrides


def build_registered_training_command(
    *, resolved_config: Mapping[str, Any], run_dir: Path, gpu_count: int
) -> tuple[list[str], dict[str, object], str]:
    """Build argv that composes to exactly the frozen resolved-config bytes."""

    pilot.validate_gpu_count(gpu_count)
    run_dir = run_dir.resolve(strict=False)
    if run_dir == REPOSITORY_ROOT or REPOSITORY_ROOT not in run_dir.parents:
        raise ValueError("registered run directory must be inside the repository")
    target = _json_compatible(resolved_config)
    if not isinstance(target, dict):  # pragma: no cover - caller supplies a mapping
        raise ValueError("registered resolved configuration must be an object")
    command = [
        str(pilot._python_executable()),
        "-u",
        str(REPOSITORY_ROOT / "scripts" / "train.py"),
        "--config-name",
        "udlm_categorical",
        *_resolved_config_leaf_overrides(resolved_config),
        f"hydra.run.dir={run_dir / 'hydra'}",
    ]
    composed, digest = pilot.compose_resolved_training_config(
        config_name="udlm_categorical",
        overrides=command[5:],
        gpu_count=gpu_count,
    )
    if composed != target:
        raise RuntimeError(
            "screen training argv does not reproduce the registered resolved config"
        )
    return command, composed, digest


def compose_screen_training_bundle(
    *,
    stage_id: str,
    arm_id: str,
    scheduler_arm_id: str,
    gpu_count: int,
    run_dir: Path,
    checkpoint: Path,
    checkpoint_sha256: str,
    global_batch_size: int,
    micro_batch_size: int,
    num_workers: int,
    exclude_special_tokens: bool,
) -> tuple[list[str], dict[str, object], str]:
    """Compose one prospective screen config for registry preparation/launch.

    This helper does not read GPUs or write artifacts.  A two-phase registry
    preparation tool can call it for the two scheduler arms and four
    scheduler-contingent conditioning configs, freeze their returned JSON and
    canonical digests, and later let this launcher reproduce them exactly.
    """

    if _ARM_TO_STAGE.get(arm_id) != stage_id:
        raise ValueError(f"arm {arm_id!r} does not belong to stage {stage_id!r}")
    if scheduler_arm_id not in verifier.EXPECTED_ARM_ORDER["scheduler"]:
        raise ValueError("scheduler arm must be E-L0 or E-L1")
    if stage_id == "scheduler" and scheduler_arm_id != arm_id:
        raise ValueError("scheduler screen arm must use its own scheduler")
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("screen num_workers must be a nonnegative integer")
    if exclude_special_tokens is not False:
        raise ValueError(
            "optimization screens require the registered full-vocabulary E process"
        )
    gpu_count = pilot.validate_gpu_count(gpu_count)
    accumulation = pilot.exact_accumulation_steps(
        global_batch_size, micro_batch_size, gpu_count
    )
    max_steps = verifier.EXPECTED_UPDATES[stage_id]
    command = pilot.build_training_command(
        gpu_count=gpu_count,
        run_dir=run_dir,
        max_steps=max_steps,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        num_workers=num_workers,
        seed=verifier.EXPECTED_TRAINING_SEED,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        exclude_special_tokens=exclude_special_tokens,
        training_variant=SCREEN_TRAINING_VARIANT,
    )
    is_l1 = scheduler_arm_id == "E-L1"
    is_film = arm_id == "E-A1"
    command.extend(
        [
            f"trainer.accumulate_grad_batches={accumulation}",
            "training.reseed_after_model_initialization="
            + ("true" if stage_id == "conditioning" else "false"),
            "training.udlm.conditioning_variant="
            + ("film_adaln" if is_film else "additive"),
            "training.udlm.zero_init_conditioning=" + ("false" if is_film else "true"),
            "optim.scheduler.name="
            + (
                "half_cosine_with_linear_warmup_and_floor"
                if is_l1
                else "constant_with_linear_warmup"
            ),
            f"optim.scheduler.warmup_updates={50 if is_l1 else 2500}",
            f"optim.scheduler.horizon_updates={1000 if is_l1 else 'null'}",
            f"optim.scheduler.decay_floor_lr={0.000003 if is_l1 else 'null'}",
        ]
    )
    resolved, digest = pilot.compose_resolved_training_config(
        config_name="udlm_categorical",
        overrides=command[5:],
        gpu_count=gpu_count,
    )
    return command, resolved, digest


def _repository_json_reference(path: Path, *, expected_schema: int) -> dict[str, Any]:
    path = path.resolve(strict=True)
    relative = verifier._repository_relative(path, "scheduler dependency")
    payload = verifier._stable_input_bytes(path)
    parsed = verifier.strict_json_loads(payload, label="scheduler dependency")
    if (
        not isinstance(parsed, Mapping)
        or parsed.get("schema_version") != expected_schema
    ):
        raise verifier.ScreenValidationError(
            "scheduler dependency has an unsupported schema"
        )
    return {
        "root": "repository",
        "relative_path": relative,
        "sha256": verifier.hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "schema_version": expected_schema,
        "canonical_sha256": verifier.canonical_json_sha256(parsed),
    }


def load_registry(
    path: Path, *, expected_raw_sha256: str, expected_canonical_sha256: str
) -> verifier.ValidatedRegistry:
    """Load the exact locally pinned registry through the strict verifier."""

    path = path.resolve(strict=True)
    relative = verifier._repository_relative(path, "optimization-screen registry")
    return verifier.load_validated_registry(
        verifier._stable_input_bytes(path),
        relative_path=relative,
        expected_raw_sha256=expected_raw_sha256,
        expected_canonical_sha256=expected_canonical_sha256,
        loader=verifier.local_blob_loader,
        git_blob_loader=verifier.git_blob_loader,
        git_ancestor_checker=verifier.git_ancestor_checker,
        git_sole_parent_checker=verifier.git_sole_parent_checker,
        git_tree_paths_loader=verifier.git_tree_paths_loader,
        git_pushed_checker=verifier.git_pushed_checker,
        git_diff_checker=verifier.git_diff_checker,
    )


def _validate_scheduler_publication_revision(
    registry: verifier.ValidatedRegistry, run_revision: str
) -> None:
    """Require the scheduler-run revision to add only the frozen registry."""

    source_revision = registry.data["source"]["revision"]
    if (
        run_revision == source_revision
        or not registry.git_ancestor_checker(source_revision, run_revision)
        or not registry.git_pushed_checker(run_revision)
    ):
        raise verifier.ScreenValidationError(
            "scheduler launch revision is not a pushed registry-publication descendant"
        )
    try:
        registry.git_blob_loader(source_revision, registry.relative_path)
    except Exception:
        pass
    else:
        raise verifier.ScreenValidationError(
            "registry path already existed at the pre-registry source revision"
        )
    if not registry.git_diff_checker(
        source_revision,
        run_revision,
        frozenset({registry.relative_path.as_posix()}),
    ):
        raise verifier.ScreenValidationError(
            "scheduler launch revision changes unregistered source bytes"
        )
    verifier._verify_revision_contract(registry, run_revision)


def _conditioning_dependency(
    registry: verifier.ValidatedRegistry,
    *,
    authorization_revision: str | None,
    scheduler_evidence: Path | None,
    scheduler_selection: Path | None,
) -> tuple[str, Mapping[str, Any]]:
    if (
        authorization_revision is None
        or scheduler_evidence is None
        or scheduler_selection is None
    ):
        raise ValueError(
            "conditioning launch requires authorization revision, scheduler evidence, "
            "and scheduler selection"
        )
    declared = {
        "authorization_revision": authorization_revision,
        "scheduler_evidence": _repository_json_reference(
            scheduler_evidence,
            expected_schema=verifier.EVIDENCE_SCHEMA_VERSION,
        ),
        "scheduler_selection": _repository_json_reference(
            scheduler_selection,
            expected_schema=verifier.SELECTION_SCHEMA_VERSION,
        ),
    }
    selected, validated_revision, normalized = (
        verifier._load_declared_scheduler_dependency(
            declared,
            loader=verifier.local_blob_loader,
            registry=registry,
        )
    )
    if selected not in verifier.EXPECTED_ARM_ORDER["scheduler"]:
        raise verifier.ScreenValidationError(
            "scheduler dependency selected no valid arm"
        )
    allowed_changes = frozenset(
        {
            registry.relative_path.as_posix(),
            normalized["scheduler_evidence"]["relative_path"],
            normalized["scheduler_selection"]["relative_path"],
        }
    )
    if not registry.git_diff_checker(
        registry.data["source"]["revision"],
        validated_revision,
        allowed_changes,
    ):
        raise verifier.ScreenValidationError(
            "conditioning authorization changed unregistered source bytes"
        )
    return validated_revision, normalized


def _registered_output_path(relative_path: str) -> Path:
    candidate = verifier._root_path("repository", PurePosixPath(relative_path))
    resolved = candidate.resolve(strict=False)
    repository = REPOSITORY_ROOT.resolve(strict=True)
    if resolved == repository or repository not in resolved.parents:
        raise verifier.ScreenValidationError(
            "registered output directory escapes the repository"
        )
    return candidate


def build_launch_plan(
    *,
    registry: verifier.ValidatedRegistry,
    stage_id: str,
    arm_id: str,
    current_revision: str,
    authorization_revision: str | None = None,
    scheduler_evidence: Path | None = None,
    scheduler_selection: Path | None = None,
) -> ScreenLaunchPlan:
    """Resolve one arm and prove its publication/dependency authorization."""

    if stage_id not in verifier.EXPECTED_STAGE_ORDER:
        raise ValueError("stage must be scheduler or conditioning")
    if _ARM_TO_STAGE.get(arm_id) != stage_id:
        raise ValueError(f"arm {arm_id!r} does not belong to stage {stage_id!r}")
    stage = verifier._stage(registry, stage_id)
    arm = verifier._arm(stage, arm_id)
    if stage_id == "scheduler":
        if any(
            value is not None
            for value in (
                authorization_revision,
                scheduler_evidence,
                scheduler_selection,
            )
        ):
            raise ValueError("scheduler launch must not declare a scheduler dependency")
        _validate_scheduler_publication_revision(registry, current_revision)
        scheduler_arm_id = None
        dependency = None
        source_revision = current_revision
    else:
        source_revision, dependency = _conditioning_dependency(
            registry,
            authorization_revision=authorization_revision,
            scheduler_evidence=scheduler_evidence,
            scheduler_selection=scheduler_selection,
        )
        if source_revision != current_revision:
            raise verifier.ScreenValidationError(
                "conditioning launch HEAD must equal its scheduler authorization revision"
            )
        scheduler_arm_id = dependency["selected_scheduler_arm_id"]
    config_entry = verifier._registered_config(arm, scheduler_arm_id=scheduler_arm_id)
    gpu_count = pilot.validate_gpu_count(registry.data["common_training"]["gpu_count"])
    run_dir = _registered_output_path(config_entry["output_directory"])
    checkpoint_ref = registry.data["common_training"]["initialization"]["checkpoint"]
    checkpoint = verifier._root_path(
        checkpoint_ref["root"], PurePosixPath(checkpoint_ref["relative_path"])
    ).resolve(strict=True)
    configured_checkpoint = Path(
        str(config_entry["parsed_config"]["training"]["init_from_mdlm_checkpoint"])
    ).resolve(strict=True)
    if configured_checkpoint != checkpoint:
        raise verifier.ScreenValidationError(
            "registered config warm-start path is not the bound MDLM checkpoint"
        )
    configured_callback = Path(
        str(config_entry["parsed_config"]["callback"]["dirpath"])
    ).resolve(strict=False)
    if configured_callback != (run_dir / "checkpoints").resolve(strict=False):
        raise verifier.ScreenValidationError(
            "registered config checkpoint output is not its registered directory"
        )
    target_config = config_entry["parsed_config"]
    try:
        num_workers = target_config["loader"]["num_workers"]
        exclude_special_tokens = target_config["training"]["udlm"][
            "exclude_special_tokens"
        ]
    except (KeyError, TypeError) as error:
        raise verifier.ScreenValidationError(
            "registered config omits screen execution controls"
        ) from error
    command, resolved, digest = compose_screen_training_bundle(
        stage_id=stage_id,
        arm_id=arm_id,
        scheduler_arm_id=(arm_id if scheduler_arm_id is None else scheduler_arm_id),
        gpu_count=gpu_count,
        run_dir=run_dir,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_ref["sha256"],
        global_batch_size=registry.data["common_training"]["global_batch_size"],
        micro_batch_size=registry.data["common_training"][
            "micro_batch_size_per_process"
        ],
        num_workers=num_workers,
        exclude_special_tokens=exclude_special_tokens,
    )
    if (
        resolved != _json_compatible(target_config)
        or digest != config_entry["config"]["canonical_sha256"]
    ):
        raise verifier.ScreenValidationError(
            "registered config differs from the reviewed screen training bundle"
        )
    attempt_id = str(arm["attempt_id"])
    if pilot.RUN_NAME_PATTERN.fullmatch(attempt_id) is None:
        raise verifier.ScreenValidationError("registered attempt ID is not launch-safe")
    session_name = f"genmol_screen_{attempt_id}"
    if pilot.RUN_NAME_PATTERN.fullmatch(session_name) is None:
        raise verifier.ScreenValidationError("registered tmux session name is invalid")
    log_path = (
        REPOSITORY_ROOT / "output" / "logs" / (f"{SCREEN_LOG_PREFIX}{attempt_id}.log")
    )
    return ScreenLaunchPlan(
        registry=registry,
        stage_id=stage_id,
        arm=arm,
        config_entry=config_entry,
        scheduler_dependency=dependency,
        source_revision=source_revision,
        gpu_count=gpu_count,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_ref["sha256"],
        run_dir=run_dir,
        log_path=log_path,
        session_name=session_name,
        command=command,
        resolved_config=resolved,
        resolved_config_sha256=digest,
        argv_sha256=pilot.training_argv_sha256(command),
    )


def dry_run_preview(plan: ScreenLaunchPlan) -> dict[str, Any]:
    """Return a JSON-compatible, explicitly non-launching preflight record."""

    return {
        "schema_version": 1,
        "status": "dry_run_preflight_completed_no_launch",
        "project_launch_artifact_mutation_performed": False,
        "training_job_lock_acquired": False,
        "gpu_probe_performed": False,
        "tmux_operation_performed": False,
        "source_revision": plan.source_revision,
        "registry": plan.registry.reference,
        "stage_id": plan.stage_id,
        "arm_id": plan.arm_id,
        "attempt_id": plan.attempt_id,
        "scheduler_authorization": plan.scheduler_dependency,
        "frozen_gpu_count": plan.gpu_count,
        "seed": verifier.EXPECTED_TRAINING_SEED,
        "optimizer_updates": plan.max_steps,
        "initialization_checkpoint": str(plan.checkpoint),
        "initialization_checkpoint_sha256": plan.checkpoint_sha256,
        "predicted_run_directory": str(plan.run_dir),
        "predicted_log_path": str(plan.log_path),
        "predicted_tmux_session": plan.session_name,
        "training_argv": plan.command,
        "training_argv_sha256": plan.argv_sha256,
        "resolved_training_config": plan.resolved_config,
        "resolved_training_config_sha256": plan.resolved_config_sha256,
        "optimization_screen": verifier._screen_binding(
            plan.registry,
            stage_id=plan.stage_id,
            arm=plan.arm,
            scheduler_dependency=plan.scheduler_dependency,
        ),
    }


def _launch_locked_screen(
    plan: ScreenLaunchPlan,
    *,
    max_utilization_percent: int,
    min_free_memory_mib: int,
    lock_path: Path,
    lock_record: dict[str, object],
    lock_sha256: str,
) -> tuple[bytes, str, Path]:
    """Probe, publish, and hand one registered arm to detached tmux."""

    pilot.validate_safety_thresholds(max_utilization_percent, min_free_memory_mib)
    gpu_inventory = pilot.probe_all_gpus()
    inventory_completed = datetime.now(timezone.utc).isoformat()
    initially_selected = pilot.select_idle_gpus(
        gpu_inventory,
        gpu_count=plan.gpu_count,
        max_utilization_percent=max_utilization_percent,
        min_free_memory_mib=min_free_memory_mib,
    )
    manifest_path = plan.run_dir / "launch_manifest.json"
    runtime_path = plan.run_dir / "runtime_config.json"
    summary_path = plan.run_dir / "training_summary.json"
    receipt_path = pilot.validate_pilot_exit_receipt_path(
        plan.run_dir / "pilot_exit_status.json"
    )
    final_checkpoint = plan.run_dir / "checkpoints" / f"{plan.max_steps}.ckpt"
    source_before_final_probe = pilot.require_pushed_commit()
    if source_before_final_probe != plan.source_revision:
        raise RuntimeError(
            "source revision changed before the screen's final GPU probe"
        )
    gpu_states = pilot.reprobe_selected_gpus(
        initially_selected,
        max_utilization_percent=max_utilization_percent,
        min_free_memory_mib=min_free_memory_mib,
    )
    final_probe_completed = datetime.now(timezone.utc).isoformat()
    selected_uuids = [state.uuid for state in gpu_states]
    selected_uuids_json = json.dumps(selected_uuids, separators=(",", ":"))

    plan.run_dir.mkdir(parents=True)
    pilot.reserve_log_path(plan.log_path)
    manifest = {
        "launch_manifest_schema_version": pilot.LAUNCH_MANIFEST_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": SCREEN_PURPOSE,
        "optimization_screen": verifier._screen_binding(
            plan.registry,
            stage_id=plan.stage_id,
            arm=plan.arm,
            scheduler_dependency=plan.scheduler_dependency,
        ),
        "gpu_selection_schema_version": 2,
        "git_sha": plan.source_revision,
        "source_revision_before_final_gpu_probe": source_before_final_probe,
        "run_name": plan.attempt_id,
        "training_variant": SCREEN_TRAINING_VARIANT,
        "hydra_config_name": "udlm_categorical",
        "udlm_prior_variant": "empirical_frequency",
        "udlm_comparison_role": "registered_engineering_screen_arm",
        "single_training_job_lock": {
            "path": str(lock_path),
            "sha256": lock_sha256,
            "record": lock_record,
            "acquired_before_any_gpu_probe": True,
            "stale_lock_policy": "fail_closed_and_require_manual_review",
            "release_owner": "pilot_exit_receipt_writer_after_publication",
        },
        "tmux_session": plan.session_name,
        "user_requested_gpu_count": plan.gpu_count,
        "gpu_selection_method": "dynamic_idle_discovery",
        "gpu_inventory_scope": "all_nvidia_gpus",
        "inventory_snapshot_completed_at_utc": inventory_completed,
        "gpu_inventory_at_selection": [asdict(state) for state in gpu_inventory],
        "initially_selected_gpu_states": [
            asdict(state) for state in initially_selected
        ],
        "logical_cuda_devices": list(range(plan.gpu_count)),
        "physical_gpu_indices": [state.physical_index for state in gpu_states],
        "cuda_visible_device_uuids": selected_uuids,
        "final_uuid_probes_completed_at_utc": final_probe_completed,
        "gpu_states_at_final_uuid_probe": [asdict(state) for state in gpu_states],
        "gpu_safety_policy": {
            "max_utilization_percent": max_utilization_percent,
            "utilization_comparison": "strictly_less_than",
            "min_free_memory_mib": min_free_memory_mib,
            "active_compute_processes_allowed": False,
            "compute_mode_prohibited_allowed": False,
        },
        "training_argv": plan.command,
        "training_argv_sha256": plan.argv_sha256,
        "resolved_training_config": plan.resolved_config,
        "resolved_training_config_sha256": plan.resolved_config_sha256,
        "runtime_config_path": str(runtime_path),
        "training_summary_path": str(summary_path),
        "training_summary_schema_version": pilot.TRAINING_SUMMARY_SCHEMA_VERSION,
        "pilot_exit_status_path": str(receipt_path),
        "pilot_exit_status_schema_version": pilot.PILOT_EXIT_STATUS_SCHEMA_VERSION,
        "expected_final_checkpoint_path": str(final_checkpoint),
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
        "log_path": str(plan.log_path),
        "log_reserved_exclusively_before_manifest": True,
        "checkpoint": str(plan.checkpoint),
        "checkpoint_sha256": plan.checkpoint_sha256,
        "seed": verifier.EXPECTED_TRAINING_SEED,
        "max_steps": plan.max_steps,
        "global_batch_size": plan.registry.data["common_training"]["global_batch_size"],
        "micro_batch_size_per_process": plan.registry.data["common_training"][
            "micro_batch_size_per_process"
        ],
        "accumulate_grad_batches": plan.registry.data["common_training"][
            "accumulate_grad_batches"
        ],
        "effective_global_batch_size": plan.registry.data["common_training"][
            "effective_global_batch_size"
        ],
        "exclude_special_tokens": bool(
            plan.resolved_config["training"]["udlm"]["exclude_special_tokens"]
        ),
        "dry_run": False,
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    manifest_sha256 = pilot._atomic_publish_bytes_exclusive(
        manifest_path, manifest_bytes, label="optimization-screen launch manifest"
    )
    environment_command, _environment = pilot.build_child_environment_command(
        command=plan.command,
        source_revision=plan.source_revision,
        resolved_config_sha256=plan.resolved_config_sha256,
        runtime_config_path=runtime_path,
        training_summary_path=summary_path,
        final_checkpoint_path=final_checkpoint,
        launch_manifest_path=manifest_path,
        launch_manifest_sha256=manifest_sha256,
        expected_max_steps=plan.max_steps,
        expected_world_size=plan.gpu_count,
        visible_uuids=",".join(selected_uuids),
        seed=verifier.EXPECTED_TRAINING_SEED,
    )
    shell_command = pilot.build_tmux_shell_command(
        environment_command,
        log_path=plan.log_path,
        training_summary_path=summary_path,
        exit_receipt_path=receipt_path,
        expected_source_revision=plan.source_revision,
        expected_config_sha256=plan.resolved_config_sha256,
        expected_argv_sha256=plan.argv_sha256,
        expected_summary_schema_version=pilot.TRAINING_SUMMARY_SCHEMA_VERSION,
        expected_max_steps=plan.max_steps,
        expected_world_size=plan.gpu_count,
        expected_final_checkpoint_path=final_checkpoint,
        expected_launch_manifest_path=manifest_path,
        expected_launch_manifest_sha256=manifest_sha256,
        expected_selected_gpu_uuids_json=selected_uuids_json,
        expected_training_job_lock_path=lock_path,
        expected_training_job_lock_sha256=lock_sha256,
        expected_initialization_checkpoint_sha256=plan.checkpoint_sha256,
    )
    subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            plan.session_name,
            "-c",
            str(REPOSITORY_ROOT),
            "bash",
            "-lc",
            shell_command,
        ],
        check=True,
    )
    return manifest_bytes, manifest_sha256, plan.log_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--expected-registry-sha256", required=True)
    parser.add_argument("--expected-registry-canonical-sha256", required=True)
    parser.add_argument("--stage", choices=verifier.EXPECTED_STAGE_ORDER, required=True)
    parser.add_argument("--arm", choices=tuple(_ARM_TO_STAGE), required=True)
    parser.add_argument("--authorization-revision")
    parser.add_argument("--scheduler-evidence", type=Path)
    parser.add_argument("--scheduler-selection", type=Path)
    parser.add_argument(
        "--max-utilization-percent",
        type=int,
        default=pilot.MAX_SAFE_UTILIZATION_PERCENT,
    )
    parser.add_argument(
        "--min-free-memory-mib",
        type=int,
        default=pilot.MIN_SAFE_FREE_MEMORY_MIB,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    pilot.validate_safety_thresholds(
        args.max_utilization_percent, args.min_free_memory_mib
    )
    registry = load_registry(
        args.registry,
        expected_raw_sha256=args.expected_registry_sha256,
        expected_canonical_sha256=args.expected_registry_canonical_sha256,
    )
    current_revision = pilot.require_pushed_commit()
    plan = build_launch_plan(
        registry=registry,
        stage_id=args.stage,
        arm_id=args.arm,
        current_revision=current_revision,
        authorization_revision=args.authorization_revision,
        scheduler_evidence=args.scheduler_evidence,
        scheduler_selection=args.scheduler_selection,
    )
    if os.path.lexists(plan.run_dir) or os.path.lexists(plan.log_path):
        raise FileExistsError(
            "refusing to overwrite an existing optimization-screen attempt: "
            f"{plan.run_dir} or {plan.log_path}"
        )
    if args.dry_run:
        print(
            json.dumps(dry_run_preview(plan), indent=2, sort_keys=True, allow_nan=False)
        )
        return 0
    if pilot.tmux_session_exists(plan.session_name):
        raise RuntimeError(f"tmux session already exists: {plan.session_name}")
    lock_path, lock_record, lock_sha256 = pilot.acquire_training_job_lock(
        source_revision=plan.source_revision,
        run_name=plan.attempt_id,
        training_variant=SCREEN_TRAINING_VARIANT,
        purpose=SCREEN_LOCK_PURPOSE,
    )
    try:
        manifest_bytes, manifest_sha256, log_path = _launch_locked_screen(
            plan,
            max_utilization_percent=args.max_utilization_percent,
            min_free_memory_mib=args.min_free_memory_mib,
            lock_path=lock_path,
            lock_record=lock_record,
            lock_sha256=lock_sha256,
        )
    except BaseException as launch_error:
        try:
            handoff_may_have_succeeded = pilot.tmux_session_exists(plan.session_name)
        except BaseException as verification_error:
            raise RuntimeError(
                "screen launch failed with an indeterminate tmux handoff; the exact "
                "training-job lock was retained fail-closed for manual review: "
                f"{type(verification_error).__name__}: {verification_error}"
            ) from launch_error
        if handoff_may_have_succeeded:
            raise RuntimeError(
                "screen launch raised after tmux may have accepted the detached job; "
                "the exact training-job lock was retained fail-closed"
            ) from launch_error
        try:
            pilot.release_exact_training_job_lock(
                lock_path, expected_sha256=lock_sha256
            )
        except Exception as release_error:
            raise RuntimeError(
                "screen launch failed before tmux handoff and its exact training-job "
                f"lock could not be released: {type(release_error).__name__}: "
                f"{release_error}"
            ) from launch_error
        raise
    print(manifest_bytes.decode("utf-8"), end="")
    print(f"launch manifest SHA-256: {manifest_sha256}")
    print(f"launched tmux session {plan.session_name}; log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

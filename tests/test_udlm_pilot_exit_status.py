import copy
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.udlm import launch_train_pilot as launcher
from scripts.udlm import write_pilot_exit_status as receipt_writer


def _canonical_sha256(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


RESOLVED_TRAINING_CONFIG = {
    "data": "safe",
    "seed": 7,
    "training": {"ema": 0.9999},
    "loader": {"global_batch_size": 8, "batch_size": 2},
    "optim": {
        "weight_decay": 0,
        "lr": 3e-4,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "scheduler": {
            "name": "constant_with_linear_warmup",
            "warmup_updates": 2500,
            "horizon_updates": None,
            "decay_floor_lr": None,
        },
    },
    "trainer": {
        "devices": 1,
        "num_nodes": 1,
        "max_steps": 10,
        "accumulate_grad_batches": 4,
        "detect_anomaly": True,
        "gradient_clip_val": 1.0,
        "precision": "bf16",
    },
}
TRAINING_ARGV = ["/repo/scripts/train.py", "seed=7"]
EXPECTED_CONFIG_SHA256 = _canonical_sha256(RESOLVED_TRAINING_CONFIG)
EXPECTED_ARGV_SHA256 = _canonical_sha256(TRAINING_ARGV)
EXPECTED_WARM_START_SHA256 = "e" * 64
EXPECTED_SELECTED_GPU_UUIDS = ["GPU-test-a"]
EXPECTED_SELECTED_GPU_UUIDS_JSON = json.dumps(
    EXPECTED_SELECTED_GPU_UUIDS, separators=(",", ":")
)
EXPECTED_LOCK_RECORD = {
    "schema_version": 1,
    "status": "held",
    "purpose": "receipt test fixture",
    "owner_token": "fixture-owner-token",
}
EXPECTED_LOCK_BYTES = (
    json.dumps(EXPECTED_LOCK_RECORD, indent=2, sort_keys=True, allow_nan=False) + "\n"
).encode("utf-8")
EXPECTED_LOCK_SHA256 = hashlib.sha256(EXPECTED_LOCK_BYTES).hexdigest()


def _run(command, *, cwd):
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def receipt_repository(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    writer = repository / "scripts/udlm/write_pilot_exit_status.py"
    writer.parent.mkdir(parents=True)
    shutil.copy2(
        Path(launcher.__file__).with_name("write_pilot_exit_status.py"),
        writer,
    )
    _run(["git", "init", "-b", "main"], cwd=repository)
    _run(["git", "config", "user.email", "pilot-tests@example.invalid"], cwd=repository)
    _run(["git", "config", "user.name", "Pilot Tests"], cwd=repository)
    _run(["git", "add", "scripts/udlm/write_pilot_exit_status.py"], cwd=repository)
    _run(["git", "commit", "-m", "add receipt writer"], cwd=repository)
    remote = tmp_path / "remote.git"
    _run(["git", "init", "--bare", str(remote)], cwd=tmp_path)
    _run(["git", "remote", "add", "origin", str(remote)], cwd=repository)
    _run(["git", "push", "-u", "origin", "main"], cwd=repository)
    revision = _run(["git", "rev-parse", "HEAD"], cwd=repository).stdout.strip()

    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(launcher, "_python_executable", lambda: Path(sys.executable))
    return repository, revision


def _paths(repository, run_name="test_run"):
    run_dir = repository / "output/udlm" / run_name
    return {
        "run_dir": run_dir,
        "summary": run_dir / "training_summary.json",
        "receipt": run_dir / "pilot_exit_status.json",
        "checkpoint": run_dir / "checkpoints/10.ckpt",
        "manifest": run_dir / "launch_manifest.json",
        "lock": repository / "output/udlm/.single_training_job.lock",
        "log": repository / "output/logs" / f"{run_name}.log",
    }


def _snapshot(path):
    observed = path.stat()
    return {
        "path": str(path),
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "mode": observed.st_mode,
        "link_count": observed.st_nlink,
        "size_bytes": observed.st_size,
        "mtime_ns": observed.st_mtime_ns,
        "ctime_ns": observed.st_ctime_ns,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "stable_regular_file_verified": True,
    }


def _finite_record(*, tensors=2, elements=4):
    return {
        "all_finite": True,
        "floating_tensor_count": tensors,
        "floating_element_count": elements,
    }


def _auxiliary_checkpoint_records(
    *,
    resolved_training_config,
    expected_steps=10,
    parameter_state_count=2,
):
    callback_key = (
        "ModelCheckpoint{'monitor': None, 'mode': 'min', "
        f"'every_n_train_steps': {expected_steps}, 'every_n_epochs': 0, "
        "'train_time_interval': None}"
    )
    scheduler_config = resolved_training_config["optim"]["scheduler"]
    schedule_check_count = (
        max(
            expected_steps,
            scheduler_config["warmup_updates"] + 1,
            (scheduler_config["horizon_updates"] or 0) + 1,
        )
        + 1
    )
    accumulation = resolved_training_config["trainer"]["accumulate_grad_batches"]
    configured_clip_algorithm = resolved_training_config["trainer"].get(
        "gradient_clip_algorithm"
    )
    effective_clip_algorithm = (
        "norm" if configured_clip_algorithm is None else configured_clip_algorithm
    )
    return {
        "checkpoint_python_floats": {
            "all_finite": True,
            "floating_scalar_count": 6,
        },
        "optimizer_live_state_match": {
            "exact_serialized_live_match": True,
            "optimizer_count": 1,
            "optimizer_class": "AdamW",
            "parameter_group_count": 1,
            "parameter_state_count": parameter_state_count,
            "exact_resolved_config_match": True,
        },
        "scheduler_live_state_match": {
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
        },
        "sampler_live_state_match": {
            "exact_hosted_stream_contract_match": True,
            "random_state_is_none": True,
            "live_state_dict_available": False,
            "sampler_class_module": "torch.utils.data.dataloader",
            "sampler_class_name": "_InfiniteConstantSampler",
        },
        "trainer_live_configuration_match": {
            "exact_detect_anomaly_match": True,
            "detect_anomaly": True,
            "exact_gradient_clip_val_match": True,
            "gradient_clip_val": float(
                resolved_training_config["trainer"]["gradient_clip_val"]
            ),
            "exact_gradient_clip_algorithm_match": True,
            "gradient_clip_algorithm": effective_clip_algorithm,
            "exact_precision_match": True,
            "configured_precision": str(
                resolved_training_config["trainer"]["precision"]
            ),
            "live_precision": "bf16-mixed",
        },
        "model_checkpoint_live_state_match": {
            "exact_serialized_live_match": True,
            "model_checkpoint_callback_count": 1,
            "state_key": callback_key,
            "configuration_matches_pilot_contract": True,
        },
        "checkpoint_hyperparameters_match": {
            "hparams_name": "kwargs",
            "exact_hyperparameter_keys": True,
            "exact_checkpoint_preflight_config_match": True,
            "exact_live_model_preflight_config_match": True,
            "exact_live_hparams_preflight_config_match": True,
            "exact_checkpoint_live_model_unresolved_config_match": True,
            "exact_checkpoint_live_hparams_unresolved_config_match": True,
            "resolved_config_sha256": _canonical_sha256(resolved_training_config),
        },
        "checkpoint_loop_state_match": {
            "exact_serialized_progress_match": True,
            "epoch": 0,
            "optimizer_steps": expected_steps,
            "accumulate_grad_batches": accumulation,
            "microbatches": expected_steps * accumulation,
        },
    }


def _film_gradient_contract_and_audit():
    contract = {
        "schema_version": 1,
        "observation_point": (
            "on_before_optimizer_step_global_rank_zero_after_gradient_accumulation"
        ),
        "optimizer_checks": [1, 2, 3],
        "first_positive_lr_optimizer_step": 2,
        "timestep_mlp_required_optimizer_check": 3,
        "groups": [
            {
                "group_id": "film_modulation",
                "kind": "film",
                "parameters": [
                    {"name": "backbone.layer.film_modulation.weight", "shape": [4, 2]},
                    {"name": "backbone.layer.film_modulation.bias", "shape": [4]},
                ],
            },
            {
                "group_id": "timestep_mlp",
                "kind": "timestep_mlp",
                "parameters": [
                    {"name": "backbone.time_conditioner.0.weight", "shape": [2, 2]},
                    {"name": "backbone.time_conditioner.0.bias", "shape": [2]},
                    {"name": "backbone.time_conditioner.2.weight", "shape": [2, 2]},
                    {"name": "backbone.time_conditioner.2.bias", "shape": [2]},
                ],
            },
        ],
    }
    digest = _canonical_sha256(contract)
    group_reports = {}
    for group in contract["groups"]:
        group_reports[group["group_id"]] = {
            "group_id": group["group_id"],
            "ordered_parameter_manifest_sha256": _canonical_sha256(group["parameters"]),
            "parameter_count": len(group["parameters"]),
            "gradient_element_count": sum(
                math.prod(parameter["shape"]) for parameter in group["parameters"]
            ),
            "all_gradients_present": True,
            "all_gradients_finite": True,
            "all_parameter_gradients_nonzero": True,
        }
    checks = []
    for index, learning_rate in enumerate((0.0, 3e-6, 6e-6), start=1):
        timestep_report = dict(group_reports["timestep_mlp"])
        timestep_report["all_parameter_gradients_nonzero"] = index == 3
        checks.append(
            {
                "optimizer_gradient_observation_index": index,
                "optimizer_step_index": index,
                "learning_rate_before_step": learning_rate,
                "film_groups": [dict(group_reports["film_modulation"])],
                "timestep_mlp_groups": [timestep_report],
            }
        )
    audit = {
        "schema_version": 1,
        "status": "completed",
        "observation_point": contract["observation_point"],
        "registered_contract_sha256": digest,
        "first_positive_lr_optimizer_step": 2,
        "timestep_mlp_required_optimizer_check": 3,
        "optimizer_checks": checks,
    }
    return contract, digest, audit


def _write_training_job_lock(paths):
    paths["lock"].parent.mkdir(parents=True, exist_ok=True)
    if not paths["lock"].exists():
        paths["lock"].write_bytes(EXPECTED_LOCK_BYTES)
    return EXPECTED_LOCK_RECORD, EXPECTED_LOCK_SHA256


def _write_launch_manifest(paths):
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    lock_record, lock_sha256 = _write_training_job_lock(paths)
    manifest = {
        "launch_manifest_schema_version": receipt_writer.LAUNCH_MANIFEST_SCHEMA_VERSION,
        "created_at": "2020-01-01T00:00:00+00:00",
        "user_requested_gpu_count": 1,
        "cuda_visible_device_uuids": EXPECTED_SELECTED_GPU_UUIDS,
        "single_training_job_lock": {
            "path": str(paths["lock"]),
            "sha256": lock_sha256,
            "record": lock_record,
        },
    }
    payload = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if not paths["manifest"].exists():
        paths["manifest"].write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _genesis_predecessor_binding(panel):
    return {
        "schema_version": 1,
        "state": "explicit_genesis_no_predecessor",
        "current_training_variant": "udlm",
        "current_variant_position": 0,
        "expected_predecessor_training_variant": None,
        "expected_predecessor_variant_position": None,
        "matched_panel_spec_sha256": _canonical_sha256(panel),
        "common_training_contract_sha256": _canonical_sha256(
            panel["common_training_contract"]
        ),
        "receipt_artifact": None,
        "predecessor_launch_manifest_artifact": None,
        "predecessor_training_summary_artifact": None,
        "predecessor_run_name": None,
        "chronology": None,
        "validated_before_gpu_probe": True,
    }


def _successful_pipeline_component():
    return {
        "shell_exit_status": 0,
        "succeeded": True,
        "possible_termination_signal": None,
        "shell_status_is_signal_compatible": False,
        "signal_provenance": None,
    }


def _predecessor_panel():
    representative_config = {
        "data": "safe",
        "seed": 7,
        "training": {
            "ema": 0.9999,
            "udlm": {
                "prior_variant": "release_uniform",
                "exclude_special_tokens": True,
                "empirical_uniform_mix": 0.0002,
            },
        },
        "loader": {"batch_size": 2, "global_batch_size": 8, "num_workers": 1},
        "optim": copy.deepcopy(RESOLVED_TRAINING_CONFIG["optim"]),
        "trainer": {
            "devices": 1,
            "num_nodes": 1,
            "max_steps": 10,
            "accumulate_grad_batches": 4,
            "detect_anomaly": True,
            "gradient_clip_val": 1.0,
            "precision": "bf16",
        },
        "callback": {"dirpath": "/masked/by/matched-panel-hash"},
    }
    return {
        "schema_version": 2,
        "purpose": "matched_R_S_E_UDLM_training_pilot",
        "execution": {
            "mode": "single_job_lease_with_machine_enforced_predecessor_receipt_chain",
            "maximum_concurrent_training_jobs": 1,
            "concurrency_enforcement": "atomic_global_worktree_training_job_lock",
            "registered_variant_order": [
                "udlm",
                "schedule_uniform",
                "udlm_categorical",
            ],
            "advance_policy": (
                "launcher_validates_and_binds_exact_successful_predecessor_receipt"
            ),
            "predecessor_receipt_bound_in_each_manifest": True,
            "genesis_requires_explicit_declaration": True,
            "successor_launch_requires_exact_predecessor_receipt": True,
        },
        "registered_treatments": [
            {
                "training_variant": "udlm",
                "hydra_config_name": "udlm",
                "udlm_prior_variant": "release_uniform",
                "comparison_role": "faithful_release_control",
            },
            {
                "training_variant": "schedule_uniform",
                "hydra_config_name": "udlm",
                "udlm_prior_variant": "schedule_uniform",
                "comparison_role": "schedule_repair_uniform_control",
            },
            {
                "training_variant": "udlm_categorical",
                "hydra_config_name": "udlm_categorical",
                "udlm_prior_variant": "empirical_frequency",
                "comparison_role": "empirical_prior_treatment",
            },
        ],
        "common_training_contract": {
            "source_revision": "a" * 40,
            "initialization_mode": "verified_mdlm_ema_warm_start",
            "initialization_checkpoint_path": "/synthetic/mdlm.ckpt",
            "initialization_checkpoint_sha256": EXPECTED_WARM_START_SHA256,
            "requested_gpu_count": 1,
            "max_steps": 10,
            "global_batch_size": 8,
            "micro_batch_size_per_process": 2,
            "accumulate_grad_batches": 4,
            "effective_global_batch_size": 8,
            "num_workers": 1,
            "seed": 7,
            "exclude_special_tokens": True,
            "empirical_uniform_mix": 0.0002,
            "empirical_uniform_mix_consumed_only_by": "empirical_frequency",
            "empirical_uniform_mix_audit": {
                "relative_path": "experiments/udlm/prior_geometry/floor.json",
                "sha256": "1" * 64,
                "source_revision": "a" * 40,
                "scope": "retrospective_training_only_engineering_selection",
            },
            "common_resolved_config_sha256": launcher.matched_panel_config_sha256(
                representative_config
            ),
        },
        "common_gpu_safety_policy": {
            "max_utilization_percent": 10,
            "utilization_comparison": "strictly_less_than",
            "min_free_memory_mib": 30000,
            "active_compute_processes_allowed": True,
            "compute_mode_prohibited_allowed": False,
            "physical_gpu_identity_is_per_run_provenance": True,
        },
    }


_PREDECESSOR_VARIANTS = {
    "udlm": ("udlm", "release_uniform", "faithful_release_control"),
    "schedule_uniform": (
        "udlm",
        "schedule_uniform",
        "schedule_repair_uniform_control",
    ),
}


def _write_producer_predecessor(
    repository,
    *,
    run_name,
    training_variant,
    variant_position,
    panel,
    prior_binding,
    created_at,
    completed_at,
    recorded_at,
    mutate_manifest=None,
    mutate_summary=None,
    mutate_receipt=None,
):
    run_dir = repository / "output/udlm" / run_name
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "launch_manifest.json"
    runtime_path = run_dir / "runtime_config.json"
    summary_path = run_dir / "training_summary.json"
    receipt_path = run_dir / "pilot_exit_status.json"
    checkpoint_path = run_dir / "checkpoints/10.ckpt"
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_bytes(b"producer-shaped predecessor checkpoint\n")
    checkpoint_snapshot = _snapshot(checkpoint_path)
    source_revision = panel["common_training_contract"]["source_revision"]
    hydra_name, prior_variant, comparison_role = _PREDECESSOR_VARIANTS[training_variant]
    selected_gpu_uuids = [f"GPU-{run_name}-fixture"]
    gpu_state = {
        "physical_index": 6,
        "uuid": selected_gpu_uuids[0],
        "name": "Synthetic Accelerator",
        "memory_used_mib": 1000,
        "memory_total_mib": 81920,
        "utilization_percent": 0,
        "compute_mode": "Default",
        "compute_processes": [],
    }
    resolved_config = {
        "data": "safe",
        "seed": 7,
        "training": {
            "ema": 0.9999,
            "udlm": {
                "prior_variant": prior_variant,
                "exclude_special_tokens": True,
                "empirical_uniform_mix": 0.0002,
            },
        },
        "loader": {"batch_size": 2, "global_batch_size": 8, "num_workers": 1},
        "optim": copy.deepcopy(RESOLVED_TRAINING_CONFIG["optim"]),
        "trainer": {
            "devices": 1,
            "num_nodes": 1,
            "max_steps": 10,
            "accumulate_grad_batches": 4,
            "detect_anomaly": True,
            "gradient_clip_val": 1.0,
            "precision": "bf16",
        },
        "callback": {"dirpath": str(checkpoint_path.parent)},
    }
    resolved_config_sha256 = _canonical_sha256(resolved_config)
    child_argv = [
        str(repository / "scripts/train.py"),
        "--config-name",
        hydra_name,
        "seed=7",
        "trainer.devices=1",
    ]
    full_argv = [str(repository / ".venv/bin/python"), "-u", *child_argv]
    argv_sha256 = _canonical_sha256(child_argv)
    summary_completion = {
        "summary_schema_version": receipt_writer.TRAINING_SUMMARY_SCHEMA_VERSION,
        "summary_path": str(summary_path),
        "final_checkpoint_path": str(checkpoint_path),
        "expected_max_steps": 10,
        "expected_world_size": 1,
        "fail_on_nonfinite_loss": True,
        "backward_anomaly_detection": True,
    }
    lock_path = repository / "output/udlm/.single_training_job.lock"
    lock_record = {
        "schema_version": 1,
        "status": "held",
        "purpose": "enforce_one_R_S_E_pilot_training_job_at_a_time",
        "source_revision": source_revision,
        "run_name": run_name,
        "training_variant": training_variant,
        "owner_token": ("c" if variant_position == 0 else "d") * 64,
        "launcher_pid_at_acquisition": 1234 + variant_position,
        "acquired_at_utc": (
            "2026-09-06T11:55:00+00:00"
            if variant_position == 0
            else "2026-09-06T11:59:05+00:00"
        ),
        "owner_process_exit_does_not_make_lock_stale": True,
        "stale_lock_policy": "fail_closed_and_require_manual_review",
        "release_policy": (
            "exact_owner_lock_only_after_receipt_or_before_tmux_handoff_failure"
        ),
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(lock_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lock_snapshot = _snapshot(lock_path)
    manifest_completion = {
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
    }
    inventory_at = (
        "2026-09-06T11:56:00+00:00"
        if variant_position == 0
        else ("2026-09-06T11:59:10+00:00")
    )
    probe_at = (
        "2026-09-06T11:56:30+00:00"
        if variant_position == 0
        else ("2026-09-06T11:59:20+00:00")
    )
    manifest = {
        "launch_manifest_schema_version": receipt_writer.LAUNCH_MANIFEST_SCHEMA_VERSION,
        "created_at": created_at,
        "purpose": "bounded UDLM training pilot",
        "gpu_selection_schema_version": 2,
        "git_sha": source_revision,
        "source_revision_before_final_gpu_probe": source_revision,
        "run_name": run_name,
        "training_variant": training_variant,
        "hydra_config_name": hydra_name,
        "udlm_prior_variant": prior_variant,
        "udlm_comparison_role": comparison_role,
        "matched_panel_spec": panel,
        "matched_panel_spec_sha256": _canonical_sha256(panel),
        "matched_panel_variant_position": variant_position,
        "predecessor_receipt_binding": prior_binding,
        "single_training_job_lock": {
            "path": str(lock_path),
            "sha256": lock_snapshot["sha256"],
            "record": lock_record,
            "acquired_before_any_gpu_probe": True,
            "stale_lock_policy": "fail_closed_and_require_manual_review",
            "release_owner": "pilot_exit_receipt_writer_after_publication",
        },
        "tmux_session": f"genmol_{training_variant}_{run_name}",
        "user_requested_gpu_count": 1,
        "gpu_selection_method": "dynamic_idle_discovery",
        "gpu_inventory_scope": "all_nvidia_gpus",
        "inventory_snapshot_completed_at_utc": inventory_at,
        "gpu_inventory_at_selection": [dict(gpu_state)],
        "initially_selected_gpu_states": [dict(gpu_state)],
        "logical_cuda_devices": [0],
        "physical_gpu_indices": [6],
        "cuda_visible_device_uuids": selected_gpu_uuids,
        "final_uuid_probes_completed_at_utc": probe_at,
        "gpu_states_at_final_uuid_probe": [dict(gpu_state)],
        "gpu_safety_policy": {
            "max_utilization_percent": 10,
            "utilization_comparison": "strictly_less_than",
            "min_free_memory_mib": 30000,
            "active_compute_processes_allowed": True,
            "compute_mode_prohibited_allowed": False,
        },
        "training_argv": full_argv,
        "training_argv_sha256": argv_sha256,
        "resolved_training_config": resolved_config,
        "resolved_training_config_sha256": resolved_config_sha256,
        "runtime_config_path": str(runtime_path),
        "training_summary_path": str(summary_path),
        "training_summary_schema_version": (
            receipt_writer.TRAINING_SUMMARY_SCHEMA_VERSION
        ),
        "pilot_exit_status_path": str(receipt_path),
        "pilot_exit_status_schema_version": 5,
        "expected_final_checkpoint_path": str(checkpoint_path),
        "launch_manifest_path": str(manifest_path),
        "launch_manifest_raw_sha256_transport": (
            "passed_out_of_band_to_training_and_receipt_to_avoid_self_hash"
        ),
        "completion_contract": manifest_completion,
        "log_path": str(repository / f"output/logs/{run_name}.log"),
        "log_reserved_exclusively_before_manifest": True,
        "checkpoint": "/synthetic/mdlm.ckpt",
        "checkpoint_sha256": EXPECTED_WARM_START_SHA256,
        "seed": 7,
        "max_steps": 10,
        "global_batch_size": 8,
        "micro_batch_size_per_process": 2,
        "accumulate_grad_batches": 4,
        "effective_global_batch_size": 8,
        "exclude_special_tokens": True,
        "dry_run": False,
    }
    if mutate_manifest is not None:
        mutate_manifest(manifest)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_snapshot = _snapshot(manifest_path)
    manifest_claim = {**manifest_snapshot, "selected_gpu_uuids": selected_gpu_uuids}
    runtime = {
        "schema_version": 2,
        "status": "preflight_completed",
        "source_revision": source_revision,
        "source": {"head": source_revision, "upstream": source_revision},
        "training_argv": child_argv,
        "observed_training_argv": child_argv,
        "training_argv_sha256": argv_sha256,
        "resolved_training_config": resolved_config,
        "resolved_training_config_sha256": resolved_config_sha256,
        "launch_manifest": manifest_claim,
        "completion_contract": summary_completion,
        "python_environment": {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "7",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONNOUSERSITE": "1",
            "PYTHONOPTIMIZE": "0",
            "PYTHONPATH": f"{repository / 'src'}:{repository}",
            "PYTHONUTF8": "1",
        },
    }
    runtime_path.write_text(
        json.dumps(runtime, sort_keys=True) + "\n", encoding="utf-8"
    )
    runtime_snapshot = _snapshot(runtime_path)
    accounting = {
        "training_seed": 7,
        "optimizer_updates": 10,
        "world_size": 1,
        "micro_batch_size_per_rank": 2,
        "accumulate_grad_batches": 4,
        "effective_global_examples_per_optimizer_step": 8,
        "total_requested_example_exposures": 80,
        "hosted_stream_rank_partition_policy": (
            "huggingface_split_dataset_by_node_disjoint_rank_streams"
        ),
        "trainable_parameter_counts": {
            "base_backbone": 3,
            "time_conditioner": 2,
            "total": 5,
        },
    }
    ema_metadata = {"shadow_parameter_count": 2, "decay": 0.9999, "num_updates": 10}
    summary = {
        "schema_version": receipt_writer.TRAINING_SUMMARY_SCHEMA_VERSION,
        "status": "completed",
        "completed_at_utc": completed_at,
        "source_revision": source_revision,
        "source": {"head": source_revision, "upstream": source_revision},
        "resolved_training_config_sha256": resolved_config_sha256,
        "training_argv_sha256": argv_sha256,
        "launch_manifest": manifest_claim,
        "runtime_config": {
            **runtime_snapshot,
            "schema_version": 2,
            "record_sha256": _canonical_sha256(runtime),
        },
        "completion_contract": summary_completion,
        "observed_training_state": {
            "global_rank": 0,
            "global_step": 10,
            "world_size": 1,
        },
        "training_accounting": accounting,
        "training_health": {
            "scope": "rank-zero counters",
            "all_losses_finite": True,
            "all_observed_gradients_finite": True,
            "every_optimizer_step_had_a_nonzero_gradient": True,
            "loss_checks": 10,
            "optimizer_step_checks": 10,
            "gradient_tensor_observations": 10,
            "gradient_element_observations": 20,
        },
        "conditioning_gradient_audit": None,
        "screen_initialization_state_audit": None,
        "final_checkpoint": {
            **checkpoint_snapshot,
            "semantic_audit": {
                "deserialized": True,
                "global_step": 10,
                "raw_model": _finite_record(),
                "ema": _finite_record(),
                "ema_metadata": ema_metadata,
                "optimizer": _finite_record(),
                "non_sentinel_checkpoint_tensors": _finite_record(
                    tensors=6, elements=12
                ),
                "framework_nonfinite_sentinels": (
                    receipt_writer._expected_framework_nonfinite_sentinels(
                        expected_steps=10
                    )
                ),
                **_auxiliary_checkpoint_records(
                    resolved_training_config=resolved_config
                ),
                "udlm_process_identity_verified": True,
                "live_model_match": {
                    "exact_key_set": True,
                    "exact_tensor_values": True,
                    "tensor_count": 2,
                },
                "live_ema_match": {"exact_tensor_values": True, "tensor_count": 2},
            },
        },
        "tensor_finiteness": {"raw_model": _finite_record(), "ema": _finite_record()},
        "startup": {
            "mode": "warm_start",
            "verified_mdlm_warm_start_report": {
                "source_path": "/synthetic/mdlm.ckpt",
                "source_resolved_path": "/synthetic/mdlm.ckpt",
                "source_sha256": EXPECTED_WARM_START_SHA256,
                "source_size_bytes": 123,
                "expected_source_sha256": EXPECTED_WARM_START_SHA256,
                "byte_identity_verified_before_and_after_load": True,
                "weights": "ema",
                "parameter_tensors": 2,
            },
        },
    }
    if mutate_summary is not None:
        mutate_summary(summary)
    summary_path.write_text(
        json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary_snapshot = _snapshot(summary_path)
    validated_bindings = {
        "schema_version": receipt_writer.TRAINING_SUMMARY_SCHEMA_VERSION,
        "source_revision": source_revision,
        "resolved_training_config_sha256": resolved_config_sha256,
        "training_argv_sha256": argv_sha256,
        "launch_manifest_path": str(manifest_path),
        "launch_manifest_sha256": manifest_snapshot["sha256"],
        "selected_gpu_uuids": selected_gpu_uuids,
        "observed_global_step": 10,
        "observed_world_size": 1,
        "training_accounting": accounting,
        "ema_metadata": ema_metadata,
        "final_checkpoint_path": str(checkpoint_path),
        "final_checkpoint_sha256": checkpoint_snapshot["sha256"],
        "startup_mode": "warm_start",
        "conditioning_gradient_audit": None,
        "screen_initialization_state_audit": None,
    }
    receipt = {
        "schema_version": 5,
        "status": "completed",
        "overall_status": "completed",
        "recorded_at_utc": recorded_at,
        "process_exit_status": 0,
        "expected_contract": {
            "training_summary_schema_version": (
                receipt_writer.TRAINING_SUMMARY_SCHEMA_VERSION
            ),
            "source_revision": source_revision,
            "resolved_training_config_sha256": resolved_config_sha256,
            "training_argv_sha256": argv_sha256,
            "launch_manifest_path": str(manifest_path),
            "launch_manifest_sha256": manifest_snapshot["sha256"],
            "selected_gpu_uuids": selected_gpu_uuids,
            "training_job_lock_path": str(lock_path),
            "training_job_lock_sha256": lock_snapshot["sha256"],
            "max_steps": 10,
            "world_size": 1,
            "training_summary_path": str(summary_path),
            "final_checkpoint_path": str(checkpoint_path),
            "initialization_checkpoint_sha256": EXPECTED_WARM_START_SHA256,
        },
        "pipeline": {
            "training": _successful_pipeline_component(),
            "tee": _successful_pipeline_component(),
            "pipefail_shell_exit_status": 0,
        },
        "source_at_receipt": {
            "verified": True,
            "expected_revision": source_revision,
            "head": source_revision,
            "upstream": source_revision,
            "output_directory_excluded_from_cleanliness_check": True,
        },
        "launch_manifest": {
            "path": str(manifest_path),
            "present": True,
            "matches_expected_raw_sha256": True,
            "selected_gpu_uuids_match_expected": True,
            "matches_training_summary_snapshot": True,
            "matches_runtime_config_snapshot": True,
            "valid_and_launch_bound": True,
            "expected_selected_gpu_uuids": selected_gpu_uuids,
            "observed_selected_gpu_uuids": selected_gpu_uuids,
            "artifact": manifest_snapshot,
            "validation_error": None,
        },
        "predecessor_receipt_binding": prior_binding,
        "training_job_lock": {
            "path": str(lock_path),
            "present": True,
            "expected_sha256": lock_snapshot["sha256"],
            "matches_expected_raw_sha256": True,
            "matches_launch_manifest_binding": True,
            "valid_and_launch_bound_before_receipt_publication": True,
            "artifact": lock_snapshot,
            "record": lock_record,
            "release_policy": "publish_receipt_then_unlink_only_same_stat_identity_and_sha256",
            "release_result_not_claimed_inside_pre_release_receipt": True,
            "validation_error": None,
        },
        "training_summary": {
            "path": str(summary_path),
            "present": True,
            "valid_and_launch_bound": True,
            "artifact": summary_snapshot,
            "validated_bindings": validated_bindings,
            "validation_error": None,
        },
        "runtime_config": {
            "path": str(runtime_path),
            "present": True,
            "matches_training_summary_snapshot": True,
            "semantic_validation_passed": True,
            "artifact": runtime_snapshot,
        },
        "final_checkpoint": {
            "path": str(checkpoint_path),
            "present": True,
            "matches_training_summary_snapshot": True,
            "artifact": checkpoint_snapshot,
        },
        "completion_requirements": {
            "training_exit_zero": True,
            "tee_exit_zero": True,
            "training_summary_valid_and_launch_bound": True,
            "launch_manifest_matches_summary_runtime_and_launch": True,
            "predecessor_receipt_binding_unchanged_and_valid": True,
            "training_job_lock_valid_before_receipt_publication": True,
            "runtime_config_matches_summary_and_launch": True,
            "final_checkpoint_matches_training_summary": True,
            "clean_pushed_source_still_matches_launch": True,
            "all_must_hold": True,
        },
    }
    if mutate_receipt is not None:
        mutate_receipt(receipt)
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )
    next_variant = ("schedule_uniform", "udlm_categorical")[variant_position]
    next_binding = {
        "schema_version": 1,
        "state": "validated_successful_predecessor",
        "current_training_variant": next_variant,
        "current_variant_position": variant_position + 1,
        "expected_predecessor_training_variant": training_variant,
        "expected_predecessor_variant_position": variant_position,
        "matched_panel_spec_sha256": _canonical_sha256(panel),
        "common_training_contract_sha256": _canonical_sha256(
            panel["common_training_contract"]
        ),
        "receipt_artifact": _snapshot(receipt_path),
        "predecessor_launch_manifest_artifact": manifest_snapshot,
        "predecessor_training_summary_artifact": summary_snapshot,
        "predecessor_run_name": run_name,
        "chronology": {
            "predecessor_launch_manifest_created_at_utc": created_at,
            "predecessor_training_summary_completed_at_utc": completed_at,
            "predecessor_exit_receipt_recorded_at_utc": recorded_at,
            "strictly_ordered_timestamps_verified": True,
        },
        "validated_before_gpu_probe": True,
    }
    return next_binding, summary_path


def _predecessor_chain_fixture(
    repository,
    *,
    mutate_manifest=None,
    mutate_summary=None,
    mutate_receipt=None,
):
    panel = _predecessor_panel()
    current_binding, summary_path = _write_producer_predecessor(
        repository,
        run_name="r-run",
        training_variant="udlm",
        variant_position=0,
        panel=panel,
        prior_binding=_genesis_predecessor_binding(panel),
        created_at="2026-09-06T11:57:00+00:00",
        completed_at="2026-09-06T11:58:00+00:00",
        recorded_at="2026-09-06T11:59:00+00:00",
        mutate_manifest=mutate_manifest,
        mutate_summary=mutate_summary,
        mutate_receipt=mutate_receipt,
    )
    current_manifest = {
        "created_at": "2026-09-06T12:00:00+00:00",
        "purpose": "bounded UDLM training pilot",
        "training_variant": "schedule_uniform",
        "matched_panel_variant_position": 1,
        "matched_panel_spec": panel,
        "matched_panel_spec_sha256": _canonical_sha256(panel),
        "predecessor_receipt_binding": current_binding,
        "single_training_job_lock": {
            "record": {"acquired_at_utc": "2026-09-06T11:59:10+00:00"}
        },
        "inventory_snapshot_completed_at_utc": "2026-09-06T11:59:20+00:00",
        "final_uuid_probes_completed_at_utc": "2026-09-06T11:59:30+00:00",
    }
    return current_manifest, current_binding, summary_path


def _transitive_predecessor_chain_fixture(repository):
    panel = _predecessor_panel()
    r_binding, r_summary_path = _write_producer_predecessor(
        repository,
        run_name="r-run",
        training_variant="udlm",
        variant_position=0,
        panel=panel,
        prior_binding=_genesis_predecessor_binding(panel),
        created_at="2026-09-06T11:57:00+00:00",
        completed_at="2026-09-06T11:58:00+00:00",
        recorded_at="2026-09-06T11:59:00+00:00",
    )
    e_binding, _s_summary_path = _write_producer_predecessor(
        repository,
        run_name="s-run",
        training_variant="schedule_uniform",
        variant_position=1,
        panel=panel,
        prior_binding=r_binding,
        created_at="2026-09-06T12:00:00+00:00",
        completed_at="2026-09-06T12:01:00+00:00",
        recorded_at="2026-09-06T12:02:00+00:00",
    )
    e_manifest = {
        "created_at": "2026-09-06T12:03:00+00:00",
        "purpose": "bounded UDLM training pilot",
        "training_variant": "udlm_categorical",
        "matched_panel_variant_position": 2,
        "matched_panel_spec": panel,
        "matched_panel_spec_sha256": _canonical_sha256(panel),
        "predecessor_receipt_binding": e_binding,
        "single_training_job_lock": {
            "record": {"acquired_at_utc": "2026-09-06T12:02:10+00:00"}
        },
        "inventory_snapshot_completed_at_utc": "2026-09-06T12:02:20+00:00",
        "final_uuid_probes_completed_at_utc": "2026-09-06T12:02:30+00:00",
    }
    return e_manifest, r_summary_path


def test_receipt_revalidates_bound_predecessor_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(receipt_writer, "REPOSITORY_ROOT", tmp_path)
    manifest, expected_binding, _summary_path = _predecessor_chain_fixture(tmp_path)

    observed = receipt_writer._validated_predecessor_binding_at_receipt(manifest)

    assert observed == expected_binding
    assert observed is not expected_binding
    predecessor_manifest = json.loads(
        Path(
            expected_binding["predecessor_launch_manifest_artifact"]["path"]
        ).read_text(encoding="utf-8")
    )
    assert (
        predecessor_manifest["launch_manifest_schema_version"]
        == receipt_writer.LAUNCH_MANIFEST_SCHEMA_VERSION
        == 2
    )


def _set_predecessor_gpu_state_fields(manifest, **updates):
    for field in (
        "gpu_inventory_at_selection",
        "initially_selected_gpu_states",
        "gpu_states_at_final_uuid_probe",
    ):
        manifest[field][0].update(copy.deepcopy(updates))


def test_receipt_accepts_recorded_process_below_utilization_threshold(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(receipt_writer, "REPOSITORY_ROOT", tmp_path)
    process = {
        "pid": 4321,
        "process_name": "pre-existing-workload",
        "used_memory_mib": 512,
    }
    manifest, expected_binding, _summary_path = _predecessor_chain_fixture(
        tmp_path,
        mutate_manifest=lambda value: _set_predecessor_gpu_state_fields(
            value,
            utilization_percent=9,
            compute_processes=[process],
        ),
    )

    assert (
        receipt_writer._validated_predecessor_binding_at_receipt(manifest)
        == expected_binding
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"utilization_percent": 10},
        {"memory_used_mib": 51_921},
        {"compute_mode": "Prohibited"},
    ],
)
def test_receipt_still_rejects_unsafe_predecessor_gpu_state(
    tmp_path, monkeypatch, updates
):
    monkeypatch.setattr(receipt_writer, "REPOSITORY_ROOT", tmp_path)
    manifest, _expected_binding, _summary_path = _predecessor_chain_fixture(
        tmp_path,
        mutate_manifest=lambda value: _set_predecessor_gpu_state_fields(
            value, **updates
        ),
    )

    with pytest.raises(ValueError, match="violates the safety policy"):
        receipt_writer._validated_predecessor_binding_at_receipt(manifest)


@pytest.mark.parametrize(
    ("artifact", "missing_key", "error_fragment"),
    [
        ("manifest", "source_revision_before_final_gpu_probe", "manifest keys"),
        ("summary", "runtime_config", "summary keys"),
        ("receipt", "expected_contract", "receipt keys"),
    ],
)
def test_receipt_rejects_sparse_nonproducer_predecessor_artifacts(
    tmp_path, monkeypatch, artifact, missing_key, error_fragment
):
    monkeypatch.setattr(receipt_writer, "REPOSITORY_ROOT", tmp_path)

    def remove_required_key(value):
        value.pop(missing_key)

    mutations = {f"mutate_{artifact}": remove_required_key}
    manifest, _binding, _summary_path = _predecessor_chain_fixture(
        tmp_path, **mutations
    )

    with pytest.raises(ValueError, match=error_fragment):
        receipt_writer._validated_predecessor_binding_at_receipt(manifest)


def test_receipt_rejects_legacy_predecessor_launch_manifest_schema(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(receipt_writer, "REPOSITORY_ROOT", tmp_path)
    manifest, _binding, _summary_path = _predecessor_chain_fixture(
        tmp_path,
        mutate_manifest=lambda value: value.__setitem__(
            "launch_manifest_schema_version", 1
        ),
    )

    with pytest.raises(ValueError, match="launch-manifest schema"):
        receipt_writer._validated_predecessor_binding_at_receipt(manifest)


def test_receipt_accepts_project_level_reviewed_predecessor_python(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(receipt_writer, "REPOSITORY_ROOT", tmp_path)
    alternate_python = str(tmp_path.parents[1] / ".venv/bin/python")
    manifest, expected_binding, _summary_path = _predecessor_chain_fixture(
        tmp_path,
        mutate_manifest=lambda value: value["training_argv"].__setitem__(
            0, alternate_python
        ),
    )

    assert (
        receipt_writer._validated_predecessor_binding_at_receipt(manifest)
        == expected_binding
    )


def test_receipt_rejects_arbitrary_predecessor_python_path(tmp_path, monkeypatch):
    monkeypatch.setattr(receipt_writer, "REPOSITORY_ROOT", tmp_path)
    manifest, _binding, _summary_path = _predecessor_chain_fixture(
        tmp_path,
        mutate_manifest=lambda value: value["training_argv"].__setitem__(
            0, "/usr/bin/python3"
        ),
    )

    with pytest.raises(ValueError, match="reviewed producer prefix"):
        receipt_writer._validated_predecessor_binding_at_receipt(manifest)


@pytest.mark.parametrize(
    ("mutation", "error_fragment"),
    [
        (
            lambda receipt: receipt["pipeline"]["training"].__setitem__(
                "succeeded", False
            ),
            "successful producer value",
        ),
        (
            lambda receipt: receipt["source_at_receipt"].__setitem__("head", "b" * 40),
            "exact producer value",
        ),
        (
            lambda receipt: receipt["completion_requirements"].pop("all_must_hold"),
            "completion requirements keys",
        ),
        (
            lambda receipt: receipt["training_job_lock"]["artifact"].__setitem__(
                "link_count", 2
            ),
            "training-job lock evidence is invalid",
        ),
    ],
)
def test_receipt_rejects_nonproducer_predecessor_success_evidence(
    tmp_path, monkeypatch, mutation, error_fragment
):
    monkeypatch.setattr(receipt_writer, "REPOSITORY_ROOT", tmp_path)
    manifest, _binding, _summary_path = _predecessor_chain_fixture(
        tmp_path, mutate_receipt=mutation
    )

    with pytest.raises(ValueError, match=error_fragment):
        receipt_writer._validated_predecessor_binding_at_receipt(manifest)


def test_receipt_rejects_predecessor_lock_record_not_matching_raw_digest(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(receipt_writer, "REPOSITORY_ROOT", tmp_path)

    def replace_owner_token(manifest):
        manifest["single_training_job_lock"]["record"]["owner_token"] = "e" * 64

    manifest, _binding, _summary_path = _predecessor_chain_fixture(
        tmp_path, mutate_manifest=replace_owner_token
    )

    with pytest.raises(ValueError, match="deterministic raw digest"):
        receipt_writer._validated_predecessor_binding_at_receipt(manifest)


@pytest.mark.parametrize(
    ("tamper", "error_fragment"),
    [
        ("duplicate_inventory_uuid", "inventory UUIDs must be unique"),
        ("duplicate_inventory_physical_index", "physical indices must be unique"),
        ("selected_uuid_absent", "selected GPU UUIDs are absent"),
        ("initial_state_differs", "initially selected GPUs differ"),
        ("sparse_process_telemetry", "compute process keys are invalid"),
    ],
)
def test_receipt_rejects_predecessor_gpu_provenance_tampering(
    tmp_path, monkeypatch, tamper, error_fragment
):
    monkeypatch.setattr(receipt_writer, "REPOSITORY_ROOT", tmp_path)

    def mutate_gpu_evidence(manifest):
        inventory = manifest["gpu_inventory_at_selection"]
        if tamper == "duplicate_inventory_uuid":
            duplicate = dict(inventory[0])
            duplicate["physical_index"] = 7
            inventory.append(duplicate)
        elif tamper == "duplicate_inventory_physical_index":
            duplicate = dict(inventory[0])
            duplicate["uuid"] = "GPU-distinct-fixture"
            inventory.append(duplicate)
        elif tamper == "selected_uuid_absent":
            inventory[0]["uuid"] = "GPU-not-selected"
        elif tamper == "initial_state_differs":
            manifest["initially_selected_gpu_states"][0]["memory_used_mib"] += 1
        else:
            inventory[0]["compute_processes"] = [
                {"pid": 123, "process_name": "missing-memory-field"}
            ]

    manifest, _binding, _summary_path = _predecessor_chain_fixture(
        tmp_path, mutate_manifest=mutate_gpu_evidence
    )

    with pytest.raises(ValueError, match=error_fragment):
        receipt_writer._validated_predecessor_binding_at_receipt(manifest)


def test_receipt_predecessor_revalidation_detects_late_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr(receipt_writer, "REPOSITORY_ROOT", tmp_path)
    manifest, _expected_binding, summary_path = _predecessor_chain_fixture(tmp_path)
    summary_path.write_bytes(summary_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="no longer matches"):
        receipt_writer._validated_predecessor_binding_at_receipt(manifest)


def test_receipt_predecessor_revalidation_requires_direct_run_directory(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(receipt_writer, "REPOSITORY_ROOT", tmp_path)
    manifest, _expected_binding, _summary_path = _predecessor_chain_fixture(tmp_path)
    relocated = tmp_path / "archive/r-run/pilot_exit_status.json"
    relocated.parent.mkdir(parents=True)
    original = Path(manifest["predecessor_receipt_binding"]["receipt_artifact"]["path"])
    shutil.copy2(original, relocated)
    manifest["predecessor_receipt_binding"]["receipt_artifact"] = _snapshot(relocated)

    with pytest.raises(ValueError, match="direct output/udlm run"):
        receipt_writer._validated_predecessor_binding_at_receipt(manifest)


def test_receipt_predecessor_revalidation_requires_cross_run_chronology(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(receipt_writer, "REPOSITORY_ROOT", tmp_path)
    manifest, _expected_binding, _summary_path = _predecessor_chain_fixture(tmp_path)
    manifest["created_at"] = "2026-09-06T11:59:00+00:00"

    with pytest.raises(ValueError, match="must precede the current"):
        receipt_writer._validated_predecessor_binding_at_receipt(manifest)


@pytest.mark.parametrize(
    ("current_event", "error_fragment"),
    [
        ("lock", "current training-job lock acquisition"),
        ("inventory", "current GPU inventory snapshot"),
        ("final_probe", "current final GPU probes"),
    ],
)
@pytest.mark.parametrize(
    "current_event_timestamp",
    ["2026-09-06T11:59:00+00:00", "2026-09-06T11:58:59+00:00"],
    ids=("equal_to_predecessor_receipt", "before_predecessor_receipt"),
)
def test_receipt_predecessor_must_strictly_predate_current_lock_and_gpu_probes(
    tmp_path,
    monkeypatch,
    current_event,
    error_fragment,
    current_event_timestamp,
):
    monkeypatch.setattr(receipt_writer, "REPOSITORY_ROOT", tmp_path)
    manifest, _expected_binding, _summary_path = _predecessor_chain_fixture(tmp_path)
    if current_event == "lock":
        manifest["single_training_job_lock"]["record"]["acquired_at_utc"] = (
            current_event_timestamp
        )
    elif current_event == "inventory":
        manifest["inventory_snapshot_completed_at_utc"] = current_event_timestamp
    else:
        manifest["final_uuid_probes_completed_at_utc"] = current_event_timestamp

    with pytest.raises(ValueError, match=error_fragment):
        receipt_writer._validated_predecessor_binding_at_receipt(manifest)


def test_receipt_predecessor_revalidation_is_transitive(tmp_path, monkeypatch):
    monkeypatch.setattr(receipt_writer, "REPOSITORY_ROOT", tmp_path)
    manifest, r_summary_path = _transitive_predecessor_chain_fixture(tmp_path)
    r_summary_path.write_bytes(r_summary_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="no longer matches"):
        receipt_writer._validated_predecessor_binding_at_receipt(manifest)


def test_receipt_requires_chain_binding_for_matched_panel_launch():
    with pytest.raises(ValueError, match="lacks predecessor receipt binding"):
        receipt_writer._validated_predecessor_binding_at_receipt(
            {"purpose": "bounded UDLM training pilot"}
        )
    assert (
        receipt_writer._validated_predecessor_binding_at_receipt(
            {"purpose": "registered UDLM optimization screen"}
        )
        is None
    )


def _valid_summary(paths, revision):
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    _write_launch_manifest(paths)
    manifest_evidence = {
        **_snapshot(paths["manifest"]),
        "selected_gpu_uuids": EXPECTED_SELECTED_GPU_UUIDS,
    }
    runtime_path = paths["run_dir"] / "runtime_config.json"
    paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    if not paths["checkpoint"].exists():
        paths["checkpoint"].write_bytes(b"stable checkpoint fixture\n")
    completion_contract = {
        "summary_schema_version": launcher.TRAINING_SUMMARY_SCHEMA_VERSION,
        "summary_path": str(paths["summary"]),
        "final_checkpoint_path": str(paths["checkpoint"]),
        "expected_max_steps": 10,
        "expected_world_size": 1,
        "fail_on_nonfinite_loss": True,
        "backward_anomaly_detection": True,
    }
    runtime_record = {
        "schema_version": receipt_writer.RUNTIME_CONFIG_SCHEMA_VERSION,
        "status": "preflight_completed",
        "source_revision": revision,
        "source": {"head": revision, "upstream": revision},
        "training_argv": TRAINING_ARGV,
        "observed_training_argv": TRAINING_ARGV,
        "training_argv_sha256": EXPECTED_ARGV_SHA256,
        "resolved_training_config": RESOLVED_TRAINING_CONFIG,
        "resolved_training_config_sha256": EXPECTED_CONFIG_SHA256,
        "launch_manifest": manifest_evidence,
        "completion_contract": completion_contract,
        "python_environment": {"PYTHONHASHSEED": "7"},
    }
    runtime_path.write_text(
        json.dumps(runtime_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "schema_version": launcher.TRAINING_SUMMARY_SCHEMA_VERSION,
        "status": "completed",
        "completed_at_utc": "2020-01-01T00:01:00+00:00",
        "source_revision": revision,
        "source": {"head": revision, "upstream": revision},
        "resolved_training_config_sha256": EXPECTED_CONFIG_SHA256,
        "training_argv_sha256": EXPECTED_ARGV_SHA256,
        "launch_manifest": manifest_evidence,
        "completion_contract": completion_contract,
        "runtime_config": {
            **_snapshot(runtime_path),
            "schema_version": receipt_writer.RUNTIME_CONFIG_SCHEMA_VERSION,
            "record_sha256": _canonical_sha256(runtime_record),
        },
        "observed_training_state": {
            "global_rank": 0,
            "global_step": 10,
            "world_size": 1,
        },
        "training_accounting": {
            "training_seed": 7,
            "optimizer_updates": 10,
            "world_size": 1,
            "micro_batch_size_per_rank": 2,
            "accumulate_grad_batches": 4,
            "effective_global_examples_per_optimizer_step": 8,
            "total_requested_example_exposures": 80,
            "hosted_stream_rank_partition_policy": (
                "huggingface_split_dataset_by_node_disjoint_rank_streams"
            ),
            "trainable_parameter_counts": {
                "base_backbone": 3,
                "time_conditioner": 2,
                "total": 5,
            },
        },
        "training_health": {
            "scope": "rank-zero counters",
            "all_losses_finite": True,
            "all_observed_gradients_finite": True,
            "every_optimizer_step_had_a_nonzero_gradient": True,
            "loss_checks": 10,
            "optimizer_step_checks": 10,
            "gradient_tensor_observations": 10,
            "gradient_element_observations": 20,
        },
        "conditioning_gradient_audit": None,
        "screen_initialization_state_audit": None,
        "final_checkpoint": {
            **_snapshot(paths["checkpoint"]),
            "semantic_audit": {
                "deserialized": True,
                "global_step": 10,
                "raw_model": _finite_record(),
                "ema": _finite_record(),
                "ema_metadata": {
                    "shadow_parameter_count": 2,
                    "decay": 0.9999,
                    "num_updates": 10,
                },
                "optimizer": _finite_record(),
                "non_sentinel_checkpoint_tensors": _finite_record(
                    tensors=6, elements=12
                ),
                "framework_nonfinite_sentinels": (
                    receipt_writer._expected_framework_nonfinite_sentinels(
                        expected_steps=10
                    )
                ),
                **_auxiliary_checkpoint_records(
                    resolved_training_config=RESOLVED_TRAINING_CONFIG
                ),
                "udlm_process_identity_verified": True,
                "live_model_match": {
                    "exact_key_set": True,
                    "exact_tensor_values": True,
                    "tensor_count": 2,
                },
                "live_ema_match": {
                    "exact_tensor_values": True,
                    "tensor_count": 2,
                },
            },
        },
        "tensor_finiteness": {
            "raw_model": _finite_record(),
            "ema": _finite_record(),
        },
        "startup": {
            "mode": "warm_start",
            "verified_mdlm_warm_start_report": {
                "source_path": "/project/mdlm.ckpt",
                "source_resolved_path": "/project/mdlm.ckpt",
                "source_sha256": EXPECTED_WARM_START_SHA256,
                "source_size_bytes": 123,
                "expected_source_sha256": EXPECTED_WARM_START_SHA256,
                "byte_identity_verified_before_and_after_load": True,
                "weights": "ema",
                "parameter_tensors": 2,
            },
        },
    }


def _write_summary(paths, revision):
    paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary"].write_text(
        json.dumps(_valid_summary(paths, revision), sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_summary_direct(
    paths,
    revision,
    summary,
    *,
    resolved_training_config=RESOLVED_TRAINING_CONFIG,
):
    return receipt_writer.validate_training_summary(
        summary,
        summary_path=paths["summary"],
        expected_schema_version=receipt_writer.TRAINING_SUMMARY_SCHEMA_VERSION,
        expected_source_revision=revision,
        expected_config_sha256=_canonical_sha256(resolved_training_config),
        expected_argv_sha256=EXPECTED_ARGV_SHA256,
        expected_launch_manifest_path=paths["manifest"],
        expected_launch_manifest_sha256=hashlib.sha256(
            paths["manifest"].read_bytes()
        ).hexdigest(),
        expected_selected_gpu_uuids=EXPECTED_SELECTED_GPU_UUIDS,
        expected_max_steps=10,
        expected_world_size=1,
        expected_final_checkpoint_path=paths["checkpoint"],
        expected_initialization_checkpoint_sha256=EXPECTED_WARM_START_SHA256,
        resolved_training_config=resolved_training_config,
        launch_manifest=json.loads(paths["manifest"].read_text(encoding="utf-8")),
    )


def _shell_command(
    paths,
    revision,
    *,
    training_command,
    log_path=None,
    expected_selected_gpu_uuids_json=EXPECTED_SELECTED_GPU_UUIDS_JSON,
):
    expected_manifest_sha256 = _write_launch_manifest(paths)
    _lock_record, expected_lock_sha256 = _write_training_job_lock(paths)
    return launcher.build_tmux_shell_command(
        training_command,
        log_path=paths["log"] if log_path is None else log_path,
        training_summary_path=paths["summary"],
        exit_receipt_path=paths["receipt"],
        expected_source_revision=revision,
        expected_config_sha256=EXPECTED_CONFIG_SHA256,
        expected_argv_sha256=EXPECTED_ARGV_SHA256,
        expected_summary_schema_version=launcher.TRAINING_SUMMARY_SCHEMA_VERSION,
        expected_max_steps=10,
        expected_world_size=1,
        expected_final_checkpoint_path=paths["checkpoint"],
        expected_launch_manifest_path=paths["manifest"],
        expected_launch_manifest_sha256=expected_manifest_sha256,
        expected_selected_gpu_uuids_json=expected_selected_gpu_uuids_json,
        expected_training_job_lock_path=paths["lock"],
        expected_training_job_lock_sha256=expected_lock_sha256,
        expected_initialization_checkpoint_sha256=EXPECTED_WARM_START_SHA256,
    )


def _execute_shell(repository, shell_command):
    return subprocess.run(
        ["bash", "-lc", shell_command],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("field_path", "error_fragment"),
    [
        (
            ("launch_manifest",),
            "training summary launch manifest evidence keys are invalid",
        ),
        (("runtime_config",), "runtime config evidence keys are invalid"),
        (("final_checkpoint",), "final checkpoint evidence keys are invalid"),
        (("source",), "training summary source keys are invalid"),
        (
            ("completion_contract",),
            "training summary completion contract keys are invalid",
        ),
        (("observed_training_state",), "observed training state keys are invalid"),
        (("training_health",), "training health evidence keys are invalid"),
        (
            ("tensor_finiteness",),
            "live tensor finiteness evidence keys are invalid",
        ),
        (
            ("tensor_finiteness", "raw_model"),
            "live raw model keys are invalid",
        ),
        (
            ("final_checkpoint", "semantic_audit", "live_model_match"),
            "checkpoint/live-model match keys are invalid",
        ),
        (
            ("final_checkpoint", "semantic_audit", "live_ema_match"),
            "checkpoint/live-EMA match keys are invalid",
        ),
        (
            ("startup", "verified_mdlm_warm_start_report"),
            "MDLM warm-start report keys are invalid",
        ),
    ],
)
def test_training_summary_rejects_undeclared_nested_extensions(
    receipt_repository, field_path, error_fragment
):
    repository, revision = receipt_repository
    paths = _paths(repository)
    summary = _valid_summary(paths, revision)
    target = summary
    for field in field_path:
        target = target[field]
    target["unexpected_extension"] = True

    with pytest.raises(ValueError, match=error_fragment):
        _validate_summary_direct(paths, revision, summary)


def test_additive_warm_start_rejects_film_only_variant_and_count(
    receipt_repository,
):
    repository, revision = receipt_repository
    paths = _paths(repository)
    summary = _valid_summary(paths, revision)
    summary["startup"]["verified_mdlm_warm_start_report"].update(
        {
            "conditioning_variant": "additive",
            "conditioning_parameter_tensors": 2,
        }
    )

    with pytest.raises(ValueError, match="MDLM warm-start report keys are invalid"):
        _validate_summary_direct(paths, revision, summary)


@pytest.mark.parametrize(
    ("field", "error_fragment"),
    [
        ("raw_model", "live and serialized raw-model finiteness evidence disagree"),
        ("ema", "live and serialized EMA finiteness evidence disagree"),
    ],
)
def test_top_level_finiteness_must_equal_checkpoint_semantic_record(
    receipt_repository, field, error_fragment
):
    repository, revision = receipt_repository
    paths = _paths(repository)
    summary = _valid_summary(paths, revision)
    summary["tensor_finiteness"][field]["floating_element_count"] += 1

    with pytest.raises(ValueError, match=error_fragment):
        _validate_summary_direct(paths, revision, summary)


@pytest.mark.parametrize("field_path", [(), ("source",)])
def test_runtime_config_rejects_undeclared_root_or_source_extensions(
    receipt_repository, field_path
):
    repository, revision = receipt_repository
    paths = _paths(repository)
    summary = _valid_summary(paths, revision)
    runtime_path = paths["run_dir"] / "runtime_config.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    target = runtime
    for field in field_path:
        target = target[field]
    target["unexpected_extension"] = True

    with pytest.raises(ValueError, match="keys are invalid"):
        receipt_writer.validate_runtime_config(
            runtime,
            expected_source_revision=revision,
            expected_config_sha256=EXPECTED_CONFIG_SHA256,
            expected_argv_sha256=EXPECTED_ARGV_SHA256,
            expected_launch_manifest_path=paths["manifest"],
            expected_launch_manifest_sha256=hashlib.sha256(
                paths["manifest"].read_bytes()
            ).hexdigest(),
            expected_selected_gpu_uuids=EXPECTED_SELECTED_GPU_UUIDS,
            expected_completion_contract=summary["completion_contract"],
        )


def test_explicit_null_gradient_clip_algorithm_normalizes_to_effective_norm(
    receipt_repository,
):
    repository, revision = receipt_repository
    paths = _paths(repository)
    summary = _valid_summary(paths, revision)
    resolved_config = copy.deepcopy(RESOLVED_TRAINING_CONFIG)
    resolved_config["trainer"]["gradient_clip_algorithm"] = None
    resolved_config_sha256 = _canonical_sha256(resolved_config)
    summary["resolved_training_config_sha256"] = resolved_config_sha256
    semantic = summary["final_checkpoint"]["semantic_audit"]
    semantic.update(
        _auxiliary_checkpoint_records(resolved_training_config=resolved_config)
    )

    bindings = _validate_summary_direct(
        paths,
        revision,
        summary,
        resolved_training_config=resolved_config,
    )
    assert bindings["schema_version"] == receipt_writer.TRAINING_SUMMARY_SCHEMA_VERSION
    assert (
        semantic["trainer_live_configuration_match"]["gradient_clip_algorithm"]
        == "norm"
    )

    semantic["trainer_live_configuration_match"]["gradient_clip_algorithm"] = None
    with pytest.raises(ValueError, match="live Trainer configuration match is invalid"):
        _validate_summary_direct(
            paths,
            revision,
            summary,
            resolved_training_config=resolved_config,
        )


def test_successful_pipeline_writes_launch_bound_receipt(receipt_repository):
    repository, revision = receipt_repository
    paths = _paths(repository)
    _write_summary(paths, revision)
    paths["log"].parent.mkdir(parents=True)
    summary_sha256 = hashlib.sha256(paths["summary"].read_bytes()).hexdigest()
    expected_accounting = json.loads(paths["summary"].read_text(encoding="utf-8"))[
        "training_accounting"
    ]

    result = _execute_shell(
        repository,
        _shell_command(paths, revision, training_command=["bash", "-c", "exit 0"]),
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    assert receipt["schema_version"] == receipt_writer.EXIT_STATUS_SCHEMA_VERSION == 5
    assert receipt["status"] == "completed"
    assert receipt["process_exit_status"] == 0
    assert receipt["pipeline"]["training"]["shell_exit_status"] == 0
    assert receipt["pipeline"]["tee"]["shell_exit_status"] == 0
    assert receipt["training_summary"]["valid_and_launch_bound"] is True
    assert receipt["training_summary"]["artifact"]["sha256"] == summary_sha256
    assert (
        receipt["training_summary"]["validated_bindings"]["training_accounting"]
        == expected_accounting
    )
    assert receipt["runtime_config"]["matches_training_summary_snapshot"] is True
    assert receipt["launch_manifest"]["valid_and_launch_bound"] is True
    assert receipt["predecessor_receipt_binding"] is None
    assert (
        receipt["completion_requirements"][
            "predecessor_receipt_binding_unchanged_and_valid"
        ]
        is True
    )
    assert receipt["launch_manifest"]["matches_runtime_config_snapshot"] is True
    assert (
        receipt["training_job_lock"][
            "valid_and_launch_bound_before_receipt_publication"
        ]
        is True
    )
    assert receipt["training_job_lock"]["artifact"]["sha256"] == EXPECTED_LOCK_SHA256
    assert receipt["final_checkpoint"]["matches_training_summary_snapshot"] is True
    assert receipt["source_at_receipt"]["verified"] is True
    assert not paths["lock"].exists()
    assert receipt["expected_contract"] == {
        "training_summary_schema_version": launcher.TRAINING_SUMMARY_SCHEMA_VERSION,
        "source_revision": revision,
        "resolved_training_config_sha256": EXPECTED_CONFIG_SHA256,
        "training_argv_sha256": EXPECTED_ARGV_SHA256,
        "launch_manifest_path": str(paths["manifest"]),
        "launch_manifest_sha256": hashlib.sha256(
            paths["manifest"].read_bytes()
        ).hexdigest(),
        "selected_gpu_uuids": EXPECTED_SELECTED_GPU_UUIDS,
        "training_job_lock_path": str(paths["lock"]),
        "training_job_lock_sha256": EXPECTED_LOCK_SHA256,
        "max_steps": 10,
        "world_size": 1,
        "training_summary_path": str(paths["summary"]),
        "final_checkpoint_path": str(paths["checkpoint"]),
        "initialization_checkpoint_sha256": EXPECTED_WARM_START_SHA256,
    }


@pytest.mark.parametrize(
    "completed_at_utc",
    [
        "2020-01-01T00:00:00+00:00",
        "2019-12-31T23:59:59+00:00",
        "2999-01-01T00:00:00+00:00",
    ],
    ids=("equal_to_manifest", "before_manifest", "after_receipt"),
)
def test_current_run_timestamps_must_be_strictly_ordered(
    receipt_repository, completed_at_utc
):
    repository, revision = receipt_repository
    paths = _paths(repository, f"chronology_{completed_at_utc[:4]}")
    summary = _valid_summary(paths, revision)
    summary["completed_at_utc"] = completed_at_utc
    paths["summary"].write_text(json.dumps(summary) + "\n", encoding="utf-8")
    paths["log"].parent.mkdir(parents=True, exist_ok=True)

    result = _execute_shell(
        repository,
        _shell_command(paths, revision, training_command=["bash", "-c", "exit 0"]),
    )

    assert result.returncode == receipt_writer.INCOMPLETE_EXIT_STATUS
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert (
        "timestamps must be strictly ordered"
        in receipt["training_summary"]["validation_error"]
    )


def test_training_failure_is_recorded_even_when_summary_is_valid(receipt_repository):
    repository, revision = receipt_repository
    paths = _paths(repository)
    _write_summary(paths, revision)
    paths["log"].parent.mkdir(parents=True)

    result = _execute_shell(
        repository,
        _shell_command(paths, revision, training_command=["bash", "-c", "exit 23"]),
    )

    assert result.returncode == 23
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["pipeline"]["training"]["shell_exit_status"] == 23
    assert receipt["pipeline"]["tee"]["shell_exit_status"] == 0
    assert receipt["training_summary"]["valid_and_launch_bound"] is True
    assert not paths["lock"].exists()


@pytest.mark.parametrize("replacement_mode", ["wrong_bytes", "replaced_file"])
def test_wrong_or_replaced_training_job_lock_is_never_unlinked(
    receipt_repository, replacement_mode
):
    repository, revision = receipt_repository
    paths = _paths(repository, f"wrong_lock_{replacement_mode}")
    _write_summary(paths, revision)
    paths["log"].parent.mkdir(parents=True)
    shell_command = _shell_command(
        paths,
        revision,
        training_command=["bash", "-c", "exit 0"],
    )
    wrong_bytes = b'{"status":"held","owner_token":"another-run"}\n'
    if replacement_mode == "replaced_file":
        paths["lock"].rename(paths["lock"].with_suffix(".original"))
    paths["lock"].write_bytes(wrong_bytes)

    result = _execute_shell(repository, shell_command)

    assert result.returncode != 0
    assert paths["lock"].read_bytes() == wrong_bytes
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert (
        receipt["training_job_lock"][
            "valid_and_launch_bound_before_receipt_publication"
        ]
        is False
    )
    assert (
        "training-job lock raw SHA-256"
        in receipt["training_job_lock"]["validation_error"]
    )


def test_training_job_lock_replaced_after_receipt_publication_is_not_unlinked(
    receipt_repository, monkeypatch
):
    repository, revision = receipt_repository
    paths = _paths(repository, "lock_release_race")
    _write_summary(paths, revision)
    monkeypatch.setattr(receipt_writer, "REPOSITORY_ROOT", repository)
    original_publish = receipt_writer._atomic_write_json_exclusive
    replacement_bytes = b'{"status":"held","owner_token":"replacement"}\n'

    def publish_then_replace_lock(path, value):
        original_publish(path, value)
        paths["lock"].rename(paths["lock"].with_suffix(".original"))
        paths["lock"].write_bytes(replacement_bytes)

    monkeypatch.setattr(
        receipt_writer,
        "_atomic_write_json_exclusive",
        publish_then_replace_lock,
    )
    argv = [
        "--training-exit-status",
        "0",
        "--tee-exit-status",
        "0",
        "--training-summary-path",
        str(paths["summary"]),
        "--receipt-path",
        str(paths["receipt"]),
        "--expected-summary-schema-version",
        str(receipt_writer.TRAINING_SUMMARY_SCHEMA_VERSION),
        "--expected-source-revision",
        revision,
        "--expected-config-sha256",
        EXPECTED_CONFIG_SHA256,
        "--expected-argv-sha256",
        EXPECTED_ARGV_SHA256,
        "--expected-launch-manifest-path",
        str(paths["manifest"]),
        "--expected-launch-manifest-sha256",
        hashlib.sha256(paths["manifest"].read_bytes()).hexdigest(),
        "--expected-selected-gpu-uuids-json",
        EXPECTED_SELECTED_GPU_UUIDS_JSON,
        "--training-job-lock-path",
        str(paths["lock"]),
        "--expected-training-job-lock-sha256",
        EXPECTED_LOCK_SHA256,
        "--expected-max-steps",
        "10",
        "--expected-world-size",
        "1",
        "--expected-final-checkpoint-path",
        str(paths["checkpoint"]),
        "--expected-initialization-checkpoint-sha256",
        EXPECTED_WARM_START_SHA256,
    ]

    with pytest.raises(ValueError, match="no longer matches"):
        receipt_writer.main(argv)

    assert paths["receipt"].is_file()
    assert paths["lock"].read_bytes() == replacement_bytes


@pytest.mark.parametrize(
    "training_command",
    [
        ["bash", "-c", "kill -TERM $$"],
        ["bash", "-c", "exit 143"],
    ],
)
def test_signal_compatible_training_status_does_not_overclaim_provenance(
    receipt_repository, training_command
):
    repository, revision = receipt_repository
    paths = _paths(repository)
    _write_summary(paths, revision)
    paths["log"].parent.mkdir(parents=True)

    result = _execute_shell(
        repository,
        _shell_command(
            paths,
            revision,
            training_command=training_command,
        ),
    )

    assert result.returncode == 143
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    training = receipt["pipeline"]["training"]
    assert training["shell_exit_status"] == 143
    assert training["possible_termination_signal"] == 15
    assert training["shell_status_is_signal_compatible"] is True
    assert training["signal_provenance"] == "ambiguous_exit_or_signal"


def test_tee_failure_is_recorded_separately(receipt_repository):
    repository, revision = receipt_repository
    paths = _paths(repository)
    _write_summary(paths, revision)

    result = _execute_shell(
        repository,
        _shell_command(
            paths,
            revision,
            training_command=["bash", "-c", "printf payload"],
            log_path=Path("/dev/full"),
        ),
    )

    assert result.returncode != 0
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["pipeline"]["training"]["shell_exit_status"] == 0
    assert receipt["pipeline"]["tee"]["shell_exit_status"] != 0
    assert receipt["process_exit_status"] == result.returncode


def test_missing_summary_writes_incomplete_receipt_and_exits_97(receipt_repository):
    repository, revision = receipt_repository
    paths = _paths(repository)
    paths["log"].parent.mkdir(parents=True)

    result = _execute_shell(
        repository,
        _shell_command(paths, revision, training_command=["bash", "-c", "exit 0"]),
    )

    assert result.returncode == 97
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["training_summary"]["present"] is False
    assert receipt["training_summary"]["valid_and_launch_bound"] is False
    assert "FileNotFoundError" in receipt["training_summary"]["validation_error"]


def test_summary_must_match_launch_steps_and_world_size(receipt_repository):
    repository, revision = receipt_repository
    paths = _paths(repository)
    summary = _valid_summary(paths, revision)
    summary["observed_training_state"]["global_step"] = 9
    paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary"].write_text(json.dumps(summary) + "\n", encoding="utf-8")
    paths["log"].parent.mkdir(parents=True)

    result = _execute_shell(
        repository,
        _shell_command(paths, revision, training_command=["bash", "-c", "exit 0"]),
    )

    assert result.returncode == 97
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    assert receipt["training_summary"]["valid_and_launch_bound"] is False
    assert "observed global step" in receipt["training_summary"]["validation_error"]


def test_required_health_semantic_runtime_and_startup_evidence_cannot_be_forged(
    receipt_repository,
):
    repository, revision = receipt_repository
    mutations = [
        (
            ("completion_contract", "fail_on_nonfinite_loss"),
            False,
            "completion fail-on-nonfinite-loss flag",
        ),
        (("training_health",), None, "training summary keys are invalid"),
        (
            ("training_health", "all_losses_finite"),
            False,
            "all-losses-finite flag",
        ),
        (
            ("training_health", "gradient_tensor_observations"),
            0,
            "gradient tensor observation count",
        ),
        (("training_accounting",), None, "training summary keys are invalid"),
        (
            ("training_accounting", "optimizer_updates"),
            True,
            "accounting optimizer updates",
        ),
        (
            (
                "training_accounting",
                "effective_global_examples_per_optimizer_step",
            ),
            7,
            "effective global examples do not equal",
        ),
        (
            ("training_accounting", "total_requested_example_exposures"),
            79,
            "total requested example exposures do not equal",
        ),
        (
            ("training_accounting", "hosted_stream_rank_partition_policy"),
            "unpartitioned_repeated_stream",
            "hosted-stream rank partition policy",
        ),
        (
            ("training_accounting", "trainable_parameter_counts", "total"),
            6,
            "trainable parameter counts do not add up",
        ),
        (
            (
                "training_accounting",
                "trainable_parameter_counts",
                "time_conditioner",
            ),
            None,
            "trainable parameter counts keys are invalid",
        ),
        (
            ("tensor_finiteness", "ema", "floating_tensor_count"),
            0,
            "live EMA floating tensor count",
        ),
        (
            ("final_checkpoint", "semantic_audit"),
            {},
            "final checkpoint semantic audit",
        ),
        (
            ("final_checkpoint", "semantic_audit", "optimizer", "all_finite"),
            False,
            "serialized checkpoint optimizer all-finite flag",
        ),
        (
            ("final_checkpoint", "semantic_audit", "ema_metadata", "num_updates"),
            9,
            "checkpoint EMA update count",
        ),
        (
            ("final_checkpoint", "semantic_audit", "ema_metadata", "decay"),
            0.9,
            "EMA decay disagrees with resolved config",
        ),
        (
            ("final_checkpoint", "semantic_audit", "live_model_match", "tensor_count"),
            0,
            "live-model match tensor count",
        ),
        (
            ("runtime_config", "record_sha256"),
            "not-a-sha256",
            "runtime config canonical record digest",
        ),
        (("startup",), None, "training summary keys are invalid"),
        (
            (
                "startup",
                "verified_mdlm_warm_start_report",
                "source_sha256",
            ),
            "a" * 64,
            "warm-start launch-pinned source digest",
        ),
    ]
    for index, (field_path, replacement, error_fragment) in enumerate(mutations):
        paths = _paths(repository, f"invalid_evidence_{index}")
        summary = _valid_summary(paths, revision)
        parent = summary
        for key in field_path[:-1]:
            parent = parent[key]
        if replacement is None:
            parent.pop(field_path[-1])
        else:
            parent[field_path[-1]] = replacement
        paths["summary"].write_text(json.dumps(summary) + "\n", encoding="utf-8")
        paths["log"].parent.mkdir(parents=True, exist_ok=True)

        result = _execute_shell(
            repository,
            _shell_command(
                paths,
                revision,
                training_command=["bash", "-c", "exit 0"],
            ),
        )

        assert result.returncode == 97
        receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
        assert error_fragment in receipt["training_summary"]["validation_error"]


@pytest.mark.parametrize("artifact_change", ["missing", "replaced"])
def test_checkpoint_must_still_match_the_summary_snapshot(
    receipt_repository, artifact_change
):
    repository, revision = receipt_repository
    paths = _paths(repository)
    _write_summary(paths, revision)
    if artifact_change == "missing":
        paths["checkpoint"].unlink()
    else:
        paths["checkpoint"].write_bytes(b"replacement checkpoint bytes\n")
    paths["log"].parent.mkdir(parents=True)

    result = _execute_shell(
        repository,
        _shell_command(paths, revision, training_command=["bash", "-c", "exit 0"]),
    )

    assert result.returncode == 97
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    assert receipt["training_summary"]["valid_and_launch_bound"] is False
    if artifact_change == "missing":
        assert "FileNotFoundError" in receipt["training_summary"]["validation_error"]
    else:
        assert "no longer matches" in receipt["training_summary"]["validation_error"]
        assert (
            receipt["final_checkpoint"]["artifact"]["sha256"]
            == hashlib.sha256(paths["checkpoint"].read_bytes()).hexdigest()
        )


def test_launch_manifest_must_still_match_the_launch_and_summary(
    receipt_repository,
):
    repository, revision = receipt_repository
    paths = _paths(repository)
    _write_summary(paths, revision)
    paths["log"].parent.mkdir(parents=True)
    shell_command = _shell_command(
        paths,
        revision,
        training_command=["bash", "-c", "exit 0"],
    )
    paths["manifest"].write_text(
        json.dumps(
            {
                "launch_manifest_schema_version": 1,
                "user_requested_gpu_count": 1,
                "cuda_visible_device_uuids": ["GPU-replacement"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = _execute_shell(repository, shell_command)

    assert result.returncode == 97
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    assert receipt["launch_manifest"]["valid_and_launch_bound"] is False
    assert receipt["training_summary"]["valid_and_launch_bound"] is False
    assert "raw SHA-256" in receipt["launch_manifest"]["validation_error"]


def test_launch_manifest_selected_uuids_must_match_receipt_contract(
    receipt_repository,
):
    repository, revision = receipt_repository
    paths = _paths(repository)
    _write_summary(paths, revision)
    paths["log"].parent.mkdir(parents=True)

    result = _execute_shell(
        repository,
        _shell_command(
            paths,
            revision,
            training_command=["bash", "-c", "exit 0"],
            expected_selected_gpu_uuids_json='["GPU-other"]',
        ),
    )

    assert result.returncode == 97
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    assert receipt["launch_manifest"]["valid_and_launch_bound"] is False
    assert "selected GPU UUIDs" in receipt["launch_manifest"]["validation_error"]


def test_launch_manifest_change_during_final_receipt_reread_fails_closed(
    receipt_repository, monkeypatch
):
    repository, revision = receipt_repository
    paths = _paths(repository)
    _write_summary(paths, revision)
    expected_manifest_sha256 = hashlib.sha256(
        paths["manifest"].read_bytes()
    ).hexdigest()
    args = receipt_writer._parse_args(
        [
            "--training-exit-status",
            "0",
            "--tee-exit-status",
            "0",
            "--training-summary-path",
            str(paths["summary"]),
            "--receipt-path",
            str(paths["receipt"]),
            "--expected-summary-schema-version",
            str(receipt_writer.TRAINING_SUMMARY_SCHEMA_VERSION),
            "--expected-source-revision",
            revision,
            "--expected-config-sha256",
            EXPECTED_CONFIG_SHA256,
            "--expected-argv-sha256",
            EXPECTED_ARGV_SHA256,
            "--expected-launch-manifest-path",
            str(paths["manifest"]),
            "--expected-launch-manifest-sha256",
            expected_manifest_sha256,
            "--expected-selected-gpu-uuids-json",
            EXPECTED_SELECTED_GPU_UUIDS_JSON,
            "--training-job-lock-path",
            str(paths["lock"]),
            "--expected-training-job-lock-sha256",
            EXPECTED_LOCK_SHA256,
            "--expected-max-steps",
            "10",
            "--expected-world-size",
            "1",
            "--expected-final-checkpoint-path",
            str(paths["checkpoint"]),
            "--expected-initialization-checkpoint-sha256",
            EXPECTED_WARM_START_SHA256,
        ]
    )
    monkeypatch.setattr(receipt_writer, "REPOSITORY_ROOT", repository)
    original_snapshot = receipt_writer.stable_file_snapshot
    manifest_reads = 0

    def mutate_before_final_manifest_read(path, *, capture_bytes=False):
        nonlocal manifest_reads
        if Path(path) == paths["manifest"]:
            manifest_reads += 1
            if manifest_reads == 2:
                paths["manifest"].write_bytes(paths["manifest"].read_bytes() + b" ")
        return original_snapshot(path, capture_bytes=capture_bytes)

    monkeypatch.setattr(
        receipt_writer,
        "stable_file_snapshot",
        mutate_before_final_manifest_read,
    )

    receipt, status = receipt_writer.build_exit_receipt(args)

    assert status == receipt_writer.INCOMPLETE_EXIT_STATUS
    # Initial validation, summary/runtime join, and final pre-publication
    # predecessor-chain validation each take an independent stable snapshot.
    assert manifest_reads == 3
    assert receipt["launch_manifest"]["valid_and_launch_bound"] is False
    assert (
        "changed during receipt validation"
        in receipt["launch_manifest"]["validation_error"]
    )
    assert (
        "changed during receipt validation"
        in receipt["training_summary"]["validation_error"]
    )


def test_runtime_record_must_semantically_match_the_launch(receipt_repository):
    repository, revision = receipt_repository
    paths = _paths(repository)
    summary = _valid_summary(paths, revision)
    runtime_path = paths["run_dir"] / "runtime_config.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["source_revision"] = "0" * 40
    runtime_path.write_text(json.dumps(runtime) + "\n", encoding="utf-8")
    summary["runtime_config"] = {
        **_snapshot(runtime_path),
        "schema_version": receipt_writer.RUNTIME_CONFIG_SCHEMA_VERSION,
        "record_sha256": _canonical_sha256(runtime),
    }
    paths["summary"].write_text(json.dumps(summary) + "\n", encoding="utf-8")
    paths["log"].parent.mkdir(parents=True)

    result = _execute_shell(
        repository,
        _shell_command(paths, revision, training_command=["bash", "-c", "exit 0"]),
    )

    assert result.returncode == 97
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    assert receipt["runtime_config"]["matches_training_summary_snapshot"] is True
    assert receipt["runtime_config"]["semantic_validation_passed"] is False
    assert (
        "runtime config source revision"
        in receipt["training_summary"]["validation_error"]
    )


def test_training_accounting_must_match_resolved_runtime_config(
    receipt_repository,
):
    repository, revision = receipt_repository
    paths = _paths(repository)
    summary = _valid_summary(paths, revision)
    mismatched_config = json.loads(json.dumps(RESOLVED_TRAINING_CONFIG))
    mismatched_config["loader"]["batch_size"] = 3
    mismatched_config_sha256 = _canonical_sha256(mismatched_config)
    summary["resolved_training_config_sha256"] = mismatched_config_sha256
    summary["final_checkpoint"]["semantic_audit"]["checkpoint_hyperparameters_match"][
        "resolved_config_sha256"
    ] = mismatched_config_sha256

    with pytest.raises(
        ValueError, match="accounting micro-batch size disagrees with resolved config"
    ):
        receipt_writer.validate_training_summary(
            summary,
            summary_path=paths["summary"],
            expected_schema_version=receipt_writer.TRAINING_SUMMARY_SCHEMA_VERSION,
            expected_source_revision=revision,
            expected_config_sha256=mismatched_config_sha256,
            expected_argv_sha256=EXPECTED_ARGV_SHA256,
            expected_launch_manifest_path=paths["manifest"],
            expected_launch_manifest_sha256=hashlib.sha256(
                paths["manifest"].read_bytes()
            ).hexdigest(),
            expected_selected_gpu_uuids=EXPECTED_SELECTED_GPU_UUIDS,
            expected_max_steps=10,
            expected_world_size=1,
            expected_final_checkpoint_path=paths["checkpoint"],
            expected_initialization_checkpoint_sha256=(EXPECTED_WARM_START_SHA256),
            resolved_training_config=mismatched_config,
            launch_manifest=json.loads(paths["manifest"].read_text(encoding="utf-8")),
        )


def test_receipt_v4_validates_and_echoes_film_gradient_certificate(
    receipt_repository,
):
    repository, revision = receipt_repository
    paths = _paths(repository)
    summary = _valid_summary(paths, revision)
    contract, contract_sha256, audit = _film_gradient_contract_and_audit()
    launch_manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    launch_manifest["optimization_screen"] = {
        "conditioning_gradient_contract": contract,
        "conditioning_gradient_contract_sha256": contract_sha256,
    }
    resolved_config = json.loads(json.dumps(RESOLVED_TRAINING_CONFIG))
    resolved_config["training"].update(
        {
            "reseed_after_model_initialization": True,
            "udlm": {"conditioning_variant": "film_adaln"},
        }
    )
    resolved_config_sha256 = _canonical_sha256(resolved_config)
    summary["resolved_training_config_sha256"] = resolved_config_sha256
    summary["final_checkpoint"]["semantic_audit"]["checkpoint_hyperparameters_match"][
        "resolved_config_sha256"
    ] = resolved_config_sha256
    summary["training_accounting"]["trainable_parameter_counts"] = {
        "base_backbone": 3,
        "time_conditioner": 12,
        "film_modulation": 12,
        "total": 27,
    }
    summary["startup"]["training_rng_policy"] = {
        "policy": "reseed_all_training_rng_streams_after_model_and_warm_start",
        "seed": 7,
        "purpose": "isolate_training_randomness_from_architecture_constructor_draws",
        "applied_before_dataloader_and_trainer_construction": True,
    }
    summary["startup"]["verified_mdlm_warm_start_report"].update(
        {
            "conditioning_variant": "film_adaln",
            "conditioning_parameter_tensors": 6,
        }
    )
    summary["conditioning_gradient_audit"] = audit
    state_audit = {
        "schema_version": 1,
        "phase": (
            "after_verified_mdlm_ema_warm_start_before_training_rng_reseed_and_optimizer_creation"
        ),
        "source_checkpoint_sha256": EXPECTED_WARM_START_SHA256,
        "resolved_training_config_sha256": resolved_config_sha256,
        "training_seed": 7,
        "conditioning_variant": "film_adaln",
        "common_backbone_tensor_count": 100,
        "common_backbone_state_sha256": "8" * 64,
        "full_initial_tensor_count": 128,
        "full_initial_state_sha256": "9" * 64,
    }
    summary["screen_initialization_state_audit"] = state_audit

    def validate():
        return receipt_writer.validate_training_summary(
            summary,
            summary_path=paths["summary"],
            expected_schema_version=receipt_writer.TRAINING_SUMMARY_SCHEMA_VERSION,
            expected_source_revision=revision,
            expected_config_sha256=resolved_config_sha256,
            expected_argv_sha256=EXPECTED_ARGV_SHA256,
            expected_launch_manifest_path=paths["manifest"],
            expected_launch_manifest_sha256=hashlib.sha256(
                paths["manifest"].read_bytes()
            ).hexdigest(),
            expected_selected_gpu_uuids=EXPECTED_SELECTED_GPU_UUIDS,
            expected_max_steps=10,
            expected_world_size=1,
            expected_final_checkpoint_path=paths["checkpoint"],
            expected_initialization_checkpoint_sha256=EXPECTED_WARM_START_SHA256,
            resolved_training_config=resolved_config,
            launch_manifest=launch_manifest,
        )

    bindings = validate()
    assert bindings["conditioning_gradient_audit"] == audit
    assert bindings["screen_initialization_state_audit"] == state_audit

    warm_start_report = summary["startup"]["verified_mdlm_warm_start_report"]
    warm_start_report["conditioning_variant"] = "additive"
    with pytest.raises(ValueError, match="warm-start conditioning variant"):
        validate()
    warm_start_report["conditioning_variant"] = "film_adaln"

    warm_start_report["conditioning_parameter_tensors"] = 5
    with pytest.raises(
        ValueError,
        match="conditioning parameter tensor count disagrees",
    ):
        validate()
    warm_start_report["conditioning_parameter_tensors"] = 6

    summary["conditioning_gradient_audit"]["optimizer_checks"][2][
        "timestep_mlp_groups"
    ][0]["all_parameter_gradients_nonzero"] = False
    with pytest.raises(ValueError, match="required conditioning gradients are zero"):
        validate()


def test_receipt_writer_rejects_legacy_training_summary_schema(receipt_repository):
    repository, revision = receipt_repository
    paths = _paths(repository)
    summary = _valid_summary(paths, revision)
    summary["schema_version"] = 1

    with pytest.raises(
        ValueError, match="unsupported training summary schema version 1; expected 5"
    ):
        receipt_writer.validate_training_summary(
            summary,
            summary_path=paths["summary"],
            expected_schema_version=1,
            expected_source_revision=revision,
            expected_config_sha256=EXPECTED_CONFIG_SHA256,
            expected_argv_sha256=EXPECTED_ARGV_SHA256,
            expected_launch_manifest_path=paths["manifest"],
            expected_launch_manifest_sha256=hashlib.sha256(
                paths["manifest"].read_bytes()
            ).hexdigest(),
            expected_selected_gpu_uuids=EXPECTED_SELECTED_GPU_UUIDS,
            expected_max_steps=10,
            expected_world_size=1,
            expected_final_checkpoint_path=paths["checkpoint"],
            expected_initialization_checkpoint_sha256=(EXPECTED_WARM_START_SHA256),
            resolved_training_config=RESOLVED_TRAINING_CONFIG,
            launch_manifest=json.loads(paths["manifest"].read_text(encoding="utf-8")),
        )


def test_dirty_source_at_receipt_cannot_complete(receipt_repository):
    repository, revision = receipt_repository
    paths = _paths(repository)
    _write_summary(paths, revision)
    paths["log"].parent.mkdir(parents=True)
    (repository / "uncommitted_source.py").write_text(
        "dirty = True\n", encoding="utf-8"
    )

    result = _execute_shell(
        repository,
        _shell_command(paths, revision, training_command=["bash", "-c", "exit 0"]),
    )

    assert result.returncode == 97
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    assert receipt["source_at_receipt"]["verified"] is False
    assert "dirty outside output" in receipt["source_at_receipt"]["error"]
    assert receipt["overall_status"] == "failed"


@pytest.mark.parametrize(
    ("payload", "error_fragment"),
    [
        ('{"schema_version":1,"schema_version":1}\n', "duplicate JSON"),
        ('{"value":NaN}\n', "non-finite JSON constant"),
        ('{"value":1e999}\n', "non-finite JSON number"),
    ],
)
def test_strict_json_failure_writes_incomplete_receipt(
    receipt_repository,
    payload,
    error_fragment,
):
    repository, revision = receipt_repository
    paths = _paths(repository)
    paths["summary"].parent.mkdir(parents=True)
    paths["summary"].write_text(payload, encoding="utf-8")
    paths["log"].parent.mkdir(parents=True)

    result = _execute_shell(
        repository,
        _shell_command(paths, revision, training_command=["bash", "-c", "exit 0"]),
    )

    assert result.returncode == 97
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    assert receipt["training_summary"]["present"] is True
    assert receipt["training_summary"]["valid_and_launch_bound"] is False
    assert error_fragment in receipt["training_summary"]["validation_error"]


def test_receipt_is_exclusive_and_never_overwritten(receipt_repository):
    repository, revision = receipt_repository
    paths = _paths(repository)
    _write_summary(paths, revision)
    paths["log"].parent.mkdir(parents=True)
    shell_command = _shell_command(
        paths,
        revision,
        training_command=["bash", "-c", "exit 0"],
    )
    first = _execute_shell(repository, shell_command)
    original = paths["receipt"].read_bytes()

    second = _execute_shell(repository, shell_command)

    assert first.returncode == 0
    assert second.returncode != 0
    assert paths["receipt"].read_bytes() == original
    assert "refusing to replace pilot exit receipt" in second.stderr

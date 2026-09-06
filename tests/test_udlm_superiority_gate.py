from __future__ import annotations

import copy
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

import pytest

from scripts.udlm import superiority_gate as gate


def _load_json(relative_path: Path) -> dict:
    return json.loads(
        (gate.REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
    )


def _inference_weights() -> dict:
    return {
        "source": "ema",
        "ema_applied": True,
        "ema": {
            "shadow_parameter_count": 202,
            "decay": 0.9999,
            "num_updates": 500,
        },
    }


def _successful_pipeline_component() -> dict:
    return {
        "shell_exit_status": 0,
        "succeeded": True,
        "possible_termination_signal": None,
        "shell_status_is_signal_compatible": False,
        "signal_provenance": None,
    }


def _training_health(*, optimizer_updates: int = 500) -> dict:
    return {
        "scope": (
            "global-rank-zero callback counters; identical fail-fast checks "
            "execute independently on every rank"
        ),
        "all_losses_finite": True,
        "all_observed_gradients_finite": True,
        "every_optimizer_step_had_a_nonzero_gradient": True,
        "loss_checks": optimizer_updates,
        "optimizer_step_checks": optimizer_updates,
        "gradient_tensor_observations": optimizer_updates,
        "gradient_element_observations": optimizer_updates,
    }


def _tensor_finiteness() -> dict:
    return {
        "raw_model": {
            "all_finite": True,
            "floating_tensor_count": 202,
            "floating_element_count": 1_000_000,
        },
        "ema": {
            "all_finite": True,
            "floating_tensor_count": 202,
            "floating_element_count": 1_000_000,
        },
    }


def _auxiliary_checkpoint_records(
    *,
    expected_steps: int,
    resolved_training_config: dict,
    parameter_state_count: int = 202,
) -> dict:
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
            "gradient_clip_algorithm": (
                "norm"
                if resolved_training_config["trainer"].get("gradient_clip_algorithm")
                is None
                else resolved_training_config["trainer"]["gradient_clip_algorithm"]
            ),
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
            "resolved_config_sha256": gate.canonical_json_sha256(
                resolved_training_config
            ),
        },
        "checkpoint_loop_state_match": {
            "exact_serialized_progress_match": True,
            "epoch": 0,
            "optimizer_steps": expected_steps,
            "accumulate_grad_batches": accumulation,
            "microbatches": expected_steps * accumulation,
        },
    }


def _python_environment(repository_root: Path, *, seed: int = 7) -> dict:
    return {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": str(seed),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONOPTIMIZE": "0",
        "PYTHONPATH": f"{repository_root / 'src'}:{repository_root}",
        "PYTHONUTF8": "1",
    }


def _candidate_lock(
    *, startup_mode: str = "warm_start", conditioning_variant: str = "additive"
) -> dict:
    sampling = {"diffusion_type": "udlm", "num_steps": gate.EXPECTED_NFE}
    implementation_inputs = {"sampler_source": {"sha256": "d" * 64}}
    metric_inputs = {"schema_version": 1}
    initialization = (
        gate.EXPECTED_BASELINE_CHECKPOINT_SHA256
        if startup_mode == "warm_start"
        else None
    )
    parameter_counts = {
        "base_model_trainable": 86_000_000,
        "time_conditioner_trainable": 787_968,
        "total_trainable": 86_787_968,
    }
    if conditioning_variant == "film_adaln":
        parameter_counts["film_modulation_trainable"] = 262_656
        parameter_counts["total_trainable"] = 87_050_624
    return {
        "schema_version": gate.CANDIDATE_LOCK_SCHEMA_VERSION,
        "candidate_id": "schedule-uniform-synthetic",
        "status": "locked_before_final_evaluation",
        "locked_at_utc": "2026-09-06T01:00:00+00:00",
        "protocol": {
            "id": gate.EXPECTED_PROTOCOL_ID,
            "sha256": gate.PROTOCOL_SHA256,
        },
        "selection": {
            "candidate_ledger": {
                "relative_path": "experiments/udlm/candidates/ledger.json",
                "sha256": "1" * 64,
                "schema_version": gate.CANDIDATE_LEDGER_SCHEMA_VERSION,
            },
            "terminal_e_exit_receipt": {
                "relative_path": "output/udlm/e-terminal/pilot_exit_status.json",
                "sha256": "0" * 64,
                "schema_version": gate.PILOT_EXIT_STATUS_SCHEMA_VERSION,
            },
            "selection_rule": gate.CANDIDATE_SELECTION_RULE,
            "checkpoint_selection_rule": gate.CHECKPOINT_SELECTION_RULE,
            "all_pilot_attempts_disclosed": True,
            "selected_without_final_seed_results": True,
            "final_seeds_used_during_selection": [],
        },
        "training": {
            "source_revision": "a" * 40,
            "training_summary": {
                "relative_path": (
                    "output/udlm/synthetic-candidate/training_summary.json"
                ),
                "sha256": "2" * 64,
                "schema_version": gate.TRAINING_SUMMARY_SCHEMA_VERSION,
            },
            "exit_receipt": {
                "relative_path": (
                    "output/udlm/synthetic-candidate/pilot_exit_status.json"
                ),
                "sha256": "3" * 64,
                "schema_version": gate.PILOT_EXIT_STATUS_SCHEMA_VERSION,
            },
            "runtime_config": {
                "relative_path": "output/udlm/synthetic-candidate/runtime_config.json",
                "sha256": "6" * 64,
                "schema_version": gate.RUNTIME_CONFIG_SCHEMA_VERSION,
            },
            "launch_manifest": {
                "relative_path": (
                    "output/udlm/synthetic-candidate/launch_manifest.json"
                ),
                "sha256": "9" * 64,
                "schema_version": gate.LAUNCH_MANIFEST_SCHEMA_VERSION,
            },
            "resolved_training_config_sha256": "7" * 64,
            "training_argv_sha256": "8" * 64,
            "checkpoint": {
                "relative_path": (
                    "output/udlm/synthetic-candidate/checkpoints/500.ckpt"
                ),
                "sha256": "4" * 64,
                "size_bytes": 1234,
                "global_step": 500,
                "weights": "ema",
            },
            "startup": {
                "mode": startup_mode,
                "initialization_checkpoint_sha256": initialization,
            },
            "training_seed": 7,
            "optimizer_updates": 500,
            "world_size": 1,
            "data_exposure": {
                "global_examples_per_optimizer_step": 2046,
                "optimizer_updates": 500,
                "total_requested_examples": 1_023_000,
                "stream_partition_policy": (
                    "huggingface_split_dataset_by_node_disjoint_rank_streams"
                ),
            },
            "parameter_counts": parameter_counts,
        },
        "inference": {
            "evaluation_config_relative_path": (
                "scripts/exps/denovo/hparams_udlm_schedule_uniform.yaml"
            ),
            "evaluation_config_sha256": "5" * 64,
            "sampling_config": sampling,
            "sampling_sha256": gate.canonical_json_sha256(sampling),
            "checkpoint_sha256": "4" * 64,
            "weights": "ema",
            "inference_weights": _inference_weights(),
            "nfe": 128,
            "final_seeds": [0, 1, 2],
            "samples_per_seed": 1000,
            "sampler_source_sha256": "d" * 64,
            "benchmark_runner_sha256": "e" * 64,
            "implementation_inputs_sha256": gate.canonical_json_sha256(
                implementation_inputs
            ),
            "metric_inputs_sha256": gate.canonical_json_sha256(metric_inputs),
            "final_run_directories_by_seed": [
                {
                    "seed": seed,
                    "relative_path": (
                        "output/udlm/final/schedule-uniform-synthetic/" f"seed_{seed}"
                    ),
                }
                for seed in gate.EXPECTED_SEEDS
            ],
        },
        "analysis": {
            "gate_source_sha256": "a" * 64,
            "report_source_sha256": "b" * 64,
            "rescore_source_sha256": "c" * 64,
            "rescore_dependency_sha256": "d" * 64,
            "benchmark_launcher_source_sha256": "e" * 64,
            "pilot_evidence_writer_source_sha256": "f" * 64,
            "scipy_version": "1.15.3",
        },
        "claim_scope": gate.CLAIM_SCOPE_BY_STARTUP[startup_mode],
    }


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def _stable_snapshot(path: Path) -> dict:
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


def _write_r_predecessor_chain(
    tmp_path: Path,
    panel: dict,
    *,
    mutate_manifest=None,
    mutate_summary=None,
    mutate_receipt=None,
    conditioning_variant="additive",
) -> dict:
    predecessor_dir = tmp_path / "output/udlm/r-predecessor"
    predecessor_dir.mkdir(parents=True)
    manifest_path = predecessor_dir / "launch_manifest.json"
    runtime_path = predecessor_dir / "runtime_config.json"
    summary_path = predecessor_dir / "training_summary.json"
    receipt_path = predecessor_dir / "pilot_exit_status.json"
    checkpoint_path = predecessor_dir / "checkpoints/500.ckpt"
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_bytes(b"synthetic predecessor checkpoint")
    checkpoint_snapshot = _stable_snapshot(checkpoint_path)
    panel_sha256 = gate.canonical_json_sha256(panel)
    source_revision = "a" * 40
    selected_gpu_uuids = ["GPU-synthetic-r-0001"]
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
                "prior_variant": "release_uniform",
                "conditioning_variant": conditioning_variant,
                "exclude_special_tokens": True,
                "empirical_uniform_mix": gate.PILOT_EMPIRICAL_UNIFORM_MIX,
            },
        },
        "loader": {
            "batch_size": 2046,
            "global_batch_size": 2046,
            "num_workers": 1,
        },
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
            "max_steps": 500,
            "accumulate_grad_batches": 1,
            "detect_anomaly": True,
            "gradient_clip_val": 1.0,
            "precision": "bf16",
        },
        "callback": {"dirpath": str(checkpoint_path.parent)},
    }
    resolved_config_sha256 = gate.canonical_json_sha256(resolved_config)
    child_training_argv = [
        str(tmp_path / "scripts/train.py"),
        "--config-name",
        "udlm",
        "seed=7",
        "trainer.devices=1",
    ]
    manifest_training_argv = [
        str(tmp_path / ".venv/bin/python"),
        "-u",
        *child_training_argv,
    ]
    training_argv_sha256 = gate.canonical_json_sha256(child_training_argv)
    completion_contract = {
        "summary_schema_version": gate.TRAINING_SUMMARY_SCHEMA_VERSION,
        "summary_path": str(summary_path),
        "final_checkpoint_path": str(checkpoint_path),
        "expected_max_steps": 500,
        "expected_world_size": 1,
        "fail_on_nonfinite_loss": True,
        "backward_anomaly_detection": True,
    }
    lock_path = tmp_path / "output/udlm/.single_training_job.lock"
    lock_record = {
        "schema_version": 1,
        "status": "held",
        "purpose": "enforce_one_R_S_E_pilot_training_job_at_a_time",
        "source_revision": source_revision,
        "run_name": "r-predecessor",
        "training_variant": "udlm",
        "owner_token": "c" * 64,
        "launcher_pid_at_acquisition": 1233,
        "acquired_at_utc": "2026-09-05T23:56:00+00:00",
        "owner_process_exit_does_not_make_lock_stale": True,
        "stale_lock_policy": "fail_closed_and_require_manual_review",
        "release_policy": (
            "exact_owner_lock_only_after_receipt_or_before_tmux_handoff_failure"
        ),
    }
    lock_payload = (
        json.dumps(lock_record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    lock_snapshot = _stable_snapshot(lock_path)
    lock_snapshot["sha256"] = hashlib.sha256(lock_payload).hexdigest()
    lock_snapshot["size_bytes"] = len(lock_payload)
    genesis = {
        "schema_version": 1,
        "state": "explicit_genesis_no_predecessor",
        "current_training_variant": "udlm",
        "current_variant_position": 0,
        "expected_predecessor_training_variant": None,
        "expected_predecessor_variant_position": None,
        "matched_panel_spec_sha256": panel_sha256,
        "common_training_contract_sha256": gate.canonical_json_sha256(
            panel["common_training_contract"]
        ),
        "receipt_artifact": None,
        "predecessor_launch_manifest_artifact": None,
        "predecessor_training_summary_artifact": None,
        "predecessor_run_name": None,
        "chronology": None,
        "validated_before_gpu_probe": True,
    }
    predecessor_manifest = {
        "launch_manifest_schema_version": gate.LAUNCH_MANIFEST_SCHEMA_VERSION,
        "created_at": "2026-09-05T23:57:02+00:00",
        "purpose": "bounded UDLM training pilot",
        "gpu_selection_schema_version": 2,
        "git_sha": source_revision,
        "source_revision_before_final_gpu_probe": source_revision,
        "run_name": "r-predecessor",
        "training_variant": "udlm",
        "hydra_config_name": "udlm",
        "udlm_prior_variant": "release_uniform",
        "udlm_comparison_role": "faithful_release_control",
        "matched_panel_spec": panel,
        "matched_panel_spec_sha256": panel_sha256,
        "matched_panel_variant_position": 0,
        "predecessor_receipt_binding": genesis,
        "single_training_job_lock": {
            "path": str(lock_path),
            "sha256": lock_snapshot["sha256"],
            "record": lock_record,
            "acquired_before_any_gpu_probe": True,
            "stale_lock_policy": "fail_closed_and_require_manual_review",
            "release_owner": "pilot_exit_receipt_writer_after_publication",
        },
        "tmux_session": "genmol_udlm_r-predecessor",
        "user_requested_gpu_count": 1,
        "gpu_selection_method": "dynamic_idle_discovery",
        "gpu_inventory_scope": "all_nvidia_gpus",
        "inventory_snapshot_completed_at_utc": "2026-09-05T23:57:00+00:00",
        "gpu_inventory_at_selection": [gpu_state],
        "initially_selected_gpu_states": [gpu_state],
        "logical_cuda_devices": [0],
        "physical_gpu_indices": [6],
        "cuda_visible_device_uuids": selected_gpu_uuids,
        "final_uuid_probes_completed_at_utc": "2026-09-05T23:57:01+00:00",
        "gpu_states_at_final_uuid_probe": [gpu_state],
        "gpu_safety_policy": {
            "max_utilization_percent": 10,
            "utilization_comparison": "strictly_less_than",
            "min_free_memory_mib": 30000,
            "active_compute_processes_allowed": True,
            "compute_mode_prohibited_allowed": False,
        },
        "training_argv": manifest_training_argv,
        "training_argv_sha256": training_argv_sha256,
        "resolved_training_config": resolved_config,
        "resolved_training_config_sha256": resolved_config_sha256,
        "runtime_config_path": str(runtime_path),
        "training_summary_path": str(summary_path),
        "training_summary_schema_version": gate.TRAINING_SUMMARY_SCHEMA_VERSION,
        "pilot_exit_status_path": str(receipt_path),
        "pilot_exit_status_schema_version": gate.PILOT_EXIT_STATUS_SCHEMA_VERSION,
        "expected_final_checkpoint_path": str(checkpoint_path),
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
        "log_path": str(tmp_path / "output/logs/r-predecessor.log"),
        "log_reserved_exclusively_before_manifest": True,
        "checkpoint": "/synthetic/mdlm.ckpt",
        "checkpoint_sha256": gate.EXPECTED_BASELINE_CHECKPOINT_SHA256,
        "seed": 7,
        "max_steps": 500,
        "global_batch_size": 2046,
        "micro_batch_size_per_process": 2046,
        "accumulate_grad_batches": 1,
        "effective_global_batch_size": 2046,
        "exclude_special_tokens": True,
        "dry_run": False,
    }
    if mutate_manifest is not None:
        mutate_manifest(predecessor_manifest)
    manifest_path.write_bytes(_json_bytes(predecessor_manifest))
    manifest_snapshot = _stable_snapshot(manifest_path)
    manifest_claim = {
        **manifest_snapshot,
        "selected_gpu_uuids": selected_gpu_uuids,
    }
    runtime = {
        "schema_version": gate.RUNTIME_CONFIG_SCHEMA_VERSION,
        "status": "preflight_completed",
        "source_revision": source_revision,
        "source": {"head": source_revision, "upstream": source_revision},
        "training_argv": list(child_training_argv),
        "observed_training_argv": list(child_training_argv),
        "training_argv_sha256": training_argv_sha256,
        "resolved_training_config": resolved_config,
        "resolved_training_config_sha256": resolved_config_sha256,
        "launch_manifest": manifest_claim,
        "completion_contract": completion_contract,
        "python_environment": _python_environment(tmp_path),
    }
    runtime_path.write_bytes(_json_bytes(runtime))
    runtime_snapshot = _stable_snapshot(runtime_path)
    trainable_parameter_counts = {
        "base_backbone": 86_000_000,
        "time_conditioner": 787_968,
        "total": 86_787_968,
    }
    if conditioning_variant == "film_adaln":
        trainable_parameter_counts["film_modulation"] = 262_656
        trainable_parameter_counts["total"] = 87_050_624
    accounting = {
        "training_seed": 7,
        "optimizer_updates": 500,
        "world_size": 1,
        "micro_batch_size_per_rank": 2046,
        "accumulate_grad_batches": 1,
        "effective_global_examples_per_optimizer_step": 2046,
        "total_requested_example_exposures": 1_023_000,
        "hosted_stream_rank_partition_policy": (
            "huggingface_split_dataset_by_node_disjoint_rank_streams"
        ),
        "trainable_parameter_counts": trainable_parameter_counts,
    }
    predecessor_summary = {
        "schema_version": gate.TRAINING_SUMMARY_SCHEMA_VERSION,
        "status": "completed",
        "completed_at_utc": "2026-09-05T23:58:00+00:00",
        "source_revision": source_revision,
        "source": {"head": source_revision, "upstream": source_revision},
        "resolved_training_config_sha256": resolved_config_sha256,
        "training_argv_sha256": training_argv_sha256,
        "launch_manifest": manifest_claim,
        "runtime_config": {
            **runtime_snapshot,
            "schema_version": gate.RUNTIME_CONFIG_SCHEMA_VERSION,
            "record_sha256": gate.canonical_json_sha256(runtime),
        },
        "completion_contract": completion_contract,
        "observed_training_state": {
            "global_rank": 0,
            "global_step": 500,
            "world_size": 1,
        },
        "training_accounting": accounting,
        "training_health": _training_health(),
        "conditioning_gradient_audit": None,
        "screen_initialization_state_audit": None,
        "final_checkpoint": {
            **checkpoint_snapshot,
            "semantic_audit": {
                "deserialized": True,
                "global_step": 500,
                "raw_model": _tensor_finiteness()["raw_model"],
                "ema": _tensor_finiteness()["ema"],
                "ema_metadata": _inference_weights()["ema"],
                "optimizer": _tensor_finiteness()["raw_model"],
                "non_sentinel_checkpoint_tensors": _tensor_finiteness()["raw_model"],
                "framework_nonfinite_sentinels": (
                    gate._expected_framework_nonfinite_sentinels(expected_steps=500)
                ),
                **_auxiliary_checkpoint_records(
                    expected_steps=500,
                    resolved_training_config=resolved_config,
                ),
                "udlm_process_identity_verified": True,
                "live_ema_match": {
                    "exact_tensor_values": True,
                    "tensor_count": 202,
                },
                "live_model_match": {
                    "exact_key_set": True,
                    "exact_tensor_values": True,
                    "tensor_count": 206,
                },
            },
        },
        "tensor_finiteness": _tensor_finiteness(),
        "startup": {
            "mode": "warm_start",
            "verified_mdlm_warm_start_report": {
                "source_path": "/synthetic/mdlm.ckpt",
                "source_resolved_path": "/synthetic/mdlm.ckpt",
                "source_sha256": gate.EXPECTED_BASELINE_CHECKPOINT_SHA256,
                "source_size_bytes": 1_396_998_679,
                "expected_source_sha256": gate.EXPECTED_BASELINE_CHECKPOINT_SHA256,
                "byte_identity_verified_before_and_after_load": True,
                "weights": "ema",
                "parameter_tensors": 202,
                **(
                    {
                        "conditioning_variant": "film_adaln",
                        "conditioning_parameter_tensors": 28,
                    }
                    if conditioning_variant == "film_adaln"
                    else {}
                ),
            },
        },
    }
    if mutate_summary is not None:
        mutate_summary(predecessor_summary)
    summary_path.write_bytes(_json_bytes(predecessor_summary))
    summary_snapshot = _stable_snapshot(summary_path)
    validated_bindings = {
        "schema_version": gate.TRAINING_SUMMARY_SCHEMA_VERSION,
        "source_revision": source_revision,
        "resolved_training_config_sha256": resolved_config_sha256,
        "training_argv_sha256": training_argv_sha256,
        "launch_manifest_path": str(manifest_path),
        "launch_manifest_sha256": manifest_snapshot["sha256"],
        "selected_gpu_uuids": selected_gpu_uuids,
        "observed_global_step": 500,
        "observed_world_size": 1,
        "training_accounting": accounting,
        "ema_metadata": _inference_weights()["ema"],
        "final_checkpoint_path": str(checkpoint_path),
        "final_checkpoint_sha256": checkpoint_snapshot["sha256"],
        "startup_mode": "warm_start",
        "conditioning_gradient_audit": None,
        "screen_initialization_state_audit": None,
    }
    predecessor_receipt = {
        "schema_version": gate.PILOT_EXIT_STATUS_SCHEMA_VERSION,
        "status": "completed",
        "overall_status": "completed",
        "recorded_at_utc": "2026-09-05T23:59:00+00:00",
        "process_exit_status": 0,
        "expected_contract": {
            "training_summary_schema_version": gate.TRAINING_SUMMARY_SCHEMA_VERSION,
            "source_revision": source_revision,
            "resolved_training_config_sha256": resolved_config_sha256,
            "training_argv_sha256": training_argv_sha256,
            "launch_manifest_path": str(manifest_path),
            "launch_manifest_sha256": manifest_snapshot["sha256"],
            "selected_gpu_uuids": selected_gpu_uuids,
            "training_job_lock_path": str(lock_path),
            "training_job_lock_sha256": lock_snapshot["sha256"],
            "max_steps": 500,
            "world_size": 1,
            "training_summary_path": str(summary_path),
            "final_checkpoint_path": str(checkpoint_path),
            "initialization_checkpoint_sha256": (
                gate.EXPECTED_BASELINE_CHECKPOINT_SHA256
            ),
        },
        "pipeline": {
            "training": _successful_pipeline_component(),
            "tee": _successful_pipeline_component(),
            "pipefail_shell_exit_status": 0,
        },
        "source_at_receipt": {
            "verified": True,
            "expected_revision": "a" * 40,
            "head": "a" * 40,
            "upstream": "a" * 40,
            "output_directory_excluded_from_cleanliness_check": True,
        },
        "launch_manifest": {
            "path": str(manifest_path),
            "present": True,
            "matches_expected_raw_sha256": True,
            "selected_gpu_uuids_match_expected": True,
            "matches_training_summary_snapshot": True,
            "matches_runtime_config_snapshot": True,
            "artifact": manifest_snapshot,
            "valid_and_launch_bound": True,
            "expected_selected_gpu_uuids": selected_gpu_uuids,
            "observed_selected_gpu_uuids": selected_gpu_uuids,
            "validation_error": None,
        },
        "predecessor_receipt_binding": genesis,
        "training_job_lock": {
            "path": str(lock_path),
            "present": True,
            "expected_sha256": lock_snapshot["sha256"],
            "matches_expected_raw_sha256": True,
            "matches_launch_manifest_binding": True,
            "valid_and_launch_bound_before_receipt_publication": True,
            "artifact": lock_snapshot,
            "record": lock_record,
            "release_policy": (
                "publish_receipt_then_unlink_only_same_stat_identity_and_sha256"
            ),
            "release_result_not_claimed_inside_pre_release_receipt": True,
            "validation_error": None,
        },
        "training_summary": {
            "path": str(summary_path),
            "present": True,
            "artifact": summary_snapshot,
            "valid_and_launch_bound": True,
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
        mutate_receipt(predecessor_receipt)
    receipt_path.write_bytes(_json_bytes(predecessor_receipt))
    return {
        "schema_version": 1,
        "state": "validated_successful_predecessor",
        "current_training_variant": "schedule_uniform",
        "current_variant_position": 1,
        "expected_predecessor_training_variant": "udlm",
        "expected_predecessor_variant_position": 0,
        "matched_panel_spec_sha256": panel_sha256,
        "common_training_contract_sha256": gate.canonical_json_sha256(
            panel["common_training_contract"]
        ),
        "receipt_artifact": _stable_snapshot(receipt_path),
        "predecessor_launch_manifest_artifact": manifest_snapshot,
        "predecessor_training_summary_artifact": summary_snapshot,
        "predecessor_run_name": "r-predecessor",
        "chronology": {
            "predecessor_launch_manifest_created_at_utc": ("2026-09-05T23:57:02+00:00"),
            "predecessor_training_summary_completed_at_utc": (
                "2026-09-05T23:58:00+00:00"
            ),
            "predecessor_exit_receipt_recorded_at_utc": ("2026-09-05T23:59:00+00:00"),
            "strictly_ordered_timestamps_verified": True,
        },
        "validated_before_gpu_probe": True,
    }


def _write_valid_training_evidence(
    tmp_path: Path,
    candidate_lock: dict,
    *,
    lock_acquired_at_utc="2026-09-05T23:59:57+00:00",
    mutate_manifest=None,
    mutate_runtime=None,
    mutate_summary=None,
    mutate_receipt=None,
    mutate_predecessor_manifest=None,
    mutate_predecessor_summary=None,
    mutate_predecessor_receipt=None,
    conditioning_variant=None,
) -> dict:
    training = candidate_lock["training"]
    if conditioning_variant is None:
        conditioning_variant = (
            "film_adaln"
            if "film_modulation_trainable" in training["parameter_counts"]
            else "additive"
        )
    manifest_path = tmp_path / training["launch_manifest"]["relative_path"]
    runtime_path = tmp_path / training["runtime_config"]["relative_path"]
    summary_path = tmp_path / training["training_summary"]["relative_path"]
    receipt_path = tmp_path / training["exit_receipt"]["relative_path"]
    checkpoint_path = tmp_path / training["checkpoint"]["relative_path"]
    manifest_path.parent.mkdir(parents=True)
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"synthetic candidate checkpoint")
    checkpoint_snapshot = _stable_snapshot(checkpoint_path)
    training["checkpoint"]["sha256"] = checkpoint_snapshot["sha256"]
    training["checkpoint"]["size_bytes"] = checkpoint_snapshot["size_bytes"]
    candidate_lock["inference"]["checkpoint_sha256"] = checkpoint_snapshot["sha256"]

    resolved_config = {
        "data": "safe",
        "seed": 7,
        "training": {
            "ema": 0.9999,
            "udlm": {
                "prior_variant": "schedule_uniform",
                "conditioning_variant": conditioning_variant,
                "exclude_special_tokens": True,
                "empirical_uniform_mix": gate.PILOT_EMPIRICAL_UNIFORM_MIX,
            },
        },
        "loader": {
            "batch_size": 2046,
            "global_batch_size": 2046,
            "num_workers": 1,
        },
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
            "max_steps": 500,
            "accumulate_grad_batches": 1,
            "detect_anomaly": True,
            "gradient_clip_val": 1.0,
            "precision": "bf16",
        },
        "callback": {"dirpath": str(manifest_path.parent / "checkpoints")},
    }
    python_path = tmp_path / ".venv/bin/python"
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.touch(exist_ok=True)
    child_training_argv = [
        str(tmp_path / "scripts/train.py"),
        "--config-name",
        "udlm",
        "seed=7",
        "trainer.devices=1",
    ]
    manifest_training_argv = [str(python_path), "-u", *child_training_argv]
    resolved_config_sha = gate.canonical_json_sha256(resolved_config)
    training_argv_sha = gate.canonical_json_sha256(child_training_argv)
    training["resolved_training_config_sha256"] = resolved_config_sha
    training["training_argv_sha256"] = training_argv_sha

    gpu_state = {
        "physical_index": 7,
        "uuid": "GPU-synthetic-0001",
        "name": "Synthetic Accelerator",
        "memory_used_mib": 1000,
        "memory_total_mib": 81920,
        "utilization_percent": 0,
        "compute_mode": "Default",
        "compute_processes": [],
    }
    panel = {
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
            "initialization_checkpoint_sha256": (
                gate.EXPECTED_BASELINE_CHECKPOINT_SHA256
            ),
            "requested_gpu_count": 1,
            "max_steps": 500,
            "global_batch_size": 2046,
            "micro_batch_size_per_process": 2046,
            "accumulate_grad_batches": 1,
            "effective_global_batch_size": 2046,
            "num_workers": 1,
            "seed": 7,
            "exclude_special_tokens": True,
            "empirical_uniform_mix": gate.PILOT_EMPIRICAL_UNIFORM_MIX,
            "empirical_uniform_mix_consumed_only_by": "empirical_frequency",
            "empirical_uniform_mix_audit": copy.deepcopy(
                gate.PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT
            ),
            "common_resolved_config_sha256": gate.matched_panel_config_sha256(
                resolved_config
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
    lock_path = tmp_path / "output/udlm/.single_training_job.lock"
    lock_record = {
        "schema_version": 1,
        "status": "held",
        "purpose": "enforce_one_R_S_E_pilot_training_job_at_a_time",
        "source_revision": "a" * 40,
        "run_name": "synthetic-candidate",
        "training_variant": "schedule_uniform",
        "owner_token": "b" * 64,
        "launcher_pid_at_acquisition": 1234,
        "acquired_at_utc": lock_acquired_at_utc,
        "owner_process_exit_does_not_make_lock_stale": True,
        "stale_lock_policy": "fail_closed_and_require_manual_review",
        "release_policy": (
            "exact_owner_lock_only_after_receipt_or_before_tmux_handoff_failure"
        ),
    }
    lock_bytes = (
        json.dumps(lock_record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    lock_path.write_bytes(lock_bytes)
    lock_snapshot = _stable_snapshot(lock_path)
    predecessor_binding = _write_r_predecessor_chain(
        tmp_path,
        panel,
        mutate_manifest=mutate_predecessor_manifest,
        mutate_summary=mutate_predecessor_summary,
        mutate_receipt=mutate_predecessor_receipt,
        conditioning_variant=conditioning_variant,
    )
    manifest = {
        "launch_manifest_schema_version": gate.LAUNCH_MANIFEST_SCHEMA_VERSION,
        "created_at": "2026-09-06T00:00:00+00:00",
        "purpose": "bounded UDLM training pilot",
        "gpu_selection_schema_version": 2,
        "git_sha": "a" * 40,
        "source_revision_before_final_gpu_probe": "a" * 40,
        "run_name": "synthetic-candidate",
        "training_variant": "schedule_uniform",
        "hydra_config_name": "udlm",
        "udlm_prior_variant": "schedule_uniform",
        "udlm_comparison_role": "schedule_repair_uniform_control",
        "matched_panel_spec": panel,
        "matched_panel_spec_sha256": gate.canonical_json_sha256(panel),
        "matched_panel_variant_position": 1,
        "predecessor_receipt_binding": predecessor_binding,
        "single_training_job_lock": {
            "path": str(lock_path),
            "sha256": lock_snapshot["sha256"],
            "record": lock_record,
            "acquired_before_any_gpu_probe": True,
            "stale_lock_policy": "fail_closed_and_require_manual_review",
            "release_owner": "pilot_exit_receipt_writer_after_publication",
        },
        "tmux_session": "genmol_schedule_uniform_synthetic-candidate",
        "user_requested_gpu_count": 1,
        "gpu_selection_method": "dynamic_idle_discovery",
        "gpu_inventory_scope": "all_nvidia_gpus",
        "inventory_snapshot_completed_at_utc": "2026-09-05T23:59:58+00:00",
        "gpu_inventory_at_selection": [gpu_state],
        "initially_selected_gpu_states": [gpu_state],
        "logical_cuda_devices": [0],
        "physical_gpu_indices": [7],
        "cuda_visible_device_uuids": ["GPU-synthetic-0001"],
        "final_uuid_probes_completed_at_utc": "2026-09-05T23:59:59+00:00",
        "gpu_states_at_final_uuid_probe": [gpu_state],
        "gpu_safety_policy": {
            "max_utilization_percent": 10,
            "utilization_comparison": "strictly_less_than",
            "min_free_memory_mib": 30000,
            "active_compute_processes_allowed": True,
            "compute_mode_prohibited_allowed": False,
        },
        "training_argv": manifest_training_argv,
        "training_argv_sha256": training_argv_sha,
        "resolved_training_config": resolved_config,
        "resolved_training_config_sha256": resolved_config_sha,
        "runtime_config_path": str(runtime_path),
        "training_summary_path": str(summary_path),
        "training_summary_schema_version": gate.TRAINING_SUMMARY_SCHEMA_VERSION,
        "pilot_exit_status_path": str(receipt_path),
        "pilot_exit_status_schema_version": gate.PILOT_EXIT_STATUS_SCHEMA_VERSION,
        "expected_final_checkpoint_path": str(checkpoint_path),
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
        "log_path": str(tmp_path / "output/logs/synthetic-candidate.log"),
        "log_reserved_exclusively_before_manifest": True,
        "checkpoint": "/synthetic/mdlm.ckpt",
        "checkpoint_sha256": gate.EXPECTED_BASELINE_CHECKPOINT_SHA256,
        "seed": 7,
        "max_steps": 500,
        "global_batch_size": 2046,
        "micro_batch_size_per_process": 2046,
        "accumulate_grad_batches": 1,
        "effective_global_batch_size": 2046,
        "exclude_special_tokens": True,
        "dry_run": False,
    }
    if mutate_manifest is not None:
        mutate_manifest(manifest)
    manifest_path.write_bytes(_json_bytes(manifest))
    manifest_snapshot = _stable_snapshot(manifest_path)
    training["launch_manifest"]["sha256"] = manifest_snapshot["sha256"]
    manifest_claim = {
        **manifest_snapshot,
        "selected_gpu_uuids": ["GPU-synthetic-0001"],
    }

    completion_contract = {
        "summary_schema_version": gate.TRAINING_SUMMARY_SCHEMA_VERSION,
        "summary_path": str(summary_path),
        "final_checkpoint_path": str(checkpoint_path),
        "expected_max_steps": 500,
        "expected_world_size": 1,
        "fail_on_nonfinite_loss": True,
        "backward_anomaly_detection": True,
    }
    runtime = {
        "schema_version": gate.RUNTIME_CONFIG_SCHEMA_VERSION,
        "status": "preflight_completed",
        "source_revision": "a" * 40,
        "source": {"head": "a" * 40, "upstream": "a" * 40},
        "resolved_training_config": resolved_config,
        "resolved_training_config_sha256": resolved_config_sha,
        "training_argv": list(child_training_argv),
        "observed_training_argv": list(child_training_argv),
        "training_argv_sha256": training_argv_sha,
        "launch_manifest": manifest_claim,
        "completion_contract": completion_contract,
        "python_environment": _python_environment(tmp_path),
    }
    if mutate_runtime is not None:
        mutate_runtime(runtime)
    runtime_path.write_bytes(_json_bytes(runtime))
    runtime_snapshot = _stable_snapshot(runtime_path)
    training["runtime_config"]["sha256"] = runtime_snapshot["sha256"]

    locked_parameter_counts = training["parameter_counts"]
    trainable_parameter_counts = {
        "base_backbone": locked_parameter_counts["base_model_trainable"],
        "time_conditioner": locked_parameter_counts["time_conditioner_trainable"],
        "total": locked_parameter_counts["total_trainable"],
    }
    if "film_modulation_trainable" in locked_parameter_counts:
        trainable_parameter_counts["film_modulation"] = locked_parameter_counts[
            "film_modulation_trainable"
        ]
    training_accounting = {
        "training_seed": 7,
        "optimizer_updates": 500,
        "world_size": 1,
        "micro_batch_size_per_rank": 2046,
        "accumulate_grad_batches": 1,
        "effective_global_examples_per_optimizer_step": 2046,
        "total_requested_example_exposures": 1_023_000,
        "hosted_stream_rank_partition_policy": (
            "huggingface_split_dataset_by_node_disjoint_rank_streams"
        ),
        "trainable_parameter_counts": trainable_parameter_counts,
    }
    summary = {
        "schema_version": gate.TRAINING_SUMMARY_SCHEMA_VERSION,
        "status": "completed",
        "completed_at_utc": "2026-09-06T00:10:00+00:00",
        "source_revision": "a" * 40,
        "source": {"head": "a" * 40, "upstream": "a" * 40},
        "resolved_training_config_sha256": resolved_config_sha,
        "training_argv_sha256": training_argv_sha,
        "launch_manifest": manifest_claim,
        "runtime_config": {
            **runtime_snapshot,
            "schema_version": gate.RUNTIME_CONFIG_SCHEMA_VERSION,
            "record_sha256": gate.canonical_json_sha256(runtime),
        },
        "completion_contract": completion_contract,
        "observed_training_state": {
            "global_rank": 0,
            "global_step": 500,
            "world_size": 1,
        },
        "training_accounting": training_accounting,
        "training_health": _training_health(),
        "final_checkpoint": {
            **checkpoint_snapshot,
            "semantic_audit": {
                "deserialized": True,
                "global_step": 500,
                "raw_model": _tensor_finiteness()["raw_model"],
                "ema": _tensor_finiteness()["ema"],
                "ema_metadata": _inference_weights()["ema"],
                "optimizer": _tensor_finiteness()["raw_model"],
                "non_sentinel_checkpoint_tensors": _tensor_finiteness()["raw_model"],
                "framework_nonfinite_sentinels": (
                    gate._expected_framework_nonfinite_sentinels(expected_steps=500)
                ),
                **_auxiliary_checkpoint_records(
                    expected_steps=500,
                    resolved_training_config=resolved_config,
                ),
                "udlm_process_identity_verified": True,
                "live_ema_match": {
                    "exact_tensor_values": True,
                    "tensor_count": 202,
                },
                "live_model_match": {
                    "exact_key_set": True,
                    "exact_tensor_values": True,
                    "tensor_count": 206,
                },
            },
        },
        "startup": {
            "mode": "warm_start",
            "verified_mdlm_warm_start_report": {
                "source_path": "/synthetic/mdlm.ckpt",
                "source_resolved_path": "/synthetic/mdlm.ckpt",
                "source_sha256": gate.EXPECTED_BASELINE_CHECKPOINT_SHA256,
                "source_size_bytes": 1_396_998_679,
                "expected_source_sha256": gate.EXPECTED_BASELINE_CHECKPOINT_SHA256,
                "byte_identity_verified_before_and_after_load": True,
                "weights": "ema",
                "parameter_tensors": 202,
                **(
                    {
                        "conditioning_variant": "film_adaln",
                        "conditioning_parameter_tensors": 28,
                    }
                    if conditioning_variant == "film_adaln"
                    else {}
                ),
            },
        },
        "conditioning_gradient_audit": None,
        "screen_initialization_state_audit": None,
        "tensor_finiteness": _tensor_finiteness(),
    }
    if mutate_summary is not None:
        mutate_summary(summary)
    summary_path.write_bytes(_json_bytes(summary))
    summary_snapshot = _stable_snapshot(summary_path)
    training["training_summary"]["sha256"] = summary_snapshot["sha256"]

    validated_bindings = {
        "schema_version": gate.TRAINING_SUMMARY_SCHEMA_VERSION,
        "source_revision": "a" * 40,
        "resolved_training_config_sha256": resolved_config_sha,
        "training_argv_sha256": training_argv_sha,
        "launch_manifest_path": str(manifest_path),
        "launch_manifest_sha256": manifest_snapshot["sha256"],
        "selected_gpu_uuids": ["GPU-synthetic-0001"],
        "observed_global_step": 500,
        "observed_world_size": 1,
        "training_accounting": training_accounting,
        "ema_metadata": _inference_weights()["ema"],
        "final_checkpoint_path": str(checkpoint_path),
        "final_checkpoint_sha256": checkpoint_snapshot["sha256"],
        "startup_mode": "warm_start",
        "conditioning_gradient_audit": None,
        "screen_initialization_state_audit": None,
    }
    receipt = {
        "schema_version": gate.PILOT_EXIT_STATUS_SCHEMA_VERSION,
        "status": "completed",
        "overall_status": "completed",
        "recorded_at_utc": "2026-09-06T00:10:01+00:00",
        "process_exit_status": 0,
        "expected_contract": {
            "training_summary_schema_version": gate.TRAINING_SUMMARY_SCHEMA_VERSION,
            "source_revision": "a" * 40,
            "resolved_training_config_sha256": resolved_config_sha,
            "training_argv_sha256": training_argv_sha,
            "launch_manifest_path": str(manifest_path),
            "launch_manifest_sha256": manifest_snapshot["sha256"],
            "selected_gpu_uuids": ["GPU-synthetic-0001"],
            "training_job_lock_path": str(lock_path),
            "training_job_lock_sha256": lock_snapshot["sha256"],
            "max_steps": 500,
            "world_size": 1,
            "training_summary_path": str(summary_path),
            "final_checkpoint_path": str(checkpoint_path),
            "initialization_checkpoint_sha256": (
                gate.EXPECTED_BASELINE_CHECKPOINT_SHA256
            ),
        },
        "pipeline": {
            "training": _successful_pipeline_component(),
            "tee": _successful_pipeline_component(),
            "pipefail_shell_exit_status": 0,
        },
        "source_at_receipt": {
            "verified": True,
            "expected_revision": "a" * 40,
            "head": "a" * 40,
            "upstream": "a" * 40,
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
            "expected_selected_gpu_uuids": ["GPU-synthetic-0001"],
            "observed_selected_gpu_uuids": ["GPU-synthetic-0001"],
            "artifact": manifest_snapshot,
            "validation_error": None,
        },
        "predecessor_receipt_binding": copy.deepcopy(predecessor_binding),
        "training_job_lock": {
            "path": str(lock_path),
            "present": True,
            "expected_sha256": lock_snapshot["sha256"],
            "matches_expected_raw_sha256": True,
            "matches_launch_manifest_binding": True,
            "valid_and_launch_bound_before_receipt_publication": True,
            "artifact": lock_snapshot,
            "record": lock_record,
            "release_policy": (
                "publish_receipt_then_unlink_only_same_stat_identity_and_sha256"
            ),
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
        "final_checkpoint": {
            "path": str(checkpoint_path),
            "present": True,
            "matches_training_summary_snapshot": True,
            "artifact": checkpoint_snapshot,
        },
        "runtime_config": {
            "path": str(runtime_path),
            "present": True,
            "matches_training_summary_snapshot": True,
            "semantic_validation_passed": True,
            "artifact": runtime_snapshot,
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
    receipt_path.write_bytes(_json_bytes(receipt))
    training["exit_receipt"]["sha256"] = hashlib.sha256(
        receipt_path.read_bytes()
    ).hexdigest()
    return {
        "manifest": manifest,
        "runtime": runtime,
        "summary": summary,
        "receipt": receipt,
        "training_accounting": training_accounting,
    }


def _write_terminal_e_training_evidence(
    tmp_path: Path,
    candidate_lock: dict,
    selected_documents: dict,
) -> dict:
    """Write a producer-shaped E run whose predecessor is the selected S run."""

    selected_manifest = selected_documents["manifest"]
    selected_summary = selected_documents["summary"]
    selected_receipt = selected_documents["receipt"]
    selected_run_dir = Path(selected_manifest["launch_manifest_path"]).parent
    selected_manifest_path = selected_run_dir / "launch_manifest.json"
    selected_summary_path = selected_run_dir / "training_summary.json"
    selected_receipt_path = selected_run_dir / "pilot_exit_status.json"
    panel = copy.deepcopy(selected_manifest["matched_panel_spec"])
    panel_sha256 = gate.canonical_json_sha256(panel)
    common_sha256 = gate.canonical_json_sha256(panel["common_training_contract"])

    run_name = "e-terminal"
    run_dir = tmp_path / "output/udlm" / run_name
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "launch_manifest.json"
    runtime_path = run_dir / "runtime_config.json"
    summary_path = run_dir / "training_summary.json"
    receipt_path = run_dir / "pilot_exit_status.json"
    checkpoint_path = run_dir / "checkpoints/500.ckpt"
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_bytes(b"synthetic terminal E checkpoint")
    checkpoint_snapshot = _stable_snapshot(checkpoint_path)

    predecessor_binding = {
        "schema_version": 1,
        "state": "validated_successful_predecessor",
        "current_training_variant": "udlm_categorical",
        "current_variant_position": 2,
        "expected_predecessor_training_variant": "schedule_uniform",
        "expected_predecessor_variant_position": 1,
        "matched_panel_spec_sha256": panel_sha256,
        "common_training_contract_sha256": common_sha256,
        "receipt_artifact": _stable_snapshot(selected_receipt_path),
        "predecessor_launch_manifest_artifact": _stable_snapshot(
            selected_manifest_path
        ),
        "predecessor_training_summary_artifact": _stable_snapshot(
            selected_summary_path
        ),
        "predecessor_run_name": selected_manifest["run_name"],
        "chronology": {
            "predecessor_launch_manifest_created_at_utc": selected_manifest[
                "created_at"
            ],
            "predecessor_training_summary_completed_at_utc": selected_summary[
                "completed_at_utc"
            ],
            "predecessor_exit_receipt_recorded_at_utc": selected_receipt[
                "recorded_at_utc"
            ],
            "strictly_ordered_timestamps_verified": True,
        },
        "validated_before_gpu_probe": True,
    }
    gpu_uuid = "GPU-synthetic-e-0001"
    gpu_state = {
        "physical_index": 8,
        "uuid": gpu_uuid,
        "name": "Synthetic Accelerator",
        "memory_used_mib": 1000,
        "memory_total_mib": 81920,
        "utilization_percent": 0,
        "compute_mode": "Default",
        "compute_processes": [],
    }
    resolved_config = copy.deepcopy(selected_manifest["resolved_training_config"])
    resolved_config["training"]["udlm"]["prior_variant"] = "empirical_frequency"
    resolved_config["callback"]["dirpath"] = str(checkpoint_path.parent)
    resolved_config_sha256 = gate.canonical_json_sha256(resolved_config)
    child_argv = [
        str(tmp_path / "scripts/train.py"),
        "--config-name",
        "udlm_categorical",
        "seed=7",
        "trainer.devices=1",
    ]
    manifest_argv = [str(tmp_path / ".venv/bin/python"), "-u", *child_argv]
    argv_sha256 = gate.canonical_json_sha256(child_argv)
    completion_contract = {
        "summary_schema_version": gate.TRAINING_SUMMARY_SCHEMA_VERSION,
        "summary_path": str(summary_path),
        "final_checkpoint_path": str(checkpoint_path),
        "expected_max_steps": 500,
        "expected_world_size": 1,
        "fail_on_nonfinite_loss": True,
        "backward_anomaly_detection": True,
    }

    lock_path = tmp_path / "output/udlm/.single_training_job.lock"
    lock_record = {
        "schema_version": 1,
        "status": "held",
        "purpose": "enforce_one_R_S_E_pilot_training_job_at_a_time",
        "source_revision": "a" * 40,
        "run_name": run_name,
        "training_variant": "udlm_categorical",
        "owner_token": "e" * 64,
        "launcher_pid_at_acquisition": 1235,
        "acquired_at_utc": "2026-09-06T00:19:00+00:00",
        "owner_process_exit_does_not_make_lock_stale": True,
        "stale_lock_policy": "fail_closed_and_require_manual_review",
        "release_policy": (
            "exact_owner_lock_only_after_receipt_or_before_tmux_handoff_failure"
        ),
    }
    lock_path.write_bytes(
        (
            json.dumps(lock_record, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
    )
    lock_snapshot = _stable_snapshot(lock_path)
    manifest = copy.deepcopy(selected_manifest)
    manifest.update(
        {
            "created_at": "2026-09-06T00:20:00+00:00",
            "run_name": run_name,
            "training_variant": "udlm_categorical",
            "hydra_config_name": "udlm_categorical",
            "udlm_prior_variant": "empirical_frequency",
            "udlm_comparison_role": "empirical_prior_treatment",
            "matched_panel_spec": panel,
            "matched_panel_spec_sha256": panel_sha256,
            "matched_panel_variant_position": 2,
            "predecessor_receipt_binding": predecessor_binding,
            "single_training_job_lock": {
                "path": str(lock_path),
                "sha256": lock_snapshot["sha256"],
                "record": lock_record,
                "acquired_before_any_gpu_probe": True,
                "stale_lock_policy": "fail_closed_and_require_manual_review",
                "release_owner": "pilot_exit_receipt_writer_after_publication",
            },
            "tmux_session": "genmol_udlm_categorical_e-terminal",
            "inventory_snapshot_completed_at_utc": ("2026-09-06T00:19:58+00:00"),
            "gpu_inventory_at_selection": [gpu_state],
            "initially_selected_gpu_states": [gpu_state],
            "physical_gpu_indices": [8],
            "cuda_visible_device_uuids": [gpu_uuid],
            "final_uuid_probes_completed_at_utc": ("2026-09-06T00:19:59+00:00"),
            "gpu_states_at_final_uuid_probe": [gpu_state],
            "training_argv": manifest_argv,
            "training_argv_sha256": argv_sha256,
            "resolved_training_config": resolved_config,
            "resolved_training_config_sha256": resolved_config_sha256,
            "runtime_config_path": str(runtime_path),
            "training_summary_path": str(summary_path),
            "pilot_exit_status_path": str(receipt_path),
            "expected_final_checkpoint_path": str(checkpoint_path),
            "launch_manifest_path": str(manifest_path),
            "log_path": str(tmp_path / "output/logs/e-terminal.log"),
        }
    )
    manifest_path.write_bytes(_json_bytes(manifest))
    manifest_snapshot = _stable_snapshot(manifest_path)
    manifest_claim = {**manifest_snapshot, "selected_gpu_uuids": [gpu_uuid]}

    runtime = copy.deepcopy(selected_documents["runtime"])
    runtime.update(
        {
            "training_argv": child_argv,
            "observed_training_argv": child_argv,
            "training_argv_sha256": argv_sha256,
            "resolved_training_config": resolved_config,
            "resolved_training_config_sha256": resolved_config_sha256,
            "launch_manifest": manifest_claim,
            "completion_contract": completion_contract,
        }
    )
    runtime_path.write_bytes(_json_bytes(runtime))
    runtime_snapshot = _stable_snapshot(runtime_path)

    summary = copy.deepcopy(selected_summary)
    summary.update(
        {
            "completed_at_utc": "2026-09-06T00:30:00+00:00",
            "resolved_training_config_sha256": resolved_config_sha256,
            "training_argv_sha256": argv_sha256,
            "launch_manifest": manifest_claim,
            "runtime_config": {
                **runtime_snapshot,
                "schema_version": gate.RUNTIME_CONFIG_SCHEMA_VERSION,
                "record_sha256": gate.canonical_json_sha256(runtime),
            },
            "completion_contract": completion_contract,
            "final_checkpoint": {
                **checkpoint_snapshot,
                "semantic_audit": copy.deepcopy(
                    selected_summary["final_checkpoint"]["semantic_audit"]
                ),
            },
        }
    )
    summary["final_checkpoint"]["semantic_audit"]["checkpoint_hyperparameters_match"][
        "resolved_config_sha256"
    ] = resolved_config_sha256
    summary_path.write_bytes(_json_bytes(summary))
    summary_snapshot = _stable_snapshot(summary_path)

    validated_bindings = copy.deepcopy(
        selected_receipt["training_summary"]["validated_bindings"]
    )
    validated_bindings.update(
        {
            "resolved_training_config_sha256": resolved_config_sha256,
            "training_argv_sha256": argv_sha256,
            "launch_manifest_path": str(manifest_path),
            "launch_manifest_sha256": manifest_snapshot["sha256"],
            "selected_gpu_uuids": [gpu_uuid],
            "final_checkpoint_path": str(checkpoint_path),
            "final_checkpoint_sha256": checkpoint_snapshot["sha256"],
        }
    )
    receipt = copy.deepcopy(selected_receipt)
    receipt.update(
        {
            "recorded_at_utc": "2026-09-06T00:30:01+00:00",
            "expected_contract": {
                "training_summary_schema_version": (
                    gate.TRAINING_SUMMARY_SCHEMA_VERSION
                ),
                "source_revision": "a" * 40,
                "resolved_training_config_sha256": resolved_config_sha256,
                "training_argv_sha256": argv_sha256,
                "launch_manifest_path": str(manifest_path),
                "launch_manifest_sha256": manifest_snapshot["sha256"],
                "selected_gpu_uuids": [gpu_uuid],
                "training_job_lock_path": str(lock_path),
                "training_job_lock_sha256": lock_snapshot["sha256"],
                "max_steps": 500,
                "world_size": 1,
                "training_summary_path": str(summary_path),
                "final_checkpoint_path": str(checkpoint_path),
                "initialization_checkpoint_sha256": (
                    gate.EXPECTED_BASELINE_CHECKPOINT_SHA256
                ),
            },
            "launch_manifest": {
                "path": str(manifest_path),
                "present": True,
                "matches_expected_raw_sha256": True,
                "selected_gpu_uuids_match_expected": True,
                "matches_training_summary_snapshot": True,
                "matches_runtime_config_snapshot": True,
                "valid_and_launch_bound": True,
                "expected_selected_gpu_uuids": [gpu_uuid],
                "observed_selected_gpu_uuids": [gpu_uuid],
                "artifact": manifest_snapshot,
                "validation_error": None,
            },
            "predecessor_receipt_binding": predecessor_binding,
            "training_job_lock": {
                "path": str(lock_path),
                "present": True,
                "expected_sha256": lock_snapshot["sha256"],
                "matches_expected_raw_sha256": True,
                "matches_launch_manifest_binding": True,
                "valid_and_launch_bound_before_receipt_publication": True,
                "artifact": lock_snapshot,
                "record": lock_record,
                "release_policy": (
                    "publish_receipt_then_unlink_only_same_stat_identity_and_sha256"
                ),
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
        }
    )
    receipt_path.write_bytes(_json_bytes(receipt))
    receipt_snapshot = _stable_snapshot(receipt_path)
    candidate_lock["selection"]["terminal_e_exit_receipt"] = {
        "relative_path": str(receipt_path.relative_to(tmp_path)),
        "sha256": receipt_snapshot["sha256"],
        "schema_version": gate.PILOT_EXIT_STATUS_SCHEMA_VERSION,
    }
    return {
        "manifest": manifest,
        "runtime": runtime,
        "summary": summary,
        "receipt": receipt,
        "receipt_path": receipt_path,
        "receipt_snapshot": receipt_snapshot,
    }


_PILOT_RUN_RESULTS: dict[tuple[str, int], dict] = {}


def _synthetic_digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _completed_pilot_attempt(
    *,
    attempt_id: str,
    candidate_id: str,
    seeds: tuple[int, ...],
    qualities: tuple[float, ...],
    diversities: tuple[float, ...],
    requested_samples: int = gate.REGISTERED_SELECTION_SAMPLES_PER_SEED,
    nfe: int = gate.REGISTERED_SELECTION_NFE,
    metric_branch: str = gate.REGISTERED_SELECTION_METRIC_BRANCH,
    checkpoint_sha256: str = "4" * 64,
) -> tuple[dict, dict[str, bytes]]:
    assert len(seeds) == len(qualities) == len(diversities)
    refs = []
    receipt_payload = _json_bytes(
        {
            "schema_version": gate.PILOT_EXIT_STATUS_SCHEMA_VERSION,
            "status": "completed",
            "candidate_id": candidate_id,
        }
    )
    receipt_ref = {
        "relative_path": (
            "output/udlm/synthetic-candidate/pilot_exit_status.json"
            if candidate_id == "schedule-uniform-synthetic"
            else f"output/udlm/{candidate_id}/pilot_exit_status.json"
        ),
        "sha256": hashlib.sha256(receipt_payload).hexdigest(),
        "schema_version": gate.PILOT_EXIT_STATUS_SCHEMA_VERSION,
    }
    blobs = {receipt_ref["relative_path"]: receipt_payload}
    for seed, quality, diversity in zip(seeds, qualities, diversities, strict=True):
        relative_path = f"experiments/udlm/pilots/{attempt_id}/seed_{seed}.json"
        run_directory = f"output/benchmarks/selection/{attempt_id}/seed_{seed}"
        summary_payload = _json_bytes(
            {"attempt_id": attempt_id, "pilot_seed": seed, "status": "completed"}
        )
        raw_payload = f"sample_index,seed\n0,{seed}\n".encode()
        summary_ref = {
            "relative_path": f"{run_directory}/summary.json",
            "sha256": hashlib.sha256(summary_payload).hexdigest(),
            "schema_version": gate.denovo_report.RUN_SCHEMA_VERSION,
        }
        raw_ref = {
            "relative_path": f"{run_directory}/raw_samples.csv",
            "sha256": hashlib.sha256(raw_payload).hexdigest(),
        }
        blobs[summary_ref["relative_path"]] = summary_payload
        blobs[raw_ref["relative_path"]] = raw_payload
        evidence = {
            "schema_version": gate.PILOT_EVIDENCE_SCHEMA_VERSION,
            "artifact_kind": "pilot_evaluation",
            "status": "completed",
            "attempt_id": attempt_id,
            "candidate_id": candidate_id,
            "pilot_seed": seed,
            "final_seed_results_included": False,
            "training_exit_receipt": receipt_ref,
            "benchmark_artifacts": {
                "summary_json": summary_ref,
                "raw_samples_csv": raw_ref,
            },
        }
        sampling = {"diffusion_type": "udlm", "num_steps": nfe}
        _PILOT_RUN_RESULTS[(attempt_id, seed)] = {
            "attempt_id": attempt_id,
            "candidate_id": candidate_id,
            "pilot_seed": seed,
            "requested_samples": requested_samples,
            "nfe": nfe,
            "metric_branch": metric_branch,
            "checkpoint": {
                "sha256": checkpoint_sha256,
                "size_bytes": 1234,
                "global_step": 500,
            },
            "evaluation_config": {
                "relative_path": (
                    "scripts/exps/denovo/hparams_udlm_schedule_uniform.yaml"
                ),
                "sha256": "5" * 64,
            },
            "sampling": {
                "config": sampling,
                "sha256": gate.canonical_json_sha256(sampling),
            },
            "inference_weights": _inference_weights(),
            "runner_sha256": "e" * 64,
            "sampler_source_sha256": "d" * 64,
            "implementation_inputs_sha256": gate.canonical_json_sha256(
                {"sampler_source": {"sha256": "d" * 64}}
            ),
            "metric_inputs_sha256": gate.canonical_json_sha256({"schema_version": 1}),
            "benchmark_revision": "b" * 40,
            "started_at_utc": "2026-09-06T00:20:00+00:00",
            "completed_at_utc": "2026-09-06T00:30:00+00:00",
            "training_exit_receipt": {
                **receipt_ref,
                "recorded_at_utc": "2026-09-06T00:10:00+00:00",
            },
            "summary_json_sha256": summary_ref["sha256"],
            "raw_samples_csv_sha256": raw_ref["sha256"],
            "quality": quality,
            "diversity": diversity,
            "independent_rescore": {
                "all_21_fields_match": True,
                "both_metric_branches_match": True,
                "failure_counts_match": True,
                "raw_model_text_redecoded": True,
            },
        }
        payload = _json_bytes(evidence)
        blobs[relative_path] = payload
        refs.append(
            {
                "artifact_kind": "pilot_evaluation",
                "pilot_seed": seed,
                "relative_path": relative_path,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "schema_version": gate.PILOT_EVIDENCE_SCHEMA_VERSION,
            }
        )
    registered_operating_point = (
        seeds == gate.REGISTERED_SELECTION_PILOT_SEEDS
        and requested_samples == gate.REGISTERED_SELECTION_SAMPLES_PER_SEED
        and nfe == gate.REGISTERED_SELECTION_NFE
        and metric_branch == gate.REGISTERED_SELECTION_METRIC_BRANCH
    )
    registered = registered_operating_point and all(
        diversity is not None for diversity in diversities
    )
    return (
        {
            "attempt_id": attempt_id,
            "candidate_id": candidate_id,
            "status": "completed",
            "eligible_for_selection": registered,
            "ineligibility_reason": (
                None
                if registered
                else (
                    gate.UNDEFINED_SELECTION_METRIC_REASON
                    if registered_operating_point
                    else gate.NONREGISTERED_OPERATING_POINT_REASON
                )
            ),
            "pilot_seeds": list(seeds),
            "selection_score": (
                {
                    "mean_released_quality": statistics.fmean(qualities),
                    "mean_released_diversity": statistics.fmean(diversities),
                }
                if registered
                else None
            ),
            "artifact_refs": refs,
        },
        blobs,
    )


def _failed_pilot_attempt(
    *,
    attempt_id: str,
    candidate_id: str,
    seed: int,
    source_revision: str = "b" * 40,
    pilot_mode: str = "engineering",
    requested_samples: int = 32,
) -> tuple[dict, dict[str, bytes]]:
    relative_path = f"experiments/udlm/pilots/{attempt_id}/seed_{seed}.json"
    root = Path(gate.REPOSITORY_ROOT).absolute()
    project_root = root.parent.parent if root.parent.name == "run_sources" else root
    run_directory = f"output/benchmarks/selection/{attempt_id}/seed_{seed}"
    receipt_relative_path = f"{run_directory}/failure_receipt.json"
    config_relative_path = "scripts/exps/denovo/hparams_udlm_schedule_uniform.yaml"
    checkpoint_relative_path = f"output/udlm/{candidate_id}/checkpoints/500.ckpt"
    checkpoint_payload = b"synthetic failed-pilot checkpoint\n"
    checkpoint_sha256 = hashlib.sha256(checkpoint_payload).hexdigest()
    log_name = f"denovo_step500_{checkpoint_sha256[:12]}_seed{seed}.log"
    log_relative_path = f"output/logs/selection/{attempt_id}/{log_name}"
    launcher_payload = b"# synthetic benchmark launcher\n"
    config_payload = (
        b"diffusion_type: udlm\n"
        b"softmax_temp: 1.0\n"
        b"randomness: 0.5\n"
        b"min_add_len: 40\n"
        b"num_steps: 128\n"
        b"inference_eps: 0.00001\n"
        b"exclude_special_tokens: true\n"
        b"prior_variant: release_uniform\n"
        b"prior_metadata_sha256: null\n"
    )
    log_payload = b'{"event":"launch"}\nsynthetic failure\n'
    sampling = {
        "diffusion_type": "udlm",
        "softmax_temp": 1.0,
        "randomness": 0.5,
        "min_add_len": 40,
        "num_steps": 128,
        "inference_eps": 1e-5,
        "exclude_special_tokens": True,
        "prior_variant": "release_uniform",
        "prior_metadata_sha256": None,
    }
    checkpoint_path = root / checkpoint_relative_path
    config_path = root / config_relative_path
    run_dir = root / run_directory
    receipt = {
        "schema_version": gate.PILOT_FAILURE_RECEIPT_SCHEMA_VERSION,
        "artifact_kind": gate.PILOT_FAILURE_RECEIPT_KIND,
        "status": "failed",
        "attempt_id": attempt_id,
        "candidate_id": candidate_id,
        "pilot_seed": seed,
        "pilot_mode": pilot_mode,
        "requested_samples": requested_samples,
        "stage": "benchmark_child_process",
        "reason": "synthetic pilot process exited nonzero",
        "started_at_utc": "2026-09-06T00:20:00+00:00",
        "failed_at_utc": "2026-09-06T00:21:00+00:00",
        "process_exit_status": 17,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "size_bytes": len(checkpoint_payload),
            "global_step": 500,
        },
        "config": {
            "path": str(config_path),
            "sha256": hashlib.sha256(config_payload).hexdigest(),
            "sampling": sampling,
            "sampling_sha256": gate.canonical_json_sha256(sampling),
        },
        "command": [
            str(project_root / ".venv/bin/python"),
            str(root / "scripts/exps/denovo/benchmark.py"),
            "--checkpoint",
            str(checkpoint_path),
            "--expected-checkpoint-sha256",
            checkpoint_sha256,
            "--expected-source-revision",
            source_revision,
            "--config",
            str(config_path),
            "--expected-config-sha256",
            hashlib.sha256(config_payload).hexdigest(),
            "--num-samples",
            str(requested_samples),
            "--seed",
            str(seed),
            "--device",
            "cuda:0",
            "--output-dir",
            str(run_dir),
        ],
        "source_revision": {
            "head": source_revision,
            "upstream": source_revision,
        },
        "launcher_source": {
            "path": str(root / gate.DENOVO_LAUNCHER_RELATIVE_PATH),
            "sha256": hashlib.sha256(launcher_payload).hexdigest(),
            "size_bytes": len(launcher_payload),
        },
        "log": {
            "path": str(root / log_relative_path),
            "sha256": hashlib.sha256(log_payload).hexdigest(),
            "size_bytes": len(log_payload),
        },
        "partial_artifacts": {
            "summary_json": None,
            "raw_samples_csv": None,
        },
    }
    receipt_payload = _json_bytes(receipt)
    evidence = {
        "schema_version": gate.PILOT_EVIDENCE_SCHEMA_VERSION,
        "artifact_kind": "pilot_failure",
        "status": "failed",
        "attempt_id": attempt_id,
        "candidate_id": candidate_id,
        "pilot_seed": seed,
        "final_seed_results_included": False,
        "failure_receipt": {
            "relative_path": receipt_relative_path,
            "sha256": hashlib.sha256(receipt_payload).hexdigest(),
            "schema_version": gate.PILOT_FAILURE_RECEIPT_SCHEMA_VERSION,
        },
    }
    payload = _json_bytes(evidence)
    return (
        {
            "attempt_id": attempt_id,
            "candidate_id": candidate_id,
            "status": "failed",
            "eligible_for_selection": False,
            "ineligibility_reason": gate.FAILED_PILOT_REASON,
            "pilot_seeds": [seed],
            "selection_score": None,
            "artifact_refs": [
                {
                    "artifact_kind": "pilot_failure",
                    "pilot_seed": seed,
                    "relative_path": relative_path,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "schema_version": gate.PILOT_EVIDENCE_SCHEMA_VERSION,
                }
            ],
        },
        {
            relative_path: payload,
            receipt_relative_path: receipt_payload,
            gate.DENOVO_LAUNCHER_RELATIVE_PATH.as_posix(): launcher_payload,
            config_relative_path: config_payload,
            checkpoint_relative_path: checkpoint_payload,
            log_relative_path: log_payload,
        },
    )


def _pilot_ledger(*, attempts: list[dict], selected_attempt_id: str) -> dict:
    selected = next(
        attempt for attempt in attempts if attempt["attempt_id"] == selected_attempt_id
    )
    return {
        "schema_version": gate.CANDIDATE_LEDGER_SCHEMA_VERSION,
        "protocol_id": gate.EXPECTED_PROTOCOL_ID,
        "status": "closed_before_final_evaluation",
        "final_seed_results_included": False,
        "attempts": attempts,
        "selection": {
            "candidate_id": selected["candidate_id"],
            "selected_attempt_id": selected_attempt_id,
            "rule": gate.CANDIDATE_SELECTION_RULE,
            "checkpoint_selection_rule": gate.CHECKPOINT_SELECTION_RULE,
            "selected_without_final_seed_results": True,
        },
    }


def _artifact_loader(blobs: dict[str, bytes]):
    return lambda relative_path: blobs[relative_path.as_posix()]


def _pilot_run_validator(evidence: dict) -> dict:
    return copy.deepcopy(
        _PILOT_RUN_RESULTS[(evidence["attempt_id"], evidence["pilot_seed"])]
    )


def _metric_branch(*, unique_count: int, quality_count: int, diversity: float) -> dict:
    return {
        "validity": 1.0,
        "valid_count": 1000,
        "validity_denominator": 1000,
        "uniqueness": unique_count / 1000,
        "unique_count": unique_count,
        "uniqueness_denominator": 1000,
        "quality": quality_count / 1000,
        "quality_count": quality_count,
        "quality_denominator": 1000,
        "diversity": diversity,
    }


def _aggregate(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values),
        "values_by_seed": [
            {"seed": seed, "value": value}
            for seed, value in zip(gate.EXPECTED_SEEDS, values, strict=True)
        ],
    }


def _candidate_report(
    *,
    unique_counts: tuple[int, int, int] = (1000, 1000, 1000),
    quality_counts: tuple[int, int, int] = (900, 900, 900),
    diversities: tuple[float, float, float] = (0.84, 0.84, 0.84),
) -> dict:
    lock = _candidate_lock()
    seed_runs = []
    released_values = {metric: [] for metric in gate.METRICS}
    for seed, unique_count, quality_count, diversity in zip(
        gate.EXPECTED_SEEDS,
        unique_counts,
        quality_counts,
        diversities,
        strict=True,
    ):
        branch = _metric_branch(
            unique_count=unique_count,
            quality_count=quality_count,
            diversity=diversity,
        )
        for metric in gate.METRICS:
            released_values[metric].append(branch[metric])
        seed_runs.append(
            {
                "seed": seed,
                "started_at_utc": f"2026-09-06T01:0{seed + 1}:00+00:00",
                "summary_path": str(
                    gate.REPOSITORY_ROOT
                    / "output/udlm/final/schedule-uniform-synthetic"
                    / f"seed_{seed}/summary.json"
                ),
                "raw_samples_sha256": f"{seed + 6:x}" * 64,
                "summary_sha256": f"{seed + 9:x}" * 64,
                "metrics": {
                    "released_comparable": branch,
                    "strict": dict(branch),
                },
                "git": {
                    "commit": "b" * 40,
                    "dirty": False,
                },
                "inference_weights": _inference_weights(),
            }
        )
    aggregates = {
        metric: _aggregate(values) for metric, values in released_values.items()
    }
    return {
        "schema_version": gate.denovo_report.REPORT_SCHEMA_VERSION,
        "status": "completed",
        "required_protocol": {
            "seeds": [0, 1, 2],
            "samples_per_seed": 1000,
            "seed_count": 3,
            "total_requested_samples": 3000,
        },
        "checkpoint": {
            "diffusion_type": "udlm",
            "sha256": lock["training"]["checkpoint"]["sha256"],
            "size_bytes": lock["training"]["checkpoint"]["size_bytes"],
            "global_step": lock["training"]["checkpoint"]["global_step"],
        },
        "config": {
            "sha256": lock["inference"]["evaluation_config_sha256"],
            "sampling_sha256": lock["inference"]["sampling_sha256"],
            "sampling": lock["inference"]["sampling_config"],
            "git_tracking": {
                "relative_path": lock["inference"]["evaluation_config_relative_path"]
            },
        },
        "generation_protocol": {
            "diffusion_type": "udlm",
            "nfe": 128,
            "num_steps": 128,
            "nfe_by_seed": [{"seed": seed, "nfe": 128} for seed in gate.EXPECTED_SEEDS],
            "inference_weights": _inference_weights(),
        },
        "inference_weights": _inference_weights(),
        "runner_sha256": "e" * 64,
        "implementation_inputs": {"sampler_source": {"sha256": "d" * 64}},
        "metric_inputs": {"schema_version": 1},
        "seed_runs": seed_runs,
        "aggregate_metrics": {
            "released_comparable": aggregates,
            "strict": copy.deepcopy(aggregates),
        },
        "environment_consistency": {
            "all_seed_signatures_equal": True,
            "all_launch_policies_equal": True,
            "idle_gpu_policy_verified": True,
            "distinct_raw_sample_csv_sha256": True,
        },
    }


def _candidate_rescore_attestation(report: dict) -> dict:
    return {
        "status": "completed_exact_match",
        "seed_results": [
            {
                "seed": row["seed"],
                "summary_sha256": row["summary_sha256"],
                "raw_samples_sha256": row["raw_samples_sha256"],
                "worker_environment": {
                    "python_hash_seed": str(row["seed"]),
                    "device": "cpu",
                    "cuda_visible_devices": "",
                    "nvidia_visible_devices": "",
                },
                "all_21_fields_match": True,
                "both_metric_branches_match": True,
                "failure_counts_match": True,
            }
            for row in report["seed_runs"]
        ],
        "fresh_seed_specific_cpu_workers": True,
        "raw_model_text_redecoded": True,
        "qed_sa_and_diversity_recomputed": True,
    }


def _evaluate_candidate_report(
    report: dict,
    baseline: dict,
    baseline_rescore: dict,
    protocol: dict,
    candidate_lock: dict,
) -> dict:
    return gate.evaluate_candidate_report(
        report,
        baseline,
        baseline_rescore,
        protocol,
        candidate_lock,
        independent_candidate_rescore=_candidate_rescore_attestation(report),
    )


@pytest.fixture
def protocol() -> dict:
    return _load_json(gate.PROTOCOL_RELATIVE_PATH)


@pytest.fixture
def baseline() -> dict:
    return _load_json(gate.BASELINE_RELATIVE_PATH)


@pytest.fixture
def baseline_rescore() -> dict:
    return _load_json(gate.BASELINE_RESCORE_RELATIVE_PATH)


def test_pinned_protocol_and_baseline_hashes_and_semantics(
    protocol, baseline, baseline_rescore
):
    protocol_bytes = (gate.REPOSITORY_ROOT / gate.PROTOCOL_RELATIVE_PATH).read_bytes()
    previous_protocol_bytes = (
        gate.REPOSITORY_ROOT / gate.PREVIOUS_PROTOCOL_RELATIVE_PATH
    ).read_bytes()
    baseline_bytes = (gate.REPOSITORY_ROOT / gate.BASELINE_RELATIVE_PATH).read_bytes()
    baseline_rescore_bytes = (
        gate.REPOSITORY_ROOT / gate.BASELINE_RESCORE_RELATIVE_PATH
    ).read_bytes()
    floor_audit_bytes = (
        gate.REPOSITORY_ROOT / gate.PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT["relative_path"]
    ).read_bytes()
    floor_audit = gate.strict_json_loads(
        floor_audit_bytes, label="empirical-prior floor audit"
    )

    assert hashlib.sha256(protocol_bytes).hexdigest() == gate.PROTOCOL_SHA256
    assert (
        hashlib.sha256(previous_protocol_bytes).hexdigest()
        == gate.PREVIOUS_PROTOCOL_SHA256
    )
    assert hashlib.sha256(baseline_bytes).hexdigest() == gate.BASELINE_SHA256
    assert (
        hashlib.sha256(baseline_rescore_bytes).hexdigest()
        == gate.BASELINE_RESCORE_SHA256
    )
    assert (
        hashlib.sha256(floor_audit_bytes).hexdigest()
        == (gate.PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT["sha256"])
    )
    assert gate.canonical_json_sha256(floor_audit) == (
        gate.PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_CANONICAL_SHA256
    )
    gate.validate_protocol(protocol)
    validated = gate.validate_baseline_manifest(baseline)
    rescore = gate.validate_baseline_rescore_attestation(baseline_rescore, baseline)
    assert validated["means"]["quality"] == pytest.approx(0.858)
    assert validated["pooled_valid"] == 3000
    assert rescore["source_revision"] == gate.EXPECTED_BASELINE_RESCORE_SOURCE_REVISION
    assert rescore["all_rows_and_manifest_values_exact_match"] is True
    assert rescore["network_controls"]["os_or_process_network_isolation"] is False


def test_protocol_validation_requires_live_empirical_prior_floor_audit(
    tmp_path, monkeypatch, protocol
):
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", tmp_path)

    with pytest.raises(gate.GateValidationError, match="floor audit is unavailable"):
        gate.validate_protocol(protocol)


def test_protocol_validation_rejects_tampered_empirical_prior_floor_audit(
    tmp_path, monkeypatch, protocol
):
    source_path = (
        gate.REPOSITORY_ROOT / gate.PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT["relative_path"]
    )
    tampered = json.loads(source_path.read_bytes())
    tampered["recommendation"]["recommended_uniform_mixture_weight"] = 0.01
    destination = tmp_path / gate.PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT["relative_path"]
    destination.parent.mkdir(parents=True)
    destination.write_bytes(_json_bytes(tampered))
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", tmp_path)

    with pytest.raises(gate.GateValidationError, match="floor audit SHA-256 mismatch"):
        gate.validate_protocol(protocol)


@pytest.mark.parametrize(
    ("section", "field", "value", "error_match"),
    [
        ("git", "commit", "0" * 40, "source revision is unexpected"),
        ("data_use", "split", "validation", "data-use scope is unexpected"),
        (
            "recommendation",
            "recommended_uniform_mixture_weight",
            0.01,
            "recommendation is unexpected",
        ),
    ],
)
def test_protocol_validation_rechecks_floor_audit_semantics_after_hashes(
    monkeypatch, protocol, section, field, value, error_match
):
    original_load = gate.load_pinned_json

    def tampered_load(relative_path, expected_sha256, *, label):
        document, payload = original_load(relative_path, expected_sha256, label=label)
        if relative_path == Path(
            gate.PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT["relative_path"]
        ):
            document = copy.deepcopy(document)
            document[section][field] = value
            monkeypatch.setattr(
                gate,
                "PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_CANONICAL_SHA256",
                gate.canonical_json_sha256(document),
            )
        return document, payload

    monkeypatch.setattr(gate, "load_pinned_json", tampered_load)

    with pytest.raises(gate.GateValidationError, match=error_match):
        gate.validate_protocol(protocol)


def test_public_evaluator_rejects_weakened_protocol_with_frozen_hash_label(
    protocol, baseline, baseline_rescore
):
    protocol["uncertainty_gates"]["quality"][
        "candidate_minus_baseline_lower_bound_strictly_greater_than"
    ] = -1.0

    with pytest.raises(gate.GateValidationError, match="not the frozen value"):
        _evaluate_candidate_report(
            _candidate_report(quality_counts=(875, 876, 877)),
            baseline,
            baseline_rescore,
            protocol,
            _candidate_lock(),
        )


def test_protocol_semantically_pins_registered_pilot_operating_point(
    protocol, monkeypatch
):
    protocol["selection_firewall"]["eligible_nfe"] = 1
    monkeypatch.setattr(
        gate, "PROTOCOL_CANONICAL_SHA256", gate.canonical_json_sha256(protocol)
    )

    with pytest.raises(gate.GateValidationError, match="eligible pilot NFE"):
        gate.validate_protocol(protocol)


@pytest.mark.parametrize(
    "payload",
    [b'{"a": 1, "a": 2}', b'{"a": NaN}', b'{"a": Infinity}'],
)
def test_strict_json_rejects_duplicates_and_nonfinite_numbers(payload):
    with pytest.raises(gate.GateValidationError):
        gate.strict_json_loads(payload, label="test payload")


def test_boundary_safe_newcombe_interval_for_two_all_success_samples():
    result = gate.newcombe_wilson_lower_difference(
        3000,
        3000,
        3000,
        3000,
        z=1.6448536269514722,
    )

    assert result["difference"] == 0.0
    assert result["lower_bound"] == pytest.approx(-0.0009010352213835171)
    assert result["lower_bound"] > -0.005


def test_welch_interval_uses_independent_seed_level_estimates():
    result = gate.welch_lower_difference(
        [0.90, 0.90, 0.90],
        [0.849, 0.872, 0.853],
        confidence=0.95,
    )

    assert result["difference"] == pytest.approx(0.042)
    assert result["degrees_of_freedom"] == pytest.approx(2.0)
    assert result["lower_bound"] == pytest.approx(0.021283873558568735)


def test_both_zero_variance_welch_inputs_fail_closed():
    with pytest.raises(gate.GateValidationError, match="both sample variances"):
        gate.welch_lower_difference([0.9, 0.9, 0.9], [0.8, 0.8, 0.8], confidence=0.95)


def test_one_zero_variance_welch_input_remains_defined():
    result = gate.welch_lower_difference(
        [0.9, 0.9, 0.9], [0.8, 0.81, 0.82], confidence=0.95
    )

    assert result["degrees_of_freedom"] == pytest.approx(2.0)
    assert result["lower_bound"] == pytest.approx(0.07314145539151921)


def test_complete_registered_gate_passes_strong_candidate(
    protocol, baseline, baseline_rescore
):
    report = _candidate_report()
    decision = _evaluate_candidate_report(
        report,
        baseline,
        baseline_rescore,
        protocol,
        _candidate_lock(),
    )

    assert decision["superiority_gate_passed"] is True
    assert decision["all_point_estimate_gates_passed"] is True
    assert decision["all_uncertainty_gates_passed"] is True
    assert decision["candidate"]["nfe"] == 128
    assert decision["candidate"]["inference_weights"] == _inference_weights()
    assert decision["baseline"]["rescore_attestation"]["status"] == (
        "completed_exact_match"
    )
    assert decision["claim"]["scope"] == "operational_continuation_only"
    assert decision["claim"]["method_only_claim_supported"] is False


def test_public_evaluator_requires_matching_candidate_raw_rescore(
    protocol, baseline, baseline_rescore
):
    report = _candidate_report()
    with pytest.raises(TypeError, match="independent_candidate_rescore"):
        gate.evaluate_candidate_report(
            report,
            baseline,
            baseline_rescore,
            protocol,
            _candidate_lock(),
        )

    attestation = _candidate_rescore_attestation(report)
    attestation["seed_results"][1]["raw_samples_sha256"] = "0" * 64
    with pytest.raises(gate.GateValidationError, match="artifact hashes differ"):
        gate.evaluate_candidate_report(
            report,
            baseline,
            baseline_rescore,
            protocol,
            _candidate_lock(),
            independent_candidate_rescore=attestation,
        )


def test_quality_equality_fails_strict_point_and_overall_gate(
    protocol, baseline, baseline_rescore
):
    report = _candidate_report(quality_counts=(858, 858, 858))
    decision = _evaluate_candidate_report(
        report,
        baseline,
        baseline_rescore,
        protocol,
        _candidate_lock(),
    )

    assert decision["metrics"]["quality"]["point_gate_passed"] is False
    assert decision["superiority_gate_passed"] is False


def test_diversity_point_margin_is_inclusive(protocol, baseline, baseline_rescore):
    threshold = baseline["released_comparable"]["mean"]["diversity"] - 0.005
    report = _candidate_report(diversities=(threshold, threshold, threshold))
    decision = _evaluate_candidate_report(
        report,
        baseline,
        baseline_rescore,
        protocol,
        _candidate_lock(),
    )

    assert decision["metrics"]["diversity"]["point_gate_passed"] is True


def test_any_wrong_final_nfe_is_rejected(protocol, baseline, baseline_rescore):
    report = _candidate_report()
    report["generation_protocol"]["nfe_by_seed"][1]["nfe"] = 64

    with pytest.raises(gate.GateValidationError, match="NFE differs"):
        _evaluate_candidate_report(
            report, baseline, baseline_rescore, protocol, _candidate_lock()
        )


def test_final_run_must_start_strictly_after_candidate_lock(
    protocol, baseline, baseline_rescore
):
    candidate_lock = _candidate_lock()
    report = _candidate_report()
    report["seed_runs"][0]["started_at_utc"] = candidate_lock["locked_at_utc"]

    with pytest.raises(gate.GateValidationError, match="start strictly after"):
        _evaluate_candidate_report(
            report, baseline, baseline_rescore, protocol, candidate_lock
        )


def test_runtime_ema_application_receipt_is_required(
    protocol, baseline, baseline_rescore
):
    report = _candidate_report()
    report["generation_protocol"]["inference_weights"] = {
        "source": "raw_model",
        "ema_applied": False,
        "ema": None,
    }

    with pytest.raises(gate.GateValidationError, match="EMA inference weights"):
        _evaluate_candidate_report(
            report, baseline, baseline_rescore, protocol, _candidate_lock()
        )


@pytest.mark.parametrize("location", ["top_level", "seed"])
def test_runtime_ema_receipt_must_be_identical_everywhere(
    protocol, baseline, baseline_rescore, location
):
    report = _candidate_report()
    if location == "top_level":
        report["inference_weights"]["ema"]["num_updates"] = 499
    else:
        report["seed_runs"][1]["inference_weights"]["ema"]["decay"] = 0.9

    with pytest.raises(
        gate.GateValidationError, match="inference-weight receipt disagrees"
    ):
        _evaluate_candidate_report(
            report, baseline, baseline_rescore, protocol, _candidate_lock()
        )


def test_observed_runner_must_equal_prelocked_runner(
    protocol, baseline, baseline_rescore
):
    report = _candidate_report()
    report["runner_sha256"] = "0" * 64

    with pytest.raises(gate.GateValidationError, match="runner differs"):
        _evaluate_candidate_report(
            report, baseline, baseline_rescore, protocol, _candidate_lock()
        )


def test_full_implementation_and_metric_maps_must_be_prelocked(
    protocol, baseline, baseline_rescore
):
    report = _candidate_report()
    report["implementation_inputs"]["extra_source"] = {"sha256": "0" * 64}
    with pytest.raises(gate.GateValidationError, match="implementation inputs differ"):
        _evaluate_candidate_report(
            report, baseline, baseline_rescore, protocol, _candidate_lock()
        )

    report = _candidate_report()
    report["metric_inputs"]["unexpected"] = True
    with pytest.raises(gate.GateValidationError, match="metric inputs differ"):
        _evaluate_candidate_report(
            report, baseline, baseline_rescore, protocol, _candidate_lock()
        )


def _final_rescore_fixture(protocol: dict) -> tuple[dict, dict]:
    candidate_lock = _candidate_lock()
    report = _candidate_report()
    report["implementation_inputs"]["ema_source"] = {"sha256": "c" * 64}
    candidate_lock["inference"]["implementation_inputs_sha256"] = (
        gate.canonical_json_sha256(report["implementation_inputs"])
    )
    for row in report["seed_runs"]:
        seed = row["seed"]
        row["raw_samples_path"] = str(
            gate.REPOSITORY_ROOT
            / "output/udlm/final/schedule-uniform-synthetic"
            / f"seed_{seed}/raw_samples.csv"
        )
        row["failure_counts"] = {"released_decode_failed": 0}
    return report, gate.validate_candidate_lock(candidate_lock, protocol)


def _final_rescore_worker_result(
    report: dict, lock: dict, call: dict, *, tamper_quality: bool = False
) -> dict:
    seed = call["expected_seed"]
    row = next(item for item in report["seed_runs"] if item["seed"] == seed)
    metrics = copy.deepcopy(row["metrics"])
    if tamper_quality and seed == 1:
        metrics["released_comparable"]["quality"] -= 0.1
    return {
        "status": "exact_match",
        "seed": seed,
        "summary_sha256": row["summary_sha256"],
        "raw_samples_sha256": row["raw_samples_sha256"],
        "row_comparison": {
            "all_match": True,
            "row_count": gate.EXPECTED_SAMPLES_PER_SEED,
            "field_count": gate.EXPECTED_BASELINE_RAW_SAMPLE_FIELD_COUNT,
            "cell_count": (
                gate.EXPECTED_SAMPLES_PER_SEED
                * gate.EXPECTED_BASELINE_RAW_SAMPLE_FIELD_COUNT
            ),
            "numeric_absolute_tolerance": 1e-12,
            "field_results": [
                {
                    "field": field,
                    "comparison": (
                        "finite_numeric_absolute_tolerance_1e-12_or_exact_null"
                        if field in gate.NUMERIC_RAW_SAMPLE_FIELDS
                        else "exact_value_and_type"
                    ),
                    "compared_rows": gate.EXPECTED_SAMPLES_PER_SEED,
                    "mismatch_count": 0,
                    "max_absolute_difference": (
                        0.0 if field in gate.NUMERIC_RAW_SAMPLE_FIELDS else None
                    ),
                }
                for field in gate.denovo_report.RAW_SAMPLE_FIELDS
            ],
        },
        "metrics": metrics,
        "failure_counts": row["failure_counts"],
        "identity": {
            "seed": seed,
            "sample_count": gate.EXPECTED_SAMPLES_PER_SEED,
            "started_at_utc": "2026-09-06T00:00:00+00:00",
            "completed_at_utc": "2026-09-06T00:01:00+00:00",
            "checkpoint": {
                key: lock["checkpoint"][key]
                for key in ("sha256", "size_bytes", "global_step")
            },
            "config": {
                "sha256": lock["evaluation_config_sha256"],
                "sampling": lock["sampling_config"],
                "sampling_sha256": lock["sampling_sha256"],
            },
            "generation": {
                "nfe": gate.EXPECTED_NFE,
                "inference_weights": lock["inference_weights"],
            },
            "source": {
                "revision": row["git"]["commit"],
                "runner_sha256": lock["benchmark_runner_sha256"],
                "sampler_source_sha256": lock["sampler_source_sha256"],
                "ema_source_sha256": "c" * 64,
                "implementation_inputs_sha256": lock["implementation_inputs_sha256"],
                "metric_inputs_sha256": lock["metric_inputs_sha256"],
            },
            "artifacts": {},
        },
        "independent_recomputation": {
            "raw_input_field": "raw_model_text",
            "decoder": "benchmark.decode_records",
            "metric_evaluator": "benchmark.evaluate_records",
            "qed_recomputed": True,
            "sa_recomputed": True,
            "released_diversity_recomputed": True,
            "strict_branch_recomputed": True,
            "all_21_raw_fields_compared": True,
            "numeric_absolute_tolerance": 1e-12,
        },
        "stable_inputs": {
            name: {
                "path": str(call[path_key]),
                "sha256": row[hash_key],
                "size_bytes": 123,
                "read_policy": (
                    "regular_file_no_symlink_stable_descriptor_bytes_retained_in_memory"
                ),
            }
            for name, path_key, hash_key in (
                ("summary_json", "summary_path", "summary_sha256"),
                ("raw_samples_csv", "raw_samples_path", "raw_samples_sha256"),
            )
        }
        | {
            "recorded_path_bindings": {
                name: {
                    "recorded_path": str(call[path_key]),
                    "supplied_resolved_path": str(call[path_key]),
                    "exact_path_match": True,
                }
                for name, path_key in (
                    ("summary_json", "summary_path"),
                    ("raw_samples_csv", "raw_samples_path"),
                )
            },
            "revalidated_unchanged_after_rescore": True,
        },
        "worker_environment": {
            "python_hash_seed": str(seed),
            "device": "cpu",
            "cuda_visible_devices": "",
            "nvidia_visible_devices": "",
            "offline_environment": {
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "WANDB_MODE": "offline",
                "WANDB_DISABLED": "true",
            },
            "python_network_guard_during_computation": {
                "guarded_apis": [
                    "socket.create_connection",
                    "socket.getaddrinfo",
                    "socket.socket.connect",
                    "socket.socket.connect_ex",
                ],
                "scope_limitation": "synthetic test proof",
            },
            "executable": "/synthetic/.venv/bin/python",
            "python": "3.synthetic",
            "platform": "synthetic",
            "pid": 123,
        },
    }


def test_final_candidate_is_independently_rescored_from_each_raw_csv(protocol):
    report, lock = _final_rescore_fixture(protocol)
    calls = []

    def worker(**call):
        calls.append(call)
        return _final_rescore_worker_result(report, lock, call)

    result = gate.independently_rescore_candidate_runs(
        report, lock, worker_invoker=worker
    )

    assert [call["expected_seed"] for call in calls] == [0, 1, 2]
    assert all(call["expected_sample_count"] == 1000 for call in calls)
    assert all(call["expected_ema_source_sha256"] == "c" * 64 for call in calls)
    assert result["status"] == "completed_exact_match"
    assert result["fresh_seed_specific_cpu_workers"] is True
    assert result["raw_model_text_redecoded"] is True
    assert len(result["seed_results"]) == 3


def test_final_candidate_rescore_rejects_recomputed_metric_tampering(protocol):
    report, lock = _final_rescore_fixture(protocol)

    def worker(**call):
        return _final_rescore_worker_result(report, lock, call, tamper_quality=True)

    with pytest.raises(gate.GateValidationError, match="released_comparable.quality"):
        gate.independently_rescore_candidate_runs(report, lock, worker_invoker=worker)


def test_final_candidate_rescore_rejects_shallow_worker_proof(protocol):
    report, lock = _final_rescore_fixture(protocol)

    def worker(**call):
        result = _final_rescore_worker_result(report, lock, call)
        result.pop("stable_inputs")
        return result

    with pytest.raises(gate.GateValidationError, match="fields are invalid"):
        gate.independently_rescore_candidate_runs(report, lock, worker_invoker=worker)


def test_lock_rejects_final_seed_leak_and_non_ema_weights(protocol):
    candidate_lock = _candidate_lock()
    candidate_lock["selection"]["selection_rule"] = "highest eligible quality"
    with pytest.raises(gate.GateValidationError, match="frozen enum"):
        gate.validate_candidate_lock(candidate_lock, protocol)

    candidate_lock = _candidate_lock()
    candidate_lock["selection"]["final_seeds_used_during_selection"] = [0]
    with pytest.raises(gate.GateValidationError, match="must not be used"):
        gate.validate_candidate_lock(candidate_lock, protocol)

    candidate_lock = _candidate_lock()
    candidate_lock["inference"]["weights"] = "raw"
    with pytest.raises(gate.GateValidationError, match="must be EMA"):
        gate.validate_candidate_lock(candidate_lock, protocol)

    candidate_lock = _candidate_lock()
    candidate_lock["inference"]["inference_weights"]["ema"]["num_updates"] = 1
    with pytest.raises(gate.GateValidationError, match="must equal optimizer updates"):
        gate.validate_candidate_lock(candidate_lock, protocol)

    candidate_lock = _candidate_lock()
    candidate_lock["training"]["training_summary"]["schema_version"] = 1
    with pytest.raises(gate.GateValidationError, match="summary schema version"):
        gate.validate_candidate_lock(candidate_lock, protocol)

    candidate_lock = _candidate_lock()
    candidate_lock["training"]["exit_receipt"]["schema_version"] = 1
    with pytest.raises(gate.GateValidationError, match="receipt schema version"):
        gate.validate_candidate_lock(candidate_lock, protocol)

    candidate_lock = _candidate_lock()
    candidate_lock["training"]["runtime_config"]["schema_version"] = 1
    with pytest.raises(gate.GateValidationError, match="runtime-config schema version"):
        gate.validate_candidate_lock(candidate_lock, protocol)

    candidate_lock = _candidate_lock()
    candidate_lock["training"]["launch_manifest"]["schema_version"] = 1
    with pytest.raises(
        gate.GateValidationError, match="launch-manifest schema version"
    ):
        gate.validate_candidate_lock(candidate_lock, protocol)

    candidate_lock = _candidate_lock()
    candidate_lock["training"]["launch_manifest"]["relative_path"] = (
        "output/udlm/synthetic-candidate/not_the_launch.json"
    )
    with pytest.raises(gate.GateValidationError, match="launch_manifest.json"):
        gate.validate_candidate_lock(candidate_lock, protocol)


def test_candidate_lock_accepts_exact_film_parameter_schema(protocol):
    candidate_lock = _candidate_lock(conditioning_variant="film_adaln")

    normalized = gate.validate_candidate_lock(candidate_lock, protocol)

    assert normalized["parameter_counts"] == {
        "base_model_trainable": 86_000_000,
        "time_conditioner_trainable": 787_968,
        "film_modulation_trainable": 262_656,
        "total_trainable": 87_050_624,
    }


@pytest.mark.parametrize(
    ("mutation", "error_fragment"),
    [
        ("extra", "exactly the additive 3-key or film_adaln 4-key schema"),
        ("zero_film", "must be at least 1"),
        ("bad_total", "do not add up"),
    ],
)
def test_candidate_lock_rejects_invalid_parameter_schema(
    protocol, mutation, error_fragment
):
    candidate_lock = _candidate_lock(conditioning_variant="film_adaln")
    counts = candidate_lock["training"]["parameter_counts"]
    if mutation == "extra":
        counts["unexpected"] = 1
    elif mutation == "zero_film":
        counts["film_modulation_trainable"] = 0
    else:
        counts["total_trainable"] -= 1

    with pytest.raises(gate.GateValidationError, match=error_fragment):
        gate.validate_candidate_lock(candidate_lock, protocol)


def test_scratch_lock_has_narrow_single_trajectory_scope(protocol):
    candidate_lock = _candidate_lock(startup_mode="scratch")

    validated = gate.validate_candidate_lock(candidate_lock, protocol)

    assert validated["startup_mode"] == "scratch"
    assert validated["claim_scope"] == (
        "single_training_trajectory_checkpoint_comparison_only"
    )


def test_tampered_baseline_count_is_rejected(baseline):
    baseline["released_comparable"]["per_seed"][0]["quality_count"] += 1

    with pytest.raises(gate.GateValidationError, match="quality"):
        gate.validate_baseline_manifest(baseline)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("status",), "completed", "not completed exact-match"),
        (
            ("source", "clean_pushed_checks", "before_computation"),
            False,
            "clean-source check",
        ),
        (("source", "revision"), "0" * 40, "source revision is unexpected"),
        (
            ("source", "files", "benchmark_runner", "sha256"),
            "0" * 64,
            "runtime module.*unbound from source",
        ),
        (
            ("implementation", "benchmark_schema_version"),
            6,
            "schemas are stale",
        ),
        (
            ("implementation", "metric_inputs_sha256"),
            "0" * 64,
            "metric-input self-hash",
        ),
        (
            ("implementation", "runtime_modules_sha256"),
            "0" * 64,
            "runtime-module self-hash",
        ),
        (
            ("protocol", "network_controls", "os_or_process_network_isolation"),
            True,
            "network-control claim",
        ),
        (
            ("seed_results", 0, "row_comparison", "all_match"),
            False,
            "row comparison all_match",
        ),
        (
            ("seed_results", 0, "inputs", "raw_samples_csv", "sha256"),
            "0" * 64,
            "digest disagrees with frozen manifest",
        ),
        (
            ("seed_results", 0, "metrics", "strict", "quality_count"),
            1,
            "quality_count",
        ),
        (
            ("seed_results", 0, "failure_counts", "strict_decode_failed"),
            11,
            "strict_decode_failed",
        ),
        (
            ("aggregate_metrics", "strict", "quality", "mean"),
            0.1,
            "mean",
        ),
        (
            ("strict_vs_repaired_funnel", "strict_valid"),
            1,
            "funnel disagrees",
        ),
        (
            (
                "manifest_comparison",
                "all_seed_rows_metrics_failures_hashes_and_aggregates_match",
            ),
            False,
            "manifest comparison is incomplete",
        ),
    ],
)
def test_baseline_rescore_attestation_tampering_fails_closed(
    baseline, baseline_rescore, path, value, message
):
    cursor = baseline_rescore
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] = value

    with pytest.raises(gate.GateValidationError, match=message):
        gate.validate_baseline_rescore_attestation(baseline_rescore, baseline)


def test_public_evaluator_cannot_bypass_baseline_rescore_attestation(
    protocol, baseline, baseline_rescore
):
    baseline_rescore["seed_results"][2]["manifest_comparison"][
        "all_counts_metrics_and_artifact_hashes_match"
    ] = False

    with pytest.raises(gate.GateValidationError, match="manifest comparison"):
        _evaluate_candidate_report(
            _candidate_report(),
            baseline,
            baseline_rescore,
            protocol,
            _candidate_lock(),
        )


def test_candidate_ledger_recomputes_scores_and_selects_deterministically():
    lower, lower_blobs = _completed_pilot_attempt(
        attempt_id="a1",
        candidate_id="schedule-low",
        seeds=(1000, 1001),
        qualities=(0.80, 0.84),
        diversities=(0.70, 0.72),
    )
    winner, winner_blobs = _completed_pilot_attempt(
        attempt_id="a2",
        candidate_id="schedule-uniform-synthetic",
        seeds=(1000, 1001),
        qualities=(0.85, 0.85),
        diversities=(0.74, 0.74),
    )
    lexical_loser, loser_blobs = _completed_pilot_attempt(
        attempt_id="z2",
        candidate_id="schedule-equal",
        seeds=(1000, 1001),
        qualities=(0.85, 0.85),
        diversities=(0.74, 0.74),
    )
    failed, failed_blobs = _failed_pilot_attempt(
        attempt_id="f1", candidate_id="schedule-failed", seed=1004
    )
    blobs = lower_blobs | winner_blobs | loser_blobs | failed_blobs
    ledger = _pilot_ledger(
        attempts=[failed, lexical_loser, lower, winner],
        selected_attempt_id="a2",
    )

    result = gate.validate_candidate_ledger(
        ledger,
        candidate_id="schedule-uniform-synthetic",
        artifact_loader=_artifact_loader(blobs),
        pilot_run_validator=_pilot_run_validator,
    )

    assert result["attempt_count"] == 4
    assert result["eligible_attempt_count"] == 3
    assert result["committed_pilot_artifact_count"] == 7
    assert result["ineligible_completed_attempt_count"] == 0
    assert result["failed_attempt_count"] == 1
    assert result["selected_attempt_id"] == "a2"
    assert result["selected_checkpoint_sha256"] == "4" * 64
    assert result["selected_score"] == {
        "mean_released_quality": 0.85,
        "mean_released_diversity": 0.74,
    }
    assert result["selection_recomputed_from_pilot_evidence"] is True


def test_completed_pilot_references_must_be_git_blobs_at_benchmark_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt, blobs = _completed_pilot_attempt(
        attempt_id="git-proof",
        candidate_id="schedule-uniform-synthetic",
        seeds=(1000,),
        qualities=(0.85,),
        diversities=(0.74,),
    )
    envelope_ref = attempt["artifact_refs"][0]
    evidence = json.loads(blobs[envelope_ref["relative_path"]])

    def git_blob(_revision: str, relative_path: Path) -> bytes:
        try:
            return blobs[relative_path.as_posix()]
        except KeyError as error:
            raise gate.GateValidationError(
                f"missing Git blob: {relative_path}"
            ) from error

    monkeypatch.setattr(gate, "_git_blob", git_blob)
    gate._require_completed_pilot_references_at_revision(evidence, "a" * 40)

    raw_path = evidence["benchmark_artifacts"]["raw_samples_csv"]["relative_path"]
    raw_bytes = blobs.pop(raw_path)
    with pytest.raises(gate.GateValidationError, match="missing Git blob"):
        gate._require_completed_pilot_references_at_revision(evidence, "a" * 40)

    blobs[raw_path] = raw_bytes + b"tampered"
    with pytest.raises(gate.GateValidationError, match="Git blob digest differs"):
        gate._require_completed_pilot_references_at_revision(evidence, "a" * 40)


def test_partially_failed_attempt_retains_and_validates_completed_seed():
    winner, winner_blobs = _completed_pilot_attempt(
        attempt_id="winner",
        candidate_id="schedule-uniform-synthetic",
        seeds=gate.REGISTERED_SELECTION_PILOT_SEEDS,
        qualities=(0.85, 0.85),
        diversities=(0.74, 0.74),
    )
    partial, partial_success_blobs = _completed_pilot_attempt(
        attempt_id="partial",
        candidate_id="schedule-partial",
        seeds=(1000,),
        qualities=(0.99,),
        diversities=(0.99,),
    )
    failure, failure_blobs = _failed_pilot_attempt(
        attempt_id="partial",
        candidate_id="schedule-partial",
        seed=1001,
        pilot_mode="registered_selection",
        requested_samples=gate.REGISTERED_SELECTION_SAMPLES_PER_SEED,
    )
    partial.update(
        {
            "status": "failed",
            "eligible_for_selection": False,
            "ineligibility_reason": gate.FAILED_PILOT_REASON,
            "pilot_seeds": [1000, 1001],
            "selection_score": None,
            "artifact_refs": partial["artifact_refs"] + failure["artifact_refs"],
        }
    )
    ledger = _pilot_ledger(attempts=[partial, winner], selected_attempt_id="winner")

    result = gate.validate_candidate_ledger(
        ledger,
        candidate_id="schedule-uniform-synthetic",
        artifact_loader=_artifact_loader(
            winner_blobs | partial_success_blobs | failure_blobs
        ),
        pilot_run_validator=_pilot_run_validator,
    )

    assert result["partially_failed_attempt_count"] == 1
    assert result["failed_attempt_count"] == 1
    assert any(
        row["attempt_id"] == "partial" and row["pilot_seed"] == 1000
        for row in result["completed_outcomes"]
    )
    assert result["failed_outcomes"][0]["attempt_id"] == "partial"
    assert result["selected_attempt_id"] == "winner"


def test_candidate_ledger_accepts_producer_valid_empty_failure_support():
    winner, winner_blobs = _completed_pilot_attempt(
        attempt_id="winner-empty-support",
        candidate_id="schedule-uniform-synthetic",
        seeds=gate.REGISTERED_SELECTION_PILOT_SEEDS,
        qualities=(0.85, 0.85),
        diversities=(0.74, 0.74),
    )
    failed, failed_blobs = _failed_pilot_attempt(
        attempt_id="empty-support", candidate_id="schedule-failed", seed=1004
    )
    ledger = _pilot_ledger(
        attempts=[failed, winner], selected_attempt_id="winner-empty-support"
    )
    failure_ref = ledger["attempts"][0]["artifact_refs"][0]
    evidence = json.loads(failed_blobs[failure_ref["relative_path"]])
    receipt_ref = evidence["failure_receipt"]
    receipt = json.loads(failed_blobs[receipt_ref["relative_path"]])
    empty_sha256 = hashlib.sha256(b"").hexdigest()
    log_path = Path(receipt["log"]["path"]).relative_to(gate.REPOSITORY_ROOT)
    failed_blobs[log_path.as_posix()] = b""
    receipt["log"].update({"sha256": empty_sha256, "size_bytes": 0})
    summary_path = Path(receipt_ref["relative_path"]).parent / "summary.json"
    failed_blobs[summary_path.as_posix()] = b""
    receipt["partial_artifacts"]["summary_json"] = {
        "path": str(Path(gate.REPOSITORY_ROOT).absolute() / summary_path),
        "sha256": empty_sha256,
        "size_bytes": 0,
    }
    receipt_bytes = _json_bytes(receipt)
    failed_blobs[receipt_ref["relative_path"]] = receipt_bytes
    receipt_ref["sha256"] = hashlib.sha256(receipt_bytes).hexdigest()
    envelope_bytes = _json_bytes(evidence)
    failed_blobs[failure_ref["relative_path"]] = envelope_bytes
    failure_ref["sha256"] = hashlib.sha256(envelope_bytes).hexdigest()

    result = gate.validate_candidate_ledger(
        ledger,
        candidate_id="schedule-uniform-synthetic",
        artifact_loader=_artifact_loader(winner_blobs | failed_blobs),
        pilot_run_validator=_pilot_run_validator,
    )

    assert result["failed_attempt_count"] == 1


def test_candidate_ledger_rejects_failure_log_in_benchmark_attempt_root():
    winner, winner_blobs = _completed_pilot_attempt(
        attempt_id="winner-distinct-log",
        candidate_id="schedule-uniform-synthetic",
        seeds=gate.REGISTERED_SELECTION_PILOT_SEEDS,
        qualities=(0.85, 0.85),
        diversities=(0.74, 0.74),
    )
    failed, failed_blobs = _failed_pilot_attempt(
        attempt_id="same-log-root", candidate_id="schedule-failed", seed=1004
    )
    ledger = _pilot_ledger(
        attempts=[failed, winner], selected_attempt_id="winner-distinct-log"
    )
    failure_ref = ledger["attempts"][0]["artifact_refs"][0]
    evidence = json.loads(failed_blobs[failure_ref["relative_path"]])
    receipt_ref = evidence["failure_receipt"]
    receipt = json.loads(failed_blobs[receipt_ref["relative_path"]])
    old_log_path = Path(receipt["log"]["path"]).relative_to(gate.REPOSITORY_ROOT)
    log_bytes = failed_blobs[old_log_path.as_posix()]
    new_log_path = Path(receipt_ref["relative_path"]).parent.parent / old_log_path.name
    failed_blobs[new_log_path.as_posix()] = log_bytes
    receipt["log"]["path"] = str(Path(gate.REPOSITORY_ROOT).absolute() / new_log_path)
    receipt_bytes = _json_bytes(receipt)
    failed_blobs[receipt_ref["relative_path"]] = receipt_bytes
    receipt_ref["sha256"] = hashlib.sha256(receipt_bytes).hexdigest()
    envelope_bytes = _json_bytes(evidence)
    failed_blobs[failure_ref["relative_path"]] = envelope_bytes
    failure_ref["sha256"] = hashlib.sha256(envelope_bytes).hexdigest()

    with pytest.raises(gate.GateValidationError, match="distinct attempt root"):
        gate.validate_candidate_ledger(
            ledger,
            candidate_id="schedule-uniform-synthetic",
            artifact_loader=_artifact_loader(winner_blobs | failed_blobs),
            pilot_run_validator=_pilot_run_validator,
        )


def test_candidate_ledger_rejects_worktree_local_failure_interpreter():
    winner, winner_blobs = _completed_pilot_attempt(
        attempt_id="winner-interpreter",
        candidate_id="schedule-uniform-synthetic",
        seeds=gate.REGISTERED_SELECTION_PILOT_SEEDS,
        qualities=(0.85, 0.85),
        diversities=(0.74, 0.74),
    )
    failed, failed_blobs = _failed_pilot_attempt(
        attempt_id="wrong-interpreter", candidate_id="schedule-failed", seed=1004
    )
    ledger = _pilot_ledger(
        attempts=[failed, winner], selected_attempt_id="winner-interpreter"
    )
    failure_ref = ledger["attempts"][0]["artifact_refs"][0]
    evidence = json.loads(failed_blobs[failure_ref["relative_path"]])
    receipt_ref = evidence["failure_receipt"]
    receipt = json.loads(failed_blobs[receipt_ref["relative_path"]])
    worktree_python = Path(gate.REPOSITORY_ROOT).absolute() / ".venv/bin/python"
    assert worktree_python != gate._project_training_python_executable()
    receipt["command"][0] = str(worktree_python)
    receipt_bytes = _json_bytes(receipt)
    failed_blobs[receipt_ref["relative_path"]] = receipt_bytes
    receipt_ref["sha256"] = hashlib.sha256(receipt_bytes).hexdigest()
    envelope_bytes = _json_bytes(evidence)
    failed_blobs[failure_ref["relative_path"]] = envelope_bytes
    failure_ref["sha256"] = hashlib.sha256(envelope_bytes).hexdigest()

    with pytest.raises(
        gate.GateValidationError, match="interpreter is not project .venv"
    ):
        gate.validate_candidate_ledger(
            ledger,
            candidate_id="schedule-uniform-synthetic",
            artifact_loader=_artifact_loader(winner_blobs | failed_blobs),
            pilot_run_validator=_pilot_run_validator,
        )


def test_candidate_ledger_enforces_seed_status_and_frozen_rule():
    attempt, blobs = _completed_pilot_attempt(
        attempt_id="a1",
        candidate_id="schedule-uniform-synthetic",
        seeds=(1000, 1001),
        qualities=(0.85, 0.85),
        diversities=(0.74, 0.74),
    )
    ledger = _pilot_ledger(attempts=[attempt], selected_attempt_id="a1")
    ledger["attempts"][0]["pilot_seeds"] = [0]
    with pytest.raises(gate.GateValidationError, match="greater than or equal to 1000"):
        gate.validate_candidate_ledger(
            ledger,
            candidate_id="schedule-uniform-synthetic",
            artifact_loader=_artifact_loader(blobs),
            pilot_run_validator=_pilot_run_validator,
        )

    attempt["pilot_seeds"] = [1000, 1001]
    ledger = _pilot_ledger(attempts=[attempt], selected_attempt_id="a1")
    ledger["selection"]["rule"] = "highest eligible quality"
    with pytest.raises(gate.GateValidationError, match="frozen enum"):
        gate.validate_candidate_ledger(
            ledger,
            candidate_id="schedule-uniform-synthetic",
            artifact_loader=_artifact_loader(blobs),
            pilot_run_validator=_pilot_run_validator,
        )


def test_exact_registered_attempt_cannot_be_arbitrarily_excluded():
    attempt, blobs = _completed_pilot_attempt(
        attempt_id="a1",
        candidate_id="schedule-uniform-synthetic",
        seeds=gate.REGISTERED_SELECTION_PILOT_SEEDS,
        qualities=(0.85, 0.85),
        diversities=(0.74, 0.74),
    )
    attempt["eligible_for_selection"] = False
    attempt["selection_score"] = None
    attempt["ineligibility_reason"] = gate.NONREGISTERED_OPERATING_POINT_REASON
    ledger = _pilot_ledger(attempts=[attempt], selected_attempt_id="a1")

    with pytest.raises(gate.GateValidationError, match="eligibility disagrees"):
        gate.validate_candidate_ledger(
            ledger,
            candidate_id="schedule-uniform-synthetic",
            artifact_loader=_artifact_loader(blobs),
            pilot_run_validator=_pilot_run_validator,
        )


@pytest.mark.parametrize(
    "operating_point_override",
    [
        {"requested_samples": 1},
        {"nfe": 64},
    ],
)
def test_one_sample_or_mismatched_nfe_attempt_is_disclosed_but_ineligible(
    operating_point_override,
):
    registered, registered_blobs = _completed_pilot_attempt(
        attempt_id="registered",
        candidate_id="schedule-uniform-synthetic",
        seeds=gate.REGISTERED_SELECTION_PILOT_SEEDS,
        qualities=(0.80, 0.80),
        diversities=(0.70, 0.70),
    )
    engineering, engineering_blobs = _completed_pilot_attempt(
        attempt_id="engineering",
        candidate_id="schedule-noisy-health",
        seeds=gate.REGISTERED_SELECTION_PILOT_SEEDS,
        qualities=(1.0, 1.0),
        diversities=(1.0, 1.0),
        **operating_point_override,
    )
    ledger = _pilot_ledger(
        attempts=[engineering, registered], selected_attempt_id="registered"
    )

    result = gate.validate_candidate_ledger(
        ledger,
        candidate_id="schedule-uniform-synthetic",
        artifact_loader=_artifact_loader(registered_blobs | engineering_blobs),
        pilot_run_validator=_pilot_run_validator,
    )

    assert result["eligible_attempt_count"] == 1
    assert result["ineligible_completed_attempt_count"] == 1
    assert result["selected_attempt_id"] == "registered"

    engineering["eligible_for_selection"] = True
    engineering["ineligibility_reason"] = None
    engineering["selection_score"] = {
        "mean_released_quality": 1.0,
        "mean_released_diversity": 1.0,
    }
    with pytest.raises(gate.GateValidationError, match="eligibility disagrees"):
        gate.validate_candidate_ledger(
            ledger,
            candidate_id="schedule-uniform-synthetic",
            artifact_loader=_artifact_loader(registered_blobs | engineering_blobs),
            pilot_run_validator=_pilot_run_validator,
        )


def test_valid_health_pilot_with_nonregistered_seed_is_disclosed():
    registered, registered_blobs = _completed_pilot_attempt(
        attempt_id="registered",
        candidate_id="schedule-uniform-synthetic",
        seeds=gate.REGISTERED_SELECTION_PILOT_SEEDS,
        qualities=(0.80, 0.80),
        diversities=(0.70, 0.70),
    )
    health, health_blobs = _completed_pilot_attempt(
        attempt_id="health",
        candidate_id="schedule-health-only",
        seeds=(1002,),
        qualities=(1.0,),
        diversities=(1.0,),
        requested_samples=1,
        nfe=10,
    )
    ledger = _pilot_ledger(
        attempts=[health, registered], selected_attempt_id="registered"
    )

    result = gate.validate_candidate_ledger(
        ledger,
        candidate_id="schedule-uniform-synthetic",
        artifact_loader=_artifact_loader(registered_blobs | health_blobs),
        pilot_run_validator=_pilot_run_validator,
    )

    assert health["ineligibility_reason"] == (gate.NONREGISTERED_OPERATING_POINT_REASON)
    assert result["eligible_attempt_count"] == 1
    assert result["ineligible_completed_attempt_count"] == 1
    assert result["selected_attempt_id"] == "registered"


@pytest.mark.parametrize(
    ("field", "tampered_value"),
    [
        ("attempt_id", "different-attempt"),
        ("candidate_id", "different-candidate"),
        ("pilot_seed", 1001),
        ("final_seed_results_included", True),
    ],
)
def test_candidate_ledger_rejects_semantically_misbound_artifact(field, tampered_value):
    attempt, blobs = _completed_pilot_attempt(
        attempt_id="a1",
        candidate_id="schedule-uniform-synthetic",
        seeds=(1000, 1001),
        qualities=(0.85, 0.85),
        diversities=(0.74, 0.74),
    )
    ledger = _pilot_ledger(attempts=[attempt], selected_attempt_id="a1")
    ref = ledger["attempts"][0]["artifact_refs"][0]
    evidence = json.loads(blobs[ref["relative_path"]])
    evidence[field] = tampered_value
    tampered = _json_bytes(evidence)
    blobs[ref["relative_path"]] = tampered
    ref["sha256"] = hashlib.sha256(tampered).hexdigest()

    with pytest.raises(gate.GateValidationError, match=f"{field} disagrees"):
        gate.validate_candidate_ledger(
            ledger,
            candidate_id="schedule-uniform-synthetic",
            artifact_loader=_artifact_loader(blobs),
            pilot_run_validator=_pilot_run_validator,
        )


def test_candidate_ledger_rejects_declared_score_tampering():
    attempt, blobs = _completed_pilot_attempt(
        attempt_id="a1",
        candidate_id="schedule-uniform-synthetic",
        seeds=(1000, 1001),
        qualities=(0.80, 0.90),
        diversities=(0.70, 0.80),
    )
    ledger = _pilot_ledger(attempts=[attempt], selected_attempt_id="a1")
    ledger["attempts"][0]["selection_score"]["mean_released_quality"] = 0.99

    with pytest.raises(gate.GateValidationError, match="mean released quality"):
        gate.validate_candidate_ledger(
            ledger,
            candidate_id="schedule-uniform-synthetic",
            artifact_loader=_artifact_loader(blobs),
            pilot_run_validator=_pilot_run_validator,
        )


def test_candidate_ledger_rejects_wrong_winner_including_lexical_tie_break():
    first, first_blobs = _completed_pilot_attempt(
        attempt_id="a-first",
        candidate_id="schedule-uniform-synthetic",
        seeds=(1000, 1001),
        qualities=(0.85, 0.85),
        diversities=(0.74, 0.74),
    )
    later, later_blobs = _completed_pilot_attempt(
        attempt_id="z-later",
        candidate_id="schedule-uniform-synthetic",
        seeds=(1000, 1001),
        qualities=(0.85, 0.85),
        diversities=(0.74, 0.74),
    )
    ledger = _pilot_ledger(attempts=[later, first], selected_attempt_id="z-later")

    with pytest.raises(
        gate.GateValidationError, match="deterministic pilot-score winner"
    ):
        gate.validate_candidate_ledger(
            ledger,
            candidate_id="schedule-uniform-synthetic",
            artifact_loader=_artifact_loader(first_blobs | later_blobs),
            pilot_run_validator=_pilot_run_validator,
        )


def test_candidate_ledger_rejects_content_free_and_malformed_failure_evidence():
    attempt, blobs = _completed_pilot_attempt(
        attempt_id="a1",
        candidate_id="schedule-uniform-synthetic",
        seeds=(1000, 1001),
        qualities=(0.85, 0.85),
        diversities=(0.74, 0.74),
    )
    ledger = _pilot_ledger(attempts=[attempt], selected_attempt_id="a1")
    ref = ledger["attempts"][0]["artifact_refs"][0]
    content_free = b'{"status":"completed"}\n'
    blobs[ref["relative_path"]] = content_free
    ref["sha256"] = hashlib.sha256(content_free).hexdigest()
    with pytest.raises(gate.GateValidationError, match="fields are invalid"):
        gate.validate_candidate_ledger(
            ledger,
            candidate_id="schedule-uniform-synthetic",
            artifact_loader=_artifact_loader(blobs),
            pilot_run_validator=_pilot_run_validator,
        )

    winner, winner_blobs = _completed_pilot_attempt(
        attempt_id="a1",
        candidate_id="schedule-uniform-synthetic",
        seeds=(1000, 1001),
        qualities=(0.85, 0.85),
        diversities=(0.74, 0.74),
    )
    failed, failed_blobs = _failed_pilot_attempt(
        attempt_id="f1", candidate_id="schedule-failed", seed=1001
    )
    ledger = _pilot_ledger(attempts=[winner, failed], selected_attempt_id="a1")
    failure_ref = ledger["attempts"][1]["artifact_refs"][0]
    failure_evidence = json.loads(failed_blobs[failure_ref["relative_path"]])
    receipt_ref = failure_evidence["failure_receipt"]
    failure_receipt = json.loads(failed_blobs[receipt_ref["relative_path"]])
    failure_receipt["reason"] = ""
    malformed_receipt = _json_bytes(failure_receipt)
    failed_blobs[receipt_ref["relative_path"]] = malformed_receipt
    receipt_ref["sha256"] = hashlib.sha256(malformed_receipt).hexdigest()
    malformed_envelope = _json_bytes(failure_evidence)
    failed_blobs[failure_ref["relative_path"]] = malformed_envelope
    failure_ref["sha256"] = hashlib.sha256(malformed_envelope).hexdigest()
    with pytest.raises(gate.GateValidationError, match="reason must be nonempty"):
        gate.validate_candidate_ledger(
            ledger,
            candidate_id="schedule-uniform-synthetic",
            artifact_loader=_artifact_loader(winner_blobs | failed_blobs),
            pilot_run_validator=_pilot_run_validator,
        )


def test_completed_matched_panel_accepts_full_r_s_e_with_selected_s(
    tmp_path, monkeypatch, protocol
):
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", tmp_path)
    candidate_lock = _candidate_lock()
    selected_documents = _write_valid_training_evidence(tmp_path, candidate_lock)
    terminal_documents = _write_terminal_e_training_evidence(
        tmp_path, candidate_lock, selected_documents
    )
    normalized = gate.validate_candidate_lock(candidate_lock, protocol)
    selected_evidence = gate.validate_training_evidence(normalized)

    completion = gate.validate_completed_matched_panel(normalized, selected_evidence)
    r_receipt_snapshot = selected_documents["manifest"]["predecessor_receipt_binding"][
        "receipt_artifact"
    ]

    assert completion == {
        "terminal_e_run_name": "e-terminal",
        "terminal_e_exit_receipt_sha256": terminal_documents["receipt_snapshot"][
            "sha256"
        ],
        "terminal_e_exit_receipt_recorded_at_utc": ("2026-09-06T00:30:01+00:00"),
        "selected_exit_receipt_recorded_at_utc": ("2026-09-06T00:10:01+00:00"),
        "candidate_locked_at_utc": "2026-09-06T01:00:00+00:00",
        "matched_panel_spec_sha256": selected_evidence["matched_panel_spec_sha256"],
        "chain_depth": 3,
        "variant_order": ["udlm", "schedule_uniform", "udlm_categorical"],
        "receipt_members": [
            {
                "variant": "udlm",
                "relative_path": Path(r_receipt_snapshot["path"])
                .relative_to(tmp_path)
                .as_posix(),
                "sha256": r_receipt_snapshot["sha256"],
            },
            {
                "variant": "schedule_uniform",
                "relative_path": candidate_lock["training"]["exit_receipt"][
                    "relative_path"
                ],
                "sha256": candidate_lock["training"]["exit_receipt"]["sha256"],
            },
            {
                "variant": "udlm_categorical",
                "relative_path": candidate_lock["selection"]["terminal_e_exit_receipt"][
                    "relative_path"
                ],
                "sha256": candidate_lock["selection"]["terminal_e_exit_receipt"][
                    "sha256"
                ],
            },
        ],
        "selected_receipt_membership_proved": True,
        "complete_registered_panel_proved": True,
    }


@pytest.mark.parametrize("terminal_variant", ["r", "s"])
def test_completed_matched_panel_rejects_nonterminal_r_or_s_receipt(
    tmp_path, monkeypatch, protocol, terminal_variant
):
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", tmp_path)
    candidate_lock = _candidate_lock()
    selected_documents = _write_valid_training_evidence(tmp_path, candidate_lock)
    if terminal_variant == "r":
        r_snapshot = selected_documents["manifest"]["predecessor_receipt_binding"][
            "receipt_artifact"
        ]
        terminal_reference = {
            "relative_path": str(Path(r_snapshot["path"]).relative_to(tmp_path)),
            "sha256": r_snapshot["sha256"],
            "schema_version": gate.PILOT_EXIT_STATUS_SCHEMA_VERSION,
        }
    else:
        terminal_reference = copy.deepcopy(candidate_lock["training"]["exit_receipt"])
    candidate_lock["selection"]["terminal_e_exit_receipt"] = terminal_reference
    normalized = gate.validate_candidate_lock(candidate_lock, protocol)
    selected_evidence = gate.validate_training_evidence(normalized)

    with pytest.raises(gate.GateValidationError, match="complete registered R/S/E"):
        gate.validate_completed_matched_panel(normalized, selected_evidence)


def test_completed_matched_panel_requires_selected_and_terminal_same_panel(
    tmp_path, monkeypatch, protocol
):
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", tmp_path)
    candidate_lock = _candidate_lock()
    selected_documents = _write_valid_training_evidence(tmp_path, candidate_lock)
    _write_terminal_e_training_evidence(tmp_path, candidate_lock, selected_documents)
    normalized = gate.validate_candidate_lock(candidate_lock, protocol)
    selected_evidence = gate.validate_training_evidence(normalized)
    selected_evidence["matched_panel_spec_sha256"] = "f" * 64

    with pytest.raises(gate.GateValidationError, match="share one matched panel"):
        gate.validate_completed_matched_panel(normalized, selected_evidence)


def test_completed_matched_panel_rejects_same_spec_selected_s_from_other_chain(
    tmp_path, monkeypatch, protocol
):
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", tmp_path)
    candidate_lock = _candidate_lock()
    selected_documents = _write_valid_training_evidence(tmp_path, candidate_lock)
    _write_terminal_e_training_evidence(tmp_path, candidate_lock, selected_documents)
    normalized = gate.validate_candidate_lock(candidate_lock, protocol)
    selected_evidence = gate.validate_training_evidence(normalized)

    # Model a separately completed S run with the same matched-panel digest.
    # Its receipt may be valid on its own, but it is not the S member named by
    # this terminal E chain and therefore cannot be cherry-picked as winner.
    normalized["receipt"] = {
        "relative_path": Path("output/udlm/other-s/pilot_exit_status.json"),
        "sha256": "f" * 64,
        "schema_version": gate.PILOT_EXIT_STATUS_SCHEMA_VERSION,
    }

    with pytest.raises(gate.GateValidationError, match="not a member"):
        gate.validate_completed_matched_panel(normalized, selected_evidence)


@pytest.mark.parametrize("tamper", ["locked_hash", "late_mutation"])
def test_completed_matched_panel_rejects_terminal_receipt_hash_tampering(
    tmp_path, monkeypatch, protocol, tamper
):
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", tmp_path)
    candidate_lock = _candidate_lock()
    selected_documents = _write_valid_training_evidence(tmp_path, candidate_lock)
    terminal_documents = _write_terminal_e_training_evidence(
        tmp_path, candidate_lock, selected_documents
    )
    if tamper == "locked_hash":
        candidate_lock["selection"]["terminal_e_exit_receipt"]["sha256"] = "f" * 64
    normalized = gate.validate_candidate_lock(candidate_lock, protocol)
    selected_evidence = gate.validate_training_evidence(normalized)
    if tamper == "late_mutation":
        terminal_documents["receipt_path"].write_bytes(
            terminal_documents["receipt_path"].read_bytes() + b" "
        )

    with pytest.raises(
        gate.GateValidationError, match="digest disagrees with candidate lock"
    ):
        gate.validate_completed_matched_panel(normalized, selected_evidence)


@pytest.mark.parametrize(
    "locked_at_utc",
    ["2026-09-06T00:30:01+00:00", "2026-09-06T00:30:00+00:00"],
)
def test_completed_matched_panel_terminal_receipt_must_strictly_predate_lock(
    tmp_path, monkeypatch, protocol, locked_at_utc
):
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", tmp_path)
    candidate_lock = _candidate_lock()
    candidate_lock["locked_at_utc"] = locked_at_utc
    selected_documents = _write_valid_training_evidence(tmp_path, candidate_lock)
    _write_terminal_e_training_evidence(tmp_path, candidate_lock, selected_documents)
    normalized = gate.validate_candidate_lock(candidate_lock, protocol)
    selected_evidence = gate.validate_training_evidence(normalized)

    with pytest.raises(gate.GateValidationError, match="strictly predate"):
        gate.validate_completed_matched_panel(normalized, selected_evidence)


@pytest.mark.parametrize(
    "selected_recorded_at_utc",
    ["2026-09-06T01:00:00+00:00", "2026-09-06T01:00:01+00:00"],
)
def test_completed_matched_panel_selected_receipt_must_strictly_predate_lock(
    tmp_path, monkeypatch, protocol, selected_recorded_at_utc
):
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", tmp_path)
    candidate_lock = _candidate_lock()
    selected_documents = _write_valid_training_evidence(tmp_path, candidate_lock)
    _write_terminal_e_training_evidence(tmp_path, candidate_lock, selected_documents)
    normalized = gate.validate_candidate_lock(candidate_lock, protocol)
    selected_evidence = gate.validate_training_evidence(normalized)
    selected_evidence["exit_receipt_recorded_at_utc"] = selected_recorded_at_utc

    with pytest.raises(gate.GateValidationError, match="strictly predate"):
        gate.validate_completed_matched_panel(normalized, selected_evidence)


@pytest.mark.parametrize(
    ("acquired_at_utc", "accepted"),
    [
        ("2026-09-05T23:59:57+00:00", True),
        ("2026-09-05T23:59:58+00:00", True),
        ("2026-09-05T23:59:59+00:00", False),
    ],
)
def test_launch_manifest_lock_must_be_acquired_no_later_than_gpu_inventory(
    tmp_path, monkeypatch, protocol, acquired_at_utc, accepted
):
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", tmp_path)
    candidate_lock = _candidate_lock()
    _write_valid_training_evidence(
        tmp_path,
        candidate_lock,
        lock_acquired_at_utc=acquired_at_utc,
    )
    normalized = gate.validate_candidate_lock(candidate_lock, protocol)

    if accepted:
        assert (
            gate.validate_training_evidence(normalized)["successful_exit_receipt"]
            is True
        )
    else:
        with pytest.raises(
            gate.GateValidationError,
            match="lock was acquired after GPU inventory probing",
        ):
            gate.validate_training_evidence(normalized)


@pytest.mark.parametrize(
    "lock_acquired_at_utc",
    ["2026-09-05T23:59:00+00:00", "2026-09-05T23:58:59+00:00"],
)
def test_predecessor_receipt_must_predate_successor_lock_and_gpu_probe(
    tmp_path, monkeypatch, protocol, lock_acquired_at_utc
):
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", tmp_path)
    candidate_lock = _candidate_lock()
    _write_valid_training_evidence(
        tmp_path,
        candidate_lock,
        lock_acquired_at_utc=lock_acquired_at_utc,
    )
    normalized = gate.validate_candidate_lock(candidate_lock, protocol)

    with pytest.raises(
        gate.GateValidationError,
        match="predecessor receipt must predate successor lock acquisition",
    ):
        gate.validate_training_evidence(normalized)


def test_training_summary_and_exit_receipt_are_joined_to_lock(
    tmp_path, monkeypatch, protocol
):
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", tmp_path)
    candidate_lock = _candidate_lock()
    documents = _write_valid_training_evidence(tmp_path, candidate_lock)
    normalized = gate.validate_candidate_lock(candidate_lock, protocol)

    evidence = gate.validate_training_evidence(normalized)

    manifest_argv = documents["manifest"]["training_argv"]
    assert manifest_argv[:3] == [
        str(tmp_path / ".venv/bin/python"),
        "-u",
        str(tmp_path / "scripts/train.py"),
    ]
    assert (
        gate.canonical_json_sha256(manifest_argv[2:])
        == (candidate_lock["training"]["training_argv_sha256"])
    )
    assert documents["runtime"]["training_argv"] == manifest_argv[2:]
    assert evidence["ema_finite_and_checkpoint_bound"] is True
    assert evidence["successful_exit_receipt"] is True
    assert evidence["training_accounting"] == documents["training_accounting"]
    assert (
        evidence["launch_manifest_sha256"]
        == candidate_lock["training"]["launch_manifest"]["sha256"]
    )
    assert evidence["selected_gpu_uuids"] == ["GPU-synthetic-0001"]
    wrong_lock = copy.deepcopy(normalized)
    wrong_lock["parameter_counts"]["total_trainable"] += 1
    with pytest.raises(
        gate.GateValidationError, match="parameter counts disagree with lock"
    ):
        gate.validate_training_evidence(wrong_lock)
    wrong_lock = copy.deepcopy(normalized)
    wrong_lock["inference_weights"]["ema"]["decay"] = 0.9
    with pytest.raises(
        gate.GateValidationError, match="EMA metadata disagrees with lock"
    ):
        gate.validate_training_evidence(wrong_lock)
    receipt = documents["receipt"]
    receipt["overall_status"] = "failed"
    receipt_bytes = _json_bytes(receipt)
    receipt_path = (
        tmp_path / candidate_lock["training"]["exit_receipt"]["relative_path"]
    )
    receipt_path.write_bytes(receipt_bytes)
    normalized["receipt"]["sha256"] = hashlib.sha256(receipt_bytes).hexdigest()
    with pytest.raises(gate.GateValidationError, match="not completed"):
        gate.validate_training_evidence(normalized)


def test_training_evidence_accepts_film_parameter_schema_across_predecessor_chain(
    tmp_path, monkeypatch, protocol
):
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", tmp_path)
    candidate_lock = _candidate_lock(conditioning_variant="film_adaln")
    documents = _write_valid_training_evidence(tmp_path, candidate_lock)

    normalized = gate.validate_candidate_lock(candidate_lock, protocol)
    evidence = gate.validate_training_evidence(normalized)

    assert evidence["successful_exit_receipt"] is True
    assert evidence["training_accounting"]["trainable_parameter_counts"] == {
        "base_backbone": 86_000_000,
        "time_conditioner": 787_968,
        "film_modulation": 262_656,
        "total": 87_050_624,
    }
    assert (
        documents["manifest"]["resolved_training_config"]["training"]["udlm"][
            "conditioning_variant"
        ]
        == "film_adaln"
    )


@pytest.mark.parametrize(
    ("mutation", "error_fragment"),
    [
        ("wrong_variant", "warm-start conditioning variant disagrees"),
        ("wrong_count", "conditioning parameter tensor count disagrees"),
        ("missing_count", "warm-start report fields"),
        ("extra", "warm-start report fields"),
    ],
)
def test_training_evidence_rejects_invalid_film_warm_start_topology(
    tmp_path, monkeypatch, protocol, mutation, error_fragment
):
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", tmp_path)
    candidate_lock = _candidate_lock(conditioning_variant="film_adaln")

    def mutate_summary(summary):
        report = summary["startup"]["verified_mdlm_warm_start_report"]
        if mutation == "wrong_variant":
            report["conditioning_variant"] = "additive"
        elif mutation == "wrong_count":
            report["conditioning_parameter_tensors"] = 27
        elif mutation == "missing_count":
            report.pop("conditioning_parameter_tensors")
        else:
            report["unexpected"] = True

    _write_valid_training_evidence(
        tmp_path,
        candidate_lock,
        mutate_summary=mutate_summary,
    )
    normalized = gate.validate_candidate_lock(candidate_lock, protocol)

    with pytest.raises(gate.GateValidationError, match=error_fragment):
        gate.validate_training_evidence(normalized)


@pytest.mark.parametrize(
    ("lock_variant", "runtime_variant"),
    [("additive", "film_adaln"), ("film_adaln", "additive")],
)
def test_training_evidence_rejects_cross_topology_parameter_schema(
    tmp_path, monkeypatch, protocol, lock_variant, runtime_variant
):
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", tmp_path)
    candidate_lock = _candidate_lock(conditioning_variant=lock_variant)
    _write_valid_training_evidence(
        tmp_path,
        candidate_lock,
        conditioning_variant=runtime_variant,
    )
    normalized = gate.validate_candidate_lock(candidate_lock, protocol)

    with pytest.raises(
        gate.GateValidationError,
        match=("warm-start report fields|" "summary trainable parameter counts fields"),
    ):
        gate.validate_training_evidence(normalized)


def test_training_evidence_rejects_extra_locked_parameter_key(
    tmp_path, monkeypatch, protocol
):
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", tmp_path)
    candidate_lock = _candidate_lock()
    _write_valid_training_evidence(tmp_path, candidate_lock)
    normalized = gate.validate_candidate_lock(candidate_lock, protocol)
    normalized["parameter_counts"]["unexpected"] = 1

    with pytest.raises(
        gate.GateValidationError, match="locked trainable parameter counts fields"
    ):
        gate.validate_training_evidence(normalized)


def _set_manifest_gpu_state_fields(manifest, **updates):
    for field in (
        "gpu_inventory_at_selection",
        "initially_selected_gpu_states",
        "gpu_states_at_final_uuid_probe",
    ):
        manifest[field][0].update(copy.deepcopy(updates))


def _set_manifest_initial_gpu_state_fields(manifest, **updates):
    for field in ("gpu_inventory_at_selection", "initially_selected_gpu_states"):
        manifest[field][0].update(copy.deepcopy(updates))


def test_gate_accepts_recorded_process_below_utilization_threshold(
    tmp_path, monkeypatch, protocol
):
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", tmp_path)
    candidate_lock = _candidate_lock()
    process = {
        "pid": 4321,
        "process_name": "pre-existing-workload",
        "used_memory_mib": 512,
    }
    _write_valid_training_evidence(
        tmp_path,
        candidate_lock,
        mutate_manifest=lambda value: _set_manifest_gpu_state_fields(
            value,
            utilization_percent=9,
            compute_processes=[process],
        ),
    )
    normalized = gate.validate_candidate_lock(candidate_lock, protocol)

    evidence = gate.validate_training_evidence(normalized)

    assert evidence["successful_exit_receipt"] is True
    assert evidence["selected_gpu_uuids"] == ["GPU-synthetic-0001"]


def test_gate_rejects_sparse_recorded_process_evidence(tmp_path, monkeypatch, protocol):
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", tmp_path)
    candidate_lock = _candidate_lock()
    _write_valid_training_evidence(
        tmp_path,
        candidate_lock,
        mutate_manifest=lambda value: _set_manifest_gpu_state_fields(
            value,
            utilization_percent=9,
            compute_processes=[{"pid": 4321, "process_name": "missing-memory-field"}],
        ),
    )
    normalized = gate.validate_candidate_lock(candidate_lock, protocol)

    with pytest.raises(gate.GateValidationError, match="compute_processes.*fields"):
        gate.validate_training_evidence(normalized)


@pytest.mark.parametrize(
    "updates",
    [
        {"utilization_percent": 10},
        {"memory_used_mib": 51_921},
        {"compute_mode": "Prohibited"},
    ],
)
def test_gate_still_rejects_unsafe_final_gpu_state(
    tmp_path, monkeypatch, protocol, updates
):
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", tmp_path)
    candidate_lock = _candidate_lock()
    _write_valid_training_evidence(
        tmp_path,
        candidate_lock,
        mutate_manifest=lambda value: _set_manifest_gpu_state_fields(value, **updates),
    )
    normalized = gate.validate_candidate_lock(candidate_lock, protocol)

    with pytest.raises(gate.GateValidationError, match="violates safety policy"):
        gate.validate_training_evidence(normalized)


def test_gate_rejects_unsafe_initial_state_even_when_final_probe_is_safe(
    tmp_path, monkeypatch, protocol
):
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", tmp_path)
    candidate_lock = _candidate_lock()
    _write_valid_training_evidence(
        tmp_path,
        candidate_lock,
        mutate_manifest=lambda value: _set_manifest_initial_gpu_state_fields(
            value, utilization_percent=10
        ),
    )
    normalized = gate.validate_candidate_lock(candidate_lock, protocol)

    with pytest.raises(gate.GateValidationError, match="violates safety policy"):
        gate.validate_training_evidence(normalized)


def test_gate_accepts_either_fixed_reviewed_venv_path(tmp_path, monkeypatch, protocol):
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", tmp_path)
    candidate_lock = _candidate_lock()
    project_venv_python = tmp_path.parents[1] / ".venv/bin/python"
    _write_valid_training_evidence(
        tmp_path,
        candidate_lock,
        mutate_manifest=lambda document: document["training_argv"].__setitem__(
            0, str(project_venv_python)
        ),
    )
    normalized = gate.validate_candidate_lock(candidate_lock, protocol)

    evidence = gate.validate_training_evidence(normalized)

    assert evidence["successful_exit_receipt"] is True


@pytest.mark.parametrize("mutation", ["missing", "replaced_bytes"])
def test_gate_stream_verifies_the_live_training_checkpoint(
    tmp_path, monkeypatch, protocol, mutation
):
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", tmp_path)
    candidate_lock = _candidate_lock()
    _write_valid_training_evidence(tmp_path, candidate_lock)
    normalized = gate.validate_candidate_lock(candidate_lock, protocol)
    checkpoint_path = (
        tmp_path / candidate_lock["training"]["checkpoint"]["relative_path"]
    )
    if mutation == "missing":
        checkpoint_path.unlink()
        error_match = "training checkpoint is unavailable"
    else:
        checkpoint_path.write_bytes(b"x" * checkpoint_path.stat().st_size)
        error_match = "checkpoint bytes disagree"

    with pytest.raises(gate.GateValidationError, match=error_match):
        gate.validate_training_evidence(normalized)


@pytest.mark.parametrize(
    ("mutation_layer", "mutator", "error_match"),
    [
        (
            "manifest_panel",
            lambda document: document["matched_panel_spec"].__setitem__(
                "purpose", "tampered"
            ),
            "matched-panel specification is unbound",
        ),
        (
            "manifest_gpu_count",
            lambda document: document["cuda_visible_device_uuids"].append(
                "GPU-synthetic-0002"
            ),
            "count 2 disagrees with world size 1",
        ),
        (
            "manifest_unsafe_gpu_threshold",
            lambda document: document["gpu_safety_policy"].__setitem__(
                "max_utilization_percent", 99
            ),
            "GPU safety policy is unexpected",
        ),
        (
            "manifest_legacy_zero_process_policy",
            lambda document: document["gpu_safety_policy"].__setitem__(
                "active_compute_processes_allowed", False
            ),
            "GPU safety policy is unexpected",
        ),
        (
            "manifest_initial_inventory_mismatch",
            lambda document: document["initially_selected_gpu_states"].__setitem__(
                0,
                {
                    **document["initially_selected_gpu_states"][0],
                    "memory_used_mib": 1001,
                },
            ),
            "initial selection differs from its inventory snapshot",
        ),
        (
            "manifest_treatment_registry",
            lambda document: (
                document["matched_panel_spec"]["registered_treatments"][0].__setitem__(
                    "comparison_role", "forged_control"
                ),
                document.__setitem__(
                    "matched_panel_spec_sha256",
                    gate.canonical_json_sha256(document["matched_panel_spec"]),
                ),
            ),
            "matched-panel treatments are not canonical",
        ),
        (
            "manifest_empirical_floor_audit",
            lambda document: (
                document["matched_panel_spec"]["common_training_contract"][
                    "empirical_uniform_mix_audit"
                ].__setitem__("sha256", "0" * 64),
                document.__setitem__(
                    "matched_panel_spec_sha256",
                    gate.canonical_json_sha256(document["matched_panel_spec"]),
                ),
            ),
            "matched-panel contract disagrees with lock",
        ),
        (
            "manifest_empirical_floor",
            lambda document: (
                document["matched_panel_spec"]["common_training_contract"].__setitem__(
                    "empirical_uniform_mix", 0.01
                ),
                document.__setitem__(
                    "matched_panel_spec_sha256",
                    gate.canonical_json_sha256(document["matched_panel_spec"]),
                ),
            ),
            "matched-panel contract disagrees with lock",
        ),
        (
            "manifest_truncated_training_argv",
            lambda document: document.__setitem__(
                "training_argv", document["training_argv"][2:]
            ),
            "training argv producer prefix is unexpected",
        ),
        (
            "manifest_wrong_training_python",
            lambda document: document["training_argv"].__setitem__(
                0, "/unreviewed/python"
            ),
            "training argv producer prefix is unexpected",
        ),
        (
            "manifest_mismatched_run_name",
            lambda document: document.__setitem__("run_name", "different-run"),
            "path does not match launch run name",
        ),
        (
            "manifest_predecessor_completion_false",
            lambda document: document["completion_contract"][
                "successful_exit_receipt_requires"
            ].__setitem__("predecessor_receipt_binding_unchanged_and_valid", False),
            "completion contract is unexpected",
        ),
        (
            "manifest_predecessor_completion_missing",
            lambda document: document["completion_contract"][
                "successful_exit_receipt_requires"
            ].pop("predecessor_receipt_binding_unchanged_and_valid"),
            "completion contract is unexpected",
        ),
        (
            "manifest_final_probe_after_manifest",
            lambda document: document.__setitem__(
                "final_uuid_probes_completed_at_utc",
                "2026-09-06T00:00:01+00:00",
            ),
            "GPU-probe chronology is invalid",
        ),
        (
            "manifest_predecessor_state",
            lambda document: document["predecessor_receipt_binding"].__setitem__(
                "state", "explicit_genesis_no_predecessor"
            ),
            "S/E predecessor identity is not the prior arm",
        ),
        (
            "manifest_resolved_empirical_floor",
            lambda document: (
                document["resolved_training_config"]["training"]["udlm"].__setitem__(
                    "empirical_uniform_mix", 0.01
                ),
                document.__setitem__(
                    "resolved_training_config_sha256",
                    gate.canonical_json_sha256(document["resolved_training_config"]),
                ),
            ),
            "resolved training config is unbound",
        ),
        (
            "manifest_lock_record",
            lambda document: document["single_training_job_lock"]["record"].__setitem__(
                "owner_token", "c" * 64
            ),
            "training-job lock record digest is unbound",
        ),
        (
            "runtime_gpu_claim",
            lambda document: document["launch_manifest"].__setitem__(
                "selected_gpu_uuids", ["GPU-synthetic-other"]
            ),
            "selected GPU UUIDs disagree with launch manifest",
        ),
        (
            "runtime_source",
            lambda document: document["source"].__setitem__("head", "0" * 40),
            "training runtime source is invalid",
        ),
        (
            "runtime_observed_argv",
            lambda document: document["observed_training_argv"].append(
                "unexpected=true"
            ),
            "runtime observed training argv disagrees",
        ),
        (
            "runtime_python_environment",
            lambda document: document["python_environment"].__setitem__(
                "PYTHONINSPECT", "1"
            ),
            "runtime Python environment fields",
        ),
        (
            "summary_digest_claim",
            lambda document: document["launch_manifest"].__setitem__(
                "sha256", "0" * 64
            ),
            "summary launch-manifest evidence digest disagrees",
        ),
        (
            "summary_source",
            lambda document: document["source"].pop("upstream"),
            "training summary source fields",
        ),
        (
            "summary_checkpoint_path",
            lambda document: document["final_checkpoint"].__setitem__(
                "path", "/wrong/checkpoint.ckpt"
            ),
            "summary checkpoint artifact path disagrees",
        ),
        (
            "summary_sparse_live_model_match",
            lambda document: document["final_checkpoint"]["semantic_audit"].__setitem__(
                "live_model_match", {"exact_tensor_values": True}
            ),
            "live model checkpoint match fields",
        ),
        (
            "summary_false_live_model_key_set",
            lambda document: document["final_checkpoint"]["semantic_audit"][
                "live_model_match"
            ].__setitem__("exact_key_set", False),
            "live model checkpoint exact key set must be true",
        ),
        (
            "summary_zero_live_ema_count",
            lambda document: document["final_checkpoint"]["semantic_audit"][
                "live_ema_match"
            ].__setitem__("tensor_count", 0),
            "live EMA checkpoint tensor count must be at least 1",
        ),
        (
            "summary_warm_start_expected_digest",
            lambda document: document["startup"][
                "verified_mdlm_warm_start_report"
            ].__setitem__("expected_source_sha256", "0" * 64),
            "warm-start expected source digest disagrees",
        ),
        (
            "summary_warm_start_byte_identity",
            lambda document: document["startup"][
                "verified_mdlm_warm_start_report"
            ].__setitem__("byte_identity_verified_before_and_after_load", False),
            "warm-start byte identity must be true",
        ),
        (
            "summary_warm_start_conditioning_variant",
            lambda document: document["startup"][
                "verified_mdlm_warm_start_report"
            ].__setitem__("conditioning_variant", "film_adaln"),
            "warm-start report fields",
        ),
        (
            "summary_warm_start_conditioning_tensor_count",
            lambda document: document["startup"][
                "verified_mdlm_warm_start_report"
            ].__setitem__("conditioning_parameter_tensors", 28),
            "warm-start report fields",
        ),
        (
            "summary_warm_start_extra_field",
            lambda document: document["startup"][
                "verified_mdlm_warm_start_report"
            ].__setitem__("unexpected", True),
            "warm-start report fields",
        ),
        (
            "summary_live_serialized_finiteness",
            lambda document: document["tensor_finiteness"]["raw_model"].__setitem__(
                "floating_element_count", 999_999
            ),
            "live and serialized raw_model finiteness evidence disagree",
        ),
        (
            "summary_impossible_finiteness_counts",
            lambda document: document["tensor_finiteness"]["raw_model"].update(
                {"floating_tensor_count": 10, "floating_element_count": 9}
            ),
            "fewer elements than tensors",
        ),
        (
            "summary_impossible_gradient_counts",
            lambda document: document["training_health"].update(
                {
                    "gradient_tensor_observations": 499,
                    "gradient_element_observations": 499,
                }
            ),
            "gradient observation counts are inconsistent",
        ),
        (
            "receipt_expected_gpu_claim",
            lambda document: document["expected_contract"].__setitem__(
                "selected_gpu_uuids", ["GPU-synthetic-other"]
            ),
            "exit receipt selected_gpu_uuids disagrees",
        ),
        (
            "receipt_predecessor_binding",
            lambda document: document["predecessor_receipt_binding"].__setitem__(
                "expected_predecessor_training_variant", "udlm_categorical"
            ),
            "does not mirror the launch predecessor binding",
        ),
        (
            "receipt_observed_gpu_claim",
            lambda document: document["launch_manifest"].__setitem__(
                "observed_selected_gpu_uuids", ["GPU-synthetic-other"]
            ),
            "observed_selected_gpu_uuids disagrees with manifest",
        ),
        (
            "receipt_lock_claim",
            lambda document: document["training_job_lock"].__setitem__(
                "matches_launch_manifest_binding", False
            ),
            "receipt training-job lock matches_launch_manifest_binding must be true",
        ),
        (
            "receipt_process_status_false",
            lambda document: document.__setitem__("process_exit_status", False),
            "process status must be an integer",
        ),
        (
            "receipt_pipeline_nonzero",
            lambda document: document["pipeline"]["training"].__setitem__(
                "shell_exit_status", 1
            ),
            "training component is not successful",
        ),
        (
            "receipt_pipeline_missing",
            lambda document: document["pipeline"].pop("tee"),
            "pipeline fields",
        ),
        (
            "receipt_missing_pipeline",
            lambda document: document.pop("pipeline"),
            "training exit receipt fields",
        ),
        (
            "receipt_source_revision",
            lambda document: document["source_at_receipt"].__setitem__(
                "upstream", "0" * 40
            ),
            "receipt source evidence is invalid",
        ),
        (
            "receipt_checkpoint_sparse_artifact",
            lambda document: document["final_checkpoint"]["artifact"].pop("device"),
            "receipt checkpoint artifact fields",
        ),
        (
            "receipt_checkpoint_snapshot_disagreement",
            lambda document: document["final_checkpoint"]["artifact"].__setitem__(
                "inode", document["final_checkpoint"]["artifact"]["inode"] + 1
            ),
            "receipt and summary checkpoint snapshots disagree",
        ),
    ],
)
def test_launch_manifest_semantic_and_cross_artifact_tampering_fails_closed(
    tmp_path,
    monkeypatch,
    protocol,
    mutation_layer,
    mutator,
    error_match,
):
    repository = tmp_path / mutation_layer
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", repository)
    candidate_lock = _candidate_lock()
    keyword = mutation_layer.split("_", maxsplit=1)[0]
    _write_valid_training_evidence(
        repository,
        candidate_lock,
        **{f"mutate_{keyword}": mutator},
    )
    normalized = gate.validate_candidate_lock(candidate_lock, protocol)

    with pytest.raises(gate.GateValidationError, match=error_match):
        gate.validate_training_evidence(normalized)


def test_gate_rejects_predecessor_artifact_mutated_after_successor_launch(
    tmp_path, monkeypatch, protocol
):
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", tmp_path)
    candidate_lock = _candidate_lock()
    documents = _write_valid_training_evidence(tmp_path, candidate_lock)
    normalized = gate.validate_candidate_lock(candidate_lock, protocol)
    predecessor_path = Path(
        documents["manifest"]["predecessor_receipt_binding"]["receipt_artifact"]["path"]
    )
    predecessor_path.write_bytes(predecessor_path.read_bytes() + b" ")

    with pytest.raises(gate.GateValidationError, match="no longer matches"):
        gate.validate_training_evidence(normalized)


def test_gate_rejects_candidate_artifacts_nested_below_launch_run_directory(
    tmp_path, monkeypatch, protocol
):
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", tmp_path)
    candidate_lock = _candidate_lock()
    training = candidate_lock["training"]
    nested = "output/udlm/synthetic-candidate/nested"
    for field, basename in (
        ("launch_manifest", "launch_manifest.json"),
        ("runtime_config", "runtime_config.json"),
        ("training_summary", "training_summary.json"),
        ("exit_receipt", "pilot_exit_status.json"),
    ):
        training[field]["relative_path"] = f"{nested}/{basename}"
    training["checkpoint"]["relative_path"] = f"{nested}/checkpoints/500.ckpt"
    _write_valid_training_evidence(tmp_path, candidate_lock)
    normalized = gate.validate_candidate_lock(candidate_lock, protocol)

    with pytest.raises(
        gate.GateValidationError, match="path does not match launch run name"
    ):
        gate.validate_training_evidence(normalized)


@pytest.mark.parametrize(
    ("mutation_keyword", "mutator", "error_match"),
    [
        (
            "receipt",
            lambda document: document.__setitem__("process_exit_status", False),
            "bound predecessor process exit status must be an integer",
        ),
        (
            "receipt",
            lambda document: document.__setitem__("process_exit_status", 1),
            "bound predecessor exit receipt is not successful",
        ),
        (
            "receipt",
            lambda document: document["pipeline"].pop("tee"),
            "bound predecessor pipeline fields",
        ),
        (
            "receipt",
            lambda document: document.pop("pipeline"),
            "bound predecessor exit receipt fields",
        ),
        (
            "receipt",
            lambda document: document["source_at_receipt"].pop("upstream"),
            "bound predecessor source evidence fields",
        ),
        (
            "manifest",
            lambda document: document.pop("tmux_session"),
            "bound predecessor launch manifest fields",
        ),
        (
            "summary",
            lambda document: document.pop("training_health"),
            "bound predecessor training summary fields",
        ),
        (
            "receipt",
            lambda document: document.pop("expected_contract"),
            "bound predecessor exit receipt fields",
        ),
    ],
)
def test_gate_rejects_nonproducer_predecessor_artifacts(
    tmp_path, monkeypatch, protocol, mutation_keyword, mutator, error_match
):
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", tmp_path)
    candidate_lock = _candidate_lock()
    _write_valid_training_evidence(
        tmp_path,
        candidate_lock,
        **{f"mutate_predecessor_{mutation_keyword}": mutator},
    )
    normalized = gate.validate_candidate_lock(candidate_lock, protocol)

    with pytest.raises(gate.GateValidationError, match=error_match):
        gate.validate_training_evidence(normalized)


@pytest.mark.parametrize(
    ("summary_time", "receipt_time"),
    [
        ("2026-09-06T00:00:00+00:00", "2026-09-06T00:10:01+00:00"),
        ("2026-09-06T00:10:02+00:00", "2026-09-06T00:10:01+00:00"),
        ("2026-09-06T00:10:00+00:00", "2026-09-06T00:09:59+00:00"),
    ],
)
def test_gate_requires_strict_launch_summary_receipt_chronology(
    tmp_path, monkeypatch, protocol, summary_time, receipt_time
):
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", tmp_path)
    candidate_lock = _candidate_lock()
    _write_valid_training_evidence(
        tmp_path,
        candidate_lock,
        mutate_summary=lambda document: document.__setitem__(
            "completed_at_utc", summary_time
        ),
        mutate_receipt=lambda document: document.__setitem__(
            "recorded_at_utc", receipt_time
        ),
    )
    normalized = gate.validate_candidate_lock(candidate_lock, protocol)

    with pytest.raises(gate.GateValidationError, match="chronology is not strict"):
        gate.validate_training_evidence(normalized)


def test_gate_rejects_predecessor_artifact_relocated_outside_bound_run_directory(
    tmp_path, monkeypatch, protocol
):
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", tmp_path)
    candidate_lock = _candidate_lock()

    def relocate_predecessor_receipt(manifest):
        binding = manifest["predecessor_receipt_binding"]
        original_path = Path(binding["receipt_artifact"]["path"])
        relocated_path = tmp_path / "output/udlm/relocated/pilot_exit_status.json"
        relocated_path.parent.mkdir(parents=True)
        relocated_path.write_bytes(original_path.read_bytes())
        binding["receipt_artifact"] = _stable_snapshot(relocated_path)

    _write_valid_training_evidence(
        tmp_path,
        candidate_lock,
        mutate_manifest=relocate_predecessor_receipt,
    )
    normalized = gate.validate_candidate_lock(candidate_lock, protocol)

    with pytest.raises(
        gate.GateValidationError,
        match=("path must equal output/udlm/r-predecessor/pilot_exit_status.json"),
    ):
        gate.validate_training_evidence(normalized)


@pytest.mark.parametrize(
    "run_name",
    ["../r-predecessor", "r/predecessor", ".r-predecessor", "r" * 81],
)
def test_gate_rejects_non_launcher_predecessor_run_names(
    tmp_path, monkeypatch, protocol, run_name
):
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", tmp_path)
    candidate_lock = _candidate_lock()

    def mutate_run_name(manifest):
        manifest["predecessor_receipt_binding"]["predecessor_run_name"] = run_name

    _write_valid_training_evidence(
        tmp_path,
        candidate_lock,
        mutate_manifest=mutate_run_name,
    )
    normalized = gate.validate_candidate_lock(candidate_lock, protocol)

    with pytest.raises(gate.GateValidationError, match="launcher-normalized syntax"):
        gate.validate_training_evidence(normalized)


def test_gate_rejects_symlinked_predecessor_artifact(tmp_path, monkeypatch, protocol):
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", tmp_path)
    candidate_lock = _candidate_lock()

    def symlink_predecessor_receipt(manifest):
        binding = manifest["predecessor_receipt_binding"]
        receipt_path = Path(binding["receipt_artifact"]["path"])
        target_path = receipt_path.with_name("pilot_exit_status.target.json")
        target_path.write_bytes(receipt_path.read_bytes())
        receipt_path.unlink()
        receipt_path.symlink_to(target_path.name)
        binding["receipt_artifact"] = _stable_snapshot(receipt_path)

    _write_valid_training_evidence(
        tmp_path,
        candidate_lock,
        mutate_manifest=symlink_predecessor_receipt,
    )
    normalized = gate.validate_candidate_lock(candidate_lock, protocol)

    with pytest.raises(gate.GateValidationError, match="not a regular file"):
        gate.validate_training_evidence(normalized)


def test_launch_manifest_raw_digest_must_match_candidate_lock(
    tmp_path, monkeypatch, protocol
):
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", tmp_path)
    candidate_lock = _candidate_lock()
    _write_valid_training_evidence(tmp_path, candidate_lock)
    normalized = gate.validate_candidate_lock(candidate_lock, protocol)
    normalized["launch_manifest"]["sha256"] = "0" * 64

    with pytest.raises(
        gate.GateValidationError,
        match="launch manifest digest disagrees with candidate lock",
    ):
        gate.validate_training_evidence(normalized)


def test_git_firewall_requires_preexisting_exact_lock_ledger_and_config(
    tmp_path, monkeypatch, protocol
):
    original_root = gate.REPOSITORY_ROOT
    original_subprocess_run = subprocess.run
    protocol_bytes = (original_root / gate.PROTOCOL_RELATIVE_PATH).read_bytes()
    baseline_bytes = (original_root / gate.BASELINE_RELATIVE_PATH).read_bytes()
    baseline_rescore_bytes = (
        original_root / gate.BASELINE_RESCORE_RELATIVE_PATH
    ).read_bytes()
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Gate Test"],
        check=True,
    )
    sampler_path = tmp_path / "src/genmol/sampler.py"
    runner_path = tmp_path / "scripts/exps/denovo/benchmark.py"
    launcher_path = tmp_path / gate.DENOVO_LAUNCHER_RELATIVE_PATH
    report_path = tmp_path / "scripts/exps/denovo/report.py"
    gate_path = tmp_path / "scripts/udlm/superiority_gate.py"
    rescore_path = tmp_path / gate.DENOVO_RESCORE_RELATIVE_PATH
    rescore_dependency_path = tmp_path / gate.DENOVO_RESCORE_DEPENDENCY_RELATIVE_PATH
    pilot_writer_path = tmp_path / gate.PILOT_EVIDENCE_WRITER_RELATIVE_PATH
    config_path = tmp_path / "scripts/exps/denovo/hparams_udlm_schedule_uniform.yaml"
    sampler_path.parent.mkdir(parents=True)
    config_path.parent.mkdir(parents=True)
    protocol_path = tmp_path / gate.PROTOCOL_RELATIVE_PATH
    baseline_path = tmp_path / gate.BASELINE_RELATIVE_PATH
    baseline_rescore_path = tmp_path / gate.BASELINE_RESCORE_RELATIVE_PATH
    protocol_path.parent.mkdir(parents=True)
    baseline_path.parent.mkdir(parents=True)
    protocol_path.write_bytes(protocol_bytes)
    baseline_path.write_bytes(baseline_bytes)
    baseline_rescore_path.write_bytes(baseline_rescore_bytes)
    sampler_bytes = b"# synthetic audited EMA sampler\n"
    runner_bytes = b"# synthetic benchmark runner\n"
    launcher_bytes = b"# synthetic benchmark launcher\n"
    report_bytes = b"# synthetic report implementation\n"
    gate_bytes = b"# synthetic superiority gate\n"
    rescore_bytes = b"# synthetic independent rescore\n"
    rescore_dependency_bytes = b"# synthetic rescore dependency\n"
    pilot_writer_bytes = b"# synthetic pilot evidence writer\n"
    config_bytes = (
        b"diffusion_type: udlm\n"
        b"softmax_temp: 1.0\n"
        b"randomness: 0.5\n"
        b"min_add_len: 40\n"
        b"num_steps: 128\n"
        b"inference_eps: 0.00001\n"
        b"exclude_special_tokens: true\n"
        b"prior_variant: release_uniform\n"
        b"prior_metadata_sha256: null\n"
    )
    sampler_path.write_bytes(sampler_bytes)
    runner_path.write_bytes(runner_bytes)
    launcher_path.write_bytes(launcher_bytes)
    report_path.write_bytes(report_bytes)
    gate_path.parent.mkdir(parents=True)
    gate_path.write_bytes(gate_bytes)
    rescore_path.write_bytes(rescore_bytes)
    rescore_dependency_path.write_bytes(rescore_dependency_bytes)
    pilot_writer_path.write_bytes(pilot_writer_bytes)
    config_path.write_bytes(config_bytes)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-qm", "training source"],
        check=True,
    )
    training_revision = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    pilot_attempt, pilot_blobs = _completed_pilot_attempt(
        attempt_id="a1",
        candidate_id="schedule-uniform-synthetic",
        seeds=(1000, 1001),
        qualities=(0.85, 0.85),
        diversities=(0.74, 0.74),
    )
    failed_attempt, failed_blobs = _failed_pilot_attempt(
        attempt_id="failed-a1",
        candidate_id="schedule-failed",
        seed=1004,
        source_revision=training_revision,
    )
    pilot_blobs |= failed_blobs
    for relative_path, artifact_bytes in pilot_blobs.items():
        pilot_artifact_path = tmp_path / relative_path
        pilot_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        pilot_artifact_path.write_bytes(artifact_bytes)
    ledger = _pilot_ledger(
        attempts=[failed_attempt, pilot_attempt], selected_attempt_id="a1"
    )
    ledger_bytes = _json_bytes(ledger)
    ledger_path = tmp_path / "experiments/udlm/candidates/ledger.json"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_bytes(ledger_bytes)
    candidate_lock = _candidate_lock()
    candidate_lock["training"]["exit_receipt"]["sha256"] = _PILOT_RUN_RESULTS[
        ("a1", gate.REGISTERED_SELECTION_PILOT_SEEDS[0])
    ]["training_exit_receipt"]["sha256"]
    candidate_lock["training"]["source_revision"] = training_revision
    candidate_lock["selection"]["candidate_ledger"]["sha256"] = hashlib.sha256(
        ledger_bytes
    ).hexdigest()
    candidate_lock["inference"]["evaluation_config_sha256"] = hashlib.sha256(
        config_bytes
    ).hexdigest()
    candidate_lock["inference"]["sampler_source_sha256"] = hashlib.sha256(
        sampler_bytes
    ).hexdigest()
    candidate_lock["inference"]["benchmark_runner_sha256"] = hashlib.sha256(
        runner_bytes
    ).hexdigest()
    candidate_lock["analysis"]["gate_source_sha256"] = hashlib.sha256(
        gate_bytes
    ).hexdigest()
    candidate_lock["analysis"]["report_source_sha256"] = hashlib.sha256(
        report_bytes
    ).hexdigest()
    candidate_lock["analysis"]["rescore_source_sha256"] = hashlib.sha256(
        rescore_bytes
    ).hexdigest()
    candidate_lock["analysis"]["rescore_dependency_sha256"] = hashlib.sha256(
        rescore_dependency_bytes
    ).hexdigest()
    candidate_lock["analysis"]["benchmark_launcher_source_sha256"] = hashlib.sha256(
        launcher_bytes
    ).hexdigest()
    candidate_lock["analysis"]["pilot_evidence_writer_source_sha256"] = hashlib.sha256(
        pilot_writer_bytes
    ).hexdigest()
    for seed in gate.REGISTERED_SELECTION_PILOT_SEEDS:
        pilot_result = _PILOT_RUN_RESULTS[("a1", seed)]
        pilot_result["benchmark_revision"] = training_revision
        pilot_result["evaluation_config"] = {
            "relative_path": candidate_lock["inference"][
                "evaluation_config_relative_path"
            ],
            "sha256": candidate_lock["inference"]["evaluation_config_sha256"],
        }
        pilot_result["runner_sha256"] = candidate_lock["inference"][
            "benchmark_runner_sha256"
        ]
        pilot_result["sampler_source_sha256"] = candidate_lock["inference"][
            "sampler_source_sha256"
        ]
    lock_path = tmp_path / "experiments/udlm/candidates/lock.json"
    lock_bytes = (json.dumps(candidate_lock, sort_keys=True) + "\n").encode()
    lock_path.write_bytes(lock_bytes)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-qm", "lock candidate"],
        check=True,
    )
    benchmark_revision = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    normalized = gate.validate_candidate_lock(candidate_lock, protocol)

    synthetic_git_blob = gate._git_blob

    def git_blob_with_real_rescore_source(revision, relative_path):
        if revision == gate.EXPECTED_BASELINE_RESCORE_SOURCE_REVISION:
            return original_subprocess_run(
                [
                    "git",
                    "-C",
                    str(original_root),
                    "show",
                    f"{revision}:{relative_path.as_posix()}",
                ],
                check=True,
                capture_output=True,
            ).stdout
        return synthetic_git_blob(revision, relative_path)

    monkeypatch.setattr(gate, "_git_blob", git_blob_with_real_rescore_source)

    def run_with_external_rescore_ancestry(args, *run_args, **run_kwargs):
        if (
            isinstance(args, list)
            and "merge-base" in args
            and args[-2] == gate.EXPECTED_BASELINE_RESCORE_SOURCE_REVISION
        ):
            return subprocess.CompletedProcess(args=args, returncode=0)
        return original_subprocess_run(args, *run_args, **run_kwargs)

    monkeypatch.setattr(gate.subprocess, "run", run_with_external_rescore_ancestry)

    evidence = gate.validate_git_lock_firewall(
        benchmark_revision=benchmark_revision,
        candidate_lock_path=Path("experiments/udlm/candidates/lock.json"),
        candidate_lock_bytes=lock_bytes,
        lock=normalized,
        pilot_run_validator=_pilot_run_validator,
    )

    assert evidence["candidate_lock_exact_blob_at_benchmark_revision"] is True
    assert (
        evidence["baseline_rescore_attestation_exact_blob_at_benchmark_revision"]
        is True
    )
    assert evidence["baseline_rescore_source_exact_blobs_verified"] is True
    assert evidence["baseline_rescore_source_revision_is_ancestor"] is True
    assert evidence["candidate_ledger_exact_blob_at_benchmark_revision"] is True
    assert evidence["ema_sampler_source_exact_blob_at_benchmark_revision"] is True
    assert evidence["benchmark_runner_exact_blob_at_benchmark_revision"] is True
    assert evidence["committed_pilot_artifact_count"] == 3
    assert evidence["eligible_pilot_attempt_count"] == 1
    assert evidence["ineligible_completed_pilot_attempt_count"] == 0
    assert evidence["failed_pilot_attempt_count"] == 1
    assert evidence["registered_selection_operating_point"] == {
        "generation_seeds": [1000, 1001],
        "requested_samples_per_seed": 256,
        "nfe": 128,
        "metric_branch": "released_comparable",
    }
    assert evidence["selected_attempt_id"] == "a1"
    assert evidence["selected_checkpoint_sha256"] == "4" * 64
    assert evidence["selection_recomputed_from_committed_pilot_evidence"] is True

    wrong_checkpoint_lock = dict(normalized)
    wrong_checkpoint_lock["checkpoint"] = {
        **normalized["checkpoint"],
        "sha256": "f" * 64,
    }
    with pytest.raises(
        gate.GateValidationError,
        match="not the pilot-ledger winner checkpoint",
    ):
        gate.validate_git_lock_firewall(
            benchmark_revision=benchmark_revision,
            candidate_lock_path=Path("experiments/udlm/candidates/lock.json"),
            candidate_lock_bytes=lock_bytes,
            lock=wrong_checkpoint_lock,
            pilot_run_validator=_pilot_run_validator,
        )

    baseline_rescore_path.write_bytes(baseline_rescore_bytes + b" ")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", baseline_rescore_path.as_posix()],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-qm", "tamper rescore evidence"],
        check=True,
    )
    tampered_revision = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    with pytest.raises(
        gate.GateValidationError, match="rescore-attestation blob differs"
    ):
        gate.validate_git_lock_firewall(
            benchmark_revision=tampered_revision,
            candidate_lock_path=Path("experiments/udlm/candidates/lock.json"),
            candidate_lock_bytes=lock_bytes,
            lock=normalized,
        )

    with pytest.raises(gate.GateValidationError, match="candidate-lock blob differs"):
        gate.validate_git_lock_firewall(
            benchmark_revision=benchmark_revision,
            candidate_lock_path=Path("experiments/udlm/candidates/lock.json"),
            candidate_lock_bytes=lock_bytes + b" ",
            lock=normalized,
        )
    wrong_runner_lock = dict(normalized)
    wrong_runner_lock["benchmark_runner_sha256"] = "0" * 64
    with pytest.raises(
        gate.GateValidationError,
        match="runner_sha256 is not the pilot-ledger winner identity",
    ):
        gate.validate_git_lock_firewall(
            benchmark_revision=benchmark_revision,
            candidate_lock_path=Path("experiments/udlm/candidates/lock.json"),
            candidate_lock_bytes=lock_bytes,
            lock=wrong_runner_lock,
            pilot_run_validator=_pilot_run_validator,
        )


def test_no_clobber_decision_writer(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", tmp_path)
    output = tmp_path / "output" / "decision.json"
    output.parent.mkdir()

    gate._atomic_write_json_exclusive(output, {"status": "first"})
    original = output.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to replace"):
        gate._atomic_write_json_exclusive(output, {"status": "second"})

    assert output.read_bytes() == original


def test_decision_writer_does_not_create_through_symlinked_ancestor(
    tmp_path, monkeypatch
):
    repository = tmp_path / "repository"
    outside = tmp_path / "outside"
    repository.mkdir()
    outside.mkdir()
    (repository / "output").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", repository)

    with pytest.raises(gate.GateValidationError, match="output parent"):
        gate._atomic_write_json_exclusive(
            repository / "output/new/decision.json", {"status": "forbidden"}
        )

    assert not (outside / "new").exists()

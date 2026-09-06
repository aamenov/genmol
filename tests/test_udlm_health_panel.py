from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.udlm import launch_train_pilot
from scripts.udlm import validate_health_panel as health
from scripts.udlm import write_pilot_evidence as pilot_evidence_writer


SOURCE_REVISION = "a" * 40


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _snapshot(path: Path) -> dict[str, object]:
    return pilot_evidence_writer.read_stable_regular_file(
        path, label=path.name, capture_payload=False
    ).snapshot()


def _resolved_config(
    *, run_dir: Path, variant: str, gpu_count: int
) -> dict[str, object]:
    return {
        "seed": health.HEALTH_PANEL_SEED,
        "training": {
            "ema": 0.9999,
            "diffusion": "udlm",
            "T": 0,
            "udlm": {
                "prior_variant": launch_train_pilot.TRAINING_VARIANTS[variant][
                    "prior_variant"
                ],
                "exclude_special_tokens": False,
                "empirical_uniform_mix": (
                    launch_train_pilot.PILOT_EMPIRICAL_UNIFORM_MIX
                ),
                "conditioning_variant": "additive",
                "zero_init_conditioning": True,
            },
        },
        "model": {
            "vocab_size": 1880,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
        },
        "optim": {
            "lr": 3e-4,
            "scheduler": {
                "name": "constant_with_linear_warmup",
                "warmup_updates": 2500,
                "horizon_updates": None,
                "decay_floor_lr": None,
            },
        },
        "noise": {"type": "loglinear"},
        "loader": {
            "batch_size": health.HEALTH_PANEL_MICRO_BATCH_SIZE,
            "global_batch_size": health.HEALTH_PANEL_GLOBAL_BATCH_SIZE,
            "num_workers": health.HEALTH_PANEL_NUM_WORKERS,
        },
        "trainer": {
            "devices": gpu_count,
            "num_nodes": 1,
            "max_steps": health.HEALTH_PANEL_MAX_STEPS,
            "accumulate_grad_batches": 8 if gpu_count == 1 else 4,
        },
        "callback": {"dirpath": str(run_dir / "checkpoints")},
    }


def _audit_binding() -> dict[str, object]:
    return {
        "relative_path": launch_train_pilot.PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_PATH,
        "sha256": launch_train_pilot.PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_SHA256,
        "source_revision": (
            launch_train_pilot.PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_SOURCE_REVISION
        ),
        "scope": launch_train_pilot.PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_SCOPE,
    }


def _genesis_binding(panel_sha256: str, common_sha256: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "state": "explicit_genesis_no_predecessor",
        "current_training_variant": "udlm",
        "current_variant_position": 0,
        "expected_predecessor_training_variant": None,
        "expected_predecessor_variant_position": None,
        "matched_panel_spec_sha256": panel_sha256,
        "common_training_contract_sha256": common_sha256,
        "receipt_artifact": None,
        "predecessor_launch_manifest_artifact": None,
        "predecessor_training_summary_artifact": None,
        "predecessor_run_name": None,
        "chronology": None,
        "validated_before_gpu_probe": True,
    }


def _successor_binding(
    *,
    position: int,
    predecessor: dict[str, object],
    panel_sha256: str,
    common_sha256: str,
) -> dict[str, object]:
    variant = launch_train_pilot.MATCHED_PANEL_VARIANT_ORDER[position]
    predecessor_variant = launch_train_pilot.MATCHED_PANEL_VARIANT_ORDER[position - 1]
    predecessor_receipt = predecessor["receipt"]
    predecessor_manifest = predecessor["manifest"]
    predecessor_summary = predecessor["summary"]
    return {
        "schema_version": 1,
        "state": "validated_successful_predecessor",
        "current_training_variant": variant,
        "current_variant_position": position,
        "expected_predecessor_training_variant": predecessor_variant,
        "expected_predecessor_variant_position": position - 1,
        "matched_panel_spec_sha256": panel_sha256,
        "common_training_contract_sha256": common_sha256,
        "receipt_artifact": _snapshot(predecessor["receipt_path"]),
        "predecessor_launch_manifest_artifact": _snapshot(predecessor["manifest_path"]),
        "predecessor_training_summary_artifact": _snapshot(predecessor["summary_path"]),
        "predecessor_run_name": predecessor_manifest["run_name"],
        "chronology": {
            "predecessor_launch_manifest_created_at_utc": predecessor_manifest[
                "created_at"
            ],
            "predecessor_training_summary_completed_at_utc": predecessor_summary[
                "completed_at_utc"
            ],
            "predecessor_exit_receipt_recorded_at_utc": predecessor_receipt[
                "recorded_at_utc"
            ],
            "strictly_ordered_timestamps_verified": True,
        },
        "validated_before_gpu_probe": True,
    }


def _write_member(
    *,
    root: Path,
    position: int,
    panel: dict[str, object],
    panel_sha256: str,
    gpu_count: int,
    predecessor: dict[str, object] | None,
) -> dict[str, object]:
    variant = launch_train_pilot.MATCHED_PANEL_VARIANT_ORDER[position]
    treatment = launch_train_pilot.TRAINING_VARIANTS[variant]
    run_name = health.health_run_name(gpu_count, variant, SOURCE_REVISION)
    run_dir = root / "output" / "udlm" / run_name
    manifest_path = run_dir / "launch_manifest.json"
    summary_path = run_dir / "training_summary.json"
    receipt_path = run_dir / "pilot_exit_status.json"
    checkpoint_path = run_dir / "checkpoints" / "10.ckpt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(f"checkpoint-{variant}".encode())
    checkpoint_snapshot = _snapshot(checkpoint_path)

    common_sha256 = launch_train_pilot.canonical_json_sha256(
        panel["common_training_contract"]
    )
    if predecessor is None:
        binding = _genesis_binding(panel_sha256, common_sha256)
    else:
        binding = _successor_binding(
            position=position,
            predecessor=predecessor,
            panel_sha256=panel_sha256,
            common_sha256=common_sha256,
        )
    training_argv = launch_train_pilot.build_training_command(
        gpu_count=gpu_count,
        run_dir=run_dir,
        max_steps=health.HEALTH_PANEL_MAX_STEPS,
        global_batch_size=health.HEALTH_PANEL_GLOBAL_BATCH_SIZE,
        micro_batch_size=health.HEALTH_PANEL_MICRO_BATCH_SIZE,
        num_workers=health.HEALTH_PANEL_NUM_WORKERS,
        seed=health.HEALTH_PANEL_SEED,
        checkpoint=health.EXPECTED_MDLM_CHECKPOINT_PATH,
        checkpoint_sha256=health.EXPECTED_MDLM_CHECKPOINT_SHA256,
        exclude_special_tokens=False,
        training_variant=variant,
    )
    resolved, resolved_sha256 = launch_train_pilot.compose_resolved_training_config(
        config_name=str(treatment["config_name"]),
        overrides=training_argv[5:],
        gpu_count=gpu_count,
    )
    gpu_states = [
        {
            "physical_index": 4 + gpu_index,
            "uuid": f"GPU-health-{position}-{gpu_index}",
            "name": "Synthetic Accelerator",
            "memory_used_mib": 1_000,
            "memory_total_mib": 81_920,
            "utilization_percent": 9,
            "compute_mode": "Default",
            "compute_processes": [
                {
                    "pid": 4_000 + position * 10 + gpu_index,
                    "process_name": "pre-existing-workload",
                    "used_memory_mib": 512,
                }
            ],
        }
        for gpu_index in range(gpu_count)
    ]
    manifest = {
        "created_at": f"2026-09-06T00:{position * 10:02d}:03+00:00",
        "git_sha": SOURCE_REVISION,
        "source_revision_before_final_gpu_probe": SOURCE_REVISION,
        "run_name": run_name,
        "training_variant": variant,
        "hydra_config_name": treatment["config_name"],
        "udlm_prior_variant": treatment["prior_variant"],
        "udlm_comparison_role": treatment["comparison_role"],
        "matched_panel_spec": copy.deepcopy(panel),
        "matched_panel_spec_sha256": panel_sha256,
        "matched_panel_variant_position": position,
        "predecessor_receipt_binding": copy.deepcopy(binding),
        "user_requested_gpu_count": gpu_count,
        "gpu_selection_schema_version": 2,
        "gpu_selection_method": "dynamic_idle_discovery",
        "gpu_inventory_scope": "all_nvidia_gpus",
        "inventory_snapshot_completed_at_utc": (
            f"2026-09-06T00:{position * 10:02d}:01+00:00"
        ),
        "gpu_inventory_at_selection": copy.deepcopy(gpu_states),
        "initially_selected_gpu_states": copy.deepcopy(gpu_states),
        "logical_cuda_devices": list(range(gpu_count)),
        "physical_gpu_indices": [state["physical_index"] for state in gpu_states],
        "cuda_visible_device_uuids": [state["uuid"] for state in gpu_states],
        "final_uuid_probes_completed_at_utc": (
            f"2026-09-06T00:{position * 10:02d}:02+00:00"
        ),
        "gpu_states_at_final_uuid_probe": copy.deepcopy(gpu_states),
        "gpu_safety_policy": {
            "max_utilization_percent": 10,
            "utilization_comparison": "strictly_less_than",
            "min_free_memory_mib": 30_000,
            "active_compute_processes_allowed": True,
            "compute_mode_prohibited_allowed": False,
        },
        "training_argv": training_argv,
        "training_argv_sha256": launch_train_pilot.training_argv_sha256(training_argv),
        "resolved_training_config": resolved,
        "resolved_training_config_sha256": resolved_sha256,
        "checkpoint": str(health.EXPECTED_MDLM_CHECKPOINT_PATH),
        "checkpoint_sha256": health.EXPECTED_MDLM_CHECKPOINT_SHA256,
        "seed": 1,
        "max_steps": 10,
        "global_batch_size": 16,
        "micro_batch_size_per_process": 2,
        "accumulate_grad_batches": 8 if gpu_count == 1 else 4,
        "effective_global_batch_size": 16,
        "exclude_special_tokens": False,
        "dry_run": False,
        "launch_manifest_path": str(manifest_path),
        "training_summary_path": str(summary_path),
        "pilot_exit_status_path": str(receipt_path),
        "expected_final_checkpoint_path": str(checkpoint_path),
    }
    summary = {
        "completed_at_utc": f"2026-09-06T00:{position * 10 + 5:02d}:00+00:00",
        "startup": {
            "mode": "warm_start",
            "verified_mdlm_warm_start_report": {
                "source_path": str(health.EXPECTED_MDLM_CHECKPOINT_PATH),
                "source_resolved_path": str(health.EXPECTED_MDLM_CHECKPOINT_PATH),
                "source_sha256": health.EXPECTED_MDLM_CHECKPOINT_SHA256,
                "expected_source_sha256": health.EXPECTED_MDLM_CHECKPOINT_SHA256,
                "weights": "ema",
                "byte_identity_verified_before_and_after_load": True,
                "source_size_bytes": (health.EXPECTED_MDLM_CHECKPOINT_SIZE_BYTES),
                "parameter_tensors": 202,
            },
        },
    }
    _write_json(manifest_path, manifest)
    _write_json(summary_path, summary)
    receipt = {
        "schema_version": 5,
        "status": "completed",
        "overall_status": "completed",
        "process_exit_status": 0,
        "recorded_at_utc": f"2026-09-06T00:{position * 10 + 6:02d}:00+00:00",
        "expected_contract": {
            "source_revision": SOURCE_REVISION,
            "max_steps": 10,
            "world_size": gpu_count,
            "launch_manifest_path": str(manifest_path),
            "training_summary_path": str(summary_path),
            "final_checkpoint_path": str(checkpoint_path),
            "initialization_checkpoint_sha256": (
                health.EXPECTED_MDLM_CHECKPOINT_SHA256
            ),
        },
        "predecessor_receipt_binding": copy.deepcopy(binding),
        "final_checkpoint": {
            "path": str(checkpoint_path),
            "present": True,
            "matches_training_summary_snapshot": True,
            "artifact": checkpoint_snapshot,
        },
    }
    _write_json(receipt_path, receipt)
    return {
        "variant": variant,
        "run_dir": run_dir,
        "manifest_path": manifest_path,
        "summary_path": summary_path,
        "receipt_path": receipt_path,
        "checkpoint_path": checkpoint_path,
        "manifest": manifest,
        "summary": summary,
        "receipt": receipt,
    }


@pytest.fixture
def health_panel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> dict[str, object]:
    monkeypatch.setattr(health, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(pilot_evidence_writer, "REPOSITORY_ROOT", tmp_path)
    checkpoint_path = tmp_path.parent / "outputs/paper_v1/checkpoints/50000.ckpt"
    monkeypatch.setattr(health, "EXPECTED_MDLM_CHECKPOINT_PATH", checkpoint_path)
    monkeypatch.setattr(
        launch_train_pilot,
        "verify_pilot_empirical_uniform_mix_audit",
        _audit_binding,
    )
    reconstructed_configs: dict[tuple[str, ...], dict[str, object]] = {}

    def fake_build_training_command(**kwargs) -> list[str]:
        variant = kwargs["training_variant"]
        run_dir = kwargs["run_dir"]
        gpu_count = kwargs["gpu_count"]
        assert kwargs == {
            "gpu_count": gpu_count,
            "run_dir": run_dir,
            "max_steps": health.HEALTH_PANEL_MAX_STEPS,
            "global_batch_size": health.HEALTH_PANEL_GLOBAL_BATCH_SIZE,
            "micro_batch_size": health.HEALTH_PANEL_MICRO_BATCH_SIZE,
            "num_workers": health.HEALTH_PANEL_NUM_WORKERS,
            "seed": health.HEALTH_PANEL_SEED,
            "checkpoint": health.EXPECTED_MDLM_CHECKPOINT_PATH,
            "checkpoint_sha256": health.EXPECTED_MDLM_CHECKPOINT_SHA256,
            "exclude_special_tokens": False,
            "training_variant": variant,
        }
        command = [
            "/fixture/.venv/bin/python",
            "-u",
            str(launch_train_pilot.REPOSITORY_ROOT / "scripts/train.py"),
            "--config-name",
            str(launch_train_pilot.TRAINING_VARIANTS[variant]["config_name"]),
            f"fixture.training_variant={variant}",
            f"fixture.run_dir={run_dir}",
            f"fixture.gpu_count={gpu_count}",
        ]
        reconstructed_configs[tuple(command[5:])] = _resolved_config(
            run_dir=run_dir,
            variant=variant,
            gpu_count=gpu_count,
        )
        return command

    def fake_compose_resolved_training_config(
        *, config_name: str, overrides: list[str], gpu_count: int
    ) -> tuple[dict[str, object], str]:
        config = copy.deepcopy(reconstructed_configs[tuple(overrides)])
        assert config_name in {"udlm", "udlm_categorical"}
        assert config["trainer"]["devices"] == gpu_count
        return config, launch_train_pilot.canonical_json_sha256(config)

    monkeypatch.setattr(
        launch_train_pilot,
        "build_training_command",
        fake_build_training_command,
    )
    monkeypatch.setattr(
        launch_train_pilot,
        "compose_resolved_training_config",
        fake_compose_resolved_training_config,
    )
    deep_calls: list[dict[str, object]] = []

    def fake_deep_validator(*args, **kwargs):
        deep_calls.append(kwargs)
        return None, ()

    monkeypatch.setattr(
        pilot_evidence_writer,
        "validate_successful_training_receipt",
        fake_deep_validator,
    )

    gpu_count = getattr(request, "param", 1)
    template_run = tmp_path / "output/udlm/template"
    template = _resolved_config(
        run_dir=template_run, variant="udlm", gpu_count=gpu_count
    )
    common_config_sha256 = launch_train_pilot.matched_panel_config_sha256(template)
    panel, panel_sha256 = launch_train_pilot.build_matched_panel_spec(
        source_revision=SOURCE_REVISION,
        checkpoint=checkpoint_path,
        checkpoint_sha256=health.EXPECTED_MDLM_CHECKPOINT_SHA256,
        gpu_count=gpu_count,
        max_steps=10,
        global_batch_size=16,
        micro_batch_size=2,
        num_workers=1,
        seed=1,
        exclude_special_tokens=False,
        max_utilization_percent=10,
        min_free_memory_mib=30_000,
        common_resolved_config_sha256=common_config_sha256,
    )
    members = []
    predecessor = None
    for position in range(3):
        predecessor = _write_member(
            root=tmp_path,
            position=position,
            panel=panel,
            panel_sha256=panel_sha256,
            gpu_count=gpu_count,
            predecessor=predecessor,
        )
        members.append(predecessor)
    return {
        "root": tmp_path,
        "gpu_count": gpu_count,
        "panel": panel,
        "panel_sha256": panel_sha256,
        "members": members,
        "terminal_path": members[-1]["receipt_path"],
        "deep_calls": deep_calls,
    }


def _rewrite_manifest(member: dict[str, object]) -> None:
    _write_json(member["manifest_path"], member["manifest"])


def _rewrite_summary(member: dict[str, object]) -> None:
    _write_json(member["summary_path"], member["summary"])


def _rewrite_receipt(member: dict[str, object]) -> None:
    _write_json(member["receipt_path"], member["receipt"])


def test_health_run_name_is_deterministic_and_uses_full_revision() -> None:
    assert health.health_run_name(2, "udlm", SOURCE_REVISION) == (
        f"health-w2-r-{SOURCE_REVISION}"
    )
    assert health.health_run_name(2, "schedule_uniform", SOURCE_REVISION) == (
        f"health-w2-s-{SOURCE_REVISION}"
    )
    assert health.health_run_name(2, "udlm_categorical", SOURCE_REVISION) == (
        f"health-w2-e-{SOURCE_REVISION}"
    )
    with pytest.raises(health.HealthPanelValidationError, match="gpu-count"):
        health.health_run_name(3, "udlm", SOURCE_REVISION)


def test_validate_health_panel_normalizes_exact_ordered_evidence(
    health_panel: dict[str, object],
) -> None:
    result = health.validate_health_panel(
        health_panel["terminal_path"],
        expected_gpu_count=1,
        expected_source_revision=SOURCE_REVISION,
    )

    assert result["schema_version"] == 1
    assert result["status"] == "validated"
    assert result["claim_scope"] == "training_health_and_provenance_only"
    assert result["health_source_revision"] == SOURCE_REVISION
    assert result["gpu_count"] == 1
    assert result["matched_panel_spec_sha256"] == health_panel["panel_sha256"]
    assert result["terminal_receipt"]["root"] == "repository"
    assert result["terminal_receipt"]["size_bytes"] > 0
    assert len(result["terminal_receipt"]["canonical_sha256"]) == 64
    assert [member["training_variant"] for member in result["receipt_members"]] == [
        "udlm",
        "schedule_uniform",
        "udlm_categorical",
    ]
    assert [member["position"] for member in result["checkpoint_members"]] == [
        0,
        1,
        2,
    ]
    assert result["eligibility"] == {
        "generation": False,
        "ranking": False,
        "superiority": False,
        "candidate_lock": False,
        "screen_authorization": True,
    }
    assert len(health_panel["deep_calls"]) == 1
    deep_call = health_panel["deep_calls"][0]
    assert deep_call["receipt_path"] == health_panel["terminal_path"]
    assert deep_call["structural"]["checkpoint"]["global_step"] == 10
    assert (
        deep_call["structural"]["checkpoint"]["sha256"]
        == (result["checkpoint_members"][-1]["sha256"])
    )
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert health.validate_health_panel(health_panel["terminal_path"]) == result
    for member in health_panel["members"]:
        state = member["manifest"]["gpu_states_at_final_uuid_probe"][0]
        assert state["utilization_percent"] == 9
        assert state["compute_processes"][0]["process_name"] == (
            "pre-existing-workload"
        )


@pytest.mark.parametrize("health_panel", [2], indirect=True)
def test_validate_health_panel_accepts_exact_two_gpu_contract(
    health_panel: dict[str, object],
) -> None:
    result = health.validate_health_panel(
        health_panel["terminal_path"], expected_gpu_count=2
    )
    assert result["gpu_count"] == 2
    for member in health_panel["members"]:
        assert member["manifest"]["accumulate_grad_batches"] == 4


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_steps", 11),
        ("global_batch_size", 15),
        ("micro_batch_size_per_process", 1),
        ("accumulate_grad_batches", 7),
        ("effective_global_batch_size", 15),
        ("num_workers", 0),
        ("seed", 2),
        ("exclude_special_tokens", True),
        ("empirical_uniform_mix", 0.01),
    ],
)
def test_health_panel_rejects_common_contract_mutations(
    health_panel: dict[str, object], field: str, value: object
) -> None:
    terminal = health_panel["members"][-1]
    panel = terminal["manifest"]["matched_panel_spec"]
    panel["common_training_contract"][field] = value
    terminal["manifest"]["matched_panel_spec_sha256"] = (
        launch_train_pilot.canonical_json_sha256(panel)
    )
    _rewrite_manifest(terminal)

    with pytest.raises(
        health.HealthPanelValidationError, match="exact registered ten-update"
    ):
        health.validate_health_panel(health_panel["terminal_path"])


def test_health_panel_rejects_safety_and_audit_mutations(
    health_panel: dict[str, object],
) -> None:
    terminal = health_panel["members"][-1]
    panel = terminal["manifest"]["matched_panel_spec"]
    panel["common_gpu_safety_policy"]["max_utilization_percent"] = 9
    panel["common_training_contract"]["empirical_uniform_mix_audit"]["scope"] = (
        "unregistered"
    )
    terminal["manifest"]["matched_panel_spec_sha256"] = (
        launch_train_pilot.canonical_json_sha256(panel)
    )
    _rewrite_manifest(terminal)

    with pytest.raises(
        health.HealthPanelValidationError, match="exact registered ten-update"
    ):
        health.validate_health_panel(health_panel["terminal_path"])


def test_health_threshold_contract_is_independent_of_launcher_default_drift(
    health_panel: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(launch_train_pilot, "MAX_SAFE_UTILIZATION_PERCENT", 99)
    monkeypatch.setattr(launch_train_pilot, "MIN_SAFE_FREE_MEMORY_MIB", 1)

    result = health.validate_health_panel(health_panel["terminal_path"])

    assert result["status"] == "validated"
    assert health.HEALTH_PANEL_MAX_UTILIZATION_PERCENT == 10
    assert health.HEALTH_PANEL_MIN_FREE_MEMORY_MIB == 30_000


def test_health_panel_rejects_legacy_zero_process_manifest_policy(
    health_panel: dict[str, object],
) -> None:
    terminal = health_panel["members"][-1]
    terminal["manifest"]["gpu_safety_policy"]["active_compute_processes_allowed"] = (
        False
    )
    _rewrite_manifest(terminal)

    with pytest.raises(
        health.HealthPanelValidationError, match="GPU safety policy is not exact"
    ):
        health.validate_health_panel(health_panel["terminal_path"])


def _set_health_gpu_state_fields(manifest: dict[str, object], **updates) -> None:
    for field in (
        "gpu_inventory_at_selection",
        "initially_selected_gpu_states",
        "gpu_states_at_final_uuid_probe",
    ):
        manifest[field][0].update(copy.deepcopy(updates))


@pytest.mark.parametrize(
    "updates",
    [
        {"utilization_percent": 10},
        {"memory_used_mib": 51_921},
        {"compute_mode": "Prohibited"},
    ],
)
def test_health_panel_rejects_unsafe_selected_gpu_state(
    health_panel: dict[str, object], updates: dict[str, object]
) -> None:
    terminal = health_panel["members"][-1]
    _set_health_gpu_state_fields(terminal["manifest"], **updates)
    _rewrite_manifest(terminal)

    with pytest.raises(
        health.HealthPanelValidationError,
        match="violates the health safety policy",
    ):
        health.validate_health_panel(health_panel["terminal_path"])


def test_health_panel_rejects_sparse_process_evidence(
    health_panel: dict[str, object],
) -> None:
    terminal = health_panel["members"][-1]
    _set_health_gpu_state_fields(
        terminal["manifest"],
        compute_processes=[{"pid": 4321, "process_name": "missing-memory-field"}],
    )
    _rewrite_manifest(terminal)

    with pytest.raises(
        health.HealthPanelValidationError, match="process 0 fields are not exact"
    ):
        health.validate_health_panel(health_panel["terminal_path"])


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("inventory_snapshot_completed_at_utc", None, "must be a nonempty UTC"),
        (
            "final_uuid_probes_completed_at_utc",
            "2026-09-06T00:00:00+00:00",
            "timestamps are out of order",
        ),
    ],
)
def test_health_panel_rejects_missing_or_reversed_gpu_probe_timestamps(
    health_panel: dict[str, object], field: str, value: object, error: str
) -> None:
    terminal = health_panel["members"][-1]
    terminal["manifest"][field] = value
    _rewrite_manifest(terminal)

    with pytest.raises(health.HealthPanelValidationError, match=error):
        health.validate_health_panel(health_panel["terminal_path"])


def test_health_panel_rejects_non_ema_warm_start(
    health_panel: dict[str, object],
) -> None:
    r_member = health_panel["members"][0]
    r_member["summary"]["startup"]["verified_mdlm_warm_start_report"]["weights"] = "raw"
    _rewrite_summary(r_member)

    with pytest.raises(health.HealthPanelValidationError, match="warm start weights"):
        health.validate_health_panel(health_panel["terminal_path"])


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        ("parameter_tensors", 201, "parameter_tensors"),
        ("conditioning_variant", "film_adaln", "conditioning_variant"),
        ("conditioning_parameter_tensors", 28, "conditioning_parameter_tensors"),
    ],
)
def test_health_panel_rejects_wrong_warm_start_topology(
    health_panel: dict[str, object],
    field: str,
    replacement: object,
    error: str,
) -> None:
    r_member = health_panel["members"][0]
    r_member["summary"]["startup"]["verified_mdlm_warm_start_report"][field] = (
        replacement
    )
    _rewrite_summary(r_member)

    with pytest.raises(health.HealthPanelValidationError, match=error):
        health.validate_health_panel(health_panel["terminal_path"])


def test_health_panel_rejects_extra_warm_start_field(
    health_panel: dict[str, object],
) -> None:
    r_member = health_panel["members"][0]
    r_member["summary"]["startup"]["verified_mdlm_warm_start_report"][
        "opaque_extra"
    ] = {"scientific_result": True}
    _rewrite_summary(r_member)

    with pytest.raises(
        health.HealthPanelValidationError, match="warm-start report keys are invalid"
    ):
        health.validate_health_panel(health_panel["terminal_path"])


def test_health_panel_rejects_wrong_mdlm_size_and_vocabulary(
    health_panel: dict[str, object],
) -> None:
    r_member = health_panel["members"][0]
    r_member["summary"]["startup"]["verified_mdlm_warm_start_report"][
        "source_size_bytes"
    ] -= 1
    _rewrite_summary(r_member)
    with pytest.raises(health.HealthPanelValidationError, match="source_size_bytes"):
        health.validate_health_panel(health_panel["terminal_path"])

    r_member["summary"]["startup"]["verified_mdlm_warm_start_report"][
        "source_size_bytes"
    ] = health.EXPECTED_MDLM_CHECKPOINT_SIZE_BYTES
    _rewrite_summary(r_member)
    r_member["manifest"]["resolved_training_config"]["model"]["vocab_size"] = 1879
    _rewrite_manifest(r_member)
    with pytest.raises(health.HealthPanelValidationError, match="vocab_size"):
        health.validate_health_panel(health_panel["terminal_path"])


def test_health_panel_rejects_variant_order_mutation(
    health_panel: dict[str, object],
) -> None:
    terminal = health_panel["members"][-1]
    binding = terminal["receipt"]["predecessor_receipt_binding"]
    binding["expected_predecessor_training_variant"] = "udlm"
    terminal["manifest"]["predecessor_receipt_binding"] = copy.deepcopy(binding)
    _rewrite_receipt(terminal)
    _rewrite_manifest(terminal)

    with pytest.raises(
        health.HealthPanelValidationError, match="expected predecessor variant"
    ):
        health.validate_health_panel(health_panel["terminal_path"])


def test_health_panel_rejects_per_run_override_hidden_behind_panel(
    health_panel: dict[str, object],
) -> None:
    r_member = health_panel["members"][0]
    r_member["manifest"]["resolved_training_config"]["loader"]["num_workers"] = 2
    _rewrite_manifest(r_member)

    with pytest.raises(
        health.HealthPanelValidationError, match="resolved config num_workers"
    ):
        health.validate_health_panel(health_panel["terminal_path"])


@pytest.mark.parametrize(
    ("field_path", "value", "argv_override"),
    [
        (("optim", "lr"), 1e-4, "optim.lr=0.0001"),
        (
            ("optim", "scheduler", "name"),
            "cosine_decay",
            "optim.scheduler.name=cosine_decay",
        ),
        (("model", "hidden_size"), 512, "model.hidden_size=512"),
    ],
)
def test_health_panel_rejects_self_consistent_common_config_tampering(
    health_panel: dict[str, object],
    field_path: tuple[str, ...],
    value: object,
    argv_override: str,
) -> None:
    common_sha256: str | None = None
    for member in health_panel["members"]:
        manifest = member["manifest"]
        config = manifest["resolved_training_config"]
        target = config
        for key in field_path[:-1]:
            target = target[key]
        target[field_path[-1]] = value
        manifest["resolved_training_config_sha256"] = (
            launch_train_pilot.canonical_json_sha256(config)
        )
        manifest["training_argv"].append(argv_override)
        manifest["training_argv_sha256"] = launch_train_pilot.training_argv_sha256(
            manifest["training_argv"]
        )
        member_common_sha256 = launch_train_pilot.matched_panel_config_sha256(config)
        if common_sha256 is None:
            common_sha256 = member_common_sha256
        else:
            assert member_common_sha256 == common_sha256

    assert common_sha256 is not None
    for member in health_panel["members"]:
        manifest = member["manifest"]
        panel = manifest["matched_panel_spec"]
        panel["common_training_contract"]["common_resolved_config_sha256"] = (
            common_sha256
        )
        manifest["matched_panel_spec_sha256"] = (
            launch_train_pilot.canonical_json_sha256(panel)
        )
        _rewrite_manifest(member)

    with pytest.raises(
        health.HealthPanelValidationError, match="exact registered ten-update"
    ):
        health.validate_health_panel(health_panel["terminal_path"])


def test_health_panel_cross_binds_expected_gpu_count_and_source(
    health_panel: dict[str, object],
) -> None:
    with pytest.raises(health.HealthPanelValidationError, match="health GPU count"):
        health.validate_health_panel(
            health_panel["terminal_path"], expected_gpu_count=2
        )
    with pytest.raises(
        health.HealthPanelValidationError, match="health source revision"
    ):
        health.validate_health_panel(
            health_panel["terminal_path"], expected_source_revision="b" * 40
        )


@pytest.mark.parametrize(
    "relative_path",
    [
        "output/udlm/run/not_the_receipt.json",
        "output/udlm/run/nested/pilot_exit_status.json",
        "elsewhere/run/pilot_exit_status.json",
    ],
)
def test_health_panel_path_contract_fails_closed(
    health_panel: dict[str, object], relative_path: str
) -> None:
    candidate = health_panel["root"] / relative_path
    with pytest.raises(health.HealthPanelValidationError, match="health receipt"):
        health.validate_health_panel(candidate)


def test_health_panel_rejects_symlinked_parent(
    health_panel: dict[str, object], tmp_path: Path
) -> None:
    terminal = health_panel["terminal_path"]
    linked_run = health_panel["root"] / "output/udlm/linked-run"
    linked_run.symlink_to(terminal.parent, target_is_directory=True)
    linked_receipt = linked_run / "pilot_exit_status.json"

    with pytest.raises(health.HealthPanelValidationError, match="symlink"):
        health.validate_health_panel(linked_receipt)


def test_health_panel_rejects_bound_predecessor_snapshot_change(
    health_panel: dict[str, object],
) -> None:
    s_member = health_panel["members"][1]
    s_member["receipt"]["unexpected"] = "mutation"
    _rewrite_receipt(s_member)

    with pytest.raises(
        health.HealthPanelValidationError, match="bound stable snapshot"
    ):
        health.validate_health_panel(health_panel["terminal_path"])

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.udlm import launch_optimization_screen as launcher
from scripts.udlm import launch_train_pilot as pilot


def _registry(gpu_count: int = 1):
    return SimpleNamespace(
        relative_path=Path("experiments/udlm/protocols/screen.json"),
        git_diff_checker=lambda _ancestor, _descendant, _allowed: True,
        reference={
            "relative_path": "experiments/udlm/protocols/screen.json",
            "sha256": "a" * 64,
            "canonical_sha256": "b" * 64,
            "schema_version": 1,
        },
        data={
            "source": {"revision": "a" * 40},
            "common_training": {
                "gpu_count": gpu_count,
                "global_batch_size": 32,
                "micro_batch_size_per_process": 4,
                "accumulate_grad_batches": 8 // gpu_count,
                "effective_global_batch_size": 32,
            },
        },
    )


def _plan(tmp_path: Path, *, gpu_count: int = 1):
    return launcher.ScreenLaunchPlan(
        registry=_registry(gpu_count),
        stage_id="scheduler",
        arm={"arm_id": "E-L0", "attempt_id": "scheduler-e-l0"},
        config_entry={"output_directory": "output/udlm/screens/e-l0"},
        scheduler_dependency=None,
        source_revision="c" * 40,
        gpu_count=gpu_count,
        checkpoint=tmp_path / "mdlm.ckpt",
        checkpoint_sha256="d" * 64,
        run_dir=tmp_path / "output/udlm/screens/e-l0",
        log_path=tmp_path / "output/logs/optimization_screen_scheduler-e-l0.log",
        session_name="genmol_screen_scheduler-e-l0",
        command=[
            sys.executable,
            "-u",
            str(launcher.REPOSITORY_ROOT / "scripts/train.py"),
            "--config-name",
            "udlm_categorical",
        ],
        resolved_config={"training": {"udlm": {"exclude_special_tokens": False}}},
        resolved_config_sha256="e" * 64,
        argv_sha256="f" * 64,
    )


def test_cli_exposes_only_registered_choices_not_training_or_physical_gpu_knobs():
    parsed = launcher._parse_args(
        [
            "--registry",
            "registry.json",
            "--expected-registry-sha256",
            "a" * 64,
            "--expected-registry-canonical-sha256",
            "b" * 64,
            "--stage",
            "scheduler",
            "--arm",
            "E-L0",
            "--dry-run",
        ]
    )
    assert parsed.stage == "scheduler"
    assert parsed.arm == "E-L0"
    assert not hasattr(parsed, "gpu_count")
    assert not hasattr(parsed, "seed")
    assert not hasattr(parsed, "max_steps")

    with pytest.raises(SystemExit):
        launcher._parse_args(
            [
                "--registry",
                "registry.json",
                "--expected-registry-sha256",
                "a" * 64,
                "--expected-registry-canonical-sha256",
                "b" * 64,
                "--stage",
                "scheduler",
                "--arm",
                "E-L0",
                "--gpu-count",
                "1",
            ]
        )


def test_registered_command_reconstructs_every_resolved_config_leaf(tmp_path):
    target, expected_digest = pilot.compose_resolved_training_config(
        config_name="udlm_categorical",
        overrides=[
            "seed=17",
            "trainer.devices=1",
            "trainer.max_steps=100",
            "trainer.accumulate_grad_batches=8",
            "loader.global_batch_size=32",
            "loader.batch_size=4",
            "loader.num_workers=1",
            "callback.every_n_train_steps=100",
            f"callback.dirpath={tmp_path / 'screen/checkpoints'}",
            "training.init_from_mdlm_checkpoint=/tmp/mdlm.ckpt",
            "training.init_from_mdlm_checkpoint_sha256=" + "a" * 64,
            "training.init_from_mdlm_ema=true",
        ],
        gpu_count=1,
    )
    run_dir = launcher.REPOSITORY_ROOT / "output/udlm/screens/test-command"
    command, recomposed, observed_digest = launcher.build_registered_training_command(
        resolved_config=target,
        run_dir=run_dir,
        gpu_count=1,
    )

    assert recomposed == target
    assert observed_digest == expected_digest
    assert pilot.training_argv_sha256(command)
    assert sum(item.startswith("hydra.run.dir=") for item in command) == 1
    assert any(item == "seed=17" for item in command)


def test_pure_bundle_composer_generates_a1_l1_and_scheduler_controls():
    run_dir = launcher.REPOSITORY_ROOT / "output/udlm/screens/e-a1_e-l1"
    command, resolved, digest = launcher.compose_screen_training_bundle(
        stage_id="conditioning",
        arm_id="E-A1",
        scheduler_arm_id="E-L1",
        gpu_count=2,
        run_dir=run_dir,
        checkpoint=Path("/tmp/mdlm.ckpt"),
        checkpoint_sha256="a" * 64,
        global_batch_size=32,
        micro_batch_size=4,
        num_workers=1,
        exclude_special_tokens=False,
    )
    assert resolved["seed"] == 17
    assert resolved["trainer"]["devices"] == 2
    assert resolved["trainer"]["max_steps"] == 500
    assert resolved["trainer"]["accumulate_grad_batches"] == 4
    assert resolved["training"]["udlm"]["prior_variant"] == "empirical_frequency"
    assert resolved["training"]["udlm"]["conditioning_variant"] == "film_adaln"
    assert resolved["training"]["udlm"]["zero_init_conditioning"] is False
    assert resolved["training"]["reseed_after_model_initialization"] is True
    assert resolved["optim"]["scheduler"] == {
        "name": "half_cosine_with_linear_warmup_and_floor",
        "warmup_updates": 50,
        "horizon_updates": 1000,
        "decay_floor_lr": 0.000003,
    }
    assert digest == pilot.canonical_json_sha256(resolved)
    assert pilot.training_argv_sha256(command)

    with pytest.raises(ValueError, match="use its own scheduler"):
        launcher.compose_screen_training_bundle(
            stage_id="scheduler",
            arm_id="E-L0",
            scheduler_arm_id="E-L1",
            gpu_count=1,
            run_dir=run_dir,
            checkpoint=Path("/tmp/mdlm.ckpt"),
            checkpoint_sha256="a" * 64,
            global_batch_size=32,
            micro_batch_size=4,
            num_workers=1,
            exclude_special_tokens=False,
        )


def test_conditioning_dependency_is_mandatory_and_normalized(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="requires authorization"):
        launcher._conditioning_dependency(
            _registry(),
            authorization_revision=None,
            scheduler_evidence=None,
            scheduler_selection=None,
        )

    evidence = tmp_path / "evidence.json"
    selection = tmp_path / "selection.json"
    evidence.write_text('{"schema_version":1}\n')
    selection.write_text('{"schema_version":1}\n')
    monkeypatch.setattr(
        launcher,
        "_repository_json_reference",
        lambda path, expected_schema: {
            "root": "repository",
            "relative_path": path.name,
            "sha256": "1" * 64,
            "size_bytes": 1,
            "schema_version": expected_schema,
            "canonical_sha256": "2" * 64,
        },
    )
    normalized = {
        "authorization_revision": "c" * 40,
        "scheduler_evidence": {"relative_path": "evidence.json"},
        "scheduler_selection": {"relative_path": "selection.json"},
        "selected_scheduler_arm_id": "E-L1",
    }
    monkeypatch.setattr(
        launcher.verifier,
        "_load_declared_scheduler_dependency",
        lambda *_args, **_kwargs: ("E-L1", "c" * 40, normalized),
    )
    registry = _registry()
    observed_diff_checks = []
    registry.git_diff_checker = (
        lambda ancestor, descendant, allowed: observed_diff_checks.append(
            (ancestor, descendant, allowed)
        )
        or True
    )
    revision, observed = launcher._conditioning_dependency(
        registry,
        authorization_revision="c" * 40,
        scheduler_evidence=evidence,
        scheduler_selection=selection,
    )
    assert revision == "c" * 40
    assert observed == normalized
    assert observed_diff_checks == [
        (
            "a" * 40,
            "c" * 40,
            frozenset(
                {
                    "experiments/udlm/protocols/screen.json",
                    "evidence.json",
                    "selection.json",
                }
            ),
        )
    ]


def test_conditioning_dependency_rejects_unregistered_authorization_change(
    monkeypatch, tmp_path
):
    evidence = tmp_path / "evidence.json"
    selection = tmp_path / "selection.json"
    evidence.write_text('{"schema_version":1}\n')
    selection.write_text('{"schema_version":1}\n')
    monkeypatch.setattr(
        launcher,
        "_repository_json_reference",
        lambda path, expected_schema: {
            "root": "repository",
            "relative_path": path.name,
            "sha256": "1" * 64,
            "size_bytes": 1,
            "schema_version": expected_schema,
            "canonical_sha256": "2" * 64,
        },
    )
    normalized = {
        "authorization_revision": "c" * 40,
        "scheduler_evidence": {"relative_path": "evidence.json"},
        "scheduler_selection": {"relative_path": "selection.json"},
        "selected_scheduler_arm_id": "E-L1",
    }
    monkeypatch.setattr(
        launcher.verifier,
        "_load_declared_scheduler_dependency",
        lambda *_args, **_kwargs: ("E-L1", "c" * 40, normalized),
    )
    registry = _registry()
    observed_diff_checks = []
    registry.git_diff_checker = (
        lambda ancestor, descendant, allowed: observed_diff_checks.append(
            (ancestor, descendant, allowed)
        )
        or False
    )

    with pytest.raises(
        launcher.verifier.ScreenValidationError,
        match="conditioning authorization changed unregistered source bytes",
    ):
        launcher._conditioning_dependency(
            registry,
            authorization_revision="c" * 40,
            scheduler_evidence=evidence,
            scheduler_selection=selection,
        )

    assert observed_diff_checks == [
        (
            "a" * 40,
            "c" * 40,
            frozenset(
                {
                    "experiments/udlm/protocols/screen.json",
                    "evidence.json",
                    "selection.json",
                }
            ),
        )
    ]


def test_screen_uses_the_shared_lock_with_an_accurate_reviewed_purpose(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(pilot, "REPOSITORY_ROOT", tmp_path)
    path, record, digest = pilot.acquire_training_job_lock(
        source_revision="a" * 40,
        run_name="scheduler-e-l0",
        training_variant="udlm_categorical",
        purpose=launcher.SCREEN_LOCK_PURPOSE,
    )
    assert record["purpose"] == (
        "enforce_one_registered_optimization_screen_job_at_a_time"
    )
    pilot.release_exact_training_job_lock(path, expected_sha256=digest)
    assert not path.exists()


def test_dry_run_main_never_probes_gpus_locks_artifacts_or_tmux(
    monkeypatch, tmp_path, capsys
):
    plan = _plan(tmp_path)
    monkeypatch.setattr(
        launcher, "load_registry", lambda *_args, **_kwargs: plan.registry
    )
    monkeypatch.setattr(pilot, "require_pushed_commit", lambda: plan.source_revision)
    monkeypatch.setattr(launcher, "build_launch_plan", lambda **_kwargs: plan)
    monkeypatch.setattr(
        launcher.verifier,
        "_screen_binding",
        lambda *_args, **_kwargs: {"arm_id": "E-L0"},
    )
    for name in (
        "probe_all_gpus",
        "reprobe_selected_gpus",
        "acquire_training_job_lock",
        "tmux_session_exists",
    ):
        monkeypatch.setattr(
            pilot,
            name,
            lambda *_args, _name=name, **_kwargs: pytest.fail(
                f"dry run called {_name}"
            ),
        )

    result = launcher.main(
        [
            "--registry",
            "ignored.json",
            "--expected-registry-sha256",
            "a" * 64,
            "--expected-registry-canonical-sha256",
            "b" * 64,
            "--stage",
            "scheduler",
            "--arm",
            "E-L0",
            "--dry-run",
        ]
    )

    preview = json.loads(capsys.readouterr().out)
    assert result == 0
    assert preview["status"] == "dry_run_preflight_completed_no_launch"
    assert preview["gpu_probe_performed"] is False
    assert preview["training_job_lock_acquired"] is False
    assert preview["project_launch_artifact_mutation_performed"] is False
    assert not (tmp_path / "output").exists()


def test_real_launch_uses_final_uuid_probe_before_manifest_and_tmux(
    monkeypatch, tmp_path
):
    plan = _plan(tmp_path)
    state = pilot.GPUState(
        physical_index=7,
        uuid="GPU-idle-seven",
        name="test card",
        memory_used_mib=100,
        memory_total_mib=80_000,
        utilization_percent=1,
        compute_mode="Default",
        compute_processes=(),
    )
    events: list[str] = []
    monkeypatch.setattr(
        pilot, "probe_all_gpus", lambda: events.append("inventory") or [state]
    )
    monkeypatch.setattr(
        pilot,
        "select_idle_gpus",
        lambda *_args, **_kwargs: events.append("select") or (state,),
    )
    monkeypatch.setattr(
        pilot,
        "require_pushed_commit",
        lambda: events.append("source") or plan.source_revision,
    )
    monkeypatch.setattr(
        pilot,
        "reprobe_selected_gpus",
        lambda *_args, **_kwargs: events.append("final_probe") or (state,),
    )
    monkeypatch.setattr(
        pilot,
        "validate_pilot_exit_receipt_path",
        lambda path: path,
    )
    original_publish = pilot._atomic_publish_bytes_exclusive

    def publish(path, payload, *, label):
        events.append("manifest")
        return original_publish(path, payload, label=label)

    monkeypatch.setattr(pilot, "_atomic_publish_bytes_exclusive", publish)
    monkeypatch.setattr(
        pilot,
        "build_child_environment_command",
        lambda **_kwargs: (["env", "python"], {}),
    )
    monkeypatch.setattr(
        pilot, "build_tmux_shell_command", lambda *_args, **_kwargs: "true"
    )
    monkeypatch.setattr(
        launcher.verifier,
        "_screen_binding",
        lambda *_args, **_kwargs: {"arm_id": "E-L0", "scheduler_authorization": None},
    )

    def run(command, **_kwargs):
        assert command[:3] == ["tmux", "new-session", "-d"]
        events.append("tmux")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(launcher.subprocess, "run", run)
    payload, digest, log_path = launcher._launch_locked_screen(
        plan,
        max_utilization_percent=10,
        min_free_memory_mib=30_000,
        lock_path=tmp_path / ".single_training_job.lock",
        lock_record={"status": "held"},
        lock_sha256="9" * 64,
    )

    manifest = json.loads(payload)
    assert digest == launcher.verifier.hashlib.sha256(payload).hexdigest()
    assert manifest["purpose"] == "registered UDLM optimization screen"
    assert manifest["optimization_screen"]["arm_id"] == "E-L0"
    assert manifest["cuda_visible_device_uuids"] == ["GPU-idle-seven"]
    assert manifest["physical_gpu_indices"] == [7]
    assert manifest["gpu_safety_policy"]["active_compute_processes_allowed"] is True
    assert manifest["checkpoint_sha256"] == plan.checkpoint_sha256
    assert events == [
        "inventory",
        "select",
        "source",
        "final_probe",
        "manifest",
        "tmux",
    ]
    assert log_path.is_file()


def test_real_main_acquires_shared_lock_before_entering_gpu_launch_phase(
    monkeypatch, tmp_path, capsys
):
    plan = _plan(tmp_path)
    events: list[str] = []
    monkeypatch.setattr(
        launcher, "load_registry", lambda *_args, **_kwargs: plan.registry
    )
    monkeypatch.setattr(pilot, "require_pushed_commit", lambda: plan.source_revision)
    monkeypatch.setattr(launcher, "build_launch_plan", lambda **_kwargs: plan)
    monkeypatch.setattr(pilot, "tmux_session_exists", lambda _name: False)

    def acquire(**kwargs):
        events.append("lock")
        assert kwargs["purpose"] == launcher.SCREEN_LOCK_PURPOSE
        return tmp_path / ".lock", {"status": "held"}, "9" * 64

    def launch(_plan, **_kwargs):
        assert events == ["lock"]
        events.append("gpu_launch_phase")
        return b'{"launched":true}\n', "8" * 64, plan.log_path

    monkeypatch.setattr(pilot, "acquire_training_job_lock", acquire)
    monkeypatch.setattr(launcher, "_launch_locked_screen", launch)
    result = launcher.main(
        [
            "--registry",
            "ignored.json",
            "--expected-registry-sha256",
            "a" * 64,
            "--expected-registry-canonical-sha256",
            "b" * 64,
            "--stage",
            "scheduler",
            "--arm",
            "E-L0",
        ]
    )
    assert result == 0
    assert events == ["lock", "gpu_launch_phase"]
    assert "launch manifest SHA-256" in capsys.readouterr().out


def test_help_works_without_site_packages_or_training_imports():
    source = launcher.REPOSITORY_ROOT / "scripts/udlm/launch_optimization_screen.py"
    result = subprocess.run(
        [sys.executable, "-S", str(source), "--help"],
        cwd=launcher.REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--expected-registry-canonical-sha256" in result.stdout
    assert "--gpu-count" not in result.stdout

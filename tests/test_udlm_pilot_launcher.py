import subprocess
import threading
from pathlib import Path

import pytest

from scripts.udlm import launch_train_pilot as launcher


def _gpu(
    *,
    index: int,
    uuid: str,
    memory_used_mib: int = 1_000,
    utilization_percent: int = 2,
    processes: tuple[dict[str, object], ...] = (),
) -> launcher.GPUState:
    return launcher.GPUState(
        physical_index=index,
        uuid=uuid,
        name="Example",
        memory_used_mib=memory_used_mib,
        memory_total_mib=48_000,
        utilization_percent=utilization_percent,
        compute_mode="Default",
        compute_processes=processes,
    )


def test_gpu_request_accepts_only_a_count_capped_at_two():
    assert launcher.validate_gpu_count(1) == 1
    assert launcher.validate_gpu_count(2) == 2
    with pytest.raises(ValueError, match="1 or 2"):
        launcher.validate_gpu_count(3)
    with pytest.raises(ValueError, match="1 or 2"):
        launcher.validate_gpu_count(True)

    parsed = launcher._parse_args(
        ["--run-name", "count_only", "--gpu-count", "2", "--scratch"]
    )
    assert parsed.gpu_count == 2
    assert parsed.training_variant == "udlm"
    assert not hasattr(parsed, "gpu_indices")
    with pytest.raises(SystemExit):
        launcher._parse_args(
            [
                "--run-name",
                "ids_are_forbidden",
                "--gpu-count",
                "1",
                "--gpu-indices",
                "3",
                "--scratch",
            ]
        )
    with pytest.raises(SystemExit):
        launcher._parse_args(
            [
                "--run-name",
                "arbitrary_config_forbidden",
                "--gpu-count",
                "1",
                "--config-name",
                "anything",
                "--scratch",
            ]
        )


def test_gpu_with_compute_process_is_not_genuinely_idle():
    state = _gpu(
        index=2,
        uuid="GPU-example",
        processes=({"pid": 123, "process_name": "python", "used_memory_mib": 900},),
    )

    reasons = state.rejection_reasons(
        max_utilization_percent=10,
        min_free_memory_mib=30_000,
    )

    assert any("active compute" in reason for reason in reasons)


def test_full_inventory_probe_records_uuid_telemetry_and_processes(monkeypatch):
    status = subprocess.CompletedProcess(
        [],
        0,
        "0, GPU-zero, Card A, 20, 48000, 1, Default\n"
        "3, GPU-three, Card B, 2000, 48000, 7, Default\n",
        "",
    )
    processes = subprocess.CompletedProcess(
        [],
        0,
        "GPU-three, 991, /other/user/python, 1800\n",
        "",
    )
    monkeypatch.setattr(launcher, "_run", lambda _command: status)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: processes)

    states = launcher.probe_all_gpus()

    assert [(state.physical_index, state.uuid) for state in states] == [
        (0, "GPU-zero"),
        (3, "GPU-three"),
    ]
    assert states[0].compute_processes == ()
    assert states[1].compute_processes[0]["pid"] == 991


@pytest.mark.parametrize(
    "process_output",
    [
        (
            "GPU-three, 991, /other/user/python, 1800\n"
            "GPU-three, 991, /other/user/python, 1800\n"
        ),
        ("No running processes found\n" "GPU-three, 991, /other/user/python, 1800\n"),
        "GPU-unknown, 991, /other/user/python, 1800\n",
    ],
)
def test_inventory_probe_rejects_duplicate_or_ambiguous_process_telemetry(
    monkeypatch,
    process_output,
):
    status = subprocess.CompletedProcess(
        [],
        0,
        "0, GPU-zero, Card A, 20, 48000, 1, Default\n"
        "3, GPU-three, Card B, 2000, 48000, 7, Default\n",
        "",
    )
    processes = subprocess.CompletedProcess([], 0, process_output, "")
    monkeypatch.setattr(launcher, "_run", lambda _command: status)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: processes)

    with pytest.raises(RuntimeError, match="ambiguous|invalid"):
        launcher.probe_all_gpus()


def test_dynamic_selection_ranks_only_genuinely_idle_devices():
    states = [
        _gpu(index=0, uuid="GPU-busy", processes=({"pid": 1},)),
        _gpu(index=1, uuid="GPU-low-free", memory_used_mib=19_000),
        _gpu(index=2, uuid="GPU-best", memory_used_mib=100),
        _gpu(index=3, uuid="GPU-next", memory_used_mib=500),
    ]

    selected = launcher.select_idle_gpus(
        states,
        gpu_count=2,
        max_utilization_percent=10,
        min_free_memory_mib=30_000,
    )

    assert [state.uuid for state in selected] == ["GPU-best", "GPU-next"]
    with pytest.raises(RuntimeError, match="only 1 are genuinely idle"):
        launcher.select_idle_gpus(
            states,
            gpu_count=2,
            max_utilization_percent=10,
            min_free_memory_mib=47_700,
        )


def test_final_probe_addresses_exact_selected_uuids(monkeypatch):
    initial = (
        _gpu(index=2, uuid="GPU-two"),
        _gpu(index=5, uuid="GPU-five"),
    )
    probed = []

    def probe(device_uuid):
        probed.append(device_uuid)
        return next(state for state in initial if state.uuid == device_uuid)

    monkeypatch.setattr(launcher, "probe_gpu_uuid", probe)
    current = launcher.reprobe_selected_gpus(
        initial,
        max_utilization_percent=10,
        min_free_memory_mib=30_000,
    )

    assert current == initial
    assert probed == ["GPU-two", "GPU-five"]

    monkeypatch.setattr(
        launcher,
        "probe_gpu_uuid",
        lambda uuid: _gpu(index=2, uuid=uuid, processes=({"pid": 99},)),
    )
    with pytest.raises(RuntimeError, match="final idle probe"):
        launcher.reprobe_selected_gpus(
            (initial[0],),
            max_utilization_percent=10,
            min_free_memory_mib=30_000,
        )

    monkeypatch.setattr(
        launcher,
        "probe_gpu_uuid",
        lambda _uuid: _gpu(index=2, uuid="GPU-different"),
    )
    with pytest.raises(RuntimeError, match="UUID identity changed"):
        launcher.reprobe_selected_gpus(
            (initial[0],),
            max_utilization_percent=10,
            min_free_memory_mib=30_000,
        )


def test_project_gpu_safety_thresholds_cannot_be_relaxed():
    with pytest.raises(ValueError, match="max-utilization-percent"):
        launcher.validate_safety_thresholds(101, 30_000)
    with pytest.raises(ValueError, match="min-free-memory-mib"):
        launcher.validate_safety_thresholds(10, 0)
    launcher.validate_safety_thresholds(5, 40_000)


def test_accumulation_requires_an_exact_global_batch():
    assert launcher.exact_accumulation_steps(16, 2, 2) == 4
    with pytest.raises(ValueError, match="exact positive multiple"):
        launcher.exact_accumulation_steps(15, 2, 2)
    with pytest.raises(ValueError, match="exact positive multiple"):
        launcher.exact_accumulation_steps(2, 2, 2)
    with pytest.raises(ValueError, match="must be integers"):
        launcher.exact_accumulation_steps(16, 2, True)


def test_training_command_records_bounded_pilot_controls(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "_python_executable", lambda: Path("/venv/python"))
    command = launcher.build_training_command(
        gpu_count=2,
        run_dir=tmp_path / "pilot",
        max_steps=10,
        global_batch_size=16,
        micro_batch_size=2,
        num_workers=1,
        seed=7,
        checkpoint=Path("/project/50000.ckpt"),
        checkpoint_sha256="a" * 64,
        exclude_special_tokens=True,
    )
    joined = " ".join(str(part) for part in command)

    assert "--config-name udlm" in joined
    assert "trainer.devices=2" in joined
    assert "trainer.max_steps=10" in joined
    assert "trainer.detect_anomaly=true" in joined
    assert "loader.global_batch_size=16" in joined
    assert "loader.batch_size=2" in joined
    assert "training.pilot_fail_on_nonfinite_loss=true" in joined
    assert "training.init_from_mdlm_checkpoint=/project/50000.ckpt" in joined
    assert f"training.init_from_mdlm_checkpoint_sha256={'a' * 64}" in joined
    assert "training.udlm.exclude_special_tokens=true" in joined
    with pytest.raises(ValueError, match="must be 1 or 2"):
        launcher.build_training_command(
            gpu_count=True,
            run_dir=tmp_path / "pilot",
            max_steps=10,
            global_batch_size=16,
            micro_batch_size=2,
            num_workers=1,
            seed=7,
            checkpoint=None,
            exclude_special_tokens=False,
        )


def test_training_command_checkpoint_digest_validation(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "_python_executable", lambda: Path("/venv/python"))
    common = {
        "gpu_count": 1,
        "run_dir": tmp_path / "pilot",
        "max_steps": 3,
        "global_batch_size": 4,
        "micro_batch_size": 2,
        "num_workers": 0,
        "seed": 1,
        "exclude_special_tokens": False,
    }

    legacy = launcher.build_training_command(
        **common,
        checkpoint=Path("/project/50000.ckpt"),
    )
    assert not any("checkpoint_sha256" in value for value in legacy)

    with pytest.raises(ValueError, match="requires a checkpoint"):
        launcher.build_training_command(
            **common,
            checkpoint=None,
            checkpoint_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="64 lowercase"):
        launcher.build_training_command(
            **common,
            checkpoint=Path("/project/50000.ckpt"),
            checkpoint_sha256="INVALID",
        )


def test_resolved_hydra_config_is_bound_before_launch(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "_python_executable", lambda: Path("/venv/python"))
    command = launcher.build_training_command(
        gpu_count=2,
        run_dir=tmp_path / "pilot",
        max_steps=10,
        global_batch_size=16,
        micro_batch_size=2,
        num_workers=1,
        seed=7,
        checkpoint=None,
        exclude_special_tokens=False,
        training_variant="schedule_uniform",
    )

    config, digest = launcher.compose_resolved_training_config(
        config_name="udlm",
        overrides=command[5:],
        gpu_count=2,
    )

    assert len(digest) == 64
    assert digest == launcher.canonical_json_sha256(config)
    assert config["seed"] == 7
    assert config["trainer"]["devices"] == 2
    assert config["trainer"]["accumulate_grad_batches"] == 4
    assert config["trainer"]["detect_anomaly"] is True
    assert config["training"]["pilot_fail_on_nonfinite_loss"] is True
    assert config["training"]["udlm"]["prior_variant"] == "schedule_uniform"
    assert "hydra" not in config


def test_child_command_sanitizes_python_and_binds_source_argv_config(
    monkeypatch,
):
    monkeypatch.setenv("PYTHONHOME", "/hostile/home")
    monkeypatch.setenv("PYTHONWARNINGS", "error")
    monkeypatch.setenv("PYTHONARBITRARY", "hostile")
    monkeypatch.setenv("LOCAL_RANK", "7")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("MASTER_ADDR", "untrusted.example")
    monkeypatch.setenv("GENMOL_TRAIN_UNEXPECTED", "stale")
    command = [
        "/venv/python",
        "-u",
        str(launcher.REPOSITORY_ROOT / "scripts/train.py"),
        "--config-name",
        "udlm",
        "seed=7",
    ]
    runtime_path = launcher.REPOSITORY_ROOT / "output/udlm/test/runtime_config.json"
    summary_path = launcher.REPOSITORY_ROOT / "output/udlm/test/training_summary.json"
    checkpoint_path = launcher.REPOSITORY_ROOT / "output/udlm/test/checkpoints/10.ckpt"
    manifest_path = launcher.REPOSITORY_ROOT / "output/udlm/test/launch_manifest.json"

    child_command, environment = launcher.build_child_environment_command(
        command=command,
        source_revision="a" * 40,
        resolved_config_sha256="b" * 64,
        runtime_config_path=runtime_path,
        training_summary_path=summary_path,
        final_checkpoint_path=checkpoint_path,
        launch_manifest_path=manifest_path,
        launch_manifest_sha256="c" * 64,
        expected_max_steps=10,
        expected_world_size=2,
        visible_uuids="GPU-one,GPU-two",
        seed=7,
    )

    assert child_command[-len(command) :] == command
    assert environment["GENMOL_TRAIN_EXPECTED_SOURCE_REVISION"] == "a" * 40
    assert environment["GENMOL_TRAIN_EXPECTED_CONFIG_SHA256"] == "b" * 64
    assert environment["GENMOL_TRAIN_EXPECTED_ARGV_SHA256"] == (
        launcher.canonical_json_sha256(command[2:])
    )
    assert environment["PYTHONHASHSEED"] == "7"
    assert environment["GENMOL_TRAIN_SUMMARY_PATH"] == str(summary_path)
    assert environment["GENMOL_TRAIN_EXPECTED_SUMMARY_SCHEMA_VERSION"] == "4"
    assert environment["GENMOL_TRAIN_EXPECTED_FINAL_CHECKPOINT_PATH"] == str(
        checkpoint_path
    )
    assert environment["GENMOL_TRAIN_EXPECTED_MAX_STEPS"] == "10"
    assert environment["GENMOL_TRAIN_EXPECTED_WORLD_SIZE"] == "2"
    assert environment["GENMOL_TRAIN_LAUNCH_MANIFEST_PATH"] == str(manifest_path)
    assert environment["GENMOL_TRAIN_EXPECTED_LAUNCH_MANIFEST_SHA256"] == "c" * 64
    assert environment["GENMOL_TRAIN_SELECTED_GPU_UUIDS_JSON"] == (
        '["GPU-one","GPU-two"]'
    )
    assert environment["PYTHONPATH"] == launcher.os.pathsep.join(
        [
            str(launcher.REPOSITORY_ROOT / "src"),
            str(launcher.REPOSITORY_ROOT),
        ]
    )
    for hostile_key in ("PYTHONHOME", "PYTHONWARNINGS", "PYTHONARBITRARY"):
        assert ["-u", hostile_key] in [
            child_command[index : index + 2] for index in range(len(child_command) - 1)
        ]
    assert ["-u", "GENMOL_TRAIN_UNEXPECTED"] in [
        child_command[index : index + 2] for index in range(len(child_command) - 1)
    ]
    for distributed_key in launcher.DISTRIBUTED_ENVIRONMENT_KEYS:
        assert ["-u", distributed_key] in [
            child_command[index : index + 2] for index in range(len(child_command) - 1)
        ]


def test_tmux_command_captures_both_pipeline_statuses_for_receipt(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(launcher, "_python_executable", lambda: Path("/venv/python"))
    log_path = tmp_path / "output/logs/pilot.log"
    summary_path = tmp_path / "output/udlm/pilot/training_summary.json"
    receipt_path = tmp_path / "output/udlm/pilot/pilot_exit_status.json"
    shell_command = launcher.build_tmux_shell_command(
        ["bash", "-c", "exit 0"],
        log_path=log_path,
        training_summary_path=summary_path,
        exit_receipt_path=receipt_path,
        expected_source_revision="a" * 40,
        expected_config_sha256="b" * 64,
        expected_argv_sha256="c" * 64,
        expected_summary_schema_version=launcher.TRAINING_SUMMARY_SCHEMA_VERSION,
        expected_max_steps=10,
        expected_world_size=2,
        expected_final_checkpoint_path=(
            tmp_path / "output/udlm/pilot/checkpoints/10.ckpt"
        ),
        expected_launch_manifest_path=(
            tmp_path / "output/udlm/pilot/launch_manifest.json"
        ),
        expected_launch_manifest_sha256="d" * 64,
        expected_selected_gpu_uuids_json='["GPU-one","GPU-two"]',
        expected_training_job_lock_path=(
            tmp_path / "output/udlm/.single_training_job.lock"
        ),
        expected_training_job_lock_sha256="e" * 64,
    )

    assert 'pipeline_status=("${PIPESTATUS[@]}")' in shell_command
    assert 'training_status="${pipeline_status[0]}"' in shell_command
    assert 'tee_status="${pipeline_status[1]}"' in shell_command
    assert "write_pilot_exit_status.py" in shell_command
    assert str(receipt_path) in shell_command
    assert "--expected-launch-manifest-sha256" in shell_command
    assert "--expected-selected-gpu-uuids-json" in shell_command


def test_tmux_command_rejects_legacy_training_summary_schema(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", tmp_path)
    with pytest.raises(ValueError, match="unexpected training summary schema version"):
        launcher.build_tmux_shell_command(
            ["bash", "-c", "exit 0"],
            log_path=tmp_path / "output/logs/pilot.log",
            training_summary_path=(
                tmp_path / "output/udlm/pilot/training_summary.json"
            ),
            exit_receipt_path=(tmp_path / "output/udlm/pilot/pilot_exit_status.json"),
            expected_source_revision="a" * 40,
            expected_config_sha256="b" * 64,
            expected_argv_sha256="c" * 64,
            expected_summary_schema_version=1,
            expected_max_steps=10,
            expected_world_size=1,
            expected_final_checkpoint_path=(
                tmp_path / "output/udlm/pilot/checkpoints/10.ckpt"
            ),
            expected_launch_manifest_path=(
                tmp_path / "output/udlm/pilot/launch_manifest.json"
            ),
            expected_launch_manifest_sha256="d" * 64,
            expected_selected_gpu_uuids_json='["GPU-one"]',
            expected_training_job_lock_path=(
                tmp_path / "output/udlm/.single_training_job.lock"
            ),
            expected_training_job_lock_sha256="e" * 64,
        )


def test_exit_receipt_path_must_be_new_and_inside_repository(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", tmp_path)
    valid = tmp_path / "output/udlm/pilot/pilot_exit_status.json"
    assert launcher.validate_pilot_exit_receipt_path(valid) == valid
    with pytest.raises(ValueError, match="in-repository"):
        launcher.validate_pilot_exit_receipt_path(
            tmp_path.parent / "outside/pilot_exit_status.json"
        )
    valid.parent.mkdir(parents=True)
    valid.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to replace"):
        launcher.validate_pilot_exit_receipt_path(valid)


@pytest.mark.parametrize(
    ("training_variant", "config_name", "fixed_prior_override"),
    [
        ("udlm", "udlm", None),
        (
            "schedule_uniform",
            "udlm",
            "training.udlm.prior_variant=schedule_uniform",
        ),
        ("udlm_categorical", "udlm_categorical", None),
    ],
)
def test_training_command_allows_only_reviewed_prior_variants(
    monkeypatch,
    tmp_path,
    training_variant,
    config_name,
    fixed_prior_override,
):
    monkeypatch.setattr(launcher, "_python_executable", lambda: Path("/venv/python"))
    command = launcher.build_training_command(
        gpu_count=1,
        run_dir=tmp_path / "pilot",
        max_steps=3,
        global_batch_size=4,
        micro_batch_size=2,
        num_workers=0,
        seed=1,
        checkpoint=None,
        exclude_special_tokens=False,
        training_variant=training_variant,
    )

    assert command[command.index("--config-name") + 1] == config_name
    if fixed_prior_override is None:
        assert not any(
            value.startswith("training.udlm.prior_variant=") for value in command
        )
    else:
        assert fixed_prior_override in command
    assert "trainer.max_steps=3" in command

    with pytest.raises(ValueError, match="training-variant"):
        launcher.build_training_command(
            gpu_count=1,
            run_dir=tmp_path / "pilot",
            max_steps=3,
            global_batch_size=4,
            micro_batch_size=2,
            num_workers=0,
            seed=1,
            checkpoint=None,
            exclude_special_tokens=False,
            training_variant="../../arbitrary.yaml",
        )


def test_pushed_commit_check_rejects_untracked_source_but_allows_output(monkeypatch):
    monkeypatch.setattr(
        launcher,
        "_run",
        lambda command: subprocess.CompletedProcess(
            command,
            0,
            stdout=("?? output/logs/run.log\0?? scripts/new_launcher.py\0"),
            stderr="",
        ),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
    )

    with pytest.raises(RuntimeError, match="scripts/new_launcher.py"):
        launcher.require_pushed_commit()


def test_matched_panel_digest_masks_only_registered_treatment_and_run_path(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(launcher, "_python_executable", lambda: Path("/venv/python"))
    common_digests = []
    panel_digests = []
    for training_variant in launcher.MATCHED_PANEL_VARIANT_ORDER:
        run_dir = tmp_path / training_variant
        command = launcher.build_training_command(
            gpu_count=1,
            run_dir=run_dir,
            max_steps=10,
            global_batch_size=16,
            micro_batch_size=2,
            num_workers=1,
            seed=1,
            checkpoint=None,
            exclude_special_tokens=False,
            training_variant=training_variant,
        )
        definition = launcher.TRAINING_VARIANTS[training_variant]
        resolved, _digest = launcher.compose_resolved_training_config(
            config_name=str(definition["config_name"]),
            overrides=command[5:],
            gpu_count=1,
        )
        common_digest = launcher.matched_panel_config_sha256(resolved)
        common_digests.append(common_digest)
        spec, panel_digest = launcher.build_matched_panel_spec(
            source_revision="a" * 40,
            checkpoint=None,
            checkpoint_sha256=None,
            gpu_count=1,
            max_steps=10,
            global_batch_size=16,
            micro_batch_size=2,
            num_workers=1,
            seed=1,
            exclude_special_tokens=False,
            max_utilization_percent=10,
            min_free_memory_mib=30_000,
            common_resolved_config_sha256=common_digest,
        )
        assert spec["execution"] == {
            "mode": "single_job_lease_with_registered_order_policy",
            "maximum_concurrent_training_jobs": 1,
            "concurrency_enforcement": "atomic_global_worktree_training_job_lock",
            "registered_variant_order": list(launcher.MATCHED_PANEL_VARIANT_ORDER),
            "advance_policy": "operator_validates_successful_predecessor_receipt",
            "predecessor_receipt_bound_in_each_manifest": False,
        }
        panel_digests.append(panel_digest)

    assert len(set(common_digests)) == 1
    assert len(set(panel_digests)) == 1

    altered_resolved = launcher.json.loads(launcher.json.dumps(resolved))
    altered_resolved["trainer"]["max_steps"] = 11
    assert launcher.matched_panel_config_sha256(altered_resolved) != common_digests[0]

    changed_spec, changed_digest = launcher.build_matched_panel_spec(
        source_revision="a" * 40,
        checkpoint=None,
        checkpoint_sha256=None,
        gpu_count=1,
        max_steps=10,
        global_batch_size=16,
        micro_batch_size=2,
        num_workers=1,
        seed=2,
        exclude_special_tokens=False,
        max_utilization_percent=10,
        min_free_memory_mib=30_000,
        common_resolved_config_sha256=common_digests[0],
    )
    assert changed_spec["common_training_contract"]["seed"] == 2
    assert changed_digest != panel_digests[0]


@pytest.mark.parametrize("seed", [-1, launcher.MAX_TRAINING_SEED + 1, True])
def test_matched_panel_rejects_seed_outside_exact_lightning_range(seed):
    with pytest.raises(ValueError, match="training controls are invalid"):
        launcher.build_matched_panel_spec(
            source_revision="a" * 40,
            checkpoint=None,
            checkpoint_sha256=None,
            gpu_count=1,
            max_steps=10,
            global_batch_size=16,
            micro_batch_size=2,
            num_workers=1,
            seed=seed,
            exclude_special_tokens=False,
            max_utilization_percent=10,
            min_free_memory_mib=30_000,
            common_resolved_config_sha256="b" * 64,
        )


def test_dry_run_is_nonmutating_and_never_probes_gpus_or_tmux(
    monkeypatch, tmp_path, capsys
):
    repository_root = tmp_path / "worktree"
    repository_root.mkdir()
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository_root)
    monkeypatch.setattr(launcher, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        launcher,
        "_parse_args",
        lambda: launcher.argparse.Namespace(
            run_name="dry_preview",
            training_variant="udlm",
            gpu_count=1,
            max_steps=10,
            global_batch_size=16,
            micro_batch_size=2,
            num_workers=1,
            seed=1,
            checkpoint=tmp_path / "unused.ckpt",
            scratch=True,
            exclude_special_tokens=False,
            max_utilization_percent=10,
            min_free_memory_mib=30_000,
            dry_run=True,
        ),
    )
    monkeypatch.setattr(launcher, "_python_executable", lambda: Path("/venv/python"))
    monkeypatch.setattr(launcher, "require_pushed_commit", lambda: "a" * 40)
    monkeypatch.setattr(
        launcher,
        "compose_resolved_training_config",
        lambda **_kwargs: (
            {
                "seed": 1,
                "training": {"udlm": {"prior_variant": "release_uniform"}},
                "callback": {
                    "dirpath": str(
                        repository_root / "output/udlm/dry_preview/checkpoints"
                    )
                },
            },
            "b" * 64,
        ),
    )
    monkeypatch.setattr(
        launcher,
        "probe_all_gpus",
        lambda: pytest.fail("dry run must not probe GPU inventory"),
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("dry run must not invoke tmux"),
    )

    launcher.main()

    preview = launcher.json.loads(capsys.readouterr().out)
    assert preview["status"] == "dry_run_preflight_completed_no_launch"
    assert preview["project_launch_artifact_mutation_performed"] is False
    assert preview["gpu_probe_performed"] is False
    assert not (repository_root / "output").exists()


def test_manifest_publication_and_log_reservation_are_exclusive(tmp_path):
    manifest_path = tmp_path / "run/launch_manifest.json"
    payload = b'{"complete":true}\n'
    digest = launcher._atomic_publish_bytes_exclusive(
        manifest_path, payload, label="pilot launch manifest"
    )
    assert manifest_path.read_bytes() == payload
    assert digest == launcher.hashlib.sha256(payload).hexdigest()
    with pytest.raises(FileExistsError, match="refusing to replace"):
        launcher._atomic_publish_bytes_exclusive(
            manifest_path, b"other", label="pilot launch manifest"
        )

    log_path = tmp_path / "logs/pilot.log"
    launcher.reserve_log_path(log_path)
    assert log_path.is_file()
    assert log_path.read_bytes() == b""
    with pytest.raises(FileExistsError, match="refusing to replace"):
        launcher.reserve_log_path(log_path)

    dangling = tmp_path / "logs/dangling.log"
    dangling.symlink_to(tmp_path / "missing-target.log")
    assert launcher.os.path.lexists(dangling)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        launcher.reserve_log_path(dangling)


def test_training_job_lock_race_has_one_owner_and_exact_release(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", tmp_path)
    barrier = threading.Barrier(2)
    successes = []
    failures = []

    def acquire(run_name):
        barrier.wait()
        try:
            successes.append(
                launcher.acquire_training_job_lock(
                    source_revision="a" * 40,
                    run_name=run_name,
                    training_variant="udlm",
                )
            )
        except RuntimeError as error:
            failures.append(str(error))

    threads = [
        threading.Thread(target=acquire, args=(f"racer_{index}",)) for index in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(successes) == 1
    assert len(failures) == 1
    assert "fail closed" in failures[0]
    lock_path, _record, digest = successes[0]
    with pytest.raises(RuntimeError, match="owned by another run"):
        launcher.release_exact_training_job_lock(lock_path, expected_sha256="0" * 64)
    assert lock_path.is_file()
    launcher.release_exact_training_job_lock(lock_path, expected_sha256=digest)
    assert not launcher.os.path.lexists(lock_path)


def test_exact_lock_release_ignores_read_updated_atime(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", tmp_path)
    lock_path, _record, digest = launcher.acquire_training_job_lock(
        source_revision="a" * 40,
        run_name="old_atime",
        training_variant="udlm",
    )
    state = lock_path.stat()
    launcher.os.utime(
        lock_path,
        ns=(state.st_mtime_ns - 86_400_000_000_000, state.st_mtime_ns),
    )

    launcher.release_exact_training_job_lock(lock_path, expected_sha256=digest)

    assert not launcher.os.path.lexists(lock_path)


def _mock_main_cpu_preflight(monkeypatch, tmp_path, *, run_name):
    repository_root = tmp_path / "worktree"
    repository_root.mkdir(exist_ok=True)
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository_root)
    monkeypatch.setattr(launcher, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        launcher,
        "_parse_args",
        lambda: launcher.argparse.Namespace(
            run_name=run_name,
            training_variant="udlm",
            gpu_count=1,
            max_steps=10,
            global_batch_size=16,
            micro_batch_size=2,
            num_workers=1,
            seed=1,
            checkpoint=tmp_path / "unused.ckpt",
            scratch=True,
            exclude_special_tokens=False,
            max_utilization_percent=10,
            min_free_memory_mib=30_000,
            dry_run=False,
        ),
    )
    monkeypatch.setattr(launcher, "_python_executable", lambda: Path("/venv/python"))
    monkeypatch.setattr(launcher, "require_pushed_commit", lambda: "a" * 40)
    monkeypatch.setattr(
        launcher,
        "compose_resolved_training_config",
        lambda **_kwargs: (
            {
                "seed": 1,
                "training": {"udlm": {"prior_variant": "release_uniform"}},
                "callback": {
                    "dirpath": str(
                        repository_root / f"output/udlm/{run_name}/checkpoints"
                    )
                },
            },
            "b" * 64,
        ),
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda command, **_kwargs: (
            subprocess.CompletedProcess(command, 1)
            if command[:2] == ["tmux", "has-session"]
            else pytest.fail(f"unexpected subprocess: {command}")
        ),
    )
    return repository_root


def test_existing_or_stale_training_lock_fails_before_gpu_probe(monkeypatch, tmp_path):
    repository_root = _mock_main_cpu_preflight(
        monkeypatch, tmp_path, run_name="blocked"
    )
    lock_path, _record, _digest = launcher.acquire_training_job_lock(
        source_revision="a" * 40,
        run_name="existing",
        training_variant="schedule_uniform",
    )
    monkeypatch.setattr(
        launcher,
        "probe_all_gpus",
        lambda: pytest.fail("overlap must fail before any GPU probe"),
    )

    with pytest.raises(RuntimeError, match="another or stale.*fail closed"):
        launcher.main()

    assert lock_path == repository_root / "output/udlm/.single_training_job.lock"
    assert lock_path.is_file()


def test_pre_tmux_launch_failure_releases_only_acquired_lock(monkeypatch, tmp_path):
    repository_root = _mock_main_cpu_preflight(
        monkeypatch, tmp_path, run_name="pre_handoff_failure"
    )

    def fail_before_handoff(**kwargs):
        lock_path = kwargs["lock_path"]
        assert lock_path.is_file()
        assert (
            launcher.hashlib.sha256(lock_path.read_bytes()).hexdigest()
            == kwargs["lock_sha256"]
        )
        raise RuntimeError("synthetic pre-tmux failure")

    monkeypatch.setattr(launcher, "_launch_locked_pilot", fail_before_handoff)

    with pytest.raises(RuntimeError, match="synthetic pre-tmux failure"):
        launcher.main()

    assert not launcher.os.path.lexists(
        repository_root / "output/udlm/.single_training_job.lock"
    )


def test_ambiguous_tmux_handoff_retains_lock_fail_closed(monkeypatch, tmp_path):
    repository_root = _mock_main_cpu_preflight(
        monkeypatch, tmp_path, run_name="ambiguous_handoff"
    )
    has_session_calls = 0

    def tmux_state(command, **_kwargs):
        nonlocal has_session_calls
        assert command[:2] == ["tmux", "has-session"]
        has_session_calls += 1
        return subprocess.CompletedProcess(
            command,
            1 if has_session_calls == 1 else 0,
        )

    monkeypatch.setattr(launcher.subprocess, "run", tmux_state)
    monkeypatch.setattr(
        launcher,
        "_launch_locked_pilot",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic post-handoff ambiguity")
        ),
    )

    with pytest.raises(RuntimeError, match="lock was retained fail-closed"):
        launcher.main()

    assert has_session_calls == 2
    assert (repository_root / "output/udlm/.single_training_job.lock").is_file()


def test_main_keeps_final_uuid_probe_adjacent_to_tmux_spawn(monkeypatch, tmp_path):
    repository_root = tmp_path / "worktree"
    repository_root.mkdir()
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository_root)
    monkeypatch.setattr(launcher, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        launcher,
        "_parse_args",
        lambda: launcher.argparse.Namespace(
            run_name="ordering",
            training_variant="udlm",
            gpu_count=1,
            max_steps=1,
            global_batch_size=2,
            micro_batch_size=2,
            num_workers=0,
            seed=1,
            checkpoint=tmp_path / "unused.ckpt",
            scratch=True,
            exclude_special_tokens=False,
            max_utilization_percent=10,
            min_free_memory_mib=30_000,
            dry_run=False,
        ),
    )
    monkeypatch.setattr(
        launcher, "_python_executable", lambda: tmp_path / ".venv/bin/python"
    )
    events = []

    def pushed_commit():
        events.append("source_check")
        return "a" * 40

    gpu = _gpu(index=3, uuid="GPU-idle")
    monkeypatch.setattr(launcher, "require_pushed_commit", pushed_commit)
    monkeypatch.setattr(
        launcher,
        "probe_all_gpus",
        lambda: events.append("inventory") or [gpu],
    )
    monkeypatch.setattr(
        launcher,
        "reprobe_selected_gpus",
        lambda *_args, **_kwargs: events.append("final_uuid_probe") or (gpu,),
    )
    monkeypatch.setattr(
        launcher,
        "compose_resolved_training_config",
        lambda **_kwargs: (
            {
                "seed": 1,
                "training": {"udlm": {"prior_variant": "release_uniform"}},
                "callback": {"dirpath": str(repository_root / "checkpoints")},
            },
            "b" * 64,
        ),
    )

    def subprocess_run(command, **_kwargs):
        if command[:2] == ["tmux", "has-session"]:
            events.append("tmux_preflight")
            return subprocess.CompletedProcess(command, 1)
        if command[:2] == ["tmux", "new-session"]:
            events.append("tmux_spawn")
            assert command[-2] == "-lc"
            assert "write_pilot_exit_status.py" in command[-1]
            assert command[-1].count("--training-exit-status") == 1
            assert "--expected-initialization-checkpoint-sha256" not in command[-1]
            manifest_path = (
                repository_root / "output/udlm/ordering/launch_manifest.json"
            )
            assert manifest_path.is_file()
            manifest = launcher.json.loads(manifest_path.read_text(encoding="utf-8"))
            summary_path = (
                repository_root / "output/udlm/ordering/training_summary.json"
            )
            checkpoint_path = (
                repository_root / "output/udlm/ordering/checkpoints/1.ckpt"
            )
            assert manifest["training_summary_path"] == str(summary_path)
            assert manifest["training_summary_schema_version"] == 4
            receipt_path = (
                repository_root / "output/udlm/ordering/pilot_exit_status.json"
            )
            assert manifest["pilot_exit_status_path"] == str(receipt_path)
            assert manifest["pilot_exit_status_schema_version"] == 4
            assert manifest["expected_final_checkpoint_path"] == str(checkpoint_path)
            assert manifest["completion_contract"] == {
                "status_at_launch": "pending",
                "complete_only_if_valid_training_summary_exists": True,
                "complete_only_if_successful_exit_receipt_exists": True,
                "valid_training_summary_and_successful_exit_receipt_both_required": (
                    True
                ),
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
            }
            assert manifest["launch_manifest_schema_version"] == 1
            assert manifest["cuda_visible_device_uuids"] == ["GPU-idle"]
            assert manifest["matched_panel_spec"]["execution"]["mode"] == (
                "single_job_lease_with_registered_order_policy"
            )
            manifest_sha256 = launcher.hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest()
            assert manifest_sha256 in command[-1]
            assert str(manifest_path) in command[-1]
            assert '["GPU-idle"]' in command[-1]
            lock = manifest["single_training_job_lock"]
            assert lock["acquired_before_any_gpu_probe"] is True
            assert lock["sha256"] in command[-1]
            assert lock["path"] in command[-1]
            return subprocess.CompletedProcess(command, 0)
        raise AssertionError(f"unexpected subprocess: {command}")

    monkeypatch.setattr(subprocess, "run", subprocess_run)

    launcher.main()

    assert events == [
        "source_check",
        "tmux_preflight",
        "inventory",
        "source_check",
        "final_uuid_probe",
        "tmux_spawn",
    ]

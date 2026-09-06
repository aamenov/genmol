import subprocess
import threading
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from scripts.udlm import launch_train_pilot as launcher


def _expected_floor_audit_binding():
    return {
        "relative_path": launcher.PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_PATH,
        "sha256": launcher.PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_SHA256,
        "source_revision": launcher.PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_SOURCE_REVISION,
        "scope": launcher.PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_SCOPE,
    }


def _mock_verified_floor_audit(monkeypatch):
    monkeypatch.setattr(
        launcher,
        "verify_pilot_empirical_uniform_mix_audit",
        lambda: _expected_floor_audit_binding(),
    )


def test_pilot_floor_is_an_override_and_manual_defaults_remain_historical():
    for relative_path in ("configs/base.yaml", "configs/udlm_categorical.yaml"):
        config = OmegaConf.load(launcher.REPOSITORY_ROOT / relative_path)
        assert config.training.udlm.empirical_uniform_mix == 0.01

    assert launcher.PILOT_EMPIRICAL_UNIFORM_MIX == 0.0002
    assert len(launcher.PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_SHA256) == 64
    assert len(launcher.PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_CANONICAL_SHA256) == 64
    assert len(launcher.PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_SOURCE_REVISION) == 40


def test_pinned_floor_audit_raw_and_canonical_hashes_match_actual_artifact():
    path = launcher.REPOSITORY_ROOT / launcher.PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_PATH
    payload = path.read_bytes()
    audit = launcher.strict_json_loads(payload, label="actual prior-floor audit")

    assert launcher.hashlib.sha256(payload).hexdigest() == (
        launcher.PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_SHA256
    )
    assert launcher.canonical_json_sha256(audit) == (
        launcher.PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_CANONICAL_SHA256
    )
    assert launcher.verify_pilot_empirical_uniform_mix_audit() == (
        _expected_floor_audit_binding()
    )


def test_checkpoint_snapshot_streams_hash_without_retaining_payload(
    monkeypatch, tmp_path
):
    repository_root = tmp_path / "worktree"
    repository_root.mkdir()
    checkpoint_path = repository_root / "output/udlm/R/checkpoints/10.ckpt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_bytes = b"checkpoint-block\n" * 131_073
    checkpoint_path.write_bytes(checkpoint_bytes)
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository_root)

    snapshot, payload = launcher.stable_repository_artifact_snapshot(
        checkpoint_path,
        suffix=".ckpt",
        label="streamed checkpoint",
        capture_bytes=False,
    )

    assert payload == b""
    assert snapshot["size_bytes"] == len(checkpoint_bytes)
    assert snapshot["sha256"] == launcher.hashlib.sha256(checkpoint_bytes).hexdigest()


@pytest.mark.parametrize(
    ("artifact_state", "expected_error", "error_fragment"),
    [
        ("missing", FileNotFoundError, None),
        ("tampered", ValueError, "raw SHA-256"),
    ],
)
def test_main_fails_closed_on_missing_or_tampered_floor_audit_before_source_or_gpu(
    monkeypatch, tmp_path, artifact_state, expected_error, error_fragment
):
    actual_payload = (
        launcher.REPOSITORY_ROOT / launcher.PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_PATH
    ).read_bytes()
    repository_root = tmp_path / "worktree"
    repository_root.mkdir()
    if artifact_state == "tampered":
        audit_path = repository_root / launcher.PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_PATH
        audit_path.parent.mkdir(parents=True)
        audit_path.write_bytes(actual_payload + b" ")
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository_root)
    monkeypatch.setattr(launcher, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        launcher,
        "_parse_args",
        lambda: launcher.argparse.Namespace(
            run_name="floor_audit_blocked",
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
            dry_run=True,
            genesis=True,
            predecessor_receipt=None,
        ),
    )
    monkeypatch.setattr(
        launcher,
        "require_pushed_commit",
        lambda: pytest.fail("floor audit must fail before source verification"),
    )
    monkeypatch.setattr(
        launcher,
        "probe_all_gpus",
        lambda: pytest.fail("floor audit must fail before a GPU query"),
    )

    context = (
        pytest.raises(expected_error)
        if error_fragment is None
        else pytest.raises(expected_error, match=error_fragment)
    )
    with context:
        launcher.main()

    assert not (repository_root / "output").exists()


def test_floor_audit_rejects_symlink_and_canonical_tamper(monkeypatch, tmp_path):
    actual_path = (
        launcher.REPOSITORY_ROOT / launcher.PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_PATH
    )
    actual_payload = actual_path.read_bytes()
    repository_root = tmp_path / "worktree"
    audit_path = repository_root / launcher.PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_PATH
    audit_path.parent.mkdir(parents=True)
    audit_path.symlink_to(actual_path)
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository_root)

    with pytest.raises(ValueError, match="single-link regular file"):
        launcher.verify_pilot_empirical_uniform_mix_audit()

    audit_path.unlink()
    audit = launcher.strict_json_loads(actual_payload, label="actual prior-floor audit")
    audit["recommendation"]["recommended_uniform_mixture_weight"] = 0.0003
    tampered = (launcher.json.dumps(audit, sort_keys=True) + "\n").encode("utf-8")
    audit_path.write_bytes(tampered)
    monkeypatch.setattr(
        launcher,
        "PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_SHA256",
        launcher.hashlib.sha256(tampered).hexdigest(),
    )

    with pytest.raises(ValueError, match="canonical SHA-256"):
        launcher.verify_pilot_empirical_uniform_mix_audit()


def test_floor_audit_semantics_reject_wrong_selected_value_even_with_forged_pins(
    monkeypatch, tmp_path
):
    actual_path = (
        launcher.REPOSITORY_ROOT / launcher.PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_PATH
    )
    audit = launcher.strict_json_loads(
        actual_path.read_bytes(), label="actual prior-floor audit"
    )
    audit["recommendation"]["recommended_uniform_mixture_weight"] = 0.0003
    tampered = (launcher.json.dumps(audit, sort_keys=True) + "\n").encode("utf-8")
    repository_root = tmp_path / "worktree"
    audit_path = repository_root / launcher.PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_PATH
    audit_path.parent.mkdir(parents=True)
    audit_path.write_bytes(tampered)
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository_root)
    monkeypatch.setattr(
        launcher,
        "PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_SHA256",
        launcher.hashlib.sha256(tampered).hexdigest(),
    )
    monkeypatch.setattr(
        launcher,
        "PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_CANONICAL_SHA256",
        launcher.canonical_json_sha256(audit),
    )

    with pytest.raises(ValueError, match="recommended_uniform_mixture_weight"):
        launcher.verify_pilot_empirical_uniform_mix_audit()


@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.parametrize("target_location", ["inside", "outside"])
@pytest.mark.parametrize("ancestor", ["output", "udlm", "logs"])
def test_main_rejects_symlinked_output_parent_before_source_or_gpu(
    monkeypatch, tmp_path, dry_run, target_location, ancestor
):
    _mock_verified_floor_audit(monkeypatch)
    repository_root = tmp_path / "worktree"
    repository_root.mkdir()
    target = (
        repository_root / "real_output"
        if target_location == "inside"
        else tmp_path / "outside_output"
    )
    target.mkdir()
    if ancestor == "output":
        link = repository_root / "output"
    else:
        (repository_root / "output").mkdir()
        if ancestor == "udlm":
            link = repository_root / "output/udlm"
        else:
            (repository_root / "output/udlm").mkdir()
            link = repository_root / "output/logs"
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository_root)
    monkeypatch.setattr(launcher, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        launcher,
        "_parse_args",
        lambda: launcher.argparse.Namespace(
            run_name="symlink_blocked",
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
            dry_run=dry_run,
            genesis=True,
            predecessor_receipt=None,
        ),
    )
    monkeypatch.setattr(
        launcher,
        "require_pushed_commit",
        lambda: pytest.fail("output path must fail before source verification"),
    )
    monkeypatch.setattr(
        launcher,
        "probe_all_gpus",
        lambda: pytest.fail("output path must fail before a GPU query"),
    )

    with pytest.raises(ValueError, match="output ancestor"):
        launcher.main()

    assert list(target.iterdir()) == []


def _gpu(
    *,
    index: int,
    uuid: str,
    memory_used_mib: int = 1_000,
    utilization_percent: int = 2,
    compute_mode: str = "Default",
    processes: tuple[dict[str, object], ...] = (),
) -> launcher.GPUState:
    return launcher.GPUState(
        physical_index=index,
        uuid=uuid,
        name="Example",
        memory_used_mib=memory_used_mib,
        memory_total_mib=48_000,
        utilization_percent=utilization_percent,
        compute_mode=compute_mode,
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
        [
            "--run-name",
            "count_only",
            "--gpu-count",
            "2",
            "--scratch",
            "--genesis",
        ]
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
                "--genesis",
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
                "--genesis",
            ]
        )


def test_gpu_with_compute_process_below_utilization_threshold_is_eligible():
    state = _gpu(
        index=2,
        uuid="GPU-example",
        utilization_percent=9,
        processes=({"pid": 123, "process_name": "python", "used_memory_mib": 900},),
    )

    reasons = state.rejection_reasons(
        max_utilization_percent=10,
        min_free_memory_mib=30_000,
    )

    assert launcher.ACTIVE_COMPUTE_PROCESSES_ALLOWED is True
    assert reasons == []
    assert launcher.select_idle_gpus(
        [state],
        gpu_count=1,
        max_utilization_percent=10,
        min_free_memory_mib=30_000,
    ) == (state,)
    assert state.compute_processes[0]["pid"] == 123


@pytest.mark.parametrize(
    ("state", "reason_fragment"),
    [
        (
            _gpu(index=2, uuid="GPU-at-threshold", utilization_percent=10),
            "utilization 10% is not below 10%",
        ),
        (
            _gpu(index=2, uuid="GPU-low-free", memory_used_mib=18_001),
            "free memory 29999 MiB is below 30000 MiB",
        ),
        (
            _gpu(index=2, uuid="GPU-prohibited", compute_mode="Prohibited"),
            "compute mode is prohibited",
        ),
    ],
)
def test_gpu_eligibility_retains_strict_utilization_memory_and_mode_guards(
    state, reason_fragment
):
    assert reason_fragment in state.rejection_reasons(
        max_utilization_percent=10,
        min_free_memory_mib=30_000,
    )


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
        lambda uuid: _gpu(
            index=2,
            uuid=uuid,
            utilization_percent=9,
            processes=({"pid": 99},),
        ),
    )
    shared = launcher.reprobe_selected_gpus(
        (initial[0],),
        max_utilization_percent=10,
        min_free_memory_mib=30_000,
    )
    assert shared[0].compute_processes == ({"pid": 99},)

    monkeypatch.setattr(
        launcher,
        "probe_gpu_uuid",
        lambda uuid: _gpu(index=2, uuid=uuid, utilization_percent=10),
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
    assert launcher.PILOT_EMPIRICAL_UNIFORM_MIX_OVERRIDE in command
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
    assert (
        config["training"]["udlm"]["empirical_uniform_mix"]
        == launcher.PILOT_EMPIRICAL_UNIFORM_MIX
    )
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
    assert command.count(launcher.PILOT_EMPIRICAL_UNIFORM_MIX_OVERRIDE) == 1
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
            "mode": (
                "single_job_lease_with_machine_enforced_predecessor_receipt_chain"
            ),
            "maximum_concurrent_training_jobs": 1,
            "concurrency_enforcement": "atomic_global_worktree_training_job_lock",
            "registered_variant_order": list(launcher.MATCHED_PANEL_VARIANT_ORDER),
            "advance_policy": (
                "launcher_validates_and_binds_exact_successful_predecessor_receipt"
            ),
            "predecessor_receipt_bound_in_each_manifest": True,
            "genesis_requires_explicit_declaration": True,
            "successor_launch_requires_exact_predecessor_receipt": True,
        }
        assert spec["common_training_contract"]["empirical_uniform_mix"] == 0.0002
        assert (
            spec["common_training_contract"]["empirical_uniform_mix_consumed_only_by"]
            == "empirical_frequency"
        )
        assert spec["common_training_contract"]["empirical_uniform_mix_audit"] == {
            "relative_path": launcher.PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_PATH,
            "sha256": launcher.PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_SHA256,
            "source_revision": (
                launcher.PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_SOURCE_REVISION
            ),
            "scope": "retrospective_training_only_engineering_selection",
        }
        assert spec["common_gpu_safety_policy"] == {
            "max_utilization_percent": 10,
            "utilization_comparison": "strictly_less_than",
            "min_free_memory_mib": 30_000,
            "active_compute_processes_allowed": True,
            "compute_mode_prohibited_allowed": False,
            "physical_gpu_identity_is_per_run_provenance": True,
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


def _predecessor_test_config(
    repository_root, run_name, training_variant, *, seed=7, gpu_count=1
):
    return {
        "seed": seed,
        "data": "safe",
        "training": {
            "ema": 0.9999,
            "pilot_fail_on_nonfinite_loss": True,
            "reseed_after_model_initialization": False,
            "udlm": {
                "prior_variant": launcher.TRAINING_VARIANTS[training_variant][
                    "prior_variant"
                ],
                "conditioning_variant": "additive",
            },
        },
        "trainer": {
            "devices": gpu_count,
            "num_nodes": 1,
            "max_steps": 10,
            "accumulate_grad_batches": 16 // (2 * gpu_count),
            "detect_anomaly": True,
        },
        "loader": {
            "global_batch_size": 16,
            "batch_size": 2,
            "num_workers": 1,
        },
        "callback": {
            "dirpath": str(repository_root / f"output/udlm/{run_name}/checkpoints"),
            "filename": "{step}",
            "every_n_train_steps": 10,
            "save_top_k": -1,
        },
    }


def _predecessor_test_panel(
    repository_root,
    *,
    seed=7,
    gpu_count=1,
    checkpoint=None,
    checkpoint_sha256=None,
):
    resolved = _predecessor_test_config(
        repository_root,
        "panel_template",
        "udlm",
        seed=seed,
        gpu_count=gpu_count,
    )
    common_digest = launcher.matched_panel_config_sha256(resolved)
    return launcher.build_matched_panel_spec(
        source_revision="a" * 40,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        gpu_count=gpu_count,
        max_steps=10,
        global_batch_size=16,
        micro_batch_size=2,
        num_workers=1,
        seed=seed,
        exclude_special_tokens=False,
        max_utilization_percent=10,
        min_free_memory_mib=30_000,
        common_resolved_config_sha256=common_digest,
    )


def _write_successful_predecessor(
    repository_root,
    *,
    run_name,
    training_variant,
    panel_spec,
    panel_sha256,
    predecessor_binding,
    python_executable_path=None,
    lock_acquired=None,
    manifest_created="2026-09-06T12:00:00+00:00",
    summary_completed="2026-09-06T12:01:00+00:00",
    receipt_recorded="2026-09-06T12:02:00+00:00",
):
    if lock_acquired is None:
        lock_acquired = manifest_created
    run_dir = repository_root / f"output/udlm/{run_name}"
    run_dir.mkdir(parents=True)
    receipt_path = run_dir / "pilot_exit_status.json"
    manifest_path = run_dir / "launch_manifest.json"
    runtime_path = run_dir / "runtime_config.json"
    summary_path = run_dir / "training_summary.json"
    checkpoint_path = run_dir / "checkpoints/10.ckpt"
    lock_path = repository_root / "output/udlm/.single_training_job.lock"
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_bytes(b"completed predecessor checkpoint\n")
    definition = launcher.TRAINING_VARIANTS[training_variant]
    position = launcher.MATCHED_PANEL_VARIANT_ORDER.index(training_variant)
    common = panel_spec["common_training_contract"]
    resolved_config = _predecessor_test_config(
        repository_root,
        run_name,
        training_variant,
        seed=common["seed"],
        gpu_count=common["requested_gpu_count"],
    )
    resolved_sha256 = launcher.canonical_json_sha256(resolved_config)
    if python_executable_path is None:
        python_executable_path = repository_root / ".venv/bin/python"
    training_argv = [
        str(python_executable_path),
        "-u",
        str(repository_root / "scripts/train.py"),
        "--config-name",
        str(definition["config_name"]),
        f"seed={common['seed']}",
    ]
    argv_sha256 = launcher.canonical_json_sha256(training_argv[2:])
    selected_uuids = [
        "GPU-predecessor"
        if common["requested_gpu_count"] == 1
        else f"GPU-predecessor-{index}"
        for index in range(common["requested_gpu_count"])
    ]
    gpu_states = [
        {
            "physical_index": 3 + index,
            "uuid": uuid,
            "name": "Fixture GPU",
            "memory_used_mib": 1_000,
            "memory_total_mib": 48_000,
            "utilization_percent": 2,
            "compute_mode": "Default",
            "compute_processes": [
                {
                    "pid": 7_000 + index,
                    "process_name": "/other/user/python",
                    "used_memory_mib": 256,
                }
            ],
        }
        for index, uuid in enumerate(selected_uuids)
    ]
    lock_record = {
        "schema_version": launcher.TRAINING_JOB_LOCK_SCHEMA_VERSION,
        "status": "held",
        "purpose": "enforce_one_R_S_E_pilot_training_job_at_a_time",
        "source_revision": common["source_revision"],
        "run_name": run_name,
        "training_variant": training_variant,
        "owner_token": "1" * 64,
        "launcher_pid_at_acquisition": 123,
        "acquired_at_utc": lock_acquired,
        "owner_process_exit_does_not_make_lock_stale": True,
        "stale_lock_policy": "fail_closed_and_require_manual_review",
        "release_policy": (
            "exact_owner_lock_only_after_receipt_or_before_tmux_handoff_failure"
        ),
    }
    lock_path.write_text(
        launcher.json.dumps(lock_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lock_snapshot, _payload = launcher.stable_repository_artifact_snapshot(
        lock_path,
        suffix=".lock",
        label="test predecessor lock",
    )
    summary_completion = {
        "summary_schema_version": launcher.TRAINING_SUMMARY_SCHEMA_VERSION,
        "summary_path": str(summary_path),
        "final_checkpoint_path": str(checkpoint_path),
        "expected_max_steps": common["max_steps"],
        "expected_world_size": common["requested_gpu_count"],
        "fail_on_nonfinite_loss": True,
        "backward_anomaly_detection": True,
    }
    manifest = {
        "launch_manifest_schema_version": launcher.LAUNCH_MANIFEST_SCHEMA_VERSION,
        "created_at": manifest_created,
        "purpose": "bounded UDLM training pilot",
        "gpu_selection_schema_version": 2,
        "git_sha": common["source_revision"],
        "source_revision_before_final_gpu_probe": common["source_revision"],
        "run_name": run_name,
        "training_variant": training_variant,
        "hydra_config_name": definition["config_name"],
        "udlm_prior_variant": definition["prior_variant"],
        "udlm_comparison_role": definition["comparison_role"],
        "matched_panel_spec": panel_spec,
        "matched_panel_spec_sha256": panel_sha256,
        "matched_panel_variant_position": position,
        "predecessor_receipt_binding": predecessor_binding,
        "single_training_job_lock": {
            "path": str(lock_path),
            "sha256": lock_snapshot["sha256"],
            "record": lock_record,
            "acquired_before_any_gpu_probe": True,
            "stale_lock_policy": "fail_closed_and_require_manual_review",
            "release_owner": "pilot_exit_receipt_writer_after_publication",
        },
        "tmux_session": f"genmol_{training_variant}_{run_name}",
        "user_requested_gpu_count": common["requested_gpu_count"],
        "gpu_selection_method": "dynamic_idle_discovery",
        "gpu_inventory_scope": "all_nvidia_gpus",
        "inventory_snapshot_completed_at_utc": manifest_created,
        "gpu_inventory_at_selection": gpu_states,
        "initially_selected_gpu_states": gpu_states,
        "logical_cuda_devices": list(range(len(gpu_states))),
        "physical_gpu_indices": [state["physical_index"] for state in gpu_states],
        "cuda_visible_device_uuids": selected_uuids,
        "final_uuid_probes_completed_at_utc": manifest_created,
        "gpu_states_at_final_uuid_probe": gpu_states,
        "gpu_safety_policy": {
            "max_utilization_percent": 10,
            "utilization_comparison": "strictly_less_than",
            "min_free_memory_mib": 30_000,
            "active_compute_processes_allowed": (
                launcher.ACTIVE_COMPUTE_PROCESSES_ALLOWED
            ),
            "compute_mode_prohibited_allowed": False,
        },
        "training_argv": training_argv,
        "training_argv_sha256": argv_sha256,
        "resolved_training_config": resolved_config,
        "resolved_training_config_sha256": resolved_sha256,
        "runtime_config_path": str(runtime_path),
        "training_summary_path": str(summary_path),
        "training_summary_schema_version": launcher.TRAINING_SUMMARY_SCHEMA_VERSION,
        "pilot_exit_status_path": str(receipt_path),
        "pilot_exit_status_schema_version": (launcher.PILOT_EXIT_STATUS_SCHEMA_VERSION),
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
        "log_path": str(repository_root / f"output/logs/{run_name}.log"),
        "log_reserved_exclusively_before_manifest": True,
        "checkpoint": common["initialization_checkpoint_path"],
        "checkpoint_sha256": common["initialization_checkpoint_sha256"],
        "seed": common["seed"],
        "max_steps": common["max_steps"],
        "global_batch_size": common["global_batch_size"],
        "micro_batch_size_per_process": common["micro_batch_size_per_process"],
        "accumulate_grad_batches": common["accumulate_grad_batches"],
        "effective_global_batch_size": common["effective_global_batch_size"],
        "exclude_special_tokens": common["exclude_special_tokens"],
        "dry_run": False,
    }
    manifest_path.write_text(
        launcher.json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_snapshot, _payload = launcher.stable_repository_artifact_snapshot(
        manifest_path,
        suffix=".json",
        label="test predecessor manifest",
    )
    manifest_claim = {**manifest_snapshot, "selected_gpu_uuids": selected_uuids}
    runtime_record = {
        "schema_version": 2,
        "status": "preflight_completed",
        "source_revision": common["source_revision"],
        "source": {
            "head": common["source_revision"],
            "upstream": common["source_revision"],
        },
        "training_argv": training_argv[2:],
        "observed_training_argv": training_argv[2:],
        "training_argv_sha256": argv_sha256,
        "resolved_training_config": resolved_config,
        "resolved_training_config_sha256": resolved_sha256,
        "launch_manifest": manifest_claim,
        "completion_contract": summary_completion,
        "python_environment": {
            **launcher.CONTROLLED_PYTHON_ENVIRONMENT,
            "PYTHONHASHSEED": str(common["seed"]),
            "PYTHONPATH": launcher.os.pathsep.join(
                (str(repository_root / "src"), str(repository_root))
            ),
        },
    }
    runtime_path.write_text(
        launcher.json.dumps(runtime_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    runtime_snapshot, _payload = launcher.stable_repository_artifact_snapshot(
        runtime_path,
        suffix=".json",
        label="test predecessor runtime config",
    )
    checkpoint_snapshot, _payload = launcher.stable_repository_artifact_snapshot(
        checkpoint_path,
        suffix=".ckpt",
        label="test predecessor checkpoint",
        capture_bytes=False,
    )
    finiteness = {
        "all_finite": True,
        "floating_tensor_count": 2,
        "floating_element_count": 10,
    }
    ema_metadata = {
        "shadow_parameter_count": 2,
        "decay": 0.9999,
        "num_updates": common["max_steps"],
    }
    startup_mode = (
        "scratch"
        if common["initialization_checkpoint_sha256"] is None
        else "warm_start"
    )
    warm_start_report = None
    if startup_mode == "warm_start":
        warm_start_report = {
            "source_path": common["initialization_checkpoint_path"],
            "source_resolved_path": common["initialization_checkpoint_path"],
            "source_sha256": common["initialization_checkpoint_sha256"],
            "source_size_bytes": 123,
            "expected_source_sha256": common["initialization_checkpoint_sha256"],
            "byte_identity_verified_before_and_after_load": True,
            "weights": "ema",
            "parameter_tensors": 2,
        }
    training_accounting = {
        "training_seed": common["seed"],
        "optimizer_updates": common["max_steps"],
        "world_size": common["requested_gpu_count"],
        "micro_batch_size_per_rank": common["micro_batch_size_per_process"],
        "accumulate_grad_batches": common["accumulate_grad_batches"],
        "effective_global_examples_per_optimizer_step": common[
            "effective_global_batch_size"
        ],
        "total_requested_example_exposures": (
            common["effective_global_batch_size"] * common["max_steps"]
        ),
        "hosted_stream_rank_partition_policy": (
            launcher._HOSTED_STREAM_RANK_PARTITION_POLICY
        ),
        "trainable_parameter_counts": {
            "base_backbone": 3,
            "time_conditioner": 2,
            "total": 5,
        },
    }
    summary = {
        "schema_version": launcher.TRAINING_SUMMARY_SCHEMA_VERSION,
        "status": "completed",
        "completed_at_utc": summary_completed,
        "source_revision": common["source_revision"],
        "source": runtime_record["source"],
        "resolved_training_config_sha256": resolved_sha256,
        "training_argv_sha256": argv_sha256,
        "launch_manifest": manifest_claim,
        "runtime_config": {
            **runtime_snapshot,
            "schema_version": 2,
            "record_sha256": launcher.canonical_json_sha256(runtime_record),
        },
        "completion_contract": summary_completion,
        "observed_training_state": {
            "global_rank": 0,
            "global_step": common["max_steps"],
            "world_size": common["requested_gpu_count"],
        },
        "training_accounting": training_accounting,
        "training_health": {
            "scope": (
                "global-rank-zero callback counters; identical fail-fast checks "
                "execute independently on every rank"
            ),
            "all_losses_finite": True,
            "all_observed_gradients_finite": True,
            "every_optimizer_step_had_a_nonzero_gradient": True,
            "loss_checks": common["max_steps"],
            "optimizer_step_checks": common["max_steps"],
            "gradient_tensor_observations": common["max_steps"],
            "gradient_element_observations": common["max_steps"] * 2,
        },
        "conditioning_gradient_audit": None,
        "screen_initialization_state_audit": None,
        "final_checkpoint": {
            **checkpoint_snapshot,
            "semantic_audit": {
                "deserialized": True,
                "global_step": common["max_steps"],
                "raw_model": finiteness,
                "ema": finiteness,
                "ema_metadata": ema_metadata,
                "optimizer": finiteness,
                "all_checkpoint_tensors": finiteness,
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
        "tensor_finiteness": {"raw_model": finiteness, "ema": finiteness},
        "startup": {
            "mode": startup_mode,
            "verified_mdlm_warm_start_report": warm_start_report,
        },
    }
    summary_path.write_text(
        launcher.json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_snapshot, _payload = launcher.stable_repository_artifact_snapshot(
        summary_path,
        suffix=".json",
        label="test predecessor summary",
    )
    completion_requirements = {
        "training_exit_zero": True,
        "tee_exit_zero": True,
        "training_summary_valid_and_launch_bound": True,
        "launch_manifest_matches_summary_runtime_and_launch": True,
        "training_job_lock_valid_before_receipt_publication": True,
        "runtime_config_matches_summary_and_launch": True,
        "final_checkpoint_matches_training_summary": True,
        "clean_pushed_source_still_matches_launch": True,
        "predecessor_receipt_binding_unchanged_and_valid": True,
        "all_must_hold": True,
    }
    validated_bindings = {
        "schema_version": launcher.TRAINING_SUMMARY_SCHEMA_VERSION,
        "source_revision": common["source_revision"],
        "resolved_training_config_sha256": resolved_sha256,
        "training_argv_sha256": argv_sha256,
        "launch_manifest_path": str(manifest_path),
        "launch_manifest_sha256": manifest_snapshot["sha256"],
        "selected_gpu_uuids": selected_uuids,
        "observed_global_step": common["max_steps"],
        "observed_world_size": common["requested_gpu_count"],
        "training_accounting": training_accounting,
        "ema_metadata": ema_metadata,
        "final_checkpoint_path": str(checkpoint_path),
        "final_checkpoint_sha256": checkpoint_snapshot["sha256"],
        "startup_mode": startup_mode,
        "conditioning_gradient_audit": None,
        "screen_initialization_state_audit": None,
    }
    receipt = {
        "schema_version": launcher.PILOT_EXIT_STATUS_SCHEMA_VERSION,
        "status": "completed",
        "overall_status": "completed",
        "recorded_at_utc": receipt_recorded,
        "process_exit_status": 0,
        "expected_contract": {
            "training_summary_schema_version": (
                launcher.TRAINING_SUMMARY_SCHEMA_VERSION
            ),
            "source_revision": common["source_revision"],
            "resolved_training_config_sha256": resolved_sha256,
            "training_argv_sha256": argv_sha256,
            "launch_manifest_path": str(manifest_path),
            "launch_manifest_sha256": manifest_snapshot["sha256"],
            "selected_gpu_uuids": selected_uuids,
            "training_job_lock_path": str(
                repository_root / "output/udlm/.single_training_job.lock"
            ),
            "training_job_lock_sha256": lock_snapshot["sha256"],
            "max_steps": common["max_steps"],
            "world_size": common["requested_gpu_count"],
            "training_summary_path": str(summary_path),
            "final_checkpoint_path": str(checkpoint_path),
            "initialization_checkpoint_sha256": common[
                "initialization_checkpoint_sha256"
            ],
        },
        "pipeline": {
            "training": {
                "shell_exit_status": 0,
                "succeeded": True,
                "possible_termination_signal": None,
                "shell_status_is_signal_compatible": False,
                "signal_provenance": None,
            },
            "tee": {
                "shell_exit_status": 0,
                "succeeded": True,
                "possible_termination_signal": None,
                "shell_status_is_signal_compatible": False,
                "signal_provenance": None,
            },
            "pipefail_shell_exit_status": 0,
        },
        "source_at_receipt": {
            "verified": True,
            "expected_revision": common["source_revision"],
            "head": common["source_revision"],
            "upstream": common["source_revision"],
            "output_directory_excluded_from_cleanliness_check": True,
        },
        "launch_manifest": {
            "path": str(manifest_path),
            "present": True,
            "artifact": manifest_snapshot,
            "matches_expected_raw_sha256": True,
            "selected_gpu_uuids_match_expected": True,
            "matches_training_summary_snapshot": True,
            "matches_runtime_config_snapshot": True,
            "valid_and_launch_bound": True,
            "expected_selected_gpu_uuids": selected_uuids,
            "observed_selected_gpu_uuids": selected_uuids,
            "validation_error": None,
        },
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
            "artifact": checkpoint_snapshot,
            "matches_training_summary_snapshot": True,
        },
        "predecessor_receipt_binding": predecessor_binding,
        "completion_requirements": completion_requirements,
    }
    receipt_path.write_text(
        launcher.json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lock_path.unlink()
    return receipt_path


def _rebind_mutated_predecessor_manifest(receipt_path, manifest):
    manifest_path = receipt_path.parent / "launch_manifest.json"
    manifest_path.write_text(
        launcher.json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest_snapshot, _payload = launcher.stable_repository_artifact_snapshot(
        manifest_path,
        suffix=".json",
        label="mutated predecessor manifest",
    )
    receipt = launcher.json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["expected_contract"]["launch_manifest_sha256"] = manifest_snapshot["sha256"]
    receipt["launch_manifest"]["artifact"] = manifest_snapshot
    return receipt


def _rebind_mutated_predecessor_summary(receipt_path, summary):
    summary_path = receipt_path.parent / "training_summary.json"
    summary_path.write_text(
        launcher.json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary_snapshot, _payload = launcher.stable_repository_artifact_snapshot(
        summary_path,
        suffix=".json",
        label="mutated predecessor summary",
    )
    receipt = launcher.json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["training_summary"]["artifact"] = summary_snapshot
    return receipt


def test_cli_requires_exactly_one_explicit_predecessor_state():
    common = ["--run-name", "chain", "--gpu-count", "1", "--scratch"]
    with pytest.raises(SystemExit):
        launcher._parse_args(common)
    with pytest.raises(SystemExit):
        launcher._parse_args(
            [
                *common,
                "--genesis",
                "--predecessor-receipt",
                "output/udlm/R/pilot_exit_status.json",
            ]
        )
    parsed = launcher._parse_args([*common, "--genesis"])
    assert parsed.genesis is True
    assert parsed.predecessor_receipt is None


def test_r_requires_and_records_explicit_genesis(monkeypatch, tmp_path):
    repository_root = tmp_path / "worktree"
    repository_root.mkdir()
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository_root)
    panel, panel_sha256 = _predecessor_test_panel(repository_root)

    binding = launcher.build_predecessor_receipt_binding(
        training_variant="udlm",
        explicit_genesis=True,
        predecessor_receipt_path=None,
        matched_panel_spec=panel,
        matched_panel_spec_sha256=panel_sha256,
    )

    assert binding == {
        "schema_version": 1,
        "state": "explicit_genesis_no_predecessor",
        "current_training_variant": "udlm",
        "current_variant_position": 0,
        "expected_predecessor_training_variant": None,
        "expected_predecessor_variant_position": None,
        "matched_panel_spec_sha256": panel_sha256,
        "common_training_contract_sha256": launcher.canonical_json_sha256(
            panel["common_training_contract"]
        ),
        "receipt_artifact": None,
        "predecessor_launch_manifest_artifact": None,
        "predecessor_training_summary_artifact": None,
        "predecessor_run_name": None,
        "chronology": None,
        "validated_before_gpu_probe": True,
    }
    with pytest.raises(ValueError, match="requires --genesis"):
        launcher.build_predecessor_receipt_binding(
            training_variant="udlm",
            explicit_genesis=False,
            predecessor_receipt_path=None,
            matched_panel_spec=panel,
            matched_panel_spec_sha256=panel_sha256,
        )
    with pytest.raises(ValueError, match="cannot declare genesis"):
        launcher.build_predecessor_receipt_binding(
            training_variant="schedule_uniform",
            explicit_genesis=True,
            predecessor_receipt_path=None,
            matched_panel_spec=panel,
            matched_panel_spec_sha256=panel_sha256,
        )


def test_s_and_e_require_exact_successful_immediate_predecessors(monkeypatch, tmp_path):
    repository_root = tmp_path / "worktree"
    repository_root.mkdir()
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository_root)
    panel, panel_sha256 = _predecessor_test_panel(repository_root)
    genesis = launcher.build_predecessor_receipt_binding(
        training_variant="udlm",
        explicit_genesis=True,
        predecessor_receipt_path=None,
        matched_panel_spec=panel,
        matched_panel_spec_sha256=panel_sha256,
    )
    r_receipt = _write_successful_predecessor(
        repository_root,
        run_name="R",
        training_variant="udlm",
        panel_spec=panel,
        panel_sha256=panel_sha256,
        predecessor_binding=genesis,
    )

    s_binding = launcher.build_predecessor_receipt_binding(
        training_variant="schedule_uniform",
        explicit_genesis=False,
        predecessor_receipt_path=r_receipt,
        matched_panel_spec=panel,
        matched_panel_spec_sha256=panel_sha256,
    )
    assert s_binding["expected_predecessor_training_variant"] == "udlm"
    assert s_binding["receipt_artifact"]["path"] == str(r_receipt)
    assert (
        s_binding["receipt_artifact"]["sha256"]
        == launcher.hashlib.sha256(r_receipt.read_bytes()).hexdigest()
    )
    s_receipt = _write_successful_predecessor(
        repository_root,
        run_name="S",
        training_variant="schedule_uniform",
        panel_spec=panel,
        panel_sha256=panel_sha256,
        predecessor_binding=s_binding,
        manifest_created="2026-09-06T12:03:00+00:00",
        summary_completed="2026-09-06T12:04:00+00:00",
        receipt_recorded="2026-09-06T12:05:00+00:00",
    )
    with pytest.raises(ValueError, match="predecessor training variant"):
        launcher.build_predecessor_receipt_binding(
            training_variant="schedule_uniform",
            explicit_genesis=False,
            predecessor_receipt_path=s_receipt,
            matched_panel_spec=panel,
            matched_panel_spec_sha256=panel_sha256,
        )

    e_binding = launcher.build_predecessor_receipt_binding(
        training_variant="udlm_categorical",
        explicit_genesis=False,
        predecessor_receipt_path=s_receipt,
        matched_panel_spec=panel,
        matched_panel_spec_sha256=panel_sha256,
    )

    assert e_binding["expected_predecessor_training_variant"] == "schedule_uniform"
    assert e_binding["expected_predecessor_variant_position"] == 1
    assert e_binding["predecessor_run_name"] == "S"
    assert e_binding["chronology"]["strictly_ordered_timestamps_verified"] is True


def test_successor_has_no_fail_open_missing_receipt(monkeypatch, tmp_path):
    repository_root = tmp_path / "worktree"
    repository_root.mkdir()
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository_root)
    panel, panel_sha256 = _predecessor_test_panel(repository_root)

    for training_variant in ("schedule_uniform", "udlm_categorical"):
        with pytest.raises(ValueError, match="requires the immediately preceding"):
            launcher.build_predecessor_receipt_binding(
                training_variant=training_variant,
                explicit_genesis=False,
                predecessor_receipt_path=None,
                matched_panel_spec=panel,
                matched_panel_spec_sha256=panel_sha256,
            )


def test_main_rejects_missing_successor_receipt_before_gpu_or_tmux(
    monkeypatch, tmp_path
):
    _mock_verified_floor_audit(monkeypatch)
    repository_root = tmp_path / "worktree"
    repository_root.mkdir()
    missing_receipt = repository_root / "output/udlm/R/pilot_exit_status.json"
    missing_receipt.parent.mkdir(parents=True)
    resolved = _predecessor_test_config(
        repository_root, "blocked_S", "schedule_uniform", seed=7
    )
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository_root)
    monkeypatch.setattr(launcher, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        launcher,
        "_parse_args",
        lambda: launcher.argparse.Namespace(
            run_name="blocked_S",
            training_variant="schedule_uniform",
            gpu_count=1,
            max_steps=10,
            global_batch_size=16,
            micro_batch_size=2,
            num_workers=1,
            seed=7,
            checkpoint=tmp_path / "unused.ckpt",
            scratch=True,
            exclude_special_tokens=False,
            max_utilization_percent=10,
            min_free_memory_mib=30_000,
            dry_run=False,
            genesis=False,
            predecessor_receipt=missing_receipt,
        ),
    )
    monkeypatch.setattr(launcher, "_python_executable", lambda: Path("/venv/python"))
    monkeypatch.setattr(launcher, "require_pushed_commit", lambda: "a" * 40)
    monkeypatch.setattr(
        launcher,
        "compose_resolved_training_config",
        lambda **_kwargs: (resolved, launcher.canonical_json_sha256(resolved)),
    )
    monkeypatch.setattr(
        launcher,
        "probe_all_gpus",
        lambda: pytest.fail("missing predecessor must fail before a GPU query"),
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "missing predecessor must fail before a tmux operation"
        ),
    )

    with pytest.raises(FileNotFoundError):
        launcher.main()

    assert not (repository_root / "output/udlm/blocked_S").exists()


def test_predecessor_receipt_must_match_the_same_common_contract(monkeypatch, tmp_path):
    repository_root = tmp_path / "worktree"
    repository_root.mkdir()
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository_root)
    panel, panel_sha256 = _predecessor_test_panel(repository_root, seed=7)
    genesis = launcher.build_predecessor_receipt_binding(
        training_variant="udlm",
        explicit_genesis=True,
        predecessor_receipt_path=None,
        matched_panel_spec=panel,
        matched_panel_spec_sha256=panel_sha256,
    )
    r_receipt = _write_successful_predecessor(
        repository_root,
        run_name="R_seed7",
        training_variant="udlm",
        panel_spec=panel,
        panel_sha256=panel_sha256,
        predecessor_binding=genesis,
    )
    different_panel, different_sha256 = _predecessor_test_panel(repository_root, seed=8)

    with pytest.raises(ValueError, match="different matched-panel"):
        launcher.build_predecessor_receipt_binding(
            training_variant="schedule_uniform",
            explicit_genesis=False,
            predecessor_receipt_path=r_receipt,
            matched_panel_spec=different_panel,
            matched_panel_spec_sha256=different_sha256,
        )


@pytest.mark.parametrize(
    ("field_path", "replacement", "error_fragment"),
    [
        (("schema_version",), 4, "receipt schema"),
        (("status",), "failed", "receipt status"),
        (("process_exit_status",), 1, "process exit status"),
        (
            (
                "completion_requirements",
                "predecessor_receipt_binding_unchanged_and_valid",
            ),
            False,
            "receipt-binding completion requirement",
        ),
        (("pipeline", "training", "succeeded"), False, "training success flag"),
        (("expected_contract", "max_steps"), 9, "expected optimizer steps"),
        (("source_at_receipt", "head"), "b" * 40, "source head"),
        (
            (
                "training_summary",
                "validated_bindings",
                "training_accounting",
                "training_seed",
            ),
            8,
            "accounting training_seed",
        ),
    ],
)
def test_predecessor_receipt_requires_schema5_completed_success(
    monkeypatch, tmp_path, field_path, replacement, error_fragment
):
    repository_root = tmp_path / "worktree"
    repository_root.mkdir()
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository_root)
    panel, panel_sha256 = _predecessor_test_panel(repository_root)
    genesis = launcher.build_predecessor_receipt_binding(
        training_variant="udlm",
        explicit_genesis=True,
        predecessor_receipt_path=None,
        matched_panel_spec=panel,
        matched_panel_spec_sha256=panel_sha256,
    )
    receipt_path = _write_successful_predecessor(
        repository_root,
        run_name="R_tamper",
        training_variant="udlm",
        panel_spec=panel,
        panel_sha256=panel_sha256,
        predecessor_binding=genesis,
    )
    receipt = launcher.json.loads(receipt_path.read_text(encoding="utf-8"))
    parent = receipt
    for key in field_path[:-1]:
        parent = parent[key]
    parent[field_path[-1]] = replacement
    receipt_path.write_text(
        launcher.json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=error_fragment):
        launcher.build_predecessor_receipt_binding(
            training_variant="schedule_uniform",
            explicit_genesis=False,
            predecessor_receipt_path=receipt_path,
            matched_panel_spec=panel,
            matched_panel_spec_sha256=panel_sha256,
        )


@pytest.mark.parametrize(
    ("artifact", "missing_key"),
    [
        ("receipt", "runtime_config"),
        ("manifest", "gpu_inventory_at_selection"),
        ("summary", "training_health"),
        ("runtime", "python_environment"),
    ],
)
def test_predecessor_rejects_missing_producer_schema_fields(
    monkeypatch, tmp_path, artifact, missing_key
):
    repository_root = tmp_path / "worktree"
    repository_root.mkdir()
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository_root)
    panel, panel_sha256 = _predecessor_test_panel(repository_root)
    genesis = launcher.build_predecessor_receipt_binding(
        training_variant="udlm",
        explicit_genesis=True,
        predecessor_receipt_path=None,
        matched_panel_spec=panel,
        matched_panel_spec_sha256=panel_sha256,
    )
    receipt_path = _write_successful_predecessor(
        repository_root,
        run_name=f"R_missing_{artifact}",
        training_variant="udlm",
        panel_spec=panel,
        panel_sha256=panel_sha256,
        predecessor_binding=genesis,
    )
    run_dir = receipt_path.parent
    receipt = launcher.json.loads(receipt_path.read_text(encoding="utf-8"))

    if artifact == "receipt":
        receipt.pop(missing_key)
    elif artifact == "manifest":
        path = run_dir / "launch_manifest.json"
        value = launcher.json.loads(path.read_text(encoding="utf-8"))
        value.pop(missing_key)
        path.write_text(
            launcher.json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        snapshot, _payload = launcher.stable_repository_artifact_snapshot(
            path, suffix=".json", label="missing-field manifest"
        )
        receipt["expected_contract"]["launch_manifest_sha256"] = snapshot["sha256"]
        receipt["launch_manifest"]["artifact"] = snapshot
    elif artifact == "summary":
        path = run_dir / "training_summary.json"
        value = launcher.json.loads(path.read_text(encoding="utf-8"))
        value.pop(missing_key)
        path.write_text(
            launcher.json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        snapshot, _payload = launcher.stable_repository_artifact_snapshot(
            path, suffix=".json", label="missing-field summary"
        )
        receipt["training_summary"]["artifact"] = snapshot
    else:
        runtime_path = run_dir / "runtime_config.json"
        runtime = launcher.json.loads(runtime_path.read_text(encoding="utf-8"))
        runtime.pop(missing_key)
        runtime_path.write_text(
            launcher.json.dumps(runtime, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        runtime_snapshot, _payload = launcher.stable_repository_artifact_snapshot(
            runtime_path, suffix=".json", label="missing-field runtime"
        )
        summary_path = run_dir / "training_summary.json"
        summary = launcher.json.loads(summary_path.read_text(encoding="utf-8"))
        summary["runtime_config"] = {
            **runtime_snapshot,
            "schema_version": 2,
            "record_sha256": launcher.canonical_json_sha256(runtime),
        }
        summary_path.write_text(
            launcher.json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary_snapshot, _payload = launcher.stable_repository_artifact_snapshot(
            summary_path, suffix=".json", label="runtime-linked summary"
        )
        receipt["runtime_config"]["artifact"] = runtime_snapshot
        receipt["training_summary"]["artifact"] = summary_snapshot
    receipt_path.write_text(
        launcher.json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="keys differ"):
        launcher.build_predecessor_receipt_binding(
            training_variant="schedule_uniform",
            explicit_genesis=False,
            predecessor_receipt_path=receipt_path,
            matched_panel_spec=panel,
            matched_panel_spec_sha256=panel_sha256,
        )


def test_predecessor_lock_record_must_hash_to_bound_producer_bytes(
    monkeypatch, tmp_path
):
    repository_root = tmp_path / "worktree"
    repository_root.mkdir()
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository_root)
    panel, panel_sha256 = _predecessor_test_panel(repository_root)
    genesis = launcher.build_predecessor_receipt_binding(
        training_variant="udlm",
        explicit_genesis=True,
        predecessor_receipt_path=None,
        matched_panel_spec=panel,
        matched_panel_spec_sha256=panel_sha256,
    )
    receipt_path = _write_successful_predecessor(
        repository_root,
        run_name="R_bad_lock_record_digest",
        training_variant="udlm",
        panel_spec=panel,
        panel_sha256=panel_sha256,
        predecessor_binding=genesis,
    )
    manifest_path = receipt_path.parent / "launch_manifest.json"
    manifest = launcher.json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["single_training_job_lock"]["record"]["owner_token"] = "2" * 64
    receipt = _rebind_mutated_predecessor_manifest(receipt_path, manifest)
    receipt["training_job_lock"]["record"] = manifest["single_training_job_lock"][
        "record"
    ]
    receipt_path.write_text(
        launcher.json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="lock record digest"):
        launcher.build_predecessor_receipt_binding(
            training_variant="schedule_uniform",
            explicit_genesis=False,
            predecessor_receipt_path=receipt_path,
            matched_panel_spec=panel,
            matched_panel_spec_sha256=panel_sha256,
        )


def test_predecessor_lock_snapshot_requires_single_link(monkeypatch, tmp_path):
    repository_root = tmp_path / "worktree"
    repository_root.mkdir()
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository_root)
    panel, panel_sha256 = _predecessor_test_panel(repository_root)
    genesis = launcher.build_predecessor_receipt_binding(
        training_variant="udlm",
        explicit_genesis=True,
        predecessor_receipt_path=None,
        matched_panel_spec=panel,
        matched_panel_spec_sha256=panel_sha256,
    )
    receipt_path = _write_successful_predecessor(
        repository_root,
        run_name="R_bad_lock_links",
        training_variant="udlm",
        panel_spec=panel,
        panel_sha256=panel_sha256,
        predecessor_binding=genesis,
    )
    receipt = launcher.json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["training_job_lock"]["artifact"]["link_count"] = 2
    receipt_path.write_text(
        launcher.json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="lock snapshot link count"):
        launcher.build_predecessor_receipt_binding(
            training_variant="schedule_uniform",
            explicit_genesis=False,
            predecessor_receipt_path=receipt_path,
            matched_panel_spec=panel,
            matched_panel_spec_sha256=panel_sha256,
        )


def test_predecessor_lock_must_predate_gpu_inventory(monkeypatch, tmp_path):
    repository_root = tmp_path / "worktree"
    repository_root.mkdir()
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository_root)
    panel, panel_sha256 = _predecessor_test_panel(repository_root)
    genesis = launcher.build_predecessor_receipt_binding(
        training_variant="udlm",
        explicit_genesis=True,
        predecessor_receipt_path=None,
        matched_panel_spec=panel,
        matched_panel_spec_sha256=panel_sha256,
    )
    receipt_path = _write_successful_predecessor(
        repository_root,
        run_name="R_late_lock",
        training_variant="udlm",
        panel_spec=panel,
        panel_sha256=panel_sha256,
        predecessor_binding=genesis,
    )
    manifest_path = receipt_path.parent / "launch_manifest.json"
    manifest = launcher.json.loads(manifest_path.read_text(encoding="utf-8"))
    lock_binding = manifest["single_training_job_lock"]
    lock_binding["record"]["acquired_at_utc"] = "2026-09-06T12:00:01+00:00"
    lock_payload = (
        launcher.json.dumps(
            lock_binding["record"], indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")
    lock_sha256 = launcher.hashlib.sha256(lock_payload).hexdigest()
    lock_binding["sha256"] = lock_sha256
    receipt = _rebind_mutated_predecessor_manifest(receipt_path, manifest)
    receipt["expected_contract"]["training_job_lock_sha256"] = lock_sha256
    receipt["training_job_lock"]["expected_sha256"] = lock_sha256
    receipt["training_job_lock"]["artifact"]["sha256"] = lock_sha256
    receipt["training_job_lock"]["record"] = lock_binding["record"]
    receipt_path.write_text(
        launcher.json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="acquired after GPU inventory"):
        launcher.build_predecessor_receipt_binding(
            training_variant="schedule_uniform",
            explicit_genesis=False,
            predecessor_receipt_path=receipt_path,
            matched_panel_spec=panel,
            matched_panel_spec_sha256=panel_sha256,
        )


@pytest.mark.parametrize(
    ("mutation", "gpu_count", "error_fragment"),
    [
        ("memory_over_total", 1, "used memory exceeds total"),
        ("utilization_over_100", 1, "utilization exceeds 100"),
        ("duplicate_inventory_uuid", 1, "inventory UUIDs must be unique"),
        (
            "duplicate_inventory_physical_index",
            1,
            "inventory physical indices must be unique",
        ),
        ("selected_absent_from_inventory", 1, "absent from inventory"),
        ("initial_differs_from_inventory", 1, "differ from inventory rows"),
        ("duplicate_final_physical_index", 2, "final GPU physical indices"),
    ],
)
def test_predecessor_rejects_impossible_gpu_producer_evidence(
    monkeypatch, tmp_path, mutation, gpu_count, error_fragment
):
    repository_root = tmp_path / "worktree"
    repository_root.mkdir()
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository_root)
    panel, panel_sha256 = _predecessor_test_panel(repository_root, gpu_count=gpu_count)
    genesis = launcher.build_predecessor_receipt_binding(
        training_variant="udlm",
        explicit_genesis=True,
        predecessor_receipt_path=None,
        matched_panel_spec=panel,
        matched_panel_spec_sha256=panel_sha256,
    )
    receipt_path = _write_successful_predecessor(
        repository_root,
        run_name=f"R_bad_gpu_{mutation}",
        training_variant="udlm",
        panel_spec=panel,
        panel_sha256=panel_sha256,
        predecessor_binding=genesis,
    )
    manifest_path = receipt_path.parent / "launch_manifest.json"
    manifest = launcher.json.loads(manifest_path.read_text(encoding="utf-8"))
    inventory = manifest["gpu_inventory_at_selection"]
    if mutation == "memory_over_total":
        inventory[0]["memory_used_mib"] = inventory[0]["memory_total_mib"] + 1
    elif mutation == "utilization_over_100":
        inventory[0]["utilization_percent"] = 101
    elif mutation == "duplicate_inventory_uuid":
        duplicate = dict(inventory[0])
        duplicate["physical_index"] += 1
        inventory.append(duplicate)
    elif mutation == "duplicate_inventory_physical_index":
        duplicate = dict(inventory[0])
        duplicate["uuid"] = "GPU-unselected"
        inventory.append(duplicate)
    elif mutation == "selected_absent_from_inventory":
        inventory[0]["uuid"] = "GPU-unselected"
    elif mutation == "initial_differs_from_inventory":
        manifest["initially_selected_gpu_states"][0]["memory_used_mib"] += 1
    else:
        manifest["gpu_states_at_final_uuid_probe"][1]["physical_index"] = manifest[
            "gpu_states_at_final_uuid_probe"
        ][0]["physical_index"]
        manifest["physical_gpu_indices"][1] = manifest["physical_gpu_indices"][0]
    receipt = _rebind_mutated_predecessor_manifest(receipt_path, manifest)
    receipt_path.write_text(
        launcher.json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=error_fragment):
        launcher.build_predecessor_receipt_binding(
            training_variant="schedule_uniform",
            explicit_genesis=False,
            predecessor_receipt_path=receipt_path,
            matched_panel_spec=panel,
            matched_panel_spec_sha256=panel_sha256,
        )


@pytest.mark.parametrize(
    ("mutation", "error_fragment"),
    [
        ("shadow_count", "live EMA tensor count"),
        ("live_ema_count", "live EMA tensor count"),
        ("serialized_ema_count", "serialized EMA tensor count"),
        ("ema_decay", "EMA decay disagrees"),
        ("health_scope", "training-health scope"),
    ],
)
def test_predecessor_rejects_incoherent_ema_or_health_evidence(
    monkeypatch, tmp_path, mutation, error_fragment
):
    repository_root = tmp_path / "worktree"
    repository_root.mkdir()
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository_root)
    panel, panel_sha256 = _predecessor_test_panel(repository_root)
    genesis = launcher.build_predecessor_receipt_binding(
        training_variant="udlm",
        explicit_genesis=True,
        predecessor_receipt_path=None,
        matched_panel_spec=panel,
        matched_panel_spec_sha256=panel_sha256,
    )
    receipt_path = _write_successful_predecessor(
        repository_root,
        run_name=f"R_bad_{mutation}",
        training_variant="udlm",
        panel_spec=panel,
        panel_sha256=panel_sha256,
        predecessor_binding=genesis,
    )
    summary_path = receipt_path.parent / "training_summary.json"
    summary = launcher.json.loads(summary_path.read_text(encoding="utf-8"))
    semantic = summary["final_checkpoint"]["semantic_audit"]
    if mutation == "shadow_count":
        semantic["ema_metadata"]["shadow_parameter_count"] = 3
    elif mutation == "live_ema_count":
        semantic["live_ema_match"]["tensor_count"] = 3
    elif mutation == "serialized_ema_count":
        semantic["ema"]["floating_tensor_count"] = 3
    elif mutation == "ema_decay":
        semantic["ema_metadata"]["decay"] = 0.9
    else:
        summary["training_health"]["scope"] = "rank-zero counters"
    receipt = _rebind_mutated_predecessor_summary(receipt_path, summary)
    receipt_path.write_text(
        launcher.json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=error_fragment):
        launcher.build_predecessor_receipt_binding(
            training_variant="schedule_uniform",
            explicit_genesis=False,
            predecessor_receipt_path=receipt_path,
            matched_panel_spec=panel,
            matched_panel_spec_sha256=panel_sha256,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "error_fragment"),
    [
        ("source_path", "/wrong/source.ckpt", "source_path"),
        ("source_resolved_path", "/wrong/resolved.ckpt", "source_resolved_path"),
        ("source_size_bytes", 0, "source size"),
        ("weights", "raw", "warm-start weights"),
        ("parameter_tensors", 0, "parameter tensor count"),
        (
            "byte_identity_verified_before_and_after_load",
            False,
            "byte identity",
        ),
    ],
)
def test_predecessor_warm_start_report_is_source_bound(
    monkeypatch, tmp_path, field, replacement, error_fragment
):
    repository_root = tmp_path / "worktree"
    repository_root.mkdir()
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository_root)
    initialization_checkpoint = repository_root / "initialization/mdlm.ckpt"
    initialization_checkpoint.parent.mkdir()
    initialization_checkpoint.write_bytes(b"fixture MDLM checkpoint\n")
    initialization_sha256 = launcher.hashlib.sha256(
        initialization_checkpoint.read_bytes()
    ).hexdigest()
    panel, panel_sha256 = _predecessor_test_panel(
        repository_root,
        checkpoint=initialization_checkpoint,
        checkpoint_sha256=initialization_sha256,
    )
    genesis = launcher.build_predecessor_receipt_binding(
        training_variant="udlm",
        explicit_genesis=True,
        predecessor_receipt_path=None,
        matched_panel_spec=panel,
        matched_panel_spec_sha256=panel_sha256,
    )
    receipt_path = _write_successful_predecessor(
        repository_root,
        run_name=f"R_bad_warm_{field}",
        training_variant="udlm",
        panel_spec=panel,
        panel_sha256=panel_sha256,
        predecessor_binding=genesis,
    )
    launcher.build_predecessor_receipt_binding(
        training_variant="schedule_uniform",
        explicit_genesis=False,
        predecessor_receipt_path=receipt_path,
        matched_panel_spec=panel,
        matched_panel_spec_sha256=panel_sha256,
    )
    summary_path = receipt_path.parent / "training_summary.json"
    summary = launcher.json.loads(summary_path.read_text(encoding="utf-8"))
    summary["startup"]["verified_mdlm_warm_start_report"][field] = replacement
    receipt = _rebind_mutated_predecessor_summary(receipt_path, summary)
    receipt_path.write_text(
        launcher.json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=error_fragment):
        launcher.build_predecessor_receipt_binding(
            training_variant="schedule_uniform",
            explicit_genesis=False,
            predecessor_receipt_path=receipt_path,
            matched_panel_spec=panel,
            matched_panel_spec_sha256=panel_sha256,
        )


@pytest.mark.parametrize(
    ("mutation", "error_fragment"),
    [
        ("unreviewed_interpreter", "training argv prefix"),
        ("short_prefix", "training argv prefix"),
        ("empty_element", "nonempty string array"),
    ],
)
def test_predecessor_rejects_unmatched_training_argv_prefix(
    monkeypatch, tmp_path, mutation, error_fragment
):
    repository_root = tmp_path / "worktree"
    repository_root.mkdir()
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository_root)
    panel, panel_sha256 = _predecessor_test_panel(repository_root)
    genesis = launcher.build_predecessor_receipt_binding(
        training_variant="udlm",
        explicit_genesis=True,
        predecessor_receipt_path=None,
        matched_panel_spec=panel,
        matched_panel_spec_sha256=panel_sha256,
    )
    receipt_path = _write_successful_predecessor(
        repository_root,
        run_name=f"R_bad_argv_{mutation}",
        training_variant="udlm",
        panel_spec=panel,
        panel_sha256=panel_sha256,
        predecessor_binding=genesis,
    )
    manifest_path = receipt_path.parent / "launch_manifest.json"
    manifest = launcher.json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "unreviewed_interpreter":
        manifest["training_argv"][0] = "/unreviewed/python"
    elif mutation == "short_prefix":
        manifest["training_argv"] = manifest["training_argv"][:4]
    else:
        manifest["training_argv"][-1] = ""
    manifest["training_argv_sha256"] = launcher.canonical_json_sha256(
        manifest["training_argv"][2:]
    )
    manifest_path.write_text(
        launcher.json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_snapshot, _payload = launcher.stable_repository_artifact_snapshot(
        manifest_path,
        suffix=".json",
        label="argv-mutated predecessor manifest",
    )
    receipt = launcher.json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["expected_contract"]["training_argv_sha256"] = manifest[
        "training_argv_sha256"
    ]
    receipt["expected_contract"]["launch_manifest_sha256"] = manifest_snapshot["sha256"]
    receipt["launch_manifest"]["artifact"] = manifest_snapshot
    receipt_path.write_text(
        launcher.json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=error_fragment):
        launcher.build_predecessor_receipt_binding(
            training_variant="schedule_uniform",
            explicit_genesis=False,
            predecessor_receipt_path=receipt_path,
            matched_panel_spec=panel,
            matched_panel_spec_sha256=panel_sha256,
        )


def test_predecessor_python_path_is_stable_when_worktree_venv_appears(
    monkeypatch, tmp_path
):
    repository_root = tmp_path / "worktree"
    repository_root.mkdir()
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository_root)
    panel, panel_sha256 = _predecessor_test_panel(repository_root)
    genesis = launcher.build_predecessor_receipt_binding(
        training_variant="udlm",
        explicit_genesis=True,
        predecessor_receipt_path=None,
        matched_panel_spec=panel,
        matched_panel_spec_sha256=panel_sha256,
    )
    receipt_path = _write_successful_predecessor(
        repository_root,
        run_name="R_project_venv",
        training_variant="udlm",
        panel_spec=panel,
        panel_sha256=panel_sha256,
        predecessor_binding=genesis,
        python_executable_path=launcher.PROJECT_ROOT / ".venv/bin/python",
    )
    worktree_python = repository_root / ".venv/bin/python"
    worktree_python.parent.mkdir(parents=True)
    worktree_python.touch()

    binding = launcher.build_predecessor_receipt_binding(
        training_variant="schedule_uniform",
        explicit_genesis=False,
        predecessor_receipt_path=receipt_path,
        matched_panel_spec=panel,
        matched_panel_spec_sha256=panel_sha256,
    )

    assert binding["predecessor_run_name"] == "R_project_venv"


def test_predecessor_chronology_must_be_strict(monkeypatch, tmp_path):
    repository_root = tmp_path / "worktree"
    repository_root.mkdir()
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository_root)
    panel, panel_sha256 = _predecessor_test_panel(repository_root)
    genesis = launcher.build_predecessor_receipt_binding(
        training_variant="udlm",
        explicit_genesis=True,
        predecessor_receipt_path=None,
        matched_panel_spec=panel,
        matched_panel_spec_sha256=panel_sha256,
    )
    receipt_path = _write_successful_predecessor(
        repository_root,
        run_name="R_bad_chronology",
        training_variant="udlm",
        panel_spec=panel,
        panel_sha256=panel_sha256,
        predecessor_binding=genesis,
        summary_completed="2026-09-06T12:03:00+00:00",
        receipt_recorded="2026-09-06T12:02:00+00:00",
    )

    with pytest.raises(ValueError, match="strictly ordered"):
        launcher.build_predecessor_receipt_binding(
            training_variant="schedule_uniform",
            explicit_genesis=False,
            predecessor_receipt_path=receipt_path,
            matched_panel_spec=panel,
            matched_panel_spec_sha256=panel_sha256,
        )


def test_bound_predecessor_bytes_are_revalidated_before_gpu_probe(
    monkeypatch, tmp_path
):
    repository_root = tmp_path / "worktree"
    repository_root.mkdir()
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository_root)
    panel, panel_sha256 = _predecessor_test_panel(repository_root)
    genesis = launcher.build_predecessor_receipt_binding(
        training_variant="udlm",
        explicit_genesis=True,
        predecessor_receipt_path=None,
        matched_panel_spec=panel,
        matched_panel_spec_sha256=panel_sha256,
    )
    receipt_path = _write_successful_predecessor(
        repository_root,
        run_name="R_revalidate",
        training_variant="udlm",
        panel_spec=panel,
        panel_sha256=panel_sha256,
        predecessor_binding=genesis,
    )
    binding = launcher.build_predecessor_receipt_binding(
        training_variant="schedule_uniform",
        explicit_genesis=False,
        predecessor_receipt_path=receipt_path,
        matched_panel_spec=panel,
        matched_panel_spec_sha256=panel_sha256,
    )
    receipt_path.write_bytes(receipt_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="no longer matches"):
        launcher.revalidate_predecessor_receipt_binding(
            binding,
            training_variant="schedule_uniform",
            matched_panel_spec=panel,
            matched_panel_spec_sha256=panel_sha256,
        )


def test_e_revalidation_rebuilds_s_and_rejects_mutated_r_chain(monkeypatch, tmp_path):
    repository_root = tmp_path / "worktree"
    repository_root.mkdir()
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository_root)
    panel, panel_sha256 = _predecessor_test_panel(repository_root)
    genesis = launcher.build_predecessor_receipt_binding(
        training_variant="udlm",
        explicit_genesis=True,
        predecessor_receipt_path=None,
        matched_panel_spec=panel,
        matched_panel_spec_sha256=panel_sha256,
    )
    r_receipt = _write_successful_predecessor(
        repository_root,
        run_name="R_transitive",
        training_variant="udlm",
        panel_spec=panel,
        panel_sha256=panel_sha256,
        predecessor_binding=genesis,
    )
    s_binding = launcher.build_predecessor_receipt_binding(
        training_variant="schedule_uniform",
        explicit_genesis=False,
        predecessor_receipt_path=r_receipt,
        matched_panel_spec=panel,
        matched_panel_spec_sha256=panel_sha256,
    )
    s_receipt = _write_successful_predecessor(
        repository_root,
        run_name="S_transitive",
        training_variant="schedule_uniform",
        panel_spec=panel,
        panel_sha256=panel_sha256,
        predecessor_binding=s_binding,
        manifest_created="2026-09-06T12:03:00+00:00",
        summary_completed="2026-09-06T12:04:00+00:00",
        receipt_recorded="2026-09-06T12:05:00+00:00",
    )
    e_binding = launcher.build_predecessor_receipt_binding(
        training_variant="udlm_categorical",
        explicit_genesis=False,
        predecessor_receipt_path=s_receipt,
        matched_panel_spec=panel,
        matched_panel_spec_sha256=panel_sha256,
    )
    launcher.revalidate_predecessor_receipt_binding(
        e_binding,
        training_variant="udlm_categorical",
        matched_panel_spec=panel,
        matched_panel_spec_sha256=panel_sha256,
        current_manifest_created_at_utc="2026-09-06T12:06:00+00:00",
    )

    r_receipt.write_bytes(r_receipt.read_bytes() + b" ")

    with pytest.raises(ValueError, match="no longer matches"):
        launcher.revalidate_predecessor_receipt_binding(
            e_binding,
            training_variant="udlm_categorical",
            matched_panel_spec=panel,
            matched_panel_spec_sha256=panel_sha256,
            current_manifest_created_at_utc="2026-09-06T12:06:00+00:00",
        )


@pytest.mark.parametrize(
    "s_lock_acquired",
    [
        "2026-09-06T12:02:00+00:00",
        "2026-09-06T12:01:59.999999+00:00",
    ],
    ids=("r_receipt_equals_s_lock", "r_receipt_postdates_s_lock"),
)
def test_e_rejects_r_receipt_not_strictly_before_s_lock(
    monkeypatch, tmp_path, s_lock_acquired
):
    repository_root = tmp_path / "worktree"
    repository_root.mkdir()
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository_root)
    panel, panel_sha256 = _predecessor_test_panel(repository_root)
    genesis = launcher.build_predecessor_receipt_binding(
        training_variant="udlm",
        explicit_genesis=True,
        predecessor_receipt_path=None,
        matched_panel_spec=panel,
        matched_panel_spec_sha256=panel_sha256,
    )
    r_receipt = _write_successful_predecessor(
        repository_root,
        run_name="R_bad_S_lock_chronology",
        training_variant="udlm",
        panel_spec=panel,
        panel_sha256=panel_sha256,
        predecessor_binding=genesis,
    )
    s_binding = launcher.build_predecessor_receipt_binding(
        training_variant="schedule_uniform",
        explicit_genesis=False,
        predecessor_receipt_path=r_receipt,
        matched_panel_spec=panel,
        matched_panel_spec_sha256=panel_sha256,
    )
    s_receipt = _write_successful_predecessor(
        repository_root,
        run_name="S_bad_lock_chronology",
        training_variant="schedule_uniform",
        panel_spec=panel,
        panel_sha256=panel_sha256,
        predecessor_binding=s_binding,
        lock_acquired=s_lock_acquired,
        manifest_created="2026-09-06T12:03:00+00:00",
        summary_completed="2026-09-06T12:04:00+00:00",
        receipt_recorded="2026-09-06T12:05:00+00:00",
    )

    with pytest.raises(ValueError, match="strictly earlier.*lock acquisition"):
        launcher.build_predecessor_receipt_binding(
            training_variant="udlm_categorical",
            explicit_genesis=False,
            predecessor_receipt_path=s_receipt,
            matched_panel_spec=panel,
            matched_panel_spec_sha256=panel_sha256,
        )


def test_post_probe_identity_check_reaches_r_checkpoint_through_s(
    monkeypatch, tmp_path
):
    repository_root = tmp_path / "worktree"
    repository_root.mkdir()
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository_root)
    panel, panel_sha256 = _predecessor_test_panel(repository_root)
    genesis = launcher.build_predecessor_receipt_binding(
        training_variant="udlm",
        explicit_genesis=True,
        predecessor_receipt_path=None,
        matched_panel_spec=panel,
        matched_panel_spec_sha256=panel_sha256,
    )
    r_receipt = _write_successful_predecessor(
        repository_root,
        run_name="R_post_probe",
        training_variant="udlm",
        panel_spec=panel,
        panel_sha256=panel_sha256,
        predecessor_binding=genesis,
    )
    s_binding = launcher.build_predecessor_receipt_binding(
        training_variant="schedule_uniform",
        explicit_genesis=False,
        predecessor_receipt_path=r_receipt,
        matched_panel_spec=panel,
        matched_panel_spec_sha256=panel_sha256,
    )
    s_receipt = _write_successful_predecessor(
        repository_root,
        run_name="S_post_probe",
        training_variant="schedule_uniform",
        panel_spec=panel,
        panel_sha256=panel_sha256,
        predecessor_binding=s_binding,
        manifest_created="2026-09-06T12:03:00+00:00",
        summary_completed="2026-09-06T12:04:00+00:00",
        receipt_recorded="2026-09-06T12:05:00+00:00",
    )
    e_binding = launcher.build_predecessor_receipt_binding(
        training_variant="udlm_categorical",
        explicit_genesis=False,
        predecessor_receipt_path=s_receipt,
        matched_panel_spec=panel,
        matched_panel_spec_sha256=panel_sha256,
    )
    launcher.revalidate_predecessor_chain_artifact_identities(
        e_binding,
        current_manifest_created_at_utc="2026-09-06T12:06:00+00:00",
    )

    r_checkpoint = repository_root / "output/udlm/R_post_probe/checkpoints/10.ckpt"
    r_checkpoint.write_bytes(r_checkpoint.read_bytes() + b"mutated")

    with pytest.raises(ValueError, match="bound stat identity"):
        launcher.revalidate_predecessor_chain_artifact_identities(
            e_binding,
            current_manifest_created_at_utc="2026-09-06T12:06:00+00:00",
        )


def test_successor_manifest_timestamp_must_strictly_postdate_predecessor_receipt(
    monkeypatch, tmp_path
):
    repository_root = tmp_path / "worktree"
    repository_root.mkdir()
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository_root)
    panel, panel_sha256 = _predecessor_test_panel(repository_root)
    genesis = launcher.build_predecessor_receipt_binding(
        training_variant="udlm",
        explicit_genesis=True,
        predecessor_receipt_path=None,
        matched_panel_spec=panel,
        matched_panel_spec_sha256=panel_sha256,
    )
    receipt_path = _write_successful_predecessor(
        repository_root,
        run_name="R_chronology_boundary",
        training_variant="udlm",
        panel_spec=panel,
        panel_sha256=panel_sha256,
        predecessor_binding=genesis,
    )
    binding = launcher.build_predecessor_receipt_binding(
        training_variant="schedule_uniform",
        explicit_genesis=False,
        predecessor_receipt_path=receipt_path,
        matched_panel_spec=panel,
        matched_panel_spec_sha256=panel_sha256,
    )

    with pytest.raises(ValueError, match="strictly earlier"):
        launcher.revalidate_predecessor_receipt_binding(
            binding,
            training_variant="schedule_uniform",
            matched_panel_spec=panel,
            matched_panel_spec_sha256=panel_sha256,
            current_manifest_created_at_utc="2026-09-06T12:02:00+00:00",
        )

    launcher.revalidate_predecessor_receipt_binding(
        binding,
        training_variant="schedule_uniform",
        matched_panel_spec=panel,
        matched_panel_spec_sha256=panel_sha256,
        current_manifest_created_at_utc="2026-09-06T12:02:00.000001+00:00",
    )


@pytest.mark.parametrize(
    "tampered_receipt_time",
    [
        "2026-09-06T12:03:00+00:00",
        "2026-09-06T12:03:00.000001+00:00",
    ],
    ids=("equal_to_lock", "after_lock"),
)
def test_successor_receipt_must_strictly_predate_current_training_lock(
    monkeypatch, tmp_path, tampered_receipt_time
):
    repository_root = tmp_path / "worktree"
    repository_root.mkdir()
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository_root)
    panel, panel_sha256 = _predecessor_test_panel(repository_root)
    genesis = launcher.build_predecessor_receipt_binding(
        training_variant="udlm",
        explicit_genesis=True,
        predecessor_receipt_path=None,
        matched_panel_spec=panel,
        matched_panel_spec_sha256=panel_sha256,
    )
    receipt_path = _write_successful_predecessor(
        repository_root,
        run_name="R_lock_chronology",
        training_variant="udlm",
        panel_spec=panel,
        panel_sha256=panel_sha256,
        predecessor_binding=genesis,
    )
    binding = launcher.build_predecessor_receipt_binding(
        training_variant="schedule_uniform",
        explicit_genesis=False,
        predecessor_receipt_path=receipt_path,
        matched_panel_spec=panel,
        matched_panel_spec_sha256=panel_sha256,
    )
    launcher._require_predecessor_receipt_before_lock(
        binding,
        current_lock_acquired_at_utc="2026-09-06T12:03:00+00:00",
    )
    tampered = launcher.json.loads(launcher.json.dumps(binding))
    tampered["chronology"]["predecessor_exit_receipt_recorded_at_utc"] = (
        tampered_receipt_time
    )
    (repository_root / "output/logs").mkdir()
    monkeypatch.setattr(
        launcher,
        "probe_all_gpus",
        lambda: pytest.fail("lock chronology must fail before GPU inventory"),
    )

    with pytest.raises(ValueError, match="strictly earlier.*lock acquisition"):
        launcher._launch_locked_pilot(
            args=launcher.argparse.Namespace(
                gpu_count=1,
                training_variant="schedule_uniform",
                run_name="S_lock_chronology",
            ),
            git_sha="a" * 40,
            checkpoint=None,
            checkpoint_sha256=None,
            command=[],
            resolved_config={},
            resolved_config_sha256="b" * 64,
            argv_sha256="c" * 64,
            matched_panel_spec=panel,
            matched_panel_spec_sha256=panel_sha256,
            predecessor_receipt_binding=tampered,
            accumulation_steps=1,
            session_name="genmol_schedule_uniform_S_lock_chronology",
            lock_path=repository_root / "output/udlm/.single_training_job.lock",
            lock_record={"acquired_at_utc": "2026-09-06T12:03:00+00:00"},
            lock_sha256="d" * 64,
        )


def test_successor_dry_run_binds_receipt_without_gpu_query_or_mutation(
    monkeypatch, tmp_path, capsys
):
    _mock_verified_floor_audit(monkeypatch)
    repository_root = tmp_path / "worktree"
    repository_root.mkdir()
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository_root)
    monkeypatch.setattr(launcher, "PROJECT_ROOT", tmp_path)
    panel, panel_sha256 = _predecessor_test_panel(repository_root, seed=7)
    genesis = launcher.build_predecessor_receipt_binding(
        training_variant="udlm",
        explicit_genesis=True,
        predecessor_receipt_path=None,
        matched_panel_spec=panel,
        matched_panel_spec_sha256=panel_sha256,
    )
    receipt_path = _write_successful_predecessor(
        repository_root,
        run_name="R_for_dry_S",
        training_variant="udlm",
        panel_spec=panel,
        panel_sha256=panel_sha256,
        predecessor_binding=genesis,
    )
    current_run = "dry_S"
    resolved = _predecessor_test_config(
        repository_root, current_run, "schedule_uniform", seed=7
    )
    before = {
        path.relative_to(repository_root): path.read_bytes()
        for path in repository_root.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(
        launcher,
        "_parse_args",
        lambda: launcher.argparse.Namespace(
            run_name=current_run,
            training_variant="schedule_uniform",
            gpu_count=1,
            max_steps=10,
            global_batch_size=16,
            micro_batch_size=2,
            num_workers=1,
            seed=7,
            checkpoint=tmp_path / "unused.ckpt",
            scratch=True,
            exclude_special_tokens=False,
            max_utilization_percent=10,
            min_free_memory_mib=30_000,
            dry_run=True,
            genesis=False,
            predecessor_receipt=receipt_path,
        ),
    )
    monkeypatch.setattr(
        launcher,
        "_python_executable",
        lambda: repository_root / ".venv/bin/python",
    )
    monkeypatch.setattr(launcher, "require_pushed_commit", lambda: "a" * 40)
    monkeypatch.setattr(
        launcher,
        "compose_resolved_training_config",
        lambda **_kwargs: (resolved, launcher.canonical_json_sha256(resolved)),
    )
    monkeypatch.setattr(
        launcher,
        "probe_all_gpus",
        lambda: pytest.fail("successor dry-run must not query GPUs"),
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("successor dry-run must not use tmux"),
    )

    launcher.main()

    preview = launcher.json.loads(capsys.readouterr().out)
    binding = preview["predecessor_receipt_binding"]
    assert binding["state"] == "validated_successful_predecessor"
    assert binding["expected_predecessor_training_variant"] == "udlm"
    assert binding["receipt_artifact"]["path"] == str(receipt_path)
    after = {
        path.relative_to(repository_root): path.read_bytes()
        for path in repository_root.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert not (repository_root / f"output/udlm/{current_run}").exists()


def test_dry_run_is_nonmutating_and_never_probes_gpus_or_tmux(
    monkeypatch, tmp_path, capsys
):
    _mock_verified_floor_audit(monkeypatch)
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
            genesis=True,
            predecessor_receipt=None,
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
    assert preview["predecessor_receipt_binding"]["state"] == (
        "explicit_genesis_no_predecessor"
    )
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
    _mock_verified_floor_audit(monkeypatch)
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
            genesis=True,
            predecessor_receipt=None,
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
            genesis=True,
            predecessor_receipt=None,
        ),
    )
    monkeypatch.setattr(
        launcher, "_python_executable", lambda: tmp_path / ".venv/bin/python"
    )
    events = []
    manifest_creation_timestamps = []
    identity_check_timestamps = []
    monkeypatch.setattr(
        launcher,
        "verify_pilot_empirical_uniform_mix_audit",
        lambda: events.append("prior_floor_audit") or _expected_floor_audit_binding(),
    )
    original_revalidate = launcher.revalidate_predecessor_receipt_binding

    def revalidate(*args, **kwargs):
        events.append("predecessor_revalidation")
        manifest_creation_timestamps.append(
            kwargs.get("current_manifest_created_at_utc")
        )
        return original_revalidate(*args, **kwargs)

    monkeypatch.setattr(launcher, "revalidate_predecessor_receipt_binding", revalidate)
    original_identity_check = launcher.revalidate_predecessor_chain_artifact_identities

    def revalidate_identities(*args, **kwargs):
        events.append("predecessor_identity_revalidation")
        identity_check_timestamps.append(kwargs.get("current_manifest_created_at_utc"))
        return original_identity_check(*args, **kwargs)

    monkeypatch.setattr(
        launcher,
        "revalidate_predecessor_chain_artifact_identities",
        revalidate_identities,
    )
    original_lock_chronology_check = launcher._require_predecessor_receipt_before_lock

    def require_receipt_before_lock(*args, **kwargs):
        events.append("predecessor_before_current_lock")
        return original_lock_chronology_check(*args, **kwargs)

    monkeypatch.setattr(
        launcher,
        "_require_predecessor_receipt_before_lock",
        require_receipt_before_lock,
    )

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
            assert manifest_creation_timestamps == [None]
            assert identity_check_timestamps == [manifest["created_at"]]
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
            assert manifest["pilot_exit_status_schema_version"] == 5
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
                    "predecessor_receipt_binding_unchanged_and_valid": True,
                },
                "training_job_lock_release": (
                    "after_exit_receipt_publication_for_completed_or_failed_pipeline"
                ),
            }
            assert manifest["launch_manifest_schema_version"] == 2
            assert manifest["predecessor_receipt_binding"]["state"] == (
                "explicit_genesis_no_predecessor"
            )
            assert manifest["cuda_visible_device_uuids"] == ["GPU-idle"]
            assert manifest["gpu_safety_policy"] == {
                "max_utilization_percent": 10,
                "utilization_comparison": "strictly_less_than",
                "min_free_memory_mib": 30_000,
                "active_compute_processes_allowed": True,
                "compute_mode_prohibited_allowed": False,
            }
            assert manifest["matched_panel_spec"]["execution"]["mode"] == (
                "single_job_lease_with_machine_enforced_predecessor_receipt_chain"
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
        "prior_floor_audit",
        "source_check",
        "tmux_preflight",
        "predecessor_before_current_lock",
        "inventory",
        "source_check",
        "predecessor_revalidation",
        "final_uuid_probe",
        "predecessor_identity_revalidation",
        "tmux_spawn",
    ]


def test_output_parent_swap_after_final_probe_fails_before_reservation(
    monkeypatch, tmp_path
):
    repository_root = _mock_main_cpu_preflight(
        monkeypatch, tmp_path, run_name="parent_swap"
    )
    outside = tmp_path / "outside_logs"
    outside.mkdir()
    gpu = _gpu(index=3, uuid="GPU-idle")
    monkeypatch.setattr(launcher, "probe_all_gpus", lambda: [gpu])

    def swap_log_parent(*_args, **_kwargs):
        log_parent = repository_root / "output/logs"
        log_parent.rmdir()
        log_parent.symlink_to(outside, target_is_directory=True)
        return (gpu,)

    monkeypatch.setattr(launcher, "reprobe_selected_gpus", swap_log_parent)

    with pytest.raises(ValueError, match="output ancestor"):
        launcher.main()

    assert list(outside.iterdir()) == []
    assert not (repository_root / "output/udlm/parent_swap").exists()
    assert not launcher.os.path.lexists(
        repository_root / "output/udlm/.single_training_job.lock"
    )

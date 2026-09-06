import subprocess
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
    assert "loader.global_batch_size=16" in joined
    assert "loader.batch_size=2" in joined
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
    assert config["training"]["udlm"]["prior_variant"] == "schedule_uniform"
    assert "hydra" not in config


def test_child_command_sanitizes_python_and_binds_source_argv_config(
    monkeypatch,
):
    monkeypatch.setenv("PYTHONHOME", "/hostile/home")
    monkeypatch.setenv("PYTHONWARNINGS", "error")
    monkeypatch.setenv("PYTHONARBITRARY", "hostile")
    command = [
        "/venv/python",
        "-u",
        str(launcher.REPOSITORY_ROOT / "scripts/train.py"),
        "--config-name",
        "udlm",
        "seed=7",
    ]
    runtime_path = launcher.REPOSITORY_ROOT / "output/udlm/test/runtime_config.json"

    child_command, environment = launcher.build_child_environment_command(
        command=command,
        source_revision="a" * 40,
        resolved_config_sha256="b" * 64,
        runtime_config_path=runtime_path,
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
        lambda **_kwargs: ({"seed": 1}, "b" * 64),
    )

    def subprocess_run(command, **_kwargs):
        if command[:2] == ["tmux", "has-session"]:
            events.append("tmux_preflight")
            return subprocess.CompletedProcess(command, 1)
        if command[:2] == ["tmux", "new-session"]:
            events.append("tmux_spawn")
            assert (
                repository_root / "output/udlm/ordering/launch_manifest.json"
            ).is_file()
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

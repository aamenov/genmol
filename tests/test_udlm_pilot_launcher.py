import subprocess
from pathlib import Path

import pytest

from scripts.udlm import launch_train_pilot as launcher


def test_gpu_request_is_capped_at_two_and_requires_exact_indices():
    assert launcher.validate_gpu_request(1, [3]) == (3,)
    assert launcher.validate_gpu_request(2, [6, 7]) == (6, 7)
    with pytest.raises(ValueError, match="1 or 2"):
        launcher.validate_gpu_request(3, [0, 1, 2])
    with pytest.raises(ValueError, match="equal"):
        launcher.validate_gpu_request(2, [1])
    with pytest.raises(ValueError, match="unique"):
        launcher.validate_gpu_request(2, [1, 1])


def test_gpu_with_compute_process_is_not_genuinely_idle():
    state = launcher.GPUState(
        physical_index=2,
        uuid="GPU-example",
        name="Example",
        memory_used_mib=1_000,
        memory_total_mib=48_000,
        utilization_percent=2,
        compute_processes=(
            {"pid": 123, "process_name": "python", "used_memory_mib": 900},
        ),
    )

    reasons = state.rejection_reasons(
        max_utilization_percent=10,
        min_free_memory_mib=30_000,
    )

    assert any("active compute" in reason for reason in reasons)


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
        exclude_special_tokens=True,
    )
    joined = " ".join(str(part) for part in command)

    assert "--config-name udlm" in joined
    assert "trainer.devices=2" in joined
    assert "trainer.max_steps=10" in joined
    assert "loader.global_batch_size=16" in joined
    assert "loader.batch_size=2" in joined
    assert "training.init_from_mdlm_checkpoint=/project/50000.ckpt" in joined
    assert "training.udlm.exclude_special_tokens=true" in joined


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

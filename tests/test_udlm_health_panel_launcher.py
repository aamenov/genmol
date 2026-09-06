from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.udlm import launch_health_panel as launcher
from scripts.udlm import launch_train_pilot as pilot


SOURCE_REVISION = "a" * 40


@pytest.fixture
def isolated_repository(monkeypatch, tmp_path):
    repository_root = tmp_path / "worktree"
    repository_root.mkdir()
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", repository_root)
    monkeypatch.setattr(pilot, "REPOSITORY_ROOT", repository_root)
    monkeypatch.setattr(pilot, "require_pushed_commit", lambda: SOURCE_REVISION)
    return repository_root


def _paths(repository_root: Path, *, gpu_count: int = 1):
    return launcher._run_paths(
        gpu_count=gpu_count,
        source_revision=SOURCE_REVISION,
    )


def _complete_prefix(
    repository_root: Path, count: int, *, gpu_count: int = 1
) -> tuple[tuple[str, Path, Path], ...]:
    paths = _paths(repository_root, gpu_count=gpu_count)
    for _variant, run_dir, receipt in paths[:count]:
        run_dir.mkdir(parents=True)
        receipt.write_text(
            json.dumps(
                {
                    "schema_version": pilot.PILOT_EXIT_STATUS_SCHEMA_VERSION,
                    "status": "completed",
                    "overall_status": "completed",
                    "process_exit_status": 0,
                }
            )
            + "\n"
        )
    return paths


def _value_after(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_cli_exposes_only_gpu_count_and_non_scientific_dry_run():
    parsed = launcher._parse_args(["--gpu-count", "2", "--dry-run"])
    assert vars(parsed) == {"gpu_count": 2, "dry_run": True}


@pytest.mark.parametrize(
    "forbidden",
    [
        ["--max-steps", "11"],
        ["--global-batch-size", "32"],
        ["--micro-batch-size", "4"],
        ["--num-workers", "2"],
        ["--seed", "17"],
        ["--checkpoint", "/tmp/other.ckpt"],
        ["--exclude-special-tokens"],
        ["--max-utilization-percent", "5"],
        ["--min-free-memory-mib", "40000"],
        ["--training-variant", "udlm_categorical"],
        ["--run-name", "other"],
    ],
)
def test_cli_rejects_every_scientific_or_safety_tuning_knob(forbidden):
    with pytest.raises(SystemExit):
        launcher._parse_args(["--gpu-count", "1", *forbidden])


@pytest.mark.parametrize("value", ["0", "3", "cuda:0"])
def test_cli_rejects_non_registered_gpu_counts(value):
    with pytest.raises(SystemExit):
        launcher._parse_args(["--gpu-count", value])


@pytest.mark.parametrize(
    ("gpu_count", "expected_accumulation"),
    [(1, 8), (2, 4)],
)
def test_first_arm_forwards_the_exact_health_contract(
    monkeypatch,
    isolated_repository,
    gpu_count,
    expected_accumulation,
):
    observed: list[list[str]] = []
    monkeypatch.setattr(pilot, "MAX_SAFE_UTILIZATION_PERCENT", 99)
    monkeypatch.setattr(pilot, "MIN_SAFE_FREE_MEMORY_MIB", 1)
    monkeypatch.setattr(pilot, "main", lambda argv: observed.append(argv))

    assert launcher.main(["--gpu-count", str(gpu_count)]) == 0

    assert len(observed) == 1
    argv = observed[0]
    expected_run_name = launcher.health_run_name(gpu_count, "udlm", SOURCE_REVISION)
    assert _value_after(argv, "--run-name") == expected_run_name
    assert _value_after(argv, "--training-variant") == "udlm"
    assert _value_after(argv, "--gpu-count") == str(gpu_count)
    assert _value_after(argv, "--max-steps") == str(launcher.HEALTH_PANEL_MAX_STEPS)
    assert _value_after(argv, "--global-batch-size") == str(
        launcher.HEALTH_PANEL_GLOBAL_BATCH_SIZE
    )
    assert _value_after(argv, "--micro-batch-size") == str(
        launcher.HEALTH_PANEL_MICRO_BATCH_SIZE
    )
    assert _value_after(argv, "--num-workers") == str(launcher.HEALTH_PANEL_NUM_WORKERS)
    assert _value_after(argv, "--seed") == str(launcher.HEALTH_PANEL_SEED)
    assert _value_after(argv, "--checkpoint") == str(
        launcher.EXPECTED_MDLM_CHECKPOINT_PATH
    )
    assert _value_after(argv, "--max-utilization-percent") == str(
        launcher.HEALTH_PANEL_MAX_UTILIZATION_PERCENT
    )
    assert _value_after(argv, "--min-free-memory-mib") == str(
        launcher.HEALTH_PANEL_MIN_FREE_MEMORY_MIB
    )
    assert "--genesis" in argv
    assert "--predecessor-receipt" not in argv
    assert "--exclude-special-tokens" not in argv
    assert "--scratch" not in argv
    assert "--dry-run" not in argv
    assert (
        pilot.exact_accumulation_steps(
            launcher.HEALTH_PANEL_GLOBAL_BATCH_SIZE,
            launcher.HEALTH_PANEL_MICRO_BATCH_SIZE,
            gpu_count,
        )
        == expected_accumulation
    )


@pytest.mark.parametrize(
    ("completed", "expected_variant"),
    [(1, "schedule_uniform"), (2, "udlm_categorical")],
)
def test_each_invocation_advances_only_the_first_missing_arm(
    monkeypatch,
    isolated_repository,
    completed,
    expected_variant,
):
    paths = _complete_prefix(isolated_repository, completed)
    observed: list[list[str]] = []
    monkeypatch.setattr(pilot, "main", lambda argv: observed.append(argv))

    assert launcher.main(["--gpu-count", "1"]) == 0

    assert len(observed) == 1
    argv = observed[0]
    assert _value_after(argv, "--training-variant") == expected_variant
    assert _value_after(argv, "--run-name") == paths[completed][1].name
    assert "--genesis" not in argv
    assert _value_after(argv, "--predecessor-receipt") == str(paths[completed - 1][2])


def test_existing_run_without_receipt_fails_before_pilot(
    monkeypatch, isolated_repository
):
    _variant, run_dir, _receipt = _paths(isolated_repository)[0]
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(
        pilot,
        "main",
        lambda _argv: pytest.fail("incomplete health arm must not call pilot"),
    )

    with pytest.raises(RuntimeError, match="without its completion receipt"):
        launcher.main(["--gpu-count", "1"])


@pytest.mark.parametrize("existing_positions", [(1,), (2,), (0, 2)])
def test_out_of_order_run_directories_fail_before_pilot(
    monkeypatch,
    isolated_repository,
    existing_positions,
):
    paths = _paths(isolated_repository)
    for position in existing_positions:
        _variant, run_dir, receipt = paths[position]
        run_dir.mkdir(parents=True)
        receipt.write_text(
            json.dumps(
                {
                    "schema_version": pilot.PILOT_EXIT_STATUS_SCHEMA_VERSION,
                    "status": "completed",
                    "overall_status": "completed",
                    "process_exit_status": 0,
                }
            )
            + "\n"
        )
    monkeypatch.setattr(
        pilot,
        "main",
        lambda _argv: pytest.fail("out-of-order health arm must not call pilot"),
    )

    with pytest.raises(RuntimeError, match="out of R -> S -> E order"):
        launcher.main(["--gpu-count", "1"])


@pytest.mark.parametrize(
    "receipt",
    [
        {
            "schema_version": pilot.PILOT_EXIT_STATUS_SCHEMA_VERSION,
            "status": "failed",
            "overall_status": "failed",
            "process_exit_status": 97,
        },
        {
            "schema_version": pilot.PILOT_EXIT_STATUS_SCHEMA_VERSION,
            "status": "completed",
            "overall_status": "completed",
            "process_exit_status": True,
        },
        {"status": "completed"},
    ],
)
def test_failed_or_malformed_receipt_closes_revision_without_advancing(
    monkeypatch, isolated_repository, receipt
):
    _variant, run_dir, receipt_path = _paths(isolated_repository)[0]
    run_dir.mkdir(parents=True)
    receipt_path.write_text(json.dumps(receipt) + "\n")
    monkeypatch.setattr(
        pilot,
        "main",
        lambda _argv: pytest.fail("failed health arm must not call pilot"),
    )

    with pytest.raises(RuntimeError, match="failed or malformed completion receipt"):
        launcher.main(["--gpu-count", "1"])


def test_indirect_run_or_receipt_is_rejected_before_pilot(
    monkeypatch, isolated_repository, tmp_path
):
    paths = _paths(isolated_repository)
    outside = tmp_path / "outside"
    outside.mkdir()
    paths[0][1].parent.mkdir(parents=True)
    paths[0][1].symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(
        pilot,
        "main",
        lambda _argv: pytest.fail("indirect health path must not call pilot"),
    )

    with pytest.raises(RuntimeError, match="direct real directory"):
        launcher.main(["--gpu-count", "1"])


def test_completed_terminal_arm_is_validated_and_never_relaunched(
    monkeypatch, isolated_repository, capsys
):
    paths = _complete_prefix(isolated_repository, 3, gpu_count=2)
    normalized = {
        "schema_version": 1,
        "status": "validated",
        "claim_scope": "training_health_and_provenance_only",
    }
    calls = []

    def fake_validate(path, **expectations):
        calls.append((path, expectations))
        return normalized

    monkeypatch.setattr(launcher, "validate_health_panel", fake_validate)
    monkeypatch.setattr(
        pilot,
        "main",
        lambda _argv: pytest.fail("completed health panel must not call pilot"),
    )

    assert launcher.main(["--gpu-count", "2"]) == 0

    assert calls == [
        (
            paths[-1][2],
            {
                "expected_gpu_count": 2,
                "expected_source_revision": SOURCE_REVISION,
            },
        )
    ]
    assert json.loads(capsys.readouterr().out) == normalized


def test_dry_run_is_forwarded_without_wrapper_gpu_tmux_or_output_actions(
    monkeypatch, isolated_repository
):
    observed: list[list[str]] = []
    monkeypatch.setattr(pilot, "main", lambda argv: observed.append(argv))
    monkeypatch.setattr(
        pilot,
        "probe_all_gpus",
        lambda: pytest.fail("wrapper must not probe GPUs"),
    )
    monkeypatch.setattr(
        pilot,
        "tmux_session_exists",
        lambda _name: pytest.fail("wrapper must not inspect tmux"),
    )

    assert launcher.main(["--gpu-count", "1", "--dry-run"]) == 0

    assert len(observed) == 1
    assert observed[0][-1] == "--dry-run"
    assert not (isolated_repository / "output").exists()

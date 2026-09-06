import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.udlm import launch_train_pilot as launcher


def _canonical_sha256(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


RESOLVED_TRAINING_CONFIG = {"seed": 7}
TRAINING_ARGV = ["/repo/scripts/train.py", "seed=7"]
EXPECTED_CONFIG_SHA256 = _canonical_sha256(RESOLVED_TRAINING_CONFIG)
EXPECTED_ARGV_SHA256 = _canonical_sha256(TRAINING_ARGV)
EXPECTED_WARM_START_SHA256 = "e" * 64


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


def _valid_summary(paths, revision):
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    runtime_path = paths["run_dir"] / "runtime_config.json"
    paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    if not paths["checkpoint"].exists():
        paths["checkpoint"].write_bytes(b"stable checkpoint fixture\n")
    completion_contract = {
        "summary_schema_version": 1,
        "summary_path": str(paths["summary"]),
        "final_checkpoint_path": str(paths["checkpoint"]),
        "expected_max_steps": 10,
        "expected_world_size": 1,
        "fail_on_nonfinite_loss": True,
        "backward_anomaly_detection": True,
    }
    runtime_record = {
        "schema_version": 1,
        "status": "preflight_completed",
        "source_revision": revision,
        "source": {"head": revision, "upstream": revision},
        "training_argv": TRAINING_ARGV,
        "observed_training_argv": TRAINING_ARGV,
        "training_argv_sha256": EXPECTED_ARGV_SHA256,
        "resolved_training_config": RESOLVED_TRAINING_CONFIG,
        "resolved_training_config_sha256": EXPECTED_CONFIG_SHA256,
        "completion_contract": completion_contract,
        "python_environment": {"PYTHONHASHSEED": "7"},
    }
    runtime_path.write_text(
        json.dumps(runtime_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": "2026-09-06T12:00:00+00:00",
        "source_revision": revision,
        "source": {"head": revision, "upstream": revision},
        "resolved_training_config_sha256": EXPECTED_CONFIG_SHA256,
        "training_argv_sha256": EXPECTED_ARGV_SHA256,
        "completion_contract": completion_contract,
        "runtime_config": {
            **_snapshot(runtime_path),
            "schema_version": 1,
            "record_sha256": _canonical_sha256(runtime_record),
        },
        "observed_training_state": {
            "global_rank": 0,
            "global_step": 10,
            "world_size": 1,
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
        "final_checkpoint": {
            **_snapshot(paths["checkpoint"]),
            "semantic_audit": {
                "deserialized": True,
                "global_step": 10,
                "raw_model": _finite_record(),
                "ema": _finite_record(),
                "optimizer": _finite_record(),
                "all_checkpoint_tensors": _finite_record(tensors=6, elements=12),
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


def _shell_command(
    paths,
    revision,
    *,
    training_command,
    log_path=None,
):
    return launcher.build_tmux_shell_command(
        training_command,
        log_path=paths["log"] if log_path is None else log_path,
        training_summary_path=paths["summary"],
        exit_receipt_path=paths["receipt"],
        expected_source_revision=revision,
        expected_config_sha256=EXPECTED_CONFIG_SHA256,
        expected_argv_sha256=EXPECTED_ARGV_SHA256,
        expected_summary_schema_version=1,
        expected_max_steps=10,
        expected_world_size=1,
        expected_final_checkpoint_path=paths["checkpoint"],
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


def test_successful_pipeline_writes_launch_bound_receipt(receipt_repository):
    repository, revision = receipt_repository
    paths = _paths(repository)
    _write_summary(paths, revision)
    paths["log"].parent.mkdir(parents=True)
    summary_sha256 = hashlib.sha256(paths["summary"].read_bytes()).hexdigest()

    result = _execute_shell(
        repository,
        _shell_command(paths, revision, training_command=["bash", "-c", "exit 0"]),
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    assert receipt["schema_version"] == 1
    assert receipt["status"] == "completed"
    assert receipt["process_exit_status"] == 0
    assert receipt["pipeline"]["training"]["shell_exit_status"] == 0
    assert receipt["pipeline"]["tee"]["shell_exit_status"] == 0
    assert receipt["training_summary"]["valid_and_launch_bound"] is True
    assert receipt["training_summary"]["artifact"]["sha256"] == summary_sha256
    assert receipt["runtime_config"]["matches_training_summary_snapshot"] is True
    assert receipt["final_checkpoint"]["matches_training_summary_snapshot"] is True
    assert receipt["source_at_receipt"]["verified"] is True
    assert receipt["expected_contract"] == {
        "training_summary_schema_version": 1,
        "source_revision": revision,
        "resolved_training_config_sha256": EXPECTED_CONFIG_SHA256,
        "training_argv_sha256": EXPECTED_ARGV_SHA256,
        "max_steps": 10,
        "world_size": 1,
        "training_summary_path": str(paths["summary"]),
        "final_checkpoint_path": str(paths["checkpoint"]),
        "initialization_checkpoint_sha256": EXPECTED_WARM_START_SHA256,
    }


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
        (("training_health",), None, "training health evidence"),
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
        (
            ("tensor_finiteness", "ema", "floating_tensor_count"),
            0,
            "live EMA floating tensor count",
        ),
        (
            ("final_checkpoint", "semantic_audit"),
            None,
            "final checkpoint semantic audit",
        ),
        (
            ("final_checkpoint", "semantic_audit", "optimizer", "all_finite"),
            False,
            "serialized checkpoint optimizer all-finite flag",
        ),
        (
            ("final_checkpoint", "semantic_audit", "live_model_match", "tensor_count"),
            0,
            "live-model match tensor count",
        ),
        (
            ("runtime_config", "record_sha256"),
            None,
            "runtime config canonical record digest",
        ),
        (("startup",), None, "startup evidence"),
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
        "schema_version": 1,
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

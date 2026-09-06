import hashlib
import json
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
    "trainer": {
        "devices": 1,
        "num_nodes": 1,
        "max_steps": 10,
        "accumulate_grad_batches": 4,
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


def _write_training_job_lock(paths):
    paths["lock"].parent.mkdir(parents=True, exist_ok=True)
    if not paths["lock"].exists():
        paths["lock"].write_bytes(EXPECTED_LOCK_BYTES)
    return EXPECTED_LOCK_RECORD, EXPECTED_LOCK_SHA256


def _write_launch_manifest(paths):
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    lock_record, lock_sha256 = _write_training_job_lock(paths)
    manifest = {
        "launch_manifest_schema_version": 1,
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
        "completed_at_utc": "2026-09-06T12:00:00+00:00",
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
    assert receipt["schema_version"] == receipt_writer.EXIT_STATUS_SCHEMA_VERSION == 3
    assert receipt["status"] == "completed"
    assert receipt["process_exit_status"] == 0
    assert receipt["pipeline"]["training"]["shell_exit_status"] == 0
    assert receipt["pipeline"]["tee"]["shell_exit_status"] == 0
    assert receipt["training_summary"]["valid_and_launch_bound"] is True
    assert receipt["training_summary"]["artifact"]["sha256"] == summary_sha256
    assert receipt["training_summary"]["validated_bindings"][
        "training_accounting"
    ] == expected_accounting
    assert receipt["runtime_config"]["matches_training_summary_snapshot"] is True
    assert receipt["launch_manifest"]["valid_and_launch_bound"] is True
    assert receipt["launch_manifest"]["matches_runtime_config_snapshot"] is True
    assert receipt["training_job_lock"][
        "valid_and_launch_bound_before_receipt_publication"
    ] is True
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
    assert receipt["training_job_lock"][
        "valid_and_launch_bound_before_receipt_publication"
    ] is False
    assert "training-job lock raw SHA-256" in receipt["training_job_lock"][
        "validation_error"
    ]


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
        (("training_accounting",), None, "training accounting"),
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
            None,
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
    assert manifest_reads == 2
    assert receipt["launch_manifest"]["valid_and_launch_bound"] is False
    assert "changed during receipt validation" in receipt["launch_manifest"][
        "validation_error"
    ]
    assert "changed during receipt validation" in receipt["training_summary"][
        "validation_error"
    ]


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

    with pytest.raises(
        ValueError, match="accounting micro-batch size disagrees with resolved config"
    ):
        receipt_writer.validate_training_summary(
            summary,
            summary_path=paths["summary"],
            expected_schema_version=receipt_writer.TRAINING_SUMMARY_SCHEMA_VERSION,
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
            expected_initialization_checkpoint_sha256=(
                EXPECTED_WARM_START_SHA256
            ),
            resolved_training_config=mismatched_config,
        )


def test_receipt_writer_rejects_legacy_training_summary_schema(receipt_repository):
    repository, revision = receipt_repository
    paths = _paths(repository)
    summary = _valid_summary(paths, revision)
    summary["schema_version"] = 1

    with pytest.raises(
        ValueError, match="unsupported training summary schema version 1; expected 3"
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
            expected_initialization_checkpoint_sha256=(
                EXPECTED_WARM_START_SHA256
            ),
            resolved_training_config=RESOLVED_TRAINING_CONFIG,
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

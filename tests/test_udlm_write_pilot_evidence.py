from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.exps.denovo import launch_benchmark as launcher
from scripts.udlm import write_pilot_evidence as writer


ATTEMPT = "schedule-l1-a1"
CANDIDATE = "candidate-a1"
SEED = 1000
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
REVISION = "d" * 40


def _bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def _snapshot(
    path: str, *, sha256: str = SHA_A, size_bytes: int = 123
) -> dict[str, Any]:
    return {
        "path": path,
        "device": 1,
        "inode": 2,
        "mode": 33188,
        "link_count": 1,
        "size_bytes": size_bytes,
        "mtime_ns": 3,
        "ctime_ns": 4,
        "sha256": sha256,
        "stable_regular_file_verified": True,
    }


def _live_snapshot(path: Path) -> dict[str, Any]:
    return writer.read_stable_regular_file(path, label=path.name).snapshot()


def _successful_receipt(root: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    run_root = root / "output" / "udlm" / "training-r"
    summary_path = run_root / "training_summary.json"
    manifest_path = run_root / "launch_manifest.json"
    runtime_path = run_root / "runtime_config.json"
    lock_path = root / "output" / "udlm" / ".single_training_job.lock"
    success_component = {
        "shell_exit_status": 0,
        "succeeded": True,
        "possible_termination_signal": None,
        "shell_status_is_signal_compatible": False,
        "signal_provenance": None,
    }
    return {
        "schema_version": 5,
        "status": "completed",
        "overall_status": "completed",
        "recorded_at_utc": "2026-09-06T09:00:00+00:00",
        "process_exit_status": 0,
        "expected_contract": {
            "training_summary_schema_version": 5,
            "source_revision": REVISION,
            "resolved_training_config_sha256": SHA_A,
            "training_argv_sha256": SHA_B,
            "launch_manifest_path": str(manifest_path),
            "launch_manifest_sha256": SHA_C,
            "selected_gpu_uuids": ["GPU-example"],
            "training_job_lock_path": str(lock_path),
            "training_job_lock_sha256": SHA_A,
            "max_steps": checkpoint["global_step"],
            "world_size": 1,
            "training_summary_path": str(summary_path),
            "final_checkpoint_path": checkpoint["path"],
            "initialization_checkpoint_sha256": None,
        },
        "pipeline": {
            "training": dict(success_component),
            "tee": dict(success_component),
            "pipefail_shell_exit_status": 0,
        },
        "source_at_receipt": {
            "verified": True,
            "expected_revision": REVISION,
            "head": REVISION,
            "upstream": REVISION,
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
            "expected_selected_gpu_uuids": ["GPU-example"],
            "observed_selected_gpu_uuids": ["GPU-example"],
            "artifact": _snapshot(str(manifest_path), sha256=SHA_C),
            "validation_error": None,
        },
        "predecessor_receipt_binding": {"state": "explicit_genesis_no_predecessor"},
        "training_job_lock": {
            "path": str(lock_path),
            "present": True,
            "expected_sha256": SHA_A,
            "matches_expected_raw_sha256": True,
            "matches_launch_manifest_binding": True,
            "valid_and_launch_bound_before_receipt_publication": True,
            "artifact": _snapshot(str(lock_path), sha256=SHA_A),
            "record": {"status": "held"},
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
            "artifact": _snapshot(str(summary_path), sha256=SHA_B),
            "validated_bindings": {"valid": True},
            "validation_error": None,
        },
        "runtime_config": {
            "path": str(runtime_path),
            "present": True,
            "matches_training_summary_snapshot": True,
            "semantic_validation_passed": True,
            "artifact": _snapshot(str(runtime_path), sha256=SHA_A),
        },
        "final_checkpoint": {
            "path": checkpoint["path"],
            "present": True,
            "matches_training_summary_snapshot": True,
            "artifact": _snapshot(
                checkpoint["path"],
                sha256=checkpoint["sha256"],
                size_bytes=checkpoint["size_bytes"],
            ),
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


class Harness:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.run_dir = (
            root / "output" / "benchmarks" / "selection" / ATTEMPT / f"seed_{SEED}"
        )
        self.run_dir.mkdir(parents=True)
        (root / "output" / "udlm" / "training-r").mkdir(parents=True)
        (root / "experiments" / "udlm").mkdir(parents=True)
        self.summary_path = self.run_dir / "summary.json"
        self.raw_path = self.run_dir / "raw_samples.csv"
        self.receipt_path = (
            root / "output" / "udlm" / "training-r" / "pilot_exit_status.json"
        )
        self.output = (
            root / "experiments" / "udlm" / "pilots" / ATTEMPT / f"seed_{SEED}.json"
        )
        training_root = root / "output" / "udlm" / "training-r"
        checkpoint_path = training_root / "checkpoints" / "500.ckpt"
        checkpoint_path.parent.mkdir()
        checkpoint_path.write_bytes(b"checkpoint bytes")
        checkpoint_snapshot = _live_snapshot(checkpoint_path)
        self.checkpoint = {
            "path": str(checkpoint_path),
            "sha256": checkpoint_snapshot["sha256"],
            "size_bytes": checkpoint_snapshot["size_bytes"],
            "global_step": 500,
            "extra_producer_field": True,
        }
        self.summary_path.write_bytes(
            _bytes({"schema_version": 7, "seed": SEED, "num_samples": 256})
        )
        self.raw_path.write_text("raw_model_text\nexample\n", encoding="utf-8")
        predecessor = {"state": "explicit_genesis_no_predecessor"}
        lock_record = {"owner": "test", "status": "held"}
        lock_bytes = (
            json.dumps(lock_record, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
        lock_sha = hashlib.sha256(lock_bytes).hexdigest()
        lock_path = root / "output" / "udlm" / ".single_training_job.lock"
        manifest_path = training_root / "launch_manifest.json"
        manifest_document = {
            "created_at": "2026-09-06T08:00:00+00:00",
            "predecessor_receipt_binding": predecessor,
            "single_training_job_lock": {
                "path": str(lock_path),
                "sha256": lock_sha,
                "record": lock_record,
            },
        }
        manifest_path.write_bytes(_bytes(manifest_document))
        manifest_snapshot = _live_snapshot(manifest_path)
        runtime_path = training_root / "runtime_config.json"
        runtime_document = {
            "resolved_training_config": {"training": "test"},
            "launch_manifest": {
                **manifest_snapshot,
                "selected_gpu_uuids": ["GPU-example"],
            },
        }
        runtime_path.write_bytes(_bytes(runtime_document))
        runtime_snapshot = _live_snapshot(runtime_path)
        training_summary_path = training_root / "training_summary.json"
        training_summary_document = {
            "completed_at_utc": "2026-09-06T08:30:00+00:00",
            "launch_manifest": {
                **manifest_snapshot,
                "selected_gpu_uuids": ["GPU-example"],
            },
            "runtime_config": {
                **runtime_snapshot,
                "schema_version": 2,
                "record_sha256": writer.canonical_json_sha256(runtime_document),
            },
            "final_checkpoint": {**checkpoint_snapshot, "semantic_audit": {}},
        }
        training_summary_path.write_bytes(_bytes(training_summary_document))
        training_summary_snapshot = _live_snapshot(training_summary_path)

        self.receipt = _successful_receipt(root, self.checkpoint)
        expected = self.receipt["expected_contract"]
        expected["launch_manifest_sha256"] = manifest_snapshot["sha256"]
        expected["training_job_lock_sha256"] = lock_sha
        self.receipt["predecessor_receipt_binding"] = predecessor
        self.receipt["launch_manifest"]["artifact"] = manifest_snapshot
        self.receipt["training_summary"]["artifact"] = training_summary_snapshot
        self.receipt["runtime_config"]["artifact"] = runtime_snapshot
        self.receipt["final_checkpoint"]["artifact"] = checkpoint_snapshot
        self.receipt["training_job_lock"]["expected_sha256"] = lock_sha
        self.receipt["training_job_lock"]["record"] = lock_record
        self.receipt["training_job_lock"]["artifact"] = {
            **_snapshot(str(lock_path), sha256=lock_sha, size_bytes=len(lock_bytes)),
            "link_count": 1,
        }
        self.receipt_path.write_bytes(_bytes(self.receipt))
        self.report_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.worker_calls: list[dict[str, Any]] = []
        self.training_validation_calls: list[dict[str, Any]] = []

    @property
    def summary_sha(self) -> str:
        return hashlib.sha256(self.summary_path.read_bytes()).hexdigest()

    @property
    def raw_sha(self) -> str:
        return hashlib.sha256(self.raw_path.read_bytes()).hexdigest()

    def structural(self) -> dict[str, Any]:
        return {
            "seed": SEED,
            "started_at_utc": "2026-09-06T10:00:00+00:00",
            "completed_at_utc": "2026-09-06T10:01:00+00:00",
            "run_dir": str(self.run_dir),
            "summary_path": str(self.summary_path),
            "summary_sha256": self.summary_sha,
            "raw_samples_path": str(self.raw_path),
            "raw_samples_sha256": self.raw_sha,
            "checkpoint": copy.deepcopy(self.checkpoint),
            "config": {"sha256": SHA_B},
            "metrics": {
                "released_comparable": {"quality": 0.8, "diversity": 0.7},
                "strict": {"quality": 0.75, "diversity": 0.69},
            },
            "failure_counts": {"released_decode_failed": 0},
            "git": {"commit": REVISION},
            "runner_sha256": SHA_C,
            "implementation_inputs": {
                "sampler_source": {"sha256": SHA_A},
                "ema_source": {"sha256": SHA_B},
            },
            "metric_inputs": {"sa": {"sha256": SHA_C}},
            "inference_weights": {"source": "ema", "ema_applied": True},
        }

    def report_validator(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.report_calls.append((args, kwargs))
        return self.structural()

    def rescore_worker(self, **kwargs: Any) -> dict[str, Any]:
        self.worker_calls.append(kwargs)
        structural = self.structural()
        return {
            "status": "exact_match",
            "seed": SEED,
            "summary_sha256": self.summary_sha,
            "raw_samples_sha256": self.raw_sha,
            "row_comparison": {"all_match": True, "field_count": 21},
            "metrics": structural["metrics"],
            "failure_counts": structural["failure_counts"],
            "identity": {
                "seed": SEED,
                "sample_count": 256,
                "started_at_utc": structural["started_at_utc"],
                "completed_at_utc": structural["completed_at_utc"],
                "checkpoint": {
                    key: structural["checkpoint"][key]
                    for key in ("path", "sha256", "size_bytes", "global_step")
                },
            },
            "independent_recomputation": {
                "raw_input_field": "raw_model_text",
                "qed_recomputed": True,
                "sa_recomputed": True,
                "released_diversity_recomputed": True,
                "strict_branch_recomputed": True,
                "all_21_raw_fields_compared": True,
            },
        }

    def training_artifact_validator(self, **kwargs: Any) -> dict[str, Any]:
        self.training_validation_calls.append(kwargs)
        return {
            "validated_bindings": {"valid": True},
            "predecessor_receipt_binding": kwargs["manifest"][
                "predecessor_receipt_binding"
            ],
        }

    def collect(self, **overrides: Any) -> dict[str, Any]:
        arguments = {
            "attempt_id": ATTEMPT,
            "candidate_id": CANDIDATE,
            "pilot_seed": SEED,
            "run_dir": self.run_dir,
            "training_exit_receipt": self.receipt_path,
            "output": self.output,
            "report_validator": self.report_validator,
            "rescore_worker": self.rescore_worker,
            "training_artifact_validator": self.training_artifact_validator,
        }
        arguments.update(overrides)
        return writer.collect_completed_pilot_evidence(**arguments)


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Harness:
    monkeypatch.setattr(writer, "REPOSITORY_ROOT", tmp_path)
    return Harness(tmp_path)


def test_collects_exact_reference_only_schema2_envelope(harness: Harness) -> None:
    envelope = harness.collect()

    assert json.loads(harness.output.read_bytes()) == envelope
    assert set(envelope) == {
        "schema_version",
        "artifact_kind",
        "status",
        "attempt_id",
        "candidate_id",
        "pilot_seed",
        "final_seed_results_included",
        "training_exit_receipt",
        "benchmark_artifacts",
    }
    assert envelope["schema_version"] == 2
    assert envelope["artifact_kind"] == "pilot_evaluation"
    assert envelope["status"] == "completed"
    assert envelope["final_seed_results_included"] is False
    assert "quality" not in envelope and "diversity" not in envelope
    assert envelope["benchmark_artifacts"]["summary_json"] == {
        "relative_path": f"output/benchmarks/selection/{ATTEMPT}/seed_{SEED}/summary.json",
        "sha256": harness.summary_sha,
        "schema_version": 7,
    }
    assert harness.report_calls == [
        (
            (harness.run_dir, SEED),
            {
                "expected_samples": 256,
                "expected_tier": "pilot",
                "final_protocol_eligible": False,
            },
        )
    ]
    worker = harness.worker_calls[0]
    assert worker["expected_seed"] == SEED
    assert worker["expected_sample_count"] == 256
    assert worker["expected_checkpoint_sha256"] == harness.checkpoint["sha256"]
    assert worker["expected_config_sha256"] == SHA_B
    assert worker["expected_source_revision"] == REVISION
    assert worker["allowed_root"] == harness.root


def test_public_training_receipt_validator_returns_timestamp_and_inputs(
    harness: Harness,
) -> None:
    recorded_at, inputs = writer.validate_successful_training_receipt(
        harness.receipt,
        receipt_path=harness.receipt_path,
        structural=harness.structural(),
        artifact_validator=harness.training_artifact_validator,
    )

    assert recorded_at.isoformat() == harness.receipt["recorded_at_utc"]
    assert [artifact.path.name for artifact in inputs] == [
        "training_summary.json",
        "launch_manifest.json",
        "runtime_config.json",
        "500.ckpt",
    ]


def test_accepts_checkpoint_inside_containing_project(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = harness.root.parent
    external_checkpoint = (
        project_root
        / f"{harness.root.name}-project-output"
        / "training-r"
        / "checkpoints"
        / "500.ckpt"
    )
    external_checkpoint.parent.mkdir(parents=True)
    external_checkpoint.write_bytes(b"external project checkpoint bytes")
    monkeypatch.setattr(writer, "_project_root", lambda: project_root)
    checkpoint_snapshot = writer.read_stable_regular_file(
        external_checkpoint,
        label="external training checkpoint",
        capture_payload=False,
        project_scope=True,
    ).snapshot()

    training_summary_path = harness.receipt_path.parent / "training_summary.json"
    training_summary = json.loads(training_summary_path.read_bytes())
    training_summary["final_checkpoint"] = {
        **checkpoint_snapshot,
        "semantic_audit": {},
    }
    training_summary_path.write_bytes(_bytes(training_summary))
    harness.receipt["training_summary"]["artifact"] = _live_snapshot(
        training_summary_path
    )
    harness.checkpoint.update(
        {
            "path": str(external_checkpoint),
            "sha256": checkpoint_snapshot["sha256"],
            "size_bytes": checkpoint_snapshot["size_bytes"],
        }
    )
    harness.receipt["expected_contract"]["final_checkpoint_path"] = str(
        external_checkpoint
    )
    harness.receipt["final_checkpoint"] = {
        "path": str(external_checkpoint),
        "present": True,
        "matches_training_summary_snapshot": True,
        "artifact": checkpoint_snapshot,
    }
    harness.receipt_path.write_bytes(_bytes(harness.receipt))

    envelope = harness.collect()

    assert envelope["status"] == "completed"


def test_never_clobbers_existing_output(harness: Harness) -> None:
    harness.output.parent.mkdir(parents=True)
    harness.output.write_bytes(b"preexisting\n")

    with pytest.raises(FileExistsError, match="refusing to replace"):
        harness.collect()

    assert harness.output.read_bytes() == b"preexisting\n"
    assert not harness.report_calls
    assert not harness.worker_calls


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 4),
        ("status", "failed"),
        ("overall_status", "failed"),
        ("process_exit_status", 1),
    ],
)
def test_rejects_non_successful_or_non_schema5_receipt(
    harness: Harness, field: str, value: object
) -> None:
    harness.receipt[field] = value
    harness.receipt_path.write_bytes(_bytes(harness.receipt))

    with pytest.raises(writer.PilotEvidenceError, match="successful schema-5"):
        harness.collect()

    assert not harness.output.exists()
    assert not harness.worker_calls


def test_rejects_receipt_checkpoint_or_chronology_mismatch(harness: Harness) -> None:
    harness.receipt["final_checkpoint"]["artifact"]["sha256"] = SHA_C
    harness.receipt_path.write_bytes(_bytes(harness.receipt))
    with pytest.raises(writer.PilotEvidenceError, match="checkpoint.*differs"):
        harness.collect()

    harness.receipt["final_checkpoint"]["artifact"]["sha256"] = harness.checkpoint[
        "sha256"
    ]
    harness.receipt["recorded_at_utc"] = "2026-09-06T10:00:00+00:00"
    harness.receipt_path.write_bytes(_bytes(harness.receipt))
    with pytest.raises(writer.PilotEvidenceError, match="must predate pilot start"):
        harness.collect()


def test_rejects_fabricated_receipt_predecessor_binding(harness: Harness) -> None:
    harness.receipt["predecessor_receipt_binding"] = {
        "state": "validated_successful_predecessor",
        "predecessor_run_name": "fabricated",
    }
    harness.receipt_path.write_bytes(_bytes(harness.receipt))

    with pytest.raises(writer.PilotEvidenceError, match="predecessor binding differs"):
        harness.collect()

    assert not harness.output.exists()
    assert not harness.worker_calls


@pytest.mark.parametrize(
    "relative_path",
    [
        "training_summary.json",
        "launch_manifest.json",
        "runtime_config.json",
        "checkpoints/500.ckpt",
    ],
)
def test_rejects_missing_live_training_artifact(
    harness: Harness, relative_path: str
) -> None:
    target = harness.receipt_path.parent / relative_path
    target.unlink()

    with pytest.raises(writer.PilotEvidenceError, match="is unavailable"):
        harness.collect()

    assert not harness.output.exists()
    assert not harness.worker_calls


@pytest.mark.parametrize(
    "relative_path",
    [
        "training_summary.json",
        "launch_manifest.json",
        "runtime_config.json",
        "checkpoints/500.ckpt",
    ],
)
def test_rejects_mismatched_live_training_artifact(
    harness: Harness, relative_path: str
) -> None:
    target = harness.receipt_path.parent / relative_path
    target.write_bytes(target.read_bytes() + b" \n")

    with pytest.raises(writer.PilotEvidenceError, match="differs"):
        harness.collect()

    assert not harness.output.exists()
    assert not harness.worker_calls


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda receipt: receipt["source_at_receipt"].__setitem__("head", "0" * 40),
            "pushed-source evidence",
        ),
        (
            lambda receipt: receipt["launch_manifest"].__setitem__(
                "matches_runtime_config_snapshot", False
            ),
            "matches_runtime_config_snapshot",
        ),
        (
            lambda receipt: receipt["runtime_config"].__setitem__(
                "semantic_validation_passed", False
            ),
            "semantic_validation_passed",
        ),
        (
            lambda receipt: receipt["training_job_lock"].__setitem__(
                "matches_launch_manifest_binding", False
            ),
            "matches_launch_manifest_binding",
        ),
        (
            lambda receipt: receipt["training_summary"].__setitem__(
                "validation_error", "synthetic error"
            ),
            "recorded an error",
        ),
        (
            lambda receipt: receipt.__setitem__("predecessor_receipt_binding", None),
            "predecessor binding",
        ),
    ],
)
def test_rejects_incomplete_training_receipt_provenance(
    harness: Harness, mutator, message: str
) -> None:
    mutator(harness.receipt)
    harness.receipt_path.write_bytes(_bytes(harness.receipt))

    with pytest.raises(writer.PilotEvidenceError, match=message):
        harness.collect()

    assert not harness.output.exists()
    assert not harness.worker_calls


def test_rejects_wrong_output_and_run_identity(harness: Harness) -> None:
    wrong_output = harness.output.with_name("seed_1001.json")
    with pytest.raises(writer.PilotEvidenceError, match="output must be exactly"):
        harness.collect(output=wrong_output)

    wrong_run = harness.root / "output" / "benchmarks" / "other" / f"seed_{SEED}"
    wrong_run.mkdir(parents=True)
    with pytest.raises(writer.PilotEvidenceError, match="pilot run directory must"):
        harness.collect(run_dir=wrong_run)


@pytest.mark.parametrize("target_name", ["summary.json", "raw_samples.csv"])
def test_rejects_symlinked_run_inputs(harness: Harness, target_name: str) -> None:
    target = harness.run_dir / target_name
    real = target.with_name(f"real-{target.name}")
    target.rename(real)
    target.symlink_to(real.name)

    with pytest.raises(writer.PilotEvidenceError, match="symlink"):
        harness.collect()

    assert not harness.output.exists()


def test_rejects_incomplete_or_metric_mismatched_rescore(harness: Harness) -> None:
    def mismatched(**kwargs: Any) -> dict[str, Any]:
        result = harness.rescore_worker(**kwargs)
        result["metrics"] = {"fabricated": True}
        return result

    with pytest.raises(writer.PilotEvidenceError, match="rescore differs"):
        harness.collect(rescore_worker=mismatched)

    assert not harness.output.exists()


def test_revalidates_inputs_after_worker_before_publication(harness: Harness) -> None:
    def mutating_worker(**kwargs: Any) -> dict[str, Any]:
        result = harness.rescore_worker(**kwargs)
        harness.raw_path.write_bytes(harness.raw_path.read_bytes() + b"changed\n")
        return result

    with pytest.raises(
        writer.PilotEvidenceError, match="input changed before publication"
    ):
        harness.collect(rescore_worker=mutating_worker)

    assert not harness.output.exists()


def test_atomic_publication_loses_race_without_clobber(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_link = os.link

    def racing_link(source: Path, destination: Path, **kwargs: Any) -> None:
        destination.write_bytes(b"racer\n")
        real_link(source, destination, **kwargs)

    monkeypatch.setattr(writer.os, "link", racing_link)
    with pytest.raises(FileExistsError, match="concurrently created"):
        harness.collect()
    assert harness.output.read_bytes() == b"racer\n"


def test_cli_help_is_gpu_inert_and_lists_exact_interface() -> None:
    source = Path(writer.__file__).resolve()
    result = subprocess.run(
        [sys.executable, "-S", str(source), "--help"],
        cwd=source.parents[2],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    for flag in (
        "--outcome",
        "--attempt-id",
        "--candidate-id",
        "--pilot-seed",
        "--run-dir",
        "--training-exit-receipt",
        "--failure-receipt",
        "--output",
    ):
        assert flag in result.stdout


def test_direct_completed_cli_bootstraps_repository_imports(tmp_path: Path) -> None:
    source = Path(writer.__file__).resolve()
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            str(source),
            "--outcome",
            "completed",
            "--attempt-id",
            "direct-cli",
            "--candidate-id",
            "candidate-cli",
            "--pilot-seed",
            "1000",
            "--run-dir",
            "output/benchmarks/missing/direct-cli/seed_1000",
            "--training-exit-receipt",
            "output/udlm/missing/pilot_exit_status.json",
            "--output",
            "experiments/udlm/pilots/direct-cli/seed_1000.json",
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "ModuleNotFoundError" not in result.stderr
    assert "is unavailable" in result.stderr


class FailureHarness:
    """A live launcher-produced failed-pilot receipt and its support files."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.output_root = root / "output" / "benchmarks" / "selection" / ATTEMPT
        self.run_dir = self.output_root / f"seed_{SEED}"
        self.run_dir.mkdir(parents=True)
        (root / "experiments" / "udlm").mkdir(parents=True)
        (root / ".venv" / "bin").mkdir(parents=True)
        (root / ".venv" / "bin" / "python").write_bytes(b"python fixture\n")

        launcher_source = root / "scripts" / "exps" / "denovo" / "launch_benchmark.py"
        launcher_source.parent.mkdir(parents=True)
        launcher_source.write_bytes(b"# launcher fixture\n")
        (launcher_source.parent / "benchmark.py").write_bytes(b"# benchmark fixture\n")
        self.launcher_source = launcher_source

        self.checkpoint_path = root / "output" / "udlm" / "candidate" / "500.ckpt"
        self.checkpoint_path.parent.mkdir(parents=True)
        self.checkpoint_path.write_bytes(b"checkpoint fixture\n")
        self.config_path = root / "scripts" / "exps" / "denovo" / "pilot.yaml"
        self.config_path.write_text(
            "diffusion_type: udlm\n"
            "softmax_temp: 1.0\n"
            "randomness: 0.5\n"
            "min_add_len: 40\n"
            "num_steps: 128\n"
            "inference_eps: 0.001\n"
            "exclude_special_tokens: true\n"
            "prior_variant: release_uniform\n"
            "prior_metadata_sha256: null\n",
            encoding="utf-8",
        )
        source_config = {
            "diffusion_type": "udlm",
            "softmax_temp": 1.0,
            "randomness": 0.5,
            "min_add_len": 40,
            "num_steps": 128,
            "inference_eps": 0.001,
            "exclude_special_tokens": True,
            "prior_variant": "release_uniform",
            "prior_metadata_sha256": None,
        }
        sampling = launcher.benchmark_runner.validate_sampling_config(source_config)
        self.expected = launcher.ExpectedRunIdentity(
            checkpoint_path=self.checkpoint_path,
            checkpoint_sha256=launcher._sha256_file(self.checkpoint_path),
            checkpoint_global_step=500,
            checkpoint_size_bytes=self.checkpoint_path.stat().st_size,
            checkpoint_diffusion_type="udlm",
            checkpoint_udlm_inference_eps=0.001,
            checkpoint_udlm_exclude_special_tokens=True,
            checkpoint_udlm_prior_variant="release_uniform",
            checkpoint_udlm_prior_metadata=None,
            checkpoint_udlm_prior_metadata_sha256=None,
            config_path=self.config_path,
            source_config=source_config,
            source_config_sha256=launcher._sha256_file(self.config_path),
            config_git_tracking=None,
            sampling_config=sampling,
            sampling_config_sha256=launcher._canonical_json_sha256(sampling),
            effective_config=source_config,
            effective_config_sha256=launcher._canonical_json_sha256(source_config),
            benchmark_runner_sha256=SHA_A,
            implementation_inputs={},
            metric_inputs={},
            num_samples=256,
            source_revision=REVISION,
        )

        run_label = launcher.benchmark_runner.benchmark_run_label(
            self.expected.checkpoint_global_step,
            self.expected.checkpoint_sha256,
            SEED,
        )
        self.log_path = (
            root / "output" / "logs" / "selection" / ATTEMPT / f"{run_label}.log"
        )
        self.log_path.parent.mkdir(parents=True)
        self.log_path.write_bytes(b'{"event":"launch"}\nsynthetic failure\n')
        self.summary_path = self.run_dir / "summary.json"
        self.raw_path = self.run_dir / "raw_samples.csv"
        self.summary_path.write_bytes(b'{"partial":true}\n')
        self.raw_path.write_bytes(b"sample_index,raw_model_text\n0,partial\n")
        command = launcher._command(
            checkpoint=self.expected.checkpoint_path,
            expected_checkpoint_sha256=self.expected.checkpoint_sha256,
            expected_source_revision=REVISION,
            config=self.expected.config_path,
            expected_config_sha256=self.expected.source_config_sha256,
            num_samples=self.expected.num_samples,
            seed=SEED,
            output_dir=self.run_dir,
        )
        self.job = launcher.RunningJob(
            seed=SEED,
            gpu=launcher.GPUState(
                index=2,
                uuid="GPU-failure-fixture",
                name="fixture",
                memory_used_mib=0,
                memory_total_mib=48_000,
                utilization_percent=0,
                compute_mode="Default",
                compute_processes=(),
            ),
            process=None,  # type: ignore[arg-type]
            log_handle=None,
            log_path=self.log_path,
            command=tuple(command),
            started_at_utc="2026-09-06T10:00:00+00:00",
        )
        self.receipt_path = launcher._write_pilot_failure_receipt(
            output_root=self.output_root,
            job=self.job,
            expected=self.expected,
            attempt_id=ATTEMPT,
            candidate_id=CANDIDATE,
            pilot_mode="registered_selection",
            source_revision={"head": REVISION, "upstream": REVISION},
            stage="benchmark_child_process",
            reason="synthetic benchmark child exited with status 17",
            process_exit_status=17,
            failed_at_utc="2026-09-06T10:01:00+00:00",
        )
        self.receipt = json.loads(self.receipt_path.read_bytes())
        self.output = (
            root / "experiments" / "udlm" / "pilots" / ATTEMPT / f"seed_{SEED}.json"
        )

    def write_receipt(self) -> None:
        self.receipt_path.write_bytes(_bytes(self.receipt))

    def collect(self, **overrides: Any) -> dict[str, Any]:
        arguments = {
            "attempt_id": ATTEMPT,
            "candidate_id": CANDIDATE,
            "pilot_seed": SEED,
            "failure_receipt": self.receipt_path,
            "output": self.output,
        }
        arguments.update(overrides)
        return writer.collect_failed_pilot_evidence(**arguments)


@pytest.fixture
def failure_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FailureHarness:
    monkeypatch.setattr(writer, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        launcher,
        "__file__",
        str(tmp_path / "scripts" / "exps" / "denovo" / "launch_benchmark.py"),
    )
    return FailureHarness(tmp_path)


def test_collects_launcher_failure_into_exact_reference_only_envelope(
    failure_harness: FailureHarness,
) -> None:
    envelope = failure_harness.collect()

    assert json.loads(failure_harness.output.read_bytes()) == envelope
    assert set(envelope) == {
        "schema_version",
        "artifact_kind",
        "status",
        "attempt_id",
        "candidate_id",
        "pilot_seed",
        "final_seed_results_included",
        "failure_receipt",
    }
    assert envelope == {
        "schema_version": 2,
        "artifact_kind": "pilot_failure",
        "status": "failed",
        "attempt_id": ATTEMPT,
        "candidate_id": CANDIDATE,
        "pilot_seed": SEED,
        "final_seed_results_included": False,
        "failure_receipt": {
            "relative_path": (
                f"output/benchmarks/selection/{ATTEMPT}/seed_{SEED}/"
                "failure_receipt.json"
            ),
            "sha256": hashlib.sha256(
                failure_harness.receipt_path.read_bytes()
            ).hexdigest(),
            "schema_version": 1,
        },
    }
    assert "reason" not in envelope
    assert "log" not in envelope
    assert "partial_artifacts" not in envelope


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda receipt, harness: receipt.__setitem__(
                "artifact_kind", "pilot_evaluation"
            ),
            "artifact_kind differs",
        ),
        (
            lambda receipt, harness: receipt.__setitem__("requested_samples", 100),
            "invalid seed/sample count",
        ),
        (
            lambda receipt, harness: receipt.__setitem__("process_exit_status", 0),
            "nonzero exit status",
        ),
        (
            lambda receipt, harness: receipt["source_revision"].__setitem__(
                "upstream", "0" * 40
            ),
            "not clean and pushed",
        ),
        (
            lambda receipt, harness: receipt["config"].__setitem__(
                "sampling_sha256", "0" * 64
            ),
            "sampling digest is not canonical",
        ),
        (
            lambda receipt, harness: receipt["command"].__setitem__(
                -1, str(harness.run_dir.parent / "wrong-seed")
            ),
            "exact producer launch",
        ),
        (
            lambda receipt, harness: receipt["log"].__setitem__("sha256", "0" * 64),
            "bytes differ",
        ),
        (
            lambda receipt, harness: receipt["partial_artifacts"][
                "summary_json"
            ].__setitem__("path", str(harness.run_dir / "wrong.json")),
            "producer path",
        ),
    ],
)
def test_rejects_adversarial_failure_receipt_claims(
    failure_harness: FailureHarness, mutator, message: str
) -> None:
    mutator(failure_harness.receipt, failure_harness)
    failure_harness.write_receipt()

    with pytest.raises(writer.PilotEvidenceError, match=message):
        failure_harness.collect()

    assert not failure_harness.output.exists()


@pytest.mark.parametrize(
    "target_name", ["checkpoint", "config", "launcher", "log", "partial"]
)
def test_rejects_changed_or_symlinked_failure_support(
    failure_harness: FailureHarness, target_name: str
) -> None:
    targets = {
        "checkpoint": failure_harness.checkpoint_path,
        "config": failure_harness.config_path,
        "launcher": failure_harness.launcher_source,
        "log": failure_harness.log_path,
        "partial": failure_harness.raw_path,
    }
    target = targets[target_name]
    if target_name == "log":
        real = target.with_name("real-failure.log")
        target.rename(real)
        target.symlink_to(real.name)
        message = "symlink"
    else:
        target.write_bytes(target.read_bytes() + b"changed\n")
        message = "differ"

    with pytest.raises(writer.PilotEvidenceError, match=message):
        failure_harness.collect()

    assert not failure_harness.output.exists()


def test_failure_collector_rejects_non_json_serializable_yaml(
    failure_harness: FailureHarness,
) -> None:
    failure_harness.config_path.write_text(
        failure_harness.config_path.read_text(encoding="utf-8")
        + "non_json_date: 2026-09-06\n",
        encoding="utf-8",
    )
    config_sha256 = hashlib.sha256(failure_harness.config_path.read_bytes()).hexdigest()
    failure_harness.receipt["config"]["sha256"] = config_sha256
    failure_harness.receipt["command"][11] = config_sha256
    failure_harness.write_receipt()

    with pytest.raises(writer.PilotEvidenceError, match="JSON-serializable"):
        failure_harness.collect()

    assert not failure_harness.output.exists()


def test_failure_collector_accepts_hashed_empty_partial_artifact(
    failure_harness: FailureHarness,
) -> None:
    failure_harness.raw_path.write_bytes(b"")
    raw_reference = failure_harness.receipt["partial_artifacts"]["raw_samples_csv"]
    raw_reference["sha256"] = hashlib.sha256(b"").hexdigest()
    raw_reference["size_bytes"] = 0
    failure_harness.write_receipt()

    envelope = failure_harness.collect()

    assert envelope["status"] == "failed"


def test_failure_collector_accepts_hashed_empty_log(
    failure_harness: FailureHarness,
) -> None:
    failure_harness.log_path.write_bytes(b"")
    failure_harness.receipt["log"]["sha256"] = hashlib.sha256(b"").hexdigest()
    failure_harness.receipt["log"]["size_bytes"] = 0
    failure_harness.write_receipt()

    envelope = failure_harness.collect()

    assert envelope["status"] == "failed"


def test_failure_collector_rejects_log_inside_output_attempt_root(
    failure_harness: FailureHarness,
) -> None:
    forged_log = failure_harness.output_root / failure_harness.log_path.name
    payload = failure_harness.log_path.read_bytes()
    forged_log.write_bytes(payload)
    failure_harness.receipt["log"] = {
        "path": str(forged_log),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    failure_harness.write_receipt()

    with pytest.raises(writer.PilotEvidenceError, match="distinct attempt root"):
        failure_harness.collect()

    assert not failure_harness.output.exists()


def test_failure_collector_rechecks_support_before_publication(
    failure_harness: FailureHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_revalidate = writer.revalidate_inputs

    def mutate_then_revalidate(inputs) -> None:
        failure_harness.log_path.write_bytes(b"changed before publication\n")
        real_revalidate(inputs)

    monkeypatch.setattr(writer, "revalidate_inputs", mutate_then_revalidate)
    with pytest.raises(writer.PilotEvidenceError, match="input changed"):
        failure_harness.collect()

    assert not failure_harness.output.exists()


def test_failure_collector_never_clobbers_existing_envelope(
    failure_harness: FailureHarness,
) -> None:
    failure_harness.output.parent.mkdir(parents=True)
    failure_harness.output.write_bytes(b"preexisting\n")

    with pytest.raises(FileExistsError, match="refusing to replace"):
        failure_harness.collect()

    assert failure_harness.output.read_bytes() == b"preexisting\n"


def test_cli_failed_mode_dispatch_and_exclusive_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        writer,
        "collect_failed_pilot_evidence",
        lambda **kwargs: calls.append(kwargs),
    )
    output = tmp_path / "evidence.json"
    receipt = tmp_path / "failure_receipt.json"
    arguments = [
        "--outcome",
        "failed",
        "--attempt-id",
        ATTEMPT,
        "--candidate-id",
        CANDIDATE,
        "--pilot-seed",
        str(SEED),
        "--failure-receipt",
        str(receipt),
        "--output",
        str(output),
    ]

    assert writer.main(arguments) == 0
    assert calls == [
        {
            "attempt_id": ATTEMPT,
            "candidate_id": CANDIDATE,
            "pilot_seed": SEED,
            "failure_receipt": receipt,
            "output": output,
        }
    ]
    with pytest.raises(SystemExit):
        writer.main(arguments + ["--run-dir", str(tmp_path / "run")])

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.exps.denovo import benchmark
from scripts.udlm import rescore_denovo_run as rescore


SEED = 1000
TEXTS = ["mol-a", "mol-b", "mol-a"]


def test_direct_cli_help_bootstraps_repository_imports(tmp_path: Path) -> None:
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(Path(rescore.__file__).resolve()), "--help"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--summary-path" in result.stdout


def _qed(smiles):
    return [0.8 if value == "mol-a" else 0.7 for value in smiles]


def _sa(smiles):
    return [3.0 if value == "mol-a" else 3.5 for value in smiles]


def _diversity(smiles):
    return 0.6 + 0.01 * len(smiles)


def _decode(raw_model_texts, *, use_bracket_safe):
    assert use_bracket_safe is False
    return benchmark.decode_records(
        raw_model_texts,
        use_bracket_safe=False,
        strict_decoder=lambda value: value,
        released_decoder=lambda value: value,
    )


def _decode_no_valid_molecules(raw_model_texts, *, use_bracket_safe):
    assert use_bracket_safe is False
    return benchmark.decode_records(
        raw_model_texts,
        use_bracket_safe=False,
        strict_decoder=lambda _value: None,
        released_decoder=lambda _value: None,
    )


def _json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _csv_bytes(records) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=benchmark.RAW_SAMPLE_FIELDS)
    writer.writeheader()
    writer.writerows(records)
    return handle.getvalue().encode("utf-8")


def _rewrite_csv(payload: bytes, *, row: int, field: str, value: str) -> bytes:
    with io.StringIO(payload.decode("utf-8"), newline="") as handle:
        records = list(csv.DictReader(handle))
    records[row][field] = value
    return _csv_bytes(records)


def _fixture() -> dict:
    records = _decode(TEXTS, use_bracket_safe=False)
    metrics, failures = benchmark.evaluate_records(
        records,
        requested_count=len(TEXTS),
        oracle_qed=_qed,
        oracle_sa=_sa,
        diversity_evaluator=_diversity,
    )
    raw_payload = _csv_bytes(records)
    raw_sha256 = hashlib.sha256(raw_payload).hexdigest()

    checkpoint_path = "/project/output/pilot/checkpoints/3.ckpt"
    checkpoint_sha256 = "d" * 64
    config_sha256 = "c" * 64
    source_revision = "a" * 40
    runner_sha256 = "b" * 64
    sampling = {
        "diffusion_type": "udlm",
        "softmax_temp": 0.5,
        "randomness": 0.5,
        "min_add_len": 40,
        "num_steps": 128,
        "inference_eps": 1e-5,
        "exclude_special_tokens": False,
        "prior_variant": "release_uniform",
        "prior_metadata_sha256": None,
    }
    source_config = dict(sampling)
    effective = {
        **source_config,
        "model_path": checkpoint_path,
        "num_samples": len(TEXTS),
        "device": "cuda:0",
    }
    inference_weights = {
        "source": "ema",
        "ema_applied": True,
        "ema": {
            "shadow_parameter_count": 2,
            "decay": 0.999,
            "num_updates": 3,
        },
    }
    implementation_inputs = {}
    for index, name in enumerate(sorted(benchmark.IMPLEMENTATION_INPUT_PATHS), start=1):
        implementation_inputs[name] = {
            "path": f"/project/{name}",
            "sha256": f"{index:x}"[-1] * 64,
            "size_bytes": index,
        }
    metric_inputs = {"schema_version": 1, "fixture": "pinned-test-inputs"}
    summary = {
        "schema_version": benchmark.SCHEMA_VERSION,
        "status": "completed",
        "seed": SEED,
        "num_samples": len(TEXTS),
        "run": {
            "seed": SEED,
            "requested_sample_count": len(TEXTS),
            "evaluation_tier": "pilot",
            "final_protocol_eligible": False,
            "started_at_utc": "2026-09-06T10:00:00+00:00",
            "completed_at_utc": "2026-09-06T10:01:00+00:00",
            "one_seed_per_invocation": True,
            "single_generation_batch": True,
            "generation_protocol": {
                "diffusion_type": "udlm",
                "nfe": 128,
                "nfe_definition": (
                    "one full backbone forward evaluation per reverse step"
                ),
                "num_steps": 128,
                "num_steps_source": "explicit UDLM reverse-transition count",
                "inference_eps": 1e-5,
                "exclude_special_tokens": False,
                "prior_variant": "release_uniform",
                "prior_metadata_sha256": None,
                "temperature": 0.5,
                "randomness": 0.5,
                "randomness_used_by_sampler": False,
                "model_use_bracket_safe": False,
                "single_generation_batch": True,
                "released_safe_fix": True,
                "released_largest_component": "maximum SMILES string length",
                "strict_safe_fix": False,
                "inference_weights": inference_weights,
            },
            "command": ["/project/.venv/bin/python", "benchmark.py"],
            "seed_configuration": {
                "seed": SEED,
                "seed_applied_immediately_before_generation": True,
                "python_random": True,
                "numpy": True,
                "torch_cpu": True,
                "torch_cuda_all": True,
                "python_hash_seed": str(SEED),
            },
        },
        "checkpoint": {
            "path": checkpoint_path,
            "sha256": checkpoint_sha256,
            "size_bytes": 123,
            "mtime_utc": "2026-09-06T09:00:00+00:00",
            "byte_identity_verified_before_and_after_load": True,
            "global_step": 3,
            "epoch": 0,
            "diffusion_type": "udlm",
            "udlm_inference_eps": 1e-5,
            "udlm_exclude_special_tokens": False,
            "udlm_prior_variant": "release_uniform",
            "udlm_prior_metadata": None,
            "udlm_prior_metadata_sha256": None,
        },
        "config": {
            "path": "/project/config.yaml",
            "sha256": config_sha256,
            "git_tracking": {
                "path": "/project/config.yaml",
                "relative_path": "config.yaml",
                "source_revision": source_revision,
                "sha256": config_sha256,
                "tracked_at_source_revision": True,
            },
            "sampling_sha256": rescore.baseline_rescore.canonical_json_sha256(sampling),
            "effective_sha256": rescore.baseline_rescore.canonical_json_sha256(
                effective
            ),
            "source": source_config,
            "effective": effective,
            "sampling": sampling,
        },
        "metrics": metrics,
        "failure_counts": failures,
        "runtime_seconds": {},
        "environment": {},
        "git": {
            "repo_root": "/project",
            "commit": source_revision,
            "branch": "codex/fixture",
            "remote_origin": "git@example.invalid:fixture.git",
            "dirty": False,
            "status_porcelain": [],
            "runner_sha256": runner_sha256,
            "upstream": source_revision,
            "expected_source_revision": source_revision,
            "clean_pushed_source_verified_before_and_after_run": True,
        },
        "implementation_inputs": implementation_inputs,
        "metric_inputs": metric_inputs,
        "tokenizer": {},
        "artifacts": {
            "raw_samples_csv": {
                "path": "/project/output/pilot/raw_samples.csv",
                "sha256": raw_sha256,
                "row_count": len(TEXTS),
                "fields": list(benchmark.RAW_SAMPLE_FIELDS),
            },
            "summary_json": {"path": "/project/output/pilot/summary.json"},
        },
    }
    summary_payload = _json_bytes(summary)
    return {
        "summary": summary,
        "summary_payload": summary_payload,
        "summary_sha256": hashlib.sha256(summary_payload).hexdigest(),
        "raw_payload": raw_payload,
        "raw_sha256": raw_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config_sha256,
        "source_revision": source_revision,
        "runner_sha256": runner_sha256,
        "sampler_source_sha256": implementation_inputs["sampler_source"]["sha256"],
        "ema_source_sha256": implementation_inputs["ema_source"]["sha256"],
        "implementation_inputs_sha256": (
            rescore.baseline_rescore.canonical_json_sha256(implementation_inputs)
        ),
        "metric_inputs_sha256": rescore.baseline_rescore.canonical_json_sha256(
            metric_inputs
        ),
    }


def _rescore(fixture: dict, **overrides):
    arguments = {
        "summary_payload": fixture["summary_payload"],
        "raw_samples_payload": fixture["raw_payload"],
        "expected_summary_sha256": fixture["summary_sha256"],
        "expected_raw_samples_sha256": fixture["raw_sha256"],
        "expected_seed": SEED,
        "expected_sample_count": len(TEXTS),
        "decode_function": _decode,
        "evaluate_function": benchmark.evaluate_records,
        "oracle_qed": _qed,
        "oracle_sa": _sa,
        "diversity_evaluator": _diversity,
        "expected_checkpoint_sha256": fixture["checkpoint_sha256"],
        "expected_config_sha256": fixture["config_sha256"],
        "expected_source_revision": fixture["source_revision"],
        "expected_runner_sha256": fixture["runner_sha256"],
        "expected_sampler_source_sha256": fixture["sampler_source_sha256"],
        "expected_ema_source_sha256": fixture["ema_source_sha256"],
        "expected_implementation_inputs_sha256": fixture[
            "implementation_inputs_sha256"
        ],
        "expected_metric_inputs_sha256": fixture["metric_inputs_sha256"],
    }
    arguments.update(overrides)
    return rescore.rescore_denovo_run(**arguments)


def _replace_raw(fixture: dict, raw_payload: bytes) -> dict:
    changed = copy.deepcopy(fixture)
    changed["raw_payload"] = raw_payload
    changed["raw_sha256"] = hashlib.sha256(raw_payload).hexdigest()
    changed["summary"]["artifacts"]["raw_samples_csv"]["sha256"] = changed["raw_sha256"]
    changed["summary_payload"] = _json_bytes(changed["summary"])
    changed["summary_sha256"] = hashlib.sha256(changed["summary_payload"]).hexdigest()
    return changed


def test_valid_small_schema7_fixture_is_independently_rescored():
    fixture = _fixture()

    result = _rescore(fixture)

    assert result["status"] == "exact_match"
    assert result["seed"] == SEED
    assert result["row_comparison"]["field_count"] == 21
    assert result["row_comparison"]["cell_count"] == 21 * len(TEXTS)
    assert result["metrics"]["released_comparable"]["quality"] == 2 / 3
    assert result["metrics"]["released_comparable"]["diversity"] == 0.62
    assert result["identity"]["checkpoint"]["sha256"] == fixture["checkpoint_sha256"]
    assert result["identity"]["config"]["sha256"] == fixture["config_sha256"]
    assert result["identity"]["source"]["revision"] == fixture["source_revision"]
    assert result["independent_recomputation"]["all_21_raw_fields_compared"] is True


def test_valid_zero_unique_run_accepts_undefined_diversity_sentinel():
    fixture = _fixture()
    records = _decode_no_valid_molecules(TEXTS, use_bracket_safe=False)
    metrics, failures = benchmark.evaluate_records(
        records,
        requested_count=len(TEXTS),
        oracle_qed=_qed,
        oracle_sa=_sa,
        diversity_evaluator=_diversity,
    )
    fixture["summary"]["metrics"] = metrics
    fixture["summary"]["failure_counts"] = failures
    fixture = _replace_raw(fixture, _csv_bytes(records))

    result = _rescore(
        fixture,
        decode_function=_decode_no_valid_molecules,
    )

    for branch_name in ("released_comparable", "strict"):
        branch = result["metrics"][branch_name]
        assert branch["unique_count"] == 0
        assert branch["diversity"] is None
        assert branch["diversity_undefined_reason"] == "no_unique_valid_molecules"


def test_rescore_rejects_raw_score_tampering_even_when_hashes_are_updated():
    fixture = _fixture()
    tampered_raw = _rewrite_csv(
        fixture["raw_payload"], row=0, field="released_qed", value="0.99"
    )
    fixture = _replace_raw(fixture, tampered_raw)

    with pytest.raises(rescore.RescoreValidationError, match="released_qed"):
        _rescore(fixture)


def test_rescore_rejects_summary_diversity_tampering_with_new_summary_hash():
    fixture = _fixture()
    fixture["summary"]["metrics"]["released_comparable"]["diversity"] = 0.99
    fixture["summary_payload"] = _json_bytes(fixture["summary"])
    fixture["summary_sha256"] = hashlib.sha256(fixture["summary_payload"]).hexdigest()

    with pytest.raises(rescore.RescoreValidationError, match="diversity"):
        _rescore(fixture)


def test_rescore_rejects_raw_model_text_tampering_with_updated_bindings():
    fixture = _fixture()
    tampered_raw = _rewrite_csv(
        fixture["raw_payload"], row=0, field="raw_model_text", value="mol-c"
    )
    fixture = _replace_raw(fixture, tampered_raw)

    with pytest.raises(rescore.RescoreValidationError, match="raw_safe|smiles"):
        _rescore(fixture)


def test_rescore_rejects_wrong_caller_pinned_hash():
    fixture = _fixture()

    with pytest.raises(rescore.RescoreValidationError, match="payload SHA-256"):
        _rescore(fixture, expected_raw_samples_sha256="0" * 64)


def test_stable_reader_rejects_symlinked_evidence(tmp_path: Path):
    fixture = _fixture()
    summary = tmp_path / "summary.json"
    raw = tmp_path / "raw_samples.csv"
    summary.write_bytes(fixture["summary_payload"])
    raw.write_bytes(fixture["raw_payload"])
    link = tmp_path / "summary-link.json"
    link.symlink_to(summary)

    with pytest.raises(rescore.RescoreValidationError, match="symlink"):
        rescore.read_run_artifacts(
            summary_path=link,
            raw_samples_path=raw,
            allowed_root=tmp_path,
            expected_summary_sha256=fixture["summary_sha256"],
            expected_raw_samples_sha256=fixture["raw_sha256"],
        )


def test_file_wrapper_requires_hash_seed_at_interpreter_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    for key, value in rescore.baseline_rescore.OFFLINE_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("PYTHONHASHSEED", str(SEED + 1))

    with pytest.raises(rescore.RescoreValidationError, match="interpreter start"):
        rescore.rescore_denovo_run_files(
            summary_path=tmp_path / "missing-summary.json",
            raw_samples_path=tmp_path / "missing-raw.csv",
            allowed_root=tmp_path,
            expected_summary_sha256="0" * 64,
            expected_raw_samples_sha256="1" * 64,
            expected_seed=SEED,
            expected_sample_count=len(TEXTS),
        )


def test_file_wrapper_rejects_internal_artifact_path_label_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    fixture = _fixture()
    summary_path = tmp_path / "summary.json"
    raw_path = tmp_path / "raw_samples.csv"
    summary_path.write_bytes(fixture["summary_payload"])
    raw_path.write_bytes(fixture["raw_payload"])
    monkeypatch.setattr(
        rescore.baseline_rescore,
        "validate_worker_environment",
        lambda seed: {"python_hash_seed": str(seed), "device": "cpu"},
    )

    with pytest.raises(
        rescore.RescoreValidationError,
        match="artifacts.summary_json.path differs from supplied resolved path",
    ):
        rescore.rescore_denovo_run_files(
            summary_path=summary_path,
            raw_samples_path=raw_path,
            allowed_root=tmp_path,
            expected_summary_sha256=fixture["summary_sha256"],
            expected_raw_samples_sha256=fixture["raw_sha256"],
            expected_seed=SEED,
            expected_sample_count=len(TEXTS),
        )


def test_invoker_starts_one_fresh_seed_specific_cpu_worker_per_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    fixture = _fixture()
    calls = []

    def fake_run(command, **kwargs):
        seed = int(command[command.index("--expected-seed") + 1])
        calls.append((command, kwargs))
        worker_result = {
            "status": "exact_match",
            "seed": seed,
            "summary_sha256": fixture["summary_sha256"],
            "raw_samples_sha256": fixture["raw_sha256"],
            "worker_environment": {
                "python_hash_seed": str(seed),
                "device": "cpu",
                "cuda_visible_devices": "",
                "nvidia_visible_devices": "",
            },
        }
        stdout = rescore.WORKER_RESULT_PREFIX + json.dumps(worker_result) + "\n"
        return rescore.subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(rescore.subprocess, "run", fake_run)

    for seed in (1000, 1001):
        result = rescore.invoke_rescore_worker(
            summary_path=Path("seed") / "summary.json",
            raw_samples_path=Path("seed") / "raw_samples.csv",
            allowed_root=tmp_path,
            expected_summary_sha256=fixture["summary_sha256"],
            expected_raw_samples_sha256=fixture["raw_sha256"],
            expected_seed=seed,
            expected_sample_count=len(TEXTS),
            expected_checkpoint_sha256=fixture["checkpoint_sha256"],
        )
        assert result["seed"] == seed

    assert len(calls) == 2
    for expected_seed, (command, kwargs) in zip((1000, 1001), calls, strict=True):
        assert command[0] == rescore.sys.executable
        assert command[1] == str(Path(rescore.__file__).resolve())
        assert "--expected-checkpoint-sha256" in command
        assert kwargs["cwd"] == rescore.REPOSITORY_ROOT
        assert kwargs["env"]["PYTHONHASHSEED"] == str(expected_seed)
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["env"]["NVIDIA_VISIBLE_DEVICES"] == ""
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["check"] is False

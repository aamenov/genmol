from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from scripts.exps.denovo import launch_benchmark as launcher


def _gpu(
    *,
    index: int = 2,
    uuid: str = "GPU-test-uuid",
    memory_used_mib: int = 20,
    memory_total_mib: int = 49_140,
    utilization_percent: int = 1,
    compute_mode: str = "Default",
    processes: tuple[dict[str, object], ...] = (),
) -> launcher.GPUState:
    return launcher.GPUState(
        index=index,
        uuid=uuid,
        name="NVIDIA RTX A6000",
        memory_used_mib=memory_used_mib,
        memory_total_mib=memory_total_mib,
        utilization_percent=utilization_percent,
        compute_mode=compute_mode,
        compute_processes=processes,
    )


def _expected(tmp_path: Path, *, num_samples: int = 3) -> launcher.ExpectedRunIdentity:
    checkpoint = (tmp_path / "model.ckpt").resolve()
    checkpoint.write_bytes(b"checkpoint fixture")
    config = (tmp_path / "config.yaml").resolve()
    config.write_text(
        "model_path: ignored.ckpt\n"
        "num_samples: 17\n"
        "softmax_temp: 0.5\n"
        "randomness: 0.5\n"
        "min_add_len: 40\n",
        encoding="utf-8",
    )
    source = {
        "model_path": "ignored.ckpt",
        "num_samples": 17,
        "softmax_temp": 0.5,
        "randomness": 0.5,
        "min_add_len": 40,
    }
    sampling = {
        "diffusion_type": "mdlm",
        "softmax_temp": 0.5,
        "randomness": 0.5,
        "min_add_len": 40,
        "num_steps": None,
        "inference_eps": None,
        "exclude_special_tokens": None,
    }
    effective = dict(source)
    effective.update(
        {
            "model_path": str(checkpoint),
            "num_samples": num_samples,
            "device": "cuda:0",
        }
    )
    implementation_inputs = {
        "sampler_source": {
            "path": str((tmp_path / "src/genmol/sampler.py").resolve()),
            "sha256": "b" * 64,
            "size_bytes": 123,
        },
        "length_distribution": {
            "path": str((tmp_path / "data/len.pk").resolve()),
            "sha256": "c" * 64,
            "size_bytes": 456,
            "count": 10,
            "minimum": 1,
            "median": 5.5,
            "maximum": 10,
        },
    }
    return launcher.ExpectedRunIdentity(
        checkpoint_path=checkpoint,
        checkpoint_sha256=launcher._sha256_file(checkpoint),
        checkpoint_global_step=50_000,
        checkpoint_size_bytes=checkpoint.stat().st_size,
        checkpoint_diffusion_type="mdlm",
        checkpoint_udlm_inference_eps=None,
        checkpoint_udlm_exclude_special_tokens=None,
        config_path=config,
        source_config=source,
        source_config_sha256=launcher._sha256_file(config),
        sampling_config=sampling,
        sampling_config_sha256=launcher._canonical_json_sha256(sampling),
        effective_config=effective,
        effective_config_sha256=launcher._canonical_json_sha256(effective),
        benchmark_runner_sha256="a" * 64,
        implementation_inputs=implementation_inputs,
        num_samples=num_samples,
    )


def _write_raw_csv(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=launcher.benchmark_runner.RAW_SAMPLE_FIELDS)
        writer.writeheader()
        for index in range(count):
            row = {field: "" for field in launcher.benchmark_runner.RAW_SAMPLE_FIELDS}
            row["sample_index"] = index
            row["raw_model_text"] = f"SAFE-{index}"
            writer.writerow(row)


def _write_matching_artifacts(
    output_root: Path,
    seed: int,
    expected: launcher.ExpectedRunIdentity,
) -> tuple[Path, Path]:
    run_dir = output_root / f"seed_{seed}"
    raw_path = run_dir / launcher.benchmark_runner.RAW_SAMPLES_FILENAME
    summary_path = run_dir / launcher.benchmark_runner.SUMMARY_FILENAME
    _write_raw_csv(raw_path, expected.num_samples)
    summary = {
        "schema_version": launcher.benchmark_runner.SCHEMA_VERSION,
        "status": "completed",
        "seed": seed,
        "num_samples": expected.num_samples,
        "run": {
            "seed": seed,
            "requested_sample_count": expected.num_samples,
            "evaluation_tier": (
                "final" if expected.num_samples == 1_000 else "pilot"
            ),
            "final_protocol_eligible": expected.num_samples == 1_000,
            "generation_protocol": {
                "diffusion_type": expected.sampling_config["diffusion_type"],
                "nfe": expected.sampling_config["num_steps"] or 2,
                "num_steps": expected.sampling_config["num_steps"],
                "inference_eps": expected.sampling_config["inference_eps"],
                "exclude_special_tokens": expected.sampling_config[
                    "exclude_special_tokens"
                ],
                "temperature": expected.sampling_config["softmax_temp"],
                "randomness": expected.sampling_config["randomness"],
                "randomness_used_by_sampler": (
                    expected.sampling_config["diffusion_type"] == "mdlm"
                ),
            },
        },
        "checkpoint": {
            "path": str(expected.checkpoint_path),
            "sha256": expected.checkpoint_sha256,
            "global_step": expected.checkpoint_global_step,
            "size_bytes": expected.checkpoint_size_bytes,
            "diffusion_type": expected.checkpoint_diffusion_type,
            "udlm_inference_eps": expected.checkpoint_udlm_inference_eps,
            "udlm_exclude_special_tokens": (
                expected.checkpoint_udlm_exclude_special_tokens
            ),
        },
        "config": {
            "path": str(expected.config_path),
            "sha256": expected.source_config_sha256,
            "source": expected.source_config,
            "sampling": expected.sampling_config,
            "sampling_sha256": expected.sampling_config_sha256,
            "effective": expected.effective_config,
            "effective_sha256": expected.effective_config_sha256,
        },
        "git": {"runner_sha256": expected.benchmark_runner_sha256},
        "implementation_inputs": expected.implementation_inputs,
        "artifacts": {
            "raw_samples_csv": {
                "path": str(raw_path.resolve()),
                "sha256": launcher._sha256_file(raw_path),
                "row_count": expected.num_samples,
                "fields": list(launcher.benchmark_runner.RAW_SAMPLE_FIELDS),
            },
            "summary_json": {"path": str(summary_path.resolve())},
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return raw_path, summary_path


def _read_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_summary(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_gpu_request_accepts_only_a_count_capped_at_two() -> None:
    required = [
        "--checkpoint",
        "model.ckpt",
        "--config",
        "config.yaml",
        "--seeds",
        "1",
        "--output-root",
        "runs",
        "--gpu-count",
        "2",
    ]
    parsed = launcher._parse_args(required)
    assert parsed.gpu_count == 2
    assert not hasattr(parsed, "gpu_indices")
    assert parsed.max_utilization_percent == 10
    assert parsed.min_free_memory_mib == 30_000
    assert launcher._validate_gpu_count(2) == 2

    with pytest.raises(ValueError, match="must be 1 or 2"):
        launcher._validate_gpu_count(3)
    with pytest.raises(ValueError, match="must be 1 or 2"):
        launcher._validate_gpu_count(True)
    with pytest.raises(SystemExit):
        launcher._parse_args([*required, "--gpu-indices", "4", "2"])


def test_sample_tier_requires_explicit_bounded_pilot() -> None:
    assert launcher._validate_sample_tier(1_000, pilot=False) == "final"
    assert launcher._validate_sample_tier(32, pilot=True) == "pilot"
    with pytest.raises(ValueError, match="exactly 1000"):
        launcher._validate_sample_tier(32, pilot=False)
    with pytest.raises(ValueError, match="capped at 100"):
        launcher._validate_sample_tier(101, pilot=True)
    with pytest.raises(ValueError, match="must be an integer"):
        launcher._validate_sample_tier(True, pilot=True)


def test_checkpoint_may_be_shared_from_project_but_not_escape_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    worktree = project / "run_sources" / "udlm"
    worktree.mkdir(parents=True)
    checkpoint = project / "outputs" / "model.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"model")
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", worktree)

    assert launcher._resolve_checkpoint(checkpoint) == checkpoint
    with pytest.raises(ValueError, match="escapes project root"):
        launcher._resolve_checkpoint(tmp_path / "outside.ckpt")


def test_source_revision_allows_only_output_and_requires_pushed_head() -> None:
    with mock.patch.object(
        launcher,
        "_git_text",
        side_effect=["?? output/run.json", "abc", "abc"],
    ):
        assert launcher._require_clean_pushed_source() == {
            "head": "abc",
            "upstream": "abc",
        }
    with mock.patch.object(
        launcher,
        "_git_text",
        return_value=" M src/genmol/model.py",
    ):
        with pytest.raises(RuntimeError, match="dirty outside output"):
            launcher._require_clean_pushed_source()
    with mock.patch.object(
        launcher,
        "_git_text",
        side_effect=["", "local", "remote"],
    ):
        with pytest.raises(RuntimeError, match="not the pushed upstream"):
            launcher._require_clean_pushed_source()


def test_snapshot_enumerates_and_probes_every_physical_gpu() -> None:
    with (
        mock.patch.object(launcher, "_physical_gpu_indices", return_value=[0, 2, 4]),
        mock.patch.object(
            launcher,
            "_probe_gpu",
            side_effect=lambda index: _gpu(index=index, uuid=f"GPU-{index}"),
        ) as probe,
    ):
        states = launcher._snapshot()

    assert [state.index for state in states] == [0, 2, 4]
    assert probe.call_args_list == [mock.call(0), mock.call(2), mock.call(4)]


def test_snapshot_refuses_partial_unverifiable_inventory() -> None:
    with (
        mock.patch.object(launcher, "_physical_gpu_indices", return_value=[0, 1]),
        mock.patch.object(
            launcher,
            "_probe_gpu",
            side_effect=[_gpu(index=0, uuid="GPU-0"), RuntimeError("query failed")],
        ),
        pytest.raises(RuntimeError, match="complete NVIDIA GPU inventory"),
    ):
        launcher._snapshot()


def test_selection_policy_schema_records_dynamic_full_inventory_semantics() -> None:
    assert launcher._selection_policy(
        requested_gpu_count=2,
        max_utilization_percent=10,
        min_free_memory_mib=30_000,
    ) == {
        "selection_method": "dynamic_idle_discovery",
        "inventory_scope": "all_nvidia_gpus",
        "requested_gpu_count": 2,
        "max_utilization_percent": 10,
        "utilization_comparison": "strictly_less_than",
        "min_free_memory_mib": 30_000,
        "active_compute_processes_allowed": False,
    }
    with pytest.raises(ValueError, match="must be 1 or 2"):
        launcher._selection_policy(
            requested_gpu_count=True,
            max_utilization_percent=10,
            min_free_memory_mib=30_000,
        )


def test_probe_gpu_parses_csv_and_rejects_compute_processes_under_policy() -> None:
    responses = [
        subprocess.CompletedProcess(
            [],
            0,
            '2, GPU-test-uuid, "NVIDIA RTX A6000", 23, 49140, 2, Default\n',
            "",
        ),
        subprocess.CompletedProcess(
            [],
            0,
            "GPU-test-uuid, 1234, /other/user/python, 1500\n",
            "",
        ),
    ]
    with mock.patch.object(launcher, "_run_nvidia_smi", side_effect=responses):
        state = launcher._probe_gpu(2)

    assert state.index == 2
    assert state.uuid == "GPU-test-uuid"
    assert state.name == "NVIDIA RTX A6000"
    assert state.memory_total_mib == 49_140
    assert state.compute_processes == (
        {
            "pid": 1234,
            "process_name": "/other/user/python",
            "used_memory_mib": 1500,
        },
    )
    assert not launcher._eligible(
        state,
        max_utilization_percent=15,
        min_free_memory_mib=30_000,
    )
    assert state.rejection_reasons(
        max_utilization_percent=15,
        min_free_memory_mib=30_000,
    ) == ["1 active compute process(es) detected"]


@pytest.mark.parametrize(
    "process_output",
    [
        (
            "GPU-test-uuid, 1234, /other/user/python, 1500\n"
            "GPU-test-uuid, 1234, /other/user/python, 1500\n"
        ),
        (
            "No running processes found\n"
            "GPU-test-uuid, 1234, /other/user/python, 1500\n"
        ),
        "GPU-different, 1234, /other/user/python, 1500\n",
    ],
)
def test_probe_gpu_rejects_duplicate_or_ambiguous_process_telemetry(
    process_output: str,
) -> None:
    responses = [
        subprocess.CompletedProcess(
            [],
            0,
            '2, GPU-test-uuid, "NVIDIA RTX A6000", 23, 49140, 2, Default\n',
            "",
        ),
        subprocess.CompletedProcess([], 0, process_output, ""),
    ]
    with (
        mock.patch.object(launcher, "_run_nvidia_smi", side_effect=responses),
        pytest.raises(RuntimeError, match="ambiguous|invalid"),
    ):
        launcher._probe_gpu("GPU-test-uuid")


def test_snapshot_rejects_duplicate_uuids_across_inventory() -> None:
    with (
        mock.patch.object(launcher, "_physical_gpu_indices", return_value=[0, 1]),
        mock.patch.object(
            launcher,
            "_probe_gpu",
            side_effect=[
                _gpu(index=0, uuid="GPU-same"),
                _gpu(index=1, uuid="GPU-same"),
            ],
        ),
        pytest.raises(RuntimeError, match="duplicate UUIDs"),
    ):
        launcher._snapshot()


def test_gpu_eligibility_enforces_free_memory_and_exclusive_utilization() -> None:
    assert launcher._eligible(
        _gpu(memory_used_mib=19_140, utilization_percent=14),
        max_utilization_percent=15,
        min_free_memory_mib=30_000,
    )
    for state in (
        _gpu(memory_used_mib=19_141),
        _gpu(utilization_percent=15),
        _gpu(compute_mode="Prohibited"),
    ):
        assert not launcher._eligible(
            state,
            max_utilization_percent=15,
            min_free_memory_mib=30_000,
        )


def test_final_probe_rechecks_policy_uuid_and_active_processes() -> None:
    candidate = _gpu()
    reached_threshold = _gpu(utilization_percent=15)
    with mock.patch.object(
        launcher, "_probe_gpu", return_value=reached_threshold
    ) as probe:
        selected, reasons = launcher._recheck_gpu_for_launch(
            candidate,
            max_utilization_percent=15,
            min_free_memory_mib=30_000,
        )
    assert selected is None
    assert any("not strictly below" in reason for reason in reasons)
    probe.assert_called_once_with(candidate.uuid)

    shared_but_below_threshold = _gpu(
        processes=(
            {"pid": 9, "process_name": "/other/python", "used_memory_mib": 20},
        )
    )
    with mock.patch.object(
        launcher,
        "_probe_gpu",
        return_value=shared_but_below_threshold,
    ):
        selected, reasons = launcher._recheck_gpu_for_launch(
            candidate,
            max_utilization_percent=15,
            min_free_memory_mib=30_000,
        )
    assert selected is None
    assert any("active compute process" in reason for reason in reasons)

    with mock.patch.object(
        launcher,
        "_probe_gpu",
        return_value=_gpu(uuid="GPU-replaced-at-same-index"),
    ):
        selected, reasons = launcher._recheck_gpu_for_launch(
            candidate,
            max_utilization_percent=15,
            min_free_memory_mib=30_000,
        )
    assert selected is None
    assert any("identity changed" in reason for reason in reasons)


def test_child_environment_maps_uuid_and_uses_logical_cuda_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setenv("UNRELATED", "preserved")
    hostile_main = "/hostile/main-checkout/src"
    monkeypatch.setenv("PYTHONPATH", hostile_main)
    selection = {"event": "launch", "physical_gpu": _gpu().as_dict()}
    environment = launcher._child_environment(
        seed=17,
        gpu=_gpu(),
        selection=selection,
        run_label="denovo-test",
    )

    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-test-uuid"
    assert environment["GENMOL_BENCHMARK_GPU_PHYSICAL_INDEX"] == "2"
    assert environment["GENMOL_BENCHMARK_GPU_UUID"] == "GPU-test-uuid"
    assert environment["PYTHONHASHSEED"] == "17"
    assert environment["UNRELATED"] == "preserved"
    assert environment["PYTHONPATH"].split(launcher.os.pathsep) == [
        str(tmp_path / "src"),
        str(tmp_path),
        hostile_main,
    ]
    assert json.loads(environment["GENMOL_BENCHMARK_GPU_SELECTION_SNAPSHOT"]) == selection
    command = launcher._command(
        checkpoint=tmp_path / "model.ckpt",
        config=tmp_path / "config.yaml",
        num_samples=1_000,
        seed=17,
        output_dir=tmp_path / "seed_17",
    )
    assert command[command.index("--device") + 1] == "cuda:0"
    assert "GPU-test-uuid" not in command


def test_child_environment_sets_worktree_pythonpath_when_unset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    environment = launcher._child_environment(
        seed=1,
        gpu=_gpu(),
        selection={"event": "launch"},
        run_label="denovo-test",
    )
    assert environment["PYTHONPATH"].split(launcher.os.pathsep) == [
        str(tmp_path / "src"),
        str(tmp_path),
    ]


def test_matching_artifacts_are_the_only_skippable_state(tmp_path: Path) -> None:
    expected = _expected(tmp_path)
    output_root = tmp_path / "runs"
    assert launcher._completed(output_root, 7, expected) is False
    _write_matching_artifacts(output_root, 7, expected)
    assert launcher._completed(output_root, 7, expected) is True


def test_expected_identity_uses_checkpoint_metadata_and_normalized_sampling(
    tmp_path: Path,
) -> None:
    expected_fixture = _expected(tmp_path)
    with (
        mock.patch.object(
            launcher.benchmark_runner,
            "checkpoint_metadata",
            return_value={
                "sha256": expected_fixture.checkpoint_sha256,
                "global_step": 50_000,
                "size_bytes": expected_fixture.checkpoint_size_bytes,
            },
        ),
        mock.patch.object(
            launcher.benchmark_runner,
            "implementation_input_provenance",
            return_value=expected_fixture.implementation_inputs,
        ),
    ):
        expected = launcher._build_expected_run_identity(
            expected_fixture.checkpoint_path,
            expected_fixture.config_path,
            expected_fixture.num_samples,
        )

    assert expected.checkpoint_global_step == 50_000
    assert expected.checkpoint_sha256 == expected_fixture.checkpoint_sha256
    assert expected.sampling_config == {
        "diffusion_type": "mdlm",
        "softmax_temp": 0.5,
        "randomness": 0.5,
        "min_add_len": 40,
        "num_steps": None,
        "inference_eps": None,
        "exclude_special_tokens": None,
    }
    assert expected.sampling_config_sha256 == launcher._canonical_json_sha256(
        expected.sampling_config
    )
    assert expected.effective_config["num_samples"] == expected_fixture.num_samples
    assert expected.effective_config["device"] == "cuda:0"
    assert expected.benchmark_runner_sha256 == launcher._sha256_file(
        Path(launcher.benchmark_runner.__file__).resolve()
    )
    assert expected.implementation_inputs == expected_fixture.implementation_inputs


def test_expected_identity_supports_udlm_and_rejects_endpoint_mismatch(
    tmp_path: Path,
) -> None:
    expected_fixture = _expected(tmp_path)
    expected_fixture.config_path.write_text(
        "diffusion_type: udlm\n"
        "softmax_temp: 1.0\n"
        "randomness: 0.0\n"
        "min_add_len: 40\n"
        "num_steps: 32\n"
        "inference_eps: 1.0e-5\n"
        "exclude_special_tokens: false\n",
        encoding="utf-8",
    )
    metadata = {
        "sha256": expected_fixture.checkpoint_sha256,
        "global_step": 100,
        "size_bytes": expected_fixture.checkpoint_size_bytes,
        "diffusion_type": "udlm",
        "udlm_inference_eps": 1e-5,
        "udlm_exclude_special_tokens": False,
    }
    with (
        mock.patch.object(
            launcher.benchmark_runner,
            "checkpoint_metadata",
            return_value=metadata,
        ),
        mock.patch.object(
            launcher.benchmark_runner,
            "implementation_input_provenance",
            return_value=expected_fixture.implementation_inputs,
        ),
    ):
        expected = launcher._build_expected_run_identity(
            expected_fixture.checkpoint_path,
            expected_fixture.config_path,
            32,
        )
        assert expected.checkpoint_diffusion_type == "udlm"
        assert expected.sampling_config["num_steps"] == 32

        metadata["udlm_inference_eps"] = 2e-5
        with pytest.raises(ValueError, match="inference_eps"):
            launcher._build_expected_run_identity(
                expected_fixture.checkpoint_path,
                expected_fixture.config_path,
                32,
            )


def test_completed_artifacts_reject_checkpoint_and_sampling_mismatches(
    tmp_path: Path,
) -> None:
    expected = _expected(tmp_path)
    output_root = tmp_path / "runs"
    _, summary_path = _write_matching_artifacts(output_root, 3, expected)
    summary = _read_summary(summary_path)
    summary["checkpoint"]["sha256"] = "0" * 64
    summary["checkpoint"]["global_step"] = 40_000
    summary["config"]["sampling"]["randomness"] = 2.0
    summary["config"]["sampling_sha256"] = launcher._canonical_json_sha256(
        summary["config"]["sampling"]
    )
    _write_summary(summary_path, summary)

    with pytest.raises(launcher.CompletionArtifactError) as caught:
        launcher._completed(output_root, 3, expected)
    message = str(caught.value)
    assert "checkpoint.sha256" in message
    assert "checkpoint.global_step" in message
    assert "config.sampling" in message
    assert "config.sampling_sha256" in message
    assert "Refusing to skip or relaunch" in message


def test_completed_artifacts_reject_runner_and_implementation_input_mismatches(
    tmp_path: Path,
) -> None:
    expected = _expected(tmp_path)
    output_root = tmp_path / "runs"
    _, summary_path = _write_matching_artifacts(output_root, 12, expected)
    summary = _read_summary(summary_path)
    summary["git"]["runner_sha256"] = "0" * 64
    summary["implementation_inputs"]["sampler_source"]["sha256"] = "1" * 64
    _write_summary(summary_path, summary)

    with pytest.raises(launcher.CompletionArtifactError) as caught:
        launcher._completed(output_root, 12, expected)
    message = str(caught.value)
    assert "git.runner_sha256" in message
    assert "implementation_inputs" in message


@pytest.mark.parametrize("missing_name", ["raw_samples.csv", "summary.json"])
def test_partial_artifacts_fail_before_launch(
    tmp_path: Path,
    missing_name: str,
) -> None:
    expected = _expected(tmp_path)
    output_root = tmp_path / "runs"
    raw_path, summary_path = _write_matching_artifacts(output_root, 1, expected)
    (raw_path if missing_name == "raw_samples.csv" else summary_path).unlink()

    with pytest.raises(launcher.CompletionArtifactError, match="partial benchmark artifacts"):
        launcher._completed(output_root, 1, expected)


def test_raw_csv_digest_row_count_and_schema_are_verified(tmp_path: Path) -> None:
    expected = _expected(tmp_path)
    output_root = tmp_path / "runs"
    raw_path, _ = _write_matching_artifacts(output_root, 5, expected)

    rows = list(csv.reader(raw_path.open("r", encoding="utf-8", newline="")))
    rows[0][1] = "unexpected_field"
    rows.pop()
    with raw_path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)

    with pytest.raises(launcher.CompletionArtifactError) as caught:
        launcher._completed(output_root, 5, expected)
    message = str(caught.value)
    assert "header/schema" in message
    assert "data rows" in message
    assert "artifacts.raw_samples_csv.sha256" in message


def test_summary_artifact_path_identity_is_verified(tmp_path: Path) -> None:
    expected = _expected(tmp_path)
    output_root = tmp_path / "runs"
    _, summary_path = _write_matching_artifacts(output_root, 9, expected)
    summary = _read_summary(summary_path)
    summary["artifacts"]["summary_json"]["path"] = str(
        (tmp_path / "different-summary.json").resolve()
    )
    _write_summary(summary_path, summary)

    with pytest.raises(
        launcher.CompletionArtifactError,
        match="artifacts.summary_json.path",
    ):
        launcher._completed(output_root, 9, expected)


def test_main_skips_matching_run_without_probing_gpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = _expected(tmp_path)
    output_root = tmp_path / "runs"
    _write_matching_artifacts(output_root, 4, expected)
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr(
        launcher,
        "_require_clean_pushed_source",
        lambda: {"head": "a" * 40, "upstream": "a" * 40},
    )
    monkeypatch.setattr(
        launcher,
        "_build_expected_run_identity",
        lambda checkpoint, config, num_samples: expected,
    )
    monkeypatch.setattr(
        launcher,
        "_snapshot",
        lambda *_: pytest.fail("matching completed runs must not probe GPUs"),
    )

    launcher.main(
        [
            "--checkpoint",
            expected.checkpoint_path.name,
            "--config",
            expected.config_path.name,
            "--num-samples",
            str(expected.num_samples),
            "--pilot",
            "--seeds",
            "4",
            "--output-root",
            output_root.name,
            "--gpu-count",
            "1",
            "--log-root",
            "logs",
        ]
    )

    assert "already have matching, integrity-checked artifacts" in capsys.readouterr().out


def test_main_requires_tmux_only_for_real_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = _expected(tmp_path)
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr(
        launcher,
        "_require_clean_pushed_source",
        lambda: {"head": "a" * 40, "upstream": "a" * 40},
    )
    monkeypatch.setattr(
        launcher,
        "_build_expected_run_identity",
        lambda checkpoint, config, num_samples: expected,
    )
    monkeypatch.setattr(launcher, "_snapshot", lambda: [_gpu()])
    common = [
        "--checkpoint",
        expected.checkpoint_path.name,
        "--config",
        expected.config_path.name,
        "--num-samples",
        str(expected.num_samples),
        "--pilot",
        "--seeds",
        "4",
        "--gpu-count",
        "1",
        "--log-root",
        "logs",
    ]

    launcher.main([*common, "--output-root", "dry-runs", "--dry-run"])
    assert "DRY RUN" in capsys.readouterr().out
    assert not (tmp_path / "dry-runs").exists()
    assert not (tmp_path / "logs").exists()

    monkeypatch.setattr(
        launcher,
        "_snapshot",
        lambda: pytest.fail("tmux guard must run before a real GPU probe"),
    )
    with pytest.raises(RuntimeError, match="must run inside tmux"):
        launcher.main([*common, "--output-root", "real-runs"])
    assert not (tmp_path / "real-runs").exists()
    assert not (tmp_path / "logs").exists()


def test_main_rejects_partial_output_before_probing_gpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _expected(tmp_path)
    output_root = tmp_path / "runs"
    raw_path, summary_path = _write_matching_artifacts(output_root, 6, expected)
    summary_path.unlink()
    assert raw_path.exists()
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        launcher,
        "_require_clean_pushed_source",
        lambda: {"head": "a" * 40, "upstream": "a" * 40},
    )
    monkeypatch.setattr(
        launcher,
        "_build_expected_run_identity",
        lambda checkpoint, config, num_samples: expected,
    )
    monkeypatch.setattr(
        launcher,
        "_snapshot",
        lambda *_: pytest.fail("partial artifacts must fail before a GPU probe"),
    )

    with pytest.raises(launcher.CompletionArtifactError, match="partial benchmark artifacts"):
        launcher.main(
            [
                "--checkpoint",
                expected.checkpoint_path.name,
                "--config",
                expected.config_path.name,
                "--num-samples",
                str(expected.num_samples),
                "--pilot",
                "--seeds",
                "6",
                "--output-root",
                output_root.name,
                "--gpu-count",
                "1",
                "--log-root",
                "logs",
            ]
        )


def test_main_rejects_mismatched_completion_before_probing_gpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _expected(tmp_path)
    output_root = tmp_path / "runs"
    _, summary_path = _write_matching_artifacts(output_root, 8, expected)
    summary = _read_summary(summary_path)
    summary["checkpoint"]["global_step"] = 45_000
    _write_summary(summary_path, summary)
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        launcher,
        "_require_clean_pushed_source",
        lambda: {"head": "a" * 40, "upstream": "a" * 40},
    )
    monkeypatch.setattr(
        launcher,
        "_build_expected_run_identity",
        lambda checkpoint, config, num_samples: expected,
    )
    monkeypatch.setattr(
        launcher,
        "_snapshot",
        lambda *_: pytest.fail("mismatched artifacts must fail before a GPU probe"),
    )

    with pytest.raises(launcher.CompletionArtifactError, match="checkpoint.global_step"):
        launcher.main(
            [
                "--checkpoint",
                expected.checkpoint_path.name,
                "--config",
                expected.config_path.name,
                "--num-samples",
                str(expected.num_samples),
                "--pilot",
                "--seeds",
                "8",
                "--output-root",
                output_root.name,
                "--gpu-count",
                "1",
                "--log-root",
                "logs",
            ]
        )

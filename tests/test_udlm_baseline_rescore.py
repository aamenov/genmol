from __future__ import annotations

import copy
import hashlib
import json
import socket
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from scripts.udlm import rescore_mdlm_baseline as rescore


@pytest.fixture(scope="module")
def manifest_artifact() -> rescore.RetainedArtifact:
    return rescore.read_stable_regular_file(
        rescore.REPOSITORY_ROOT / rescore.MANIFEST_RELATIVE_PATH,
        allowed_root=rescore.WORKSPACE_SCOPE_ROOT,
        label="test manifest",
        expected_sha256=rescore.EXPECTED_MANIFEST_SHA256,
    )


@pytest.fixture(scope="module")
def validated_manifest(manifest_artifact) -> dict:
    document = rescore.strict_json_loads(
        manifest_artifact.payload, label="test manifest"
    )
    return rescore.validate_manifest(document)


def _seed_inputs(
    seed: int, validated_manifest: dict
) -> tuple[rescore.RetainedArtifact, rescore.RetainedArtifact]:
    if not rescore.DEFAULT_HISTORICAL_RUNS_DIR.is_dir():
        pytest.skip(
            "host-local immutable MDLM row artifacts are absent in this checkout"
        )
    return rescore._seed_artifacts(  # noqa: SLF001
        rescore.DEFAULT_HISTORICAL_RUNS_DIR,
        seed=seed,
        validated_manifest=validated_manifest,
    )


def _seed_rows(
    seed: int, validated_manifest: dict
) -> tuple[list[dict[str, str]], rescore.RetainedArtifact, rescore.RetainedArtifact]:
    raw, summary = _seed_inputs(seed, validated_manifest)
    rows = rescore.parse_raw_rows(
        raw.payload,
        expected_count=rescore.EXPECTED_SAMPLES_PER_SEED,
        label=f"test seed {seed} rows",
    )
    return rows, raw, summary


def test_frozen_manifest_and_source_aggregate_are_exactly_pinned(
    manifest_artifact, validated_manifest
):
    assert manifest_artifact.sha256 == rescore.EXPECTED_MANIFEST_SHA256
    assert validated_manifest["document"]["schema_version"] == 1
    if not rescore.DEFAULT_HISTORICAL_RUNS_DIR.is_dir():
        pytest.skip(
            "host-local immutable MDLM aggregate artifact is absent in this checkout"
        )
    aggregate = rescore.read_stable_regular_file(
        rescore.DEFAULT_HISTORICAL_RUNS_DIR / "report/aggregate.json",
        allowed_root=rescore.WORKSPACE_SCOPE_ROOT,
        label="test historical aggregate",
        expected_sha256=rescore.EXPECTED_SOURCE_AGGREGATE_SHA256,
    )
    aggregate_document = rescore.strict_json_loads(
        aggregate.payload, label="test historical aggregate"
    )

    rescore.validate_source_aggregate(aggregate_document, validated_manifest)


@pytest.mark.parametrize(
    "payload",
    [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}', b"\xff"],
)
def test_strict_json_rejects_ambiguous_or_non_utf8_payloads(payload):
    with pytest.raises(rescore.RescoreValidationError):
        rescore.strict_json_loads(payload, label="bad fixture")


def test_stable_reader_rejects_symlink_and_wrong_hash(tmp_path):
    target = tmp_path / "target.json"
    target.write_bytes(b"{}\n")
    symlink = tmp_path / "link.json"
    symlink.symlink_to(target)

    with pytest.raises(rescore.RescoreValidationError, match="symlink"):
        rescore.read_stable_regular_file(
            symlink, allowed_root=tmp_path, label="symlink fixture"
        )
    with pytest.raises(rescore.RescoreValidationError, match="SHA-256 mismatch"):
        rescore.read_stable_regular_file(
            target,
            allowed_root=tmp_path,
            label="hash fixture",
            expected_sha256="0" * 64,
        )


def test_current_report_row_validators_accept_retained_schema2_bytes(
    validated_manifest, monkeypatch
):
    rows, raw, summary_artifact = _seed_rows(0, validated_manifest)
    summary_before = summary_artifact.payload
    summary = rescore.strict_json_loads(summary_before, label="seed 0 summary")
    branch_calls: list[str] = []
    cross_calls: list[int] = []
    original_branch = rescore.denovo_report._validate_branch_rows
    original_cross = rescore.denovo_report._validate_cross_branch_rows

    def branch_spy(records, *, prefix, seed):
        branch_calls.append(prefix)
        return original_branch(records, prefix=prefix, seed=seed)

    def cross_spy(records, *, seed):
        cross_calls.append(seed)
        return original_cross(records, seed=seed)

    monkeypatch.setattr(
        rescore.denovo_report, "_validate_branch_rows", branch_spy
    )
    monkeypatch.setattr(
        rescore.denovo_report, "_validate_cross_branch_rows", cross_spy
    )

    result = rescore.validate_legacy_summary_and_rows(
        summary,
        rows,
        seed=0,
        raw_sha256=raw.sha256,
    )

    assert branch_calls == ["strict", "released"]
    assert cross_calls == [0]
    assert result["row_validator_counts"]["strict"]["valid_count"] == 990
    assert summary_artifact.path.read_bytes() == summary_before
    assert hashlib.sha256(summary_artifact.path.read_bytes()).hexdigest() == (
        summary_artifact.sha256
    )


def test_schema2_summary_is_never_silently_upgraded(validated_manifest):
    rows, raw, summary_artifact = _seed_rows(0, validated_manifest)
    summary = rescore.strict_json_loads(
        summary_artifact.payload, label="seed 0 summary"
    )
    summary["schema_version"] = rescore.benchmark.SCHEMA_VERSION

    with pytest.raises(rescore.RescoreValidationError, match="must remain schema 2"):
        rescore.validate_legacy_summary_and_rows(
            summary, rows, seed=0, raw_sha256=raw.sha256
        )


def test_all_21_fields_use_exact_or_absolute_tolerance_comparison():
    row = {}
    for field in rescore.benchmark.RAW_SAMPLE_FIELDS:
        if field == "sample_index":
            row[field] = "0"
        elif field == "raw_model_text":
            row[field] = "synthetic"
        elif field in rescore.NUMERIC_ROW_FIELDS:
            row[field] = "0.5"
        elif field in rescore.REQUIRED_BOOLEAN_ROW_FIELDS:
            row[field] = "False"
        else:
            row[field] = ""
    historical = [row]
    rescored = [rescore.normalize_historical_row(row, row_index=0)]
    rescored[0]["strict_qed"] += 0.5e-12

    result = rescore.compare_raw_records(historical, rescored)

    assert result["all_match"] is True
    assert result["field_count"] == 21
    assert result["cell_count"] == 21
    strict_qed = next(
        row for row in result["field_results"] if row["field"] == "strict_qed"
    )
    assert strict_qed["max_absolute_difference"] == pytest.approx(0.5e-12)

    rescored[0]["strict_qed"] += 1.0e-12
    with pytest.raises(rescore.RescoreValidationError, match="exceeding absolute"):
        rescore.compare_raw_records(historical, rescored)

    rescored = [rescore.normalize_historical_row(row, row_index=0)]
    rescored[0]["raw_model_text"] += "x"
    with pytest.raises(rescore.RescoreValidationError, match="differs exactly"):
        rescore.compare_raw_records(historical, rescored)


def test_mocked_seed_rescore_redecodes_raw_text_without_bracket_safe(
    validated_manifest,
):
    rows, raw, summary_artifact = _seed_rows(0, validated_manifest)
    typed_rows = [
        rescore.normalize_historical_row(row, row_index=index)
        for index, row in enumerate(rows)
    ]
    summary = rescore.strict_json_loads(
        summary_artifact.payload, label="seed 0 summary"
    )
    decode_calls: list[tuple[list[str], bool]] = []
    evaluation_calls: list[int] = []

    def fake_decode(raw_model_texts, *, use_bracket_safe):
        decode_calls.append((list(raw_model_texts), use_bracket_safe))
        return copy.deepcopy(typed_rows)

    def fake_evaluate(
        decoded,
        *,
        requested_count,
        oracle_qed,
        oracle_sa,
        diversity_evaluator,
    ):
        assert decoded is not typed_rows
        assert callable(oracle_qed)
        assert callable(oracle_sa)
        assert callable(diversity_evaluator)
        evaluation_calls.append(requested_count)
        return (
            copy.deepcopy(summary["metrics"]),
            copy.deepcopy(summary["failure_counts"]),
        )

    result = rescore.rescore_retained_seed(
        seed=0,
        raw_artifact=raw,
        summary_artifact=summary_artifact,
        validated_manifest=validated_manifest,
        decode_function=fake_decode,
        evaluate_function=fake_evaluate,
        oracle_qed=lambda smiles: smiles,
        oracle_sa=lambda smiles: smiles,
        diversity_evaluator=lambda smiles: smiles,
    )

    assert len(decode_calls) == 1
    assert decode_calls[0][0] == [row["raw_model_text"] for row in rows]
    assert decode_calls[0][1] is False
    assert evaluation_calls == [1_000]
    assert result["status"] == "exact_match"
    assert result["row_comparison"]["cell_count"] == 21_000
    assert result["manifest_comparison"] == {
        "all_counts_metrics_and_artifact_hashes_match": True
    }


def test_seed_rescore_rejects_manifest_hash_disagreement(validated_manifest):
    _, raw, summary = _seed_rows(0, validated_manifest)
    bad_raw = rescore.RetainedArtifact(
        path=raw.path,
        payload=raw.payload,
        sha256="0" * 64,
        size_bytes=raw.size_bytes,
    )

    with pytest.raises(rescore.RescoreValidationError, match="not manifest-pinned"):
        rescore.rescore_retained_seed(
            seed=0,
            raw_artifact=bad_raw,
            summary_artifact=summary,
            validated_manifest=validated_manifest,
            decode_function=lambda *_args, **_kwargs: [],
            evaluate_function=lambda *_args, **_kwargs: ({}, {}),
            oracle_qed=lambda _value: None,
            oracle_sa=lambda _value: None,
            diversity_evaluator=lambda _value: None,
        )


def test_worker_environment_is_cpu_only_offline_and_seed_specific(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-must-not-leak")
    monkeypatch.setenv("PYTHONHASHSEED", "999")

    environment = rescore.worker_environment(2)

    assert environment["PYTHONHASHSEED"] == "2"
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["NVIDIA_VISIBLE_DEVICES"] == ""
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["HF_DATASETS_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONPATH"].split(rescore.os.pathsep) == [
        str(rescore.REPOSITORY_SRC),
        str(rescore.REPOSITORY_ROOT),
    ]


def test_worker_environment_validation_requires_interpreter_start_hash_seed(
    monkeypatch,
):
    for key, value in rescore.OFFLINE_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("PYTHONHASHSEED", "1")
    environment = rescore.validate_worker_environment(1)
    assert environment["device"] == "cpu"
    network_guard = environment["python_network_guard_during_computation"]
    assert network_guard["guarded_apis"] == list(
        rescore.PYTHON_NETWORK_GUARDED_APIS
    )
    assert network_guard["scope_limitation"] == (
        rescore.PYTHON_NETWORK_GUARD_LIMITATION
    )
    assert "not OS-level or process-level" in network_guard["scope_limitation"]

    monkeypatch.setenv("PYTHONHASHSEED", "0")
    with pytest.raises(rescore.RescoreValidationError, match="interpreter start"):
        rescore.validate_worker_environment(1)


def test_network_guard_blocks_connect_without_touching_the_network():
    test_socket = socket.socket()
    unguarded_gethostbyname = socket.gethostbyname
    unguarded_sendto = socket.socket.sendto
    try:
        with rescore.network_disabled():
            with pytest.raises(RuntimeError, match="guarded Python network API"):
                test_socket.connect(("127.0.0.1", 9))
            with pytest.raises(RuntimeError, match="guarded Python network API"):
                test_socket.connect_ex(("127.0.0.1", 9))
            with pytest.raises(RuntimeError, match="guarded Python network API"):
                socket.create_connection(("127.0.0.1", 9))
            with pytest.raises(RuntimeError, match="guarded Python network API"):
                socket.getaddrinfo("localhost", 80)
            assert socket.gethostbyname is unguarded_gethostbyname
            assert socket.socket.sendto is unguarded_sendto
    finally:
        test_socket.close()


def test_runtime_module_provenance_includes_loaded_safe_and_rdkit_implementations(
    monkeypatch,
):
    calls = []

    def fake_module_provenance(module_name):
        calls.append(module_name)
        return {"module": module_name, "sha256": "0" * 64}

    monkeypatch.setattr(rescore, "_module_provenance", fake_module_provenance)

    result = rescore.runtime_module_provenance()

    assert calls == list(result)
    assert {
        "safe",
        "safe.converter",
        "rdkit",
        "rdkit.Chem",
        "rdkit.Chem.rdchem",
        "rdkit.Chem.rdmolfiles",
        "rdkit.Chem.rdmolops",
    } <= set(result)


def test_worker_command_pins_each_source_hash_and_seed(tmp_path):
    args = rescore.build_parser().parse_args(
        [
            "--expected-source-revision",
            "a" * 40,
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--historical-runs-dir",
            str(tmp_path / "runs"),
        ]
    )
    expected_source_files = {
        "rescore_runner": {"sha256": "1" * 64},
        "benchmark_runner": {"sha256": "2" * 64},
        "report_validator": {"sha256": "3" * 64},
    }

    command = rescore._worker_command(  # noqa: SLF001
        args, seed=2, source_files=expected_source_files
    )

    assert command[0] == rescore.sys.executable
    assert command[command.index("--worker-seed") + 1] == "2"
    assert command[command.index("--expected-rescore-sha256") + 1] == "1" * 64
    assert command[command.index("--expected-benchmark-sha256") + 1] == "2" * 64
    assert command[command.index("--expected-report-sha256") + 1] == "3" * 64


def test_invoke_worker_parses_one_canonical_marker_and_pins_process_controls(
    tmp_path, monkeypatch
):
    args = rescore.build_parser().parse_args(
        [
            "--expected-source-revision",
            "a" * 40,
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--historical-runs-dir",
            str(tmp_path / "runs"),
        ]
    )
    source_files = {
        "rescore_runner": {"sha256": "1" * 64},
        "benchmark_runner": {"sha256": "2" * 64},
        "report_validator": {"sha256": "3" * 64},
    }
    expected_result = {"seed": 1, "status": "exact_match", "value": 7}
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        marker = rescore.WORKER_RESULT_PREFIX + json.dumps(expected_result)
        return SimpleNamespace(
            returncode=0,
            stdout=f"worker diagnostic\n{marker}\n",
            stderr="",
        )

    monkeypatch.setattr(rescore.subprocess, "run", fake_run)

    result = rescore._invoke_worker(  # noqa: SLF001
        args, seed=1, source_files=source_files
    )

    assert result == expected_result
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[command.index("--worker-seed") + 1] == "1"
    assert kwargs["cwd"] == rescore.REPOSITORY_ROOT
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["check"] is False
    assert kwargs["env"]["PYTHONHASHSEED"] == "1"
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""


@pytest.mark.parametrize(
    ("stdout", "message"),
    [
        ("diagnostic only\n", "emitted 0 result records"),
        (
            rescore.WORKER_RESULT_PREFIX
            + '{"seed":0,"status":"exact_match"}\n'
            + rescore.WORKER_RESULT_PREFIX
            + '{"seed":0,"status":"exact_match"}\n',
            "emitted 2 result records",
        ),
    ],
)
def test_invoke_worker_rejects_missing_or_ambiguous_markers(
    tmp_path, monkeypatch, stdout, message
):
    args = rescore.build_parser().parse_args(
        [
            "--expected-source-revision",
            "a" * 40,
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--historical-runs-dir",
            str(tmp_path / "runs"),
        ]
    )
    source_files = {
        "rescore_runner": {"sha256": "1" * 64},
        "benchmark_runner": {"sha256": "2" * 64},
        "report_validator": {"sha256": "3" * 64},
    }
    monkeypatch.setattr(
        rescore.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=stdout, stderr=""
        ),
    )

    with pytest.raises(RuntimeError, match=message):
        rescore._invoke_worker(  # noqa: SLF001
            args, seed=0, source_files=source_files
        )


def test_run_worker_synthetic_orchestration_order_without_historical_rows(
    tmp_path, monkeypatch
):
    args = rescore.build_parser().parse_args(
        [
            "--expected-source-revision",
            "a" * 40,
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--historical-runs-dir",
            str(tmp_path / "runs"),
            "--worker-seed",
            "1",
            "--expected-rescore-sha256",
            "1" * 64,
            "--expected-benchmark-sha256",
            "2" * 64,
            "--expected-report-sha256",
            "3" * 64,
        ]
    )
    events = []
    source_receipt = {"revision": "a" * 40, "upstream": "origin/test"}
    manifest_artifact = rescore.RetainedArtifact(
        path=args.manifest,
        payload=b"{}",
        sha256=rescore.EXPECTED_MANIFEST_SHA256,
        size_bytes=2,
    )
    raw_artifact = rescore.RetainedArtifact(
        path=tmp_path / "raw.csv", payload=b"raw", sha256="4" * 64, size_bytes=3
    )
    summary_artifact = rescore.RetainedArtifact(
        path=tmp_path / "summary.json",
        payload=b"{}",
        sha256="5" * 64,
        size_bytes=2,
    )

    monkeypatch.setattr(
        rescore,
        "validate_worker_environment",
        lambda seed: events.append("environment") or {"python_hash_seed": str(seed)},
    )

    def fake_clean_source(_revision):
        events.append("clean_source")
        return source_receipt

    monkeypatch.setattr(
        rescore.benchmark, "require_clean_pushed_source", fake_clean_source
    )
    monkeypatch.setattr(
        rescore,
        "_verify_worker_sources",
        lambda _hashes: events.append("verify_sources"),
    )
    monkeypatch.setattr(
        rescore,
        "read_stable_regular_file",
        lambda *_args, **_kwargs: events.append("read_manifest")
        or manifest_artifact,
    )
    monkeypatch.setattr(
        rescore,
        "validate_manifest",
        lambda _manifest: events.append("validate_manifest") or {"validated": True},
    )

    def fake_seed_artifacts(*_args, **_kwargs):
        events.append("seed_artifacts")
        return raw_artifact, summary_artifact

    monkeypatch.setattr(rescore, "_seed_artifacts", fake_seed_artifacts)
    sa_snapshot = SimpleNamespace(
        provenance={"relative_path": "oracle/fpscores.pkl", "sha256": "6" * 64}
    )
    monkeypatch.setattr(
        rescore.benchmark,
        "load_pinned_sa_metric_input",
        lambda: events.append("load_sa") or sa_snapshot,
    )
    monkeypatch.setattr(
        rescore.benchmark,
        "assert_local_genmol_import",
        lambda: events.append("assert_local_import"),
    )

    @contextmanager
    def fake_network_guard():
        events.append("network_guard_enter")
        try:
            yield
        finally:
            events.append("network_guard_exit")

    @contextmanager
    def fake_pinned_sa(received_snapshot, _oracle_factory):
        assert received_snapshot is sa_snapshot
        events.append("pinned_sa_enter")
        try:
            yield "sa-oracle"
        finally:
            events.append("pinned_sa_exit")

    monkeypatch.setattr(rescore, "network_disabled", fake_network_guard)
    monkeypatch.setattr(rescore.benchmark, "pinned_tdc_sa_oracle", fake_pinned_sa)
    fake_tdc = ModuleType("tdc")
    fake_tdc.Oracle = lambda name: events.append(f"oracle:{name}") or f"oracle:{name}"
    fake_tdc.Evaluator = (
        lambda name: events.append(f"evaluator:{name}") or f"evaluator:{name}"
    )
    monkeypatch.setitem(sys.modules, "tdc", fake_tdc)

    def fake_rescore_seed(**kwargs):
        events.append("rescore_seed")
        assert kwargs["seed"] == 1
        assert kwargs["raw_artifact"] is raw_artifact
        assert kwargs["summary_artifact"] is summary_artifact
        assert kwargs["oracle_qed"] == "oracle:qed"
        assert kwargs["oracle_sa"] == "sa-oracle"
        assert kwargs["diversity_evaluator"] == "evaluator:diversity"
        return {"seed": 1, "status": "exact_match", "timings_seconds": {}}

    monkeypatch.setattr(rescore, "rescore_retained_seed", fake_rescore_seed)
    monkeypatch.setattr(
        rescore.benchmark,
        "assert_runtime_tdc_metric_provenance",
        lambda metric_inputs: events.append("assert_metric_provenance"),
    )
    runtime_modules = {"safe.converter": {"sha256": "7" * 64}}
    monkeypatch.setattr(
        rescore,
        "runtime_module_provenance",
        lambda: events.append("runtime_modules") or runtime_modules,
    )
    monkeypatch.setattr(
        rescore,
        "environment_versions",
        lambda: events.append("environment_versions") or {"safe": "test"},
    )

    result = rescore.run_worker(args)

    assert events == [
        "environment",
        "clean_source",
        "verify_sources",
        "read_manifest",
        "validate_manifest",
        "seed_artifacts",
        "load_sa",
        "assert_local_import",
        "network_guard_enter",
        "pinned_sa_enter",
        "oracle:qed",
        "evaluator:diversity",
        "rescore_seed",
        "pinned_sa_exit",
        "network_guard_exit",
        "assert_metric_provenance",
        "runtime_modules",
        "clean_source",
        "verify_sources",
        "seed_artifacts",
        "environment_versions",
    ]
    assert result["seed"] == 1
    assert result["status"] == "exact_match"
    assert result["source_verification"][
        "clean_pushed_before_and_after_computation"
    ] is True
    assert result["runtime_modules"] == runtime_modules
    assert result["metric_inputs"] == sa_snapshot.provenance
    assert result["metric_inputs_sha256"] == rescore.canonical_json_sha256(
        sa_snapshot.provenance
    )


def test_atomic_exclusive_publication_never_clobbers(tmp_path, monkeypatch):
    monkeypatch.setattr(rescore, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(rescore, "WORKSPACE_SCOPE_ROOT", tmp_path)
    output = tmp_path / "output/udlm/attestation.json"
    output.parent.mkdir(parents=True)

    rescore.atomic_exclusive_write_json(output, {"schema_version": 1, "value": 7})

    assert json.loads(output.read_text(encoding="utf-8"))["value"] == 7
    with pytest.raises(FileExistsError, match="clobber"):
        rescore.atomic_exclusive_write_json(output, {"value": 8})
    assert json.loads(output.read_text(encoding="utf-8"))["value"] == 7
    assert list(output.parent.glob(f".{output.name}.*.tmp")) == []


def test_attestation_writer_does_not_create_through_symlinked_ancestor(
    tmp_path, monkeypatch
):
    repository = tmp_path / "repository"
    outside = tmp_path / "outside"
    repository.mkdir()
    outside.mkdir()
    (repository / "output").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(rescore, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(rescore, "WORKSPACE_SCOPE_ROOT", repository)

    with pytest.raises(rescore.RescoreValidationError, match="output parent"):
        rescore.atomic_exclusive_write_json(
            repository / "output/new/attestation.json", {"status": "forbidden"}
        )

    assert not (outside / "new").exists()


def test_output_path_cannot_enter_legacy_input_tree(
    tmp_path, monkeypatch
):
    runs = tmp_path / "output/benchmarks/denovo_50000"
    runs.mkdir(parents=True)
    monkeypatch.setattr(rescore, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(rescore, "WORKSPACE_SCOPE_ROOT", tmp_path)

    with pytest.raises(rescore.RescoreValidationError, match="legacy inputs"):
        rescore._validate_output_path(  # noqa: SLF001
            runs / "seed_0/new.json", runs
        )


def _synthetic_parent_seed_result(seed):
    metrics = {
        "released_comparable": {
            "validity": 0.9,
            "valid_count": 900,
            "validity_denominator": 1_000,
            "uniqueness": 1.0,
            "unique_count": 900,
            "uniqueness_denominator": 900,
            "quality": 0.8,
            "quality_count": 800,
            "quality_denominator": 1_000,
            "diversity": 0.82,
        },
        "strict": {
            "validity": 0.88,
            "valid_count": 880,
            "validity_denominator": 1_000,
            "uniqueness": 1.0,
            "unique_count": 880,
            "uniqueness_denominator": 880,
            "quality": 0.78,
            "quality_count": 780,
            "quality_denominator": 1_000,
            "diversity": 0.81,
        },
    }
    metric_inputs = {
        "relative_path": "oracle/fpscores.pkl",
        "sha256": "6" * 64,
    }
    return {
        "seed": seed,
        "status": "exact_match",
        "metrics": metrics,
        "failure_counts": {
            "released_recovered_strict_failure": 20,
            "released_largest_component_applied": 22,
        },
        "metric_inputs": metric_inputs,
        "metric_inputs_sha256": rescore.canonical_json_sha256(metric_inputs),
        "runtime_modules": {"safe.converter": {"sha256": "7" * 64}},
        "source_verification": {
            "expected_file_sha256": {
                "rescore_runner": "1" * 64,
                "benchmark_runner": "2" * 64,
                "report_validator": "3" * 64,
            }
        },
        "timings_seconds": {"worker_total": float(seed + 1)},
    }


def test_run_parent_synthetic_orders_seeds_rechecks_then_publishes_no_clobber(
    tmp_path, monkeypatch
):
    runs = tmp_path / "runs"
    (runs / "report").mkdir(parents=True)
    output = tmp_path / "evidence/rescore.json"
    output.parent.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    aggregate = runs / "report/aggregate.json"
    aggregate.write_text("{}\n", encoding="utf-8")
    args = rescore.build_parser().parse_args(
        [
            "--expected-source-revision",
            "a" * 40,
            "--manifest",
            str(manifest),
            "--historical-runs-dir",
            str(runs),
            "--output",
            str(output),
            "--jobs",
            "2",
        ]
    )
    monkeypatch.setattr(rescore, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(rescore, "WORKSPACE_SCOPE_ROOT", tmp_path)
    events = []
    clean_count = 0
    provenance_count = 0
    source_receipt = {"revision": "a" * 40, "upstream": "origin/test"}
    expected_source_files = {
        "rescore_runner": {"sha256": "1" * 64},
        "benchmark_runner": {"sha256": "2" * 64},
        "report_validator": {"sha256": "3" * 64},
        "chemistry_utils_module": {"sha256": "4" * 64},
    }

    def fake_clean_source(_revision):
        nonlocal clean_count
        clean_count += 1
        events.append(f"clean_{clean_count}")
        return copy.deepcopy(source_receipt)

    def fake_source_provenance(_revision):
        nonlocal provenance_count
        provenance_count += 1
        events.append(f"provenance_{provenance_count}")
        return copy.deepcopy(expected_source_files)

    def fake_read(path, **kwargs):
        if kwargs["label"] == "frozen MDLM manifest":
            events.append("read_manifest")
            return rescore.RetainedArtifact(
                path=Path(path),
                payload=b"{}",
                sha256=rescore.EXPECTED_MANIFEST_SHA256,
                size_bytes=2,
            )
        assert kwargs["label"] == "historical source aggregate"
        events.append("read_aggregate")
        return rescore.RetainedArtifact(
            path=Path(path),
            payload=b"{}",
            sha256=rescore.EXPECTED_SOURCE_AGGREGATE_SHA256,
            size_bytes=2,
        )

    validated_manifest = {
        "source_aggregate": {"sha256": rescore.EXPECTED_SOURCE_AGGREGATE_SHA256}
    }
    monkeypatch.setattr(
        rescore.benchmark, "require_clean_pushed_source", fake_clean_source
    )
    monkeypatch.setattr(rescore, "source_provenance", fake_source_provenance)
    monkeypatch.setattr(rescore, "read_stable_regular_file", fake_read)
    monkeypatch.setattr(
        rescore,
        "validate_manifest",
        lambda _document: events.append("validate_manifest")
        or validated_manifest,
    )
    monkeypatch.setattr(
        rescore,
        "validate_source_aggregate",
        lambda *_args: events.append("validate_source_aggregate"),
    )

    def fake_invoke_worker(_args, *, seed, source_files):
        assert source_files == expected_source_files
        events.append(f"worker_{seed}")
        return copy.deepcopy(_synthetic_parent_seed_result(seed))

    monkeypatch.setattr(rescore, "_invoke_worker", fake_invoke_worker)
    monkeypatch.setattr(
        rescore,
        "_compare_aggregate_with_manifest",
        lambda *_args: events.append("compare_aggregate"),
    )
    monkeypatch.setattr(
        rescore,
        "_revalidate_all_inputs",
        lambda **_kwargs: events.append("revalidate_inputs"),
    )
    real_writer = rescore.atomic_exclusive_write_json

    def recording_writer(path, payload):
        events.append("publish")
        real_writer(path, payload)

    monkeypatch.setattr(rescore, "atomic_exclusive_write_json", recording_writer)

    output_path, payload = rescore.run_parent(args)

    assert output_path == output
    assert [row["seed"] for row in payload["seed_results"]] == [0, 1, 2]
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == (
        "completed_exact_match"
    )
    assert payload["protocol"]["network_controls"] == {
        "offline_environment_variables": rescore.OFFLINE_ENVIRONMENT,
        "python_runtime_guarded_apis": list(rescore.PYTHON_NETWORK_GUARDED_APIS),
        "os_or_process_network_isolation": False,
        "scope_limitation": rescore.PYTHON_NETWORK_GUARD_LIMITATION,
    }
    milestones = [event for event in events if not event.startswith("worker_")]
    assert milestones == [
        "clean_1",
        "provenance_1",
        "read_manifest",
        "validate_manifest",
        "read_aggregate",
        "validate_source_aggregate",
        "compare_aggregate",
        "clean_2",
        "provenance_2",
        "revalidate_inputs",
        "clean_3",
        "publish",
    ]

    published_bytes = output.read_bytes()
    with pytest.raises(FileExistsError, match="clobber"):
        rescore.run_parent(args)
    assert output.read_bytes() == published_bytes
    assert clean_count == 3


def test_aggregate_recomputation_matches_manifest(validated_manifest):
    manifest = validated_manifest["document"]
    recovery = (10, 14, 12)
    components = (14, 14, 13)
    seed_results = []
    for seed in rescore.EXPECTED_SEEDS:
        metrics = {}
        for branch_name in ("released_comparable", "strict"):
            row = validated_manifest["rows"][branch_name][seed]
            metrics[branch_name] = {
                "validity": row["validity"],
                "valid_count": row["valid_count"],
                "validity_denominator": row["requested"],
                "uniqueness": row["uniqueness"],
                "unique_count": row["unique_count"],
                "uniqueness_denominator": row["valid_count"],
                "quality": row["quality"],
                "quality_count": row["quality_count"],
                "quality_denominator": row["requested"],
                "diversity": row["diversity"],
            }
        seed_results.append(
            {
                "seed": seed,
                "metrics": metrics,
                "failure_counts": {
                    "released_recovered_strict_failure": recovery[seed],
                    "released_largest_component_applied": components[seed],
                },
            }
        )

    aggregate = rescore._aggregate_rescore_metrics(seed_results)  # noqa: SLF001
    funnel = rescore._aggregate_funnel(seed_results)  # noqa: SLF001
    rescore._compare_aggregate_with_manifest(  # noqa: SLF001
        aggregate, funnel, validated_manifest
    )

    assert aggregate["released_comparable"]["quality"]["mean"] == (
        manifest["released_comparable"]["mean"]["quality"]
    )
    assert funnel == manifest["strict_vs_repaired_funnel"]

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.udlm import launch_optimization_screen as launcher
from scripts.udlm import prepare_optimization_screen_registry as prepare
from scripts.udlm import verify_optimization_screen as verifier


def _config_refs(gpu_count: int) -> dict[str, dict[str, object]]:
    return {
        spec.filename: {
            "root": "repository",
            "relative_path": (
                f"experiments/udlm/protocols/optimization_screen_configs_gpu"
                f"{gpu_count}/{spec.filename}"
            ),
            "sha256": hashlib.sha256(spec.filename.encode()).hexdigest(),
            "size_bytes": 10,
            "canonical_sha256": hashlib.sha256(
                ("canonical:" + spec.filename).encode()
            ).hexdigest(),
        }
        for spec in prepare._config_specs(gpu_count)
    }


@pytest.mark.parametrize("gpu_count,accumulation", [(1, 8), (2, 4)])
def test_specs_are_exactly_six_gpu_specific_unique_configs(gpu_count, accumulation):
    specs = prepare._config_specs(gpu_count)

    assert len(specs) == 6
    assert len({spec.filename for spec in specs}) == 6
    assert len({spec.output_directory for spec in specs}) == 6
    assert all(f"gpu{gpu_count}" in spec.output_directory for spec in specs)
    assert [spec.arm_id for spec in specs] == [
        "E-L0",
        "E-L1",
        "E-A0",
        "E-A0",
        "E-A1",
        "E-A1",
    ]
    assert prepare.GLOBAL_BATCH_SIZE == 16
    assert prepare.MICRO_BATCH_SIZE == 2
    assert prepare.GLOBAL_BATCH_SIZE == (
        prepare.MICRO_BATCH_SIZE * gpu_count * accumulation
    )


def test_launcher_and_registry_share_the_pinned_prior_floor_provenance():
    assert str(launcher.pilot.PILOT_EMPIRICAL_UNIFORM_MIX) == str(
        verifier.EXPECTED_EMPIRICAL_UNIFORM_MIX
    )
    assert (
        launcher.pilot.PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_PATH
        == verifier.EXPECTED_PRIOR_FLOOR_AUDIT_PATH
    )
    assert (
        launcher.pilot.PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_SHA256
        == verifier.EXPECTED_PRIOR_FLOOR_AUDIT_SHA256
    )
    assert (
        launcher.pilot.PILOT_EMPIRICAL_UNIFORM_MIX_AUDIT_SOURCE_REVISION
        == verifier.EXPECTED_PRIOR_FLOOR_AUDIT_SOURCE_REVISION
    )


@pytest.mark.parametrize("gpu_count,accumulation", [(1, 8), (2, 4)])
def test_actual_hydra_composition_and_launcher_replay_for_all_six(
    monkeypatch, gpu_count, accumulation
):
    def forbidden_gpu_probe(*_args, **_kwargs):  # pragma: no cover - failure path
        raise AssertionError("registry preparation must not query GPUs")

    monkeypatch.setattr(launcher.pilot, "probe_all_gpus", forbidden_gpu_probe)
    monkeypatch.setattr(launcher.pilot, "_probe_gpus", forbidden_gpu_probe)
    documents = prepare._compose_config_documents(
        gpu_count=gpu_count,
        checkpoint_reference={"sha256": prepare.CHECKPOINT_SHA256},
        launcher=launcher,
        verifier=verifier,
    )

    assert len(documents) == 6
    for spec in prepare._config_specs(gpu_count):
        config = documents[spec.filename]
        assert config["seed"] == 17
        assert config["trainer"]["devices"] == gpu_count
        assert config["trainer"]["max_steps"] == (
            100 if spec.stage_id == "scheduler" else 500
        )
        assert config["trainer"]["accumulate_grad_batches"] == accumulation
        assert config["loader"]["global_batch_size"] == 16
        assert config["loader"]["batch_size"] == 2
        assert config["training"]["init_from_mdlm_ema"] is True
        assert (
            config["training"]["udlm"]["empirical_uniform_mix"]
            == launcher.pilot.PILOT_EMPIRICAL_UNIFORM_MIX
        )
        assert (
            config["training"]["init_from_mdlm_checkpoint_sha256"]
            == prepare.CHECKPOINT_SHA256
        )
        assert config["callback"]["dirpath"].endswith(
            f"/{spec.output_directory}/checkpoints"
        )
        assert config["training"]["reseed_after_model_initialization"] is (
            spec.stage_id == "conditioning"
        )


def test_config_set_publication_is_exact_and_never_overwrites(monkeypatch, tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(prepare, "REPOSITORY_ROOT", repository)
    documents = {f"config_{index}.json": {"index": index} for index in range(6)}
    destination = repository / "protocols" / "configs_gpu1"

    prepare._publish_config_set_exclusive(destination, documents)

    assert {path.name for path in destination.iterdir()} == set(documents)
    assert all(
        json.loads((destination / name).read_bytes()) == value
        for name, value in documents.items()
    )
    before = {path.name: path.read_bytes() for path in destination.iterdir()}
    with pytest.raises(FileExistsError, match="refusing to replace"):
        prepare._publish_config_set_exclusive(destination, documents)
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == before


def test_registry_publication_is_exclusive(monkeypatch, tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(prepare, "REPOSITORY_ROOT", repository)
    destination = repository / "protocols" / "registry.json"

    prepare._publish_bytes_exclusive(destination, b"first\n")

    with pytest.raises(FileExistsError, match="refusing to replace"):
        prepare._publish_bytes_exclusive(destination, b"second\n")
    assert destination.read_bytes() == b"first\n"


def test_freeze_source_preflight_rejects_dirty_and_unpushed(monkeypatch):
    revision = "a" * 40

    def dirty(arguments, *, check=True):
        del check
        assert arguments[0] == "status"
        return SimpleNamespace(stdout="?? unreviewed.txt\n", returncode=0)

    monkeypatch.setattr(prepare, "_run_git", dirty)
    with pytest.raises(prepare.PreparationError, match="completely clean"):
        prepare._require_clean_pushed_source()

    def unpushed(arguments, *, check=True):
        del check
        if arguments[0] == "status":
            return SimpleNamespace(stdout="", returncode=0)
        if arguments[:2] == ["rev-parse", "--verify"]:
            value = revision if arguments[2] == "HEAD" else "b" * 40
            return SimpleNamespace(stdout=value + "\n", returncode=0)
        assert arguments[0] == "merge-base"
        return SimpleNamespace(stdout="", returncode=1)

    monkeypatch.setattr(prepare, "_run_git", unpushed)
    with pytest.raises(prepare.PreparationError, match="has not been pushed"):
        prepare._require_clean_pushed_source()


def test_registry_must_not_exist_in_source_revision(monkeypatch):
    monkeypatch.setattr(
        prepare,
        "_run_git",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="", returncode=0),
    )
    prepare._ensure_registry_absent_at_revision("a" * 40, "registry.json")

    monkeypatch.setattr(
        prepare,
        "_run_git",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout="100644 blob deadbeef\tregistry.json\0", returncode=0
        ),
    )
    with pytest.raises(prepare.PreparationError, match="already existed at R0"):
        prepare._ensure_registry_absent_at_revision("a" * 40, "registry.json")


@pytest.mark.parametrize("gpu_count,accumulation", [(1, 8), (2, 4)])
def test_registry_document_encodes_strict_batch_and_stage_contracts(
    gpu_count, accumulation
):
    source_refs = [
        {
            "root": "repository",
            "relative_path": path,
            "sha256": hashlib.sha256(path.encode()).hexdigest(),
            "size_bytes": 1,
        }
        for path in prepare.SOURCE_PATHS
    ]
    json_ref = {
        "root": "repository",
        "relative_path": "artifact.json",
        "sha256": "a" * 64,
        "size_bytes": 1,
        "schema_version": 1,
        "canonical_sha256": "b" * 64,
    }
    checkpoint = {
        "root": "project",
        "relative_path": prepare.CHECKPOINT_RELATIVE_PATH,
        "sha256": prepare.CHECKPOINT_SHA256,
        "size_bytes": prepare.CHECKPOINT_SIZE_BYTES,
    }

    document = prepare._build_registry_document(
        revision="c" * 40,
        gpu_count=gpu_count,
        checkpoint_reference=checkpoint,
        source_references=source_refs,
        panel_reference=json_ref,
        frequency_reference=json_ref,
        fixture_reference=json_ref,
        gradient_reference=json_ref,
        config_references=_config_refs(gpu_count),
        verifier=verifier,
    )

    common = document["common_training"]
    assert common["global_batch_size"] == 16
    assert common["micro_batch_size_per_process"] == 2
    assert common["accumulate_grad_batches"] == accumulation
    assert common["effective_global_batch_size"] == 16
    assert common["training_seed"] == 17
    assert common["initialization"]["checkpoint"] == checkpoint
    assert document["source"]["revision"] == "c" * 40
    assert document["source"]["clean"] is True
    assert document["source"]["pushed"] is True
    assert [stage["optimizer_updates"] for stage in document["stages"]] == [
        100,
        500,
    ]
    all_entries = [
        entry
        for stage in document["stages"]
        for arm in stage["arms"]
        for entry in arm["resolved_configs"]
    ]
    assert len(all_entries) == 6
    assert len({entry["output_directory"] for entry in all_entries}) == 6
    assert any(
        ref["relative_path"] == "scripts/udlm/prepare_optimization_screen_registry.py"
        for ref in document["source"]["blobs"]
    )
    assert {
        verifier.EXPECTED_PRIOR_FLOOR_AUDIT_SOURCE_PATH,
        verifier.EXPECTED_PRIOR_FLOOR_AUDIT_PATH,
    }.issubset({ref["relative_path"] for ref in document["source"]["blobs"]})


def test_generated_registry_document_passes_authoritative_strict_validator():
    revision = "d" * 40
    local_blobs: dict[tuple[str, str], bytes | verifier.BlobSnapshot] = {}
    git_blobs: dict[tuple[str, str], bytes] = {}

    def add_repository_blob(path: str, payload: bytes) -> dict[str, object]:
        local_blobs[("repository", path)] = payload
        git_blobs[(revision, path)] = payload
        return {
            "root": "repository",
            "relative_path": path,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }

    def add_repository_json(path: str) -> dict[str, object]:
        payload = (prepare.REPOSITORY_ROOT / path).read_bytes()
        parsed = json.loads(payload)
        return {
            **add_repository_blob(path, payload),
            "schema_version": parsed["schema_version"],
            "canonical_sha256": verifier.canonical_json_sha256(parsed),
        }

    source_refs = [
        add_repository_blob(path, (prepare.REPOSITORY_ROOT / path).read_bytes())
        for path in prepare.SOURCE_PATHS
    ]
    panel = add_repository_json(verifier.EXPECTED_PANEL_PATH)
    frequency = add_repository_json(verifier.EXPECTED_FREQUENCY_PATH)
    fixture = add_repository_json(verifier.EXPECTED_INITIALIZATION_FIXTURE_PATH)
    gradient = add_repository_json(verifier.EXPECTED_GRADIENT_CONTRACT_PATH)
    checkpoint = {
        "root": "project",
        "relative_path": prepare.CHECKPOINT_RELATIVE_PATH,
        "sha256": prepare.CHECKPOINT_SHA256,
        "size_bytes": prepare.CHECKPOINT_SIZE_BYTES,
    }
    local_blobs[("project", prepare.CHECKPOINT_RELATIVE_PATH)] = verifier.BlobSnapshot(
        size_bytes=prepare.CHECKPOINT_SIZE_BYTES,
        sha256=prepare.CHECKPOINT_SHA256,
    )
    documents = prepare._compose_config_documents(
        gpu_count=1,
        checkpoint_reference=checkpoint,
        launcher=launcher,
        verifier=verifier,
    )
    config_refs: dict[str, dict[str, object]] = {}
    config_directory = prepare.CONFIG_DIRECTORY_TEMPLATE.format(gpu_count=1)
    for name, document in documents.items():
        relative = f"{config_directory}/{name}"
        config_refs[name] = {
            **add_repository_blob(relative, prepare._json_bytes(document)),
            "canonical_sha256": verifier.canonical_json_sha256(document),
        }
    candidate = prepare._build_registry_document(
        revision=revision,
        gpu_count=1,
        checkpoint_reference=checkpoint,
        source_references=source_refs,
        panel_reference=panel,
        frequency_reference=frequency,
        fixture_reference=fixture,
        gradient_reference=gradient,
        config_references=config_refs,
        verifier=verifier,
    )

    normalized = verifier.validate_registry(
        candidate,
        loader=lambda root, path: local_blobs[(root, path.as_posix())],
        git_blob_loader=lambda source, path: git_blobs[(source, path.as_posix())],
    )

    assert normalized["common_training"]["global_batch_size"] == 16
    assert normalized["common_training"]["micro_batch_size_per_process"] == 2
    assert normalized["common_training"]["accumulate_grad_batches"] == 8
    assert (
        sum(
            len(arm["resolved_configs"])
            for stage in normalized["stages"]
            for arm in stage["arms"]
        )
        == 6
    )


def test_freeze_validates_candidate_before_exclusive_publication(monkeypatch, tmp_path):
    repository = tmp_path / "repository"
    project = tmp_path / "project"
    repository.mkdir()
    project.mkdir()
    events: list[str] = []
    revision = "c" * 40
    monkeypatch.setattr(prepare, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(prepare, "PROJECT_ROOT", project)
    monkeypatch.setattr(prepare, "_validate_fresh_output_paths", lambda _specs: None)
    monkeypatch.setattr(prepare, "_require_clean_pushed_source", lambda: revision)
    monkeypatch.setattr(
        prepare,
        "_ensure_registry_absent_at_revision",
        lambda *_args: events.append("absence-checked"),
    )
    checkpoint = {
        "root": "project",
        "relative_path": prepare.CHECKPOINT_RELATIVE_PATH,
        "sha256": prepare.CHECKPOINT_SHA256,
        "size_bytes": prepare.CHECKPOINT_SIZE_BYTES,
    }
    monkeypatch.setattr(prepare, "_validate_checkpoint", lambda _verifier: checkpoint)
    monkeypatch.setattr(
        prepare,
        "_load_committed_configs",
        lambda **_kwargs: (_config_refs(1), {name: {} for name in _config_refs(1)}),
    )
    monkeypatch.setattr(
        prepare,
        "_validate_committed_config_reconstruction",
        lambda **_kwargs: events.append("configs-replayed"),
    )
    monkeypatch.setattr(
        prepare,
        "_blob_reference_from_git",
        lambda _revision, path: {
            "root": "repository",
            "relative_path": path,
            "sha256": hashlib.sha256(path.encode()).hexdigest(),
            "size_bytes": 1,
        },
    )

    def json_reference(_revision, path, *, verifier):
        del verifier
        return {
            "root": "repository",
            "relative_path": path,
            "sha256": hashlib.sha256(path.encode()).hexdigest(),
            "size_bytes": 1,
            "schema_version": 1,
            "canonical_sha256": hashlib.sha256(
                ("canonical:" + path).encode()
            ).hexdigest(),
        }

    monkeypatch.setattr(prepare, "_json_reference_from_git", json_reference)

    def validate(payload, **kwargs):
        events.append("strict-verifier")
        assert hashlib.sha256(payload).hexdigest() == kwargs["expected_raw_sha256"]
        assert kwargs["relative_path"] == prepare.REGISTRY_RELATIVE_PATH
        parsed = json.loads(payload)
        assert parsed["common_training"]["global_batch_size"] == 16
        assert parsed["common_training"]["micro_batch_size_per_process"] == 2
        assert parsed["common_training"]["accumulate_grad_batches"] == 8
        return object()

    monkeypatch.setattr(verifier, "load_validated_registry", validate)

    def publish(path, payload):
        events.append("exclusive-publication")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    monkeypatch.setattr(prepare, "_publish_bytes_exclusive", publish)
    monkeypatch.setattr(
        prepare,
        "_assert_only_registry_change",
        lambda _path: events.append("registry-only-checked"),
    )

    result = prepare.freeze_registry(1)

    assert events == [
        "absence-checked",
        "configs-replayed",
        "strict-verifier",
        "exclusive-publication",
        "registry-only-checked",
    ]
    assert result["source_revision"] == revision
    assert result["config_count"] == 6
    assert result["gpu_probe_performed"] is False
    assert result["training_launched"] is False


def test_failed_strict_validation_publishes_nothing(monkeypatch, tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(prepare, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(prepare, "_validate_fresh_output_paths", lambda _specs: None)
    monkeypatch.setattr(prepare, "_require_clean_pushed_source", lambda: "c" * 40)
    monkeypatch.setattr(
        prepare, "_ensure_registry_absent_at_revision", lambda *_args: None
    )
    monkeypatch.setattr(
        prepare,
        "_validate_checkpoint",
        lambda _verifier: {
            "root": "project",
            "relative_path": prepare.CHECKPOINT_RELATIVE_PATH,
            "sha256": prepare.CHECKPOINT_SHA256,
            "size_bytes": prepare.CHECKPOINT_SIZE_BYTES,
        },
    )
    monkeypatch.setattr(
        prepare,
        "_load_committed_configs",
        lambda **_kwargs: (_config_refs(1), {name: {} for name in _config_refs(1)}),
    )
    monkeypatch.setattr(
        prepare, "_validate_committed_config_reconstruction", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        prepare,
        "_blob_reference_from_git",
        lambda _revision, path: {
            "root": "repository",
            "relative_path": path,
            "sha256": "a" * 64,
            "size_bytes": 1,
        },
    )
    monkeypatch.setattr(
        prepare,
        "_json_reference_from_git",
        lambda _revision, path, *, verifier: {
            "root": "repository",
            "relative_path": path,
            "sha256": "a" * 64,
            "size_bytes": 1,
            "schema_version": 1,
            "canonical_sha256": "b" * 64,
        },
    )
    monkeypatch.setattr(
        verifier,
        "load_validated_registry",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            verifier.ScreenValidationError("candidate rejected")
        ),
    )
    monkeypatch.setattr(
        prepare,
        "_publish_bytes_exclusive",
        lambda *_args: pytest.fail("publication happened before validation"),
    )

    with pytest.raises(verifier.ScreenValidationError, match="candidate rejected"):
        prepare.freeze_registry(1)
    assert not (repository / prepare.REGISTRY_RELATIVE_PATH).exists()


def test_cli_help_is_standard_library_only_and_exposes_two_phases():
    script = Path(prepare.__file__)
    completed = subprocess.run(
        [sys.executable, "-S", str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "materialize-configs" in completed.stdout
    assert "freeze-registry" in completed.stdout
    assert "gpu" in completed.stdout.lower()

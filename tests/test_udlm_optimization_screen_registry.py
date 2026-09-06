from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from scripts.udlm import launch_optimization_screen as launcher
from scripts.udlm import prepare_optimization_screen_registry as prepare
from scripts.udlm import validate_health_panel as health
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


def _health_evidence(
    gpu_count: int,
    *,
    source_revision: str = "a" * 40,
    terminal_payload: bytes = b'{"schema_version":5}\n',
) -> dict[str, object]:
    variants = tuple(launcher.pilot.MATCHED_PANEL_VARIANT_ORDER)
    run_names = [
        health.health_run_name(gpu_count, variant, source_revision)
        for variant in variants
    ]
    terminal_document = json.loads(terminal_payload)
    return {
        "schema_version": health.HEALTH_PANEL_EVIDENCE_SCHEMA_VERSION,
        "status": "validated",
        "claim_scope": "training_health_and_provenance_only",
        "health_source_revision": source_revision,
        "gpu_count": gpu_count,
        "matched_panel_spec_sha256": "f" * 64,
        "terminal_receipt": {
            "root": "repository",
            "relative_path": (f"output/udlm/{run_names[-1]}/pilot_exit_status.json"),
            "sha256": hashlib.sha256(terminal_payload).hexdigest(),
            "size_bytes": len(terminal_payload),
            "schema_version": 5,
            "canonical_sha256": verifier.canonical_json_sha256(terminal_document),
            "training_variant": "udlm_categorical",
            "position": 2,
        },
        "receipt_members": [
            {
                "position": position,
                "training_variant": variant,
                "run_name": run_names[position],
                "relative_path": (
                    f"output/udlm/{run_names[position]}/pilot_exit_status.json"
                ),
                "sha256": (
                    hashlib.sha256(terminal_payload).hexdigest()
                    if position == 2
                    else str(position + 1) * 64
                ),
                "schema_version": 5,
                "recorded_at_utc": f"2026-09-06T00:0{position}:00+00:00",
            }
            for position, variant in enumerate(variants)
        ],
        "checkpoint_members": [
            {
                "position": position,
                "training_variant": variant,
                "run_name": run_names[position],
                "relative_path": (
                    f"output/udlm/{run_names[position]}/checkpoints/10.ckpt"
                ),
                "sha256": chr(ord("a") + position) * 64,
                "size_bytes": 100 + position,
                "global_step": 10,
            }
            for position, variant in enumerate(variants)
        ],
        "eligibility": {
            "generation": False,
            "ranking": False,
            "superiority": False,
            "candidate_lock": False,
            "screen_authorization": True,
        },
    }


def _health_transition(
    gpu_count: int,
    *,
    source_revision: str = "a" * 40,
    registry_revision: str = "c" * 40,
) -> dict[str, object]:
    directory = prepare.CONFIG_DIRECTORY_TEMPLATE.format(gpu_count=gpu_count)
    return {
        "health_source_revision": source_revision,
        "registry_source_revision": registry_revision,
        "allowed_config_paths": sorted(
            f"{directory}/{spec.filename}" for spec in prepare._config_specs(gpu_count)
        ),
        "health_source_is_registry_source_parent": True,
        "exact_config_only_transition_verified": True,
        "opposite_gpu_config_family_absent": True,
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


def test_materialize_requires_clean_pushed_input_before_any_work(monkeypatch):
    monkeypatch.setattr(
        prepare,
        "_require_clean_pushed_source",
        lambda: (_ for _ in ()).throw(
            prepare.PreparationError("registry preparation requires a clean input")
        ),
    )
    monkeypatch.setattr(
        prepare,
        "_runtime_modules",
        lambda: pytest.fail("materialization continued past its Git precondition"),
    )

    with pytest.raises(prepare.PreparationError, match="clean input"):
        prepare.materialize_configs(1)


def test_health_prerequisite_uses_deterministic_terminal_path_and_expectations(
    monkeypatch, tmp_path
):
    repository = tmp_path / "repository"
    repository.mkdir()
    source_revision = "a" * 40
    expected = _health_evidence(2, source_revision=source_revision)
    calls: list[object] = []

    def run_name(gpu_count, variant, revision):
        calls.append(("run-name", gpu_count, variant, revision))
        return f"health-w{gpu_count}-e-{revision}"

    def validate(path, **kwargs):
        calls.append(("validate", path, kwargs))
        return expected

    monkeypatch.setattr(prepare, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(
        prepare,
        "_health_module",
        lambda: SimpleNamespace(
            health_run_name=run_name,
            validate_health_panel=validate,
        ),
    )

    observed = prepare._validate_health_prerequisite(
        gpu_count=2,
        health_source_revision=source_revision,
    )

    assert observed == expected
    assert calls == [
        ("run-name", 2, "udlm_categorical", source_revision),
        (
            "validate",
            repository
            / "output"
            / "udlm"
            / f"health-w2-e-{source_revision}"
            / "pilot_exit_status.json",
            {
                "expected_gpu_count": 2,
                "expected_source_revision": source_revision,
            },
        ),
    ]


def test_health_prerequisite_wraps_validator_failure(monkeypatch):
    source_revision = "a" * 40

    def fail(*_args, **_kwargs):
        raise health.HealthPanelValidationError("recursive chain invalid")

    monkeypatch.setattr(
        prepare,
        "_health_module",
        lambda: SimpleNamespace(
            health_run_name=lambda *_args: "health-w1-e-" + source_revision,
            validate_health_panel=fail,
        ),
    )

    with pytest.raises(
        prepare.PreparationError,
        match="terminal-E ten-update health receipt.*recursive chain invalid",
    ):
        prepare._validate_health_prerequisite(
            gpu_count=1,
            health_source_revision=source_revision,
        )


def test_materialize_rechecks_exact_source_before_publication(monkeypatch):
    revisions = iter(("a" * 40, "b" * 40))
    monkeypatch.setattr(
        prepare, "_require_clean_pushed_source", lambda: next(revisions)
    )
    monkeypatch.setattr(
        prepare, "_require_no_config_family_at_health_source", lambda _revision: None
    )
    monkeypatch.setattr(
        prepare,
        "_validate_health_prerequisite",
        lambda **_kwargs: _health_evidence(1),
    )
    monkeypatch.setattr(prepare, "_runtime_modules", lambda: (object(), object()))
    monkeypatch.setattr(prepare, "_validate_fresh_output_paths", lambda _specs: None)
    monkeypatch.setattr(
        prepare,
        "_validate_checkpoint",
        lambda _verifier: {"sha256": prepare.CHECKPOINT_SHA256},
    )
    documents = {
        spec.filename: {"config": spec.filename} for spec in prepare._config_specs(1)
    }
    monkeypatch.setattr(
        prepare, "_compose_config_documents", lambda **_kwargs: documents
    )
    monkeypatch.setattr(
        prepare,
        "_publish_config_set_exclusive",
        lambda *_args: pytest.fail("publication happened after the revision changed"),
    )

    with pytest.raises(prepare.PreparationError, match="revision changed"):
        prepare.materialize_configs(1)


def test_materialize_checks_only_six_changes_after_publication(monkeypatch, tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    revision = "a" * 40
    events: list[str] = []
    documents = {
        spec.filename: {"config": spec.filename} for spec in prepare._config_specs(1)
    }
    monkeypatch.setattr(prepare, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(
        prepare,
        "_require_clean_pushed_source",
        lambda: events.append("clean-pushed") or revision,
    )
    monkeypatch.setattr(
        prepare,
        "_require_no_config_family_at_health_source",
        lambda _revision: events.append("config-families-absent"),
    )
    health_evidence = _health_evidence(1, source_revision=revision)
    monkeypatch.setattr(
        prepare,
        "_validate_health_prerequisite",
        lambda **_kwargs: events.append("health-validated") or health_evidence,
    )
    verifier = SimpleNamespace(canonical_json_sha256=lambda _document: "d" * 64)
    monkeypatch.setattr(prepare, "_runtime_modules", lambda: (object(), verifier))
    monkeypatch.setattr(
        prepare,
        "_validate_fresh_output_paths",
        lambda _specs: events.append("outputs-checked"),
    )
    monkeypatch.setattr(
        prepare,
        "_validate_checkpoint",
        lambda _verifier: {"sha256": prepare.CHECKPOINT_SHA256},
    )
    monkeypatch.setattr(
        prepare, "_compose_config_documents", lambda **_kwargs: documents
    )
    real_publish = prepare._publish_config_set_exclusive

    def publish(directory, values):
        events.append("published")
        real_publish(directory, values)

    monkeypatch.setattr(prepare, "_publish_config_set_exclusive", publish)

    def boundary(**kwargs):
        events.append("boundary-checked")
        assert kwargs == {
            "gpu_count": 1,
            "documents": documents,
            "source_revision": revision,
        }

    monkeypatch.setattr(prepare, "_assert_only_materialized_config_changes", boundary)

    result = prepare.materialize_configs(1)

    assert events == [
        "clean-pushed",
        "config-families-absent",
        "health-validated",
        "outputs-checked",
        "clean-pushed",
        "published",
        "boundary-checked",
    ]
    assert result["source_revision"] == revision
    assert result["health_gate"] == health_evidence


def test_materialize_requires_terminal_health_before_composition(monkeypatch):
    revision = "a" * 40
    monkeypatch.setattr(prepare, "_require_clean_pushed_source", lambda: revision)
    monkeypatch.setattr(
        prepare, "_require_no_config_family_at_health_source", lambda _revision: None
    )
    monkeypatch.setattr(
        prepare,
        "_validate_health_prerequisite",
        lambda **_kwargs: (_ for _ in ()).throw(
            prepare.PreparationError("terminal-E ten-update health receipt is required")
        ),
    )
    monkeypatch.setattr(
        prepare,
        "_runtime_modules",
        lambda: pytest.fail("composition started without health authorization"),
    )

    with pytest.raises(prepare.PreparationError, match="health receipt"):
        prepare.materialize_configs(1)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("extra_file", "exactly six expected files"),
        ("missing_file", "exactly six expected files"),
        ("changed_bytes", "bytes changed"),
        ("unrelated_change", "only worktree changes"),
        ("missing_status_entry", "only worktree changes"),
        ("changed_revision", "source revision changed"),
    ],
)
def test_materialized_config_boundary_rejects_concurrent_mutation(
    monkeypatch, tmp_path, mutation, message
):
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(prepare, "REPOSITORY_ROOT", repository)
    revision = "a" * 40
    documents = {
        spec.filename: {"config": spec.filename} for spec in prepare._config_specs(1)
    }
    directory = prepare._config_directory(1)
    prepare._publish_config_set_exclusive(directory, documents)
    expected_paths = sorted(
        (directory / name).relative_to(repository).as_posix() for name in documents
    )

    if mutation == "extra_file":
        (directory / "extra.json").write_text("{}\n", encoding="utf-8")
    elif mutation == "missing_file":
        (directory / expected_paths[0].rsplit("/", 1)[-1]).unlink()
    elif mutation == "changed_bytes":
        (directory / expected_paths[0].rsplit("/", 1)[-1]).write_bytes(b"{}\n")
    elif mutation == "unrelated_change":
        (repository / "unrelated.txt").write_text("concurrent\n", encoding="utf-8")

    def run_git(arguments, *, check=True):
        del check
        if arguments[:2] == ["rev-parse", "--verify"]:
            observed = (
                "b" * 40
                if mutation == "changed_revision" and arguments[2] == "HEAD"
                else revision
            )
            return SimpleNamespace(stdout=observed + "\n", returncode=0)
        assert arguments == [
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ]
        paths = [
            path.relative_to(repository).as_posix()
            for path in repository.rglob("*")
            if path.is_file()
        ]
        if mutation == "missing_status_entry":
            paths.remove(expected_paths[0])
        return SimpleNamespace(
            stdout="".join(f"?? {path}\0" for path in sorted(paths)),
            returncode=0,
        )

    monkeypatch.setattr(prepare, "_run_git", run_git)

    with pytest.raises(prepare.PreparationError, match=message):
        prepare._assert_only_materialized_config_changes(
            gpu_count=1,
            documents=documents,
            source_revision=revision,
        )


def test_materialized_config_boundary_accepts_exact_six_untracked_files(
    monkeypatch, tmp_path
):
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(prepare, "REPOSITORY_ROOT", repository)
    revision = "a" * 40
    documents = {
        spec.filename: {"config": spec.filename} for spec in prepare._config_specs(2)
    }
    directory = prepare._config_directory(2)
    prepare._publish_config_set_exclusive(directory, documents)
    expected_paths = sorted(
        (directory / name).relative_to(repository).as_posix() for name in documents
    )

    def run_git(arguments, *, check=True):
        del check
        if arguments[:2] == ["rev-parse", "--verify"]:
            return SimpleNamespace(stdout=revision + "\n", returncode=0)
        return SimpleNamespace(
            stdout="".join(f"?? {path}\0" for path in expected_paths), returncode=0
        )

    monkeypatch.setattr(prepare, "_run_git", run_git)

    prepare._assert_only_materialized_config_changes(
        gpu_count=2,
        documents=documents,
        source_revision=revision,
    )


def test_materialized_config_boundary_rechecks_bytes_after_git_status(
    monkeypatch, tmp_path
):
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(prepare, "REPOSITORY_ROOT", repository)
    revision = "a" * 40
    documents = {
        spec.filename: {"config": spec.filename} for spec in prepare._config_specs(1)
    }
    directory = prepare._config_directory(1)
    prepare._publish_config_set_exclusive(directory, documents)
    expected_paths = sorted(
        (directory / name).relative_to(repository).as_posix() for name in documents
    )
    target = directory / sorted(documents)[0]

    def run_git(arguments, *, check=True):
        del check
        if arguments[:2] == ["rev-parse", "--verify"]:
            return SimpleNamespace(stdout=revision + "\n", returncode=0)
        target.write_bytes(b"{}\n")
        return SimpleNamespace(
            stdout="".join(f"?? {path}\0" for path in expected_paths), returncode=0
        )

    monkeypatch.setattr(prepare, "_run_git", run_git)

    with pytest.raises(prepare.PreparationError, match="bytes changed"):
        prepare._assert_only_materialized_config_changes(
            gpu_count=1,
            documents=documents,
            source_revision=revision,
        )


def test_registry_publication_is_exclusive(monkeypatch, tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(prepare, "REPOSITORY_ROOT", repository)
    destination = repository / "protocols" / "registry.json"

    prepare._publish_bytes_exclusive(destination, b"first\n")

    with pytest.raises(FileExistsError, match="refusing to replace"):
        prepare._publish_bytes_exclusive(destination, b"second\n")
    assert destination.read_bytes() == b"first\n"


def test_registry_publication_boundary_rechecks_refs_and_retains_failed_candidate(
    monkeypatch, tmp_path
):
    revision = "a" * 40

    def exact_refs(arguments, *, check=True):
        del check
        assert arguments[:2] == ["rev-parse", "--verify"]
        return SimpleNamespace(stdout=revision + "\n", returncode=0)

    monkeypatch.setattr(prepare, "_run_git", exact_refs)
    prepare._require_exact_pushed_revision(revision)

    candidate = {"schema_version": 1, "status": "candidate"}
    payload = prepare._json_bytes(candidate)
    path = tmp_path / "registry.json"
    path.write_bytes(payload)
    prepare._require_registry_candidate_bytes(
        candidate=candidate,
        payload=payload,
        raw_sha256=hashlib.sha256(payload).hexdigest(),
        canonical_sha256=verifier.canonical_json_sha256(candidate),
        verifier=verifier,
        published_path=path,
    )

    retained_corruption = b'{"corrupt":true}\n'
    path.write_bytes(retained_corruption)
    with pytest.raises(prepare.PreparationError, match="published registry bytes"):
        prepare._require_registry_candidate_bytes(
            candidate=candidate,
            payload=payload,
            raw_sha256=hashlib.sha256(payload).hexdigest(),
            canonical_sha256=verifier.canonical_json_sha256(candidate),
            verifier=verifier,
            published_path=path,
        )
    assert path.read_bytes() == retained_corruption

    def changed_upstream(arguments, *, check=True):
        del check
        value = revision if arguments[-1] == "HEAD" else "b" * 40
        return SimpleNamespace(stdout=value + "\n", returncode=0)

    monkeypatch.setattr(prepare, "_run_git", changed_upstream)
    with pytest.raises(prepare.PreparationError, match="HEAD or upstream changed"):
        prepare._require_exact_pushed_revision(revision)


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


@pytest.mark.parametrize("gpu_count", [1, 2])
def test_health_to_r0_transition_accepts_exact_selected_config_commit(
    monkeypatch, gpu_count
):
    health_revision = "a" * 40
    r0_revision = "b" * 40
    selected_directory = prepare.CONFIG_DIRECTORY_TEMPLATE.format(gpu_count=gpu_count)
    expected_paths = sorted(
        f"{selected_directory}/{spec.filename}"
        for spec in prepare._config_specs(gpu_count)
    )

    def run_git(arguments, *, check=True):
        del check
        if arguments[0] == "rev-list":
            return SimpleNamespace(
                stdout=f"{r0_revision} {health_revision}\n", returncode=0
            )
        if arguments[0] == "diff":
            return SimpleNamespace(
                stdout="".join(f"{path}\0" for path in expected_paths), returncode=0
            )
        revision = arguments[4]
        directory = arguments[6]
        paths = (
            expected_paths
            if revision == r0_revision and directory == selected_directory
            else []
        )
        return SimpleNamespace(
            stdout="".join(f"{path}\0" for path in paths), returncode=0
        )

    monkeypatch.setattr(prepare, "_run_git", run_git)

    transition = prepare._validate_health_to_r0_transition(
        health_source_revision=health_revision,
        r0_revision=r0_revision,
        gpu_count=gpu_count,
    )

    assert transition == _health_transition(
        gpu_count,
        source_revision=health_revision,
        registry_revision=r0_revision,
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong_parent", "parent"),
        ("extra_diff", "exactly the selected six"),
        ("missing_diff", "exactly the selected six"),
        ("selected_at_health", "already existed"),
        ("missing_at_r0", "exact six-file set"),
        ("opposite_family", "unselected GPU-count"),
    ],
)
def test_health_to_r0_transition_rejects_chronology_or_path_mutation(
    monkeypatch, mutation, message
):
    gpu_count = 1
    health_revision = "a" * 40
    r0_revision = "b" * 40
    selected_directory = prepare.CONFIG_DIRECTORY_TEMPLATE.format(gpu_count=gpu_count)
    opposite_directory = prepare.CONFIG_DIRECTORY_TEMPLATE.format(gpu_count=2)
    expected_paths = sorted(
        f"{selected_directory}/{spec.filename}"
        for spec in prepare._config_specs(gpu_count)
    )

    def run_git(arguments, *, check=True):
        del check
        if arguments[0] == "rev-list":
            parent = "c" * 40 if mutation == "wrong_parent" else health_revision
            return SimpleNamespace(stdout=f"{r0_revision} {parent}\n", returncode=0)
        if arguments[0] == "diff":
            paths = list(expected_paths)
            if mutation == "extra_diff":
                paths.append("unrelated.py")
            elif mutation == "missing_diff":
                paths.pop()
            return SimpleNamespace(
                stdout="".join(f"{path}\0" for path in paths), returncode=0
            )
        revision = arguments[4]
        directory = arguments[6]
        paths: list[str] = []
        if revision == r0_revision and directory == selected_directory:
            paths = list(expected_paths)
            if mutation == "missing_at_r0":
                paths.pop()
        elif (
            mutation == "selected_at_health"
            and revision == health_revision
            and directory == selected_directory
        ):
            paths = [expected_paths[0]]
        elif mutation == "opposite_family" and directory == opposite_directory:
            paths = [f"{opposite_directory}/unexpected.json"]
        return SimpleNamespace(
            stdout="".join(f"{path}\0" for path in paths), returncode=0
        )

    monkeypatch.setattr(prepare, "_run_git", run_git)

    with pytest.raises(prepare.PreparationError, match=message):
        prepare._validate_health_to_r0_transition(
            health_source_revision=health_revision,
            r0_revision=r0_revision,
            gpu_count=gpu_count,
        )


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
        health_evidence=_health_evidence(gpu_count),
        health_source_transition=_health_transition(gpu_count),
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
    assert document["prerequisite_health_gate"] == {
        "evidence": _health_evidence(gpu_count),
        "source_transition": _health_transition(gpu_count),
    }
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


def test_generated_registry_document_passes_authoritative_loader_and_live_health_replay():
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
    terminal_payload = b'{"schema_version":5}\n'
    health_evidence = _health_evidence(
        1,
        source_revision="c" * 40,
        terminal_payload=terminal_payload,
    )
    local_blobs[
        ("repository", health_evidence["terminal_receipt"]["relative_path"])
    ] = terminal_payload
    candidate = prepare._build_registry_document(
        revision=revision,
        gpu_count=1,
        health_evidence=health_evidence,
        health_source_transition=_health_transition(
            1,
            source_revision="c" * 40,
            registry_revision=revision,
        ),
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
    payload = prepare._json_bytes(candidate)
    live_calls: list[tuple[Path, int, str]] = []

    def tree_paths(source: str, directory: PurePosixPath) -> frozenset[str]:
        prefix = directory.as_posix() + "/"
        return frozenset(
            path
            for observed_source, path in git_blobs
            if observed_source == source and path.startswith(prefix)
        )

    def live_health(path, *, expected_gpu_count, expected_source_revision):
        live_calls.append((path, expected_gpu_count, expected_source_revision))
        return health_evidence

    loaded = verifier.load_validated_registry(
        payload,
        relative_path=prepare.REGISTRY_RELATIVE_PATH,
        expected_raw_sha256=hashlib.sha256(payload).hexdigest(),
        expected_canonical_sha256=verifier.canonical_json_sha256(candidate),
        loader=lambda root, path: local_blobs[(root, path.as_posix())],
        git_blob_loader=lambda source, path: git_blobs[(source, path.as_posix())],
        git_ancestor_checker=lambda ancestor, descendant: (
            ancestor == "c" * 40 and descendant == revision
        ),
        git_sole_parent_checker=lambda child, parent: (
            child == revision and parent == "c" * 40
        ),
        git_tree_paths_loader=tree_paths,
        git_pushed_checker=lambda source: source == revision,
        git_diff_checker=lambda ancestor, descendant, paths: (
            ancestor == "c" * 40
            and descendant == revision
            and paths
            == frozenset(
                _health_transition(1, registry_revision=revision)[
                    "allowed_config_paths"
                ]
            )
        ),
        health_gate_validator=live_health,
    )

    assert normalized["common_training"]["global_batch_size"] == 16
    assert normalized["common_training"]["micro_batch_size_per_process"] == 2
    assert normalized["common_training"]["accumulate_grad_batches"] == 8
    assert (
        loaded.data["prerequisite_health_gate"]
        == normalized["prerequisite_health_gate"]
    )
    assert live_calls == [
        (
            prepare.REPOSITORY_ROOT
            / health_evidence["terminal_receipt"]["relative_path"],
            1,
            "c" * 40,
        )
    ]
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
    health_revision = "a" * 40
    health_evidence = _health_evidence(1, source_revision=health_revision)
    health_transition = _health_transition(
        1,
        source_revision=health_revision,
        registry_revision=revision,
    )
    monkeypatch.setattr(prepare, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(prepare, "PROJECT_ROOT", project)
    monkeypatch.setattr(prepare, "_validate_fresh_output_paths", lambda _specs: None)
    monkeypatch.setattr(
        prepare,
        "_require_clean_pushed_source",
        lambda: events.append("clean-pushed") or revision,
    )
    monkeypatch.setattr(
        prepare,
        "_single_parent_revision",
        lambda _revision: events.append("parent-checked") or health_revision,
    )
    monkeypatch.setattr(
        prepare,
        "_validate_health_prerequisite",
        lambda **_kwargs: events.append("health-validated") or health_evidence,
    )
    monkeypatch.setattr(
        prepare,
        "_validate_health_to_r0_transition",
        lambda **_kwargs: events.append("transition-validated") or health_transition,
    )
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
        "_require_exact_pushed_revision",
        lambda _revision: events.append("refs-rechecked"),
    )
    monkeypatch.setattr(
        prepare,
        "_assert_only_registry_change",
        lambda _path: events.append("registry-only-checked"),
    )

    result = prepare.freeze_registry(1)

    assert events == [
        "clean-pushed",
        "parent-checked",
        "health-validated",
        "transition-validated",
        "absence-checked",
        "configs-replayed",
        "strict-verifier",
        "clean-pushed",
        "exclusive-publication",
        "refs-rechecked",
        "registry-only-checked",
        "refs-rechecked",
    ]
    assert result["source_revision"] == revision
    assert result["health_source_revision"] == health_revision
    assert (
        result["health_terminal_receipt_sha256"]
        == health_evidence["terminal_receipt"]["sha256"]
    )
    assert result["config_count"] == 6
    assert result["gpu_probe_performed"] is False
    assert result["training_launched"] is False


def test_failed_strict_validation_publishes_nothing(monkeypatch, tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(prepare, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(prepare, "_validate_fresh_output_paths", lambda _specs: None)
    monkeypatch.setattr(prepare, "_require_clean_pushed_source", lambda: "c" * 40)
    monkeypatch.setattr(prepare, "_single_parent_revision", lambda _revision: "a" * 40)
    monkeypatch.setattr(
        prepare,
        "_validate_health_prerequisite",
        lambda **_kwargs: _health_evidence(1),
    )
    monkeypatch.setattr(
        prepare,
        "_validate_health_to_r0_transition",
        lambda **_kwargs: _health_transition(1),
    )
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


def test_freeze_requires_terminal_health_before_runtime_or_publication(monkeypatch):
    monkeypatch.setattr(prepare, "_require_clean_pushed_source", lambda: "b" * 40)
    monkeypatch.setattr(prepare, "_single_parent_revision", lambda _revision: "a" * 40)
    monkeypatch.setattr(
        prepare,
        "_validate_health_prerequisite",
        lambda **_kwargs: (_ for _ in ()).throw(
            prepare.PreparationError("terminal-E ten-update health receipt is required")
        ),
    )
    monkeypatch.setattr(
        prepare,
        "_runtime_modules",
        lambda: pytest.fail("freeze imported runtime after failed health gate"),
    )
    monkeypatch.setattr(
        prepare,
        "_publish_bytes_exclusive",
        lambda *_args: pytest.fail("freeze published without health authorization"),
    )

    with pytest.raises(prepare.PreparationError, match="health receipt"):
        prepare.freeze_registry(1)


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

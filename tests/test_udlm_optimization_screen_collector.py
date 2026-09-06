from __future__ import annotations

import ast
import copy
import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest

import test_udlm_optimization_screen as cases
from scripts.udlm import collect_optimization_screen_evidence as collector
from scripts.udlm import verify_optimization_screen as verifier


REPOSITORY_ROOT = Path(__file__).parents[1]


def _overlay(harness: cases.Harness, bundle: collector.CollectionBundle):
    payloads = {
        ("repository", output.relative_path): output.payload
        for output in bundle.outputs
    }

    def load(root: str, relative_path: PurePosixPath):
        key = (root, relative_path.as_posix())
        if key in payloads:
            return payloads[key]
        return harness.loader(root, relative_path)

    return load


def test_scheduler_collection_derives_exact_refs_and_passes_strict_verifier() -> None:
    harness = cases.build_harness()
    expected = cases._scheduler_evidence(harness)
    bundle = collector.build_collection_bundle(
        registry=harness.registry,
        stage_id="scheduler",
        run_source_revision=cases.PUBLICATION_REVISION,
        evidence_relative_path=(
            "experiments/udlm/screens/scheduler_evidence_collected.json"
        ),
        loader=harness.loader,
    )

    assert bundle.evidence == expected
    assert bundle.decision["status"] == "completed"
    assert bundle.decision["selected_arm_id"] == "E-L1"
    assert [PurePosixPath(output.relative_path).name for output in bundle.outputs] == [
        "denoising_binding.json",
        "denoising_binding.json",
        "scheduler_evidence_collected.json",
    ]
    for attempt in bundle.evidence["attempts"]:
        for ref in attempt["artifacts"].values():
            payload = harness.loader("repository", PurePosixPath(ref["relative_path"]))
            assert isinstance(payload, bytes)
            assert ref["size_bytes"] == len(payload)
        assert attempt["initialization"]["state_audit"] == cases._state_audit(
            harness,
            arm_id=attempt["arm_id"],
            config_sha256=attempt["resolved_config"]["canonical_sha256"],
            stage_id="scheduler",
        )

    decision = verifier.evaluate_evidence_bytes(
        bundle.outputs[-1].payload,
        evidence_relative_path=bundle.evidence_relative_path,
        stage_id="scheduler",
        registry=harness.registry,
        loader=_overlay(harness, bundle),
    )
    assert decision == bundle.decision


def test_conditioning_collection_reproduces_committed_scheduler_dependency() -> None:
    harness = cases.build_harness()
    expected = cases._conditioning_evidence(harness)
    dependency = expected["scheduler_dependency"]
    declaration, selected, normalized = collector.derive_scheduler_dependency(
        registry=harness.registry,
        authorization_revision=cases.AUTHORIZATION_REVISION,
        scheduler_evidence_relative_path=dependency["scheduler_evidence"][
            "relative_path"
        ],
        scheduler_selection_relative_path=dependency["scheduler_selection"][
            "relative_path"
        ],
        loader=harness.loader,
    )

    assert selected == "E-L1"
    assert normalized["selected_scheduler_arm_id"] == selected
    assert declaration == dependency
    bundle = collector.build_collection_bundle(
        registry=harness.registry,
        stage_id="conditioning",
        run_source_revision=cases.AUTHORIZATION_REVISION,
        evidence_relative_path=(
            "experiments/udlm/screens/conditioning_evidence_collected.json"
        ),
        loader=harness.loader,
        scheduler_dependency_declaration=declaration,
        scheduler_arm_id=selected,
        initialization_audit=expected["initialization_audit"],
    )

    assert bundle.evidence == expected
    assert bundle.decision["status"] == "completed"
    assert bundle.decision["selected_arm_id"] == "E-A1"
    assert (
        bundle.evidence["attempts"][1]["conditioning_gradient_audit"]
        == (expected["attempts"][1]["conditioning_gradient_audit"])
    )


def test_collection_rejects_receipt_summary_mismatch_before_publication() -> None:
    harness = cases.build_harness()
    evidence = cases._scheduler_evidence(harness)
    summary_ref = evidence["attempts"][0]["artifacts"]["training_summary"]
    key = (summary_ref["root"], summary_ref["relative_path"])
    summary = json.loads(harness.blobs[key])
    summary["observed_training_state"]["global_step"] = 99
    harness.blobs[key] = cases._bytes(summary)

    with pytest.raises(verifier.ScreenValidationError):
        collector.build_collection_bundle(
            registry=harness.registry,
            stage_id="scheduler",
            run_source_revision=cases.PUBLICATION_REVISION,
            evidence_relative_path=(
                "experiments/udlm/screens/invalid_scheduler_evidence.json"
            ),
            loader=harness.loader,
        )


def test_collection_uses_summary_attestations_not_caller_values() -> None:
    harness = cases.build_harness()
    evidence = cases._scheduler_evidence(harness)
    expected_audit = copy.deepcopy(
        evidence["attempts"][0]["initialization"]["state_audit"]
    )

    bundle = collector.build_collection_bundle(
        registry=harness.registry,
        stage_id="scheduler",
        run_source_revision=cases.PUBLICATION_REVISION,
        evidence_relative_path="experiments/udlm/screens/derived_audits.json",
        loader=harness.loader,
    )

    assert (
        bundle.evidence["attempts"][0]["initialization"]["state_audit"]
        == expected_audit
    )
    assert bundle.evidence["attempts"][0]["conditioning_gradient_audit"] is None


def test_exclusive_multi_output_publish_never_clobbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(collector, "REPOSITORY_ROOT", tmp_path)
    outputs = (
        collector.PendingOutput("one/a.json", b'{"schema_version":1}\n'),
        collector.PendingOutput("two/b.json", b'{"schema_version":1}\n'),
    )
    blocking = tmp_path / "two" / "b.json"
    blocking.parent.mkdir(parents=True)
    blocking.write_bytes(b"preexisting\n")

    with pytest.raises(FileExistsError, match="refusing to replace"):
        collector.publish_outputs_exclusive(outputs)
    assert not (tmp_path / "one" / "a.json").exists()
    assert blocking.read_bytes() == b"preexisting\n"

    blocking.unlink()
    collector.publish_outputs_exclusive(outputs)
    assert (tmp_path / "one" / "a.json").read_bytes() == outputs[0].payload
    assert blocking.read_bytes() == outputs[1].payload
    with pytest.raises(FileExistsError, match="refusing to replace"):
        collector.publish_outputs_exclusive(outputs)
    assert (tmp_path / "one" / "a.json").read_bytes() == outputs[0].payload


def test_publish_rolls_back_only_new_links_on_late_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(collector, "REPOSITORY_ROOT", tmp_path)
    outputs = (
        collector.PendingOutput("one/a.json", b"first\n"),
        collector.PendingOutput("two/b.json", b"second\n"),
    )
    real_link = os.link
    calls = 0

    def collide_on_second(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            destination.write_bytes(b"racer\n")
        real_link(source, destination)

    monkeypatch.setattr(collector.os, "link", collide_on_second)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        collector.publish_outputs_exclusive(outputs)
    assert not (tmp_path / "one" / "a.json").exists()
    assert (tmp_path / "two" / "b.json").read_bytes() == b"racer\n"


def test_collector_import_and_help_are_stdlib_only_and_gpu_inert() -> None:
    source_path = (
        REPOSITORY_ROOT / "scripts/udlm/collect_optimization_screen_evidence.py"
    )
    tree = ast.parse(source_path.read_text())
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not imported_roots & {"torch", "lightning", "numpy", "omegaconf"}

    environment = {
        **os.environ,
        "PYTHONPATH": f"{REPOSITORY_ROOT / 'src'}:{REPOSITORY_ROOT}",
        "CUDA_VISIBLE_DEVICES": "",
    }
    result = subprocess.run(
        [sys.executable, "-S", str(source_path), "--help"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--expected-registry-sha256" in result.stdout
    assert "--initialization-audit" in result.stdout

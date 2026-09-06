"""Collect immutable evidence for one registered UDLM optimization screen.

The collector does not train, evaluate a model, or inspect GPUs.  It reads the
frozen registry, derives every artifact reference from the registered output
directories, constructs the small denoising-report wrappers, and asks the
independent screen verifier to validate the complete document before anything
is published.  Wrapper and evidence files are then created with exclusive
no-clobber semantics.

Scheduler evidence must be collected from the pushed registry-only run
revision.  Conditioning evidence additionally derives and validates its
authorization from the already committed scheduler evidence and selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.udlm import verify_optimization_screen as screen  # noqa: E402


EVIDENCE_SCHEMA_VERSION = 1
DENOISING_WRAPPER_SCHEMA_VERSION = 1
ARTIFACT_FILENAMES = {
    "launch_manifest": "launch_manifest.json",
    "runtime_config": "runtime_config.json",
    "training_summary": "training_summary.json",
    "exit_receipt": "pilot_exit_status.json",
    "evaluator_report": "denoising_evaluator.json",
    "denoising_wrapper": "denoising_binding.json",
}
ARTIFACT_SCHEMA_FIELDS = {
    "launch_manifest": "launch_manifest_schema_version",
    "runtime_config": "schema_version",
    "training_summary": "schema_version",
    "exit_receipt": "schema_version",
    "evaluator_report": "schema_version",
}


class EvidenceCollectionError(RuntimeError):
    """Raised before publication when evidence is absent or not authoritative."""


ArtifactLoader = Callable[[str, PurePosixPath], bytes | screen.BlobSnapshot]


@dataclass(frozen=True)
class PendingOutput:
    """One repository-relative byte payload awaiting exclusive publication."""

    relative_path: str
    payload: bytes

    @property
    def reference(self) -> dict[str, Any]:
        parsed = screen._mapping(
            screen.strict_json_loads(self.payload, label=self.relative_path),
            self.relative_path,
        )
        return {
            "root": "repository",
            "relative_path": self.relative_path,
            "sha256": hashlib.sha256(self.payload).hexdigest(),
            "size_bytes": len(self.payload),
            "schema_version": screen._integer(
                parsed.get("schema_version"),
                f"{self.relative_path} schema_version",
                minimum=1,
            ),
            "canonical_sha256": screen.canonical_json_sha256(parsed),
        }


@dataclass(frozen=True)
class CollectionBundle:
    """A preflight-validated evidence document and all files it publishes."""

    evidence: Mapping[str, Any]
    evidence_relative_path: str
    outputs: tuple[PendingOutput, ...]
    decision: Mapping[str, Any]


def _json_compatible(item: object) -> object:
    if isinstance(item, Decimal):
        return float(item)
    if isinstance(item, Mapping):
        return {key: _json_compatible(child) for key, child in item.items()}
    if isinstance(item, (list, tuple)):
        return [_json_compatible(child) for child in item]
    return item


def _json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                _json_compatible(value), indent=2, sort_keys=True, allow_nan=False
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise EvidenceCollectionError("evidence contains non-JSON data") from error


def _artifact_relative_path(output_directory: str, filename: str) -> str:
    directory = screen._relative_path(output_directory, "registered output directory")
    name = PurePosixPath(filename)
    if len(name.parts) != 1 or name.suffix not in {".json", ".ckpt"}:
        raise EvidenceCollectionError("artifact filename is not a single safe name")
    return (directory / name).as_posix()


def _load_payload(
    loader: ArtifactLoader,
    *,
    root: str,
    relative_path: str,
    label: str,
) -> bytes | screen.BlobSnapshot:
    try:
        payload = loader(root, PurePosixPath(relative_path))
    except screen.ScreenValidationError:
        raise
    except Exception as error:
        raise EvidenceCollectionError(f"{label} is unavailable") from error
    if not isinstance(payload, (bytes, screen.BlobSnapshot)):
        raise EvidenceCollectionError(f"{label} loader returned an invalid value")
    if isinstance(payload, bytes) and not payload:
        raise EvidenceCollectionError(f"{label} is empty")
    if isinstance(payload, screen.BlobSnapshot) and payload.size_bytes < 1:
        raise EvidenceCollectionError(f"{label} is empty")
    return payload


def _blob_reference(
    loader: ArtifactLoader,
    *,
    relative_path: str,
    label: str,
) -> dict[str, Any]:
    payload = _load_payload(
        loader, root="repository", relative_path=relative_path, label=label
    )
    if isinstance(payload, screen.BlobSnapshot):
        digest = screen._sha256(payload.sha256, f"{label} digest")
        size_bytes = screen._integer(payload.size_bytes, f"{label} size", minimum=1)
    else:
        digest = hashlib.sha256(payload).hexdigest()
        size_bytes = len(payload)
    return {
        "root": "repository",
        "relative_path": relative_path,
        "sha256": digest,
        "size_bytes": size_bytes,
    }


def _json_artifact(
    loader: ArtifactLoader,
    *,
    relative_path: str,
    expected_schema_version: int,
    schema_field: str = "schema_version",
    label: str,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    payload = _load_payload(
        loader, root="repository", relative_path=relative_path, label=label
    )
    if not isinstance(payload, bytes):
        raise EvidenceCollectionError(f"{label} JSON bytes were not retained")
    parsed = screen._mapping(screen.strict_json_loads(payload, label=label), label)
    schema_version = screen._integer(
        parsed.get(schema_field), f"{label} schema", minimum=1
    )
    if schema_version != expected_schema_version:
        raise EvidenceCollectionError(
            f"{label} schema {schema_version} != {expected_schema_version}"
        )
    return (
        {
            "root": "repository",
            "relative_path": relative_path,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "schema_version": schema_version,
            "canonical_sha256": screen.canonical_json_sha256(parsed),
        },
        parsed,
    )


def _producer_reference(registry: screen.ValidatedRegistry) -> Mapping[str, Any]:
    matches = [
        ref
        for ref in registry.data["source"]["blobs"]
        if ref["relative_path"]
        == "scripts/udlm/collect_optimization_screen_evidence.py"
    ]
    if len(matches) != 1:
        raise EvidenceCollectionError(
            "registry must contain exactly one collector source binding"
        )
    return matches[0]


def _overlay_loader(
    loader: ArtifactLoader, outputs: Sequence[PendingOutput]
) -> ArtifactLoader:
    overlay = {
        ("repository", output.relative_path): output.payload for output in outputs
    }
    if len(overlay) != len(outputs):
        raise EvidenceCollectionError("collector output paths collide")

    def load(root: str, relative_path: PurePosixPath) -> bytes | screen.BlobSnapshot:
        key = (root, relative_path.as_posix())
        if key in overlay:
            return overlay[key]
        return loader(root, relative_path)

    return load


def derive_scheduler_dependency(
    *,
    registry: screen.ValidatedRegistry,
    authorization_revision: str,
    scheduler_evidence_relative_path: str,
    scheduler_selection_relative_path: str,
    loader: ArtifactLoader,
) -> tuple[Mapping[str, Any], str, Mapping[str, Any]]:
    """Derive, load, and independently reproduce a scheduler authorization."""

    evidence_ref, _ = _json_artifact(
        loader,
        relative_path=screen._relative_path(
            scheduler_evidence_relative_path,
            "scheduler evidence path",
            suffix=".json",
        ).as_posix(),
        expected_schema_version=screen.EVIDENCE_SCHEMA_VERSION,
        label="scheduler evidence dependency",
    )
    selection_ref, _ = _json_artifact(
        loader,
        relative_path=screen._relative_path(
            scheduler_selection_relative_path,
            "scheduler selection path",
            suffix=".json",
        ).as_posix(),
        expected_schema_version=screen.SELECTION_SCHEMA_VERSION,
        label="scheduler selection dependency",
    )
    declaration = {
        "authorization_revision": screen._git_revision(
            authorization_revision, "conditioning authorization revision"
        ),
        "scheduler_evidence": evidence_ref,
        "scheduler_selection": selection_ref,
    }
    selected_arm_id, validated_revision, normalized = (
        screen._load_declared_scheduler_dependency(
            declaration, loader=loader, registry=registry
        )
    )
    if validated_revision != authorization_revision:
        raise EvidenceCollectionError("scheduler authorization revision changed")
    return declaration, selected_arm_id, normalized


def _attempt_and_wrapper(
    *,
    registry: screen.ValidatedRegistry,
    stage_id: str,
    arm: Mapping[str, Any],
    scheduler_arm_id: str | None,
    run_source_revision: str,
    loader: ArtifactLoader,
    producer_ref: Mapping[str, Any],
) -> tuple[Mapping[str, Any], PendingOutput]:
    config_entry = screen._registered_config(arm, scheduler_arm_id=scheduler_arm_id)
    output_directory = config_entry["output_directory"]
    refs: dict[str, Mapping[str, Any]] = {}
    documents: dict[str, Mapping[str, Any]] = {}
    for artifact_name in (
        "launch_manifest",
        "runtime_config",
        "training_summary",
        "exit_receipt",
    ):
        path = _artifact_relative_path(
            output_directory, ARTIFACT_FILENAMES[artifact_name]
        )
        ref, document = _json_artifact(
            loader,
            relative_path=path,
            expected_schema_version=screen.EXPECTED_ARTIFACT_SCHEMA_VERSIONS[
                artifact_name
            ],
            schema_field=ARTIFACT_SCHEMA_FIELDS[artifact_name],
            label=f"{arm['arm_id']} {artifact_name}",
        )
        refs[artifact_name] = ref
        documents[artifact_name] = document

    # The filename helper deliberately accepts only one path component.
    # Checkpoint ownership is assembled from separately normalized components.
    checkpoint_relative_path = (
        screen._relative_path(output_directory, "registered output directory")
        / "checkpoints"
        / f"{screen.EXPECTED_UPDATES[stage_id]}.ckpt"
    ).as_posix()
    checkpoint = {
        **_blob_reference(
            loader,
            relative_path=checkpoint_relative_path,
            label=f"{arm['arm_id']} final checkpoint",
        ),
        "global_step": screen.EXPECTED_UPDATES[stage_id],
    }

    summary = documents["training_summary"]
    state_audit = summary.get("screen_initialization_state_audit")
    gradient_audit = summary.get("conditioning_gradient_audit")
    registered_initialization = registry.data["common_training"]["initialization"]
    initialization = {
        "mode": registered_initialization["mode"],
        "source_checkpoint_sha256": registered_initialization["checkpoint"]["sha256"],
        "weights": registered_initialization["weights"],
        "optimizer_reset": registered_initialization["optimizer_reset"],
        "scheduler_reset": registered_initialization["scheduler_reset"],
        "global_step_reset": registered_initialization["global_step_reset"],
        "ema_reset": registered_initialization["ema_reset"],
        "state_audit": state_audit,
    }

    evaluator_path = _artifact_relative_path(
        output_directory, ARTIFACT_FILENAMES["evaluator_report"]
    )
    evaluator_ref, _ = _json_artifact(
        loader,
        relative_path=evaluator_path,
        expected_schema_version=screen.EVALUATOR_REPORT_SCHEMA_VERSION,
        label=f"{arm['arm_id']} denoising evaluator report",
    )
    wrapper = {
        "schema_version": DENOISING_WRAPPER_SCHEMA_VERSION,
        "artifact_kind": "optimization_screen_denoising_evaluation_binding",
        "registry_sha256": registry.raw_sha256,
        "registry_canonical_sha256": registry.canonical_sha256,
        "stage_id": stage_id,
        "arm_id": arm["arm_id"],
        "attempt_id": arm["attempt_id"],
        "source_revision": run_source_revision,
        "resolved_config_canonical_sha256": config_entry["config"]["canonical_sha256"],
        "checkpoint_sha256": checkpoint["sha256"],
        "producer_source": dict(producer_ref),
        "evaluator_report": evaluator_ref,
        "generation_metrics_included": False,
        "final_generation_seeds_included": [],
    }
    wrapper_path = _artifact_relative_path(
        output_directory, ARTIFACT_FILENAMES["denoising_wrapper"]
    )
    wrapper_output = PendingOutput(wrapper_path, _json_bytes(wrapper))
    attempt = {
        "attempt_id": arm["attempt_id"],
        "arm_id": arm["arm_id"],
        "status": "completed",
        "failure_reason": None,
        "training_seed": registry.data["common_training"]["training_seed"],
        "optimizer_updates": screen.EXPECTED_UPDATES[stage_id],
        "gpu_count": registry.data["common_training"]["gpu_count"],
        "source_revision": run_source_revision,
        "output_directory": output_directory,
        "resolved_config": config_entry["config"],
        "artifacts": refs,
        "checkpoint": checkpoint,
        "initialization": initialization,
        "denoising_report": wrapper_output.reference,
        "conditioning_gradient_audit": gradient_audit,
    }
    return attempt, wrapper_output


def build_collection_bundle(
    *,
    registry: screen.ValidatedRegistry,
    stage_id: str,
    run_source_revision: str,
    evidence_relative_path: str,
    loader: ArtifactLoader,
    scheduler_dependency_declaration: Mapping[str, Any] | None = None,
    scheduler_arm_id: str | None = None,
    initialization_audit: Mapping[str, Any] | None = None,
) -> CollectionBundle:
    """Build and preflight a complete evidence bundle without writing files."""

    if stage_id not in screen.EXPECTED_STAGE_ORDER:
        raise EvidenceCollectionError("stage must be scheduler or conditioning")
    run_source_revision = screen._git_revision(
        run_source_revision, "screen run source revision"
    )
    evidence_relative_path = screen._relative_path(
        evidence_relative_path, "screen evidence output", suffix=".json"
    ).as_posix()
    if not evidence_relative_path.startswith("experiments/udlm/screens/"):
        raise EvidenceCollectionError(
            "screen evidence must live under experiments/udlm/screens"
        )
    if stage_id == "scheduler":
        if any(
            value is not None
            for value in (
                scheduler_dependency_declaration,
                scheduler_arm_id,
                initialization_audit,
            )
        ):
            raise EvidenceCollectionError(
                "scheduler collection cannot contain conditioning prerequisites"
            )
    elif (
        scheduler_dependency_declaration is None
        or scheduler_arm_id not in screen.EXPECTED_ARM_ORDER["scheduler"]
        or initialization_audit is None
    ):
        raise EvidenceCollectionError(
            "conditioning collection requires a validated scheduler dependency "
            "and initialization audit"
        )

    producer_ref = _producer_reference(registry)
    stage = screen._stage(registry, stage_id)
    attempts: list[Mapping[str, Any]] = []
    wrapper_outputs: list[PendingOutput] = []
    for arm in stage["arms"]:
        attempt, wrapper = _attempt_and_wrapper(
            registry=registry,
            stage_id=stage_id,
            arm=arm,
            scheduler_arm_id=scheduler_arm_id,
            run_source_revision=run_source_revision,
            loader=loader,
            producer_ref=producer_ref,
        )
        attempts.append(attempt)
        wrapper_outputs.append(wrapper)

    evidence = _json_compatible(
        {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "registry": registry.reference,
            "stage_id": stage_id,
            "run_source_revision": run_source_revision,
            "producer_source": dict(producer_ref),
            "status": "closed_after_registered_attempts",
            "generation_metrics_included": False,
            "final_generation_seeds_included": [],
            "scheduler_dependency": (
                None
                if scheduler_dependency_declaration is None
                else dict(scheduler_dependency_declaration)
            ),
            "initialization_audit": (
                None if initialization_audit is None else dict(initialization_audit)
            ),
            "attempts": attempts,
        }
    )
    evidence = screen._mapping(evidence, "assembled screen evidence")
    evidence_output = PendingOutput(evidence_relative_path, _json_bytes(evidence))
    outputs = (*wrapper_outputs, evidence_output)
    preflight_loader = _overlay_loader(loader, outputs)
    # Use the verifier's strict validator first so collection errors retain a
    # useful exception instead of being collapsed into an incomplete decision.
    preflight_evidence = screen._mapping(
        screen.strict_json_loads(
            evidence_output.payload, label="assembled screen evidence"
        ),
        "assembled screen evidence",
    )
    screen._validate_evidence_document(
        preflight_evidence,
        loader=preflight_loader,
        registry=registry,
        stage_id=stage_id,
    )
    decision = screen.evaluate_evidence_bytes(
        evidence_output.payload,
        evidence_relative_path=evidence_relative_path,
        stage_id=stage_id,
        registry=registry,
        loader=preflight_loader,
    )
    if decision.get("status") != "completed":
        raise EvidenceCollectionError(
            "independent verifier rejected the assembled screen evidence"
        )
    return CollectionBundle(
        evidence=evidence,
        evidence_relative_path=evidence_relative_path,
        outputs=outputs,
        decision=decision,
    )


def _repository_path(relative_path: str) -> Path:
    relative = screen._relative_path(relative_path, "collector output path")
    candidate = REPOSITORY_ROOT.joinpath(*relative.parts)
    root = REPOSITORY_ROOT.resolve(strict=True)
    current = REPOSITORY_ROOT
    for part in relative.parts[:-1]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            break
        if stat.S_ISLNK(mode):
            raise EvidenceCollectionError(
                f"collector output has a symlink ancestor: {candidate}"
            )
    if (
        candidate.resolve(strict=False)
        .parent.resolve(strict=False)
        .is_relative_to(root)
    ):
        return candidate
    raise EvidenceCollectionError(f"collector output escapes repository: {candidate}")


def publish_outputs_exclusive(outputs: Sequence[PendingOutput]) -> None:
    """Publish all outputs without replacing any existing filesystem entry."""

    if not outputs:
        raise EvidenceCollectionError("collector has no outputs to publish")
    paths = [_repository_path(output.relative_path) for output in outputs]
    if len(paths) != len(set(paths)):
        raise EvidenceCollectionError("collector output paths collide")
    existing = [path for path in paths if os.path.lexists(path)]
    if existing:
        raise FileExistsError(f"refusing to replace collector output: {existing[0]}")

    prepared: list[tuple[Path, Path]] = []
    linked: list[tuple[Path, Path]] = []
    committed = False
    try:
        for path, output in zip(paths, outputs, strict=True):
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.parent.resolve(
                strict=True
            ) != path.parent or not path.parent.resolve(strict=True).is_relative_to(
                REPOSITORY_ROOT.resolve(strict=True)
            ):
                raise EvidenceCollectionError(
                    f"collector output parent is unsafe: {path.parent}"
                )
            descriptor, temporary_name = tempfile.mkstemp(
                dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
            )
            temporary = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(output.payload)
                handle.flush()
                os.fsync(handle.fileno())
            prepared.append((path, temporary))
        for path, temporary in prepared:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise FileExistsError(
                    f"refusing to replace collector output: {path}"
                ) from error
            linked.append((path, temporary))
        for directory in sorted({path.parent for path in paths}):
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        committed = True
    finally:
        if not committed:
            for path, temporary in reversed(linked):
                try:
                    path_stat = path.stat(follow_symlinks=False)
                    temporary_stat = temporary.stat(follow_symlinks=False)
                    if (
                        path_stat.st_dev,
                        path_stat.st_ino,
                    ) == (temporary_stat.st_dev, temporary_stat.st_ino):
                        path.unlink()
                except FileNotFoundError:
                    pass
        for _path, temporary in prepared:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _run_git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _current_clean_pushed_revision() -> str:
    revision = screen._git_revision(_run_git("rev-parse", "HEAD"), "Git HEAD")
    if _run_git("status", "--porcelain", "--untracked-files=no"):
        raise EvidenceCollectionError(
            "collector requires a clean tracked worktree at the run revision"
        )
    if not screen.git_pushed_checker(revision):
        raise EvidenceCollectionError("collector requires a pushed run revision")
    return revision


def _repository_relative_input(path: Path, label: str) -> str:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or not resolved.is_relative_to(
        REPOSITORY_ROOT.resolve(strict=True)
    ):
        raise EvidenceCollectionError(f"{label} must be a repository file")
    if resolved != path.absolute():
        raise EvidenceCollectionError(f"{label} cannot be a symlink")
    return resolved.relative_to(REPOSITORY_ROOT.resolve(strict=True)).as_posix()


def _repository_relative_output(path: Path) -> str:
    normalized = Path(os.path.abspath(os.fspath(path)))
    if normalized.suffix != ".json" or not normalized.is_relative_to(
        REPOSITORY_ROOT.resolve(strict=True)
    ):
        raise EvidenceCollectionError(
            "evidence output must be a JSON path inside the repository"
        )
    return normalized.relative_to(REPOSITORY_ROOT.resolve(strict=True)).as_posix()


def _load_initialization_audit(path: Path) -> Mapping[str, Any]:
    relative_path = _repository_relative_input(path, "initialization audit")
    _ref, document = _json_artifact(
        screen.local_blob_loader,
        relative_path=relative_path,
        expected_schema_version=screen.INITIALIZATION_AUDIT_SCHEMA_VERSION,
        label="conditioning initialization audit",
    )
    return document


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--expected-registry-sha256", required=True)
    parser.add_argument("--expected-registry-canonical-sha256", required=True)
    parser.add_argument("--stage", choices=screen.EXPECTED_STAGE_ORDER, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scheduler-evidence", type=Path)
    parser.add_argument("--scheduler-selection", type=Path)
    parser.add_argument("--initialization-audit", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    registry_path = args.registry.resolve(strict=True)
    registry_relative_path = _repository_relative_input(registry_path, "registry")
    registry_payload = screen._stable_input_bytes(registry_path)
    registry = screen.load_validated_registry(
        registry_payload,
        relative_path=registry_relative_path,
        expected_raw_sha256=args.expected_registry_sha256,
        expected_canonical_sha256=args.expected_registry_canonical_sha256,
        loader=screen.local_blob_loader,
        git_blob_loader=screen.git_blob_loader,
        git_ancestor_checker=screen.git_ancestor_checker,
        git_sole_parent_checker=screen.git_sole_parent_checker,
        git_tree_paths_loader=screen.git_tree_paths_loader,
        git_pushed_checker=screen.git_pushed_checker,
        git_diff_checker=screen.git_diff_checker,
    )
    run_source_revision = _current_clean_pushed_revision()
    evidence_relative_path = _repository_relative_output(args.output)

    dependency_declaration = None
    scheduler_arm_id = None
    initialization_audit = None
    dependency_arguments = (
        args.scheduler_evidence,
        args.scheduler_selection,
        args.initialization_audit,
    )
    if args.stage == "scheduler":
        if any(value is not None for value in dependency_arguments):
            raise EvidenceCollectionError(
                "scheduler collection does not accept dependency or init-audit inputs"
            )
    elif any(value is None for value in dependency_arguments):
        raise EvidenceCollectionError(
            "conditioning collection requires --scheduler-evidence, "
            "--scheduler-selection, and --initialization-audit"
        )
    else:
        scheduler_evidence_path = _repository_relative_input(
            args.scheduler_evidence, "scheduler evidence"
        )
        scheduler_selection_path = _repository_relative_input(
            args.scheduler_selection, "scheduler selection"
        )
        (
            dependency_declaration,
            scheduler_arm_id,
            _normalized_dependency,
        ) = derive_scheduler_dependency(
            registry=registry,
            authorization_revision=run_source_revision,
            scheduler_evidence_relative_path=scheduler_evidence_path,
            scheduler_selection_relative_path=scheduler_selection_path,
            loader=screen.local_blob_loader,
        )
        initialization_audit = _load_initialization_audit(args.initialization_audit)

    bundle = build_collection_bundle(
        registry=registry,
        stage_id=args.stage,
        run_source_revision=run_source_revision,
        evidence_relative_path=evidence_relative_path,
        loader=screen.local_blob_loader,
        scheduler_dependency_declaration=dependency_declaration,
        scheduler_arm_id=scheduler_arm_id,
        initialization_audit=initialization_audit,
    )
    publish_outputs_exclusive(bundle.outputs)
    print(json.dumps(bundle.evidence, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

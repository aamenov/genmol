"""Prepare the prospective UDLM optimization screen without using a GPU.

The workflow is deliberately split across two Git revisions:

1. ``materialize-configs`` composes and exclusively publishes the six resolved
   Hydra configurations for the user-selected GPU count.  Commit and push
   those files together with the reviewed implementation (revision R0).
2. ``freeze-registry`` requires a clean, pushed R0, proves that every config is
   its exact Git blob and is reproducible by the registered launcher, validates
   the complete candidate with the strict screen verifier, and exclusively
   writes the registry as the sole R0 -> R1 change.

Neither phase inventories GPUs or starts a training/evaluation process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPOSITORY_ROOT.parents[1]
CHECKPOINT_RELATIVE_PATH = "outputs/paper_v1/checkpoints/50000.ckpt"
CHECKPOINT_SHA256 = "8d00aa47b02f64bf39ff6b0b2e786f213587366fc2c3d29712a00f3f84108dd6"
CHECKPOINT_SIZE_BYTES = 1_396_998_679
REGISTRY_RELATIVE_PATH = (
    "experiments/udlm/protocols/optimization_screen_registry_v1.json"
)
CONFIG_DIRECTORY_TEMPLATE = (
    "experiments/udlm/protocols/optimization_screen_configs_gpu{gpu_count}"
)
GLOBAL_BATCH_SIZE = 16
MICRO_BATCH_SIZE = 2
NUM_WORKERS = 1
EXCLUDE_SPECIAL_TOKENS = False
PANEL_BATCH_SIZE = 4
HEX_REVISION = re.compile(r"[0-9a-f]{40}\Z")

# This is intentionally a superset of the verifier's required source closure.
# In particular, the producer itself is bound so that the frozen registry can
# be reconstructed from R0 without trusting whatever happens to be on disk.
SOURCE_PATHS = (
    "configs/base.yaml",
    "configs/udlm.yaml",
    "configs/udlm_categorical.yaml",
    "scripts/train.py",
    "scripts/udlm/audit_conditioning_initialization.py",
    "scripts/udlm/collect_optimization_screen_evidence.py",
    "scripts/udlm/evaluate_denoising_panel.py",
    "scripts/udlm/launch_optimization_screen.py",
    "scripts/udlm/launch_train_pilot.py",
    "scripts/udlm/materialize_validation_panel.py",
    "scripts/udlm/prepare_optimization_screen_registry.py",
    "scripts/udlm/token_frequency_audit.py",
    "scripts/udlm/verify_optimization_screen.py",
    "scripts/udlm/write_pilot_exit_status.py",
    "src/genmol/backbone.py",
    "src/genmol/diffusion.py",
    "src/genmol/model.py",
    "src/genmol/utils/ema.py",
    "src/genmol/utils/utils_data.py",
)


class PreparationError(RuntimeError):
    """Raised when a prospective registry cannot be frozen safely."""


@dataclass(frozen=True)
class ConfigSpec:
    """One of the six immutable scheduler/conditioner configurations."""

    filename: str
    stage_id: str
    arm_id: str
    scheduler_arm_id: str
    registry_scheduler_arm_id: str | None
    output_directory: str


def _runtime_modules() -> tuple[Any, Any]:
    """Import project modules lazily so ``python -S ... --help`` stays usable."""

    import sys

    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    from scripts.udlm import launch_optimization_screen as launcher
    from scripts.udlm import verify_optimization_screen as verifier

    return launcher, verifier


def _config_specs(gpu_count: int) -> tuple[ConfigSpec, ...]:
    if type(gpu_count) is not int or gpu_count not in {1, 2}:
        raise ValueError("gpu-count must be 1 or 2")
    prefix = f"gpu{gpu_count}"
    specs = (
        ConfigSpec(
            "scheduler_e_l0.json",
            "scheduler",
            "E-L0",
            "E-L0",
            None,
            f"output/udlm/screens/{prefix}_scheduler_e_l0",
        ),
        ConfigSpec(
            "scheduler_e_l1.json",
            "scheduler",
            "E-L1",
            "E-L1",
            None,
            f"output/udlm/screens/{prefix}_scheduler_e_l1",
        ),
        ConfigSpec(
            "conditioning_e_a0__e_l0.json",
            "conditioning",
            "E-A0",
            "E-L0",
            "E-L0",
            f"output/udlm/screens/{prefix}_conditioning_e_a0__e_l0",
        ),
        ConfigSpec(
            "conditioning_e_a0__e_l1.json",
            "conditioning",
            "E-A0",
            "E-L1",
            "E-L1",
            f"output/udlm/screens/{prefix}_conditioning_e_a0__e_l1",
        ),
        ConfigSpec(
            "conditioning_e_a1__e_l0.json",
            "conditioning",
            "E-A1",
            "E-L0",
            "E-L0",
            f"output/udlm/screens/{prefix}_conditioning_e_a1__e_l0",
        ),
        ConfigSpec(
            "conditioning_e_a1__e_l1.json",
            "conditioning",
            "E-A1",
            "E-L1",
            "E-L1",
            f"output/udlm/screens/{prefix}_conditioning_e_a1__e_l1",
        ),
    )
    if (
        len({spec.filename for spec in specs}) != 6
        or len({spec.output_directory for spec in specs}) != 6
    ):
        raise AssertionError("optimization-screen config paths are not unique")
    return specs


def _config_directory(gpu_count: int) -> Path:
    return REPOSITORY_ROOT / CONFIG_DIRECTORY_TEMPLATE.format(gpu_count=gpu_count)


def _checkpoint_path() -> Path:
    return PROJECT_ROOT / CHECKPOINT_RELATIVE_PATH


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _validate_checkpoint(verifier: Any) -> dict[str, Any]:
    reference = {
        "root": "project",
        "relative_path": CHECKPOINT_RELATIVE_PATH,
        "sha256": CHECKPOINT_SHA256,
        "size_bytes": CHECKPOINT_SIZE_BYTES,
    }
    # Streaming through the verifier checks regular-file identity, link count,
    # size, and digest without loading the 1.4 GB checkpoint into memory.
    verifier._load_bound_blob(
        reference,
        loader=verifier.local_blob_loader,
        label="screen initialization checkpoint",
    )
    return reference


def _validate_fresh_output_paths(specs: Sequence[ConfigSpec]) -> None:
    for spec in specs:
        output = REPOSITORY_ROOT.joinpath(*PurePosixPath(spec.output_directory).parts)
        if os.path.lexists(output):
            raise FileExistsError(
                f"refusing to reuse registered screen output directory: {output}"
            )


def _compose_config_documents(
    *,
    gpu_count: int,
    checkpoint_reference: Mapping[str, Any],
    launcher: Any,
    verifier: Any,
) -> dict[str, Mapping[str, Any]]:
    """Compose all six documents and prove both launcher reconstruction paths."""

    accumulation = 8 if gpu_count == 1 else 4
    documents: dict[str, Mapping[str, Any]] = {}
    for spec in _config_specs(gpu_count):
        run_dir = REPOSITORY_ROOT.joinpath(*PurePosixPath(spec.output_directory).parts)
        _command, resolved, digest = launcher.compose_screen_training_bundle(
            stage_id=spec.stage_id,
            arm_id=spec.arm_id,
            scheduler_arm_id=spec.scheduler_arm_id,
            gpu_count=gpu_count,
            run_dir=run_dir,
            checkpoint=_checkpoint_path().resolve(strict=False),
            checkpoint_sha256=checkpoint_reference["sha256"],
            global_batch_size=GLOBAL_BATCH_SIZE,
            micro_batch_size=MICRO_BATCH_SIZE,
            num_workers=NUM_WORKERS,
            exclude_special_tokens=EXCLUDE_SPECIAL_TOKENS,
        )
        _replay_command, replayed, replay_digest = (
            launcher.build_registered_training_command(
                resolved_config=resolved,
                run_dir=run_dir,
                gpu_count=gpu_count,
            )
        )
        if (
            replayed != resolved
            or replay_digest != digest
            or digest != verifier.canonical_json_sha256(resolved)
            or resolved["seed"] != 17
            or resolved["trainer"]["max_steps"]
            != (100 if spec.stage_id == "scheduler" else 500)
            or resolved["trainer"]["accumulate_grad_batches"] != accumulation
            or resolved["loader"]["global_batch_size"] != GLOBAL_BATCH_SIZE
            or resolved["loader"]["batch_size"] != MICRO_BATCH_SIZE
            or resolved["training"]["init_from_mdlm_ema"] is not True
            or resolved["training"]["init_from_mdlm_checkpoint_sha256"]
            != checkpoint_reference["sha256"]
        ):
            raise PreparationError(
                f"launcher reconstruction failed for {spec.filename}"
            )
        documents[spec.filename] = resolved
    if set(documents) != {spec.filename for spec in _config_specs(gpu_count)}:
        raise AssertionError("did not compose exactly the six registered configs")
    return documents


def _publish_config_set_exclusive(
    directory: Path, documents: Mapping[str, Mapping[str, Any]]
) -> None:
    """Publish a complete six-file directory without replacing any path."""

    expected_names = set(documents)
    if len(expected_names) != 6 or any(
        Path(name).name != name for name in expected_names
    ):
        raise PreparationError("config set must contain exactly six plain filenames")
    directory = Path(os.path.abspath(os.fspath(directory)))
    repository = REPOSITORY_ROOT.resolve(strict=True)
    if not directory.resolve(strict=False).is_relative_to(repository):
        raise PreparationError("config directory must be inside the repository")
    if os.path.lexists(directory):
        raise FileExistsError(f"refusing to replace config directory: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=directory.parent, prefix=".screen-configs."))
    created_target = False
    try:
        for name in sorted(documents):
            payload = _json_bytes(documents[name])
            with (staging / name).open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        os.mkdir(directory, mode=0o755)
        created_target = True
        for name in sorted(documents):
            os.link(staging / name, directory / name)
        shutil.rmtree(staging)
        target_descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(target_descriptor)
        finally:
            os.close(target_descriptor)
        parent_descriptor = os.open(directory.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except Exception:
        if created_target:
            for name in expected_names:
                try:
                    (directory / name).unlink()
                except FileNotFoundError:
                    pass
            try:
                directory.rmdir()
            except OSError:
                pass
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def materialize_configs(gpu_count: int) -> dict[str, Any]:
    """Compose and exclusively publish the six reviewed R0 config candidates."""

    launcher, verifier = _runtime_modules()
    specs = _config_specs(gpu_count)
    _validate_fresh_output_paths(specs)
    checkpoint = _validate_checkpoint(verifier)
    documents = _compose_config_documents(
        gpu_count=gpu_count,
        checkpoint_reference=checkpoint,
        launcher=launcher,
        verifier=verifier,
    )
    directory = _config_directory(gpu_count)
    _publish_config_set_exclusive(directory, documents)
    return {
        "status": "six_resolved_configs_materialized_no_gpu_operation",
        "gpu_count": gpu_count,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "micro_batch_size_per_process": MICRO_BATCH_SIZE,
        "accumulate_grad_batches": 8 if gpu_count == 1 else 4,
        "checkpoint_sha256": checkpoint["sha256"],
        "config_directory": directory.relative_to(REPOSITORY_ROOT).as_posix(),
        "configs": [
            {
                "relative_path": (directory / spec.filename)
                .relative_to(REPOSITORY_ROOT)
                .as_posix(),
                "canonical_sha256": verifier.canonical_json_sha256(
                    documents[spec.filename]
                ),
                "output_directory": spec.output_directory,
            }
            for spec in specs
        ],
        "gpu_probe_performed": False,
        "training_launched": False,
    }


def _run_git(
    arguments: Sequence[str], *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments],
        check=check,
        capture_output=True,
        text=True,
    )


def _require_clean_pushed_source() -> str:
    status = _run_git(["status", "--porcelain=v1", "--untracked-files=all"]).stdout
    if status:
        raise PreparationError(
            "freeze-registry requires a completely clean R0 worktree"
        )
    revision = _run_git(["rev-parse", "--verify", "HEAD"]).stdout.strip()
    if HEX_REVISION.fullmatch(revision) is None:
        raise PreparationError("HEAD did not resolve to a full lowercase Git revision")
    upstream = _run_git(["rev-parse", "--verify", "@{upstream}"]).stdout.strip()
    if HEX_REVISION.fullmatch(upstream) is None:
        raise PreparationError(
            "configured upstream did not resolve to a full lowercase Git revision"
        )
    if revision != upstream:
        raise PreparationError(
            "R0 HEAD has not been pushed exactly to the configured upstream"
        )
    return revision


def _ensure_registry_absent_at_revision(revision: str, relative_path: str) -> None:
    result = _run_git(["ls-tree", "-z", "--full-tree", revision, "--", relative_path])
    if result.stdout:
        raise PreparationError(
            "registry path already existed at R0; refusing to violate R0/R1 chronology"
        )


def _git_blob_bytes(revision: str, relative_path: str) -> bytes:
    object_name = f"{revision}:{relative_path}"
    kind = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "cat-file", "-t", object_name],
        check=True,
        capture_output=True,
    ).stdout.strip()
    if kind != b"blob":
        raise PreparationError(f"R0 object is not a blob: {relative_path}")
    return subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "cat-file", "blob", object_name],
        check=True,
        capture_output=True,
    ).stdout


def _blob_reference_from_git(revision: str, relative_path: str) -> dict[str, Any]:
    payload = _git_blob_bytes(revision, relative_path)
    if not payload:
        raise PreparationError(f"R0 blob is empty: {relative_path}")
    return {
        "root": "repository",
        "relative_path": relative_path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _json_reference_from_git(
    revision: str, relative_path: str, *, verifier: Any
) -> dict[str, Any]:
    reference = _blob_reference_from_git(revision, relative_path)
    payload = _git_blob_bytes(revision, relative_path)
    parsed = verifier.strict_json_loads(payload, label=relative_path)
    if not isinstance(parsed, Mapping):
        raise PreparationError(f"R0 JSON blob is not an object: {relative_path}")
    schema_version = parsed.get("schema_version")
    if type(schema_version) is not int or schema_version < 1:
        raise PreparationError(f"R0 JSON blob has no positive schema: {relative_path}")
    return {
        **reference,
        "schema_version": schema_version,
        "canonical_sha256": verifier.canonical_json_sha256(parsed),
    }


def _config_reference_from_git(
    revision: str, relative_path: str, *, verifier: Any
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    reference = _blob_reference_from_git(revision, relative_path)
    payload = _git_blob_bytes(revision, relative_path)
    strict_parsed = verifier.strict_json_loads(payload, label=relative_path)
    if not isinstance(strict_parsed, Mapping):
        raise PreparationError(f"R0 config is not an object: {relative_path}")
    # The strict parser deliberately represents decimal leaves as Decimal.
    # Replay uses Hydra's ordinary int/float/string leaves, so return a second
    # parse only after the strict parse has rejected duplicates/nonfinite input.
    parsed = json.loads(payload)
    return (
        {
            **reference,
            "canonical_sha256": verifier.canonical_json_sha256(strict_parsed),
        },
        parsed,
    )


def _load_committed_configs(
    *, revision: str, gpu_count: int, verifier: Any
) -> tuple[dict[str, dict[str, Any]], dict[str, Mapping[str, Any]]]:
    directory = _config_directory(gpu_count)
    expected = {spec.filename for spec in _config_specs(gpu_count)}
    if not directory.is_dir() or directory.is_symlink():
        raise PreparationError("the GPU-specific R0 config directory is unavailable")
    observed = {entry.name for entry in directory.iterdir()}
    if observed != expected or any(
        not (directory / name).is_file() or (directory / name).is_symlink()
        for name in observed
    ):
        raise PreparationError("R0 config directory must contain exactly six files")
    references: dict[str, dict[str, Any]] = {}
    documents: dict[str, Mapping[str, Any]] = {}
    for name in sorted(expected):
        relative = (directory / name).relative_to(REPOSITORY_ROOT).as_posix()
        reference, document = _config_reference_from_git(
            revision, relative, verifier=verifier
        )
        if (directory / name).read_bytes() != _git_blob_bytes(revision, relative):
            raise PreparationError(f"working config differs from R0: {relative}")
        references[name] = reference
        documents[name] = document
    return references, documents


def _validate_committed_config_reconstruction(
    *,
    gpu_count: int,
    checkpoint_reference: Mapping[str, Any],
    documents: Mapping[str, Mapping[str, Any]],
    launcher: Any,
    verifier: Any,
) -> None:
    expected = _compose_config_documents(
        gpu_count=gpu_count,
        checkpoint_reference=checkpoint_reference,
        launcher=launcher,
        verifier=verifier,
    )
    for spec in _config_specs(gpu_count):
        if dict(documents[spec.filename]) != dict(expected[spec.filename]):
            raise PreparationError(
                f"committed config is not the reviewed launcher bundle: {spec.filename}"
            )


def _conditioner(candidate: bool) -> dict[str, Any]:
    return {
        "kind": "film_adaln" if candidate else "additive",
        "post_timestep_mlp_silu": candidate,
        "zero_initialized_per_layer_film": candidate,
    }


def _scheduler(arm_id: str) -> dict[str, Any]:
    if arm_id == "E-L0":
        return {
            "kind": "constant_with_warmup",
            "warmup_updates": 2500,
            "horizon_updates": None,
            "peak_learning_rate": 0.0003,
            "minimum_learning_rate": None,
        }
    return {
        "kind": "cosine_with_minimum",
        "warmup_updates": 50,
        "horizon_updates": 1000,
        "peak_learning_rate": 0.0003,
        "minimum_learning_rate": 0.000003,
    }


def _config_registry_entries(
    *,
    gpu_count: int,
    references: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, str | None], dict[str, Any]]:
    return {
        (spec.arm_id, spec.registry_scheduler_arm_id): {
            "scheduler_arm_id": spec.registry_scheduler_arm_id,
            "output_directory": spec.output_directory,
            "config": dict(references[spec.filename]),
        }
        for spec in _config_specs(gpu_count)
    }


def _build_registry_document(
    *,
    revision: str,
    gpu_count: int,
    checkpoint_reference: Mapping[str, Any],
    source_references: Sequence[Mapping[str, Any]],
    panel_reference: Mapping[str, Any],
    frequency_reference: Mapping[str, Any],
    fixture_reference: Mapping[str, Any],
    gradient_reference: Mapping[str, Any],
    config_references: Mapping[str, Mapping[str, Any]],
    verifier: Any,
) -> dict[str, Any]:
    entries = _config_registry_entries(
        gpu_count=gpu_count, references=config_references
    )
    evaluator_reference = next(
        dict(reference)
        for reference in source_references
        if reference["relative_path"] == "scripts/udlm/evaluate_denoising_panel.py"
    )
    scheduler_stage = {
        "stage_id": "scheduler",
        "order_index": 0,
        "training_seed": 17,
        "optimizer_updates": 100,
        "arm_order": ["E-L0", "E-L1"],
        "arms": [
            {
                "arm_id": arm_id,
                "attempt_id": f"scheduler-{arm_id.lower()}-g{gpu_count}",
                "role": "control" if index == 0 else "candidate",
                "scheduler": _scheduler(arm_id),
                "conditioner": _conditioner(False),
                "resolved_configs": [entries[(arm_id, None)]],
            }
            for index, arm_id in enumerate(("E-L0", "E-L1"))
        ],
        "selection_rule": {
            "rule_id": (
                "l1_if_pooled_loss_le_98pct_l0_and_two_bins_strictly_better_"
                "and_each_bin_le_102pct"
            ),
            "pooled_candidate_max_percent_of_control": 98,
            "per_bin_candidate_max_percent_of_control": 102,
            "minimum_strictly_better_bins": 2,
            "complete_failure_fallback_arm": "E-L0",
            "incomplete_evidence_winner": None,
        },
        "gradient_contract": None,
        "initialization_fixture": None,
    }
    conditioning_stage = {
        "stage_id": "conditioning",
        "order_index": 1,
        "training_seed": 17,
        "optimizer_updates": 500,
        "arm_order": ["E-A0", "E-A1"],
        "arms": [
            {
                "arm_id": arm_id,
                "attempt_id": f"conditioning-{arm_id.lower()}-g{gpu_count}",
                "role": "control" if index == 0 else "candidate",
                "scheduler": "selected_scheduler_arm",
                "conditioner": _conditioner(index == 1),
                "resolved_configs": [
                    entries[(arm_id, scheduler_id)] for scheduler_id in ("E-L0", "E-L1")
                ],
            }
            for index, arm_id in enumerate(("E-A0", "E-A1"))
        ],
        "selection_rule": {
            "rule_id": (
                "a1_if_exact_init_and_pooled_loss_le_98pct_a0_and_each_bin_le_"
                "102pct_and_pooled_accuracy_nondecreasing_and_registered_gradients_pass"
            ),
            "pooled_candidate_max_percent_of_control": 98,
            "per_bin_candidate_max_percent_of_control": 102,
            "pooled_clean_token_accuracy_nondecreasing": True,
            "exact_initialization_equality_required": True,
            "gradient_contract_sha256": gradient_reference["canonical_sha256"],
            "complete_failure_fallback_arm": "E-A0",
            "incomplete_evidence_winner": None,
        },
        "gradient_contract": dict(gradient_reference),
        "initialization_fixture": dict(fixture_reference),
    }
    accumulation = 8 if gpu_count == 1 else 4
    return {
        "schema_version": verifier.REGISTRY_SCHEMA_VERSION,
        "registry_id": verifier.EXPECTED_REGISTRY_ID,
        "status": verifier.EXPECTED_REGISTRY_STATUS,
        "claim_scope": verifier.EXPECTED_CLAIM_SCOPE,
        "firewall": {
            "final_generation_seeds": list(verifier.FINAL_GENERATION_SEEDS),
            "final_generation_seeds_forbidden": True,
            "generation_metrics_allowed": False,
            "health_gate_evidence_eligible": False,
            "superiority_evidence_eligible": False,
            "unregistered_attempts_allowed": False,
            "failed_or_missing_evidence_policy": "incomplete_no_winner",
        },
        "source": {
            "revision": revision,
            "clean": True,
            "pushed": True,
            "blobs": [dict(reference) for reference in source_references],
        },
        "common_training": {
            "training_seed": 17,
            "gpu_count": gpu_count,
            "prior_variant": "empirical_frequency",
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "micro_batch_size_per_process": MICRO_BATCH_SIZE,
            "accumulate_grad_batches": accumulation,
            "effective_global_batch_size": GLOBAL_BATCH_SIZE,
            "initialization": {
                "mode": "fresh_independent_mdlm_ema_warm_start_each_arm",
                "checkpoint": dict(checkpoint_reference),
                "weights": "ema",
                "optimizer_reset": True,
                "scheduler_reset": True,
                "global_step_reset": True,
                "ema_reset": True,
            },
            "artifact_schema_versions": dict(
                verifier.EXPECTED_ARTIFACT_SCHEMA_VERSIONS
            ),
        },
        "panel": {
            "artifact": dict(panel_reference),
            "ordered_token_ids_sha256": verifier.EXPECTED_PANEL_TOKEN_IDS_SHA256,
            "rows": verifier.EXPECTED_PANEL_ROWS,
            "content_tokens_per_time_bin": verifier.EXPECTED_PANEL_CONTENT_TOKENS,
            "time_bins": [0.1, 0.5, 0.9],
            "corruption_seed": verifier.EXPECTED_CORRUPTION_SEED,
            "weights": "ema",
            "device": "cpu",
            "batch_size": PANEL_BATCH_SIZE,
            "frequency_artifact": dict(frequency_reference),
            "frequency_ordered_text_sha256": (
                verifier.EXPECTED_FREQUENCY_ORDERED_TEXT_SHA256
            ),
            "evaluator_report_schema_version": verifier.EVALUATOR_REPORT_SCHEMA_VERSION,
            "evaluator_source": evaluator_reference,
        },
        "stages": [scheduler_stage, conditioning_stage],
    }


def _publish_bytes_exclusive(path: Path, payload: bytes) -> None:
    path = Path(os.path.abspath(os.fspath(path)))
    if not path.resolve(strict=False).is_relative_to(REPOSITORY_ROOT.resolve()):
        raise PreparationError("registry output must be inside the repository")
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to replace registry: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o644)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to replace registry: {path}") from error
        temporary.unlink()
        parent_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _assert_only_registry_change(relative_path: str) -> None:
    status = _run_git(
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"]
    ).stdout
    if status != f"?? {relative_path}\0":
        raise PreparationError(
            "concurrent worktree mutation detected after registry publication"
        )


def freeze_registry(gpu_count: int) -> dict[str, Any]:
    """Freeze a verifier-approved registry as the sole prospective R1 file."""

    launcher, verifier = _runtime_modules()
    specs = _config_specs(gpu_count)
    _validate_fresh_output_paths(specs)
    registry_path = REPOSITORY_ROOT / REGISTRY_RELATIVE_PATH
    revision = _require_clean_pushed_source()
    _ensure_registry_absent_at_revision(revision, REGISTRY_RELATIVE_PATH)
    if os.path.lexists(registry_path):
        raise FileExistsError(f"refusing to replace registry: {registry_path}")
    checkpoint = _validate_checkpoint(verifier)
    config_references, documents = _load_committed_configs(
        revision=revision, gpu_count=gpu_count, verifier=verifier
    )
    _validate_committed_config_reconstruction(
        gpu_count=gpu_count,
        checkpoint_reference=checkpoint,
        documents=documents,
        launcher=launcher,
        verifier=verifier,
    )
    source_references = [
        _blob_reference_from_git(revision, path) for path in SOURCE_PATHS
    ]
    panel = _json_reference_from_git(
        revision, verifier.EXPECTED_PANEL_PATH, verifier=verifier
    )
    frequency = _json_reference_from_git(
        revision, verifier.EXPECTED_FREQUENCY_PATH, verifier=verifier
    )
    fixture = _json_reference_from_git(
        revision, verifier.EXPECTED_INITIALIZATION_FIXTURE_PATH, verifier=verifier
    )
    gradient = _json_reference_from_git(
        revision, verifier.EXPECTED_GRADIENT_CONTRACT_PATH, verifier=verifier
    )
    candidate = _build_registry_document(
        revision=revision,
        gpu_count=gpu_count,
        checkpoint_reference=checkpoint,
        source_references=source_references,
        panel_reference=panel,
        frequency_reference=frequency,
        fixture_reference=fixture,
        gradient_reference=gradient,
        config_references=config_references,
        verifier=verifier,
    )
    payload = _json_bytes(candidate)
    raw_sha256 = hashlib.sha256(payload).hexdigest()
    canonical_sha256 = verifier.canonical_json_sha256(candidate)
    # This invokes the authoritative whole-registry validator, including all
    # local bytes, R0 Git blobs, checkpoint identity, config semantics, and R0
    # push status.  Publication happens only after it returns successfully.
    verifier.load_validated_registry(
        payload,
        relative_path=REGISTRY_RELATIVE_PATH,
        expected_raw_sha256=raw_sha256,
        expected_canonical_sha256=canonical_sha256,
        loader=verifier.local_blob_loader,
        git_blob_loader=verifier.git_blob_loader,
        git_ancestor_checker=verifier.git_ancestor_checker,
        git_pushed_checker=verifier.git_pushed_checker,
        git_diff_checker=verifier.git_diff_checker,
    )
    _publish_bytes_exclusive(registry_path, payload)
    if registry_path.read_bytes() != payload:
        raise PreparationError("published registry bytes differ from validation")
    _assert_only_registry_change(REGISTRY_RELATIVE_PATH)
    return {
        "status": "validated_registry_frozen_as_only_r1_candidate",
        "source_revision": revision,
        "gpu_count": gpu_count,
        "registry_relative_path": REGISTRY_RELATIVE_PATH,
        "registry_sha256": raw_sha256,
        "registry_canonical_sha256": canonical_sha256,
        "registry_size_bytes": len(payload),
        "config_count": 6,
        "gpu_probe_performed": False,
        "training_launched": False,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("materialize-configs", "freeze-registry"):
        child = subparsers.add_parser(command)
        child.add_argument("--gpu-count", type=int, choices=(1, 2), required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "materialize-configs":
        result = materialize_configs(args.gpu_count)
    else:
        result = freeze_registry(args.gpu_count)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

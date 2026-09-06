"""Prepare the prospective UDLM optimization screen without using a GPU.

The workflow starts only after the deterministic ten-update health panel has a
fully validated terminal-E receipt, then splits publication across two Git
revisions:

1. ``materialize-configs`` composes and exclusively publishes the six resolved
   Hydra configurations for the user-selected GPU count.  Commit and push
   those files as the exact sole child change after the reviewed health source
   (revision R0).
2. ``freeze-registry`` revalidates that health receipt, requires a clean,
   pushed R0, proves that every config is
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
    "experiments/udlm/protocols/optimization_screen_registry_v2.json"
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
    "experiments/udlm/prior_geometry/floor_selection_train_rows_10001_30000.json",
    "scripts/train.py",
    "scripts/udlm/audit_conditioning_initialization.py",
    "scripts/udlm/audit_empirical_prior_floor.py",
    "scripts/udlm/collect_optimization_screen_evidence.py",
    "scripts/udlm/evaluate_denoising_panel.py",
    "scripts/udlm/launch_optimization_screen.py",
    "scripts/udlm/launch_health_panel.py",
    "scripts/udlm/launch_train_pilot.py",
    "scripts/udlm/materialize_validation_panel.py",
    "scripts/udlm/prepare_optimization_screen_registry.py",
    "scripts/udlm/token_frequency_audit.py",
    "scripts/udlm/validate_health_panel.py",
    "scripts/udlm/verify_optimization_screen.py",
    "scripts/udlm/write_pilot_evidence.py",
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


def _health_module() -> Any:
    """Import the CPU-only health validator lazily for lightweight ``--help``."""

    import sys

    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    from scripts.udlm import validate_health_panel as health

    return health


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


def _validate_health_prerequisite(
    *, gpu_count: int, health_source_revision: str
) -> dict[str, Any]:
    """Validate the deterministic terminal-E receipt without probing a GPU."""

    health = _health_module()
    terminal_run = health.health_run_name(
        gpu_count, "udlm_categorical", health_source_revision
    )
    terminal_receipt = (
        REPOSITORY_ROOT / "output" / "udlm" / terminal_run / "pilot_exit_status.json"
    )
    try:
        evidence = health.validate_health_panel(
            terminal_receipt,
            expected_gpu_count=gpu_count,
            expected_source_revision=health_source_revision,
        )
    except (OSError, ValueError) as error:
        raise PreparationError(
            "the exact terminal-E ten-update health receipt is required before "
            f"screen preparation: {error}"
        ) from error
    if not isinstance(evidence, Mapping):
        raise PreparationError("health validator returned a non-object result")
    return dict(evidence)


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

    source_revision = _require_clean_pushed_source()
    _require_no_config_family_at_health_source(source_revision)
    health_evidence = _validate_health_prerequisite(
        gpu_count=gpu_count,
        health_source_revision=source_revision,
    )
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
    if _require_clean_pushed_source() != source_revision:
        raise PreparationError(
            "source revision changed while materialized configs were composed"
        )
    _publish_config_set_exclusive(directory, documents)
    _assert_only_materialized_config_changes(
        gpu_count=gpu_count,
        documents=documents,
        source_revision=source_revision,
    )
    return {
        "status": "six_resolved_configs_materialized_no_gpu_operation",
        "source_revision": source_revision,
        "gpu_count": gpu_count,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "micro_batch_size_per_process": MICRO_BATCH_SIZE,
        "accumulate_grad_batches": 8 if gpu_count == 1 else 4,
        "checkpoint_sha256": checkpoint["sha256"],
        "health_gate": health_evidence,
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
            "registry preparation requires a completely clean input worktree"
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
            "input HEAD has not been pushed exactly to the configured upstream"
        )
    return revision


def _require_exact_pushed_revision(revision: str) -> None:
    """Require both live Git refs to remain the exact reviewed revision."""

    if HEX_REVISION.fullmatch(revision) is None:
        raise PreparationError("expected source revision is invalid")
    for reference in ("HEAD", "@{upstream}"):
        observed = _run_git(["rev-parse", "--verify", reference]).stdout.strip()
        if observed != revision:
            raise PreparationError(
                "HEAD or upstream changed during registry publication"
            )


def _require_registry_candidate_bytes(
    *,
    candidate: Mapping[str, Any],
    payload: bytes,
    raw_sha256: str,
    canonical_sha256: str,
    verifier: Any,
    published_path: Path | None = None,
) -> None:
    """Recheck in-memory and, when present, exclusively published registry bytes."""

    if (
        _json_bytes(candidate) != payload
        or hashlib.sha256(payload).hexdigest() != raw_sha256
        or verifier.canonical_json_sha256(candidate) != canonical_sha256
    ):
        raise PreparationError("registry candidate bytes changed after validation")
    if published_path is None:
        return
    try:
        retained = verifier._stable_input_bytes(published_path)
    except (OSError, ValueError) as error:
        raise PreparationError(
            "published registry cannot be read as an exclusive stable file"
        ) from error
    if retained != payload:
        raise PreparationError("published registry bytes differ from validation")


def _assert_only_materialized_config_changes(
    *,
    gpu_count: int,
    documents: Mapping[str, Mapping[str, Any]],
    source_revision: str,
) -> None:
    """Require the exact generated config set to be the sole R0 candidate change."""

    if HEX_REVISION.fullmatch(source_revision) is None:
        raise PreparationError("materialization source revision is invalid")
    expected_names = {spec.filename for spec in _config_specs(gpu_count)}
    if set(documents) != expected_names:
        raise PreparationError("materialized config documents are not the exact six")

    directory = _config_directory(gpu_count)
    expected_payloads = {name: _json_bytes(documents[name]) for name in expected_names}

    def require_exact_files() -> None:
        if not directory.is_dir() or directory.is_symlink():
            raise PreparationError(
                "materialized config directory is unavailable or unsafe"
            )
        observed_names = {entry.name for entry in directory.iterdir()}
        if observed_names != expected_names:
            raise PreparationError(
                "materialized config directory must contain exactly six expected files"
            )
        for name in sorted(expected_names):
            path = directory / name
            if not path.is_file() or path.is_symlink():
                raise PreparationError(
                    f"materialized config is not a regular file: {path}"
                )
            if path.read_bytes() != expected_payloads[name]:
                raise PreparationError(
                    f"materialized config bytes changed during publication: {path}"
                )

    def require_source_revision() -> None:
        for reference in ("HEAD", "@{upstream}"):
            observed_revision = _run_git(
                ["rev-parse", "--verify", reference]
            ).stdout.strip()
            if observed_revision != source_revision:
                raise PreparationError(
                    "source revision changed during materialized-config publication"
                )

    require_exact_files()
    require_source_revision()

    repository = REPOSITORY_ROOT.resolve(strict=True)
    expected_status = {
        "?? " + (directory / name).relative_to(repository).as_posix()
        for name in expected_names
    }
    raw_status = _run_git(
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"]
    ).stdout
    if not raw_status.endswith("\0"):
        raise PreparationError(
            "materialized-config Git status is incomplete or unexpectedly empty"
        )
    observed_status = raw_status[:-1].split("\0")
    if (
        len(observed_status) != len(expected_status)
        or set(observed_status) != expected_status
    ):
        raise PreparationError(
            "exactly the six materialized configs must be the only worktree changes"
        )

    # Recheck path bytes and refs after Git inspection.  This makes a mutation
    # concurrent with the boundary observable instead of trusting stale reads.
    require_exact_files()
    require_source_revision()


def _single_parent_revision(revision: str) -> str:
    """Return the sole parent of R0, rejecting merges and root commits."""

    if HEX_REVISION.fullmatch(revision) is None:
        raise PreparationError("R0 revision is invalid")
    fields = (
        _run_git(["rev-list", "--parents", "-n", "1", revision]).stdout.strip().split()
    )
    if len(fields) != 2 or fields[0] != revision:
        raise PreparationError(
            "R0 must be a single-parent commit immediately after the health source"
        )
    parent = fields[1]
    if HEX_REVISION.fullmatch(parent) is None:
        raise PreparationError("R0 parent revision is invalid")
    return parent


def _git_tree_paths(revision: str, directory: str) -> set[str]:
    """List exact blob paths below one committed directory."""

    raw = _run_git(
        ["ls-tree", "-r", "--name-only", "-z", revision, "--", directory]
    ).stdout
    if not raw:
        return set()
    if not raw.endswith("\0"):
        raise PreparationError("Git tree path listing is truncated")
    paths = raw[:-1].split("\0")
    if len(paths) != len(set(paths)) or any(not path for path in paths):
        raise PreparationError("Git tree path listing is malformed")
    return set(paths)


def _validate_health_to_r0_transition(
    *, health_source_revision: str, r0_revision: str, gpu_count: int
) -> dict[str, Any]:
    """Prove R0 is exactly H plus the selected-W six-config family."""

    if _single_parent_revision(r0_revision) != health_source_revision:
        raise PreparationError(
            "the screen-config R0 parent must equal the validated health source"
        )
    selected_directory = CONFIG_DIRECTORY_TEMPLATE.format(gpu_count=gpu_count)
    expected_paths = {
        f"{selected_directory}/{spec.filename}" for spec in _config_specs(gpu_count)
    }
    changed_raw = _run_git(
        [
            "diff",
            "--name-only",
            "--no-renames",
            "-z",
            health_source_revision,
            r0_revision,
            "--",
        ]
    ).stdout
    if not changed_raw.endswith("\0"):
        raise PreparationError("H-to-R0 Git diff is empty or truncated")
    changed_paths = changed_raw[:-1].split("\0")
    if (
        len(changed_paths) != len(expected_paths)
        or set(changed_paths) != expected_paths
    ):
        raise PreparationError(
            "R0 must differ from the health source by exactly the selected six configs"
        )
    if _git_tree_paths(health_source_revision, selected_directory):
        raise PreparationError(
            "selected GPU-count configs already existed at the health source"
        )
    if _git_tree_paths(r0_revision, selected_directory) != expected_paths:
        raise PreparationError(
            "R0 selected GPU-count config directory is not the exact six-file set"
        )
    other_gpu_count = 2 if gpu_count == 1 else 1
    other_directory = CONFIG_DIRECTORY_TEMPLATE.format(gpu_count=other_gpu_count)
    if _git_tree_paths(health_source_revision, other_directory) or _git_tree_paths(
        r0_revision, other_directory
    ):
        raise PreparationError(
            "the unselected GPU-count config family must be absent from H and R0"
        )
    return {
        "health_source_revision": health_source_revision,
        "registry_source_revision": r0_revision,
        "allowed_config_paths": sorted(expected_paths),
        "health_source_is_registry_source_parent": True,
        "exact_config_only_transition_verified": True,
        "opposite_gpu_config_family_absent": True,
    }


def _require_no_config_family_at_health_source(source_revision: str) -> None:
    """Keep W as the sole materialization choice at the health revision."""

    for gpu_count in (1, 2):
        directory = CONFIG_DIRECTORY_TEMPLATE.format(gpu_count=gpu_count)
        if _git_tree_paths(source_revision, directory) or os.path.lexists(
            REPOSITORY_ROOT / directory
        ):
            raise PreparationError(
                "optimization-screen config families must both be absent before "
                "materialization"
            )


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
    health_evidence: Mapping[str, Any],
    health_source_transition: Mapping[str, Any],
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
        "prerequisite_health_gate": {
            "evidence": dict(health_evidence),
            "source_transition": dict(health_source_transition),
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

    revision = _require_clean_pushed_source()
    health_source_revision = _single_parent_revision(revision)
    health_evidence = _validate_health_prerequisite(
        gpu_count=gpu_count,
        health_source_revision=health_source_revision,
    )
    health_source_transition = _validate_health_to_r0_transition(
        health_source_revision=health_source_revision,
        r0_revision=revision,
        gpu_count=gpu_count,
    )
    launcher, verifier = _runtime_modules()
    specs = _config_specs(gpu_count)
    _validate_fresh_output_paths(specs)
    registry_path = REPOSITORY_ROOT / REGISTRY_RELATIVE_PATH
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
        health_evidence=health_evidence,
        health_source_transition=health_source_transition,
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
        git_sole_parent_checker=verifier.git_sole_parent_checker,
        git_tree_paths_loader=verifier.git_tree_paths_loader,
        git_pushed_checker=verifier.git_pushed_checker,
        git_diff_checker=verifier.git_diff_checker,
    )
    if _require_clean_pushed_source() != revision:
        raise PreparationError(
            "source revision changed while the registry candidate was validated"
        )
    _require_registry_candidate_bytes(
        candidate=candidate,
        payload=payload,
        raw_sha256=raw_sha256,
        canonical_sha256=canonical_sha256,
        verifier=verifier,
    )
    _publish_bytes_exclusive(registry_path, payload)
    _require_exact_pushed_revision(revision)
    _require_registry_candidate_bytes(
        candidate=candidate,
        payload=payload,
        raw_sha256=raw_sha256,
        canonical_sha256=canonical_sha256,
        verifier=verifier,
        published_path=registry_path,
    )
    _assert_only_registry_change(REGISTRY_RELATIVE_PATH)
    _require_exact_pushed_revision(revision)
    _require_registry_candidate_bytes(
        candidate=candidate,
        payload=payload,
        raw_sha256=raw_sha256,
        canonical_sha256=canonical_sha256,
        verifier=verifier,
        published_path=registry_path,
    )
    return {
        "status": "validated_registry_frozen_as_only_r1_candidate",
        "source_revision": revision,
        "health_source_revision": health_source_revision,
        "health_terminal_receipt_sha256": health_evidence["terminal_receipt"]["sha256"],
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

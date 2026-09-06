"""Prove exact A0/A1 logits at the registered pre-optimizer warm start.

This producer is intentionally CPU-only.  It loads the two full conditioning
models one at a time from their exact registered resolved configurations,
applies the same verified MDLM EMA checkpoint independently to each model,
and evaluates a Git-bound literal probe.  The retained evidence is two raw
little-endian float32 logit blobs plus one strict JSON audit object.

The audit is an engineering-screen prerequisite, not a quality metric.  It
does not train either arm and does not inspect accelerator inventory.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPOSITORY_ROOT.parents[1]
SCHEMA_VERSION = 1
FIXTURE_SCHEMA_VERSION = 1
EXPECTED_FIXTURE_PURPOSE = (
    "pre_optimizer_exact_additive_vs_film_warm_start_logit_identity"
)
EXPECTED_FIXTURE_PATH = "experiments/udlm/protocols/conditioning_init_fixture_v1.json"
EXPECTED_FIXTURE_SHA256 = (
    "a7069082bd9a5a7d76d7d52b324345f4a46ba2dea386b87411b1fcb8112cc989"
)
EXPECTED_FIXTURE_SIZE_BYTES = 643
EXPECTED_FIXTURE_CANONICAL_SHA256 = (
    "96ed170d2e3db2c68101dd07d603ff44db7f77a95c3b9578d6c378d1e1f85d1b"
)
PROBE_PHASE = "after_mdlm_ema_load_before_training_rng_reseed_and_optimizer_creation"
LOGITS_DTYPE = "float32-little-endian-c-order"
ARM_IDS = ("E-A0", "E-A1")
SCHEDULER_ARM_IDS = ("E-L0", "E-L1")
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
HEX_GIT_REVISION = re.compile(r"[0-9a-f]{40}\Z")


class InitializationAuditError(RuntimeError):
    """Raised when the audit is not exactly bound to its frozen contract."""


@dataclass(frozen=True)
class ArmContract:
    """One registered conditioning arm and its exact resolved config."""

    arm_id: str
    output_directory: str
    config_ref: Mapping[str, Any]
    config: Mapping[str, Any]


@dataclass(frozen=True)
class AuditContract:
    """All immutable inputs needed for one A0-versus-A1 audit."""

    registry: Any
    source_revision: str
    checkpoint_ref: Mapping[str, Any]
    checkpoint_path: Path
    fixture_ref: Mapping[str, Any]
    fixture: Mapping[str, Any]
    producer_ref: Mapping[str, Any]
    reference: ArmContract
    candidate: ArmContract


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise InitializationAuditError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise InitializationAuditError(f"non-finite JSON constant: {value}")


def strict_json_loads(payload: bytes, *, label: str) -> object:
    """Decode strict finite UTF-8 JSON while rejecting duplicate keys."""

    try:
        source = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise InitializationAuditError(f"{label} is not UTF-8") from error
    try:
        return json.loads(
            source,
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, OverflowError, ValueError) as error:
        if isinstance(error, InitializationAuditError):
            raise
        raise InitializationAuditError(f"{label} is not strict JSON") from error


def canonical_json_sha256(value: object) -> str:
    """Return the canonical JSON digest used by the screen verifier."""

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise InitializationAuditError("value is not canonical finite JSON") from error
    return hashlib.sha256(encoded).hexdigest()


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or HEX_SHA256.fullmatch(value) is None:
        raise InitializationAuditError(f"{label} must be a lowercase SHA-256")
    return value


def _git_revision(value: object, label: str) -> str:
    if not isinstance(value, str) or HEX_GIT_REVISION.fullmatch(value) is None:
        raise InitializationAuditError(f"{label} must be a full Git revision")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InitializationAuditError(f"{label} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise InitializationAuditError(f"{label} has invalid keys")


def _relative_path(value: object, label: str, *, suffix: str | None = None) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise InitializationAuditError(f"{label} must be a normalized relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or (suffix is not None and path.suffix != suffix)
    ):
        raise InitializationAuditError(f"{label} must be a normalized relative path")
    return value


def _root_path(root: str, relative_path: str, *, must_exist: bool) -> Path:
    if root == "repository":
        base = REPOSITORY_ROOT
    elif root == "project":
        base = PROJECT_ROOT
    else:
        raise InitializationAuditError("artifact root must be repository or project")
    relative = _relative_path(relative_path, "artifact path")
    candidate = base.joinpath(*PurePosixPath(relative).parts)
    current = base
    for part in PurePosixPath(relative).parts[:-1]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            break
        if stat.S_ISLNK(mode):
            raise InitializationAuditError(
                f"artifact path has a symlink ancestor: {candidate}"
            )
    if must_exist:
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as error:
            raise InitializationAuditError(
                f"artifact is missing: {candidate}"
            ) from error
        if not resolved.is_relative_to(base.resolve(strict=True)):
            raise InitializationAuditError(f"artifact escapes its root: {candidate}")
        if resolved != candidate:
            raise InitializationAuditError(f"artifact path is a symlink: {candidate}")
    return candidate


def _read_stable_file(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise InitializationAuditError(f"cannot open artifact: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise InitializationAuditError(
                f"artifact must be one exclusive regular file: {path}"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        payload = b"".join(chunks)
        if identity_before != identity_after or len(payload) != before.st_size:
            raise InitializationAuditError(f"artifact changed while read: {path}")
        return payload
    finally:
        os.close(descriptor)


def _validate_blob_ref(
    value: object,
    label: str,
    *,
    root: str | None = None,
    suffix: str | None = None,
    json_ref: bool = False,
    config_ref: bool = False,
) -> dict[str, Any]:
    if json_ref and config_ref:
        raise InitializationAuditError("artifact cannot be both JSON and config ref")
    ref = _mapping(value, label)
    expected = {"root", "relative_path", "sha256", "size_bytes"}
    if json_ref:
        expected |= {"schema_version", "canonical_sha256"}
    elif config_ref:
        expected |= {"canonical_sha256"}
    _exact_keys(ref, expected, label)
    observed_root = ref.get("root")
    if observed_root not in {"repository", "project"} or (
        root is not None and observed_root != root
    ):
        raise InitializationAuditError(f"{label} has an invalid root")
    relative_path = _relative_path(
        ref.get("relative_path"), f"{label}.relative_path", suffix=suffix
    )
    digest = _sha256(ref.get("sha256"), f"{label}.sha256")
    size_bytes = ref.get("size_bytes")
    if type(size_bytes) is not int or size_bytes <= 0:
        raise InitializationAuditError(f"{label}.size_bytes must be positive")
    normalized = {
        "root": observed_root,
        "relative_path": relative_path,
        "sha256": digest,
        "size_bytes": size_bytes,
    }
    if json_ref:
        schema_version = ref.get("schema_version")
        if type(schema_version) is not int or schema_version <= 0:
            raise InitializationAuditError(f"{label}.schema_version is invalid")
        normalized.update(
            {
                "schema_version": schema_version,
                "canonical_sha256": _sha256(
                    ref.get("canonical_sha256"), f"{label}.canonical_sha256"
                ),
            }
        )
    elif config_ref:
        normalized["canonical_sha256"] = _sha256(
            ref.get("canonical_sha256"), f"{label}.canonical_sha256"
        )
    return normalized


def _load_exact_json_ref(
    value: object, label: str, *, config_ref: bool = False
) -> tuple[dict[str, Any], Any]:
    ref = _validate_blob_ref(
        value,
        label,
        json_ref=not config_ref,
        config_ref=config_ref,
        suffix=".json",
    )
    path = _root_path(ref["root"], ref["relative_path"], must_exist=True)
    payload = _read_stable_file(path)
    if (
        len(payload) != ref["size_bytes"]
        or hashlib.sha256(payload).hexdigest() != ref["sha256"]
    ):
        raise InitializationAuditError(f"{label} bytes differ from the registry")
    parsed = strict_json_loads(payload, label=label)
    if canonical_json_sha256(parsed) != ref["canonical_sha256"]:
        raise InitializationAuditError(f"{label} canonical digest is unmatched")
    if not isinstance(parsed, Mapping):
        raise InitializationAuditError(f"{label} must contain a JSON object")
    if not config_ref and parsed.get("schema_version") != ref["schema_version"]:
        raise InitializationAuditError(f"{label} schema is unmatched")
    return ref, parsed


def _validate_fixture(value: object, checkpoint_sha256: str) -> Mapping[str, Any]:
    fixture = _mapping(value, "conditioning initialization fixture")
    _exact_keys(
        fixture,
        {
            "schema_version",
            "purpose",
            "checkpoint_sha256",
            "probe_phase",
            "input_ids",
            "attention_mask",
            "noise_tensor",
            "timestep_tensor",
        },
        "conditioning initialization fixture",
    )
    if (
        fixture.get("schema_version") != FIXTURE_SCHEMA_VERSION
        or fixture.get("purpose") != EXPECTED_FIXTURE_PURPOSE
        or fixture.get("checkpoint_sha256") != checkpoint_sha256
        or fixture.get("probe_phase") != PROBE_PHASE
    ):
        raise InitializationAuditError(
            "conditioning initialization fixture is unmatched"
        )
    input_ids = fixture.get("input_ids")
    attention_mask = fixture.get("attention_mask")
    if (
        not isinstance(input_ids, list)
        or not input_ids
        or any(
            not isinstance(row, list)
            or not row
            or any(type(token) is not int or token < 0 for token in row)
            for row in input_ids
        )
    ):
        raise InitializationAuditError("fixture input_ids are invalid")
    sequence_length = len(input_ids[0])
    if any(len(row) != sequence_length for row in input_ids):
        raise InitializationAuditError("fixture input_ids must be rectangular")
    if (
        not isinstance(attention_mask, list)
        or len(attention_mask) != len(input_ids)
        or any(
            not isinstance(row, list)
            or len(row) != sequence_length
            or any(type(item) is not int or item not in {0, 1} for item in row)
            for row in attention_mask
        )
    ):
        raise InitializationAuditError("fixture attention_mask is invalid")
    for name in ("noise_tensor", "timestep_tensor"):
        values = fixture.get(name)
        if (
            not isinstance(values, list)
            or len(values) != len(input_ids)
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                or float(item) < 0
                for item in values
            )
        ):
            raise InitializationAuditError(f"fixture {name} is invalid")
    return fixture


def _screen_module() -> Any:
    """Import the standard-library verifier after making the repo importable."""

    repository = str(REPOSITORY_ROOT)
    if repository not in sys.path:
        sys.path.insert(0, repository)
    return importlib.import_module("scripts.udlm.verify_optimization_screen")


def _load_validated_registry(
    registry_path: Path,
    *,
    expected_raw_sha256: str,
    expected_canonical_sha256: str,
) -> Any:
    screen = _screen_module()
    normalized = Path(os.path.abspath(os.fspath(registry_path)))
    if not normalized.is_relative_to(REPOSITORY_ROOT) or normalized.suffix != ".json":
        raise InitializationAuditError("registry must be a repository JSON file")
    payload = _read_stable_file(normalized)
    relative_path = normalized.relative_to(REPOSITORY_ROOT).as_posix()
    try:
        return screen.load_validated_registry(
            payload,
            relative_path=relative_path,
            expected_raw_sha256=_sha256(
                expected_raw_sha256, "expected registry raw digest"
            ),
            expected_canonical_sha256=_sha256(
                expected_canonical_sha256, "expected registry canonical digest"
            ),
            loader=screen.local_blob_loader,
            git_blob_loader=screen.git_blob_loader,
            git_ancestor_checker=screen.git_ancestor_checker,
            git_pushed_checker=screen.git_pushed_checker,
            git_diff_checker=screen.git_diff_checker,
        )
    except Exception as error:
        if isinstance(error, InitializationAuditError):
            raise
        raise InitializationAuditError("registry validation failed") from error


def _git_stdout(*arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise InitializationAuditError("Git identity check failed") from error
    return result.stdout.strip()


def _validate_source_revision(registry: Any, source_revision: str) -> Mapping[str, Any]:
    source_revision = _git_revision(source_revision, "audit source revision")
    if _git_stdout("rev-parse", "HEAD") != source_revision:
        raise InitializationAuditError("audit source revision is not checked out")
    screen = _screen_module()
    registry_revision = _git_revision(
        registry.data["source"]["revision"], "registry source revision"
    )
    if not screen.git_ancestor_checker(registry_revision, source_revision):
        raise InitializationAuditError("audit source is not a registry descendant")
    if not screen.git_pushed_checker(source_revision):
        raise InitializationAuditError("audit source revision is not pushed")
    producer_ref = next(
        (
            _validate_blob_ref(
                ref,
                "conditioning initialization producer",
                root="repository",
                suffix=".py",
            )
            for ref in registry.data["source"]["blobs"]
            if ref.get("relative_path")
            == "scripts/udlm/audit_conditioning_initialization.py"
        ),
        None,
    )
    if producer_ref is None:
        raise InitializationAuditError("registry omits the audit producer source")
    producer_path = _root_path(
        producer_ref["root"], producer_ref["relative_path"], must_exist=True
    )
    live_payload = _read_stable_file(producer_path)
    try:
        committed_payload = screen.git_blob_loader(
            source_revision, PurePosixPath(producer_ref["relative_path"])
        )
    except Exception as error:
        raise InitializationAuditError(
            "audit producer is absent from the source revision"
        ) from error
    for label, payload in (("live", live_payload), ("committed", committed_payload)):
        if (
            len(payload) != producer_ref["size_bytes"]
            or hashlib.sha256(payload).hexdigest() != producer_ref["sha256"]
        ):
            raise InitializationAuditError(
                f"{label} audit producer differs from the registry source binding"
            )
    return producer_ref


def _registered_arm(
    conditioning_stage: Mapping[str, Any], arm_id: str, scheduler_arm_id: str
) -> ArmContract:
    arms = conditioning_stage.get("arms")
    if not isinstance(arms, list):
        raise InitializationAuditError("conditioning stage has no registered arms")
    arm = next(
        (
            item
            for item in arms
            if isinstance(item, Mapping) and item.get("arm_id") == arm_id
        ),
        None,
    )
    if arm is None:
        raise InitializationAuditError(f"conditioning stage omits {arm_id}")
    entries = arm.get("resolved_configs")
    if not isinstance(entries, list):
        raise InitializationAuditError(f"{arm_id} has no resolved configs")
    entry = next(
        (
            item
            for item in entries
            if isinstance(item, Mapping)
            and item.get("scheduler_arm_id") == scheduler_arm_id
        ),
        None,
    )
    if entry is None:
        raise InitializationAuditError(
            f"{arm_id} has no config for scheduler {scheduler_arm_id}"
        )
    output_directory = _relative_path(
        entry.get("output_directory"), f"{arm_id} output directory"
    )
    if not output_directory.startswith("output/udlm/screens/"):
        raise InitializationAuditError(f"{arm_id} output directory is outside screens")
    config_ref, config = _load_exact_json_ref(
        entry.get("config"),
        f"{arm_id} registered resolved config",
        config_ref=True,
    )
    config = _mapping(config, f"{arm_id} registered resolved config")
    training = _mapping(config.get("training"), f"{arm_id} training config")
    udlm = _mapping(training.get("udlm"), f"{arm_id} UDLM config")
    expected_variant = "additive" if arm_id == "E-A0" else "film_adaln"
    expected_zero = arm_id == "E-A0"
    if (
        config.get("seed") != 17
        or training.get("diffusion") != "udlm"
        or training.get("init_from_mdlm_ema") is not True
        or training.get("reseed_after_model_initialization") is not True
        or udlm.get("prior_variant") != "empirical_frequency"
        or udlm.get("conditioning_variant") != expected_variant
        or udlm.get("zero_init_conditioning") is not expected_zero
    ):
        raise InitializationAuditError(f"{arm_id} config semantics are unmatched")
    return ArmContract(
        arm_id=arm_id,
        output_directory=output_directory,
        config_ref=config_ref,
        config=dict(config),
    )


def build_audit_contract(
    registry: Any, *, scheduler_arm_id: str, source_revision: str
) -> AuditContract:
    """Select and recheck the two registered contingent A0/A1 configs."""

    if scheduler_arm_id not in SCHEDULER_ARM_IDS:
        raise InitializationAuditError("scheduler arm must be E-L0 or E-L1")
    producer_ref = _validate_source_revision(registry, source_revision)
    stages = registry.data.get("stages")
    if not isinstance(stages, list) or len(stages) != 2:
        raise InitializationAuditError("registry must contain two stages")
    conditioning_stage = stages[1]
    if conditioning_stage.get("stage_id") != "conditioning":
        raise InitializationAuditError("registry conditioning stage is out of order")
    checkpoint_ref = _validate_blob_ref(
        registry.data["common_training"]["initialization"]["checkpoint"],
        "registered MDLM checkpoint",
        root="project",
        suffix=".ckpt",
    )
    checkpoint_path = _root_path(
        checkpoint_ref["root"], checkpoint_ref["relative_path"], must_exist=True
    )
    fixture_ref, raw_fixture = _load_exact_json_ref(
        conditioning_stage.get("initialization_fixture"),
        "conditioning initialization fixture",
    )
    expected_fixture_ref = {
        "root": "repository",
        "relative_path": EXPECTED_FIXTURE_PATH,
        "sha256": EXPECTED_FIXTURE_SHA256,
        "size_bytes": EXPECTED_FIXTURE_SIZE_BYTES,
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "canonical_sha256": EXPECTED_FIXTURE_CANONICAL_SHA256,
    }
    if fixture_ref != expected_fixture_ref:
        raise InitializationAuditError("conditioning fixture reference is unmatched")
    fixture = _validate_fixture(raw_fixture, checkpoint_ref["sha256"])
    reference = _registered_arm(conditioning_stage, "E-A0", scheduler_arm_id)
    candidate = _registered_arm(conditioning_stage, "E-A1", scheduler_arm_id)
    for arm in (reference, candidate):
        training = _mapping(arm.config.get("training"), f"{arm.arm_id} training")
        if training.get("init_from_mdlm_checkpoint_sha256") != checkpoint_ref["sha256"]:
            raise InitializationAuditError(
                f"{arm.arm_id} checkpoint digest differs from the registry"
            )
    return AuditContract(
        registry=registry,
        source_revision=source_revision,
        checkpoint_ref=checkpoint_ref,
        checkpoint_path=checkpoint_path,
        fixture_ref=fixture_ref,
        fixture=fixture,
        producer_ref=producer_ref,
        reference=reference,
        candidate=candidate,
    )


def _load_ml_stack() -> SimpleNamespace:
    """Load the model stack only after masking every accelerator from this process."""

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    repository = str(REPOSITORY_ROOT)
    source = str(REPOSITORY_ROOT / "src")
    for path in (repository, source):
        if path not in sys.path:
            sys.path.insert(0, path)
    import lightning as lightning
    import numpy
    import torch
    from omegaconf import OmegaConf

    from genmol.model import GenMol

    return SimpleNamespace(
        torch=torch,
        numpy=numpy,
        OmegaConf=OmegaConf,
        GenMol=GenMol,
        seed_everything=lightning.seed_everything,
    )


def _assert_model_is_cpu(model: Any) -> None:
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise InitializationAuditError("registered model has no backbone")
    state = backbone.state_dict()
    if not state or any(tensor.device.type != "cpu" for tensor in state.values()):
        raise InitializationAuditError("initialization audit model must remain on CPU")


def _serialize_logits(logits: Any, stack: SimpleNamespace) -> tuple[bytes, list[int]]:
    torch = stack.torch
    if not isinstance(logits, torch.Tensor) or logits.dtype != torch.float32:
        raise InitializationAuditError("initialization logits must be float32")
    if logits.device.type != "cpu" or logits.is_sparse or logits.is_quantized:
        raise InitializationAuditError(
            "initialization logits must be dense CPU tensors"
        )
    if logits.ndim != 3 or any(int(size) <= 0 for size in logits.shape):
        raise InitializationAuditError("initialization logits must have [B,L,V] shape")
    if not bool(torch.isfinite(logits).all().item()):
        raise InitializationAuditError(
            "initialization logits contain non-finite values"
        )
    contiguous = logits.detach().contiguous()
    array = stack.numpy.asarray(contiguous.numpy(), dtype="<f4", order="C")
    payload = array.tobytes(order="C")
    shape = [int(size) for size in contiguous.shape]
    if len(payload) != math.prod(shape) * 4:
        raise InitializationAuditError("serialized logit byte size is inconsistent")
    return payload, shape


def probe_registered_arm(
    arm: ArmContract,
    *,
    fixture: Mapping[str, Any],
    checkpoint_path: Path,
    checkpoint_sha256: str,
    stack: SimpleNamespace | None = None,
) -> tuple[bytes, list[int]]:
    """Construct, warm-start, and probe one full registered model on CPU."""

    stack = _load_ml_stack() if stack is None else stack
    torch = stack.torch
    seed = arm.config.get("seed")
    if type(seed) is not int or seed != 17:
        raise InitializationAuditError("conditioning audit requires training seed 17")
    applied_seed = stack.seed_everything(seed, workers=True)
    if type(applied_seed) is not int or applied_seed != seed:
        raise InitializationAuditError(
            "model-construction seed was not applied exactly"
        )
    model = stack.GenMol(stack.OmegaConf.create(dict(arm.config)))
    _assert_model_is_cpu(model)
    expected_variant = "additive" if arm.arm_id == "E-A0" else "film_adaln"
    if (
        getattr(model, "diffusion_type", None) != "udlm"
        or getattr(model.backbone, "conditioning_variant", None) != expected_variant
    ):
        raise InitializationAuditError(f"{arm.arm_id} constructed the wrong topology")
    report = model.initialize_from_mdlm_checkpoint(
        checkpoint_path,
        use_ema=True,
        expected_sha256=checkpoint_sha256,
    )
    if (
        not isinstance(report, Mapping)
        or report.get("source_sha256") != checkpoint_sha256
        or report.get("expected_source_sha256") != checkpoint_sha256
        or report.get("weights") != "ema"
        or report.get("byte_identity_verified_before_and_after_load") is not True
    ):
        raise InitializationAuditError(
            f"{arm.arm_id} did not complete the verified MDLM EMA warm start"
        )
    _assert_model_is_cpu(model)
    input_ids = torch.tensor(fixture["input_ids"], dtype=torch.long, device="cpu")
    attention_mask = torch.tensor(
        fixture["attention_mask"], dtype=torch.long, device="cpu"
    )
    timestep = torch.tensor(
        fixture["timestep_tensor"], dtype=torch.float32, device="cpu"
    )
    noise = torch.tensor(fixture["noise_tensor"], dtype=torch.float32, device="cpu")
    model_noise = model.mdlm.sigma(timestep)
    if model_noise.dtype != torch.float32 or not torch.equal(model_noise, noise):
        raise InitializationAuditError(
            "fixture noise is not the registered model's exact float32 sigma(t)"
        )
    model.backbone.eval()
    with torch.inference_mode():
        logits = model.backbone(
            input_ids,
            attention_mask,
            noise_level=noise,
        ).logits
    return _serialize_logits(logits, stack)


def _output_path(relative_directory: str, arm_id: str) -> Path:
    directory = _root_path("repository", relative_directory, must_exist=False)
    path = directory / f"initialization_logits_{arm_id}.bin"
    if not path.is_relative_to(REPOSITORY_ROOT):
        raise InitializationAuditError("logit output escapes the repository")
    return path


def _normalize_report_path(path: Path) -> Path:
    normalized = Path(os.path.abspath(os.fspath(path)))
    if normalized.suffix != ".json" or not normalized.is_relative_to(REPOSITORY_ROOT):
        raise InitializationAuditError("audit output must be repository JSON")
    _root_path(
        "repository",
        normalized.relative_to(REPOSITORY_ROOT).as_posix(),
        must_exist=False,
    )
    return normalized


def _write_exclusive(path: Path, payload: bytes) -> None:
    if not isinstance(payload, bytes) or not payload:
        raise InitializationAuditError("refusing to publish an empty artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    _root_path(
        "repository",
        path.relative_to(REPOSITORY_ROOT).as_posix(),
        must_exist=False,
    )
    if os.path.lexists(path):
        raise FileExistsError(
            f"refusing to replace initialization audit artifact: {path}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to replace initialization audit artifact: {path}"
            ) from error
        temporary.unlink()
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _artifact_ref(path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "root": "repository",
        "relative_path": path.relative_to(REPOSITORY_ROOT).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def run_audit(
    contract: AuditContract,
    *,
    output_path: Path,
    stack: SimpleNamespace | None = None,
) -> Mapping[str, Any]:
    """Execute both probes sequentially and publish immutable evidence."""

    output_path = _normalize_report_path(output_path)
    reference_path = _output_path(
        contract.reference.output_directory, contract.reference.arm_id
    )
    candidate_path = _output_path(
        contract.candidate.output_directory, contract.candidate.arm_id
    )
    outputs = (reference_path, candidate_path, output_path)
    if len(set(outputs)) != 3:
        raise InitializationAuditError("initialization audit output paths collide")
    existing = [path for path in outputs if os.path.lexists(path)]
    if existing:
        raise FileExistsError(
            f"refusing to replace initialization audit artifact: {existing[0]}"
        )

    shared_stack = _load_ml_stack() if stack is None else stack
    reference_payload, reference_shape = probe_registered_arm(
        contract.reference,
        fixture=contract.fixture,
        checkpoint_path=contract.checkpoint_path,
        checkpoint_sha256=contract.checkpoint_ref["sha256"],
        stack=shared_stack,
    )
    gc.collect()
    candidate_payload, candidate_shape = probe_registered_arm(
        contract.candidate,
        fixture=contract.fixture,
        checkpoint_path=contract.checkpoint_path,
        checkpoint_sha256=contract.checkpoint_ref["sha256"],
        stack=shared_stack,
    )
    gc.collect()
    if reference_shape != candidate_shape:
        raise InitializationAuditError("A0/A1 initialization logit shapes differ")

    reference_ref = _artifact_ref(reference_path, reference_payload)
    candidate_ref = _artifact_ref(candidate_path, candidate_payload)
    exact_equal = reference_payload == candidate_payload
    audit = {
        "schema_version": SCHEMA_VERSION,
        "reference_arm_id": contract.reference.arm_id,
        "candidate_arm_id": contract.candidate.arm_id,
        "fixture": dict(contract.fixture_ref),
        "source_revision": contract.source_revision,
        "checkpoint_sha256": contract.checkpoint_ref["sha256"],
        "reference_config_canonical_sha256": contract.reference.config_ref[
            "canonical_sha256"
        ],
        "candidate_config_canonical_sha256": contract.candidate.config_ref[
            "canonical_sha256"
        ],
        "probe_phase": PROBE_PHASE,
        "input_ids_sha256": canonical_json_sha256(contract.fixture["input_ids"]),
        "attention_mask_sha256": canonical_json_sha256(
            contract.fixture["attention_mask"]
        ),
        "noise_tensor_sha256": canonical_json_sha256(contract.fixture["noise_tensor"]),
        "timestep_tensor_sha256": canonical_json_sha256(
            contract.fixture["timestep_tensor"]
        ),
        "logits_dtype": LOGITS_DTYPE,
        "logits_shape": reference_shape,
        "reference_logits": reference_ref,
        "candidate_logits": candidate_ref,
        "producer_source": dict(contract.producer_ref),
        "exact_equal": exact_equal,
    }
    report_payload = (
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _write_exclusive(reference_path, reference_payload)
    _write_exclusive(candidate_path, candidate_payload)
    _write_exclusive(output_path, report_payload)
    return audit


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--expected-registry-sha256", required=True)
    parser.add_argument("--expected-registry-canonical-sha256", required=True)
    parser.add_argument("--scheduler-arm", choices=SCHEDULER_ARM_IDS, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        registry = _load_validated_registry(
            args.registry,
            expected_raw_sha256=args.expected_registry_sha256,
            expected_canonical_sha256=args.expected_registry_canonical_sha256,
        )
        contract = build_audit_contract(
            registry,
            scheduler_arm_id=args.scheduler_arm,
            source_revision=args.source_revision,
        )
        audit = run_audit(contract, output_path=args.output)
    except (InitializationAuditError, FileExistsError, OSError) as error:
        print(f"conditioning initialization audit failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(audit, sort_keys=True, allow_nan=False))
    return 0 if audit["exact_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

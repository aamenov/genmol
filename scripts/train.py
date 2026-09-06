# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import hashlib
import json
import math
import os
import re
import stat
import struct
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_PILOT_ENVIRONMENT_KEYS = {
    "GENMOL_TRAIN_EXPECTED_SOURCE_REVISION",
    "GENMOL_TRAIN_EXPECTED_CONFIG_SHA256",
    "GENMOL_TRAIN_EXPECTED_ARGV_SHA256",
    "GENMOL_TRAIN_RUNTIME_CONFIG_PATH",
    "GENMOL_TRAIN_SUMMARY_PATH",
    "GENMOL_TRAIN_EXPECTED_SUMMARY_SCHEMA_VERSION",
    "GENMOL_TRAIN_EXPECTED_FINAL_CHECKPOINT_PATH",
    "GENMOL_TRAIN_EXPECTED_MAX_STEPS",
    "GENMOL_TRAIN_EXPECTED_WORLD_SIZE",
    "GENMOL_TRAIN_LAUNCH_MANIFEST_PATH",
    "GENMOL_TRAIN_EXPECTED_LAUNCH_MANIFEST_SHA256",
    "GENMOL_TRAIN_SELECTED_GPU_UUIDS_JSON",
}
_RUNTIME_CONFIG_SCHEMA_VERSION = 2
_TRAINING_SUMMARY_SCHEMA_VERSION = 5
_MAX_TRAINING_SEED = 2**32 - 1
_PINNED_LIGHTNING_VERSION = "2.5.1"
_HOSTED_STREAM_RANK_PARTITION_POLICY = (
    "huggingface_split_dataset_by_node_disjoint_rank_streams"
)
_CONTROLLED_PYTHON_ENVIRONMENT = {
    "PYTHONPATH": os.pathsep.join(
        [str(_REPOSITORY_ROOT / "src"), str(_REPOSITORY_ROOT)]
    ),
    "PYTHONNOUSERSITE": "1",
    "PYTHONOPTIMIZE": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
}


def _canonical_json_sha256(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _strict_json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value):
    raise RuntimeError(f"non-finite JSON constant: {value}")


def _finite_json_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise RuntimeError(f"non-finite JSON number: {value}")
    return parsed


def _strict_json_loads(payload, *, label):
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"pilot {label} is not valid UTF-8") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_json_pairs,
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_float,
        )
    except json.JSONDecodeError as error:
        raise RuntimeError(f"pilot {label} is not valid JSON") from error


def _parse_selected_gpu_uuids(value):
    try:
        parsed = _strict_json_loads(
            value.encode("utf-8"), label="selected GPU UUID contract"
        )
    except UnicodeEncodeError as error:
        raise RuntimeError(
            "GENMOL_TRAIN_SELECTED_GPU_UUIDS_JSON must be UTF-8 JSON"
        ) from error
    if (
        not isinstance(parsed, list)
        or not parsed
        or any(
            not isinstance(uuid, str)
            or not uuid.startswith("GPU-")
            or len(uuid) <= len("GPU-")
            or "," in uuid
            for uuid in parsed
        )
        or len(set(parsed)) != len(parsed)
    ):
        raise RuntimeError(
            "GENMOL_TRAIN_SELECTED_GPU_UUIDS_JSON must be a nonempty array of "
            "unique NVIDIA GPU UUID strings"
        )
    return parsed


def _validate_selected_gpu_exposure(selected_gpu_uuids):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(selected_gpu_uuids):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES disagrees with the selected GPU UUID contract"
        )
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("CUDA_DEVICE_ORDER must be PCI_BUS_ID for the pilot")


def _in_repository_artifact_path(value, *, suffix, label):
    path = Path(os.path.abspath(os.fspath(value)))
    if (
        path == _REPOSITORY_ROOT
        or _REPOSITORY_ROOT not in path.parents
        or path.suffix != suffix
    ):
        raise RuntimeError(f"pilot {label} must be an in-repository {suffix} file")
    return path


def _pilot_base_argv():
    """Remove only Lightning's exact rank-specific Hydra suffix."""
    argv = list(sys.argv)
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank in (None, "0"):
        return argv
    if not local_rank.isdigit() or int(local_rank) <= 0 or len(argv) < 4:
        raise RuntimeError("pilot child has an invalid distributed LOCAL_RANK")
    base_argv, suffix = argv[:-3], argv[-3:]
    expected_job_name = f"hydra.job.name=train_ddp_process_{local_rank}"
    if suffix[1:] != [expected_job_name, "hydra.output_subdir=null"]:
        raise RuntimeError("pilot DDP child has unexpected Hydra argv overrides")
    if not suffix[0].startswith("hydra.run.dir="):
        raise RuntimeError("pilot DDP child lacks Lightning's Hydra run directory")
    base_run_overrides = [
        value for value in base_argv if value.startswith("hydra.run.dir=")
    ]
    if len(base_run_overrides) != 1:
        raise RuntimeError("pilot base argv must contain one Hydra run directory")
    base_run_dir = base_run_overrides[0].split("=", 1)[1]
    child_run_dir = suffix[0].split("=", 1)[1].strip('"')
    if child_run_dir != base_run_dir:
        raise RuntimeError("pilot DDP child changed the Hydra run directory")
    return base_argv


def _pilot_environment_contract():
    present = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("GENMOL_TRAIN_")
    }
    if not present:
        return None
    if set(present) != _PILOT_ENVIRONMENT_KEYS:
        raise RuntimeError(
            "partial or unexpected GENMOL_TRAIN_ pilot environment contract"
        )
    for key in (
        "GENMOL_TRAIN_EXPECTED_CONFIG_SHA256",
        "GENMOL_TRAIN_EXPECTED_ARGV_SHA256",
        "GENMOL_TRAIN_EXPECTED_LAUNCH_MANIFEST_SHA256",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", present[key]):
            raise RuntimeError(f"{key} must be 64 lowercase hexadecimal digits")
    if not re.fullmatch(
        r"[0-9a-f]{40}", present["GENMOL_TRAIN_EXPECTED_SOURCE_REVISION"]
    ):
        raise RuntimeError(
            "GENMOL_TRAIN_EXPECTED_SOURCE_REVISION must be a full commit hash"
        )
    runtime_path = _in_repository_artifact_path(
        present["GENMOL_TRAIN_RUNTIME_CONFIG_PATH"],
        suffix=".json",
        label="runtime config path",
    )
    summary_path = _in_repository_artifact_path(
        present["GENMOL_TRAIN_SUMMARY_PATH"],
        suffix=".json",
        label="training summary path",
    )
    final_checkpoint_path = _in_repository_artifact_path(
        present["GENMOL_TRAIN_EXPECTED_FINAL_CHECKPOINT_PATH"],
        suffix=".ckpt",
        label="final checkpoint path",
    )
    launch_manifest_path = _in_repository_artifact_path(
        present["GENMOL_TRAIN_LAUNCH_MANIFEST_PATH"],
        suffix=".json",
        label="launch manifest path",
    )
    if launch_manifest_path != summary_path.with_name("launch_manifest.json"):
        raise RuntimeError(
            "pilot launch manifest must be launch_manifest.json beside the summary"
        )
    if (
        len({runtime_path, summary_path, final_checkpoint_path, launch_manifest_path})
        != 4
    ):
        raise RuntimeError("pilot completion artifact paths must be distinct")
    integer_fields = {
        "summary_schema_version": (
            "GENMOL_TRAIN_EXPECTED_SUMMARY_SCHEMA_VERSION",
            _TRAINING_SUMMARY_SCHEMA_VERSION,
        ),
        "expected_max_steps": ("GENMOL_TRAIN_EXPECTED_MAX_STEPS", None),
        "expected_world_size": ("GENMOL_TRAIN_EXPECTED_WORLD_SIZE", None),
    }
    parsed_integers = {}
    for output_name, (environment_name, exact_value) in integer_fields.items():
        raw_value = present[environment_name]
        if not raw_value.isdigit() or str(int(raw_value)) != raw_value:
            raise RuntimeError(f"{environment_name} must be a canonical integer")
        value = int(raw_value)
        if value <= 0 or (exact_value is not None and value != exact_value):
            raise RuntimeError(f"{environment_name} has an invalid value")
        parsed_integers[output_name] = value
    if parsed_integers["expected_world_size"] not in (1, 2):
        raise RuntimeError("GENMOL_TRAIN_EXPECTED_WORLD_SIZE must be 1 or 2")
    selected_gpu_uuids = _parse_selected_gpu_uuids(
        present["GENMOL_TRAIN_SELECTED_GPU_UUIDS_JSON"]
    )
    if len(selected_gpu_uuids) != parsed_integers["expected_world_size"]:
        raise RuntimeError(
            "selected GPU UUID count disagrees with the expected world size"
        )
    _validate_selected_gpu_exposure(selected_gpu_uuids)
    python_environment = {
        key: value for key, value in os.environ.items() if key.startswith("PYTHON")
    }
    expected_python_environment = {
        **_CONTROLLED_PYTHON_ENVIRONMENT,
        "PYTHONHASHSEED": python_environment.get("PYTHONHASHSEED"),
    }
    if (
        not expected_python_environment["PYTHONHASHSEED"]
        or not expected_python_environment["PYTHONHASHSEED"].isdigit()
        or python_environment != expected_python_environment
    ):
        raise RuntimeError("pilot child has an unexpected Python environment")
    if (
        sys.flags.optimize != 0
        or sys.flags.no_user_site != 1
        or sys.flags.utf8_mode != 1
        or not sys.dont_write_bytecode
    ):
        raise RuntimeError("pilot child Python flags disagree with its environment")
    if (
        _canonical_json_sha256(_pilot_base_argv())
        != present["GENMOL_TRAIN_EXPECTED_ARGV_SHA256"]
    ):
        raise RuntimeError("pilot child argv disagrees with the launch manifest")
    launch_manifest_snapshot, launch_manifest = _validate_launch_manifest(
        launch_manifest_path,
        expected_sha256=present["GENMOL_TRAIN_EXPECTED_LAUNCH_MANIFEST_SHA256"],
        expected_selected_gpu_uuids=selected_gpu_uuids,
    )
    return {
        **present,
        **parsed_integers,
        "runtime_path": runtime_path,
        "summary_path": summary_path,
        "final_checkpoint_path": final_checkpoint_path,
        "launch_manifest_path": launch_manifest_path,
        "launch_manifest_snapshot": launch_manifest_snapshot,
        "launch_manifest": launch_manifest,
        "selected_gpu_uuids": selected_gpu_uuids,
    }


def _require_pilot_source_revision(expected_revision):
    status = subprocess.run(
        [
            "git",
            "-C",
            str(_REPOSITORY_ROOT),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "-z",
            "--",
            ".",
            ":(exclude)output",
            ":(exclude)output/**",
        ],
        check=True,
        capture_output=True,
    )
    if status.stdout:
        raise RuntimeError("pilot child source worktree is dirty outside output/")
    revisions = []
    for reference in ("HEAD", "@{upstream}"):
        result = subprocess.run(
            ["git", "-C", str(_REPOSITORY_ROOT), "rev-parse", reference],
            check=True,
            capture_output=True,
            text=True,
        )
        revisions.append(result.stdout.strip())
    if revisions != [expected_revision, expected_revision]:
        raise RuntimeError(
            "pilot child source revision disagrees with clean pushed launch revision"
        )
    return {"head": revisions[0], "upstream": revisions[1]}


def _stat_identity(value):
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode": int(value.st_mode),
        "link_count": int(value.st_nlink),
        "size_bytes": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _stable_file_snapshot(path, *, capture_bytes=False):
    """Hash one regular file while rejecting symlinks and path replacement."""

    path = Path(os.path.abspath(os.fspath(path)))
    before_path = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before_path.st_mode):
        raise RuntimeError(f"pilot artifact is not a regular file: {path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    payload = bytearray() if capture_bytes else None
    digest = hashlib.sha256()
    try:
        before_descriptor = os.fstat(descriptor)
        if not stat.S_ISREG(before_descriptor.st_mode) or _stat_identity(
            before_descriptor
        ) != _stat_identity(before_path):
            raise RuntimeError(f"pilot artifact changed before open: {path}")
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if payload is not None:
                payload.extend(chunk)
        after_descriptor = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = path.stat(follow_symlinks=False)
    identities = {
        tuple(_stat_identity(value).items())
        for value in (
            before_path,
            before_descriptor,
            after_descriptor,
            after_path,
        )
    }
    if len(identities) != 1:
        raise RuntimeError(f"pilot artifact changed while being hashed: {path}")
    snapshot = {
        "path": str(path),
        **_stat_identity(after_path),
        "sha256": digest.hexdigest(),
        "stable_regular_file_verified": True,
    }
    return snapshot, None if payload is None else bytes(payload)


def _validate_launch_manifest(path, *, expected_sha256, expected_selected_gpu_uuids):
    snapshot, payload = _stable_file_snapshot(path, capture_bytes=True)
    if snapshot["sha256"] != expected_sha256:
        raise RuntimeError(
            "pilot launch manifest raw SHA-256 disagrees with the launch contract"
        )
    manifest = _strict_json_loads(payload, label="launch manifest")
    if not isinstance(manifest, dict):
        raise RuntimeError("pilot launch manifest root must be a JSON object")
    if manifest.get("cuda_visible_device_uuids") != expected_selected_gpu_uuids:
        raise RuntimeError(
            "pilot launch manifest selected GPU UUIDs disagree with the launch contract"
        )
    requested_gpu_count = manifest.get("user_requested_gpu_count")
    if (
        isinstance(requested_gpu_count, bool)
        or not isinstance(requested_gpu_count, int)
        or requested_gpu_count != len(expected_selected_gpu_uuids)
    ):
        raise RuntimeError(
            "pilot launch manifest GPU count disagrees with its selected UUIDs"
        )
    return snapshot, manifest


def _launch_manifest_evidence():
    _validate_selected_gpu_exposure(_PILOT_CONTRACT["selected_gpu_uuids"])
    snapshot, _manifest = _validate_launch_manifest(
        _PILOT_CONTRACT["launch_manifest_path"],
        expected_sha256=_PILOT_CONTRACT["GENMOL_TRAIN_EXPECTED_LAUNCH_MANIFEST_SHA256"],
        expected_selected_gpu_uuids=_PILOT_CONTRACT["selected_gpu_uuids"],
    )
    if snapshot != _PILOT_CONTRACT["launch_manifest_snapshot"]:
        raise RuntimeError("pilot launch manifest changed after process startup")
    return {
        **snapshot,
        "selected_gpu_uuids": list(_PILOT_CONTRACT["selected_gpu_uuids"]),
    }


def _atomic_write_json_exclusive(path, value):
    """Publish a complete JSON certificate exactly once."""

    path = Path(os.path.abspath(os.fspath(path)))
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to replace pilot training summary: {path}")
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o644)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to replace pilot training summary: {path}"
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


_PILOT_CONTRACT = _pilot_environment_contract()
if _PILOT_CONTRACT is not None:
    _require_pilot_source_revision(
        _PILOT_CONTRACT["GENMOL_TRAIN_EXPECTED_SOURCE_REVISION"]
    )

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import hydra
import lightning as L
import omegaconf
import torch
from genmol.backbone import is_conditioning_parameter_name
from genmol.diffusion import ContinuousCategoricalDiffusion
from genmol.model import (
    GenMol,
    UDLM_CONDITIONING_CHECKPOINT_KEY,
    UDLM_PRIOR_CHECKPOINT_KEY,
    optimizer_scheduler_multiplier,
    optimizer_scheduler_spec,
)
from genmol.utils.checkpoint_io import verified_checkpoint_file
from genmol.utils.utils_data import get_dataloader, get_last_checkpoint

omegaconf.OmegaConf.register_new_resolver("cwd", os.getcwd)
omegaconf.OmegaConf.register_new_resolver("device_count", torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver("eval", eval)
omegaconf.OmegaConf.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)


class _PilotFiniteLossCallback(L.Callback):
    """Fail fast on unhealthy loss/gradients and retain a rank-local receipt."""

    def __init__(self):
        super().__init__()
        self.loss_checks = 0
        self.optimizer_step_checks = 0
        self.gradient_tensor_observations = 0
        self.gradient_element_observations = 0

    def on_before_backward(self, trainer, pl_module, loss):
        del trainer, pl_module
        if not isinstance(loss, torch.Tensor) or not bool(
            torch.isfinite(loss.detach()).all().item()
        ):
            raise FloatingPointError(
                "pilot training loss is non-finite before backward"
            )
        self.loss_checks += 1

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        del trainer, optimizer
        gradient_tensors = 0
        gradient_elements = 0
        has_nonzero_gradient = False
        for name, parameter in pl_module.named_parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            gradient = gradient.detach()
            if not (torch.is_floating_point(gradient) or torch.is_complex(gradient)):
                raise RuntimeError(f"pilot gradient is not floating point: {name}")
            if not bool(torch.isfinite(gradient).all().item()):
                raise FloatingPointError(f"pilot gradient is non-finite: {name}")
            gradient_tensors += 1
            gradient_elements += gradient.numel()
            has_nonzero_gradient = has_nonzero_gradient or bool(
                torch.count_nonzero(gradient).item()
            )
        if gradient_tensors == 0:
            raise RuntimeError("pilot optimizer step has no gradients")
        if not has_nonzero_gradient:
            raise RuntimeError("pilot optimizer step has only zero gradients")
        self.optimizer_step_checks += 1
        self.gradient_tensor_observations += gradient_tensors
        self.gradient_element_observations += gradient_elements

    def completion_report(self, *, expected_optimizer_steps):
        if self.loss_checks < expected_optimizer_steps:
            raise RuntimeError(
                "pilot observed fewer finite losses than optimizer steps: "
                f"{self.loss_checks} < {expected_optimizer_steps}"
            )
        if self.optimizer_step_checks != expected_optimizer_steps:
            raise RuntimeError(
                "pilot gradient-check count disagrees with completed steps: "
                f"{self.optimizer_step_checks} != {expected_optimizer_steps}"
            )
        return {
            "scope": (
                "global-rank-zero callback counters; identical fail-fast checks "
                "execute independently on every rank"
            ),
            "all_losses_finite": True,
            "all_observed_gradients_finite": True,
            "every_optimizer_step_had_a_nonzero_gradient": True,
            "loss_checks": self.loss_checks,
            "optimizer_step_checks": self.optimizer_step_checks,
            "gradient_tensor_observations": self.gradient_tensor_observations,
            "gradient_element_observations": self.gradient_element_observations,
        }


_FILM_GRADIENT_AUDIT_SCHEMA_VERSION = 1
_FILM_GRADIENT_AUDIT_OBSERVATION_POINT = (
    "on_before_optimizer_step_global_rank_zero_after_gradient_accumulation"
)
_FILM_GRADIENT_AUDIT_CHECKS = (1, 2, 3)
_FILM_GRADIENT_FIRST_POSITIVE_LR_STEP = 2
_TIMESTEP_MLP_REQUIRED_OPTIMIZER_CHECK = 3
_SCREEN_INITIALIZATION_STATE_AUDIT_SCHEMA_VERSION = 1
_SCREEN_INITIALIZATION_STATE_AUDIT_PHASE = "after_verified_mdlm_ema_warm_start_before_training_rng_reseed_and_optimizer_creation"
_BACKBONE_STATE_HASH_DOMAIN = b"genmol-backbone-state-v1\0"


def _hash_framed_bytes(digest, value):
    """Hash bytes with an unambiguous unsigned-64-bit length prefix."""

    if not isinstance(value, bytes):
        raise TypeError("framed state-hash values must be bytes")
    digest.update(struct.pack(">Q", len(value)))
    digest.update(value)


def _backbone_state_identity(named_tensors):
    """Return a deterministic, exact identity for a named tensor state."""

    items = sorted(named_tensors, key=lambda item: item[0])
    if not items:
        raise RuntimeError("backbone state identity cannot hash an empty state")
    names = [name for name, _tensor in items]
    if any(not isinstance(name, str) or not name for name in names):
        raise RuntimeError("backbone state identity contains an invalid tensor name")
    if len(set(names)) != len(names):
        raise RuntimeError("backbone state identity contains duplicate tensor names")

    digest = hashlib.sha256()
    digest.update(_BACKBONE_STATE_HASH_DOMAIN)
    digest.update(struct.pack(">Q", len(items)))
    for name, tensor in items:
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"backbone state value is not a tensor: {name}")
        if tensor.device.type == "meta" or tensor.is_sparse or tensor.is_quantized:
            raise RuntimeError(
                f"backbone state tensor has an unsupported representation: {name}"
            )
        detached = tensor.detach().cpu().contiguous()
        raw_bytes = detached.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        _hash_framed_bytes(digest, name.encode("utf-8"))
        _hash_framed_bytes(digest, str(detached.dtype).encode("ascii"))
        digest.update(struct.pack(">Q", detached.ndim))
        for dimension in detached.shape:
            digest.update(struct.pack(">Q", dimension))
        _hash_framed_bytes(digest, raw_bytes)
    return {"tensor_count": len(items), "state_sha256": digest.hexdigest()}


def _screen_initialization_state_audit(config, model, startup_mode, warm_start_report):
    """Bind a screen arm to its exact post-warm-start backbone state."""

    if _PILOT_CONTRACT is None:
        return None
    manifest = _PILOT_CONTRACT.get("launch_manifest")
    screen = (
        manifest.get("optimization_screen") if isinstance(manifest, Mapping) else None
    )
    if screen is None:
        return None
    if not isinstance(screen, Mapping):
        raise RuntimeError("optimization-screen launch contract must be an object")
    if startup_mode != "warm_start":
        raise RuntimeError(
            "optimization-screen state audit requires a verified MDLM warm start"
        )
    retained_warm_start = _verified_warm_start_report(
        config, startup_mode, warm_start_report
    )
    if retained_warm_start.get("weights") != "ema":
        raise RuntimeError("optimization screen must warm-start from MDLM EMA weights")
    backbone = getattr(model, "backbone", None)
    state_dict = None if backbone is None else getattr(backbone, "state_dict", None)
    if not callable(state_dict):
        raise RuntimeError("optimization-screen model has no backbone state dictionary")
    full_state = list(state_dict().items())
    common_state = [
        (name, tensor)
        for name, tensor in full_state
        if not is_conditioning_parameter_name(name)
    ]
    full_identity = _backbone_state_identity(full_state)
    common_identity = _backbone_state_identity(common_state)
    conditioning_variant = _configured_conditioning_variant(config)
    if conditioning_variant not in {"additive", "film_adaln"}:
        raise RuntimeError(
            "optimization-screen conditioning variant must be additive or film_adaln"
        )
    return {
        "schema_version": _SCREEN_INITIALIZATION_STATE_AUDIT_SCHEMA_VERSION,
        "phase": _SCREEN_INITIALIZATION_STATE_AUDIT_PHASE,
        "source_checkpoint_sha256": retained_warm_start["source_sha256"],
        "resolved_training_config_sha256": _PILOT_CONTRACT[
            "GENMOL_TRAIN_EXPECTED_CONFIG_SHA256"
        ],
        "training_seed": _exact_training_seed(
            config.get("seed", 1), "optimization-screen training seed"
        ),
        "conditioning_variant": conditioning_variant,
        "common_backbone_tensor_count": common_identity["tensor_count"],
        "common_backbone_state_sha256": common_identity["state_sha256"],
        "full_initial_tensor_count": full_identity["tensor_count"],
        "full_initial_state_sha256": full_identity["state_sha256"],
    }


def _validate_film_gradient_audit_contract(value, expected_sha256):
    """Validate the immutable A1 gradient topology registered before launch."""

    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ):
        raise RuntimeError(
            "conditioning gradient contract SHA-256 must be lowercase hexadecimal"
        )
    if not isinstance(value, Mapping):
        raise RuntimeError("conditioning gradient contract must be an object")
    contract = dict(value)
    expected_keys = {
        "schema_version",
        "observation_point",
        "optimizer_checks",
        "first_positive_lr_optimizer_step",
        "timestep_mlp_required_optimizer_check",
        "groups",
    }
    if set(contract) != expected_keys:
        raise RuntimeError("conditioning gradient contract has invalid keys")
    if (
        type(contract["schema_version"]) is not int
        or contract["schema_version"] != _FILM_GRADIENT_AUDIT_SCHEMA_VERSION
    ):
        raise RuntimeError("conditioning gradient contract has invalid schema")
    if contract["observation_point"] != _FILM_GRADIENT_AUDIT_OBSERVATION_POINT:
        raise RuntimeError(
            "conditioning gradient contract has invalid observation point"
        )
    if contract["optimizer_checks"] != list(_FILM_GRADIENT_AUDIT_CHECKS):
        raise RuntimeError("conditioning gradient contract has invalid check schedule")
    if (
        type(contract["first_positive_lr_optimizer_step"]) is not int
        or contract["first_positive_lr_optimizer_step"]
        != _FILM_GRADIENT_FIRST_POSITIVE_LR_STEP
    ):
        raise RuntimeError("conditioning gradient contract has invalid LR transition")
    if (
        type(contract["timestep_mlp_required_optimizer_check"]) is not int
        or contract["timestep_mlp_required_optimizer_check"]
        != _TIMESTEP_MLP_REQUIRED_OPTIMIZER_CHECK
    ):
        raise RuntimeError(
            "conditioning gradient contract has invalid timestep activation check"
        )

    groups = contract["groups"]
    if not isinstance(groups, list) or len(groups) != 2:
        raise RuntimeError("conditioning gradient contract must contain two groups")
    expected_group_identity = (
        ("film_modulation", "film"),
        ("timestep_mlp", "timestep_mlp"),
    )
    globally_seen_names = set()
    normalized_groups = []
    for group, (expected_group_id, expected_kind) in zip(
        groups, expected_group_identity, strict=True
    ):
        if not isinstance(group, Mapping) or set(group) != {
            "group_id",
            "kind",
            "parameters",
        }:
            raise RuntimeError("conditioning gradient contract group has invalid keys")
        if group["group_id"] != expected_group_id or group["kind"] != expected_kind:
            raise RuntimeError("conditioning gradient contract group order is invalid")
        parameters = group["parameters"]
        if not isinstance(parameters, list) or not parameters:
            raise RuntimeError("conditioning gradient contract group has no parameters")
        normalized_parameters = []
        for parameter in parameters:
            if not isinstance(parameter, Mapping) or set(parameter) != {
                "name",
                "shape",
            }:
                raise RuntimeError(
                    "conditioning gradient parameter manifest has invalid keys"
                )
            name = parameter["name"]
            shape = parameter["shape"]
            if (
                not isinstance(name, str)
                or not name
                or name in globally_seen_names
                or not isinstance(shape, list)
                or not shape
                or any(
                    type(dimension) is not int or dimension <= 0 for dimension in shape
                )
            ):
                raise RuntimeError(
                    "conditioning gradient parameter manifest is invalid"
                )
            globally_seen_names.add(name)
            normalized_parameters.append({"name": name, "shape": list(shape)})
        normalized_groups.append(
            {
                "group_id": expected_group_id,
                "kind": expected_kind,
                "parameters": normalized_parameters,
            }
        )
    contract["groups"] = normalized_groups
    if _canonical_json_sha256(contract) != expected_sha256:
        raise RuntimeError(
            "conditioning gradient contract disagrees with its registered SHA-256"
        )
    return contract


def _registered_film_gradient_audit_contract():
    if _PILOT_CONTRACT is None:
        raise RuntimeError("conditioning gradient audit requires a launch contract")
    manifest = _PILOT_CONTRACT.get("launch_manifest")
    screen = (
        manifest.get("optimization_screen") if isinstance(manifest, Mapping) else None
    )
    if not isinstance(screen, Mapping):
        raise RuntimeError(
            "FiLM pilot launch manifest lacks its optimization-screen contract"
        )
    return _validate_film_gradient_audit_contract(
        screen.get("conditioning_gradient_contract"),
        screen.get("conditioning_gradient_contract_sha256"),
    )


def _runtime_conditioning_parameter_groups(pl_module):
    named_parameters = getattr(pl_module, "named_parameters", None)
    if not callable(named_parameters):
        raise RuntimeError("FiLM gradient audit model has no named parameters")
    film_parameters = []
    timestep_parameters = []
    for name, parameter in named_parameters():
        if ".film_modulation." in name:
            film_parameters.append((name, parameter))
        elif name.removeprefix("backbone.").startswith("time_conditioner."):
            timestep_parameters.append((name, parameter))
    if not film_parameters or not timestep_parameters:
        raise RuntimeError("FiLM gradient audit found an empty conditioning group")
    return {
        "film_modulation": film_parameters,
        "timestep_mlp": timestep_parameters,
    }


class _FilmGradientActivationCallback(L.Callback):
    """Attest the staged gradient path created by zero-initialized FiLM layers."""

    def __init__(self, registered_contract):
        super().__init__()
        self.registered_contract = registered_contract
        self.registered_contract_sha256 = _canonical_json_sha256(registered_contract)
        self.optimizer_checks = []

    def _group_report(self, *, group_contract, runtime_parameters):
        runtime_manifest = [
            {"name": name, "shape": list(parameter.shape)}
            for name, parameter in runtime_parameters
        ]
        if runtime_manifest != group_contract["parameters"]:
            raise RuntimeError(
                f"runtime {group_contract['group_id']} parameter manifest changed"
            )
        gradient_element_count = 0
        all_parameter_gradients_nonzero = True
        for name, parameter in runtime_parameters:
            gradient = parameter.grad
            if gradient is None:
                raise RuntimeError(f"conditioning gradient is missing: {name}")
            gradient = gradient.detach()
            if not (torch.is_floating_point(gradient) or torch.is_complex(gradient)):
                raise RuntimeError(
                    f"conditioning gradient is not floating point: {name}"
                )
            if not bool(torch.isfinite(gradient).all().item()):
                raise FloatingPointError(f"conditioning gradient is non-finite: {name}")
            gradient_element_count += gradient.numel()
            all_parameter_gradients_nonzero = all_parameter_gradients_nonzero and bool(
                torch.count_nonzero(gradient).item()
            )
        return {
            "group_id": group_contract["group_id"],
            "ordered_parameter_manifest_sha256": _canonical_json_sha256(
                runtime_manifest
            ),
            "parameter_count": len(runtime_parameters),
            "gradient_element_count": gradient_element_count,
            "all_gradients_present": True,
            "all_gradients_finite": True,
            "all_parameter_gradients_nonzero": all_parameter_gradients_nonzero,
        }

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        del trainer
        check_index = len(self.optimizer_checks) + 1
        if check_index > _FILM_GRADIENT_AUDIT_CHECKS[-1]:
            return
        parameter_groups = getattr(optimizer, "param_groups", None)
        if not isinstance(parameter_groups, list) or len(parameter_groups) != 1:
            raise RuntimeError(
                "FiLM gradient audit requires one optimizer parameter group"
            )
        learning_rate = parameter_groups[0].get("lr")
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or not math.isfinite(float(learning_rate))
            or learning_rate < 0
        ):
            raise RuntimeError("FiLM gradient audit observed an invalid learning rate")
        learning_rate = float(learning_rate)
        if check_index == 1 and learning_rate != 0.0:
            raise RuntimeError(
                "FiLM gradient audit expected zero LR at optimizer step 1"
            )
        if check_index in (2, 3) and learning_rate <= 0.0:
            raise RuntimeError(
                "FiLM gradient audit expected positive LR at optimizer steps 2 and 3"
            )

        runtime_groups = _runtime_conditioning_parameter_groups(pl_module)
        contract_groups = {
            group["group_id"]: group for group in self.registered_contract["groups"]
        }
        film_report = self._group_report(
            group_contract=contract_groups["film_modulation"],
            runtime_parameters=runtime_groups["film_modulation"],
        )
        timestep_report = self._group_report(
            group_contract=contract_groups["timestep_mlp"],
            runtime_parameters=runtime_groups["timestep_mlp"],
        )
        if check_index == 1 and not film_report["all_parameter_gradients_nonzero"]:
            raise RuntimeError(
                "every FiLM parameter must receive a nonzero gradient at the first "
                "optimizer observation"
            )
        if check_index == 3 and not timestep_report["all_parameter_gradients_nonzero"]:
            raise RuntimeError(
                "every timestep-MLP parameter must receive a nonzero gradient at the "
                "third optimizer observation"
            )
        self.optimizer_checks.append(
            {
                "optimizer_gradient_observation_index": check_index,
                "optimizer_step_index": check_index,
                "learning_rate_before_step": learning_rate,
                "film_groups": [film_report],
                "timestep_mlp_groups": [timestep_report],
            }
        )

    def completion_report(self):
        observed_checks = [
            check["optimizer_gradient_observation_index"]
            for check in self.optimizer_checks
        ]
        if observed_checks != list(_FILM_GRADIENT_AUDIT_CHECKS):
            raise RuntimeError(
                "FiLM gradient audit did not observe its first three optimizer checks"
            )
        return {
            "schema_version": _FILM_GRADIENT_AUDIT_SCHEMA_VERSION,
            "status": "completed",
            "observation_point": _FILM_GRADIENT_AUDIT_OBSERVATION_POINT,
            "registered_contract_sha256": self.registered_contract_sha256,
            "first_positive_lr_optimizer_step": (_FILM_GRADIENT_FIRST_POSITIVE_LR_STEP),
            "timestep_mlp_required_optimizer_check": (
                _TIMESTEP_MLP_REQUIRED_OPTIMIZER_CHECK
            ),
            "optimizer_checks": list(self.optimizer_checks),
        }


def _configured_conditioning_variant(config):
    training = config.get("training", {})
    udlm = training.get("udlm", {}) if isinstance(training, Mapping) else {}
    variant = udlm.get("conditioning_variant", "additive")
    if not isinstance(variant, str):
        raise RuntimeError("training.udlm.conditioning_variant must be a string")
    return variant


def _pilot_callbacks(config):
    if _PILOT_CONTRACT is None:
        return []
    if config.training.get("pilot_fail_on_nonfinite_loss") is not True:
        raise RuntimeError(
            "pilot config must enable fail-fast non-finite loss validation"
        )
    if config.trainer.get("detect_anomaly") is not True:
        raise RuntimeError("pilot config must enable backward anomaly detection")
    callbacks = [_PilotFiniteLossCallback()]
    if _configured_conditioning_variant(config) == "film_adaln":
        if config.training.get("reseed_after_model_initialization") is not True:
            raise RuntimeError(
                "FiLM optimization screen requires post-initialization reseeding"
            )
        callbacks.append(
            _FilmGradientActivationCallback(_registered_film_gradient_audit_contract())
        )
    return callbacks


def _exact_positive_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"{label} must be a positive integer")
    return value


def _exact_nonnegative_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{label} must be a nonnegative integer")
    return value


def _exact_training_seed(value, label):
    seed = _exact_nonnegative_integer(value, label)
    if seed > _MAX_TRAINING_SEED:
        raise RuntimeError(
            f"{label} must be at most {_MAX_TRAINING_SEED} for NumPy/Lightning"
        )
    return seed


def _pilot_trainable_parameter_counts(model):
    """Count the exact trainable split represented by the live pilot model."""

    backbone = getattr(model, "backbone", None)
    if backbone is None or not callable(getattr(backbone, "named_parameters", None)):
        raise RuntimeError("pilot model has no countable backbone parameters")
    if not callable(getattr(model, "named_parameters", None)):
        raise RuntimeError("pilot model has no countable trainable parameters")

    model_parameters = {
        id(parameter): (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    backbone_parameters = {
        id(parameter): (name, parameter)
        for name, parameter in backbone.named_parameters()
        if parameter.requires_grad
    }
    if not model_parameters:
        raise RuntimeError("pilot model has no trainable parameters")
    if set(model_parameters) != set(backbone_parameters):
        raise RuntimeError(
            "pilot trainable parameters are not exactly the backbone parameters"
        )

    base_backbone = 0
    time_conditioner = 0
    film_modulation = 0
    for name, parameter in backbone_parameters.values():
        count = parameter.numel()
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise RuntimeError(f"pilot trainable parameter has invalid size: {name}")
        if name.startswith("time_conditioner."):
            time_conditioner += count
        elif is_conditioning_parameter_name(name):
            film_modulation += count
        else:
            base_backbone += count
    if base_backbone <= 0:
        raise RuntimeError("pilot backbone has no base trainable parameters")
    if time_conditioner <= 0:
        raise RuntimeError("pilot backbone has no trainable time_conditioner")
    total = sum(parameter.numel() for _name, parameter in model_parameters.values())
    if total != base_backbone + time_conditioner + film_modulation:
        raise RuntimeError("pilot trainable parameter counts do not add up")
    result = {
        "base_backbone": base_backbone,
        "time_conditioner": time_conditioner,
    }
    if film_modulation:
        result["film_modulation"] = film_modulation
    result["total"] = total
    return result


def _pilot_training_accounting(config, trainer, model, train_dataloader):
    """Derive the requested-example and parameter receipt at completion."""

    if _PILOT_CONTRACT is None:
        return None
    training_seed = _exact_training_seed(config.get("seed", 1), "pilot training seed")
    optimizer_updates = _exact_positive_integer(
        getattr(trainer, "global_step", None), "pilot optimizer updates"
    )
    configured_updates = _exact_positive_integer(
        config.trainer.get("max_steps"), "pilot configured optimizer updates"
    )
    runtime_max_steps = _exact_positive_integer(
        getattr(trainer, "max_steps", None), "pilot runtime max steps"
    )
    expected_updates = _PILOT_CONTRACT["expected_max_steps"]
    if not (
        optimizer_updates == configured_updates == runtime_max_steps == expected_updates
    ):
        raise RuntimeError(
            "pilot optimizer updates disagree across completion, runtime, config, "
            "and launch contract"
        )

    world_size = _exact_positive_integer(
        getattr(trainer, "world_size", None), "pilot accounting world size"
    )
    configured_devices = _exact_positive_integer(
        config.trainer.get("devices"), "pilot configured devices"
    )
    configured_nodes = _exact_positive_integer(
        config.trainer.get("num_nodes"), "pilot configured nodes"
    )
    runtime_nodes = _exact_positive_integer(
        getattr(trainer, "num_nodes", None), "pilot runtime nodes"
    )
    expected_world_size = _PILOT_CONTRACT["expected_world_size"]
    if (
        configured_nodes != 1
        or runtime_nodes != configured_nodes
        or configured_devices * configured_nodes != world_size
        or world_size != expected_world_size
    ):
        raise RuntimeError(
            "pilot accounting world size disagrees across runtime, config, and "
            "launch contract"
        )

    if train_dataloader is None:
        train_dataloader = getattr(trainer, "train_dataloader", None)
    micro_batch_size = _exact_positive_integer(
        config.loader.get("batch_size"), "pilot micro-batch size per rank"
    )
    runtime_micro_batch_size = _exact_positive_integer(
        getattr(train_dataloader, "batch_size", None),
        "pilot runtime micro-batch size per rank",
    )
    if runtime_micro_batch_size != micro_batch_size:
        raise RuntimeError(
            "pilot runtime micro-batch size disagrees with the resolved config"
        )
    accumulation = _exact_positive_integer(
        config.trainer.get("accumulate_grad_batches"),
        "pilot configured gradient accumulation",
    )
    runtime_accumulation = _exact_positive_integer(
        getattr(trainer, "accumulate_grad_batches", None),
        "pilot runtime gradient accumulation",
    )
    if runtime_accumulation != accumulation:
        raise RuntimeError(
            "pilot runtime gradient accumulation disagrees with the resolved config"
        )
    effective_global_examples = micro_batch_size * world_size * accumulation
    configured_global_batch = _exact_positive_integer(
        config.loader.get("global_batch_size"), "pilot configured global batch size"
    )
    if configured_global_batch != effective_global_examples:
        raise RuntimeError(
            "pilot configured global batch size does not equal micro-batch per rank "
            "times world size times accumulation"
        )
    if config.get("data") != "safe":
        raise RuntimeError("pilot accounting requires the hosted SAFE training stream")

    return {
        "training_seed": training_seed,
        "optimizer_updates": optimizer_updates,
        "world_size": world_size,
        "micro_batch_size_per_rank": micro_batch_size,
        "accumulate_grad_batches": accumulation,
        "effective_global_examples_per_optimizer_step": effective_global_examples,
        "total_requested_example_exposures": (
            effective_global_examples * optimizer_updates
        ),
        "hosted_stream_rank_partition_policy": (_HOSTED_STREAM_RANK_PARTITION_POLICY),
        "trainable_parameter_counts": _pilot_trainable_parameter_counts(model),
    }


def _pilot_streaming_partition(trainer):
    """Resolve a strict single-node rank identity before loading the stream."""

    if _PILOT_CONTRACT is None:
        return None
    expected_world_size = _PILOT_CONTRACT["expected_world_size"]
    actual_world_size = _exact_positive_integer(
        getattr(trainer, "world_size", None), "pilot runtime world size"
    )
    num_nodes = _exact_positive_integer(
        getattr(trainer, "num_nodes", None), "pilot trainer num_nodes"
    )
    global_rank = getattr(trainer, "global_rank", None)
    if (
        isinstance(global_rank, bool)
        or not isinstance(global_rank, int)
        or not 0 <= global_rank < actual_world_size
    ):
        raise RuntimeError("pilot trainer has an invalid global rank")
    if num_nodes != 1:
        raise RuntimeError("pilot streaming partition requires exactly one node")
    if actual_world_size != expected_world_size:
        raise RuntimeError(
            "pilot streaming world size disagrees with the launch contract"
        )

    distributed_environment = {
        key: os.environ.get(key) for key in ("LOCAL_RANK", "WORLD_SIZE", "NODE_RANK")
    }
    present_values = {
        key: value
        for key, value in distributed_environment.items()
        if value is not None
    }
    if present_values:
        expected_environment = {
            "LOCAL_RANK": str(global_rank),
            "WORLD_SIZE": str(actual_world_size),
            "NODE_RANK": "0",
        }
        if distributed_environment != expected_environment:
            raise RuntimeError(
                "pilot distributed environment is partial or inconsistent: "
                f"{distributed_environment!r}"
            )
    elif global_rank != 0:
        raise RuntimeError("pilot nonzero rank lacks Lightning's DDP environment")
    return global_rank, actual_world_size


def _training_strategy():
    """Use a self-spawning environment only for the reviewed local pilot."""

    cluster_environment = None
    if _PILOT_CONTRACT is not None:
        cluster_environment = L.fabric.plugins.environments.LightningEnvironment()
    return L.pytorch.strategies.DDPStrategy(
        find_unused_parameters=False,
        cluster_environment=cluster_environment,
    )


def _validate_pilot_completion_config(config):
    if _PILOT_CONTRACT is None:
        return None
    expected_steps = _PILOT_CONTRACT["expected_max_steps"]
    expected_world_size = _PILOT_CONTRACT["expected_world_size"]
    configured_steps = _exact_positive_integer(
        config.trainer.get("max_steps"), "pilot trainer.max_steps"
    )
    devices = _exact_positive_integer(
        config.trainer.get("devices"), "pilot trainer.devices"
    )
    nodes = _exact_positive_integer(
        config.trainer.get("num_nodes"), "pilot trainer.num_nodes"
    )
    if configured_steps != expected_steps:
        raise RuntimeError("pilot max step disagrees with the launch contract")
    if devices * nodes != expected_world_size:
        raise RuntimeError("pilot world size disagrees with the launch contract")
    callback_dir = Path(os.path.abspath(os.fspath(config.callback.get("dirpath"))))
    if (
        callback_dir / f"{expected_steps}.ckpt"
        != _PILOT_CONTRACT["final_checkpoint_path"]
    ):
        raise RuntimeError("pilot final checkpoint path disagrees with its config")
    if (
        config.callback.get("filename") != "{step}"
        or config.callback.get("every_n_train_steps") != expected_steps
        or config.callback.get("save_top_k") != -1
    ):
        raise RuntimeError(
            "pilot checkpoint callback does not guarantee the final step"
        )
    return {
        "summary_schema_version": _PILOT_CONTRACT["summary_schema_version"],
        "summary_path": str(_PILOT_CONTRACT["summary_path"]),
        "final_checkpoint_path": str(_PILOT_CONTRACT["final_checkpoint_path"]),
        "expected_max_steps": expected_steps,
        "expected_world_size": expected_world_size,
        "fail_on_nonfinite_loss": True,
        "backward_anomaly_detection": True,
    }


def checkpoint_startup_mode(resume_checkpoint, initialization_checkpoint):
    """Choose exactly one of Lightning resume, one-time warm-start, or scratch."""
    if resume_checkpoint is not None:
        return "resume"
    if initialization_checkpoint:
        return "warm_start"
    return "scratch"


def _reseed_training_rng_after_model_initialization(config, startup_mode):
    """Make architecture-screen training randomness independent of init draws."""

    enabled = config.training.get("reseed_after_model_initialization", False)
    if type(enabled) is not bool:
        raise RuntimeError(
            "training.reseed_after_model_initialization must be a boolean"
        )
    if not enabled:
        return None
    if _PILOT_CONTRACT is None:
        raise RuntimeError(
            "post-initialization reseeding is restricted to a launch-bound pilot"
        )
    if startup_mode != "warm_start":
        raise RuntimeError(
            "post-initialization reseeding requires the common verified MDLM "
            "warm-start checkpoint"
        )
    seed = _exact_training_seed(
        config.get("seed", 1), "post-initialization training seed"
    )
    applied_seed = L.seed_everything(seed, workers=True)
    if type(applied_seed) is not int or applied_seed != seed:
        raise RuntimeError(
            "Lightning did not apply the exact post-initialization training seed"
        )
    return {
        "policy": "reseed_all_training_rng_streams_after_model_and_warm_start",
        "seed": seed,
        "purpose": "isolate_training_randomness_from_architecture_constructor_draws",
        "applied_before_dataloader_and_trainer_construction": True,
    }


def _validate_and_record_pilot_config(config):
    if _PILOT_CONTRACT is None:
        return None
    completion_contract = _validate_pilot_completion_config(config)
    _pilot_callbacks(config)
    _exact_training_seed(config.get("seed", 1), "pilot training seed")
    expected_revision = _PILOT_CONTRACT["GENMOL_TRAIN_EXPECTED_SOURCE_REVISION"]
    source = _require_pilot_source_revision(expected_revision)
    resolved = omegaconf.OmegaConf.to_container(
        config,
        resolve=True,
        enum_to_str=True,
    )
    digest = _canonical_json_sha256(resolved)
    if digest != _PILOT_CONTRACT["GENMOL_TRAIN_EXPECTED_CONFIG_SHA256"]:
        raise RuntimeError(
            "resolved Hydra config disagrees with the launch-pinned config digest"
        )
    if os.environ["PYTHONHASHSEED"] != str(config.get("seed", 1)):
        raise RuntimeError("PYTHONHASHSEED disagrees with the resolved training seed")
    launch_manifest_evidence = _launch_manifest_evidence()
    record = {
        "schema_version": _RUNTIME_CONFIG_SCHEMA_VERSION,
        "status": "preflight_completed",
        "source_revision": expected_revision,
        "source": source,
        "training_argv": _pilot_base_argv(),
        "observed_training_argv": list(sys.argv),
        "training_argv_sha256": _PILOT_CONTRACT["GENMOL_TRAIN_EXPECTED_ARGV_SHA256"],
        "resolved_training_config": resolved,
        "resolved_training_config_sha256": digest,
        "launch_manifest": launch_manifest_evidence,
        "completion_contract": completion_contract,
        "python_environment": {
            key: value for key, value in os.environ.items() if key.startswith("PYTHON")
        },
    }
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank not in (None, "0"):
        return record
    runtime_path = _PILOT_CONTRACT["runtime_path"]
    encoded = json.dumps(record, indent=2, sort_keys=True) + "\n"
    try:
        with runtime_path.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if runtime_path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(
                "pilot runtime config record already exists with other data"
            )
    return record


def _floating_tensor_finiteness(named_tensors, *, label):
    floating_tensor_count = 0
    floating_element_count = 0
    for name, tensor in named_tensors:
        if not isinstance(tensor, torch.Tensor):
            continue
        if not (torch.is_floating_point(tensor) or torch.is_complex(tensor)):
            continue
        floating_tensor_count += 1
        floating_element_count += tensor.numel()
        if not bool(torch.isfinite(tensor.detach()).all().item()):
            raise FloatingPointError(f"non-finite {label} tensor: {name}")
    if floating_tensor_count == 0:
        raise RuntimeError(f"pilot {label} contains no floating tensors")
    return {
        "all_finite": True,
        "floating_tensor_count": floating_tensor_count,
        "floating_element_count": floating_element_count,
    }


def _nested_path_leaves(value, *, path):
    """Yield nested leaves with structural path components and no interpolation."""

    if isinstance(value, omegaconf.DictConfig):
        for key, item in value.items_ex(resolve=False):
            yield from _nested_path_leaves(item, path=(*path, key))
        return
    if isinstance(value, omegaconf.ListConfig):
        for index, item in enumerate(value._iter_ex(resolve=False)):
            yield from _nested_path_leaves(item, path=(*path, index))
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _nested_path_leaves(item, path=(*path, key))
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _nested_path_leaves(item, path=(*path, index))
        return
    yield path, value


def _format_structural_path(path):
    rendered = str(path[0])
    for component in path[1:]:
        if isinstance(component, int):
            rendered += f"[{component}]"
        else:
            rendered += f"[{json.dumps(str(component), ensure_ascii=False)}]"
    return rendered


def _nested_named_tensors(value, *, prefix):
    """Yield every tensor with an unambiguous structural diagnostic path."""

    for path, item in _nested_path_leaves(value, path=(prefix,)):
        if isinstance(item, torch.Tensor):
            yield _format_structural_path(path), item


def _python_float_finiteness(value, *, prefix, label):
    floating_scalar_count = 0
    for path, item in _nested_path_leaves(value, path=(prefix,)):
        if isinstance(item, (np.ndarray, np.generic)):
            raise RuntimeError(
                f"unsupported NumPy numeric leaf in {label}: "
                f"{_format_structural_path(path)}"
            )
        if type(item) is not float:
            continue
        floating_scalar_count += 1
        if not math.isfinite(item):
            raise FloatingPointError(
                f"non-finite {label} Python float: {_format_structural_path(path)}"
            )
    return {
        "all_finite": True,
        "floating_scalar_count": floating_scalar_count,
    }


def _exact_nested_state_match(serialized, live, *, label, path=()):
    """Require exact recursive types, keys, shapes, dtypes, and values."""

    diagnostic = label if not path else f"{label} {_format_structural_path(path)}"
    if isinstance(serialized, torch.Tensor) or isinstance(live, torch.Tensor):
        if not isinstance(serialized, torch.Tensor) or not isinstance(
            live, torch.Tensor
        ):
            raise RuntimeError(f"{diagnostic} tensor type disagrees")
        if (
            serialized.device.type == "meta"
            or live.device.type == "meta"
            or serialized.shape != live.shape
            or serialized.dtype != live.dtype
            or not torch.equal(serialized.detach().cpu(), live.detach().cpu())
        ):
            raise RuntimeError(f"{diagnostic} tensor value disagrees")
        return
    if isinstance(serialized, Mapping) or isinstance(live, Mapping):
        if not isinstance(serialized, Mapping) or not isinstance(live, Mapping):
            raise RuntimeError(f"{diagnostic} mapping type disagrees")
        if set(serialized) != set(live):
            raise RuntimeError(f"{diagnostic} mapping keys disagree")
        for key in serialized:
            _exact_nested_state_match(
                serialized[key], live[key], label=label, path=(*path, key)
            )
        return
    if isinstance(serialized, (list, tuple)) or isinstance(live, (list, tuple)):
        if type(serialized) is not type(live) or len(serialized) != len(live):
            raise RuntimeError(f"{diagnostic} sequence structure disagrees")
        for index, (serialized_item, live_item) in enumerate(
            zip(serialized, live, strict=True)
        ):
            _exact_nested_state_match(
                serialized_item,
                live_item,
                label=label,
                path=(*path, index),
            )
        return
    if type(serialized) is not type(live) or serialized != live:
        raise RuntimeError(f"{diagnostic} scalar value disagrees")


def _validated_checkpoint_top_level_schema(checkpoint, *, model):
    """Require exactly the Lightning and conditional UDLM checkpoint records."""

    expected_keys = {
        "callbacks",
        "ema",
        "epoch",
        "global_step",
        "hparams_name",
        "hyper_parameters",
        "loops",
        "lr_schedulers",
        "optimizer_states",
        "pytorch-lightning_version",
        "sampler",
        "state_dict",
    }
    if type(getattr(model, "mdlm", None)) is ContinuousCategoricalDiffusion:
        expected_keys.add(UDLM_PRIOR_CHECKPOINT_KEY)
    if getattr(model, "udlm_conditioning_metadata", None) is not None:
        expected_keys.add(UDLM_CONDITIONING_CHECKPOINT_KEY)
    observed_keys = set(checkpoint)
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        unexpected = sorted(repr(key) for key in observed_keys - expected_keys)
        raise RuntimeError(
            "pilot checkpoint top-level keys disagree with the exact schema: "
            f"missing={missing}, unexpected={unexpected}"
        )


def _model_checkpoint_callback_state_key(expected_steps):
    return (
        "ModelCheckpoint{'monitor': None, 'mode': 'min', "
        f"'every_n_train_steps': {expected_steps}, 'every_n_epochs': 0, "
        "'train_time_interval': None}"
    )


def _validated_lightning_checkpoint_sentinel(
    checkpoint, *, checkpoint_path, expected_steps, trainer
):
    """Verify Lightning's sole legitimate non-finite bookkeeping tensor.

    Lightning 2.5.1 initializes ``ModelCheckpoint.kth_value`` to positive
    infinity for ``mode='min'``.  With the pilot's ``monitor=None`` and
    ``save_top_k=-1`` contract, that value remains the unranked-checkpoint
    sentinel when the final checkpoint is serialized.  It is framework state,
    not model, EMA, optimizer, or scheduler state, and is accepted only after
    the complete callback record is matched to the final checkpoint contract.
    """

    framework_version = checkpoint.get("pytorch-lightning_version")
    if framework_version != _PINNED_LIGHTNING_VERSION:
        raise RuntimeError(
            "pilot checkpoint Lightning version "
            f"{framework_version!r}; expected {_PINNED_LIGHTNING_VERSION!r}"
        )
    if L.__version__ != _PINNED_LIGHTNING_VERSION:
        raise RuntimeError(
            "pilot runtime Lightning version "
            f"{L.__version__!r}; expected {_PINNED_LIGHTNING_VERSION!r}"
        )
    callbacks = checkpoint.get("callbacks")
    if not isinstance(callbacks, Mapping):
        raise RuntimeError("pilot checkpoint has no callback-state mapping")
    callback_key = _model_checkpoint_callback_state_key(expected_steps)
    if set(callbacks) != {callback_key}:
        raise RuntimeError(
            "pilot checkpoint callback-state keys disagree with the exact "
            "ModelCheckpoint contract"
        )
    state = callbacks[callback_key]
    if not isinstance(state, Mapping):
        raise RuntimeError("pilot ModelCheckpoint state is not a mapping")
    expected_state_keys = {
        "monitor",
        "best_model_score",
        "best_model_path",
        "current_score",
        "dirpath",
        "best_k_models",
        "kth_best_model_path",
        "kth_value",
        "last_model_path",
    }
    if set(state) != expected_state_keys:
        raise RuntimeError("pilot ModelCheckpoint state keys are invalid")
    checkpoint_path = Path(os.path.abspath(os.fspath(checkpoint_path)))
    if (
        state.get("monitor") is not None
        or state.get("best_model_score") is not None
        or state.get("current_score") is not None
        or state.get("best_model_path") != str(checkpoint_path)
        or state.get("dirpath") != str(checkpoint_path.parent)
        or state.get("best_k_models") != {}
        or state.get("kth_best_model_path") != ""
        or state.get("last_model_path") != ""
    ):
        raise RuntimeError(
            "pilot ModelCheckpoint state disagrees with the unranked final-step "
            "checkpoint contract"
        )
    sentinel = state.get("kth_value")
    if (
        not isinstance(sentinel, torch.Tensor)
        or sentinel.dtype != torch.float32
        or sentinel.device.type != "cpu"
        or sentinel.shape != torch.Size([])
        or not bool(torch.isposinf(sentinel).item())
    ):
        raise FloatingPointError(
            "pilot ModelCheckpoint kth_value is not the exact scalar float32 "
            "positive-infinity sentinel"
        )
    live_callbacks = getattr(trainer, "callbacks", None)
    model_checkpoint_callbacks = (
        []
        if not isinstance(live_callbacks, (list, tuple))
        else [
            callback
            for callback in live_callbacks
            if isinstance(callback, L.pytorch.callbacks.ModelCheckpoint)
        ]
    )
    if len(model_checkpoint_callbacks) != 1:
        raise RuntimeError("pilot must have exactly one live ModelCheckpoint callback")
    live_callback = model_checkpoint_callbacks[0]
    if (
        live_callback.state_key != callback_key
        or live_callback.monitor is not None
        or live_callback.mode != "min"
        or live_callback.save_top_k != -1
        or live_callback.save_last is not None
        or live_callback.save_weights_only is not False
        or live_callback.filename != "{step}"
        or live_callback.auto_insert_metric_name is not False
        or live_callback._enable_version_counter is not False
        or live_callback._every_n_train_steps != expected_steps
        or live_callback._every_n_epochs != 0
        or live_callback._train_time_interval is not None
        or live_callback._save_on_train_epoch_end is not None
        or Path(os.path.abspath(os.fspath(live_callback.dirpath)))
        != checkpoint_path.parent
    ):
        raise RuntimeError(
            "pilot live ModelCheckpoint disagrees with the exact final-step contract"
        )
    _exact_nested_state_match(
        state,
        live_callback.state_dict(),
        label="pilot serialized/live ModelCheckpoint state",
    )
    tensor_path = ("checkpoint", "callbacks", callback_key, "kth_value")
    return (
        tensor_path,
        sentinel,
        {
            "all_expected_and_only_expected_verified": True,
            "nonfinite_tensor_count": 1,
            "nonfinite_element_count": 1,
            "records": [
                {
                    "tensor_path_components": list(tensor_path),
                    "framework": "lightning",
                    "framework_version": _PINNED_LIGHTNING_VERSION,
                    "callback": "ModelCheckpoint",
                    "field": "kth_value",
                    "dtype": "float32",
                    "shape": [],
                    "value": "+inf",
                    "meaning": "unranked_min_mode_checkpoint_sentinel",
                    "excluded_from_non_sentinel_finiteness": True,
                }
            ],
        },
        {
            "exact_serialized_live_match": True,
            "model_checkpoint_callback_count": 1,
            "state_key": callback_key,
            "configuration_matches_pilot_contract": True,
        },
    )


def _resolved_omegaconf_mapping(value, *, label):
    if not isinstance(value, omegaconf.DictConfig):
        raise RuntimeError(f"{label} must be an OmegaConf DictConfig")
    try:
        resolved = omegaconf.OmegaConf.to_container(
            value,
            resolve=True,
            enum_to_str=True,
        )
    except Exception as error:
        raise RuntimeError(f"{label} cannot be fully resolved") from error
    if not isinstance(resolved, Mapping):
        raise RuntimeError(f"{label} did not resolve to a mapping")
    return dict(resolved)


def _unresolved_omegaconf_mapping(value, *, label):
    if not isinstance(value, omegaconf.DictConfig):
        raise RuntimeError(f"{label} must be an OmegaConf DictConfig")
    try:
        unresolved = omegaconf.OmegaConf.to_container(
            value,
            resolve=False,
            enum_to_str=True,
        )
    except Exception as error:
        raise RuntimeError(f"{label} cannot be structurally serialized") from error
    if not isinstance(unresolved, Mapping):
        raise RuntimeError(f"{label} did not serialize to a mapping")
    return dict(unresolved)


def _validated_checkpoint_hyperparameters(
    checkpoint,
    *,
    model,
    expected_resolved_config,
    expected_config_sha256,
):
    """Bind reload-time checkpoint configuration to preflight and the live model."""

    if not isinstance(expected_resolved_config, Mapping):
        raise RuntimeError("pilot preflight resolved config is not a mapping")
    expected_resolved_config = dict(expected_resolved_config)
    if _canonical_json_sha256(expected_resolved_config) != expected_config_sha256:
        raise RuntimeError("pilot preflight resolved config digest changed")
    if checkpoint.get("hparams_name") != "kwargs":
        raise RuntimeError("pilot checkpoint hparams_name must be exactly 'kwargs'")
    hyperparameters = checkpoint.get("hyper_parameters")
    if not isinstance(hyperparameters, Mapping) or set(hyperparameters) != {"config"}:
        raise RuntimeError(
            "pilot checkpoint hyper_parameters must contain exactly config"
        )
    checkpoint_config = _resolved_omegaconf_mapping(
        hyperparameters["config"],
        label="pilot checkpoint hyper_parameters.config",
    )
    _exact_nested_state_match(
        checkpoint_config,
        expected_resolved_config,
        label="pilot checkpoint/preflight resolved config",
    )

    live_config = _resolved_omegaconf_mapping(
        getattr(model, "config", None),
        label="pilot live model config",
    )
    _exact_nested_state_match(
        live_config,
        expected_resolved_config,
        label="pilot live-model/preflight resolved config",
    )
    live_hyperparameters = getattr(model, "hparams", None)
    if not isinstance(live_hyperparameters, Mapping) or set(live_hyperparameters) != {
        "config"
    }:
        raise RuntimeError("pilot live model hparams must contain exactly config")
    live_hparams_config = _resolved_omegaconf_mapping(
        live_hyperparameters["config"],
        label="pilot live model hparams.config",
    )
    _exact_nested_state_match(
        live_hparams_config,
        expected_resolved_config,
        label="pilot live-hparams/preflight resolved config",
    )
    checkpoint_unresolved_config = _unresolved_omegaconf_mapping(
        hyperparameters["config"],
        label="pilot checkpoint hyper_parameters.config",
    )
    live_unresolved_config = _unresolved_omegaconf_mapping(
        getattr(model, "config", None),
        label="pilot live model config",
    )
    live_hparams_unresolved_config = _unresolved_omegaconf_mapping(
        live_hyperparameters["config"],
        label="pilot live model hparams.config",
    )
    _exact_nested_state_match(
        checkpoint_unresolved_config,
        live_unresolved_config,
        label="pilot checkpoint/live-model unresolved config",
    )
    _exact_nested_state_match(
        checkpoint_unresolved_config,
        live_hparams_unresolved_config,
        label="pilot checkpoint/live-hparams unresolved config",
    )
    return {
        "hparams_name": "kwargs",
        "exact_hyperparameter_keys": True,
        "exact_checkpoint_preflight_config_match": True,
        "exact_live_model_preflight_config_match": True,
        "exact_live_hparams_preflight_config_match": True,
        "exact_checkpoint_live_model_unresolved_config_match": True,
        "exact_checkpoint_live_hparams_unresolved_config_match": True,
        "resolved_config_sha256": expected_config_sha256,
    }


def _normalized_lightning_precision(value):
    aliases = {
        "16": "16-mixed",
        "bf16": "bf16-mixed",
        "32": "32-true",
        "64": "64-true",
        16: "16-mixed",
        32: "32-true",
        64: "64-true",
    }
    return aliases.get(value, value)


def _validated_live_trainer_configuration(config, trainer):
    """Bind safety- and numerics-relevant live Trainer settings to config."""

    trainer_config = config.get("trainer")
    if not isinstance(trainer_config, Mapping):
        raise RuntimeError("pilot resolved trainer config is not a mapping")
    if trainer_config.get("detect_anomaly") is not True:
        raise RuntimeError("pilot resolved config must enable anomaly detection")
    if getattr(trainer, "_detect_anomaly", None) is not True:
        raise RuntimeError("pilot live Trainer did not enable anomaly detection")

    configured_clip = trainer_config.get("gradient_clip_val")
    live_clip = getattr(trainer, "gradient_clip_val", None)
    if (
        isinstance(configured_clip, bool)
        or not isinstance(configured_clip, (int, float))
        or not math.isfinite(configured_clip)
        or configured_clip < 0.0
        or isinstance(live_clip, bool)
        or not isinstance(live_clip, (int, float))
        or float(live_clip) != float(configured_clip)
    ):
        raise RuntimeError(
            "pilot live Trainer gradient clipping disagrees with resolved config"
        )
    configured_clip_algorithm = trainer_config.get("gradient_clip_algorithm")
    live_clip_algorithm_object = getattr(trainer, "gradient_clip_algorithm", None)
    if configured_clip_algorithm is None:
        # Lightning 2.5.1 retains ``None`` on Trainer when the field is omitted
        # or null, then applies its documented effective default, norm, at the
        # clipping call site.
        exact_clip_algorithm_match = live_clip_algorithm_object is None
        effective_clip_algorithm = "norm"
    else:
        live_clip_algorithm = getattr(live_clip_algorithm_object, "value", None)
        exact_clip_algorithm_match = (
            configured_clip_algorithm in {"norm", "value"}
            and live_clip_algorithm == configured_clip_algorithm
        )
        effective_clip_algorithm = configured_clip_algorithm
    if not exact_clip_algorithm_match:
        raise RuntimeError(
            "pilot live Trainer gradient-clip algorithm disagrees with resolved "
            "config"
        )

    configured_precision = trainer_config.get("precision")
    expected_live_precision = _normalized_lightning_precision(configured_precision)
    live_precision = getattr(trainer, "precision", None)
    if (
        not isinstance(configured_precision, (str, int))
        or isinstance(configured_precision, bool)
        or not isinstance(expected_live_precision, str)
        or live_precision != expected_live_precision
    ):
        raise RuntimeError(
            "pilot live Trainer precision disagrees with resolved config"
        )
    return {
        "exact_detect_anomaly_match": True,
        "detect_anomaly": True,
        "exact_gradient_clip_val_match": True,
        "gradient_clip_val": float(configured_clip),
        "exact_gradient_clip_algorithm_match": True,
        "gradient_clip_algorithm": effective_clip_algorithm,
        "exact_precision_match": True,
        "configured_precision": str(configured_precision),
        "live_precision": live_precision,
    }


def _expected_lightning_loop_state(*, optimizer_steps, accumulation):
    microbatches = optimizer_steps * accumulation
    zero_batch_progress = {
        "total": {"ready": 0, "completed": 0, "started": 0, "processed": 0},
        "current": {"ready": 0, "completed": 0, "started": 0, "processed": 0},
        "is_last_batch": False,
    }
    return {
        "fit_loop": {
            "state_dict": {},
            "epoch_loop.state_dict": {"_batches_that_stepped": optimizer_steps},
            "epoch_loop.batch_progress": {
                "total": {
                    "ready": microbatches,
                    "completed": microbatches,
                    "started": microbatches,
                    "processed": microbatches,
                },
                "current": {
                    "ready": microbatches,
                    "completed": microbatches,
                    "started": microbatches,
                    "processed": microbatches,
                },
                "is_last_batch": False,
            },
            "epoch_loop.scheduler_progress": {
                "total": {"ready": optimizer_steps, "completed": optimizer_steps},
                "current": {"ready": optimizer_steps, "completed": optimizer_steps},
            },
            "epoch_loop.automatic_optimization.state_dict": {},
            "epoch_loop.automatic_optimization.optim_progress": {
                "optimizer": {
                    "step": {
                        "total": {
                            "ready": optimizer_steps,
                            "completed": optimizer_steps,
                        },
                        "current": {
                            "ready": optimizer_steps,
                            "completed": optimizer_steps,
                        },
                    },
                    "zero_grad": {
                        "total": {
                            "ready": optimizer_steps,
                            "completed": optimizer_steps,
                            "started": optimizer_steps,
                        },
                        "current": {
                            "ready": optimizer_steps,
                            "completed": optimizer_steps,
                            "started": optimizer_steps,
                        },
                    },
                }
            },
            "epoch_loop.manual_optimization.state_dict": {},
            "epoch_loop.manual_optimization.optim_step_progress": {
                "total": {"ready": 0, "completed": 0},
                "current": {"ready": 0, "completed": 0},
            },
            "epoch_loop.val_loop.state_dict": {},
            "epoch_loop.val_loop.batch_progress": zero_batch_progress,
            "epoch_progress": {
                "total": {"ready": 1, "completed": 0, "started": 1, "processed": 0},
                "current": {
                    "ready": 1,
                    "completed": 0,
                    "started": 1,
                    "processed": 0,
                },
            },
        },
        "validate_loop": {"state_dict": {}, "batch_progress": zero_batch_progress},
        "test_loop": {"state_dict": {}, "batch_progress": zero_batch_progress},
        "predict_loop": {
            "state_dict": {},
            "batch_progress": {
                "total": {
                    "ready": 0,
                    "completed": 0,
                    "started": 0,
                    "processed": 0,
                },
                "current": {
                    "ready": 0,
                    "completed": 0,
                    "started": 0,
                    "processed": 0,
                },
            },
        },
    }


def _validated_checkpoint_loop_state(
    checkpoint, *, expected_steps, expected_resolved_config
):
    trainer_config = expected_resolved_config.get("trainer")
    if not isinstance(trainer_config, Mapping):
        raise RuntimeError("pilot resolved config has no trainer mapping")
    accumulation = trainer_config.get("accumulate_grad_batches")
    if type(accumulation) is not int or accumulation <= 0:
        raise RuntimeError("pilot resolved gradient accumulation is invalid")
    epoch = checkpoint.get("epoch")
    if type(epoch) is not int or epoch != 0:
        raise RuntimeError("pilot checkpoint epoch must be zero for the hosted stream")
    expected_loops = _expected_lightning_loop_state(
        optimizer_steps=expected_steps,
        accumulation=accumulation,
    )
    _exact_nested_state_match(
        checkpoint.get("loops"),
        expected_loops,
        label="pilot serialized Lightning loop state",
    )
    return {
        "exact_serialized_progress_match": True,
        "epoch": 0,
        "optimizer_steps": expected_steps,
        "accumulate_grad_batches": accumulation,
        "microbatches": expected_steps * accumulation,
    }


def _validated_live_training_state(
    checkpoint,
    *,
    trainer,
    model,
    expected_steps,
    expected_resolved_config,
):
    optimizer_states = checkpoint.get("optimizer_states")
    live_optimizers = getattr(trainer, "optimizers", None)
    if (
        not isinstance(optimizer_states, list)
        or len(optimizer_states) != 1
        or not isinstance(live_optimizers, (list, tuple))
        or len(live_optimizers) != 1
        or not isinstance(live_optimizers[0], torch.optim.AdamW)
    ):
        raise RuntimeError("pilot requires exactly one serialized/live AdamW optimizer")
    live_optimizer = live_optimizers[0]
    backbone = getattr(model, "backbone", None)
    backbone_parameters = (
        None if backbone is None else getattr(backbone, "parameters", None)
    )
    if not callable(backbone_parameters):
        raise RuntimeError("pilot model backbone does not expose its parameters")
    expected_parameters = list(backbone_parameters())
    expected_optim = expected_resolved_config.get("optim")
    if not isinstance(expected_optim, Mapping) or set(expected_optim) != {
        "weight_decay",
        "lr",
        "beta1",
        "beta2",
        "eps",
        "scheduler",
    }:
        raise RuntimeError("pilot resolved optimizer config is invalid")
    try:
        expected_schedule_spec = optimizer_scheduler_spec(expected_optim)
    except (TypeError, ValueError) as error:
        raise RuntimeError("pilot resolved optimizer schedule is invalid") from error
    if getattr(model, "optimizer_scheduler_spec", None) != expected_schedule_spec:
        raise RuntimeError(
            "pilot live model optimizer-scheduler spec disagrees with resolved config"
        )
    beta1 = expected_optim.get("beta1")
    beta2 = expected_optim.get("beta2")
    eps = expected_optim.get("eps")
    weight_decay = expected_optim.get("weight_decay")
    if (
        any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in (beta1, beta2, eps, weight_decay)
        )
        or not 0.0 <= beta1 < 1.0
        or not 0.0 <= beta2 < 1.0
        or eps <= 0.0
        or weight_decay < 0.0
    ):
        raise RuntimeError("pilot resolved AdamW hyperparameters are invalid")
    live_parameter_groups = live_optimizer.param_groups
    if (
        len(live_parameter_groups) != 1
        or list(live_parameter_groups[0].get("params", [])) != expected_parameters
    ):
        raise RuntimeError(
            "pilot AdamW does not optimize exactly the backbone parameters"
        )
    serialized_optimizer = optimizer_states[0]
    live_optimizer_state = live_optimizer.state_dict()
    _exact_nested_state_match(
        serialized_optimizer,
        live_optimizer_state,
        label="pilot serialized/live AdamW state",
    )
    if (
        not isinstance(serialized_optimizer, Mapping)
        or set(serialized_optimizer) != {"state", "param_groups"}
        or not isinstance(serialized_optimizer["state"], Mapping)
        or not serialized_optimizer["state"]
        or not isinstance(serialized_optimizer["param_groups"], list)
        or len(serialized_optimizer["param_groups"]) != 1
    ):
        raise RuntimeError("pilot serialized AdamW state structure is invalid")
    serialized_group = serialized_optimizer["param_groups"][0]
    if not isinstance(serialized_group, Mapping):
        raise RuntimeError("pilot serialized AdamW parameter group is invalid")
    expected_current_lr = expected_schedule_spec.peak_lr * (
        optimizer_scheduler_multiplier(expected_schedule_spec, expected_steps)
    )
    expected_group_without_parameters = {
        "lr": expected_current_lr,
        "betas": (beta1, beta2),
        "eps": eps,
        "weight_decay": weight_decay,
        "amsgrad": False,
        "foreach": None,
        "maximize": False,
        "capturable": False,
        "differentiable": False,
        "fused": None,
        "initial_lr": expected_schedule_spec.peak_lr,
    }
    serialized_group_without_parameters = {
        key: value for key, value in serialized_group.items() if key != "params"
    }
    _exact_nested_state_match(
        serialized_group_without_parameters,
        expected_group_without_parameters,
        label="pilot serialized AdamW/resolved-config parameter group",
    )
    serialized_parameter_ids = serialized_group.get("params")
    if (
        not isinstance(serialized_parameter_ids, list)
        or not serialized_parameter_ids
        or any(
            type(value) is not int or value < 0 for value in serialized_parameter_ids
        )
        or len(set(serialized_parameter_ids)) != len(serialized_parameter_ids)
        or set(serialized_parameter_ids) != set(serialized_optimizer["state"])
    ):
        raise RuntimeError(
            "pilot serialized AdamW parameter IDs/state are inconsistent"
        )
    for parameter_id, state in serialized_optimizer["state"].items():
        if not isinstance(state, Mapping) or set(state) != {
            "step",
            "exp_avg",
            "exp_avg_sq",
        }:
            raise RuntimeError(
                f"pilot serialized AdamW state entry is invalid: {parameter_id}"
            )
        step = state["step"]
        if (
            not isinstance(step, torch.Tensor)
            or step.shape != torch.Size([])
            or not bool((step == expected_steps).item())
        ):
            raise RuntimeError(
                f"pilot serialized AdamW step is invalid: {parameter_id}"
            )

    serialized_schedulers = checkpoint.get("lr_schedulers")
    live_scheduler_configs = getattr(trainer, "lr_scheduler_configs", None)
    if (
        not isinstance(serialized_schedulers, list)
        or len(serialized_schedulers) != 1
        or not isinstance(live_scheduler_configs, (list, tuple))
        or len(live_scheduler_configs) != 1
    ):
        raise RuntimeError("pilot requires exactly one serialized/live LR scheduler")
    live_scheduler_config = live_scheduler_configs[0]
    live_scheduler = getattr(live_scheduler_config, "scheduler", None)
    if (
        not isinstance(live_scheduler, torch.optim.lr_scheduler.LambdaLR)
        or live_scheduler.optimizer is not live_optimizer
        or getattr(live_scheduler_config, "interval", None) != "step"
        or getattr(live_scheduler_config, "name", None) != "lr"
    ):
        raise RuntimeError("pilot live LR scheduler contract is invalid")
    serialized_scheduler = serialized_schedulers[0]
    live_scheduler_state = live_scheduler.state_dict()
    _exact_nested_state_match(
        serialized_scheduler,
        live_scheduler_state,
        label="pilot serialized/live LambdaLR state",
    )
    if (
        not isinstance(serialized_scheduler, Mapping)
        or serialized_scheduler.get("last_epoch") != expected_steps
        or serialized_scheduler.get("_step_count") != expected_steps + 1
        or serialized_scheduler.get("base_lrs") != [serialized_group.get("initial_lr")]
        or serialized_scheduler.get("_last_lr") != [serialized_group.get("lr")]
    ):
        raise RuntimeError("pilot serialized LambdaLR step/LR state is invalid")
    live_lambdas = getattr(live_scheduler, "lr_lambdas", None)
    if (
        not isinstance(live_lambdas, list)
        or len(live_lambdas) != 1
        or not callable(live_lambdas[0])
    ):
        raise RuntimeError("pilot live LambdaLR callable structure is invalid")
    schedule_check_maximum = max(
        expected_steps,
        expected_schedule_spec.warmup_updates + 1,
        (expected_schedule_spec.horizon_updates or 0) + 1,
    )
    for scheduler_index in range(schedule_check_maximum + 1):
        observed_multiplier = live_lambdas[0](scheduler_index)
        expected_multiplier = optimizer_scheduler_multiplier(
            expected_schedule_spec, scheduler_index
        )
        if (
            isinstance(observed_multiplier, bool)
            or not isinstance(observed_multiplier, (int, float))
            or not math.isfinite(observed_multiplier)
            or float(observed_multiplier) != expected_multiplier
        ):
            raise RuntimeError(
                "pilot live LambdaLR callable disagrees with resolved schedule at "
                f"index {scheduler_index}"
            )

    serialized_sampler = checkpoint.get("sampler")
    train_dataloader = getattr(trainer, "train_dataloader", None)
    if train_dataloader is None:
        raise RuntimeError("pilot trainer has no hosted-stream train dataloader")
    sampler = getattr(train_dataloader, "sampler", None)
    if (
        sampler is None
        or type(sampler).__module__ != "torch.utils.data.dataloader"
        or type(sampler).__name__ != "_InfiniteConstantSampler"
    ):
        raise RuntimeError("pilot trainer has an unexpected hosted-stream sampler")
    sampler_state_dict = (
        None if sampler is None else getattr(sampler, "state_dict", None)
    )
    if callable(sampler_state_dict):
        raise RuntimeError(
            "pilot hosted-stream sampler unexpectedly exposes mutable state"
        )
    if serialized_sampler != {"random_state": None}:
        raise RuntimeError(
            "pilot hosted-stream sampler must have exact null random-state evidence"
        )

    return {
        "optimizer": {
            "exact_serialized_live_match": True,
            "optimizer_count": 1,
            "optimizer_class": "AdamW",
            "parameter_group_count": 1,
            "parameter_state_count": len(serialized_optimizer["state"]),
            "exact_resolved_config_match": True,
        },
        "scheduler": {
            "exact_serialized_live_match": True,
            "scheduler_count": 1,
            "scheduler_class": "LambdaLR",
            "interval": "step",
            "name": "lr",
            "last_epoch": expected_steps,
            "step_count": expected_steps + 1,
            "exact_model_spec_match": True,
            "exact_callable_schedule_match": True,
            "callable_schedule_index_checks": schedule_check_maximum + 1,
        },
        "sampler": {
            "exact_hosted_stream_contract_match": True,
            "random_state_is_none": True,
            "live_state_dict_available": False,
            "sampler_class_module": "torch.utils.data.dataloader",
            "sampler_class_name": "_InfiniteConstantSampler",
        },
    }


def _validate_checkpoint_matches_live_model(checkpoint_state, model):
    live_state = model.state_dict()
    if not isinstance(live_state, Mapping) or not live_state:
        raise RuntimeError("pilot live model has no state_dict tensors")
    if set(checkpoint_state) != set(live_state):
        missing = sorted(set(live_state) - set(checkpoint_state))
        unexpected = sorted(set(checkpoint_state) - set(live_state))
        raise RuntimeError(
            "pilot checkpoint state_dict keys disagree with the live model: "
            f"missing={missing}, unexpected={unexpected}"
        )
    compared = 0
    for name, live_tensor in live_state.items():
        checkpoint_tensor = checkpoint_state[name]
        if not isinstance(live_tensor, torch.Tensor) or not isinstance(
            checkpoint_tensor, torch.Tensor
        ):
            raise RuntimeError(f"pilot checkpoint state entry is not a tensor: {name}")
        if (
            checkpoint_tensor.device.type == "meta"
            or live_tensor.device.type == "meta"
            or checkpoint_tensor.shape != live_tensor.shape
            or checkpoint_tensor.dtype != live_tensor.dtype
            or not torch.equal(
                checkpoint_tensor.detach().cpu(), live_tensor.detach().cpu()
            )
        ):
            raise RuntimeError(
                f"pilot checkpoint tensor disagrees with the live model: {name}"
            )
        compared += 1
    return {
        "exact_key_set": True,
        "exact_tensor_values": True,
        "tensor_count": compared,
    }


def _validate_checkpoint_ema_matches_live(checkpoint_shadows, model):
    ema = getattr(model, "ema", None)
    live_shadows = None if ema is None else getattr(ema, "shadow_params", None)
    if not isinstance(live_shadows, (list, tuple)) or not live_shadows:
        raise RuntimeError("pilot live model has no EMA shadow tensors")
    if len(checkpoint_shadows) != len(live_shadows):
        raise RuntimeError("pilot checkpoint EMA count disagrees with the live model")
    for index, (checkpoint_tensor, live_tensor) in enumerate(
        zip(checkpoint_shadows, live_shadows, strict=True)
    ):
        if not isinstance(checkpoint_tensor, torch.Tensor) or not isinstance(
            live_tensor, torch.Tensor
        ):
            raise RuntimeError(f"pilot EMA entry is not a tensor: {index}")
        if (
            checkpoint_tensor.device.type == "meta"
            or live_tensor.device.type == "meta"
            or checkpoint_tensor.shape != live_tensor.shape
            or checkpoint_tensor.dtype != live_tensor.dtype
            or not torch.equal(
                checkpoint_tensor.detach().cpu(), live_tensor.detach().cpu()
            )
        ):
            raise RuntimeError(
                f"pilot checkpoint EMA tensor disagrees with the live model: {index}"
            )
    return {
        "exact_tensor_values": True,
        "tensor_count": len(checkpoint_shadows),
    }


def _validated_ema_metadata(ema_state, *, label, expected_updates):
    expected_keys = {"shadow_params", "decay", "num_updates"}
    if not isinstance(ema_state, Mapping) or set(ema_state) != expected_keys:
        observed_keys = set(ema_state) if isinstance(ema_state, Mapping) else set()
        missing = sorted(expected_keys - observed_keys)
        unexpected = sorted(repr(key) for key in observed_keys - expected_keys)
        raise RuntimeError(
            f"{label} EMA keys disagree with the exact schema: "
            f"missing={missing}, unexpected={unexpected}"
        )
    shadows = ema_state.get("shadow_params")
    if not isinstance(shadows, (list, tuple)) or not shadows:
        raise RuntimeError(f"{label} has no EMA shadow tensors")
    decay = ema_state.get("decay")
    if isinstance(decay, bool) or not isinstance(decay, (int, float)):
        raise RuntimeError(f"{label} EMA decay is not a real scalar")
    decay = float(decay)
    if not math.isfinite(decay) or not 0.0 < decay < 1.0:
        raise RuntimeError(f"{label} EMA decay must be finite and in (0, 1)")
    num_updates = ema_state.get("num_updates")
    if type(num_updates) is not int or num_updates != expected_updates:
        raise RuntimeError(
            f"{label} EMA update count {num_updates!r}; expected {expected_updates}"
        )
    return {
        "shadow_parameter_count": len(shadows),
        "decay": decay,
        "num_updates": num_updates,
    }


def _audit_pilot_checkpoint(
    path,
    *,
    expected_steps,
    model,
    trainer,
    expected_resolved_config,
    expected_config_sha256,
):
    """Load and semantically audit the exact stable checkpoint bytes."""

    path = Path(os.path.abspath(os.fspath(path)))
    snapshot_before, _unused = _stable_file_snapshot(path)
    if snapshot_before["size_bytes"] <= 0:
        raise RuntimeError("pilot final checkpoint is empty")
    with verified_checkpoint_file(
        path,
        expected_sha256=snapshot_before["sha256"],
    ) as (checkpoint_file, checkpoint_identity):
        identity_snapshot = {
            "path": str(path),
            "device": checkpoint_identity.device,
            "inode": checkpoint_identity.inode,
            "mode": checkpoint_identity.mode,
            "link_count": checkpoint_identity.link_count,
            "size_bytes": checkpoint_identity.size_bytes,
            "mtime_ns": checkpoint_identity.mtime_ns,
            "ctime_ns": checkpoint_identity.ctime_ns,
            "sha256": checkpoint_identity.sha256,
            "stable_regular_file_verified": True,
        }
        if identity_snapshot != snapshot_before:
            raise RuntimeError(
                "pilot final checkpoint changed before descriptor-bound loading"
            )
        try:
            checkpoint = torch.load(
                checkpoint_file,
                map_location="cpu",
                weights_only=False,
            )
        except Exception as error:
            raise RuntimeError(
                "pilot final checkpoint cannot be deserialized"
            ) from error
    snapshot_after, _unused = _stable_file_snapshot(path)
    if snapshot_after != snapshot_before:
        raise RuntimeError("pilot final checkpoint changed while it was audited")
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("pilot final checkpoint must deserialize to a mapping")
    _validated_checkpoint_top_level_schema(checkpoint, model=model)
    checkpoint_python_floats = _python_float_finiteness(
        checkpoint,
        prefix="checkpoint",
        label="serialized checkpoint",
    )
    checkpoint_hyperparameters_match = _validated_checkpoint_hyperparameters(
        checkpoint,
        model=model,
        expected_resolved_config=expected_resolved_config,
        expected_config_sha256=expected_config_sha256,
    )
    checkpoint_loop_state_match = _validated_checkpoint_loop_state(
        checkpoint,
        expected_steps=expected_steps,
        expected_resolved_config=expected_resolved_config,
    )
    checkpoint_step = checkpoint.get("global_step")
    if (
        isinstance(checkpoint_step, bool)
        or not isinstance(checkpoint_step, int)
        or checkpoint_step != expected_steps
    ):
        raise RuntimeError(
            "pilot checkpoint global_step "
            f"{checkpoint_step!r}; expected {expected_steps}"
        )
    checkpoint_state = checkpoint.get("state_dict")
    if not isinstance(checkpoint_state, Mapping) or not checkpoint_state:
        raise RuntimeError("pilot checkpoint has no nonempty state_dict mapping")
    raw_tensors = _floating_tensor_finiteness(
        checkpoint_state.items(), label="serialized checkpoint model state"
    )
    ema_state = checkpoint.get("ema")
    if not isinstance(ema_state, Mapping):
        raise RuntimeError("pilot checkpoint has no EMA mapping")
    checkpoint_shadows = ema_state.get("shadow_params")
    if not isinstance(checkpoint_shadows, (list, tuple)) or not checkpoint_shadows:
        raise RuntimeError("pilot checkpoint has no EMA shadow tensors")
    ema_tensors = _floating_tensor_finiteness(
        (
            (f"shadow_params[{index}]", tensor)
            for index, tensor in enumerate(checkpoint_shadows)
        ),
        label="serialized checkpoint EMA state",
    )
    ema_metadata = _validated_ema_metadata(
        ema_state,
        label="pilot checkpoint",
        expected_updates=expected_steps,
    )
    optimizer_states = checkpoint.get("optimizer_states")
    if not isinstance(optimizer_states, list) or not optimizer_states:
        raise RuntimeError("pilot checkpoint has no optimizer state")
    optimizer_tensors = _floating_tensor_finiteness(
        _nested_named_tensors(optimizer_states, prefix="optimizer_states"),
        label="serialized checkpoint optimizer state",
    )
    live_training_state_match = _validated_live_training_state(
        checkpoint,
        trainer=trainer,
        model=model,
        expected_steps=expected_steps,
        expected_resolved_config=expected_resolved_config,
    )
    trainer_live_configuration_match = _validated_live_trainer_configuration(
        expected_resolved_config,
        trainer,
    )
    (
        framework_sentinel_path,
        framework_sentinel,
        framework_sentinels,
        model_checkpoint_live_state_match,
    ) = _validated_lightning_checkpoint_sentinel(
        checkpoint,
        checkpoint_path=path,
        expected_steps=expected_steps,
        trainer=trainer,
    )
    sentinel_occurrences = 0

    def non_sentinel_checkpoint_tensors():
        nonlocal sentinel_occurrences
        for tensor_path, item in _nested_path_leaves(checkpoint, path=("checkpoint",)):
            if not isinstance(item, torch.Tensor):
                continue
            if tensor_path == framework_sentinel_path:
                if item is not framework_sentinel:
                    raise RuntimeError(
                        "pilot checkpoint sentinel structural identity changed"
                    )
                sentinel_occurrences += 1
                continue
            yield _format_structural_path(tensor_path), item

    non_sentinel_tensors = _floating_tensor_finiteness(
        non_sentinel_checkpoint_tensors(),
        label="serialized non-sentinel checkpoint",
    )
    if sentinel_occurrences != 1:
        raise RuntimeError(
            "pilot checkpoint did not consume exactly one framework sentinel"
        )
    prior_validator = getattr(model, "_validate_udlm_prior_checkpoint", None)
    if not callable(prior_validator):
        raise RuntimeError("pilot model has no UDLM checkpoint identity validator")
    prior_validator(checkpoint)
    conditioning_validator = getattr(
        model, "_validate_udlm_conditioning_checkpoint", None
    )
    if not callable(conditioning_validator):
        raise RuntimeError(
            "pilot model has no UDLM conditioning checkpoint identity validator"
        )
    conditioning_validator(checkpoint)
    live_match = _validate_checkpoint_matches_live_model(checkpoint_state, model)
    live_ema_match = _validate_checkpoint_ema_matches_live(checkpoint_shadows, model)
    live_ema = getattr(model, "ema", None)
    live_ema_metadata = _validated_ema_metadata(
        {
            "shadow_params": getattr(live_ema, "shadow_params", None),
            "decay": getattr(live_ema, "decay", None),
            "num_updates": getattr(live_ema, "num_updates", None),
        },
        label="pilot live model",
        expected_updates=expected_steps,
    )
    if live_ema_metadata != ema_metadata:
        raise RuntimeError("pilot checkpoint EMA metadata disagrees with live model")
    return snapshot_before, {
        "deserialized": True,
        "global_step": checkpoint_step,
        "raw_model": raw_tensors,
        "ema": ema_tensors,
        "ema_metadata": ema_metadata,
        "optimizer": optimizer_tensors,
        "non_sentinel_checkpoint_tensors": non_sentinel_tensors,
        "checkpoint_python_floats": checkpoint_python_floats,
        "framework_nonfinite_sentinels": framework_sentinels,
        "checkpoint_hyperparameters_match": checkpoint_hyperparameters_match,
        "checkpoint_loop_state_match": checkpoint_loop_state_match,
        "optimizer_live_state_match": live_training_state_match["optimizer"],
        "scheduler_live_state_match": live_training_state_match["scheduler"],
        "sampler_live_state_match": live_training_state_match["sampler"],
        "trainer_live_configuration_match": trainer_live_configuration_match,
        "model_checkpoint_live_state_match": model_checkpoint_live_state_match,
        "udlm_process_identity_verified": True,
        "live_model_match": live_match,
        "live_ema_match": live_ema_match,
    }


def _verified_warm_start_report(config, startup_mode, warm_start_report):
    if startup_mode != "warm_start":
        if warm_start_report is not None:
            raise RuntimeError(
                "non-warm-start pilot unexpectedly retained a warm-start report"
            )
        return None
    if not isinstance(warm_start_report, Mapping):
        raise RuntimeError("warm-start pilot did not retain its verified MDLM report")
    report = dict(warm_start_report)
    expected_sha256 = config.training.get("init_from_mdlm_checkpoint_sha256")
    if (
        not isinstance(expected_sha256, str)
        or report.get("source_sha256") != expected_sha256
        or report.get("expected_source_sha256") != expected_sha256
        or report.get("byte_identity_verified_before_and_after_load") is not True
    ):
        raise RuntimeError("warm-start report disagrees with the verified MDLM source")
    return report


def _validated_screen_initialization_state_audit(
    value, *, config, startup_mode, warm_start_report
):
    """Validate the retained pre-optimizer state certificate at completion."""

    manifest = _PILOT_CONTRACT.get("launch_manifest")
    screen = (
        manifest.get("optimization_screen") if isinstance(manifest, Mapping) else None
    )
    if screen is None:
        if value is not None:
            raise RuntimeError(
                "non-screen pilot unexpectedly retained an initialization-state audit"
            )
        return None
    if not isinstance(screen, Mapping):
        raise RuntimeError("optimization-screen launch contract must be an object")
    if not isinstance(value, Mapping):
        raise RuntimeError(
            "optimization-screen completion lacks its initialization-state audit"
        )
    audit = dict(value)
    expected_keys = {
        "schema_version",
        "phase",
        "source_checkpoint_sha256",
        "resolved_training_config_sha256",
        "training_seed",
        "conditioning_variant",
        "common_backbone_tensor_count",
        "common_backbone_state_sha256",
        "full_initial_tensor_count",
        "full_initial_state_sha256",
    }
    if set(audit) != expected_keys:
        raise RuntimeError(
            "optimization-screen initialization-state audit has invalid keys"
        )
    warm_start = _verified_warm_start_report(config, startup_mode, warm_start_report)
    expected_values = {
        "schema_version": _SCREEN_INITIALIZATION_STATE_AUDIT_SCHEMA_VERSION,
        "phase": _SCREEN_INITIALIZATION_STATE_AUDIT_PHASE,
        "source_checkpoint_sha256": warm_start["source_sha256"],
        "resolved_training_config_sha256": _PILOT_CONTRACT[
            "GENMOL_TRAIN_EXPECTED_CONFIG_SHA256"
        ],
        "training_seed": _exact_training_seed(
            config.get("seed", 1), "optimization-screen training seed"
        ),
        "conditioning_variant": _configured_conditioning_variant(config),
    }
    for key, expected in expected_values.items():
        if audit.get(key) != expected:
            raise RuntimeError(
                f"optimization-screen initialization-state audit changed {key}"
            )
    for key in ("common_backbone_tensor_count", "full_initial_tensor_count"):
        _exact_positive_integer(audit.get(key), f"initialization-state {key}")
    if audit["full_initial_tensor_count"] < audit["common_backbone_tensor_count"]:
        raise RuntimeError(
            "initialization-state full tensor count is smaller than common backbone"
        )
    for key in ("common_backbone_state_sha256", "full_initial_state_sha256"):
        if not isinstance(audit.get(key), str) or not re.fullmatch(
            r"[0-9a-f]{64}", audit[key]
        ):
            raise RuntimeError(f"initialization-state {key} is not a SHA-256")
    return audit


def _write_pilot_training_summary(
    *,
    config,
    trainer,
    model,
    preflight_record,
    startup_mode,
    warm_start_report,
    screen_initialization_state_audit=None,
    train_dataloader=None,
    training_rng_policy=None,
):
    """Publish the sole rank-zero certificate that a pilot completed."""

    if _PILOT_CONTRACT is None:
        return None
    if not bool(getattr(trainer, "is_global_zero", False)):
        return None
    global_rank = getattr(trainer, "global_rank", 0)
    if (
        isinstance(global_rank, bool)
        or not isinstance(global_rank, int)
        or global_rank != 0
    ):
        raise RuntimeError("pilot global-zero process reports an invalid global rank")
    expected_steps = _PILOT_CONTRACT["expected_max_steps"]
    expected_world_size = _PILOT_CONTRACT["expected_world_size"]
    observed_steps = _exact_positive_integer(
        getattr(trainer, "global_step", None), "pilot completed global step"
    )
    observed_world_size = _exact_positive_integer(
        getattr(trainer, "world_size", None), "pilot completed world size"
    )
    if observed_steps != expected_steps:
        raise RuntimeError(
            f"pilot stopped at global step {observed_steps!r}; expected {expected_steps}"
        )
    if observed_world_size != expected_world_size:
        raise RuntimeError(
            "pilot runtime world size "
            f"{observed_world_size!r}; expected {expected_world_size}"
        )
    callbacks = getattr(trainer, "callbacks", None)
    health_callbacks = (
        []
        if not isinstance(callbacks, (list, tuple))
        else [
            callback
            for callback in callbacks
            if isinstance(callback, _PilotFiniteLossCallback)
        ]
    )
    if len(health_callbacks) != 1:
        raise RuntimeError("pilot trainer must retain exactly one health callback")
    training_health = health_callbacks[0].completion_report(
        expected_optimizer_steps=expected_steps
    )
    film_callbacks = (
        []
        if not isinstance(callbacks, (list, tuple))
        else [
            callback
            for callback in callbacks
            if isinstance(callback, _FilmGradientActivationCallback)
        ]
    )
    configured_conditioning_variant = _configured_conditioning_variant(config)
    if configured_conditioning_variant == "film_adaln":
        if len(film_callbacks) != 1:
            raise RuntimeError(
                "FiLM pilot trainer must retain exactly one gradient-audit callback"
            )
        conditioning_gradient_audit = film_callbacks[0].completion_report()
    else:
        if film_callbacks:
            raise RuntimeError(
                "non-FiLM pilot unexpectedly retained a gradient-audit callback"
            )
        conditioning_gradient_audit = None
    if not isinstance(preflight_record, Mapping):
        raise RuntimeError("pilot completion lacks its runtime config record")
    completion_contract = _validate_pilot_completion_config(config)
    if preflight_record.get("completion_contract") != completion_contract:
        raise RuntimeError("pilot runtime config completion contract changed")

    expected_revision = _PILOT_CONTRACT["GENMOL_TRAIN_EXPECTED_SOURCE_REVISION"]
    source = _require_pilot_source_revision(expected_revision)
    if preflight_record.get("source") != source:
        raise RuntimeError("pilot source identity changed after runtime preflight")
    if (
        _canonical_json_sha256(_pilot_base_argv())
        != _PILOT_CONTRACT["GENMOL_TRAIN_EXPECTED_ARGV_SHA256"]
    ):
        raise RuntimeError("pilot argv changed before completion")

    runtime_snapshot, runtime_bytes = _stable_file_snapshot(
        _PILOT_CONTRACT["runtime_path"], capture_bytes=True
    )
    try:
        recorded_runtime = _strict_json_loads(runtime_bytes, label="runtime config")
    except RuntimeError as error:
        raise RuntimeError("pilot runtime config is not valid strict JSON") from error
    if recorded_runtime != dict(preflight_record):
        raise RuntimeError("pilot runtime config bytes disagree with runtime preflight")
    launch_manifest_evidence = _launch_manifest_evidence()
    if preflight_record.get("launch_manifest") != launch_manifest_evidence:
        raise RuntimeError("pilot launch manifest changed after runtime preflight")

    raw_tensors = _floating_tensor_finiteness(
        model.state_dict().items(), label="raw model state"
    )
    ema = getattr(model, "ema", None)
    shadows = None if ema is None else getattr(ema, "shadow_params", None)
    if not isinstance(shadows, (list, tuple)):
        raise RuntimeError("pilot model has no verifiable EMA shadow tensors")
    ema_tensors = _floating_tensor_finiteness(
        ((f"shadow_params[{index}]", tensor) for index, tensor in enumerate(shadows)),
        label="EMA state",
    )
    training_accounting = _pilot_training_accounting(
        config, trainer, model, train_dataloader
    )
    checkpoint_snapshot, checkpoint_audit = _audit_pilot_checkpoint(
        _PILOT_CONTRACT["final_checkpoint_path"],
        expected_steps=expected_steps,
        model=model,
        trainer=trainer,
        expected_resolved_config=preflight_record.get("resolved_training_config"),
        expected_config_sha256=_PILOT_CONTRACT["GENMOL_TRAIN_EXPECTED_CONFIG_SHA256"],
    )
    retained_warm_start = _verified_warm_start_report(
        config, startup_mode, warm_start_report
    )
    retained_screen_initialization = _validated_screen_initialization_state_audit(
        screen_initialization_state_audit,
        config=config,
        startup_mode=startup_mode,
        warm_start_report=warm_start_report,
    )

    startup_record = {
        "mode": startup_mode,
        "verified_mdlm_warm_start_report": retained_warm_start,
    }
    if training_rng_policy is not None:
        startup_record["training_rng_policy"] = training_rng_policy

    summary = {
        "schema_version": _TRAINING_SUMMARY_SCHEMA_VERSION,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": expected_revision,
        "source": source,
        "resolved_training_config_sha256": _PILOT_CONTRACT[
            "GENMOL_TRAIN_EXPECTED_CONFIG_SHA256"
        ],
        "training_argv_sha256": _PILOT_CONTRACT["GENMOL_TRAIN_EXPECTED_ARGV_SHA256"],
        "launch_manifest": launch_manifest_evidence,
        "runtime_config": {
            **runtime_snapshot,
            "schema_version": preflight_record.get("schema_version"),
            "record_sha256": _canonical_json_sha256(recorded_runtime),
        },
        "completion_contract": completion_contract,
        "observed_training_state": {
            "global_rank": global_rank,
            "global_step": observed_steps,
            "world_size": observed_world_size,
        },
        "training_accounting": training_accounting,
        "training_health": training_health,
        "conditioning_gradient_audit": conditioning_gradient_audit,
        "screen_initialization_state_audit": retained_screen_initialization,
        "final_checkpoint": {
            **checkpoint_snapshot,
            "semantic_audit": checkpoint_audit,
        },
        "tensor_finiteness": {
            "raw_model": raw_tensors,
            "ema": ema_tensors,
        },
        "startup": startup_record,
    }
    if summary["schema_version"] != _PILOT_CONTRACT["summary_schema_version"]:
        raise RuntimeError("pilot training summary schema disagrees with launch")
    if _launch_manifest_evidence() != summary["launch_manifest"]:
        raise RuntimeError("pilot launch manifest changed before summary publication")
    _atomic_write_json_exclusive(_PILOT_CONTRACT["summary_path"], summary)
    written_snapshot, written_bytes = _stable_file_snapshot(
        _PILOT_CONTRACT["summary_path"], capture_bytes=True
    )
    if json.loads(written_bytes) != summary:
        raise RuntimeError(
            "published pilot training summary failed its post-write check"
        )
    return {**summary, "artifact": written_snapshot}


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="base",
)
def train(config):
    pilot_preflight = _validate_and_record_pilot_config(config)
    pilot_callbacks = _pilot_callbacks(config)
    L.seed_everything(config.get("seed", 1), workers=True)
    wandb_logger = None
    if config.wandb.name is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config), **config.wandb
        )

    if config.training.get("use_bracket_safe"):
        config.model.vocab_size += 2

    model = GenMol(config)
    ckpt_path = get_last_checkpoint(config.callback.dirpath)
    init_from_mdlm = config.training.get("init_from_mdlm_checkpoint")
    startup_mode = checkpoint_startup_mode(ckpt_path, init_from_mdlm)
    warm_start_report = None
    if startup_mode == "warm_start":
        source_path = hydra.utils.to_absolute_path(init_from_mdlm)
        warm_start_report = model.initialize_from_mdlm_checkpoint(
            source_path,
            use_ema=bool(config.training.get("init_from_mdlm_ema", True)),
            expected_sha256=config.training.get("init_from_mdlm_checkpoint_sha256"),
        )
        print(
            'Initialized UDLM backbone from MDLM: '
            f"{warm_start_report['source_path']} "
            f"(weights={warm_start_report['weights']}, "
            f"parameters={warm_start_report['parameter_tensors']}, "
            f"sha256={warm_start_report['source_sha256']})"
        )
    elif startup_mode == "resume":
        print(
            f"Resuming {ckpt_path}; the configured MDLM initialization is "
            "a one-time provenance field and will not be reapplied."
        )

    screen_initialization_state_audit = _screen_initialization_state_audit(
        config, model, startup_mode, warm_start_report
    )
    training_rng_policy = _reseed_training_rng_after_model_initialization(
        config, startup_mode
    )

    train_dataloader = None
    if _PILOT_CONTRACT is None:
        train_dataloader = get_dataloader(config)
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=[hydra.utils.instantiate(config.callback), *pilot_callbacks],
        strategy=_training_strategy(),
        logger=wandb_logger,
        enable_progress_bar=True,
    )
    if _PILOT_CONTRACT is not None:
        # Lightning cannot inject a DistributedSampler into an iterable
        # dataset. Resolve the pilot rank after Trainer construction and split
        # the hosted stream explicitly; manual/released paths remain unchanged.
        streaming_rank, streaming_world_size = _pilot_streaming_partition(trainer)
        train_dataloader = get_dataloader(
            config,
            streaming_rank=streaming_rank,
            streaming_world_size=streaming_world_size,
        )
    trainer.fit(model, train_dataloader, ckpt_path=ckpt_path)
    _write_pilot_training_summary(
        config=config,
        trainer=trainer,
        model=model,
        preflight_record=pilot_preflight,
        startup_mode=startup_mode,
        warm_start_report=warm_start_report,
        screen_initialization_state_audit=screen_initialization_state_audit,
        train_dataloader=train_dataloader,
        training_rng_policy=training_rng_policy,
    )


if __name__ == "__main__":
    train()

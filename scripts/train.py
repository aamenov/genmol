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
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path


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
}
_TRAINING_SUMMARY_SCHEMA_VERSION = 1
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
    if len({runtime_path, summary_path, final_checkpoint_path}) != 3:
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
    if _canonical_json_sha256(_pilot_base_argv()) != present[
        "GENMOL_TRAIN_EXPECTED_ARGV_SHA256"
    ]:
        raise RuntimeError("pilot child argv disagrees with the launch manifest")
    return {
        **present,
        **parsed_integers,
        "runtime_path": runtime_path,
        "summary_path": summary_path,
        "final_checkpoint_path": final_checkpoint_path,
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
        if (
            not stat.S_ISREG(before_descriptor.st_mode)
            or _stat_identity(before_descriptor) != _stat_identity(before_path)
        ):
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

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import hydra
import lightning as L
import omegaconf
import torch
from genmol.model import GenMol
from genmol.utils.utils_data import get_dataloader, get_last_checkpoint

omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)


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


def _pilot_callbacks(config):
    if _PILOT_CONTRACT is None:
        return []
    if config.training.get('pilot_fail_on_nonfinite_loss') is not True:
        raise RuntimeError(
            "pilot config must enable fail-fast non-finite loss validation"
        )
    if config.trainer.get('detect_anomaly') is not True:
        raise RuntimeError("pilot config must enable backward anomaly detection")
    return [_PilotFiniteLossCallback()]


def _exact_positive_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"{label} must be a positive integer")
    return value


def _validate_pilot_completion_config(config):
    if _PILOT_CONTRACT is None:
        return None
    expected_steps = _PILOT_CONTRACT["expected_max_steps"]
    expected_world_size = _PILOT_CONTRACT["expected_world_size"]
    configured_steps = _exact_positive_integer(
        config.trainer.get('max_steps'), "pilot trainer.max_steps"
    )
    devices = _exact_positive_integer(
        config.trainer.get('devices'), "pilot trainer.devices"
    )
    nodes = _exact_positive_integer(
        config.trainer.get('num_nodes'), "pilot trainer.num_nodes"
    )
    if configured_steps != expected_steps:
        raise RuntimeError("pilot max step disagrees with the launch contract")
    if devices * nodes != expected_world_size:
        raise RuntimeError("pilot world size disagrees with the launch contract")
    callback_dir = Path(os.path.abspath(os.fspath(config.callback.get('dirpath'))))
    if callback_dir / f"{expected_steps}.ckpt" != _PILOT_CONTRACT[
        "final_checkpoint_path"
    ]:
        raise RuntimeError("pilot final checkpoint path disagrees with its config")
    if (
        config.callback.get('filename') != '{step}'
        or config.callback.get('every_n_train_steps') != expected_steps
        or config.callback.get('save_top_k') != -1
    ):
        raise RuntimeError("pilot checkpoint callback does not guarantee the final step")
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
        return 'resume'
    if initialization_checkpoint:
        return 'warm_start'
    return 'scratch'


def _validate_and_record_pilot_config(config):
    if _PILOT_CONTRACT is None:
        return None
    completion_contract = _validate_pilot_completion_config(config)
    _pilot_callbacks(config)
    expected_revision = _PILOT_CONTRACT[
        "GENMOL_TRAIN_EXPECTED_SOURCE_REVISION"
    ]
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
    if os.environ["PYTHONHASHSEED"] != str(config.get('seed', 1)):
        raise RuntimeError("PYTHONHASHSEED disagrees with the resolved training seed")
    record = {
        "schema_version": 1,
        "status": "preflight_completed",
        "source_revision": expected_revision,
        "source": source,
        "training_argv": _pilot_base_argv(),
        "observed_training_argv": list(sys.argv),
        "training_argv_sha256": _PILOT_CONTRACT[
            "GENMOL_TRAIN_EXPECTED_ARGV_SHA256"
        ],
        "resolved_training_config": resolved,
        "resolved_training_config_sha256": digest,
        "completion_contract": completion_contract,
        "python_environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith("PYTHON")
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
            raise RuntimeError("pilot runtime config record already exists with other data")
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


def _nested_named_tensors(value, *, prefix):
    """Yield every tensor in a checkpoint with a deterministic diagnostic name."""

    if isinstance(value, torch.Tensor):
        yield prefix, value
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _nested_named_tensors(item, prefix=f"{prefix}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _nested_named_tensors(item, prefix=f"{prefix}[{index}]")


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


def _audit_pilot_checkpoint(path, *, expected_steps, model):
    """Load and semantically audit the exact stable checkpoint bytes."""

    snapshot_before, _unused = _stable_file_snapshot(path)
    if snapshot_before["size_bytes"] <= 0:
        raise RuntimeError("pilot final checkpoint is empty")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise RuntimeError("pilot final checkpoint cannot be deserialized") from error
    snapshot_after, _unused = _stable_file_snapshot(path)
    if snapshot_after != snapshot_before:
        raise RuntimeError("pilot final checkpoint changed while it was audited")
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("pilot final checkpoint must deserialize to a mapping")
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
    optimizer_states = checkpoint.get("optimizer_states")
    if not isinstance(optimizer_states, list) or not optimizer_states:
        raise RuntimeError("pilot checkpoint has no optimizer state")
    optimizer_tensors = _floating_tensor_finiteness(
        _nested_named_tensors(optimizer_states, prefix="optimizer_states"),
        label="serialized checkpoint optimizer state",
    )
    all_tensors = _floating_tensor_finiteness(
        _nested_named_tensors(checkpoint, prefix="checkpoint"),
        label="serialized checkpoint",
    )
    prior_validator = getattr(model, "_validate_udlm_prior_checkpoint", None)
    if not callable(prior_validator):
        raise RuntimeError("pilot model has no UDLM checkpoint identity validator")
    prior_validator(checkpoint)
    live_match = _validate_checkpoint_matches_live_model(checkpoint_state, model)
    live_ema_match = _validate_checkpoint_ema_matches_live(
        checkpoint_shadows, model
    )
    return snapshot_before, {
        "deserialized": True,
        "global_step": checkpoint_step,
        "raw_model": raw_tensors,
        "ema": ema_tensors,
        "optimizer": optimizer_tensors,
        "all_checkpoint_tensors": all_tensors,
        "udlm_process_identity_verified": True,
        "live_model_match": live_match,
        "live_ema_match": live_ema_match,
    }


def _verified_warm_start_report(config, startup_mode, warm_start_report):
    if startup_mode != 'warm_start':
        if warm_start_report is not None:
            raise RuntimeError("non-warm-start pilot unexpectedly retained a warm-start report")
        return None
    if not isinstance(warm_start_report, Mapping):
        raise RuntimeError("warm-start pilot did not retain its verified MDLM report")
    report = dict(warm_start_report)
    expected_sha256 = config.training.get('init_from_mdlm_checkpoint_sha256')
    if (
        not isinstance(expected_sha256, str)
        or report.get('source_sha256') != expected_sha256
        or report.get('expected_source_sha256') != expected_sha256
        or report.get('byte_identity_verified_before_and_after_load') is not True
    ):
        raise RuntimeError("warm-start report disagrees with the verified MDLM source")
    return report


def _write_pilot_training_summary(
    *,
    config,
    trainer,
    model,
    preflight_record,
    startup_mode,
    warm_start_report,
):
    """Publish the sole rank-zero certificate that a pilot completed."""

    if _PILOT_CONTRACT is None:
        return None
    if not bool(getattr(trainer, 'is_global_zero', False)):
        return None
    global_rank = getattr(trainer, 'global_rank', 0)
    if isinstance(global_rank, bool) or not isinstance(global_rank, int) or global_rank != 0:
        raise RuntimeError("pilot global-zero process reports an invalid global rank")
    expected_steps = _PILOT_CONTRACT["expected_max_steps"]
    expected_world_size = _PILOT_CONTRACT["expected_world_size"]
    observed_steps = getattr(trainer, 'global_step', None)
    observed_world_size = getattr(trainer, 'world_size', None)
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
    if not isinstance(preflight_record, Mapping):
        raise RuntimeError("pilot completion lacks its runtime config record")
    completion_contract = _validate_pilot_completion_config(config)
    if preflight_record.get("completion_contract") != completion_contract:
        raise RuntimeError("pilot runtime config completion contract changed")

    expected_revision = _PILOT_CONTRACT[
        "GENMOL_TRAIN_EXPECTED_SOURCE_REVISION"
    ]
    source = _require_pilot_source_revision(expected_revision)
    if preflight_record.get("source") != source:
        raise RuntimeError("pilot source identity changed after runtime preflight")
    if _canonical_json_sha256(_pilot_base_argv()) != _PILOT_CONTRACT[
        "GENMOL_TRAIN_EXPECTED_ARGV_SHA256"
    ]:
        raise RuntimeError("pilot argv changed before completion")

    runtime_snapshot, runtime_bytes = _stable_file_snapshot(
        _PILOT_CONTRACT["runtime_path"], capture_bytes=True
    )
    try:
        recorded_runtime = json.loads(runtime_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("pilot runtime config is not valid UTF-8 JSON") from error
    if recorded_runtime != dict(preflight_record):
        raise RuntimeError("pilot runtime config bytes disagree with runtime preflight")

    raw_tensors = _floating_tensor_finiteness(
        model.state_dict().items(), label="raw model state"
    )
    ema = getattr(model, 'ema', None)
    shadows = None if ema is None else getattr(ema, 'shadow_params', None)
    if not isinstance(shadows, (list, tuple)):
        raise RuntimeError("pilot model has no verifiable EMA shadow tensors")
    ema_tensors = _floating_tensor_finiteness(
        ((f"shadow_params[{index}]", tensor) for index, tensor in enumerate(shadows)),
        label="EMA state",
    )
    checkpoint_snapshot, checkpoint_audit = _audit_pilot_checkpoint(
        _PILOT_CONTRACT["final_checkpoint_path"],
        expected_steps=expected_steps,
        model=model,
    )
    retained_warm_start = _verified_warm_start_report(
        config, startup_mode, warm_start_report
    )

    summary = {
        "schema_version": _TRAINING_SUMMARY_SCHEMA_VERSION,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": expected_revision,
        "source": source,
        "resolved_training_config_sha256": _PILOT_CONTRACT[
            "GENMOL_TRAIN_EXPECTED_CONFIG_SHA256"
        ],
        "training_argv_sha256": _PILOT_CONTRACT[
            "GENMOL_TRAIN_EXPECTED_ARGV_SHA256"
        ],
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
        "training_health": training_health,
        "final_checkpoint": {
            **checkpoint_snapshot,
            "semantic_audit": checkpoint_audit,
        },
        "tensor_finiteness": {
            "raw_model": raw_tensors,
            "ema": ema_tensors,
        },
        "startup": {
            "mode": startup_mode,
            "verified_mdlm_warm_start_report": retained_warm_start,
        },
    }
    if summary["schema_version"] != _PILOT_CONTRACT["summary_schema_version"]:
        raise RuntimeError("pilot training summary schema disagrees with launch")
    _atomic_write_json_exclusive(_PILOT_CONTRACT["summary_path"], summary)
    written_snapshot, written_bytes = _stable_file_snapshot(
        _PILOT_CONTRACT["summary_path"], capture_bytes=True
    )
    if json.loads(written_bytes) != summary:
        raise RuntimeError("published pilot training summary failed its post-write check")
    return {**summary, "artifact": written_snapshot}


@hydra.main(version_base=None,
    config_path="../configs",
    config_name="base",
)
def train(config):
    pilot_preflight = _validate_and_record_pilot_config(config)
    pilot_callbacks = _pilot_callbacks(config)
    L.seed_everything(config.get('seed', 1), workers=True)
    wandb_logger = None
    if config.wandb.name is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config),
            **config.wandb)
    
    if config.training.get('use_bracket_safe'):
        config.model.vocab_size += 2

    model = GenMol(config)
    ckpt_path = get_last_checkpoint(config.callback.dirpath)
    init_from_mdlm = config.training.get('init_from_mdlm_checkpoint')
    startup_mode = checkpoint_startup_mode(ckpt_path, init_from_mdlm)
    warm_start_report = None
    if startup_mode == 'warm_start':
        source_path = hydra.utils.to_absolute_path(init_from_mdlm)
        warm_start_report = model.initialize_from_mdlm_checkpoint(
            source_path,
            use_ema=bool(config.training.get('init_from_mdlm_ema', True)),
            expected_sha256=config.training.get(
                'init_from_mdlm_checkpoint_sha256'
            ),
        )
        print(
            'Initialized UDLM backbone from MDLM: '
            f"{warm_start_report['source_path']} "
            f"(weights={warm_start_report['weights']}, "
            f"parameters={warm_start_report['parameter_tensors']}, "
            f"sha256={warm_start_report['source_sha256']})"
        )
    elif startup_mode == 'resume':
        print(
            f'Resuming {ckpt_path}; the configured MDLM initialization is '
            'a one-time provenance field and will not be reapplied.'
        )
    
    train_dataloader = get_dataloader(config)
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=[hydra.utils.instantiate(config.callback), *pilot_callbacks],
        strategy=hydra.utils.instantiate({'_target_': 'lightning.pytorch.strategies.DDPStrategy',
                                          'find_unused_parameters': False}),
        logger=wandb_logger,
        enable_progress_bar=True)
    trainer.fit(model, train_dataloader, ckpt_path=ckpt_path)
    _write_pilot_training_summary(
        config=config,
        trainer=trainer,
        model=model,
        preflight_record=pilot_preflight,
        startup_mode=startup_mode,
        warm_start_report=warm_start_report,
    )
    

if __name__ == '__main__':
    train()

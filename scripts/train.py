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
import subprocess
import sys
from pathlib import Path


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_PILOT_ENVIRONMENT_KEYS = {
    "GENMOL_TRAIN_EXPECTED_SOURCE_REVISION",
    "GENMOL_TRAIN_EXPECTED_CONFIG_SHA256",
    "GENMOL_TRAIN_EXPECTED_ARGV_SHA256",
    "GENMOL_TRAIN_RUNTIME_CONFIG_PATH",
}
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
    runtime_path = Path(present["GENMOL_TRAIN_RUNTIME_CONFIG_PATH"]).resolve()
    if (
        runtime_path == _REPOSITORY_ROOT
        or _REPOSITORY_ROOT not in runtime_path.parents
        or runtime_path.suffix != ".json"
    ):
        raise RuntimeError("pilot runtime config path must be an in-repository JSON file")
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
    return {**present, "runtime_path": runtime_path}


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


@hydra.main(version_base=None,
    config_path="../configs",
    config_name="base",
)
def train(config):
    _validate_and_record_pilot_config(config)
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
    if startup_mode == 'warm_start':
        source_path = hydra.utils.to_absolute_path(init_from_mdlm)
        report = model.initialize_from_mdlm_checkpoint(
            source_path,
            use_ema=bool(config.training.get('init_from_mdlm_ema', True)),
            expected_sha256=config.training.get(
                'init_from_mdlm_checkpoint_sha256'
            ),
        )
        print(
            'Initialized UDLM backbone from MDLM: '
            f"{report['source_path']} (weights={report['weights']}, "
            f"parameters={report['parameter_tensors']}, "
            f"sha256={report['source_sha256']})"
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
        callbacks=[hydra.utils.instantiate(config.callback)],
        strategy=hydra.utils.instantiate({'_target_': 'lightning.pytorch.strategies.DDPStrategy',
                                          'find_unused_parameters': False}),
        logger=wandb_logger,
        enable_progress_bar=True)
    trainer.fit(model, train_dataloader, ckpt_path=ckpt_path)
    

if __name__ == '__main__':
    train()

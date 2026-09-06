import json
import sys

import pytest
from omegaconf import OmegaConf

from scripts import train as train_entrypoint
from scripts.train import checkpoint_startup_mode


def test_existing_training_checkpoint_takes_precedence_over_warm_start():
    assert checkpoint_startup_mode("step-100.ckpt", "mdlm.ckpt") == "resume"


def test_warm_start_is_used_only_without_a_resume_checkpoint():
    assert checkpoint_startup_mode(None, "mdlm.ckpt") == "warm_start"
    assert checkpoint_startup_mode(None, None) == "scratch"


def test_manual_training_preserves_absent_pilot_contract(monkeypatch):
    for key in train_entrypoint._PILOT_ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)

    assert train_entrypoint._pilot_environment_contract() is None


def test_partial_pilot_environment_is_rejected(monkeypatch):
    for key in train_entrypoint._PILOT_ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GENMOL_TRAIN_EXPECTED_SOURCE_REVISION", "a" * 40)

    with pytest.raises(RuntimeError, match="partial or unexpected"):
        train_entrypoint._pilot_environment_contract()


def test_ddp_child_accepts_only_lightning_exact_hydra_suffix(monkeypatch):
    run_dir = "/repo/output/udlm/pilot/hydra"
    base = [
        "/repo/scripts/train.py",
        "--config-name",
        "udlm",
        f"hydra.run.dir={run_dir}",
    ]
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            *base,
            f'hydra.run.dir="{run_dir}"',
            "hydra.job.name=train_ddp_process_1",
            "hydra.output_subdir=null",
        ],
    )

    assert train_entrypoint._pilot_base_argv() == base

    sys.argv[-2] = "hydra.job.name=unreviewed"
    with pytest.raises(RuntimeError, match="unexpected Hydra argv"):
        train_entrypoint._pilot_base_argv()


def test_pilot_config_digest_is_checked_and_recorded_once(tmp_path, monkeypatch):
    config = OmegaConf.create({"seed": 7, "trainer": {"devices": 2}})
    resolved = OmegaConf.to_container(config, resolve=True, enum_to_str=True)
    runtime_path = tmp_path / "runtime_config.json"
    argv = ["/repo/scripts/train.py", "seed=7"]
    contract = {
        "GENMOL_TRAIN_EXPECTED_SOURCE_REVISION": "a" * 40,
        "GENMOL_TRAIN_EXPECTED_CONFIG_SHA256": (
            train_entrypoint._canonical_json_sha256(resolved)
        ),
        "GENMOL_TRAIN_EXPECTED_ARGV_SHA256": (
            train_entrypoint._canonical_json_sha256(argv)
        ),
        "GENMOL_TRAIN_RUNTIME_CONFIG_PATH": str(runtime_path),
        "runtime_path": runtime_path,
    }
    monkeypatch.setattr(train_entrypoint, "_PILOT_CONTRACT", contract)
    monkeypatch.setattr(
        train_entrypoint,
        "_require_pilot_source_revision",
        lambda revision: {"head": revision, "upstream": revision},
    )
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("PYTHONHASHSEED", "7")
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    first = train_entrypoint._validate_and_record_pilot_config(config)
    second = train_entrypoint._validate_and_record_pilot_config(config)

    assert first == second
    assert json.loads(runtime_path.read_text(encoding="utf-8")) == first
    assert (
        first["resolved_training_config_sha256"]
        == contract["GENMOL_TRAIN_EXPECTED_CONFIG_SHA256"]
    )

    contract["GENMOL_TRAIN_EXPECTED_CONFIG_SHA256"] = "b" * 64
    with pytest.raises(RuntimeError, match="launch-pinned config digest"):
        train_entrypoint._validate_and_record_pilot_config(config)

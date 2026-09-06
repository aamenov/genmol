import json
import sys
from types import SimpleNamespace

import pytest
import torch
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
    checkpoint_dir = tmp_path / "checkpoints"
    config = OmegaConf.create(
        {
            "seed": 7,
            "trainer": {
                "devices": 2,
                "num_nodes": 1,
                "max_steps": 10,
                "detect_anomaly": True,
            },
            "callback": {
                "dirpath": str(checkpoint_dir),
                "filename": "{step}",
                "every_n_train_steps": 10,
                "save_top_k": -1,
            },
            "training": {"pilot_fail_on_nonfinite_loss": True},
        }
    )
    resolved = OmegaConf.to_container(config, resolve=True, enum_to_str=True)
    runtime_path = tmp_path / "runtime_config.json"
    summary_path = tmp_path / "training_summary.json"
    final_checkpoint_path = checkpoint_dir / "10.ckpt"
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
        "expected_max_steps": 10,
        "expected_world_size": 2,
        "summary_schema_version": 1,
        "runtime_path": runtime_path,
        "summary_path": summary_path,
        "final_checkpoint_path": final_checkpoint_path,
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


def _completion_fixture(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    runtime_path = tmp_path / "runtime_config.json"
    summary_path = tmp_path / "training_summary.json"
    checkpoint_path = checkpoint_dir / "10.ckpt"
    checkpoint_state = {"weight": torch.tensor([1.0, -2.0])}
    checkpoint_ema = [torch.tensor([0.5, 3.0])]
    torch.save(
        {
            "global_step": 10,
            "state_dict": checkpoint_state,
            "ema": {"shadow_params": checkpoint_ema},
            "optimizer_states": [
                {
                    "state": {
                        0: {
                            "step": torch.tensor(10.0),
                            "exp_avg": torch.tensor([0.1, -0.2]),
                            "exp_avg_sq": torch.tensor([0.01, 0.04]),
                        }
                    }
                }
            ],
        },
        checkpoint_path,
    )
    argv = ["/repo/scripts/train.py", "seed=7"]
    config = OmegaConf.create(
        {
            "seed": 7,
            "trainer": {
                "devices": 2,
                "num_nodes": 1,
                "max_steps": 10,
                "detect_anomaly": True,
            },
            "callback": {
                "dirpath": str(checkpoint_dir),
                "filename": "{step}",
                "every_n_train_steps": 10,
                "save_top_k": -1,
            },
            "training": {
                "pilot_fail_on_nonfinite_loss": True,
                "init_from_mdlm_checkpoint": "/project/mdlm.ckpt",
                "init_from_mdlm_checkpoint_sha256": "c" * 64,
            },
        }
    )
    resolved = OmegaConf.to_container(config, resolve=True, enum_to_str=True)
    contract = {
        "GENMOL_TRAIN_EXPECTED_SOURCE_REVISION": "a" * 40,
        "GENMOL_TRAIN_EXPECTED_CONFIG_SHA256": (
            train_entrypoint._canonical_json_sha256(resolved)
        ),
        "GENMOL_TRAIN_EXPECTED_ARGV_SHA256": (
            train_entrypoint._canonical_json_sha256(argv)
        ),
        "expected_max_steps": 10,
        "expected_world_size": 2,
        "summary_schema_version": train_entrypoint._TRAINING_SUMMARY_SCHEMA_VERSION,
        "runtime_path": runtime_path,
        "summary_path": summary_path,
        "final_checkpoint_path": checkpoint_path,
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
    preflight = train_entrypoint._validate_and_record_pilot_config(config)
    health_callback = train_entrypoint._PilotFiniteLossCallback()
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    health_module = SimpleNamespace(named_parameters=lambda: [("weight", parameter)])
    for _index in range(10):
        health_callback.on_before_backward(None, None, torch.tensor(1.25))
        parameter.grad = torch.tensor([0.5])
        health_callback.on_before_optimizer_step(None, health_module, None)
    trainer = SimpleNamespace(
        is_global_zero=True,
        global_rank=0,
        global_step=10,
        world_size=2,
        callbacks=[health_callback],
    )
    model = SimpleNamespace(
        state_dict=lambda: {"weight": torch.tensor([1.0, -2.0])},
        ema=SimpleNamespace(shadow_params=[torch.tensor([0.5, 3.0])]),
        _validate_udlm_prior_checkpoint=lambda checkpoint: None,
    )
    warm_start = {
        "source_path": "/project/mdlm.ckpt",
        "source_resolved_path": "/project/mdlm.ckpt",
        "source_sha256": "c" * 64,
        "source_size_bytes": 123,
        "expected_source_sha256": "c" * 64,
        "byte_identity_verified_before_and_after_load": True,
        "weights": "ema",
        "parameter_tensors": 2,
    }
    return config, contract, preflight, trainer, model, warm_start


def test_pilot_nonfinite_loss_fails_before_backward():
    callback = train_entrypoint._PilotFiniteLossCallback()
    callback.on_before_backward(None, None, torch.tensor(1.25))

    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(FloatingPointError, match="before backward"):
            callback.on_before_backward(None, None, torch.tensor(value))


def test_pilot_health_callback_rejects_nonfinite_or_zero_gradients():
    callback = train_entrypoint._PilotFiniteLossCallback()
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    module = SimpleNamespace(named_parameters=lambda: [("weight", parameter)])

    parameter.grad = torch.tensor([float("nan")])
    with pytest.raises(FloatingPointError, match="gradient is non-finite"):
        callback.on_before_optimizer_step(None, module, None)
    parameter.grad = torch.tensor([0.0])
    with pytest.raises(RuntimeError, match="only zero gradients"):
        callback.on_before_optimizer_step(None, module, None)
    parameter.grad = None
    with pytest.raises(RuntimeError, match="no gradients"):
        callback.on_before_optimizer_step(None, module, None)


def test_pilot_completion_summary_binds_and_verifies_every_artifact(
    tmp_path, monkeypatch
):
    config, contract, preflight, trainer, model, warm_start = _completion_fixture(
        tmp_path, monkeypatch
    )

    result = train_entrypoint._write_pilot_training_summary(
        config=config,
        trainer=trainer,
        model=model,
        preflight_record=preflight,
        startup_mode="warm_start",
        warm_start_report=warm_start,
    )
    summary = json.loads(contract["summary_path"].read_text(encoding="utf-8"))

    assert result["status"] == "completed"
    assert (
        summary["schema_version"] == train_entrypoint._TRAINING_SUMMARY_SCHEMA_VERSION
    )
    assert summary["source_revision"] == "a" * 40
    assert (
        summary["resolved_training_config_sha256"]
        == contract["GENMOL_TRAIN_EXPECTED_CONFIG_SHA256"]
    )
    assert (
        summary["training_argv_sha256"] == contract["GENMOL_TRAIN_EXPECTED_ARGV_SHA256"]
    )
    assert summary["runtime_config"]["sha256"]
    assert summary["final_checkpoint"]["sha256"]
    checkpoint_audit = summary["final_checkpoint"]["semantic_audit"]
    assert checkpoint_audit["deserialized"] is True
    assert checkpoint_audit["global_step"] == 10
    assert checkpoint_audit["raw_model"]["all_finite"] is True
    assert checkpoint_audit["ema"]["all_finite"] is True
    assert checkpoint_audit["optimizer"]["all_finite"] is True
    assert checkpoint_audit["all_checkpoint_tensors"]["all_finite"] is True
    assert checkpoint_audit["udlm_process_identity_verified"] is True
    assert checkpoint_audit["live_model_match"]["exact_tensor_values"] is True
    assert checkpoint_audit["live_ema_match"]["exact_tensor_values"] is True
    assert summary["observed_training_state"] == {
        "global_rank": 0,
        "global_step": 10,
        "world_size": 2,
    }
    assert summary["training_health"]["loss_checks"] == 10
    assert summary["training_health"]["optimizer_step_checks"] == 10
    assert (
        summary["training_health"]["every_optimizer_step_had_a_nonzero_gradient"]
        is True
    )
    assert summary["tensor_finiteness"]["raw_model"]["all_finite"] is True
    assert summary["tensor_finiteness"]["ema"]["all_finite"] is True
    assert (
        summary["startup"]["verified_mdlm_warm_start_report"]["source_sha256"]
        == "c" * 64
    )
    with pytest.raises(FileExistsError, match="refusing to replace"):
        train_entrypoint._write_pilot_training_summary(
            config=config,
            trainer=trainer,
            model=model,
            preflight_record=preflight,
            startup_mode="warm_start",
            warm_start_report=warm_start,
        )


@pytest.mark.parametrize("state", ["raw", "ema"])
def test_pilot_completion_rejects_nonfinite_model_or_ema_state(
    tmp_path, monkeypatch, state
):
    config, _contract, preflight, trainer, model, warm_start = _completion_fixture(
        tmp_path, monkeypatch
    )
    if state == "raw":
        model.state_dict = lambda: {"weight": torch.tensor([float("nan")])}
    else:
        model.ema.shadow_params = [torch.tensor([float("inf")])]

    with pytest.raises(FloatingPointError, match="non-finite"):
        train_entrypoint._write_pilot_training_summary(
            config=config,
            trainer=trainer,
            model=model,
            preflight_record=preflight,
            startup_mode="warm_start",
            warm_start_report=warm_start,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("global_step", 9, "stopped at global step"),
        ("world_size", 1, "runtime world size"),
        ("global_rank", 1, "invalid global rank"),
    ],
)
def test_pilot_completion_rejects_wrong_step_or_world_size(
    tmp_path, monkeypatch, field, value, message
):
    config, _contract, preflight, trainer, model, warm_start = _completion_fixture(
        tmp_path, monkeypatch
    )
    setattr(trainer, field, value)

    with pytest.raises(RuntimeError, match=message):
        train_entrypoint._write_pilot_training_summary(
            config=config,
            trainer=trainer,
            model=model,
            preflight_record=preflight,
            startup_mode="warm_start",
            warm_start_report=warm_start,
        )


def test_pilot_completion_rejects_missing_or_nonregular_checkpoint(
    tmp_path, monkeypatch
):
    config, contract, preflight, trainer, model, warm_start = _completion_fixture(
        tmp_path, monkeypatch
    )
    contract["final_checkpoint_path"].unlink()
    replacement = tmp_path / "replacement.ckpt"
    replacement.write_bytes(b"wrong checkpoint")
    contract["final_checkpoint_path"].symlink_to(replacement)

    with pytest.raises(RuntimeError, match="not a regular file"):
        train_entrypoint._write_pilot_training_summary(
            config=config,
            trainer=trainer,
            model=model,
            preflight_record=preflight,
            startup_mode="warm_start",
            warm_start_report=warm_start,
        )


def test_pilot_completion_rejects_undecodable_checkpoint(tmp_path, monkeypatch):
    config, contract, preflight, trainer, model, warm_start = _completion_fixture(
        tmp_path, monkeypatch
    )
    contract["final_checkpoint_path"].write_bytes(b"not a torch checkpoint")

    with pytest.raises(RuntimeError, match="cannot be deserialized"):
        train_entrypoint._write_pilot_training_summary(
            config=config,
            trainer=trainer,
            model=model,
            preflight_record=preflight,
            startup_mode="warm_start",
            warm_start_report=warm_start,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("global_step", 9, "checkpoint global_step"),
        ("state_dict", {"weight": torch.tensor([float("nan"), -2.0])}, "non-finite"),
        (
            "ema",
            {"shadow_params": [torch.tensor([float("inf")])]},
            "non-finite",
        ),
        (
            "ema",
            {"shadow_params": [torch.tensor([0.5, 2.5])]},
            "EMA tensor disagrees",
        ),
        ("optimizer_states", [], "no optimizer state"),
    ],
)
def test_pilot_completion_rejects_invalid_serialized_checkpoint(
    tmp_path, monkeypatch, field, value, message
):
    config, contract, preflight, trainer, model, warm_start = _completion_fixture(
        tmp_path, monkeypatch
    )
    checkpoint = torch.load(
        contract["final_checkpoint_path"], map_location="cpu", weights_only=False
    )
    checkpoint[field] = value
    torch.save(checkpoint, contract["final_checkpoint_path"])

    with pytest.raises((RuntimeError, FloatingPointError), match=message):
        train_entrypoint._write_pilot_training_summary(
            config=config,
            trainer=trainer,
            model=model,
            preflight_record=preflight,
            startup_mode="warm_start",
            warm_start_report=warm_start,
        )


def test_nonpilot_completion_and_callbacks_are_no_ops(tmp_path, monkeypatch):
    monkeypatch.setattr(train_entrypoint, "_PILOT_CONTRACT", None)

    assert train_entrypoint._pilot_callbacks(OmegaConf.create({})) == []
    assert (
        train_entrypoint._write_pilot_training_summary(
            config=None,
            trainer=None,
            model=None,
            preflight_record=None,
            startup_mode="scratch",
            warm_start_report=None,
        )
        is None
    )
    assert not (tmp_path / "training_summary.json").exists()

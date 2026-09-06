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


def test_pilot_streaming_partition_accepts_parent_and_lightning_child(
    monkeypatch,
):
    monkeypatch.setattr(
        train_entrypoint,
        "_PILOT_CONTRACT",
        {"expected_world_size": 2},
    )
    for key in ("LOCAL_RANK", "WORLD_SIZE", "NODE_RANK"):
        monkeypatch.delenv(key, raising=False)
    parent = SimpleNamespace(global_rank=0, world_size=2, num_nodes=1)
    assert train_entrypoint._pilot_streaming_partition(parent) == (0, 2)

    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("NODE_RANK", "0")
    child = SimpleNamespace(global_rank=1, world_size=2, num_nodes=1)
    assert train_entrypoint._pilot_streaming_partition(child) == (1, 2)


@pytest.mark.parametrize(
    ("trainer", "environment", "message"),
    [
        (
            SimpleNamespace(global_rank=1, world_size=2, num_nodes=1),
            {},
            "nonzero rank lacks",
        ),
        (
            SimpleNamespace(global_rank=0, world_size=2, num_nodes=1),
            {"LOCAL_RANK": "0"},
            "partial or inconsistent",
        ),
        (
            SimpleNamespace(global_rank=1, world_size=2, num_nodes=1),
            {"LOCAL_RANK": "01", "WORLD_SIZE": "2", "NODE_RANK": "0"},
            "partial or inconsistent",
        ),
        (
            SimpleNamespace(global_rank=0, world_size=1, num_nodes=2),
            {},
            "exactly one node",
        ),
        (
            SimpleNamespace(global_rank=0, world_size=1, num_nodes=1),
            {},
            "world size disagrees",
        ),
    ],
)
def test_pilot_streaming_partition_rejects_ambiguous_identity(
    monkeypatch, trainer, environment, message
):
    monkeypatch.setattr(
        train_entrypoint,
        "_PILOT_CONTRACT",
        {"expected_world_size": 2},
    )
    for key in ("LOCAL_RANK", "WORLD_SIZE", "NODE_RANK"):
        monkeypatch.delenv(key, raising=False)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)

    with pytest.raises(RuntimeError, match=message):
        train_entrypoint._pilot_streaming_partition(trainer)


def test_pilot_strategy_ignores_inherited_scheduler_environment(monkeypatch):
    monkeypatch.setattr(
        train_entrypoint,
        "_PILOT_CONTRACT",
        {"expected_world_size": 2},
    )
    monkeypatch.setenv("SLURM_NTASKS", "2")
    monkeypatch.setenv("SLURM_JOB_NAME", "hostile-allocation")
    monkeypatch.setenv("SLURM_NODEID", "0")
    monkeypatch.setenv("SLURM_LOCALID", "0")
    monkeypatch.setenv("SLURM_PROCID", "0")

    strategy = train_entrypoint._training_strategy()
    trainer = train_entrypoint.L.Trainer(
        accelerator="cpu",
        devices=2,
        strategy=strategy,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    assert isinstance(
        trainer.strategy.cluster_environment,
        train_entrypoint.L.fabric.plugins.environments.LightningEnvironment,
    )
    assert trainer.strategy.cluster_environment.creates_processes_externally is False


def test_manual_strategy_retains_lightning_environment_autodetection(monkeypatch):
    monkeypatch.setattr(train_entrypoint, "_PILOT_CONTRACT", None)

    strategy = train_entrypoint._training_strategy()

    assert strategy.cluster_environment is None


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
            "data": "safe",
            "seed": 7,
            "loader": {"global_batch_size": 16, "batch_size": 2},
            "trainer": {
                "devices": 2,
                "num_nodes": 1,
                "max_steps": 10,
                "accumulate_grad_batches": 4,
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
        "summary_schema_version": 2,
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
            "ema": {
                "shadow_params": checkpoint_ema,
                "decay": 0.9999,
                "num_updates": 10,
            },
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
            "data": "safe",
            "seed": 7,
            "loader": {"global_batch_size": 16, "batch_size": 2},
            "trainer": {
                "devices": 2,
                "num_nodes": 1,
                "max_steps": 10,
                "accumulate_grad_batches": 4,
                "detect_anomaly": True,
            },
            "callback": {
                "dirpath": str(checkpoint_dir),
                "filename": "{step}",
                "every_n_train_steps": 10,
                "save_top_k": -1,
            },
            "training": {
                "ema": 0.9999,
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
        num_nodes=1,
        max_steps=10,
        accumulate_grad_batches=4,
        train_dataloader=SimpleNamespace(batch_size=2),
        callbacks=[health_callback],
    )
    base_parameter = torch.nn.Parameter(torch.ones(3))
    conditioner_parameter = torch.nn.Parameter(torch.ones(2))
    backbone = SimpleNamespace(
        named_parameters=lambda: [
            ("base_weight", base_parameter),
            ("time_conditioner.weight", conditioner_parameter),
        ]
    )
    model = SimpleNamespace(
        backbone=backbone,
        named_parameters=lambda: [
            ("backbone.base_weight", base_parameter),
            ("backbone.time_conditioner.weight", conditioner_parameter),
        ],
        state_dict=lambda: {"weight": torch.tensor([1.0, -2.0])},
        ema=SimpleNamespace(
            shadow_params=[torch.tensor([0.5, 3.0])],
            decay=0.9999,
            num_updates=10,
        ),
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
    assert checkpoint_audit["ema_metadata"] == {
        "shadow_parameter_count": 1,
        "decay": 0.9999,
        "num_updates": 10,
    }
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
    assert summary["training_accounting"] == {
        "training_seed": 7,
        "optimizer_updates": 10,
        "world_size": 2,
        "micro_batch_size_per_rank": 2,
        "accumulate_grad_batches": 4,
        "effective_global_examples_per_optimizer_step": 16,
        "total_requested_example_exposures": 160,
        "hosted_stream_rank_partition_policy": (
            "huggingface_split_dataset_by_node_disjoint_rank_streams"
        ),
        "trainable_parameter_counts": {
            "base_backbone": 3,
            "time_conditioner": 2,
            "total": 5,
        },
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


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("boolean_seed", "training seed"),
        ("runtime_micro_batch", "micro-batch size disagrees"),
        ("runtime_accumulation", "gradient accumulation disagrees"),
        ("configured_global_batch", "does not equal micro-batch"),
    ],
)
def test_pilot_training_accounting_rejects_type_or_config_mismatch(
    tmp_path, monkeypatch, mutation, message
):
    config, _contract, _preflight, trainer, model, _warm_start = (
        _completion_fixture(tmp_path, monkeypatch)
    )
    if mutation == "boolean_seed":
        config.seed = True
    elif mutation == "runtime_micro_batch":
        trainer.train_dataloader.batch_size = 1
    elif mutation == "runtime_accumulation":
        trainer.accumulate_grad_batches = 3
    elif mutation == "configured_global_batch":
        config.loader.global_batch_size = 15

    with pytest.raises(RuntimeError, match=message):
        train_entrypoint._pilot_training_accounting(
            config,
            trainer,
            model,
            trainer.train_dataloader,
        )


def test_pilot_training_accounting_rejects_trainable_parameters_outside_backbone(
    tmp_path, monkeypatch
):
    config, _contract, _preflight, trainer, model, _warm_start = (
        _completion_fixture(tmp_path, monkeypatch)
    )
    outside_parameter = torch.nn.Parameter(torch.ones(1))
    original_named_parameters = model.named_parameters
    model.named_parameters = lambda: [
        *original_named_parameters(),
        ("outside_backbone", outside_parameter),
    ]

    with pytest.raises(RuntimeError, match="not exactly the backbone"):
        train_entrypoint._pilot_training_accounting(
            config,
            trainer,
            model,
            trainer.train_dataloader,
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


def test_pilot_checkpoint_audit_rejects_path_swap_during_deserialization(
    tmp_path, monkeypatch
):
    _config, contract, _preflight, _trainer, model, _warm_start = (
        _completion_fixture(tmp_path, monkeypatch)
    )
    checkpoint_path = contract["final_checkpoint_path"]
    displaced_path = tmp_path / "displaced.ckpt"
    replacement_path = tmp_path / "replacement.ckpt"
    replacement_path.write_bytes(checkpoint_path.read_bytes())
    original_torch_load = torch.load
    swapped = False

    def swap_path_while_loading(checkpoint_file, *args, **kwargs):
        nonlocal swapped
        assert hasattr(checkpoint_file, "read")
        checkpoint_path.rename(displaced_path)
        replacement_path.rename(checkpoint_path)
        swapped = True
        return original_torch_load(checkpoint_file, *args, **kwargs)

    monkeypatch.setattr(
        train_entrypoint.torch,
        "load",
        swap_path_while_loading,
    )
    with pytest.raises(
        RuntimeError,
        match="identity changed during deserialization",
    ):
        train_entrypoint._audit_pilot_checkpoint(
            checkpoint_path,
            expected_steps=10,
            model=model,
        )
    assert swapped is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("global_step", 9, "checkpoint global_step"),
        ("state_dict", {"weight": torch.tensor([float("nan"), -2.0])}, "non-finite"),
        (
            "ema",
            {
                "shadow_params": [torch.tensor([float("inf")])],
                "decay": 0.9999,
                "num_updates": 10,
            },
            "non-finite",
        ),
        (
            "ema",
            {
                "shadow_params": [torch.tensor([0.5, 2.5])],
                "decay": 0.9999,
                "num_updates": 10,
            },
            "EMA tensor disagrees",
        ),
        (
            "ema",
            {
                "shadow_params": [torch.tensor([0.5, 3.0])],
                "decay": 0.9999,
                "num_updates": 9,
            },
            "EMA update count",
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

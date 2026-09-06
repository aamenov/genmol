import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from scripts import train as train_entrypoint
from scripts.train import checkpoint_startup_mode


def _launch_manifest_fixture(tmp_path, selected_gpu_uuids):
    path = tmp_path / "launch_manifest.json"
    manifest = {
        "launch_manifest_schema_version": 1,
        "user_requested_gpu_count": len(selected_gpu_uuids),
        "cuda_visible_device_uuids": list(selected_gpu_uuids),
    }
    payload = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.write_bytes(payload)
    digest = train_entrypoint.hashlib.sha256(payload).hexdigest()
    snapshot, parsed = train_entrypoint._validate_launch_manifest(
        path,
        expected_sha256=digest,
        expected_selected_gpu_uuids=list(selected_gpu_uuids),
    )
    assert parsed == manifest
    return path, digest, snapshot


def _bind_launch_manifest(contract, path, digest, snapshot, selected_gpu_uuids):
    selected_json = json.dumps(list(selected_gpu_uuids), separators=(",", ":"))
    contract.update(
        {
            "GENMOL_TRAIN_LAUNCH_MANIFEST_PATH": str(path),
            "GENMOL_TRAIN_EXPECTED_LAUNCH_MANIFEST_SHA256": digest,
            "GENMOL_TRAIN_SELECTED_GPU_UUIDS_JSON": selected_json,
            "launch_manifest_path": path,
            "launch_manifest_snapshot": snapshot,
            "selected_gpu_uuids": list(selected_gpu_uuids),
        }
    )


def test_existing_training_checkpoint_takes_precedence_over_warm_start():
    assert checkpoint_startup_mode("step-100.ckpt", "mdlm.ckpt") == "resume"


def test_warm_start_is_used_only_without_a_resume_checkpoint():
    assert checkpoint_startup_mode(None, "mdlm.ckpt") == "warm_start"
    assert checkpoint_startup_mode(None, None) == "scratch"


def test_post_initialization_reseed_is_launch_bound_and_explicit(monkeypatch):
    config = OmegaConf.create(
        {
            "seed": 17,
            "training": {"reseed_after_model_initialization": True},
        }
    )
    calls = []
    monkeypatch.setattr(
        train_entrypoint.L,
        "seed_everything",
        lambda seed, workers: (calls.append((seed, workers)), seed)[1],
    )

    monkeypatch.setattr(train_entrypoint, "_PILOT_CONTRACT", {})
    record = train_entrypoint._reseed_training_rng_after_model_initialization(
        config, "warm_start"
    )
    assert calls == [(17, True)]
    assert record == {
        "policy": "reseed_all_training_rng_streams_after_model_and_warm_start",
        "seed": 17,
        "purpose": "isolate_training_randomness_from_architecture_constructor_draws",
        "applied_before_dataloader_and_trainer_construction": True,
    }

    with pytest.raises(RuntimeError, match="common verified MDLM warm-start"):
        train_entrypoint._reseed_training_rng_after_model_initialization(
            config, "resume"
        )
    with pytest.raises(RuntimeError, match="common verified MDLM warm-start"):
        train_entrypoint._reseed_training_rng_after_model_initialization(
            config, "scratch"
        )
    monkeypatch.setattr(train_entrypoint, "_PILOT_CONTRACT", None)
    with pytest.raises(RuntimeError, match="launch-bound pilot"):
        train_entrypoint._reseed_training_rng_after_model_initialization(
            config, "warm_start"
        )


def test_post_initialization_reseed_default_is_noop_and_rejects_nonboolean():
    config = OmegaConf.create({"seed": 17, "training": {}})
    assert (
        train_entrypoint._reseed_training_rng_after_model_initialization(
            config, "warm_start"
        )
        is None
    )
    config.training.reseed_after_model_initialization = "true"
    with pytest.raises(RuntimeError, match="must be a boolean"):
        train_entrypoint._reseed_training_rng_after_model_initialization(
            config, "warm_start"
        )


def test_post_initialization_reseed_rejects_seed_coercion_and_out_of_range(
    monkeypatch,
):
    config = OmegaConf.create(
        {
            "seed": train_entrypoint._MAX_TRAINING_SEED + 1,
            "training": {"reseed_after_model_initialization": True},
        }
    )
    monkeypatch.setattr(train_entrypoint, "_PILOT_CONTRACT", {})
    with pytest.raises(RuntimeError, match="must be at most"):
        train_entrypoint._reseed_training_rng_after_model_initialization(
            config, "warm_start"
        )

    config.seed = 17
    monkeypatch.setattr(train_entrypoint.L, "seed_everything", lambda _seed, workers: 0)
    with pytest.raises(RuntimeError, match="did not apply the exact"):
        train_entrypoint._reseed_training_rng_after_model_initialization(
            config, "warm_start"
        )


def test_backbone_state_identity_is_order_independent_and_value_exact():
    state = [
        ("z.weight", torch.tensor([[1.0, -0.0]], dtype=torch.float32)),
        ("a.index", torch.tensor([1, 2], dtype=torch.int64)),
    ]
    identity = train_entrypoint._backbone_state_identity(state)

    assert identity == train_entrypoint._backbone_state_identity(reversed(state))
    assert identity["tensor_count"] == 2
    changed = [
        ("z.weight", torch.tensor([[1.0, 0.0]], dtype=torch.float32)),
        state[1],
    ]
    assert (
        train_entrypoint._backbone_state_identity(changed)["state_sha256"]
        != identity["state_sha256"]
    )
    with pytest.raises(RuntimeError, match="duplicate tensor names"):
        train_entrypoint._backbone_state_identity([state[0], state[0]])


def test_screen_initialization_state_audit_separates_common_conditioning_state(
    monkeypatch,
):
    class TinyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
            self.time_conditioner = torch.nn.Linear(2, 2)
            self.block = torch.nn.Module()
            self.block.film_modulation = torch.nn.Linear(2, 4)

    config = OmegaConf.create(
        {
            "seed": 17,
            "training": {
                "init_from_mdlm_checkpoint_sha256": "c" * 64,
                "udlm": {"conditioning_variant": "film_adaln"},
            },
        }
    )
    monkeypatch.setattr(
        train_entrypoint,
        "_PILOT_CONTRACT",
        {
            "GENMOL_TRAIN_EXPECTED_CONFIG_SHA256": "d" * 64,
            "launch_manifest": {"optimization_screen": {"stage_id": "conditioning"}},
        },
    )
    model = SimpleNamespace(backbone=TinyBackbone())
    warm_start = {
        "source_sha256": "c" * 64,
        "expected_source_sha256": "c" * 64,
        "byte_identity_verified_before_and_after_load": True,
        "weights": "ema",
    }

    initial = train_entrypoint._screen_initialization_state_audit(
        config, model, "warm_start", warm_start
    )
    assert set(initial) == {
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
    assert initial["common_backbone_tensor_count"] == 1
    assert initial["full_initial_tensor_count"] == 5

    with torch.no_grad():
        model.backbone.time_conditioner.weight.add_(1.0)
    conditioning_changed = train_entrypoint._screen_initialization_state_audit(
        config, model, "warm_start", warm_start
    )
    assert (
        conditioning_changed["common_backbone_state_sha256"]
        == initial["common_backbone_state_sha256"]
    )
    assert (
        conditioning_changed["full_initial_state_sha256"]
        != initial["full_initial_state_sha256"]
    )

    with torch.no_grad():
        model.backbone.base_weight.add_(1.0)
    base_changed = train_entrypoint._screen_initialization_state_audit(
        config, model, "warm_start", warm_start
    )
    assert (
        base_changed["common_backbone_state_sha256"]
        != initial["common_backbone_state_sha256"]
    )


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


def test_pilot_launch_manifest_requires_exact_raw_hash_and_selected_uuids(tmp_path):
    selected_gpu_uuids = ["GPU-test-a", "GPU-test-b"]
    path, digest, snapshot = _launch_manifest_fixture(tmp_path, selected_gpu_uuids)

    assert snapshot["sha256"] == digest
    with pytest.raises(RuntimeError, match="raw SHA-256"):
        train_entrypoint._validate_launch_manifest(
            path,
            expected_sha256="0" * 64,
            expected_selected_gpu_uuids=selected_gpu_uuids,
        )

    mutated = {
        "launch_manifest_schema_version": 1,
        "user_requested_gpu_count": 2,
        "cuda_visible_device_uuids": ["GPU-test-a", "GPU-other"],
    }
    path.write_text(
        json.dumps(mutated, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    mutated_digest = train_entrypoint.hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="selected GPU UUIDs disagree"):
        train_entrypoint._validate_launch_manifest(
            path,
            expected_sha256=mutated_digest,
            expected_selected_gpu_uuids=selected_gpu_uuids,
        )


def test_pilot_selected_uuid_contract_must_equal_actual_cuda_exposure(monkeypatch):
    selected_gpu_uuids = ["GPU-test-a", "GPU-test-b"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test-a,GPU-test-b")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    train_entrypoint._validate_selected_gpu_exposure(selected_gpu_uuids)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test-b,GPU-test-a")
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
        train_entrypoint._validate_selected_gpu_exposure(selected_gpu_uuids)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test-a,GPU-test-b")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    with pytest.raises(RuntimeError, match="CUDA_DEVICE_ORDER"):
        train_entrypoint._validate_selected_gpu_exposure(selected_gpu_uuids)


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
            "optim": {
                "weight_decay": 0,
                "lr": 3e-4,
                "beta1": 0.9,
                "beta2": 0.999,
                "eps": 1e-8,
                "scheduler": {
                    "name": "constant_with_linear_warmup",
                    "warmup_updates": 2500,
                    "horizon_updates": None,
                    "decay_floor_lr": None,
                },
            },
            "trainer": {
                "devices": 2,
                "num_nodes": 1,
                "max_steps": 10,
                "accumulate_grad_batches": 4,
                "detect_anomaly": True,
                "gradient_clip_val": 1.0,
                "precision": "bf16",
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
    selected_gpu_uuids = ["GPU-test-a", "GPU-test-b"]
    manifest_path, manifest_sha256, manifest_snapshot = _launch_manifest_fixture(
        tmp_path, selected_gpu_uuids
    )
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
        "summary_schema_version": train_entrypoint._TRAINING_SUMMARY_SCHEMA_VERSION,
        "runtime_path": runtime_path,
        "summary_path": summary_path,
        "final_checkpoint_path": final_checkpoint_path,
    }
    _bind_launch_manifest(
        contract,
        manifest_path,
        manifest_sha256,
        manifest_snapshot,
        selected_gpu_uuids,
    )
    monkeypatch.setattr(train_entrypoint, "_PILOT_CONTRACT", contract)
    monkeypatch.setattr(
        train_entrypoint,
        "_require_pilot_source_revision",
        lambda revision: {"head": revision, "upstream": revision},
    )
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("PYTHONHASHSEED", "7")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(selected_gpu_uuids))
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    first = train_entrypoint._validate_and_record_pilot_config(config)
    second = train_entrypoint._validate_and_record_pilot_config(config)

    assert first == second
    assert json.loads(runtime_path.read_text(encoding="utf-8")) == first
    assert (
        first["resolved_training_config_sha256"]
        == contract["GENMOL_TRAIN_EXPECTED_CONFIG_SHA256"]
    )
    assert first["schema_version"] == train_entrypoint._RUNTIME_CONFIG_SCHEMA_VERSION
    assert first["launch_manifest"]["sha256"] == manifest_sha256
    assert first["launch_manifest"]["selected_gpu_uuids"] == selected_gpu_uuids

    contract["GENMOL_TRAIN_EXPECTED_CONFIG_SHA256"] = "b" * 64
    with pytest.raises(RuntimeError, match="launch-pinned config digest"):
        train_entrypoint._validate_and_record_pilot_config(config)


class _FixtureIterableDataset(torch.utils.data.IterableDataset):
    def __iter__(self):
        return iter(())


def _completion_fixture(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    runtime_path = tmp_path / "runtime_config.json"
    summary_path = tmp_path / "training_summary.json"
    checkpoint_path = checkpoint_dir / "10.ckpt"
    checkpoint_state = {"weight": torch.tensor([1.0, -2.0])}
    checkpoint_ema = [torch.tensor([0.5, 3.0])]
    callback_key = train_entrypoint._model_checkpoint_callback_state_key(10)
    callback_state = {
        "monitor": None,
        "best_model_score": None,
        "best_model_path": str(checkpoint_path),
        "current_score": None,
        "dirpath": str(checkpoint_dir),
        "best_k_models": {},
        "kth_best_model_path": "",
        "kth_value": torch.tensor(float("inf"), dtype=torch.float32),
        "last_model_path": "",
    }
    base_parameter = torch.nn.Parameter(torch.ones(3))
    conditioner_parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW(
        [base_parameter, conditioner_parameter],
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda index: index / 2500 if index < 2500 else 1.0,
    )
    for _index in range(10):
        base_parameter.grad = torch.full_like(base_parameter, 0.25)
        conditioner_parameter.grad = torch.full_like(conditioner_parameter, 0.5)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    torch.save(
        {
            "pytorch-lightning_version": "2.5.1",
            "global_step": 10,
            "state_dict": checkpoint_state,
            "ema": {
                "shadow_params": checkpoint_ema,
                "decay": 0.9999,
                "num_updates": 10,
            },
            "optimizer_states": [optimizer.state_dict()],
            "lr_schedulers": [scheduler.state_dict()],
            "sampler": {"random_state": None},
            "callbacks": {callback_key: callback_state},
        },
        checkpoint_path,
    )
    argv = ["/repo/scripts/train.py", "seed=7"]
    config = OmegaConf.create(
        {
            "data": "safe",
            "seed": 7,
            "loader": {"global_batch_size": 16, "batch_size": 2},
            "optim": {
                "weight_decay": 0,
                "lr": 3e-4,
                "beta1": 0.9,
                "beta2": 0.999,
                "eps": 1e-8,
                "scheduler": {
                    "name": "constant_with_linear_warmup",
                    "warmup_updates": 2500,
                    "horizon_updates": None,
                    "decay_floor_lr": None,
                },
            },
            "trainer": {
                "devices": 2,
                "num_nodes": 1,
                "max_steps": 10,
                "accumulate_grad_batches": 4,
                "detect_anomaly": True,
                "gradient_clip_val": 1.0,
                "precision": "bf16",
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
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint.update(
        {
            "epoch": 0,
            "hparams_name": "kwargs",
            "hyper_parameters": {"config": config},
            "loops": train_entrypoint._expected_lightning_loop_state(
                optimizer_steps=10,
                accumulation=4,
            ),
        }
    )
    torch.save(checkpoint, checkpoint_path)
    selected_gpu_uuids = ["GPU-test-a", "GPU-test-b"]
    manifest_path, manifest_sha256, manifest_snapshot = _launch_manifest_fixture(
        tmp_path, selected_gpu_uuids
    )
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
    _bind_launch_manifest(
        contract,
        manifest_path,
        manifest_sha256,
        manifest_snapshot,
        selected_gpu_uuids,
    )
    monkeypatch.setattr(train_entrypoint, "_PILOT_CONTRACT", contract)
    monkeypatch.setattr(
        train_entrypoint,
        "_require_pilot_source_revision",
        lambda revision: {"head": revision, "upstream": revision},
    )
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("PYTHONHASHSEED", "7")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(selected_gpu_uuids))
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    preflight = train_entrypoint._validate_and_record_pilot_config(config)
    health_callback = train_entrypoint._PilotFiniteLossCallback()
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    health_module = SimpleNamespace(named_parameters=lambda: [("weight", parameter)])
    for _index in range(10):
        health_callback.on_before_backward(None, None, torch.tensor(1.25))
        parameter.grad = torch.tensor([0.5])
        health_callback.on_before_optimizer_step(None, health_module, None)
    checkpoint_callback = train_entrypoint.L.pytorch.callbacks.ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="{step}",
        save_top_k=-1,
        auto_insert_metric_name=False,
        enable_version_counter=False,
        every_n_train_steps=10,
    )
    checkpoint_callback.load_state_dict(callback_state)
    trainer = SimpleNamespace(
        is_global_zero=True,
        global_rank=0,
        global_step=10,
        world_size=2,
        num_nodes=1,
        max_steps=10,
        accumulate_grad_batches=4,
        _detect_anomaly=True,
        gradient_clip_val=1.0,
        gradient_clip_algorithm=None,
        precision="bf16-mixed",
        train_dataloader=torch.utils.data.DataLoader(
            _FixtureIterableDataset(),
            batch_size=2,
        ),
        callbacks=[checkpoint_callback, health_callback],
        optimizers=[optimizer],
        lr_scheduler_configs=[
            SimpleNamespace(scheduler=scheduler, interval="step", name="lr")
        ],
    )
    backbone = SimpleNamespace(
        parameters=lambda: iter([base_parameter, conditioner_parameter]),
        named_parameters=lambda: [
            ("base_weight", base_parameter),
            ("time_conditioner.weight", conditioner_parameter),
        ],
    )
    model = SimpleNamespace(
        config=config,
        hparams={"config": config},
        optimizer_scheduler_spec=train_entrypoint.optimizer_scheduler_spec(
            resolved["optim"]
        ),
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
        _validate_udlm_conditioning_checkpoint=lambda checkpoint: None,
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


class _TinyFilmAuditModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Module()
        self.backbone.time_conditioner = torch.nn.Sequential(
            torch.nn.Linear(2, 2),
            torch.nn.SiLU(),
            torch.nn.Linear(2, 2),
        )
        self.backbone.block = torch.nn.Module()
        self.backbone.block.film_modulation = torch.nn.Linear(2, 4)


def _film_audit_contract(model):
    groups = []
    for group_id, kind, predicate in (
        (
            "film_modulation",
            "film",
            lambda name: ".film_modulation." in name,
        ),
        (
            "timestep_mlp",
            "timestep_mlp",
            lambda name: name.removeprefix("backbone.").startswith("time_conditioner."),
        ),
    ):
        groups.append(
            {
                "group_id": group_id,
                "kind": kind,
                "parameters": [
                    {"name": name, "shape": list(parameter.shape)}
                    for name, parameter in model.named_parameters()
                    if predicate(name)
                ],
            }
        )
    contract = {
        "schema_version": 1,
        "observation_point": (
            "on_before_optimizer_step_global_rank_zero_after_gradient_accumulation"
        ),
        "optimizer_checks": [1, 2, 3],
        "first_positive_lr_optimizer_step": 2,
        "timestep_mlp_required_optimizer_check": 3,
        "groups": groups,
    }
    return contract, train_entrypoint._canonical_json_sha256(contract)


def _set_conditioning_gradients(model, *, timestep_nonzero):
    for name, parameter in model.named_parameters():
        if ".film_modulation." in name:
            parameter.grad = torch.ones_like(parameter)
        elif name.removeprefix("backbone.").startswith("time_conditioner."):
            parameter.grad = (
                torch.ones_like(parameter)
                if timestep_nonzero
                else torch.zeros_like(parameter)
            )


def test_film_gradient_audit_binds_topology_lr_transition_and_staged_gradients(
    monkeypatch,
):
    model = _TinyFilmAuditModel()
    contract, digest = _film_audit_contract(model)
    monkeypatch.setattr(
        train_entrypoint,
        "_PILOT_CONTRACT",
        {
            "launch_manifest": {
                "optimization_screen": {
                    "conditioning_gradient_contract": contract,
                    "conditioning_gradient_contract_sha256": digest,
                }
            }
        },
    )
    config = OmegaConf.create(
        {
            "training": {
                "pilot_fail_on_nonfinite_loss": True,
                "reseed_after_model_initialization": True,
                "udlm": {"conditioning_variant": "film_adaln"},
            },
            "trainer": {"detect_anomaly": True},
        }
    )
    callbacks = train_entrypoint._pilot_callbacks(config)
    assert [type(callback) for callback in callbacks] == [
        train_entrypoint._PilotFiniteLossCallback,
        train_entrypoint._FilmGradientActivationCallback,
    ]
    callback = callbacks[1]
    for learning_rate, timestep_nonzero in (
        (0.0, False),
        (3e-6, False),
        (6e-6, True),
    ):
        _set_conditioning_gradients(model, timestep_nonzero=timestep_nonzero)
        callback.on_before_optimizer_step(
            None,
            model,
            SimpleNamespace(param_groups=[{"lr": learning_rate}]),
        )
    report = callback.completion_report()
    assert report == {
        "schema_version": 1,
        "status": "completed",
        "observation_point": (
            "on_before_optimizer_step_global_rank_zero_after_gradient_accumulation"
        ),
        "registered_contract_sha256": digest,
        "first_positive_lr_optimizer_step": 2,
        "timestep_mlp_required_optimizer_check": 3,
        "optimizer_checks": callback.optimizer_checks,
    }
    assert [
        check["learning_rate_before_step"] for check in report["optimizer_checks"]
    ] == [0.0, 3e-6, 6e-6]
    assert (
        report["optimizer_checks"][0]["film_groups"][0][
            "all_parameter_gradients_nonzero"
        ]
        is True
    )
    assert (
        report["optimizer_checks"][0]["timestep_mlp_groups"][0][
            "all_parameter_gradients_nonzero"
        ]
        is False
    )
    assert (
        report["optimizer_checks"][2]["timestep_mlp_groups"][0][
            "all_parameter_gradients_nonzero"
        ]
        is True
    )


def test_film_gradient_audit_rejects_a_dead_timestep_path_on_third_backward():
    model = _TinyFilmAuditModel()
    contract, digest = _film_audit_contract(model)
    contract = train_entrypoint._validate_film_gradient_audit_contract(contract, digest)
    callback = train_entrypoint._FilmGradientActivationCallback(contract)
    for learning_rate in (0.0, 3e-6):
        _set_conditioning_gradients(model, timestep_nonzero=False)
        callback.on_before_optimizer_step(
            None,
            model,
            SimpleNamespace(param_groups=[{"lr": learning_rate}]),
        )
    _set_conditioning_gradients(model, timestep_nonzero=False)
    with pytest.raises(RuntimeError, match="third optimizer observation"):
        callback.on_before_optimizer_step(
            None,
            model,
            SimpleNamespace(param_groups=[{"lr": 6e-6}]),
        )


def test_film_gradient_audit_rejects_unregistered_topology_or_lr_schedule():
    model = _TinyFilmAuditModel()
    contract, digest = _film_audit_contract(model)
    contract["groups"][0]["parameters"][0]["shape"] = [999]
    altered_digest = train_entrypoint._canonical_json_sha256(contract)
    callback = train_entrypoint._FilmGradientActivationCallback(
        train_entrypoint._validate_film_gradient_audit_contract(
            contract, altered_digest
        )
    )
    _set_conditioning_gradients(model, timestep_nonzero=False)
    with pytest.raises(RuntimeError, match="parameter manifest changed"):
        callback.on_before_optimizer_step(
            None,
            model,
            SimpleNamespace(param_groups=[{"lr": 0.0}]),
        )

    valid_contract, _valid_digest = _film_audit_contract(model)
    callback = train_entrypoint._FilmGradientActivationCallback(valid_contract)
    with pytest.raises(RuntimeError, match="expected zero LR"):
        callback.on_before_optimizer_step(
            None,
            model,
            SimpleNamespace(param_groups=[{"lr": 3e-6}]),
        )

    with pytest.raises(RuntimeError, match="registered SHA-256"):
        train_entrypoint._validate_film_gradient_audit_contract(
            valid_contract, "0" * 64
        )


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
    assert (
        summary["launch_manifest"]["sha256"]
        == contract["GENMOL_TRAIN_EXPECTED_LAUNCH_MANIFEST_SHA256"]
    )
    assert summary["launch_manifest"]["selected_gpu_uuids"] == [
        "GPU-test-a",
        "GPU-test-b",
    ]
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
    assert checkpoint_audit["non_sentinel_checkpoint_tensors"]["all_finite"] is True
    callback_key = train_entrypoint._model_checkpoint_callback_state_key(10)
    assert checkpoint_audit["framework_nonfinite_sentinels"] == {
        "all_expected_and_only_expected_verified": True,
        "nonfinite_tensor_count": 1,
        "nonfinite_element_count": 1,
        "records": [
            {
                "tensor_path_components": [
                    "checkpoint",
                    "callbacks",
                    callback_key,
                    "kth_value",
                ],
                "framework": "lightning",
                "framework_version": "2.5.1",
                "callback": "ModelCheckpoint",
                "field": "kth_value",
                "dtype": "float32",
                "shape": [],
                "value": "+inf",
                "meaning": "unranked_min_mode_checkpoint_sentinel",
                "excluded_from_non_sentinel_finiteness": True,
            }
        ],
    }
    assert checkpoint_audit["checkpoint_python_floats"]["all_finite"] is True
    assert checkpoint_audit["checkpoint_python_floats"]["floating_scalar_count"] > 0
    assert checkpoint_audit["optimizer_live_state_match"] == {
        "exact_serialized_live_match": True,
        "optimizer_count": 1,
        "optimizer_class": "AdamW",
        "parameter_group_count": 1,
        "parameter_state_count": 2,
        "exact_resolved_config_match": True,
    }
    assert checkpoint_audit["scheduler_live_state_match"] == {
        "exact_serialized_live_match": True,
        "scheduler_count": 1,
        "scheduler_class": "LambdaLR",
        "interval": "step",
        "name": "lr",
        "last_epoch": 10,
        "step_count": 11,
        "exact_model_spec_match": True,
        "exact_callable_schedule_match": True,
        "callable_schedule_index_checks": 2502,
    }
    assert checkpoint_audit["sampler_live_state_match"] == {
        "exact_hosted_stream_contract_match": True,
        "random_state_is_none": True,
        "live_state_dict_available": False,
        "sampler_class_module": "torch.utils.data.dataloader",
        "sampler_class_name": "_InfiniteConstantSampler",
    }
    assert checkpoint_audit["trainer_live_configuration_match"] == {
        "exact_detect_anomaly_match": True,
        "detect_anomaly": True,
        "exact_gradient_clip_val_match": True,
        "gradient_clip_val": 1.0,
        "exact_gradient_clip_algorithm_match": True,
        "gradient_clip_algorithm": "norm",
        "exact_precision_match": True,
        "configured_precision": "bf16",
        "live_precision": "bf16-mixed",
    }
    assert checkpoint_audit["model_checkpoint_live_state_match"] == {
        "exact_serialized_live_match": True,
        "model_checkpoint_callback_count": 1,
        "state_key": callback_key,
        "configuration_matches_pilot_contract": True,
    }
    assert checkpoint_audit["checkpoint_hyperparameters_match"] == {
        "hparams_name": "kwargs",
        "exact_hyperparameter_keys": True,
        "exact_checkpoint_preflight_config_match": True,
        "exact_live_model_preflight_config_match": True,
        "exact_live_hparams_preflight_config_match": True,
        "exact_checkpoint_live_model_unresolved_config_match": True,
        "exact_checkpoint_live_hparams_unresolved_config_match": True,
        "resolved_config_sha256": contract["GENMOL_TRAIN_EXPECTED_CONFIG_SHA256"],
    }
    assert checkpoint_audit["checkpoint_loop_state_match"] == {
        "exact_serialized_progress_match": True,
        "epoch": 0,
        "optimizer_steps": 10,
        "accumulate_grad_batches": 4,
        "microbatches": 40,
    }
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
    assert summary["conditioning_gradient_audit"] is None
    assert summary["screen_initialization_state_audit"] is None
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
    config, _contract, _preflight, trainer, model, _warm_start = _completion_fixture(
        tmp_path, monkeypatch
    )
    if mutation == "boolean_seed":
        config.seed = True
    elif mutation == "runtime_micro_batch":
        trainer.train_dataloader = torch.utils.data.DataLoader(
            _FixtureIterableDataset(), batch_size=1
        )
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


def test_pilot_parameter_accounting_separates_film_from_base_backbone():
    class FilmLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.film_modulation = torch.nn.Linear(2, 4)

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_weight = torch.nn.Parameter(torch.ones(3))
            self.time_conditioner = torch.nn.Linear(2, 1, bias=False)
            self.layer = torch.nn.ModuleList([FilmLayer()])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Backbone()

    assert train_entrypoint._pilot_trainable_parameter_counts(Model()) == {
        "base_backbone": 3,
        "time_conditioner": 2,
        "film_modulation": 12,
        "total": 17,
    }


def test_pilot_training_accounting_rejects_trainable_parameters_outside_backbone(
    tmp_path, monkeypatch
):
    config, _contract, _preflight, trainer, model, _warm_start = _completion_fixture(
        tmp_path, monkeypatch
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


def test_pilot_completion_rejects_launch_manifest_changed_after_preflight(
    tmp_path, monkeypatch
):
    config, contract, preflight, trainer, model, warm_start = _completion_fixture(
        tmp_path, monkeypatch
    )
    contract["launch_manifest_path"].write_text(
        json.dumps(
            {
                "launch_manifest_schema_version": 1,
                "user_requested_gpu_count": 2,
                "cuda_visible_device_uuids": ["GPU-test-a", "GPU-replacement"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="launch manifest raw SHA-256"):
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
    _config, contract, preflight, trainer, model, _warm_start = _completion_fixture(
        tmp_path, monkeypatch
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
            trainer=trainer,
            expected_resolved_config=preflight["resolved_training_config"],
            expected_config_sha256=contract["GENMOL_TRAIN_EXPECTED_CONFIG_SHA256"],
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


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("nan", "exact scalar float32 positive-infinity sentinel"),
        ("negative_inf", "exact scalar float32 positive-infinity sentinel"),
        ("vector_inf", "exact scalar float32 positive-infinity sentinel"),
        ("wrong_checkpoint_path", "unranked final-step checkpoint contract"),
        ("nonempty_top_k", "unranked final-step checkpoint contract"),
        ("wrong_callback_key", "callback-state keys disagree"),
        ("wrong_lightning_version", "Lightning version"),
        ("second_nonfinite", "serialized checkpoint optimizer state"),
        ("nonfinite_optimizer", "serialized checkpoint optimizer state"),
    ],
)
def test_pilot_checkpoint_allows_only_the_exact_lightning_sentinel(
    tmp_path, monkeypatch, mutation, message
):
    config, contract, preflight, trainer, model, warm_start = _completion_fixture(
        tmp_path, monkeypatch
    )
    checkpoint = torch.load(
        contract["final_checkpoint_path"], map_location="cpu", weights_only=False
    )
    callback_key = train_entrypoint._model_checkpoint_callback_state_key(10)
    callback_state = checkpoint["callbacks"][callback_key]
    if mutation == "nan":
        callback_state["kth_value"] = torch.tensor(float("nan"))
    elif mutation == "negative_inf":
        callback_state["kth_value"] = torch.tensor(float("-inf"))
    elif mutation == "vector_inf":
        callback_state["kth_value"] = torch.tensor([float("inf")])
    elif mutation == "wrong_checkpoint_path":
        callback_state["best_model_path"] = str(tmp_path / "other.ckpt")
    elif mutation == "nonempty_top_k":
        callback_state["best_k_models"] = {"other.ckpt": torch.tensor(1.0)}
    elif mutation == "wrong_callback_key":
        checkpoint["callbacks"][f"{callback_key}-tampered"] = checkpoint[
            "callbacks"
        ].pop(callback_key)
    elif mutation == "wrong_lightning_version":
        checkpoint["pytorch-lightning_version"] = "2.5.0"
    elif mutation == "second_nonfinite":
        checkpoint["optimizer_states"][0]["state"][0]["unexpected_nonfinite"] = (
            torch.tensor(float("inf"))
        )
    elif mutation == "nonfinite_optimizer":
        checkpoint["optimizer_states"][0]["state"][0]["exp_avg"] = torch.tensor(
            [float("nan")]
        )
    else:  # pragma: no cover - the parameter table is exhaustive
        raise AssertionError(mutation)
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


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_hyperparameters", "hyper_parameters must contain exactly config"),
        ("resolved_hyperparameter_drift", "checkpoint/preflight resolved config"),
        ("alternate_interpolation", "checkpoint/live-model unresolved config"),
        ("loop_progress", "serialized Lightning loop state"),
        ("optimizer_config", "AdamW/resolved-config parameter group"),
        ("scheduler_callable", "LambdaLR callable disagrees"),
        ("wrong_sampler", "unexpected hosted-stream sampler"),
        ("live_callback", "live ModelCheckpoint disagrees"),
        ("python_float", "non-finite serialized checkpoint Python float"),
        ("numpy_array", "unsupported NumPy numeric leaf"),
        ("unknown_top_level", "top-level keys disagree"),
        ("opaque_top_level", "top-level keys disagree"),
        ("ema_finite_extra", "EMA keys disagree"),
        ("ema_opaque_extra", "EMA keys disagree"),
    ],
)
def test_pilot_checkpoint_rejects_semantic_state_drift(
    tmp_path, monkeypatch, mutation, message
):
    config, contract, preflight, trainer, model, warm_start = _completion_fixture(
        tmp_path, monkeypatch
    )
    checkpoint_path = contract["final_checkpoint_path"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if mutation == "missing_hyperparameters":
        checkpoint["hyper_parameters"] = {}
    elif mutation == "resolved_hyperparameter_drift":
        checkpoint["hyper_parameters"]["config"].data = "other"
    elif mutation == "alternate_interpolation":
        monkeypatch.delenv("UDLM_AUDIT_FAKE_DATA", raising=False)
        alternate = OmegaConf.create(
            OmegaConf.to_container(config, resolve=False, enum_to_str=True)
        )
        alternate.data = "${oc.env:UDLM_AUDIT_FAKE_DATA,safe}"
        assert OmegaConf.to_container(alternate, resolve=True)["data"] == "safe"
        checkpoint["hyper_parameters"]["config"] = alternate
    elif mutation == "loop_progress":
        checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"]["total"][
            "ready"
        ] -= 1
    elif mutation == "optimizer_config":
        trainer.optimizers[0].param_groups[0]["betas"] = (0.8, 0.999)
        checkpoint["optimizer_states"] = [trainer.optimizers[0].state_dict()]
    elif mutation == "scheduler_callable":
        trainer.lr_scheduler_configs[0].scheduler.lr_lambdas[0] = lambda _index: 1.0
    elif mutation == "wrong_sampler":
        trainer.train_dataloader = SimpleNamespace(
            batch_size=2,
            sampler=SimpleNamespace(),
        )
    elif mutation == "live_callback":
        trainer.callbacks[0].mode = "max"
    elif mutation == "python_float":
        checkpoint["ema"]["unexpected_python_float"] = float("inf")
    elif mutation == "numpy_array":
        checkpoint["ema"]["unexpected_numpy"] = np.array([np.nan], dtype=np.float32)
    elif mutation == "unknown_top_level":
        checkpoint["unexpected"] = "finite but outside the exact schema"
    elif mutation == "opaque_top_level":
        checkpoint["opaque_extra"] = SimpleNamespace(hidden=torch.tensor(float("nan")))
    elif mutation == "ema_finite_extra":
        checkpoint["ema"]["unexpected"] = "finite but outside the exact schema"
    elif mutation == "ema_opaque_extra":
        checkpoint["ema"]["opaque_extra"] = SimpleNamespace(
            hidden=torch.tensor(float("nan"))
        )
    else:  # pragma: no cover - the parameter table is exhaustive
        raise AssertionError(mutation)
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises((RuntimeError, FloatingPointError), match=message):
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
        ("_detect_anomaly", False, "did not enable anomaly detection"),
        ("gradient_clip_val", 0.5, "gradient clipping disagrees"),
        (
            "gradient_clip_algorithm",
            SimpleNamespace(value="value"),
            "gradient-clip algorithm disagrees",
        ),
        ("precision", "32-true", "precision disagrees"),
    ],
)
def test_pilot_completion_rejects_live_trainer_config_drift(
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


@pytest.mark.parametrize(
    ("algorithm_config", "live_algorithm", "accepted"),
    [
        ({}, None, True),
        ({"gradient_clip_algorithm": None}, None, True),
        ({"gradient_clip_algorithm": "norm"}, SimpleNamespace(value="norm"), True),
        (
            {"gradient_clip_algorithm": "value"},
            SimpleNamespace(value="value"),
            True,
        ),
        ({"gradient_clip_algorithm": "norm"}, None, False),
        ({}, SimpleNamespace(value="norm"), False),
        (
            {"gradient_clip_algorithm": "norm"},
            SimpleNamespace(value="value"),
            False,
        ),
    ],
)
def test_live_trainer_gradient_clip_algorithm_contract(
    algorithm_config, live_algorithm, accepted
):
    trainer_config = {
        "detect_anomaly": True,
        "gradient_clip_val": 1.0,
        "precision": "bf16",
        **algorithm_config,
    }
    trainer = SimpleNamespace(
        _detect_anomaly=True,
        gradient_clip_val=1.0,
        gradient_clip_algorithm=live_algorithm,
        precision="bf16-mixed",
    )

    if accepted:
        result = train_entrypoint._validated_live_trainer_configuration(
            {"trainer": trainer_config}, trainer
        )
        assert result["gradient_clip_algorithm"] == (
            algorithm_config.get("gradient_clip_algorithm") or "norm"
        )
    else:
        with pytest.raises(RuntimeError, match="gradient-clip algorithm disagrees"):
            train_entrypoint._validated_live_trainer_configuration(
                {"trainer": trainer_config}, trainer
            )


def test_checkpoint_top_level_schema_requires_conditional_udlm_metadata(
    tmp_path, monkeypatch
):
    _config, contract, _preflight, _trainer, model, _warm_start = _completion_fixture(
        tmp_path, monkeypatch
    )

    class _CategoricalProcess:
        pass

    monkeypatch.setattr(
        train_entrypoint, "ContinuousCategoricalDiffusion", _CategoricalProcess
    )
    model.mdlm = _CategoricalProcess()
    model.udlm_conditioning_metadata = object()
    checkpoint = torch.load(
        contract["final_checkpoint_path"], map_location="cpu", weights_only=False
    )
    checkpoint[train_entrypoint.UDLM_PRIOR_CHECKPOINT_KEY] = {}
    checkpoint[train_entrypoint.UDLM_CONDITIONING_CHECKPOINT_KEY] = {}

    train_entrypoint._validated_checkpoint_top_level_schema(checkpoint, model=model)

    checkpoint.pop(train_entrypoint.UDLM_PRIOR_CHECKPOINT_KEY)
    with pytest.raises(RuntimeError, match="top-level keys disagree"):
        train_entrypoint._validated_checkpoint_top_level_schema(checkpoint, model=model)


def test_checkpoint_tensor_walk_does_not_resolve_hydra_interpolations():
    unresolved = OmegaConf.create({"value": "${not_registered:anything}"})

    assert list(
        train_entrypoint._nested_named_tensors(
            {"hyper_parameters": unresolved, "weight": torch.tensor([1.0])},
            prefix="checkpoint",
        )
    ) == [('checkpoint["weight"]', torch.tensor([1.0]))]


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

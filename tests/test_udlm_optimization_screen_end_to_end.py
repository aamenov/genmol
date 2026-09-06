from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from argparse import Namespace
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

import test_udlm_optimization_screen as cases
from scripts import train as train_entrypoint
from scripts.udlm import audit_conditioning_initialization as initialization_producer
from scripts.udlm import collect_optimization_screen_evidence as collector
from scripts.udlm import launch_optimization_screen as screen_launcher
from scripts.udlm import launch_train_pilot as pilot_launcher
from scripts.udlm import verify_optimization_screen as verifier
from scripts.udlm import write_pilot_exit_status as receipt_writer


def _screen_config_factory(repository: Path, original):
    """Add producer-required leaves to the verifier's compact config fixture."""

    def build(**kwargs):
        config = original(**kwargs)
        config["data"] = "safe"
        config["training"].update(
            {
                "ema": 0.9999,
                "pilot_fail_on_nonfinite_loss": True,
            }
        )
        config["training"]["udlm"]["exclude_special_tokens"] = False
        config["loader"]["num_workers"] = 1
        config["trainer"]["detect_anomaly"] = True
        config["callback"].update(
            {
                "dirpath": str(repository / kwargs["output_directory"] / "checkpoints"),
                "filename": "{step}",
                "save_top_k": -1,
                "every_n_train_steps": kwargs["updates"],
            }
        )
        return config

    return build


class _CountedParameter:
    """Parameter-count stub; tensor bytes are exercised through ``state_dict``."""

    requires_grad = True

    def __init__(self, count: int):
        self._count = count

    def numel(self) -> int:
        return self._count


def _fake_model_and_checkpoint(
    path: Path,
    *,
    optimizer_updates: int,
    conditioning_counts: tuple[int, int] | None = None,
):
    if conditioning_counts is None:
        base_parameter = torch.nn.Parameter(torch.ones(3))
        time_parameter = torch.nn.Parameter(torch.ones(2))
        film_parameters = []
    else:
        time_count, film_count = conditioning_counts
        base_parameter = _CountedParameter(3)
        time_parameter = _CountedParameter(time_count)
        film_parameters = [
            ("block.film_modulation.weight", _CountedParameter(film_count))
        ]
    named_parameters = [
        ("base_weight", base_parameter),
        ("time_conditioner.weight", time_parameter),
        *film_parameters,
    ]
    raw_state = {"weight": torch.tensor([1.0, -2.0])}
    ema_state = [torch.tensor([0.5, 3.0])]
    backbone = SimpleNamespace(
        named_parameters=lambda: named_parameters,
        state_dict=lambda: {
            "base_weight": torch.ones(3),
            "time_conditioner.weight": torch.ones(2),
            **(
                {"block.film_modulation.weight": torch.zeros(1)}
                if film_parameters
                else {}
            ),
        },
    )
    model = SimpleNamespace(
        backbone=backbone,
        named_parameters=lambda: [
            (f"backbone.{name}", parameter) for name, parameter in named_parameters
        ],
        state_dict=lambda: dict(raw_state),
        ema=SimpleNamespace(
            shadow_params=list(ema_state),
            decay=0.9999,
            num_updates=optimizer_updates,
        ),
        _validate_udlm_prior_checkpoint=lambda _checkpoint: None,
        _validate_udlm_conditioning_checkpoint=lambda _checkpoint: None,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "global_step": optimizer_updates,
            "state_dict": raw_state,
            "ema": {
                "shadow_params": ema_state,
                "decay": 0.9999,
                "num_updates": optimizer_updates,
            },
            "optimizer_states": [
                {
                    "state": {
                        0: {
                            "step": torch.tensor(float(optimizer_updates)),
                            "exp_avg": torch.tensor([0.1, -0.2]),
                            "exp_avg_sq": torch.tensor([0.01, 0.04]),
                        }
                    }
                }
            ],
        },
        path,
    )
    return model


def _completed_health_callback(optimizer_updates: int):
    callback = train_entrypoint._PilotFiniteLossCallback()
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    module = SimpleNamespace(named_parameters=lambda: [("weight", parameter)])
    for _ in range(optimizer_updates):
        callback.on_before_backward(None, None, torch.tensor(1.25))
        parameter.grad = torch.tensor([0.5])
        callback.on_before_optimizer_step(None, module, None)
    return callback


def _add_file_to_harness(
    harness: cases.Harness, repository: Path, relative_path: str
) -> None:
    harness.blobs[("repository", relative_path)] = (
        repository / relative_path
    ).read_bytes()


def _produce_screen_arm(
    *,
    harness: cases.Harness,
    repository: Path,
    stage_id: str,
    arm_id: str,
    source_revision: str,
    losses: list[float],
    monkeypatch: pytest.MonkeyPatch,
    scheduler_arm_id: str | None = None,
    scheduler_dependency: dict | None = None,
) -> None:
    """Run current launch, summary, and receipt producers without real hardware."""

    assert harness.registry is not None
    stage = verifier._stage(harness.registry, stage_id)
    arm = verifier._arm(stage, arm_id)
    config_entry = verifier._registered_config(
        arm,
        scheduler_arm_id=(None if stage_id == "scheduler" else scheduler_arm_id),
    )
    output_directory = config_entry["output_directory"]
    run_dir = repository / output_directory
    selected_uuid = "GPU-cpu-fixture"
    optimizer_updates = verifier.EXPECTED_UPDATES[stage_id]
    resolved_config = screen_launcher._json_compatible(config_entry["parsed_config"])
    command = [
        "/fixture/python",
        "-u",
        str(repository / "scripts/train.py"),
        "--config-name",
        "udlm_categorical",
        f"fixture_arm={arm_id}",
    ]
    argv_sha256 = pilot_launcher.training_argv_sha256(command)
    checkpoint_path = run_dir / "checkpoints" / f"{optimizer_updates}.ckpt"
    monkeypatch.setattr(
        pilot_launcher, "require_pushed_commit", lambda: source_revision
    )
    lock_path, lock_record, lock_sha256 = pilot_launcher.acquire_training_job_lock(
        source_revision=source_revision,
        run_name=arm["attempt_id"],
        training_variant=screen_launcher.SCREEN_TRAINING_VARIANT,
        purpose=screen_launcher.SCREEN_LOCK_PURPOSE,
    )
    plan = screen_launcher.ScreenLaunchPlan(
        registry=harness.registry,
        stage_id=stage_id,
        arm=arm,
        config_entry=config_entry,
        scheduler_dependency=scheduler_dependency,
        source_revision=source_revision,
        gpu_count=1,
        checkpoint=repository / "mdlm.ckpt",
        checkpoint_sha256=harness.registry.data["common_training"]["initialization"][
            "checkpoint"
        ]["sha256"],
        run_dir=run_dir,
        log_path=repository / "output/logs" / f"{arm['attempt_id']}.log",
        session_name=f"genmol_screen_{arm['attempt_id']}",
        command=command,
        resolved_config=resolved_config,
        resolved_config_sha256=config_entry["config"]["canonical_sha256"],
        argv_sha256=argv_sha256,
    )
    manifest_payload, manifest_sha256, _log_path = (
        screen_launcher._launch_locked_screen(
            plan,
            max_utilization_percent=10,
            min_free_memory_mib=30_000,
            lock_path=lock_path,
            lock_record=lock_record,
            lock_sha256=lock_sha256,
        )
    )
    manifest_path = run_dir / "launch_manifest.json"
    assert manifest_path.read_bytes() == manifest_payload

    conditioning_counts = None
    if arm_id == "E-A1":
        contract_groups = {
            group["group_id"]: group
            for group in harness.registry.data["stages"][1]["gradient_contract"][
                "groups"
            ]
        }
        conditioning_counts = (
            sum(
                math.prod(parameter["shape"])
                for parameter in contract_groups["timestep_mlp"]["parameters"]
            ),
            sum(
                math.prod(parameter["shape"])
                for parameter in contract_groups["film_modulation"]["parameters"]
            ),
        )
    model = _fake_model_and_checkpoint(
        checkpoint_path,
        optimizer_updates=optimizer_updates,
        conditioning_counts=conditioning_counts,
    )
    selected_uuids = [selected_uuid]
    manifest_snapshot, manifest = train_entrypoint._validate_launch_manifest(
        manifest_path,
        expected_sha256=manifest_sha256,
        expected_selected_gpu_uuids=selected_uuids,
    )
    runtime_path = run_dir / "runtime_config.json"
    summary_path = run_dir / "training_summary.json"
    pilot_contract = {
        "GENMOL_TRAIN_EXPECTED_SOURCE_REVISION": source_revision,
        "GENMOL_TRAIN_EXPECTED_CONFIG_SHA256": config_entry["config"][
            "canonical_sha256"
        ],
        "GENMOL_TRAIN_EXPECTED_ARGV_SHA256": argv_sha256,
        "GENMOL_TRAIN_RUNTIME_CONFIG_PATH": str(runtime_path),
        "GENMOL_TRAIN_SUMMARY_PATH": str(summary_path),
        "GENMOL_TRAIN_EXPECTED_SUMMARY_SCHEMA_VERSION": str(
            train_entrypoint._TRAINING_SUMMARY_SCHEMA_VERSION
        ),
        "GENMOL_TRAIN_EXPECTED_FINAL_CHECKPOINT_PATH": str(checkpoint_path),
        "GENMOL_TRAIN_EXPECTED_MAX_STEPS": str(optimizer_updates),
        "GENMOL_TRAIN_EXPECTED_WORLD_SIZE": "1",
        "GENMOL_TRAIN_LAUNCH_MANIFEST_PATH": str(manifest_path),
        "GENMOL_TRAIN_EXPECTED_LAUNCH_MANIFEST_SHA256": manifest_sha256,
        "GENMOL_TRAIN_SELECTED_GPU_UUIDS_JSON": json.dumps(selected_uuids),
        "expected_max_steps": optimizer_updates,
        "expected_world_size": 1,
        "summary_schema_version": train_entrypoint._TRAINING_SUMMARY_SCHEMA_VERSION,
        "runtime_path": runtime_path,
        "summary_path": summary_path,
        "final_checkpoint_path": checkpoint_path,
        "launch_manifest_path": manifest_path,
        "launch_manifest_snapshot": manifest_snapshot,
        "launch_manifest": manifest,
        "selected_gpu_uuids": selected_uuids,
    }
    monkeypatch.setattr(train_entrypoint, "_PILOT_CONTRACT", pilot_contract)
    monkeypatch.setattr(
        train_entrypoint,
        "_require_pilot_source_revision",
        lambda revision: {"head": revision, "upstream": revision},
    )
    # ``training_argv_sha256`` intentionally fingerprints the argv observed by
    # scripts/train.py, excluding the interpreter and ``-u`` prefix.
    monkeypatch.setattr(sys, "argv", command[2:])
    monkeypatch.setenv("PYTHONHASHSEED", "17")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", selected_uuid)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    config = OmegaConf.create(resolved_config)
    preflight = train_entrypoint._validate_and_record_pilot_config(config)
    warm_start = {
        "source_path": "/project/mdlm.ckpt",
        "source_resolved_path": "/project/mdlm.ckpt",
        "source_sha256": plan.checkpoint_sha256,
        "source_size_bytes": 1_396_998_679,
        "expected_source_sha256": plan.checkpoint_sha256,
        "byte_identity_verified_before_and_after_load": True,
        "weights": "ema",
        "parameter_tensors": 2,
    }
    state_audit = train_entrypoint._screen_initialization_state_audit(
        config, model, "warm_start", warm_start
    )
    health_callback = _completed_health_callback(optimizer_updates)
    callbacks = [health_callback]
    if arm_id == "E-A1":
        gradient_callback = train_entrypoint._FilmGradientActivationCallback(
            harness.registry.data["stages"][1]["gradient_contract"]
        )
        gradient_callback.optimizer_checks = cases._gradient_audit(
            harness, scheduler_arm_id=scheduler_arm_id or "E-L0"
        )["optimizer_checks"]
        callbacks.append(gradient_callback)
    trainer = SimpleNamespace(
        is_global_zero=True,
        global_rank=0,
        global_step=optimizer_updates,
        world_size=1,
        num_nodes=1,
        max_steps=optimizer_updates,
        accumulate_grad_batches=resolved_config["trainer"]["accumulate_grad_batches"],
        train_dataloader=SimpleNamespace(batch_size=4),
        callbacks=callbacks,
    )
    training_rng_policy = None
    if stage_id == "conditioning":
        training_rng_policy = {
            "policy": "reseed_all_training_rng_streams_after_model_and_warm_start",
            "seed": 17,
            "purpose": "isolate_training_randomness_from_architecture_constructor_draws",
            "applied_before_dataloader_and_trainer_construction": True,
        }
    train_entrypoint._write_pilot_training_summary(
        config=config,
        trainer=trainer,
        model=model,
        preflight_record=preflight,
        startup_mode="warm_start",
        warm_start_report=warm_start,
        screen_initialization_state_audit=state_audit,
        train_dataloader=trainer.train_dataloader,
        training_rng_policy=training_rng_policy,
    )

    receipt_path = run_dir / "pilot_exit_status.json"
    receipt_args = Namespace(
        training_exit_status=0,
        tee_exit_status=0,
        training_summary_path=summary_path,
        receipt_path=receipt_path,
        expected_summary_schema_version=receipt_writer.TRAINING_SUMMARY_SCHEMA_VERSION,
        expected_source_revision=source_revision,
        expected_config_sha256=config_entry["config"]["canonical_sha256"],
        expected_argv_sha256=argv_sha256,
        expected_launch_manifest_path=manifest_path,
        expected_launch_manifest_sha256=manifest_sha256,
        expected_selected_gpu_uuids=selected_uuids,
        training_job_lock_path=lock_path,
        expected_training_job_lock_sha256=lock_sha256,
        expected_max_steps=optimizer_updates,
        expected_world_size=1,
        expected_final_checkpoint_path=checkpoint_path,
        expected_initialization_checkpoint_sha256=plan.checkpoint_sha256,
    )
    receipt, exit_status = receipt_writer.build_exit_receipt(receipt_args)
    assert exit_status == 0
    receipt_writer._atomic_write_json_exclusive(receipt_path, receipt)
    pilot_launcher.release_exact_training_job_lock(
        lock_path, expected_sha256=lock_sha256
    )

    checkpoint_payload = checkpoint_path.read_bytes()
    checkpoint_ref = {
        "root": "repository",
        "relative_path": checkpoint_path.relative_to(repository).as_posix(),
        "sha256": hashlib.sha256(checkpoint_payload).hexdigest(),
        "size_bytes": len(checkpoint_payload),
        "global_step": optimizer_updates,
    }
    evaluator = cases._make_evaluator_report(
        harness,
        arm_id=arm_id,
        source_revision=source_revision,
        config_sha256=config_entry["config"]["canonical_sha256"],
        checkpoint=checkpoint_ref,
        losses=losses,
        correct=[8000, 8000, 8000],
    )
    evaluator_path = run_dir / "denoising_evaluator.json"
    evaluator_path.write_bytes(cases._bytes(evaluator))
    for relative_path in (
        f"{output_directory}/launch_manifest.json",
        f"{output_directory}/runtime_config.json",
        f"{output_directory}/training_summary.json",
        f"{output_directory}/pilot_exit_status.json",
        f"{output_directory}/checkpoints/{optimizer_updates}.ckpt",
        f"{output_directory}/denoising_evaluator.json",
    ):
        _add_file_to_harness(harness, repository, relative_path)


def _overlay_loader(harness: cases.Harness, bundle: collector.CollectionBundle):
    overlay = {
        ("repository", output.relative_path): output.payload
        for output in bundle.outputs
    }

    def load(root: str, relative_path: PurePosixPath):
        return overlay.get((root, relative_path.as_posix())) or harness.loader(
            root, relative_path
        )

    return load


def _retain_bundle_outputs(
    harness: cases.Harness, bundle: collector.CollectionBundle
) -> None:
    for output in bundle.outputs:
        harness.blobs[("repository", output.relative_path)] = output.payload


def _produce_initialization_audit(
    *,
    harness: cases.Harness,
    repository: Path,
    scheduler_arm_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    """Run the current CPU-only initialization-audit publication path."""

    assert harness.registry is not None
    stage = verifier._stage(harness.registry, "conditioning")
    fixture_ref = stage["initialization_fixture"]
    fixture = json.loads(
        harness.blobs[(fixture_ref["root"], fixture_ref["relative_path"])]
    )
    producer_ref = next(
        ref
        for ref in harness.registry.data["source"]["blobs"]
        if ref["relative_path"] == "scripts/udlm/audit_conditioning_initialization.py"
    )

    def arm_contract(arm_id: str):
        arm = verifier._arm(stage, arm_id)
        entry = verifier._registered_config(arm, scheduler_arm_id=scheduler_arm_id)
        return initialization_producer.ArmContract(
            arm_id=arm_id,
            output_directory=entry["output_directory"],
            config_ref=entry["config"],
            config=entry["parsed_config"],
        )

    checkpoint_ref = harness.registry.data["common_training"]["initialization"][
        "checkpoint"
    ]
    contract = initialization_producer.AuditContract(
        registry=harness.registry,
        source_revision=cases.AUTHORIZATION_REVISION,
        checkpoint_ref=checkpoint_ref,
        checkpoint_path=repository / "mdlm.ckpt",
        fixture_ref=fixture_ref,
        fixture=fixture,
        producer_ref=producer_ref,
        reference=arm_contract("E-A0"),
        candidate=arm_contract("E-A1"),
    )
    monkeypatch.setattr(initialization_producer, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(
        initialization_producer,
        "probe_registered_arm",
        lambda *_args, **_kwargs: (b"\x00\x00\x80?", [1]),
    )
    output_path = (
        repository / "output/udlm/screens/conditioning_initialization_audit.json"
    )
    audit = initialization_producer.run_audit(
        contract, output_path=output_path, stack=SimpleNamespace()
    )
    for ref in (audit["reference_logits"], audit["candidate_logits"]):
        _add_file_to_harness(harness, repository, ref["relative_path"])
    _add_file_to_harness(
        harness,
        repository,
        output_path.relative_to(repository).as_posix(),
    )
    return dict(audit)


def test_current_producers_feed_two_phase_collector_and_verifier_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    original_config_factory = cases._resolved_config
    monkeypatch.setattr(
        cases,
        "_resolved_config",
        _screen_config_factory(repository, original_config_factory),
    )
    harness = cases.build_harness(gpu_count=1)
    assert harness.registry is not None

    monkeypatch.setattr(screen_launcher, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(pilot_launcher, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(receipt_writer, "REPOSITORY_ROOT", repository)
    gpu_state = pilot_launcher.GPUState(
        physical_index=7,
        uuid="GPU-cpu-fixture",
        name="fixture only",
        memory_used_mib=0,
        memory_total_mib=80_000,
        utilization_percent=0,
        compute_mode="Default",
        compute_processes=(),
    )
    monkeypatch.setattr(pilot_launcher, "probe_all_gpus", lambda: [gpu_state])
    monkeypatch.setattr(
        pilot_launcher,
        "select_idle_gpus",
        lambda *_args, **_kwargs: (gpu_state,),
    )
    monkeypatch.setattr(
        pilot_launcher,
        "reprobe_selected_gpus",
        lambda *_args, **_kwargs: (gpu_state,),
    )
    monkeypatch.setattr(
        pilot_launcher,
        "build_child_environment_command",
        lambda **_kwargs: (["env", "python"], {}),
    )
    monkeypatch.setattr(
        pilot_launcher, "build_tmux_shell_command", lambda *_args, **_kwargs: "true"
    )
    monkeypatch.setattr(
        screen_launcher.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0),
    )
    monkeypatch.setattr(
        receipt_writer,
        "verify_clean_pushed_source",
        lambda revision: {
            "verified": True,
            "expected_revision": revision,
            "head": revision,
            "upstream": revision,
            "output_directory_excluded_from_cleanliness_check": True,
        },
    )

    _produce_screen_arm(
        harness=harness,
        repository=repository,
        stage_id="scheduler",
        arm_id="E-L0",
        source_revision=cases.PUBLICATION_REVISION,
        losses=[100.0, 100.0, 100.0],
        monkeypatch=monkeypatch,
    )
    _produce_screen_arm(
        harness=harness,
        repository=repository,
        stage_id="scheduler",
        arm_id="E-L1",
        source_revision=cases.PUBLICATION_REVISION,
        losses=[98.0, 98.0, 98.0],
        monkeypatch=monkeypatch,
    )

    bundle = collector.build_collection_bundle(
        registry=harness.registry,
        stage_id="scheduler",
        run_source_revision=cases.PUBLICATION_REVISION,
        evidence_relative_path="experiments/udlm/screens/e2e_scheduler.json",
        loader=harness.loader,
    )
    decision = verifier.evaluate_evidence_bytes(
        bundle.outputs[-1].payload,
        evidence_relative_path=bundle.evidence_relative_path,
        stage_id="scheduler",
        registry=harness.registry,
        loader=_overlay_loader(harness, bundle),
    )
    assert decision == bundle.decision
    assert decision["status"] == "completed"
    assert decision["selected_arm_id"] == "E-L1"
    for attempt in bundle.evidence["attempts"]:
        receipt_ref = attempt["artifacts"]["exit_receipt"]
        receipt = json.loads(
            harness.blobs[(receipt_ref["root"], receipt_ref["relative_path"])]
        )
        assert receipt["training_summary"]["valid_and_launch_bound"] is True
        assert (
            receipt["training_summary"]["validated_bindings"][
                "screen_initialization_state_audit"
            ]
            == attempt["initialization"]["state_audit"]
        )

    _retain_bundle_outputs(harness, bundle)
    harness.add_blob(
        "repository",
        bundle.evidence_relative_path,
        bundle.outputs[-1].payload,
        revisions=(cases.AUTHORIZATION_REVISION,),
    )
    scheduler_selection_path = "experiments/udlm/screens/e2e_scheduler_selection.json"
    harness.add_blob(
        "repository",
        scheduler_selection_path,
        cases._bytes(bundle.decision),
        revisions=(cases.AUTHORIZATION_REVISION,),
    )
    dependency_declaration, scheduler_arm_id, scheduler_dependency = (
        collector.derive_scheduler_dependency(
            registry=harness.registry,
            authorization_revision=cases.AUTHORIZATION_REVISION,
            scheduler_evidence_relative_path=bundle.evidence_relative_path,
            scheduler_selection_relative_path=scheduler_selection_path,
            loader=harness.loader,
        )
    )
    assert scheduler_arm_id == "E-L1"

    for arm_id, losses in (
        ("E-A0", [100.0, 100.0, 100.0]),
        ("E-A1", [98.0, 98.0, 98.0]),
    ):
        _produce_screen_arm(
            harness=harness,
            repository=repository,
            stage_id="conditioning",
            arm_id=arm_id,
            source_revision=cases.AUTHORIZATION_REVISION,
            losses=losses,
            monkeypatch=monkeypatch,
            scheduler_arm_id=scheduler_arm_id,
            scheduler_dependency=dict(scheduler_dependency),
        )
    initialization_audit = _produce_initialization_audit(
        harness=harness,
        repository=repository,
        scheduler_arm_id=scheduler_arm_id,
        monkeypatch=monkeypatch,
    )
    conditioning_bundle = collector.build_collection_bundle(
        registry=harness.registry,
        stage_id="conditioning",
        run_source_revision=cases.AUTHORIZATION_REVISION,
        evidence_relative_path="experiments/udlm/screens/e2e_conditioning.json",
        loader=harness.loader,
        scheduler_dependency_declaration=dependency_declaration,
        scheduler_arm_id=scheduler_arm_id,
        initialization_audit=initialization_audit,
    )
    conditioning_decision = verifier.evaluate_evidence_bytes(
        conditioning_bundle.outputs[-1].payload,
        evidence_relative_path=conditioning_bundle.evidence_relative_path,
        stage_id="conditioning",
        registry=harness.registry,
        loader=_overlay_loader(harness, conditioning_bundle),
    )
    assert conditioning_decision == conditioning_bundle.decision
    assert conditioning_decision["status"] == "completed"
    assert conditioning_decision["selected_arm_id"] == "E-A1"
    assert conditioning_decision["dependency"]["selected_scheduler_arm_id"] == "E-L1"

    first_receipt_ref = bundle.evidence["attempts"][0]["artifacts"]["exit_receipt"]
    receipt_key = (
        first_receipt_ref["root"],
        first_receipt_ref["relative_path"],
    )
    tampered_receipt = json.loads(harness.blobs[receipt_key])
    tampered_receipt["training_summary"]["validated_bindings"][
        "screen_initialization_state_audit"
    ]["full_initial_state_sha256"] = "0" * 64
    harness.blobs[receipt_key] = cases._bytes(tampered_receipt)
    rejected = verifier.evaluate_evidence_bytes(
        conditioning_bundle.outputs[-1].payload,
        evidence_relative_path=conditioning_bundle.evidence_relative_path,
        stage_id="conditioning",
        registry=harness.registry,
        loader=_overlay_loader(harness, conditioning_bundle),
    )
    assert rejected["status"] == "incomplete"
    assert rejected["selected_arm_id"] is None
    assert rejected["complete_threshold_fallback_used"] is False
    with pytest.raises(verifier.ScreenValidationError):
        collector.build_collection_bundle(
            registry=harness.registry,
            stage_id="conditioning",
            run_source_revision=cases.AUTHORIZATION_REVISION,
            evidence_relative_path="experiments/udlm/screens/tampered_e2e.json",
            loader=harness.loader,
            scheduler_dependency_declaration=dependency_declaration,
            scheduler_arm_id=scheduler_arm_id,
            initialization_audit=initialization_audit,
        )

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from scripts.udlm import verify_optimization_screen as screen


HEALTH_REVISION = "d" * 40
REGISTRY_REVISION = "a" * 40
PUBLICATION_REVISION = "b" * 40
AUTHORIZATION_REVISION = "c" * 40
REGISTRY_PATH = "experiments/udlm/protocols/optimization_screen_registry_v2.json"
FIXTURE_PATH = Path(__file__).parents[1] / screen.EXPECTED_INITIALIZATION_FIXTURE_PATH


def _bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass
class Harness:
    blobs: dict[tuple[str, str], bytes | screen.BlobSnapshot] = field(
        default_factory=dict
    )
    git_blobs: dict[tuple[str, str], bytes] = field(default_factory=dict)
    registry_document: dict[str, Any] | None = None
    registry_payload: bytes | None = None
    registry: screen.ValidatedRegistry | None = None
    validated_health_evidence: dict[str, Any] | None = None

    def add_blob(
        self,
        root: str,
        path: str,
        payload: bytes,
        *,
        revisions: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        self.blobs[(root, path)] = payload
        if root == "repository":
            for revision in revisions:
                self.git_blobs[(revision, path)] = payload
        return {
            "root": root,
            "relative_path": path,
            "sha256": _sha(payload),
            "size_bytes": len(payload),
        }

    def add_json(
        self,
        root: str,
        path: str,
        value: object,
        schema_version: int,
        *,
        revisions: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        payload = _bytes(value)
        ref = self.add_blob(root, path, payload, revisions=revisions)
        return {
            **ref,
            "schema_version": schema_version,
            "canonical_sha256": screen.canonical_json_sha256(value),
        }

    def add_config(
        self,
        path: str,
        value: object,
        *,
        revisions: tuple[str, ...] = (
            REGISTRY_REVISION,
            PUBLICATION_REVISION,
            AUTHORIZATION_REVISION,
        ),
    ) -> dict[str, Any]:
        payload = _bytes(value)
        ref = self.add_blob("repository", path, payload, revisions=revisions)
        return {**ref, "canonical_sha256": screen.canonical_json_sha256(value)}

    def loader(self, root: str, path: PurePosixPath) -> bytes | screen.BlobSnapshot:
        return self.blobs[(root, path.as_posix())]

    def git_loader(self, revision: str, path: PurePosixPath) -> bytes:
        return self.git_blobs[(revision, path.as_posix())]

    @staticmethod
    def ancestor(ancestor: str, descendant: str) -> bool:
        return (
            ancestor == descendant
            or (
                ancestor == HEALTH_REVISION
                and descendant
                in {
                    REGISTRY_REVISION,
                    PUBLICATION_REVISION,
                    AUTHORIZATION_REVISION,
                }
            )
            or (
                ancestor == REGISTRY_REVISION
                and descendant in {PUBLICATION_REVISION, AUTHORIZATION_REVISION}
            )
            or (
                ancestor == PUBLICATION_REVISION
                and descendant == AUTHORIZATION_REVISION
            )
        )

    @staticmethod
    def sole_parent(revision: str, expected_parent: str) -> bool:
        parents = {
            REGISTRY_REVISION: (HEALTH_REVISION,),
            PUBLICATION_REVISION: (REGISTRY_REVISION,),
            AUTHORIZATION_REVISION: (PUBLICATION_REVISION,),
        }
        return parents.get(revision) == (expected_parent,)

    def tree_paths(self, revision: str, directory: PurePosixPath) -> frozenset[str]:
        prefix = directory.as_posix() + "/"
        return frozenset(
            path
            for observed_revision, path in self.git_blobs
            if observed_revision == revision and path.startswith(prefix)
        )

    @staticmethod
    def pushed(_revision: str) -> bool:
        return True

    @staticmethod
    def allowed_diff(
        _ancestor: str, _descendant: str, _allowed_paths: frozenset[str]
    ) -> bool:
        return True

    def health_validator(
        self,
        terminal_receipt_path: Path,
        *,
        expected_gpu_count: int,
        expected_source_revision: str,
    ) -> dict[str, Any]:
        assert self.validated_health_evidence is not None
        evidence = self.validated_health_evidence
        assert terminal_receipt_path == (
            screen.REPOSITORY_ROOT / evidence["terminal_receipt"]["relative_path"]
        )
        assert expected_gpu_count == evidence["gpu_count"]
        assert expected_source_revision == evidence["health_source_revision"]
        return copy.deepcopy(evidence)

    def freeze(self) -> screen.ValidatedRegistry:
        assert self.registry_document is not None
        payload = _bytes(self.registry_document)
        self.registry_payload = payload
        self.git_blobs[(PUBLICATION_REVISION, REGISTRY_PATH)] = payload
        self.git_blobs[(AUTHORIZATION_REVISION, REGISTRY_PATH)] = payload
        self.registry = screen.load_validated_registry(
            payload,
            relative_path=REGISTRY_PATH,
            expected_raw_sha256=_sha(payload),
            expected_canonical_sha256=screen.canonical_json_sha256(
                self.registry_document
            ),
            loader=self.loader,
            git_blob_loader=self.git_loader,
            git_ancestor_checker=self.ancestor,
            git_sole_parent_checker=self.sole_parent,
            git_tree_paths_loader=self.tree_paths,
            git_pushed_checker=self.pushed,
            git_diff_checker=self.allowed_diff,
            health_gate_validator=self.health_validator,
        )
        return self.registry


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


def _resolved_config(
    *,
    arm_id: str,
    scheduler_arm_id: str,
    updates: int,
    output_directory: str,
    checkpoint_path: str,
    checkpoint_sha256: str,
    gpu_count: int,
) -> dict[str, Any]:
    l1 = scheduler_arm_id == "E-L1"
    return {
        "seed": 17,
        "training": {
            "diffusion": "udlm",
            "init_from_mdlm_checkpoint": f"/project/{checkpoint_path}",
            "init_from_mdlm_checkpoint_sha256": checkpoint_sha256,
            "init_from_mdlm_ema": True,
            "reseed_after_model_initialization": updates == 500,
            "udlm": {
                "prior_variant": "empirical_frequency",
                "empirical_uniform_mix": 0.0002,
                "conditioning_variant": (
                    "film_adaln" if arm_id == "E-A1" else "additive"
                ),
                "zero_init_conditioning": arm_id != "E-A1",
            },
        },
        "loader": {"global_batch_size": 32, "batch_size": 4},
        "trainer": {
            "accelerator": "cuda",
            "num_nodes": 1,
            "devices": gpu_count,
            "accumulate_grad_batches": 8 // gpu_count,
            "max_steps": updates,
        },
        "optim": {
            "lr": 0.0003,
            "scheduler": {
                "name": (
                    "half_cosine_with_linear_warmup_and_floor"
                    if l1
                    else "constant_with_linear_warmup"
                ),
                "warmup_updates": 50 if l1 else 2500,
                "horizon_updates": 1000 if l1 else None,
                "decay_floor_lr": 0.000003 if l1 else None,
            },
        },
        "callback": {"dirpath": f"/repo/{output_directory}/checkpoints"},
    }


def _source_ref(harness: Harness, path: str) -> dict[str, Any]:
    if path in {
        screen.EXPECTED_PRIOR_FLOOR_AUDIT_SOURCE_PATH,
        screen.EXPECTED_PRIOR_FLOOR_AUDIT_PATH,
    }:
        payload = (screen.REPOSITORY_ROOT / path).read_bytes()
    else:
        payload = f"source:{path}\n".encode()
    return harness.add_blob(
        "repository",
        path,
        payload,
        revisions=(
            REGISTRY_REVISION,
            PUBLICATION_REVISION,
            AUTHORIZATION_REVISION,
        ),
    )


def _health_gate(harness: Harness, *, gpu_count: int) -> dict[str, Any]:
    variants = screen.EXPECTED_HEALTH_VARIANT_ORDER
    slugs = screen.EXPECTED_HEALTH_VARIANT_SLUGS
    run_names = [f"health-w{gpu_count}-{slug}-{HEALTH_REVISION}" for slug in slugs]
    terminal_receipt_value = {
        "schema_version": screen.EXPECTED_HEALTH_RECEIPT_SCHEMA_VERSION,
        "status": "success",
        "recorded_at_utc": "2026-09-06T00:00:02Z",
    }
    terminal_ref = harness.add_json(
        "repository",
        f"output/udlm/{run_names[-1]}/pilot_exit_status.json",
        terminal_receipt_value,
        screen.EXPECTED_HEALTH_RECEIPT_SCHEMA_VERSION,
    )
    receipt_members = []
    checkpoint_members = []
    for position, (variant, run_name) in enumerate(
        zip(variants, run_names, strict=True)
    ):
        receipt_sha256 = (
            terminal_ref["sha256"]
            if position == 2
            else _sha(f"health-receipt-{position}\n".encode())
        )
        receipt_members.append(
            {
                "position": position,
                "training_variant": variant,
                "run_name": run_name,
                "relative_path": (f"output/udlm/{run_name}/pilot_exit_status.json"),
                "sha256": receipt_sha256,
                "schema_version": screen.EXPECTED_HEALTH_RECEIPT_SCHEMA_VERSION,
                "recorded_at_utc": f"2026-09-06T00:00:0{position}Z",
            }
        )
        checkpoint_members.append(
            {
                "position": position,
                "training_variant": variant,
                "run_name": run_name,
                "relative_path": f"output/udlm/{run_name}/checkpoints/10.ckpt",
                "sha256": _sha(f"health-checkpoint-{position}\n".encode()),
                "size_bytes": 1000 + position,
                "global_step": 10,
            }
        )
    evidence = {
        "schema_version": screen.EXPECTED_HEALTH_SCHEMA_VERSION,
        "status": screen.EXPECTED_HEALTH_STATUS,
        "claim_scope": screen.EXPECTED_HEALTH_CLAIM_SCOPE,
        "health_source_revision": HEALTH_REVISION,
        "gpu_count": gpu_count,
        "matched_panel_spec_sha256": _sha(b"health-panel\n"),
        "terminal_receipt": {
            **terminal_ref,
            "training_variant": variants[-1],
            "position": 2,
        },
        "receipt_members": receipt_members,
        "checkpoint_members": checkpoint_members,
        "eligibility": dict(screen.EXPECTED_HEALTH_ELIGIBILITY),
    }
    harness.validated_health_evidence = copy.deepcopy(evidence)
    config_paths = list(screen._expected_screen_config_paths(gpu_count))
    return {
        "evidence": evidence,
        "source_transition": {
            "health_source_revision": HEALTH_REVISION,
            "registry_source_revision": REGISTRY_REVISION,
            "allowed_config_paths": config_paths,
            "health_source_is_registry_source_parent": True,
            "exact_config_only_transition_verified": True,
            "opposite_gpu_config_family_absent": True,
        },
    }


def build_harness(*, gpu_count: int = 1) -> Harness:
    harness = Harness()
    required_sources = (
        "configs/base.yaml",
        "configs/udlm.yaml",
        "configs/udlm_categorical.yaml",
        "scripts/train.py",
        "scripts/udlm/audit_conditioning_initialization.py",
        screen.EXPECTED_PRIOR_FLOOR_AUDIT_SOURCE_PATH,
        screen.EXPECTED_PRIOR_FLOOR_AUDIT_PATH,
        "scripts/udlm/collect_optimization_screen_evidence.py",
        "scripts/udlm/evaluate_denoising_panel.py",
        "scripts/udlm/launch_health_panel.py",
        "scripts/udlm/launch_optimization_screen.py",
        "scripts/udlm/launch_train_pilot.py",
        "scripts/udlm/validate_health_panel.py",
        "scripts/udlm/verify_optimization_screen.py",
        "scripts/udlm/write_pilot_evidence.py",
        "scripts/udlm/write_pilot_exit_status.py",
        "src/genmol/backbone.py",
        "src/genmol/diffusion.py",
        "src/genmol/model.py",
        "src/genmol/utils/ema.py",
        "src/genmol/utils/utils_data.py",
        "scripts/udlm/materialize_validation_panel.py",
        "scripts/udlm/token_frequency_audit.py",
    )
    source_refs = [_source_ref(harness, path) for path in required_sources]

    repository = screen.REPOSITORY_ROOT
    panel_payload = (repository / screen.EXPECTED_PANEL_PATH).read_bytes()
    panel_value = json.loads(panel_payload)
    harness.blobs[("repository", screen.EXPECTED_PANEL_PATH)] = panel_payload
    for revision in (
        REGISTRY_REVISION,
        PUBLICATION_REVISION,
        AUTHORIZATION_REVISION,
    ):
        harness.git_blobs[(revision, screen.EXPECTED_PANEL_PATH)] = panel_payload
    panel_ref = {
        "root": "repository",
        "relative_path": screen.EXPECTED_PANEL_PATH,
        "sha256": _sha(panel_payload),
        "size_bytes": len(panel_payload),
        "schema_version": 1,
        "canonical_sha256": screen.canonical_json_sha256(panel_value),
    }
    frequency_payload = (repository / screen.EXPECTED_FREQUENCY_PATH).read_bytes()
    frequency_value = json.loads(frequency_payload)
    harness.blobs[("repository", screen.EXPECTED_FREQUENCY_PATH)] = frequency_payload
    for revision in (
        REGISTRY_REVISION,
        PUBLICATION_REVISION,
        AUTHORIZATION_REVISION,
    ):
        harness.git_blobs[(revision, screen.EXPECTED_FREQUENCY_PATH)] = (
            frequency_payload
        )
    frequency_ref = {
        "root": "repository",
        "relative_path": screen.EXPECTED_FREQUENCY_PATH,
        "sha256": _sha(frequency_payload),
        "size_bytes": len(frequency_payload),
        "schema_version": 1,
        "canonical_sha256": screen.canonical_json_sha256(frequency_value),
    }
    contract_payload = (
        repository / screen.EXPECTED_GRADIENT_CONTRACT_PATH
    ).read_bytes()
    contract_value = json.loads(contract_payload)
    harness.blobs[("repository", screen.EXPECTED_GRADIENT_CONTRACT_PATH)] = (
        contract_payload
    )
    for revision in (
        REGISTRY_REVISION,
        PUBLICATION_REVISION,
        AUTHORIZATION_REVISION,
    ):
        harness.git_blobs[(revision, screen.EXPECTED_GRADIENT_CONTRACT_PATH)] = (
            contract_payload
        )
    contract_ref = {
        "root": "repository",
        "relative_path": screen.EXPECTED_GRADIENT_CONTRACT_PATH,
        "sha256": _sha(contract_payload),
        "size_bytes": len(contract_payload),
        "schema_version": 1,
        "canonical_sha256": screen.canonical_json_sha256(contract_value),
    }

    init_ref = {
        "root": "project",
        "relative_path": "outputs/paper_v1/checkpoints/50000.ckpt",
        "sha256": ("8d00aa47b02f64bf39ff6b0b2e786f213587366fc2c3d29712a00f3f84108dd6"),
        "size_bytes": 1_396_998_679,
    }
    harness.blobs[("project", init_ref["relative_path"])] = screen.BlobSnapshot(
        size_bytes=init_ref["size_bytes"], sha256=init_ref["sha256"]
    )
    fixture_payload = FIXTURE_PATH.read_bytes()
    fixture_value = json.loads(fixture_payload)
    assert len(fixture_payload) == screen.EXPECTED_INITIALIZATION_FIXTURE_SIZE_BYTES
    assert _sha(fixture_payload) == screen.EXPECTED_INITIALIZATION_FIXTURE_SHA256
    assert (
        screen.canonical_json_sha256(fixture_value)
        == screen.EXPECTED_INITIALIZATION_FIXTURE_CANONICAL_SHA256
    )
    fixture_ref = {
        "root": "repository",
        "relative_path": screen.EXPECTED_INITIALIZATION_FIXTURE_PATH,
        "sha256": screen.EXPECTED_INITIALIZATION_FIXTURE_SHA256,
        "size_bytes": screen.EXPECTED_INITIALIZATION_FIXTURE_SIZE_BYTES,
        "schema_version": 1,
        "canonical_sha256": screen.EXPECTED_INITIALIZATION_FIXTURE_CANONICAL_SHA256,
    }
    harness.blobs[("repository", fixture_ref["relative_path"])] = fixture_payload
    for revision in (
        REGISTRY_REVISION,
        PUBLICATION_REVISION,
        AUTHORIZATION_REVISION,
    ):
        harness.git_blobs[(revision, fixture_ref["relative_path"])] = fixture_payload

    config_filenames = {
        ("E-L0", None): "scheduler_e_l0.json",
        ("E-L1", None): "scheduler_e_l1.json",
        ("E-A0", "E-L0"): "conditioning_e_a0__e_l0.json",
        ("E-A0", "E-L1"): "conditioning_e_a0__e_l1.json",
        ("E-A1", "E-L0"): "conditioning_e_a1__e_l0.json",
        ("E-A1", "E-L1"): "conditioning_e_a1__e_l1.json",
    }
    config_directory = screen.EXPECTED_SCREEN_CONFIG_DIRECTORY_TEMPLATE.format(
        gpu_count=gpu_count
    )
    config_entries: dict[tuple[str, str | None], dict[str, Any]] = {}
    for stage_id, arms in (
        ("scheduler", ("E-L0", "E-L1")),
        ("conditioning", ("E-A0", "E-A1")),
    ):
        scheduler_ids: tuple[str | None, ...] = (
            (None,) if stage_id == "scheduler" else ("E-L0", "E-L1")
        )
        for arm_id in arms:
            for contingent in scheduler_ids:
                selected_scheduler = arm_id if contingent is None else contingent
                suffix = "" if contingent is None else f"_{contingent.lower()}"
                output_directory = f"output/udlm/screens/{arm_id.lower()}{suffix}"
                config = _resolved_config(
                    arm_id=arm_id,
                    scheduler_arm_id=selected_scheduler,
                    updates=screen.EXPECTED_UPDATES[stage_id],
                    output_directory=output_directory,
                    checkpoint_path=init_ref["relative_path"],
                    checkpoint_sha256=init_ref["sha256"],
                    gpu_count=gpu_count,
                )
                config_ref = harness.add_config(
                    f"{config_directory}/{config_filenames[(arm_id, contingent)]}",
                    config,
                )
                config_entries[(arm_id, contingent)] = {
                    "scheduler_arm_id": contingent,
                    "output_directory": output_directory,
                    "config": config_ref,
                }

    scheduler_stage = {
        "stage_id": "scheduler",
        "order_index": 0,
        "training_seed": 17,
        "optimizer_updates": 100,
        "arm_order": ["E-L0", "E-L1"],
        "arms": [
            {
                "arm_id": arm_id,
                "attempt_id": f"scheduler-{arm_id.lower()}",
                "role": "control" if index == 0 else "candidate",
                "scheduler": _scheduler(arm_id),
                "conditioner": _conditioner(False),
                "resolved_configs": [config_entries[(arm_id, None)]],
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
                "attempt_id": f"conditioning-{arm_id.lower()}",
                "role": "control" if index == 0 else "candidate",
                "scheduler": "selected_scheduler_arm",
                "conditioner": _conditioner(index == 1),
                "resolved_configs": [
                    config_entries[(arm_id, scheduler_id)]
                    for scheduler_id in ("E-L0", "E-L1")
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
            "gradient_contract_sha256": screen.canonical_json_sha256(contract_value),
            "complete_failure_fallback_arm": "E-A0",
            "incomplete_evidence_winner": None,
        },
        "gradient_contract": contract_ref,
        "initialization_fixture": fixture_ref,
    }
    evaluator_ref = next(
        ref
        for ref in source_refs
        if ref["relative_path"] == "scripts/udlm/evaluate_denoising_panel.py"
    )
    prerequisite_health_gate = _health_gate(harness, gpu_count=gpu_count)
    harness.registry_document = {
        "schema_version": screen.REGISTRY_SCHEMA_VERSION,
        "registry_id": screen.EXPECTED_REGISTRY_ID,
        "status": screen.EXPECTED_REGISTRY_STATUS,
        "claim_scope": screen.EXPECTED_CLAIM_SCOPE,
        "firewall": {
            "final_generation_seeds": [0, 1, 2],
            "final_generation_seeds_forbidden": True,
            "generation_metrics_allowed": False,
            "health_gate_evidence_eligible": False,
            "superiority_evidence_eligible": False,
            "unregistered_attempts_allowed": False,
            "failed_or_missing_evidence_policy": "incomplete_no_winner",
        },
        "source": {
            "revision": REGISTRY_REVISION,
            "clean": True,
            "pushed": True,
            "blobs": source_refs,
        },
        "prerequisite_health_gate": prerequisite_health_gate,
        "common_training": {
            "training_seed": 17,
            "gpu_count": gpu_count,
            "prior_variant": "empirical_frequency",
            "global_batch_size": 32,
            "micro_batch_size_per_process": 4,
            "accumulate_grad_batches": 8 // gpu_count,
            "effective_global_batch_size": 32,
            "initialization": {
                "mode": "fresh_independent_mdlm_ema_warm_start_each_arm",
                "checkpoint": init_ref,
                "weights": "ema",
                "optimizer_reset": True,
                "scheduler_reset": True,
                "global_step_reset": True,
                "ema_reset": True,
            },
            "artifact_schema_versions": dict(screen.EXPECTED_ARTIFACT_SCHEMA_VERSIONS),
        },
        "panel": {
            "artifact": panel_ref,
            "ordered_token_ids_sha256": screen.EXPECTED_PANEL_TOKEN_IDS_SHA256,
            "rows": 256,
            "content_tokens_per_time_bin": 13627,
            "time_bins": [0.1, 0.5, 0.9],
            "corruption_seed": 17,
            "weights": "ema",
            "device": "cpu",
            "batch_size": 4,
            "frequency_artifact": frequency_ref,
            "frequency_ordered_text_sha256": (
                screen.EXPECTED_FREQUENCY_ORDERED_TEXT_SHA256
            ),
            "evaluator_report_schema_version": 4,
            "evaluator_source": evaluator_ref,
        },
        "stages": [scheduler_stage, conditioning_stage],
    }
    harness.freeze()
    return harness


def _artifact_json(
    harness: Harness,
    output_directory: str,
    name: str,
    value: object,
    schema_version: int,
) -> dict[str, Any]:
    return harness.add_json(
        "repository",
        f"{output_directory}/{name}.json",
        value,
        schema_version,
    )


def _load_registry_document(
    harness: Harness,
    document: Mapping[str, Any],
    *,
    git_ancestor_checker=None,
    git_sole_parent_checker=None,
    git_tree_paths_loader=None,
    git_diff_checker=None,
    health_gate_validator=None,
) -> screen.ValidatedRegistry:
    payload = _bytes(document)
    return screen.load_validated_registry(
        payload,
        relative_path=REGISTRY_PATH,
        expected_raw_sha256=_sha(payload),
        expected_canonical_sha256=screen.canonical_json_sha256(document),
        loader=harness.loader,
        git_blob_loader=harness.git_loader,
        git_ancestor_checker=(
            harness.ancestor if git_ancestor_checker is None else git_ancestor_checker
        ),
        git_sole_parent_checker=(
            harness.sole_parent
            if git_sole_parent_checker is None
            else git_sole_parent_checker
        ),
        git_tree_paths_loader=(
            harness.tree_paths
            if git_tree_paths_loader is None
            else git_tree_paths_loader
        ),
        git_pushed_checker=harness.pushed,
        git_diff_checker=(
            harness.allowed_diff if git_diff_checker is None else git_diff_checker
        ),
        health_gate_validator=(
            harness.health_validator
            if health_gate_validator is None
            else health_gate_validator
        ),
    )


def _gradient_audit(
    harness: Harness, *, scheduler_arm_id: str, passing: bool = True
) -> dict[str, Any]:
    assert harness.registry is not None
    stage = harness.registry.data["stages"][1]
    contract = stage["gradient_contract"]
    checks = []
    warmup = 2500 if scheduler_arm_id == "E-L0" else 50
    for index in (1, 2, 3):
        collections: dict[str, list[dict[str, Any]]] = {
            "film_groups": [],
            "timestep_mlp_groups": [],
        }
        for group in contract["groups"]:
            kind = group["kind"]
            nonzero = True
            if not passing and kind == "timestep_mlp" and index == 3:
                nonzero = False
            collections[
                "film_groups" if kind == "film" else "timestep_mlp_groups"
            ].append(
                {
                    "group_id": group["group_id"],
                    "ordered_parameter_manifest_sha256": (
                        screen.canonical_json_sha256(group["parameters"])
                    ),
                    "parameter_count": len(group["parameters"]),
                    "gradient_element_count": sum(
                        math.prod(parameter["shape"])
                        for parameter in group["parameters"]
                    ),
                    "all_gradients_present": True,
                    "all_gradients_finite": True,
                    "all_parameter_gradients_nonzero": nonzero,
                }
            )
        checks.append(
            {
                "optimizer_gradient_observation_index": index,
                "optimizer_step_index": index,
                "learning_rate_before_step": 0.0003 * ((index - 1) / warmup),
                **collections,
            }
        )
    return {
        "schema_version": 1,
        "status": "completed",
        "observation_point": screen.EXPECTED_OBSERVATION_POINT,
        "registered_contract_sha256": stage["gradient_contract_sha256"],
        "first_positive_lr_optimizer_step": 2,
        "timestep_mlp_required_optimizer_check": 3,
        "optimizer_checks": checks,
    }


def _conditioning_record(
    harness: Harness, arm_id: str
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    validation = {
        "before_strict_state_load": True,
        "strict_state_load": True,
        "after_strict_state_load": True,
        "runtime_identity_unchanged": True,
    }
    if arm_id != "E-A1":
        return (
            {
                "runtime_conditioning_variant": "additive",
                "checkpoint_metadata_required": False,
                "checkpoint_metadata_present": False,
                "checkpoint_metadata_validation": "correctly_absent_for_additive",
                "runtime_metadata": None,
                "runtime_metadata_canonical_sha256": None,
                "checkpoint_metadata": None,
                "checkpoint_metadata_canonical_sha256": None,
                "validation": validation,
            },
            None,
        )
    assert harness.registry is not None
    manifest = [
        {
            "name": parameter["name"].removeprefix("backbone."),
            "shape": parameter["shape"],
        }
        for group in harness.registry.data["stages"][1]["gradient_contract"]["groups"]
        for parameter in group["parameters"]
    ]
    metadata = {
        "schema_version": 1,
        "variant": "film_adaln",
        "hidden_size": 768,
        "layer_count": 12,
        "conditioning_parameter_manifest": manifest,
    }
    digest = screen.canonical_json_sha256(metadata)
    return (
        {
            "runtime_conditioning_variant": "film_adaln",
            "checkpoint_metadata_required": True,
            "checkpoint_metadata_present": True,
            "checkpoint_metadata_validation": "required_record_matches_runtime_exactly",
            "runtime_metadata": metadata,
            "runtime_metadata_canonical_sha256": digest,
            "checkpoint_metadata": copy.deepcopy(metadata),
            "checkpoint_metadata_canonical_sha256": digest,
            "validation": validation,
        },
        metadata,
    )


def _state_audit(
    harness: Harness,
    *,
    arm_id: str,
    config_sha256: str,
    stage_id: str,
) -> dict[str, Any]:
    assert harness.registry is not None
    common_digest = "5" * 64 if stage_id == "scheduler" else "6" * 64
    full_digest = (
        "5" * 64
        if stage_id == "scheduler"
        else ("7" * 64 if arm_id == "E-A0" else "8" * 64)
    )
    return {
        "schema_version": 1,
        "phase": (
            "after_verified_mdlm_ema_warm_start_before_training_rng_reseed_"
            "and_optimizer_creation"
        ),
        "source_checkpoint_sha256": harness.registry.data["common_training"][
            "initialization"
        ]["checkpoint"]["sha256"],
        "resolved_training_config_sha256": config_sha256,
        "training_seed": 17,
        "conditioning_variant": "film_adaln" if arm_id == "E-A1" else "additive",
        "common_backbone_tensor_count": 200,
        "common_backbone_state_sha256": common_digest,
        "full_initial_tensor_count": 228 if arm_id == "E-A1" else 200,
        "full_initial_state_sha256": full_digest,
    }


def _make_evaluator_report(
    harness: Harness,
    *,
    arm_id: str,
    source_revision: str,
    config_sha256: str,
    checkpoint: dict[str, Any],
    losses: list[float],
    correct: list[int],
) -> dict[str, Any]:
    assert harness.registry is not None
    conditioning, metadata = _conditioning_record(harness, arm_id)
    grid = [
        {
            "time": time,
            "row_corruption_seeds_sha256": str(index + 1) * 64,
            "corrupted_token_ids_sha256": chr(ord("a") + index) * 64,
        }
        for index, time in enumerate((0.1, 0.5, 0.9))
    ]
    bins = [
        {
            **item,
            "metrics": {
                "overall": {
                    "production_loss_sum": loss,
                    "denominator_tokens": 13627,
                    "clean_token_top1_correct": hits,
                }
            },
        }
        for item, loss, hits in zip(grid, losses, correct, strict=True)
    ]
    registry_hashes = {
        ref["relative_path"]: ref["sha256"]
        for ref in harness.registry.data["source"]["blobs"]
    }
    source_hashes = {
        path: registry_hashes[path] for path in screen.EVALUATOR_SOURCE_PATHS
    }
    return {
        "schema_version": 4,
        "conditioning": conditioning,
        "source": {
            "git_commit": source_revision,
            "git_dirty": False,
            "git_worktree_state": "clean",
            "files_sha256": source_hashes,
            "postcheck": {"status": "unchanged_after_checkpoint_load_and_evaluation"},
        },
        "checkpoint": {
            "sha256": checkpoint["sha256"],
            "size_bytes": checkpoint["size_bytes"],
            "global_step": checkpoint["global_step"],
            "config_sha256": config_sha256,
            "weights_evaluated": "ema",
            "diffusion_type": "udlm",
            "udlm_conditioning_metadata_declared": metadata is not None,
            "udlm_conditioning_metadata": metadata,
            "udlm_conditioning_metadata_sha256": (
                None if metadata is None else screen.canonical_json_sha256(metadata)
            ),
            "weight_application": {
                "source": "checkpoint.ema.shadow_params",
                "udlm_conditioning_identity": conditioning,
            },
        },
        "artifacts": {
            "panel": {
                "sha256": harness.registry.data["panel"]["artifact"]["sha256"],
                "ordered_token_ids_sha256": screen.EXPECTED_PANEL_TOKEN_IDS_SHA256,
            },
            "training_frequency": {
                "sha256": harness.registry.data["panel"]["frequency_artifact"]["sha256"]
            },
        },
        "evaluation": {
            "rows_evaluated": 256,
            "content_tokens_per_time_bin": 13627,
            "time_bins": [0.1, 0.5, 0.9],
            "seed": 17,
            "batch_size": 4,
            "device": "cpu",
            "corruption_grid_sha256": screen.canonical_json_sha256(grid),
            "process": {"prior_variant": "empirical_frequency"},
            "metrics_by_time": bins,
        },
    }


def _make_attempt(
    harness: Harness,
    *,
    stage_id: str,
    arm_id: str,
    scheduler_arm_id: str | None,
    source_revision: str,
    losses: list[float],
    correct: list[int],
    gradients_pass: bool = True,
    scheduler_dependency: dict[str, Any] | None = None,
    gpu_state_updates: Mapping[str, Any] | None = None,
    gpu_policy_updates: Mapping[str, Any] | None = None,
    gpu_timestamp_updates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    assert harness.registry is not None
    stage = screen._stage(harness.registry, stage_id)
    arm = screen._arm(stage, arm_id)
    config_entry = screen._registered_config(arm, scheduler_arm_id=scheduler_arm_id)
    output_directory = config_entry["output_directory"]
    checkpoint_payload = f"checkpoint:{stage_id}:{arm_id}".encode()
    checkpoint_base = harness.add_blob(
        "repository",
        f"{output_directory}/checkpoints/{screen.EXPECTED_UPDATES[stage_id]}.ckpt",
        checkpoint_payload,
    )
    checkpoint = {
        **checkpoint_base,
        "global_step": screen.EXPECTED_UPDATES[stage_id],
    }
    gradient_audit = (
        _gradient_audit(
            harness,
            scheduler_arm_id=scheduler_arm_id or "E-L0",
            passing=gradients_pass,
        )
        if arm_id == "E-A1"
        else None
    )
    state_audit = _state_audit(
        harness,
        arm_id=arm_id,
        config_sha256=config_entry["config"]["canonical_sha256"],
        stage_id=stage_id,
    )
    initialization = {
        "mode": "fresh_independent_mdlm_ema_warm_start_each_arm",
        "source_checkpoint_sha256": harness.registry.data["common_training"][
            "initialization"
        ]["checkpoint"]["sha256"],
        "weights": "ema",
        "optimizer_reset": True,
        "scheduler_reset": True,
        "global_step_reset": True,
        "ema_reset": True,
        "state_audit": state_audit,
    }
    gpu_state = {
        "physical_index": 7,
        "uuid": "GPU-test-idle-1",
        "name": "Synthetic Accelerator",
        "memory_used_mib": 1_000,
        "memory_total_mib": 81_920,
        "utilization_percent": 0,
        "compute_mode": "Default",
        "compute_processes": [],
    }
    if gpu_state_updates is not None:
        gpu_state.update(copy.deepcopy(dict(gpu_state_updates)))
    gpu_policy = {
        "max_utilization_percent": 10,
        "utilization_comparison": "strictly_less_than",
        "min_free_memory_mib": 30_000,
        "active_compute_processes_allowed": True,
        "compute_mode_prohibited_allowed": False,
    }
    if gpu_policy_updates is not None:
        gpu_policy.update(copy.deepcopy(dict(gpu_policy_updates)))
    gpu_timestamps = {
        "created_at": "2026-09-06T00:00:03+00:00",
        "inventory_snapshot_completed_at_utc": "2026-09-06T00:00:01+00:00",
        "final_uuid_probes_completed_at_utc": "2026-09-06T00:00:02+00:00",
    }
    if gpu_timestamp_updates is not None:
        gpu_timestamps.update(copy.deepcopy(dict(gpu_timestamp_updates)))
    manifest = {
        **gpu_timestamps,
        "launch_manifest_schema_version": 2,
        "purpose": "registered UDLM optimization screen",
        "gpu_selection_schema_version": 2,
        "git_sha": source_revision,
        "source_revision_before_final_gpu_probe": source_revision,
        "optimization_screen": screen._screen_binding(
            harness.registry,
            stage_id=stage_id,
            arm=arm,
            scheduler_dependency=scheduler_dependency,
        ),
        "seed": 17,
        "max_steps": screen.EXPECTED_UPDATES[stage_id],
        "udlm_prior_variant": "empirical_frequency",
        "resolved_training_config_sha256": config_entry["config"]["canonical_sha256"],
        "user_requested_gpu_count": 1,
        "gpu_selection_method": "dynamic_idle_discovery",
        "gpu_inventory_scope": "all_nvidia_gpus",
        "gpu_inventory_at_selection": [copy.deepcopy(gpu_state)],
        "initially_selected_gpu_states": [copy.deepcopy(gpu_state)],
        "logical_cuda_devices": [0],
        "physical_gpu_indices": [gpu_state["physical_index"]],
        "cuda_visible_device_uuids": ["GPU-test-idle-1"],
        "gpu_states_at_final_uuid_probe": [copy.deepcopy(gpu_state)],
        "gpu_safety_policy": gpu_policy,
    }
    manifest_ref = _artifact_json(
        harness, output_directory, "launch_manifest", manifest, 2
    )
    runtime = {
        "schema_version": 2,
        "status": "preflight_completed",
        "source_revision": source_revision,
        "resolved_training_config_sha256": config_entry["config"]["canonical_sha256"],
        "launch_manifest": {
            "sha256": manifest_ref["sha256"],
            "selected_gpu_uuids": ["GPU-test-idle-1"],
        },
    }
    runtime_ref = _artifact_json(
        harness, output_directory, "runtime_config", runtime, 2
    )
    startup = {
        "mode": "warm_start",
        "verified_mdlm_warm_start_report": {
            "source_sha256": harness.registry.data["common_training"]["initialization"][
                "checkpoint"
            ]["sha256"],
            "weights": "ema",
        },
    }
    if stage_id == "conditioning":
        startup["training_rng_policy"] = {
            "policy": "reseed_all_training_rng_streams_after_model_and_warm_start",
            "seed": 17,
            "purpose": "isolate_training_randomness_from_architecture_constructor_draws",
            "applied_before_dataloader_and_trainer_construction": True,
        }
    summary = {
        "schema_version": 4,
        "status": "completed",
        "source_revision": source_revision,
        "resolved_training_config_sha256": config_entry["config"]["canonical_sha256"],
        "launch_manifest": {"sha256": manifest_ref["sha256"]},
        "observed_training_state": {
            "global_step": screen.EXPECTED_UPDATES[stage_id],
            "world_size": 1,
        },
        "startup": startup,
        "conditioning_gradient_audit": gradient_audit,
        "screen_initialization_state_audit": state_audit,
        "final_checkpoint": {
            "sha256": checkpoint["sha256"],
            "size_bytes": checkpoint["size_bytes"],
        },
    }
    summary_ref = _artifact_json(
        harness, output_directory, "training_summary", summary, 4
    )
    receipt = {
        "schema_version": 5,
        "status": "completed",
        "overall_status": "completed",
        "process_exit_status": 0,
        "predecessor_receipt_binding": None,
        "completion_requirements": {
            "predecessor_receipt_binding_unchanged_and_valid": True,
            "all_must_hold": True,
        },
        "expected_contract": {
            "source_revision": source_revision,
            "resolved_training_config_sha256": config_entry["config"][
                "canonical_sha256"
            ],
            "launch_manifest_sha256": manifest_ref["sha256"],
            "selected_gpu_uuids": ["GPU-test-idle-1"],
            "max_steps": screen.EXPECTED_UPDATES[stage_id],
            "world_size": 1,
            "initialization_checkpoint_sha256": initialization[
                "source_checkpoint_sha256"
            ],
        },
        "source_at_receipt": {"verified": True},
        "launch_manifest": {"valid_and_launch_bound": True},
        "runtime_config": {
            "matches_training_summary_snapshot": True,
            "semantic_validation_passed": True,
        },
        "training_summary": {
            "valid_and_launch_bound": True,
            "artifact": {"sha256": summary_ref["sha256"]},
            "validated_bindings": {
                "conditioning_gradient_audit": gradient_audit,
                "screen_initialization_state_audit": state_audit,
            },
        },
        "final_checkpoint": {
            "matches_training_summary_snapshot": True,
            "artifact": {"sha256": checkpoint["sha256"]},
        },
    }
    receipt_ref = _artifact_json(
        harness, output_directory, "pilot_exit_status", receipt, 5
    )
    evaluator = _make_evaluator_report(
        harness,
        arm_id=arm_id,
        source_revision=source_revision,
        config_sha256=config_entry["config"]["canonical_sha256"],
        checkpoint=checkpoint,
        losses=losses,
        correct=correct,
    )
    evaluator_ref = _artifact_json(
        harness, output_directory, "denoising_evaluator", evaluator, 4
    )
    wrapper = {
        "schema_version": 1,
        "artifact_kind": "optimization_screen_denoising_evaluation_binding",
        "registry_sha256": harness.registry.raw_sha256,
        "registry_canonical_sha256": harness.registry.canonical_sha256,
        "stage_id": stage_id,
        "arm_id": arm_id,
        "attempt_id": arm["attempt_id"],
        "source_revision": source_revision,
        "resolved_config_canonical_sha256": config_entry["config"]["canonical_sha256"],
        "checkpoint_sha256": checkpoint["sha256"],
        "producer_source": next(
            ref
            for ref in harness.registry.data["source"]["blobs"]
            if ref["relative_path"]
            == "scripts/udlm/collect_optimization_screen_evidence.py"
        ),
        "evaluator_report": evaluator_ref,
        "generation_metrics_included": False,
        "final_generation_seeds_included": [],
    }
    wrapper_ref = _artifact_json(
        harness, output_directory, "denoising_binding", wrapper, 1
    )
    return {
        "attempt_id": arm["attempt_id"],
        "arm_id": arm_id,
        "status": "completed",
        "failure_reason": None,
        "training_seed": 17,
        "optimizer_updates": screen.EXPECTED_UPDATES[stage_id],
        "gpu_count": 1,
        "source_revision": source_revision,
        "output_directory": output_directory,
        "resolved_config": config_entry["config"],
        "artifacts": {
            "launch_manifest": manifest_ref,
            "runtime_config": runtime_ref,
            "training_summary": summary_ref,
            "exit_receipt": receipt_ref,
        },
        "checkpoint": checkpoint,
        "initialization": initialization,
        "denoising_report": wrapper_ref,
        "conditioning_gradient_audit": gradient_audit,
    }


def _scheduler_evidence(
    harness: Harness,
    *,
    control_losses: list[float] | None = None,
    candidate_losses: list[float] | None = None,
    gpu_state_updates: Mapping[str, Any] | None = None,
    gpu_policy_updates: Mapping[str, Any] | None = None,
    gpu_timestamp_updates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    assert harness.registry is not None
    collector = next(
        ref
        for ref in harness.registry.data["source"]["blobs"]
        if ref["relative_path"]
        == "scripts/udlm/collect_optimization_screen_evidence.py"
    )
    return {
        "schema_version": 1,
        "registry": harness.registry.reference,
        "stage_id": "scheduler",
        "run_source_revision": PUBLICATION_REVISION,
        "producer_source": collector,
        "status": "closed_after_registered_attempts",
        "generation_metrics_included": False,
        "final_generation_seeds_included": [],
        "scheduler_dependency": None,
        "initialization_audit": None,
        "attempts": [
            _make_attempt(
                harness,
                stage_id="scheduler",
                arm_id="E-L0",
                scheduler_arm_id=None,
                source_revision=PUBLICATION_REVISION,
                losses=control_losses or [100.0, 100.0, 100.0],
                correct=[8000, 8000, 8000],
                gpu_state_updates=gpu_state_updates,
                gpu_policy_updates=gpu_policy_updates,
                gpu_timestamp_updates=gpu_timestamp_updates,
            ),
            _make_attempt(
                harness,
                stage_id="scheduler",
                arm_id="E-L1",
                scheduler_arm_id=None,
                source_revision=PUBLICATION_REVISION,
                losses=candidate_losses or [98.0, 98.0, 98.0],
                correct=[8000, 8000, 8000],
                gpu_state_updates=gpu_state_updates,
                gpu_policy_updates=gpu_policy_updates,
                gpu_timestamp_updates=gpu_timestamp_updates,
            ),
        ],
    }


def _publish_scheduler_decision(
    harness: Harness, evidence: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    assert harness.registry is not None
    evidence_payload = _bytes(evidence)
    decision = screen.evaluate_evidence_bytes(
        evidence_payload,
        evidence_relative_path="experiments/udlm/screens/scheduler_evidence.json",
        stage_id="scheduler",
        registry=harness.registry,
        loader=harness.loader,
    )
    evidence_ref = harness.add_json(
        "repository",
        "experiments/udlm/screens/scheduler_evidence.json",
        evidence,
        1,
        revisions=(AUTHORIZATION_REVISION,),
    )
    decision_ref = harness.add_json(
        "repository",
        "experiments/udlm/screens/scheduler_selection.json",
        decision,
        1,
        revisions=(AUTHORIZATION_REVISION,),
    )
    return evidence_ref, decision_ref


def _conditioning_init_audit(
    harness: Harness,
    *,
    scheduler_arm_id: str,
    exact_equal: bool = True,
) -> dict[str, Any]:
    assert harness.registry is not None
    stage = harness.registry.data["stages"][1]
    a0_config = screen._registered_config(
        screen._arm(stage, "E-A0"), scheduler_arm_id=scheduler_arm_id
    )
    a1_config = screen._registered_config(
        screen._arm(stage, "E-A1"), scheduler_arm_id=scheduler_arm_id
    )
    fixture_ref = stage["initialization_fixture"]
    fixture = json.loads(
        harness.blobs[(fixture_ref["root"], fixture_ref["relative_path"])]
    )
    reference_payload = b"\x00\x00\x80?"
    candidate_payload = reference_payload if exact_equal else b"\x00\x00\x00@"
    reference_ref = harness.add_blob(
        "repository",
        f"{a0_config['output_directory']}/E-A0.initial_logits.bin",
        reference_payload,
    )
    candidate_ref = harness.add_blob(
        "repository",
        f"{a1_config['output_directory']}/E-A1.initial_logits.bin",
        candidate_payload,
    )
    producer = next(
        ref
        for ref in harness.registry.data["source"]["blobs"]
        if ref["relative_path"] == "scripts/udlm/audit_conditioning_initialization.py"
    )
    return {
        "schema_version": 1,
        "reference_arm_id": "E-A0",
        "candidate_arm_id": "E-A1",
        "fixture": fixture_ref,
        "source_revision": AUTHORIZATION_REVISION,
        "checkpoint_sha256": harness.registry.data["common_training"]["initialization"][
            "checkpoint"
        ]["sha256"],
        "reference_config_canonical_sha256": a0_config["config"]["canonical_sha256"],
        "candidate_config_canonical_sha256": a1_config["config"]["canonical_sha256"],
        "probe_phase": (
            "after_mdlm_ema_load_before_training_rng_reseed_and_optimizer_creation"
        ),
        "input_ids_sha256": screen.canonical_json_sha256(fixture["input_ids"]),
        "attention_mask_sha256": screen.canonical_json_sha256(
            fixture["attention_mask"]
        ),
        "noise_tensor_sha256": screen.canonical_json_sha256(fixture["noise_tensor"]),
        "timestep_tensor_sha256": screen.canonical_json_sha256(
            fixture["timestep_tensor"]
        ),
        "logits_dtype": "float32-little-endian-c-order",
        "logits_shape": [1],
        "reference_logits": reference_ref,
        "candidate_logits": candidate_ref,
        "producer_source": producer,
        "exact_equal": exact_equal,
    }


def _conditioning_evidence(
    harness: Harness,
    *,
    scheduler_evidence: dict[str, Any] | None = None,
    control_losses: list[float] | None = None,
    candidate_losses: list[float] | None = None,
    control_correct: list[int] | None = None,
    candidate_correct: list[int] | None = None,
    gradients_pass: bool = True,
    exact_init: bool = True,
) -> dict[str, Any]:
    assert harness.registry is not None
    scheduler_document = scheduler_evidence or _scheduler_evidence(harness)
    evidence_ref, decision_ref = _publish_scheduler_decision(
        harness, scheduler_document
    )
    scheduler_decision = json.loads(
        harness.blobs[(decision_ref["root"], decision_ref["relative_path"])]
    )
    scheduler_arm_id = scheduler_decision["selected_arm_id"]
    normalized_dependency = {
        "authorization_revision": AUTHORIZATION_REVISION,
        "scheduler_evidence": evidence_ref,
        "scheduler_selection": decision_ref,
        "selected_scheduler_arm_id": scheduler_arm_id,
    }
    collector = next(
        ref
        for ref in harness.registry.data["source"]["blobs"]
        if ref["relative_path"]
        == "scripts/udlm/collect_optimization_screen_evidence.py"
    )
    return {
        "schema_version": 1,
        "registry": harness.registry.reference,
        "stage_id": "conditioning",
        "run_source_revision": AUTHORIZATION_REVISION,
        "producer_source": collector,
        "status": "closed_after_registered_attempts",
        "generation_metrics_included": False,
        "final_generation_seeds_included": [],
        "scheduler_dependency": {
            "authorization_revision": AUTHORIZATION_REVISION,
            "scheduler_evidence": evidence_ref,
            "scheduler_selection": decision_ref,
        },
        "initialization_audit": _conditioning_init_audit(
            harness,
            scheduler_arm_id=scheduler_arm_id,
            exact_equal=exact_init,
        ),
        "attempts": [
            _make_attempt(
                harness,
                stage_id="conditioning",
                arm_id="E-A0",
                scheduler_arm_id=scheduler_arm_id,
                source_revision=AUTHORIZATION_REVISION,
                losses=control_losses or [100.0, 100.0, 100.0],
                correct=control_correct or [8000, 8000, 8000],
                scheduler_dependency=normalized_dependency,
            ),
            _make_attempt(
                harness,
                stage_id="conditioning",
                arm_id="E-A1",
                scheduler_arm_id=scheduler_arm_id,
                source_revision=AUTHORIZATION_REVISION,
                losses=candidate_losses or [98.0, 98.0, 98.0],
                correct=candidate_correct or [8000, 8000, 8000],
                gradients_pass=gradients_pass,
                scheduler_dependency=normalized_dependency,
            ),
        ],
    }


def _evaluate(
    harness: Harness, evidence: dict[str, Any], stage_id: str
) -> dict[str, Any]:
    assert harness.registry is not None
    return screen.evaluate_evidence_bytes(
        _bytes(evidence),
        evidence_relative_path=f"experiments/udlm/screens/{stage_id}_input.json",
        stage_id=stage_id,
        registry=harness.registry,
        loader=harness.loader,
    )


def _evaluator_context(
    harness: Harness, evidence: dict[str, Any], attempt_index: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    assert harness.registry is not None
    attempt = evidence["attempts"][attempt_index]
    wrapper_ref = attempt["denoising_report"]
    wrapper = json.loads(
        harness.blobs[(wrapper_ref["root"], wrapper_ref["relative_path"])]
    )
    evaluator_ref = wrapper["evaluator_report"]
    evaluator = json.loads(
        harness.blobs[(evaluator_ref["root"], evaluator_ref["relative_path"])]
    )
    stage = screen._stage(harness.registry, evidence["stage_id"])
    arm = screen._arm(stage, attempt["arm_id"])
    scheduler_arm = (
        None
        if evidence["stage_id"] == "scheduler"
        else json.loads(
            harness.blobs[
                (
                    evidence["scheduler_dependency"]["scheduler_selection"]["root"],
                    evidence["scheduler_dependency"]["scheduler_selection"][
                        "relative_path"
                    ],
                )
            ]
        )["selected_arm_id"]
    )
    config_entry = screen._registered_config(arm, scheduler_arm_id=scheduler_arm)
    return evaluator, config_entry, attempt["checkpoint"], attempt["arm_id"]


def test_registry_and_scheduler_happy_path_select_candidate() -> None:
    harness = build_harness()
    decision = _evaluate(harness, _scheduler_evidence(harness), "scheduler")
    assert decision["status"] == "completed"
    assert decision["selected_arm_id"] == "E-L1"
    assert decision["complete_threshold_fallback_used"] is False


def test_screen_gpu_policy_accepts_recorded_process_below_ten_percent() -> None:
    harness = build_harness()
    evidence = _scheduler_evidence(
        harness,
        gpu_state_updates={
            "utilization_percent": 9,
            "compute_processes": [
                {
                    "pid": 4321,
                    "process_name": "pre-existing-workload",
                    "used_memory_mib": 512,
                }
            ],
        },
    )

    decision = _evaluate(harness, evidence, "scheduler")

    assert decision["status"] == "completed"
    assert decision["selected_arm_id"] == "E-L1"


def test_screen_gpu_policy_accepts_stricter_recorded_thresholds_round_trip() -> None:
    harness = build_harness()
    evidence = _scheduler_evidence(
        harness,
        gpu_state_updates={"utilization_percent": 4},
        gpu_policy_updates={
            "max_utilization_percent": 5,
            "min_free_memory_mib": 40_000,
        },
    )

    decision = _evaluate(harness, evidence, "scheduler")

    assert decision["status"] == "completed"
    assert decision["selected_arm_id"] == "E-L1"


@pytest.mark.parametrize(
    "gpu_state_updates",
    [
        {"utilization_percent": 10},
        {"memory_used_mib": 51_921},
        {"compute_mode": "Prohibited"},
    ],
)
def test_screen_gpu_policy_rejects_unsafe_selected_state(
    gpu_state_updates: dict[str, Any],
) -> None:
    harness = build_harness()
    evidence = _scheduler_evidence(harness, gpu_state_updates=gpu_state_updates)

    decision = _evaluate(harness, evidence, "scheduler")

    assert decision["status"] == "incomplete"
    assert decision["selected_arm_id"] is None
    assert decision["reason_codes"] == ["evidence_invalid_or_unmatched"]


def test_screen_gpu_policy_rejects_sparse_process_evidence() -> None:
    harness = build_harness()
    evidence = _scheduler_evidence(
        harness,
        gpu_state_updates={
            "utilization_percent": 9,
            "compute_processes": [
                {"pid": 4321, "process_name": "missing-memory-field"}
            ],
        },
    )

    decision = _evaluate(harness, evidence, "scheduler")

    assert decision["status"] == "incomplete"
    assert decision["selected_arm_id"] is None


@pytest.mark.parametrize(
    "gpu_policy_updates",
    [
        {"max_utilization_percent": 11},
        {"min_free_memory_mib": 29_999},
        {"active_compute_processes_allowed": False},
    ],
)
def test_screen_gpu_policy_rejects_out_of_bounds_policy(
    gpu_policy_updates: dict[str, Any],
) -> None:
    harness = build_harness()
    evidence = _scheduler_evidence(harness, gpu_policy_updates=gpu_policy_updates)

    decision = _evaluate(harness, evidence, "scheduler")

    assert decision["status"] == "incomplete"
    assert decision["selected_arm_id"] is None


@pytest.mark.parametrize(
    "gpu_timestamp_updates",
    [
        {"inventory_snapshot_completed_at_utc": None},
        {"final_uuid_probes_completed_at_utc": "2026-09-06T00:00:00+00:00"},
    ],
)
def test_screen_gpu_policy_rejects_missing_or_reversed_probe_timestamps(
    gpu_timestamp_updates: dict[str, Any],
) -> None:
    harness = build_harness()
    evidence = _scheduler_evidence(harness, gpu_timestamp_updates=gpu_timestamp_updates)

    decision = _evaluate(harness, evidence, "scheduler")

    assert decision["status"] == "incomplete"
    assert decision["selected_arm_id"] is None


def test_complete_scheduler_threshold_miss_selects_registered_control() -> None:
    harness = build_harness()
    evidence = _scheduler_evidence(
        harness,
        candidate_losses=[94.0, 100.0, 100.0],
    )
    decision = _evaluate(harness, evidence, "scheduler")
    assert decision["status"] == "completed"
    assert decision["selected_arm_id"] == "E-L0"
    assert decision["complete_threshold_fallback_used"] is True
    assert decision["gates"]["strictly_better_bin_count"] == 1


def test_scheduler_exact_per_bin_boundary_with_unequal_denominators() -> None:
    def attempt(arm_id: str, values: list[str], denominator: int) -> dict[str, Any]:
        return {
            "arm_id": arm_id,
            "bins": [
                {
                    "loss_sum": Decimal(value),
                    "denominator": denominator,
                    "correct": 0,
                }
                for value in values
            ],
        }

    control = attempt("E-L0", ["100", "100", "100"], 100)
    at_boundary = attempt("E-L1", ["51", "48", "48"], 50)
    gates = screen._scheduler_gates([control, at_boundary])
    assert gates["pooled_candidate_le_98_percent_control"] is True
    assert gates["every_bin_candidate_le_102_percent_control"] is True
    assert gates["all_required_gates_pass"] is True

    just_above = attempt(
        "E-L1",
        ["51.0000000000000001", "47.9999999999999999", "48"],
        50,
    )
    gates = screen._scheduler_gates([control, just_above])
    assert gates["pooled_candidate_le_98_percent_control"] is True
    assert gates["every_bin_candidate_le_102_percent_control"] is False
    assert gates["all_required_gates_pass"] is False


@pytest.mark.parametrize(
    "payload",
    [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}', b'{"a":1e1000000}'],
)
def test_strict_json_rejects_duplicate_and_nonfinite_numbers(payload: bytes) -> None:
    with pytest.raises(screen.ScreenValidationError):
        screen.strict_json_loads(payload, label="test")


def test_exact_rational_boundaries_do_not_round() -> None:
    assert screen._loss_ratio_le(Fraction(98, 1), 1, Fraction(100, 1), 1, 98)
    assert not screen._loss_ratio_le(
        Fraction(98 * 10**40 + 1, 10**40),
        1,
        Fraction(100, 1),
        1,
        98,
    )
    parsed = screen.strict_json_loads(
        b'{"x":0.9800000000000000000000000001}', label="x"
    )
    assert parsed["x"] > Decimal("0.98")


def test_missing_malformed_and_failed_evidence_never_falls_back() -> None:
    harness = build_harness()
    assert harness.registry is not None
    for payload in (b"", b"{}", b'{"schema_version":NaN}'):
        decision = screen.evaluate_evidence_bytes(
            payload,
            evidence_relative_path="experiments/udlm/screens/missing.json",
            stage_id="scheduler",
            registry=harness.registry,
            loader=harness.loader,
        )
        assert decision["status"] == "incomplete"
        assert decision["selected_arm_id"] is None
        assert decision["complete_threshold_fallback_used"] is False
    evidence = _scheduler_evidence(harness)
    evidence["attempts"][1].update(
        {
            "status": "failed",
            "failure_reason": "registered run failed",
            "artifacts": None,
            "checkpoint": None,
            "initialization": None,
            "denoising_report": None,
            "conditioning_gradient_audit": None,
        }
    )
    decision = _evaluate(harness, evidence, "scheduler")
    assert decision["status"] == "incomplete"
    assert decision["selected_arm_id"] is None


def test_generation_metrics_or_final_seed_leakage_is_incomplete() -> None:
    harness = build_harness()
    evidence = _scheduler_evidence(harness)
    evidence["generation_metrics_included"] = True
    evidence["final_generation_seeds_included"] = [0]
    decision = _evaluate(harness, evidence, "scheduler")
    assert decision["status"] == "incomplete"
    assert decision["selected_arm_id"] is None


def test_conditioning_candidate_happy_path_and_registered_control_paths() -> None:
    harness = build_harness()
    candidate = _evaluate(harness, _conditioning_evidence(harness), "conditioning")
    assert candidate["status"] == "completed"
    assert candidate["selected_arm_id"] == "E-A1"

    harness = build_harness()
    gradient_miss = _evaluate(
        harness,
        _conditioning_evidence(harness, gradients_pass=False),
        "conditioning",
    )
    assert gradient_miss["status"] == "completed"
    assert gradient_miss["selected_arm_id"] == "E-A0"
    assert gradient_miss["gates"]["registered_gradient_contract_pass"] is False

    harness = build_harness()
    init_miss = _evaluate(
        harness,
        _conditioning_evidence(harness, exact_init=False),
        "conditioning",
    )
    assert init_miss["status"] == "completed"
    assert init_miss["selected_arm_id"] == "E-A0"


def test_conditioning_revalidates_l0_contingency_and_blocks_incomplete_scheduler() -> (
    None
):
    harness = build_harness()
    scheduler_l0 = _scheduler_evidence(harness, candidate_losses=[94.0, 100.0, 100.0])
    conditioning = _conditioning_evidence(harness, scheduler_evidence=scheduler_l0)
    decision = _evaluate(harness, conditioning, "conditioning")
    assert decision["status"] == "completed"
    assert decision["dependency"]["selected_scheduler_arm_id"] == "E-L0"
    assert decision["selected_arm_id"] == "E-A1"

    harness = build_harness()
    conditioning = _conditioning_evidence(harness)
    failed_scheduler = _scheduler_evidence(harness)
    failed_scheduler["attempts"][1].update(
        {
            "status": "failed",
            "failure_reason": "failed",
            "artifacts": None,
            "checkpoint": None,
            "initialization": None,
            "denoising_report": None,
            "conditioning_gradient_audit": None,
        }
    )
    evidence_ref, decision_ref = _publish_scheduler_decision(harness, failed_scheduler)
    conditioning["scheduler_dependency"]["scheduler_evidence"] = evidence_ref
    conditioning["scheduler_dependency"]["scheduler_selection"] = decision_ref
    decision = _evaluate(harness, conditioning, "conditioning")
    assert decision["status"] == "incomplete"
    assert decision["selected_arm_id"] is None


def test_conditioning_uses_pooled_integer_accuracy_cross_product() -> None:
    harness = build_harness()
    equality = _evaluate(
        harness,
        _conditioning_evidence(
            harness,
            control_correct=[8000, 8000, 8000],
            candidate_correct=[7999, 8000, 8001],
        ),
        "conditioning",
    )
    assert equality["selected_arm_id"] == "E-A1"

    harness = build_harness()
    regression = _evaluate(
        harness,
        _conditioning_evidence(
            harness,
            control_correct=[8000, 8000, 8000],
            candidate_correct=[7999, 8000, 8000],
        ),
        "conditioning",
    )
    assert regression["status"] == "completed"
    assert regression["selected_arm_id"] == "E-A0"


def test_gradient_lr_or_scheduler_dependency_tamper_is_incomplete() -> None:
    harness = build_harness()
    evidence = _conditioning_evidence(harness)
    evidence["attempts"][1]["conditioning_gradient_audit"]["optimizer_checks"][1][
        "learning_rate_before_step"
    ] = 0.000006000000000001
    decision = _evaluate(harness, evidence, "conditioning")
    assert decision["status"] == "incomplete"
    assert decision["selected_arm_id"] is None

    harness = build_harness()
    evidence = _conditioning_evidence(harness)
    selection_ref = evidence["scheduler_dependency"]["scheduler_selection"]
    declared = json.loads(
        harness.blobs[(selection_ref["root"], selection_ref["relative_path"])]
    )
    declared["selected_arm_id"] = "E-L0"
    harness.blobs[(selection_ref["root"], selection_ref["relative_path"])] = _bytes(
        declared
    )
    decision = _evaluate(harness, evidence, "conditioning")
    assert decision["status"] == "incomplete"
    assert decision["selected_arm_id"] is None


def test_registry_requires_exact_health_gate_shape_and_world_size() -> None:
    harness = build_harness()
    assert harness.registry_document is not None

    missing = copy.deepcopy(harness.registry_document)
    del missing["prerequisite_health_gate"]
    with pytest.raises(screen.ScreenValidationError, match="missing=.*health"):
        _load_registry_document(harness, missing)

    wrong_world_size = copy.deepcopy(harness.registry_document)
    wrong_world_size["prerequisite_health_gate"]["evidence"]["gpu_count"] = 2
    with pytest.raises(screen.ScreenValidationError, match="GPU count differs"):
        _load_registry_document(harness, wrong_world_size)


def test_registry_rejects_health_eligibility_or_order_tampering() -> None:
    harness = build_harness()
    assert harness.registry_document is not None

    eligible_for_ranking = copy.deepcopy(harness.registry_document)
    eligible_for_ranking["prerequisite_health_gate"]["evidence"]["eligibility"][
        "ranking"
    ] = True
    with pytest.raises(screen.ScreenValidationError, match="eligibility"):
        _load_registry_document(harness, eligible_for_ranking)

    wrong_order = copy.deepcopy(harness.registry_document)
    wrong_order["prerequisite_health_gate"]["evidence"]["receipt_members"].reverse()
    with pytest.raises(screen.ScreenValidationError, match="receipt member"):
        _load_registry_document(harness, wrong_order)


def test_registry_revalidates_live_health_evidence() -> None:
    harness = build_harness()
    assert harness.registry_document is not None
    assert harness.validated_health_evidence is not None
    stale = copy.deepcopy(harness.validated_health_evidence)
    stale["matched_panel_spec_sha256"] = "f" * 64

    def stale_validator(*_args, **_kwargs):
        return stale

    with pytest.raises(screen.ScreenValidationError, match="live health-gate"):
        _load_registry_document(
            harness,
            harness.registry_document,
            health_gate_validator=stale_validator,
        )


def test_registry_revalidates_health_to_r0_git_transition() -> None:
    harness = build_harness()
    assert harness.registry_document is not None

    with pytest.raises(screen.ScreenValidationError, match="sole immediate parent"):
        _load_registry_document(
            harness,
            harness.registry_document,
            git_sole_parent_checker=lambda _revision, _parent: False,
        )

    with pytest.raises(screen.ScreenValidationError, match="not an ancestor"):
        _load_registry_document(
            harness,
            harness.registry_document,
            git_ancestor_checker=lambda _ancestor, _descendant: False,
        )

    with pytest.raises(screen.ScreenValidationError, match="not limited"):
        _load_registry_document(
            harness,
            harness.registry_document,
            git_diff_checker=lambda _ancestor, _descendant, _paths: False,
        )

    selected_path = screen._expected_screen_config_paths(1)[0]
    harness.git_blobs[(HEALTH_REVISION, selected_path)] = b"preexisting"
    with pytest.raises(screen.ScreenValidationError, match="unexpectedly exists"):
        _load_registry_document(harness, harness.registry_document)

    harness = build_harness()
    assert harness.registry_document is not None
    selected_directory = screen.EXPECTED_SCREEN_CONFIG_DIRECTORY_TEMPLATE.format(
        gpu_count=1
    )
    harness.git_blobs[
        (REGISTRY_REVISION, f"{selected_directory}/unregistered.json")
    ] = b"extra"
    with pytest.raises(screen.ScreenValidationError, match="exact six-file set"):
        _load_registry_document(harness, harness.registry_document)

    harness = build_harness()
    assert harness.registry_document is not None
    opposite_path = screen._expected_screen_config_paths(2)[0]
    harness.git_blobs[(REGISTRY_REVISION, opposite_path)] = b"wrong-family"
    with pytest.raises(screen.ScreenValidationError, match="unexpectedly exists"):
        _load_registry_document(harness, harness.registry_document)

    harness = build_harness()
    assert harness.registry_document is not None
    opposite_directory = screen.EXPECTED_SCREEN_CONFIG_DIRECTORY_TEMPLATE.format(
        gpu_count=2
    )
    harness.git_blobs[
        (HEALTH_REVISION, f"{opposite_directory}/arbitrary-extra.txt")
    ] = b"wrong-family"
    with pytest.raises(screen.ScreenValidationError, match="unexpectedly exists"):
        _load_registry_document(harness, harness.registry_document)


def test_registry_semantic_and_git_root_tampering_is_rejected() -> None:
    two_gpu = build_harness(gpu_count=2)
    assert two_gpu.registry is not None
    assert two_gpu.registry.data["common_training"]["gpu_count"] == 2

    harness = build_harness()
    assert harness.registry_document is not None
    bad = copy.deepcopy(harness.registry_document)
    bad["common_training"]["gpu_count"] = 3
    payload = _bytes(bad)
    with pytest.raises(screen.ScreenValidationError):
        screen.load_validated_registry(
            payload,
            relative_path=REGISTRY_PATH,
            expected_raw_sha256=_sha(payload),
            expected_canonical_sha256=screen.canonical_json_sha256(bad),
            loader=harness.loader,
            git_blob_loader=harness.git_loader,
            git_ancestor_checker=harness.ancestor,
            git_sole_parent_checker=harness.sole_parent,
            git_tree_paths_loader=harness.tree_paths,
            git_pushed_checker=harness.pushed,
            git_diff_checker=harness.allowed_diff,
            health_gate_validator=harness.health_validator,
        )


@pytest.mark.parametrize(
    "relative_path",
    [
        screen.EXPECTED_PRIOR_FLOOR_AUDIT_SOURCE_PATH,
        screen.EXPECTED_PRIOR_FLOOR_AUDIT_PATH,
    ],
)
def test_registry_rejects_self_consistent_prior_floor_audit_tampering(
    relative_path: str,
) -> None:
    harness = build_harness()
    assert harness.registry_document is not None
    bad = copy.deepcopy(harness.registry_document)
    ref = next(
        entry
        for entry in bad["source"]["blobs"]
        if entry["relative_path"] == relative_path
    )
    tampered = harness.blobs[("repository", relative_path)] + b"\n"
    harness.blobs[("repository", relative_path)] = tampered
    for revision in (
        REGISTRY_REVISION,
        PUBLICATION_REVISION,
        AUTHORIZATION_REVISION,
    ):
        harness.git_blobs[(revision, relative_path)] = tampered
    ref["sha256"] = _sha(tampered)
    ref["size_bytes"] = len(tampered)
    payload = _bytes(bad)

    with pytest.raises(screen.ScreenValidationError, match="audit .*identity"):
        screen.load_validated_registry(
            payload,
            relative_path=REGISTRY_PATH,
            expected_raw_sha256=_sha(payload),
            expected_canonical_sha256=screen.canonical_json_sha256(bad),
            loader=harness.loader,
            git_blob_loader=harness.git_loader,
            git_ancestor_checker=harness.ancestor,
            git_sole_parent_checker=harness.sole_parent,
            git_tree_paths_loader=harness.tree_paths,
            git_pushed_checker=harness.pushed,
            git_diff_checker=harness.allowed_diff,
            health_gate_validator=harness.health_validator,
        )


def test_registry_rejects_self_consistent_empirical_floor_config_drift() -> None:
    harness = build_harness()
    assert harness.registry_document is not None
    bad = copy.deepcopy(harness.registry_document)
    config_ref = bad["stages"][0]["arms"][0]["resolved_configs"][0]["config"]
    config_path = config_ref["relative_path"]
    config = json.loads(harness.blobs[("repository", config_path)])
    config["training"]["udlm"]["empirical_uniform_mix"] = 0.01
    config_payload = _bytes(config)
    harness.blobs[("repository", config_path)] = config_payload
    for revision in (
        REGISTRY_REVISION,
        PUBLICATION_REVISION,
        AUTHORIZATION_REVISION,
    ):
        harness.git_blobs[(revision, config_path)] = config_payload
    config_ref.update(
        {
            "sha256": _sha(config_payload),
            "size_bytes": len(config_payload),
            "canonical_sha256": screen.canonical_json_sha256(config),
        }
    )
    payload = _bytes(bad)

    with pytest.raises(screen.ScreenValidationError, match="common training semantics"):
        screen.load_validated_registry(
            payload,
            relative_path=REGISTRY_PATH,
            expected_raw_sha256=_sha(payload),
            expected_canonical_sha256=screen.canonical_json_sha256(bad),
            loader=harness.loader,
            git_blob_loader=harness.git_loader,
            git_ancestor_checker=harness.ancestor,
            git_sole_parent_checker=harness.sole_parent,
            git_tree_paths_loader=harness.tree_paths,
            git_pushed_checker=harness.pushed,
            git_diff_checker=harness.allowed_diff,
            health_gate_validator=harness.health_validator,
        )


def test_registry_rejects_self_consistent_mislabeled_config_and_source_omission() -> (
    None
):
    harness = build_harness()
    assert harness.registry_document is not None
    bad = copy.deepcopy(harness.registry_document)
    config_ref = bad["stages"][0]["arms"][0]["resolved_configs"][0]["config"]
    config_path = config_ref["relative_path"]
    config = json.loads(harness.blobs[("repository", config_path)])
    config["training"]["diffusion"] = "mdlm"
    payload = _bytes(config)
    harness.blobs[("repository", config_path)] = payload
    for revision in (
        REGISTRY_REVISION,
        PUBLICATION_REVISION,
        AUTHORIZATION_REVISION,
    ):
        harness.git_blobs[(revision, config_path)] = payload
    config_ref.update(
        {
            "sha256": _sha(payload),
            "size_bytes": len(payload),
            "canonical_sha256": screen.canonical_json_sha256(config),
        }
    )
    registry_payload = _bytes(bad)
    with pytest.raises(screen.ScreenValidationError):
        screen.load_validated_registry(
            registry_payload,
            relative_path=REGISTRY_PATH,
            expected_raw_sha256=_sha(registry_payload),
            expected_canonical_sha256=screen.canonical_json_sha256(bad),
            loader=harness.loader,
            git_blob_loader=harness.git_loader,
            git_ancestor_checker=harness.ancestor,
            git_sole_parent_checker=harness.sole_parent,
            git_tree_paths_loader=harness.tree_paths,
            git_pushed_checker=harness.pushed,
            git_diff_checker=harness.allowed_diff,
            health_gate_validator=harness.health_validator,
        )

    harness = build_harness()
    assert harness.registry_document is not None
    omitted = copy.deepcopy(harness.registry_document)
    omitted["source"]["blobs"] = [
        ref
        for ref in omitted["source"]["blobs"]
        if ref["relative_path"] != "scripts/udlm/launch_optimization_screen.py"
    ]
    omitted_payload = _bytes(omitted)
    with pytest.raises(screen.ScreenValidationError):
        screen.load_validated_registry(
            omitted_payload,
            relative_path=REGISTRY_PATH,
            expected_raw_sha256=_sha(omitted_payload),
            expected_canonical_sha256=screen.canonical_json_sha256(omitted),
            loader=harness.loader,
            git_blob_loader=harness.git_loader,
            git_ancestor_checker=harness.ancestor,
            git_sole_parent_checker=harness.sole_parent,
            git_tree_paths_loader=harness.tree_paths,
            git_pushed_checker=harness.pushed,
            git_diff_checker=harness.allowed_diff,
            health_gate_validator=harness.health_validator,
        )

    bad = copy.deepcopy(harness.registry_document)
    bad["stages"][0]["arms"].reverse()
    payload = _bytes(bad)
    with pytest.raises(screen.ScreenValidationError):
        screen.load_validated_registry(
            payload,
            relative_path=REGISTRY_PATH,
            expected_raw_sha256=_sha(payload),
            expected_canonical_sha256=screen.canonical_json_sha256(bad),
            loader=harness.loader,
            git_blob_loader=harness.git_loader,
            git_ancestor_checker=harness.ancestor,
            git_sole_parent_checker=harness.sole_parent,
            git_tree_paths_loader=harness.tree_paths,
            git_pushed_checker=harness.pushed,
            git_diff_checker=harness.allowed_diff,
            health_gate_validator=harness.health_validator,
        )


def test_artifact_hash_or_owner_tamper_is_incomplete() -> None:
    harness = build_harness()
    evidence = _scheduler_evidence(harness)
    evidence["attempts"][1]["checkpoint"]["relative_path"] = evidence["attempts"][0][
        "checkpoint"
    ]["relative_path"]
    decision = _evaluate(harness, evidence, "scheduler")
    assert decision["status"] == "incomplete"
    assert decision["selected_arm_id"] is None

    harness = build_harness()
    evidence = _scheduler_evidence(harness)
    ref = evidence["attempts"][0]["artifacts"]["runtime_config"]
    harness.blobs[(ref["root"], ref["relative_path"])] += b"tamper"
    decision = _evaluate(harness, evidence, "scheduler")
    assert decision["status"] == "incomplete"


def test_evaluator_schema_process_and_conditioning_identity_are_revalidated() -> None:
    harness = build_harness()
    evidence = _conditioning_evidence(harness)
    evaluator, config_entry, checkpoint, arm_id = _evaluator_context(
        harness, evidence, 1
    )
    assert harness.registry is not None
    screen._validate_evaluator_report(
        evaluator,
        registry=harness.registry,
        config_entry=config_entry,
        checkpoint=checkpoint,
        source_revision=AUTHORIZATION_REVISION,
        arm_id=arm_id,
    )
    for mutation in ("legacy_schema", "wrong_process", "typed_metadata"):
        bad = copy.deepcopy(evaluator)
        if mutation == "legacy_schema":
            bad["schema_version"] = 3
        elif mutation == "wrong_process":
            bad["evaluation"]["process"]["prior_variant"] = "release_uniform"
        else:
            bad["conditioning"]["runtime_metadata"]["hidden_size"] = True
            bad["conditioning"]["checkpoint_metadata"]["hidden_size"] = True
            digest = screen.canonical_json_sha256(
                bad["conditioning"]["runtime_metadata"]
            )
            bad["conditioning"]["runtime_metadata_canonical_sha256"] = digest
            bad["conditioning"]["checkpoint_metadata_canonical_sha256"] = digest
        with pytest.raises(screen.ScreenValidationError):
            screen._validate_evaluator_report(
                bad,
                registry=harness.registry,
                config_entry=config_entry,
                checkpoint=checkpoint,
                source_revision=AUTHORIZATION_REVISION,
                arm_id=arm_id,
            )


def test_unpushed_or_nonchronological_run_revision_is_incomplete() -> None:
    harness = build_harness()
    evidence = _scheduler_evidence(harness)
    evidence["run_source_revision"] = REGISTRY_REVISION
    for attempt in evidence["attempts"]:
        attempt["source_revision"] = REGISTRY_REVISION
    decision = _evaluate(harness, evidence, "scheduler")
    assert decision["status"] == "incomplete"
    assert decision["selected_arm_id"] is None

    harness = build_harness()
    assert harness.registry is not None
    object.__setattr__(
        harness.registry,
        "git_pushed_checker",
        lambda revision: revision != PUBLICATION_REVISION,
    )
    decision = _evaluate(harness, _scheduler_evidence(harness), "scheduler")
    assert decision["status"] == "incomplete"
    assert decision["selected_arm_id"] is None


def test_checkpoint_hashing_streams_without_deserialization(tmp_path: Path) -> None:
    path = tmp_path / "large.ckpt"
    payload = b"0123456789abcdef" * (1024 * 128)
    path.write_bytes(payload)
    snapshot = screen._read_stable_file(path, retain=False)
    assert isinstance(snapshot, screen.BlobSnapshot)
    assert snapshot.size_bytes == len(payload)
    assert snapshot.sha256 == _sha(payload)


def test_atomic_output_refuses_existing_file_and_symlink(tmp_path: Path) -> None:
    repository_output = screen.REPOSITORY_ROOT / "output" / "udlm" / "screens"
    repository_output.mkdir(parents=True, exist_ok=True)
    target = repository_output / f"pytest-selection-{tmp_path.name}.json"
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        screen._atomic_write_json_exclusive(target, {"status": "completed"})
        with pytest.raises(FileExistsError):
            screen._atomic_write_json_exclusive(target, {"status": "changed"})
        target.unlink()
        target.symlink_to(tmp_path / "missing")
        with pytest.raises(FileExistsError):
            screen._atomic_write_json_exclusive(target, {"status": "changed"})
    finally:
        if target.exists() or target.is_symlink():
            target.unlink()


def test_verifier_has_only_standard_library_imports_and_no_gpu_probe() -> None:
    source_path = screen.REPOSITORY_ROOT / "scripts/udlm/verify_optimization_screen.py"
    source_text = source_path.read_text()
    tree = ast.parse(source_text)
    imports = {
        node.names[0].name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert imports <= {
        "__future__",
        "argparse",
        "collections",
        "copy",
        "dataclasses",
        "datetime",
        "decimal",
        "fractions",
        "hashlib",
        "importlib",
        "json",
        "math",
        "os",
        "pathlib",
        "re",
        "stat",
        "subprocess",
        "sys",
        "tempfile",
        "typing",
    }
    assert "nvidia-smi" not in source_text
    assert "torch" not in source_text
    assert "CUDA" not in source_text
    result = subprocess.run(
        [sys.executable, "-S", str(source_path), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--expected-registry-canonical-sha256" in result.stdout
    assert screen.INCOMPLETE_EXIT_STATUS == 97

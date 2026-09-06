from __future__ import annotations

import copy
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

import pytest

from scripts.udlm import superiority_gate as gate


def _load_json(relative_path: Path) -> dict:
    return json.loads(
        (gate.REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
    )


def _inference_weights() -> dict:
    return {
        "source": "ema",
        "ema_applied": True,
        "ema": {
            "shadow_parameter_count": 202,
            "decay": 0.9999,
            "num_updates": 500,
        },
    }


def _candidate_lock(*, startup_mode: str = "warm_start") -> dict:
    sampling = {"diffusion_type": "udlm", "num_steps": gate.EXPECTED_NFE}
    implementation_inputs = {"sampler_source": {"sha256": "d" * 64}}
    metric_inputs = {"schema_version": 1}
    initialization = (
        gate.EXPECTED_BASELINE_CHECKPOINT_SHA256
        if startup_mode == "warm_start"
        else None
    )
    return {
        "schema_version": 1,
        "candidate_id": "schedule-uniform-synthetic",
        "status": "locked_before_final_evaluation",
        "locked_at_utc": "2026-09-06T00:00:00+00:00",
        "protocol": {
            "id": gate.EXPECTED_PROTOCOL_ID,
            "sha256": gate.PROTOCOL_SHA256,
        },
        "selection": {
            "candidate_ledger": {
                "relative_path": "experiments/udlm/candidates/ledger.json",
                "sha256": "1" * 64,
                "schema_version": 1,
            },
            "selection_rule": gate.CANDIDATE_SELECTION_RULE,
            "checkpoint_selection_rule": gate.CHECKPOINT_SELECTION_RULE,
            "all_pilot_attempts_disclosed": True,
            "selected_without_final_seed_results": True,
            "final_seeds_used_during_selection": [],
        },
        "training": {
            "source_revision": "a" * 40,
            "training_summary": {
                "relative_path": "output/udlm/candidate/training_summary.json",
                "sha256": "2" * 64,
                "schema_version": gate.TRAINING_SUMMARY_SCHEMA_VERSION,
            },
            "exit_receipt": {
                "relative_path": "output/udlm/candidate/pilot_exit_status.json",
                "sha256": "3" * 64,
                "schema_version": gate.PILOT_EXIT_STATUS_SCHEMA_VERSION,
            },
            "runtime_config": {
                "relative_path": "output/udlm/candidate/runtime_config.json",
                "sha256": "6" * 64,
                "schema_version": 1,
            },
            "resolved_training_config_sha256": "7" * 64,
            "training_argv_sha256": "8" * 64,
            "checkpoint": {
                "relative_path": "output/udlm/candidate/final.ckpt",
                "sha256": "4" * 64,
                "size_bytes": 1234,
                "global_step": 500,
                "weights": "ema",
            },
            "startup": {
                "mode": startup_mode,
                "initialization_checkpoint_sha256": initialization,
            },
            "training_seed": 7,
            "optimizer_updates": 500,
            "world_size": 1,
            "data_exposure": {
                "global_examples_per_optimizer_step": 2046,
                "optimizer_updates": 500,
                "total_requested_examples": 1_023_000,
                "stream_partition_policy": (
                    "huggingface_split_dataset_by_node_disjoint_rank_streams"
                ),
            },
            "parameter_counts": {
                "base_model_trainable": 86_000_000,
                "time_conditioner_trainable": 787_968,
                "total_trainable": 86_787_968,
            },
        },
        "inference": {
            "evaluation_config_relative_path": (
                "scripts/exps/denovo/hparams_udlm_schedule_uniform.yaml"
            ),
            "evaluation_config_sha256": "5" * 64,
            "sampling_config": sampling,
            "sampling_sha256": gate.canonical_json_sha256(sampling),
            "checkpoint_sha256": "4" * 64,
            "weights": "ema",
            "inference_weights": _inference_weights(),
            "nfe": 128,
            "final_seeds": [0, 1, 2],
            "samples_per_seed": 1000,
            "sampler_source_sha256": "d" * 64,
            "benchmark_runner_sha256": "e" * 64,
            "implementation_inputs_sha256": gate.canonical_json_sha256(
                implementation_inputs
            ),
            "metric_inputs_sha256": gate.canonical_json_sha256(metric_inputs),
            "final_run_directories_by_seed": [
                {
                    "seed": seed,
                    "relative_path": (
                        "output/udlm/final/schedule-uniform-synthetic/" f"seed_{seed}"
                    ),
                }
                for seed in gate.EXPECTED_SEEDS
            ],
        },
        "analysis": {
            "gate_source_sha256": "a" * 64,
            "report_source_sha256": "b" * 64,
            "scipy_version": "1.15.3",
        },
        "claim_scope": gate.CLAIM_SCOPE_BY_STARTUP[startup_mode],
    }


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def _completed_pilot_attempt(
    *,
    attempt_id: str,
    candidate_id: str,
    seeds: tuple[int, ...],
    qualities: tuple[float, ...],
    diversities: tuple[float, ...],
    requested_samples: int = gate.REGISTERED_SELECTION_SAMPLES_PER_SEED,
    nfe: int = gate.REGISTERED_SELECTION_NFE,
    metric_branch: str = gate.REGISTERED_SELECTION_METRIC_BRANCH,
) -> tuple[dict, dict[str, bytes]]:
    assert len(seeds) == len(qualities) == len(diversities)
    refs = []
    blobs = {}
    for seed, quality, diversity in zip(seeds, qualities, diversities, strict=True):
        relative_path = f"experiments/udlm/pilots/{attempt_id}/seed_{seed}.json"
        evidence = {
            "schema_version": gate.PILOT_EVIDENCE_SCHEMA_VERSION,
            "artifact_kind": "pilot_evaluation",
            "status": "completed",
            "attempt_id": attempt_id,
            "candidate_id": candidate_id,
            "pilot_seed": seed,
            "final_seed_results_included": False,
            "checkpoint_sha256": "c" * 64,
            "evaluation": {
                "metric_branch": metric_branch,
                "requested_samples": requested_samples,
                "nfe": nfe,
                "quality": quality,
                "diversity": diversity,
            },
        }
        payload = _json_bytes(evidence)
        blobs[relative_path] = payload
        refs.append(
            {
                "artifact_kind": "pilot_evaluation",
                "pilot_seed": seed,
                "relative_path": relative_path,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "schema_version": gate.PILOT_EVIDENCE_SCHEMA_VERSION,
            }
        )
    registered = (
        seeds == gate.REGISTERED_SELECTION_PILOT_SEEDS
        and requested_samples == gate.REGISTERED_SELECTION_SAMPLES_PER_SEED
        and nfe == gate.REGISTERED_SELECTION_NFE
        and metric_branch == gate.REGISTERED_SELECTION_METRIC_BRANCH
    )
    return (
        {
            "attempt_id": attempt_id,
            "candidate_id": candidate_id,
            "status": "completed",
            "eligible_for_selection": registered,
            "ineligibility_reason": (
                None if registered else gate.NONREGISTERED_OPERATING_POINT_REASON
            ),
            "pilot_seeds": list(seeds),
            "selection_score": (
                {
                    "mean_released_quality": statistics.fmean(qualities),
                    "mean_released_diversity": statistics.fmean(diversities),
                }
                if registered
                else None
            ),
            "artifact_refs": refs,
        },
        blobs,
    )


def _failed_pilot_attempt(
    *, attempt_id: str, candidate_id: str, seed: int
) -> tuple[dict, dict[str, bytes]]:
    relative_path = f"experiments/udlm/pilots/{attempt_id}/seed_{seed}.json"
    evidence = {
        "schema_version": gate.PILOT_EVIDENCE_SCHEMA_VERSION,
        "artifact_kind": "pilot_failure",
        "status": "failed",
        "attempt_id": attempt_id,
        "candidate_id": candidate_id,
        "pilot_seed": seed,
        "final_seed_results_included": False,
        "failure": {
            "stage": "sampling",
            "reason": "synthetic pilot process exited nonzero",
        },
    }
    payload = _json_bytes(evidence)
    return (
        {
            "attempt_id": attempt_id,
            "candidate_id": candidate_id,
            "status": "failed",
            "eligible_for_selection": False,
            "ineligibility_reason": gate.FAILED_PILOT_REASON,
            "pilot_seeds": [seed],
            "selection_score": None,
            "artifact_refs": [
                {
                    "artifact_kind": "pilot_failure",
                    "pilot_seed": seed,
                    "relative_path": relative_path,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "schema_version": gate.PILOT_EVIDENCE_SCHEMA_VERSION,
                }
            ],
        },
        {relative_path: payload},
    )


def _pilot_ledger(*, attempts: list[dict], selected_attempt_id: str) -> dict:
    selected = next(
        attempt for attempt in attempts if attempt["attempt_id"] == selected_attempt_id
    )
    return {
        "schema_version": gate.CANDIDATE_LEDGER_SCHEMA_VERSION,
        "protocol_id": gate.EXPECTED_PROTOCOL_ID,
        "status": "closed_before_final_evaluation",
        "final_seed_results_included": False,
        "attempts": attempts,
        "selection": {
            "candidate_id": selected["candidate_id"],
            "selected_attempt_id": selected_attempt_id,
            "rule": gate.CANDIDATE_SELECTION_RULE,
            "checkpoint_selection_rule": gate.CHECKPOINT_SELECTION_RULE,
            "selected_without_final_seed_results": True,
        },
    }


def _artifact_loader(blobs: dict[str, bytes]):
    return lambda relative_path: blobs[relative_path.as_posix()]


def _metric_branch(*, unique_count: int, quality_count: int, diversity: float) -> dict:
    return {
        "validity": 1.0,
        "valid_count": 1000,
        "validity_denominator": 1000,
        "uniqueness": unique_count / 1000,
        "unique_count": unique_count,
        "uniqueness_denominator": 1000,
        "quality": quality_count / 1000,
        "quality_count": quality_count,
        "quality_denominator": 1000,
        "diversity": diversity,
    }


def _aggregate(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values),
        "values_by_seed": [
            {"seed": seed, "value": value}
            for seed, value in zip(gate.EXPECTED_SEEDS, values, strict=True)
        ],
    }


def _candidate_report(
    *,
    unique_counts: tuple[int, int, int] = (1000, 1000, 1000),
    quality_counts: tuple[int, int, int] = (900, 900, 900),
    diversities: tuple[float, float, float] = (0.84, 0.84, 0.84),
) -> dict:
    lock = _candidate_lock()
    seed_runs = []
    released_values = {metric: [] for metric in gate.METRICS}
    for seed, unique_count, quality_count, diversity in zip(
        gate.EXPECTED_SEEDS,
        unique_counts,
        quality_counts,
        diversities,
        strict=True,
    ):
        branch = _metric_branch(
            unique_count=unique_count,
            quality_count=quality_count,
            diversity=diversity,
        )
        for metric in gate.METRICS:
            released_values[metric].append(branch[metric])
        seed_runs.append(
            {
                "seed": seed,
                "started_at_utc": f"2026-09-06T00:0{seed + 1}:00+00:00",
                "summary_path": str(
                    gate.REPOSITORY_ROOT
                    / "output/udlm/final/schedule-uniform-synthetic"
                    / f"seed_{seed}/summary.json"
                ),
                "raw_samples_sha256": f"{seed + 6:x}" * 64,
                "summary_sha256": f"{seed + 9:x}" * 64,
                "metrics": {
                    "released_comparable": branch,
                    "strict": dict(branch),
                },
                "git": {
                    "commit": "b" * 40,
                    "dirty": False,
                },
                "inference_weights": _inference_weights(),
            }
        )
    aggregates = {
        metric: _aggregate(values) for metric, values in released_values.items()
    }
    return {
        "schema_version": gate.denovo_report.REPORT_SCHEMA_VERSION,
        "status": "completed",
        "required_protocol": {
            "seeds": [0, 1, 2],
            "samples_per_seed": 1000,
            "seed_count": 3,
            "total_requested_samples": 3000,
        },
        "checkpoint": {
            "diffusion_type": "udlm",
            "sha256": lock["training"]["checkpoint"]["sha256"],
            "size_bytes": lock["training"]["checkpoint"]["size_bytes"],
            "global_step": lock["training"]["checkpoint"]["global_step"],
        },
        "config": {
            "sha256": lock["inference"]["evaluation_config_sha256"],
            "sampling_sha256": lock["inference"]["sampling_sha256"],
            "sampling": lock["inference"]["sampling_config"],
            "git_tracking": {
                "relative_path": lock["inference"]["evaluation_config_relative_path"]
            },
        },
        "generation_protocol": {
            "diffusion_type": "udlm",
            "nfe": 128,
            "num_steps": 128,
            "nfe_by_seed": [{"seed": seed, "nfe": 128} for seed in gate.EXPECTED_SEEDS],
            "inference_weights": _inference_weights(),
        },
        "inference_weights": _inference_weights(),
        "runner_sha256": "e" * 64,
        "implementation_inputs": {"sampler_source": {"sha256": "d" * 64}},
        "metric_inputs": {"schema_version": 1},
        "seed_runs": seed_runs,
        "aggregate_metrics": {
            "released_comparable": aggregates,
            "strict": copy.deepcopy(aggregates),
        },
        "environment_consistency": {
            "all_seed_signatures_equal": True,
            "all_launch_policies_equal": True,
            "idle_gpu_policy_verified": True,
            "distinct_raw_sample_csv_sha256": True,
        },
    }


@pytest.fixture
def protocol() -> dict:
    return _load_json(gate.PROTOCOL_RELATIVE_PATH)


@pytest.fixture
def baseline() -> dict:
    return _load_json(gate.BASELINE_RELATIVE_PATH)


@pytest.fixture
def baseline_rescore() -> dict:
    return _load_json(gate.BASELINE_RESCORE_RELATIVE_PATH)


def test_pinned_protocol_and_baseline_hashes_and_semantics(
    protocol, baseline, baseline_rescore
):
    protocol_bytes = (gate.REPOSITORY_ROOT / gate.PROTOCOL_RELATIVE_PATH).read_bytes()
    baseline_bytes = (gate.REPOSITORY_ROOT / gate.BASELINE_RELATIVE_PATH).read_bytes()
    baseline_rescore_bytes = (
        gate.REPOSITORY_ROOT / gate.BASELINE_RESCORE_RELATIVE_PATH
    ).read_bytes()

    assert hashlib.sha256(protocol_bytes).hexdigest() == gate.PROTOCOL_SHA256
    assert hashlib.sha256(baseline_bytes).hexdigest() == gate.BASELINE_SHA256
    assert (
        hashlib.sha256(baseline_rescore_bytes).hexdigest()
        == gate.BASELINE_RESCORE_SHA256
    )
    gate.validate_protocol(protocol)
    validated = gate.validate_baseline_manifest(baseline)
    rescore = gate.validate_baseline_rescore_attestation(baseline_rescore, baseline)
    assert validated["means"]["quality"] == pytest.approx(0.858)
    assert validated["pooled_valid"] == 3000
    assert rescore["source_revision"] == gate.EXPECTED_BASELINE_RESCORE_SOURCE_REVISION
    assert rescore["all_rows_and_manifest_values_exact_match"] is True
    assert rescore["network_controls"]["os_or_process_network_isolation"] is False


def test_public_evaluator_rejects_weakened_protocol_with_frozen_hash_label(
    protocol, baseline, baseline_rescore
):
    protocol["uncertainty_gates"]["quality"][
        "candidate_minus_baseline_lower_bound_strictly_greater_than"
    ] = -1.0

    with pytest.raises(gate.GateValidationError, match="not the frozen value"):
        gate.evaluate_candidate_report(
            _candidate_report(quality_counts=(875, 876, 877)),
            baseline,
            baseline_rescore,
            protocol,
            _candidate_lock(),
        )


def test_protocol_semantically_pins_registered_pilot_operating_point(
    protocol, monkeypatch
):
    protocol["selection_firewall"]["eligible_nfe"] = 1
    monkeypatch.setattr(
        gate, "PROTOCOL_CANONICAL_SHA256", gate.canonical_json_sha256(protocol)
    )

    with pytest.raises(gate.GateValidationError, match="eligible pilot NFE"):
        gate.validate_protocol(protocol)


@pytest.mark.parametrize(
    "payload",
    [b'{"a": 1, "a": 2}', b'{"a": NaN}', b'{"a": Infinity}'],
)
def test_strict_json_rejects_duplicates_and_nonfinite_numbers(payload):
    with pytest.raises(gate.GateValidationError):
        gate.strict_json_loads(payload, label="test payload")


def test_boundary_safe_newcombe_interval_for_two_all_success_samples():
    result = gate.newcombe_wilson_lower_difference(
        3000,
        3000,
        3000,
        3000,
        z=1.6448536269514722,
    )

    assert result["difference"] == 0.0
    assert result["lower_bound"] == pytest.approx(-0.0009010352213835171)
    assert result["lower_bound"] > -0.005


def test_welch_interval_uses_independent_seed_level_estimates():
    result = gate.welch_lower_difference(
        [0.90, 0.90, 0.90],
        [0.849, 0.872, 0.853],
        confidence=0.95,
    )

    assert result["difference"] == pytest.approx(0.042)
    assert result["degrees_of_freedom"] == pytest.approx(2.0)
    assert result["lower_bound"] == pytest.approx(0.021283873558568735)


def test_both_zero_variance_welch_inputs_fail_closed():
    with pytest.raises(gate.GateValidationError, match="both sample variances"):
        gate.welch_lower_difference([0.9, 0.9, 0.9], [0.8, 0.8, 0.8], confidence=0.95)


def test_one_zero_variance_welch_input_remains_defined():
    result = gate.welch_lower_difference(
        [0.9, 0.9, 0.9], [0.8, 0.81, 0.82], confidence=0.95
    )

    assert result["degrees_of_freedom"] == pytest.approx(2.0)
    assert result["lower_bound"] == pytest.approx(0.07314145539151921)


def test_complete_registered_gate_passes_strong_candidate(
    protocol, baseline, baseline_rescore
):
    decision = gate.evaluate_candidate_report(
        _candidate_report(),
        baseline,
        baseline_rescore,
        protocol,
        _candidate_lock(),
    )

    assert decision["superiority_gate_passed"] is True
    assert decision["all_point_estimate_gates_passed"] is True
    assert decision["all_uncertainty_gates_passed"] is True
    assert decision["candidate"]["nfe"] == 128
    assert decision["candidate"]["inference_weights"] == _inference_weights()
    assert decision["baseline"]["rescore_attestation"]["status"] == (
        "completed_exact_match"
    )
    assert decision["claim"]["scope"] == "operational_continuation_only"
    assert decision["claim"]["method_only_claim_supported"] is False


def test_quality_equality_fails_strict_point_and_overall_gate(
    protocol, baseline, baseline_rescore
):
    decision = gate.evaluate_candidate_report(
        _candidate_report(quality_counts=(858, 858, 858)),
        baseline,
        baseline_rescore,
        protocol,
        _candidate_lock(),
    )

    assert decision["metrics"]["quality"]["point_gate_passed"] is False
    assert decision["superiority_gate_passed"] is False


def test_diversity_point_margin_is_inclusive(protocol, baseline, baseline_rescore):
    threshold = baseline["released_comparable"]["mean"]["diversity"] - 0.005
    decision = gate.evaluate_candidate_report(
        _candidate_report(diversities=(threshold, threshold, threshold)),
        baseline,
        baseline_rescore,
        protocol,
        _candidate_lock(),
    )

    assert decision["metrics"]["diversity"]["point_gate_passed"] is True


def test_any_wrong_final_nfe_is_rejected(protocol, baseline, baseline_rescore):
    report = _candidate_report()
    report["generation_protocol"]["nfe_by_seed"][1]["nfe"] = 64

    with pytest.raises(gate.GateValidationError, match="NFE differs"):
        gate.evaluate_candidate_report(
            report, baseline, baseline_rescore, protocol, _candidate_lock()
        )


def test_runtime_ema_application_receipt_is_required(
    protocol, baseline, baseline_rescore
):
    report = _candidate_report()
    report["generation_protocol"]["inference_weights"] = {
        "source": "raw_model",
        "ema_applied": False,
        "ema": None,
    }

    with pytest.raises(gate.GateValidationError, match="EMA inference weights"):
        gate.evaluate_candidate_report(
            report, baseline, baseline_rescore, protocol, _candidate_lock()
        )


@pytest.mark.parametrize("location", ["top_level", "seed"])
def test_runtime_ema_receipt_must_be_identical_everywhere(
    protocol, baseline, baseline_rescore, location
):
    report = _candidate_report()
    if location == "top_level":
        report["inference_weights"]["ema"]["num_updates"] = 499
    else:
        report["seed_runs"][1]["inference_weights"]["ema"]["decay"] = 0.9

    with pytest.raises(
        gate.GateValidationError, match="inference-weight receipt disagrees"
    ):
        gate.evaluate_candidate_report(
            report, baseline, baseline_rescore, protocol, _candidate_lock()
        )


def test_observed_runner_must_equal_prelocked_runner(
    protocol, baseline, baseline_rescore
):
    report = _candidate_report()
    report["runner_sha256"] = "0" * 64

    with pytest.raises(gate.GateValidationError, match="runner differs"):
        gate.evaluate_candidate_report(
            report, baseline, baseline_rescore, protocol, _candidate_lock()
        )


def test_full_implementation_and_metric_maps_must_be_prelocked(
    protocol, baseline, baseline_rescore
):
    report = _candidate_report()
    report["implementation_inputs"]["extra_source"] = {"sha256": "0" * 64}
    with pytest.raises(gate.GateValidationError, match="implementation inputs differ"):
        gate.evaluate_candidate_report(
            report, baseline, baseline_rescore, protocol, _candidate_lock()
        )

    report = _candidate_report()
    report["metric_inputs"]["unexpected"] = True
    with pytest.raises(gate.GateValidationError, match="metric inputs differ"):
        gate.evaluate_candidate_report(
            report, baseline, baseline_rescore, protocol, _candidate_lock()
        )


def test_lock_rejects_final_seed_leak_and_non_ema_weights(protocol):
    candidate_lock = _candidate_lock()
    candidate_lock["selection"]["selection_rule"] = "highest eligible quality"
    with pytest.raises(gate.GateValidationError, match="frozen enum"):
        gate.validate_candidate_lock(candidate_lock, protocol)

    candidate_lock = _candidate_lock()
    candidate_lock["selection"]["final_seeds_used_during_selection"] = [0]
    with pytest.raises(gate.GateValidationError, match="must not be used"):
        gate.validate_candidate_lock(candidate_lock, protocol)

    candidate_lock = _candidate_lock()
    candidate_lock["inference"]["weights"] = "raw"
    with pytest.raises(gate.GateValidationError, match="must be EMA"):
        gate.validate_candidate_lock(candidate_lock, protocol)

    candidate_lock = _candidate_lock()
    candidate_lock["inference"]["inference_weights"]["ema"]["num_updates"] = 1
    with pytest.raises(gate.GateValidationError, match="must equal optimizer updates"):
        gate.validate_candidate_lock(candidate_lock, protocol)

    candidate_lock = _candidate_lock()
    candidate_lock["training"]["training_summary"]["schema_version"] = 1
    with pytest.raises(gate.GateValidationError, match="summary schema version"):
        gate.validate_candidate_lock(candidate_lock, protocol)

    candidate_lock = _candidate_lock()
    candidate_lock["training"]["exit_receipt"]["schema_version"] = 1
    with pytest.raises(gate.GateValidationError, match="receipt schema version"):
        gate.validate_candidate_lock(candidate_lock, protocol)


def test_scratch_lock_has_narrow_single_trajectory_scope(protocol):
    candidate_lock = _candidate_lock(startup_mode="scratch")

    validated = gate.validate_candidate_lock(candidate_lock, protocol)

    assert validated["startup_mode"] == "scratch"
    assert validated["claim_scope"] == (
        "single_training_trajectory_checkpoint_comparison_only"
    )


def test_tampered_baseline_count_is_rejected(baseline):
    baseline["released_comparable"]["per_seed"][0]["quality_count"] += 1

    with pytest.raises(gate.GateValidationError, match="quality"):
        gate.validate_baseline_manifest(baseline)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("status",), "completed", "not completed exact-match"),
        (
            ("source", "clean_pushed_checks", "before_computation"),
            False,
            "clean-source check",
        ),
        (("source", "revision"), "0" * 40, "source revision is unexpected"),
        (
            ("source", "files", "benchmark_runner", "sha256"),
            "0" * 64,
            "runtime module.*unbound from source",
        ),
        (
            ("implementation", "benchmark_schema_version"),
            6,
            "schemas are stale",
        ),
        (
            ("implementation", "metric_inputs_sha256"),
            "0" * 64,
            "metric-input self-hash",
        ),
        (
            ("implementation", "runtime_modules_sha256"),
            "0" * 64,
            "runtime-module self-hash",
        ),
        (
            ("protocol", "network_controls", "os_or_process_network_isolation"),
            True,
            "network-control claim",
        ),
        (
            ("seed_results", 0, "row_comparison", "all_match"),
            False,
            "row comparison all_match",
        ),
        (
            ("seed_results", 0, "inputs", "raw_samples_csv", "sha256"),
            "0" * 64,
            "digest disagrees with frozen manifest",
        ),
        (
            ("seed_results", 0, "metrics", "strict", "quality_count"),
            1,
            "quality_count",
        ),
        (
            ("seed_results", 0, "failure_counts", "strict_decode_failed"),
            11,
            "strict_decode_failed",
        ),
        (
            ("aggregate_metrics", "strict", "quality", "mean"),
            0.1,
            "mean",
        ),
        (
            ("strict_vs_repaired_funnel", "strict_valid"),
            1,
            "funnel disagrees",
        ),
        (
            (
                "manifest_comparison",
                "all_seed_rows_metrics_failures_hashes_and_aggregates_match",
            ),
            False,
            "manifest comparison is incomplete",
        ),
    ],
)
def test_baseline_rescore_attestation_tampering_fails_closed(
    baseline, baseline_rescore, path, value, message
):
    cursor = baseline_rescore
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] = value

    with pytest.raises(gate.GateValidationError, match=message):
        gate.validate_baseline_rescore_attestation(baseline_rescore, baseline)


def test_public_evaluator_cannot_bypass_baseline_rescore_attestation(
    protocol, baseline, baseline_rescore
):
    baseline_rescore["seed_results"][2]["manifest_comparison"][
        "all_counts_metrics_and_artifact_hashes_match"
    ] = False

    with pytest.raises(gate.GateValidationError, match="manifest comparison"):
        gate.evaluate_candidate_report(
            _candidate_report(),
            baseline,
            baseline_rescore,
            protocol,
            _candidate_lock(),
        )


def test_candidate_ledger_recomputes_scores_and_selects_deterministically():
    lower, lower_blobs = _completed_pilot_attempt(
        attempt_id="a1",
        candidate_id="schedule-low",
        seeds=(1000, 1001),
        qualities=(0.80, 0.84),
        diversities=(0.70, 0.72),
    )
    winner, winner_blobs = _completed_pilot_attempt(
        attempt_id="a2",
        candidate_id="schedule-uniform-synthetic",
        seeds=(1000, 1001),
        qualities=(0.85, 0.85),
        diversities=(0.74, 0.74),
    )
    lexical_loser, loser_blobs = _completed_pilot_attempt(
        attempt_id="z2",
        candidate_id="schedule-equal",
        seeds=(1000, 1001),
        qualities=(0.85, 0.85),
        diversities=(0.74, 0.74),
    )
    failed, failed_blobs = _failed_pilot_attempt(
        attempt_id="f1", candidate_id="schedule-failed", seed=1004
    )
    blobs = lower_blobs | winner_blobs | loser_blobs | failed_blobs
    ledger = _pilot_ledger(
        attempts=[failed, lexical_loser, lower, winner],
        selected_attempt_id="a2",
    )

    result = gate.validate_candidate_ledger(
        ledger,
        candidate_id="schedule-uniform-synthetic",
        artifact_loader=_artifact_loader(blobs),
    )

    assert result["attempt_count"] == 4
    assert result["eligible_attempt_count"] == 3
    assert result["committed_pilot_artifact_count"] == 7
    assert result["ineligible_completed_attempt_count"] == 0
    assert result["failed_attempt_count"] == 1
    assert result["selected_attempt_id"] == "a2"
    assert result["selected_score"] == {
        "mean_released_quality": 0.85,
        "mean_released_diversity": 0.74,
    }
    assert result["selection_recomputed_from_pilot_evidence"] is True


def test_candidate_ledger_enforces_seed_status_and_frozen_rule():
    attempt, blobs = _completed_pilot_attempt(
        attempt_id="a1",
        candidate_id="schedule-uniform-synthetic",
        seeds=(1000, 1001),
        qualities=(0.85, 0.85),
        diversities=(0.74, 0.74),
    )
    ledger = _pilot_ledger(attempts=[attempt], selected_attempt_id="a1")
    ledger["attempts"][0]["pilot_seeds"] = [0]
    with pytest.raises(gate.GateValidationError, match="greater than or equal to 1000"):
        gate.validate_candidate_ledger(
            ledger,
            candidate_id="schedule-uniform-synthetic",
            artifact_loader=_artifact_loader(blobs),
        )

    attempt["pilot_seeds"] = [1000, 1001]
    ledger = _pilot_ledger(attempts=[attempt], selected_attempt_id="a1")
    ledger["selection"]["rule"] = "highest eligible quality"
    with pytest.raises(gate.GateValidationError, match="frozen enum"):
        gate.validate_candidate_ledger(
            ledger,
            candidate_id="schedule-uniform-synthetic",
            artifact_loader=_artifact_loader(blobs),
        )


def test_exact_registered_attempt_cannot_be_arbitrarily_excluded():
    attempt, blobs = _completed_pilot_attempt(
        attempt_id="a1",
        candidate_id="schedule-uniform-synthetic",
        seeds=gate.REGISTERED_SELECTION_PILOT_SEEDS,
        qualities=(0.85, 0.85),
        diversities=(0.74, 0.74),
    )
    attempt["eligible_for_selection"] = False
    attempt["selection_score"] = None
    attempt["ineligibility_reason"] = gate.NONREGISTERED_OPERATING_POINT_REASON
    ledger = _pilot_ledger(attempts=[attempt], selected_attempt_id="a1")

    with pytest.raises(gate.GateValidationError, match="eligibility disagrees"):
        gate.validate_candidate_ledger(
            ledger,
            candidate_id="schedule-uniform-synthetic",
            artifact_loader=_artifact_loader(blobs),
        )


@pytest.mark.parametrize(
    "operating_point_override",
    [
        {"requested_samples": 1},
        {"nfe": 64},
    ],
)
def test_one_sample_or_mismatched_nfe_attempt_is_disclosed_but_ineligible(
    operating_point_override,
):
    registered, registered_blobs = _completed_pilot_attempt(
        attempt_id="registered",
        candidate_id="schedule-uniform-synthetic",
        seeds=gate.REGISTERED_SELECTION_PILOT_SEEDS,
        qualities=(0.80, 0.80),
        diversities=(0.70, 0.70),
    )
    engineering, engineering_blobs = _completed_pilot_attempt(
        attempt_id="engineering",
        candidate_id="schedule-noisy-health",
        seeds=gate.REGISTERED_SELECTION_PILOT_SEEDS,
        qualities=(1.0, 1.0),
        diversities=(1.0, 1.0),
        **operating_point_override,
    )
    ledger = _pilot_ledger(
        attempts=[engineering, registered], selected_attempt_id="registered"
    )

    result = gate.validate_candidate_ledger(
        ledger,
        candidate_id="schedule-uniform-synthetic",
        artifact_loader=_artifact_loader(registered_blobs | engineering_blobs),
    )

    assert result["eligible_attempt_count"] == 1
    assert result["ineligible_completed_attempt_count"] == 1
    assert result["selected_attempt_id"] == "registered"

    engineering["eligible_for_selection"] = True
    engineering["ineligibility_reason"] = None
    engineering["selection_score"] = {
        "mean_released_quality": 1.0,
        "mean_released_diversity": 1.0,
    }
    with pytest.raises(gate.GateValidationError, match="eligibility disagrees"):
        gate.validate_candidate_ledger(
            ledger,
            candidate_id="schedule-uniform-synthetic",
            artifact_loader=_artifact_loader(registered_blobs | engineering_blobs),
        )


def test_valid_health_pilot_with_nonregistered_seed_is_disclosed():
    registered, registered_blobs = _completed_pilot_attempt(
        attempt_id="registered",
        candidate_id="schedule-uniform-synthetic",
        seeds=gate.REGISTERED_SELECTION_PILOT_SEEDS,
        qualities=(0.80, 0.80),
        diversities=(0.70, 0.70),
    )
    health, health_blobs = _completed_pilot_attempt(
        attempt_id="health",
        candidate_id="schedule-health-only",
        seeds=(1002,),
        qualities=(1.0,),
        diversities=(1.0,),
        requested_samples=1,
        nfe=10,
    )
    ledger = _pilot_ledger(
        attempts=[health, registered], selected_attempt_id="registered"
    )

    result = gate.validate_candidate_ledger(
        ledger,
        candidate_id="schedule-uniform-synthetic",
        artifact_loader=_artifact_loader(registered_blobs | health_blobs),
    )

    assert health["ineligibility_reason"] == (gate.NONREGISTERED_OPERATING_POINT_REASON)
    assert result["eligible_attempt_count"] == 1
    assert result["ineligible_completed_attempt_count"] == 1
    assert result["selected_attempt_id"] == "registered"


@pytest.mark.parametrize(
    ("field", "tampered_value"),
    [
        ("attempt_id", "different-attempt"),
        ("candidate_id", "different-candidate"),
        ("pilot_seed", 1001),
        ("final_seed_results_included", True),
    ],
)
def test_candidate_ledger_rejects_semantically_misbound_artifact(field, tampered_value):
    attempt, blobs = _completed_pilot_attempt(
        attempt_id="a1",
        candidate_id="schedule-uniform-synthetic",
        seeds=(1000, 1001),
        qualities=(0.85, 0.85),
        diversities=(0.74, 0.74),
    )
    ledger = _pilot_ledger(attempts=[attempt], selected_attempt_id="a1")
    ref = ledger["attempts"][0]["artifact_refs"][0]
    evidence = json.loads(blobs[ref["relative_path"]])
    evidence[field] = tampered_value
    tampered = _json_bytes(evidence)
    blobs[ref["relative_path"]] = tampered
    ref["sha256"] = hashlib.sha256(tampered).hexdigest()

    with pytest.raises(gate.GateValidationError, match=f"{field} disagrees"):
        gate.validate_candidate_ledger(
            ledger,
            candidate_id="schedule-uniform-synthetic",
            artifact_loader=_artifact_loader(blobs),
        )


def test_candidate_ledger_rejects_declared_score_tampering():
    attempt, blobs = _completed_pilot_attempt(
        attempt_id="a1",
        candidate_id="schedule-uniform-synthetic",
        seeds=(1000, 1001),
        qualities=(0.80, 0.90),
        diversities=(0.70, 0.80),
    )
    ledger = _pilot_ledger(attempts=[attempt], selected_attempt_id="a1")
    ledger["attempts"][0]["selection_score"]["mean_released_quality"] = 0.99

    with pytest.raises(gate.GateValidationError, match="mean released quality"):
        gate.validate_candidate_ledger(
            ledger,
            candidate_id="schedule-uniform-synthetic",
            artifact_loader=_artifact_loader(blobs),
        )


def test_candidate_ledger_rejects_wrong_winner_including_lexical_tie_break():
    first, first_blobs = _completed_pilot_attempt(
        attempt_id="a-first",
        candidate_id="schedule-uniform-synthetic",
        seeds=(1000, 1001),
        qualities=(0.85, 0.85),
        diversities=(0.74, 0.74),
    )
    later, later_blobs = _completed_pilot_attempt(
        attempt_id="z-later",
        candidate_id="schedule-uniform-synthetic",
        seeds=(1000, 1001),
        qualities=(0.85, 0.85),
        diversities=(0.74, 0.74),
    )
    ledger = _pilot_ledger(attempts=[later, first], selected_attempt_id="z-later")

    with pytest.raises(
        gate.GateValidationError, match="deterministic pilot-score winner"
    ):
        gate.validate_candidate_ledger(
            ledger,
            candidate_id="schedule-uniform-synthetic",
            artifact_loader=_artifact_loader(first_blobs | later_blobs),
        )


def test_candidate_ledger_rejects_content_free_and_malformed_failure_evidence():
    attempt, blobs = _completed_pilot_attempt(
        attempt_id="a1",
        candidate_id="schedule-uniform-synthetic",
        seeds=(1000, 1001),
        qualities=(0.85, 0.85),
        diversities=(0.74, 0.74),
    )
    ledger = _pilot_ledger(attempts=[attempt], selected_attempt_id="a1")
    ref = ledger["attempts"][0]["artifact_refs"][0]
    content_free = b'{"status":"completed"}\n'
    blobs[ref["relative_path"]] = content_free
    ref["sha256"] = hashlib.sha256(content_free).hexdigest()
    with pytest.raises(gate.GateValidationError, match="fields are invalid"):
        gate.validate_candidate_ledger(
            ledger,
            candidate_id="schedule-uniform-synthetic",
            artifact_loader=_artifact_loader(blobs),
        )

    winner, winner_blobs = _completed_pilot_attempt(
        attempt_id="a1",
        candidate_id="schedule-uniform-synthetic",
        seeds=(1000, 1001),
        qualities=(0.85, 0.85),
        diversities=(0.74, 0.74),
    )
    failed, failed_blobs = _failed_pilot_attempt(
        attempt_id="f1", candidate_id="schedule-failed", seed=1001
    )
    ledger = _pilot_ledger(attempts=[winner, failed], selected_attempt_id="a1")
    failure_ref = ledger["attempts"][1]["artifact_refs"][0]
    failure_evidence = json.loads(failed_blobs[failure_ref["relative_path"]])
    failure_evidence["failure"]["reason"] = ""
    malformed_failure = _json_bytes(failure_evidence)
    failed_blobs[failure_ref["relative_path"]] = malformed_failure
    failure_ref["sha256"] = hashlib.sha256(malformed_failure).hexdigest()
    with pytest.raises(gate.GateValidationError, match="reason must be nonempty"):
        gate.validate_candidate_ledger(
            ledger,
            candidate_id="schedule-uniform-synthetic",
            artifact_loader=_artifact_loader(winner_blobs | failed_blobs),
        )


def test_training_summary_and_exit_receipt_are_joined_to_lock(
    tmp_path, monkeypatch, protocol
):
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", tmp_path)
    candidate_lock = _candidate_lock()
    summary_path = (
        tmp_path / candidate_lock["training"]["training_summary"]["relative_path"]
    )
    receipt_path = (
        tmp_path / candidate_lock["training"]["exit_receipt"]["relative_path"]
    )
    runtime_path = (
        tmp_path / candidate_lock["training"]["runtime_config"]["relative_path"]
    )
    summary_path.parent.mkdir(parents=True)
    resolved_config = {
        "data": "safe",
        "seed": 7,
        "training": {"ema": 0.9999},
        "loader": {"batch_size": 2046, "global_batch_size": 2046},
        "trainer": {
            "devices": 1,
            "num_nodes": 1,
            "max_steps": 500,
            "accumulate_grad_batches": 1,
        },
    }
    training_argv = ["/repo/scripts/train.py", "seed=7"]
    resolved_config_sha = gate.canonical_json_sha256(resolved_config)
    training_argv_sha = gate.canonical_json_sha256(training_argv)
    runtime = {
        "schema_version": 1,
        "status": "preflight_completed",
        "source_revision": "a" * 40,
        "resolved_training_config": resolved_config,
        "resolved_training_config_sha256": resolved_config_sha,
        "training_argv": training_argv,
        "training_argv_sha256": training_argv_sha,
    }
    runtime_bytes = (json.dumps(runtime, sort_keys=True) + "\n").encode()
    runtime_path.write_bytes(runtime_bytes)
    runtime_sha = hashlib.sha256(runtime_bytes).hexdigest()
    candidate_lock["training"]["runtime_config"]["sha256"] = runtime_sha
    candidate_lock["training"]["resolved_training_config_sha256"] = resolved_config_sha
    candidate_lock["training"]["training_argv_sha256"] = training_argv_sha
    training_accounting = {
        "training_seed": 7,
        "optimizer_updates": 500,
        "world_size": 1,
        "micro_batch_size_per_rank": 2046,
        "accumulate_grad_batches": 1,
        "effective_global_examples_per_optimizer_step": 2046,
        "total_requested_example_exposures": 1_023_000,
        "hosted_stream_rank_partition_policy": (
            "huggingface_split_dataset_by_node_disjoint_rank_streams"
        ),
        "trainable_parameter_counts": {
            "base_backbone": 86_000_000,
            "time_conditioner": 787_968,
            "total": 86_787_968,
        },
    }
    summary = {
        "schema_version": gate.TRAINING_SUMMARY_SCHEMA_VERSION,
        "status": "completed",
        "source_revision": "a" * 40,
        "resolved_training_config_sha256": resolved_config_sha,
        "training_argv_sha256": training_argv_sha,
        "runtime_config": {"sha256": runtime_sha},
        "observed_training_state": {"global_step": 500, "world_size": 1},
        "training_accounting": training_accounting,
        "final_checkpoint": {
            "sha256": "4" * 64,
            "size_bytes": 1234,
            "semantic_audit": {
                "global_step": 500,
                "ema": {
                    "all_finite": True,
                    "floating_tensor_count": 202,
                },
                "ema_metadata": _inference_weights()["ema"],
                "live_ema_match": {
                    "exact_tensor_values": True,
                    "tensor_count": 202,
                },
                "live_model_match": {"exact_tensor_values": True},
            },
        },
        "startup": {
            "mode": "warm_start",
            "verified_mdlm_warm_start_report": {
                "source_sha256": gate.EXPECTED_BASELINE_CHECKPOINT_SHA256,
                "weights": "ema",
            },
        },
    }
    summary_bytes = (json.dumps(summary, sort_keys=True) + "\n").encode()
    summary_path.write_bytes(summary_bytes)
    summary_sha = hashlib.sha256(summary_bytes).hexdigest()
    candidate_lock["training"]["training_summary"]["sha256"] = summary_sha
    receipt = {
        "schema_version": gate.PILOT_EXIT_STATUS_SCHEMA_VERSION,
        "status": "completed",
        "overall_status": "completed",
        "process_exit_status": 0,
        "expected_contract": {
            "source_revision": "a" * 40,
            "resolved_training_config_sha256": resolved_config_sha,
            "training_argv_sha256": training_argv_sha,
            "max_steps": 500,
            "world_size": 1,
            "initialization_checkpoint_sha256": (
                gate.EXPECTED_BASELINE_CHECKPOINT_SHA256
            ),
        },
        "source_at_receipt": {"verified": True},
        "training_summary": {
            "valid_and_launch_bound": True,
            "artifact": {"sha256": summary_sha},
            "validated_bindings": {
                "training_accounting": training_accounting,
                "ema_metadata": _inference_weights()["ema"],
            },
        },
        "final_checkpoint": {
            "matches_training_summary_snapshot": True,
            "artifact": {"sha256": "4" * 64, "size_bytes": 1234},
        },
        "runtime_config": {
            "matches_training_summary_snapshot": True,
            "semantic_validation_passed": True,
            "artifact": {"sha256": runtime_sha},
        },
    }
    receipt_bytes = (json.dumps(receipt, sort_keys=True) + "\n").encode()
    receipt_path.write_bytes(receipt_bytes)
    candidate_lock["training"]["exit_receipt"]["sha256"] = hashlib.sha256(
        receipt_bytes
    ).hexdigest()
    normalized = gate.validate_candidate_lock(candidate_lock, protocol)

    evidence = gate.validate_training_evidence(normalized)

    assert evidence["ema_finite_and_checkpoint_bound"] is True
    assert evidence["successful_exit_receipt"] is True
    assert evidence["training_accounting"] == training_accounting
    wrong_lock = copy.deepcopy(normalized)
    wrong_lock["parameter_counts"]["total_trainable"] += 1
    with pytest.raises(
        gate.GateValidationError, match="parameter counts disagree with lock"
    ):
        gate.validate_training_evidence(wrong_lock)
    wrong_lock = copy.deepcopy(normalized)
    wrong_lock["inference_weights"]["ema"]["decay"] = 0.9
    with pytest.raises(
        gate.GateValidationError, match="EMA metadata disagrees with lock"
    ):
        gate.validate_training_evidence(wrong_lock)
    receipt["overall_status"] = "failed"
    receipt_bytes = (json.dumps(receipt, sort_keys=True) + "\n").encode()
    receipt_path.write_bytes(receipt_bytes)
    normalized["receipt"]["sha256"] = hashlib.sha256(receipt_bytes).hexdigest()
    with pytest.raises(gate.GateValidationError, match="not completed"):
        gate.validate_training_evidence(normalized)


def test_git_firewall_requires_preexisting_exact_lock_ledger_and_config(
    tmp_path, monkeypatch, protocol
):
    original_root = gate.REPOSITORY_ROOT
    original_subprocess_run = subprocess.run
    protocol_bytes = (original_root / gate.PROTOCOL_RELATIVE_PATH).read_bytes()
    baseline_bytes = (original_root / gate.BASELINE_RELATIVE_PATH).read_bytes()
    baseline_rescore_bytes = (
        original_root / gate.BASELINE_RESCORE_RELATIVE_PATH
    ).read_bytes()
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Gate Test"],
        check=True,
    )
    sampler_path = tmp_path / "src/genmol/sampler.py"
    runner_path = tmp_path / "scripts/exps/denovo/benchmark.py"
    report_path = tmp_path / "scripts/exps/denovo/report.py"
    gate_path = tmp_path / "scripts/udlm/superiority_gate.py"
    config_path = tmp_path / "scripts/exps/denovo/hparams_udlm_schedule_uniform.yaml"
    sampler_path.parent.mkdir(parents=True)
    config_path.parent.mkdir(parents=True)
    protocol_path = tmp_path / gate.PROTOCOL_RELATIVE_PATH
    baseline_path = tmp_path / gate.BASELINE_RELATIVE_PATH
    baseline_rescore_path = tmp_path / gate.BASELINE_RESCORE_RELATIVE_PATH
    protocol_path.parent.mkdir(parents=True)
    baseline_path.parent.mkdir(parents=True)
    protocol_path.write_bytes(protocol_bytes)
    baseline_path.write_bytes(baseline_bytes)
    baseline_rescore_path.write_bytes(baseline_rescore_bytes)
    sampler_bytes = b"# synthetic audited EMA sampler\n"
    runner_bytes = b"# synthetic benchmark runner\n"
    report_bytes = b"# synthetic report implementation\n"
    gate_bytes = b"# synthetic superiority gate\n"
    config_bytes = b"diffusion_type: udlm\nnum_steps: 128\n"
    sampler_path.write_bytes(sampler_bytes)
    runner_path.write_bytes(runner_bytes)
    report_path.write_bytes(report_bytes)
    gate_path.parent.mkdir(parents=True)
    gate_path.write_bytes(gate_bytes)
    config_path.write_bytes(config_bytes)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-qm", "training source"],
        check=True,
    )
    training_revision = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    pilot_attempt, pilot_blobs = _completed_pilot_attempt(
        attempt_id="a1",
        candidate_id="schedule-uniform-synthetic",
        seeds=(1000, 1001),
        qualities=(0.85, 0.85),
        diversities=(0.74, 0.74),
    )
    for relative_path, artifact_bytes in pilot_blobs.items():
        pilot_artifact_path = tmp_path / relative_path
        pilot_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        pilot_artifact_path.write_bytes(artifact_bytes)
    ledger = _pilot_ledger(attempts=[pilot_attempt], selected_attempt_id="a1")
    ledger_bytes = _json_bytes(ledger)
    ledger_path = tmp_path / "experiments/udlm/candidates/ledger.json"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_bytes(ledger_bytes)
    candidate_lock = _candidate_lock()
    candidate_lock["training"]["source_revision"] = training_revision
    candidate_lock["selection"]["candidate_ledger"]["sha256"] = hashlib.sha256(
        ledger_bytes
    ).hexdigest()
    candidate_lock["inference"]["evaluation_config_sha256"] = hashlib.sha256(
        config_bytes
    ).hexdigest()
    candidate_lock["inference"]["sampler_source_sha256"] = hashlib.sha256(
        sampler_bytes
    ).hexdigest()
    candidate_lock["inference"]["benchmark_runner_sha256"] = hashlib.sha256(
        runner_bytes
    ).hexdigest()
    candidate_lock["analysis"]["gate_source_sha256"] = hashlib.sha256(
        gate_bytes
    ).hexdigest()
    candidate_lock["analysis"]["report_source_sha256"] = hashlib.sha256(
        report_bytes
    ).hexdigest()
    lock_path = tmp_path / "experiments/udlm/candidates/lock.json"
    lock_bytes = (json.dumps(candidate_lock, sort_keys=True) + "\n").encode()
    lock_path.write_bytes(lock_bytes)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-qm", "lock candidate"],
        check=True,
    )
    benchmark_revision = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    normalized = gate.validate_candidate_lock(candidate_lock, protocol)

    synthetic_git_blob = gate._git_blob

    def git_blob_with_real_rescore_source(revision, relative_path):
        if revision == gate.EXPECTED_BASELINE_RESCORE_SOURCE_REVISION:
            return original_subprocess_run(
                [
                    "git",
                    "-C",
                    str(original_root),
                    "show",
                    f"{revision}:{relative_path.as_posix()}",
                ],
                check=True,
                capture_output=True,
            ).stdout
        return synthetic_git_blob(revision, relative_path)

    monkeypatch.setattr(gate, "_git_blob", git_blob_with_real_rescore_source)

    def run_with_external_rescore_ancestry(args, *run_args, **run_kwargs):
        if (
            isinstance(args, list)
            and "merge-base" in args
            and args[-2] == gate.EXPECTED_BASELINE_RESCORE_SOURCE_REVISION
        ):
            return subprocess.CompletedProcess(args=args, returncode=0)
        return original_subprocess_run(args, *run_args, **run_kwargs)

    monkeypatch.setattr(gate.subprocess, "run", run_with_external_rescore_ancestry)

    evidence = gate.validate_git_lock_firewall(
        benchmark_revision=benchmark_revision,
        candidate_lock_path=Path("experiments/udlm/candidates/lock.json"),
        candidate_lock_bytes=lock_bytes,
        lock=normalized,
    )

    assert evidence["candidate_lock_exact_blob_at_benchmark_revision"] is True
    assert (
        evidence["baseline_rescore_attestation_exact_blob_at_benchmark_revision"]
        is True
    )
    assert evidence["baseline_rescore_source_exact_blobs_verified"] is True
    assert evidence["baseline_rescore_source_revision_is_ancestor"] is True
    assert evidence["candidate_ledger_exact_blob_at_benchmark_revision"] is True
    assert evidence["ema_sampler_source_exact_blob_at_benchmark_revision"] is True
    assert evidence["benchmark_runner_exact_blob_at_benchmark_revision"] is True
    assert evidence["committed_pilot_artifact_count"] == 2
    assert evidence["eligible_pilot_attempt_count"] == 1
    assert evidence["ineligible_completed_pilot_attempt_count"] == 0
    assert evidence["failed_pilot_attempt_count"] == 0
    assert evidence["registered_selection_operating_point"] == {
        "generation_seeds": [1000, 1001],
        "requested_samples_per_seed": 256,
        "nfe": 128,
        "metric_branch": "released_comparable",
    }
    assert evidence["selected_attempt_id"] == "a1"
    assert evidence["selection_recomputed_from_committed_pilot_evidence"] is True

    baseline_rescore_path.write_bytes(baseline_rescore_bytes + b" ")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", baseline_rescore_path.as_posix()],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-qm", "tamper rescore evidence"],
        check=True,
    )
    tampered_revision = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    with pytest.raises(
        gate.GateValidationError, match="rescore-attestation blob differs"
    ):
        gate.validate_git_lock_firewall(
            benchmark_revision=tampered_revision,
            candidate_lock_path=Path("experiments/udlm/candidates/lock.json"),
            candidate_lock_bytes=lock_bytes,
            lock=normalized,
        )

    with pytest.raises(gate.GateValidationError, match="candidate-lock blob differs"):
        gate.validate_git_lock_firewall(
            benchmark_revision=benchmark_revision,
            candidate_lock_path=Path("experiments/udlm/candidates/lock.json"),
            candidate_lock_bytes=lock_bytes + b" ",
            lock=normalized,
        )
    wrong_runner_lock = dict(normalized)
    wrong_runner_lock["benchmark_runner_sha256"] = "0" * 64
    with pytest.raises(gate.GateValidationError, match="runner source blob differs"):
        gate.validate_git_lock_firewall(
            benchmark_revision=benchmark_revision,
            candidate_lock_path=Path("experiments/udlm/candidates/lock.json"),
            candidate_lock_bytes=lock_bytes,
            lock=wrong_runner_lock,
        )


def test_no_clobber_decision_writer(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", tmp_path)
    output = tmp_path / "output" / "decision.json"
    output.parent.mkdir()

    gate._atomic_write_json_exclusive(output, {"status": "first"})
    original = output.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to replace"):
        gate._atomic_write_json_exclusive(output, {"status": "second"})

    assert output.read_bytes() == original


def test_decision_writer_does_not_create_through_symlinked_ancestor(
    tmp_path, monkeypatch
):
    repository = tmp_path / "repository"
    outside = tmp_path / "outside"
    repository.mkdir()
    outside.mkdir()
    (repository / "output").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(gate, "REPOSITORY_ROOT", repository)

    with pytest.raises(gate.GateValidationError, match="output parent"):
        gate._atomic_write_json_exclusive(
            repository / "output/new/decision.json", {"status": "forbidden"}
        )

    assert not (outside / "new").exists()

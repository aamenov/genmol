"""Validate and report the final three-seed GenMol de novo benchmark.

The report consumes only completed per-seed artifacts produced by
``scripts/exps/denovo/benchmark.py``.  Raw CSV rows are treated as the
authoritative count ledger: all validity, uniqueness, quality, repair, and
largest-component counts are recomputed before any aggregate is published.

This script is intentionally CPU-only. It never loads the model checkpoint or
imports RDKit, SAFE, or TDC. Categorical-prior validation imports Torch only to
reproduce the checkpoint constructor's exact float64 normalization and digest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

# Importing the schema constants does not import any GPU or chemistry package.
# Keeping one source of truth makes schema drift fail immediately.
from scripts.exps.denovo.benchmark import (  # noqa: E402
    METRIC_INPUT_SCHEMA_VERSION,
    RAW_SAMPLE_FIELDS,
    RAW_SAMPLES_FILENAME,
    SA_FINGERPRINT_SCORE_COUNT,
    SA_FRAGMENT_SCORE_ROW_COUNT,
    SA_FRAGMENT_SCORES_RELATIVE_PATH,
    SA_FRAGMENT_SCORES_SHA256,
    SA_FRAGMENT_SCORES_SIZE_BYTES,
    SCHEMA_VERSION as RUN_SCHEMA_VERSION,
    SUMMARY_FILENAME,
    TDC_METRIC_DISTRIBUTION_VERSION,
    TDC_METRIC_IMPLEMENTATION_SHA256,
    TDC_METRIC_IMPLEMENTATION_SIZE_BYTES,
    TDC_METRIC_IMPLEMENTATION_PATHS,
    UDLM_CATEGORICAL_PRIOR_VARIANTS,
    UDLM_PRIOR_VARIANT_IDENTITIES,
    UDLM_PRIOR_VARIANTS,
    benchmark_run_label,
    validate_udlm_prior_metadata_record,
    validate_sampling_config,
)


REPORT_SCHEMA_VERSION = 4
EXPECTED_SEEDS = (0, 1, 2)
EXPECTED_SAMPLES_PER_SEED = 1_000
EXPECTED_GLOBAL_STEP = 50_000
EXPECTED_CHECKPOINT_SHA256 = (
    "8d00aa47b02f64bf39ff6b0b2e786f213587366fc2c3d29712a00f3f84108dd6"
)
EXPECTED_CHECKPOINT_SIZE_BYTES = 1_396_998_679
PAPER_V1_SAMPLING_CONFIG = {
    "diffusion_type": "mdlm",
    "softmax_temp": 0.5,
    "randomness": 0.5,
    "min_add_len": 40,
    "num_steps": None,
    "inference_eps": None,
    "exclude_special_tokens": None,
    "prior_variant": None,
    "prior_metadata_sha256": None,
}
EXPECTED_GENERATION_PROTOCOL = {
    "diffusion_type": "mdlm",
    "nfe": 2,
    "nfe_definition": "one full backbone forward evaluation per reverse step",
    "num_steps": None,
    "num_steps_source": (
        "MDLM.get_num_steps_confidence on the single padded generation batch"
    ),
    "inference_eps": None,
    "exclude_special_tokens": None,
    "prior_variant": None,
    "prior_metadata_sha256": None,
    "temperature": 0.5,
    "randomness": 0.5,
    "randomness_used_by_sampler": True,
    "model_use_bracket_safe": False,
    "single_generation_batch": True,
    "released_safe_fix": True,
    "released_largest_component": "maximum SMILES string length",
    "strict_safe_fix": False,
}
AGGREGATE_JSON_FILENAME = "aggregate.json"
AGGREGATE_CSV_FILENAME = "aggregate.csv"

PAPER_PRIMARY_URL = (
    "https://raw.githubusercontent.com/mlresearch/v267/main/assets/lee25o/lee25o.pdf"
)
PAPER_REFERENCE = {
    "model": "GenMol V1",
    "paper": "GenMol: A Drug Discovery Generalist with Discrete Diffusion",
    "venue": "Proceedings of ICML 2025 (PMLR 267)",
    "table": "Table 1, de novo generation",
    "primary_url": PAPER_PRIMARY_URL,
    "protocol": {
        "runs": 3,
        "samples_per_run": 1_000,
        "seed_values": None,
        "seed_disclosure": (
            "The paper reports three runs but does not disclose their seed values."
        ),
    },
    # Fractions are used for rate metrics throughout the machine-readable files.
    "metrics": {
        "validity": {"mean": 1.000, "reported_sd": 0.000, "unit": "fraction"},
        "uniqueness": {"mean": 0.997, "reported_sd": 0.001, "unit": "fraction"},
        "quality": {"mean": 0.846, "reported_sd": 0.008, "unit": "fraction"},
        "diversity": {"mean": 0.818, "reported_sd": 0.001, "unit": "score"},
        "generation_time": {
            "mean": 21.1,
            "reported_sd": 0.4,
            "unit": "seconds",
        },
    },
    "dispersion_note": (
        "The plus/minus values are transcribed as published. This report does not "
        "infer an unreported seed list or a paired-run relationship."
    ),
}

TRAINING_CONTEXT = {
    "checkpoint": {
        "relative_path": "outputs/paper_v1/checkpoints/50000.ckpt",
        "sha256": EXPECTED_CHECKPOINT_SHA256,
        "size_bytes": EXPECTED_CHECKPOINT_SIZE_BYTES,
        "global_step": EXPECTED_GLOBAL_STEP,
        "ema": {
            "finite_shadow_tensors": 202,
            "num_updates": 50_000,
            "decay": 0.9999,
        },
    },
    "architecture": {
        "safe_version": "V1 (use_bracket_safe=false)",
        "vocabulary_size": 1_880,
        "bert_layers": 12,
        "hidden_size": 768,
        "maximum_sequence_length": 256,
        "training_precision": "bf16",
    },
    "training": {
        "local_global_batch_size": 2_046,
        "paper_global_batch_size": 2_048,
        "local_hardware": "3 NVIDIA RTX A6000 GPUs (physical IDs 5, 6, 7 at launch)",
        "paper_hardware": "8 NVIDIA A100 GPUs",
        "local_elapsed": "46:19:39",
        "paper_elapsed": "approximately 5 hours",
        "training_rng_seed": None,
        "training_rng_seed_status": (
            "Not recorded in the Hydra training configuration or retained training log."
        ),
    },
    "data_and_tokenizer": {
        "dataset": "datamol-io/safe-gpt",
        "tokenizer": "datamol-io/safe-gpt",
        "revision_status": (
            "Unpinned in the local training configuration; exact dataset and "
            "tokenizer revisions are unproven."
        ),
        "length_file": {
            "path": "data/len.pk",
            "sha256": (
                "9795cde72c60e9e58cd6dc99551801bdf47d00de6e060cbafb5f30584b19cf27"
            ),
            "count": 249_455,
            "minimum": 10,
            "median": 49,
            "maximum": 87,
        },
    },
}

METRIC_NAMES = ("validity", "uniqueness", "quality", "diversity")
METRIC_LABELS = {
    "validity": "Validity",
    "uniqueness": "Uniqueness",
    "quality": "Quality",
    "diversity": "Diversity",
    "generation_time": "Generation time",
}
BRANCH_PREFIX = {"released_comparable": "released", "strict": "strict"}
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
HEX_GIT_REVISION = re.compile(r"[0-9a-f]{40}\Z")


class ReportValidationError(ValueError):
    """Raised when benchmark artifacts do not support a valid comparison."""


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReportValidationError(f"{context} must be a JSON object")
    return value


def _required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ReportValidationError(f"{context} is missing required field {key!r}")
    return mapping[key]


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReportValidationError(f"{context} must be an integer")
    return value


def _finite_number(value: Any, context: str, *, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportValidationError(f"{context} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ReportValidationError(f"{context} must be finite")
    return number


def _utc_datetime(value: Any, context: str) -> datetime:
    """Parse an explicit UTC timestamp used in launch-safety provenance."""

    if not isinstance(value, str) or not value:
        raise ReportValidationError(f"{context} must be a non-empty UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ReportValidationError(f"{context} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ReportValidationError(f"{context} must include an explicit UTC offset")
    return parsed


def _validate_compute_processes(value: Any, context: str) -> list[dict[str, Any]]:
    """Validate an unambiguous per-GPU compute-process snapshot."""

    if not isinstance(value, list):
        raise ReportValidationError(f"{context} must be a list")
    validated = []
    seen_pids: set[int] = set()
    for index, process_value in enumerate(value):
        process_context = f"{context}[{index}]"
        process = _mapping(process_value, process_context)
        if set(process) != {"pid", "process_name", "used_memory_mib"}:
            raise ReportValidationError(
                f"{process_context} must contain pid, process_name, and used_memory_mib"
            )
        pid = _integer(process.get("pid"), f"{process_context}.pid")
        process_name = process.get("process_name")
        process_memory = _integer(
            process.get("used_memory_mib"), f"{process_context}.used_memory_mib"
        )
        if (
            pid <= 0
            or pid in seen_pids
            or not isinstance(process_name, str)
            or not process_name.strip()
        ):
            raise ReportValidationError(
                f"{process_context} has invalid or duplicate process identity"
            )
        if process_memory < 0:
            raise ReportValidationError(f"{process_context} has negative GPU memory")
        seen_pids.add(pid)
        validated.append(
            {
                "pid": pid,
                "process_name": process_name,
                "used_memory_mib": process_memory,
            }
        )
    return validated


def _sha256_value(value: Any, context: str) -> str:
    if not isinstance(value, str) or not HEX_SHA256.fullmatch(value):
        raise ReportValidationError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _git_revision_value(value: Any, context: str) -> str:
    if not isinstance(value, str) or not HEX_GIT_REVISION.fullmatch(value):
        raise ReportValidationError(
            f"{context} must be a 40-digit lowercase Git revision"
        )
    return value


def _csv_bool(value: str, context: str, *, optional: bool = False) -> bool | None:
    if optional and value == "":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    qualifier = "True, False, or empty" if optional else "True or False"
    raise ReportValidationError(f"{context} must be {qualifier}; got {value!r}")


def _csv_number(value: str, context: str, *, optional: bool = False) -> float | None:
    if optional and value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ReportValidationError(f"{context} must be numeric; got {value!r}") from exc
    if not math.isfinite(number):
        raise ReportValidationError(f"{context} must be finite")
    return number


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _assert_close(actual: Any, expected: float | None, context: str) -> None:
    if expected is None:
        if actual is not None:
            raise ReportValidationError(f"{context} must be null when its denominator is zero")
        return
    number = _finite_number(actual, context)
    if not math.isclose(number, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ReportValidationError(
            f"{context}={number!r} disagrees with the raw-row value {expected!r}"
        )


def _resolve_in_repository(path: Path) -> Path:
    resolved = path if path.is_absolute() else REPOSITORY_ROOT / path
    resolved = resolved.resolve()
    if resolved != REPOSITORY_ROOT and REPOSITORY_ROOT not in resolved.parents:
        raise ReportValidationError(f"path escapes repository scope: {resolved}")
    return resolved


def discover_run_directories(runs_dir: Path) -> dict[int, Path]:
    """Find exactly one complete artifact pair for each required seed."""
    runs_dir = _resolve_in_repository(runs_dir)
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"benchmark runs directory does not exist: {runs_dir}")

    summary_dirs = {path.parent for path in runs_dir.rglob(SUMMARY_FILENAME)}
    sample_dirs = {path.parent for path in runs_dir.rglob(RAW_SAMPLES_FILENAME)}
    candidate_dirs = summary_dirs | sample_dirs
    incomplete = sorted(
        str(path)
        for path in candidate_dirs
        if not (path / SUMMARY_FILENAME).is_file()
        or not (path / RAW_SAMPLES_FILENAME).is_file()
    )
    if incomplete:
        raise ReportValidationError(
            "every discovered run directory must contain both summary.json and "
            f"raw_samples.csv; incomplete: {', '.join(incomplete)}"
        )
    if len(candidate_dirs) != len(EXPECTED_SEEDS):
        raise ReportValidationError(
            f"expected exactly {len(EXPECTED_SEEDS)} completed run directories, "
            f"found {len(candidate_dirs)} under {runs_dir}"
        )

    by_seed: dict[int, Path] = {}
    for run_dir in sorted(candidate_dirs):
        try:
            summary = json.loads((run_dir / SUMMARY_FILENAME).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ReportValidationError(f"invalid JSON in {run_dir / SUMMARY_FILENAME}") from exc
        summary = _mapping(summary, str(run_dir / SUMMARY_FILENAME))
        run = _mapping(_required(summary, "run", str(run_dir)), f"{run_dir}: run")
        seed = _integer(_required(run, "seed", f"{run_dir}: run"), f"{run_dir}: run.seed")
        if seed in by_seed:
            raise ReportValidationError(f"duplicate benchmark summary for seed {seed}")
        by_seed[seed] = run_dir

    if tuple(sorted(by_seed)) != EXPECTED_SEEDS:
        raise ReportValidationError(
            f"required seeds are exactly {list(EXPECTED_SEEDS)}; found {sorted(by_seed)}"
        )
    return by_seed


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(RAW_SAMPLE_FIELDS):
            raise ReportValidationError(
                f"{path} has unexpected columns/order; expected {list(RAW_SAMPLE_FIELDS)}, "
                f"found {reader.fieldnames}"
            )
        records = list(reader)
    if len(records) != EXPECTED_SAMPLES_PER_SEED:
        raise ReportValidationError(
            f"{path} must contain exactly {EXPECTED_SAMPLES_PER_SEED} data rows; "
            f"found {len(records)}"
        )
    for expected_index, record in enumerate(records):
        try:
            sample_index = int(record["sample_index"])
        except ValueError as exc:
            raise ReportValidationError(
                f"{path}: row {expected_index + 2} has invalid sample_index"
            ) from exc
        if str(sample_index) != record["sample_index"] or sample_index != expected_index:
            raise ReportValidationError(
                f"{path}: sample_index must be the ordered range 0..999; "
                f"row {expected_index + 2} contains {record['sample_index']!r}"
            )
    return records


def _validate_branch_rows(
    records: Sequence[Mapping[str, str]],
    *,
    prefix: str,
    seed: int,
) -> dict[str, int]:
    context = f"seed {seed} {prefix}"
    seen: set[str] = set()
    valid_count = 0
    unique_count = 0
    quality_count = 0

    for index, record in enumerate(records):
        row_context = f"{context} row {index}"
        smiles = record[f"{prefix}_smiles"] or None
        decode_error = record[f"{prefix}_decode_error"] or None
        is_unique = _csv_bool(
            record[f"{prefix}_is_first_unique"],
            f"{row_context} is_first_unique",
        )
        quality_pass = _csv_bool(
            record[f"{prefix}_quality_pass"],
            f"{row_context} quality_pass",
            optional=True,
        )
        quality_counted = _csv_bool(
            record[f"{prefix}_quality_counted"],
            f"{row_context} quality_counted",
        )
        qed = _csv_number(
            record[f"{prefix}_qed"],
            f"{row_context} QED",
            optional=True,
        )
        sa = _csv_number(
            record[f"{prefix}_sa"],
            f"{row_context} SA",
            optional=True,
        )

        if smiles is None:
            if decode_error is None:
                raise ReportValidationError(f"{row_context} lacks both SMILES and decode error")
            if any(value is not None for value in (qed, sa, quality_pass)):
                raise ReportValidationError(f"{row_context} scores an invalid decoded sample")
            if is_unique or quality_counted:
                raise ReportValidationError(f"{row_context} counts an invalid decoded sample")
            continue

        valid_count += 1
        if decode_error is not None:
            raise ReportValidationError(f"{row_context} has both SMILES and a decode error")
        if qed is None or sa is None or quality_pass is None:
            raise ReportValidationError(f"{row_context} valid sample lacks QED/SA/quality")
        expected_pass = bool(qed >= 0.6 and sa <= 4.0)
        if quality_pass != expected_pass:
            raise ReportValidationError(
                f"{row_context} quality flag disagrees with QED>=0.6 and SA<=4"
            )
        expected_unique = smiles not in seen
        if is_unique != expected_unique:
            raise ReportValidationError(
                f"{row_context} first-unique flag disagrees with preceding rows"
            )
        if expected_unique:
            seen.add(smiles)
            unique_count += 1
        expected_counted = expected_unique and expected_pass
        if quality_counted != expected_counted:
            raise ReportValidationError(
                f"{row_context} quality_counted must equal first_unique AND quality_pass"
            )
        quality_count += int(expected_counted)

    return {
        "valid_count": valid_count,
        "unique_count": unique_count,
        "quality_count": quality_count,
    }


def _validate_cross_branch_rows(
    records: Sequence[Mapping[str, str]], *, seed: int
) -> dict[str, int]:
    counts = {
        "raw_safe_conversion_failed": 0,
        "released_recovered_strict_failure": 0,
        "strict_valid_but_released_failed": 0,
        "released_largest_component_applied": 0,
    }
    for index, record in enumerate(records):
        context = f"seed {seed} row {index}"
        raw_safe_error = record["raw_safe_error"] or None
        strict_valid = bool(record["strict_smiles"])
        released_valid = bool(record["released_smiles"])
        recovered = _csv_bool(
            record["released_was_recovered"], f"{context} released_was_recovered"
        )
        component_applied = _csv_bool(
            record["released_largest_component_applied"],
            f"{context} released_largest_component_applied",
        )
        repaired_smiles = record["released_repaired_smiles"] or None

        if raw_safe_error is not None:
            counts["raw_safe_conversion_failed"] += 1
            if strict_valid or released_valid:
                raise ReportValidationError(
                    f"{context} cannot decode after raw SAFE conversion failed"
                )
        expected_recovered = not strict_valid and released_valid
        if recovered != expected_recovered:
            raise ReportValidationError(
                f"{context} recovery flag disagrees with strict/released validity"
            )
        counts["released_recovered_strict_failure"] += int(expected_recovered)
        counts["strict_valid_but_released_failed"] += int(
            strict_valid and not released_valid
        )

        if released_valid:
            if repaired_smiles is None:
                raise ReportValidationError(
                    f"{context} released-valid row lacks pre-component repaired SMILES"
                )
            expected_smiles = sorted(repaired_smiles.split("."), key=len)[-1]
            if record["released_smiles"] != expected_smiles:
                raise ReportValidationError(
                    f"{context} released SMILES is not the released largest component"
                )
            expected_component = expected_smiles != repaired_smiles
        else:
            if repaired_smiles is not None:
                raise ReportValidationError(
                    f"{context} has repaired SMILES but no released final SMILES"
                )
            expected_component = False
        if component_applied != expected_component:
            raise ReportValidationError(
                f"{context} largest-component flag disagrees with pre/post strings"
            )
        counts["released_largest_component_applied"] += int(expected_component)
    return counts


def _validate_metric_branch(
    branch: Mapping[str, Any],
    derived: Mapping[str, int],
    *,
    seed: int,
    branch_name: str,
) -> None:
    context = f"seed {seed} metrics.{branch_name}"
    requested = EXPECTED_SAMPLES_PER_SEED
    valid_count = derived["valid_count"]
    unique_count = derived["unique_count"]
    quality_count = derived["quality_count"]
    expected_integers = {
        "valid_count": valid_count,
        "validity_denominator": requested,
        "unique_count": unique_count,
        "uniqueness_denominator": valid_count,
        "diversity_input_count": unique_count,
        "quality_count": quality_count,
        "quality_denominator": requested,
    }
    for key, expected in expected_integers.items():
        actual = _integer(_required(branch, key, context), f"{context}.{key}")
        if actual != expected:
            raise ReportValidationError(
                f"{context}.{key}={actual} disagrees with raw rows ({expected})"
            )
    _assert_close(branch.get("validity"), _ratio(valid_count, requested), f"{context}.validity")
    _assert_close(
        branch.get("uniqueness"),
        _ratio(unique_count, valid_count),
        f"{context}.uniqueness",
    )
    _assert_close(branch.get("quality"), _ratio(quality_count, requested), f"{context}.quality")

    diversity = _finite_number(
        branch.get("diversity"), f"{context}.diversity", allow_none=True
    )
    reason = branch.get("diversity_undefined_reason")
    if unique_count:
        if diversity is None or not 0.0 <= diversity <= 1.0:
            raise ReportValidationError(f"{context}.diversity must lie in [0, 1]")
        if reason is not None:
            raise ReportValidationError(
                f"{context}.diversity_undefined_reason must be null for a finite score"
            )
    elif diversity is not None or reason != "no_unique_valid_molecules":
        raise ReportValidationError(
            f"{context} must explain undefined diversity when no unique sample exists"
        )

    thresholds = _mapping(
        _required(branch, "quality_thresholds", context), f"{context}.quality_thresholds"
    )
    if thresholds != {"qed_min_inclusive": 0.6, "sa_max_inclusive": 4.0}:
        raise ReportValidationError(f"{context} has unexpected quality thresholds")
    if not isinstance(branch.get("definition"), str) or not branch["definition"].strip():
        raise ReportValidationError(f"{context}.definition must be non-empty")


def _command_option(command: Sequence[str], option: str, context: str) -> str:
    positions = [index for index, value in enumerate(command) if value == option]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ReportValidationError(f"{context} must contain exactly one {option} value")
    return command[positions[0] + 1]


def _validate_cuda_provenance(
    environment: Mapping[str, Any],
    run: Mapping[str, Any],
    *,
    seed: int,
    checkpoint_global_step: int,
    checkpoint_sha256: str,
    checkpoint_path: Any,
    config_path: Any,
    config_sha256: str,
    git_commit: Any,
    summary_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the launcher's point-in-time idle-GPU selection evidence."""
    context = f"{summary_path}: environment"
    if environment.get("requested_device") != "cuda:0":
        raise ReportValidationError(f"{context}.requested_device must be logical cuda:0")
    if environment.get("resolved_model_device") != "cuda:0":
        raise ReportValidationError(
            f"{context}.resolved_model_device must be logical cuda:0"
        )
    if environment.get("torch_cuda_available") is not True:
        raise ReportValidationError(f"{context} does not confirm CUDA availability")

    cuda_device = _mapping(environment.get("cuda_device"), f"{context}.cuda_device")
    if _integer(cuda_device.get("logical_index"), f"{context}.cuda_device.logical_index") != 0:
        raise ReportValidationError(f"{context}.cuda_device.logical_index must be zero")
    device_name = cuda_device.get("name")
    if not isinstance(device_name, str) or "RTX A6000" not in device_name:
        raise ReportValidationError(
            f"{context}.cuda_device.name must identify the expected RTX A6000 hardware"
        )
    total_memory = _integer(
        cuda_device.get("total_memory_bytes"),
        f"{context}.cuda_device.total_memory_bytes",
    )
    if total_memory <= 0:
        raise ReportValidationError(f"{context}.cuda_device total memory must be positive")
    compute_capability = cuda_device.get("compute_capability")
    if (
        not isinstance(compute_capability, list)
        or len(compute_capability) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in compute_capability
        )
    ):
        raise ReportValidationError(
            f"{context}.cuda_device.compute_capability must contain two integers"
        )

    versions = _mapping(environment.get("versions"), f"{context}.versions")
    expected_version_keys = {
        "torch",
        "lightning",
        "transformers",
        "numpy",
        "pandas",
        "pyyaml",
        "safe",
        "rdkit",
        "tdc",
        "bionemo_moco",
    }
    if set(versions) != expected_version_keys:
        raise ReportValidationError(
            f"{context}.versions must contain exactly {sorted(expected_version_keys)}"
        )
    if any(not isinstance(value, str) or not value for value in versions.values()):
        raise ReportValidationError(f"{context}.versions contains an unrecorded dependency")
    for name in ("python", "platform", "torch_cuda_version"):
        if not isinstance(environment.get(name), str) or not environment[name]:
            raise ReportValidationError(f"{context}.{name} must be recorded")
    cudnn_version = _integer(environment.get("cudnn_version"), f"{context}.cudnn_version")
    if cudnn_version <= 0:
        raise ReportValidationError(f"{context}.cudnn_version must be positive")

    launch = _mapping(
        environment.get("launch_environment"), f"{context}.launch_environment"
    )
    visible_uuid = launch.get("CUDA_VISIBLE_DEVICES")
    selected_uuid = launch.get("GENMOL_BENCHMARK_GPU_UUID")
    if (
        not isinstance(visible_uuid, str)
        or not visible_uuid.startswith("GPU-")
        or "," in visible_uuid
        or visible_uuid != selected_uuid
    ):
        raise ReportValidationError(
            f"{context} must map the launcher's selected GPU UUID to logical cuda:0"
        )
    expected_run_label = benchmark_run_label(
        checkpoint_global_step,
        checkpoint_sha256,
        seed,
    )
    if launch.get("GENMOL_BENCHMARK_RUN_LABEL") != expected_run_label:
        raise ReportValidationError(f"{context} has an unexpected benchmark run label")
    physical_index_text = launch.get("GENMOL_BENCHMARK_GPU_PHYSICAL_INDEX")
    if not isinstance(physical_index_text, str) or not physical_index_text.isdigit():
        raise ReportValidationError(f"{context} lacks a physical GPU index")
    physical_index = int(physical_index_text)

    snapshot_text = launch.get("GENMOL_BENCHMARK_GPU_SELECTION_SNAPSHOT")
    if not isinstance(snapshot_text, str) or not snapshot_text:
        raise ReportValidationError(f"{context} lacks the launch selection snapshot")
    try:
        snapshot_value = json.loads(snapshot_text)
    except json.JSONDecodeError as exc:
        raise ReportValidationError(f"{context} launch snapshot is not valid JSON") from exc
    snapshot = _mapping(snapshot_value, f"{context} launch snapshot")
    if snapshot.get("event") != "launch":
        raise ReportValidationError(f"{context} launch snapshot has the wrong event")
    selection_timestamp = _utc_datetime(
        snapshot.get("timestamp_utc"), f"{context} launch snapshot timestamp_utc"
    )
    selection_schema_version = snapshot.get("gpu_selection_schema_version")
    inventory_snapshot_completed_at_utc = None
    final_uuid_probe_completed_at_utc = None
    if selection_schema_version is None:
        raise ReportValidationError(
            f"{context} schema-{RUN_SCHEMA_VERSION} runs require the dynamic "
            "gpu_selection_schema_version 2 launch contract"
        )
    if (
        _integer(
            selection_schema_version,
            f"{context} launch snapshot gpu_selection_schema_version",
        )
        == 2
    ):
        expected_snapshot_fields = {
            "event",
            "gpu_selection_schema_version",
            "timestamp_utc",
            "inventory_snapshot_completed_at_utc",
            "final_uuid_probe_completed_at_utc",
            "source_revision",
            "gpu_inventory_at_selection",
            "running_physical_indices_at_selection",
            "physical_gpu_at_final_uuid_probe",
            "policy",
            "command",
        }
        if set(snapshot) != expected_snapshot_fields:
            raise ReportValidationError(
                f"{context} schema-v2 launch snapshot fields must be exactly "
                f"{sorted(expected_snapshot_fields)}"
            )
        physical_gpu_field = "physical_gpu_at_final_uuid_probe"
        selected_gpu_telemetry_stage = "final_exact_uuid_probe"
        inventory_snapshot_completed_at_utc = snapshot.get(
            "inventory_snapshot_completed_at_utc"
        )
        final_uuid_probe_completed_at_utc = snapshot.get(
            "final_uuid_probe_completed_at_utc"
        )
        inventory_timestamp = _utc_datetime(
            inventory_snapshot_completed_at_utc,
            f"{context} launch snapshot inventory_snapshot_completed_at_utc",
        )
        final_probe_timestamp = _utc_datetime(
            final_uuid_probe_completed_at_utc,
            f"{context} launch snapshot final_uuid_probe_completed_at_utc",
        )
        if final_probe_timestamp != selection_timestamp:
            raise ReportValidationError(
                f"{context} schema-v2 launch and final-probe timestamps disagree"
            )
        if final_probe_timestamp < inventory_timestamp:
            raise ReportValidationError(
                f"{context} final UUID probe predates the inventory snapshot"
            )
    else:
        raise ReportValidationError(
            f"{context} has unsupported gpu_selection_schema_version "
            f"{selection_schema_version!r}"
        )
    source_revision = _mapping(
        snapshot.get("source_revision"), f"{context} launch snapshot source_revision"
    )
    if set(source_revision) != {"head", "upstream"}:
        raise ReportValidationError(
            f"{context} launch snapshot lacks exact source revision provenance"
        )
    source_head = _git_revision_value(
        source_revision.get("head"), f"{context} launch snapshot source_revision.head"
    )
    source_upstream = _git_revision_value(
        source_revision.get("upstream"),
        f"{context} launch snapshot source_revision.upstream",
    )
    if source_head != source_upstream or source_head != git_commit:
        raise ReportValidationError(
            f"{context} benchmark source was not the recorded pushed commit"
        )
    physical_gpu = _mapping(
        snapshot.get(physical_gpu_field),
        f"{context} launch snapshot {physical_gpu_field}",
    )
    expected_physical_gpu_fields = {
        "index",
        "uuid",
        "name",
        "memory_used_mib",
        "memory_total_mib",
        "utilization_percent",
        "compute_mode",
        "compute_processes",
    }
    if set(physical_gpu) != expected_physical_gpu_fields:
        raise ReportValidationError(
            f"{context} {physical_gpu_field} must contain exact GPU telemetry"
        )
    if _integer(physical_gpu.get("index"), f"{context} physical_gpu.index") != physical_index:
        raise ReportValidationError(f"{context} physical GPU indices disagree")
    if physical_gpu.get("uuid") != visible_uuid:
        raise ReportValidationError(f"{context} physical GPU UUIDs disagree")
    if physical_gpu.get("name") != device_name:
        raise ReportValidationError(f"{context} physical and logical GPU names disagree")
    validated_processes = _validate_compute_processes(
        physical_gpu.get("compute_processes"),
        f"{context} {physical_gpu_field}.compute_processes",
    )
    memory_used = _integer(
        physical_gpu.get("memory_used_mib"), f"{context} physical_gpu.memory_used_mib"
    )
    memory_total = _integer(
        physical_gpu.get("memory_total_mib"), f"{context} physical_gpu.memory_total_mib"
    )
    utilization = _integer(
        physical_gpu.get("utilization_percent"),
        f"{context} physical_gpu.utilization_percent",
    )
    if (
        memory_used < 0
        or memory_total <= 0
        or memory_used > memory_total
        or not 0 <= utilization <= 100
    ):
        raise ReportValidationError(f"{context} contains impossible GPU utilization data")
    memory_free = memory_total - memory_used
    compute_mode = physical_gpu.get("compute_mode")
    if (
        not isinstance(compute_mode, str)
        or not compute_mode.strip()
        or compute_mode.lower() == "prohibited"
    ):
        raise ReportValidationError(f"{context} selected a prohibited compute-mode GPU")

    policy = _mapping(snapshot.get("policy"), f"{context} launch snapshot policy")
    common_policy_fields = {
        "max_utilization_percent",
        "utilization_comparison",
        "min_free_memory_mib",
        "active_compute_processes_allowed",
    }
    legacy_policy_fields = {
        *common_policy_fields,
        "selected_physical_indices",
    }
    dynamic_policy_fields = {
        *common_policy_fields,
        "selection_method",
        "inventory_scope",
        "requested_gpu_count",
    }
    policy_fields = set(policy)
    selected_inventory_item = None
    if policy_fields == legacy_policy_fields:
        if selection_schema_version is not None:
            raise ReportValidationError(
                f"{context} schema-v2 snapshot cannot use the legacy explicit policy"
            )
        selection_method = "explicit_physical_indices"
        inventory_scope = "user_selected_physical_indices"
        selected_indices = policy.get("selected_physical_indices")
        if (
            not isinstance(selected_indices, list)
            or not selected_indices
            or any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in selected_indices
            )
            or len(selected_indices) != len(set(selected_indices))
            or len(selected_indices) > 2
        ):
            raise ReportValidationError(
                f"{context} policy.selected_physical_indices must be unique physical IDs"
            )
        if physical_index not in selected_indices:
            raise ReportValidationError(
                f"{context} selected physical GPU was not explicitly chosen by the user"
            )
        requested_gpu_count = len(selected_indices)
    elif policy_fields == dynamic_policy_fields:
        selection_method = policy.get("selection_method")
        inventory_scope = policy.get("inventory_scope")
        if selection_method != "dynamic_idle_discovery":
            raise ReportValidationError(
                f"{context} dynamic launch policy has an invalid selection method"
            )
        if inventory_scope != "all_nvidia_gpus":
            raise ReportValidationError(
                f"{context} dynamic launch policy must inspect all NVIDIA GPUs"
            )
        if selection_schema_version is None:
            selected_gpu_telemetry_stage = "legacy_dynamic_final_uuid_probe"
        requested_gpu_count = _integer(
            policy.get("requested_gpu_count"),
            f"{context} policy.requested_gpu_count",
        )
        if requested_gpu_count not in (1, 2):
            raise ReportValidationError(
                f"{context} policy.requested_gpu_count must be 1 or 2"
            )
        selected_indices = None
        inventory_value = snapshot.get("gpu_inventory_at_selection")
        if not isinstance(inventory_value, list) or not inventory_value:
            raise ReportValidationError(
                f"{context} dynamic launch snapshot must record the full GPU inventory"
            )
        inventory_identities = []
        validated_inventory = []
        expected_inventory_fields = {
            "index",
            "uuid",
            "name",
            "memory_used_mib",
            "memory_total_mib",
            "utilization_percent",
            "compute_mode",
            "compute_processes",
        }
        for inventory_index, inventory_item in enumerate(inventory_value):
            item_context = (
                f"{context} gpu_inventory_at_selection[{inventory_index}]"
            )
            item = _mapping(inventory_item, item_context)
            if set(item) != expected_inventory_fields:
                raise ReportValidationError(
                    f"{item_context} must contain exact GPU telemetry and processes"
                )
            item_index = _integer(item.get("index"), f"{item_context}.index")
            item_uuid = item.get("uuid")
            item_name = item.get("name")
            item_memory_used = _integer(
                item.get("memory_used_mib"), f"{item_context}.memory_used_mib"
            )
            item_memory_total = _integer(
                item.get("memory_total_mib"), f"{item_context}.memory_total_mib"
            )
            item_utilization = _integer(
                item.get("utilization_percent"),
                f"{item_context}.utilization_percent",
            )
            item_compute_mode = item.get("compute_mode")
            item_processes = item.get("compute_processes")
            if (
                item_index < 0
                or not isinstance(item_uuid, str)
                or not item_uuid.startswith("GPU-")
                or not isinstance(item_name, str)
                or not item_name
                or item_memory_used < 0
                or item_memory_total <= 0
                or item_memory_used > item_memory_total
                or not 0 <= item_utilization <= 100
                or not isinstance(item_compute_mode, str)
                or not item_compute_mode.strip()
                or not isinstance(item_processes, list)
            ):
                raise ReportValidationError(
                    f"{item_context} contains invalid GPU inventory telemetry"
                )
            _validate_compute_processes(
                item_processes, f"{item_context}.compute_processes"
            )
            identity = (item_index, item_uuid)
            inventory_identities.append(identity)
            validated_inventory.append(dict(item))
        if (
            len({index for index, _ in inventory_identities})
            != len(inventory_identities)
            or len({uuid for _, uuid in inventory_identities})
            != len(inventory_identities)
        ):
            raise ReportValidationError(
                f"{context} dynamic GPU inventory contains duplicate identities"
            )
        if [index for index, _ in inventory_identities] != sorted(
            index for index, _ in inventory_identities
        ):
            raise ReportValidationError(
                f"{context} dynamic GPU inventory must be ordered by physical index"
            )
        if (physical_index, visible_uuid) not in inventory_identities:
            raise ReportValidationError(
                f"{context} dynamically selected GPU is absent from the full inventory"
            )
        selected_inventory_item = next(
            item
            for item in validated_inventory
            if item["index"] == physical_index and item["uuid"] == visible_uuid
        )
        running_indices = snapshot.get("running_physical_indices_at_selection")
        if (
            not isinstance(running_indices, list)
            or any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in running_indices
            )
            or len(running_indices) != len(set(running_indices))
            or any(
                index not in {item_index for item_index, _ in inventory_identities}
                for index in running_indices
            )
            or physical_index in running_indices
        ):
            raise ReportValidationError(
                f"{context} dynamic launch snapshot has invalid running GPU indices"
            )
    else:
        raise ReportValidationError(
            f"{context} launch policy fields must match either the legacy explicit "
            f"schema {sorted(legacy_policy_fields)} or dynamic schema "
            f"{sorted(dynamic_policy_fields)}"
        )
    max_utilization = _integer(
        policy.get("max_utilization_percent"),
        f"{context} policy.max_utilization_percent",
    )
    min_free_memory = _integer(
        policy.get("min_free_memory_mib"),
        f"{context} policy.min_free_memory_mib",
    )
    if not 0 < max_utilization <= 10 or min_free_memory < 30_000:
        raise ReportValidationError(f"{context} launch policy is invalid")
    if policy.get("utilization_comparison") != "strictly_less_than":
        raise ReportValidationError(
            f"{context} launch policy must use an exclusive utilization threshold"
        )
    if policy.get("active_compute_processes_allowed") is not False:
        raise ReportValidationError(
            f"{context} launch policy must forbid active compute processes"
        )
    if selected_inventory_item is not None:
        initial_processes = _validate_compute_processes(
            selected_inventory_item["compute_processes"],
            f"{context} selected inventory GPU compute_processes",
        )
        initial_free_memory = (
            selected_inventory_item["memory_total_mib"]
            - selected_inventory_item["memory_used_mib"]
        )
        if (
            initial_processes
            or selected_inventory_item["utilization_percent"] >= max_utilization
            or initial_free_memory < min_free_memory
            or selected_inventory_item["compute_mode"].lower() == "prohibited"
        ):
            raise ReportValidationError(
                f"{context} dynamically selected GPU was not idle in the full inventory"
            )
    if validated_processes:
        raise ReportValidationError(
            f"{context} selected a GPU with active compute processes"
        )
    if utilization >= max_utilization:
        raise ReportValidationError(
            f"{context} selected GPU does not satisfy the strict utilization threshold"
        )
    if memory_free < min_free_memory:
        raise ReportValidationError(
            f"{context} selected GPU does not satisfy the free-memory threshold"
        )

    command = run.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(value, str) for value in command)
    ):
        raise ReportValidationError(f"{summary_path}: run.command must be a string list")
    if snapshot.get("command") != command:
        raise ReportValidationError(
            f"{context} snapshot command disagrees with the executed run command"
        )
    expected_command_values = {
        "--seed": str(seed),
        "--num-samples": str(EXPECTED_SAMPLES_PER_SEED),
        "--device": "cuda:0",
    }
    for option, expected in expected_command_values.items():
        if _command_option(command, option, f"{summary_path}: run.command") != expected:
            raise ReportValidationError(
                f"{summary_path}: run.command {option} must equal {expected}"
            )
    pinned_digest = _command_option(
        command,
        "--expected-checkpoint-sha256",
        f"{summary_path}: run.command",
    )
    if pinned_digest != checkpoint_sha256:
        raise ReportValidationError(
            f"{summary_path}: run.command --expected-checkpoint-sha256 must "
            "equal the loaded checkpoint digest"
        )
    pinned_source_revision = _command_option(
        command,
        "--expected-source-revision",
        f"{summary_path}: run.command",
    )
    if pinned_source_revision != source_head:
        raise ReportValidationError(
            f"{summary_path}: run.command --expected-source-revision must "
            "equal the launcher's pushed source revision"
        )
    pinned_config_digest = _command_option(
        command,
        "--expected-config-sha256",
        f"{summary_path}: run.command",
    )
    if pinned_config_digest != config_sha256:
        raise ReportValidationError(
            f"{summary_path}: run.command --expected-config-sha256 must "
            "equal the loaded inference-config digest"
        )
    expected_paths = {
        "--checkpoint": checkpoint_path,
        "--config": config_path,
        "--output-dir": summary_path.parent,
    }
    for option, expected_path in expected_paths.items():
        raw_path = _command_option(
            command,
            option,
            f"{summary_path}: run.command",
        )
        if not isinstance(expected_path, str | Path) or not str(expected_path):
            raise ReportValidationError(
                f"{summary_path}: recorded path for {option} is invalid"
            )
        if Path(raw_path).resolve() != Path(expected_path).resolve():
            raise ReportValidationError(
                f"{summary_path}: run.command {option} disagrees with its "
                "recorded artifact path"
            )
    if len(command) != 20:
        raise ReportValidationError(
            f"{summary_path}: run.command must contain only the nine bound options"
        )
    working_directory = environment.get("working_directory")
    executable = environment.get("executable")
    if not isinstance(working_directory, str) or not working_directory:
        raise ReportValidationError(f"{context}.working_directory must be recorded")
    if Path(working_directory).resolve() != REPOSITORY_ROOT:
        raise ReportValidationError(
            f"{context}.working_directory must be the benchmark repository root"
        )
    if not isinstance(executable, str) or not executable:
        raise ReportValidationError(f"{context}.executable must be recorded")

    def command_path(value: str) -> Path:
        path = Path(value)
        return (Path(working_directory) / path).resolve() if not path.is_absolute() else path.resolve()

    if command_path(command[0]) != command_path(executable):
        raise ReportValidationError(
            f"{summary_path}: run.command interpreter disagrees with environment"
        )
    project_root = (
        REPOSITORY_ROOT.parent.parent
        if REPOSITORY_ROOT.parent.name == "run_sources"
        else REPOSITORY_ROOT
    )
    if command_path(command[0]) != (project_root / ".venv/bin/python").resolve():
        raise ReportValidationError(
            f"{summary_path}: run.command did not use the project .venv Python"
        )
    if command_path(command[1]) != (
        REPOSITORY_ROOT / "scripts/exps/denovo/benchmark.py"
    ).resolve():
        raise ReportValidationError(
            f"{summary_path}: run.command does not execute the pinned benchmark script"
        )

    environment_signature = {
        "python": environment["python"],
        "platform": environment["platform"],
        "versions": dict(versions),
        "torch_cuda_version": environment["torch_cuda_version"],
        "cudnn_version": cudnn_version,
        "gpu_name": device_name,
        "gpu_total_memory_bytes": total_memory,
        "gpu_compute_capability": list(compute_capability),
    }
    launch_provenance = {
        "physical_index": physical_index,
        "uuid": visible_uuid,
        "name": device_name,
        "selection_timestamp_utc": snapshot.get("timestamp_utc"),
        "gpu_selection_schema_version": selection_schema_version,
        "selected_gpu_telemetry_stage": selected_gpu_telemetry_stage,
        "inventory_snapshot_completed_at_utc": inventory_snapshot_completed_at_utc,
        "final_uuid_probe_completed_at_utc": final_uuid_probe_completed_at_utc,
        "memory_used_mib_at_selection": memory_used,
        "memory_total_mib_at_selection": memory_total,
        "memory_free_mib_at_selection": memory_free,
        "utilization_percent_at_selection": utilization,
        "compute_process_count_at_selection": len(validated_processes),
        "compute_processes_at_selection": validated_processes,
        "selection_method": selection_method,
        "inventory_scope": inventory_scope,
        "user_requested_gpu_count": requested_gpu_count,
        "user_selected_physical_indices": (
            None if selected_indices is None else list(selected_indices)
        ),
        "gpu_inventory_at_selection": (
            validated_inventory
            if selection_method == "dynamic_idle_discovery"
            else None
        ),
        "running_physical_indices_at_selection": (
            list(running_indices)
            if selection_method == "dynamic_idle_discovery"
            else None
        ),
        "policy": dict(policy),
        "source_revision": dict(source_revision),
        "visibility": (
            "one dynamically selected idle GPU UUID from the full NVIDIA inventory "
            "mapped to logical cuda:0; no active compute process was present at the "
            "final UUID probe"
            if selection_method == "dynamic_idle_discovery"
            else "one user-selected physical GPU UUID mapped to logical cuda:0; "
            "no active compute process was present at the final launch probe"
        ),
    }
    return environment_signature, launch_provenance


def _validate_tokenizer_provenance(
    value: Any, *, summary_path: Path
) -> dict[str, Any]:
    context = f"{summary_path}: tokenizer"
    tokenizer = dict(_mapping(value, context))
    expected_fields = {
        "requested_identifier",
        "class",
        "name_or_path",
        "declared_revision",
        "resolved_commit_hash",
        "base_vocab_size",
        "effective_size",
        "vocabulary_sha256",
        "added_vocabulary_sha256",
        "backend_json_sha256",
        "backend_serialization_error",
        "special_token_ids",
    }
    if set(tokenizer) != expected_fields:
        raise ReportValidationError(
            f"{context} fields must be exactly {sorted(expected_fields)}"
        )
    if not isinstance(tokenizer["class"], str) or not tokenizer["class"]:
        raise ReportValidationError(f"{context}.class must be recorded")
    if tokenizer["requested_identifier"] != "datamol-io/safe-gpt":
        raise ReportValidationError(
            f"{context}.requested_identifier must be the V1 "
            "datamol-io/safe-gpt tokenizer"
        )
    name_or_path = tokenizer["name_or_path"]
    if name_or_path is not None and (
        not isinstance(name_or_path, str) or not name_or_path
    ):
        raise ReportValidationError(f"{context}.name_or_path must be null or a string")
    for name in ("declared_revision", "resolved_commit_hash"):
        revision = tokenizer[name]
        if revision is not None and (not isinstance(revision, str) or not revision):
            raise ReportValidationError(f"{context}.{name} must be null or a string")
    expected_sizes = {"base_vocab_size": 1_880, "effective_size": 1_882}
    for name, expected_size in expected_sizes.items():
        if _integer(tokenizer[name], f"{context}.{name}") != expected_size:
            raise ReportValidationError(
                f"{context}.{name} must equal {expected_size} for this V1 runtime"
            )
    for name in ("vocabulary_sha256", "added_vocabulary_sha256"):
        _sha256_value(tokenizer[name], f"{context}.{name}")
    backend_hash = tokenizer["backend_json_sha256"]
    if backend_hash is not None:
        _sha256_value(backend_hash, f"{context}.backend_json_sha256")
    backend_error = tokenizer["backend_serialization_error"]
    if backend_error is not None and (
        not isinstance(backend_error, str) or not backend_error
    ):
        raise ReportValidationError(
            f"{context}.backend_serialization_error must be null or a string"
        )
    special_ids = _mapping(tokenizer["special_token_ids"], f"{context}.special_token_ids")
    expected_special_ids = {"pad": 3, "bos": 1, "eos": 2, "mask": 4}
    if dict(special_ids) != expected_special_ids:
        raise ReportValidationError(
            f"{context}.special_token_ids must equal {expected_special_ids}"
        )
    return tokenizer


def _validate_metric_inputs(value: Any, *, summary_path: Path) -> dict[str, Any]:
    """Validate the immutable TDC/SA inputs behind the quality metric."""

    context = f"{summary_path}: metric_inputs"
    metric_inputs = dict(_mapping(value, context))
    expected_top_level = {
        "schema_version",
        "sa_fragment_scores",
        "tdc_metric_implementation",
        "sa_loading_policy",
        "affected_outputs",
    }
    if set(metric_inputs) != expected_top_level:
        raise ReportValidationError(
            f"{context} fields must be exactly {sorted(expected_top_level)}"
        )
    if metric_inputs["schema_version"] != METRIC_INPUT_SCHEMA_VERSION:
        raise ReportValidationError(
            f"{context}.schema_version must equal {METRIC_INPUT_SCHEMA_VERSION}"
        )

    fragment_scores = _mapping(
        metric_inputs["sa_fragment_scores"], f"{context}.sa_fragment_scores"
    )
    expected_fragment_fields = {
        "path",
        "relative_path",
        "sha256",
        "size_bytes",
        "serialization",
        "top_level_row_count",
        "fingerprint_score_count",
        "duplicate_fingerprint_count",
    }
    if set(fragment_scores) != expected_fragment_fields:
        raise ReportValidationError(
            f"{context}.sa_fragment_scores fields must be exactly "
            f"{sorted(expected_fragment_fields)}"
        )
    expected_fragment_values = {
        "relative_path": SA_FRAGMENT_SCORES_RELATIVE_PATH.as_posix(),
        "sha256": SA_FRAGMENT_SCORES_SHA256,
        "size_bytes": SA_FRAGMENT_SCORES_SIZE_BYTES,
        "serialization": "python_pickle_verified_before_deserialization",
        "top_level_row_count": SA_FRAGMENT_SCORE_ROW_COUNT,
        "fingerprint_score_count": SA_FINGERPRINT_SCORE_COUNT,
        "duplicate_fingerprint_count": 0,
    }
    for key, expected in expected_fragment_values.items():
        if fragment_scores.get(key) != expected:
            raise ReportValidationError(
                f"{context}.sa_fragment_scores.{key}={fragment_scores.get(key)!r}; "
                f"expected pinned value {expected!r}"
            )
    recorded_fragment_path = fragment_scores.get("path")
    if not isinstance(recorded_fragment_path, str) or not Path(
        recorded_fragment_path
    ).is_absolute():
        raise ReportValidationError(
            f"{context}.sa_fragment_scores.path must be absolute"
        )
    expected_fragment_path = (
        REPOSITORY_ROOT / SA_FRAGMENT_SCORES_RELATIVE_PATH
    ).resolve()
    if Path(recorded_fragment_path).resolve() != expected_fragment_path:
        raise ReportValidationError(
            f"{context}.sa_fragment_scores.path is not the pinned repository path"
        )

    tdc = _mapping(
        metric_inputs["tdc_metric_implementation"],
        f"{context}.tdc_metric_implementation",
    )
    if set(tdc) != {"distribution", "version", "implementation_files"}:
        raise ReportValidationError(
            f"{context}.tdc_metric_implementation has unexpected fields"
        )
    if (
        tdc.get("distribution") != "PyTDC"
        or tdc.get("version") != TDC_METRIC_DISTRIBUTION_VERSION
    ):
        raise ReportValidationError(
            f"{context} must record the audited PyTDC "
            f"{TDC_METRIC_DISTRIBUTION_VERSION} metric backend"
        )
    implementation_files = _mapping(
        tdc.get("implementation_files"),
        f"{context}.tdc_metric_implementation.implementation_files",
    )
    if set(implementation_files) != set(TDC_METRIC_IMPLEMENTATION_PATHS):
        raise ReportValidationError(
            f"{context} has incomplete TDC metric implementation fingerprints"
        )
    for name, relative_path in TDC_METRIC_IMPLEMENTATION_PATHS.items():
        source = _mapping(
            implementation_files[name],
            f"{context}.tdc_metric_implementation.implementation_files.{name}",
        )
        if set(source) != {"path", "sha256", "size_bytes"}:
            raise ReportValidationError(
                f"{context} TDC source {name} has unexpected fields"
            )
        source_path = source.get("path")
        if (
            not isinstance(source_path, str)
            or not Path(source_path).is_absolute()
            or not Path(source_path).as_posix().endswith(relative_path.as_posix())
        ):
            raise ReportValidationError(
                f"{context} TDC source {name} has an invalid path"
            )
        digest = _sha256_value(
            source.get("sha256"), f"{context} TDC source {name} SHA-256"
        )
        size_bytes = _integer(
            source.get("size_bytes"), f"{context} TDC source {name} size"
        )
        if digest != TDC_METRIC_IMPLEMENTATION_SHA256[name]:
            raise ReportValidationError(
                f"{context} TDC source {name} does not match the audited SHA-256"
            )
        if size_bytes != TDC_METRIC_IMPLEMENTATION_SIZE_BYTES[name]:
            raise ReportValidationError(
                f"{context} TDC source {name} does not match the audited size"
            )

    expected_loading_policy = {
        "requested_oracle": "sa",
        "oracle_class": "tdc.oracles.Oracle",
        "sa_callable": "tdc.chem_utils.oracle.oracle.SA",
        "network_download_allowed": False,
        "tdc_oracle_load_invoked": False,
        "resident_scores_loaded_from_verified_bytes": True,
        "artifact_mutation_after_resident_load_affects_current_run": False,
    }
    if metric_inputs["sa_loading_policy"] != expected_loading_policy:
        raise ReportValidationError(
            f"{context}.sa_loading_policy is not the fail-closed pinned policy"
        )
    expected_affected_outputs = [
        "raw_samples_csv.strict_sa",
        "raw_samples_csv.released_sa",
        "metrics.strict.quality",
        "metrics.released_comparable.quality",
    ]
    if metric_inputs["affected_outputs"] != expected_affected_outputs:
        raise ReportValidationError(
            f"{context}.affected_outputs does not match the SA-dependent fields"
        )
    return metric_inputs


def _validate_generation_protocol(
    protocol: Mapping[str, Any],
    sampling: Mapping[str, Any],
    *,
    context: str,
) -> None:
    expected_keys = set(EXPECTED_GENERATION_PROTOCOL)
    if set(protocol) != expected_keys:
        raise ReportValidationError(
            f"{context} fields must be exactly {sorted(expected_keys)}"
        )
    diffusion_type = sampling["diffusion_type"]
    if protocol.get("diffusion_type") != diffusion_type:
        raise ReportValidationError(f"{context}.diffusion_type disagrees with config")
    nfe = _integer(protocol.get("nfe"), f"{context}.nfe")
    if nfe <= 0:
        raise ReportValidationError(f"{context}.nfe must be positive")
    if protocol.get("nfe_definition") != (
        "one full backbone forward evaluation per reverse step"
    ):
        raise ReportValidationError(f"{context}.nfe_definition is unexpected")
    if protocol.get("temperature") != sampling["softmax_temp"]:
        raise ReportValidationError(f"{context}.temperature disagrees with config")
    if protocol.get("randomness") != sampling["randomness"]:
        raise ReportValidationError(f"{context}.randomness disagrees with config")
    if protocol.get("prior_variant") != sampling["prior_variant"]:
        raise ReportValidationError(f"{context}.prior_variant disagrees with config")
    if protocol.get("prior_metadata_sha256") != sampling["prior_metadata_sha256"]:
        raise ReportValidationError(
            f"{context}.prior_metadata_sha256 disagrees with config"
        )

    if diffusion_type == "udlm":
        if protocol.get("num_steps") != sampling["num_steps"] or nfe != sampling["num_steps"]:
            raise ReportValidationError(
                f"{context} UDLM NFE must equal the explicit num_steps"
            )
        if protocol.get("inference_eps") != sampling["inference_eps"]:
            raise ReportValidationError(f"{context}.inference_eps disagrees with config")
        if protocol.get("exclude_special_tokens") is not sampling["exclude_special_tokens"]:
            raise ReportValidationError(
                f"{context}.exclude_special_tokens disagrees with config"
            )
        if protocol.get("num_steps_source") != (
            "explicit UDLM reverse-transition count"
        ):
            raise ReportValidationError(f"{context}.num_steps_source is unexpected")
        if protocol.get("randomness_used_by_sampler") is not False:
            raise ReportValidationError(
                f"{context} must record that UDLM ignores randomness"
            )
    else:
        if (
            protocol.get("num_steps") is not None
            or protocol.get("inference_eps") is not None
            or protocol.get("exclude_special_tokens") is not None
            or protocol.get("prior_variant") is not None
            or protocol.get("prior_metadata_sha256") is not None
        ):
            raise ReportValidationError(
                f"{context} MDLM UDLM-only settings must be null"
            )
        if protocol.get("num_steps_source") != (
            "MDLM.get_num_steps_confidence on the single padded generation batch"
        ):
            raise ReportValidationError(f"{context}.num_steps_source is unexpected")
        if protocol.get("randomness_used_by_sampler") is not True:
            raise ReportValidationError(f"{context} must record MDLM randomness use")

    common_expected = {
        "model_use_bracket_safe": False,
        "single_generation_batch": True,
        "released_safe_fix": True,
        "released_largest_component": "maximum SMILES string length",
        "strict_safe_fix": False,
    }
    for key, expected in common_expected.items():
        if protocol.get(key) != expected:
            raise ReportValidationError(f"{context}.{key}={protocol.get(key)!r}; expected {expected!r}")


def _validate_summary_and_rows(
    run_dir: Path,
    expected_seed: int,
) -> dict[str, Any]:
    summary_path = run_dir / SUMMARY_FILENAME
    samples_path = run_dir / RAW_SAMPLES_FILENAME
    try:
        summary_value = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReportValidationError(f"invalid JSON: {summary_path}") from exc
    summary = dict(_mapping(summary_value, str(summary_path)))
    required_top_level = {
        "schema_version",
        "status",
        "seed",
        "num_samples",
        "run",
        "checkpoint",
        "config",
        "metrics",
        "failure_counts",
        "runtime_seconds",
        "environment",
        "git",
        "implementation_inputs",
        "metric_inputs",
        "tokenizer",
        "artifacts",
    }
    missing = required_top_level - summary.keys()
    if missing:
        raise ReportValidationError(
            f"{summary_path} is missing top-level field(s): {sorted(missing)}"
        )
    if summary["schema_version"] != RUN_SCHEMA_VERSION:
        raise ReportValidationError(
            f"{summary_path} schema_version={summary['schema_version']!r}; "
            f"expected {RUN_SCHEMA_VERSION}"
        )
    if summary["status"] != "completed":
        raise ReportValidationError(
            f"{summary_path} is not a completion marker (status={summary['status']!r})"
        )

    run = _mapping(summary["run"], f"{summary_path}: run")
    nested_seed = _integer(_required(run, "seed", "run"), "run.seed")
    alias_seed = _integer(summary["seed"], "seed")
    nested_count = _integer(
        _required(run, "requested_sample_count", "run"), "run.requested_sample_count"
    )
    alias_count = _integer(summary["num_samples"], "num_samples")
    if nested_seed != alias_seed or nested_seed != expected_seed:
        raise ReportValidationError(
            f"{summary_path} seed aliases/directory disagree: "
            f"run={nested_seed}, top-level={alias_seed}, expected={expected_seed}"
        )
    if nested_count != alias_count or nested_count != EXPECTED_SAMPLES_PER_SEED:
        raise ReportValidationError(
            f"{summary_path} must describe exactly {EXPECTED_SAMPLES_PER_SEED} samples"
        )
    seed_config = _mapping(
        _required(run, "seed_configuration", "run"), "run.seed_configuration"
    )
    expected_seed_provenance = {
        "seed": expected_seed,
        "seed_applied_immediately_before_generation": True,
        "python_random": True,
        "numpy": True,
        "torch_cpu": True,
        "torch_cuda_all": True,
        "python_hash_seed": str(expected_seed),
    }
    for key, expected in expected_seed_provenance.items():
        if seed_config.get(key) != expected:
            raise ReportValidationError(
                f"{summary_path} seed_configuration.{key}={seed_config.get(key)!r}; "
                f"expected {expected!r}"
            )
    if run.get("one_seed_per_invocation") is not True:
        raise ReportValidationError(
            f"{summary_path} did not use exactly one seed per invocation"
        )
    if run.get("single_generation_batch") is not True:
        raise ReportValidationError(f"{summary_path} did not use one released-style batch")
    if run.get("evaluation_tier") != "final" or run.get("final_protocol_eligible") is not True:
        raise ReportValidationError(
            f"{summary_path} is a pilot artifact and cannot enter the final report"
        )
    started_at = _utc_datetime(
        run.get("started_at_utc"), f"{summary_path}: run.started_at_utc"
    )
    completed_at = _utc_datetime(
        run.get("completed_at_utc"), f"{summary_path}: run.completed_at_utc"
    )
    if completed_at < started_at:
        raise ReportValidationError(
            f"{summary_path} run.completed_at_utc predates run.started_at_utc"
        )
    generation_protocol = _mapping(
        run.get("generation_protocol"), "run.generation_protocol"
    )

    checkpoint = _mapping(summary["checkpoint"], f"{summary_path}: checkpoint")
    checkpoint_sha = _sha256_value(checkpoint.get("sha256"), "checkpoint.sha256")
    global_step = _integer(checkpoint.get("global_step"), "checkpoint.global_step")
    size_bytes = _integer(checkpoint.get("size_bytes"), "checkpoint.size_bytes")
    if global_step < 0 or size_bytes <= 0:
        raise ReportValidationError(
            f"seed {expected_seed} has invalid checkpoint step or size metadata"
        )
    if checkpoint.get("byte_identity_verified_before_and_after_load") is not True:
        raise ReportValidationError(
            f"seed {expected_seed} did not verify stable checkpoint bytes around loading"
        )
    checkpoint_diffusion_type = str(checkpoint.get("diffusion_type", "")).lower()
    if checkpoint_diffusion_type not in {"mdlm", "udlm"}:
        raise ReportValidationError(
            f"seed {expected_seed} has invalid checkpoint diffusion_type"
        )
    checkpoint_udlm_inference_eps = checkpoint.get("udlm_inference_eps")
    checkpoint_udlm_exclude_special_tokens = checkpoint.get(
        "udlm_exclude_special_tokens"
    )
    checkpoint_prior_variant = checkpoint.get("udlm_prior_variant")
    checkpoint_prior_metadata = checkpoint.get("udlm_prior_metadata")
    checkpoint_prior_digest = checkpoint.get("udlm_prior_metadata_sha256")
    if checkpoint_diffusion_type == "mdlm":
        if (
            checkpoint_sha != EXPECTED_CHECKPOINT_SHA256
            or global_step != EXPECTED_GLOBAL_STEP
            or size_bytes != EXPECTED_CHECKPOINT_SIZE_BYTES
        ):
            raise ReportValidationError(
                "MDLM reports are restricted to the audited 50k baseline checkpoint; "
                "its checkpoint hash, step, and byte size must all match"
            )
        if any(
            value is not None
            for value in (
                checkpoint_udlm_inference_eps,
                checkpoint_udlm_exclude_special_tokens,
                checkpoint_prior_variant,
                checkpoint_prior_metadata,
                checkpoint_prior_digest,
            )
        ):
            raise ReportValidationError(
                "MDLM checkpoint must record null UDLM endpoint and prior fields"
            )
    else:
        if checkpoint_prior_variant not in UDLM_PRIOR_VARIANTS:
            raise ReportValidationError(
                "UDLM checkpoint has an invalid udlm_prior_variant"
            )
        if checkpoint_prior_variant == "release_uniform":
            if checkpoint_prior_metadata is not None or checkpoint_prior_digest is not None:
                raise ReportValidationError(
                    "release_uniform checkpoint must not declare categorical prior metadata"
                )
        else:
            try:
                checkpoint_prior_digest = _sha256_value(
                    checkpoint_prior_digest,
                    "checkpoint.udlm_prior_metadata_sha256",
                )
                validated_prior = validate_udlm_prior_metadata_record(
                    checkpoint_prior_metadata,
                    expected_variant=checkpoint_prior_variant,
                )
            except (RuntimeError, ValueError) as exc:
                raise ReportValidationError(
                    f"categorical checkpoint prior metadata is invalid: {exc}"
                ) from exc
            if _sha256_json(validated_prior) != checkpoint_prior_digest:
                raise ReportValidationError(
                    "checkpoint categorical prior metadata digest is invalid"
                )

    config = _mapping(summary["config"], f"{summary_path}: config")
    for key in ("sha256", "sampling_sha256", "effective_sha256"):
        _sha256_value(config.get(key), f"config.{key}")
    sampling = _mapping(config.get("sampling"), "config.sampling")
    if _sha256_json(sampling) != config["sampling_sha256"]:
        raise ReportValidationError(f"{summary_path} config.sampling_sha256 is invalid")
    try:
        normalized_sampling = validate_sampling_config(sampling)
    except ValueError as exc:
        raise ReportValidationError(
            f"{summary_path} sampling config is invalid: {exc}"
        ) from exc
    if dict(sampling) != normalized_sampling:
        raise ReportValidationError(
            f"{summary_path} sampling config is not canonical: "
            f"expected {normalized_sampling}, found {dict(sampling)}"
        )
    if sampling["diffusion_type"] == "mdlm" and dict(sampling) != PAPER_V1_SAMPLING_CONFIG:
        raise ReportValidationError(
            f"{summary_path} sampling config is not the exact GenMol V1 MDLM protocol: "
            f"expected {PAPER_V1_SAMPLING_CONFIG}, found {dict(sampling)}"
        )
    if sampling["diffusion_type"] != checkpoint_diffusion_type:
        raise ReportValidationError(
            f"{summary_path} sampling/checkpoint diffusion types disagree"
        )
    if checkpoint_diffusion_type == "udlm":
        if checkpoint_udlm_inference_eps is None or not math.isclose(
            float(checkpoint_udlm_inference_eps),
            float(sampling["inference_eps"]),
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ReportValidationError(
                f"{summary_path} UDLM checkpoint/config inference_eps values disagree"
            )
        if (
            checkpoint_udlm_exclude_special_tokens
            is not sampling["exclude_special_tokens"]
        ):
            raise ReportValidationError(
                f"{summary_path} UDLM checkpoint/config special-token policies disagree"
            )
        if checkpoint_prior_variant != sampling["prior_variant"]:
            raise ReportValidationError(
                f"{summary_path} UDLM checkpoint/config prior variants disagree"
            )
        if checkpoint_prior_digest != sampling["prior_metadata_sha256"]:
            raise ReportValidationError(
                f"{summary_path} UDLM checkpoint/config prior metadata hashes disagree"
            )
        if checkpoint_prior_variant in UDLM_CATEGORICAL_PRIOR_VARIANTS:
            try:
                validate_udlm_prior_metadata_record(
                    checkpoint_prior_metadata,
                    expected_variant=checkpoint_prior_variant,
                    expected_exclude_special_tokens=sampling[
                        "exclude_special_tokens"
                    ],
                )
            except RuntimeError as exc:
                raise ReportValidationError(
                    f"categorical checkpoint/config prior identity is invalid: {exc}"
                ) from exc
    _validate_generation_protocol(
        generation_protocol,
        sampling,
        context=f"{summary_path}: run.generation_protocol",
    )
    source = _mapping(config.get("source"), "config.source")
    for key, expected in sampling.items():
        source_value = source.get(key)
        if (
            key == "prior_variant"
            and key not in source
            and sampling["diffusion_type"] == "udlm"
        ):
            source_value = "release_uniform"
        if source_value != expected:
            raise ReportValidationError(
                f"{summary_path} config.source.{key}={source.get(key)!r}; "
                f"expected exact V1 value {expected!r}"
            )
    if "num_samples" in source and source["num_samples"] != EXPECTED_SAMPLES_PER_SEED:
        raise ReportValidationError(
            f"{summary_path} config.source.num_samples must be 1000 when present"
        )
    effective = _mapping(config.get("effective"), "config.effective")
    if _sha256_json(effective) != config["effective_sha256"]:
        raise ReportValidationError(f"{summary_path} config.effective_sha256 is invalid")
    expected_effective = dict(source)
    expected_effective.update(
        {
            "model_path": checkpoint.get("path"),
            "num_samples": EXPECTED_SAMPLES_PER_SEED,
            "device": "cuda:0",
        }
    )
    if dict(effective) != expected_effective:
        raise ReportValidationError(
            f"{summary_path} effective config does not equal source config plus the "
            "evaluated checkpoint, 1000 samples, and logical cuda:0"
        )

    records = _load_csv(samples_path)
    raw_sha256 = _sha256_file(samples_path)
    artifacts = _mapping(summary["artifacts"], f"{summary_path}: artifacts")
    raw_artifact = _mapping(
        _required(artifacts, "raw_samples_csv", "artifacts"),
        "artifacts.raw_samples_csv",
    )
    if _sha256_value(raw_artifact.get("sha256"), "raw_samples_csv.sha256") != raw_sha256:
        raise ReportValidationError(f"{samples_path} SHA-256 disagrees with summary")
    if raw_artifact.get("row_count") != EXPECTED_SAMPLES_PER_SEED:
        raise ReportValidationError(f"{samples_path} summary row_count is not 1000")
    if raw_artifact.get("fields") != list(RAW_SAMPLE_FIELDS):
        raise ReportValidationError(f"{samples_path} summary field list disagrees with CSV")
    artifact_path = Path(str(raw_artifact.get("path", "")))
    if not artifact_path.is_absolute():
        artifact_path = (REPOSITORY_ROOT / artifact_path).resolve()
    else:
        artifact_path = artifact_path.resolve()
    if artifact_path != samples_path.resolve():
        raise ReportValidationError(
            f"{summary_path} points at {artifact_path}, not adjacent {samples_path.resolve()}"
        )

    strict_counts = _validate_branch_rows(records, prefix="strict", seed=expected_seed)
    released_counts = _validate_branch_rows(records, prefix="released", seed=expected_seed)
    cross_counts = _validate_cross_branch_rows(records, seed=expected_seed)
    metrics = _mapping(summary["metrics"], f"{summary_path}: metrics")
    if set(metrics) != {"released_comparable", "strict"}:
        raise ReportValidationError(f"{summary_path} has unexpected metric branches")
    _validate_metric_branch(
        _mapping(metrics["strict"], "metrics.strict"),
        strict_counts,
        seed=expected_seed,
        branch_name="strict",
    )
    _validate_metric_branch(
        _mapping(metrics["released_comparable"], "metrics.released_comparable"),
        released_counts,
        seed=expected_seed,
        branch_name="released_comparable",
    )

    expected_failures = {
        **cross_counts,
        "strict_decode_failed": EXPECTED_SAMPLES_PER_SEED - strict_counts["valid_count"],
        "released_decode_failed": (
            EXPECTED_SAMPLES_PER_SEED - released_counts["valid_count"]
        ),
        "strict_duplicates": strict_counts["valid_count"] - strict_counts["unique_count"],
        "released_duplicates": (
            released_counts["valid_count"] - released_counts["unique_count"]
        ),
    }
    failure_counts = _mapping(summary["failure_counts"], f"{summary_path}: failure_counts")
    if dict(failure_counts) != expected_failures:
        raise ReportValidationError(
            f"{summary_path} failure_counts disagree with raw rows: expected "
            f"{expected_failures}, found {dict(failure_counts)}"
        )

    runtime = _mapping(summary["runtime_seconds"], f"{summary_path}: runtime_seconds")
    required_runtime = {
        "model_load_and_device_move",
        "model_sampling_and_tokenizer",
        "released_postprocessing",
        "generation",
        "decode_and_metrics",
        "total_before_summary_write",
    }
    if set(runtime) != required_runtime:
        raise ReportValidationError(
            f"{summary_path} runtime fields must be exactly {sorted(required_runtime)}"
        )
    for name, value in runtime.items():
        number = _finite_number(value, f"runtime_seconds.{name}")
        if number < 0:
            raise ReportValidationError(f"runtime_seconds.{name} cannot be negative")
    expected_generation = (
        runtime["model_sampling_and_tokenizer"]
        + runtime["released_postprocessing"]
    )
    if not math.isclose(
        runtime["generation"], expected_generation, rel_tol=1e-9, abs_tol=1e-6
    ):
        raise ReportValidationError(
            f"{summary_path} generation runtime must equal model/tokenizer plus "
            "released postprocessing"
        )
    if runtime["total_before_summary_write"] + 1e-9 < sum(
        runtime[name]
        for name in (
            "model_load_and_device_move",
            "model_sampling_and_tokenizer",
            "decode_and_metrics",
        )
    ):
        raise ReportValidationError(
            f"{summary_path} total runtime is smaller than its timed components"
        )

    git = _mapping(summary["git"], f"{summary_path}: git")
    runner_sha = _sha256_value(git.get("runner_sha256"), "git.runner_sha256")
    git_commit = _git_revision_value(git.get("commit"), "git.commit")
    git_upstream = _git_revision_value(git.get("upstream"), "git.upstream")
    expected_source_revision = _git_revision_value(
        git.get("expected_source_revision"), "git.expected_source_revision"
    )
    if not (git_commit == git_upstream == expected_source_revision):
        raise ReportValidationError(
            f"{summary_path} Git commit, upstream, and expected source revision disagree"
        )
    if git.get("dirty") is not False:
        raise ReportValidationError(
            f"{summary_path} benchmark source was dirty during provenance capture"
        )
    if git.get("clean_pushed_source_verified_before_and_after_run") is not True:
        raise ReportValidationError(
            f"{summary_path} lacks child-side pre/post clean pushed-source verification"
        )
    environment = _mapping(summary["environment"], f"{summary_path}: environment")
    environment_signature, launch_provenance = _validate_cuda_provenance(
        environment,
        run,
        seed=expected_seed,
        checkpoint_global_step=global_step,
        checkpoint_sha256=checkpoint_sha,
        checkpoint_path=checkpoint.get("path"),
        config_path=config.get("path"),
        config_sha256=config["sha256"],
        git_commit=git_commit,
        summary_path=summary_path,
    )
    tokenizer = _validate_tokenizer_provenance(
        summary["tokenizer"], summary_path=summary_path
    )
    metric_inputs = _validate_metric_inputs(
        summary["metric_inputs"], summary_path=summary_path
    )
    implementation_inputs = _mapping(
        summary["implementation_inputs"], f"{summary_path}: implementation_inputs"
    )
    if set(implementation_inputs) != {
        "sampler_source",
        "model_source",
        "ema_source",
        "checkpoint_io_source",
        "diffusion_source",
        "backbone_source",
        "chemistry_utils_source",
        "data_utils_source",
        "moco_utils_source",
        "save_utils_source",
        "bracket_safe_converter_source",
        "length_distribution",
    }:
        raise ReportValidationError(
            f"{summary_path} has unexpected direct implementation-input fields"
        )
    for name, value in implementation_inputs.items():
        item = _mapping(value, f"implementation_inputs.{name}")
        _sha256_value(item.get("sha256"), f"implementation_inputs.{name}.sha256")
        size = _integer(item.get("size_bytes"), f"implementation_inputs.{name}.size_bytes")
        if size <= 0:
            raise ReportValidationError(
                f"implementation_inputs.{name}.size_bytes must be positive"
            )
    length_input = _mapping(
        implementation_inputs["length_distribution"],
        "implementation_inputs.length_distribution",
    )
    expected_length_stats = {
        "count": 249_455,
        "minimum": 10,
        "median": 49.0,
        "maximum": 87,
    }
    for key, expected in expected_length_stats.items():
        if length_input.get(key) != expected:
            raise ReportValidationError(
                f"implementation_inputs.length_distribution.{key}="
                f"{length_input.get(key)!r}; expected verified value {expected!r}"
            )
    expected_length_sha = TRAINING_CONTEXT["data_and_tokenizer"]["length_file"][
        "sha256"
    ]
    if length_input["sha256"] != expected_length_sha:
        raise ReportValidationError(
            "length-distribution SHA-256 differs from the verified training artifact"
        )

    return {
        "seed": expected_seed,
        "started_at_utc": run["started_at_utc"],
        "completed_at_utc": run["completed_at_utc"],
        "run_dir": str(run_dir.resolve()),
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": _sha256_file(summary_path),
        "raw_samples_path": str(samples_path.resolve()),
        "raw_samples_sha256": raw_sha256,
        "summary": summary,
        "checkpoint": dict(checkpoint),
        "config": dict(config),
        "metrics": {name: dict(_mapping(value, name)) for name, value in metrics.items()},
        "failure_counts": expected_failures,
        "runtime_seconds": dict(runtime),
        "environment": dict(environment),
        "git": dict(git),
        "runner_sha256": runner_sha,
        "environment_signature": environment_signature,
        "launch_provenance": launch_provenance,
        "tokenizer": tokenizer,
        "implementation_inputs": {
            name: dict(_mapping(value, f"implementation_inputs.{name}"))
            for name, value in implementation_inputs.items()
        },
        "metric_inputs": metric_inputs,
    }


def _aggregate(values: Sequence[float]) -> dict[str, Any]:
    if len(values) != len(EXPECTED_SEEDS):
        raise ReportValidationError("every aggregate must contain exactly three seed values")
    numbers = [float(value) for value in values]
    if not all(math.isfinite(value) for value in numbers):
        raise ReportValidationError("aggregate inputs must be finite")
    return {
        "n": len(numbers),
        "mean": statistics.fmean(numbers),
        "sample_sd": statistics.stdev(numbers),
        "sd_definition": (
            "Sample standard deviation across the three seed-level estimates "
            "(Bessel correction, denominator n-1, ddof=1)."
        ),
        "values_by_seed": [
            {"seed": seed, "value": value}
            for seed, value in zip(EXPECTED_SEEDS, numbers)
        ],
    }


def _common_identity(runs: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    first = runs[0]
    checkpoint = first["checkpoint"]
    config = first["config"]
    generation_protocol = first["summary"]["run"]["generation_protocol"]
    raw_hashes = [run["raw_samples_sha256"] for run in runs]
    if len(set(raw_hashes)) != len(raw_hashes):
        raise ReportValidationError(
            "two or more seeds have identical ordered raw-sample CSV SHA-256 digests; "
            "independent stochastic outputs are not established"
        )
    for run in runs[1:]:
        if run["checkpoint"] != checkpoint:
            raise ReportValidationError(
                "checkpoint metadata differs across seeds; aggregation is forbidden"
            )
        for key in ("sha256", "sampling_sha256", "effective_sha256"):
            if run["config"].get(key) != config.get(key):
                raise ReportValidationError(
                    f"config.{key} differs across seeds; aggregation is forbidden"
                )
        for key in ("source", "sampling", "effective"):
            if run["config"].get(key) != config.get(key):
                raise ReportValidationError(
                    f"config.{key} differs across seeds despite its fingerprint"
                )
        if run["runner_sha256"] != first["runner_sha256"]:
            raise ReportValidationError(
                "benchmark runner SHA-256 differs across seeds; aggregation is forbidden"
            )
        if run["implementation_inputs"] != first["implementation_inputs"]:
            raise ReportValidationError(
                "direct implementation/data input fingerprints differ across seeds; "
                "aggregation is forbidden"
            )
        if run["metric_inputs"] != first["metric_inputs"]:
            raise ReportValidationError(
                "metric-input artifact or TDC implementation fingerprints differ "
                "across seeds; aggregation is forbidden"
            )
        if run["tokenizer"] != first["tokenizer"]:
            raise ReportValidationError(
                "loaded tokenizer metadata/fingerprints differ across seeds; "
                "aggregation is forbidden"
            )
        if run["environment_signature"] != first["environment_signature"]:
            raise ReportValidationError(
                "dependency, CUDA, Python, platform, or GPU model metadata differs "
                "across seeds; aggregation is forbidden"
            )
        if run["launch_provenance"]["policy"] != first["launch_provenance"]["policy"]:
            raise ReportValidationError(
                "idle-GPU launch policy differs across seeds; aggregation is forbidden"
            )
        current_protocol = run["summary"]["run"]["generation_protocol"]
        if {
            key: value for key, value in current_protocol.items() if key != "nfe"
        } != {key: value for key, value in generation_protocol.items() if key != "nfe"}:
            raise ReportValidationError(
                "generation protocol differs across seeds; aggregation is forbidden"
            )
    return dict(checkpoint), dict(config)


def _funnel_for_run(run: Mapping[str, Any]) -> dict[str, int]:
    strict = run["metrics"]["strict"]
    released = run["metrics"]["released_comparable"]
    failures = run["failure_counts"]
    return {
        "requested": EXPECTED_SAMPLES_PER_SEED,
        "raw_safe_available": (
            EXPECTED_SAMPLES_PER_SEED - failures["raw_safe_conversion_failed"]
        ),
        "strict_valid": strict["valid_count"],
        "strict_unique": strict["unique_count"],
        "strict_quality": strict["quality_count"],
        "released_valid": released["valid_count"],
        "released_unique": released["unique_count"],
        "released_quality": released["quality_count"],
        "released_recovered_strict_failure": failures[
            "released_recovered_strict_failure"
        ],
        "strict_valid_but_released_failed": failures[
            "strict_valid_but_released_failed"
        ],
        "released_largest_component_applied": failures[
            "released_largest_component_applied"
        ],
    }


def _prior_interpretation(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Return an explicit causal label for the evaluated corruption process."""

    if checkpoint["diffusion_type"] == "mdlm":
        return {
            "variant": None,
            "label": "audited_local_mdlm_evaluation",
            "comparison_role": "local_mdlm_control",
            "process_family": "masked_diffusion_language_model",
            "schedule_variant": None,
            "prior_source": None,
            "objective_scope": None,
            "prior_metadata_sha256": None,
            "matched_prior_effect_control": None,
            "causal_claim_boundary": (
                "This is a local checkpoint evaluation, not the published GenMol run. "
                "Published GenMol V1 values remain an external paper reference, and no "
                "categorical-prior effect is estimated by this MDLM benchmark."
            ),
        }
    variant = checkpoint["udlm_prior_variant"]
    identity = UDLM_PRIOR_VARIANT_IDENTITIES[variant]
    labels = {
        "release_uniform": "faithful_release_uniform_control",
        "schedule_uniform": "schedule_repair_uniform_control",
        "empirical_frequency": "smoothed_empirical_prior_treatment",
    }
    if variant == "schedule_uniform":
        matched_control = "release_uniform diagnoses the schedule/process change"
        boundary = (
            "This is the uniform schedule-repair control. Any improvement over "
            "release_uniform or MDLM is not evidence of empirical-prior benefit. "
            "Its training objective excludes the parameter-independent endpoint KL, "
            "which must be reported separately."
        )
    elif variant == "empirical_frequency":
        matched_control = "schedule_uniform with the same categorical process and schedule"
        boundary = (
            "Only empirical_frequency minus a matched schedule_uniform checkpoint can "
            "estimate a stationary-prior effect. Comparison with release_uniform or MDLM "
            "conflates the prior with schedule/process changes and is not prior-benefit "
            "evidence. Its training objective excludes the parameter-independent endpoint "
            "KL, which must be reported separately."
        )
    else:
        matched_control = None
        boundary = (
            "This is the faithful released uniform UDLM control; it contains neither "
            "the categorical schedule repair nor an empirical prior."
        )
    return {
        "variant": variant,
        "label": labels[variant],
        "comparison_role": identity["comparison_role"],
        "process_family": identity["process_family"],
        "schedule_variant": identity["schedule_variant"],
        "prior_source": identity["prior_source"],
        "objective_scope": identity["objective_scope"],
        "prior_metadata_sha256": checkpoint["udlm_prior_metadata_sha256"],
        "matched_prior_effect_control": matched_control,
        "causal_claim_boundary": boundary,
    }


def collect_report(runs_dir: Path) -> dict[str, Any]:
    """Validate three run directories and return the report data model."""
    by_seed = discover_run_directories(runs_dir)
    runs = [
        _validate_summary_and_rows(by_seed[seed], seed)
        for seed in EXPECTED_SEEDS
    ]
    checkpoint, config = _common_identity(runs)
    prior_interpretation = _prior_interpretation(checkpoint)

    local_metrics: dict[str, Any] = {}
    for branch_name in BRANCH_PREFIX:
        local_metrics[branch_name] = {}
        for metric_name in METRIC_NAMES:
            values = [run["metrics"][branch_name][metric_name] for run in runs]
            if any(value is None for value in values):
                raise ReportValidationError(
                    f"cannot aggregate undefined {branch_name}.{metric_name}"
                )
            local_metrics[branch_name][metric_name] = _aggregate(values)
            local_metrics[branch_name][metric_name]["unit"] = (
                "score" if metric_name == "diversity" else "fraction"
            )
    local_metrics["runtime_seconds"] = {
        name: {**_aggregate([run["runtime_seconds"][name] for run in runs]), "unit": "seconds"}
        for name in (
            "model_load_and_device_move",
            "model_sampling_and_tokenizer",
            "released_postprocessing",
            "generation",
            "decode_and_metrics",
            "total_before_summary_write",
        )
    }

    comparison: dict[str, Any] = {}
    for metric_name in METRIC_NAMES:
        local = local_metrics["released_comparable"][metric_name]
        published = PAPER_REFERENCE["metrics"][metric_name]
        difference = local["mean"] - published["mean"]
        comparison[metric_name] = {
            "local_mean": local["mean"],
            "local_sample_sd": local["sample_sd"],
            "published_mean": published["mean"],
            "published_reported_sd": published["reported_sd"],
            "difference_local_minus_published": difference,
            "difference_unit": (
                "percentage_points" if published["unit"] == "fraction" else "score"
            ),
            "difference_display_value": (
                100.0 * difference if published["unit"] == "fraction" else difference
            ),
        }
    local_time = local_metrics["runtime_seconds"]["generation"]
    paper_time = PAPER_REFERENCE["metrics"]["generation_time"]
    comparison["generation_time"] = {
        "local_mean": local_time["mean"],
        "local_sample_sd": local_time["sample_sd"],
        "published_mean": paper_time["mean"],
        "published_reported_sd": paper_time["reported_sd"],
        "difference_local_minus_published": local_time["mean"] - paper_time["mean"],
        "difference_unit": "seconds",
        "difference_display_value": local_time["mean"] - paper_time["mean"],
        "comparability": (
            "Descriptive only: the local timer matches the released de_novo_generation "
            "boundary (model/tokenizer plus SAFE repair and largest-component selection), "
            "but local inference uses a recorded idle RTX A6000 while the "
            "paper used an A100. Point-in-time selection below the recorded exclusive "
            "utilization threshold cannot guarantee identical timing conditions, and "
            "the software environments are not identical."
        ),
    }

    per_seed_funnel = [
        {"seed": run["seed"], **_funnel_for_run(run)} for run in runs
    ]
    pooled_funnel = {
        key: sum(row[key] for row in per_seed_funnel)
        for key in per_seed_funnel[0]
        if key != "seed"
    }

    seed_rows = []
    for run in runs:
        environment = run["environment"]
        cuda_device = environment.get("cuda_device") or {}
        launch = run["launch_provenance"]
        seed_rows.append(
            {
                "seed": run["seed"],
                "started_at_utc": run["started_at_utc"],
                "completed_at_utc": run["completed_at_utc"],
                "summary_path": run["summary_path"],
                "summary_sha256": run["summary_sha256"],
                "raw_samples_path": run["raw_samples_path"],
                "raw_samples_sha256": run["raw_samples_sha256"],
                "metrics": run["metrics"],
                "failure_counts": run["failure_counts"],
                "runtime_seconds": run["runtime_seconds"],
                "device": {
                    "requested": environment.get("requested_device"),
                    "resolved": environment.get("resolved_model_device"),
                    "name": cuda_device.get("name"),
                    "logical_index": cuda_device.get("logical_index"),
                    "physical_index_at_launch": launch["physical_index"],
                    "uuid_at_launch": launch["uuid"],
                },
                "git": {
                    "commit": run["git"].get("commit"),
                    "branch": run["git"].get("branch"),
                    "dirty": run["git"].get("dirty"),
                    "runner_sha256": run["runner_sha256"],
                },
                "launch_provenance": run["launch_provenance"],
            }
        )

    metric_definitions = {
        "aggregation": (
            "Headline values are the arithmetic mean and sample SD of the three "
            "seed-level metrics. SD uses sqrt(sum((x_i - mean)^2)/(n-1)), n=3 "
            "(ddof=1); rows are not pooled before computing headline rates."
        ),
        "released_comparable": (
            "Released-code-compatible path: repair SAFE with fix=True, canonicalize, "
            "retain the largest disconnected component, then evaluate."
        ),
        "strict": (
            "Strict path: canonical SAFE decode with fix=False and no largest-component "
            "selection. It is a diagnostic local addition and has no Table 1 counterpart."
        ),
        "validity": "valid_count / 1,000 requested samples, independently per seed.",
        "uniqueness": "first-occurrence unique valid molecules / valid_count, per seed.",
        "quality": (
            "first-occurrence unique valid molecules with QED >= 0.6 and SA <= 4.0 / "
            "1,000 requested samples, per seed. SA uses TDC Oracle('sa') with the "
            f"pinned fragment-score artifact SHA-256 {SA_FRAGMENT_SCORES_SHA256}."
        ),
        "diversity": (
            "TDC Diversity on first-occurrence unique valid molecules; "
            "diversity_input_count records the input size."
        ),
        "generation_time": (
            "Released-compatible wall-clock boundary: one 1,000-sample model/tokenizer "
            "call plus SAFE fix=True repair and largest-component selection. It equals "
            "model_sampling_and_tokenizer + released_postprocessing; model loading, "
            "strict decoding, and metric evaluation are excluded."
        ),
    }
    deviations = [
        {
            "item": "Training global batch size",
            "local": "2,046 (62 microbatch x 3 GPUs x 11 accumulation)",
            "published": "2,048",
            "consequence": (
                "Small protocol deviation; checkpoint is not byte-identical to the "
                "released model."
            ),
        },
        {
            "item": "Training hardware",
            "local": "3 NVIDIA RTX A6000 GPUs; 46:19:39 wall time",
            "published": "8 NVIDIA A100 GPUs; approximately 5 hours",
            "consequence": (
                "Training and inference wall times are hardware-dependent and not "
                "speed reproductions."
            ),
        },
        {
            "item": "Dataset and tokenizer revisions",
            "local": "datamol-io/safe-gpt identifiers were unpinned",
            "published": "Exact local revisions cannot be proven from this run",
            "consequence": (
                "Possible corpus/tokenizer drift remains an uncontrolled "
                "reproducibility variable."
            ),
        },
        {
            "item": "Evaluation seeds",
            "local": "Explicit seeds 0, 1, 2",
            "published": "Three runs; seed values undisclosed",
            "consequence": (
                "Runs are repeatable locally but cannot be paired seed-for-seed with "
                "Table 1."
            ),
        },
        {
            "item": "Training RNG seed",
            "local": "Not recorded in the Hydra training config or retained training log",
            "published": "Not disclosed",
            "consequence": "The training trajectory cannot be recreated seed-for-seed.",
        },
    ]
    evaluated_diffusion_type = config["sampling"]["diffusion_type"]
    if evaluated_diffusion_type == "udlm":
        prior_variant = checkpoint["udlm_prior_variant"]
        prior_identity = UDLM_PRIOR_VARIANT_IDENTITIES[prior_variant]
        deviations = [
            {
                "item": "Diffusion formulation",
                "local": (
                    f"{prior_interpretation['label']}: "
                    f"{prior_identity['process_family']} / "
                    f"{prior_identity['schedule_variant']}"
                ),
                "published": "GenMol V1 masked diffusion (MDLM)",
                "consequence": (
                    prior_interpretation["causal_claim_boundary"]
                ),
            },
            {
                "item": "Evaluation seeds",
                "local": "Explicit seeds 0, 1, 2",
                "published": "Three runs; seed values undisclosed",
                "consequence": (
                    "Runs are repeatable locally but cannot be paired seed-for-seed."
                ),
            },
            {
                "item": "Training provenance",
                "local": "Checkpoint identity is exact; training details are external",
                "published": "GenMol V1 paper protocol",
                "consequence": (
                    "Interpret performance as a checkpoint comparison, not a controlled "
                    "training-system reproduction."
                ),
            },
        ]
    caveats = [
        (
            "This is a from-scratch implementation benchmark against published "
            "reference values, not a claim of exact model or environment reproduction."
        ),
        (
            "The released-comparable path intentionally repairs malformed SAFE and may "
            "drop disconnected components. Strict results expose the effect; neither "
            "path should be silently substituted for the other."
        ),
        (
            "With three local seeds, SD is descriptive. No hypothesis test, confidence "
            "interval, or equivalence claim is made. Paper seed values are unavailable."
        ),
        comparison["generation_time"]["comparability"],
        TRAINING_CONTEXT["data_and_tokenizer"]["revision_status"],
        (
            "Training RNG and full training-system provenance are not inferred from "
            "evaluation artifacts. Evaluation seeds are explicit but do not repair "
            "that gap."
        ),
        (
            "Implementation hashes establish that all local seeds used the same code. "
            "This reporter does not assert that those hashes equal a pristine NVIDIA "
            "release checkout."
        ),
        (
            "Quality depends on the pinned TDC synthetic-accessibility fragment-score "
            f"artifact ({SA_FRAGMENT_SCORES_RELATIVE_PATH.as_posix()}, SHA-256 "
            f"{SA_FRAGMENT_SCORES_SHA256}). The benchmark loads verified resident bytes "
            "and disables TDC's implicit downloader during scoring."
        ),
    ]
    if evaluated_diffusion_type == "udlm":
        caveats.append(prior_interpretation["causal_claim_boundary"])

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "completed",
        "scientific_status": (
            "Three-seed comparative benchmark; not an exact reproduction claim."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(_resolve_in_repository(runs_dir)),
        "required_protocol": {
            "seeds": list(EXPECTED_SEEDS),
            "samples_per_seed": EXPECTED_SAMPLES_PER_SEED,
            "seed_count": len(EXPECTED_SEEDS),
            "total_requested_samples": len(EXPECTED_SEEDS)
            * EXPECTED_SAMPLES_PER_SEED,
        },
        "checkpoint": checkpoint,
        "udlm_prior_interpretation": prior_interpretation,
        "config": config,
        "generation_protocol": {
            **runs[0]["summary"]["run"]["generation_protocol"],
            "nfe": (
                runs[0]["summary"]["run"]["generation_protocol"]["nfe"]
                if len(
                    {
                        run["summary"]["run"]["generation_protocol"]["nfe"]
                        for run in runs
                    }
                )
                == 1
                else None
            ),
            "nfe_by_seed": [
                {
                    "seed": run["seed"],
                    "nfe": run["summary"]["run"]["generation_protocol"]["nfe"],
                }
                for run in runs
            ],
        },
        "runner_sha256": runs[0]["runner_sha256"],
        "implementation_inputs": runs[0]["implementation_inputs"],
        "metric_inputs": runs[0]["metric_inputs"],
        "tokenizer": runs[0]["tokenizer"],
        "environment_consistency": {
            "shared_signature": runs[0]["environment_signature"],
            "all_seed_signatures_equal": True,
            "all_launch_policies_equal": True,
            "idle_gpu_policy_verified": True,
            "distinct_raw_sample_csv_sha256": True,
            "per_seed_launch_provenance": [
                {"seed": run["seed"], **run["launch_provenance"]} for run in runs
            ],
        },
        "training_context": (
            TRAINING_CONTEXT
            if evaluated_diffusion_type == "mdlm"
            else {
                "scope_note": (
                    "The bundled detailed training context describes the audited MDLM "
                    "baseline, not the evaluated UDLM checkpoint."
                ),
                "evaluated_checkpoint": checkpoint,
                "mdlm_baseline_reference": TRAINING_CONTEXT,
            }
        ),
        "published_reference": PAPER_REFERENCE,
        "seed_runs": seed_rows,
        "aggregate_metrics": local_metrics,
        "comparison_to_published_genmol_v1": comparison,
        "strict_vs_repaired_funnel": {
            "per_seed": per_seed_funnel,
            "sum_across_seeds": pooled_funnel,
            "pooling_note": (
                "Unique counts are summed within seed; molecules are not deduplicated "
                "across independent runs."
            ),
        },
        "metric_definitions": metric_definitions,
        "documented_deviations": deviations,
        "caveats": caveats,
        "artifacts": {},
    }


CSV_FIELDS = (
    "row_type",
    "evaluation_path",
    "seed",
    "metric",
    "value",
    "numerator",
    "denominator",
    "n_seeds",
    "mean",
    "sample_sd",
    "published_mean",
    "published_reported_sd",
    "difference_local_minus_published",
    "unit",
    "checkpoint_sha256",
    "global_step",
    "udlm_prior_variant",
    "udlm_prior_metadata_sha256",
    "udlm_comparison_role",
    "udlm_objective_scope",
    "note",
)


def aggregate_csv_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    checkpoint = payload["checkpoint"]
    common = {
        "checkpoint_sha256": checkpoint["sha256"],
        "global_step": checkpoint["global_step"],
        "udlm_prior_variant": checkpoint.get("udlm_prior_variant"),
        "udlm_prior_metadata_sha256": checkpoint.get(
            "udlm_prior_metadata_sha256"
        ),
        "udlm_comparison_role": payload["udlm_prior_interpretation"][
            "comparison_role"
        ],
        "udlm_objective_scope": payload["udlm_prior_interpretation"][
            "objective_scope"
        ],
    }
    count_names = {
        "validity": ("valid_count", "validity_denominator"),
        "uniqueness": ("unique_count", "uniqueness_denominator"),
        "quality": ("quality_count", "quality_denominator"),
        "diversity": (None, "diversity_input_count"),
    }
    for seed_run in payload["seed_runs"]:
        for branch_name in BRANCH_PREFIX:
            branch = seed_run["metrics"][branch_name]
            for metric_name in METRIC_NAMES:
                numerator_name, denominator_name = count_names[metric_name]
                rows.append(
                    {
                        "row_type": "seed_metric",
                        "evaluation_path": branch_name,
                        "seed": seed_run["seed"],
                        "metric": metric_name,
                        "value": branch[metric_name],
                        "numerator": branch[numerator_name] if numerator_name else "",
                        "denominator": branch[denominator_name],
                        "unit": "score" if metric_name == "diversity" else "fraction",
                        "note": branch["definition"],
                        **common,
                    }
                )
        rows.append(
            {
                "row_type": "seed_metric",
                "evaluation_path": "timing",
                "seed": seed_run["seed"],
                "metric": "generation_time",
                "value": seed_run["runtime_seconds"]["generation"],
                "unit": "seconds",
                "note": payload["metric_definitions"]["generation_time"],
                **common,
            }
        )

    for branch_name in BRANCH_PREFIX:
        for metric_name in METRIC_NAMES:
            aggregate = payload["aggregate_metrics"][branch_name][metric_name]
            comparison = (
                payload["comparison_to_published_genmol_v1"][metric_name]
                if branch_name == "released_comparable"
                else None
            )
            rows.append(
                {
                    "row_type": "aggregate_metric",
                    "evaluation_path": branch_name,
                    "metric": metric_name,
                    "n_seeds": aggregate["n"],
                    "mean": aggregate["mean"],
                    "sample_sd": aggregate["sample_sd"],
                    "published_mean": comparison["published_mean"] if comparison else "",
                    "published_reported_sd": (
                        comparison["published_reported_sd"] if comparison else ""
                    ),
                    "difference_local_minus_published": (
                        comparison["difference_local_minus_published"]
                        if comparison
                        else ""
                    ),
                    "unit": aggregate["unit"],
                    "note": aggregate["sd_definition"],
                    **common,
                }
            )
    time_aggregate = payload["aggregate_metrics"]["runtime_seconds"]["generation"]
    time_comparison = payload["comparison_to_published_genmol_v1"]["generation_time"]
    rows.append(
        {
            "row_type": "aggregate_metric",
            "evaluation_path": "timing",
            "metric": "generation_time",
            "n_seeds": time_aggregate["n"],
            "mean": time_aggregate["mean"],
            "sample_sd": time_aggregate["sample_sd"],
            "published_mean": time_comparison["published_mean"],
            "published_reported_sd": time_comparison["published_reported_sd"],
            "difference_local_minus_published": time_comparison[
                "difference_local_minus_published"
            ],
            "unit": "seconds",
            "note": time_comparison["comparability"],
            **common,
        }
    )
    return [{field: row.get(field, "") for field in CSV_FIELDS} for row in rows]


def _atomic_write_text(path: Path, writer: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    def writer(handle: Any) -> None:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")

    _atomic_write_text(path, writer)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    def writer(handle: Any) -> None:
        csv_writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="raise")
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    _atomic_write_text(path, writer)


def _format_metric(metric_name: str, mean: float, sd: float) -> str:
    if metric_name in {"validity", "uniqueness", "quality"}:
        return f"{100.0 * mean:.2f} +/- {100.0 * sd:.2f}%"
    if metric_name == "diversity":
        return f"{mean:.4f} +/- {sd:.4f}"
    return f"{mean:.2f} +/- {sd:.2f} s"


def _format_difference(metric_name: str, difference: float) -> str:
    if metric_name in {"validity", "uniqueness", "quality"}:
        return f"{100.0 * difference:+.2f} pp"
    if metric_name == "diversity":
        return f"{difference:+.4f}"
    return f"{difference:+.2f} s"


def _comparison_chart(payload: Mapping[str, Any]) -> Any:
    from reportlab.graphics.shapes import Drawing, Line, Rect, String
    from reportlab.lib import colors

    drawing = Drawing(480, 205)
    drawing.add(
        Rect(
            0,
            0,
            480,
            205,
            fillColor=colors.HexColor("#F5F8FA"),
            strokeColor=None,
        )
    )
    left, bottom, top = 54, 36, 178
    width = 402
    drawing.add(Line(left, bottom, left, top, strokeColor=colors.HexColor("#82929C")))
    drawing.add(
        Line(
            left,
            bottom,
            left + width,
            bottom,
            strokeColor=colors.HexColor("#82929C"),
        )
    )
    for tick in (0, 25, 50, 75, 100):
        y = bottom + (top - bottom) * tick / 105.0
        drawing.add(
            Line(
                left,
                y,
                left + width,
                y,
                strokeColor=colors.HexColor("#DDE5E9"),
            )
        )
        drawing.add(
            String(
                left - 8,
                y - 3,
                str(tick),
                fontSize=7,
                textAnchor="end",
                fillColor=colors.HexColor("#53636D"),
            )
        )
    metrics = ("validity", "uniqueness", "quality", "diversity")
    comparison = payload["comparison_to_published_genmol_v1"]
    group_width = width / len(metrics)
    bar_width = 24
    local_color = colors.HexColor("#006D77")
    paper_color = colors.HexColor("#E29578")
    for index, metric in enumerate(metrics):
        item = comparison[metric]
        local = 100.0 * item["local_mean"]
        paper = 100.0 * item["published_mean"]
        center = left + group_width * (index + 0.5)
        for offset, value, color in (
            (-bar_width, local, local_color),
            (2, paper, paper_color),
        ):
            height = (top - bottom) * value / 105.0
            drawing.add(
                Rect(
                    center + offset,
                    bottom,
                    bar_width - 3,
                    height,
                    fillColor=color,
                    strokeColor=None,
                )
            )
        drawing.add(
            String(
                center,
                18,
                METRIC_LABELS[metric],
                fontSize=7.2,
                textAnchor="middle",
                fillColor=colors.HexColor("#25343B"),
            )
        )
    drawing.add(Rect(304, 188, 9, 9, fillColor=local_color, strokeColor=None))
    drawing.add(String(318, 189, "Local released-comparable", fontSize=7.2))
    drawing.add(Rect(411, 188, 9, 9, fillColor=paper_color, strokeColor=None))
    drawing.add(String(425, 189, "Paper", fontSize=7.2))
    drawing.add(String(12, 187, "score x 100", fontSize=7.2, fillColor=colors.HexColor("#53636D")))
    return drawing


def render_pdf(payload: Mapping[str, Any], pdf_path: Path) -> None:
    """Render a dedicated, self-validating benchmark PDF."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        raise RuntimeError(
            "PDF generation requires reportlab from requirements-stage0.lock"
        ) from exc

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{pdf_path.stem}.", suffix=".pdf", dir=pdf_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)

    palette = {
        "ink": colors.HexColor("#1F3038"),
        "muted": colors.HexColor("#5E7079"),
        "teal": colors.HexColor("#006D77"),
        "pale_teal": colors.HexColor("#E5F3F3"),
        "coral": colors.HexColor("#E29578"),
        "pale_coral": colors.HexColor("#FFF0EA"),
        "line": colors.HexColor("#D9E2E6"),
        "paper": colors.HexColor("#F7F9FA"),
        "white": colors.white,
    }
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=25,
            leading=29,
            textColor=palette["ink"],
            alignment=TA_LEFT,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=palette["muted"],
            spaceAfter=12,
        ),
        "kicker": ParagraphStyle(
            "Kicker",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=7.5,
            leading=9,
            textColor=palette["teal"],
            spaceAfter=6,
        ),
        "h1": ParagraphStyle(
            "Section",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=19,
            textColor=palette["teal"],
            spaceBefore=3,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "Subsection",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=palette["ink"],
            spaceBefore=8,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.8,
            leading=12.2,
            textColor=palette["ink"],
            spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.1,
            leading=9.4,
            textColor=palette["muted"],
        ),
        "table": ParagraphStyle(
            "TableText",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.2,
            leading=9,
            textColor=palette["ink"],
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.2,
            leading=9,
            textColor=palette["white"],
        ),
        "callout": ParagraphStyle(
            "Callout",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=13,
            textColor=palette["ink"],
        ),
    }

    def table(
        raw_rows: Sequence[Sequence[Any]],
        widths: Sequence[float],
        *,
        header: bool = True,
        highlight_last: bool = False,
    ) -> Any:
        converted = []
        for row_index, row in enumerate(raw_rows):
            style = styles["table_header"] if header and row_index == 0 else styles["table"]
            converted.append(
                [
                    item if hasattr(item, "wrap") else Paragraph(escape(str(item)), style)
                    for item in row
                ]
            )
        result = Table(
            converted,
            colWidths=widths,
            repeatRows=1 if header else 0,
            hAlign="LEFT",
        )
        commands = [
            ("BACKGROUND", (0, 0), (-1, 0), palette["teal"]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.35, palette["line"]),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
        if len(raw_rows) > 2:
            for row_index in range(1, len(raw_rows)):
                if row_index % 2 == 0:
                    commands.append(
                        (
                            "BACKGROUND",
                            (0, row_index),
                            (-1, row_index),
                            palette["paper"],
                        )
                    )
        if highlight_last:
            commands.append(("BACKGROUND", (0, -1), (-1, -1), palette["pale_teal"]))
        result.setStyle(TableStyle(commands))
        return result

    def callout(text: str, *, caution: bool = False) -> Any:
        background = palette["pale_coral"] if caution else palette["pale_teal"]
        accent = palette["coral"] if caution else palette["teal"]
        item = Table([[Paragraph(escape(text), styles["callout"])]], colWidths=[174 * mm])
        item.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), background),
                    ("BOX", (0, 0), (-1, -1), 0.8, accent),
                    ("LEFTPADDING", (0, 0), (-1, -1), 9),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        return item

    checkpoint = payload["checkpoint"]
    comparison = payload["comparison_to_published_genmol_v1"]
    story: list[Any] = [
        Spacer(1, 11 * mm),
        Paragraph("AUDITABLE 3 x 1,000 DE NOVO EVALUATION", styles["kicker"]),
        Paragraph("GenMol from-scratch benchmark", styles["title"]),
        Paragraph(
            f"Evaluated {checkpoint['diffusion_type'].upper()} checkpoint at step "
            f"{checkpoint['global_step']:,} compared with published GenMol V1 Table 1",
            styles["subtitle"],
        ),
        callout(
            "COMPARATIVE RESULT - NOT AN EXACT REPRODUCTION CLAIM. All three local "
            "seeds completed with 1,000 retained raw rows and passed independent count, "
            "checkpoint, configuration, and artifact-hash validation."
        ),
        callout(
            payload["udlm_prior_interpretation"]["causal_claim_boundary"],
            caution=checkpoint["diffusion_type"] == "udlm",
        ),
        Spacer(1, 7 * mm),
        Paragraph("Headline comparison", styles["h1"]),
    ]
    headline = [["Metric", "Local released-comparable", "Published GenMol V1", "Local - paper"]]
    for metric_name in (*METRIC_NAMES, "generation_time"):
        item = comparison[metric_name]
        headline.append(
            [
                METRIC_LABELS[metric_name],
                _format_metric(metric_name, item["local_mean"], item["local_sample_sd"]),
                _format_metric(
                    metric_name,
                    item["published_mean"],
                    item["published_reported_sd"],
                ),
                _format_difference(metric_name, item["difference_local_minus_published"]),
            ]
        )
    story.extend(
        [
            table(headline, [37 * mm, 55 * mm, 50 * mm, 32 * mm]),
            Spacer(1, 3 * mm),
            Paragraph(
                escape(payload["metric_definitions"]["aggregation"]), styles["small"]
            ),
            Spacer(1, 4 * mm),
            _comparison_chart(payload),
            Paragraph(
                "Bars show means on a 0-100 scale (diversity multiplied by 100). "
                "Uncertainty remains in the tables to avoid suggesting paired estimates.",
                styles["small"],
            ),
            Spacer(1, 5 * mm),
            Paragraph("Interpretation boundary", styles["h2"]),
            Paragraph(
                "Differences are descriptive. The evaluated checkpoint and its "
                "backend-specific protocol are identified below, while the published "
                "run seeds are undisclosed. No significance, parity, superiority, or "
                "exact reproduction conclusion follows from this table.",
                styles["body"],
            ),
            PageBreak(),
            Paragraph("Seed-level results and decoding funnel", styles["h1"]),
            Paragraph(
                "The released-comparable branch is the Table 1 comparison target. The "
                "strict branch exposes output that decodes without SAFE repair or "
                "largest-component selection.",
                styles["body"],
            ),
            Paragraph("Released-comparable metrics", styles["h2"]),
        ]
    )

    released_rows = [
        ["Seed", "Validity", "Uniqueness", "Quality", "Diversity", "Generation"]
    ]
    strict_rows = [["Seed", "Validity", "Uniqueness", "Quality", "Diversity"]]
    for seed_run in payload["seed_runs"]:
        released = seed_run["metrics"]["released_comparable"]
        strict = seed_run["metrics"]["strict"]
        released_rows.append(
            [
                seed_run["seed"],
                f"{100 * released['validity']:.2f}% ({released['valid_count']}/1000)",
                f"{100 * released['uniqueness']:.2f}% "
                f"({released['unique_count']}/{released['uniqueness_denominator']})",
                f"{100 * released['quality']:.2f}% ({released['quality_count']}/1000)",
                f"{released['diversity']:.4f} (n={released['diversity_input_count']})",
                f"{seed_run['runtime_seconds']['generation']:.2f} s",
            ]
        )
        strict_rows.append(
            [
                seed_run["seed"],
                f"{100 * strict['validity']:.2f}% ({strict['valid_count']}/1000)",
                f"{100 * strict['uniqueness']:.2f}% "
                f"({strict['unique_count']}/{strict['uniqueness_denominator']})",
                f"{100 * strict['quality']:.2f}% ({strict['quality_count']}/1000)",
                f"{strict['diversity']:.4f} (n={strict['diversity_input_count']})",
            ]
        )
    released_aggregate = payload["aggregate_metrics"]["released_comparable"]
    strict_aggregate = payload["aggregate_metrics"]["strict"]
    released_rows.append(
        [
            "Mean +/- sample SD",
            _format_metric(
                "validity",
                released_aggregate["validity"]["mean"],
                released_aggregate["validity"]["sample_sd"],
            ),
            _format_metric(
                "uniqueness",
                released_aggregate["uniqueness"]["mean"],
                released_aggregate["uniqueness"]["sample_sd"],
            ),
            _format_metric(
                "quality",
                released_aggregate["quality"]["mean"],
                released_aggregate["quality"]["sample_sd"],
            ),
            _format_metric(
                "diversity",
                released_aggregate["diversity"]["mean"],
                released_aggregate["diversity"]["sample_sd"],
            ),
            _format_metric(
                "generation_time",
                payload["aggregate_metrics"]["runtime_seconds"]["generation"][
                    "mean"
                ],
                payload["aggregate_metrics"]["runtime_seconds"]["generation"][
                    "sample_sd"
                ],
            ),
        ]
    )
    strict_rows.append(
        [
            "Mean +/- sample SD",
            _format_metric(
                "validity",
                strict_aggregate["validity"]["mean"],
                strict_aggregate["validity"]["sample_sd"],
            ),
            _format_metric(
                "uniqueness",
                strict_aggregate["uniqueness"]["mean"],
                strict_aggregate["uniqueness"]["sample_sd"],
            ),
            _format_metric(
                "quality",
                strict_aggregate["quality"]["mean"],
                strict_aggregate["quality"]["sample_sd"],
            ),
            _format_metric(
                "diversity",
                strict_aggregate["diversity"]["mean"],
                strict_aggregate["diversity"]["sample_sd"],
            ),
        ]
    )
    story.extend(
        [
            table(
                released_rows,
                [18 * mm, 31 * mm, 34 * mm, 31 * mm, 33 * mm, 27 * mm],
                highlight_last=True,
            ),
            Paragraph("Strict metrics", styles["h2"]),
            table(
                strict_rows,
                [30 * mm, 35 * mm, 38 * mm, 35 * mm, 36 * mm],
                highlight_last=True,
            ),
            Paragraph("Strict vs repaired funnel (counts)", styles["h2"]),
        ]
    )
    funnel_rows = [
        [
            "Seed",
            "Requested / raw SAFE",
            "Strict valid > unique > quality",
            "Repair recovered / released lost",
            "Released valid > unique > quality",
            "Largest component applied",
        ]
    ]
    for row in payload["strict_vs_repaired_funnel"]["per_seed"]:
        funnel_rows.append(
            [
                row["seed"],
                f"{row['requested']} / {row['raw_safe_available']}",
                f"{row['strict_valid']} > {row['strict_unique']} > {row['strict_quality']}",
                f"{row['released_recovered_strict_failure']} / "
                f"{row['strict_valid_but_released_failed']}",
                f"{row['released_valid']} > {row['released_unique']} > "
                f"{row['released_quality']}",
                row["released_largest_component_applied"],
            ]
        )
    pooled = payload["strict_vs_repaired_funnel"]["sum_across_seeds"]
    funnel_rows.append(
        [
            "Sum",
            f"{pooled['requested']} / {pooled['raw_safe_available']}",
            f"{pooled['strict_valid']} > {pooled['strict_unique']} > {pooled['strict_quality']}",
            f"{pooled['released_recovered_strict_failure']} / "
            f"{pooled['strict_valid_but_released_failed']}",
            f"{pooled['released_valid']} > {pooled['released_unique']} > "
            f"{pooled['released_quality']}",
            pooled["released_largest_component_applied"],
        ]
    )
    story.extend(
        [
            table(
                funnel_rows,
                [13 * mm, 27 * mm, 38 * mm, 34 * mm, 40 * mm, 22 * mm],
                highlight_last=True,
            ),
            Paragraph(
                escape(payload["strict_vs_repaired_funnel"]["pooling_note"]), styles["small"]
            ),
            Paragraph("Exact metric definitions", styles["h2"]),
        ]
    )
    definitions = payload["metric_definitions"]
    definition_rows = [["Metric/path", "Definition and denominator"]]
    for key in (
        "released_comparable",
        "strict",
        "validity",
        "uniqueness",
        "quality",
        "diversity",
        "generation_time",
    ):
        definition_rows.append([key.replace("_", " ").title(), definitions[key]])
    story.extend(
        [
            table(definition_rows, [39 * mm, 135 * mm]),
            PageBreak(),
            Paragraph("Checkpoint and protocol provenance", styles["h1"]),
            callout(
                f"Checkpoint identity: global step {checkpoint['global_step']:,}; "
                f"SHA-256 {checkpoint['sha256']}. All three summaries agree."
            ),
            Paragraph("Verified checkpoint", styles["h2"]),
        ]
    )
    checkpoint_rows = [
        ["Field", "Value"],
        ["Resolved path", checkpoint.get("path")],
        ["SHA-256", checkpoint["sha256"]],
        ["Size", f"{checkpoint['size_bytes']:,} bytes"],
        ["Global step", f"{checkpoint['global_step']:,}"],
        [
            "Diffusion backend",
            checkpoint["diffusion_type"].upper(),
        ],
        [
            "UDLM inference epsilon",
            checkpoint.get("udlm_inference_eps") or "not applicable",
        ],
        [
            "UDLM excludes special tokens",
            (
                checkpoint.get("udlm_exclude_special_tokens")
                if checkpoint["diffusion_type"] == "udlm"
                else "not applicable"
            ),
        ],
        [
            "UDLM prior variant",
            checkpoint.get("udlm_prior_variant") or "not applicable",
        ],
        [
            "UDLM prior metadata SHA-256",
            checkpoint.get("udlm_prior_metadata_sha256") or "not applicable",
        ],
        [
            "UDLM comparison role",
            payload["udlm_prior_interpretation"]["comparison_role"],
        ],
        [
            "UDLM objective scope",
            payload["udlm_prior_interpretation"]["objective_scope"]
            or "not applicable",
        ],
        ["Config SHA-256", payload["config"]["sha256"]],
        ["Effective-config SHA-256", payload["config"]["effective_sha256"]],
        ["Sampling-config SHA-256", payload["config"]["sampling_sha256"]],
        ["Runner SHA-256", payload["runner_sha256"]],
        [
            "Sampler source SHA-256",
            payload["implementation_inputs"]["sampler_source"]["sha256"],
        ],
        [
            "Model source SHA-256",
            payload["implementation_inputs"]["model_source"]["sha256"],
        ],
        [
            "EMA source SHA-256",
            payload["implementation_inputs"]["ema_source"]["sha256"],
        ],
        [
            "Checkpoint I/O source SHA-256",
            payload["implementation_inputs"]["checkpoint_io_source"]["sha256"],
        ],
        [
            "Diffusion source SHA-256",
            payload["implementation_inputs"]["diffusion_source"]["sha256"],
        ],
        [
            "Backbone source SHA-256",
            payload["implementation_inputs"]["backbone_source"]["sha256"],
        ],
        [
            "Chemistry utilities SHA-256",
            payload["implementation_inputs"]["chemistry_utils_source"]["sha256"],
        ],
        [
            "Data utilities SHA-256",
            payload["implementation_inputs"]["data_utils_source"]["sha256"],
        ],
        [
            "MoCo utilities SHA-256",
            payload["implementation_inputs"]["moco_utils_source"]["sha256"],
        ],
        [
            "Checkpoint-save utilities SHA-256",
            payload["implementation_inputs"]["save_utils_source"]["sha256"],
        ],
        [
            "Bracket converter SHA-256",
            payload["implementation_inputs"]["bracket_safe_converter_source"]["sha256"],
        ],
        [
            "Length distribution",
            f"n=249,455; min/median/max=10/49/87; SHA-256 "
            f"{payload['implementation_inputs']['length_distribution']['sha256']}",
        ],
        [
            "SA fragment scores",
            f"{payload['metric_inputs']['sa_fragment_scores']['relative_path']}; "
            f"{payload['metric_inputs']['sa_fragment_scores']['size_bytes']:,} bytes; "
            f"SHA-256 {payload['metric_inputs']['sa_fragment_scores']['sha256']}",
        ],
        [
            "SA metric backend",
            f"{payload['metric_inputs']['tdc_metric_implementation']['distribution']} "
            f"{payload['metric_inputs']['tdc_metric_implementation']['version']}; "
            "verified resident scores; network downloader disabled",
        ],
    ]
    story.extend(
        [
            table(checkpoint_rows, [50 * mm, 124 * mm]),
            Paragraph("Audited evaluation protocol", styles["h2"]),
        ]
    )
    sampling = payload["config"]["sampling"]
    generation_protocol = payload["generation_protocol"]
    protocol_rows = [
        ["Setting", "Audited local value"],
        ["Runs / seeds", "3 independent invocations; explicit seeds 0, 1, 2"],
        ["Samples", "1,000 per seed; one generation batch; 3,000 requested total"],
        ["Diffusion backend", sampling["diffusion_type"].upper()],
        ["Prior variant", sampling["prior_variant"] or "not applicable"],
        [
            "Prior metadata SHA-256",
            sampling["prior_metadata_sha256"] or "not applicable",
        ],
        ["Softmax temperature", sampling["softmax_temp"]],
        ["Randomness", sampling["randomness"]],
        [
            "Randomness effective",
            generation_protocol["randomness_used_by_sampler"],
        ],
        ["Minimum added length", sampling["min_add_len"]],
        [
            "Reverse steps",
            sampling["num_steps"] or "MDLM confidence-derived",
        ],
        [
            "Inference epsilon",
            sampling["inference_eps"] or "not applicable",
        ],
        [
            "NFE by seed",
            ", ".join(
                f"{row['seed']}: {row['nfe']}"
                for row in generation_protocol["nfe_by_seed"]
            ),
        ],
        [
            "Released postprocessing",
            "SAFE fix=True; canonical decode; retain component with maximum SMILES string length",
        ],
        ["Strict diagnostic", "SAFE fix=False; no component selection"],
        [
            "Generation timer",
            "model_sampling_and_tokenizer + released_postprocessing",
        ],
    ]
    tokenizer = payload["tokenizer"]
    tokenizer_rows = [
        ["Tokenizer field", "Recorded inference value"],
        [
            "Identity",
            f"requested={tokenizer['requested_identifier']}; "
            f"runtime name={tokenizer['name_or_path'] or 'not retained'}; "
            f"class {tokenizer['class']}",
        ],
        [
            "Revision",
            f"declared={tokenizer['declared_revision'] or 'not declared'}; "
            f"resolved={tokenizer['resolved_commit_hash'] or 'not recorded'}",
        ],
        [
            "Vocabulary",
            f"base=1,880; effective=1,882 after repository-added tokens; "
            f"SHA-256 {tokenizer['vocabulary_sha256']}",
        ],
        [
            "Added/backend hashes",
            f"added={tokenizer['added_vocabulary_sha256']}; "
            f"backend={tokenizer['backend_json_sha256'] or 'not serializable'}",
        ],
        ["Special IDs", "PAD=3, BOS=1, EOS=2, MASK=4"],
    ]
    story.extend(
        [
            table(protocol_rows, [50 * mm, 124 * mm]),
            callout(
                payload["udlm_prior_interpretation"]["causal_claim_boundary"]
            ),
            Paragraph("Loaded tokenizer provenance", styles["h2"]),
            table(tokenizer_rows, [50 * mm, 124 * mm]),
            PageBreak(),
            Paragraph("Documented deviations and published source", styles["h1"]),
            Paragraph("Known training and evaluation deviations", styles["h2"]),
        ]
    )
    deviation_rows = [["Item", "Local", "Published/reference", "Why it matters"]]
    for item in payload["documented_deviations"]:
        deviation_rows.append(
            [item["item"], item["local"], item["published"], item["consequence"]]
        )
    story.extend(
        [
            table(deviation_rows, [34 * mm, 45 * mm, 42 * mm, 53 * mm]),
            Spacer(1, 4 * mm),
            callout(
                "HARDWARE TIMING CAVEAT. Local generation time is measured on "
                "recorded idle RTX A6000 inference devices, using the released "
                "de_novo_generation timer boundary (model/tokenizer + SAFE repair + "
                "largest component). Each launch snapshot satisfied an exclusive "
                "utilization threshold of at most 10%, at least 30,000 MiB free memory, "
                "and no active compute process. Table 1 used an "
                "A100 and a different software environment. Hardware can affect runtime, "
                "so the time delta is descriptive and is not a speed claim.",
                caution=True,
            ),
            Paragraph("Published source", styles["h2"]),
            Paragraph(
                "Primary paper PDF: "
                f'<link href="{PAPER_PRIMARY_URL}" color="#006D77">{PAPER_PRIMARY_URL}</link>. '
                "Published values transcribed here: validity 100.0 +/- 0.0%, uniqueness "
                "99.7 +/- 0.1%, quality 84.6 +/- 0.8%, diversity 0.818 +/- 0.001, and "
                "time 21.1 +/- 0.4 seconds.",
                styles["body"],
            ),
            Paragraph(
                "The paper reports three 1,000-sample runs, but its seed values are "
                "undisclosed. Local seeds 0, 1, and 2 are therefore explicit repeatability "
                "choices, not matched replicas of paper runs.",
                styles["body"],
            ),
            PageBreak(),
            Paragraph("Audit trail and claims boundary", styles["h1"]),
            Paragraph("Per-seed artifact identities", styles["h2"]),
        ]
    )
    artifact_rows = [["Seed", "Raw CSV SHA-256", "Summary SHA-256", "Inference device"]]
    for seed_run in payload["seed_runs"]:
        device = seed_run["device"]
        device_label = device.get("name") or device.get("resolved") or "not recorded"
        physical = device.get("physical_index_at_launch")
        if physical is not None:
            device_label += f"; physical {physical}, logical {device.get('logical_index')}"
        artifact_rows.append(
            [
                seed_run["seed"],
                seed_run["raw_samples_sha256"],
                seed_run["summary_sha256"],
                device_label,
            ]
        )
    selection_rows = [
        [
            "Seed",
            "Physical GPU selection",
            "Load / free memory at final launch probe",
            "Active compute processes at final launch probe",
        ]
    ]
    for seed_run in payload["seed_runs"]:
        launch = seed_run["launch_provenance"]
        policy = launch["policy"]
        processes = launch["compute_processes_at_selection"]
        process_details = "none"
        if processes:
            process_details = "; ".join(
                f"PID {process['pid']} {process['process_name']} "
                f"({process['used_memory_mib']} MiB)"
                for process in processes
            )
        if launch["selection_method"] == "dynamic_idle_discovery":
            selection_text = (
                f"{launch['physical_index']} (dynamic; full inventory; "
                f"requested {launch['user_requested_gpu_count']})"
            )
        else:
            selected_ids = ", ".join(
                str(index) for index in launch["user_selected_physical_indices"]
            )
            selection_text = f"{launch['physical_index']} (allowed: {selected_ids})"
        selection_rows.append(
            [
                seed_run["seed"],
                selection_text,
                f"{launch['utilization_percent_at_selection']}% "
                f"(< {policy['max_utilization_percent']}%); "
                f"{launch['memory_free_mib_at_selection']:,} MiB free "
                f"(minimum {policy['min_free_memory_mib']:,})",
                f"{launch['compute_process_count_at_selection']}: {process_details}",
            ]
        )
    story.extend(
        [
            table(artifact_rows, [14 * mm, 61 * mm, 61 * mm, 38 * mm]),
            Paragraph("Idle-GPU final launch probes", styles["h2"]),
            table(selection_rows, [13 * mm, 40 * mm, 56 * mm, 65 * mm]),
            Paragraph("Validation performed before aggregation", styles["h2"]),
            Paragraph(
                "Exactly seeds 0/1/2; completed summary markers; top-level/nested seed "
                "and sample-count agreement; 1,000 ordered CSV rows per seed; exact CSV "
                "schema; raw-file SHA-256; consistent checkpoint hash/size/global step; "
                "backend-specific sampling/generation protocol; distinct ordered raw outputs; "
                "complete RNG seeding; physical-GPU selection provenance and UUID to "
                "logical cuda:0 mapping; recorded GPU-selection method plus final launch "
                "probes satisfying the exclusive utilization and minimum-free-memory policy, "
                "with process inventories preserved; cross-seed dependency, tokenizer, "
                "implementation, pinned SA artifact, TDC metric-source, "
                "checkpoint, effective-config, and runner identity; recomputed "
                "strict/released validity, first-occurrence uniqueness, quality gates, "
                "repair recovery, largest-component flags, failure counts, ratios, and "
                "timing invariants.",
                styles["body"],
            ),
            Paragraph("Caveats", styles["h2"]),
        ]
    )
    for number, caveat in enumerate(payload["caveats"], start=1):
        story.append(Paragraph(f"{number}. {escape(caveat)}", styles["body"]))
    story.extend(
        [
            Spacer(1, 5 * mm),
            callout(
                "BOTTOM LINE. This document faithfully describes the completed local "
                "benchmark and its distance from the published headline values. It does "
                "not assert that the released checkpoint, original seeds, training data "
                "revision, hardware, or full environment were reproduced exactly."
            ),
            Spacer(1, 5 * mm),
            Paragraph(
                f"Generated at {escape(payload['generated_at_utc'])}. Machine-readable "
                "JSON and CSV are emitted in the caller-selected aggregate directory.",
                styles["small"],
            ),
        ]
    )

    def page_decor(canvas: Any, document: Any) -> None:
        canvas.saveState()
        width, height = A4
        canvas.setStrokeColor(palette["line"])
        canvas.line(18 * mm, 14 * mm, width - 18 * mm, 14 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(palette["muted"])
        canvas.drawString(
            18 * mm,
            9 * mm,
            f"GenMol {checkpoint['diffusion_type'].upper()} de novo benchmark | auditable comparison",
        )
        canvas.drawRightString(width - 18 * mm, 9 * mm, f"Page {document.page}")
        canvas.restoreState()

    try:
        document = SimpleDocTemplate(
            str(temporary_path),
            pagesize=A4,
            rightMargin=18 * mm,
            leftMargin=18 * mm,
            topMargin=16 * mm,
            bottomMargin=19 * mm,
            title=(
                "GenMol from-scratch "
                f"{checkpoint['diffusion_type'].upper()} de novo benchmark"
            ),
            author="GenMol v2 benchmark workflow",
            subject="Three-seed comparison with published GenMol V1 Table 1",
        )
        document.build(story, onFirstPage=page_decor, onLaterPages=page_decor)
        validate_pdf(
            temporary_path,
            expected_checkpoint_sha256=payload["checkpoint"]["sha256"],
        )
        os.replace(temporary_path, pdf_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def validate_pdf(
    path: Path,
    *,
    expected_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF validation requires pypdf from requirements-stage0.lock"
        ) from exc
    reader = PdfReader(str(path))
    if len(reader.pages) < 3:
        raise RuntimeError(f"report PDF unexpectedly has only {len(reader.pages)} page(s)")
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    required_fragments = (
        "GenMol from-scratch benchmark",
        "Headline comparison",
        "Strict vs repaired funnel",
        "Checkpoint and protocol provenance",
        "Softmax temperature",
        "Evaluation seeds",
        "Idle-GPU final launch probes",
        "model_sampling_and_tokenizer + released_postprocessing",
        "NOT AN EXACT REPRODUCTION CLAIM",
        PAPER_PRIMARY_URL,
    )
    if expected_checkpoint_sha256 is not None:
        required_fragments = (*required_fragments, expected_checkpoint_sha256)
    missing = [fragment for fragment in required_fragments if fragment not in text]
    if missing:
        raise RuntimeError(f"report PDF text validation failed; missing {missing}")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
        "page_count": len(reader.pages),
        "validation": "pypdf opened every page and found all required report sections",
    }


def write_report_bundle(
    payload: dict[str, Any],
    *,
    output_dir: Path,
    pdf_path: Path,
    overwrite: bool = False,
) -> dict[str, Path]:
    output_dir = _resolve_in_repository(output_dir)
    pdf_path = _resolve_in_repository(pdf_path)
    json_path = output_dir / AGGREGATE_JSON_FILENAME
    csv_path = output_dir / AGGREGATE_CSV_FILENAME
    targets = (json_path, csv_path, pdf_path)
    duplicate_targets = len({path.resolve() for path in targets}) != len(targets)
    if duplicate_targets:
        raise ReportValidationError("JSON, CSV, and PDF output paths must be distinct")
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite report artifact(s): "
            + ", ".join(str(path) for path in existing)
            + "; pass --overwrite only after reviewing the existing files"
        )

    render_pdf(payload, pdf_path)
    pdf_metadata = validate_pdf(
        pdf_path,
        expected_checkpoint_sha256=payload["checkpoint"]["sha256"],
    )
    payload["artifacts"] = {
        "report_pdf": pdf_metadata,
        "aggregate_json": {"path": str(json_path)},
        "aggregate_csv": {"path": str(csv_path)},
    }
    rows = aggregate_csv_rows(payload)
    _write_csv(csv_path, rows)
    payload["artifacts"]["aggregate_csv"].update(
        {
            "sha256": _sha256_file(csv_path),
            "row_count": len(rows),
            "fields": list(CSV_FIELDS),
        }
    )
    _write_json(json_path, payload)

    # Re-open final artifacts rather than trusting the writers.
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    if loaded.get("status") != "completed" or loaded.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise RuntimeError("aggregate JSON failed post-write validation")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        csv_reader = csv.DictReader(handle)
        final_rows = list(csv_reader)
    if tuple(csv_reader.fieldnames or ()) != CSV_FIELDS or len(final_rows) != len(rows):
        raise RuntimeError("aggregate CSV failed post-write validation")
    validate_pdf(
        pdf_path,
        expected_checkpoint_sha256=payload["checkpoint"]["sha256"],
    )
    return {"json": json_path, "csv": csv_path, "pdf": pdf_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        required=True,
        help="Tree containing exactly the seed 0, 1, and 2 completed run directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for aggregate.json and aggregate.csv.",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        required=True,
        help="Explicit destination for the dedicated benchmark PDF.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace all three report artifacts after input validation.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = collect_report(args.runs_dir)
    outputs = write_report_bundle(
        payload,
        output_dir=args.output_dir,
        pdf_path=args.pdf,
        overwrite=args.overwrite,
    )
    print(f"Validated seeds: {', '.join(str(seed) for seed in EXPECTED_SEEDS)}")
    print(f"Aggregate JSON: {outputs['json']}")
    print(f"Aggregate CSV:  {outputs['csv']}")
    print(f"Report PDF:    {outputs['pdf']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

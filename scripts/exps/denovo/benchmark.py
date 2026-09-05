# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproducible, auditable de novo benchmark for one checkpoint and one seed.

This runner intentionally generates the entire requested sample count in one
batch, just like the released ``scripts/exps/denovo/run.py``.  Splitting a run
into smaller batches changes both Python and Torch random-number consumption and
would therefore no longer be the released sampling procedure.

The released metric path repairs invalid SAFE fragments before decoding and
then retains the largest disconnected SMILES component.  We report those
paper-comparable metrics unchanged, while also retaining the model text, the
corresponding SAFE string, and a strict ``safe.decode(..., fix=False)`` result
for every requested sample.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import pickle
import platform
import random
import stat
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
REPO_SRC = REPO_ROOT / "src"
for import_root in (REPO_ROOT, REPO_SRC):
    while str(import_root) in sys.path:
        sys.path.remove(str(import_root))
    sys.path.insert(0, str(import_root))

SCHEMA_VERSION = 4
TOKENIZER_REQUESTED_IDENTIFIER = "datamol-io/safe-gpt"
RAW_SAMPLES_FILENAME = "raw_samples.csv"
SUMMARY_FILENAME = "summary.json"
LOCK_FILENAME = ".benchmark.lock"

METRIC_INPUT_SCHEMA_VERSION = 1
SA_FRAGMENT_SCORES_RELATIVE_PATH = Path("oracle/fpscores.pkl")
SA_FRAGMENT_SCORES_SHA256 = (
    "24a4392f5c673e79c0446af3c4d8e458293b5fecaa244328e76741ead9d21dbf"
)
SA_FRAGMENT_SCORES_SIZE_BYTES = 9_048_931
SA_FRAGMENT_SCORE_ROW_COUNT = 3_549
SA_FINGERPRINT_SCORE_COUNT = 705_292
TDC_METRIC_IMPLEMENTATION_PATHS = {
    "oracle_dispatch": Path("tdc/oracles.py"),
    "sa_qed_scoring": Path("tdc/chem_utils/oracle/oracle.py"),
    "evaluator_dispatch": Path("tdc/evaluator.py"),
    "diversity_scoring": Path("tdc/chem_utils/evaluator.py"),
}
TDC_METRIC_DISTRIBUTION_VERSION = "0.4.1"
TDC_METRIC_IMPLEMENTATION_SHA256 = {
    "oracle_dispatch": "03b52abdc8a1446f903238009fd9682842e04479eac8e989ed2395147938de2b",
    "sa_qed_scoring": "d266c89b5ea5f67135d0fa04f3348c4e67e3b8946c4ffe13ee8d6b96a5335e4f",
    "evaluator_dispatch": "3531d60f2b128417429f2e510d994c1c72a2e441124ea43505224f4120819549",
    "diversity_scoring": "eb61d9c258be6ad1a8a2297395f6519ff89d7651013fc8a2c72831e572d574e3",
}
TDC_METRIC_IMPLEMENTATION_SIZE_BYTES = {
    "oracle_dispatch": 25_879,
    "sa_qed_scoring": 59_584,
    "evaluator_dispatch": 15_901,
    "diversity_scoring": 13_620,
}

LAUNCH_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "GENMOL_BENCHMARK_GPU_PHYSICAL_INDEX",
    "GENMOL_BENCHMARK_GPU_UUID",
    "GENMOL_BENCHMARK_GPU_SELECTION_SNAPSHOT",
    "GENMOL_BENCHMARK_RUN_LABEL",
)

IMPLEMENTATION_INPUT_PATHS = {
    "sampler_source": REPO_ROOT / "src/genmol/sampler.py",
    "model_source": REPO_ROOT / "src/genmol/model.py",
    "diffusion_source": REPO_ROOT / "src/genmol/diffusion.py",
    "backbone_source": REPO_ROOT / "src/genmol/backbone.py",
    "chemistry_utils_source": REPO_ROOT / "src/genmol/utils/utils_chem.py",
    "data_utils_source": REPO_ROOT / "src/genmol/utils/utils_data.py",
    "bracket_safe_converter_source": (
        REPO_ROOT / "src/genmol/utils/bracket_safe_converter.py"
    ),
    "length_distribution": REPO_ROOT / "data/len.pk",
}

RAW_SAMPLE_FIELDS = (
    "sample_index",
    "raw_model_text",
    "raw_safe",
    "raw_safe_error",
    "strict_smiles",
    "strict_decode_error",
    "strict_qed",
    "strict_sa",
    "strict_is_first_unique",
    "strict_quality_pass",
    "strict_quality_counted",
    "released_repaired_smiles",
    "released_smiles",
    "released_decode_error",
    "released_qed",
    "released_sa",
    "released_is_first_unique",
    "released_quality_pass",
    "released_quality_counted",
    "released_was_recovered",
    "released_largest_component_applied",
)


class BenchmarkConfigurationError(ValueError):
    """Raised before model loading when a requested run is not well formed."""


@dataclass(frozen=True)
class PinnedSAMetricInput:
    """Verified, resident SA fragment scores and their immutable provenance."""

    provenance: Mapping[str, Any]
    fragment_scores: Mapping[int, float]


def benchmark_run_label(global_step: int, checkpoint_sha256: str, seed: int) -> str:
    """Return the shared launcher/report identity label for one seed."""
    return f"denovo_step{int(global_step)}_{checkpoint_sha256[:12]}_seed{int(seed)}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_pinned_regular_file(
    *,
    repository_root: Path,
    relative_path: Path,
    expected_sha256: str,
    expected_size_bytes: int,
) -> tuple[Path, bytes]:
    """Read one exact in-repository regular file through a stable descriptor.

    The resolved path, inode, metadata, byte count, and digest all have to agree.
    ``O_NOFOLLOW`` closes the final-component symlink race on platforms that
    provide it; the canonical-path and post-read inode checks also reject a
    symlinked parent or a path replacement during the read.
    """

    if not isinstance(relative_path, Path) or relative_path.is_absolute():
        raise ValueError("pinned metric-input path must be repository-relative")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("expected metric-input SHA-256 must be 64 lowercase hex digits")
    if type(expected_size_bytes) is not int or expected_size_bytes <= 0:
        raise ValueError("expected metric-input size must be a positive integer")

    try:
        root = repository_root.resolve(strict=True)
    except OSError as error:
        raise FileNotFoundError(
            f"Benchmark repository root is unavailable: {repository_root}"
        ) from error
    candidate = root / relative_path
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"metric-input path escapes repository root: {relative_path}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise FileNotFoundError(
            "Pinned TDC SA fragment-score artifact is missing; refusing TDC's "
            f"implicit downloader: {candidate}"
        ) from error
    if resolved != candidate:
        raise RuntimeError(
            "Pinned TDC SA fragment-score artifact must not traverse a symlink: "
            f"{candidate} resolves to {resolved}"
        )
    try:
        path_before = candidate.stat(follow_symlinks=False)
    except OSError as error:
        raise RuntimeError(
            f"Pinned metric-input path changed before it was opened: {candidate}"
        ) from error
    if not stat.S_ISREG(path_before.st_mode):
        raise RuntimeError(f"Pinned metric input is not a regular file: {candidate}")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as error:
        raise RuntimeError(
            f"Could not securely open pinned metric input {candidate}: {error}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"Pinned metric input is not a regular file: {candidate}")
        if (
            path_before.st_dev != before.st_dev
            or path_before.st_ino != before.st_ino
        ):
            raise RuntimeError(
                f"Pinned metric-input path was replaced before open: {candidate}"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise RuntimeError(f"Pinned metric input changed while being read: {candidate}")
    try:
        path_after = candidate.stat(follow_symlinks=False)
    except OSError as error:
        raise RuntimeError(
            f"Pinned metric-input path changed after it was read: {candidate}"
        ) from error
    if (
        not stat.S_ISREG(path_after.st_mode)
        or path_after.st_dev != after.st_dev
        or path_after.st_ino != after.st_ino
    ):
        raise RuntimeError(
            f"Pinned metric-input path was replaced while being read: {candidate}"
        )

    payload = b"".join(chunks)
    actual_size = len(payload)
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_size != expected_size_bytes:
        raise RuntimeError(
            "Pinned TDC SA fragment-score artifact has the wrong size: "
            f"{actual_size} bytes != {expected_size_bytes} bytes"
        )
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "Pinned TDC SA fragment-score artifact has the wrong SHA-256: "
            f"{actual_sha256} != {expected_sha256}"
        )
    return candidate, payload


def _decode_sa_fragment_scores(
    payload: bytes,
    *,
    expected_row_count: int,
    expected_fingerprint_count: int,
) -> dict[int, float]:
    """Decode the pinned pickle using TDC's row-to-fingerprint semantics."""

    try:
        rows = pickle.loads(payload)
    except Exception as error:
        raise RuntimeError("Pinned TDC SA fragment scores are not a valid pickle") from error
    if type(rows) is not list or len(rows) != expected_row_count:
        raise RuntimeError(
            "Pinned TDC SA fragment-score row count is invalid: "
            f"{len(rows) if isinstance(rows, list) else type(rows).__name__} "
            f"!= {expected_row_count}"
        )

    fragment_scores: dict[int, float] = {}
    for row_index, row in enumerate(rows):
        if type(row) is not list or len(row) < 2:
            raise RuntimeError(
                f"Pinned TDC SA fragment-score row {row_index} is malformed"
            )
        score_value = row[0]
        if isinstance(score_value, bool) or not isinstance(score_value, (int, float)):
            raise RuntimeError(
                f"Pinned TDC SA fragment-score row {row_index} has a nonnumeric score"
            )
        score = float(score_value)
        if not math.isfinite(score):
            raise RuntimeError(
                f"Pinned TDC SA fragment-score row {row_index} has a nonfinite score"
            )
        for fingerprint in row[1:]:
            if isinstance(fingerprint, bool) or not isinstance(fingerprint, int):
                raise RuntimeError(
                    f"Pinned TDC SA fragment-score row {row_index} has a noninteger key"
                )
            if fingerprint in fragment_scores:
                raise RuntimeError(
                    "Pinned TDC SA fragment scores contain a duplicate fingerprint key"
                )
            fragment_scores[fingerprint] = score
    if len(fragment_scores) != expected_fingerprint_count:
        raise RuntimeError(
            "Pinned TDC SA fragment-score fingerprint count is invalid: "
            f"{len(fragment_scores)} != {expected_fingerprint_count}"
        )
    return fragment_scores


def _tdc_metric_implementation_provenance() -> dict[str, Any]:
    """Fingerprint the installed TDC files that define reported metrics."""

    try:
        distribution = importlib.metadata.distribution("PyTDC")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("The pinned benchmark requires the PyTDC distribution") from error
    if distribution.version != TDC_METRIC_DISTRIBUTION_VERSION:
        raise RuntimeError(
            "PyTDC version does not match the audited metric backend: "
            f"{distribution.version} != {TDC_METRIC_DISTRIBUTION_VERSION}"
        )
    files: dict[str, Any] = {}
    for name, relative_path in TDC_METRIC_IMPLEMENTATION_PATHS.items():
        path = Path(distribution.locate_file(relative_path)).resolve()
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Required TDC metric implementation is not regular: {path}")
        digest = _sha256(path)
        size_bytes = path.stat().st_size
        if digest != TDC_METRIC_IMPLEMENTATION_SHA256[name]:
            raise RuntimeError(
                f"TDC metric implementation {name} has an unaudited SHA-256: {digest}"
            )
        if size_bytes != TDC_METRIC_IMPLEMENTATION_SIZE_BYTES[name]:
            raise RuntimeError(
                f"TDC metric implementation {name} has an unaudited size: {size_bytes}"
            )
        files[name] = {
            "path": str(path),
            "sha256": digest,
            "size_bytes": size_bytes,
        }
    return {
        "distribution": "PyTDC",
        "version": TDC_METRIC_DISTRIBUTION_VERSION,
        "implementation_files": files,
    }


def _load_pinned_sa_metric_input(
    *,
    repository_root: Path,
    relative_path: Path,
    expected_sha256: str,
    expected_size_bytes: int,
    expected_row_count: int,
    expected_fingerprint_count: int,
    include_tdc_provenance: bool = True,
) -> PinnedSAMetricInput:
    """Return verified resident scores without calling TDC's network loader."""

    path, payload = _read_pinned_regular_file(
        repository_root=repository_root,
        relative_path=relative_path,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
    )
    fragment_scores = _decode_sa_fragment_scores(
        payload,
        expected_row_count=expected_row_count,
        expected_fingerprint_count=expected_fingerprint_count,
    )
    tdc_provenance = (
        _tdc_metric_implementation_provenance()
        if include_tdc_provenance
        else {
            "distribution": "PyTDC",
            "version": "test-fixture",
            "implementation_files": {},
        }
    )
    provenance = {
        "schema_version": METRIC_INPUT_SCHEMA_VERSION,
        "sa_fragment_scores": {
            "path": str(path),
            "relative_path": relative_path.as_posix(),
            "sha256": expected_sha256,
            "size_bytes": expected_size_bytes,
            "serialization": "python_pickle_verified_before_deserialization",
            "top_level_row_count": expected_row_count,
            "fingerprint_score_count": expected_fingerprint_count,
            "duplicate_fingerprint_count": 0,
        },
        "tdc_metric_implementation": tdc_provenance,
        "sa_loading_policy": {
            "requested_oracle": "sa",
            "oracle_class": "tdc.oracles.Oracle",
            "sa_callable": "tdc.chem_utils.oracle.oracle.SA",
            "network_download_allowed": False,
            "tdc_oracle_load_invoked": False,
            "resident_scores_loaded_from_verified_bytes": True,
            "artifact_mutation_after_resident_load_affects_current_run": False,
        },
        "affected_outputs": [
            "raw_samples_csv.strict_sa",
            "raw_samples_csv.released_sa",
            "metrics.strict.quality",
            "metrics.released_comparable.quality",
        ],
    }
    return PinnedSAMetricInput(
        provenance=provenance,
        fragment_scores=fragment_scores,
    )


def load_pinned_sa_metric_input() -> PinnedSAMetricInput:
    """Load the only SA artifact authorized for benchmark quality metrics."""

    return _load_pinned_sa_metric_input(
        repository_root=REPO_ROOT,
        relative_path=SA_FRAGMENT_SCORES_RELATIVE_PATH,
        expected_sha256=SA_FRAGMENT_SCORES_SHA256,
        expected_size_bytes=SA_FRAGMENT_SCORES_SIZE_BYTES,
        expected_row_count=SA_FRAGMENT_SCORE_ROW_COUNT,
        expected_fingerprint_count=SA_FINGERPRINT_SCORE_COUNT,
    )


def metric_input_provenance() -> dict[str, Any]:
    """Validate and fingerprint every external input to reported metrics."""

    return dict(load_pinned_sa_metric_input().provenance)


def _json_compatible_number(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    result = float(value)
    return result if math.isfinite(result) else None


def _normalise_scores(values: Any, expected_length: int, name: str) -> list[float]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if expected_length == 1 and not isinstance(values, (list, tuple)):
        values = [values]
    values = list(values)
    if len(values) != expected_length:
        raise RuntimeError(
            f"{name} returned {len(values)} values for {expected_length} molecules"
        )
    scores: list[float] = []
    for value in values:
        score = _json_compatible_number(value)
        if score is None:
            raise RuntimeError(f"{name} returned a non-finite score")
        scores.append(score)
    return scores


def _error_text(exc: BaseException) -> str:
    message = str(exc).replace("\n", " ").strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def seed_sampling(seed: int, device: str) -> dict[str, Any]:
    """Seed RNGs immediately before sampling, isolating model-load RNG use."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    requested_device = torch.device(device)
    if requested_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return {
        "seed": seed,
        "seed_applied_immediately_before_generation": True,
        "python_random": True,
        "numpy": True,
        "torch_cpu": True,
        "torch_cuda_all": requested_device.type == "cuda",
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
    }


def synchronize_device(device: Any) -> None:
    """Make CUDA timing boundaries measure completed sampler work only."""
    import torch

    parsed = torch.device(device)
    if parsed.type == "cuda":
        torch.cuda.synchronize(parsed)


def _sampler_udlm_inference_eps(sampler: Any) -> float:
    """Return the inference endpoint stored in the loaded UDLM checkpoint."""

    training = sampler.model.config.training
    udlm_config = training.get("udlm", {})
    inference_eps = float(udlm_config.get("inference_eps", 1e-5))
    if not 0 < inference_eps < 1:
        raise BenchmarkConfigurationError(
            "Loaded UDLM checkpoint has inference_eps outside (0, 1)"
        )
    return inference_eps


def generate_raw_model_text(
    sampler: Any,
    num_samples: int,
    *,
    diffusion_type: str,
    softmax_temp: float,
    randomness: float,
    min_add_len: int,
    num_steps: int | None,
    inference_eps: float | None,
    exclude_special_tokens: bool | None,
) -> tuple[list[str], dict[str, Any]]:
    """Run either diffusion backend through the shared raw-token sampler API.

    ``Sampler.generate(..., return_token_ids=True)`` is the single source of
    denoising semantics.  This helper only constructs the released de-novo
    template and stops before chemical decoding so failed rows remain auditable.
    """
    import torch

    loaded_diffusion_type = str(
        getattr(sampler, "diffusion_type", "mdlm")
    ).lower()
    if loaded_diffusion_type != diffusion_type:
        raise BenchmarkConfigurationError(
            "Inference config requests diffusion_type="
            f"{diffusion_type!r}, but checkpoint loaded as {loaded_diffusion_type!r}"
        )

    # This is the body of Sampler.de_novo_generation up to the raw token
    # boundary. Do not split a final run into smaller batches.
    with torch.no_grad():
        x = torch.hstack(
            [
                torch.full((1, 1), sampler.model.bos_index),
                torch.full((1, 1), sampler.model.eos_index),
            ]
        )
        x = sampler._insert_mask(x, num_samples, min_add_len=min_add_len)
        x = x.to(sampler.model.device)

        if diffusion_type == "udlm":
            assert num_steps is not None
            loaded_inference_eps = _sampler_udlm_inference_eps(sampler)
            if not math.isclose(
                loaded_inference_eps,
                float(inference_eps),
                rel_tol=0,
                abs_tol=1e-12,
            ):
                raise BenchmarkConfigurationError(
                    "Inference config inference_eps does not match the loaded UDLM "
                    f"checkpoint ({inference_eps} != {loaded_inference_eps})"
                )
            loaded_exclusion = bool(
                sampler.model.config.training.get("udlm", {}).get(
                    "exclude_special_tokens", False
                )
            )
            if loaded_exclusion is not exclude_special_tokens:
                raise BenchmarkConfigurationError(
                    "Inference config exclude_special_tokens does not match the "
                    f"loaded UDLM checkpoint ({exclude_special_tokens} != "
                    f"{loaded_exclusion})"
                )
            nfe = num_steps
            num_steps_source = "explicit UDLM reverse-transition count"
        else:
            nfe = max(int(sampler.mdlm.get_num_steps_confidence(x)), 2)
            num_steps_source = (
                "MDLM.get_num_steps_confidence on the single padded generation batch"
            )

        token_ids = sampler.generate(
            x,
            softmax_temp=softmax_temp,
            randomness=randomness,
            num_steps=num_steps,
            return_token_ids=True,
        )
        decoded = sampler.model.tokenizer.batch_decode(
            token_ids, skip_special_tokens=True
        )

    if len(decoded) != num_samples:
        raise RuntimeError(
            f"Tokenizer returned {len(decoded)} rows for {num_samples} requested samples"
        )
    return [str(value) for value in decoded], {
        "diffusion_type": diffusion_type,
        "nfe": nfe,
        "nfe_definition": "one full backbone forward evaluation per reverse step",
        "num_steps": num_steps,
        "num_steps_source": num_steps_source,
        "inference_eps": inference_eps,
        "exclude_special_tokens": exclude_special_tokens,
        "temperature": softmax_temp,
        "randomness": randomness,
        "randomness_used_by_sampler": diffusion_type == "mdlm",
    }


def _canonicalize_chemically_valid_smiles(decoded: str) -> str:
    from rdkit import Chem

    # SAFE decoding normally returns a sanitized canonical SMILES already.  An
    # explicit RDKit round trip makes chemical validity an enforced invariant,
    # rather than assuming every non-None decoder string is chemically valid.
    molecule = Chem.MolFromSmiles(decoded, sanitize=True)
    if molecule is None:
        raise ValueError("RDKit rejected the strict decoded SMILES")
    Chem.SanitizeMol(molecule)
    return Chem.MolToSmiles(molecule, canonical=True)


def _default_strict_decoder(safe_text: str) -> str | None:
    import safe as sf

    # This exact call defines strict SAFE decoding for this benchmark.
    decoded = sf.decode(
        safe_text,
        canonical=True,
        ignore_errors=True,
        fix=False,
    )
    if decoded is None:
        return None
    return _canonicalize_chemically_valid_smiles(decoded)


def _default_released_decoder(safe_text: str) -> str | None:
    from genmol.utils.utils_chem import safe_to_smiles

    return safe_to_smiles(safe_text, fix=True)


def _default_bracket_converter(model_text: str) -> str:
    from genmol.utils.bracket_safe_converter import bracketsafe2safe

    return bracketsafe2safe(model_text)


def decode_records(
    raw_model_texts: Sequence[str],
    *,
    use_bracket_safe: bool,
    strict_decoder: Callable[[str], str | None] | None = None,
    released_decoder: Callable[[str], str | None] | None = None,
    bracket_converter: Callable[[str], str] | None = None,
    timing: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Retain strict and released post-processing for every generated row.

    When ``timing`` is supplied, ``released_postprocessing`` measures the part
    of the released ``Sampler.generate`` call that follows tokenizer decoding:
    Bracket-SAFE conversion when applicable, ``safe_to_smiles(..., fix=True)``,
    failed-row removal, and largest-component selection.  The strict diagnostic
    path is deliberately executed after that boundary so it cannot inflate the
    paper-comparable generation time.
    """
    strict_decoder = strict_decoder or _default_strict_decoder
    released_decoder = released_decoder or _default_released_decoder
    bracket_converter = bracket_converter or _default_bracket_converter

    records: list[dict[str, Any]] = []
    for sample_index, raw_model_text in enumerate(raw_model_texts):
        record: dict[str, Any] = {
            field: None for field in RAW_SAMPLE_FIELDS
        }
        record.update(
            {
                "sample_index": sample_index,
                "raw_model_text": raw_model_text,
                "strict_is_first_unique": False,
                "strict_quality_counted": False,
                "released_is_first_unique": False,
                "released_quality_counted": False,
                "released_was_recovered": False,
                "released_largest_component_applied": False,
            }
        )

        records.append(record)

    # Keep the released path contiguous and ahead of strict diagnostics.  This
    # gives the benchmark the same timing endpoint as released run.py, whose
    # timer wraps Sampler.de_novo_generation (including repair and component
    # filtering), while retaining every requested row for auditability.
    released_postprocessing_start = time.perf_counter()
    for record in records:
        raw_model_text = record["raw_model_text"]
        try:
            raw_safe = (
                bracket_converter(raw_model_text)
                if use_bracket_safe
                else raw_model_text
            )
            if not isinstance(raw_safe, str):
                raise TypeError("SAFE conversion did not return a string")
            record["raw_safe"] = raw_safe
        except Exception as exc:  # preserve the row and account for the failure
            record["raw_safe_error"] = _error_text(exc)
            record["strict_decode_error"] = "SAFE conversion failed"
            record["released_decode_error"] = "SAFE conversion failed"
            continue

    for record in records:
        raw_safe = record["raw_safe"]
        if raw_safe is None:
            continue
        try:
            repaired_smiles = released_decoder(raw_safe)
            if repaired_smiles:
                repaired_smiles = str(repaired_smiles)
                record["released_repaired_smiles"] = repaired_smiles
            else:
                record["released_decode_error"] = "decode_returned_none"
        except Exception as exc:
            record["released_decode_error"] = _error_text(exc)

    for record in records:
        repaired_smiles = record["released_repaired_smiles"]
        if repaired_smiles is None:
            continue
        # Match released Sampler.generate exactly: split on '.', sort by
        # string length, and retain the final (largest) component.
        released_smiles = sorted(repaired_smiles.split("."), key=len)[-1]
        record["released_smiles"] = released_smiles
        record["released_largest_component_applied"] = (
            released_smiles != repaired_smiles
        )
    released_postprocessing_seconds = (
        time.perf_counter() - released_postprocessing_start
    )

    # Strict decoding is a diagnostic addition and is outside the released
    # generation timer by construction.
    for record in records:
        raw_safe = record["raw_safe"]
        if raw_safe is None:
            continue
        try:
            strict_smiles = strict_decoder(raw_safe)
            if strict_smiles:
                record["strict_smiles"] = str(strict_smiles)
            else:
                record["strict_decode_error"] = "decode_returned_none"
        except Exception as exc:
            record["strict_decode_error"] = _error_text(exc)

    for record in records:
        record["released_was_recovered"] = bool(
            record["strict_smiles"] is None and record["released_smiles"] is not None
        )

    if timing is not None:
        timing["released_postprocessing"] = released_postprocessing_seconds

    return records


def _first_unique_indices(records: Sequence[Mapping[str, Any]], smiles_key: str) -> list[int]:
    seen: set[str] = set()
    indices: list[int] = []
    for index, record in enumerate(records):
        smiles = record[smiles_key]
        if smiles is not None and smiles not in seen:
            seen.add(smiles)
            indices.append(index)
    return indices


def _evaluate_metric_branch(
    records: list[dict[str, Any]],
    *,
    prefix: str,
    requested_count: int,
    oracle_qed: Callable[[Sequence[str]], Any],
    oracle_sa: Callable[[Sequence[str]], Any],
    diversity_evaluator: Callable[[Sequence[str]], Any],
) -> dict[str, Any]:
    smiles_key = f"{prefix}_smiles"
    valid_indices = [
        index for index, record in enumerate(records) if record[smiles_key] is not None
    ]
    valid_smiles = [records[index][smiles_key] for index in valid_indices]

    if valid_smiles:
        qed_scores = _normalise_scores(oracle_qed(valid_smiles), len(valid_smiles), "QED")
        sa_scores = _normalise_scores(oracle_sa(valid_smiles), len(valid_smiles), "SA")
        for index, qed, sa in zip(valid_indices, qed_scores, sa_scores):
            records[index][f"{prefix}_qed"] = qed
            records[index][f"{prefix}_sa"] = sa
            records[index][f"{prefix}_quality_pass"] = bool(qed >= 0.6 and sa <= 4.0)

    unique_indices = _first_unique_indices(records, smiles_key)
    for index in unique_indices:
        records[index][f"{prefix}_is_first_unique"] = True
        records[index][f"{prefix}_quality_counted"] = bool(
            records[index][f"{prefix}_quality_pass"]
        )
    unique_smiles = [records[index][smiles_key] for index in unique_indices]
    quality_count = sum(
        bool(records[index][f"{prefix}_quality_counted"])
        for index in unique_indices
    )

    if unique_smiles:
        diversity = _json_compatible_number(diversity_evaluator(unique_smiles))
        diversity_undefined_reason = (
            None if diversity is not None else "evaluator_returned_non_finite"
        )
    else:
        diversity = None
        diversity_undefined_reason = "no_unique_valid_molecules"

    valid_count = len(valid_indices)
    unique_count = len(unique_indices)
    return {
        "validity": valid_count / requested_count,
        "valid_count": valid_count,
        "validity_denominator": requested_count,
        "uniqueness": unique_count / valid_count if valid_count else None,
        "unique_count": unique_count,
        "uniqueness_denominator": valid_count,
        "diversity": diversity,
        "diversity_input_count": unique_count,
        "diversity_undefined_reason": diversity_undefined_reason,
        "quality": quality_count / requested_count,
        "quality_count": quality_count,
        "quality_denominator": requested_count,
        "quality_thresholds": {"qed_min_inclusive": 0.6, "sa_max_inclusive": 4.0},
    }


def evaluate_records(
    records: list[dict[str, Any]],
    *,
    requested_count: int,
    oracle_qed: Callable[[Sequence[str]], Any],
    oracle_sa: Callable[[Sequence[str]], Any],
    diversity_evaluator: Callable[[Sequence[str]], Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Calculate released-comparable and strict metrics with named denominators."""
    if len(records) != requested_count:
        raise ValueError(
            f"Have {len(records)} decoded records for {requested_count} requested samples"
        )

    released_metrics = _evaluate_metric_branch(
        records,
        prefix="released",
        requested_count=requested_count,
        oracle_qed=oracle_qed,
        oracle_sa=oracle_sa,
        diversity_evaluator=diversity_evaluator,
    )
    strict_metrics = _evaluate_metric_branch(
        records,
        prefix="strict",
        requested_count=requested_count,
        oracle_qed=oracle_qed,
        oracle_sa=oracle_sa,
        diversity_evaluator=diversity_evaluator,
    )

    metrics = {
        "released_comparable": {
            **released_metrics,
            "definition": (
                "Released GenMol path: SAFE fragment repair with fix=True, canonical "
                "decode, largest disconnected component, deduplicate before diversity "
                "and quality; validity and quality divide by requested samples, while "
                "uniqueness divides by released-valid samples."
            ),
        },
        "strict": {
            **strict_metrics,
            "definition": (
                "Direct canonical sf.decode(raw_safe, fix=False, "
                "ignore_errors=True), followed by explicit RDKit parsing, sanitization, "
                "and canonicalization, without fragment repair or largest-component "
                "selection; other metric denominators mirror the released path."
            ),
        },
    }
    failure_counts = {
        "raw_safe_conversion_failed": sum(
            record["raw_safe_error"] is not None for record in records
        ),
        "strict_decode_failed": requested_count - strict_metrics["valid_count"],
        "released_decode_failed": requested_count - released_metrics["valid_count"],
        "released_recovered_strict_failure": sum(
            bool(record["released_was_recovered"]) for record in records
        ),
        "strict_valid_but_released_failed": sum(
            record["strict_smiles"] is not None and record["released_smiles"] is None
            for record in records
        ),
        "released_largest_component_applied": sum(
            bool(record["released_largest_component_applied"]) for record in records
        ),
        "strict_duplicates": strict_metrics["valid_count"] - strict_metrics["unique_count"],
        "released_duplicates": (
            released_metrics["valid_count"] - released_metrics["unique_count"]
        ),
    }
    return metrics, failure_counts


def _atomic_write(path: Path, writer: Callable[[Any], None], *, newline: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline=newline) as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        try:
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError:
            # Directory fsync is unavailable on some filesystems; the file was
            # still flushed and atomically renamed on the local filesystem.
            pass
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    def write(handle: Any) -> None:
        csv_writer = csv.DictWriter(handle, fieldnames=RAW_SAMPLE_FIELDS, extrasaction="raise")
        csv_writer.writeheader()
        csv_writer.writerows(records)

    _atomic_write(path, write, newline="")


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    def write(handle: Any) -> None:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")

    _atomic_write(path, write)


@contextmanager
def output_lock(output_dir: Path) -> Iterable[None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / LOCK_FILENAME
    try:
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Output directory is locked: {lock_path}. Verify no benchmark is running "
            "before removing a stale lock."
        ) from exc
    try:
        lock_payload = json.dumps({"pid": os.getpid(), "created_at_utc": _utc_now()}) + "\n"
        os.write(descriptor, lock_payload.encode("utf-8"))
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def validate_output_target(output_dir: Path, *, overwrite: bool) -> None:
    existing = [
        path
        for path in (
            output_dir / RAW_SAMPLES_FILENAME,
            output_dir / SUMMARY_FILENAME,
        )
        if path.exists()
    ]
    if existing and not overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing benchmark artifact(s): {paths}. "
            "Choose a fresh output directory or pass --overwrite explicitly."
        )


def load_yaml_config(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise BenchmarkConfigurationError("Config must contain a YAML mapping")
    try:
        _canonical_json_sha256(config)
    except (TypeError, ValueError) as exc:
        raise BenchmarkConfigurationError(
            "Config values must be JSON-serializable for provenance recording"
        ) from exc
    return config


def validate_sampling_config(config: Mapping[str, Any]) -> dict[str, Any]:
    missing = [
        key for key in ("softmax_temp", "randomness", "min_add_len") if key not in config
    ]
    if missing:
        raise BenchmarkConfigurationError(
            f"Config is missing required sampling key(s): {', '.join(missing)}"
        )
    diffusion_type = str(config.get("diffusion_type", "mdlm")).lower()
    if diffusion_type not in {"mdlm", "udlm"}:
        raise BenchmarkConfigurationError(
            "diffusion_type must be either 'mdlm' or 'udlm'"
        )
    try:
        softmax_temp = float(config["softmax_temp"])
        randomness = float(config["randomness"])
        min_add_len = int(config["min_add_len"])
    except (TypeError, ValueError) as exc:
        raise BenchmarkConfigurationError("Sampling parameters have invalid types") from exc
    if not math.isfinite(softmax_temp) or softmax_temp <= 0:
        raise BenchmarkConfigurationError("softmax_temp must be finite and positive")
    if not math.isfinite(randomness) or randomness < 0:
        raise BenchmarkConfigurationError("randomness must be finite and non-negative")
    if min_add_len < 0 or isinstance(config["min_add_len"], bool):
        raise BenchmarkConfigurationError("min_add_len must be a non-negative integer")
    if float(min_add_len) != float(config["min_add_len"]):
        raise BenchmarkConfigurationError("min_add_len must be an integer")

    num_steps: int | None = None
    inference_eps: float | None = None
    if diffusion_type == "udlm":
        missing_udlm = [
            key
            for key in ("num_steps", "inference_eps", "exclude_special_tokens")
            if key not in config
        ]
        if missing_udlm:
            raise BenchmarkConfigurationError(
                "UDLM config is missing required sampling key(s): "
                + ", ".join(missing_udlm)
            )
        try:
            num_steps = int(config["num_steps"])
            inference_eps = float(config["inference_eps"])
        except (TypeError, ValueError) as exc:
            raise BenchmarkConfigurationError(
                "UDLM sampling parameters have invalid types"
            ) from exc
        if (
            isinstance(config["num_steps"], bool)
            or num_steps <= 0
            or float(num_steps) != float(config["num_steps"])
        ):
            raise BenchmarkConfigurationError("num_steps must be a positive integer")
        if not math.isfinite(inference_eps) or not 0 < inference_eps < 1:
            raise BenchmarkConfigurationError(
                "inference_eps must be finite and lie strictly between 0 and 1"
            )
        if not isinstance(config["exclude_special_tokens"], bool):
            raise BenchmarkConfigurationError(
                "exclude_special_tokens must be a boolean"
            )
        exclude_special_tokens: bool | None = config["exclude_special_tokens"]
    elif (
        config.get("num_steps") is not None
        or config.get("inference_eps") is not None
        or config.get("exclude_special_tokens") is not None
    ):
        raise BenchmarkConfigurationError(
            "MDLM inference config must leave UDLM-only settings null"
        )
    else:
        exclude_special_tokens = None
    return {
        "diffusion_type": diffusion_type,
        "softmax_temp": softmax_temp,
        "randomness": randomness,
        "min_add_len": min_add_len,
        "num_steps": num_steps,
        "inference_eps": inference_eps,
        "exclude_special_tokens": exclude_special_tokens,
    }


def validate_device(device: str) -> None:
    import torch

    try:
        parsed = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise BenchmarkConfigurationError(f"Invalid Torch device: {device!r}") from exc
    if parsed.type == "cuda":
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices is None or not visible_devices.strip():
            raise BenchmarkConfigurationError(
                "CUDA benchmark runs require an explicit, non-empty "
                "CUDA_VISIBLE_DEVICES mapping. Select and isolate an idle physical GPU "
                "immediately before launch; use logical --device cuda:0 inside it."
            )
        if not torch.cuda.is_available():
            raise BenchmarkConfigurationError(
                "A CUDA device was requested but torch.cuda.is_available() is false"
            )


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    import torch

    load_kwargs: dict[str, Any] = {
        "map_location": "cpu",
        # Lightning checkpoints contain trusted config objects in addition to tensors.
        "weights_only": False,
    }
    try:
        checkpoint = torch.load(path, mmap=True, **load_kwargs)
    except (TypeError, RuntimeError):
        checkpoint = torch.load(path, **load_kwargs)
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("Checkpoint root is not a mapping")
    global_step = checkpoint.get("global_step")
    epoch = checkpoint.get("epoch")
    if hasattr(global_step, "item"):
        global_step = global_step.item()
    if hasattr(epoch, "item"):
        epoch = epoch.item()
    if global_step is None:
        raise RuntimeError("Checkpoint does not contain global_step")
    hyper_parameters = checkpoint.get("hyper_parameters", {})
    checkpoint_config = (
        hyper_parameters.get("config", {})
        if isinstance(hyper_parameters, Mapping)
        else {}
    )
    checkpoint_training = (
        checkpoint_config.get("training", {})
        if isinstance(checkpoint_config, Mapping)
        else {}
    )
    diffusion_type = str(checkpoint_training.get("diffusion", "mdlm")).lower()
    if diffusion_type not in {"mdlm", "udlm"}:
        raise RuntimeError(
            f"Checkpoint declares unsupported diffusion type {diffusion_type!r}"
        )
    checkpoint_udlm = (
        checkpoint_training.get("udlm", {})
        if isinstance(checkpoint_training, Mapping)
        else {}
    )
    udlm_inference_eps = (
        float(checkpoint_udlm.get("inference_eps", 1e-5))
        if diffusion_type == "udlm"
        else None
    )
    if udlm_inference_eps is not None and not 0 < udlm_inference_eps < 1:
        raise RuntimeError("Checkpoint UDLM inference_eps must lie in (0, 1)")
    udlm_exclude_special_tokens = (
        bool(checkpoint_udlm.get("exclude_special_tokens", False))
        if diffusion_type == "udlm"
        else None
    )
    metadata = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "mtime_utc": datetime.fromtimestamp(
            path.stat().st_mtime, timezone.utc
        ).isoformat(),
        "global_step": int(global_step),
        "epoch": int(epoch) if epoch is not None else None,
        "diffusion_type": diffusion_type,
        "udlm_inference_eps": udlm_inference_eps,
        "udlm_exclude_special_tokens": udlm_exclude_special_tokens,
    }
    del checkpoint
    return metadata


def _git_command(arguments: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def git_provenance() -> dict[str, Any]:
    status = _git_command(["status", "--porcelain=v1", "--untracked-files=normal"])
    return {
        "repo_root": str(REPO_ROOT),
        "commit": _git_command(["rev-parse", "HEAD"]),
        "branch": _git_command(["branch", "--show-current"]),
        "remote_origin": _git_command(["remote", "get-url", "origin"]),
        "dirty": bool(status) if status is not None else None,
        "status_porcelain": status.splitlines() if status else [],
        "runner_sha256": _sha256(Path(__file__).resolve()),
    }


def implementation_input_provenance() -> dict[str, Any]:
    """Fingerprint source/data inputs that directly define generation semantics."""
    import pickle

    result: dict[str, Any] = {}
    for name, path in IMPLEMENTATION_INPUT_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required benchmark input does not exist: {path}")
        result[name] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }

    length_path = IMPLEMENTATION_INPUT_PATHS["length_distribution"]
    with length_path.open("rb") as handle:
        lengths = pickle.load(handle)
    if not isinstance(lengths, Sequence) or isinstance(lengths, (str, bytes)):
        raise RuntimeError("data/len.pk must contain a sequence of lengths")
    if not lengths:
        raise RuntimeError("data/len.pk contains no lengths")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in lengths):
        raise RuntimeError("data/len.pk contains a non-integer length")
    result["length_distribution"].update(
        {
            "count": len(lengths),
            "minimum": min(lengths),
            "median": _json_compatible_number(statistics.median(lengths)),
            "maximum": max(lengths),
        }
    )
    return result


def tokenizer_provenance(tokenizer: Any) -> dict[str, Any]:
    """Fingerprint the tokenizer instance that actually drives this run."""
    vocabulary = tokenizer.get_vocab()
    if not isinstance(vocabulary, Mapping) or not vocabulary:
        raise RuntimeError("Loaded tokenizer did not expose a non-empty vocabulary")
    normalized_vocabulary = {
        str(token): int(index) for token, index in vocabulary.items()
    }
    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_json = None
    backend_serialization_error = None
    if backend is not None:
        try:
            backend_json = backend.to_str()
        except Exception as exc:
            # SAFE installs a custom pre-tokenizer that some tokenizers builds
            # cannot serialize.  The effective vocabulary and package version
            # remain independently fingerprinted; preserve the limitation.
            backend_serialization_error = _error_text(exc)
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    if not isinstance(init_kwargs, Mapping):
        init_kwargs = {}
    added_vocabulary = tokenizer.get_added_vocab()
    normalized_added_vocabulary = {
        str(token): int(index) for token, index in added_vocabulary.items()
    }
    return {
        "requested_identifier": TOKENIZER_REQUESTED_IDENTIFIER,
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")) or None,
        "declared_revision": init_kwargs.get("revision"),
        "resolved_commit_hash": init_kwargs.get("_commit_hash"),
        "base_vocab_size": int(tokenizer.vocab_size),
        "effective_size": int(len(tokenizer)),
        "vocabulary_sha256": _canonical_json_sha256(normalized_vocabulary),
        "added_vocabulary_sha256": _canonical_json_sha256(
            normalized_added_vocabulary
        ),
        "backend_json_sha256": (
            hashlib.sha256(backend_json.encode("utf-8")).hexdigest()
            if backend_json is not None
            else None
        ),
        "backend_serialization_error": backend_serialization_error,
        "special_token_ids": {
            "pad": tokenizer.pad_token_id,
            "bos": tokenizer.bos_token_id,
            "eos": tokenizer.eos_token_id,
            "mask": tokenizer.mask_token_id,
        },
    }


def _package_version(*distribution_names: str) -> str | None:
    for distribution_name in distribution_names:
        try:
            return importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def environment_metadata(requested_device: str, resolved_device: str) -> dict[str, Any]:
    import torch

    metadata: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "working_directory": str(Path.cwd()),
        "versions": {
            "torch": _package_version("torch"),
            "lightning": _package_version("lightning"),
            "transformers": _package_version("transformers"),
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "pyyaml": _package_version("PyYAML"),
            "safe": _package_version("safe-mol", "safe"),
            "rdkit": _package_version("rdkit"),
            "tdc": _package_version("PyTDC", "tdc"),
            "bionemo_moco": _package_version("bionemo-moco"),
        },
        "requested_device": requested_device,
        "resolved_model_device": resolved_device,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "launch_environment": {
            key: os.environ.get(key) for key in LAUNCH_ENVIRONMENT_KEYS
        },
    }
    resolved = torch.device(resolved_device)
    if resolved.type == "cuda":
        logical_index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(logical_index)
        metadata["cuda_device"] = {
            "logical_index": logical_index,
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
        }
    else:
        metadata["cuda_device"] = None
    return metadata


def _uses_bracket_safe(sampler: Any) -> bool:
    return bool(sampler.model.config.training.get("use_bracket_safe"))


def assert_local_genmol_import() -> Path:
    """Fail before model loading if Python resolved GenMol outside this worktree."""

    import genmol

    module_file = getattr(genmol, "__file__", None)
    if not module_file:
        raise RuntimeError("Loaded genmol package has no inspectable __file__")
    resolved = Path(module_file).resolve()
    if resolved != REPO_SRC and REPO_SRC not in resolved.parents:
        raise RuntimeError(
            f"Refusing non-worktree genmol import: {resolved}; expected under {REPO_SRC}"
        )
    return resolved


def assert_runtime_module_provenance(
    implementation_inputs: Mapping[str, Mapping[str, Any]],
) -> None:
    """Bind loaded generation modules to the source hashes in the summary."""

    import genmol.backbone as backbone_module
    import genmol.diffusion as diffusion_module
    import genmol.model as model_module
    import genmol.sampler as sampler_module

    modules = {
        "sampler_source": sampler_module,
        "model_source": model_module,
        "diffusion_source": diffusion_module,
        "backbone_source": backbone_module,
    }
    for source_name, module in modules.items():
        module_path = Path(module.__file__).resolve()
        recorded = implementation_inputs[source_name]
        if module_path != Path(str(recorded["path"])).resolve():
            raise RuntimeError(
                f"Runtime {module.__name__} path {module_path} does not match "
                f"recorded {source_name} path {recorded['path']}"
            )
        runtime_sha256 = _sha256(module_path)
        if runtime_sha256 != recorded["sha256"]:
            raise RuntimeError(
                f"Runtime {module.__name__} source changed after provenance capture"
            )


def assert_runtime_tdc_metric_provenance(
    metric_inputs: Mapping[str, Any],
) -> None:
    """Bind imported TDC metric modules to the files fingerprinted preflight."""

    module_names = {
        "oracle_dispatch": "tdc.oracles",
        "sa_qed_scoring": "tdc.chem_utils.oracle.oracle",
        "evaluator_dispatch": "tdc.evaluator",
        "diversity_scoring": "tdc.chem_utils.evaluator",
    }
    tdc_provenance = metric_inputs.get("tdc_metric_implementation")
    if not isinstance(tdc_provenance, Mapping):
        raise RuntimeError("Metric provenance lacks TDC implementation metadata")
    recorded_files = tdc_provenance.get("implementation_files")
    if not isinstance(recorded_files, Mapping) or set(recorded_files) != set(
        module_names
    ):
        raise RuntimeError("Metric provenance has incomplete TDC implementation files")
    for source_name, module_name in module_names.items():
        module = importlib.import_module(module_name)
        module_file = getattr(module, "__file__", None)
        if not module_file:
            raise RuntimeError(f"Runtime TDC module {module_name} has no source path")
        runtime_path = Path(module_file).resolve()
        recorded = recorded_files[source_name]
        if not isinstance(recorded, Mapping):
            raise RuntimeError(f"TDC metric provenance {source_name} is malformed")
        if runtime_path != Path(str(recorded.get("path"))).resolve():
            raise RuntimeError(
                f"Runtime TDC module {module_name} path does not match preflight"
            )
        if _sha256(runtime_path) != recorded.get("sha256"):
            raise RuntimeError(
                f"Runtime TDC module {module_name} changed after preflight"
            )


@contextmanager
def pinned_tdc_sa_oracle(
    snapshot: PinnedSAMetricInput,
    oracle_class: type,
) -> Iterable[Any]:
    """Yield ``Oracle('sa')`` with verified resident scores and no downloader.

    TDC normally calls ``oracle_load('fpscores')`` lazily on the first SA score.
    We instead populate the same module-level mapping from already verified
    bytes and replace both reachable downloader hooks with a fail-closed stub
    for the duration of scoring.
    """

    assert_runtime_tdc_metric_provenance(snapshot.provenance)
    oracle_dispatch = importlib.import_module("tdc.oracles")
    scoring_module = importlib.import_module("tdc.chem_utils.oracle.oracle")

    def downloader_disabled(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(
            "TDC oracle downloading is disabled; only the pinned resident SA "
            "fragment scores may be used"
        )

    previous_dispatch_loader = oracle_dispatch.oracle_load
    previous_scoring_loader = scoring_module.oracle_load
    previous_scores = scoring_module._fscores
    resident_scores = snapshot.fragment_scores
    oracle_dispatch.oracle_load = downloader_disabled
    scoring_module.oracle_load = downloader_disabled
    scoring_module._fscores = resident_scores
    try:
        oracle = oracle_class("sa")
        if getattr(oracle, "name", None) != "sa":
            raise RuntimeError("TDC did not resolve the requested SA oracle exactly")
        if getattr(oracle, "evaluator_func", None) is not scoring_module.SA:
            raise RuntimeError("TDC SA oracle resolved to an unexpected implementation")
        yield oracle
        if scoring_module._fscores is not resident_scores:
            raise RuntimeError("TDC replaced the pinned resident SA fragment scores")
    finally:
        scoring_module._fscores = previous_scores
        scoring_module.oracle_load = previous_scoring_loader
        oracle_dispatch.oracle_load = previous_dispatch_loader


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = args.checkpoint.resolve()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config does not exist: {config_path}")
    if args.num_samples <= 0:
        raise BenchmarkConfigurationError("num_samples must be a positive integer")
    if args.seed < 0 or args.seed > 2**32 - 1:
        raise BenchmarkConfigurationError("seed must be in [0, 2**32 - 1]")

    started_at = _utc_now()
    total_start = time.perf_counter()
    validate_device(args.device)
    # Verify and retain the metric-defining bytes before loading a checkpoint,
    # importing CUDA-facing model code, or moving any tensor onto a GPU.
    sa_metric_snapshot = load_pinned_sa_metric_input()
    metric_inputs = dict(sa_metric_snapshot.provenance)
    source_config_sha256 = _sha256(config_path)
    source_config = load_yaml_config(config_path)
    if _sha256(config_path) != source_config_sha256:
        raise RuntimeError(f"Config changed while it was being read: {config_path}")
    sampling_config = validate_sampling_config(source_config)
    effective_config = dict(source_config)
    effective_config.update(
        {
            "model_path": str(checkpoint_path),
            "num_samples": args.num_samples,
            "device": args.device,
        }
    )
    sampling_config_sha256 = _canonical_json_sha256(sampling_config)
    effective_config_sha256 = _canonical_json_sha256(effective_config)

    with output_lock(output_dir):
        validate_output_target(output_dir, overwrite=args.overwrite)
        checkpoint_info = checkpoint_metadata(checkpoint_path)
        if checkpoint_info["diffusion_type"] != sampling_config["diffusion_type"]:
            raise BenchmarkConfigurationError(
                "Inference config diffusion_type does not match checkpoint metadata: "
                f"{sampling_config['diffusion_type']!r} != "
                f"{checkpoint_info['diffusion_type']!r}"
            )
        if sampling_config["diffusion_type"] == "udlm" and not math.isclose(
            float(checkpoint_info["udlm_inference_eps"]),
            float(sampling_config["inference_eps"]),
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise BenchmarkConfigurationError(
                "Inference config inference_eps does not match checkpoint metadata: "
                f"{sampling_config['inference_eps']} != "
                f"{checkpoint_info['udlm_inference_eps']}"
            )
        if (
            sampling_config["diffusion_type"] == "udlm"
            and checkpoint_info["udlm_exclude_special_tokens"]
            is not sampling_config["exclude_special_tokens"]
        ):
            raise BenchmarkConfigurationError(
                "Inference config exclude_special_tokens does not match checkpoint "
                "metadata"
            )
        implementation_inputs = implementation_input_provenance()
        git_info = git_provenance()

        # Heavy imports are intentionally below argument/output validation.
        assert_local_genmol_import()
        from tdc import Evaluator, Oracle

        from genmol.sampler import Sampler

        assert_runtime_module_provenance(implementation_inputs)

        model_load_start = time.perf_counter()
        sampler = Sampler(str(checkpoint_path))
        sampler.model.to(args.device)
        sampler.mdlm.to_device(sampler.model.device)
        model_load_seconds = time.perf_counter() - model_load_start
        tokenizer_info = tokenizer_provenance(sampler.model.tokenizer)
        environment_info = environment_metadata(args.device, str(sampler.model.device))
        use_bracket_safe = _uses_bracket_safe(sampler)

        seed_info = seed_sampling(args.seed, args.device)
        synchronize_device(sampler.model.device)
        model_sampling_start = time.perf_counter()
        raw_model_texts, denoising_protocol = generate_raw_model_text(
            sampler,
            args.num_samples,
            **sampling_config,
        )
        synchronize_device(sampler.model.device)
        model_sampling_seconds = time.perf_counter() - model_sampling_start

        scoring_start = time.perf_counter()
        decode_timing: dict[str, float] = {}
        records = decode_records(
            raw_model_texts,
            use_bracket_safe=use_bracket_safe,
            timing=decode_timing,
        )
        with pinned_tdc_sa_oracle(sa_metric_snapshot, Oracle) as sa_oracle:
            metrics, failure_counts = evaluate_records(
                records,
                requested_count=args.num_samples,
                oracle_qed=Oracle("qed"),
                oracle_sa=sa_oracle,
                diversity_evaluator=Evaluator("diversity"),
            )
        assert_runtime_tdc_metric_provenance(metric_inputs)
        scoring_seconds = time.perf_counter() - scoring_start
        released_postprocessing_seconds = decode_timing["released_postprocessing"]
        # Released run.py times the full de_novo_generation call.  Its endpoint
        # includes tokenizer decoding (already in model_sampling_seconds), SAFE
        # repair, failed-row removal, and largest-component selection.
        generation_seconds = model_sampling_seconds + released_postprocessing_seconds

        raw_samples_path = output_dir / RAW_SAMPLES_FILENAME
        summary_path = output_dir / SUMMARY_FILENAME
        atomic_write_csv(raw_samples_path, records)

        total_seconds = time.perf_counter() - total_start
        summary: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            # Top-level aliases make launcher completion checks cheap.  The
            # structured copies below are retained for schema clarity.
            "seed": args.seed,
            "num_samples": args.num_samples,
            "run": {
                "seed": args.seed,
                "requested_sample_count": args.num_samples,
                "evaluation_tier": (
                    "final" if args.num_samples == 1_000 else "pilot"
                ),
                "final_protocol_eligible": args.num_samples == 1_000,
                "started_at_utc": started_at,
                "completed_at_utc": _utc_now(),
                "one_seed_per_invocation": True,
                "single_generation_batch": True,
                "generation_protocol": {
                    **denoising_protocol,
                    "model_use_bracket_safe": use_bracket_safe,
                    "single_generation_batch": True,
                    "released_safe_fix": True,
                    "released_largest_component": "maximum SMILES string length",
                    "strict_safe_fix": False,
                },
                "command": [sys.executable, *sys.argv],
                "seed_configuration": seed_info,
            },
            "checkpoint": checkpoint_info,
            "config": {
                "path": str(config_path),
                "sha256": source_config_sha256,
                "sampling_sha256": sampling_config_sha256,
                "effective_sha256": effective_config_sha256,
                "source": source_config,
                "effective": effective_config,
                "sampling": sampling_config,
            },
            "metrics": metrics,
            "failure_counts": failure_counts,
            "runtime_seconds": {
                "model_load_and_device_move": model_load_seconds,
                "model_sampling_and_tokenizer": model_sampling_seconds,
                "released_postprocessing": released_postprocessing_seconds,
                "generation": generation_seconds,
                "decode_and_metrics": scoring_seconds,
                "total_before_summary_write": total_seconds,
            },
            "environment": environment_info,
            "git": git_info,
            "implementation_inputs": implementation_inputs,
            "metric_inputs": metric_inputs,
            "tokenizer": tokenizer_info,
            "artifacts": {
                "raw_samples_csv": {
                    "path": str(raw_samples_path),
                    "sha256": _sha256(raw_samples_path),
                    "row_count": len(records),
                    "fields": list(RAW_SAMPLE_FIELDS),
                },
                "summary_json": {"path": str(summary_path)},
            },
        }
        # The summary is written last and is the completion marker.  Its own
        # digest is intentionally omitted because a file cannot contain its
        # final self-hash.
        atomic_write_json(summary_path, summary)

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one reproducible GenMol de novo benchmark seed and retain raw, "
            "strict, and released-repaired outputs."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--num-samples", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace raw_samples.csv and summary.json if they exist.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_benchmark(args)
    released = summary["metrics"]["released_comparable"]
    strict = summary["metrics"]["strict"]
    print(f"Completed: {args.output_dir.resolve()}")

    def format_metric(value: float | None) -> str:
        return "undefined" if value is None else f"{value:.6f}"

    print(
        "Released-comparable: "
        f"validity={format_metric(released['validity'])}, "
        f"uniqueness={format_metric(released['uniqueness'])}, "
        f"diversity={format_metric(released['diversity'])}, "
        f"quality={format_metric(released['quality'])}"
    )
    strict_uniqueness = strict["uniqueness"]
    strict_diversity = strict["diversity"]
    print(
        "Strict: "
        f"validity={format_metric(strict['validity'])}, "
        f"uniqueness={format_metric(strict_uniqueness)}, "
        f"diversity={format_metric(strict_diversity)}, "
        f"quality={format_metric(strict['quality'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Audit empirical-UDLM prior geometry on frozen train/validation token IDs.

This is a CPU-only analytical diagnostic.  It neither trains nor samples a
model, and its held-out unigram cross-entropy cannot establish molecular
quality or select a winning generator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FREQUENCY_PATH = (
    REPOSITORY_ROOT
    / "experiments"
    / "udlm"
    / "token_frequency"
    / "train_first_10000.json"
)
VALIDATION_PANEL_PATH = (
    REPOSITORY_ROOT / "experiments" / "udlm" / "validation_panel" / "first_256.json"
)
FREQUENCY_SHA256 = "088c78e75611f3cc42c4011e1da6f65a377e673b9cba07a28b126b0fc62f06ed"
VALIDATION_PANEL_SHA256 = (
    "e2493da4f3cb3217b48c78dc2901dc7524d7d90dc3959cffa87a1b6f8a9a7658"
)
TOKENIZER_SHA256 = "0db5f4dbdc7e8ff759e98483759611a426e187ee7f3f0a91edc8800abe7bf140"
DEFAULT_MIXTURES = (
    0.0001,
    0.001,
    0.003,
    0.01,
    0.03,
    0.05,
    0.1,
    0.2,
    0.5,
    0.75,
)
FOLLOWUP_MIXTURES = (0.001, 0.01, 0.05)
SOURCE_PATHS = (
    Path("scripts/udlm/audit_prior_geometry.py"),
    Path("experiments/udlm/token_frequency/train_first_10000.json"),
    Path("experiments/udlm/validation_panel/first_256.json"),
    Path("configs/base.yaml"),
    Path("configs/udlm.yaml"),
    Path("configs/udlm_categorical.yaml"),
    Path("src/genmol/diffusion.py"),
    Path("src/genmol/model.py"),
    Path("src/genmol/utils/utils_data.py"),
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _read_regular_file(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"input is not a regular file: {path}")
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        chunks = []
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        final_identity = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        payload = b"".join(chunks)
        if final_identity != identity or len(payload) != before.st_size:
            raise RuntimeError(f"input changed while it was read: {path}")
        return payload
    finally:
        os.close(descriptor)


def _load_frozen_json(path: Path, expected_sha256: str) -> tuple[dict[str, Any], bytes]:
    payload = _read_regular_file(path)
    actual = _sha256(payload)
    if actual != expected_sha256:
        raise ValueError(f"frozen input hash mismatch for {path}: {actual}")

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    value = json.loads(
        payload,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=reject_nonfinite,
    )
    if not isinstance(value, dict):
        raise ValueError(f"frozen input must be a JSON object: {path}")
    return value, payload


def _length_prefixed_digest(values: Sequence[bytes]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


def validate_inputs(frequencies: dict[str, Any], panel: dict[str, Any]) -> None:
    """Validate the two frozen artifacts used by the analytical audit."""

    if frequencies.get("schema_version") != 1 or frequencies.get("purpose") != (
        "CPU-only token-frequency diagnostic; not benchmark evidence"
    ):
        raise ValueError("unexpected frequency artifact schema or purpose")
    if panel.get("schema_version") != 1 or panel.get("purpose") != (
        "fixed held-out denoising panel; not generative benchmark evidence"
    ):
        raise ValueError("unexpected validation-panel schema or purpose")

    frequency_tokenizer = frequencies.get("tokenizer")
    panel_tokenizer = panel.get("tokenizer")
    if not isinstance(frequency_tokenizer, dict) or not isinstance(
        panel_tokenizer, dict
    ):
        raise ValueError("missing tokenizer provenance")
    for field in (
        "repo_id",
        "revision",
        "tokenizer_json_sha256",
        "base_vocab_size",
        "special_token_ids",
    ):
        if frequency_tokenizer.get(field) != panel_tokenizer.get(field):
            raise ValueError(f"frequency/panel tokenizer mismatch for {field}")
    if frequency_tokenizer.get("tokenizer_json_sha256") != TOKENIZER_SHA256:
        raise ValueError("unexpected tokenizer bytes")

    frequency_dataset = frequencies.get("dataset")
    panel_dataset = panel.get("dataset")
    if not isinstance(frequency_dataset, dict) or not isinstance(panel_dataset, dict):
        raise ValueError("missing dataset provenance")
    if frequency_dataset.get("repo_id") != panel_dataset.get("repo_id") or (
        frequency_dataset.get("revision") != panel_dataset.get("revision")
    ):
        raise ValueError("frequency/panel dataset identity mismatch")
    if frequency_dataset.get("split") != "train" or panel_dataset.get("split") != (
        "validation"
    ):
        raise ValueError("frequency and panel artifacts must use train/validation")

    vocab_size = frequency_tokenizer.get("base_vocab_size")
    counts = frequencies.get("counts_by_token_id")
    if (
        type(vocab_size) is not int
        or vocab_size < 2
        or not isinstance(counts, list)
        or len(counts) != vocab_size
        or any(type(count) is not int or count < 0 for count in counts)
    ):
        raise ValueError("invalid training token counts")
    if sum(counts) != frequencies.get("content_token_count"):
        raise ValueError("training content-token total mismatch")
    if sum(count > 0 for count in counts) != frequencies.get("observed_token_types"):
        raise ValueError("training observed-type count mismatch")
    if sum(count == 0 for count in counts) != frequencies.get("unobserved_token_types"):
        raise ValueError("training unobserved-type count mismatch")

    rows = panel.get("rows")
    sample_count = panel.get("sample_count")
    if (
        not isinstance(rows, list)
        or not rows
        or type(sample_count) is not int
        or sample_count != len(rows)
    ):
        raise ValueError("invalid validation rows")
    ordered_digest = panel.get("ordered_token_ids_sha256")
    if not isinstance(ordered_digest, str) or not SHA256_PATTERN.fullmatch(
        ordered_digest
    ):
        raise ValueError("invalid validation ordered-token digest")
    special_ids = panel_tokenizer.get("special_token_ids")
    if (
        not isinstance(special_ids, list)
        or special_ids != sorted(set(special_ids))
        or any(
            type(token_id) is not int or not 0 <= token_id < vocab_size
            for token_id in special_ids
        )
    ):
        raise ValueError("invalid tokenizer special-token IDs")
    special_set = set(special_ids)
    if any(counts[token_id] != 0 for token_id in special_ids):
        raise ValueError("training content counts include a special token")
    if (
        type(frequencies.get("content_token_count")) is not int
        or frequencies["content_token_count"] <= 0
        or type(frequencies.get("example_count")) is not int
        or frequencies["example_count"] <= 0
        or type(frequencies.get("observed_token_types")) is not int
        or type(frequencies.get("unobserved_token_types")) is not int
    ):
        raise ValueError("training aggregate counts must be positive integers")
    encoded_rows = []
    previous_source_index = -1
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"validation row {index} is not an object")
        source_index = row.get("source_index")
        token_ids = row.get("input_ids")
        if type(source_index) is not int or source_index <= previous_source_index:
            raise ValueError("validation source indices are not strictly increasing")
        previous_source_index = source_index
        if (
            not isinstance(token_ids, list)
            or len(token_ids) < 2
            or any(
                type(token_id) is not int or not 0 <= token_id < vocab_size
                for token_id in token_ids
            )
        ):
            raise ValueError(f"invalid token IDs in validation row {index}")
        content_length = sum(token_id not in special_set for token_id in token_ids)
        recorded_content_length = row.get("content_length")
        if (
            content_length <= 0
            or type(recorded_content_length) is not int
            or recorded_content_length != content_length
        ):
            raise ValueError(f"validation content length mismatch at row {index}")
        encoded_rows.append(
            json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
        )
    if _length_prefixed_digest(encoded_rows) != ordered_digest:
        raise ValueError("validation ordered-token digest mismatch")


def _validate_mixtures(mixtures: Sequence[float]) -> tuple[float, ...]:
    if not mixtures:
        raise ValueError("at least one mixture weight is required")
    normalized = []
    for value in mixtures:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("mixture weights must be real numbers")
        weight = float(value)
        if not math.isfinite(weight) or not 0.0 < weight < 1.0:
            raise ValueError(
                "mixture weights must be finite and strictly between 0 and 1"
            )
        normalized.append(weight)
    if normalized != sorted(normalized) or len(normalized) != len(set(normalized)):
        raise ValueError("mixture weights must be unique and strictly increasing")
    return tuple(normalized)


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256(payload)


def load_empirical_process_contract(
    frequencies: dict[str, Any],
) -> tuple[dict[str, Any], tuple[int, ...], tuple[float, ...]]:
    """Construct the configured process without allocating the BERT backbone."""

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from genmol.model import _build_udlm_process

    resolvers = {
        "cwd": lambda: str(REPOSITORY_ROOT),
        "device_count": lambda: 1,
        "eval": lambda expression: eval(expression, {"__builtins__": {}}, {}),
        "div_up": lambda x, y: (x + y - 1) // y,
    }
    for name, resolver in resolvers.items():
        if OmegaConf.has_resolver(name):
            OmegaConf.clear_resolver(name)
        OmegaConf.register_new_resolver(name, resolver)
    with initialize_config_dir(
        version_base=None,
        config_dir=str(REPOSITORY_ROOT / "configs"),
    ):
        config = compose(config_name="udlm_categorical")
    resolved = OmegaConf.to_container(config, resolve=True, enum_to_str=True)
    if not isinstance(resolved, dict):
        raise RuntimeError("resolved UDLM categorical config is not a mapping")
    training = resolved.get("training")
    model = resolved.get("model")
    if not isinstance(training, dict) or not isinstance(model, dict):
        raise RuntimeError("resolved UDLM categorical config is malformed")
    udlm = training.get("udlm")
    if not isinstance(udlm, dict):
        raise RuntimeError("resolved UDLM categorical config has no UDLM section")
    expected = {
        "diffusion": "udlm",
        "prior_variant": "empirical_frequency",
        "empirical_uniform_mix": 0.01,
        "exclude_special_tokens": False,
        "sampling_eps": 1e-3,
        "noise_eps": 1e-3,
        "antithetic_sampling": True,
        "vocab_size": 1880,
    }
    observed = {
        "diffusion": training.get("diffusion"),
        "prior_variant": udlm.get("prior_variant"),
        "empirical_uniform_mix": udlm.get("empirical_uniform_mix"),
        "exclude_special_tokens": udlm.get("exclude_special_tokens"),
        "sampling_eps": training.get("sampling_eps"),
        "noise_eps": udlm.get("noise_eps"),
        "antithetic_sampling": training.get("antithetic_sampling"),
        "vocab_size": model.get("vocab_size"),
    }
    if observed != expected:
        raise RuntimeError(f"configured empirical UDLM contract changed: {observed!r}")

    tokenizer_record = frequencies["tokenizer"]

    class PinnedTokenizerIdentity:
        vocab_size = tokenizer_record["base_vocab_size"]
        all_special_ids = tuple(tokenizer_record["special_token_ids"])

    excluded = (
        PinnedTokenizerIdentity.all_special_ids
        if observed["exclude_special_tokens"]
        else ()
    )
    process, metadata = _build_udlm_process(
        variant=observed["prior_variant"],
        model_vocab_size=observed["vocab_size"],
        excluded_token_ids=excluded,
        sampling_eps=observed["sampling_eps"],
        noise_eps=observed["noise_eps"],
        antithetic_sampling=observed["antithetic_sampling"],
        empirical_uniform_mix=observed["empirical_uniform_mix"],
        tokenizer=PinnedTokenizerIdentity(),
    )
    active_ids = tuple(int(value) for value in process.diffusion_token_ids.tolist())
    probabilities = tuple(float(value) for value in process.stationary_probs.tolist())
    metadata_record = metadata.to_dict()
    if (
        metadata_record["variant"] != "empirical_frequency"
        or metadata_record["uniform_mixture_weight"] != 0.01
        or metadata_record["frequency_artifact_sha256"] != FREQUENCY_SHA256
        or metadata_record["active_vocab_size"] != len(active_ids)
        or metadata_record["excluded_token_ids"] != list(excluded)
        or len(probabilities) != len(active_ids)
    ):
        raise RuntimeError("live empirical process metadata disagrees with its config")
    relevant_config = {
        "model": {"vocab_size": observed["vocab_size"]},
        "training": {
            "diffusion": observed["diffusion"],
            "sampling_eps": observed["sampling_eps"],
            "antithetic_sampling": observed["antithetic_sampling"],
            "udlm": {
                "prior_variant": observed["prior_variant"],
                "empirical_uniform_mix": observed["empirical_uniform_mix"],
                "exclude_special_tokens": observed["exclude_special_tokens"],
                "noise_eps": observed["noise_eps"],
            },
        },
    }
    contract = {
        "resolved_relevant_config": relevant_config,
        "resolved_relevant_config_sha256": _canonical_json_sha256(relevant_config),
        "exact_process_backend": (
            f"{type(process).__module__}.{type(process).__qualname__}"
        ),
        "prior_metadata": metadata_record,
        "prior_metadata_sha256": _canonical_json_sha256(metadata_record),
        "active_token_count": len(active_ids),
        "configured_probability_sum": math.fsum(probabilities),
    }
    return contract, active_ids, probabilities


def _mixture_probabilities(
    counts: Sequence[int], active_ids: Sequence[int], weight: float
) -> list[float]:
    active_total = sum(counts[token_id] for token_id in active_ids)
    if active_total <= 0:
        raise ValueError("configured active alphabet has no training observations")
    active_size = len(active_ids)
    probabilities = [
        (1.0 - weight) * counts[token_id] / active_total + weight / active_size
        for token_id in active_ids
    ]
    normalizer = math.fsum(probabilities)
    return [probability / normalizer for probability in probabilities]


def analyze_prior_geometry(
    frequencies: dict[str, Any],
    panel: dict[str, Any],
    mixtures: Sequence[float],
    *,
    active_token_ids: Sequence[int],
    configured_probabilities: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Compute held-out unigram fit and stationary-prior concentration."""

    validate_inputs(frequencies, panel)
    mixture_grid = _validate_mixtures(mixtures)
    counts = frequencies["counts_by_token_id"]
    full_vocab_size = len(counts)
    if (
        not active_token_ids
        or any(
            type(token_id) is not int or not 0 <= token_id < full_vocab_size
            for token_id in active_token_ids
        )
        or list(active_token_ids) != sorted(set(active_token_ids))
    ):
        raise ValueError("active token IDs must be unique, sorted, and in range")
    active_ids = tuple(active_token_ids)
    active_set = set(active_ids)
    active_size = len(active_ids)
    if active_size < 2:
        raise ValueError("active alphabet must contain at least two token IDs")
    active_training_total = sum(counts[token_id] for token_id in active_ids)
    special_ids = set(panel["tokenizer"]["special_token_ids"])
    validation_counts: Counter[int] = Counter()
    for row in panel["rows"]:
        validation_counts.update(
            token_id for token_id in row["input_ids"] if token_id not in special_ids
        )
    validation_total = sum(validation_counts.values())
    if validation_total <= 0:
        raise ValueError("validation panel has no content tokens")
    inactive_validation_ids = sorted(set(validation_counts) - active_set)
    if inactive_validation_ids:
        raise ValueError(
            "validation content IDs are outside the configured active alphabet: "
            f"{inactive_validation_ids}"
        )

    unseen_ids = {token_id for token_id in active_ids if counts[token_id] == 0}
    validation_unseen = sum(
        count for token_id, count in validation_counts.items() if token_id in unseen_ids
    )
    rows = []
    for weight in mixture_grid:
        probabilities = _mixture_probabilities(counts, active_ids, weight)
        if not all(math.isfinite(value) and value > 0.0 for value in probabilities):
            raise RuntimeError("mixture did not produce a full-support finite prior")
        probability_by_token = dict(zip(active_ids, probabilities, strict=True))
        validation_nll = (
            -math.fsum(
                count * math.log(probability_by_token[token_id])
                for token_id, count in validation_counts.items()
            )
            / validation_total
        )
        entropy = -math.fsum(
            probability * math.log(probability) for probability in probabilities
        )
        ordered = sorted(probabilities, reverse=True)
        rows.append(
            {
                "uniform_mixture_weight": weight,
                "validation_content_token_nll_nats": validation_nll,
                "validation_content_token_perplexity": math.exp(validation_nll),
                "stationary_entropy_nats": entropy,
                "stationary_effective_vocabulary": math.exp(entropy),
                "training_unseen_token_mass": math.fsum(
                    probability_by_token[token_id] for token_id in unseen_ids
                ),
                "maximum_token_probability": ordered[0],
                "top_10_token_mass": math.fsum(ordered[:10]),
                "minimum_token_probability": ordered[-1],
            }
        )

    best = min(rows, key=lambda row: row["validation_content_token_nll_nats"])
    configured = next(
        (row for row in rows if row["uniform_mixture_weight"] == 0.01),
        None,
    )
    configured_agreement = None
    if configured_probabilities is not None:
        configured_values = [float(value) for value in configured_probabilities]
        if len(configured_values) != active_size or not all(
            math.isfinite(value) and value > 0.0 for value in configured_values
        ):
            raise ValueError("configured process probabilities are invalid")
        formula_values = _mixture_probabilities(counts, active_ids, 0.01)
        max_abs_difference = max(
            abs(expected - observed)
            for expected, observed in zip(
                formula_values, configured_values, strict=True
            )
        )
        if max_abs_difference > 2e-15:
            raise RuntimeError(
                "audited formula disagrees with the live configured prior"
            )
        configured_agreement = {
            "uniform_mixture_weight": 0.01,
            "probability_count": len(configured_values),
            "probability_sum": math.fsum(configured_values),
            "maximum_absolute_formula_difference": max_abs_difference,
            "status": "live_process_matches_audited_formula",
        }
    panel_unseen_qualification = (
        "This panel has no tokens unseen in the pinned training prefix, so it "
        "cannot evaluate the proposed unseen-token-support benefit."
        if validation_unseen == 0
        else (
            f"This panel contains {validation_unseen} token observations unseen "
            "in the pinned training prefix; the grid remains exploratory."
        )
    )
    return {
        "training": {
            "examples": frequencies["example_count"],
            "full_vocabulary_size": full_vocab_size,
            "full_content_tokens": sum(counts),
            "active_vocabulary_size": active_size,
            "active_content_tokens": active_training_total,
            "active_observed_token_types": active_size - len(unseen_ids),
            "active_unobserved_token_types": len(unseen_ids),
            "excluded_token_ids": sorted(set(range(full_vocab_size)) - active_set),
        },
        "validation": {
            "examples": len(panel["rows"]),
            "content_tokens": validation_total,
            "observed_token_types": len(validation_counts),
            "tokens_unseen_in_training_prefix": validation_unseen,
        },
        "uniform_baseline": {
            "validation_content_token_nll_nats": math.log(active_size),
            "validation_content_token_perplexity": float(active_size),
            "stationary_entropy_nats": math.log(active_size),
            "stationary_effective_vocabulary": float(active_size),
            "training_unseen_token_mass": len(unseen_ids) / active_size,
            "maximum_token_probability": 1.0 / active_size,
            "top_10_token_mass": min(10, active_size) / active_size,
            "minimum_token_probability": 1.0 / active_size,
        },
        "mixture_grid": rows,
        "grid_minimum_validation_nll": {
            **best,
            "qualification": (
                "Descriptive minimum on a fixed unigram grid; not a generator "
                "selection rule. Exploratory reuse of this panel requires a fresh "
                "confirmatory panel before a prior claim. " + panel_unseen_qualification
            ),
        },
        "configured_0_01": configured,
        "configured_process_agreement": configured_agreement,
        "proposed_followup_grid": {
            "uniform_mixture_weights": list(FOLLOWUP_MIXTURES),
            "status": "hypothesis_to_test_after_the_matched_R_S_E_health_gate",
            "rationale": (
                "Span a 50-fold unseen-token-mass range around the configured "
                "0.01 treatment while holding the process family and noise "
                "schedule fixed and varying only the stationary prior pi."
            ),
            "required_matched_control": (
                "Interpret empirical-prior rows against schedule_uniform only after "
                "matching initialization, seed, data order, budget, checkpoint, "
                "sampling protocol, and uncertainty analysis. This audit alone "
                "cannot select a weight or establish a causal effect."
            ),
        },
    }


def _git_environment() -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment["PATH"] = "/usr/bin:/bin"
    return environment


def _git(*arguments: str, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["/usr/bin/git", "-C", str(REPOSITORY_ROOT), *arguments],
        check=True,
        capture_output=True,
        text=text,
        env=_git_environment(),
    )


def require_clean_pushed_commit() -> str:
    status = _git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "-z",
        "--",
        ".",
        ":(exclude)output",
        ":(exclude)output/**",
    ).stdout
    if status:
        raise RuntimeError("source worktree must be clean before the audit")
    head = _git("rev-parse", "HEAD").stdout.strip()
    upstream = _git("rev-parse", "@{upstream}").stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", head) or head != upstream:
        raise RuntimeError("audit source revision must be pushed to its upstream")
    return head


def source_inputs(commit: str) -> dict[str, dict[str, Any]]:
    records = {}
    for relative_path in SOURCE_PATHS:
        payload = _read_regular_file(REPOSITORY_ROOT / relative_path)
        blob = _git("show", f"{commit}:{relative_path.as_posix()}", text=False).stdout
        if payload != blob:
            raise RuntimeError(f"working bytes differ from Git blob: {relative_path}")
        records[relative_path.as_posix()] = {
            "sha256": _sha256(payload),
            "size_bytes": len(payload),
            "git_blob_verified": True,
        }
    return records


def build_audit(
    frequencies: dict[str, Any],
    panel: dict[str, Any],
    mixtures: Sequence[float],
    *,
    git_commit: str,
    inputs: dict[str, dict[str, Any]],
    process_contract: dict[str, Any],
    active_token_ids: Sequence[int],
    configured_probabilities: Sequence[float],
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", git_commit):
        raise ValueError("git_commit must be a full SHA-1")
    geometry = analyze_prior_geometry(
        frequencies,
        panel,
        mixtures,
        active_token_ids=active_token_ids,
        configured_probabilities=configured_probabilities,
    )
    return {
        "schema_version": 1,
        "purpose": "CPU-only empirical-UDLM stationary-prior geometry audit",
        "claim_scope": (
            "This held-out unigram/concentration analysis performs no model "
            "training or molecular generation. It cannot rank generators, "
            "estimate chemical quality, or establish that UDLM beats GenMol."
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": git_commit,
            "upstream": git_commit,
            "dirty": False,
        },
        "inputs": {
            "frequency_artifact": {
                "path": str(FREQUENCY_PATH.relative_to(REPOSITORY_ROOT)),
                "sha256": FREQUENCY_SHA256,
            },
            "validation_panel": {
                "path": str(VALIDATION_PANEL_PATH.relative_to(REPOSITORY_ROOT)),
                "sha256": VALIDATION_PANEL_SHA256,
            },
            "source_files": inputs,
        },
        "configured_process_contract": process_contract,
        "definitions": {
            "stationary_prior": (
                "For active alphabet A, pi_j(w)=(1-w)*c_j/"
                "sum_{k in A}(c_k)+w/|A| for j in A, normalized in binary64; "
                "excluded IDs are outside the stationary support and 0<w<1"
            ),
            "validation_nll": (
                "negative mean log pi_j(w) over non-special token IDs in the "
                "ordered frozen validation panel"
            ),
            "effective_vocabulary": "exp(-sum_j pi_j log pi_j)",
            "training_unseen_token_mass": (
                "sum of pi_j over active IDs with c_j=0 in the pinned 10,000-"
                "example training prefix"
            ),
        },
        "geometry": geometry,
    }


def _bind_output(path: Path) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path)))
    bound = lexical.parent.resolve() / lexical.name
    if not lexical.name or (
        bound != REPOSITORY_ROOT and REPOSITORY_ROOT not in bound.parents
    ):
        raise ValueError("output must remain inside the repository")
    if os.path.lexists(bound):
        raise FileExistsError(f"refusing to overwrite output: {bound}")
    return bound


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    """Atomically publish one no-clobber artifact in the bound directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o644)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite output: {path}") from error
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mixture",
        type=float,
        action="append",
        dest="mixtures",
        help="Repeat for an explicit strictly increasing grid; defaults are pinned.",
    )
    args = parser.parse_args()
    output = _bind_output(args.output)
    mixtures = DEFAULT_MIXTURES if args.mixtures is None else tuple(args.mixtures)
    commit = require_clean_pushed_commit()
    inputs = source_inputs(commit)
    frequencies, _ = _load_frozen_json(FREQUENCY_PATH, FREQUENCY_SHA256)
    panel, _ = _load_frozen_json(VALIDATION_PANEL_PATH, VALIDATION_PANEL_SHA256)
    validate_inputs(frequencies, panel)
    process_contract, active_token_ids, configured_probabilities = (
        load_empirical_process_contract(frequencies)
    )
    result = build_audit(
        frequencies,
        panel,
        mixtures,
        git_commit=commit,
        inputs=inputs,
        process_contract=process_contract,
        active_token_ids=active_token_ids,
        configured_probabilities=configured_probabilities,
    )
    if require_clean_pushed_commit() != commit:
        raise RuntimeError("source revision changed while the audit was computed")
    if source_inputs(commit) != inputs:
        raise RuntimeError("source inputs changed while the audit was computed")
    _write_exclusive(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "source_commit": commit,
                "training": result["geometry"]["training"],
                "validation": result["geometry"]["validation"],
                "configured_0_01": result["geometry"]["configured_0_01"],
                "grid_minimum_validation_nll": result["geometry"][
                    "grid_minimum_validation_nll"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

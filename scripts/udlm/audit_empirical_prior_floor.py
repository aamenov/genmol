"""Audit a smaller empirical-UDLM uniform floor on training-only SAFE rows.

The existing empirical prior was estimated from the first 10,000 rows of the
pinned SAFE training stream and mixed with 1% uniform mass.  This CPU-only
audit evaluates that fixed estimate on the next two ordered 10,000-row blocks.
It does not train a model, decode molecules, inspect GPUs, or provide evidence
that UDLM beats GenMol.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BASE_FREQUENCY_RELATIVE_PATH = Path(
    "experiments/udlm/token_frequency/train_first_10000.json"
)
BASE_FREQUENCY_PATH = REPOSITORY_ROOT / BASE_FREQUENCY_RELATIVE_PATH
BASE_FREQUENCY_SHA256 = (
    "088c78e75611f3cc42c4011e1da6f65a377e673b9cba07a28b126b0fc62f06ed"
)
MODEL_VOCAB_SIZE = 1_880
MAX_SEQUENCE_LENGTH = 256
PREFIX_CHECKPOINTS = (10_000, 20_000, 30_000)
CURRENT_UNIFORM_MIXTURE_WEIGHT = 0.01
CANDIDATE_UNIFORM_MIXTURE_WEIGHT = 0.0002
MIXTURE_GRID = (
    0.000001,
    0.00001,
    0.00005,
    0.0001,
    0.0002,
    0.0003,
    0.001,
    0.003,
    0.01,
    0.05,
)
SOURCE_PATHS = (
    Path("scripts/udlm/audit_empirical_prior_floor.py"),
    Path("scripts/udlm/token_frequency_audit.py"),
    Path("src/genmol/utils/utils_data.py"),
    BASE_FREQUENCY_RELATIVE_PATH,
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class PriorFloorAuditError(RuntimeError):
    """Raised when the audit cannot establish its exact input contract."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256(payload)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def strict_json_loads(payload: bytes, *, label: str) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error


def _stable_file_bytes(path: Path) -> bytes:
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
        path_state = os.stat(path, follow_symlinks=False)
        for observed in (after, path_state):
            if (
                observed.st_dev,
                observed.st_ino,
                observed.st_mode,
                observed.st_nlink,
                observed.st_size,
                observed.st_mtime_ns,
                observed.st_ctime_ns,
            ) != identity:
                raise ValueError(f"input changed while it was read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _validate_counts(counts: Sequence[int], *, label: str) -> tuple[int, ...]:
    if (
        not isinstance(counts, (list, tuple))
        or len(counts) < 2
        or any(type(value) is not int or value < 0 for value in counts)
        or sum(counts) <= 0
    ):
        raise ValueError(f"{label} must be nonnegative integer token counts")
    return tuple(counts)


def cumulative_block_counts(
    earlier: Sequence[int], later: Sequence[int]
) -> tuple[int, ...]:
    """Subtract two cumulative count vectors without accepting drift."""

    before = _validate_counts(earlier, label="earlier cumulative counts")
    after = _validate_counts(later, label="later cumulative counts")
    if len(before) != len(after):
        raise ValueError("cumulative count vectors have different vocabulary sizes")
    block = tuple(right - left for left, right in zip(before, after, strict=True))
    if any(value < 0 for value in block) or sum(block) <= 0:
        raise ValueError("later cumulative counts do not contain the earlier prefix")
    return block


def _mixture_probabilities(
    training_counts: Sequence[int], weight: float
) -> tuple[float, ...]:
    counts = _validate_counts(training_counts, label="training counts")
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise ValueError("uniform mixture weight must be a real number")
    weight = float(weight)
    if not math.isfinite(weight) or not 0.0 < weight < 1.0:
        raise ValueError("uniform mixture weight must lie strictly between 0 and 1")
    total = sum(counts)
    vocabulary_size = len(counts)
    probabilities = tuple(
        (1.0 - weight) * count / total + weight / vocabulary_size for count in counts
    )
    if not all(math.isfinite(value) and value > 0.0 for value in probabilities):
        raise RuntimeError("mixture did not yield a finite full-support prior")
    return probabilities


def _heldout_nll(
    training_counts: Sequence[int], heldout_counts: Sequence[int], weight: float
) -> float:
    probabilities = _mixture_probabilities(training_counts, weight)
    heldout = _validate_counts(heldout_counts, label="held-out counts")
    if len(probabilities) != len(heldout):
        raise ValueError("training and held-out vocabularies differ")
    return -math.fsum(
        count * math.log(probabilities[token_id])
        for token_id, count in enumerate(heldout)
        if count
    ) / sum(heldout)


def _maximum_likelihood_weight(
    training_counts: Sequence[int], heldout_counts: Sequence[int]
) -> float:
    """Find the held-out unigram MLE for the one-dimensional mixture."""

    training = _validate_counts(training_counts, label="training counts")
    heldout = _validate_counts(heldout_counts, label="held-out counts")
    if len(training) != len(heldout):
        raise ValueError("training and held-out vocabularies differ")
    total = sum(training)
    vocabulary_size = len(training)
    frequencies = tuple(count / total for count in training)

    def derivative(weight: float) -> float:
        return math.fsum(
            count
            * (1.0 / vocabulary_size - frequencies[token_id])
            / ((1.0 - weight) * frequencies[token_id] + weight / vocabulary_size)
            for token_id, count in enumerate(heldout)
            if count
        )

    lower = 1e-12
    upper = 1.0 - 1e-12
    lower_derivative = derivative(lower)
    upper_derivative = derivative(upper)
    if lower_derivative <= 0.0:
        return lower
    if upper_derivative >= 0.0:
        return upper
    for _ in range(128):
        midpoint = (lower + upper) / 2.0
        if derivative(midpoint) > 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def analyze_block(
    training_counts: Sequence[int],
    heldout_counts: Sequence[int],
    mixture_grid: Sequence[float] = MIXTURE_GRID,
) -> dict[str, Any]:
    """Evaluate one ordered training-only block under a fixed prefix prior."""

    training = _validate_counts(training_counts, label="training counts")
    heldout = _validate_counts(heldout_counts, label="held-out counts")
    if len(training) != len(heldout):
        raise ValueError("training and held-out vocabularies differ")
    weights = tuple(float(value) for value in mixture_grid)
    if (
        not weights
        or weights != tuple(sorted(set(weights)))
        or any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in weights)
    ):
        raise ValueError(
            "mixture grid must be unique, increasing, finite, and interior"
        )
    if (
        CURRENT_UNIFORM_MIXTURE_WEIGHT not in weights
        or CANDIDATE_UNIFORM_MIXTURE_WEIGHT not in weights
    ):
        raise ValueError("mixture grid must include the current and candidate weights")

    unseen_ids = {index for index, count in enumerate(training) if count == 0}
    heldout_total = sum(heldout)
    heldout_unseen = sum(heldout[index] for index in unseen_ids)
    rows = []
    for weight in weights:
        probabilities = _mixture_probabilities(training, weight)
        nll = _heldout_nll(training, heldout, weight)
        rows.append(
            {
                "uniform_mixture_weight": weight,
                "heldout_content_token_nll_nats": nll,
                "heldout_content_token_perplexity": math.exp(nll),
                "training_unseen_stationary_mass": math.fsum(
                    probabilities[index] for index in unseen_ids
                ),
            }
        )
    by_weight = {row["uniform_mixture_weight"]: row for row in rows}
    optimum = _maximum_likelihood_weight(training, heldout)
    return {
        "heldout_content_tokens": heldout_total,
        "heldout_observed_token_types": sum(value > 0 for value in heldout),
        "training_unseen_token_types": len(unseen_ids),
        "heldout_new_token_types": sum(heldout[index] > 0 for index in unseen_ids),
        "heldout_tokens_from_training_unseen_types": heldout_unseen,
        "heldout_training_unseen_token_fraction": heldout_unseen / heldout_total,
        "continuous_maximum_likelihood_uniform_mixture_weight": optimum,
        "continuous_maximum_likelihood_nll_nats": _heldout_nll(
            training, heldout, optimum
        ),
        "mixture_grid": rows,
        "candidate_minus_current_nll_nats": (
            by_weight[CANDIDATE_UNIFORM_MIXTURE_WEIGHT][
                "heldout_content_token_nll_nats"
            ]
            - by_weight[CURRENT_UNIFORM_MIXTURE_WEIGHT][
                "heldout_content_token_nll_nats"
            ]
        ),
    }


def recommend_candidate_weight(
    discovery: dict[str, Any], retrospective_replication: dict[str, Any]
) -> dict[str, Any]:
    """Apply the disclosed conservative engineering rule to two blocks."""

    try:
        optima = [
            float(discovery["continuous_maximum_likelihood_uniform_mixture_weight"]),
            float(
                retrospective_replication[
                    "continuous_maximum_likelihood_uniform_mixture_weight"
                ]
            ),
        ]
        deltas = [
            float(discovery["candidate_minus_current_nll_nats"]),
            float(retrospective_replication["candidate_minus_current_nll_nats"]),
        ]
        unseen_counts = [
            int(discovery["heldout_tokens_from_training_unseen_types"]),
            int(retrospective_replication["heldout_tokens_from_training_unseen_types"]),
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("block analyses are malformed") from error
    if not all(math.isfinite(value) for value in (*optima, *deltas)) or any(
        value <= 0 for value in unseen_counts
    ):
        raise ValueError("block analyses lack finite unseen-token evidence")
    optima_in_guardrail = all(0.0001 <= value <= 0.0003 for value in optima)
    candidate_improves_both = all(value < 0.0 for value in deltas)
    selected = (
        CANDIDATE_UNIFORM_MIXTURE_WEIGHT
        if optima_in_guardrail and candidate_improves_both
        else CURRENT_UNIFORM_MIXTURE_WEIGHT
    )
    return {
        "status": "training_only_retrospective_engineering_recommendation",
        "current_uniform_mixture_weight": CURRENT_UNIFORM_MIXTURE_WEIGHT,
        "candidate_uniform_mixture_weight": CANDIDATE_UNIFORM_MIXTURE_WEIGHT,
        "recommended_uniform_mixture_weight": selected,
        "both_block_optima_within_0_0001_to_0_0003": optima_in_guardrail,
        "candidate_nll_strictly_better_than_current_on_both_blocks": (
            candidate_improves_both
        ),
        "block_optima": optima,
        "candidate_minus_current_nll_by_block": deltas,
        "selection_rule": (
            "recommend 0.0002 only when both ordered-block continuous optima lie "
            "in [0.0001, 0.0003], both blocks contain tokens absent from the "
            "first-10000 prefix, and 0.0002 has lower unigram NLL than 0.01 on "
            "both; otherwise retain 0.01"
        ),
        "qualification": (
            "The rule was formalized after exploratory inspection of these training "
            "blocks. It is suitable only as disclosed pilot hyperparameter "
            "engineering and is not confirmatory molecular-generation evidence."
        ),
    }


def _validate_base_frequency_artifact(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("base frequency artifact root must be an object")
    counts = _validate_counts(
        value.get("counts_by_token_id"), label="base frequency counts"
    )
    dataset = value.get("dataset")
    tokenizer = value.get("tokenizer")
    if (
        value.get("schema_version") != 1
        or value.get("example_count") != PREFIX_CHECKPOINTS[0]
        or value.get("content_token_count") != sum(counts)
        or len(counts) != MODEL_VOCAB_SIZE
        or not isinstance(dataset, dict)
        or not isinstance(tokenizer, dict)
        or not SHA256_PATTERN.fullmatch(
            str(dataset.get("ordered_safe_text_sha256", ""))
        )
        or tokenizer.get("base_vocab_size") != MODEL_VOCAB_SIZE
        or tokenizer.get("special_token_ids") != [0, 1, 2, 3, 4]
    ):
        raise ValueError("base frequency artifact identity is invalid")
    return value


def collect_prefix_checkpoints(batch_size: int) -> dict[int, dict[str, Any]]:
    """Stream and tokenize 30,000 pinned rows once, recording exact prefixes."""

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch size must be a positive integer")
    from datasets import load_dataset

    from genmol.utils.utils_data import (
        SAFE_GPT_DATASET_REVISION,
        SAFE_GPT_REPO_ID,
        SAFE_GPT_TOKENIZER_REVISION,
        SAFE_GPT_TOKENIZER_SHA256,
        get_tokenizer,
    )

    tokenizer = get_tokenizer()
    if tokenizer.vocab_size != MODEL_VOCAB_SIZE or sorted(
        int(value) for value in tokenizer.all_special_ids
    ) != [0, 1, 2, 3, 4]:
        raise PriorFloorAuditError("live tokenizer identity is unexpected")
    dataset = load_dataset(
        SAFE_GPT_REPO_ID,
        revision=SAFE_GPT_DATASET_REVISION,
        streaming=True,
        split="train",
    )
    iterator = iter(dataset)
    counts = [0] * MODEL_VOCAB_SIZE
    content_tokens = 0
    text_digest = hashlib.sha256()
    special_ids = set(int(value) for value in tokenizer.all_special_ids)
    pending: list[str] = []
    checkpoints: dict[int, dict[str, Any]] = {}

    def consume_pending() -> None:
        nonlocal content_tokens
        if not pending:
            return
        encoded = tokenizer(
            pending,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
        )["input_ids"]
        if len(encoded) != len(pending):
            raise PriorFloorAuditError("tokenizer changed the batch cardinality")
        for row in encoded:
            content = [
                int(token_id)
                for token_id in row
                if int(token_id) not in special_ids and int(token_id) < MODEL_VOCAB_SIZE
            ]
            if not content:
                raise PriorFloorAuditError("tokenization yielded an empty content row")
            content_tokens += len(content)
            for token_id in content:
                counts[token_id] += 1
        pending.clear()

    for row_number in range(1, PREFIX_CHECKPOINTS[-1] + 1):
        try:
            record = next(iterator)
        except StopIteration as error:
            raise PriorFloorAuditError(
                f"dataset ended before row {PREFIX_CHECKPOINTS[-1]}"
            ) from error
        if not isinstance(record, dict):
            raise PriorFloorAuditError(f"row {row_number} is not a mapping")
        value = record.get("input") or record.get("safe")
        if not isinstance(value, str) or not value:
            raise PriorFloorAuditError(f"row {row_number} has no nonempty SAFE text")
        payload = value.encode("utf-8")
        text_digest.update(len(payload).to_bytes(8, "big"))
        text_digest.update(payload)
        pending.append(value)
        if len(pending) == batch_size or row_number in PREFIX_CHECKPOINTS:
            consume_pending()
        if row_number in PREFIX_CHECKPOINTS:
            checkpoints[row_number] = {
                "example_count": row_number,
                "content_token_count": content_tokens,
                "observed_token_types": sum(value > 0 for value in counts),
                "unobserved_token_types": sum(value == 0 for value in counts),
                "ordered_safe_text_sha256": text_digest.copy().hexdigest(),
                "counts_by_token_id": list(counts),
            }
    if pending or tuple(checkpoints) != PREFIX_CHECKPOINTS:
        raise PriorFloorAuditError(
            "prefix checkpoint collection did not finish exactly"
        )
    checkpoints[PREFIX_CHECKPOINTS[-1]]["stream_identity"] = {
        "repo_id": SAFE_GPT_REPO_ID,
        "dataset_revision": SAFE_GPT_DATASET_REVISION,
        "split": "train",
        "tokenizer_revision": SAFE_GPT_TOKENIZER_REVISION,
        "tokenizer_json_sha256": SAFE_GPT_TOKENIZER_SHA256,
        "base_vocab_size": MODEL_VOCAB_SIZE,
        "special_token_ids": sorted(special_ids),
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
    }
    return checkpoints


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
        raise PriorFloorAuditError("source worktree must be clean before the audit")
    head = _git("rev-parse", "HEAD").stdout.strip()
    upstream = _git("rev-parse", "@{upstream}").stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", head) or head != upstream:
        raise PriorFloorAuditError("audit source revision must equal its upstream")
    return head


def source_inputs(commit: str) -> dict[str, dict[str, Any]]:
    records = {}
    for relative_path in SOURCE_PATHS:
        payload = _stable_file_bytes(REPOSITORY_ROOT / relative_path)
        blob = _git("show", f"{commit}:{relative_path.as_posix()}", text=False).stdout
        if payload != blob:
            raise PriorFloorAuditError(
                f"working bytes differ from Git blob: {relative_path}"
            )
        records[relative_path.as_posix()] = {
            "sha256": _sha256(payload),
            "size_bytes": len(payload),
            "git_blob_verified": True,
        }
    return records


def build_audit(
    *,
    commit: str,
    source_records: dict[str, dict[str, Any]],
    base_frequency: dict[str, Any],
    checkpoints: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("commit must be a full lowercase SHA-1")
    base_counts = _validate_counts(
        base_frequency["counts_by_token_id"], label="base frequency counts"
    )
    if tuple(checkpoints) != PREFIX_CHECKPOINTS:
        raise ValueError("stream checkpoints are incomplete or unordered")
    first = checkpoints[10_000]
    if (
        first["counts_by_token_id"] != list(base_counts)
        or first["content_token_count"] != base_frequency["content_token_count"]
        or first["ordered_safe_text_sha256"]
        != base_frequency["dataset"]["ordered_safe_text_sha256"]
    ):
        raise PriorFloorAuditError(
            "live first-10000 prefix does not reproduce the frozen frequency artifact"
        )
    discovery_counts = cumulative_block_counts(
        checkpoints[10_000]["counts_by_token_id"],
        checkpoints[20_000]["counts_by_token_id"],
    )
    retrospective_replication_counts = cumulative_block_counts(
        checkpoints[20_000]["counts_by_token_id"],
        checkpoints[30_000]["counts_by_token_id"],
    )
    discovery = analyze_block(base_counts, discovery_counts)
    retrospective_replication = analyze_block(
        base_counts, retrospective_replication_counts
    )
    recommendation = recommend_candidate_weight(discovery, retrospective_replication)
    return {
        "schema_version": 1,
        "purpose": "CPU-only empirical-UDLM uniform-floor training-data audit",
        "claim_scope": (
            "This audit measures unigram fit on ordered SAFE training blocks. It "
            "does not train a denoiser, generate or score molecules, rank generators, "
            "or establish that UDLM beats GenMol."
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {"commit": commit, "upstream": commit, "dirty": False},
        "inputs": {
            "source_files": source_records,
            "base_frequency_artifact": {
                "relative_path": BASE_FREQUENCY_RELATIVE_PATH.as_posix(),
                "sha256": BASE_FREQUENCY_SHA256,
                "canonical_sha256": _canonical_json_sha256(base_frequency),
            },
        },
        "data_use": {
            "prior_estimation_rows": [1, 10_000],
            "exploratory_rows": [10_001, 20_000],
            "retrospective_replication_rows": [20_001, 30_000],
            "split": "training",
            "formal_preregistration_before_data_access": False,
            "final_generation_seeds_or_metrics_used": False,
        },
        "definitions": {
            "stationary_prior": (
                "pi_j(w)=(1-w)c_j/sum_k(c_k)+w/V over all V=1880 token IDs"
            ),
            "block_nll": (
                "negative mean log pi_j(w) over content tokens in the stated "
                "ordered training block"
            ),
            "continuous_optimum": (
                "the unique interior root of the held-out log-likelihood derivative, "
                "found by 128 deterministic bisection iterations"
            ),
        },
        "stream_checkpoints": {str(rows): value for rows, value in checkpoints.items()},
        "block_analyses": {
            "rows_10001_20000_exploratory": discovery,
            "rows_20001_30000_retrospective_replication": (retrospective_replication),
        },
        "recommendation": recommendation,
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


def _write_exclusive(path: Path, value: dict[str, Any]) -> tuple[int, str]:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
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
    return len(payload), _sha256(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    output = _bind_output(args.output)
    commit = require_clean_pushed_commit()
    sources = source_inputs(commit)
    base_payload = _stable_file_bytes(BASE_FREQUENCY_PATH)
    if _sha256(base_payload) != BASE_FREQUENCY_SHA256:
        raise PriorFloorAuditError("base frequency raw SHA-256 is unexpected")
    base = _validate_base_frequency_artifact(
        strict_json_loads(base_payload, label="base frequency artifact")
    )
    checkpoints = collect_prefix_checkpoints(args.batch_size)
    result = build_audit(
        commit=commit,
        source_records=sources,
        base_frequency=base,
        checkpoints=checkpoints,
    )
    if require_clean_pushed_commit() != commit or source_inputs(commit) != sources:
        raise PriorFloorAuditError("source changed while the audit was running")
    size_bytes, digest = _write_exclusive(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "size_bytes": size_bytes,
                "sha256": digest,
                "source_commit": commit,
                "recommendation": result["recommendation"],
                "block_analyses": result["block_analyses"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

"""Audit the pinned SAFE training stream's token-frequency geometry on CPU.

This diagnostic does not train a model. It quantifies how unlike the observed
SAFE target distribution the faithful 1,880-way UDLM uniform prior is.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from itertools import islice
from pathlib import Path
from typing import Iterable, Sequence

from datasets import load_dataset

from genmol.utils.utils_data import (
    SAFE_GPT_DATASET_REVISION,
    SAFE_GPT_REPO_ID,
    SAFE_GPT_TOKENIZER_REVISION,
    SAFE_GPT_TOKENIZER_SHA256,
    get_tokenizer,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODEL_VOCAB_SIZE = 1_880
MAX_SEQUENCE_LENGTH = 256
EXOTIC_ID_TAIL_START = 1_680
COVERAGE_LEVELS = (0.99, 0.999, 0.9999)


def uniform_hit_probability(category_count: int, sequence_length: int, total: int) -> float:
    """Probability at least one iid uniform draw lands in a category subset."""
    if not 0 <= category_count <= total or total <= 0 or sequence_length < 0:
        raise ValueError("invalid uniform probability arguments")
    return -math.expm1(sequence_length * math.log1p(-category_count / total))


def types_for_coverage(counts: Sequence[int], coverage: float) -> int:
    if not 0 < coverage <= 1:
        raise ValueError("coverage must lie in (0, 1]")
    total = sum(counts)
    if total <= 0:
        raise ValueError("counts must contain at least one token")
    target = coverage * total
    cumulative = 0
    for index, count in enumerate(sorted(counts, reverse=True), start=1):
        cumulative += count
        if cumulative >= target:
            return index
    return len(counts)


def summarize_counts(
    counts: Sequence[int],
    content_lengths: Sequence[int],
    id_to_token: Sequence[str],
    special_token_ids: Sequence[int],
) -> dict:
    if len(counts) != MODEL_VOCAB_SIZE or len(id_to_token) != MODEL_VOCAB_SIZE:
        raise ValueError("audit expects the GenMol 1,880-state vocabulary")
    if not content_lengths:
        raise ValueError("at least one sequence is required")
    total = sum(counts)
    if total <= 0:
        raise ValueError("at least one content token is required")

    probabilities = [count / total for count in counts if count]
    entropy_nats = -sum(probability * math.log(probability) for probability in probabilities)
    median_length = int(statistics.median_low(content_lengths))
    ranked = sorted(range(len(counts)), key=lambda token_id: (-counts[token_id], token_id))
    observed_ranked = [token_id for token_id in ranked if counts[token_id]]

    def token_record(token_id: int) -> dict:
        return {
            "token_id": token_id,
            "token": id_to_token[token_id],
            "count": counts[token_id],
            "frequency": counts[token_id] / total,
        }

    tail_count = sum(counts[EXOTIC_ID_TAIL_START:])
    special_count = len(set(special_token_ids) & set(range(MODEL_VOCAB_SIZE)))
    return {
        "example_count": len(content_lengths),
        "content_token_count": total,
        "content_length": {
            "minimum": min(content_lengths),
            "median_low": median_length,
            "mean": statistics.fmean(content_lengths),
            "maximum": max(content_lengths),
        },
        "observed_token_types": sum(count > 0 for count in counts),
        "unobserved_token_types": sum(count == 0 for count in counts),
        "empirical_entropy_nats": entropy_nats,
        "empirical_entropy_bits": entropy_nats / math.log(2),
        "empirical_perplexity": math.exp(entropy_nats),
        "types_needed_for_coverage": {
            f"{coverage:.4f}": types_for_coverage(counts, coverage)
            for coverage in COVERAGE_LEVELS
        },
        "highest_frequency_tokens": [token_record(token_id) for token_id in ranked[:25]],
        "lowest_frequency_observed_tokens": [
            token_record(token_id)
            for token_id in sorted(
                observed_ranked,
                key=lambda token_id: (counts[token_id], token_id),
            )[:25]
        ],
        "id_tail_1680_1879": {
            "category_count": MODEL_VOCAB_SIZE - EXOTIC_ID_TAIL_START,
            "empirical_count": tail_count,
            "empirical_mass": tail_count / total,
            "uniform_per_token_mass": (
                MODEL_VOCAB_SIZE - EXOTIC_ID_TAIL_START
            )
            / MODEL_VOCAB_SIZE,
            "uniform_sequence_hit_probability_at_median_length": (
                uniform_hit_probability(
                    MODEL_VOCAB_SIZE - EXOTIC_ID_TAIL_START,
                    median_length,
                    MODEL_VOCAB_SIZE,
                )
            ),
        },
        "tokenizer_control_symbols": {
            "category_count": special_count,
            "uniform_per_token_mass": special_count / MODEL_VOCAB_SIZE,
            "uniform_sequence_hit_probability_at_median_length": (
                uniform_hit_probability(
                    special_count,
                    median_length,
                    MODEL_VOCAB_SIZE,
                )
            ),
        },
        "counts_by_token_id": list(counts),
    }


def _batched(values: Iterable[str], batch_size: int) -> Iterable[list[str]]:
    batch = []
    for value in values:
        batch.append(value)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def audit_training_prefix(sample_count: int, batch_size: int = 256) -> dict:
    if sample_count <= 0 or batch_size <= 0:
        raise ValueError("sample-count and batch-size must be positive")
    tokenizer = get_tokenizer()
    if tokenizer.vocab_size != MODEL_VOCAB_SIZE:
        raise RuntimeError(
            f"expected tokenizer base vocabulary {MODEL_VOCAB_SIZE}, got "
            f"{tokenizer.vocab_size}"
        )
    dataset = load_dataset(
        SAFE_GPT_REPO_ID,
        revision=SAFE_GPT_DATASET_REVISION,
        streaming=True,
        split="train",
    )
    records = islice(dataset, sample_count)
    text_digest = hashlib.sha256()

    def texts():
        observed = 0
        for record in records:
            value = record.get("input") or record.get("safe")
            if not isinstance(value, str) or not value:
                raise RuntimeError(f"invalid SAFE row at prefix index {observed}")
            payload = value.encode("utf-8")
            text_digest.update(len(payload).to_bytes(8, "big"))
            text_digest.update(payload)
            observed += 1
            yield value
        if observed != sample_count:
            raise RuntimeError(f"dataset ended after {observed} of {sample_count} rows")

    counts = [0] * MODEL_VOCAB_SIZE
    content_lengths = []
    specials = set(tokenizer.all_special_ids)
    for batch in _batched(texts(), batch_size):
        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
        )["input_ids"]
        for row in encoded:
            content_ids = [
                int(token_id)
                for token_id in row
                if int(token_id) not in specials and int(token_id) < MODEL_VOCAB_SIZE
            ]
            if not content_ids:
                raise RuntimeError("tokenization produced an empty content sequence")
            content_lengths.append(len(content_ids))
            for token_id in content_ids:
                counts[token_id] += 1

    summary = summarize_counts(
        counts,
        content_lengths,
        [str(tokenizer.convert_ids_to_tokens(i)) for i in range(MODEL_VOCAB_SIZE)],
        tokenizer.all_special_ids,
    )
    summary.update(
        {
            "schema_version": 1,
            "purpose": "CPU-only token-frequency diagnostic; not benchmark evidence",
            "dataset": {
                "repo_id": SAFE_GPT_REPO_ID,
                "revision": SAFE_GPT_DATASET_REVISION,
                "split": "train",
                "selection": f"first {sample_count} streaming rows",
                "ordered_safe_text_sha256": text_digest.hexdigest(),
            },
            "tokenizer": {
                "repo_id": SAFE_GPT_REPO_ID,
                "revision": SAFE_GPT_TOKENIZER_REVISION,
                "tokenizer_json_sha256": SAFE_GPT_TOKENIZER_SHA256,
                "base_vocab_size": MODEL_VOCAB_SIZE,
                "special_token_ids": sorted(int(value) for value in specials),
            },
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "git_sha": subprocess.run(
                ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-count", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(REPOSITORY_ROOT):
        raise ValueError(f"output must be inside {REPOSITORY_ROOT}")
    if output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {output}; pass --force explicitly")
    result = audit_training_prefix(args.sample_count, args.batch_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps({key: value for key, value in result.items() if key != "counts_by_token_id"}, indent=2))


if __name__ == "__main__":
    main()

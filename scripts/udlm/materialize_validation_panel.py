"""Materialize a small, revision-pinned SAFE validation denoising panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

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


def _length_prefixed_digest(values: list[bytes]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


def validate_panel(panel: dict) -> None:
    rows = panel.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("panel must contain at least one row")
    if panel.get("sample_count") != len(rows):
        raise ValueError("sample_count does not match rows")
    metadata = panel.get("tokenizer", {})
    special_ids = set(metadata.get("special_token_ids", []))
    bos_id = metadata.get("bos_token_id")
    eos_id = metadata.get("eos_token_id")
    encoded_rows = []
    previous_source_index = -1
    for expected_index, row in enumerate(rows):
        source_index = row.get("source_index")
        if type(source_index) is not int or source_index <= previous_source_index:
            raise ValueError("source indices must be strictly increasing")
        previous_source_index = source_index
        ids = row.get("input_ids")
        if not isinstance(ids, list) or not 2 <= len(ids) <= MAX_SEQUENCE_LENGTH:
            raise ValueError(f"invalid token length at row {expected_index}")
        if ids[0] != bos_id or ids[-1] != eos_id:
            raise ValueError(f"missing BOS/EOS framing at row {expected_index}")
        if any(type(token_id) is not int or not 0 <= token_id < MODEL_VOCAB_SIZE for token_id in ids):
            raise ValueError(f"out-of-vocabulary ID at row {expected_index}")
        content_length = sum(token_id not in special_ids for token_id in ids)
        if row.get("content_length") != content_length or content_length <= 0:
            raise ValueError(f"content length mismatch at row {expected_index}")
        encoded_rows.append(json.dumps(ids, separators=(",", ":")).encode("utf-8"))
    actual_digest = _length_prefixed_digest(encoded_rows)
    if panel.get("ordered_token_ids_sha256") != actual_digest:
        raise ValueError("ordered token-ID digest mismatch")


def materialize_panel(sample_count: int) -> dict:
    if sample_count <= 0:
        raise ValueError("sample-count must be positive")
    tokenizer = get_tokenizer()
    if tokenizer.vocab_size != MODEL_VOCAB_SIZE:
        raise RuntimeError("unexpected tokenizer vocabulary size")
    dataset = load_dataset(
        SAFE_GPT_REPO_ID,
        revision=SAFE_GPT_DATASET_REVISION,
        streaming=True,
        split="validation",
    )
    rows = []
    safe_payloads = []
    id_payloads = []
    special_ids = set(int(value) for value in tokenizer.all_special_ids)
    skipped_invalid_source_indices = []
    for source_index, record in enumerate(dataset):
        value = record.get("input") or record.get("safe")
        if not isinstance(value, str) or not value:
            skipped_invalid_source_indices.append(source_index)
            continue
        input_ids = [
            int(token_id)
            for token_id in tokenizer(
                value,
                add_special_tokens=True,
                truncation=True,
                max_length=MAX_SEQUENCE_LENGTH,
            )["input_ids"]
        ]
        content_length = sum(token_id not in special_ids for token_id in input_ids)
        safe_payload = value.encode("utf-8")
        id_payload = json.dumps(input_ids, separators=(",", ":")).encode("utf-8")
        safe_payloads.append(safe_payload)
        id_payloads.append(id_payload)
        rows.append(
            {
                "source_index": source_index,
                "input_ids": input_ids,
                "content_length": content_length,
                "safe_sha256": hashlib.sha256(safe_payload).hexdigest(),
            }
        )
        if len(rows) == sample_count:
            break
    if len(rows) != sample_count:
        raise RuntimeError(f"validation stream ended after {len(rows)} rows")
    panel = {
        "schema_version": 1,
        "purpose": "fixed held-out denoising panel; not generative benchmark evidence",
        "sample_count": sample_count,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "dataset": {
            "repo_id": SAFE_GPT_REPO_ID,
            "revision": SAFE_GPT_DATASET_REVISION,
            "split": "validation",
            "selection": f"first {sample_count} non-empty streaming rows",
            "skipped_invalid_source_indices": skipped_invalid_source_indices,
            "ordered_safe_text_sha256": _length_prefixed_digest(safe_payloads),
            "raw_safe_text_included": False,
        },
        "tokenizer": {
            "repo_id": SAFE_GPT_REPO_ID,
            "revision": SAFE_GPT_TOKENIZER_REVISION,
            "tokenizer_json_sha256": SAFE_GPT_TOKENIZER_SHA256,
            "base_vocab_size": MODEL_VOCAB_SIZE,
            "special_token_ids": sorted(special_ids),
            "bos_token_id": int(tokenizer.bos_token_id),
            "eos_token_id": int(tokenizer.eos_token_id),
            "pad_token_id": int(tokenizer.pad_token_id),
        },
        "ordered_token_ids_sha256": _length_prefixed_digest(id_payloads),
        "git_sha": subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "rows": rows,
    }
    validate_panel(panel)
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(REPOSITORY_ROOT):
        raise ValueError(f"output must be inside {REPOSITORY_ROOT}")
    if output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {output}; pass --force explicitly")
    panel = materialize_panel(args.sample_count)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(panel, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(
        json.dumps(
            {key: value for key, value in panel.items() if key != "rows"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

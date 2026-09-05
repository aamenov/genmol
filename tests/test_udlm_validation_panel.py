import copy
import hashlib
import json

import pytest

from scripts.udlm.materialize_validation_panel import validate_panel


def _panel():
    rows = [
        {
            "source_index": 0,
            "input_ids": [1, 27, 38, 2],
            "content_length": 2,
            "safe_sha256": "unused-in-structural-check",
        },
        {
            "source_index": 1,
            "input_ids": [1, 54, 2],
            "content_length": 1,
            "safe_sha256": "unused-in-structural-check",
        },
    ]
    digest = hashlib.sha256()
    for row in rows:
        value = json.dumps(row["input_ids"], separators=(",", ":")).encode()
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return {
        "sample_count": 2,
        "tokenizer": {
            "special_token_ids": [0, 1, 2, 3, 4],
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
        "ordered_token_ids_sha256": digest.hexdigest(),
        "rows": rows,
    }


def test_fixed_panel_structural_validation_accepts_framed_content():
    validate_panel(_panel())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda panel: panel["rows"][0].update(source_index=9), "increasing"),
        (lambda panel: panel["rows"][0].update(input_ids=[27, 2]), "BOS/EOS"),
        (lambda panel: panel["rows"][0].update(content_length=9), "content length"),
        (lambda panel: panel.update(ordered_token_ids_sha256="bad"), "digest"),
    ],
)
def test_fixed_panel_validation_rejects_corruption(mutation, message):
    panel = copy.deepcopy(_panel())
    mutation(panel)
    with pytest.raises(ValueError, match=message):
        validate_panel(panel)

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.udlm import update_notebook as updater


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = REPOSITORY_ROOT / "genmol_from_scratch.ipynb"


def _notebook() -> dict:
    return json.loads(SOURCE_NOTEBOOK.read_text())


def _write_notebook(path: Path, notebook: dict) -> None:
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")


def _cells_by_id(notebook: dict) -> dict[str, dict]:
    return {cell["id"]: cell for cell in notebook["cells"]}


def test_update_replaces_generated_cells_and_preserves_stages_20_6_through_20_8(
    tmp_path: Path,
) -> None:
    notebook = _notebook()
    before = _cells_by_id(notebook)
    preserved_before = {
        cell_id: copy.deepcopy(before[cell_id])
        for cell_id in updater.PRESERVED_STAGE20_TAGS_BY_ID
    }
    generated = updater.stage_cells()
    before[generated[0]["id"]]["source"] = "corrupt generated source\n"

    source = tmp_path / "source.ipynb"
    destination = tmp_path / "updated.ipynb"
    _write_notebook(source, notebook)
    updater.update_notebook(source, destination)

    updated = json.loads(destination.read_text())
    updated_by_id = _cells_by_id(updated)
    assert updated_by_id[generated[0]["id"]] == generated[0]
    assert {
        cell_id: updated_by_id[cell_id]
        for cell_id in updater.PRESERVED_STAGE20_TAGS_BY_ID
    } == preserved_before

    stage20_ids = [
        cell["id"]
        for cell in updated["cells"]
        if cell["id"].startswith(updater.STAGE_TAG_PREFIX)
    ]
    assert stage20_ids == [
        *(cell["id"] for cell in generated),
        *updater.PRESERVED_STAGE20_TAGS_BY_ID,
    ]


def test_update_rejects_unknown_stage20_tag(tmp_path: Path) -> None:
    notebook = _notebook()
    notebook["cells"].append(
        {
            "cell_type": "markdown",
            "id": "retired-stage-20-cell",
            "metadata": {"tags": ["stage-20-udlm-retired"]},
            "source": "retired content\n",
        }
    )
    source = tmp_path / "unknown.ipynb"
    _write_notebook(source, notebook)

    with pytest.raises(ValueError, match="unknown .*prefixed cell"):
        updater.update_notebook(source, tmp_path / "must-not-exist.ipynb")


def test_update_rejects_duplicate_cell_ids(tmp_path: Path) -> None:
    notebook = _notebook()
    notebook["cells"].append(copy.deepcopy(notebook["cells"][0]))
    source = tmp_path / "duplicate.ipynb"
    _write_notebook(source, notebook)

    with pytest.raises(ValueError, match="duplicate cell id"):
        updater.update_notebook(source, tmp_path / "must-not-exist.ipynb")


def test_update_is_byte_idempotent(tmp_path: Path) -> None:
    first = tmp_path / "first.ipynb"
    second = tmp_path / "second.ipynb"

    updater.update_notebook(SOURCE_NOTEBOOK, first)
    updater.update_notebook(first, second)

    assert second.read_bytes() == first.read_bytes()


def test_prior_floor_teaching_binds_retrospective_artifact_without_rewriting_history():
    notebook = _notebook()
    cells = _cells_by_id(notebook)
    markdown = "".join(cells["stage-20-udlm-prior-geometry"]["source"])
    code = "".join(cells["stage-20-udlm-prior-geometry-code"]["source"])

    for fragment in (
        "retrospective replication, not a preregistered confirmation",
        "historical/manual categorical configuration",
        "reviewed pilot launches only",
        "Does the lower unigram NLL mean",
    ):
        assert fragment in markdown
    for fragment in (
        "floor_selection_train_rows_10001_30000.json",
        "02908dafaf589ca9a49e560aa1eab470a18d6bfe616b781164784c489f54a9f1",
        "6424b323084358ea050ba22d7e13ef8d45962496",
        '"historical_manual_weight": 0.01',
        '"reviewed_pilot_weight": 0.0002',
    ):
        assert fragment in code
    compile(code, "stage-20-udlm-prior-geometry-code", "exec")


def test_generated_schedule_teaching_distinguishes_official_recipe(tmp_path: Path):
    destination = tmp_path / "updated.ipynb"
    updater.update_notebook(SOURCE_NOTEBOOK, destination)
    cells = _cells_by_id(json.loads(destination.read_text()))
    markdown = "".join(cells["stage-20-udlm-evidence"]["source"])

    assert "QM9 recipe uses 25,000 optimizer steps" in markdown
    assert "not the released UDLM QM9 schedule" in markdown
    assert "pilot hypothesis" in markdown
    assert "not an exact replay of the official recipe" in markdown


def test_generated_health_teaching_binds_exact_gate_and_later_diagnostic(
    tmp_path: Path,
):
    destination = tmp_path / "updated.ipynb"
    updater.update_notebook(SOURCE_NOTEBOOK, destination)
    cells = _cells_by_id(json.loads(destination.read_text()))
    markdown = "".join(cells["stage-20-udlm-evidence"]["source"])
    compact_markdown = " ".join(markdown.split())
    code = "".join(cells["stage-20-udlm-evidence-code"]["source"])
    all_markdown = "\n".join(
        "".join(cell["source"])
        for cell in cells.values()
        if cell["cell_type"] == "markdown"
    )

    for fragment in (
        "scripts/udlm/launch_health_panel.py",
        "scripts/udlm/validate_health_panel.py",
        "health-w{W}-e-{H}",
        "Both config materialization at $H$ and registry freezing at R0",
        "permits only screen authorization",
        "$B_{\\mathrm{eff}}=Wma_W=16$",
        "Only after each 1,000-update training receipt validates",
        "not the 10-update health-terminal receipt",
    ):
        assert fragment in compact_markdown
    for fragment in (
        '"launcher_variant_order": ["udlm", "schedule_uniform", "udlm_categorical"]',
        '"supported_world_sizes": [1, 2]',
        '"num_nodes": 1',
        '"optimizer_updates_each": 10',
        '"training_seed": 1',
        '"loader_workers": 1',
        '"micro_batch_size_per_process": 2',
        '"gradient_accumulation_by_world_size": {1: 8, 2: 4}',
        '"effective_global_batch_size": 16',
        '"vocabulary_size": 1880',
        '"exclude_special_tokens": False',
        '"scratch_mode": False',
        '"empirical_uniform_mix": 0.0002',
        '"checkpoint_project_relative_path"',
        '"checkpoint_size_bytes": 1396998679',
        "8d00aa47b02f64bf39ff6b0b2e786f213587366fc2c3d29712a00f3f84108dd6",
        '"max_utilization_percent": 10',
        '"min_free_memory_mib": 30000',
        '"normalized_evidence_schema_version": 1',
        '"successful_exit_receipt_schema_version": 5',
        '"terminal_e_receipt_required_before"',
        '"screen_config_materialization"',
        '"screen_registry_freeze"',
        '"candidate_lock": False',
        '"post_training_decode_diagnostic"',
        '"after_optimizer_updates_each": 1000',
        '"seed": 1100',
        '"requested": 32',
    ):
        assert fragment in code
    assert '"health_generation"' not in code
    compact_all_markdown = " ".join(all_markdown.split())
    assert "deterministic `health-w{W}-{r,s,e}-{H}` names" in compact_all_markdown
    assert "Only after those training runs, seed 1100 x 32 requests" in (
        compact_all_markdown
    )
    compile(code, "stage-20-udlm-evidence-code", "exec")

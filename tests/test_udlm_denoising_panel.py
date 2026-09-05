import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from genmol.diffusion import ContinuousUniformDiffusion
from scripts.udlm import evaluate_denoising_panel as evaluator
from scripts.udlm.evaluate_denoising_panel import (
    DEFAULT_FREQUENCIES,
    DEFAULT_PANEL,
    FROZEN_FREQUENCY_SHA256,
    FROZEN_PANEL_SHA256,
    SOURCE_INPUTS,
    _atomic_write_json,
    _load_json,
    _validate_device,
    apply_ema_weights,
    checkpoint_metadata,
    corruption_seed,
    evaluate_denoising_panel,
    load_verified_checkpoint,
    load_frozen_artifacts,
    runtime_provenance,
    sha256_file,
    source_provenance,
    validate_evaluation_inputs,
    validate_time_bins,
    verify_source_provenance,
)


def _ordered_token_digest(rows):
    digest = hashlib.sha256()
    for row in rows:
        value = json.dumps(row["input_ids"], separators=(",", ":")).encode()
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


def _toy_artifacts():
    rows = [
        {
            "source_index": 0,
            "input_ids": [1, 4, 5, 6, 7, 2],
            "content_length": 4,
            "safe_sha256": "0" * 64,
        },
        {
            "source_index": 3,
            "input_ids": [1, 7, 6, 5, 4, 7, 6, 2],
            "content_length": 6,
            "safe_sha256": "1" * 64,
        },
        {
            "source_index": 9,
            "input_ids": [1, 5, 5, 4, 6, 7, 7, 4, 2],
            "content_length": 7,
            "safe_sha256": "2" * 64,
        },
    ]
    tokenizer = {
        "repo_id": "toy-safe",
        "revision": "tokenizer-revision",
        "tokenizer_json_sha256": "a" * 64,
        "base_vocab_size": 8,
        "special_token_ids": [0, 1, 2, 3],
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 3,
    }
    panel = {
        "schema_version": 1,
        "purpose": "test panel",
        "sample_count": len(rows),
        "max_sequence_length": 256,
        "dataset": {
            "repo_id": "toy-safe",
            "revision": "dataset-revision",
            "split": "validation",
        },
        "tokenizer": tokenizer,
        "ordered_token_ids_sha256": _ordered_token_digest(rows),
        "rows": rows,
    }
    counts = [0, 0, 0, 0, 0, 5, 50, 5_000]
    frequencies = {
        "schema_version": 1,
        "purpose": "test frequencies",
        "example_count": 100,
        "content_token_count": sum(counts),
        "observed_token_types": sum(count > 0 for count in counts),
        "counts_by_token_id": counts,
        "dataset": {
            "repo_id": "toy-safe",
            "revision": "dataset-revision",
            "split": "train",
        },
        "tokenizer": copy.deepcopy(tokenizer),
    }
    return panel, frequencies


class _ToyUDLM(nn.Module):
    """A tiny injected denoiser whose top-1 prediction is the noisy token."""

    diffusion_type = "udlm"

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.mdlm = ContinuousUniformDiffusion(
            8,
            excluded_token_ids=(0, 1, 2, 3),
            sampling_eps=1e-3,
            noise_eps=0.1,
            antithetic_sampling=True,
        )

    def forward(self, input_ids, attention_mask=None, t=None):
        del attention_mask
        assert t is not None
        token_ids = torch.arange(8, device=input_ids.device)
        return -4.0 * (token_ids - input_ids.unsqueeze(-1)).abs() + self.anchor


def test_committed_frozen_artifacts_match_pinned_hashes_and_validate():
    panel, frequencies, provenance = load_frozen_artifacts()

    assert sha256_file(DEFAULT_PANEL) == FROZEN_PANEL_SHA256
    assert sha256_file(DEFAULT_FREQUENCIES) == FROZEN_FREQUENCY_SHA256
    assert panel["sample_count"] == 256
    assert frequencies["example_count"] == 10_000
    assert provenance["panel"]["sha256"] == FROZEN_PANEL_SHA256


def test_input_validation_rejects_cross_artifact_tokenizer_mismatch():
    panel, frequencies = _toy_artifacts()
    frequencies["tokenizer"]["revision"] = "different"

    with pytest.raises(ValueError, match="tokenizer mismatch for revision"):
        validate_evaluation_inputs(panel, frequencies)


def test_input_validation_rejects_bad_row_hash_and_inconsistent_counts():
    panel, frequencies = _toy_artifacts()
    bad_panel = copy.deepcopy(panel)
    bad_panel["rows"][0]["safe_sha256"] = "not-a-sha"
    with pytest.raises(ValueError, match="invalid SAFE hash"):
        validate_evaluation_inputs(bad_panel, frequencies)

    bad_frequencies = copy.deepcopy(frequencies)
    bad_frequencies["content_token_count"] += 1
    with pytest.raises(ValueError, match="content_token_count"):
        validate_evaluation_inputs(panel, bad_frequencies)


def test_time_and_seed_validation_is_strict_and_seed_is_stable():
    assert validate_time_bins((0.1, 0.5, 0.9)) == (0.1, 0.5, 0.9)
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_time_bins((0.5, 0.5))
    with pytest.raises(ValueError, match="strictly in"):
        validate_time_bins((0.0, 0.5))

    assert corruption_seed(7, 0.5, 3) == corruption_seed(7, 0.5, 3)
    assert corruption_seed(7, 0.5, 3) != corruption_seed(7, 0.5, 4)


def test_toy_evaluation_is_deterministic_and_preserves_metric_denominators():
    panel, frequencies = _toy_artifacts()
    first = evaluate_denoising_panel(
        _ToyUDLM(),
        panel,
        frequencies,
        time_bins=(0.35, 0.8),
        seed=123,
        batch_size=1,
        device="cpu",
    )
    second = evaluate_denoising_panel(
        _ToyUDLM(),
        panel,
        frequencies,
        time_bins=(0.35, 0.8),
        seed=123,
        batch_size=3,
        device="cpu",
    )

    assert first["content_tokens_per_time_bin"] == 17
    assert first["pooled_token_time_metrics"]["overall"][
        "production_loss_mean"
    ] == pytest.approx(
        second["pooled_token_time_metrics"]["overall"]["production_loss_mean"],
        rel=1e-6,
        abs=1e-8,
    )
    for first_bin, second_bin in zip(
        first["metrics_by_time"], second["metrics_by_time"]
    ):
        assert (
            first_bin["row_corruption_seeds_sha256"]
            == second_bin["row_corruption_seeds_sha256"]
        )
        assert (
            first_bin["corrupted_token_ids_sha256"]
            == second_bin["corrupted_token_ids_sha256"]
        )
        # The corruption grid is exact across batch partitions.  Numerical
        # equality of real neural-network outputs is intentionally not part of
        # the contract, so metric comparisons use a tolerance.
        assert first_bin["metrics"]["overall"][
            "production_loss_mean"
        ] == pytest.approx(
            second_bin["metrics"]["overall"]["production_loss_mean"],
            rel=1e-6,
            abs=1e-8,
        )

        overall = first_bin["metrics"]["overall"]
        changed = first_bin["metrics"]["by_observed_corruption"]["observed_changed"]
        unchanged = first_bin["metrics"]["by_observed_corruption"]["observed_unchanged"]
        assert overall["denominator_tokens"] == 17
        assert changed["denominator_tokens"] > 0
        assert unchanged["denominator_tokens"] > 0
        assert changed["denominator_tokens"] + unchanged["denominator_tokens"] == 17
        assert changed["clean_token_top1_accuracy"] == 0.0
        assert unchanged["clean_token_top1_accuracy"] == 1.0
        assert overall["production_loss_mean"] >= 0

        frequency_groups = first_bin["metrics"]["by_training_frequency"]
        assert (
            sum(group["denominator_tokens"] for group in frequency_groups.values())
            == 17
        )
        assert frequency_groups["unseen"]["denominator_tokens"] > 0
        assert frequency_groups["count_ge_1000"]["denominator_tokens"] > 0

    assert first["pooled_token_time_metrics"]["overall"]["denominator_tokens"] == 2 * 17
    assert first["corruption_grid_sha256"] == second["corruption_grid_sha256"]
    assert first["seed_protocol"]["corruption_device"] == "cpu"
    assert "not claimed" in first["seed_protocol"]["qualification"]
    assert "not an integrated" in first["estimator_scope"]


def test_evaluation_can_use_a_small_prefix_without_redefining_the_panel():
    panel, frequencies = _toy_artifacts()
    result = evaluate_denoising_panel(
        _ToyUDLM(),
        panel,
        frequencies,
        time_bins=(0.5,),
        seed=5,
        batch_size=2,
        max_rows=1,
    )

    assert result["rows_evaluated"] == 1
    assert result["row_selection"] == "first 1 rows of the frozen ordered panel"
    assert result["content_tokens_per_time_bin"] == 4


def test_checkpoint_metadata_records_exact_bytes_and_udlm_identity(tmp_path: Path):
    checkpoint_path = tmp_path / "tiny.ckpt"
    torch.save(
        {
            "global_step": 12,
            "epoch": 2,
            "hyper_parameters": {
                "config": {
                    "seed": 7,
                    "training": {
                        "diffusion": "udlm",
                        "init_from_mdlm_checkpoint": "source.ckpt",
                        "init_from_mdlm_ema": False,
                    },
                }
            },
        },
        checkpoint_path,
    )

    metadata = checkpoint_metadata(checkpoint_path)

    assert metadata["sha256"] == sha256_file(checkpoint_path)
    assert metadata["size_bytes"] == checkpoint_path.stat().st_size
    assert metadata["global_step"] == 12
    assert metadata["epoch"] == 2
    assert metadata["diffusion_type"] == "udlm"
    assert len(metadata["config_sha256"]) == 64
    assert metadata["training_initialization_declaration"] == {
        "init_from_mdlm_checkpoint": "source.ckpt",
        "init_from_mdlm_ema": False,
        "qualification": metadata["training_initialization_declaration"][
            "qualification"
        ],
    }


def test_json_loader_parses_the_same_bytes_it_hashes(tmp_path: Path, monkeypatch):
    artifact = tmp_path / "artifact.json"
    artifact.write_text('{"value": 1}')
    expected = hashlib.sha256(artifact.read_bytes()).hexdigest()

    def forbidden_reread(_self):
        raise AssertionError("JSON must not be reread after hashing")

    monkeypatch.setattr(Path, "read_text", forbidden_reread)
    value, observed = _load_json(artifact, expected)

    assert value == {"value": 1}
    assert observed == expected


def test_exact_uniform_backend_and_prior_identity_are_enforced():
    panel, frequencies = _toy_artifacts()
    result = evaluate_denoising_panel(
        _ToyUDLM(),
        panel,
        frequencies,
        time_bins=(0.5,),
        max_rows=1,
    )
    process = result["process"]
    assert process["backend"].endswith(".ContinuousUniformDiffusion")
    assert process["backend_exact_type_required"] is True
    assert process["prior"]["family"] == "uniform_over_diffusion_token_ids"
    assert process["prior"]["support_size"] == 4
    assert len(process["prior"]["identity_sha256"]) == 64

    class UnsupportedUniformSubclass(ContinuousUniformDiffusion):
        pass

    model = _ToyUDLM()
    model.mdlm = UnsupportedUniformSubclass(
        8,
        excluded_token_ids=(0, 1, 2, 3),
        sampling_eps=1e-3,
        noise_eps=0.1,
    )
    with pytest.raises(ValueError, match="exact uniform UDLM backend"):
        evaluate_denoising_panel(
            model,
            panel,
            frequencies,
            time_bins=(0.5,),
            max_rows=1,
        )


class _TinyBackboneHolder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1))


def _tiny_weight_inputs():
    model = _TinyBackboneHolder()
    state = {
        f"backbone.{name}": parameter.detach().clone()
        for name, parameter in model.backbone.named_parameters()
    }
    shadows = [
        torch.full_like(parameter, index + 1)
        for index, parameter in enumerate(model.backbone.parameters())
    ]
    return model, state, shadows


def test_ema_mapping_validates_every_name_shape_and_dtype_before_copy():
    model, state, shadows = _tiny_weight_inputs()
    provenance = apply_ema_weights(model, state, {"shadow_params": shadows})
    assert provenance["parameter_tensors"] == 4
    assert len(provenance["ordered_parameter_names_sha256"]) == 64
    for parameter, expected in zip(model.backbone.parameters(), shadows, strict=True):
        assert torch.equal(parameter, expected)

    for malformed in (
        shadows[:-1],
        [*shadows[:-1], torch.zeros(2, dtype=shadows[-1].dtype)],
        [*shadows[:-1], shadows[-1].double()],
    ):
        candidate, candidate_state, _candidate_shadows = _tiny_weight_inputs()
        before = [parameter.detach().clone() for parameter in candidate.parameters()]
        with pytest.raises(ValueError, match="EMA"):
            apply_ema_weights(
                candidate, candidate_state, {"shadow_params": malformed}
            )
        assert all(
            torch.equal(parameter, original)
            for parameter, original in zip(candidate.parameters(), before, strict=True)
        )

    candidate, candidate_state, candidate_shadows = _tiny_weight_inputs()
    candidate_state.pop("backbone.0.weight")
    with pytest.raises(ValueError, match="backbone.0.weight"):
        apply_ema_weights(
            candidate, candidate_state, {"shadow_params": candidate_shadows}
        )


def test_verified_checkpoint_loads_once_and_rejects_a_changed_snapshot(
    tmp_path: Path, monkeypatch
):
    checkpoint_path = tmp_path / "checkpoint.ckpt"
    checkpoint_path.write_bytes(b"stable-placeholder")
    load_calls = []

    def fake_load(path, **_kwargs):
        load_calls.append(path)
        return {"state_dict": {}}

    monkeypatch.setattr(torch, "load", fake_load)
    checkpoint, snapshot = load_verified_checkpoint(checkpoint_path)
    assert checkpoint == {"state_dict": {}}
    assert snapshot["sha256"] == hashlib.sha256(b"stable-placeholder").hexdigest()
    assert len(load_calls) == 1

    hashes = iter(("a" * 64, "b" * 64))
    monkeypatch.setattr(evaluator, "sha256_file", lambda _path: next(hashes))
    load_calls.clear()
    with pytest.raises(RuntimeError, match="changed while it was being loaded"):
        load_verified_checkpoint(checkpoint_path)
    assert len(load_calls) == 1


def test_cli_device_validation_is_explicitly_cpu_only():
    _validate_device("cpu")
    with pytest.raises(ValueError, match="CPU-only"):
        _validate_device("cuda:0")
    with pytest.raises(ValueError, match="CPU-only"):
        _validate_device("mps")


def test_source_provenance_distinguishes_unknown_git_and_is_rechecked(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / "implementation.py"
    source.write_text("before\n")
    monkeypatch.setattr(evaluator, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(evaluator, "SOURCE_INPUTS", (Path("implementation.py"),))
    monkeypatch.setattr(evaluator, "_git_output", lambda _arguments: None)

    provenance = source_provenance()
    assert provenance["git_worktree_state"] == "unknown"
    assert provenance["git_dirty"] is None
    verify_source_provenance(provenance)

    source.write_text("after\n")
    with pytest.raises(RuntimeError, match="implementation inputs changed"):
        verify_source_provenance(provenance)


def test_runtime_and_source_inputs_cover_execution_and_training_dependencies():
    assert Path("src/genmol/utils/ema.py") in SOURCE_INPUTS
    assert Path("scripts/udlm/launch_train_pilot.py") in SOURCE_INPUTS
    assert Path("scripts/train.py") in SOURCE_INPUTS
    assert Path("configs/base.yaml") in SOURCE_INPUTS
    assert Path("configs/udlm.yaml") in SOURCE_INPUTS
    runtime = runtime_provenance()
    assert set(runtime["packages"]) == {
        "lightning",
        "transformers",
        "tokenizers",
        "omegaconf",
        "hydra-core",
        "bionemo-moco",
    }
    assert "CPU-only" in runtime["evaluation_device_policy"]


def test_atomic_json_uses_unique_private_temps_and_never_clobbers(
    tmp_path: Path, monkeypatch
):
    # The production guard confines reports to the worktree. Isolate this unit
    # test to its pytest-owned directory while retaining all write semantics.
    monkeypatch.setattr(evaluator, "REPOSITORY_ROOT", tmp_path)
    created = []
    real_mkstemp = evaluator.tempfile.mkstemp

    def recording_mkstemp(**kwargs):
        descriptor, name = real_mkstemp(**kwargs)
        created.append(name)
        return descriptor, name

    monkeypatch.setattr(evaluator.tempfile, "mkstemp", recording_mkstemp)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _atomic_write_json(first, {"value": 1}, force=False)
    _atomic_write_json(second, {"value": 2}, force=False)
    assert len(created) == len(set(created)) == 2
    assert json.loads(first.read_text()) == {"value": 1}
    assert all(not Path(name).exists() for name in created)

    with pytest.raises(FileExistsError, match="pass --force"):
        _atomic_write_json(first, {"value": 3}, force=False)
    assert json.loads(first.read_text()) == {"value": 1}
    _atomic_write_json(first, {"value": 3}, force=True)
    assert json.loads(first.read_text()) == {"value": 3}

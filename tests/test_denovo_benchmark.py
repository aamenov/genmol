from __future__ import annotations

import csv
import hashlib
import json
import pickle
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.exps.denovo import benchmark


def test_benchmark_summary_schema_includes_mandatory_inference_weights() -> None:
    assert benchmark.SCHEMA_VERSION == 7


def _tiny_sa_artifact(root: Path) -> tuple[Path, str, int]:
    path = root / "oracle" / "fpscores.pkl"
    path.parent.mkdir(parents=True)
    payload = pickle.dumps([[1.25, 7, 9], [-0.5, 11]], protocol=4)
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest(), len(payload)


def _score(values):
    return [0.7 for _ in values]


def _sa(values):
    return [3.0 for _ in values]


def _diversity(values):
    return len(values) / 10


def _categorical_checkpoint_parts(
    variant: str,
    *,
    exclude_special_tokens: bool = False,
    mixture_weight: float = 0.01,
):
    torch = pytest.importorskip("torch")
    full_vocab_size = 1880
    excluded = (
        list(benchmark.SAFE_GPT_SPECIAL_TOKEN_IDS) if exclude_special_tokens else []
    )
    active_ids = [
        token_id for token_id in range(full_vocab_size) if token_id not in excluded
    ]
    if variant == "schedule_uniform":
        probabilities = torch.full(
            (len(active_ids),), 1.0 / len(active_ids), dtype=torch.float64
        )
        probabilities /= probabilities.sum()
        frequency = {
            field: None
            for field in (
                "frequency_artifact_path",
                "frequency_artifact_sha256",
                "frequency_artifact_schema_version",
                "frequency_example_count",
                "frequency_content_token_count",
                "frequency_active_token_count",
                "frequency_dataset_repo_id",
                "frequency_dataset_revision",
                "frequency_dataset_split",
                "frequency_dataset_selection",
                "frequency_ordered_text_sha256",
                "frequency_implementation_git_sha",
            )
        }
        recorded_mix = None
    else:
        artifact, counts = benchmark._load_empirical_frequency_counts()
        active_count = sum(counts[token_id] for token_id in active_ids)
        empirical = [counts[token_id] / active_count for token_id in active_ids]
        values = [
            (1.0 - mixture_weight) * value + mixture_weight / len(active_ids)
            for value in empirical
        ]
        probabilities = torch.tensor(values, dtype=torch.float64)
        probabilities /= probabilities.sum()
        frequency = {
            "frequency_artifact_path": (
                benchmark.EMPIRICAL_FREQUENCY_RELATIVE_PATH.as_posix()
            ),
            "frequency_artifact_sha256": benchmark.EMPIRICAL_FREQUENCY_SHA256,
            "frequency_artifact_schema_version": 1,
            "frequency_example_count": 10_000,
            "frequency_content_token_count": 517_090,
            "frequency_active_token_count": active_count,
            "frequency_dataset_repo_id": benchmark.TOKENIZER_REQUESTED_IDENTIFIER,
            "frequency_dataset_revision": benchmark.SAFE_GPT_DATASET_REVISION,
            "frequency_dataset_split": "train",
            "frequency_dataset_selection": "first 10000 streaming rows",
            "frequency_ordered_text_sha256": (
                benchmark.EMPIRICAL_FREQUENCY_ORDERED_TEXT_SHA256
            ),
            "frequency_implementation_git_sha": (
                benchmark.EMPIRICAL_FREQUENCY_IMPLEMENTATION_GIT_SHA
            ),
        }
        recorded_mix = mixture_weight
        assert artifact["content_token_count"] == active_count

    identity = benchmark.UDLM_PRIOR_VARIANT_IDENTITIES[variant]
    metadata = {
        "schema_version": 1,
        "variant": variant,
        **identity,
        "full_vocab_size": full_vocab_size,
        "active_vocab_size": len(active_ids),
        "excluded_token_ids": excluded,
        "sampling_eps": 1e-3,
        "noise_eps": 1e-3,
        "antithetic_sampling": True,
        "active_token_ids_sha256": benchmark._canonical_numeric_sequence_sha256(
            active_ids
        ),
        "stationary_probs_sha256": benchmark._canonical_numeric_sequence_sha256(
            probabilities.tolist()
        ),
        "uniform_mixture_weight": recorded_mix,
        **frequency,
        "tokenizer_repo_id": benchmark.TOKENIZER_REQUESTED_IDENTIFIER,
        "tokenizer_revision": benchmark.SAFE_GPT_TOKENIZER_REVISION,
        "tokenizer_json_sha256": benchmark.SAFE_GPT_TOKENIZER_SHA256,
    }
    diffusion_ids = torch.tensor(active_ids, dtype=torch.long)
    mapping = torch.full((full_vocab_size,), -1, dtype=torch.long)
    mapping[diffusion_ids] = torch.arange(len(active_ids), dtype=torch.long)
    state = {
        "mdlm.diffusion_token_ids": diffusion_ids,
        "mdlm.token_to_diffusion_index": mapping,
        "mdlm.stationary_probs": probabilities,
    }
    config = {
        "model": {"vocab_size": full_vocab_size},
        "training": {
            "diffusion": "udlm",
            "sampling_eps": 1e-3,
            "antithetic_sampling": True,
            "udlm": {
                "inference_eps": 1e-5,
                "noise_eps": 1e-3,
                "exclude_special_tokens": exclude_special_tokens,
                "prior_variant": variant,
                "empirical_uniform_mix": mixture_weight,
            },
        },
    }
    return metadata, state, config


def test_decode_and_metrics_keep_strict_and_released_funnels() -> None:
    strict_results = {
        "ok": "A",
        "recover": None,
        "multi": "C.DD",
        "bad": None,
        "duplicate": "A",
    }
    released_results = {
        "ok": "A",
        "recover": "B",
        "multi": "C.DD",
        "bad": None,
        "duplicate": "A",
    }

    timing = {}
    records = benchmark.decode_records(
        ["ok", "recover", "multi", "bad", "duplicate"],
        use_bracket_safe=False,
        strict_decoder=strict_results.get,
        released_decoder=released_results.get,
        timing=timing,
    )
    metrics, failures = benchmark.evaluate_records(
        records,
        requested_count=5,
        oracle_qed=_score,
        oracle_sa=_sa,
        diversity_evaluator=_diversity,
    )

    released = metrics["released_comparable"]
    assert released["valid_count"] == 4
    assert released["validity"] == pytest.approx(4 / 5)
    assert released["unique_count"] == 3
    assert released["uniqueness_denominator"] == 4
    assert released["uniqueness"] == pytest.approx(3 / 4)
    assert released["diversity_input_count"] == 3
    assert released["quality_count"] == 3
    assert released["quality_denominator"] == 5
    assert released["quality"] == pytest.approx(3 / 5)

    strict = metrics["strict"]
    assert strict["valid_count"] == 3
    assert strict["validity"] == pytest.approx(3 / 5)
    assert strict["unique_count"] == 2
    assert strict["uniqueness"] == pytest.approx(2 / 3)
    assert strict["quality"] == pytest.approx(2 / 5)

    assert records[1]["released_was_recovered"] is True
    assert records[2]["released_repaired_smiles"] == "C.DD"
    assert records[2]["released_smiles"] == "DD"
    assert records[2]["released_largest_component_applied"] is True
    assert records[4]["released_is_first_unique"] is False
    assert records[4]["released_quality_counted"] is False
    assert failures == {
        "raw_safe_conversion_failed": 0,
        "strict_decode_failed": 2,
        "released_decode_failed": 1,
        "released_recovered_strict_failure": 1,
        "strict_valid_but_released_failed": 0,
        "released_largest_component_applied": 1,
        "strict_duplicates": 1,
        "released_duplicates": 1,
    }
    assert set(timing) == {"released_postprocessing"}
    assert timing["released_postprocessing"] >= 0


def test_bracket_safe_conversion_failure_is_retained() -> None:
    def converter(value: str) -> str:
        if value == "broken":
            raise ValueError("cannot convert")
        return f"safe:{value}"

    records = benchmark.decode_records(
        ["good", "broken"],
        use_bracket_safe=True,
        bracket_converter=converter,
        strict_decoder=lambda value: value,
        released_decoder=lambda value: value,
    )

    assert records[0]["raw_safe"] == "safe:good"
    assert records[1]["raw_safe"] is None
    assert records[1]["raw_safe_error"] == "ValueError: cannot convert"
    assert records[1]["strict_decode_error"] == "SAFE conversion failed"
    assert records[1]["released_decode_error"] == "SAFE conversion failed"


def test_strict_smiles_must_pass_rdkit_sanitization() -> None:
    assert benchmark._canonicalize_chemically_valid_smiles("OCC") == "CCO"
    with pytest.raises(ValueError, match="RDKit rejected"):
        benchmark._canonicalize_chemically_valid_smiles("not a SMILES")


def test_empty_valid_set_has_defined_denominators_and_failure_counts() -> None:
    records = benchmark.decode_records(
        ["x", "y"],
        use_bracket_safe=False,
        strict_decoder=lambda _: None,
        released_decoder=lambda _: None,
    )

    metrics, failures = benchmark.evaluate_records(
        records,
        requested_count=2,
        oracle_qed=lambda _: pytest.fail("oracle must not run"),
        oracle_sa=lambda _: pytest.fail("oracle must not run"),
        diversity_evaluator=lambda _: pytest.fail("evaluator must not run"),
    )

    for branch in metrics.values():
        assert branch["validity"] == 0
        assert branch["uniqueness"] is None
        assert branch["uniqueness_denominator"] == 0
        assert branch["diversity"] is None
        assert branch["quality"] == 0
    assert failures["strict_decode_failed"] == 2
    assert failures["released_decode_failed"] == 2


@pytest.mark.parametrize("diffusion_type", ["mdlm", "udlm"])
def test_generate_raw_model_text_uses_shared_raw_token_api(diffusion_type: str) -> None:
    torch = pytest.importorskip("torch")

    class FakeTokenizer:
        def batch_decode(self, values, *, skip_special_tokens):
            assert skip_special_tokens is True
            return [f"safe-{int(row[0])}" for row in values]

    class FakeModel:
        bos_index = 1
        eos_index = 2
        device = torch.device("cpu")
        tokenizer = FakeTokenizer()
        config = type(
            "Config",
            (),
            {
                "training": {
                    "udlm": {"inference_eps": 1e-5},
                }
            },
        )()

    class FakeMDLM:
        def get_num_steps_confidence(self, values):
            return 1  # released code clamps this to two steps

    class FakeSampler:
        pad_index = 0

        def __init__(self):
            self.model = FakeModel()
            self.mdlm = FakeMDLM()
            self.diffusion_type = diffusion_type
            self.insert_call = None
            self.generate_call = None

        def _insert_mask(self, values, count, *, min_add_len):
            self.insert_call = (values.clone(), count, min_add_len)
            return values.repeat(count, 1)

        def generate(self, values, **kwargs):
            self.generate_call = (values.clone(), kwargs)
            return values + 2

    sampler = FakeSampler()
    decoded, protocol = benchmark.generate_raw_model_text(
        sampler,
        3,
        diffusion_type=diffusion_type,
        softmax_temp=0.5,
        randomness=0.25,
        min_add_len=40,
        num_steps=8 if diffusion_type == "udlm" else None,
        inference_eps=1e-5 if diffusion_type == "udlm" else None,
        exclude_special_tokens=False if diffusion_type == "udlm" else None,
        prior_variant="release_uniform" if diffusion_type == "udlm" else None,
        prior_metadata_sha256=None,
    )

    assert sampler.insert_call[1:] == (3, 40)
    assert decoded == ["safe-3", "safe-3", "safe-3"]
    assert sampler.generate_call[1] == {
        "softmax_temp": 0.5,
        "randomness": 0.25,
        "num_steps": 8 if diffusion_type == "udlm" else None,
        "return_token_ids": True,
    }
    assert protocol["diffusion_type"] == diffusion_type
    assert protocol["nfe"] == (8 if diffusion_type == "udlm" else 2)
    assert protocol["randomness_used_by_sampler"] is (diffusion_type == "mdlm")


def test_loaded_categorical_model_must_match_explicit_sampling_prior_digest() -> None:
    metadata, _, _ = _categorical_checkpoint_parts("schedule_uniform")

    class PriorMetadata:
        def to_dict(self):
            return dict(metadata)

    class Model:
        config = type(
            "Config",
            (),
            {"training": {"udlm": {"prior_variant": "schedule_uniform"}}},
        )()
        udlm_prior_metadata = PriorMetadata()

    sampler = type("Sampler", (), {"model": Model()})()
    digest = benchmark._canonical_json_sha256(metadata)

    benchmark._validate_loaded_udlm_prior_identity(
        sampler,
        prior_variant="schedule_uniform",
        prior_metadata_sha256=digest,
    )
    with pytest.raises(
        benchmark.BenchmarkConfigurationError,
        match="prior_metadata_sha256",
    ):
        benchmark._validate_loaded_udlm_prior_identity(
            sampler,
            prior_variant="schedule_uniform",
            prior_metadata_sha256="0" * 64,
        )


def test_atomic_outputs_and_default_no_overwrite(tmp_path: Path) -> None:
    records = [
        {field: (0 if field == "sample_index" else None)}
        for field in benchmark.RAW_SAMPLE_FIELDS
    ]
    csv_path = tmp_path / benchmark.RAW_SAMPLES_FILENAME
    json_path = tmp_path / benchmark.SUMMARY_FILENAME

    benchmark.atomic_write_csv(csv_path, records)
    benchmark.atomic_write_json(json_path, {"status": "completed"})

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["sample_index"] == "0"
    assert json.loads(json_path.read_text(encoding="utf-8"))["status"] == "completed"
    assert not list(tmp_path.glob("*.tmp"))

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        benchmark.validate_output_target(tmp_path, overwrite=False)
    benchmark.validate_output_target(tmp_path, overwrite=True)


def test_output_lock_prevents_concurrent_writer(tmp_path: Path) -> None:
    with benchmark.output_lock(tmp_path):
        with pytest.raises(RuntimeError, match="is locked"):
            with benchmark.output_lock(tmp_path):
                pass
    assert not (tmp_path / benchmark.LOCK_FILENAME).exists()


def test_config_validation_and_fingerprints(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "softmax_temp: 0.5\nrandomness: 0.5\nmin_add_len: 40\n",
        encoding="utf-8",
    )
    config = benchmark.load_yaml_config(config_path)
    sampling = benchmark.validate_sampling_config(config)

    assert sampling == {
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
    assert benchmark._canonical_json_sha256(
        sampling
    ) == benchmark._canonical_json_sha256(dict(reversed(list(sampling.items()))))

    with pytest.raises(benchmark.BenchmarkConfigurationError, match="missing"):
        benchmark.validate_sampling_config({})
    with pytest.raises(benchmark.BenchmarkConfigurationError, match="integer"):
        benchmark.validate_sampling_config(
            {"softmax_temp": 0.5, "randomness": 0.5, "min_add_len": 40.5}
        )

    udlm = benchmark.validate_sampling_config(
        {
            "diffusion_type": "udlm",
            "softmax_temp": 1.0,
            "randomness": 99,
            "min_add_len": 12,
            "num_steps": 32,
            "inference_eps": 1e-5,
            "exclude_special_tokens": False,
        }
    )
    assert udlm["num_steps"] == 32
    assert udlm["inference_eps"] == pytest.approx(1e-5)
    assert udlm["prior_variant"] == "release_uniform"
    assert udlm["prior_metadata_sha256"] is None
    categorical = benchmark.validate_sampling_config(
        {
            **{
                key: value
                for key, value in udlm.items()
                if key not in {"prior_variant", "prior_metadata_sha256"}
            },
            "prior_variant": "schedule_uniform",
            "prior_metadata_sha256": "a" * 64,
        }
    )
    assert categorical["prior_variant"] == "schedule_uniform"
    assert categorical["prior_metadata_sha256"] == "a" * 64
    with pytest.raises(
        benchmark.BenchmarkConfigurationError,
        match="explicitly declare",
    ):
        benchmark.validate_sampling_config(
            {
                "diffusion_type": "udlm",
                "softmax_temp": 1.0,
                "randomness": 0.0,
                "min_add_len": 12,
                "num_steps": 32,
                "inference_eps": 1e-5,
                "exclude_special_tokens": False,
                "prior_variant": "schedule_uniform",
            }
        )
    with pytest.raises(benchmark.BenchmarkConfigurationError, match="64 lowercase"):
        benchmark.validate_sampling_config(
            {
                **categorical,
                "prior_metadata_sha256": "NOT-A-DIGEST",
            }
        )
    with pytest.raises(benchmark.BenchmarkConfigurationError, match="num_steps"):
        benchmark.validate_sampling_config(
            {
                "diffusion_type": "udlm",
                "softmax_temp": 1.0,
                "randomness": 0.0,
                "min_add_len": 12,
                "inference_eps": 1e-5,
                "exclude_special_tokens": False,
            }
        )


@pytest.mark.parametrize(
    ("filename", "variant"),
    [
        ("hparams_udlm_schedule_uniform.yaml", "schedule_uniform"),
        ("hparams_udlm_categorical.yaml", "empirical_frequency"),
    ],
)
def test_categorical_inference_configs_pin_exact_default_metadata_digest(
    filename: str,
    variant: str,
) -> None:
    metadata, _, _ = _categorical_checkpoint_parts(variant)
    path = benchmark.REPO_ROOT / "scripts/exps/denovo" / filename

    sampling = benchmark.validate_sampling_config(benchmark.load_yaml_config(path))

    assert sampling["prior_variant"] == variant
    assert sampling["prior_metadata_sha256"] == benchmark._canonical_json_sha256(
        metadata
    )


def test_checkpoint_metadata_records_step_and_digest(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint_path = tmp_path / "tiny.ckpt"
    torch.save({"global_step": 50_000, "epoch": 4, "state_dict": {}}, checkpoint_path)

    metadata = benchmark.checkpoint_metadata(checkpoint_path)

    assert metadata["global_step"] == 50_000
    assert metadata["epoch"] == 4
    assert metadata["size_bytes"] == checkpoint_path.stat().st_size
    assert len(metadata["sha256"]) == 64
    assert metadata["byte_identity_verified_before_and_after_load"] is True
    assert metadata["diffusion_type"] == "mdlm"
    assert metadata["udlm_exclude_special_tokens"] is None
    assert metadata["udlm_prior_variant"] is None
    assert metadata["udlm_prior_metadata"] is None
    assert metadata["udlm_prior_metadata_sha256"] is None


def test_mdlm_checkpoint_cannot_hide_categorical_prior_state(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint_path = tmp_path / "mislabeled-mdlm.ckpt"
    torch.save(
        {
            "global_step": 1,
            "state_dict": {"mdlm.stationary_probs": torch.ones(2) / 2},
        },
        checkpoint_path,
    )

    with pytest.raises(RuntimeError, match="categorical UDLM prior"):
        benchmark.checkpoint_metadata(checkpoint_path)


def test_checkpoint_metadata_records_udlm_backend_and_endpoint(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    omegaconf = pytest.importorskip("omegaconf")
    checkpoint_path = tmp_path / "tiny-udlm.ckpt"
    config = omegaconf.OmegaConf.create(
        {
            "training": {
                "diffusion": "udlm",
                "udlm": {
                    "inference_eps": 2e-5,
                    "exclude_special_tokens": True,
                },
            }
        }
    )
    torch.save(
        {
            "global_step": 100,
            "epoch": 0,
            "state_dict": {},
            "hyper_parameters": {"config": config},
        },
        checkpoint_path,
    )

    metadata = benchmark.checkpoint_metadata(checkpoint_path)

    assert metadata["diffusion_type"] == "udlm"
    assert metadata["udlm_inference_eps"] == pytest.approx(2e-5)
    assert metadata["udlm_exclude_special_tokens"] is True
    assert metadata["udlm_prior_variant"] == "release_uniform"
    assert metadata["udlm_prior_metadata"] is None


@pytest.mark.parametrize(
    ("variant", "exclude_special_tokens"),
    [("schedule_uniform", False), ("empirical_frequency", True)],
)
def test_checkpoint_metadata_validates_full_categorical_prior_identity(
    tmp_path: Path,
    variant: str,
    exclude_special_tokens: bool,
) -> None:
    torch = pytest.importorskip("torch")
    metadata, state, config = _categorical_checkpoint_parts(
        variant,
        exclude_special_tokens=exclude_special_tokens,
    )
    checkpoint_path = tmp_path / f"{variant}.ckpt"
    torch.save(
        {
            "global_step": 12,
            "epoch": 0,
            "state_dict": state,
            "hyper_parameters": {"config": config},
            benchmark.UDLM_PRIOR_CHECKPOINT_KEY: metadata,
        },
        checkpoint_path,
    )

    observed = benchmark.checkpoint_metadata(checkpoint_path)

    assert observed["udlm_prior_variant"] == variant
    assert observed["udlm_prior_metadata"] == metadata
    assert observed["udlm_prior_metadata_sha256"] == (
        benchmark._canonical_json_sha256(metadata)
    )
    assert observed["udlm_exclude_special_tokens"] is exclude_special_tokens


@pytest.mark.parametrize("variant", ["schedule_uniform", "empirical_frequency"])
def test_prior_metadata_hash_is_validated_without_state(variant: str) -> None:
    metadata, _, _ = _categorical_checkpoint_parts(variant)
    benchmark.validate_udlm_prior_metadata_record(
        metadata,
        expected_variant=variant,
    )
    metadata["stationary_probs_sha256"] = "0" * 64

    with pytest.raises(RuntimeError, match="configured prior law"):
        benchmark.validate_udlm_prior_metadata_record(
            metadata,
            expected_variant=variant,
        )


def test_checkpoint_metadata_rejects_wrong_launch_pinned_digest(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    checkpoint_path = tmp_path / "tiny.ckpt"
    torch.save({"global_step": 1, "state_dict": {}}, checkpoint_path)

    with pytest.raises(RuntimeError, match="launch-pinned digest"):
        benchmark.checkpoint_metadata(
            checkpoint_path,
            expected_sha256="0" * 64,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("metadata_hash", "stationary prior disagrees"),
        ("missing_state", "stationary_probs"),
        ("config_variant", "must not declare categorical"),
        ("config_noise", "noise_eps disagrees"),
        ("config_exclusion", "exclusions disagree"),
        ("bool_mix", "must be a real number"),
    ],
)
def test_checkpoint_metadata_rejects_categorical_prior_mismatches(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    torch = pytest.importorskip("torch")
    metadata, state, config = _categorical_checkpoint_parts("empirical_frequency")
    if mutation == "metadata_hash":
        metadata["stationary_probs_sha256"] = "0" * 64
    elif mutation == "missing_state":
        del state["mdlm.stationary_probs"]
    elif mutation == "config_variant":
        config["training"]["udlm"]["prior_variant"] = "release_uniform"
    elif mutation == "config_noise":
        config["training"]["udlm"]["noise_eps"] = 0.02
    elif mutation == "config_exclusion":
        config["training"]["udlm"]["exclude_special_tokens"] = True
    else:
        config["training"]["udlm"]["empirical_uniform_mix"] = True
    checkpoint_path = tmp_path / f"bad-{mutation}.ckpt"
    torch.save(
        {
            "global_step": 12,
            "state_dict": state,
            "hyper_parameters": {"config": config},
            benchmark.UDLM_PRIOR_CHECKPOINT_KEY: metadata,
        },
        checkpoint_path,
    )

    with pytest.raises(RuntimeError, match=message):
        benchmark.checkpoint_metadata(checkpoint_path)


def test_checkpoint_metadata_rejects_tampered_empirical_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    metadata, state, config = _categorical_checkpoint_parts("empirical_frequency")
    artifact = tmp_path / benchmark.EMPIRICAL_FREQUENCY_RELATIVE_PATH
    artifact.parent.mkdir(parents=True)
    source_artifact = benchmark.REPO_ROOT / benchmark.EMPIRICAL_FREQUENCY_RELATIVE_PATH
    artifact.write_bytes(source_artifact.read_bytes() + b"\n")
    monkeypatch.setattr(benchmark, "REPO_ROOT", tmp_path)
    checkpoint_path = tmp_path / "tampered-artifact.ckpt"
    torch.save(
        {
            "global_step": 12,
            "state_dict": state,
            "hyper_parameters": {"config": config},
            benchmark.UDLM_PRIOR_CHECKPOINT_KEY: metadata,
        },
        checkpoint_path,
    )

    with pytest.raises(RuntimeError, match="artifact SHA-256 mismatch"):
        benchmark.checkpoint_metadata(checkpoint_path)


def test_implementation_inputs_include_length_distribution_statistics() -> None:
    snapshot = benchmark.load_implementation_input_snapshot()
    inputs = snapshot.provenance

    assert set(inputs) == {
        "genmol_package_init_source",
        "genmol_utils_package_init_source",
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
    }
    for artifact in inputs.values():
        assert len(artifact["sha256"]) == 64
        assert artifact["size_bytes"] > 0
    lengths = inputs["length_distribution"]
    assert lengths["count"] > 0
    assert lengths["minimum"] <= lengths["median"] <= lengths["maximum"]
    assert lengths["count"] == len(snapshot.length_distribution)
    assert (
        lengths["sha256"]
        == hashlib.sha256(
            (benchmark.REPO_ROOT / "data/len.pk").read_bytes()
        ).hexdigest()
    )
    assert lengths["loading_policy"] == (
        "verified_bytes_retained_in_memory_for_generation"
    )


def test_pinned_sa_artifact_loads_verified_bytes_without_tdc_download(
    tmp_path: Path,
) -> None:
    path, digest, size = _tiny_sa_artifact(tmp_path)

    snapshot = benchmark._load_pinned_sa_metric_input(
        repository_root=tmp_path,
        relative_path=Path("oracle/fpscores.pkl"),
        expected_sha256=digest,
        expected_size_bytes=size,
        expected_row_count=2,
        expected_fingerprint_count=3,
        include_tdc_provenance=False,
    )

    assert snapshot.fragment_scores == {7: 1.25, 9: 1.25, 11: -0.5}
    artifact = snapshot.provenance["sa_fragment_scores"]
    assert artifact["path"] == str(path)
    assert artifact["sha256"] == digest
    assert snapshot.provenance["sa_loading_policy"]["network_download_allowed"] is False
    assert snapshot.provenance["sa_loading_policy"]["tdc_oracle_load_invoked"] is False


def test_pinned_sa_artifact_rejects_missing_symlink_nonregular_and_wrong_digest(
    tmp_path: Path,
) -> None:
    relative = Path("oracle/fpscores.pkl")
    common = {
        "repository_root": tmp_path,
        "relative_path": relative,
        "expected_sha256": "0" * 64,
        "expected_size_bytes": 1,
    }
    with pytest.raises(FileNotFoundError, match="implicit downloader"):
        benchmark._read_pinned_regular_file(**common)

    target = tmp_path / "target.pkl"
    target.write_bytes(b"x")
    (tmp_path / "oracle").mkdir()
    (tmp_path / relative).symlink_to(target)
    with pytest.raises(RuntimeError, match="symlink"):
        benchmark._read_pinned_regular_file(**common)
    (tmp_path / relative).unlink()
    (tmp_path / relative).mkdir()
    with pytest.raises(RuntimeError, match="regular file"):
        benchmark._read_pinned_regular_file(**common)
    (tmp_path / relative).rmdir()
    (tmp_path / relative).write_bytes(b"x")
    wrong_digest = {**common, "expected_sha256": "f" * 64}
    with pytest.raises(RuntimeError, match="wrong SHA-256"):
        benchmark._read_pinned_regular_file(**wrong_digest)


def test_pinned_sa_artifact_rejects_mutation_while_descriptor_is_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, digest, size = _tiny_sa_artifact(tmp_path)
    real_read = benchmark.os.read
    mutated = False

    def mutating_read(descriptor: int, count: int) -> bytes:
        nonlocal mutated
        chunk = real_read(descriptor, count)
        if chunk and not mutated:
            mutated = True
            path.write_bytes(path.read_bytes() + b"changed")
        return chunk

    monkeypatch.setattr(benchmark.os, "read", mutating_read)
    with pytest.raises(RuntimeError, match="changed while being read"):
        benchmark._read_pinned_regular_file(
            repository_root=tmp_path,
            relative_path=Path("oracle/fpscores.pkl"),
            expected_sha256=digest,
            expected_size_bytes=size,
        )


def test_tdc_sa_context_hydrates_scores_and_disables_both_download_hooks() -> None:
    tdc = pytest.importorskip("tdc")
    oracle_dispatch = pytest.importorskip("tdc.oracles")
    scoring_module = pytest.importorskip("tdc.chem_utils.oracle.oracle")
    provenance = {
        "tdc_metric_implementation": benchmark._tdc_metric_implementation_provenance()
    }
    snapshot = benchmark.PinnedSAMetricInput(
        provenance=provenance,
        fragment_scores={7: 1.25},
    )
    previous_scores = scoring_module._fscores
    previous_dispatch_loader = oracle_dispatch.oracle_load
    previous_scoring_loader = scoring_module.oracle_load

    with benchmark.pinned_tdc_sa_oracle(snapshot, tdc.Oracle) as oracle:
        assert oracle.name == "sa"
        assert scoring_module._fscores == {7: 1.25}
        with pytest.raises(RuntimeError, match="downloading is disabled"):
            oracle_dispatch.oracle_load("fpscores")
        with pytest.raises(RuntimeError, match="downloading is disabled"):
            scoring_module.oracle_load("fpscores")

    assert scoring_module._fscores is previous_scores
    assert oracle_dispatch.oracle_load is previous_dispatch_loader
    assert scoring_module.oracle_load is previous_scoring_loader


def test_runtime_generation_modules_match_recorded_paths_and_hashes() -> None:
    inputs = benchmark.implementation_input_provenance()
    benchmark.assert_local_genmol_import()
    benchmark.assert_runtime_module_provenance(inputs)

    tampered = {name: dict(value) for name, value in inputs.items()}
    tampered["diffusion_source"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="source changed"):
        benchmark.assert_runtime_module_provenance(tampered)


def test_git_status_preflight_ignores_only_repository_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("/output/\n", encoding="utf-8")
    source = tmp_path / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=GenMol Test",
            "-c",
            "user.email=genmol@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    monkeypatch.setattr(benchmark, "REPO_ROOT", tmp_path)

    output_file = tmp_path / "output" / "seed_0" / "summary.json"
    output_file.parent.mkdir(parents=True)
    output_file.write_text("{}\n", encoding="utf-8")
    assert benchmark._git_status_outside_output() == []

    source.write_text("value = 2\n", encoding="utf-8")
    assert benchmark._git_status_outside_output()


def test_child_source_preflight_binds_expected_head_and_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "a" * 40
    monkeypatch.setattr(benchmark, "_git_status_outside_output", lambda: [])
    monkeypatch.setattr(
        benchmark,
        "_git_command",
        lambda arguments: revision
        if arguments in (["rev-parse", "HEAD"], ["rev-parse", "@{upstream}"])
        else None,
    )
    assert benchmark.require_clean_pushed_source(revision) == {
        "head": revision,
        "upstream": revision,
    }

    monkeypatch.setattr(
        benchmark, "_git_status_outside_output", lambda: [" M src/genmol/model.py"]
    )
    with pytest.raises(RuntimeError, match="dirty outside output"):
        benchmark.require_clean_pushed_source(revision)

    monkeypatch.setattr(benchmark, "_git_status_outside_output", lambda: [])
    monkeypatch.setattr(
        benchmark,
        "_git_command",
        lambda arguments: revision if arguments == ["rev-parse", "HEAD"] else None,
    )
    with pytest.raises(RuntimeError, match="no inspectable upstream"):
        benchmark.require_clean_pushed_source(revision)
    with pytest.raises(benchmark.BenchmarkConfigurationError, match="40 lowercase"):
        benchmark.require_clean_pushed_source("A" * 40)


def test_tracked_source_file_provenance_rejects_external_or_modified_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    config = tmp_path / "configs" / "inference.yaml"
    config.parent.mkdir()
    config.write_text("num_steps: 16\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("/output/\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=GenMol Test",
            "-c",
            "user.email=genmol@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True
    ).strip()
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    monkeypatch.setattr(benchmark, "REPO_ROOT", tmp_path)

    provenance = benchmark.tracked_source_file_provenance(
        config,
        expected_revision=revision,
        expected_sha256=digest,
    )
    assert provenance["relative_path"] == "configs/inference.yaml"
    assert provenance["tracked_at_source_revision"] is True

    external = tmp_path / "output" / "custom.yaml"
    external.parent.mkdir()
    external.write_text("num_steps: 8\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not tracked"):
        benchmark.tracked_source_file_provenance(
            external,
            expected_revision=revision,
            expected_sha256=hashlib.sha256(external.read_bytes()).hexdigest(),
        )

    config.write_text("num_steps: 32\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="worktree config bytes"):
        benchmark.tracked_source_file_provenance(
            config,
            expected_revision=revision,
            expected_sha256=digest,
        )


def test_seed_sampling_repeats_python_numpy_and_torch_streams() -> None:
    import random

    numpy = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")

    first_info = benchmark.seed_sampling(1234, "cpu")
    first = (random.random(), numpy.random.random(), torch.rand(3))
    second_info = benchmark.seed_sampling(1234, "cpu")
    second = (random.random(), numpy.random.random(), torch.rand(3))

    assert first_info["seed_applied_immediately_before_generation"] is True
    assert second_info["seed"] == 1234
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_tokenizer_provenance_hashes_effective_vocabulary() -> None:
    class FakeBackend:
        def to_str(self):
            return '{"model":"fake"}'

    class FakeTokenizer:
        vocab_size = 2
        name_or_path = "synthetic/tokenizer"
        init_kwargs = {"revision": "main", "_commit_hash": "abc123"}
        backend_tokenizer = FakeBackend()
        pad_token_id = 0
        bos_token_id = 1
        eos_token_id = 2
        mask_token_id = 3

        def get_vocab(self):
            return {"B": 1, "A": 0, "<extra>": 2}

        def get_added_vocab(self):
            return {"<extra>": 2}

        def __len__(self):
            return 3

    provenance = benchmark.tokenizer_provenance(FakeTokenizer())

    assert provenance["requested_identifier"] == "datamol-io/safe-gpt"
    assert provenance["effective_size"] == 3
    assert provenance["base_vocab_size"] == 2
    assert provenance["resolved_commit_hash"] == "abc123"
    assert len(provenance["vocabulary_sha256"]) == 64
    assert len(provenance["backend_json_sha256"]) == 64


def test_inference_weight_receipt_validation_and_required_ema() -> None:
    assert benchmark.AUDITED_BENCHMARK_REQUIRES_EMA is True
    ema_receipt = {
        "source": "ema",
        "ema_applied": True,
        "ema": {
            "shadow_parameter_count": 2,
            "decay": 0.999,
            "num_updates": 17,
        },
    }

    validated = benchmark.validate_inference_weights(ema_receipt, require_ema=True)
    assert validated == ema_receipt
    assert validated is not ema_receipt
    assert validated["ema"] is not ema_receipt["ema"]

    raw_receipt = {"source": "raw_model", "ema_applied": False, "ema": None}
    assert benchmark.validate_inference_weights(raw_receipt) == raw_receipt
    with pytest.raises(benchmark.BenchmarkConfigurationError, match="were required"):
        benchmark.validate_inference_weights(raw_receipt, require_ema=True)

    zero_update = {
        **ema_receipt,
        "ema": {**ema_receipt["ema"], "num_updates": 0},
    }
    with pytest.raises(benchmark.BenchmarkConfigurationError, match="positive update"):
        benchmark.validate_inference_weights(zero_update, require_ema=True)


def test_sampler_load_copies_and_freezes_validated_ema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from genmol import sampler as sampler_module
    from genmol.utils.ema import ExponentialMovingAverage

    checkpoint_path = tmp_path / "synthetic.ckpt"
    checkpoint_path.write_bytes(b"synthetic checkpoint bytes")
    backbone = torch.nn.Linear(3, 2)
    with torch.no_grad():
        for parameter in backbone.parameters():
            parameter.fill_(-1.0)
    ema = ExponentialMovingAverage(backbone.parameters(), decay=0.9)
    with torch.no_grad():
        for shadow in ema.shadow_params:
            shadow.fill_(2.0)
    ema.num_updates = 11
    model = SimpleNamespace(backbone=backbone, ema=ema)
    monkeypatch.setattr(
        sampler_module.GenMol,
        "load_from_checkpoint",
        staticmethod(lambda _checkpoint_file: model),
    )

    loaded = sampler_module.load_model_from_path(
        checkpoint_path,
        require_ema=True,
    )
    for parameter in loaded.backbone.parameters():
        assert torch.equal(parameter, torch.full_like(parameter, 2.0))
    assert loaded.inference_weights == {
        "source": "ema",
        "ema_applied": True,
        "ema": {
            "shadow_parameter_count": 2,
            "decay": 0.9,
            "num_updates": 11,
        },
    }
    with pytest.raises(TypeError):
        loaded.inference_weights["source"] = "raw_model"
    with pytest.raises(TypeError):
        loaded.inference_weights["ema"]["decay"] = 0.0

    sampler = sampler_module.Sampler.__new__(sampler_module.Sampler)
    sampler._inference_weights = loaded.inference_weights
    public_receipt = sampler.inference_weights
    public_receipt["ema"]["decay"] = 0.1
    assert sampler.inference_weights["ema"]["decay"] == 0.9


def test_sampler_preserves_raw_weights_unless_ema_is_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from genmol import sampler as sampler_module

    checkpoint_path = tmp_path / "synthetic.ckpt"
    checkpoint_path.write_bytes(b"synthetic checkpoint bytes")
    model = SimpleNamespace(backbone=torch.nn.Linear(2, 1), ema=None)
    original = [parameter.detach().clone() for parameter in model.backbone.parameters()]
    monkeypatch.setattr(
        sampler_module.GenMol,
        "load_from_checkpoint",
        staticmethod(lambda _checkpoint_file: model),
    )

    loaded = sampler_module.load_model_from_path(checkpoint_path)
    assert all(
        torch.equal(before, after)
        for before, after in zip(original, loaded.backbone.parameters())
    )
    assert dict(loaded.inference_weights) == {
        "source": "raw_model",
        "ema_applied": False,
        "ema": None,
    }
    with pytest.raises(RuntimeError, match="checkpoint has no EMA"):
        sampler_module.load_model_from_path(checkpoint_path, require_ema=True)


def test_cli_requires_every_run_identity_field() -> None:
    parser = benchmark.build_parser()
    args = parser.parse_args(
        [
            "--checkpoint",
            "model.ckpt",
            "--expected-checkpoint-sha256",
            "b" * 64,
            "--expected-source-revision",
            "a" * 40,
            "--config",
            "hparams.yaml",
            "--expected-config-sha256",
            "c" * 64,
            "--num-samples",
            "1000",
            "--seed",
            "7",
            "--device",
            "cpu",
            "--output-dir",
            "run",
        ]
    )
    assert args.seed == 7
    assert args.num_samples == 1000
    assert args.overwrite is False


def test_benchmark_run_label_is_hash_qualified() -> None:
    assert benchmark.benchmark_run_label(50_000, "abcdef012345" + "0" * 52, 2) == (
        "denovo_step50000_abcdef012345_seed2"
    )

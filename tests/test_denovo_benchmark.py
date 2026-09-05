from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.exps.denovo import benchmark


def _score(values):
    return [0.7 for _ in values]


def _sa(values):
    return [3.0 for _ in values]


def _diversity(values):
    return len(values) / 10


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
    }
    assert benchmark._canonical_json_sha256(sampling) == benchmark._canonical_json_sha256(
        dict(reversed(list(sampling.items())))
    )

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


def test_checkpoint_metadata_records_step_and_digest(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint_path = tmp_path / "tiny.ckpt"
    torch.save({"global_step": 50_000, "epoch": 4, "state_dict": {}}, checkpoint_path)

    metadata = benchmark.checkpoint_metadata(checkpoint_path)

    assert metadata["global_step"] == 50_000
    assert metadata["epoch"] == 4
    assert metadata["size_bytes"] == checkpoint_path.stat().st_size
    assert len(metadata["sha256"]) == 64
    assert metadata["diffusion_type"] == "mdlm"
    assert metadata["udlm_exclude_special_tokens"] is None


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


def test_implementation_inputs_include_length_distribution_statistics() -> None:
    inputs = benchmark.implementation_input_provenance()

    assert set(inputs) == {
        "sampler_source",
        "model_source",
        "diffusion_source",
        "backbone_source",
        "chemistry_utils_source",
        "data_utils_source",
        "bracket_safe_converter_source",
        "length_distribution",
    }
    for artifact in inputs.values():
        assert len(artifact["sha256"]) == 64
        assert artifact["size_bytes"] > 0
    lengths = inputs["length_distribution"]
    assert lengths["count"] > 0
    assert lengths["minimum"] <= lengths["median"] <= lengths["maximum"]


def test_runtime_generation_modules_match_recorded_paths_and_hashes() -> None:
    inputs = benchmark.implementation_input_provenance()
    benchmark.assert_local_genmol_import()
    benchmark.assert_runtime_module_provenance(inputs)

    tampered = {name: dict(value) for name, value in inputs.items()}
    tampered["diffusion_source"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="source changed"):
        benchmark.assert_runtime_module_provenance(tampered)


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


def test_cli_requires_every_run_identity_field() -> None:
    parser = benchmark.build_parser()
    args = parser.parse_args(
        [
            "--checkpoint",
            "model.ckpt",
            "--config",
            "hparams.yaml",
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

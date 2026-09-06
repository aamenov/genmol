import hashlib
import itertools

import datasets
import pytest
from omegaconf import OmegaConf

from genmol.utils import utils_data


def test_tokenizer_download_is_revision_and_checksum_pinned(monkeypatch, tmp_path):
    tokenizer_file = tmp_path / "tokenizer.json"
    tokenizer_file.write_bytes(b"pinned tokenizer")
    expected_sha = hashlib.sha256(tokenizer_file.read_bytes()).hexdigest()
    calls = {}

    def fake_download(repo_id, *, filename, revision):
        calls.update(repo_id=repo_id, filename=filename, revision=revision)
        return str(tokenizer_file)

    class FakeSafeTokenizer:
        @classmethod
        def from_pretrained(cls, path):
            assert path == str(tokenizer_file)
            return cls()

        def get_pretrained(self):
            return self

        def add_tokens(self, tokens):
            assert tokens == ["<", ">"]

    monkeypatch.setattr(utils_data, "hf_hub_download", fake_download)
    monkeypatch.setattr(utils_data, "SAFETokenizer", FakeSafeTokenizer)
    monkeypatch.setattr(utils_data, "SAFE_GPT_TOKENIZER_SHA256", expected_sha)

    assert utils_data.get_tokenizer().__class__ is FakeSafeTokenizer
    assert calls == {
        "repo_id": utils_data.SAFE_GPT_REPO_ID,
        "filename": "tokenizer.json",
        "revision": utils_data.SAFE_GPT_TOKENIZER_REVISION,
    }


def test_hosted_training_stream_is_revision_pinned_and_rank_sharded(monkeypatch):
    calls = {}
    split_calls = []
    source_dataset = object()
    rank_dataset = object()

    def fake_load_dataset(repo_id, **kwargs):
        calls.update(repo_id=repo_id, **kwargs)
        return source_dataset

    def fake_split(dataset, *, rank, world_size):
        split_calls.append((dataset, rank, world_size))
        return rank_dataset

    class FakeDataLoader:
        def __init__(self, dataset, **kwargs):
            self.dataset = dataset
            self.kwargs = kwargs

    monkeypatch.setattr(utils_data.datasets, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(utils_data, "split_dataset_by_node", fake_split)
    monkeypatch.setattr(utils_data.torch.utils.data, "DataLoader", FakeDataLoader)
    monkeypatch.setattr(utils_data, "Collator", lambda config: "collator")
    config = OmegaConf.create(
        {
            "data": "safe",
            "loader": {"batch_size": 2, "num_workers": 0, "pin_memory": False},
        }
    )

    loader = utils_data.get_dataloader(
        config,
        streaming_rank=1,
        streaming_world_size=2,
    )

    assert loader.dataset is rank_dataset
    assert split_calls == [(source_dataset, 1, 2)]
    assert calls == {
        "repo_id": utils_data.SAFE_GPT_REPO_ID,
        "revision": utils_data.SAFE_GPT_DATASET_REVISION,
        "streaming": True,
        "split": "train",
    }


def test_single_rank_hosted_stream_preserves_dataset_identity(monkeypatch):
    source_dataset = object()
    monkeypatch.setattr(
        utils_data,
        "split_dataset_by_node",
        lambda *_args, **_kwargs: pytest.fail("single rank must not shard"),
    )

    assert (
        utils_data._shard_hosted_stream(
            source_dataset,
            streaming_rank=0,
            streaming_world_size=1,
        )
        is source_dataset
    )
    assert utils_data._shard_hosted_stream(source_dataset) is source_dataset


def test_two_rank_hosted_streams_are_disjoint_and_reconstruct_source():
    def generate_rows():
        for index in range(12):
            yield {"index": index}

    source = datasets.IterableDataset.from_generator(generate_rows)
    rank_rows = []
    for rank in (0, 1):
        shard = utils_data._shard_hosted_stream(
            source,
            streaming_rank=rank,
            streaming_world_size=2,
        )
        rank_rows.append([row["index"] for row in itertools.islice(iter(shard), 6)])

    assert set(rank_rows[0]).isdisjoint(rank_rows[1])
    assert sorted(rank_rows[0] + rank_rows[1]) == list(range(12))
    assert rank_rows == [list(range(0, 12, 2)), list(range(1, 12, 2))]


@pytest.mark.parametrize(
    ("global_rank", "world_size"),
    [(-1, 2), (2, 2), (0, 0), (True, 2), (0, True), (0.0, 1)],
)
def test_distributed_data_identity_is_fail_closed(global_rank, world_size):
    with pytest.raises(ValueError, match="distributed data identity"):
        utils_data._shard_hosted_stream(
            object(),
            streaming_rank=global_rank,
            streaming_world_size=world_size,
        )


@pytest.mark.parametrize(
    ("streaming_rank", "streaming_world_size"),
    [(None, 2), (0, None)],
)
def test_streaming_partition_arguments_must_be_paired(
    streaming_rank, streaming_world_size
):
    with pytest.raises(ValueError, match="must be provided together"):
        utils_data._shard_hosted_stream(
            object(),
            streaming_rank=streaming_rank,
            streaming_world_size=streaming_world_size,
        )

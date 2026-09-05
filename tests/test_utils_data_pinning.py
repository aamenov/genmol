import hashlib

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


def test_hosted_training_stream_is_revision_pinned(monkeypatch):
    calls = {}

    def fake_load_dataset(repo_id, **kwargs):
        calls.update(repo_id=repo_id, **kwargs)
        return []

    class FakeDataLoader:
        def __init__(self, dataset, **kwargs):
            self.dataset = dataset
            self.kwargs = kwargs

    monkeypatch.setattr(utils_data.datasets, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(utils_data.torch.utils.data, "DataLoader", FakeDataLoader)
    monkeypatch.setattr(utils_data, "Collator", lambda config: "collator")
    config = OmegaConf.create(
        {
            "data": "safe",
            "loader": {"batch_size": 2, "num_workers": 0, "pin_memory": False},
        }
    )

    loader = utils_data.get_dataloader(config)

    assert loader.dataset == []
    assert calls == {
        "repo_id": utils_data.SAFE_GPT_REPO_ID,
        "revision": utils_data.SAFE_GPT_DATASET_REVISION,
        "streaming": True,
        "split": "train",
    }

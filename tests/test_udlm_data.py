from genmol.utils.utils_data import Collator, UserDataset


def test_collator_accepts_hosted_and_local_safe_columns():
    assert Collator._read_safe({"safe": "C.C"}) == "C.C"
    assert Collator._read_safe({"input": "N.N"}) == "N.N"


def test_user_dataset_scalar_indexing(tmp_path):
    path = tmp_path / "tiny.safe"
    path.write_text("C.C\nN.N\n")
    dataset = UserDataset(path)

    assert len(dataset) == 2
    assert dataset[0] == {"input": "C.C"}
    assert dataset[1] == {"input": "N.N"}

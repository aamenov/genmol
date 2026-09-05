import hashlib
import os

import pytest
import torch

from genmol.utils.checkpoint_io import verified_checkpoint_file


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _different_sha256(digest):
    replacement = "0" if digest[0] != "0" else "1"
    return replacement + digest[1:]


def test_verified_checkpoint_file_loads_exact_pinned_bytes(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"value": torch.tensor([1, 2, 3])}, checkpoint_path)
    expected_sha256 = _sha256(checkpoint_path)

    with verified_checkpoint_file(
        checkpoint_path,
        expected_sha256=expected_sha256,
    ) as (checkpoint_file, identity):
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)

    assert torch.equal(checkpoint["value"], torch.tensor([1, 2, 3]))
    assert identity.requested_path == str(checkpoint_path)
    assert identity.resolved_path == str(checkpoint_path.resolve())
    assert identity.sha256 == expected_sha256
    assert identity.size_bytes == checkpoint_path.stat().st_size


def test_verified_checkpoint_file_rejects_wrong_digest_before_yield(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"value": torch.tensor([1])}, checkpoint_path)
    wrong_sha256 = _different_sha256(_sha256(checkpoint_path))
    body_entered = False

    with pytest.raises(RuntimeError, match="launch-pinned digest"):
        with verified_checkpoint_file(
            checkpoint_path,
            expected_sha256=wrong_sha256,
        ):
            body_entered = True

    assert not body_entered


def test_verified_checkpoint_file_detects_pathname_replacement(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    replacement_path = tmp_path / "replacement.pt"
    torch.save({"value": torch.tensor([1])}, checkpoint_path)
    torch.save({"value": torch.tensor([2])}, replacement_path)
    expected_sha256 = _sha256(checkpoint_path)

    with pytest.raises(RuntimeError, match="changed during deserialization"):
        with verified_checkpoint_file(
            checkpoint_path,
            expected_sha256=expected_sha256,
        ) as (checkpoint_file, _identity):
            os.replace(replacement_path, checkpoint_path)
            checkpoint = torch.load(
                checkpoint_file,
                map_location="cpu",
                weights_only=True,
            )
            assert torch.equal(checkpoint["value"], torch.tensor([1]))


def test_verified_checkpoint_file_detects_in_place_mutation(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"value": torch.tensor([1])}, checkpoint_path)
    expected_sha256 = _sha256(checkpoint_path)

    with pytest.raises(RuntimeError, match="changed during deserialization"):
        with verified_checkpoint_file(
            checkpoint_path,
            expected_sha256=expected_sha256,
        ) as (checkpoint_file, identity):
            checkpoint = torch.load(
                checkpoint_file,
                map_location="cpu",
                weights_only=True,
            )
            with checkpoint_path.open("r+b") as mutable_checkpoint:
                original_byte = mutable_checkpoint.read(1)
                mutable_checkpoint.seek(0)
                mutable_checkpoint.write(bytes([original_byte[0] ^ 0xFF]))
                mutable_checkpoint.flush()
                os.fsync(mutable_checkpoint.fileno())
            assert checkpoint_path.stat().st_ino == identity.inode
            assert checkpoint_path.stat().st_size == identity.size_bytes
            assert torch.equal(checkpoint["value"], torch.tensor([1]))

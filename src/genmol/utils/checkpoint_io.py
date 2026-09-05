"""Stable, byte-bound checkpoint reads for scientific runs.

Pathnames can be replaced between a provenance hash and ``torch.load``.  This
module instead opens one regular-file descriptor, fingerprints that descriptor,
and gives callers a duplicate of the same descriptor to load.  The original
descriptor and pathname binding are verified again after the caller finishes.
"""

from __future__ import annotations

import hashlib
import os
import stat
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Iterator


_SHA256_HEX_LENGTH = 64
_HASH_CHUNK_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True)
class CheckpointFileIdentity:
    """Identity of the exact regular-file bytes exposed to a loader."""

    requested_path: str
    resolved_path: str
    sha256: str
    size_bytes: int
    device: int
    inode: int
    mode: int
    link_count: int
    mtime_ns: int
    ctime_ns: int

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def validate_sha256(value: str, *, name: str = "expected checkpoint SHA-256") -> str:
    """Require the canonical digest spelling used in manifests and CLIs."""

    if (
        not isinstance(value, str)
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal digits")
    return value


def _descriptor_state(file_stat: os.stat_result) -> tuple[int, ...]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        file_stat.st_nlink,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _sha256_descriptor(file_descriptor: int) -> str:
    """Hash a descriptor without changing its shared file offset."""

    digest = hashlib.sha256()
    offset = 0
    while True:
        chunk = os.pread(file_descriptor, _HASH_CHUNK_SIZE, offset)
        if not chunk:
            break
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def _require_path_binding(
    resolved_path: Path,
    expected_state: tuple[int, ...],
    *,
    phase: str,
) -> None:
    try:
        path_stat = os.stat(resolved_path, follow_symlinks=False)
    except OSError as error:
        raise RuntimeError(
            f"checkpoint pathname disappeared during {phase}: {resolved_path}"
        ) from error
    if _descriptor_state(path_stat) != expected_state:
        raise RuntimeError(
            f"checkpoint pathname or file identity changed during {phase}: "
            f"{resolved_path}"
        )


@contextmanager
def verified_checkpoint_file(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str | None = None,
) -> Iterator[tuple[BinaryIO, CheckpointFileIdentity]]:
    """Yield one stable descriptor and fail closed if its bytes change.

    The expected digest is checked before the caller can deserialize the
    trusted checkpoint.  A second digest plus descriptor/path metadata checks
    run after deserialization, so ordinary pathname replacement and in-place
    concurrent writes cannot silently change the loaded scientific input.
    """

    if expected_sha256 is not None:
        expected_sha256 = validate_sha256(expected_sha256)
    requested_path = Path(path)
    try:
        resolved_path = requested_path.resolve(strict=True)
    except OSError as error:
        raise FileNotFoundError(f"checkpoint does not exist: {requested_path}") from error

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        file_descriptor = os.open(resolved_path, flags)
    except OSError as error:
        raise RuntimeError(f"cannot open checkpoint safely: {resolved_path}") from error

    try:
        before_hash = os.fstat(file_descriptor)
        if not stat.S_ISREG(before_hash.st_mode):
            raise RuntimeError(f"checkpoint is not a regular file: {resolved_path}")
        initial_state = _descriptor_state(before_hash)
        _require_path_binding(resolved_path, initial_state, phase="checkpoint open")

        digest = _sha256_descriptor(file_descriptor)
        after_hash = os.fstat(file_descriptor)
        if _descriptor_state(after_hash) != initial_state:
            raise RuntimeError(
                f"checkpoint changed while it was being hashed: {resolved_path}"
            )
        _require_path_binding(resolved_path, initial_state, phase="checkpoint hash")
        if expected_sha256 is not None and digest != expected_sha256:
            raise RuntimeError(
                "checkpoint SHA-256 does not match the launch-pinned digest: "
                f"{digest} != {expected_sha256}"
            )

        identity = CheckpointFileIdentity(
            requested_path=str(requested_path),
            resolved_path=str(resolved_path),
            sha256=digest,
            size_bytes=before_hash.st_size,
            device=before_hash.st_dev,
            inode=before_hash.st_ino,
            mode=before_hash.st_mode,
            link_count=before_hash.st_nlink,
            mtime_ns=before_hash.st_mtime_ns,
            ctime_ns=before_hash.st_ctime_ns,
        )
        try:
            with os.fdopen(os.dup(file_descriptor), "rb") as checkpoint_file:
                yield checkpoint_file, identity
        finally:
            after_load = os.fstat(file_descriptor)
            if _descriptor_state(after_load) != initial_state:
                raise RuntimeError(
                    "checkpoint file identity changed during deserialization: "
                    f"{resolved_path}"
                )
            _require_path_binding(
                resolved_path,
                initial_state,
                phase="checkpoint deserialization",
            )
            final_digest = _sha256_descriptor(file_descriptor)
            if final_digest != digest:
                raise RuntimeError(
                    "checkpoint bytes changed during deserialization: "
                    f"{final_digest} != {digest}"
                )
            final_stat = os.fstat(file_descriptor)
            if _descriptor_state(final_stat) != initial_state:
                raise RuntimeError(
                    f"checkpoint changed during its final verification: {resolved_path}"
                )
            _require_path_binding(
                resolved_path,
                initial_state,
                phase="checkpoint final verification",
            )
    finally:
        os.close(file_descriptor)

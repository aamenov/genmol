"""Launch the exact ten-update UDLM health panel one arm at a time.

The wrapper intentionally exposes only the user-selected world size and a
CPU-only dry-run switch.  Every scientific and safety control is fixed here,
and each invocation either previews/launches the first missing R -> S -> E arm
or validates an already completed terminal E receipt.  Health evidence can
authorize the registered optimization screen; it cannot rank models or support
a superiority claim.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Sequence

from scripts.udlm import launch_train_pilot
from scripts.udlm.validate_health_panel import (
    EXPECTED_MDLM_CHECKPOINT_PATH,
    HEALTH_PANEL_GLOBAL_BATCH_SIZE,
    HEALTH_PANEL_MAX_UTILIZATION_PERCENT,
    HEALTH_PANEL_MAX_STEPS,
    HEALTH_PANEL_MICRO_BATCH_SIZE,
    HEALTH_PANEL_MIN_FREE_MEMORY_MIB,
    HEALTH_PANEL_NUM_WORKERS,
    HEALTH_PANEL_SEED,
    health_run_name,
    validate_health_panel,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HEALTH_VARIANTS = tuple(launch_train_pilot.MATCHED_PANEL_VARIANT_ORDER)
RECEIPT_BASENAME = "pilot_exit_status.json"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu-count",
        type=int,
        choices=(1, 2),
        required=True,
        help="User-selected number of dynamically chosen idle GPUs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform CPU-only preflight without GPU, tmux, or output mutation.",
    )
    return parser.parse_args(argv)


def _run_paths(
    *, gpu_count: int, source_revision: str
) -> tuple[tuple[str, Path, Path], ...]:
    output_root = REPOSITORY_ROOT / "output" / "udlm"
    return tuple(
        (
            variant,
            output_root / health_run_name(gpu_count, variant, source_revision),
            output_root
            / health_run_name(gpu_count, variant, source_revision)
            / RECEIPT_BASENAME,
        )
        for variant in HEALTH_VARIANTS
    )


def _require_direct_directory(path: Path, *, label: str) -> None:
    try:
        state = path.stat(follow_symlinks=False)
    except OSError as error:
        raise RuntimeError(f"cannot inspect {label}: {path}: {error}") from error
    if not stat.S_ISDIR(state.st_mode) or stat.S_ISLNK(state.st_mode):
        raise RuntimeError(f"{label} must be a direct real directory: {path}")


def _require_direct_receipt(path: Path, *, label: str) -> None:
    if not os.path.lexists(path):
        raise RuntimeError(f"{label} exists without its completion receipt: {path}")
    try:
        state = path.stat(follow_symlinks=False)
    except OSError as error:
        raise RuntimeError(
            f"cannot inspect {label} receipt: {path}: {error}"
        ) from error
    if (
        not stat.S_ISREG(state.st_mode)
        or stat.S_ISLNK(state.st_mode)
        or state.st_nlink != 1
    ):
        raise RuntimeError(
            f"{label} completion receipt must be a direct single-link regular file: "
            f"{path}"
        )


def _require_successful_receipt(path: Path, *, label: str) -> None:
    _snapshot, payload = launch_train_pilot.stable_repository_artifact_snapshot(
        path,
        suffix=".json",
        label=f"{label} completion receipt",
    )
    receipt = launch_train_pilot.strict_json_loads(
        payload, label=f"{label} completion receipt"
    )
    if not isinstance(receipt, Mapping):
        raise RuntimeError(f"{label} completion receipt is not a JSON object")
    if (
        receipt.get("schema_version")
        != launch_train_pilot.PILOT_EXIT_STATUS_SCHEMA_VERSION
        or receipt.get("status") != "completed"
        or receipt.get("overall_status") != "completed"
        or type(receipt.get("process_exit_status")) is not int
        or receipt.get("process_exit_status") != 0
    ):
        raise RuntimeError(
            f"{label} has a failed or malformed completion receipt; preserve this "
            "source-revision namespace and retry the full R -> S -> E panel from "
            "a new clean pushed descendant"
        )


def _completed_prefix(
    paths: tuple[tuple[str, Path, Path], ...],
) -> int:
    """Return the number of completed leading arms, rejecting every gap."""

    existence = tuple(os.path.lexists(run_dir) for _, run_dir, _ in paths)
    missing_seen = False
    completed = 0
    for (variant, run_dir, receipt_path), exists in zip(paths, existence, strict=True):
        if not exists:
            missing_seen = True
            continue
        if missing_seen:
            raise RuntimeError(
                "health-panel run directories exist out of R -> S -> E order: "
                f"unexpected {variant} directory {run_dir}"
            )
        _require_direct_directory(run_dir, label=f"{variant} health run")
        _require_direct_receipt(receipt_path, label=f"{variant} health run")
        _require_successful_receipt(receipt_path, label=f"{variant} health run")
        completed += 1
    return completed


def _pilot_argv(
    *,
    gpu_count: int,
    variant: str,
    run_dir: Path,
    predecessor_receipt: Path | None,
    dry_run: bool,
) -> list[str]:
    argv = [
        "--run-name",
        run_dir.name,
        "--training-variant",
        variant,
        "--gpu-count",
        str(gpu_count),
        "--max-steps",
        str(HEALTH_PANEL_MAX_STEPS),
        "--global-batch-size",
        str(HEALTH_PANEL_GLOBAL_BATCH_SIZE),
        "--micro-batch-size",
        str(HEALTH_PANEL_MICRO_BATCH_SIZE),
        "--num-workers",
        str(HEALTH_PANEL_NUM_WORKERS),
        "--seed",
        str(HEALTH_PANEL_SEED),
        "--checkpoint",
        str(EXPECTED_MDLM_CHECKPOINT_PATH),
        "--max-utilization-percent",
        str(HEALTH_PANEL_MAX_UTILIZATION_PERCENT),
        "--min-free-memory-mib",
        str(HEALTH_PANEL_MIN_FREE_MEMORY_MIB),
    ]
    if predecessor_receipt is None:
        argv.append("--genesis")
    else:
        argv.extend(("--predecessor-receipt", str(predecessor_receipt)))
    if dry_run:
        argv.append("--dry-run")
    return argv


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    gpu_count = launch_train_pilot.validate_gpu_count(args.gpu_count)
    source_revision = launch_train_pilot.require_pushed_commit()
    paths = _run_paths(
        gpu_count=gpu_count,
        source_revision=source_revision,
    )

    # Read-only ancestor validation rejects output symlinks even for dry runs.
    for _variant, run_dir, _receipt_path in paths:
        launch_train_pilot.validate_pilot_output_parents(
            run_dir=run_dir,
            log_path=(REPOSITORY_ROOT / "output" / "logs" / f"{run_dir.name}.log"),
            create_missing=False,
        )

    completed = _completed_prefix(paths)
    if completed == len(paths):
        terminal_receipt = paths[-1][2]
        normalized = validate_health_panel(
            terminal_receipt,
            expected_gpu_count=gpu_count,
            expected_source_revision=source_revision,
        )
        print(json.dumps(normalized, indent=2, sort_keys=True, allow_nan=False))
        return 0

    variant, run_dir, _receipt_path = paths[completed]
    predecessor_receipt = None if completed == 0 else paths[completed - 1][2]
    launch_train_pilot.main(
        _pilot_argv(
            gpu_count=gpu_count,
            variant=variant,
            run_dir=run_dir,
            predecessor_receipt=predecessor_receipt,
            dry_run=args.dry_run,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

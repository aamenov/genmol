"""Run reproducible GenMol fragment-vocabulary ablations.

This runner is deliberately separate from the released PMO entry point.  It
keeps the released update policy as a reference arm while adding evidence-
retaining alternatives, explicit oracle accounting, manifests, event logs,
and restart checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from tdc import Oracle as TDCOracle


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from genmol.sampler import Sampler
from genmol.utils.utils_chem import cut, cut_all
from scripts.exps.pmo.main.genmol.experiment_io import (
    JsonlEventLog,
    build_manifest,
    iter_events,
    load_checkpoint,
    save_checkpoint,
    sha256_config,
    sha256_file,
    summarize_indexed_scores,
    summarize_scores,
    write_manifest,
)
from scripts.exps.pmo.main.genmol.fragment_population import (
    FragmentObservation,
    FragmentPopulation,
)


ORACLES = (
    "albuterol_similarity",
    "amlodipine_mpo",
    "celecoxib_rediscovery",
    "deco_hop",
    "drd2",
    "fexofenadine_mpo",
    "gsk3b",
    "isomers_c7h8n2o2",
    "isomers_c9h10n2o2pf2cl",
    "jnk3",
    "median1",
    "median2",
    "mestranol_similarity",
    "osimertinib_mpo",
    "perindopril_mpo",
    "qed",
    "ranolazine_mpo",
    "scaffold_hop",
    "sitagliptin_mpo",
    "thiothixene_rediscovery",
    "troglitazone_rediscovery",
    "valsartan_smarts",
    "zaleplon_mpo",
)

PAPER_GAMMA = {
    "albuterol_similarity": 0.2,
    "amlodipine_mpo": 0.3,
    "celecoxib_rediscovery": 0.0,
    "deco_hop": 0.2,
    "drd2": 0.0,
    "fexofenadine_mpo": 0.0,
    "gsk3b": 0.0,
    "isomers_c7h8n2o2": 0.5,
    "isomers_c9h10n2o2pf2cl": 0.0,
    "jnk3": 0.5,
    "median1": 0.2,
    "median2": 0.2,
    "mestranol_similarity": 0.0,
    "osimertinib_mpo": 0.0,
    "perindopril_mpo": 0.4,
    "qed": 0.0,
    "ranolazine_mpo": 0.0,
    "scaffold_hop": 0.0,
    "sitagliptin_mpo": 0.2,
    "thiothixene_rediscovery": 0.3,
    "troglitazone_rediscovery": 0.0,
    "valsartan_smarts": 0.4,
    "zaleplon_mpo": 0.4,
}

VARIANT_SETTINGS: dict[str, dict[str, Any]] = {
    "released": {"mode": "released", "min_support": 1, "parent_control": False},
    "running_mean": {"mode": "mean", "min_support": 1, "parent_control": False},
    "support3": {"mode": "mean", "min_support": 3, "parent_control": False},
    "shrink10": {"mode": "bayes", "min_support": 1, "parent_control": False},
    "delta": {"mode": "delta", "min_support": 1, "parent_control": True},
    "running_mean_parent_control": {
        "mode": "mean",
        "min_support": 1,
        "parent_control": True,
    },
}

SAFE_EXPERIMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

SMALL_MOLECULE_ORACLES = {
    "albuterol_similarity",
    "isomers_c7h8n2o2",
    "isomers_c9h10n2o2pf2cl",
    "median1",
    "qed",
    "sitagliptin_mpo",
    "zaleplon_mpo",
}
LARGE_MOLECULE_ORACLES = {"gsk3b", "jnk3"}


@dataclasses.dataclass(frozen=True)
class ScoreOutcome:
    """One canonicalized oracle lookup."""

    raw_smiles: Optional[str]
    canonical_smiles: Optional[str]
    valid: bool
    score: Optional[float]
    charged: bool
    call_index: Optional[int]
    reason: str


class CachedOracle:
    """A strict unique-canonical-molecule oracle budget."""

    def __init__(self, evaluator: Any, budget: int):
        if budget <= 0:
            raise ValueError("oracle budget must be positive")
        self.evaluator = evaluator
        self.budget = budget
        self.buffer: dict[str, list[float | int]] = {}

    @property
    def calls(self) -> int:
        return len(self.buffer)

    @property
    def finished(self) -> bool:
        return self.calls >= self.budget

    def score(self, smiles: Optional[str]) -> ScoreOutcome:
        if smiles is None:
            return ScoreOutcome(smiles, None, False, None, False, None, "missing_smiles")
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None or not smiles:
            return ScoreOutcome(smiles, None, False, None, False, None, "invalid_smiles")
        canonical = Chem.MolToSmiles(molecule)
        cached = self.buffer.get(canonical)
        if cached is not None:
            return ScoreOutcome(
                smiles,
                canonical,
                True,
                float(cached[0]),
                False,
                int(cached[1]),
                "cache_hit",
            )
        if self.finished:
            return ScoreOutcome(smiles, canonical, True, None, False, None, "budget_exhausted")

        value = float(self.evaluator(canonical))
        if not math.isfinite(value):
            raise ValueError(f"oracle returned non-finite score for {canonical!r}")
        call_index = self.calls + 1
        self.buffer[canonical] = [value, call_index]
        return ScoreOutcome(smiles, canonical, True, value, True, call_index, "scored")

    def state_dict(self) -> dict[str, Any]:
        return {"budget": self.budget, "buffer": self.buffer}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["budget"]) != self.budget:
            raise ValueError("oracle budget differs from checkpoint")
        restored: dict[str, list[float | int]] = {}
        for smiles, row in state["buffer"].items():
            restored[str(smiles)] = [float(row[0]), int(row[1])]
        indices = sorted(int(row[1]) for row in restored.values())
        if indices != list(range(1, len(indices) + 1)):
            raise ValueError("oracle checkpoint call indices are not contiguous")
        if len(restored) > self.budget:
            raise ValueError("oracle checkpoint exceeds configured budget")
        self.buffer = restored

    def scores_in_call_order(self) -> list[float]:
        rows = sorted(self.buffer.values(), key=lambda row: int(row[1]))
        return [float(row[0]) for row in rows]


class RunDirectoryLock:
    """Advisory process lock preventing duplicate writers for one run ID."""

    def __init__(self, path: Path):
        self.path = path
        self.descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(self.descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(self.descriptor)
            self.descriptor = -1
            raise
        os.ftruncate(self.descriptor, 0)
        os.write(self.descriptor, f"pid={os.getpid()}\n".encode())
        os.fsync(self.descriptor)

    def close(self) -> None:
        if self.descriptor < 0:
            return
        fcntl.flock(self.descriptor, fcntl.LOCK_UN)
        os.close(self.descriptor)
        self.descriptor = -1

    def __del__(self) -> None:
        self.close()


def _derived_seed(base_seed: int, stream: str) -> int:
    payload = f"genmol-fragment-vocab-v1:{base_seed}:{stream}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Preserve the two-column prefix emitted by ``git status --porcelain``.
    # ``str.strip()`` silently removes the leading index/worktree status space
    # from the first path, corrupting the recorded provenance for that entry.
    return completed.stdout.rstrip("\r\n")


def _git_metadata() -> dict[str, Any]:
    status = _git_output("status", "--porcelain=v1")
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    return {
        "commit": _git_output("rev-parse", "HEAD"),
        "branch": _git_output("branch", "--show-current"),
        "dirty": bool(status),
        "status": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime_metadata(device: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "rdkit": rdBase.rdkitVersion,
        "packages": {
            "bionemo-moco": _package_version("bionemo-moco"),
            "pytdc": _package_version("pytdc"),
            "safe-mol": _package_version("safe-mol"),
            "transformers": _package_version("transformers"),
        },
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        logical_index = torch.device(device).index or 0
        properties = torch.cuda.get_device_properties(logical_index)
        result["gpu"] = {
            "logical_index": logical_index,
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "capability": [properties.major, properties.minor],
        }
    return result


def _vocabulary_has_sufficient_statistics(path: str | Path) -> bool:
    with Path(path).open(newline="") as handle:
        fieldnames = csv.DictReader(handle).fieldnames or []
    fields = set(fieldnames)
    return "count" in fields and bool({"score_sum", "sum", "total"} & fields)


def _repair_event_tail(path: Path, expected_events: int) -> Optional[Path]:
    """Archive and remove JSONL bytes newer than a durable checkpoint."""

    if expected_events < 0:
        raise ValueError("expected event count cannot be negative")
    if not path.exists():
        if expected_events:
            raise ValueError("checkpoint references a missing event log")
        return None
    payload = path.read_bytes()
    line_ends = [index + 1 for index, byte in enumerate(payload) if byte == ord("\n")]
    if len(line_ends) < expected_events:
        raise ValueError("event log contains fewer durable rows than the checkpoint")
    prefix_end = 0 if expected_events == 0 else line_ends[expected_events - 1]
    if prefix_end == len(payload):
        return None

    orphaned = path.with_name(f"{path.name}.orphaned.{time.time_ns()}")
    orphaned.write_bytes(payload[prefix_end:])
    temporary = path.with_name(f".{path.name}.repair.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload[:prefix_end])
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return orphaned


def _fsync_file(path: Path) -> None:
    if not path.exists():
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _molecule_size_bounds(oracle_name: str, default_min: int, default_max: int) -> tuple[int, int]:
    if oracle_name in SMALL_MOLECULE_ORACLES:
        return 10, 30
    if oracle_name in LARGE_MOLECULE_ORACLES:
        return 30, 80
    return default_min, default_max


def _attach_fragments(frag1: str, frag2: str, rng: np.random.Generator) -> Optional[str]:
    reaction = AllChem.ReactionFromSmarts("[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]")
    reactant1 = Chem.MolFromSmiles(frag1)
    reactant2 = Chem.MolFromSmiles(frag2)
    if reactant1 is None or reactant2 is None:
        return None
    products = reaction.RunReactants((reactant1, reactant2))
    if not products:
        return None
    product = products[int(rng.integers(len(products)))][0]
    try:
        return Chem.MolToSmiles(product)
    except (RuntimeError, ValueError):
        return None


def _largest_component(smiles: Optional[str]) -> Optional[str]:
    if not smiles:
        return None
    return max(smiles.split("."), key=len)


def _score_row(outcome: Optional[ScoreOutcome]) -> Optional[dict[str, Any]]:
    return None if outcome is None else dataclasses.asdict(outcome)


def _top_means(oracle: CachedOracle) -> dict[str, Optional[float]]:
    scores = sorted(oracle.scores_in_call_order(), reverse=True)
    result: dict[str, Optional[float]] = {}
    for k in (1, 10, 100):
        result[f"top_{k}"] = None if not scores else float(np.mean(scores[:k]))
    return result


def _rng_state(
    fragment_rng: random.Random,
    attach_rng: np.random.Generator,
) -> dict[str, Any]:
    return {
        "python_global": random.getstate(),
        "numpy_global": np.random.get_state(),
        "fragment": fragment_rng.getstate(),
        "attach": attach_rng.bit_generator.state,
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(
    state: Mapping[str, Any],
    fragment_rng: random.Random,
    attach_rng: np.random.Generator,
) -> None:
    random.setstate(state["python_global"])
    np.random.set_state(state["numpy_global"])
    fragment_rng.setstate(state["fragment"])
    attach_rng.bit_generator.state = state["attach"]
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", choices=ORACLES, required=True)
    parser.add_argument("--variant", choices=tuple(VARIANT_SETTINGS), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-oracle-calls", type=int, default=10_000)
    parser.add_argument("--reporting-frequency", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--max-iterations", type=int, default=30_000)
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--legacy-warmup-off-by-one", action="store_true")
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--softmax-temp", type=float, default=1.2)
    parser.add_argument("--randomness", type=float, default=2.0)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--min-mol-size", type=int, default=20)
    parser.add_argument("--max-mol-size", type=int, default=40)
    parser.add_argument("--legacy-seed-count", type=int, default=None)
    parser.add_argument("--prior-mean", type=float, default=None)
    parser.add_argument("--prior-mean-source", default=None)
    parser.add_argument(
        "--delta-attribution",
        choices=("all_child", "novel_vs_parent"),
        default="novel_vs_parent",
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--scientific-status",
        required=True,
        help="Explicit maturity/comparability label stored in the immutable run config.",
    )
    parser.add_argument(
        "--matrix-path",
        type=Path,
        default=None,
        help="Optional launcher matrix source recorded in the resolved configuration.",
    )
    parser.add_argument(
        "--matrix-sha256",
        default=None,
        help="SHA-256 of --matrix-path; required together with that path.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "output" / "pmo_ablation",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--durable-events", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "max_oracle_calls",
        "reporting_frequency",
        "checkpoint_every",
        "max_iterations",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.population_size < 2:
        raise ValueError("population-size must be at least two")
    if args.warmup < 0:
        raise ValueError("warmup must be nonnegative")
    if args.min_mol_size <= 0 or args.max_mol_size < args.min_mol_size:
        raise ValueError("invalid molecule-size bounds")
    if not 0 <= (PAPER_GAMMA[args.oracle] if args.gamma is None else args.gamma) <= 1:
        raise ValueError("gamma must lie in [0, 1]")
    if not math.isfinite(args.guidance_scale) or args.guidance_scale <= 0:
        raise ValueError("guidance-scale must be finite and positive")
    if not SAFE_EXPERIMENT_ID.fullmatch(args.experiment_id):
        raise ValueError(
            "experiment-id must contain only letters, digits, dots, underscores, and hyphens"
        )
    if not args.scientific_status.strip():
        raise ValueError("scientific-status must be nonempty")
    matrix_path = getattr(args, "matrix_path", None)
    matrix_sha256 = getattr(args, "matrix_sha256", None)
    if (matrix_path is None) != (matrix_sha256 is None):
        raise ValueError("matrix-path and matrix-sha256 must be provided together")
    if matrix_path is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", str(matrix_sha256)):
            raise ValueError("matrix-sha256 must be a lowercase SHA-256 digest")
        if sha256_file(matrix_path.expanduser().resolve()) != matrix_sha256:
            raise ValueError("matrix-sha256 does not match matrix-path")
    settings = VARIANT_SETTINGS[args.variant]
    vocab_path = args.vocab_path or (
        REPOSITORY_ROOT / "scripts" / "exps" / "pmo" / "vocab" / f"{args.oracle}.csv"
    )
    if (
        settings["mode"] in {"mean", "bayes"}
        and args.legacy_seed_count is None
        and not _vocabulary_has_sufficient_statistics(vocab_path)
    ):
        raise ValueError(
            "statistical variants require an enriched vocabulary or an explicit "
            "--legacy-seed-count; use 1 only as a declared approximation"
        )
    if args.variant == "shrink10":
        if args.prior_mean is None or not math.isfinite(args.prior_mean):
            raise ValueError("shrink10 requires a finite frozen --prior-mean")
        if not args.prior_mean_source:
            raise ValueError("shrink10 requires --prior-mean-source provenance")
    elif args.prior_mean is not None or args.prior_mean_source is not None:
        raise ValueError("prior mean and source are only valid for shrink10")


def _resolved_config(args: argparse.Namespace) -> dict[str, Any]:
    settings = VARIANT_SETTINGS[args.variant]
    vocab_path = args.vocab_path or (
        REPOSITORY_ROOT / "scripts" / "exps" / "pmo" / "vocab" / f"{args.oracle}.csv"
    )
    min_size, max_size = _molecule_size_bounds(
        args.oracle, args.min_mol_size, args.max_mol_size
    )
    matrix_path = getattr(args, "matrix_path", None)
    return {
        "experiment_id": args.experiment_id,
        "scientific_status": args.scientific_status,
        "matrix_path": (
            None if matrix_path is None else str(matrix_path.expanduser().resolve())
        ),
        "matrix_sha256": getattr(args, "matrix_sha256", None),
        "oracle": args.oracle,
        "variant": args.variant,
        "policy_mode": settings["mode"],
        "parent_control": settings["parent_control"],
        "model_path": str(args.model_path.expanduser().resolve()),
        "vocab_path": str(vocab_path.expanduser().resolve()),
        "device": args.device,
        "seed": args.seed,
        "max_oracle_calls": args.max_oracle_calls,
        "reporting_frequency": args.reporting_frequency,
        "checkpoint_every": args.checkpoint_every,
        "max_iterations": args.max_iterations,
        "population_size": args.population_size,
        "warmup": args.warmup,
        "legacy_warmup_off_by_one": args.legacy_warmup_off_by_one,
        "gamma": PAPER_GAMMA[args.oracle] if args.gamma is None else args.gamma,
        "softmax_temp": args.softmax_temp,
        "randomness": args.randomness,
        "guidance_scale": args.guidance_scale,
        "min_mol_size": min_size,
        "max_mol_size": max_size,
        "min_support": settings["min_support"],
        "prior_strength": 10.0 if args.variant == "shrink10" else 0.0,
        "prior_mean": args.prior_mean,
        "prior_mean_source": args.prior_mean_source,
        "legacy_seed_count": args.legacy_seed_count,
        "delta_attribution": args.delta_attribution,
        "population_sampling_order": "canonical fragment string before uniform sampling",
        "statistical_duplicate_policy": "one update per unique canonical child",
        "released_duplicate_policy": "repeat cached-child decomposition, matching release",
        "durable_events": bool(args.durable_events),
    }


def _run_directory(config: Mapping[str, Any], output_root: Path) -> Path:
    return (
        output_root.expanduser().resolve()
        / str(config["experiment_id"])
        / str(config["oracle"])
        / str(config["variant"])
        / f"seed_{config['seed']}"
    )


def run(args: argparse.Namespace) -> Path:
    _validate_args(args)
    config = _resolved_config(args)
    vocabulary_has_statistics = _vocabulary_has_sufficient_statistics(config["vocab_path"])
    run_dir = _run_directory(config, args.output_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".run.lock"
    try:
        run_lock = RunDirectoryLock(lock_path)
    except BlockingIOError as error:
        raise RuntimeError(f"another process owns run directory {run_dir}") from error
    try:
        return _run_locked(args, config, vocabulary_has_statistics, run_dir)
    finally:
        run_lock.close()


def _run_locked(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    vocabulary_has_statistics: bool,
    run_dir: Path,
) -> Path:
    manifest_path = run_dir / "manifest.json"
    events_path = run_dir / "events.jsonl"
    checkpoint_path = run_dir / "state" / "latest.pkl"
    summary_path = run_dir / "summary.json"

    run_id = f"{config['experiment_id']}:{config['oracle']}:{config['variant']}:seed{config['seed']}"
    manifest_candidate = build_manifest(
        run_id=run_id,
        model_path=config["model_path"],
        config=config,
        task=str(config["oracle"]),
        variant=str(config["variant"]),
        seed=int(config["seed"]),
        oracle_budget=int(config["max_oracle_calls"]),
        extra={
            "git": _git_metadata(),
            "runtime": _runtime_metadata(str(config["device"])),
            "vocabulary": {
                "path": config["vocab_path"],
                "sha256": sha256_file(config["vocab_path"]),
            },
            "population_estimator_status": (
                "exact-count statistical continuation"
                if config["policy_mode"] in {"mean", "bayes"} and vocabulary_has_statistics
                else "approximate legacy pseudo-count"
                if config["policy_mode"] in {"mean", "bayes"}
                else "released update-policy reference with isolated RNG streams"
                if config["policy_mode"] == "released"
                else "exploratory approximate delta attribution"
            ),
        },
    )
    manifest_candidate["status"] = "running"

    if args.resume:
        if not manifest_path.exists() or not checkpoint_path.exists():
            raise FileNotFoundError("--resume requires an existing manifest and checkpoint")
        manifest = json.loads(manifest_path.read_text())
        if manifest["config_sha256"] != sha256_config(config):
            raise ValueError("resolved configuration does not match the saved run")
        if manifest["model"]["sha256"] != manifest_candidate["model"]["sha256"]:
            raise ValueError("model checkpoint hash does not match the saved run")
        if (
            manifest["extra"]["vocabulary"]["sha256"]
            != manifest_candidate["extra"]["vocabulary"]["sha256"]
        ):
            raise ValueError("vocabulary hash does not match the saved run")
        if manifest["extra"]["git"] != manifest_candidate["extra"]["git"]:
            raise ValueError("git code identity or tracked working diff changed since checkpoint")
    else:
        if (
            manifest_path.exists()
            or checkpoint_path.exists()
            or events_path.exists()
            or summary_path.exists()
        ):
            raise FileExistsError(f"run directory already contains state: {run_dir}")
        manifest = manifest_candidate

    seed = int(config["seed"])
    selection_rng = random.Random(_derived_seed(seed, "population-selection"))
    fragment_rng = random.Random(_derived_seed(seed, "vocabulary-fragmentation"))
    attach_rng = np.random.default_rng(_derived_seed(seed, "attachment-product"))
    random.seed(_derived_seed(seed, "generation-python"))
    np.random.seed(_derived_seed(seed, "legacy-numpy") % (2**32))
    torch.manual_seed(_derived_seed(seed, "diffusion-torch"))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_derived_seed(seed, "diffusion-torch-cuda"))

    fragmenter = lambda smiles: cut(smiles, rng=fragment_rng)
    population = FragmentPopulation.from_csv(
        config["vocab_path"],
        capacity=int(config["population_size"]),
        mode=str(config["policy_mode"]),
        fragmenter=fragmenter,
        rng=selection_rng,
        min_support=int(config["min_support"]),
        prior_strength=float(config["prior_strength"]),
        prior_mean=config["prior_mean"],
        legacy_seed_count=config["legacy_seed_count"],
    )
    if len(population.active_fragments) < 2:
        raise ValueError("initial vocabulary must provide at least two active fragments")
    oracle = CachedOracle(TDCOracle(name=str(config["oracle"])), int(config["max_oracle_calls"]))
    event_log = JsonlEventLog(events_path, durable=args.durable_events)
    start_iteration = 0
    event_count = 0
    elapsed_before_resume = 0.0
    next_checkpoint_call = int(config["checkpoint_every"])

    sampler = Sampler(str(config["model_path"]))
    sampler.model.to(str(config["device"]))
    sampler.mdlm.to_device(sampler.model.device)

    if args.resume:
        state, checkpoint_metadata = load_checkpoint(checkpoint_path, with_metadata=True)
        if checkpoint_metadata["config_sha256"] != manifest["config_sha256"]:
            raise ValueError("checkpoint configuration hash does not match manifest")
        if checkpoint_metadata["model_sha256"] != manifest["model"]["sha256"]:
            raise ValueError("checkpoint model hash does not match manifest")
        if (
            checkpoint_metadata.get("vocabulary_sha256")
            != manifest["extra"]["vocabulary"]["sha256"]
        ):
            raise ValueError("checkpoint vocabulary hash does not match manifest")
        if checkpoint_metadata.get("git_commit") != manifest["extra"]["git"]["commit"]:
            raise ValueError("checkpoint Git commit does not match manifest")
        if (
            checkpoint_metadata.get("tracked_diff_sha256")
            != manifest["extra"]["git"]["tracked_diff_sha256"]
        ):
            raise ValueError("checkpoint tracked working diff does not match manifest")
        population.load_state_dict(state["population"])
        oracle.load_state_dict(state["oracle"])
        _restore_rng_state(state["rng"], fragment_rng, attach_rng)
        start_iteration = int(state["next_iteration"])
        event_count = int(state["event_count"])
        elapsed_before_resume = float(state["elapsed_seconds"])
        next_checkpoint_call = int(state["next_checkpoint_call"])
        orphaned_tail = _repair_event_tail(events_path, event_count)
        if orphaned_tail is not None:
            print(f"Archived uncheckpointed event tail to {orphaned_tail}", flush=True)
        existing_events = list(iter_events(events_path))
        if len(existing_events) != event_count:
            raise ValueError("event log is not aligned with the latest checkpoint")

    manifest["status"] = "running"
    if args.resume:
        manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
        write_manifest(manifest_path, manifest, overwrite=True)
    else:
        manifest["resume_count"] = 0
        write_manifest(manifest_path, manifest)

    started = time.monotonic()
    last_checkpoint_calls = oracle.calls
    last_checkpoint_events = event_count

    def save_state(next_iteration: int) -> None:
        nonlocal last_checkpoint_calls, last_checkpoint_events
        elapsed = elapsed_before_resume + (time.monotonic() - started)
        state = {
            "next_iteration": next_iteration,
            "event_count": event_count,
            "elapsed_seconds": elapsed,
            "next_checkpoint_call": next_checkpoint_call,
            "population": population.state_dict(),
            "oracle": oracle.state_dict(),
            "rng": _rng_state(fragment_rng, attach_rng),
        }
        # A checkpoint may only refer to event bytes already flushed to disk.
        _fsync_file(events_path)
        save_checkpoint(
            checkpoint_path,
            state,
            metadata={
                "config_sha256": manifest["config_sha256"],
                "model_sha256": manifest["model"]["sha256"],
                "git_commit": manifest["extra"]["git"]["commit"],
                "tracked_diff_sha256": manifest["extra"]["git"]["tracked_diff_sha256"],
                "vocabulary_sha256": manifest["extra"]["vocabulary"]["sha256"],
            },
        )
        last_checkpoint_calls = oracle.calls
        last_checkpoint_events = event_count

    next_iteration = start_iteration
    terminal_status = "failed"
    terminal_error: Optional[str] = None
    try:
        if not args.resume:
            save_state(start_iteration)
        for iteration in range(start_iteration, int(config["max_iterations"])):
            next_iteration = iteration
            if oracle.finished:
                terminal_status = "completed"
                break

            remask_enabled = (
                iteration > int(config["warmup"])
                if config["legacy_warmup_off_by_one"]
                else iteration >= int(config["warmup"])
            )
            candidate: Optional[dict[str, Any]] = None
            for proposal_attempt in range(1, 1001):
                frag1, frag2 = population.sample(2)
                parent_smiles = _attach_fragments(frag1, frag2, attach_rng)
                if parent_smiles is None:
                    continue
                child_smiles = parent_smiles
                if remask_enabled:
                    child_smiles = sampler.mask_modification(
                        parent_smiles,
                        gamma=float(config["gamma"]),
                        softmax_temp=float(config["softmax_temp"]),
                        randomness=float(config["randomness"]),
                        w=float(config["guidance_scale"]),
                    )
                    child_smiles = _largest_component(child_smiles)
                child_molecule = Chem.MolFromSmiles(child_smiles) if child_smiles else None
                if child_molecule is None:
                    continue
                atom_count = child_molecule.GetNumAtoms()
                if int(config["min_mol_size"]) <= atom_count <= int(config["max_mol_size"]):
                    candidate = {
                        "selected_fragments": [frag1, frag2],
                        "parent_smiles": parent_smiles,
                        "child_smiles": Chem.MolToSmiles(child_molecule),
                        "atom_count": atom_count,
                        "proposal_attempts": proposal_attempt,
                        "remask_enabled": remask_enabled,
                    }
                    break

            if candidate is None:
                raise RuntimeError("failed to generate a size-valid molecule in 1000 attempts")

            parent_outcome: Optional[ScoreOutcome] = None
            if remask_enabled and bool(config["parent_control"]):
                parent_outcome = oracle.score(candidate["parent_smiles"])
                if oracle.finished:
                    while oracle.calls >= next_checkpoint_call:
                        next_checkpoint_call += int(config["checkpoint_every"])
                    event = {
                        "event_index": event_count,
                        "iteration": iteration,
                        **candidate,
                        "parent_oracle": _score_row(parent_outcome),
                        "child_oracle": None,
                        "population_update": {"updated": False, "reason": "budget_after_parent"},
                        "fragment_statistics_after": {},
                        "population_cutoff_after": population.active_rows()[-1][0],
                        "population_size_after": len(population.active_rows()),
                        "oracle_calls": oracle.calls,
                        **_top_means(oracle),
                        "elapsed_seconds": elapsed_before_resume + (time.monotonic() - started),
                    }
                    event_log.append(event)
                    event_count += 1
                    next_iteration = iteration + 1
                    terminal_status = "completed"
                    break

            child_outcome = oracle.score(candidate["child_smiles"])
            update_payload: dict[str, Any] = {"updated": False, "reason": "unscored_child"}
            fragment_statistics: dict[str, Any] = {}
            if child_outcome.score is not None and child_outcome.canonical_smiles is not None:
                if str(config["policy_mode"]) == "delta" and not remask_enabled:
                    update_payload = {"updated": False, "reason": "delta_warmup"}
                else:
                    credit_fragments = None
                    if str(config["policy_mode"]) == "delta":
                        child_fragments = set(fragmenter(child_outcome.canonical_smiles))
                        if config["delta_attribution"] == "novel_vs_parent":
                            child_fragments -= cut_all(candidate["parent_smiles"])
                        credit_fragments = frozenset(child_fragments)
                    observation = FragmentObservation(
                        observation_id=child_outcome.canonical_smiles,
                        child_smiles=child_outcome.canonical_smiles,
                        child_score=child_outcome.score,
                        parent_score=None if parent_outcome is None else parent_outcome.score,
                        credit_fragments=credit_fragments,
                    )
                    update = population.observe(observation)
                    update_payload = dataclasses.asdict(update)
                    for fragment in update.observed_fragments:
                        stats = population.get_stats(fragment)
                        if stats is not None:
                            fragment_statistics[fragment] = dataclasses.asdict(stats)

            event = {
                "event_index": event_count,
                "iteration": iteration,
                **candidate,
                "parent_oracle": _score_row(parent_outcome),
                "child_oracle": _score_row(child_outcome),
                "population_update": update_payload,
                "fragment_statistics_after": fragment_statistics,
                "population_cutoff_after": population.active_rows()[-1][0],
                "population_size_after": len(population.active_rows()),
                "oracle_calls": oracle.calls,
                **_top_means(oracle),
                "elapsed_seconds": elapsed_before_resume + (time.monotonic() - started),
            }
            event_log.append(event)
            event_count += 1
            next_iteration = iteration + 1

            if oracle.calls >= next_checkpoint_call:
                while oracle.calls >= next_checkpoint_call:
                    next_checkpoint_call += int(config["checkpoint_every"])
                save_state(next_iteration)
                print(
                    f"{oracle.calls}/{config['max_oracle_calls']} calls | "
                    f"iteration {iteration} | {_top_means(oracle)}",
                    flush=True,
                )
        else:
            terminal_status = "max_iterations_reached"

        if oracle.finished:
            terminal_status = "completed"
    except KeyboardInterrupt:
        terminal_status = "interrupted"
        terminal_error = "KeyboardInterrupt"
        raise
    except BaseException as error:
        terminal_status = "failed"
        terminal_error = f"{type(error).__name__}: {error}"
        raise
    finally:
        checkpoint_consistent = terminal_status in {"completed", "max_iterations_reached"}
        if checkpoint_consistent:
            save_state(next_iteration)
        scores = oracle.scores_in_call_order()
        score_summary = summarize_scores(
            scores,
            reporting_frequency=int(config["reporting_frequency"]),
            budget=int(config["max_oracle_calls"]),
        )
        logged_events = (
            list(iter_events(events_path, tolerate_truncated_last_line=True))
            if events_path.exists()
            else []
        )
        child_indexed_scores = [
            (
                int(event["child_oracle"]["call_index"]),
                float(event["child_oracle"]["score"]),
            )
            for event in logged_events
            if event.get("child_oracle")
            and event["child_oracle"].get("charged")
            and event["child_oracle"].get("score") is not None
        ]
        child_total_call_summary = summarize_indexed_scores(
            child_indexed_scores,
            observed_oracle_calls=oracle.calls,
            reporting_frequency=int(config["reporting_frequency"]),
            budget=int(config["max_oracle_calls"]),
        )
        child_scores = [score for _, score in child_indexed_scores]
        if child_scores:
            child_count_summary = summarize_scores(
                child_scores,
                reporting_frequency=int(config["reporting_frequency"]),
                budget=len(child_scores),
            )
            child_count_summary["axis"] = "charged_child_count"
            child_count_summary["score_count"] = child_count_summary.pop("oracle_calls")
            child_count_summary["child_count_horizon"] = child_count_summary.pop(
                "oracle_budget"
            )
        else:
            child_count_summary = {
                "axis": "charged_child_count",
                "score_count": 0,
                "child_count_horizon": 0,
                "reporting_frequency": int(config["reporting_frequency"]),
            }
            for k in (1, 10, 100):
                label = f"top_{k}"
                child_count_summary[label] = None
                child_count_summary[f"auc_{label}"] = None
                child_count_summary[f"trajectory_{label}"] = [
                    {"oracle_calls": 0, "top_k_mean": 0.0}
                ]
        summary = {
            "schema_version": 2,
            "run_id": run_id,
            "status": terminal_status,
            "error": terminal_error,
            "iterations_completed": next_iteration,
            "events": event_count,
            "elapsed_seconds": elapsed_before_resume + (time.monotonic() - started),
            "scores": {
                "all_charged_molecules": score_summary,
                "charged_children_total_call_axis": child_total_call_summary,
                "charged_children_child_count_axis": child_count_summary,
                "interpretation": (
                    "Parent-control arms include charged parents in the primary all-molecule PMO "
                    "trajectory. The total-call child trajectory retains each charged child's "
                    "actual global call position; the child-count trajectory measures proposal "
                    "quality without parent-call throughput."
                ),
            },
            "population": {
                "size": len(population.active_rows()),
                "active_rows": population.active_rows(),
            },
            "config_sha256": manifest["config_sha256"],
            "model_sha256": manifest["model"]["sha256"],
            "checkpoint_consistent": checkpoint_consistent,
            "recoverable_oracle_calls": last_checkpoint_calls,
            "recoverable_events": last_checkpoint_events,
        }
        write_manifest(summary_path, summary, overwrite=True)
        manifest["status"] = terminal_status
        manifest["error"] = terminal_error
        manifest["summary_path"] = str(summary_path)
        manifest["checkpoint_path"] = str(checkpoint_path)
        manifest["events_path"] = str(events_path)
        manifest["elapsed_seconds"] = summary["elapsed_seconds"]
        manifest["oracle_calls"] = oracle.calls
        write_manifest(manifest_path, manifest, overwrite=True)

    return run_dir


def main() -> None:
    args = _parse_args()
    run_dir = run(args)
    print(f"Results: {run_dir}", flush=True)


if __name__ == "__main__":
    main()

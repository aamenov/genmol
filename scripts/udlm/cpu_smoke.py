"""Cheap end-to-end UDLM overfit and sampling smoke test.

This is intentionally not a benchmark. It uses a tiny BERT and a fixed toy set
to catch integration, numerical, and reverse-chain failures before GPU pilots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import lightning as L
import safe as sf
import torch
from omegaconf import OmegaConf

from genmol.model import (
    EMPIRICAL_FREQUENCY_PATH,
    GenMol,
    UDLM_PRIOR_VARIANTS,
)
from genmol.sampler import Sampler


ROOT_DIR = Path(__file__).resolve().parents[2]
TOY_SMILES = (
    "CCO",
    "CCN",
    "CCS",
    "CCCl",
    "CCC",
    "CCCO",
    "CCCN",
    "CC(C)O",
    "CC(C)N",
    "CC(=O)O",
    "CCOC(=O)C",
    "c1ccccc1",
    "c1ccncc1",
    "CC1CCCCC1",
    "O=C(O)c1ccccc1",
    "CCOc1ccccc1",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_tensor_sha256(tensor: torch.Tensor) -> str:
    payload = json.dumps(
        tensor.detach().cpu().tolist(),
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _git_provenance() -> dict[str, object]:
    def command(*arguments: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(ROOT_DIR), *arguments],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = command("status", "--porcelain=v1", "--untracked-files=normal")
    return {
        "commit": command("rev-parse", "HEAD"),
        "upstream": command("rev-parse", "@{upstream}"),
        "dirty": bool(status) if status is not None else None,
        "status_porcelain": status.splitlines() if status else [],
    }


def _validate_git_provenance(provenance: dict[str, object]) -> None:
    """Require an auditable, pushed source state before collecting evidence."""
    failures = []
    commit = provenance.get("commit")
    upstream = provenance.get("upstream")
    if not commit or not upstream:
        failures.append("HEAD and its upstream must both resolve")
    elif commit != upstream:
        failures.append("HEAD must equal its upstream commit")
    if provenance.get("dirty") is not False:
        failures.append("worktree must be clean")
    if failures:
        raise RuntimeError("refusing unauditable CPU smoke run: " + "; ".join(failures))


def _source_paths(prior_variant: str) -> dict[str, Path]:
    paths = {
        "smoke_runner": Path(__file__).resolve(),
        "model": ROOT_DIR / "src/genmol/model.py",
        "diffusion": ROOT_DIR / "src/genmol/diffusion.py",
        "sampler": ROOT_DIR / "src/genmol/sampler.py",
    }
    if prior_variant == "empirical_frequency":
        paths["frequency_artifact"] = EMPIRICAL_FREQUENCY_PATH
    return paths


def _source_provenance(prior_variant: str) -> dict[str, dict[str, object]]:
    return {
        name: {
            "path": str(path.relative_to(ROOT_DIR)),
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in _source_paths(prior_variant).items()
    }


def _write_json_exclusive(path: Path, payload: dict[str, object]) -> None:
    """Create one result exactly once without following a pre-existing leaf link."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _config(
    *,
    exclude_special_tokens: bool,
    sampling_steps: int,
    prior_variant: str = "release_uniform",
    empirical_uniform_mix: float = 0.01,
):
    return OmegaConf.create(
        {
            "seed": 1,
            "model": {
                "attention_probs_dropout_prob": 0.0,
                "classifier_dropout": None,
                "hidden_act": "gelu",
                "hidden_dropout_prob": 0.0,
                "hidden_size": 64,
                "initializer_range": 0.02,
                "intermediate_size": 128,
                "layer_norm_eps": 1e-12,
                "max_position_embeddings": 48,
                "model_type": "bert",
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "pad_token_id": 3,
                "position_embedding_type": "absolute",
                "torch_dtype": "float32",
                "type_vocab_size": 2,
                "use_cache": True,
                "vocab_size": 1880,
            },
            "training": {
                "diffusion": "udlm",
                "ema": 0.0,
                "antithetic_sampling": True,
                "sampling_eps": 1e-3,
                "global_mean_loss": True,
                "use_bracket_safe": False,
                "udlm": {
                    "prior_variant": prior_variant,
                    "empirical_uniform_mix": empirical_uniform_mix,
                    "exclude_special_tokens": exclude_special_tokens,
                    "noise_eps": 1e-3,
                    "inference_eps": 1e-5,
                    "sampling_steps": sampling_steps,
                    "time_embedding_size": 64,
                    "zero_init_conditioning": True,
                },
            },
            "optim": {
                "weight_decay": 0.0,
                "lr": 3e-3,
                "beta1": 0.9,
                "beta2": 0.999,
                "eps": 1e-8,
            },
        }
    )


def _batch(model: GenMol):
    converter = sf.SAFEConverter()
    safe_strings = [converter.encoder(smiles, allow_empty=True) for smiles in TOY_SMILES]
    batch = model.tokenizer(
        safe_strings,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=model.config.model.max_position_embeddings,
    )
    return {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}


@torch.no_grad()
def _fixed_diagnostics(model: GenMol, batch, seed: int):
    diagnostics = {}
    mutable = model.diffusion_token_mask(batch["input_ids"], batch["attention_mask"])
    for time_value in (0.1, 0.5, 0.9):
        generator = torch.Generator().manual_seed(seed + round(100 * time_value))
        t = torch.full((batch["input_ids"].shape[0],), time_value)
        xt = model.mdlm.forward_process(
            batch["input_ids"], t, mutable_mask=mutable, generator=generator
        )
        logits = model(xt, batch["attention_mask"], t=t)
        loss = model.mdlm.loss(
            logits,
            batch["input_ids"],
            xt,
            t,
            mask=mutable,
            global_mean=True,
        )
        prediction = logits.argmax(-1)
        accuracy = (prediction[mutable] == batch["input_ids"][mutable]).float().mean()
        diagnostics[str(time_value)] = {
            "loss": float(loss),
            "clean_token_accuracy": float(accuracy),
            "corrupted_token_ids_sha256": _canonical_tensor_sha256(xt),
            "changed_content_tokens": int(
                ((xt != batch["input_ids"]) & mutable).sum().item()
            ),
            "content_token_count": int(mutable.sum().item()),
        }
    return diagnostics


def _sample(model: GenMol, sample_count: int, length: int, sampling_steps: int):
    sampler = Sampler.__new__(Sampler)
    sampler.model = model
    sampler.pad_index = model.pad_index
    sampler.mdlm = model.mdlm
    sampler.diffusion_type = "udlm"
    x = torch.full((sample_count, length + 2), model.mask_index, dtype=torch.long)
    x[:, 0] = model.bos_index
    x[:, -1] = model.eos_index
    return sampler.generate(
        x,
        softmax_temp=1.0,
        randomness=1.0,
        fix=False,
        num_steps=sampling_steps,
    )


def _validate_smoke_gate(losses, before, after, generated):
    """Fail before artifact creation when the declared smoke gate is not met."""
    failures = []
    if not losses or not all(math.isfinite(loss) for loss in losses):
        failures.append("all training losses must be finite")
    else:
        window = min(5, max(1, len(losses) // 2))
        if sum(losses[-window:]) / window >= sum(losses[:window]) / window:
            failures.append("last-window mean loss did not fall below first-window mean")
    if after["0.5"]["loss"] >= before["0.5"]["loss"]:
        failures.append("fixed t=0.5 diagnostic loss did not improve")
    for time_value, before_row in before.items():
        after_row = after.get(time_value, {})
        before_digest = before_row.get("corrupted_token_ids_sha256")
        if before_digest is not None and (
            after_row.get("corrupted_token_ids_sha256") != before_digest
        ):
            failures.append(
                f"fixed t={time_value} corruption changed between diagnostics"
            )
    if not generated:
        failures.append("reverse chain produced no strictly decodable molecule")
    if failures:
        raise RuntimeError("UDLM CPU smoke gate failed: " + "; ".join(failures))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--sampling-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--exclude-special-tokens", action="store_true")
    parser.add_argument(
        "--prior-variant",
        choices=sorted(UDLM_PRIOR_VARIANTS),
        default="release_uniform",
    )
    parser.add_argument("--empirical-uniform-mix", type=float, default=0.01)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    if args.steps <= 0 or args.sample_count <= 0 or args.sampling_steps <= 0:
        raise ValueError("steps, sample-count, and sampling-steps must be positive")
    if not math.isfinite(args.empirical_uniform_mix) or not (
        0.0 < args.empirical_uniform_mix < 1.0
    ):
        raise ValueError("empirical-uniform-mix must lie in (0, 1)")
    output_path = (
        args.output
        if args.output is not None
        else ROOT_DIR
        / "output"
        / "udlm"
        / "cpu_smoke"
        / (
            f"{args.prior_variant}_seed{args.seed}_steps{args.steps}_"
            f"n{args.sample_count}.json"
        )
    ).resolve()
    if output_path != ROOT_DIR and ROOT_DIR not in output_path.parents:
        raise ValueError(f"output must remain inside repository root {ROOT_DIR}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing smoke artifact: {output_path}")

    started = time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    git = _git_provenance()
    _validate_git_provenance(git)
    source_provenance = _source_provenance(args.prior_variant)
    L.seed_everything(args.seed, workers=True)
    config = _config(
        exclude_special_tokens=args.exclude_special_tokens,
        sampling_steps=args.sampling_steps,
        prior_variant=args.prior_variant,
        empirical_uniform_mix=args.empirical_uniform_mix,
    )
    model = GenMol(config)
    model.log = lambda *unused_args, **unused_kwargs: None
    batch = _batch(model)
    clean_batch_sha256 = _canonical_tensor_sha256(batch["input_ids"])
    attention_mask_sha256 = _canonical_tensor_sha256(batch["attention_mask"])
    before = _fixed_diagnostics(model.eval(), batch, args.seed)

    model.train()
    optimizer = torch.optim.AdamW(model.backbone.parameters(), lr=3e-3)
    losses = []
    gradient_norms = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model.training_step(batch, step)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.backbone.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError(f"non-finite gradient at step {step}")
        optimizer.step()
        losses.append(float(loss.detach()))
        gradient_norms.append(float(gradient_norm.detach()))

    after = _fixed_diagnostics(model.eval(), batch, args.seed)
    content_lengths = model.diffusion_token_mask(
        batch["input_ids"], batch["attention_mask"]
    ).sum(-1)
    generated = _sample(
        model,
        sample_count=args.sample_count,
        length=int(content_lengths.median()),
        sampling_steps=args.sampling_steps,
    )
    _validate_smoke_gate(losses, before, after, generated)

    completed_source_provenance = _source_provenance(args.prior_variant)
    completed_git = _git_provenance()
    if completed_source_provenance != source_provenance or completed_git != git:
        raise RuntimeError(
            "source or Git state changed during CPU smoke run; refusing artifact"
        )

    result = {
        "schema_version": 2,
        "purpose": "bounded CPU integration smoke; not benchmark or superiority evidence",
        "claim_scope": (
            "Checks finite optimization, fixed-grid denoising diagnostics, and an "
            "executable reverse chain. Loss magnitudes across release_uniform and "
            "categorical variants are not directly comparable because their schedules "
            "and objectives differ. Sample validity is descriptive at this tiny size."
        ),
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git,
        "source_inputs": source_provenance,
        "seed": args.seed,
        "device": "cpu",
        "steps": args.steps,
        "sampling_steps": args.sampling_steps,
        "sample_count_requested": args.sample_count,
        "no_repair_decodable_samples": len(generated),
        "exclude_special_tokens": args.exclude_special_tokens,
        "prior_variant": args.prior_variant,
        "empirical_uniform_mix_requested": args.empirical_uniform_mix,
        "prior_metadata": model.udlm_prior_metadata.to_dict(),
        "effective_config": OmegaConf.to_container(config, resolve=True),
        "toy_smiles": list(TOY_SMILES),
        "batch_shape": list(batch["input_ids"].shape),
        "clean_input_ids_sha256": clean_batch_sha256,
        "attention_mask_sha256": attention_mask_sha256,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_first_five_mean": sum(losses[:5]) / min(5, len(losses)),
        "loss_last_five_mean": sum(losses[-5:]) / min(5, len(losses)),
        "gradient_norm_first": gradient_norms[0],
        "gradient_norm_last": gradient_norms[-1],
        "all_losses_finite": all(math.isfinite(value) for value in losses),
        "all_gradient_norms_finite": all(
            math.isfinite(value) for value in gradient_norms
        ),
        "fixed_diagnostics_before": before,
        "fixed_diagnostics_after": after,
        "generated_smiles": generated,
        "runtime_seconds": time.time() - started,
    }
    _write_json_exclusive(output_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

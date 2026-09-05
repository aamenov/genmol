"""Cheap end-to-end UDLM overfit and sampling smoke test.

This is intentionally not a benchmark. It uses a tiny BERT and a fixed toy set
to catch integration, numerical, and reverse-chain failures before GPU pilots.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import lightning as L
import safe as sf
import torch
from omegaconf import OmegaConf

from genmol.model import GenMol
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


def _config(*, exclude_special_tokens: bool, sampling_steps: int):
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
        "--output",
        type=Path,
        default=ROOT_DIR / "output" / "udlm" / "cpu_smoke.json",
    )
    args = parser.parse_args()
    if args.steps <= 0 or args.sample_count <= 0 or args.sampling_steps <= 0:
        raise ValueError("steps, sample-count, and sampling-steps must be positive")

    started = time.time()
    L.seed_everything(args.seed, workers=True)
    model = GenMol(
        _config(
            exclude_special_tokens=args.exclude_special_tokens,
            sampling_steps=args.sampling_steps,
        )
    )
    model.log = lambda *unused_args, **unused_kwargs: None
    batch = _batch(model)
    before = _fixed_diagnostics(model.eval(), batch, args.seed)

    model.train()
    optimizer = torch.optim.AdamW(model.backbone.parameters(), lr=3e-3)
    losses = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model.training_step(batch, step)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.backbone.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError(f"non-finite gradient at step {step}")
        optimizer.step()
        losses.append(float(loss.detach()))

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
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = None

    result = {
        "purpose": "integration smoke test; not benchmark evidence",
        "git_sha": git_sha,
        "seed": args.seed,
        "device": "cpu",
        "steps": args.steps,
        "sampling_steps": args.sampling_steps,
        "sample_count_requested": args.sample_count,
        "strict_valid_samples": len(generated),
        "exclude_special_tokens": args.exclude_special_tokens,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_first_five_mean": sum(losses[:5]) / min(5, len(losses)),
        "loss_last_five_mean": sum(losses[-5:]) / min(5, len(losses)),
        "fixed_diagnostics_before": before,
        "fixed_diagnostics_after": after,
        "generated_smiles": generated,
        "runtime_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

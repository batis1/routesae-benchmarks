"""Measured, small language-model activation pilot for RouteSAE ideas.

Run from the repository root. The official RouteSAE class is used unchanged.
All reported reconstruction comparisons within a track share the same target.
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks.variants import (
    GroupedTopK,
    ResidualCascadeSAE,
    Top2RouteSAE,
    normalize_decoders,
)
from src.model import RouteSAE, TopK


def nmse(target, prediction):
    return (((prediction - target).square().mean(dim=-1)) /
            target.square().mean(dim=-1).clamp_min(1e-8)).mean()


def token_ids(tokenizer, split, count):
    rows = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    ids = []
    for row in rows:
        if row["text"].strip():
            ids.extend(tokenizer.encode(row["text"] + "\n", add_special_tokens=False))
        if len(ids) >= count:
            break
    if len(ids) < count:
        raise RuntimeError(f"Only {len(ids)} tokens in {split}; requested {count}")
    return torch.tensor(ids[:count], dtype=torch.long)


@torch.inference_mode()
def collect(model, ids, start, end, seq_length, device):
    chunks = []
    model.eval()
    for offset in range(0, len(ids), seq_length):
        chunk = ids[offset:offset + seq_length].unsqueeze(0).to(device)
        out = model(input_ids=chunk, output_hidden_states=True, use_cache=False)
        stack = torch.stack(out.hidden_states[start:end], dim=2).squeeze(0)
        mean = stack.mean(dim=-1, keepdim=True)
        std = stack.std(dim=-1, keepdim=True)
        chunks.append(((stack - mean) / (std + 1e-6)).to("cpu", dtype=torch.float16))
    return torch.cat(chunks, dim=0)


def make_activations(args, device):
    cache = Path(args.cache)
    if cache.exists():
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        expected = (args.model, args.train_tokens, args.val_tokens, args.seq_length)
        actual = tuple(payload["config"][key] for key in
                       ("model", "train_tokens", "val_tokens", "seq_length"))
        if actual != expected:
            raise ValueError(f"Activation cache config mismatch: {actual} != {expected}")
        return payload
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    lm = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)
    n_layers = lm.config.num_hidden_layers
    start, end = n_layers // 4, n_layers * 3 // 4 + 1
    print(f"Extracting {args.train_tokens}+{args.val_tokens} tokens from {args.model}; "
          f"hidden states [{start}:{end}]", flush=True)
    train_ids = token_ids(tokenizer, "train", args.train_tokens)
    val_ids = token_ids(tokenizer, "validation", args.val_tokens)
    payload = {
        "config": {"model": args.model, "train_tokens": args.train_tokens,
                   "val_tokens": args.val_tokens, "seq_length": args.seq_length,
                   "n_layers": n_layers, "start_layer": start, "end_layer": end,
                   "hidden_size": lm.config.hidden_size,
                   "model_commit": getattr(lm.config, "_commit_hash", None)},
        "train": collect(lm, train_ids, start, end, args.seq_length, device),
        "validation": collect(lm, val_ids, start, end, args.seq_length, device),
    }
    del lm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache)
    return payload


def train(model, data, objective, args, device):
    model.to(device).train()
    normalize_decoders(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    data = data.to(device, dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1000)
    start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(args.steps):
        index = torch.randint(len(data), (args.batch_tokens,), generator=generator)
        inputs = data[index.to(device)]
        loss = objective(model, inputs)[0]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        normalize_decoders(model)
        if (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            print(f"  step {step + 1}/{args.steps} train_nmse={loss.item():.4f}", flush=True)
    return {
        "train_seconds": round(time.perf_counter() - start, 2),
        "peak_gpu_mb": round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
        if torch.cuda.is_available() else None,
        "parameters": sum(p.numel() for p in model.parameters()),
    }


@torch.inference_mode()
def evaluate(model, data, objective, args, device):
    model.eval()
    data = data.to(device, dtype=torch.float32)
    losses = []
    active = None
    l0 = []
    for batch in data.split(args.batch_tokens):
        loss, latents = objective(model, batch)
        losses.append((loss.item(), len(batch)))
        nonzero = latents != 0
        l0.append((nonzero.sum(dim=-1).float().mean().item(), len(batch)))
        counts = nonzero.reshape(-1, nonzero.shape[-1]).sum(dim=0).cpu()
        active = counts if active is None else active + counts
    denominator = sum(size for _, size in losses)
    return {
        "val_nmse": sum(value * size for value, size in losses) / denominator,
        "val_l0": sum(value * size for value, size in l0) / denominator,
        "alive_features": int((active > 0).sum()),
    }


def routed_objective(model, x):
    _, routed, latents, reconstruction, _ = model(x.unsqueeze(1), "sum", "hard")
    return nmse(routed, reconstruction), latents


def dictionary_objective(model, x):
    latents, reconstruction = model(x)
    return nmse(x, reconstruction), latents


def route_goal_objective(model, x, routing):
    _, _, latents, reconstruction, _ = model(x.unsqueeze(1), "sum", routing)
    return nmse(x[:, -1:, :], reconstruction), latents


def cascade_objective(model, x):
    latents, reconstruction = model(x)
    return nmse(x[:, -1, :], reconstruction), latents


@torch.inference_mode()
def freeze_route(baseline, activations, device, batch_tokens):
    baseline.eval()
    result = []
    for chunk in activations.split(batch_tokens):
        x = chunk.to(device, dtype=torch.float32).unsqueeze(1)
        weights = baseline.get_router_weights(x, "sum")
        _, routed = baseline.get_sae_input(x, weights, "hard")
        result.append(routed.squeeze(1).to("cpu", dtype=torch.float16))
    return torch.cat(result)


def run_candidate(name, model, train_data, val_data, objective, args, device):
    print(f"Training {name}", flush=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    training = train(model, train_data, objective, args, device)
    metrics = evaluate(model, val_data, objective, args, device)
    return {**training, **metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--train-tokens", type=int, default=8192)
    parser.add_argument("--val-tokens", type=int, default=2048)
    parser.add_argument("--seq-length", type=int, default=128)
    parser.add_argument("--latent-multiplier", type=int, default=4)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-tokens", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", default="results/activations.pt")
    parser.add_argument("--output", default="results/pilot.json")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_tokens < 1:
        parser.error("steps and batch-tokens must be positive")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    payload = make_activations(args, device)
    train_stack, val_stack = payload["train"], payload["validation"]
    info = payload["config"]
    d, n = info["hidden_size"], info["n_layers"]
    width = d * args.latent_multiplier
    if width % args.groups:
        parser.error("latent width must be divisible by groups")
    results = {
        "status": "exploratory_pilot",
        "source_route_sae_commit": "d1a3c06415c00619b3cf48895ed2a569d8b0ac5f",
        "args": vars(args), "model": info,
        "hardware": {"device": str(device),
                     "gpu": torch.cuda.get_device_name(device) if torch.cuda.is_available() else None,
                     "torch": torch.__version__},
        "tracks": {},
        "limits": [
            "One small corpus and one seed do not establish interpretability or superiority.",
            "RouteSAE and top-2 routed-input NMSE have different targets and must not be ranked.",
            "Alive feature count is not monosemanticity or causal faithfulness."
        ],
    }
    baseline = RouteSAE(d, n, width, args.k)
    results["tracks"]["official_route_reconstruction"] = {
        "official_RouteSAE_hard": run_candidate(
            "official RouteSAE hard", baseline, train_stack, val_stack,
            routed_objective, args, device)
    }
    fixed_train = freeze_route(baseline, train_stack, device, args.batch_tokens)
    fixed_val = freeze_route(baseline, val_stack, device, args.batch_tokens)
    dictionary = {}
    for name, constructor in (
        ("flat_TopK", lambda: TopK(d, width, args.k)),
        ("hierarchical_1_of_8", lambda: GroupedTopK(d, width, args.k, args.groups, 1)),
        ("subspace_2_of_8", lambda: GroupedTopK(d, width, args.k, args.groups, 2)),
    ):
        torch.manual_seed(args.seed)
        dictionary[name] = run_candidate(
            name, constructor(), fixed_train, fixed_val,
            dictionary_objective, args, device)
    results["tracks"]["fixed_route_dictionary"] = dictionary
    routing = {}
    for name, constructor, mode in (
        ("top1_late_target", lambda: RouteSAE(d, n, width, args.k), "hard"),
        ("top2_late_target", lambda: Top2RouteSAE(d, n, width, args.k), "top2"),
    ):
        torch.manual_seed(args.seed)
        objective = lambda model, x, mode=mode: route_goal_objective(model, x, mode)
        routing[name] = run_candidate(name, constructor(), train_stack, val_stack,
                                      objective, args, device)
    results["tracks"]["common_late_target_routing"] = routing
    cascade = {}
    for name, constructor, objective, data_train, data_val in (
        ("late_flat_TopK", lambda: TopK(d, width, args.k), dictionary_objective,
         train_stack[:, -1, :], val_stack[:, -1, :]),
        ("residual_cascade", lambda: ResidualCascadeSAE(d, width, args.k),
         cascade_objective, train_stack, val_stack),
    ):
        torch.manual_seed(args.seed)
        cascade[name] = run_candidate(name, constructor(), data_train, data_val,
                                      objective, args, device)
    results["tracks"]["common_late_target_cascade"] = cascade
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output}", flush=True)
    print(json.dumps(results["tracks"], indent=2), flush=True)


if __name__ == "__main__":
    main()

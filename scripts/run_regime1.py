#!/usr/bin/env python3
"""
Run LineageVI regime 1 only (expression reconstruction). Saves regime1 checkpoint and adata with latent.
Writes regime1/pretrained_vae_regime1.pt, regime1/model_config.json, regime1/regime1_params.txt. AnnData is written at the end of regime 2.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import scanpy as sc

import lineagevi as lvi


def _write_params(path: str, params: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for k, v in sorted(params.items()):
            if isinstance(v, (list, tuple)):
                import json
                v = json.dumps(v)
            f.write(f"{k} = {v}\n")


def main():
    p = argparse.ArgumentParser(description="LineageVI regime 1 only (save checkpoint + adata with latent)")
    p.add_argument("--adata_path", type=str, required=True, help="Path to adata_preprocessed.h5ad")
    p.add_argument("--output_dir", type=str, default="./outputs/pancreas")
    p.add_argument("--epochs1", type=int, default=400)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr_regime1", type=float, default=1e-3)
    p.add_argument("--train_size", type=float, default=0.9)
    p.add_argument("--kl_weight_schedule", type=str, default="linear")
    p.add_argument("--kl_weight_min", type=float, default=0.0)
    p.add_argument("--kl_weight_max", type=float, default=0.1)
    p.add_argument("--kl_cycle_ramp_frac", type=float, default=0.2)
    p.add_argument("--n_hidden", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--cluster_embedding_dim", type=int, default=32)
    p.add_argument("--cluster_key", type=str, default="leiden")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_timestamp", action="store_true", help="Do not append timestamp to output_dir")
    args = p.parse_args()

    output_base = args.output_dir.rstrip("/")
    output_dir = output_base
    if not args.no_timestamp:
        output_dir = output_base + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    regime1_dir = os.path.join(output_dir, "regime1")
    os.makedirs(regime1_dir, exist_ok=True)

    adata = sc.read_h5ad(args.adata_path)
    vae = lvi.LineageVI(
        adata,
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        dropout=args.dropout,
        cluster_embedding_dim=args.cluster_embedding_dim,
        cluster_key=args.cluster_key if args.cluster_key else None,
    )
    seeds = (args.seed, args.seed + 1, args.seed + 2)
    vae.fit(
        K=args.K,
        batch_size=args.batch_size,
        lr_regime1=args.lr_regime1,
        epochs1=args.epochs1,
        epochs2=0,
        seeds=seeds,
        train_size=args.train_size,
        kl_weight_schedule=args.kl_weight_schedule,
        kl_weight_min=args.kl_weight_min,
        kl_weight_max=args.kl_weight_max,
        kl_cycle_ramp_frac=args.kl_cycle_ramp_frac,
        output_dir=regime1_dir,
        show_metrics=False,
    )

    params = {
        "adata_path": args.adata_path,
        "output_dir": output_dir,
        "epochs1": args.epochs1,
        "K": args.K,
        "batch_size": args.batch_size,
        "lr_regime1": args.lr_regime1,
        "train_size": args.train_size,
        "kl_weight_schedule": args.kl_weight_schedule,
        "kl_weight_min": args.kl_weight_min,
        "kl_weight_max": args.kl_weight_max,
        "kl_cycle_ramp_frac": args.kl_cycle_ramp_frac,
        "n_hidden": args.n_hidden,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
        "cluster_embedding_dim": args.cluster_embedding_dim,
        "cluster_key": args.cluster_key,
        "seed": args.seed,
    }
    _write_params(os.path.join(regime1_dir, "regime1_params.txt"), params)
    print(f"Saved {regime1_dir}/regime1_params.txt")


if __name__ == "__main__":
    main()

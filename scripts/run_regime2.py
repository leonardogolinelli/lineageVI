#!/usr/bin/env python3
"""
Run LineageVI regime 2 only (velocity), loading a regime-1 checkpoint. Saves final model and AnnData to output_dir.
Writes pretrained_vae.pt, model_config.json, adata_with_latent.h5ad, regime2_params.txt.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import scanpy as sc

import lineagevi as lvi
from lineagevi import utils as lvi_utils


def _write_params(path: str, params: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for k, v in sorted(params.items()):
            if isinstance(v, (list, tuple)):
                import json
                v = json.dumps(v)
            f.write(f"{k} = {v}\n")


def main():
    p = argparse.ArgumentParser(description="LineageVI regime 2 only (load regime-1 checkpoint, train velocity, save final model)")
    p.add_argument("--adata_path", type=str, required=True, help="Path to preprocessed adata (e.g. .../adata_preprocessed.h5ad); latent is computed from regime-1 encoder if missing")
    p.add_argument("--checkpoint_path", type=str, required=True, help="Path to regime-1 .pt checkpoint")
    p.add_argument("--output_dir", type=str, default="./outputs/pancreas")
    p.add_argument("--epochs2", type=int, default=150)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr_regime2", type=float, default=1e-3)
    p.add_argument("--train_size", type=float, default=0.9)
    p.add_argument("--velocity_loss_weight_gene", type=float, default=1.0)
    p.add_argument("--velocity_loss_weight_gp", type=float, default=1.0)
    p.add_argument("--seed_regime2", type=int, default=2)
    p.add_argument("--monitor_genes", type=str, nargs="*", default=None)
    p.add_argument("--monitor_every_epochs", type=int, default=25)
    p.add_argument("--no_timestamp", action="store_true", help="Do not append timestamp to output_dir")
    args = p.parse_args()

    output_dir = args.output_dir.rstrip("/")
    if not args.no_timestamp:
        output_dir = output_dir + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    adata = sc.read_h5ad(args.adata_path)
    linvi = lvi_utils.load_model(adata, args.checkpoint_path, training=True)
    linvi.fit_regime2_only(
        K=args.K,
        batch_size=args.batch_size,
        lr_regime2=args.lr_regime2,
        epochs2=args.epochs2,
        seed_regime2=args.seed_regime2,
        train_size=args.train_size,
        velocity_loss_weight_gene=args.velocity_loss_weight_gene,
        velocity_loss_weight_gp=args.velocity_loss_weight_gp,
        output_dir=output_dir,
        monitor_genes=args.monitor_genes,
        monitor_every_epochs=args.monitor_every_epochs,
        show_metrics=False,
    )

    out_adata_path = os.path.join(output_dir, "adata_with_latent.h5ad")
    linvi.adata.write_h5ad(out_adata_path)
    print(f"Saved {out_adata_path}")

    params = {
        "adata_path": args.adata_path,
        "checkpoint_path": args.checkpoint_path,
        "output_dir": output_dir,
        "epochs2": args.epochs2,
        "K": args.K,
        "batch_size": args.batch_size,
        "lr_regime2": args.lr_regime2,
        "train_size": args.train_size,
        "velocity_loss_weight_gene": args.velocity_loss_weight_gene,
        "velocity_loss_weight_gp": args.velocity_loss_weight_gp,
        "seed_regime2": args.seed_regime2,
        "monitor_every_epochs": args.monitor_every_epochs,
    }
    _write_params(os.path.join(output_dir, "regime2_params.txt"), params)
    print(f"Saved {output_dir}/regime2_params.txt")


if __name__ == "__main__":
    main()

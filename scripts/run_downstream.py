#!/usr/bin/env python3
"""
Downstream LineageVI: follows Tutorial_Pancreas_Clean_V2.ipynb exactly.
Load model and AnnData from regime2 output, get_model_outputs, velocity UMAP/pseudotime,
build_gp_adata, differential tests, plot_differential, heatmap (top velocity GPs), optional perturbation.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import scanpy as sc

import lineagevi.plots as lvi_plots
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
    p = argparse.ArgumentParser(description="LineageVI downstream (tutorial V2 order)")
    p.add_argument("--adata_path", type=str, required=True, help="Path to adata from regime2 (e.g. <regime2_output>/adata_with_latent.h5ad)")
    p.add_argument("--model_path", type=str, required=True, help="Path to model from regime2 (e.g. <regime2_output>/pretrained_vae.pt)")
    p.add_argument("--output_dir", type=str, default="./outputs/pancreas")
    p.add_argument("--groupby_key", type=str, default="leiden", help="Key in adata.obs for clusters (tutorial uses 'clusters')")
    p.add_argument("--run_perturbation", action="store_true", help="Run perturbation (tutorial: optional)")
    p.add_argument("--perturb_group", type=str, default=None, help="Group to perturb (e.g. Alpha, Beta)")
    p.add_argument("--no_timestamp", action="store_true", help="Do not append timestamp to output_dir")
    args = p.parse_args()

    output_dir = args.output_dir.rstrip("/")
    if not args.no_timestamp:
        output_dir = output_dir + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    figures_dir = os.path.join(output_dir, "figures")
    differential_dir = os.path.join(output_dir, "differential")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(differential_dir, exist_ok=True)

    adata = sc.read_h5ad(args.adata_path)
    vae = lvi_utils.load_model(adata, args.model_path, training=False)

    # Tutorial: get_model_outputs
    vae.get_model_outputs(
        adata=adata,
        return_negative_velo=False,
        base_seed=0,
        save_to_adata=True,
        unspliced_key="Mu",
        spliced_key="Ms",
        nn_key="indices",
        rescale_velocity_magnitude=False,
        max_velocity_magnitude=1,
    )

    # Tutorial: velocity UMAP and pseudotime (gene space)
    import scvelo as scv
    cluster_key = args.groupby_key
    if "X_umap" not in adata.obsm:
        sc.pp.neighbors(adata, use_rep="X_pca" if "X_pca" in adata.obsm else "z")
        sc.tl.umap(adata)
    scv.tl.velocity_graph(adata)
    scv.pl.velocity_embedding_stream(adata, color=cluster_key, save=os.path.join(figures_dir, "velocity_umap_clusters.png"), show=False)
    scv.tl.velocity_pseudotime(adata)
    scv.pl.velocity_embedding_stream(adata, color="velocity_pseudotime", save=os.path.join(figures_dir, "velocity_umap.png"), show=False)

    # Tutorial: build_gp_adata and velocity in GP space
    adata_gp = lvi_utils.build_gp_adata(adata)
    adata_gp.write_h5ad(os.path.join(output_dir, "adata_gp.h5ad"))
    sc.tl.umap(adata_gp)
    scv.tl.velocity_graph(adata_gp)
    scv.pl.velocity_embedding_stream(adata_gp, color=cluster_key, save=os.path.join(figures_dir, "velocity_umap_gp_clusters.png"), show=False)
    scv.tl.velocity_pseudotime(adata_gp)
    scv.pl.velocity_embedding_stream(adata_gp, color="velocity_pseudotime", save=os.path.join(figures_dir, "velocity_umap_gp.png"), show=False)

    # Tutorial: differential (Ms, velocity, mean, velocity_gp) and plot_differential
    de = vae.differential(adata, cluster_key, layer="Ms", ensure_model_outputs=False)
    dvelo = vae.differential(adata, cluster_key, layer="velocity", ensure_model_outputs=False)
    dlatent = vae.differential(adata, cluster_key, obsm="mean", ensure_model_outputs=False)
    dgp = vae.differential(adata, cluster_key, obsm="velocity_gp", ensure_model_outputs=False)
    adata.uns["diff_Ms"] = de
    adata.uns["diff_velocity"] = dvelo
    adata.uns["diff_latent"] = dlatent
    adata.uns["diff_gp_velo"] = dgp

    for key, label in [("diff_Ms", "Ms"), ("diff_velocity", "velocity"), ("diff_latent", "mean"), ("diff_gp_velo", "velocity_gp")]:
        fig = lvi_plots.plot_differential(adata, scores_key=key, n_points=10, lim_val=1.3)
        if fig is not None:
            fig.savefig(os.path.join(figures_dir, f"differential_{label}.png"), dpi=150, bbox_inches="tight")
        scores = adata.uns[key]
        for group_name, df in scores.items():
            safe_name = group_name.replace("/", "_").replace(" ", "_")
            df.to_csv(os.path.join(differential_dir, f"{key}_{safe_name}.csv"))

    # Tutorial: top_features_table then heatmap (top velocity GPs)
    df = lvi_plots.top_features_table(adata_gp, groupby_key=cluster_key, categories="all", layer="velocity", n=10)
    lvi_plots.heatmap(
        adata_gp,
        list(df.feature),
        sortby="velocity_pseudotime",
        layer="Ms",
        col_color=cluster_key,
        figsize=(10, 6),
        show=False,
        save=os.path.join(figures_dir, "heatmap_gp_velocity.png"),
    )

    # Tutorial: perturbation (optional)
    if args.run_perturbation and args.perturb_group and cluster_key in adata.obs.columns:
        try:
            vae.perturb(adata, mode="genes", groupby_key=cluster_key, group_to_perturb=args.perturb_group, perturb_value=1.0)
        except Exception as e:
            print(f"Perturbation skipped: {e}")

    params = {
        "adata_path": args.adata_path,
        "model_path": args.model_path,
        "output_dir": output_dir,
        "groupby_key": cluster_key,
    }
    _write_params(os.path.join(output_dir, "downstream_params.txt"), params)
    print(f"Saved {output_dir}/downstream_params.txt")


if __name__ == "__main__":
    main()

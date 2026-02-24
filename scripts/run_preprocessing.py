#!/usr/bin/env python3
"""
Preprocessing for LineageVI pipeline. Follows the exact sequence from Tutorial_Pancreas_Clean_V2.ipynb.
All steps and hyperparameters are tunable via CLI. Writes adata_preprocessed.h5ad and preprocessing_params.txt.

Uses one Enrichr library for annotations (e.g. GO_Biological_Process_2025); GMT is taken from gene_sets_dir and downloaded if not present.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import scanpy as sc
import scvelo as scv

import lineagevi.utils as lvi_utils


def _library_name_to_gmt_path(library_name: str, gene_sets_dir: str) -> str:
    """Same filename convention as download_enrichr_libraries."""
    safe = re.sub(r"[^\w\-.]", "_", library_name)
    return os.path.join(gene_sets_dir, f"{safe}.gmt")


def _write_params(path: str, params: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for k, v in sorted(params.items()):
            if isinstance(v, (list, tuple)):
                v = json.dumps(v)
            f.write(f"{k} = {v}\n")


def main():
    p = argparse.ArgumentParser(description="Preprocess data for LineageVI (tutorial sequence)")
    p.add_argument("--dataset", type=str, default="pancreas", help="scvelo dataset name (e.g. pancreas) or path to .h5ad")
    # Annotation: Enrichr library name; GMT is taken from gene_sets_dir and downloaded if missing
    p.add_argument("--gene_sets_dir", type=str, default="./gene_sets", help="Directory for GMT files (downloads here if not present)")
    p.add_argument("--annotation_library", type=str, default=None, help="Enrichr library name to use (e.g. GO_Biological_Process_2025). Downloaded to gene_sets_dir if not present.")
    p.add_argument("--min_shared_counts", type=int, default=20)
    p.add_argument("--n_top_genes", type=int, default=3000)
    p.add_argument("--min_genes_per_term", type=int, default=12)
    p.add_argument("--n_pcs", type=int, default=100)
    p.add_argument("--n_neighbors", type=int, default=200)
    p.add_argument("--K_neighbors", type=int, default=20)
    p.add_argument("--cluster_key", type=str, default="leiden")
    p.add_argument("--clean_terms", type=lambda x: x.lower() in ("true", "1", "yes"), default=False)
    p.add_argument("--output_dir", type=str, default="./outputs/pancreas")
    p.add_argument("--no_timestamp", action="store_true", help="Do not append timestamp to output_dir")
    args = p.parse_args()

    # Resolve annotation file: Enrichr library in gene_sets_dir; download if not present
    if not args.annotation_library:
        p.error("--annotation_library is required (e.g. GO_Biological_Process_2025)")
    gene_sets_dir = os.path.abspath(args.gene_sets_dir)
    os.makedirs(gene_sets_dir, exist_ok=True)
    gmt_path = _library_name_to_gmt_path(args.annotation_library, gene_sets_dir)
    if not os.path.isfile(gmt_path):
        lvi_utils.download_enrichr_libraries(
            [args.annotation_library],
            output_dir=gene_sets_dir,
            skip_existing=False,
        )
    if not os.path.isfile(gmt_path):
        raise FileNotFoundError(f"Expected GMT at {gmt_path} after download. Check library name: {args.annotation_library!r}")
    annotation_files = [gmt_path]

    output_base = args.output_dir.rstrip("/")
    output_dir = output_base
    if not args.no_timestamp:
        output_dir = output_base + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    if output_dir != output_base:
        latest_file = output_base + ".latest.txt"
        os.makedirs(os.path.dirname(latest_file) or ".", exist_ok=True)
        with open(latest_file, "w") as f:
            f.write(output_dir + "\n")

    # --- 1. Load data (tutorial: scv.datasets.pancreas() or sc.read_h5ad) ---
    if args.dataset.endswith(".h5ad") or os.path.isfile(args.dataset):
        adata = sc.read_h5ad(args.dataset)
    else:
        adata = getattr(scv.datasets, args.dataset)()

    # --- 2. Obtain raw counts for the annotation step (tutorial) ---
    adata.X = adata.layers["unspliced"].copy() + adata.layers["spliced"].copy()
    adata.layers["counts"] = adata.X.copy()

    # --- 3. filter_and_normalize (tutorial) ---
    scv.pp.filter_and_normalize(
        adata,
        min_shared_counts=args.min_shared_counts,
        n_top_genes=args.n_top_genes,
        subset_highly_variable=True,
        log=True,
    )

    # --- 4. moments (tutorial) ---
    scv.pp.moments(adata, n_pcs=args.n_pcs, n_neighbors=args.n_neighbors)

    # --- 5. leiden (tutorial) ---
    sc.tl.leiden(adata, key_added=args.cluster_key)

    # --- 6. add_annotations (tutorial: add_annotations then subset var) ---
    lvi_utils.add_annotations(
        adata,
        files=annotation_files,
        min_genes=args.min_genes_per_term,
        varm_key="mask",
        uns_key="terms",
        clean=args.clean_terms,
        genes_use_upper=True,
    )
    adata._inplace_subset_var(adata.varm["mask"].sum(1) > 0)

    # --- 7. Remove terms with too few genes, then again remove genes not in any retained term (tutorial) ---
    select_terms = adata.varm["mask"].sum(0) > args.min_genes_per_term
    adata.uns["terms"] = np.array(adata.uns["terms"])[select_terms].tolist()
    adata.varm["mask"] = adata.varm["mask"][:, select_terms]
    adata._inplace_subset_var(adata.varm["mask"].sum(1) > 0)

    # --- 8. get_neighbor_indices (tutorial) ---
    lvi_utils.get_neighbor_indices(
        adata,
        K=args.K_neighbors,
        neighbors_key="neighbors",
        indices_key="indices",
    )

    out_h5ad = os.path.join(output_dir, "adata_preprocessed.h5ad")
    adata.write_h5ad(out_h5ad)
    print(f"Saved {out_h5ad}")

    params = {
        "dataset": args.dataset,
        "annotation_library": args.annotation_library,
        "gene_sets_dir": gene_sets_dir,
        "annotation_file": annotation_files[0],
        "min_shared_counts": args.min_shared_counts,
        "n_top_genes": args.n_top_genes,
        "min_genes_per_term": args.min_genes_per_term,
        "n_pcs": args.n_pcs,
        "n_neighbors": args.n_neighbors,
        "K_neighbors": args.K_neighbors,
        "cluster_key": args.cluster_key,
        "clean_terms": args.clean_terms,
        "output_dir": output_dir,
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
        "n_terms": int(adata.varm["mask"].shape[1]),
    }
    params_path = os.path.join(output_dir, "preprocessing_params.txt")
    _write_params(params_path, params)
    print(f"Saved {params_path}")


if __name__ == "__main__":
    main()

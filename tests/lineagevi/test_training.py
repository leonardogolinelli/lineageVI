"""Test script for LineageVI model training.

This script tests the basic training workflow for the LineageVI VAE model.
Run with:
    python -m tests.lineagevi.test_training --dataset_name pancreas
    or
    python -m tests.lineagevi.test_training --adata_path <path_to_data.h5ad>
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch
import scanpy as sc
import scvelo as scv
import lineagevi as lvi
from lineagevi.utils import preprocess_for_lineagevi


def parse_seeds(seeds_str: str) -> tuple[int, int, int]:
    """Parse comma-separated seeds string into tuple."""
    seeds_list = [int(s.strip()) for s in seeds_str.split(",")]
    if len(seeds_list) != 3:
        raise ValueError(f"seeds must contain exactly 3 comma-separated integers, got: {seeds_str}")
    return tuple(seeds_list)


def main():
    parser = argparse.ArgumentParser(description="Test LineageVI model training")
    
    # Data loading (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--dataset_name", type=str, help="Load from scvelo datasets (e.g., 'pancreas')")
    data_group.add_argument("--adata_path", type=str, help="Path to AnnData file (.h5ad)")
    parser.add_argument("--annotation_file", type=str, help="Path to annotation file (.gmt)")
    
    # Preprocessing
    parser.add_argument("--min_shared_counts", type=int, default=20, help="Minimum shared counts")
    parser.add_argument("--n_top_genes", type=int, default=2000, help="Number of top HVGs")
    parser.add_argument("--min_genes_per_term", type=int, default=12, help="Min genes per annotation term")
    parser.add_argument("--n_pcs", type=int, default=100, help="Number of PCs for moments")
    parser.add_argument("--n_neighbors", type=int, default=200, help="Neighbors for moments smoothing")
    parser.add_argument("--K_neighbors", type=int, default=20, help="Neighbors for model")
    parser.add_argument("--skip_if_preprocessed", type=str, default="true", help="Skip if already preprocessed")
    parser.add_argument("--cluster_key", type=str, help="Key for cluster labels in adata.obs")
    
    # Model initialization
    parser.add_argument("--n_hidden", type=int, default=128, help="Hidden units")
    parser.add_argument("--mask_key", type=str, default="mask", help="Gene program mask key")
    parser.add_argument("--unspliced_key", type=str, default="unspliced", help="Unspliced layer key")
    parser.add_argument("--spliced_key", type=str, default="spliced", help="Spliced layer key")
    parser.add_argument("--nn_key", type=str, default="indices", help="Neighbor indices key")
    parser.add_argument("--cluster_embedding_dim", type=int, default=8, help="Cluster embedding dim")
    def seed_type(x):
        """Convert string to int or None."""
        if x is None or x == "" or (isinstance(x, str) and x.lower() == "none"):
            return None
        return int(x)
    
    parser.add_argument("--seed", type=seed_type, default=None, help="Random seed (default: None for non-deterministic)")
    
    # Training
    parser.add_argument("--K", type=int, default=10, help="Number of neighbors for loss")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs1", type=int, default=50, help="Epochs for regime 1")
    parser.add_argument("--epochs2", type=int, default=50, help="Epochs for regime 2")
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seeds (default: 0,1,2)")
    parser.add_argument("--output_dir", type=str, default="./test_outputs/lineagevi", help="Output directory")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    parser.add_argument("--monitor_genes", type=str, help="Comma-separated gene names to monitor")
    parser.add_argument("--monitor_negative_velo", type=str, default="true", help="Monitor negative velocities")
    parser.add_argument("--monitor_every_epochs", type=int, default=5, help="Monitor frequency")
    parser.add_argument("--plot_cluster_key", type=str, help="Cluster key for velocity embedding plots (default: cluster_key or 'leiden')")
    
    args = parser.parse_args()
    
    # Convert string booleans to actual booleans
    skip_if_preprocessed = args.skip_if_preprocessed.lower() in ("true", "1", "yes")
    monitor_negative_velo = args.monitor_negative_velo.lower() in ("true", "1", "yes")
    
    # Parse seeds
    seeds_tuple = parse_seeds(args.seeds)
    
    # Parse monitor_genes
    monitor_genes_list = None
    if args.monitor_genes:
        monitor_genes_list = [g.strip() for g in args.monitor_genes.split(",")]
    
    # Preprocess data
    print("Preprocessing data...")
    adata = preprocess_for_lineagevi(
        dataset_name=args.dataset_name,
        adata_path=args.adata_path,
        annotation_file=args.annotation_file,
        min_shared_counts=args.min_shared_counts,
        n_top_genes=args.n_top_genes,
        min_genes_per_term=args.min_genes_per_term,
        n_pcs=args.n_pcs,
        n_neighbors=args.n_neighbors,
        K_neighbors=args.K_neighbors,
        skip_if_preprocessed=skip_if_preprocessed,
        cluster_key=args.cluster_key,
    )
    print(f"Preprocessed data: {adata.n_obs} cells, {adata.n_vars} genes")
    
    # Initialize model
    print("Initializing LineageVI model...")
    linvi = lvi.LineageVI(
        adata,
        n_hidden=args.n_hidden,
        mask_key=args.mask_key,
        unspliced_key=args.unspliced_key,
        spliced_key=args.spliced_key,
        nn_key=args.nn_key,
        seed=args.seed,
        cluster_key=args.cluster_key,
        cluster_embedding_dim=args.cluster_embedding_dim,
    )
    
    # Train model
    print(f"Training model (regime 1: {args.epochs1} epochs, regime 2: {args.epochs2} epochs)...")
    history = linvi.fit(
        K=args.K,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs1=args.epochs1,
        epochs2=args.epochs2,
        seeds=seeds_tuple,
        output_dir=args.output_dir,
        verbose=args.verbose,
        monitor_genes=monitor_genes_list,
        monitor_negative_velo=monitor_negative_velo,
        monitor_every_epochs=args.monitor_every_epochs,
    )
    
    # Get model outputs
    print("Computing model outputs...")
    linvi.get_model_outputs(adata, save_to_adata=True)
    
    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "test_model.pt"
    torch.save(linvi.model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Save adata with outputs
    adata_path = output_dir / "test_outputs.h5ad"
    adata.write(adata_path)
    print(f"AnnData with outputs saved to {adata_path}")
    
    # Generate velocity embedding plots
    print("Generating velocity embedding plots...")
    
    # Determine cluster key for plotting (may differ from training cluster key)
    plot_cluster_key = args.plot_cluster_key
    if plot_cluster_key is None:
        # Try to use training cluster key, or fall back to 'leiden'
        plot_cluster_key = args.cluster_key if args.cluster_key else "leiden"
    
    # Check if cluster key exists
    if plot_cluster_key not in adata.obs.columns:
        print(f"Warning: Cluster key '{plot_cluster_key}' not found in adata.obs. Available keys: {list(adata.obs.columns)}")
        print("Skipping velocity embedding plots.")
    else:
        plots_dir = output_dir / "velocity_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot 1: Gene expression space
        print(f"  Computing neighbors and UMAP for gene expression space...")
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        print(f"  Computing velocity graph for gene expression space...")
        scv.tl.velocity_graph(adata)
        print(f"  Plotting velocity embedding stream (gene expression space)...")
        scv.pl.velocity_embedding_stream(
            adata, 
            color=plot_cluster_key,
            save=f"{plots_dir}/velocity_embedding_gene_space.png",
            show=False
        )
        print(f"  Saved to {plots_dir}/velocity_embedding_gene_space.png")
        
        # Plot 2: Gene program space
        print(f"  Building gene program adata...")
        adata_gp = lvi.utils.build_gp_adata(adata)
        print(f"  Computing neighbors and UMAP for gene program space...")
        sc.pp.neighbors(adata_gp)
        sc.tl.umap(adata_gp)
        print(f"  Computing velocity graph for gene program space...")
        scv.tl.velocity_graph(adata_gp)
        print(f"  Plotting velocity embedding stream (gene program space)...")
        scv.pl.velocity_embedding_stream(
            adata_gp, 
            color=plot_cluster_key,
            save=f"{plots_dir}/velocity_embedding_gp_space.png",
            show=False
        )
        print(f"  Saved to {plots_dir}/velocity_embedding_gp_space.png")
    
    print("Training test completed successfully!")


if __name__ == "__main__":
    main()

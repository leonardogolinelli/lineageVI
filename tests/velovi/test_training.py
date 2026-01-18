"""Test script for LineageVI model training.

This script tests the basic training workflow for the LineageVI VAE model.
Run with:
    python -m tests.velovi.test_training --adata_path <path_to_data.h5ad>
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch
import scanpy as sc
import lineagevi as lvi


def main():
    parser = argparse.ArgumentParser(description="Test LineageVI model training")
    parser.add_argument("--adata_path", type=str, required=True, help="Path to AnnData file")
    parser.add_argument("--output_dir", type=str, default="./test_outputs/velovi", help="Output directory")
    parser.add_argument("--epochs1", type=int, default=10, help="Epochs for regime 1 (reconstruction)")
    parser.add_argument("--epochs2", type=int, default=10, help="Epochs for regime 2 (velocity)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.adata_path}...")
    adata = sc.read_h5ad(args.adata_path)
    print(f"Loaded {adata.n_obs} cells, {adata.n_vars} genes")
    
    # Initialize model
    print("Initializing LineageVI model...")
    linvi = lvi.LineageVI(adata, seed=args.seed)
    
    # Train model
    print(f"Training model (regime 1: {args.epochs1} epochs, regime 2: {args.epochs2} epochs)...")
    history = linvi.fit(epochs1=args.epochs1, epochs2=args.epochs2)
    
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
    
    print("Training test completed successfully!")


if __name__ == "__main__":
    main()

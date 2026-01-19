"""Standalone script to fit and save a GMM on latent states."""

import argparse
from pathlib import Path
import numpy as np
import scanpy as sc
from sklearn.mixture import GaussianMixture
import joblib


def fit_gmm(
    adata_path: str,
    z_key: str,
    out_path: str,
    n_components: int = 32,
    seed: int = 0,
):
    """
    Fit a GMM on latent states and save to disk.
    
    Parameters
    ----------
    adata_path : str
        Path to AnnData file.
    z_key : str
        Key in adata.obsm for latent states.
    out_path : str
        Output path for saved GMM.
    n_components : int
        Number of GMM components.
    seed : int
        Random seed.
    """
    # Load AnnData
    print(f"Loading AnnData from {adata_path}...")
    adata = sc.read_h5ad(adata_path)
    print(f"Loaded {adata.n_obs} cells, {adata.n_vars} genes")
    
    # Extract latent states
    if z_key not in adata.obsm:
        raise ValueError(f"Key '{z_key}' not found in adata.obsm. Available keys: {list(adata.obsm.keys())}")
    
    Z = np.asarray(adata.obsm[z_key], dtype=np.float64)  # (n_cells, n_latent)
    print(f"Extracted latent states: shape {Z.shape}")
    
    # Fit GMM
    print(f"Fitting GMM with {n_components} components...")
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        reg_covar=1e-6,
        max_iter=200,
        n_init=2,
        random_state=seed,
    )
    gmm.fit(Z)
    
    # Print summary
    print(f"\nGMM Summary:")
    print(f"  Components: {n_components}")
    print(f"  Converged: {gmm.converged_}")
    print(f"  Average train log-likelihood: {gmm.score(Z):.4f}")
    print(f"  Latent dimension: {Z.shape[1]}")
    
    # Save
    out_path_obj = Path(out_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(gmm, out_path)
    print(f"\nSaved GMM to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Fit and save GMM on latent states")
    parser.add_argument("--adata_path", type=str, required=True, help="Path to AnnData file")
    parser.add_argument("--z_key", type=str, default="mean", help="Key in adata.obsm for latent states")
    parser.add_argument("--out_path", type=str, required=True, help="Output path for saved GMM (.pkl)")
    parser.add_argument("--n_components", type=int, default=32, help="Number of GMM components")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    args = parser.parse_args()
    
    fit_gmm(
        adata_path=args.adata_path,
        z_key=args.z_key,
        out_path=args.out_path,
        n_components=args.n_components,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

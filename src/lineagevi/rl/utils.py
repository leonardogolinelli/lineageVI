"""Utility functions for RL training and evaluation."""

from typing import Optional, Dict, Any
import torch
import numpy as np
import scanpy as sc
import pandas as pd
import json
import os
from pathlib import Path

from .policies import ActorCriticPolicy


def compute_lineage_centroids(
    adata: sc.AnnData,
    lineage_key: str,
    z_key: str = "mean",
) -> tuple[torch.Tensor, list]:
    """
    Compute lineage centroids from AnnData.
    
    Parameters
    ----------
    adata : AnnData
        AnnData with latent states in obsm[z_key].
    lineage_key : str
        Key in adata.obs for lineage labels.
    z_key : str, default 'mean'
        Key in adata.obsm for latent states.
    
    Returns
    -------
    centroids : torch.Tensor
        Centroids of shape (n_lineages, n_latent).
    lineage_names : list
        List of lineage names (ordered).
    """
    if z_key not in adata.obsm:
        raise ValueError(f"Key '{z_key}' not found in adata.obsm. Available keys: {list(adata.obsm.keys())}")
    
    if lineage_key not in adata.obs:
        raise ValueError(f"Key '{lineage_key}' not found in adata.obs. Available keys: {list(adata.obs.keys())}")
    
    z = adata.obsm[z_key]  # (n_cells, n_latent)
    lineage_labels = adata.obs[lineage_key]
    
    # Get unique lineages
    unique_lineages = lineage_labels.unique().tolist()
    unique_lineages = sorted(unique_lineages)  # Sort for consistency
    
    # Compute centroids
    centroids = []
    for lineage in unique_lineages:
        mask = lineage_labels == lineage
        centroid = z[mask].mean(axis=0)
        centroids.append(centroid)
    
    centroids = np.stack(centroids, axis=0)  # (n_lineages, n_latent)
    centroids = torch.from_numpy(centroids).float()
    
    return centroids, unique_lineages


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_policy_checkpoint(
    policy: ActorCriticPolicy,
    centroids: torch.Tensor,
    lineage_names: list,
    config: Dict[str, Any],
    checkpoint_dir: str,
    iteration: Optional[int] = None,
):
    """
    Save policy checkpoint.
    
    Parameters
    ----------
    policy : ActorCriticPolicy
        Trained policy.
    centroids : torch.Tensor
        Lineage centroids.
    lineage_names : list
        List of lineage names.
    config : dict
        Training configuration.
    checkpoint_dir : str
        Directory to save checkpoint.
    iteration : int, optional
        Iteration number (for naming).
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if iteration is not None:
        checkpoint_path = checkpoint_dir / f"policy_iter_{iteration}.pt"
        config_path = checkpoint_dir / f"config_iter_{iteration}.json"
    else:
        checkpoint_path = checkpoint_dir / "policy_final.pt"
        config_path = checkpoint_dir / "config_final.json"
    
    # Save policy state dict
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "centroids": centroids.cpu(),
        "lineage_names": lineage_names,
        "config": config,
    }, checkpoint_path)
    
    # Save config as JSON
    config_serializable = {}
    for k, v in config.items():
        if isinstance(v, (int, float, str, bool, list, dict, type(None))):
            config_serializable[k] = v
        elif isinstance(v, np.ndarray):
            config_serializable[k] = v.tolist()
        elif isinstance(v, torch.Tensor):
            config_serializable[k] = v.cpu().numpy().tolist()
        else:
            config_serializable[k] = str(v)
    
    with open(config_path, "w") as f:
        json.dump(config_serializable, f, indent=2)
    
    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved config to {config_path}")


def load_policy_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[ActorCriticPolicy, torch.Tensor, list, Dict[str, Any]]:
    """
    Load policy checkpoint.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to checkpoint file.
    device : torch.device
        Device to load model on.
    
    Returns
    -------
    policy : ActorCriticPolicy
        Loaded policy.
    centroids : torch.Tensor
        Lineage centroids.
    lineage_names : list
        List of lineage names.
    config : dict
        Training configuration.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    policy_state_dict = checkpoint["policy_state_dict"]
    centroids = checkpoint["centroids"].to(device)
    lineage_names = checkpoint["lineage_names"]
    config = checkpoint["config"]
    
    # Reconstruct policy from config
    obs_dim = config["obs_dim"]
    n_latent = config["n_latent"]
    hidden_sizes = config.get("hidden_sizes", [128, 128])
    delta_max = config.get("delta_max", 1.0)
    
    policy = ActorCriticPolicy(
        obs_dim=obs_dim,
        n_latent=n_latent,
        hidden_sizes=hidden_sizes,
        delta_max=delta_max,
    ).to(device)
    
    policy.load_state_dict(policy_state_dict)
    policy.eval()
    
    return policy, centroids, lineage_names, config

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
    allowed: list[str] | None = None,
    exclude: list[str] | None = None,
    min_cells: int = 1,
) -> tuple[torch.Tensor, list[str]]:
    """
    Compute lineage centroids from AnnData with filtering.
    
    Parameters
    ----------
    adata : AnnData
        AnnData with latent states in obsm[z_key].
    lineage_key : str
        Key in adata.obs for lineage labels.
    z_key : str, default 'mean'
        Key in adata.obsm for latent states.
    allowed : list[str], optional
        If provided, keep only these lineage labels.
    exclude : list[str], optional
        If provided, exclude these lineage labels.
    min_cells : int, default 1
        Minimum number of cells required per lineage.
    
    Returns
    -------
    centroids : torch.Tensor
        Centroids of shape (n_lineages, n_latent).
    goal_labels : list[str]
        List of goal labels (ordered, after filtering).
    """
    if z_key not in adata.obsm:
        raise ValueError(f"Key '{z_key}' not found in adata.obsm. Available keys: {list(adata.obsm.keys())}")
    
    if lineage_key not in adata.obs:
        raise ValueError(f"Key '{lineage_key}' not found in adata.obs. Available keys: {list(adata.obs.keys())}")
    
    z = adata.obsm[z_key]  # (n_cells, n_latent)
    lineage_labels = adata.obs[lineage_key].astype(str)  # Ensure string type for consistent comparisons
    
    # Get unique lineages
    unique_lineages = lineage_labels.unique().tolist()
    unique_lineages = sorted(unique_lineages)  # Sort for consistency
    
    # Apply filters
    filtered_lineages = []
    for lineage in unique_lineages:
        # Check allowed
        if allowed is not None and lineage not in allowed:
            continue
        
        # Check excluded
        if exclude is not None and lineage in exclude:
            continue
        
        # Check min_cells
        mask = lineage_labels == lineage
        n_cells = mask.sum()
        if n_cells < min_cells:
            continue
        
        filtered_lineages.append(lineage)
    
    if len(filtered_lineages) == 0:
        raise ValueError(
            f"No lineages passed filters. "
            f"allowed={allowed}, exclude={exclude}, min_cells={min_cells}"
        )
    
    # Compute centroids for filtered lineages
    centroids = []
    for lineage in filtered_lineages:
        mask = lineage_labels == lineage
        centroid = z[mask].mean(axis=0)
        centroids.append(centroid)
    
    centroids = np.stack(centroids, axis=0)  # (n_lineages, n_latent)
    centroids = torch.from_numpy(centroids).float()
    
    return centroids, filtered_lineages


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
    goal_labels: list,  # Changed from lineage_names for consistency
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
    goal_labels : list
        List of goal labels (ordered, aligned with centroids).
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
        "goal_labels": goal_labels,  # Changed from lineage_names
        "lineage_names": goal_labels,  # Keep for backward compatibility
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
    goal_labels : list
        List of goal labels (ordered, aligned with centroids).
    config : dict
        Training configuration.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    policy_state_dict = checkpoint["policy_state_dict"]
    centroids = checkpoint["centroids"].to(device)
    # Support both goal_labels (new) and lineage_names (old) for backward compatibility
    goal_labels = checkpoint.get("goal_labels", checkpoint.get("lineage_names", []))
    config = checkpoint["config"]
    
    # Reconstruct policy from config
    obs_dim = config["obs_dim"]
    n_latent = config["n_latent"]
    hidden_sizes = config.get("hidden_sizes", [128, 128])
    actor_hidden_sizes = config.get("actor_hidden_sizes", None)
    critic_hidden_sizes = config.get("critic_hidden_sizes", None)
    separate_trunks = config.get("separate_trunks", False)
    activation = config.get("activation", "relu")
    delta_clip = config.get("delta_clip", None)
    kl_stop_threshold = config.get("kl_stop_threshold", 0.02)
    kl_stop_immediate_threshold = config.get("kl_stop_immediate_threshold", 0.03)
    goal_cond_dim = config.get("goal_cond_dim", 32)
    use_t_norm = config.get("use_t_norm", False)
    allow_noop_action = config.get("allow_noop_action", True)
    
    policy = ActorCriticPolicy(
        obs_dim=obs_dim,
        n_latent=n_latent,
        goal_cond_dim=goal_cond_dim,
        use_t_norm=use_t_norm,
        allow_noop_action=allow_noop_action,
        hidden_sizes=hidden_sizes,
        actor_hidden_sizes=actor_hidden_sizes,
        critic_hidden_sizes=critic_hidden_sizes,
        separate_trunks=separate_trunks,
        activation=activation,
        delta_clip=delta_clip,
    ).to(device)
    
    policy.load_state_dict(policy_state_dict)
    policy.eval()
    
    return policy, centroids, goal_labels, config

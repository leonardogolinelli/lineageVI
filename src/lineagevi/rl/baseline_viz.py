"""Visualization script for baseline trajectories (no agent, just velocity flow)."""

import argparse
from pathlib import Path
from typing import Optional, Tuple
import torch
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from ..utils import load_model
from .adapter import VelocityVAEAdapter
from .envs import LatentVelocityEnv
from .utils import compute_lineage_centroids, set_seed


def build_embedding(
    Z: np.ndarray,
    method: str = "pca",
    n_components: int = 2,
    **kwargs,
) -> Tuple[np.ndarray, object]:
    """
    Build 2D embedding from latent states.
    
    Parameters
    ----------
    Z : np.ndarray
        Latent states of shape (n_cells, n_latent).
    method : str
        Embedding method: 'pca' or 'umap'.
    n_components : int
        Number of components (2 for visualization).
    **kwargs
        Additional arguments for embedding method.
    
    Returns
    -------
    embedding : np.ndarray
        2D embedding of shape (n_cells, 2).
    transformer : object
        Fitted transformer (for projecting new points).
    """
    if method == "pca":
        transformer = PCA(n_components=n_components, random_state=42)
        embedding = transformer.fit_transform(Z)
    elif method == "umap":
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not available. Install with: pip install umap-learn")
        transformer = umap.UMAP(n_components=n_components, random_state=42, **kwargs)
        embedding = transformer.fit_transform(Z)
    else:
        raise ValueError(f"Unknown embedding method: {method}")
    
    return embedding, transformer


def rollout_baseline(
    env: LatentVelocityEnv,
    z0: torch.Tensor,
    goal_idx: int,
    z_goal: torch.Tensor,
    T: int,
    x0: Optional[torch.Tensor] = None,
    cluster_idx: Optional[torch.Tensor] = None,
    process_idx: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Roll out baseline trajectory (no perturbations, just velocity flow).
    
    Parameters
    ----------
    env : LatentVelocityEnv
        Environment instance.
    z0 : torch.Tensor
        Initial latent state.
    goal_idx : int
        Goal index (for environment).
    z_goal : torch.Tensor
        Goal latent state (for distance computation).
    T : int
        Rollout horizon.
    x0 : torch.Tensor, optional
        Initial gene expression.
    cluster_idx : torch.Tensor, optional
        Cluster index.
    process_idx : torch.Tensor, optional
        Process index.
    
    Returns
    -------
    z_trajectory : np.ndarray
        Latent states of shape (T+1, n_latent).
    distances : np.ndarray
        Distances to goal of shape (T+1,).
    """
    obs, info = env.reset(z0, goal_idx, x0, cluster_idx=cluster_idx, process_idx=process_idx)
    
    z_trajectory = [z0.cpu().numpy()]
    # Compute distance to actual goal (not just centroid)
    initial_dist = torch.norm(z0 - z_goal, p=2).item()
    distances = [initial_dist]
    
    for t in range(T):
        # No perturbation: action = 0, delta = 0
        obs_next, reward, done, info_next = env.step((0, 0.0))
        
        z_current = env.z  # Keep on device
        z_trajectory.append(z_current.cpu().numpy())
        # Compute distance to actual goal
        dist = torch.norm(z_current - z_goal, p=2).item()
        distances.append(dist)
        
        if done:
            break
        
        obs = obs_next
    
    return np.array(z_trajectory), np.array(distances)


def plot_trajectory_overlay(
    embedding: np.ndarray,
    z_trajectory: np.ndarray,
    z_goal: np.ndarray,
    transformer,
    start_lineage: str,
    goal_lineage: str,
    output_path: Path,
    lineage_key: Optional[str] = None,
    lineage_labels: Optional[np.ndarray] = None,
):
    """
    Plot trajectory overlay on embedding.
    
    Parameters
    ----------
    embedding : np.ndarray
        Global embedding of all cells (n_cells, 2).
    z_trajectory : np.ndarray
        Trajectory latent states (T+1, n_latent).
    z_goal : np.ndarray
        Goal latent state (n_latent,).
    transformer
        Fitted transformer for projecting new points.
    start_lineage : str
        Start lineage label.
    goal_lineage : str
        Goal lineage label.
    output_path : Path
        Output file path.
    lineage_key : str, optional
        Key for lineage labels in adata.obs.
    lineage_labels : np.ndarray, optional
        Lineage labels for coloring.
    """
    # Project trajectory to embedding space
    z_traj_2d = transformer.transform(z_trajectory)
    z_goal_2d = transformer.transform(z_goal.reshape(1, -1))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all cells
    if lineage_labels is not None and lineage_key is not None:
        # Color by lineage
        unique_lineages = np.unique(lineage_labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_lineages)))
        lineage_to_color = {lineage: colors[i] for i, lineage in enumerate(unique_lineages)}
        
        for lineage in unique_lineages:
            mask = lineage_labels == lineage
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[lineage_to_color[lineage]],
                s=10,
                alpha=0.3,
                label=str(lineage),
            )
    else:
        # Gray scatter
        ax.scatter(embedding[:, 0], embedding[:, 1], c='gray', s=10, alpha=0.3, label='Cells')
    
    # Plot trajectory
    ax.plot(
        z_traj_2d[:, 0],
        z_traj_2d[:, 1],
        '-',
        color='blue',
        linewidth=2,
        alpha=0.7,
        label='Baseline trajectory',
    )
    
    # Mark start
    ax.scatter(
        z_traj_2d[0, 0],
        z_traj_2d[0, 1],
        marker='o',
        s=200,
        c='green',
        edgecolors='black',
        linewidths=2,
        label='Start',
        zorder=10,
    )
    
    # Mark goal
    ax.scatter(
        z_goal_2d[0, 0],
        z_goal_2d[0, 1],
        marker='*',
        s=300,
        c='gold',
        edgecolors='black',
        linewidths=2,
        label='Goal',
        zorder=10,
    )
    
    # Mark end
    ax.scatter(
        z_traj_2d[-1, 0],
        z_traj_2d[-1, 1],
        marker='s',
        s=150,
        c='blue',
        edgecolors='black',
        linewidths=2,
        label='End',
        zorder=10,
    )
    
    ax.set_xlabel('Embedding dimension 1', fontsize=12)
    ax.set_ylabel('Embedding dimension 2', fontsize=12)
    ax.set_title(
        f'Baseline Trajectory\n'
        f'Start: {start_lineage} → Goal: {goal_lineage}',
        fontsize=11,
    )
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved trajectory overlay → {output_path}")


def plot_distance_curve(
    distances: np.ndarray,
    output_path: Path,
):
    """
    Plot distance-to-goal curve.
    
    Parameters
    ----------
    distances : np.ndarray
        Distances (T+1,).
    output_path : Path
        Output file path.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(distances, linewidth=2, color='blue', label='Distance to goal')
    ax.set_xlabel('Time step', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    ax.set_title('Distance to Goal Over Time', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved distance curve → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize baseline trajectories (no agent)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained VAE model")
    parser.add_argument("--adata_path", type=str, required=True, help="Path to AnnData file")
    parser.add_argument("--lineage_key", type=str, required=True, help="Key in adata.obs for lineage labels")
    parser.add_argument("--source_lineage", type=str, default=None, help="Source lineage label (default: random)")
    parser.add_argument("--source_mode", type=str, default="sample", choices=["centroid", "sample"],
                        help="Source mode: 'centroid' (use source lineage centroid) or 'sample' (sample a cell from source lineage, default)")
    parser.add_argument("--target_lineage", type=str, required=True, help="Target lineage label")
    parser.add_argument("--target_mode", type=str, default="centroid", choices=["centroid", "sample"],
                        help="Target mode: 'centroid' (use target lineage centroid, default) or 'sample' (sample a cell from target lineage)")
    parser.add_argument("--T", type=int, default=256, help="Rollout horizon (default: 256)")
    parser.add_argument("--T_max", type=int, default=None, help="Maximum episode length (default: same as T)")
    parser.add_argument("--embedding", type=str, default="pca", choices=["pca", "umap"],
                        help="Embedding method: 'pca' or 'umap'")
    parser.add_argument("--z_key", type=str, default="mean", help="Key in adata.obsm for latent states")
    parser.add_argument("--outdir", type=str, default="./baseline_viz_results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cpu, cuda)")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step size")
    parser.add_argument("--use_negative_velocity", action="store_true", help="Use negative velocity instead of normal velocity")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Load AnnData
    print(f"Loading AnnData from {args.adata_path}...")
    adata = sc.read_h5ad(args.adata_path)
    print(f"Loaded {adata.n_obs} cells, {adata.n_vars} genes")
    
    # Load pretrained model
    print(f"Loading pretrained model from {args.model_path}...")
    vae = load_model(adata, args.model_path, map_location=device, training=False)
    print("Model loaded")
    
    # Extract latent states
    if args.z_key not in adata.obsm:
        print(f"Computing latent states (key '{args.z_key}' not found)...")
        vae.get_model_outputs(adata, save_to_adata=True)
        print("Latent states computed")
    
    z_all = torch.from_numpy(adata.obsm[args.z_key]).float().to(device)  # (n_cells, n_latent)
    n_cells = z_all.shape[0]
    n_latent = z_all.shape[1]
    
    # Compute goal centroids
    print(f"Computing goal centroids from lineage key '{args.lineage_key}'...")
    centroids, goal_labels = compute_lineage_centroids(
        adata,
        args.lineage_key,
        z_key=args.z_key,
    )
    centroids = centroids.to(device)
    print(f"Found {len(goal_labels)} goal lineages: {goal_labels}")
    
    # Create mapping from goal label to index
    label_to_goal_idx = {label: idx for idx, label in enumerate(goal_labels)}
    
    target_goal_label = args.target_lineage
    if target_goal_label not in label_to_goal_idx:
        raise ValueError(f"Target lineage '{target_goal_label}' not in goal_labels: {goal_labels}")
    goal_idx = label_to_goal_idx[target_goal_label]
    
    # Create adapter
    velocity_mode = "decode_x"  # Default, could be made configurable
    adapter = VelocityVAEAdapter(vae.model, device, velocity_mode=velocity_mode)
    print(f"Created adapter with velocity_mode='{velocity_mode}'")
    
    # Get cluster/process indices
    cluster_indices = None
    process_indices = None
    if vae.model.cluster_key is not None:
        cluster_labels = adata.obs[vae.model.cluster_key]
        cluster_indices = torch.tensor([
            vae.model.cluster_to_idx.get(str(label), 0) for label in cluster_labels
        ], dtype=torch.long, device=device)
    
    cls_encoding_key = vae.model.cls_encoding_key
    process_labels = adata.obs[cls_encoding_key]
    process_indices = torch.tensor([
        vae.model.process_to_idx.get(str(label), 0) for label in process_labels
    ], dtype=torch.long, device=device)
    
    # Select start cell based on source_lineage and source_mode
    rng = np.random.RandomState(args.seed)
    if args.source_lineage is not None:
        if args.source_mode == "centroid":
            # Use source lineage centroid
            source_mask = adata.obs[args.lineage_key] == args.source_lineage
            z0 = z_all[source_mask].mean(dim=0)  # (n_latent,)
            start_lineage = args.source_lineage
            start_cell_idx = None
            start_cell_idx_for_cluster = None
            print(f"Using source centroid for '{args.source_lineage}'")
        else:  # source_mode == "sample"
            # Sample random cell from specified lineage
            start_mask = adata.obs[args.lineage_key] == args.source_lineage
            start_indices = np.where(start_mask)[0]
            if len(start_indices) == 0:
                raise ValueError(f"No cells found with lineage '{args.source_lineage}'")
            start_cell_idx = rng.choice(start_indices)
            z0 = z_all[start_cell_idx]
            start_lineage = args.source_lineage
            start_cell_idx_for_cluster = start_cell_idx
            print(f"Randomly selected start cell from lineage '{args.source_lineage}': index {start_cell_idx}")
    else:
        # Random cell from any lineage
        start_cell_idx = rng.randint(0, n_cells)
        z0 = z_all[start_cell_idx]
        start_lineage = adata.obs[args.lineage_key].iloc[start_cell_idx]
        start_cell_idx_for_cluster = start_cell_idx
        print(f"Randomly selected start cell: index {start_cell_idx}, lineage '{start_lineage}'")
    
    # Get goal
    target_goal_label = args.target_lineage
    if args.target_mode == "centroid":
        z_goal = centroids[goal_idx].to(device)  # (n_latent,)
        print(f"Target: centroid for '{target_goal_label}'")
    else:  # target_mode == "sample"
        # Sample one cell from target lineage (seed ensures reproducibility)
        target_mask = adata.obs[args.lineage_key] == target_goal_label
        target_indices = np.where(target_mask)[0]
        if len(target_indices) == 0:
            raise ValueError(f"No cells found with lineage '{target_goal_label}'")
        goal_cell_idx = rng.choice(target_indices)
        z_goal = z_all[goal_cell_idx].to(device)  # (n_latent,)
        print(f"Goal: cell {goal_cell_idx} from lineage '{target_goal_label}' (sampled with seed={args.seed})")
    
    # Build embedding
    print(f"Building {args.embedding.upper()} embedding...")
    Z_all = z_all.cpu().numpy()
    embedding, transformer = build_embedding(Z_all, method=args.embedding)
    print(f"Embedding shape: {embedding.shape}")
    
    # Set T_max
    T_max = args.T_max if args.T_max is not None else args.T
    T_max = max(T_max, args.T)  # Ensure T_max is at least as large as T
    
    # Create environment
    env = LatentVelocityEnv(
        adapter=adapter,
        centroids=centroids,
        goal_names=goal_labels,
        dt=args.dt,
        T_max=T_max,
        eps_success=0.1,  # Not used for baseline, but required
        lambda_progress=1.0,
        lambda_act=0.01,
        lambda_mag=0.1,
        R_succ=10.0,
        use_negative_velocity=args.use_negative_velocity,
    )
    
    # Get initial x if needed
    x0 = None
    if velocity_mode == "fixed_x":
        unspliced_key = "unspliced" if "unspliced" in adata.layers else "Mu"
        spliced_key = "spliced" if "spliced" in adata.layers else "Ms"
        u = torch.from_numpy(np.asarray(adata.layers[unspliced_key][start_cell_idx])).float().to(device)
        s = torch.from_numpy(np.asarray(adata.layers[spliced_key][start_cell_idx])).float().to(device)
        x0 = torch.cat([u, s], dim=0)  # (2*n_genes,)
    
    # Get cluster/process indices for start cell
    cluster_idx = cluster_indices[start_cell_idx] if cluster_indices is not None else None
    process_idx = process_indices[start_cell_idx] if process_indices is not None else None
    
    # Roll out baseline trajectory
    print(f"Rolling out baseline trajectory for T={args.T} steps (T_max={T_max})...")
    z_trajectory, distances = rollout_baseline(
        env, z0, goal_idx, z_goal, args.T, x0, cluster_idx, process_idx
    )
    
    print(f"Trajectory length: {len(z_trajectory)} steps")
    print(f"Initial distance: {distances[0]:.4f}")
    print(f"Final distance: {distances[-1]:.4f}")
    
    # Get lineage labels for coloring
    lineage_labels = None
    if args.lineage_key in adata.obs:
        lineage_labels = adata.obs[args.lineage_key].values
    
    # Create visualizations
    # Plot A: Trajectory overlay
    plot_trajectory_overlay(
        embedding,
        z_trajectory,
        z_goal.cpu().numpy(),
        transformer,
        str(start_lineage),
        target_goal_label,
        outdir / "trajectory_overlay.png",
        lineage_key=args.lineage_key,
        lineage_labels=lineage_labels,
    )
    
    # Plot B: Distance curve
    plot_distance_curve(
        distances,
        outdir / "distance_curve.png",
    )
    
    # Save raw arrays
    np.savez(
        outdir / "trajectory.npz",
        z_trajectory=z_trajectory,
        z_goal=z_goal.cpu().numpy(),
        distances=distances,
        embedding=embedding,
        start_cell_idx=start_cell_idx,
        goal_idx=goal_idx,
        start_lineage=str(start_lineage),
        goal_lineage=target_goal_label,
    )
    print(f"Saved raw arrays → {outdir / 'trajectory.npz'}")
    
    print(f"\nVisualization complete! Outputs saved to {outdir}")


if __name__ == "__main__":
    main()

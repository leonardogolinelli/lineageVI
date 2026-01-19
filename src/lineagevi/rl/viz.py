"""Visualization script for RL agent trajectories."""

import argparse
from pathlib import Path
from typing import Optional, Tuple, Literal, List
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
from .utils import load_policy_checkpoint, set_seed


def build_embedding(
    Z: np.ndarray,
    method: Literal["pca", "umap"] = "pca",
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


def rollout_agent(
    env: LatentVelocityEnv,
    policy,
    z0: torch.Tensor,
    goal_idx: int,
    z_goal: torch.Tensor,
    T: int,
    x0: Optional[torch.Tensor] = None,
    cluster_idx: Optional[torch.Tensor] = None,
    process_idx: Optional[torch.Tensor] = None,
    deterministic: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Roll out agent trajectory using policy.
    
    Parameters
    ----------
    env : LatentVelocityEnv
        Environment instance.
    policy
        Trained policy.
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
    deterministic : bool
        If True, use argmax for discrete action and mean for magnitude.
    
    Returns
    -------
    z_trajectory : np.ndarray
        Latent states of shape (T+1, n_latent).
    distances : np.ndarray
        Distances to goal of shape (T+1,).
    actions : np.ndarray
        Actions of shape (T,).
    deltas : np.ndarray
        Perturbation magnitudes of shape (T,).
    """
    obs, info = env.reset(z0, goal_idx, x0, cluster_idx=cluster_idx, process_idx=process_idx)
    
    z_trajectory = [z0.cpu().numpy()]
    # Compute distance to actual goal (not just centroid)
    initial_dist = torch.norm(z0 - z_goal, p=2).item()
    distances = [initial_dist]
    actions = []
    deltas = []
    
    device = next(policy.parameters()).device
    
    for t in range(T):
        # Get action from policy
        obs_tensor = torch.from_numpy(obs) if isinstance(obs, np.ndarray) else obs
        obs_tensor = obs_tensor.to(device).float()
        
        with torch.no_grad():
            if deterministic:
                # Deterministic: argmax for action, mean for magnitude
                action_logits, magnitude_mu, _, _ = policy.forward(obs_tensor.unsqueeze(0))
                action = torch.argmax(action_logits, dim=1).item()
                
                # Get magnitude for chosen action
                if action > 0:
                    action_idx = action - 1
                    delta = magnitude_mu[0, action_idx].item()
                    delta = policy.delta_max * np.tanh(delta)  # Apply tanh squashing
                else:
                    delta = 0.0
            else:
                action, delta_tensor, _, _ = policy.sample(obs_tensor.unsqueeze(0), deterministic=True)
                action = action.item()
                delta = delta_tensor.item()
        
        # Step environment
        obs_next, reward, done, info_next = env.step((action, delta))
        
        z_current = env.z  # Keep on device
        z_trajectory.append(z_current.cpu().numpy())
        # Compute distance to actual goal
        dist = torch.norm(z_current - z_goal, p=2).item()
        distances.append(dist)
        actions.append(action)
        deltas.append(delta)
        
        if done:
            break
        
        obs = obs_next
    
    return np.array(z_trajectory), np.array(distances), np.array(actions), np.array(deltas)


def plot_trajectory_overlay(
    embedding: np.ndarray,
    z_baseline: np.ndarray,
    z_agent: np.ndarray,
    z_goal: np.ndarray,
    transformer,
    start_lineage: str,
    goal_lineage: str,
    success: bool,
    n_interventions: int,
    total_magnitude: float,
    output_path: Path,
    lineage_key: Optional[str] = None,
    lineage_labels: Optional[np.ndarray] = None,
):
    """
    Plot trajectory overlay on embedding.
    
    Parameters
    ----------
    embedding : np.ndarray
        2D embedding of all cells (n_cells, 2).
    z_baseline : np.ndarray
        Baseline trajectory latent states (T+1, n_latent).
    z_agent : np.ndarray
        Agent trajectory latent states (T+1, n_latent).
    z_goal : np.ndarray
        Goal latent state (n_latent,).
    transformer
        Fitted transformer for projecting new points.
    start_lineage : str
        Starting lineage label.
    goal_lineage : str
        Goal lineage label.
    success : bool
        Whether agent reached goal.
    n_interventions : int
        Number of interventions (L0).
    total_magnitude : float
        Total magnitude (L1).
    output_path : Path
        Output file path.
    lineage_key : str, optional
        Key for lineage labels in adata.
    lineage_labels : np.ndarray, optional
        Lineage labels for coloring cells.
    """
    # Project trajectories to embedding space
    z_baseline_2d = transformer.transform(z_baseline)
    z_agent_2d = transformer.transform(z_agent)
    z_goal_2d = transformer.transform(z_goal.reshape(1, -1))[0]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot all cells
    if lineage_labels is not None:
        # Color by lineage
        unique_lineages = np.unique(lineage_labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_lineages)))
        lineage_to_color = {label: colors[i] for i, label in enumerate(unique_lineages)}
        
        for label in unique_lineages:
            mask = lineage_labels == label
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[lineage_to_color[label]],
                s=10,
                alpha=0.3,
                label=label if len(unique_lineages) <= 20 else None,
            )
    else:
        # Gray scatter
        ax.scatter(embedding[:, 0], embedding[:, 1], c='gray', s=10, alpha=0.3, label='Cells')
    
    # Plot baseline trajectory
    ax.plot(
        z_baseline_2d[:, 0],
        z_baseline_2d[:, 1],
        '--',
        color='blue',
        linewidth=2,
        alpha=0.7,
        label='Baseline (no perturbations)',
    )
    
    # Plot agent trajectory
    ax.plot(
        z_agent_2d[:, 0],
        z_agent_2d[:, 1],
        '-',
        color='red',
        linewidth=2,
        alpha=0.7,
        label='Agent trajectory',
    )
    
    # Mark start
    ax.scatter(
        z_baseline_2d[0, 0],
        z_baseline_2d[0, 1],
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
        z_goal_2d[0],
        z_goal_2d[1],
        marker='*',
        s=300,
        c='gold',
        edgecolors='black',
        linewidths=2,
        label='Goal',
        zorder=10,
    )
    
    # Mark end of agent trajectory
    ax.scatter(
        z_agent_2d[-1, 0],
        z_agent_2d[-1, 1],
        marker='s',
        s=150,
        c='red',
        edgecolors='black',
        linewidths=2,
        label='Agent end',
        zorder=10,
    )
    
    ax.set_xlabel('Embedding dimension 1', fontsize=12)
    ax.set_ylabel('Embedding dimension 2', fontsize=12)
    ax.set_title(
        f'Trajectory Overlay\n'
        f'Start: {start_lineage} → Goal: {goal_lineage} | '
        f'Success: {success} | L0: {n_interventions} | L1: {total_magnitude:.3f}',
        fontsize=11,
    )
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved trajectory overlay → {output_path}")


def plot_distance_curves(
    distances_baseline: np.ndarray,
    distances_agent: np.ndarray,
    output_path: Path,
):
    """
    Plot distance-to-goal curves.
    
    Parameters
    ----------
    distances_baseline : np.ndarray
        Baseline distances (T+1,).
    distances_agent : np.ndarray
        Agent distances (T+1,).
    output_path : Path
        Output file path.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    t_baseline = np.arange(len(distances_baseline))
    t_agent = np.arange(len(distances_agent))
    
    ax.plot(t_baseline, distances_baseline, '--', color='blue', linewidth=2, label='Baseline', alpha=0.7)
    ax.plot(t_agent, distances_agent, '-', color='red', linewidth=2, label='Agent', alpha=0.7)
    
    ax.set_xlabel('Time step', fontsize=12)
    ax.set_ylabel('Distance to goal', fontsize=12)
    ax.set_title('Distance to Goal Over Time', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved distance curves → {output_path}")


def plot_interventions(
    actions: np.ndarray,
    deltas: np.ndarray,
    n_latent: int,
    output_path: Path,
    method: Literal["stem", "heatmap"] = "heatmap",
):
    """
    Plot intervention schedule.
    
    Parameters
    ----------
    actions : np.ndarray
        Actions of shape (T,).
    deltas : np.ndarray
        Perturbation magnitudes of shape (T,).
    n_latent : int
        Number of latent dimensions.
    output_path : Path
        Output file path.
    method : str
        Plotting method: 'stem' or 'heatmap'.
    """
    T = len(actions)
    
    if method == "heatmap":
        # Create heatmap: H[dim, t] = delta if action==dim else 0
        H = np.zeros((n_latent, T))
        for t in range(T):
            if actions[t] > 0:
                dim = actions[t] - 1  # action 1 -> dim 0
                H[dim, t] = deltas[t]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(H, aspect='auto', cmap='Reds', interpolation='nearest')
        ax.set_xlabel('Time step', fontsize=12)
        ax.set_ylabel('Latent dimension', fontsize=12)
        ax.set_title('Intervention Schedule (Heatmap)', fontsize=14)
        plt.colorbar(im, ax=ax, label='Perturbation magnitude')
        
    else:  # stem plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Color-code by dimension
        colors = plt.cm.tab20(np.linspace(0, 1, n_latent))
        
        for t in range(T):
            if actions[t] > 0:
                dim = actions[t] - 1
                ax.stem([t], [deltas[t]], linefmt=f'C{dim}-', markerfmt=f'C{dim}o', basefmt=' ')
        
        ax.set_xlabel('Time step', fontsize=12)
        ax.set_ylabel('Perturbation magnitude', fontsize=12)
        ax.set_title('Intervention Schedule (Stem Plot)', fontsize=14)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved intervention schedule → {output_path}")


def plot_intervention_summary(
    actions: np.ndarray,
    deltas: np.ndarray,
    n_latent: int,
    output_path: Path,
    gp_names: Optional[List[str]] = None,
):
    """
    Plot summary of interventions by gene program name.
    
    Parameters
    ----------
    actions : np.ndarray
        Actions of shape (T,).
    deltas : np.ndarray
        Perturbation magnitudes of shape (T,).
    n_latent : int
        Number of latent dimensions.
    output_path : Path
        Output file path.
    gp_names : list of str, optional
        Gene program names. If None, uses GP_0, GP_1, etc.
    """
    # Count interventions per dimension
    intervention_counts = np.zeros(n_latent)
    intervention_magnitudes = np.zeros(n_latent)
    
    for t in range(len(actions)):
        if actions[t] > 0:
            dim = actions[t] - 1  # action 1 -> dim 0
            intervention_counts[dim] += 1
            intervention_magnitudes[dim] += abs(deltas[t])
    
    # Get indices of programs that were actually targeted
    targeted_dims = np.where(intervention_counts > 0)[0]
    
    if len(targeted_dims) == 0:
        # No interventions occurred
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No interventions occurred', 
                ha='center', va='center', fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved intervention summary → {output_path}")
        return
    
    # Create labels
    if gp_names is not None and len(gp_names) == n_latent:
        labels = [gp_names[dim] for dim in targeted_dims]
    else:
        labels = [f"GP_{dim}" for dim in targeted_dims]
    
    # Sort by frequency (descending)
    sort_idx = np.argsort(intervention_counts[targeted_dims])[::-1]
    targeted_dims_sorted = targeted_dims[sort_idx]
    labels_sorted = [labels[i] for i in sort_idx]
    counts_sorted = intervention_counts[targeted_dims_sorted]
    magnitudes_sorted = intervention_magnitudes[targeted_dims_sorted]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(6, len(targeted_dims) * 0.4)))
    
    # Plot 1: Intervention frequency
    y_pos = np.arange(len(targeted_dims_sorted))
    ax1.barh(y_pos, counts_sorted, color='steelblue', alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels_sorted, fontsize=10)
    ax1.set_xlabel('Number of Interventions', fontsize=12)
    ax1.set_title('Intervention Frequency by Gene Program', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, count in enumerate(counts_sorted):
        ax1.text(count + 0.1, i, f'{int(count)}', va='center', fontsize=9)
    
    # Plot 2: Total magnitude
    ax2.barh(y_pos, magnitudes_sorted, color='coral', alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels_sorted, fontsize=10)
    ax2.set_xlabel('Total Intervention Magnitude', fontsize=12)
    ax2.set_title('Total Intervention Magnitude by Gene Program', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, mag in enumerate(magnitudes_sorted):
        ax2.text(mag + 0.01, i, f'{mag:.2f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved intervention summary → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize RL agent trajectories")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to policy checkpoint")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained VAE model")
    parser.add_argument("--adata_path", type=str, required=True, help="Path to AnnData file")
    parser.add_argument("--lineage_key", type=str, required=True, help="Key in adata.obs for lineage labels")
    parser.add_argument("--start_cell_idx", type=int, default=None, help="Start cell index (mutually exclusive with --start_lineage)")
    parser.add_argument("--start_lineage", type=str, default=None, help="Start lineage label (mutually exclusive with --start_cell_idx, default: random)")
    parser.add_argument("--target_goal", type=str, required=True, help="Target goal label")
    parser.add_argument("--goal_mode", type=str, default="centroid", choices=["centroid", "goal_cell"],
                        help="Goal mode: 'centroid' (use lineage centroid, default) or 'goal_cell' (sample a cell from target lineage)")
    parser.add_argument("--T", type=int, default=64, help="Rollout horizon")
    parser.add_argument("--embedding", type=str, default="pca", choices=["pca", "umap"],
                        help="Embedding method: 'pca' or 'umap'")
    parser.add_argument("--z_key", type=str, default="mean", help="Key in adata.obsm for latent states")
    parser.add_argument("--outdir", type=str, default="./viz_results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cpu, cuda)")
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use deterministic policy (default: True)")
    parser.add_argument("--intervention_method", type=str, default="heatmap", choices=["stem", "heatmap"],
                        help="Intervention visualization method")
    
    args = parser.parse_args()
    
    # Validate mutually exclusive arguments
    if args.start_cell_idx is not None and args.start_lineage is not None:
        raise ValueError("--start_cell_idx and --start_lineage are mutually exclusive")
    
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
    
    # Load checkpoint
    print(f"Loading policy from {args.checkpoint}...")
    policy, centroids, goal_labels, config = load_policy_checkpoint(args.checkpoint, device)
    print(f"Loaded policy for {len(goal_labels)} goal lineages: {goal_labels}")
    
    # Create mapping from goal label to index
    label_to_goal_idx = {label: idx for idx, label in enumerate(goal_labels)}
    
    if args.target_goal not in label_to_goal_idx:
        raise ValueError(f"Target goal '{args.target_goal}' not in goal_labels: {goal_labels}")
    goal_idx = label_to_goal_idx[args.target_goal]
    
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
    
    # Get environment config
    env_config = config.get("env", {})
    velocity_mode = env_config.get("velocity_mode", "decode_x")
    use_negative_velocity = env_config.get("use_negative_velocity", False)
    
    # Create adapter
    adapter = VelocityVAEAdapter(vae.model, device, velocity_mode=velocity_mode)
    
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
    
    # Select start cell
    if args.start_cell_idx is not None:
        start_cell_idx = args.start_cell_idx
        if start_cell_idx >= n_cells:
            raise ValueError(f"start_cell_idx {start_cell_idx} >= n_cells {n_cells}")
        start_lineage = adata.obs[args.lineage_key].iloc[start_cell_idx]
        print(f"Using specified start cell: index {start_cell_idx}, lineage '{start_lineage}'")
    elif args.start_lineage is not None:
        # Sample random cell from specified lineage
        start_mask = adata.obs[args.lineage_key] == args.start_lineage
        start_indices = np.where(start_mask)[0]
        if len(start_indices) == 0:
            raise ValueError(f"No cells found with lineage '{args.start_lineage}'")
        start_cell_idx = np.random.choice(start_indices)
        start_lineage = args.start_lineage
        print(f"Randomly selected start cell from lineage '{args.start_lineage}': index {start_cell_idx}")
    else:
        # Random cell from any lineage
        start_cell_idx = np.random.randint(0, n_cells)
        start_lineage = adata.obs[args.lineage_key].iloc[start_cell_idx]
        print(f"Randomly selected start cell: index {start_cell_idx}, lineage '{start_lineage}'")
    
    z0 = z_all[start_cell_idx]  # (n_latent,)
    
    # Get goal
    if args.goal_mode == "centroid":
        z_goal = centroids[goal_idx].to(device)  # (n_latent,)
        print(f"Goal: centroid for '{args.target_goal}'")
    else:  # goal_cell (default)
        # Sample one cell from target lineage (seed ensures reproducibility)
        target_mask = adata.obs[args.lineage_key] == args.target_goal
        target_indices = np.where(target_mask)[0]
        if len(target_indices) == 0:
            raise ValueError(f"No cells found with lineage '{args.target_goal}'")
        # Use seed for reproducible sampling
        rng = np.random.RandomState(args.seed)
        goal_cell_idx = rng.choice(target_indices)
        z_goal = z_all[goal_cell_idx].to(device)  # (n_latent,)
        print(f"Goal: cell {goal_cell_idx} from lineage '{args.target_goal}' (sampled with seed={args.seed})")
    
    # Build embedding
    print(f"Building {args.embedding.upper()} embedding...")
    Z_all = z_all.cpu().numpy()
    embedding, transformer = build_embedding(Z_all, method=args.embedding)
    print(f"Embedding shape: {embedding.shape}")
    
    # Get T_max from config (should match training)
    T_max_viz = env_config.get("T_max", 100)
    # Ensure T_max is at least as large as requested rollout horizon
    T_max_viz = max(T_max_viz, args.T)
    
    # Create environment
    env = LatentVelocityEnv(
        adapter=adapter,
        centroids=centroids,
        goal_names=goal_labels,
        dt=env_config.get("dt", 0.1),
        T_max=T_max_viz,
        eps_success=env_config.get("eps_success", 0.1),
        lambda_progress=env_config.get("lambda_progress", 1.0),
        lambda_act=env_config.get("lambda_act", 0.01),
        lambda_mag=env_config.get("lambda_mag", 0.1),
        R_succ=env_config.get("R_succ", 10.0),
        use_negative_velocity=use_negative_velocity,
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
    print("Rolling out baseline trajectory...")
    z_baseline, distances_baseline = rollout_baseline(
        env, z0, goal_idx, z_goal, args.T, x0, cluster_idx, process_idx
    )
    
    # Roll out agent trajectory
    print("Rolling out agent trajectory...")
    z_agent, distances_agent, actions, deltas = rollout_agent(
        env, policy, z0, goal_idx, z_goal, args.T, x0, cluster_idx, process_idx,
        deterministic=args.deterministic
    )
    
    # Compute metrics
    n_interventions = np.sum(actions != 0)
    total_magnitude = np.sum(np.abs(deltas))
    final_distance = distances_agent[-1]
    initial_distance = distances_agent[0]
    # Use environment's success threshold for consistency
    success = final_distance < env.eps_success
    
    print(f"\nTrajectory metrics:")
    print(f"  Initial distance: {initial_distance:.4f}")
    print(f"  Final distance: {final_distance:.4f}")
    print(f"  Success: {success}")
    print(f"  Interventions (L0): {n_interventions}")
    print(f"  Total magnitude (L1): {total_magnitude:.4f}")
    
    # Get lineage labels for coloring
    lineage_labels = None
    if args.lineage_key in adata.obs:
        lineage_labels = adata.obs[args.lineage_key].values
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Plot A: Trajectory overlay
    plot_trajectory_overlay(
        embedding,
        z_baseline,
        z_agent,
        z_goal.cpu().numpy(),
        transformer,
        str(start_lineage),
        args.target_goal,
        success,
        n_interventions,
        total_magnitude,
        outdir / "trajectory_overlay.png",
        lineage_key=args.lineage_key,
        lineage_labels=lineage_labels,
    )
    
    # Plot B: Distance curves
    plot_distance_curves(
        distances_baseline,
        distances_agent,
        outdir / "distance_curves.png",
    )
    
    # Plot C: Intervention schedule
    plot_interventions(
        actions,
        deltas,
        adapter.n_latent,
        outdir / "interventions.png",
        method=args.intervention_method,
    )
    
    # Plot D: Intervention summary with gene program names
    gp_names = None
    if "terms" in adata.uns and len(adata.uns["terms"]) == adapter.n_latent:
        gp_names = adata.uns["terms"]
        print(f"Found gene program names in adata.uns['terms']")
    else:
        print(f"Gene program names not found in adata.uns['terms'], using GP_0, GP_1, etc.")
    
    plot_intervention_summary(
        actions,
        deltas,
        adapter.n_latent,
        outdir / "intervention_summary.png",
        gp_names=gp_names,
    )
    
    # Save raw arrays
    np.savez(
        outdir / "trajectory.npz",
        z_baseline=z_baseline,
        z_agent=z_agent,
        z_goal=z_goal.cpu().numpy(),
        distances_baseline=distances_baseline,
        distances_agent=distances_agent,
        actions=actions,
        deltas=deltas,
        embedding=embedding,
        start_cell_idx=start_cell_idx,
        goal_idx=goal_idx,
        start_lineage=str(start_lineage),
        goal_lineage=args.target_goal,
    )
    print(f"Saved raw arrays → {outdir / 'trajectory.npz'}")
    
    print(f"\nVisualization complete! Outputs saved to {outdir}")


if __name__ == "__main__":
    main()

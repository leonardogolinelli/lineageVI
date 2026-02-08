"""Visualization script for RL agent trajectories."""

import argparse
from pathlib import Path
from typing import Optional, Tuple, Literal, List
import torch
import torch.distributions as dist
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

    Returns
    -------
    z_trajectory : np.ndarray
        Latent states of shape (T+1, n_latent).
    distances : np.ndarray
        Distances to goal of shape (T+1,).
    """
    obs, info = env.reset(z0, goal_idx, x0, cluster_idx=cluster_idx, goal_state=z_goal)

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
        distance = torch.norm(z_current - z_goal, p=2).item()
        distances.append(distance)
        
        if done:
            break
        
        obs = obs_next
    
    return np.array(z_trajectory), np.array(distances)


def rollout_reachability_baseline(
    env: LatentVelocityEnv,
    z0: torch.Tensor,
    z_goal: torch.Tensor,
    T: int,
    delta_max: float,
    x0: Optional[torch.Tensor] = None,
    cluster_idx: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Greedy coordinate-descent baseline for reachability.

    Picks the coordinate with largest absolute error and applies a clipped step.
    """
    obs, info = env.reset(z0, goal_idx=0, x0=x0, cluster_idx=cluster_idx, goal_state=z_goal)
    
    z_trajectory = [z0.cpu().numpy()]
    initial_dist = torch.norm(z0 - z_goal, p=2).item()
    distances = [initial_dist]
    actions = []
    deltas = []
    
    for t in range(T):
        z_current = env.z
        error = z_goal - z_current
        
        idx = torch.argmax(error.abs()).item()
        raw_delta = error[idx].item()
        delta = float(np.clip(raw_delta, -delta_max, delta_max))
        
        if delta == 0.0:
            action = 0
        else:
            action = idx + 1  # 0 is no-op, 1..n_latent map to dims
        
        obs_next, reward, done, info_next = env.step((action, delta))
        
        z_current = env.z
        z_trajectory.append(z_current.cpu().numpy())
        distance = torch.norm(z_current - z_goal, p=2).item()
        distances.append(distance)
        actions.append(action)
        deltas.append(delta)
        
        if done:
            break
        
        obs = obs_next
    
    return np.array(z_trajectory), np.array(distances), np.array(actions), np.array(deltas)


def inspect_policy(
    policy,
    obs: torch.Tensor,
    n_samples: int = 100,
    actions_per_step: int = 1,
) -> dict:
    """
    Inspect policy behavior: action probabilities, magnitude distributions, entropy.
    
    Parameters
    ----------
    policy
        Trained policy.
    obs : torch.Tensor
        Observation of shape (obs_dim,).
    n_samples : int
        Number of samples to draw for statistics.
    
    Returns
    -------
    info : dict
        Dictionary with policy inspection results.
    """
    device = next(policy.parameters()).device
    obs_batch = obs.unsqueeze(0).to(device).float()  # (1, obs_dim)
    
    with torch.no_grad():
        # Get policy outputs
        action_logits, _, _, value = policy.forward(obs_batch)
        
        # Convert logits to probabilities
        action_probs = torch.softmax(action_logits, dim=1).squeeze(0)  # (n_latent + 1,)
        
        # Get magnitude params for each possible action (action-conditioned)
        # For inspection, we'll compute magnitude params for each action
        magnitude_mu_list = []
        magnitude_logstd_list = []
        for a in range(policy.n_latent + 1):
            action_tensor = torch.full((1,), a, dtype=torch.long, device=obs_batch.device)
            mu_a, std_a = policy._get_magnitude_params(obs_batch, action_tensor)
            # Get log_std from std (reverse the exp)
            log_std_a = torch.log(std_a + 1e-8)  # Add small epsilon for numerical stability
            magnitude_mu_list.append(mu_a.item())
            magnitude_logstd_list.append(log_std_a.item())
        
        # Convert to tensors (for actions 1..n_latent, action 0 has no magnitude)
        magnitude_mu = torch.tensor(magnitude_mu_list[1:], device=obs_batch.device)  # (n_latent,)
        magnitude_logstd = torch.tensor(magnitude_logstd_list[1:], device=obs_batch.device)  # (n_latent,)
        
        # Get clamped magnitude std
        magnitude_logstd_clamped = torch.clamp(magnitude_logstd, policy.LOG_STD_MIN, policy.LOG_STD_MAX)
        magnitude_std = torch.exp(magnitude_logstd_clamped)  # (n_latent,)
        
        # Compute entropy of action distribution
        action_dist = torch.distributions.Categorical(logits=action_logits)
        action_entropy = action_dist.entropy().item()
        
        # Sample multiple times to see variation
        actions_sampled = []
        deltas_sampled = []
        for _ in range(n_samples):
            action, delta, _, _ = policy.sample(obs_batch, deterministic=False, n_actions=actions_per_step)
            if action.numel() == 1:
                actions_sampled.append(action.item())
                deltas_sampled.append(delta.item())
            else:
                actions_sampled.extend(action.flatten().cpu().numpy().tolist())
                deltas_sampled.extend(delta.flatten().cpu().numpy().tolist())
        
        actions_sampled = np.array(actions_sampled)
        deltas_sampled = np.array(deltas_sampled)
        
        # Statistics
        action_counts = np.bincount(actions_sampled, minlength=policy.n_latent + 1)
        action_freq = action_counts / n_samples
        
        info = {
            "action_logits": action_logits.squeeze(0).cpu().numpy(),
            "action_probs": action_probs.cpu().numpy(),
            "action_entropy": action_entropy,
            "magnitude_mu": magnitude_mu.cpu().numpy(),
            "magnitude_logstd": magnitude_logstd.squeeze(0).cpu().numpy(),
            "magnitude_logstd_clamped": magnitude_logstd_clamped.squeeze(0).cpu().numpy(),
            "magnitude_std": magnitude_std.cpu().numpy(),
            "action_freq": action_freq,
            "action_samples": actions_sampled,
            "delta_samples": deltas_sampled,
            "delta_mean": np.mean(deltas_sampled),
            "delta_std": np.std(deltas_sampled),
        }
        
        return info


def rollout_agent(
    env: LatentVelocityEnv,
    policy,
    z0: torch.Tensor,
    goal_idx: int,
    z_goal: torch.Tensor,
    T: int,
    x0: Optional[torch.Tensor] = None,
    cluster_idx: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    deterministic_action: bool = False,
    actions_per_step: int = 1,
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
    deterministic : bool
        If True, use argmax for discrete action and mean for magnitude.
    deterministic_action : bool
        If True, use argmax for action but sample magnitude.
    
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
    obs, info = env.reset(z0, goal_idx, x0, cluster_idx=cluster_idx, goal_state=z_goal)

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
            if actions_per_step == 1 and deterministic_action:
                # Argmax action, sampled magnitude
                action_logits, _, _, _ = policy.forward(obs_tensor.unsqueeze(0))
                action = torch.argmax(action_logits, dim=1).item()
                
                if action > 0:
                    action_tensor = torch.tensor([action], device=obs_tensor.device)
                    magnitude_mu, magnitude_std = policy._get_magnitude_params(obs_tensor.unsqueeze(0), action_tensor)
                    delta = dist.Normal(magnitude_mu, magnitude_std).sample()[0].item()
                else:
                    delta = 0.0
                
                # Clip magnitude if configured
                delta_clip = getattr(policy, "delta_clip", None)
                if delta_clip is not None:
                    delta = float(np.clip(delta, -delta_clip, delta_clip))
            elif actions_per_step == 1 and deterministic:
                # Deterministic: argmax for action, mean for magnitude
                action_logits, magnitude_mu, _, _ = policy.forward(obs_tensor.unsqueeze(0))
                action = torch.argmax(action_logits, dim=1).item()
                
                # Get magnitude for chosen action (conditioned on state + action)
                if action > 0:
                    action_tensor = torch.tensor([action], device=obs_tensor.device)
                    magnitude_mu, _ = policy._get_magnitude_params(obs_tensor.unsqueeze(0), action_tensor)
                    delta = magnitude_mu[0].item()  # Use mean directly (deterministic)
                else:
                    delta = 0.0
                
                # Clip magnitude if configured
                delta_clip = getattr(policy, "delta_clip", None)
                if delta_clip is not None:
                    delta = float(np.clip(delta, -delta_clip, delta_clip))
            else:
                # For k>1 or stochastic sampling, use policy sampler
                action, delta_tensor, _, _ = policy.sample(
                    obs_tensor.unsqueeze(0),
                    deterministic=deterministic or deterministic_action,
                    n_actions=actions_per_step,
                )
                delta = delta_tensor.squeeze(0)
        
        # Step environment
        obs_next, reward, done, info_next = env.step((action, delta))
        
        z_current = env.z  # Keep on device
        z_trajectory.append(z_current.cpu().numpy())
        # Compute distance to actual goal
        distance = torch.norm(z_current - z_goal, p=2).item()
        distances.append(distance)
        if torch.is_tensor(action):
            actions.append(action.cpu().numpy())
        else:
            actions.append(action)
        if torch.is_tensor(delta):
            deltas.append(delta.cpu().numpy())
        else:
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
    eps_success: Optional[float] = None,
    centroids: Optional[np.ndarray] = None,
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
    eps_success : float, optional
        Success radius in latent space (projected to embedding).
    centroids : np.ndarray, optional
        Centroid latent states (n_goals, n_latent) to plot with indices.
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
    
    # Plot all centroids with index labels if provided
    if centroids is not None and len(centroids) > 0:
        centroids_2d = transformer.transform(centroids)
        ax.scatter(
            centroids_2d[:, 0],
            centroids_2d[:, 1],
            c='black',
            s=60,
            alpha=0.8,
            label='Centroids',
            zorder=8,
        )
        for idx, (x, y) in enumerate(centroids_2d):
            ax.text(x, y + 1, str(idx), color='black', fontsize=11, fontweight='bold', ha='center', va='bottom', zorder=9)
    
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
    
    # Plot success radius if provided
    if eps_success is not None and eps_success > 0:
        rng = np.random.default_rng(0)
        n_samples = 128
        dirs = rng.normal(size=(n_samples, z_goal.shape[0]))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
        circle_latent = z_goal.reshape(1, -1) + eps_success * dirs
        circle_2d = transformer.transform(circle_latent)
        angles = np.arctan2(circle_2d[:, 1] - z_goal_2d[1], circle_2d[:, 0] - z_goal_2d[0])
        order = np.argsort(angles)
        circle_2d = circle_2d[order]
        ax.plot(circle_2d[:, 0], circle_2d[:, 1], color='black', alpha=0.6, linewidth=2.5, label='Success radius')
    
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
    parser.add_argument("--source_lineage", type=str, default=None, help="Source lineage label (default: random)")
    parser.add_argument("--source_mode", type=str, default="sample", choices=["centroid", "sample"],
                        help="Source mode: 'centroid' (use source lineage centroid) or 'sample' (sample a cell from source lineage, default)")
    parser.add_argument("--target_lineage", type=str, required=True, help="Target lineage label")
    parser.add_argument("--target_mode", type=str, default="centroid", choices=["centroid", "sample"],
                        help="Target mode: 'centroid' (use target lineage centroid, default) or 'sample' (sample a cell from target lineage)")
    parser.add_argument("--T", type=int, default=64, help="Rollout horizon")
    parser.add_argument("--embedding", type=str, default="pca", choices=["pca", "umap"],
                        help="Embedding method: 'pca' or 'umap'")
    parser.add_argument("--z_key", type=str, default="mean", help="Key in adata.obsm for latent states")
    parser.add_argument("--outdir", type=str, default="./viz_results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cpu, cuda)")
    parser.add_argument("--deterministic", action="store_true", default=False,
                        help="Use deterministic policy (default: False, uses stochastic sampling)")
    parser.add_argument("--deterministic_action", action="store_true", default=False,
                        help="Use argmax action but sample magnitude (default: False)")
    parser.add_argument("--intervention_method", type=str, default="heatmap", choices=["stem", "heatmap"],
                        help="Intervention visualization method")
    parser.add_argument("--use_negative_velocity", action="store_true",
                        help="Use negative velocity instead of normal velocity")
    parser.add_argument("--deactivate_velocity", action="store_true",
                        help="Deactivate velocity effect on next state (default: velocity affects state)")
    parser.add_argument("--n_viz_trajectories", type=int, default=1,
                        help="Number of trajectory visualizations to generate (default: 1)")
    parser.add_argument("--reachability_test", action="store_true", default=False,
                        help="Run greedy reachability baseline (default: False)")
    parser.add_argument("--baseline_delta_max", type=float, default=None,
                        help="Delta max for reachability baseline (default: delta_clip from checkpoint or 1.0)")
    parser.add_argument("--reward_mode", type=str, default=None,
                        choices=["plain", "scaled", "milestone", "multi_milestone"],
                        help="Reward mode override (default: use checkpoint config)")
    parser.add_argument("--progress_weight_p", type=float, default=None,
                        help="Near-goal emphasis exponent p for scaled progress (default: use checkpoint config)")
    parser.add_argument("--progress_weight_c", type=float, default=None,
                        help="Near-goal emphasis offset c for scaled progress (default: use checkpoint config)")
    parser.add_argument("--actions_per_step", type=int, default=None,
                        help="Number of action draws per step (default: use checkpoint config or 1)")
    
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
    
    # Load checkpoint
    print(f"Loading policy from {args.checkpoint}...")
    policy, centroids, goal_labels, config = load_policy_checkpoint(args.checkpoint, device)
    print(f"Loaded policy for {len(goal_labels)} goal lineages: {goal_labels}")
    
    # Create mapping from goal label to index
    label_to_goal_idx = {label: idx for idx, label in enumerate(goal_labels)}
    
    target_goal_label = args.target_lineage
    if target_goal_label not in label_to_goal_idx:
        raise ValueError(f"Target lineage '{target_goal_label}' not in goal_labels: {goal_labels}")
    goal_idx = label_to_goal_idx[target_goal_label]
    
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
    # CLI argument overrides config
    use_negative_velocity = args.use_negative_velocity if args.use_negative_velocity else env_config.get("use_negative_velocity", False)
    
    # Create adapter
    adapter = VelocityVAEAdapter(vae.model, device, velocity_mode=velocity_mode)
    
    # Get cluster indices
    cluster_indices = None
    if vae.model.cluster_key is not None:
        cluster_labels = adata.obs[vae.model.cluster_key]
        cluster_indices = torch.tensor([
            vae.model.cluster_to_idx.get(str(label), 0) for label in cluster_labels
        ], dtype=torch.long, device=device)

    # Build embedding (shared across all experiments)
    print(f"Building {args.embedding.upper()} embedding...")
    Z_all = z_all.cpu().numpy()
    embedding, transformer = build_embedding(Z_all, method=args.embedding)
    print(f"Embedding shape: {embedding.shape}")
    
    # Get T_max from config (should match training)
    T_max_viz = env_config.get("T_max", 100)
    # Ensure T_max is at least as large as requested rollout horizon
    T_max_viz = max(T_max_viz, args.T)
    
    # Create environment (shared across all experiments)
    reward_mode = args.reward_mode if args.reward_mode is not None else env_config.get("reward_mode", "plain")
    progress_weight_p = args.progress_weight_p if args.progress_weight_p is not None else env_config.get("progress_weight_p", 0.0)
    progress_weight_c = args.progress_weight_c if args.progress_weight_c is not None else env_config.get("progress_weight_c", 0.1)
    actions_per_step = args.actions_per_step if args.actions_per_step is not None else env_config.get("actions_per_step", 1)
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
        deactivate_velocity=args.deactivate_velocity,
        reward_mode=reward_mode,
        progress_weight_p=progress_weight_p,
        progress_weight_c=progress_weight_c,
    )
    
    # Get lineage labels for coloring (shared across all experiments)
    lineage_labels = None
    if args.lineage_key in adata.obs:
        lineage_labels = adata.obs[args.lineage_key].values
    
    # Get gene program names (shared across all experiments)
    gp_names = None
    if "terms" in adata.uns and len(adata.uns["terms"]) == adapter.n_latent:
        gp_names = adata.uns["terms"]
        print(f"Found gene program names in adata.uns['terms']")
    else:
        print(f"Gene program names not found in adata.uns['terms'], using GP_0, GP_1, etc.")
    
    # Run multiple experiments
    for example_idx in range(args.n_viz_trajectories):
        print(f"\n{'='*60}")
        print(f"Experiment {example_idx + 1}/{args.n_viz_trajectories}")
        print(f"{'='*60}")
        
        # Use different seed for each experiment for reproducibility
        experiment_seed = args.seed + example_idx
        rng = np.random.RandomState(experiment_seed)
        
        # Set PyTorch seed for this experiment to ensure stochastic policy sampling varies
        if not args.deterministic:
            torch.manual_seed(experiment_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(experiment_seed)
        
        # Select start cell based on source_lineage and source_mode
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
            print(f"Goal: cell {goal_cell_idx} from lineage '{target_goal_label}' (sampled with seed={experiment_seed})")
    
        # Get initial x if needed
        x0 = None
        if velocity_mode == "fixed_x":
            if start_cell_idx_for_cluster is not None:
                unspliced_key = "unspliced" if "unspliced" in adata.layers else "Mu"
                spliced_key = "spliced" if "spliced" in adata.layers else "Ms"
                u = torch.from_numpy(np.asarray(adata.layers[unspliced_key][start_cell_idx_for_cluster])).float().to(device)
                s = torch.from_numpy(np.asarray(adata.layers[spliced_key][start_cell_idx_for_cluster])).float().to(device)
                x0 = torch.cat([u, s], dim=0)  # (2*n_genes,)
            else:
                # When using centroid, compute mean gene expression from source lineage
                if args.source_lineage is not None:
                    source_mask = adata.obs[args.lineage_key] == args.source_lineage
                    unspliced_key = "unspliced" if "unspliced" in adata.layers else "Mu"
                    spliced_key = "spliced" if "spliced" in adata.layers else "Ms"
                    u_mean = np.asarray(adata.layers[unspliced_key][source_mask]).mean(axis=0)
                    s_mean = np.asarray(adata.layers[spliced_key][source_mask]).mean(axis=0)
                    u = torch.from_numpy(u_mean).float().to(device)
                    s = torch.from_numpy(s_mean).float().to(device)
                    x0 = torch.cat([u, s], dim=0)
        
        # Get cluster indices for start cell
        if start_cell_idx is not None:
            cluster_idx_val = cluster_indices[start_cell_idx].item() if cluster_indices is not None else None
        else:
            if args.source_lineage is not None:
                source_mask = (adata.obs[args.lineage_key] == args.source_lineage).values
                if cluster_indices is not None:
                    source_cluster_indices = cluster_indices[source_mask].cpu().numpy()
                    cluster_idx_val = int(np.bincount(source_cluster_indices).argmax()) if len(source_cluster_indices) > 0 else None
                else:
                    cluster_idx_val = None
            else:
                cluster_idx_val = None

        cluster_idx = torch.tensor(cluster_idx_val, dtype=torch.long, device=device) if cluster_idx_val is not None else None

        # Roll out baseline trajectory
        print("Rolling out baseline trajectory...")
        z_baseline, distances_baseline = rollout_baseline(
            env, z0, goal_idx, z_goal, args.T, x0, cluster_idx
        )

        if args.reachability_test:
            if args.baseline_delta_max is not None:
                baseline_delta_max = args.baseline_delta_max
            else:
                baseline_delta_max = config.get("delta_clip", None)
                if baseline_delta_max is None:
                    baseline_delta_max = 1.0
            
            print(f"Running reachability baseline (delta_max={baseline_delta_max})...")
            _, distances_greedy, _, _ = rollout_reachability_baseline(
                env, z0, z_goal, args.T, baseline_delta_max, x0, cluster_idx
            )
            best_distance = float(np.min(distances_greedy))
            print(f"Reachability baseline best distance: {best_distance:.4f}")
        
        # Inspect policy at initial state (first experiment only)
        if example_idx == 0:
            print("\nInspecting policy at initial state...")
            obs_init, _ = env.reset(z0, goal_idx, x0, cluster_idx=cluster_idx, goal_state=z_goal)
            obs_init_tensor = torch.from_numpy(obs_init) if isinstance(obs_init, np.ndarray) else obs_init
            obs_init_tensor = obs_init_tensor.to(device).float()
            
            policy_info = inspect_policy(policy, obs_init_tensor, n_samples=1000, actions_per_step=actions_per_step)
            
            print(f"  Action entropy: {policy_info['action_entropy']:.4f} (max: {np.log(policy.n_latent + 1):.4f})")
            print(f"  Top 5 action probabilities:")
            top5_idx = np.argsort(policy_info['action_probs'])[-5:][::-1]
            for idx in top5_idx:
                prob = policy_info['action_probs'][idx]
                freq = policy_info['action_freq'][idx]
                action_name = "no-op" if idx == 0 else f"dim_{idx-1}"
                print(f"    {action_name}: prob={prob:.4f}, sampled_freq={freq:.4f}")
            
            print(f"  Magnitude statistics (for non-zero actions):")
            print(f"    Mean log_std (clamped): {np.mean(policy_info['magnitude_logstd_clamped']):.4f}")
            print(f"    Min log_std (clamped): {np.min(policy_info['magnitude_logstd_clamped']):.4f}")
            print(f"    Max log_std (clamped): {np.max(policy_info['magnitude_logstd_clamped']):.4f}")
            print(f"    Mean std: {np.mean(policy_info['magnitude_std']):.4f}")
            print(f"    Min std: {np.min(policy_info['magnitude_std']):.4f}")
            print(f"    Max std: {np.max(policy_info['magnitude_std']):.4f}")
            
            # Save policy inspection to file
            import json
            policy_info_save = {
                "action_entropy": float(policy_info['action_entropy']),
                "action_probs": policy_info['action_probs'].tolist(),
                "magnitude_logstd_clamped": policy_info['magnitude_logstd_clamped'].tolist(),
                "magnitude_std": policy_info['magnitude_std'].tolist(),
                "magnitude_mu": policy_info['magnitude_mu'].tolist(),
                "action_freq": policy_info['action_freq'].tolist(),
                "delta_mean": float(policy_info['delta_mean']),
                "delta_std": float(policy_info['delta_std']),
            }
            with open(outdir / "policy_inspection.json", "w") as f:
                json.dump(policy_info_save, f, indent=2)
            print(f"  Policy inspection saved to {outdir / 'policy_inspection.json'}")
        
        # Roll out agent trajectory
        print(f"\nRolling out agent trajectory (deterministic={args.deterministic}, deterministic_action={args.deterministic_action})...")
        z_agent, distances_agent, actions, deltas = rollout_agent(
            env, policy, z0, goal_idx, z_goal, args.T, x0, cluster_idx,
            deterministic=args.deterministic,
            deterministic_action=args.deterministic_action,
            actions_per_step=actions_per_step,
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
        
        # Create output file prefix for this experiment
        if args.n_viz_trajectories > 1:
            prefix = f"example_{example_idx}_"
        else:
            prefix = ""
        
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
            args.target_lineage,
            success,
            n_interventions,
            total_magnitude,
            outdir / f"{prefix}trajectory_overlay.png",
            lineage_key=args.lineage_key,
            lineage_labels=lineage_labels,
        )
        
        # Plot B: Distance curves
        plot_distance_curves(
            distances_baseline,
            distances_agent,
            outdir / f"{prefix}distance_curves.png",
        )
        
        # Plot C: Intervention schedule
        plot_interventions(
            actions,
            deltas,
            adapter.n_latent,
            outdir / f"{prefix}interventions.png",
            method=args.intervention_method,
        )
        
        # Plot D: Intervention summary with gene program names
        plot_intervention_summary(
            actions,
            deltas,
            adapter.n_latent,
            outdir / f"{prefix}intervention_summary.png",
            gp_names=gp_names,
        )
        
        # Save raw arrays
        np.savez(
            outdir / f"{prefix}trajectory.npz",
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
            goal_lineage=args.target_lineage,
        )
        print(f"Saved raw arrays → {outdir / f'{prefix}trajectory.npz'}")
        
        print(f"\nExperiment {example_idx + 1} complete! Outputs saved to {outdir}")
    
    print(f"\n{'='*60}")
    print(f"All {args.n_viz_trajectories} experiments complete! Outputs saved to {outdir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

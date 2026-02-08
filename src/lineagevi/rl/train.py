"""Training script for goal-conditioned PPO agent."""

import argparse
import yaml
from pathlib import Path
from typing import Optional, Dict, List
import torch
import numpy as np
import scanpy as sc
from tqdm import tqdm
import matplotlib.pyplot as plt

from ..utils import load_model
from .adapter import VelocityVAEAdapter
from .envs import VectorizedLatentVelocityEnv, LatentVelocityEnv
from .policies import ActorCriticPolicy
from .ppo import PPOTrainer
from .utils import (
    compute_lineage_centroids,
    set_seed,
    save_policy_checkpoint,
)
from .gmm import SklearnGMMScorer
from .fit_gmm import fit_gmm
from .viz import (
    build_embedding,
    rollout_baseline,
    rollout_agent,
    plot_trajectory_overlay,
    plot_distance_curves,
    plot_interventions,
    plot_intervention_summary,
)


def plot_training_curves(
    metrics_history: Dict[str, List[float]],
    step_norms_history: Dict[str, List[float]],
    output_dir: Path,
):
    """
    Plot training curves for RL training metrics.
    
    Parameters
    ----------
    metrics_history : dict
        Dictionary with metric names as keys and lists of values as values.
    step_norms_history : dict
        Dictionary with step norm names as keys and lists of values as values.
    output_dir : Path
        Directory to save plots.
    """
    plots_dir = output_dir / "training_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Combined overview (3x3 grid to include mean_nll)
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle("RL Training Metrics", fontsize=16)
    
    # Policy loss (separate subplot)
    ax = axes[0, 0]
    if "policy_loss" in metrics_history:
        ax.plot(metrics_history["policy_loss"], label="Policy Loss", alpha=0.7, color="blue", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Policy Loss")
    ax.set_title("Policy Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Value loss (separate subplot)
    ax = axes[0, 1]
    if "value_loss" in metrics_history:
        ax.plot(metrics_history["value_loss"], label="Value Loss", alpha=0.7, color="red", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value Loss")
    ax.set_title("Value Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Entropy
    ax = axes[1, 0]
    if "entropy" in metrics_history:
        ax.plot(metrics_history["entropy"], label="Entropy", alpha=0.7, color="green", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Entropy")
    ax.set_title("Policy Entropy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # KL Divergence
    ax = axes[1, 1]
    if "kl" in metrics_history:
        # Use max with epsilon to avoid log(0) issues (KL is now non-negative)
        kl_plot = np.maximum(np.array(metrics_history["kl"]), 1e-12)
        ax.plot(kl_plot, label="KL Divergence", alpha=0.7, color="orange", linewidth=2)
        ax.set_yscale("log")  # KL is now always non-negative, safe for log scale
    ax.set_xlabel("Iteration")
    ax.set_ylabel("KL Divergence")
    ax.set_title("KL Divergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Clip fraction
    ax = axes[2, 0]
    if "clip_fraction" in metrics_history:
        ax.plot(metrics_history["clip_fraction"], label="Clip Fraction", alpha=0.7, color="purple", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fraction")
    ax.set_title("PPO Clip Fraction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Step norms - velocity (separate)
    ax = axes[2, 1]
    if "velocity_magnitude" in step_norms_history:
        ax.plot(step_norms_history["velocity_magnitude"], label="Velocity Norm", alpha=0.7, color="blue", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Velocity Norm")
    ax.set_title("Velocity Magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mean NLL (off-manifold penalty) - if available
    ax = axes[2, 2]
    if "mean_nll" in metrics_history:
        ax.plot(metrics_history["mean_nll"], label="Mean NLL", alpha=0.7, color="purple", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("NLL")
        ax.set_title("Mean Negative Log-Likelihood\n(Off-Manifold Penalty)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.axis('off')  # Hide empty subplot
    
    plt.tight_layout()
    plot_path = plots_dir / "training_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves → {plot_path}")
    
    # Plot 2: Losses with dual y-axes (if both exist) or single axis (if only one)
    if "policy_loss" in metrics_history or "value_loss" in metrics_history:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color1 = 'blue'
        color2 = 'red'
        lines = []
        labels = []
        
        if "policy_loss" in metrics_history and "value_loss" in metrics_history:
            # Both exist - use dual y-axes
            ax1.set_xlabel("Iteration", fontsize=12)
            ax1.set_ylabel("Policy Loss", color=color1, fontsize=12)
            line1 = ax1.plot(metrics_history["policy_loss"], label="Policy Loss", color=color1, linewidth=2, alpha=0.8)
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.grid(True, alpha=0.3)
            
            ax2 = ax1.twinx()  # Create second y-axis
            ax2.set_ylabel("Value Loss", color=color2, fontsize=12)
            line2 = ax2.plot(metrics_history["value_loss"], label="Value Loss", color=color2, linewidth=2, alpha=0.8)
            ax2.tick_params(axis='y', labelcolor=color2)
            
            lines = line1 + line2
            labels = ["Policy Loss", "Value Loss"]
            ax1.legend(lines, labels, loc='upper left', fontsize=11)
        elif "policy_loss" in metrics_history:
            # Only policy loss
            ax1.plot(metrics_history["policy_loss"], label="Policy Loss", color=color1, linewidth=2, alpha=0.8)
            ax1.set_xlabel("Iteration", fontsize=12)
            ax1.set_ylabel("Policy Loss", fontsize=12)
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)
        elif "value_loss" in metrics_history:
            # Only value loss
            ax1.plot(metrics_history["value_loss"], label="Value Loss", color=color2, linewidth=2, alpha=0.8)
            ax1.set_xlabel("Iteration", fontsize=12)
            ax1.set_ylabel("Value Loss", fontsize=12)
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)
        
        ax1.set_title("Training Losses", fontsize=14)
        plt.tight_layout()
        plot_path = plots_dir / "losses.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved losses plot → {plot_path}")
    
    # Plot 3: Step norms (separate subplots for different scales)
    if any(key in step_norms_history for key in ["velocity_magnitude", "perturbation_magnitude", "state_change"]):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Step Norms", fontsize=14)
        
        # Velocity magnitude
        ax = axes[0]
        if "velocity_magnitude" in step_norms_history:
            ax.plot(step_norms_history["velocity_magnitude"], label="Velocity Norm", alpha=0.7, color="blue", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Velocity Norm")
        ax.set_title("Velocity Magnitude")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Perturbation magnitude
        ax = axes[1]
        if "perturbation_magnitude" in step_norms_history:
            ax.plot(step_norms_history["perturbation_magnitude"], label="Perturbation Norm", alpha=0.7, color="red", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Perturbation Norm")
        ax.set_title("Perturbation Magnitude")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # State change
        ax = axes[2]
        if "state_change" in step_norms_history:
            ax.plot(step_norms_history["state_change"], label="State Change Norm", alpha=0.7, color="cyan", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("State Change Norm")
        ax.set_title("State Change Magnitude")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = plots_dir / "step_norms.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved step norms plot → {plot_path}")
    
    # Plot 4: Task metrics
    ppo_metric_keys = {"policy_loss", "value_loss", "entropy", "kl", "clip_fraction"}
    task_metric_keys = [k for k in metrics_history.keys() if k not in ppo_metric_keys]
    if any(key in metrics_history for key in task_metric_keys):
        # Use 3x3 grid, but if mean_nll exists, we'll use all 9 slots
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle("Task-Level Evaluation Metrics", fontsize=16)
        
        # Success rate
        ax = axes[0, 0]
        if "success_rate" in metrics_history:
            ax.plot(metrics_history["success_rate"], label="Success Rate", alpha=0.7, color="green", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Success Rate")
        ax.set_title("Success Rate")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # Mean final distance
        ax = axes[0, 1]
        if "mean_final_distance" in metrics_history:
            ax.plot(metrics_history["mean_final_distance"], label="Final Distance", alpha=0.7, color="blue", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Distance")
        ax.set_title("Mean Final Distance")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Mean best distance
        ax = axes[0, 2]
        if "mean_best_distance" in metrics_history:
            ax.plot(metrics_history["mean_best_distance"], label="Best Distance", alpha=0.7, color="cyan", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Distance")
        ax.set_title("Mean Best Distance")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Mean distance improvement
        ax = axes[1, 0]
        if "mean_distance_improvement" in metrics_history:
            ax.plot(metrics_history["mean_distance_improvement"], label="Distance Improvement", alpha=0.7, color="teal", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Improvement")
        ax.set_title("Mean Distance Improvement\nE[d₀ - d_T]")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Best improvement
        ax = axes[1, 1]
        if "best_improvement" in metrics_history:
            ax.plot(metrics_history["best_improvement"], label="Best Improvement", alpha=0.7, color="darkgreen", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Improvement")
        ax.set_title("Best Improvement\nE[d₀ - min_t d_t]")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # L0 interventions
        ax = axes[1, 2]
        if "L0_interventions" in metrics_history:
            ax.plot(metrics_history["L0_interventions"], label="L0 Interventions", alpha=0.7, color="orange", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Count")
        ax.set_title("L0 Interventions (per episode)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # L1 magnitude
        ax = axes[2, 0]
        if "L1_magnitude" in metrics_history:
            ax.plot(metrics_history["L1_magnitude"], label="L1 Magnitude", alpha=0.7, color="red", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Magnitude")
        ax.set_title("L1 Magnitude (per episode)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # No-op fraction
        ax = axes[2, 1]
        if "noop_fraction" in metrics_history:
            ax.plot(metrics_history["noop_fraction"], label="No-Op Fraction", alpha=0.7, color="purple", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Fraction")
        ax.set_title("No-Op Fraction")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # Reward metrics
        ax = axes[2, 2]
        if "mean_episode_return" in metrics_history:
            ax.plot(metrics_history["mean_episode_return"], label="Episode Return", alpha=0.7, color="brown", linewidth=2)
        if "mean_step_reward" in metrics_history:
            ax.plot(metrics_history["mean_step_reward"], label="Mean Step Reward", alpha=0.7, color="darkred", linewidth=2)
        if "mean_episode_return" in metrics_history or "mean_step_reward" in metrics_history:
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Reward")
            ax.set_title("Reward Metrics")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot mean NLL if available (overlay on episode return if both exist, or use separate subplot)
        if "mean_nll" in metrics_history:
            if "mean_episode_return" in metrics_history or "mean_step_reward" in metrics_history:
                # Both exist: plot NLL on secondary y-axis
                ax2 = ax.twinx()
                ax2.plot(metrics_history["mean_nll"], label="Mean NLL", alpha=0.7, color="purple", linewidth=2, linestyle="--")
                ax2.set_ylabel("NLL (Off-Manifold)", color="purple")
                ax2.tick_params(axis='y', labelcolor="purple")
                # Combine legends
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
            else:
                # Only NLL exists: use the main plot
                ax.plot(metrics_history["mean_nll"], label="Mean NLL", alpha=0.7, color="purple", linewidth=2)
                ax.set_xlabel("Iteration")
                ax.set_ylabel("NLL")
                ax.set_title("Mean Negative Log-Likelihood (Off-Manifold)")
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = plots_dir / "task_metrics.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved task metrics plot → {plot_path}")


def generate_example_visualizations(
    policy: ActorCriticPolicy,
    adapter: VelocityVAEAdapter,
    env_config: dict,
    centroids: torch.Tensor,
    goal_labels: list,
    z_all: torch.Tensor,
    adata: sc.AnnData,
    lineage_key: str,
    output_dir: Path,
    cluster_indices: Optional[torch.Tensor],
    n_examples: int = 3,
    embedding_method: str = "pca",
    T_rollout: int = 50,
    lambda_progress: float = 1.0,
    lambda_act: float = 0.01,
    lambda_mag: float = 0.1,
    R_succ: float = 10.0,
    alpha_stay: float = 0.0,
    perturb_clip: Optional[float] = None,
    use_negative_velocity: bool = False,
    deactivate_velocity: bool = False,
    terminate_on_success: bool = False,
    milestone_rewards: bool = False,
    reward_mode: str = "plain",
    progress_weight_p: float = 0.0,
    progress_weight_c: float = 0.1,
    success_reward_bonus_pct: float = 0.0,
    success_reward_bonus_w: float = 0.0,
    eps_success_decay_factor: float = 0.95,
    target_mode: str = "centroid",
    seed: int = 42,
    deterministic: bool = False,
    gmm_path: Optional[str] = None,
    lambda_off: float = 0.0,
    fixed_goal_idx: Optional[int] = None,
    source_lineage: Optional[str] = None,
    target_lineage: Optional[str] = None,
    source_mode: str = "sample",
    eps_success: Optional[float] = None,
):
    """
    Generate example trajectory visualizations after training.
    
    Parameters
    ----------
    policy : ActorCriticPolicy
        Trained policy.
    adapter : VelocityVAEAdapter
        VAE adapter.
    env_config : dict
        Environment configuration.
    centroids : torch.Tensor
        Goal centroids.
    goal_labels : list
        Goal labels.
    z_all : torch.Tensor
        All latent states.
    adata : sc.AnnData
        AnnData object.
    lineage_key : str
        Key in adata.obs for lineage labels.
    output_dir : Path
        Output directory.
    cluster_indices : torch.Tensor, optional
        Cluster indices.
    n_examples : int
        Number of example trajectories to visualize.
    embedding_method : str
        Embedding method ('pca' or 'umap').
    lambda_progress : float
        Progress reward scaling.
    lambda_act : float
        Action penalty coefficient.
    lambda_mag : float
        Magnitude penalty coefficient.
    R_succ : float
        Success reward bonus.
    use_negative_velocity : bool
        Whether to use negative velocity.
    perturb_clip : float, optional
        Clip applied perturbation magnitude in visualization env.
    """
    device = next(policy.parameters()).device
    n_cells = z_all.shape[0]
    n_goals = len(goal_labels)
    
    # Create visualization output directory
    viz_dir = output_dir / "trajectory_viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Build embedding from all cells
    print(f"Building {embedding_method.upper()} embedding...")
    Z_all = z_all.cpu().numpy()
    embedding, transformer = build_embedding(Z_all, method=embedding_method)
    print(f"Embedding shape: {embedding.shape}")
    
    # Get T_max from env_config (should match training)
    T_max_viz = env_config.get("T_max", 100)
    # Ensure T_max is at least as large as T_rollout
    T_max_viz = max(T_max_viz, T_rollout)
    
    # Get GMM path and lambda_off for visualization (use same as training)
    gmm_path_viz = None
    lambda_off_viz = 0.0
    if lambda_off > 0.0:
        # Use the same GMM that was used during training
        gmm_path_viz = str(output_dir / "gmm.pkl")
        if not Path(gmm_path_viz).exists():
            # Try to use provided gmm_path if it exists
            if gmm_path and Path(gmm_path).exists():
                gmm_path_viz = gmm_path
            else:
                print(f"Warning: GMM not found at {gmm_path_viz}, disabling off-manifold penalty in visualization")
                lambda_off_viz = 0.0
        else:
            lambda_off_viz = lambda_off
    
    # Create single environment for visualization
    eps_success_viz = eps_success if eps_success is not None else env_config.get("eps_success", 0.1)
    single_env = LatentVelocityEnv(
        adapter=adapter,
        centroids=centroids,
        goal_names=goal_labels,
        dt=env_config.get("dt", 0.1),
        T_max=T_max_viz,
        eps_success=eps_success_viz,
        lambda_progress=lambda_progress,
        lambda_act=lambda_act,
        lambda_mag=lambda_mag,
        R_succ=R_succ,
        alpha_stay=alpha_stay,
        perturb_clip=perturb_clip,
        use_negative_velocity=use_negative_velocity,
        deactivate_velocity=deactivate_velocity,
        terminate_on_success=terminate_on_success,
        milestone_rewards=milestone_rewards,
        reward_mode=reward_mode,
        progress_weight_p=progress_weight_p,
        progress_weight_c=progress_weight_c,
        milestone_decay_factor=eps_success_decay_factor,
        success_reward_bonus_pct=success_reward_bonus_pct,
        success_reward_bonus_w=success_reward_bonus_w,
        gmm_path=gmm_path_viz if lambda_off_viz > 0.0 else None,
        lambda_off=lambda_off_viz,
    )
    
    # Precompute source centroid if using centroid mode
    source_centroid_viz = None
    if source_lineage is not None and source_mode == "centroid":
        source_mask = adata.obs[lineage_key] == source_lineage
        source_centroid_viz = z_all[source_mask].mean(dim=0)  # (n_latent,)
        print(f"Using source centroid for visualization: '{source_lineage}'")
    
    # Determine target goal index if target_lineage is specified
    label_to_goal_idx_viz = {label: idx for idx, label in enumerate(goal_labels)}
    target_goal_idx_viz = None
    if target_lineage is not None:
        if target_lineage not in label_to_goal_idx_viz:
            print(f"Warning: Target lineage '{target_lineage}' not in goal_labels, using fixed_goal_idx or random")
        else:
            target_goal_idx_viz = label_to_goal_idx_viz[target_lineage]
            print(f"Using target lineage for visualization: '{target_lineage}' (index {target_goal_idx_viz})")
    
    # Sample example trajectories
    for example_idx in range(n_examples):
        print(f"\nGenerating visualization {example_idx + 1}/{n_examples}...")
        
        rng = np.random.RandomState(seed + example_idx)  # Use seed + example_idx for reproducibility
        
        # Select start cell based on source_lineage and source_mode
        if source_lineage is not None:
            if source_mode == "centroid":
                # Use source centroid
                z0 = source_centroid_viz.clone()
                start_lineage = source_lineage
                start_cell_idx = None  # Not used when using centroid
            else:  # source_mode == "sample"
                # Sample a cell from source lineage
                source_mask = adata.obs[lineage_key] == source_lineage
                source_indices = np.where(source_mask)[0]
                if len(source_indices) == 0:
                    raise ValueError(f"No cells found with source lineage '{source_lineage}'")
                start_cell_idx = rng.choice(source_indices)
                z0 = z_all[start_cell_idx]
                start_lineage = source_lineage
        else:
            # Original behavior: sample random start cell
            start_cell_idx = rng.randint(0, n_cells)
            z0 = z_all[start_cell_idx]
            start_lineage = adata.obs[lineage_key].iloc[start_cell_idx]
        
        # Select goal based on target_lineage or fixed_goal_idx
        if target_goal_idx_viz is not None:
            goal_idx = target_goal_idx_viz
            goal_label = target_lineage
        elif fixed_goal_idx is not None:
            goal_idx = fixed_goal_idx
            goal_label = goal_labels[goal_idx]
        else:
            # Random goal
            goal_idx = rng.randint(0, n_goals)
            goal_label = goal_labels[goal_idx]
        
        # Get goal: sample a cell from target lineage or use centroid
        if target_mode == "centroid":
            z_goal = centroids[goal_idx].to(device)
        else:  # target_mode == "sample"
            # Sample one cell from target lineage
            target_mask = adata.obs[lineage_key] == goal_label
            target_indices = np.where(target_mask)[0]
            if len(target_indices) > 0:
                goal_cell_idx = rng.choice(target_indices)
                z_goal = z_all[goal_cell_idx].to(device)
            else:
                # Fallback to centroid if no cells found
                z_goal = centroids[goal_idx].to(device)
        
        # Get cluster indices for start cell
        if start_cell_idx is not None:
            cluster_idx = cluster_indices[start_cell_idx] if cluster_indices is not None else None
        else:
            # When using centroid, use representative cluster indices from source lineage
            if source_lineage is not None:
                source_mask = adata.obs[lineage_key] == source_lineage
                source_cell_idx_array = np.where(source_mask)[0]
                if cluster_indices is not None:
                    source_cluster_values = cluster_indices[source_cell_idx_array].cpu().numpy()
                    unique, counts = np.unique(source_cluster_values, return_counts=True)
                    mode_cluster = unique[np.argmax(counts)]
                    cluster_idx = torch.tensor(int(mode_cluster), dtype=torch.long, device=device)
                else:
                    cluster_idx = None
            else:
                cluster_idx = cluster_indices[0] if cluster_indices is not None else None

        # Get initial x if needed
        x0 = None
        if adapter.velocity_mode == "fixed_x":
            if start_cell_idx is not None:
                unspliced_key = "unspliced" if "unspliced" in adata.layers else "Mu"
                spliced_key = "spliced" if "spliced" in adata.layers else "Ms"
                u = torch.from_numpy(np.asarray(adata.layers[unspliced_key][start_cell_idx])).float().to(device)
                s = torch.from_numpy(np.asarray(adata.layers[spliced_key][start_cell_idx])).float().to(device)
                x0 = torch.cat([u, s], dim=0)
            else:
                # When using centroid, compute mean gene expression from source lineage
                source_mask = adata.obs[lineage_key] == source_lineage
                unspliced_key = "unspliced" if "unspliced" in adata.layers else "Mu"
                spliced_key = "spliced" if "spliced" in adata.layers else "Ms"
                u_mean = np.asarray(adata.layers[unspliced_key][source_mask]).mean(axis=0)
                s_mean = np.asarray(adata.layers[spliced_key][source_mask]).mean(axis=0)
                u = torch.from_numpy(u_mean).float().to(device)
                s = torch.from_numpy(s_mean).float().to(device)
                x0 = torch.cat([u, s], dim=0)
        
        # Roll out baseline trajectory
        z_baseline, distances_baseline = rollout_baseline(
            single_env, z0, goal_idx, z_goal, T_rollout, x0, cluster_idx
        )

        # Roll out agent trajectory
        z_agent, distances_agent, actions, deltas = rollout_agent(
            single_env, policy, z0, goal_idx, z_goal, T_rollout, x0, cluster_idx,
            deterministic=deterministic
        )
        
        # Compute metrics
        n_interventions = np.sum(actions != 0)
        total_magnitude = np.sum(np.abs(deltas))
        final_distance = distances_agent[-1]
        success = final_distance < single_env.eps_success
        
        # Get lineage labels for coloring
        lineage_labels = None
        if lineage_key in adata.obs:
            lineage_labels = adata.obs[lineage_key].values
        
        # Create visualizations
        example_prefix = f"example_{example_idx + 1}"
        
        # Plot A: Trajectory overlay
        plot_trajectory_overlay(
            embedding,
            z_baseline,
            z_agent,
            z_goal.cpu().numpy(),
            transformer,
            str(start_lineage),
            goal_label,
            success,
            n_interventions,
            total_magnitude,
            viz_dir / f"{example_prefix}_trajectory_overlay.png",
            lineage_key=lineage_key,
            lineage_labels=lineage_labels,
            eps_success=single_env.eps_success,
            centroids=centroids.cpu().numpy(),
        )
        
        # Plot B: Distance curves
        plot_distance_curves(
            distances_baseline,
            distances_agent,
            viz_dir / f"{example_prefix}_distance_curves.png",
        )
        
        # Plot C: Intervention schedule
        plot_interventions(
            actions,
            deltas,
            adapter.n_latent,
            viz_dir / f"{example_prefix}_interventions.png",
            method="heatmap",
        )
        
        # Plot D: Intervention summary with gene program names
        gp_names = None
        if "terms" in adata.uns and len(adata.uns["terms"]) == adapter.n_latent:
            gp_names = adata.uns["terms"]
        
        plot_intervention_summary(
            actions,
            deltas,
            adapter.n_latent,
            viz_dir / f"{example_prefix}_intervention_summary.png",
            gp_names=gp_names,
        )
        
        # Save raw arrays
        np.savez(
            viz_dir / f"{example_prefix}_trajectory.npz",
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
            goal_lineage=goal_label,
        )
        
        print(f"  Saved visualizations for example {example_idx + 1}: start='{start_lineage}', goal='{goal_label}'")
    
    print(f"\nExample trajectory visualizations saved to {viz_dir}")


def load_config(config_path: Optional[str]) -> dict:
    """Load config from YAML file or return defaults."""
    if config_path is not None and Path(config_path).exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        # Default config
        config = {
            "env": {
                "dt": 0.1,
                "T_max": 100,
                "eps_success": 0.1,
                "lambda_progress": 1.0,
                "lambda_act": 0.01,
                "lambda_mag": 0.1,
                "R_succ": 10.0,
                "velocity_mode": "decode_x",
                "use_negative_velocity": False,
            },
            "ppo": {
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_eps": 0.2,
                "target_kl": 0.01,
                "vf_coef": 0.5,
                "ent_coef": 0.01,
                "lr": 3e-4,  # Learning rate. If KL spikes persist, consider reducing to 1e-4 or lower.
                "max_grad_norm": 0.5,
            },
            "policy": {
                "hidden_sizes": [128, 128],
                "activation": "relu",
                "delta_clip": None,
                "delta_max": 1.0,
            },
            "training": {
                "batch_size": 64,
                "T_rollout": 50,
                "epochs": 10,
                "minibatch_size": 64,
                "n_iterations": 1000,
                "save_freq": 100,
            },
        }
    return config


def main():
    parser = argparse.ArgumentParser(description="Train goal-conditioned PPO agent")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained model checkpoint")
    parser.add_argument("--adata_path", type=str, required=True, help="Path to AnnData file")
    parser.add_argument("--lineage_key", type=str, required=True, help="Key in adata.obs for lineage labels")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cpu, cuda)")
    parser.add_argument("--z_key", type=str, default="mean", help="Key in adata.obsm for latent states")
    parser.add_argument("--source_lineage", type=str, default=None, help="Source lineage label (cells will be sampled from this lineage as starting points)")
    parser.add_argument("--target_lineage", type=str, default=None, help="Target lineage label (goal for all episodes)")
    parser.add_argument("--source_mode", type=str, default="sample", choices=["centroid", "sample"],
                        help="Source mode: 'centroid' (use source lineage centroid as starting point) or 'sample' (sample a cell from source lineage, default)")
    parser.add_argument("--target_mode", type=str, default="centroid", choices=["centroid", "sample"],
                        help="Target mode: 'centroid' (use target lineage centroid, default) or 'sample' (sample a cell from target lineage)")
    parser.add_argument("--n_iterations", type=int, default=None, help="Total training iterations (overrides config)")
    parser.add_argument("--epochs", type=int, default=None, help="PPO inner epochs per iteration (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None, help="Environment batch size (overrides config)")
    parser.add_argument("--T_rollout", type=int, default=None, help="Rollout horizon (overrides config)")
    parser.add_argument("--T_max", type=int, default=None, help="Maximum episode length (overrides config, should be >= T_rollout)")
    parser.add_argument("--minibatch_size", type=int, default=None, help="Minibatch size for PPO updates (overrides config)")
    parser.add_argument("--save_freq", type=int, default=None, help="Checkpoint save frequency (overrides config)")
    parser.add_argument("--use_negative_velocity", action="store_true", help="Use negative velocity instead of normal velocity")
    parser.add_argument("--deactivate_velocity", action="store_true", help="Deactivate velocity effect on next state (default: velocity affects state)")
    parser.add_argument("--terminate_on_success", action="store_true", help="Terminate episode immediately on success (default: False)")
    parser.add_argument("--milestone_rewards", action="store_true",
                        help="Enable multi-milestone success rewards within an episode (default: False)")
    parser.add_argument(
        "--reward_mode",
        type=str,
        default="plain",
        choices=["plain", "scaled", "milestone", "multi_milestone"],
        help="Reward mode: plain, scaled, milestone (single), multi_milestone (default: plain)",
    )
    parser.add_argument(
        "--progress_weight_p",
        type=float,
        default=0.0,
        help="Near-goal emphasis exponent p for scaled progress (default: 0.0)",
    )
    parser.add_argument(
        "--progress_weight_c",
        type=float,
        default=0.1,
        help="Near-goal emphasis offset c for scaled progress (default: 0.1)",
    )
    parser.add_argument("--success_reward_bonus_pct", type=float, default=0.0,
                        help="Reward bonus pct applied on success-rate threshold or milestone (default: 0.0)")
    parser.add_argument("--success_reward_bonus_w", type=float, default=0.0,
                        help="Linear reward bonus applied on success-rate threshold or milestone (default: 0.0)")
    parser.add_argument("--dt", type=float, default=None, help="Time step size (overrides config)")
    parser.add_argument("--lambda_progress", type=float, default=None, help="Progress reward scaling factor (overrides config)")
    parser.add_argument("--lambda_act", type=float, default=None, help="Action penalty coefficient (overrides config)")
    parser.add_argument("--lambda_mag", type=float, default=None, help="Magnitude penalty coefficient (overrides config)")
    parser.add_argument("--actions_per_step", type=int, default=None, help="Number of action draws per step (default: 1)")
    parser.add_argument("--R_succ", type=float, default=None, help="Success reward bonus (overrides config)")
    parser.add_argument("--alpha_stay", type=float, default=None, help="State cost coefficient for staying near goal (overrides config, default: 0.0)")
    parser.add_argument("--eps_success", type=float, default=None, help="Success radius as fraction of initial distance (overrides config, default: 0.1)")
    parser.add_argument("--eps_success_decay_on_success", action="store_true",
                        help="Decay eps_success percentage when success rate exceeds a threshold")
    parser.add_argument("--eps_success_pct", type=float, default=0.1,
                        help="Success radius as a fraction of initial distance (default: 0.1)")
    parser.add_argument("--eps_success_success_rate_threshold", type=float, default=0.2,
                        help="Success-rate threshold to trigger eps_success decay (default: 0.2)")
    parser.add_argument("--eps_success_decay_factor", type=float, default=0.95,
                        help="Multiplicative decay factor for eps_success percentage (default: 0.95)")
    # eps_success decay uses success_reward_bonus_pct / success_reward_bonus_w for bonuses
    # eps_success_reward_linear_w removed in favor of success_reward_bonus_w
    parser.add_argument("--perturb_clip", type=float, default=None,
                        help="Clip applied perturbation magnitude (env-side, default: none)")
    parser.add_argument("--delta_max", type=float, default=None, help="Maximum action magnitude (overrides config and auto-calibration)")
    parser.add_argument("--delta_max_scale", type=float, default=0.5, help="Scale factor for auto-calibrated delta_max (default: 0.5)")
    parser.add_argument("--gamma", type=float, default=None, help="Discount factor for future rewards (overrides config, default: 0.99)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config, default: 3e-4)")
    parser.add_argument("--actor_lr", type=float, default=None, help="Actor learning rate (overrides lr)")
    parser.add_argument("--critic_lr", type=float, default=None, help="Critic learning rate (overrides lr)")
    parser.add_argument("--ent_coef", type=float, default=None, help="Entropy coefficient for exploration bonus (overrides config, default: 0.01)")
    parser.add_argument("--ent_coef_final", type=float, default=None,
                        help="Final entropy coefficient for annealing (default: ent_coef)")
    parser.add_argument("--ent_anneal_iters", type=int, default=0,
                        help="Number of iterations to linearly anneal ent_coef to ent_coef_final (default: 0)")
    parser.add_argument("--kl_stop_threshold", type=float, default=0.02,
                        help="Stop PPO epoch if KL exceeds this for 2 consecutive minibatches (default: 0.02)")
    parser.add_argument("--kl_stop_immediate_threshold", type=float, default=0.03,
                        help="Stop PPO epoch immediately if KL exceeds this once (default: 0.03)")
    parser.add_argument("--goal_cond_dim", type=int, default=32,
                        help="Goal-conditioning projection dim for (z_goal - z_t) (default: 32)")
    parser.add_argument("--use_t_norm", action="store_true",
                        help="Include normalized time in policy conditioning (default: False)")
    parser.add_argument("--n_viz_trajectories", type=int, default=3, help="Number of example trajectories to visualize (default: 3)")
    parser.add_argument("--disable_noop_action", action="store_true",
                        help="Disallow no-op action (forces a perturbation each step)")
    parser.add_argument("--viz_embedding", type=str, default="pca", choices=["pca", "umap"], help="Embedding method for visualization (default: pca)")
    parser.add_argument("--skip_viz", action="store_true", help="Skip trajectory visualization after training")
    parser.add_argument("--deterministic", action="store_true", default=False,
                        help="Use deterministic policy for visualization (default: False, uses stochastic sampling)")
    parser.add_argument("--gmm_path", type=str, default=None, help="Path to saved GMM (.pkl). If not provided and lambda_off > 0, will fit automatically")
    parser.add_argument("--gmm_components", type=int, default=32, help="Number of GMM components (default: 32)")
    parser.add_argument("--lambda_off", type=float, default=0.0, help="Off-manifold penalty coefficient (default: 0.0, disabled)")
    parser.add_argument(
        "--hidden_sizes",
        type=str,
        default=None,
        help="Comma-separated hidden layer sizes for policy MLP (overrides config, default: 128,128)",
    )
    parser.add_argument(
        "--actor_hidden_sizes",
        type=str,
        default=None,
        help="Comma-separated hidden sizes for actor trunk (overrides shared hidden_sizes)",
    )
    parser.add_argument(
        "--critic_hidden_sizes",
        type=str,
        default=None,
        help="Comma-separated hidden sizes for critic trunk (overrides shared hidden_sizes)",
    )
    parser.add_argument(
        "--separate_trunks",
        action="store_true",
        help="Use separate actor/critic trunks even if sizes are the same",
    )
    parser.add_argument(
        "--activation",
        type=str,
        choices=["relu", "tanh"],
        default=None,
        help="Activation function for policy MLP (overrides config, default: relu)",
    )
    parser.add_argument(
        "--delta_clip",
        type=float,
        default=None,
        help="Clip magnitude to [-x, x] if set (overrides config, default: none)",
    )
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load config
    config = load_config(args.config)
    env_config = config.get("env", {})
    ppo_config = config.get("ppo", {})
    policy_config = config.get("policy", {})
    training_config = config.get("training", {})

    # Apply architecture overrides
    def parse_sizes_arg(value: Optional[str], flag: str) -> Optional[list]:
        if value is None:
            return None
        try:
            sizes = [int(size.strip()) for size in value.split(",") if size.strip()]
        except ValueError as exc:
            raise ValueError(f"Invalid {flag} '{value}'. Use comma-separated integers.") from exc
        if len(sizes) == 0:
            raise ValueError(f"Invalid {flag}: must provide at least one integer.")
        return sizes
    
    hidden_sizes_cli = parse_sizes_arg(args.hidden_sizes, "--hidden_sizes")
    if hidden_sizes_cli is not None:
        policy_config["hidden_sizes"] = hidden_sizes_cli
    actor_hidden_sizes_cli = parse_sizes_arg(args.actor_hidden_sizes, "--actor_hidden_sizes")
    critic_hidden_sizes_cli = parse_sizes_arg(args.critic_hidden_sizes, "--critic_hidden_sizes")
    if args.activation is not None:
        policy_config["activation"] = args.activation
    if args.delta_clip is not None:
        if args.delta_clip <= 0:
            raise ValueError("--delta_clip must be positive.")
        policy_config["delta_clip"] = args.delta_clip
    
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
    else:
        print(f"Using existing latent states from obsm['{args.z_key}']")
    
    # Compute lineage centroids with filtering
    print(f"Computing lineage centroids from '{args.lineage_key}'...")
    centroids, goal_labels = compute_lineage_centroids(
        adata, 
        args.lineage_key, 
        z_key=args.z_key,
        allowed=None,
        exclude=None,
        min_cells=1,
    )
    centroids = centroids.to(device)
    print(f"Found {len(goal_labels)} goal lineages: {goal_labels}")
    
    # Create mapping from lineage label to goal index
    label_to_goal_idx = {label: idx for idx, label in enumerate(goal_labels)}
    n_goals = len(goal_labels)
    
    # Handle source_lineage and target_lineage
    source_lineage = args.source_lineage
    target_lineage = args.target_lineage
    
    # Validate source and target lineages if provided
    if source_lineage is not None:
        if source_lineage not in adata.obs[args.lineage_key].values:
            raise ValueError(f"Source lineage '{source_lineage}' not found in adata.obs['{args.lineage_key}']")
        source_mask = adata.obs[args.lineage_key] == source_lineage
        n_source_cells = source_mask.sum()
        if n_source_cells == 0:
            raise ValueError(f"Source lineage '{source_lineage}' has no cells")
        print(f"Using source lineage: {source_lineage} ({n_source_cells} cells)")
    
    if target_lineage is not None:
        if target_lineage not in goal_labels:
            raise ValueError(f"Target lineage '{target_lineage}' not in goal_labels: {goal_labels}")
        print(f"Using target lineage: {target_lineage}")
    
    # Handle target_lineage as fixed goal
    fixed_goal_idx = None
    if target_lineage is not None:
        # Use target_lineage as fixed goal
        fixed_goal_idx = label_to_goal_idx[target_lineage]
        print(f"Target lineage '{target_lineage}' set as fixed goal (index {fixed_goal_idx})")
    
    # Create adapter
    velocity_mode = env_config.get("velocity_mode", "decode_x")
    adapter = VelocityVAEAdapter(vae.model, device, velocity_mode=velocity_mode)
    print(f"Created adapter with velocity_mode='{velocity_mode}'")
    
    # Get cluster indices if model uses them
    cluster_indices = None
    if vae.model.cluster_key is not None:
        cluster_labels = adata.obs[vae.model.cluster_key]
        cluster_indices = torch.tensor([
            vae.model.cluster_to_idx.get(str(label), 0) for label in cluster_labels
        ], dtype=torch.long, device=device)

    # Create environment (cluster indices passed per-reset, not in constructor)
    batch_size = args.batch_size if args.batch_size is not None else training_config.get("batch_size", 64)
    dt = args.dt if args.dt is not None else env_config.get("dt", 0.1)
    # Get use_negative_velocity from CLI or config
    use_negative_velocity = args.use_negative_velocity if args.use_negative_velocity else env_config.get("use_negative_velocity", False)
    # Get deactivate_velocity from CLI or config
    deactivate_velocity = args.deactivate_velocity if args.deactivate_velocity else env_config.get("deactivate_velocity", False)
    # Get terminate_on_success from CLI or config
    terminate_on_success = args.terminate_on_success if args.terminate_on_success else env_config.get("terminate_on_success", False)
    # Get milestone_rewards from CLI or config
    reward_mode = args.reward_mode if args.reward_mode is not None else env_config.get("reward_mode", "plain")
    if args.milestone_rewards and reward_mode == "plain":
        reward_mode = "multi_milestone"
    milestone_rewards = reward_mode == "multi_milestone"
    # Get reward parameters from CLI or config
    lambda_progress = args.lambda_progress if args.lambda_progress is not None else env_config.get("lambda_progress", 1.0)
    lambda_act = args.lambda_act if args.lambda_act is not None else env_config.get("lambda_act", 0.01)
    lambda_mag = args.lambda_mag if args.lambda_mag is not None else env_config.get("lambda_mag", 0.1)
    R_succ = args.R_succ if args.R_succ is not None else env_config.get("R_succ", 10.0)
    alpha_stay = args.alpha_stay if args.alpha_stay is not None else env_config.get("alpha_stay", 0.0)
    perturb_clip = args.perturb_clip if args.perturb_clip is not None else env_config.get("perturb_clip", None)
    progress_weight_p = args.progress_weight_p if args.progress_weight_p is not None else env_config.get("progress_weight_p", 0.0)
    progress_weight_c = args.progress_weight_c if args.progress_weight_c is not None else env_config.get("progress_weight_c", 0.0)
    actions_per_step = args.actions_per_step if args.actions_per_step is not None else env_config.get("actions_per_step", 1)
    eps_success = args.eps_success if args.eps_success is not None else env_config.get("eps_success", 0.1)
    eps_success_decay_on_success = args.eps_success_decay_on_success
    eps_success_pct = args.eps_success_pct if args.eps_success is None else eps_success
    eps_success_success_rate_threshold = args.eps_success_success_rate_threshold
    eps_success_decay_factor = args.eps_success_decay_factor
    success_reward_bonus_pct = args.success_reward_bonus_pct
    success_reward_bonus_w = args.success_reward_bonus_w
    if not (0.0 < eps_success_pct <= 1.0):
        raise ValueError("eps_success_pct must be in (0, 1]")
    if milestone_rewards and terminate_on_success:
        raise ValueError("milestone_rewards requires terminate_on_success=False")
    if reward_mode == "multi_milestone" and not (0.0 < eps_success_decay_factor < 1.0):
        raise ValueError("eps_success_decay_factor must be in (0, 1) when milestone_rewards is enabled")
    if success_reward_bonus_pct < 0.0:
        raise ValueError("success_reward_bonus_pct must be >= 0")
    if success_reward_bonus_w < 0.0:
        raise ValueError("success_reward_bonus_w must be >= 0")
    if success_reward_bonus_pct > 0.0 and success_reward_bonus_w > 0.0:
        raise ValueError("success_reward_bonus_pct and success_reward_bonus_w are mutually exclusive")
    if eps_success_decay_on_success:
        if not (0.0 < eps_success_decay_factor < 1.0):
            raise ValueError("eps_success_decay_factor must be in (0, 1) when decay is enabled")
        if not (0.0 <= eps_success_success_rate_threshold <= 1.0):
            raise ValueError("eps_success_success_rate_threshold must be in [0, 1]")
        print(
            "Eps-success decay enabled: "
            f"pct={eps_success_pct}, threshold={eps_success_success_rate_threshold}, "
            f"factor={eps_success_decay_factor}"
        )
    else:
        print(f"Eps-success pct enabled: pct={eps_success_pct}")
    
    # Get source_mode and target_mode from CLI or config
    source_mode = args.source_mode if args.source_mode is not None else env_config.get("source_mode", "sample")
    target_mode = args.target_mode if args.target_mode is not None else env_config.get("target_mode", "centroid")
    
    # Validate source_mode if source_lineage is specified
    if source_lineage is not None:
        if source_mode not in ["centroid", "sample"]:
            raise ValueError(f"source_mode must be 'centroid' or 'sample', got '{source_mode}'")
        print(f"Source mode: {source_mode}")
    
    # Get T_max and T_rollout, ensure T_max >= T_rollout
    T_rollout_for_env = args.T_rollout if args.T_rollout is not None else training_config.get("T_rollout", 50)
    T_max_env = args.T_max if args.T_max is not None else env_config.get("T_max", 100)
    T_max_env = max(T_max_env, T_rollout_for_env)  # Ensure T_max is at least as large as T_rollout
    
    # Handle GMM for off-manifold penalty
    gmm_path = None
    lambda_off = args.lambda_off if args.lambda_off is not None else 0.0
    
    if lambda_off > 0.0:
        if args.gmm_path is not None:
            # Use provided GMM path
            gmm_path = args.gmm_path
            print(f"Using GMM from {gmm_path}")
        else:
            # Fit GMM automatically
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            gmm_path = str(output_dir / "gmm.pkl")
            
            print(f"Fitting GMM with {args.gmm_components} components...")
            fit_gmm(
                adata_path=args.adata_path,
                z_key=args.z_key,
                out_path=gmm_path,
                n_components=args.gmm_components,
                seed=args.seed,
            )
            print(f"Saved GMM to {gmm_path}")
    
    ent_coef = args.ent_coef if args.ent_coef is not None else ppo_config.get("ent_coef", 0.01)
    ent_coef_final = args.ent_coef_final if args.ent_coef_final is not None else ppo_config.get("ent_coef_final", ent_coef)
    ent_anneal_iters = args.ent_anneal_iters if args.ent_anneal_iters is not None else ppo_config.get("ent_anneal_iters", 0)
    kl_stop_threshold = args.kl_stop_threshold if args.kl_stop_threshold is not None else ppo_config.get("kl_stop_threshold", 0.02)
    kl_stop_immediate_threshold = args.kl_stop_immediate_threshold if args.kl_stop_immediate_threshold is not None else ppo_config.get("kl_stop_immediate_threshold", 0.03)
    delta_clip_config = policy_config.get("delta_clip", None)
    
    env = VectorizedLatentVelocityEnv(
        adapter=adapter,
        centroids=centroids,
        goal_names=goal_labels,
        batch_size=batch_size,
        dt=dt,
        T_max=T_max_env,
        eps_success=eps_success,
        lambda_progress=lambda_progress,
        lambda_act=lambda_act,
        lambda_mag=lambda_mag,
        R_succ=R_succ,
        alpha_stay=alpha_stay,
        perturb_clip=perturb_clip,
        use_negative_velocity=use_negative_velocity,
        deactivate_velocity=deactivate_velocity,
        terminate_on_success=terminate_on_success,
        milestone_rewards=milestone_rewards,
        reward_mode=reward_mode,
        progress_weight_p=progress_weight_p,
        progress_weight_c=progress_weight_c,
        milestone_decay_factor=eps_success_decay_factor,
        success_reward_bonus_pct=success_reward_bonus_pct,
        success_reward_bonus_w=success_reward_bonus_w,
        gmm_path=gmm_path,
        lambda_off=lambda_off,
    )
    print(f"Created environment with batch_size={batch_size}, dt={dt}, use_negative_velocity={use_negative_velocity}, deactivate_velocity={deactivate_velocity}, terminate_on_success={terminate_on_success}, reward_mode={reward_mode}")
    print(f"Success threshold: eps_success={eps_success}")
    print(f"Reward parameters: lambda_progress={lambda_progress}, lambda_act={lambda_act}, lambda_mag={lambda_mag}, R_succ={R_succ}, alpha_stay={alpha_stay}")
    print(f"Source mode: {source_mode}, Target mode: {target_mode}")
    
    # Auto-calibrate delta_max based on velocity field drift norms
    z_all = torch.from_numpy(adata.obsm[args.z_key]).float().to(device)  # (n_cells, n_latent)
    n_cells = z_all.shape[0]
    
    # Determine effective delta_max
    if args.delta_max is not None:
        # User provided explicit delta_max, use it as-is
        delta_max_effective = args.delta_max
        median_drift_norm = None
        print(f"Using explicit delta_max: {delta_max_effective:.6f}")
    else:
        # Auto-calibrate from velocity field
        delta_max_scale = args.delta_max_scale
        N_SAMPLE = min(1024, n_cells)
        
        # Sample latents
        sample_indices = torch.randperm(n_cells, device=device)[:N_SAMPLE]
        z_sample = z_all[sample_indices]  # (N_SAMPLE, n_latent)
        
        # Compute velocities using adapter (same as environment)
        with torch.no_grad():
            # Get cluster indices for sampled cells
            cluster_idx_sample = cluster_indices[sample_indices] if cluster_indices is not None else None

            v_sample = adapter.velocity(
                z_sample,
                cluster_indices=cluster_idx_sample,
            )  # (N_SAMPLE, n_latent)
            
            # Apply negative velocity if requested (same as environment)
            if use_negative_velocity:
                v_sample = -v_sample
            
            # Compute drift = dt * v
            drift = dt * v_sample  # (N_SAMPLE, n_latent)
            
            # Compute drift norms
            drift_norm = torch.linalg.norm(drift, dim=-1)  # (N_SAMPLE,)
            
            # Compute median
            median_drift_norm = drift_norm.median().item()
            
            # Set effective delta_max
            delta_max_effective = delta_max_scale * median_drift_norm
        
        print(f"Auto-calibrated delta_max: {delta_max_effective:.6f} (median drift norm {median_drift_norm:.6f}, scale {delta_max_scale})")
    
    # Create policy
    n_latent = adapter.n_latent
    goal_cond_dim = args.goal_cond_dim if args.goal_cond_dim is not None else policy_config.get("goal_cond_dim", 32)
    use_t_norm = args.use_t_norm if args.use_t_norm is not None else policy_config.get("use_t_norm", False)
    obs_dim = (2 * n_latent) + (1 if use_t_norm else 0)  # z + (z_goal - z_t) [+ t]
    hidden_sizes = policy_config.get("hidden_sizes", [128, 128])
    actor_hidden_sizes = actor_hidden_sizes_cli
    critic_hidden_sizes = critic_hidden_sizes_cli
    separate_trunks = args.separate_trunks
    activation = policy_config.get("activation", "relu")
    delta_clip = delta_clip_config
    allow_noop_action = not args.disable_noop_action if args.disable_noop_action is not None else policy_config.get("allow_noop_action", True)
    
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
    print(f"Created policy with obs_dim={obs_dim}, n_latent={n_latent}")
    
    
    # Get PPO hyperparameters (CLI overrides config)
    gamma = args.gamma if args.gamma is not None else ppo_config.get("gamma", 0.99)
    lr = args.lr if args.lr is not None else ppo_config.get("lr", 3e-4)
    actor_lr = args.actor_lr if args.actor_lr is not None else lr
    critic_lr = args.critic_lr if args.critic_lr is not None else lr
    
    # Create trainer
    trainer = PPOTrainer(
        policy=policy,
        env=env,
        gamma=gamma,
        gae_lambda=ppo_config.get("gae_lambda", 0.95),
        clip_eps=ppo_config.get("clip_eps", 0.2),
        target_kl=ppo_config.get("target_kl", 0.01),
        vf_coef=ppo_config.get("vf_coef", 0.5),
        ent_coef=ent_coef,
        ent_coef_final=ent_coef_final,
        ent_anneal_iters=ent_anneal_iters,
        lr=lr,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        max_grad_norm=ppo_config.get("max_grad_norm", 0.5),
        kl_stop_threshold=kl_stop_threshold,
        kl_stop_immediate_threshold=kl_stop_immediate_threshold,
        device=device,
        actions_per_step=actions_per_step,
    )
    print("Created PPO trainer")
    
    # z_all already computed above for delta_max calibration
    n_cells = z_all.shape[0]
    
    # Precompute origin labels for all cells
    origin_labels_all = adata.obs[args.lineage_key].astype(str).values  # (n_cells,)
    
    # Precompute eligible cells for each goal (cells where origin != goal)
    # This ensures goal != origin by construction
    eligible_cells = {}
    for g_idx, goal_label in enumerate(goal_labels):
        eligible_mask = origin_labels_all != goal_label
        eligible_cells[g_idx] = np.where(eligible_mask)[0]
        
        # Check that eligible cells exist
        if len(eligible_cells[g_idx]) == 0:
            if target_lineage is not None and source_lineage is not None and target_lineage == source_lineage and goal_label == target_lineage:
                # Allow same source/target lineage by permitting origin == goal for this goal
                eligible_cells[g_idx] = np.where(origin_labels_all == goal_label)[0]
            else:
                raise ValueError(
                    f"No eligible start cells for goal '{goal_label}' "
                    f"because all cells have that origin. "
                    f"Choose a different goal or dataset."
                )
    
    # If source_lineage is specified, filter eligible cells to only those from source_lineage
    source_cell_indices = None
    if source_lineage is not None:
        source_mask = adata.obs[args.lineage_key] == source_lineage
        source_cell_indices = np.where(source_mask)[0]
        if len(source_cell_indices) == 0:
            raise ValueError(f"Source lineage '{source_lineage}' has no cells")
        
        # If target_lineage is also specified, verify source != target
        if target_lineage is not None:
            target_goal_idx = label_to_goal_idx[target_lineage]
            if source_lineage == target_lineage:
                # Allow starts from the same lineage as the goal
                eligible_cells[target_goal_idx] = source_cell_indices
                print(
                    f"Source lineage '{source_lineage}' matches target lineage '{target_lineage}'. "
                    f"Allowing starts from the same lineage ({len(source_cell_indices)} cells)."
                )
            else:
                # Verify that source cells are eligible for target goal
                source_eligible_for_target = np.intersect1d(source_cell_indices, eligible_cells[target_goal_idx])
                if len(source_eligible_for_target) == 0:
                    raise ValueError(
                        f"No eligible start cells from source lineage '{source_lineage}' for target lineage '{target_lineage}'. "
                        f"This may happen if all cells from source lineage have origin '{target_lineage}'. "
                        f"Choose different source/target lineages."
                    )
                print(f"Source lineage '{source_lineage}' has {len(source_eligible_for_target)} cells eligible for target '{target_lineage}'")
    
    # If fixed_goal is set, verify it has eligible cells
    if fixed_goal_idx is not None:
        if source_lineage is not None:
            # Check if source cells are eligible for fixed goal
            source_eligible = np.intersect1d(source_cell_indices, eligible_cells[fixed_goal_idx])
            if len(source_eligible) == 0:
                raise ValueError(
                    f"No eligible start cells from source lineage '{source_lineage}' for target goal (index {fixed_goal_idx}). "
                    f"All cells from source lineage have origin matching the target goal. "
                    f"Choose different source/target lineages."
                )
        else:
            if len(eligible_cells[fixed_goal_idx]) == 0:
                raise ValueError(
                    f"Target goal '{target_lineage}' has no eligible start cells. "
                    f"All cells have origin matching the goal. "
                    f"Choose a different goal or adjust filters."
                )
    
    print(f"Precomputed eligible cells for {n_goals} goals")
    for g_idx, goal_label in enumerate(goal_labels):
        if source_lineage is not None and fixed_goal_idx == g_idx:
            # Show intersection with source cells
            source_eligible = np.intersect1d(source_cell_indices, eligible_cells[g_idx])
            print(f"  Goal '{goal_label}': {len(eligible_cells[g_idx])} eligible start cells (from source '{source_lineage}': {len(source_eligible)})")
        else:
            print(f"  Goal '{goal_label}': {len(eligible_cells[g_idx])} eligible start cells")
    
    # Training loop
    n_iterations = args.n_iterations if args.n_iterations is not None else training_config.get("n_iterations", 1000)
    T_rollout = args.T_rollout if args.T_rollout is not None else training_config.get("T_rollout", 50)
    T_max = args.T_max if args.T_max is not None else env_config.get("T_max", 100)
    # Ensure T_max is at least as large as T_rollout
    T_max = max(T_max, T_rollout)
    save_freq = args.save_freq if args.save_freq is not None else training_config.get("save_freq", 100)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Precompute source centroid if using centroid mode
    source_centroid = None
    if source_lineage is not None and source_mode == "centroid":
        source_mask = adata.obs[args.lineage_key] == source_lineage
        source_centroid = z_all[source_mask].mean(dim=0)  # (n_latent,)
        print(f"Computed source centroid for '{source_lineage}': shape {source_centroid.shape}")
    
    current_eps_success_pct = eps_success_pct
    decay_count = 0
    print(f"Starting training for {n_iterations} iterations...")
    if source_lineage is not None and target_lineage is not None:
        print(f"Training with fixed source='{source_lineage}' (mode={source_mode}) and target='{target_lineage}' (mode={target_mode})")
    elif source_lineage is not None:
        print(f"Training with fixed source='{source_lineage}' (mode={source_mode}), goals sampled uniformly (target_mode={target_mode})")
    elif target_lineage is not None:
        print(f"Training with fixed target='{target_lineage}' (mode={target_mode}), start cells sampled from eligible origins")
    else:
        print(f"Training uses uniform goal sampling (target_mode={target_mode}). Goals are sampled first, then start cells are sampled from eligible origins (origin != goal).")
    
    # Initialize metrics history
    ppo_metric_keys = [
        "policy_loss",
        "value_loss",
        "entropy",
        "kl",
        "clip_fraction",
        "mean_adv",
        "mean_abs_adv",
        "value_bias",
        "value_mse",
        "ent_coef",
    ]
    task_metric_keys = [
        "success_rate",
        "mean_pct_improvement",
        "best_pct_improvement",
        "L0_interventions",
        "L1_magnitude",
        "noop_fraction",
        "mean_episode_return",
        "mean_step_reward",
        "mean_progress",
        "mean_action_penalty",
        "mean_magnitude_penalty",
        "mean_success_bonus",
        "mean_off_manifold_penalty",
        "eps_success_pct",
        "eps_success_mean",
    ]
    if milestone_rewards:
        task_metric_keys.append("mean_milestones_reached")
    if eps_success_decay_on_success:
        task_metric_keys.extend(["success_reward_bonus_pct", "success_reward_bonus_w"])
    if not milestone_rewards:
        task_metric_keys.append("R_succ_current")
    if lambda_off > 0.0:
        task_metric_keys.append("mean_nll")
    metrics_history = {k: [] for k in (ppo_metric_keys + task_metric_keys)}
    step_norms_history = {
        "velocity_magnitude": [],
        "perturbation_magnitude": [],
        "state_change": [],
    }
    
    
    interrupted = False
    try:
        for iteration in tqdm(range(n_iterations), desc="Training"):
            # Sample goals first (uniform)
            if fixed_goal_idx is not None:
                # Fixed goal for all batch elements
                goal_idx = torch.full((batch_size,), fixed_goal_idx, dtype=torch.long, device=device)
            else:
                # Uniform sampling from all goals
                goal_idx = torch.randint(0, n_goals, (batch_size,), device=device)

            # Sample start cells conditioned on goal (origin != goal)
            if source_lineage is not None and source_mode == "centroid":
                # Use source lineage centroid as starting point for all episodes
                z0 = source_centroid.unsqueeze(0).repeat(batch_size, 1)  # (B, n_latent)
                cell_indices = None  # Not used when using centroid
            else:
                # Sample cells (either from source lineage or all eligible)
                cell_indices = np.zeros(batch_size, dtype=np.int64)
                for i in range(batch_size):
                    g_i = goal_idx[i].item()
                    if source_lineage is not None:
                        # Sample only from source lineage cells that are eligible for this goal
                        source_eligible = np.intersect1d(source_cell_indices, eligible_cells[g_i])
                        if len(source_eligible) == 0:
                            raise ValueError(
                                f"No eligible cells from source lineage '{source_lineage}' for goal index {g_i}. "
                                f"This should have been caught during validation."
                            )
                        cell_indices[i] = np.random.choice(source_eligible)
                    else:
                        # Original behavior: sample from all eligible cells
                        eligible = eligible_cells[g_i]
                        cell_indices[i] = np.random.choice(eligible)

                # Get latent states for sampled cells
                z0 = z_all[cell_indices]  # (B, n_latent)

            # Sample goal states if target_mode == "sample"
            goal_states = None
            if target_mode == "sample":
                goal_states = torch.zeros(batch_size, n_latent, device=device)
                for i in range(batch_size):
                    g_i = goal_idx[i].item()
                    goal_label = goal_labels[g_i]
                    # Sample one cell from target lineage
                    target_mask = adata.obs[args.lineage_key] == goal_label
                    target_indices = np.where(target_mask)[0]
                    if len(target_indices) > 0:
                        goal_cell_idx = np.random.choice(target_indices)
                        goal_states[i] = z_all[goal_cell_idx]
                    else:
                        # Fallback to centroid if no cells found
                        goal_states[i] = centroids[g_i]

            # Set eps_success as a fraction of per-episode initial distance
            goal_z_batch = goal_states if goal_states is not None else centroids[goal_idx]
            initial_distances = torch.norm(z0 - goal_z_batch, p=2, dim=1)
            env.eps_success = current_eps_success_pct * initial_distances

            # Get per-cell cluster indices for sampled cells
            cluster_idx_batch = None
            if cell_indices is not None:
                if cluster_indices is not None:
                    cluster_idx_batch = cluster_indices[cell_indices]
            else:
                # When using centroid, use representative cluster indices from source lineage
                if source_lineage is not None:
                    source_mask = adata.obs[args.lineage_key] == source_lineage
                    source_cell_idx_array = np.where(source_mask)[0]
                    if cluster_indices is not None:
                        source_cluster_values = cluster_indices[source_cell_idx_array].cpu().numpy()
                        unique, counts = np.unique(source_cluster_values, return_counts=True)
                        mode_cluster = unique[np.argmax(counts)]
                        cluster_idx_batch = torch.full((batch_size,), int(mode_cluster), dtype=torch.long, device=device)
                else:
                    if cluster_indices is not None:
                        cluster_idx_batch = torch.full((batch_size,), cluster_indices[0].item(), dtype=torch.long, device=device)

            # Get initial x if needed for fixed_x mode
            x0 = None
            if velocity_mode == "fixed_x":
                if cell_indices is not None:
                    # Get gene expression for sampled cells
                    unspliced_key = "unspliced" if "unspliced" in adata.layers else "Mu"
                    spliced_key = "spliced" if "spliced" in adata.layers else "Ms"
                    u = torch.from_numpy(np.asarray(adata.layers[unspliced_key][cell_indices])).float().to(device)
                    s = torch.from_numpy(np.asarray(adata.layers[spliced_key][cell_indices])).float().to(device)
                    x0 = torch.cat([u, s], dim=1)  # (B, 2*n_genes)
                else:
                    # When using centroid, compute mean gene expression from source lineage
                    source_mask = adata.obs[args.lineage_key] == source_lineage
                    unspliced_key = "unspliced" if "unspliced" in adata.layers else "Mu"
                    spliced_key = "spliced" if "spliced" in adata.layers else "Ms"
                    u_mean = np.asarray(adata.layers[unspliced_key][source_mask]).mean(axis=0)
                    s_mean = np.asarray(adata.layers[spliced_key][source_mask]).mean(axis=0)
                    u = torch.from_numpy(u_mean).float().to(device).unsqueeze(0).repeat(batch_size, 1)
                    s = torch.from_numpy(s_mean).float().to(device).unsqueeze(0).repeat(batch_size, 1)
                    x0 = torch.cat([u, s], dim=1)  # (B, 2*n_genes)

            # Collect rollouts
            batch = trainer.collect_rollouts(z0, goal_idx, T_rollout, x0, cluster_idx_batch, goal_states=goal_states)

            # Compute task metrics
            task_metrics = trainer.compute_task_metrics(batch)
            eps_val = env.eps_success
            if torch.is_tensor(eps_val):
                eps_val = eps_val.mean().item()
            task_metrics = {
                **task_metrics,
                "eps_success_pct": current_eps_success_pct,
                "eps_success_mean": eps_val,
            }
            
            # Conditionally decay eps_success based on success rate
            if eps_success_decay_on_success:
                success_rate = task_metrics.get("success_rate", 0.0)
                decay_triggered = success_rate > eps_success_success_rate_threshold
                if decay_triggered:
                    current_eps_success_pct *= eps_success_decay_factor
                    decay_count += 1
                    if success_reward_bonus_pct > 0.0:
                        env.R_succ = env.base_R_succ * (1.0 + success_reward_bonus_pct * decay_count)
                        task_metrics["success_reward_bonus_pct"] = success_reward_bonus_pct
                        task_metrics["success_reward_bonus_w"] = 0.0
                    elif success_reward_bonus_w > 0.0:
                        env.R_succ = env.base_R_succ + (success_reward_bonus_w * decay_count)
                        task_metrics["success_reward_bonus_pct"] = 0.0
                        task_metrics["success_reward_bonus_w"] = success_reward_bonus_w
                    else:
                        env.R_succ = env.base_R_succ
                        task_metrics["success_reward_bonus_pct"] = 0.0
                        task_metrics["success_reward_bonus_w"] = 0.0
                else:
                    task_metrics["success_reward_bonus_pct"] = 0.0
                    task_metrics["success_reward_bonus_w"] = 0.0
            if not eps_success_decay_on_success:
                task_metrics["success_reward_bonus_pct"] = 0.0
                task_metrics["success_reward_bonus_w"] = 0.0
                env.R_succ = env.base_R_succ
            task_metrics["R_succ_current"] = float(env.R_succ)

            # Update policy
            metrics = trainer.update(
                batch,
                epochs=args.epochs if args.epochs is not None else training_config.get("epochs", 10),
                minibatch_size=args.minibatch_size if args.minibatch_size is not None else training_config.get("minibatch_size", 64),
                iteration=iteration,
            )

            # Compute mean NLL if available
            if lambda_off > 0.0 and "nll" in batch:
                mean_nll = batch["nll"].mean().item()
                all_metrics = {**metrics, **task_metrics, "mean_nll": mean_nll}
            else:
                all_metrics = {**metrics, **task_metrics}

            # Store metrics for plotting
            for k in metrics_history.keys():
                if k in all_metrics:
                    metrics_history[k].append(all_metrics[k])

            # Store step norms (average over last batch)
            if env.step_norms["velocity_magnitude"]:
                recent_velocity = env.step_norms["velocity_magnitude"][-batch_size:] if len(env.step_norms["velocity_magnitude"]) >= batch_size else env.step_norms["velocity_magnitude"]
                recent_perturb = env.step_norms["perturbation_magnitude"][-batch_size:] if len(env.step_norms["perturbation_magnitude"]) >= batch_size else env.step_norms["perturbation_magnitude"]
                recent_state = env.step_norms["state_change"][-batch_size:] if len(env.step_norms["state_change"]) >= batch_size else env.step_norms["state_change"]

                step_norms_history["velocity_magnitude"].append(np.mean(recent_velocity) if len(recent_velocity) > 0 else 0.0)
                step_norms_history["perturbation_magnitude"].append(np.mean(recent_perturb) if len(recent_perturb) > 0 else 0.0)
                step_norms_history["state_change"].append(np.mean(recent_state) if len(recent_state) > 0 else 0.0)


            # Log metrics
            if iteration % 10 == 0:
                print(f"\nIteration {iteration} (dt={dt}):")
                print("  PPO metrics:")
                for k in ppo_metric_keys:
                    if k in all_metrics:
                        print(f"    {k}: {all_metrics[k]:.4f}")
                print("  Task metrics:")
                for k in task_metric_keys:
                    if k in all_metrics:
                        print(f"    {k}: {all_metrics[k]:.4f}")
                # Log step norms from environment
                if env.step_norms["velocity_magnitude"]:
                    print(f"  avg_velocity_norm: {np.mean(env.step_norms['velocity_magnitude'][-10:]):.4f}")
                    print(f"  avg_perturbation_norm: {np.mean(env.step_norms['perturbation_magnitude'][-10:]):.4f}")
                    print(f"  avg_state_change_norm: {np.mean(env.step_norms['state_change'][-10:]):.4f}")

            # Save checkpoint
            if (iteration + 1) % save_freq == 0 or iteration == n_iterations - 1:
                # Update env config with actual T_max used
                config_copy = config.copy()
                if "env" not in config_copy:
                    config_copy["env"] = {}
                config_copy["env"]["T_max"] = T_max_env
                if torch.is_tensor(env.eps_success):
                    config_copy["env"]["eps_success"] = env.eps_success.mean().item()
                else:
                    config_copy["env"]["eps_success"] = env.eps_success
                config_copy["env"]["eps_success_decay_on_success"] = eps_success_decay_on_success
                config_copy["env"]["eps_success_pct"] = current_eps_success_pct
                config_copy["env"]["eps_success_success_rate_threshold"] = eps_success_success_rate_threshold
                config_copy["env"]["eps_success_decay_factor"] = eps_success_decay_factor
                config_copy["env"]["terminate_on_success"] = terminate_on_success
                config_copy["env"]["milestone_rewards"] = milestone_rewards
                config_copy["env"]["reward_mode"] = reward_mode
                config_copy["env"]["milestone_decay_factor"] = eps_success_decay_factor
                config_copy["env"]["lambda_progress"] = lambda_progress
                config_copy["env"]["lambda_act"] = lambda_act
                config_copy["env"]["lambda_mag"] = lambda_mag
                config_copy["env"]["R_succ"] = R_succ
                config_copy["env"]["alpha_stay"] = alpha_stay
                config_copy["env"]["progress_weight_p"] = progress_weight_p
                config_copy["env"]["progress_weight_c"] = progress_weight_c
                config_copy["env"]["success_reward_bonus_pct"] = success_reward_bonus_pct
                config_copy["env"]["success_reward_bonus_w"] = success_reward_bonus_w
                config_copy["env"]["perturb_clip"] = perturb_clip
                config_copy["env"]["actions_per_step"] = actions_per_step

                save_config = {
                    "obs_dim": obs_dim,
                    "n_latent": n_latent,
                    "goal_cond_dim": goal_cond_dim,
                    "use_t_norm": use_t_norm,
                    "hidden_sizes": hidden_sizes,
                    "actor_hidden_sizes": actor_hidden_sizes,
                    "critic_hidden_sizes": critic_hidden_sizes,
                    "separate_trunks": separate_trunks,
                    "allow_noop_action": allow_noop_action,
                    "kl_stop_threshold": kl_stop_threshold,
                    "kl_stop_immediate_threshold": kl_stop_immediate_threshold,
                    "actor_lr": actor_lr,
                    "critic_lr": critic_lr,
                    "delta_max": delta_max_effective,
                    "ent_coef_final": ent_coef_final,
                    "ent_anneal_iters": ent_anneal_iters,
                    "delta_max_calibration": {
                        "delta_max_effective": delta_max_effective,
                        "median_drift_norm": median_drift_norm if median_drift_norm is not None else "N/A (explicit delta_max used)",
                        "delta_max_scale": args.delta_max_scale if args.delta_max is None else "N/A",
                        "dt": dt,
                    },
                    "target_mode": target_mode,
                    **config_copy,
                }
                save_policy_checkpoint(
                    policy,
                    centroids,
                    goal_labels,  # Save goal_labels (not lineage_names)
                    save_config,
                    output_dir,
                    iteration=iteration + 1,
                )
    except KeyboardInterrupt:
        interrupted = True
        print("\nTraining interrupted by user. Proceeding to plots/visualizations...")
    
    if not interrupted:
        print(f"\nTraining complete! Checkpoints saved to {output_dir}")
    
    # Plot training curves
    print("Generating training plots...")
    plot_training_curves(metrics_history, step_norms_history, output_dir)
    
    # Generate example trajectory visualizations
    if not args.skip_viz:
        print("\nGenerating example trajectory visualizations...")
        generate_example_visualizations(
            policy=policy,
            adapter=adapter,
            env_config=env_config,
            centroids=centroids,
            goal_labels=goal_labels,
            z_all=z_all,
            adata=adata,
            lineage_key=args.lineage_key,
            output_dir=output_dir,
            cluster_indices=cluster_indices,
            n_examples=args.n_viz_trajectories,
            embedding_method=args.viz_embedding,
            T_rollout=T_rollout,
            lambda_progress=lambda_progress,
            lambda_act=lambda_act,
            lambda_mag=lambda_mag,
            R_succ=R_succ,
            alpha_stay=alpha_stay,
            perturb_clip=perturb_clip,
            use_negative_velocity=use_negative_velocity,
            deactivate_velocity=deactivate_velocity,
            terminate_on_success=terminate_on_success,
            milestone_rewards=milestone_rewards,
            reward_mode=reward_mode,
            progress_weight_p=progress_weight_p,
            progress_weight_c=progress_weight_c,
            success_reward_bonus_pct=success_reward_bonus_pct,
            success_reward_bonus_w=success_reward_bonus_w,
            eps_success_decay_factor=eps_success_decay_factor,
            target_mode=target_mode,  # Use same target_mode as training
            seed=args.seed,
            gmm_path=gmm_path,
            lambda_off=lambda_off,
            fixed_goal_idx=fixed_goal_idx,
            source_lineage=source_lineage,  # Pass source_lineage from training
            target_lineage=target_lineage,  # Pass target_lineage from training
            source_mode=source_mode,  # Pass source_mode from training
            deterministic=args.deterministic,
            eps_success=eps_success,
        )


if __name__ == "__main__":
    main()

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
from .envs import VectorizedLatentVelocityEnv
from .policies import ActorCriticPolicy
from .ppo import PPOTrainer
from .utils import (
    compute_lineage_centroids,
    set_seed,
    save_policy_checkpoint,
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
    
    # Plot 1: Combined overview (3x2 grid for better organization)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
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
                "lambda_act": 0.01,
                "lambda_mag": 0.1,
                "R_succ": 10.0,
                "velocity_mode": "decode_x",
            },
            "ppo": {
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_eps": 0.2,
                "target_kl": 0.01,
                "vf_coef": 0.5,
                "ent_coef": 0.01,
                "lr": 3e-4,
                "max_grad_norm": 0.5,
            },
            "policy": {
                "hidden_sizes": [128, 128],
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
    parser.add_argument("--goal_allowed", type=str, nargs="*", default=None, help="Allowed goal labels (default: all)")
    parser.add_argument("--goal_exclude", type=str, nargs="*", default=None, help="Excluded goal labels")
    parser.add_argument("--goal_min_cells", type=int, default=1, help="Minimum cells per goal lineage")
    parser.add_argument("--fixed_goal", type=str, default=None, help="Fixed goal label for all episodes (optional)")
    parser.add_argument("--n_iterations", type=int, default=None, help="Total training iterations (overrides config)")
    parser.add_argument("--epochs", type=int, default=None, help="PPO inner epochs per iteration (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None, help="Environment batch size (overrides config)")
    parser.add_argument("--T_rollout", type=int, default=None, help="Rollout horizon (overrides config)")
    parser.add_argument("--minibatch_size", type=int, default=None, help="Minibatch size for PPO updates (overrides config)")
    parser.add_argument("--save_freq", type=int, default=None, help="Checkpoint save frequency (overrides config)")
    
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
        allowed=args.goal_allowed,
        exclude=args.goal_exclude,
        min_cells=args.goal_min_cells,
    )
    centroids = centroids.to(device)
    print(f"Found {len(goal_labels)} goal lineages: {goal_labels}")
    
    # Create mapping from lineage label to goal index
    label_to_goal_idx = {label: idx for idx, label in enumerate(goal_labels)}
    n_goals = len(goal_labels)
    
    # Handle fixed_goal
    fixed_goal_idx = None
    if args.fixed_goal is not None:
        if args.fixed_goal not in label_to_goal_idx:
            raise ValueError(f"Fixed goal '{args.fixed_goal}' not in goal_labels: {goal_labels}")
        fixed_goal_idx = label_to_goal_idx[args.fixed_goal]
        print(f"Using fixed goal: {args.fixed_goal} (index {fixed_goal_idx})")
    
    # Create adapter
    velocity_mode = env_config.get("velocity_mode", "decode_x")
    adapter = VelocityVAEAdapter(vae.model, device, velocity_mode=velocity_mode)
    print(f"Created adapter with velocity_mode='{velocity_mode}'")
    
    # Get cluster/process indices if model uses them
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
    
    # Create environment (cluster/process indices now passed per-reset, not in constructor)
    batch_size = args.batch_size if args.batch_size is not None else training_config.get("batch_size", 64)
    dt = env_config.get("dt", 0.1)
    env = VectorizedLatentVelocityEnv(
        adapter=adapter,
        centroids=centroids,
        goal_names=goal_labels,
        batch_size=batch_size,
        dt=dt,
        T_max=env_config.get("T_max", 100),
        eps_success=env_config.get("eps_success", 0.1),
        lambda_act=env_config.get("lambda_act", 0.01),
        lambda_mag=env_config.get("lambda_mag", 0.1),
        R_succ=env_config.get("R_succ", 10.0),
    )
    print(f"Created environment with batch_size={batch_size}, dt={dt}")
    
    # Create policy
    obs_dim = adapter.n_latent + n_goals + 1  # z + goal_emb + t
    n_latent = adapter.n_latent
    hidden_sizes = policy_config.get("hidden_sizes", [128, 128])
    delta_max = policy_config.get("delta_max", 1.0)
    
    policy = ActorCriticPolicy(
        obs_dim=obs_dim,
        n_latent=n_latent,
        hidden_sizes=hidden_sizes,
        delta_max=delta_max,
    ).to(device)
    print(f"Created policy with obs_dim={obs_dim}, n_latent={n_latent}")
    
    # Create trainer
    trainer = PPOTrainer(
        policy=policy,
        env=env,
        gamma=ppo_config.get("gamma", 0.99),
        gae_lambda=ppo_config.get("gae_lambda", 0.95),
        clip_eps=ppo_config.get("clip_eps", 0.2),
        target_kl=ppo_config.get("target_kl", 0.01),
        vf_coef=ppo_config.get("vf_coef", 0.5),
        ent_coef=ppo_config.get("ent_coef", 0.01),
        lr=ppo_config.get("lr", 3e-4),
        max_grad_norm=ppo_config.get("max_grad_norm", 0.5),
        device=device,
    )
    print("Created PPO trainer")
    
    # Get initial states and goals for training
    z_all = torch.from_numpy(adata.obsm[args.z_key]).float().to(device)  # (n_cells, n_latent)
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
            raise ValueError(
                f"No eligible start cells for goal '{goal_label}' "
                f"because all cells have that origin. "
                f"Adjust goal filters (--goal_allowed, --goal_exclude, --goal_min_cells) or dataset."
            )
    
    # If fixed_goal is set, verify it has eligible cells
    if fixed_goal_idx is not None:
        if len(eligible_cells[fixed_goal_idx]) == 0:
            raise ValueError(
                f"Fixed goal '{args.fixed_goal}' has no eligible start cells. "
                f"All cells have origin '{args.fixed_goal}'. "
                f"Choose a different goal or adjust filters."
            )
    
    print(f"Precomputed eligible cells for {n_goals} goals")
    for g_idx, goal_label in enumerate(goal_labels):
        print(f"  Goal '{goal_label}': {len(eligible_cells[g_idx])} eligible start cells")
    
    # Training loop
    n_iterations = args.n_iterations if args.n_iterations is not None else training_config.get("n_iterations", 1000)
    T_rollout = args.T_rollout if args.T_rollout is not None else training_config.get("T_rollout", 50)
    save_freq = args.save_freq if args.save_freq is not None else training_config.get("save_freq", 100)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting training for {n_iterations} iterations...")
    print("Training uses uniform goal sampling. Goals are sampled first, then start cells are sampled from eligible origins (origin != goal).")
    
    # Initialize metrics history
    metrics_history = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "kl": [],
        "clip_fraction": [],
    }
    step_norms_history = {
        "velocity_magnitude": [],
        "perturbation_magnitude": [],
        "state_change": [],
    }
    
    for iteration in tqdm(range(n_iterations), desc="Training"):
        # Sample goals first (uniform)
        if fixed_goal_idx is not None:
            # Fixed goal for all batch elements
            goal_idx = torch.full((batch_size,), fixed_goal_idx, dtype=torch.long, device=device)
        else:
            # Uniform sampling from all goals
            goal_idx = torch.randint(0, n_goals, (batch_size,), device=device)
        
        # Sample start cells conditioned on goal (origin != goal)
        cell_indices = np.zeros(batch_size, dtype=np.int64)
        for i in range(batch_size):
            g_i = goal_idx[i].item()
            eligible = eligible_cells[g_i]
            cell_indices[i] = np.random.choice(eligible)
        
        # Get latent states for sampled cells
        z0 = z_all[cell_indices]  # (B, n_latent)
        
        # Get per-cell cluster/process indices for sampled cells
        cluster_idx_batch = None
        process_idx_batch = None
        if cluster_indices is not None:
            cluster_idx_batch = cluster_indices[cell_indices]
        if process_indices is not None:
            process_idx_batch = process_indices[cell_indices]
        
        # Get initial x if needed for fixed_x mode
        x0 = None
        if velocity_mode == "fixed_x":
            # Get gene expression for sampled cells
            unspliced_key = "unspliced" if "unspliced" in adata.layers else "Mu"
            spliced_key = "spliced" if "spliced" in adata.layers else "Ms"
            u = torch.from_numpy(np.asarray(adata.layers[unspliced_key][cell_indices])).float().to(device)
            s = torch.from_numpy(np.asarray(adata.layers[spliced_key][cell_indices])).float().to(device)
            x0 = torch.cat([u, s], dim=1)  # (B, 2*n_genes)
        
        # Collect rollouts
        batch = trainer.collect_rollouts(z0, goal_idx, T_rollout, x0, cluster_idx_batch, process_idx_batch)
        
        # Update policy
        metrics = trainer.update(
            batch,
            epochs=args.epochs if args.epochs is not None else training_config.get("epochs", 10),
            minibatch_size=args.minibatch_size if args.minibatch_size is not None else training_config.get("minibatch_size", 64),
        )
        
        # Store metrics for plotting
        for k in metrics_history.keys():
            if k in metrics:
                metrics_history[k].append(metrics[k])
        
        # Store step norms (average over last batch)
        if env.step_norms["velocity_magnitude"]:
            # Get averages from the last batch
            recent_velocity = env.step_norms["velocity_magnitude"][-batch_size:] if len(env.step_norms["velocity_magnitude"]) >= batch_size else env.step_norms["velocity_magnitude"]
            recent_perturb = env.step_norms["perturbation_magnitude"][-batch_size:] if len(env.step_norms["perturbation_magnitude"]) >= batch_size else env.step_norms["perturbation_magnitude"]
            recent_state = env.step_norms["state_change"][-batch_size:] if len(env.step_norms["state_change"]) >= batch_size else env.step_norms["state_change"]
            
            step_norms_history["velocity_magnitude"].append(np.mean(recent_velocity) if len(recent_velocity) > 0 else 0.0)
            step_norms_history["perturbation_magnitude"].append(np.mean(recent_perturb) if len(recent_perturb) > 0 else 0.0)
            step_norms_history["state_change"].append(np.mean(recent_state) if len(recent_state) > 0 else 0.0)
        
        # Log metrics
        if iteration % 10 == 0:
            print(f"\nIteration {iteration} (dt={dt}):")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
            # Log step norms from environment
            if env.step_norms["velocity_magnitude"]:
                print(f"  avg_velocity_norm: {np.mean(env.step_norms['velocity_magnitude'][-10:]):.4f}")
                print(f"  avg_perturbation_norm: {np.mean(env.step_norms['perturbation_magnitude'][-10:]):.4f}")
                print(f"  avg_state_change_norm: {np.mean(env.step_norms['state_change'][-10:]):.4f}")
        
        # Save checkpoint
        if (iteration + 1) % save_freq == 0 or iteration == n_iterations - 1:
            save_config = {
                "obs_dim": obs_dim,
                "n_latent": n_latent,
                "hidden_sizes": hidden_sizes,
                "delta_max": delta_max,
                **config,
            }
            save_policy_checkpoint(
                policy,
                centroids,
                goal_labels,  # Save goal_labels (not lineage_names)
                save_config,
                output_dir,
                iteration=iteration + 1,
            )
    
    print(f"\nTraining complete! Checkpoints saved to {output_dir}")
    
    # Plot training curves
    print("Generating training plots...")
    plot_training_curves(metrics_history, step_norms_history, output_dir)


if __name__ == "__main__":
    main()

"""Training script for goal-conditioned PPO agent."""

import argparse
import yaml
from pathlib import Path
from typing import Optional
import torch
import numpy as np
import scanpy as sc
from tqdm import tqdm

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
    parser.add_argument("--goal_sampling", type=str, default="uniform", choices=["uniform", "balanced_batch"], help="Goal sampling strategy")
    parser.add_argument("--fixed_goal", type=str, default=None, help="Fixed goal label for all episodes (optional)")
    parser.add_argument("--allow_same_as_origin", action="store_true", help="Allow goal to be same as origin lineage")
    
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
    batch_size = training_config.get("batch_size", 64)
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
    
    # Training loop
    n_iterations = training_config.get("n_iterations", 1000)
    T_rollout = training_config.get("T_rollout", 50)
    save_freq = training_config.get("save_freq", 100)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting training for {n_iterations} iterations...")
    
    # Get origin lineage labels for all cells
    origin_labels = adata.obs[args.lineage_key].values  # (n_cells,)
    
    for iteration in tqdm(range(n_iterations), desc="Training"):
        # Sample random initial states
        cell_indices = np.random.choice(n_cells, size=batch_size, replace=True)
        z0 = z_all[cell_indices]  # (B, n_latent)
        
        # Get origin lineage labels for sampled cells
        origin_labels_batch = origin_labels[cell_indices]  # (B,)
        
        # Map origin labels to goal indices (if they exist in goal_labels)
        origin_goal_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        for i, label in enumerate(origin_labels_batch):
            origin_goal_idx[i] = label_to_goal_idx.get(str(label), -1)  # -1 if not in goal_labels
        
        # Sample goals (excluding origin unless allow_same_as_origin or fixed_goal)
        if fixed_goal_idx is not None:
            # Fixed goal for all
            goal_idx = torch.full((batch_size,), fixed_goal_idx, dtype=torch.long, device=device)
            # If fixed goal equals origin and not allowed, resample those cells
            if not args.allow_same_as_origin:
                mask_same = (goal_idx == origin_goal_idx)
                if mask_same.any():
                    # Resample goals for those cells (excluding origin)
                    for i in range(batch_size):
                        if mask_same[i]:
                            valid_goals = [g for g in range(n_goals) if g != origin_goal_idx[i]]
                            if len(valid_goals) > 0:
                                goal_idx[i] = np.random.choice(valid_goals)
        else:
            # Uniform sampling excluding origin
            goal_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
            all_goal_indices = torch.arange(n_goals, device=device)
            
            for i in range(batch_size):
                origin_idx = origin_goal_idx[i].item()
                if args.allow_same_as_origin or origin_idx == -1:
                    # Sample from all goals
                    goal_idx[i] = torch.randint(0, n_goals, (1,), device=device)
                else:
                    # Sample from all goals except origin (vectorized)
                    # Sample integer in [0, n_goals-2], then skip over origin
                    sampled = torch.randint(0, n_goals - 1, (1,), device=device).item()
                    if sampled >= origin_idx:
                        sampled += 1
                    goal_idx[i] = sampled
        
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
            epochs=training_config.get("epochs", 10),
            minibatch_size=training_config.get("minibatch_size", 64),
        )
        
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


if __name__ == "__main__":
    main()

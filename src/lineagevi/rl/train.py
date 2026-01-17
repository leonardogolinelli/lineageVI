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
    
    # Compute lineage centroids
    print(f"Computing lineage centroids from '{args.lineage_key}'...")
    centroids, lineage_names = compute_lineage_centroids(adata, args.lineage_key, z_key=args.z_key)
    centroids = centroids.to(device)
    print(f"Found {len(lineage_names)} lineages: {lineage_names}")
    
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
    
    # Create environment
    batch_size = training_config.get("batch_size", 64)
    dt = env_config.get("dt", 0.1)
    env = VectorizedLatentVelocityEnv(
        adapter=adapter,
        centroids=centroids,
        goal_names=lineage_names,
        batch_size=batch_size,
        dt=dt,
        T_max=env_config.get("T_max", 100),
        eps_success=env_config.get("eps_success", 0.1),
        lambda_act=env_config.get("lambda_act", 0.01),
        lambda_mag=env_config.get("lambda_mag", 0.1),
        R_succ=env_config.get("R_succ", 10.0),
        cluster_indices=cluster_indices,
        process_indices=process_indices,
    )
    print(f"Created environment with batch_size={batch_size}, dt={dt}")
    
    # Create policy
    obs_dim = adapter.n_latent + len(lineage_names) + 1  # z + goal_emb + t
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
    
    for iteration in tqdm(range(n_iterations), desc="Training"):
        # Sample random initial states and goals
        cell_indices = np.random.choice(n_cells, size=batch_size, replace=True)
        z0 = z_all[cell_indices]  # (B, n_latent)
        
        # Sample random goals (can exclude current lineage for harder task)
        goal_idx = torch.randint(0, len(lineage_names), (batch_size,), device=device)
        
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
        batch = trainer.collect_rollouts(z0, goal_idx, T_rollout, x0)
        
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
                lineage_names,
                save_config,
                output_dir,
                iteration=iteration + 1,
            )
    
    print(f"\nTraining complete! Checkpoints saved to {output_dir}")


if __name__ == "__main__":
    main()

"""Test script for RL agent (PPO) training.

This script tests the PPO agent training workflow.
Run with:
    python -m tests.rl.test_training --model_path <path_to_pretrained_model.pt> --adata_path <path_to_data.h5ad>
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch
import scanpy as sc
import lineagevi as lvi
from lineagevi.rl import (
    VelocityVAEAdapter,
    VectorizedLatentVelocityEnv,
    ActorCriticPolicy,
    PPOTrainer,
    compute_lineage_centroids,
    set_seed,
)


def main():
    parser = argparse.ArgumentParser(description="Test RL agent training")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained model checkpoint")
    parser.add_argument("--adata_path", type=str, required=True, help="Path to AnnData file")
    parser.add_argument("--lineage_key", type=str, required=True, help="Key in adata.obs for lineage labels")
    parser.add_argument("--output_dir", type=str, default="./test_outputs/rl", help="Output directory")
    parser.add_argument("--n_iterations", type=int, default=10, help="Number of training iterations")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (small for testing)")
    parser.add_argument("--T_rollout", type=int, default=10, help="Rollout horizon (short for testing)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--z_key", type=str, default="mean", help="Key in adata.obsm for latent states")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load AnnData
    print(f"Loading AnnData from {args.adata_path}...")
    adata = sc.read_h5ad(args.adata_path)
    print(f"Loaded {adata.n_obs} cells, {adata.n_vars} genes")
    
    # Load pretrained model
    print(f"Loading pretrained model from {args.model_path}...")
    vae = lvi.utils.load_model(adata, args.model_path, map_location=device, training=False)
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
    adapter = VelocityVAEAdapter(vae.model, device, velocity_mode="decode_x")
    print("Created adapter")
    
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
    env = VectorizedLatentVelocityEnv(
        adapter=adapter,
        centroids=centroids,
        goal_names=lineage_names,
        batch_size=args.batch_size,
        dt=0.1,
        T_max=50,
        eps_success=0.1,
        lambda_act=0.01,
        lambda_mag=0.1,
        R_succ=10.0,
    )
    print(f"Created environment with batch_size={args.batch_size}")
    
    # Create policy
    obs_dim = adapter.n_latent + len(lineage_names) + 1
    n_latent = adapter.n_latent
    policy = ActorCriticPolicy(
        obs_dim=obs_dim,
        n_latent=n_latent,
        hidden_sizes=[64, 64],  # Smaller for testing
        delta_max=1.0,
    ).to(device)
    print(f"Created policy with obs_dim={obs_dim}, n_latent={n_latent}")
    
    # Create trainer
    trainer = PPOTrainer(
        policy=policy,
        env=env,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        target_kl=0.01,
        vf_coef=0.5,
        ent_coef=0.01,
        lr=3e-4,
        max_grad_norm=0.5,
        device=device,
    )
    print("Created PPO trainer")
    
    # Get initial states
    z_all = torch.from_numpy(adata.obsm[args.z_key]).float().to(device)
    n_cells = z_all.shape[0]
    
    # Training loop (short for testing)
    print(f"Starting training for {args.n_iterations} iterations...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for iteration in range(args.n_iterations):
        # Sample random initial states and goals
        import numpy as np
        cell_indices = np.random.choice(n_cells, size=args.batch_size, replace=True)
        z0 = z_all[cell_indices]
        goal_idx = torch.randint(0, len(lineage_names), (args.batch_size,), device=device)
        
        # Get per-cell indices
        cluster_idx_batch = None
        process_idx_batch = None
        if cluster_indices is not None:
            cluster_idx_batch = cluster_indices[cell_indices]
        if process_indices is not None:
            process_idx_batch = process_indices[cell_indices]
        
        # Collect rollouts
        batch = trainer.collect_rollouts(
            z0, goal_idx, args.T_rollout, 
            x0=None, 
            cluster_idx=cluster_idx_batch, 
            process_idx=process_idx_batch
        )
        
        # Update policy
        metrics = trainer.update(batch, epochs=3, minibatch_size=args.batch_size)
        
        # Log
        if iteration % 2 == 0:
            print(f"\nIteration {iteration}:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
    
    # Save checkpoint
    checkpoint_path = output_dir / "test_policy.pt"
    from lineagevi.rl.utils import save_policy_checkpoint
    save_config = {
        "obs_dim": obs_dim,
        "n_latent": n_latent,
        "hidden_sizes": [64, 64],
        "delta_max": 1.0,
    }
    save_policy_checkpoint(
        policy, centroids, lineage_names, save_config, output_dir, iteration=None
    )
    print(f"\nPolicy checkpoint saved to {checkpoint_path}")
    print("RL training test completed successfully!")


if __name__ == "__main__":
    main()

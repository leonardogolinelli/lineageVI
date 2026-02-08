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
    centroids, goal_labels = compute_lineage_centroids(adata, args.lineage_key, z_key=args.z_key)
    centroids = centroids.to(device)
    print(f"Found {len(goal_labels)} goal lineages: {goal_labels}")
    
    # Create adapter
    adapter = VelocityVAEAdapter(vae.model, device, velocity_mode="decode_x")
    print("Created adapter")
    
    # Get cluster indices if model uses them
    cluster_indices = None
    if vae.model.cluster_key is not None:
        cluster_labels = adata.obs[vae.model.cluster_key]
        cluster_indices = torch.tensor([
            vae.model.cluster_to_idx.get(str(label), 0) for label in cluster_labels
        ], dtype=torch.long, device=device)

    # Create environment
    env = VectorizedLatentVelocityEnv(
        adapter=adapter,
        centroids=centroids,
        goal_names=goal_labels,
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
    n_goals = len(goal_labels)
    obs_dim = adapter.n_latent + n_goals + 1
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
    
    # Precompute origin labels and eligible cells (matching training logic)
    import numpy as np
    origin_labels_all = adata.obs[args.lineage_key].astype(str).values
    
    # Precompute eligible cells for each goal (cells where origin != goal)
    eligible_cells = {}
    for g_idx, goal_label in enumerate(goal_labels):
        eligible_mask = origin_labels_all != goal_label
        eligible_cells[g_idx] = np.where(eligible_mask)[0]
        if len(eligible_cells[g_idx]) == 0:
            raise ValueError(
                f"No eligible start cells for goal '{goal_label}'. "
                f"All cells have that origin. Adjust dataset or goal filters."
            )
    
    print(f"Precomputed eligible cells for {n_goals} goals")
    for g_idx, goal_label in enumerate(goal_labels):
        print(f"  Goal '{goal_label}': {len(eligible_cells[g_idx])} eligible start cells")
    
    # Training loop (short for testing)
    print(f"Starting training for {args.n_iterations} iterations...")
    print("Training uses uniform goal sampling. Goals are sampled first, then start cells from eligible origins.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for iteration in range(args.n_iterations):
        # Sample goals first (uniform)
        goal_idx = torch.randint(0, n_goals, (args.batch_size,), device=device)
        
        # Sample start cells conditioned on goal (origin != goal)
        cell_indices = np.zeros(args.batch_size, dtype=np.int64)
        for i in range(args.batch_size):
            g_i = goal_idx[i].item()
            eligible = eligible_cells[g_i]
            cell_indices[i] = np.random.choice(eligible)
        
        # Get latent states for sampled cells
        z0 = z_all[cell_indices]
        
        # Get per-cell cluster indices
        cluster_idx_batch = cluster_indices[cell_indices] if cluster_indices is not None else None

        # Collect rollouts
        batch = trainer.collect_rollouts(
            z0, goal_idx, args.T_rollout,
            x0=None,
            cluster_idx=cluster_idx_batch,
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
        policy, centroids, goal_labels, save_config, output_dir, iteration=None
    )
    print(f"\nPolicy checkpoint saved to {checkpoint_path}")
    print("RL training test completed successfully!")


if __name__ == "__main__":
    main()

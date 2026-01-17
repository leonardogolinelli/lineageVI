"""Evaluation script for trained PPO policy."""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import torch
import numpy as np
import scanpy as sc
from tqdm import tqdm
import pandas as pd

from ..utils import load_model
from .adapter import VelocityVAEAdapter
from .envs import LatentVelocityEnv
from .utils import load_policy_checkpoint
from typing import Optional


def evaluate_episode(
    env: LatentVelocityEnv,
    policy,
    z0: torch.Tensor,
    goal_idx: int,
    x0: Optional[torch.Tensor] = None,
    cluster_idx: Optional[torch.Tensor] = None,
    process_idx: Optional[torch.Tensor] = None,
    max_steps: int = 200,
) -> Dict:
    """
    Run a single evaluation episode.
    
    Returns
    -------
    episode_info : dict
        Episode statistics.
    """
    obs, info = env.reset(z0, goal_idx, x0, cluster_idx=cluster_idx, process_idx=process_idx)
    
    trajectory = {
        "z": [z0.cpu().numpy()],
        "actions": [],
        "deltas": [],
        "rewards": [],
        "distances": [info["distance"]],
        "done": False,
    }
    
    total_reward = 0.0
    steps = 0
    success = False
    
    for step in range(max_steps):
        # Get action from policy
        obs_tensor = torch.from_numpy(obs) if isinstance(obs, np.ndarray) else obs
        obs_tensor = obs_tensor.to(next(policy.parameters()).device).float()
        
        with torch.no_grad():
            action, delta, _, _ = policy.sample(obs_tensor, deterministic=False)
        
        # Step environment
        obs_next, reward, done, info_next = env.step((action.item(), delta.item()))
        
        # Store trajectory
        trajectory["actions"].append(action.item())
        trajectory["deltas"].append(delta.item())
        trajectory["rewards"].append(reward)
        trajectory["distances"].append(info_next["distance"])
        trajectory["z"].append(env.z.cpu().numpy())
        
        total_reward += reward
        steps += 1
        
        if done:
            success = info_next.get("success", False)
            trajectory["done"] = True
            break
        
        obs = obs_next
    
    # Compute metrics
    n_interventions = sum(1 for a in trajectory["actions"] if a != 0)
    total_magnitude = sum(abs(d) for d in trajectory["deltas"])
    final_distance = trajectory["distances"][-1]
    
    episode_info = {
        "success": success,
        "steps": steps,
        "total_reward": total_reward,
        "n_interventions": n_interventions,
        "total_magnitude": total_magnitude,
        "final_distance": final_distance,
        "initial_distance": trajectory["distances"][0],
        "actions": trajectory["actions"],
        "deltas": trajectory["deltas"],
        "distances": trajectory["distances"],
    }
    
    return episode_info, trajectory


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO policy")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to policy checkpoint")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained VAE model")
    parser.add_argument("--adata_path", type=str, required=True, help="Path to AnnData file")
    parser.add_argument("--lineage_key", type=str, required=True, help="Key in adata.obs for lineage labels")
    parser.add_argument("--n_episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--output_dir", type=str, default="./eval_results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cpu, cuda)")
    parser.add_argument("--z_key", type=str, default="mean", help="Key in adata.obsm for latent states")
    parser.add_argument("--max_steps", type=int, default=200, help="Maximum steps per episode")
    
    args = parser.parse_args()
    
    # Set seed
    from .utils import set_seed
    set_seed(args.seed)
    
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading policy from {args.checkpoint}...")
    policy, centroids, lineage_names, config = load_policy_checkpoint(args.checkpoint, device)
    print(f"Loaded policy for {len(lineage_names)} lineages: {lineage_names}")
    
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
    
    # Get environment config from checkpoint
    env_config = config.get("env", {})
    velocity_mode = env_config.get("velocity_mode", "decode_x")
    
    # Create adapter
    adapter = VelocityVAEAdapter(vae.model, device, velocity_mode=velocity_mode)
    
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
    
    # Create environment (single episode, indices passed per-reset)
    env = LatentVelocityEnv(
        adapter=adapter,
        centroids=centroids,
        goal_names=lineage_names,
        dt=env_config.get("dt", 0.1),
        T_max=env_config.get("T_max", 100),
        eps_success=env_config.get("eps_success", 0.1),
        lambda_act=env_config.get("lambda_act", 0.01),
        lambda_mag=env_config.get("lambda_mag", 0.1),
        R_succ=env_config.get("R_succ", 10.0),
    )
    
    # Get latent states
    z_all = torch.from_numpy(adata.obsm[args.z_key]).float().to(device)  # (n_cells, n_latent)
    n_cells = z_all.shape[0]
    
    # Run evaluation episodes
    print(f"Running {args.n_episodes} evaluation episodes...")
    all_episodes = []
    all_trajectories = []
    
    for episode in tqdm(range(args.n_episodes), desc="Evaluating"):
        # Sample random initial state and goal
        cell_idx = np.random.randint(0, n_cells)
        z0 = z_all[cell_idx]  # (n_latent,)
        goal_idx = np.random.randint(0, len(lineage_names))
        
        # Get per-episode cluster/process indices
        cluster_idx = cluster_indices[cell_idx] if cluster_indices is not None else None
        process_idx = process_indices[cell_idx] if process_indices is not None else None
        
        # Get initial x if needed
        x0 = None
        if velocity_mode == "fixed_x":
            unspliced_key = "unspliced" if "unspliced" in adata.layers else "Mu"
            spliced_key = "spliced" if "spliced" in adata.layers else "Ms"
            u = torch.from_numpy(np.asarray(adata.layers[unspliced_key][cell_idx])).float().to(device)
            s = torch.from_numpy(np.asarray(adata.layers[spliced_key][cell_idx])).float().to(device)
            x0 = torch.cat([u, s])  # (2*n_genes,)
        
        # Run episode
        episode_info, trajectory = evaluate_episode(
            env, policy, z0, goal_idx, x0, cluster_idx, process_idx, max_steps=args.max_steps
        )
        
        all_episodes.append(episode_info)
        all_trajectories.append(trajectory)
    
    # Compute aggregate metrics
    success_rate = sum(1 for e in all_episodes if e["success"]) / len(all_episodes)
    steps_to_success = [e["steps"] for e in all_episodes if e["success"]]
    median_steps = np.median(steps_to_success) if steps_to_success else None
    mean_interventions = np.mean([e["n_interventions"] for e in all_episodes])
    mean_magnitude = np.mean([e["total_magnitude"] for e in all_episodes])
    mean_return = np.mean([e["total_reward"] for e in all_episodes])
    
    # Action distribution
    all_actions = [a for e in all_episodes for a in e["actions"] if a != 0]
    action_distribution = pd.Series(all_actions).value_counts().sort_index()
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Success rate: {success_rate:.2%}")
    if median_steps is not None:
        print(f"Median steps to success: {median_steps:.1f}")
    print(f"Mean total interventions (L0): {mean_interventions:.2f}")
    print(f"Mean total magnitude (L1): {mean_magnitude:.4f}")
    print(f"Mean return: {mean_return:.4f}")
    print(f"\nAction distribution (selected latent dimensions):")
    print(action_distribution)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    metrics = {
        "success_rate": success_rate,
        "median_steps_to_success": float(median_steps) if median_steps is not None else None,
        "mean_interventions": float(mean_interventions),
        "mean_magnitude": float(mean_magnitude),
        "mean_return": float(mean_return),
        "action_distribution": action_distribution.to_dict(),
    }
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save episode details
    episodes_df = pd.DataFrame(all_episodes)
    episodes_df.to_csv(output_dir / "episodes.csv", index=False)
    
    # Save example trajectories
    example_trajectories = all_trajectories[:10]  # Save first 10
    with open(output_dir / "example_trajectories.json", "w") as f:
        # Convert numpy arrays to lists for JSON
        trajectories_serializable = []
        for traj in example_trajectories:
            traj_serializable = {
                "z": [z.tolist() for z in traj["z"]],
                "actions": traj["actions"],
                "deltas": traj["deltas"],
                "rewards": traj["rewards"],
                "distances": traj["distances"],
                "done": traj["done"],
            }
            trajectories_serializable.append(traj_serializable)
        json.dump(trajectories_serializable, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    from typing import Optional
    main()

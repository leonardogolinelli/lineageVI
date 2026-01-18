"""PPO trainer with GAE, clipping, and KL early stopping."""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

from .envs import VectorizedLatentVelocityEnv
from .policies import ActorCriticPolicy


class PPOTrainer:
    """
    PPO trainer for goal-conditioned cell reprogramming.
    
    Handles:
    - Rollout collection
    - GAE advantage computation (with terminal state handling)
    - PPO update with clipping and KL early stopping
    - Logging
    """
    
    def __init__(
        self,
        policy: ActorCriticPolicy,
        env: VectorizedLatentVelocityEnv,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        target_kl: float = 0.01,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        lr: float = 3e-4,
        max_grad_norm: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        self.policy = policy
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.target_kl = target_kl
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        
        self.device = device if device is not None else next(policy.parameters()).device
        
        # Optimizer
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        
        # Logging
        self.metrics = defaultdict(list)
    
    def collect_rollouts(
        self,
        z0: torch.Tensor,  # (B, n_latent)
        goal_idx: torch.Tensor,  # (B,)
        T_rollout: int,
        x0: Optional[torch.Tensor] = None,
        cluster_idx: Optional[torch.Tensor] = None,  # (B,)
        process_idx: Optional[torch.Tensor] = None,  # (B,)
    ) -> Dict[str, torch.Tensor]:
        """
        Collect rollout trajectories.
        
        Parameters
        ----------
        z0 : torch.Tensor
            Initial latent states of shape (batch_size, n_latent).
        goal_idx : torch.Tensor
            Goal indices of shape (batch_size,).
        T_rollout : int
            Rollout horizon.
        x0 : torch.Tensor, optional
            Initial gene expression for fixed_x mode.
        cluster_idx : torch.Tensor, optional
            Cluster indices of shape (batch_size,).
        process_idx : torch.Tensor, optional
            Process indices of shape (batch_size,).
        
        Returns
        -------
        batch : dict
            Dictionary with keys:
            - obs: (T, B, obs_dim)
            - action: (T, B)
            - delta: (T, B)
            - raw_delta: (T, B)
            - reward: (T, B)
            - done: (T, B)
            - log_prob: (T, B)
            - value: (T, B)
            - next_value: (T, B) - value at next state (0 if done)
        """
        batch_size = z0.shape[0]
        
        # Reset environment
        obs, info = self.env.reset(z0, goal_idx, x0, cluster_idx=cluster_idx, process_idx=process_idx)
        
        # Storage
        obs_list = []
        action_list = []
        delta_list = []
        raw_delta_list = []
        reward_list = []
        done_list = []
        log_prob_list = []
        value_list = []
        
        # Collect rollouts
        for t in range(T_rollout):
            # Get action from current policy
            obs_tensor = torch.from_numpy(obs) if isinstance(obs, np.ndarray) else obs
            obs_tensor = obs_tensor.to(self.device).float()
            
            action, delta, raw_delta, log_prob = self.policy.sample(obs_tensor)
            value = self.policy.value(obs_tensor).squeeze(-1)  # (B,)
            
            # Step environment (env handles device internally)
            obs_next, reward, done, info_next = self.env.step((action, delta))
            
            # Store (ensure all on CPU for consistency, detach gradients to avoid backward issues)
            obs_list.append(obs_tensor.detach().cpu())
            action_list.append(action.detach().cpu())
            delta_list.append(delta.detach().cpu())
            raw_delta_list.append(raw_delta.detach().cpu())
            reward_list.append(reward.detach().cpu() if torch.is_tensor(reward) else torch.from_numpy(reward))
            done_list.append(done.detach().cpu() if torch.is_tensor(done) else torch.from_numpy(done))
            log_prob_list.append(log_prob.detach().cpu())
            value_list.append(value.detach().cpu())
            
            # Update obs for next iteration
            obs = obs_next
            
            # Break if all done
            if isinstance(done, torch.Tensor):
                if done.all():
                    break
            elif isinstance(done, np.ndarray):
                if done.all():
                    break
        
        # Stack into tensors: (T, B, ...)
        T_actual = len(obs_list)
        obs_batch = torch.stack(obs_list, dim=0)  # (T, B, obs_dim)
        action_batch = torch.stack(action_list, dim=0)  # (T, B)
        delta_batch = torch.stack(delta_list, dim=0)  # (T, B)
        raw_delta_batch = torch.stack(raw_delta_list, dim=0)  # (T, B)
        reward_batch = torch.stack(reward_list, dim=0)  # (T, B)
        done_batch = torch.stack(done_list, dim=0)  # (T, B)
        log_prob_batch = torch.stack(log_prob_list, dim=0)  # (T, B)
        value_batch = torch.stack(value_list, dim=0)  # (T, B)
        
        # Get next value (for bootstrapping)
        # If done, next_value = 0 (terminal/absorbing)
        obs_next_tensor = torch.from_numpy(obs_next) if isinstance(obs_next, np.ndarray) else obs_next
        obs_next_tensor = obs_next_tensor.to(self.device).float()
        next_value = self.policy.value(obs_next_tensor).squeeze(-1).detach().cpu()  # (B,) on CPU
        
        # Build next_value_batch: value at t+1 (0 if done at t)
        next_value_batch = torch.zeros(T_actual, batch_size)
        for t in range(T_actual):
            done_t = done_batch[t]
            if t + 1 < T_actual:
                # Use value at t+1 if not done at t
                next_value_t = value_batch[t + 1] * (~done_t).float()
            else:
                # Last step: use final next_value if not done
                next_value_t = next_value * (~done_t).float()
            next_value_batch[t] = next_value_t
        
        batch = {
            "obs": obs_batch,
            "action": action_batch,
            "delta": delta_batch,
            "raw_delta": raw_delta_batch,
            "reward": reward_batch,
            "done": done_batch,
            "log_prob": log_prob_batch,
            "value": value_batch,
            "next_value": next_value_batch,
        }
        
        return batch
    
    def compute_gae(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and returns.
        
        Important: When done=True (success), V(s_{t+1}) = 0 (no bootstrapping past terminal).
        
        Parameters
        ----------
        batch : dict
            Batch from collect_rollouts.
        
        Returns
        -------
        advantages : torch.Tensor
            GAE advantages of shape (T*B,).
        returns : torch.Tensor
            Return targets of shape (T*B,).
        """
        rewards = batch["reward"]  # (T, B)
        values = batch["value"]  # (T, B)
        next_values = batch["next_value"]  # (T, B)
        dones = batch["done"]  # (T, B)
        
        T, B = rewards.shape
        
        # Compute TD residuals
        # δ_t = r_t + γ * (1 - done_t) * V(s_{t+1}) - V(s_t)
        # When done=True, (1 - done_t) = 0, so no bootstrapping
        td_residuals = rewards + self.gamma * (~dones).float() * next_values - values  # (T, B)
        
        # Compute GAE advantages (backward recursion)
        advantages = torch.zeros_like(td_residuals)  # (T, B)
        last_gae = torch.zeros(B, device=td_residuals.device)  # Per-environment GAE
        
        for t in reversed(range(T)):
            # A_t = δ_t + γ * λ * (1 - done_t) * A_{t+1}
            # When done=True, (1 - done_t) = 0, so advantage doesn't propagate
            advantages[t] = td_residuals[t] + self.gamma * self.gae_lambda * (~dones[t]).float() * last_gae
            last_gae = advantages[t]
        
        # Compute returns: R_t = A_t + V(s_t)
        returns = advantages + values
        
        # Flatten: (T*B,)
        advantages_flat = advantages.flatten()
        returns_flat = returns.flatten()
        
        # Normalize advantages
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
        
        return advantages_flat, returns_flat
    
    def update(
        self,
        batch: Dict[str, torch.Tensor],
        epochs: int = 10,
        minibatch_size: int = 64,
    ) -> Dict[str, float]:
        """
        Update policy using PPO.
        
        Parameters
        ----------
        batch : dict
            Batch from collect_rollouts.
        epochs : int
            Number of PPO epochs.
        minibatch_size : int
            Minibatch size for updates.
        
        Returns
        -------
        metrics : dict
            Training metrics.
        """
        # Compute advantages and returns
        advantages, returns = self.compute_gae(batch)
        
        # Flatten batch for minibatching
        T, B = batch["obs"].shape[:2]
        N = T * B
        
        # Detach all batch tensors to avoid gradient issues when reusing across epochs
        obs_flat = batch["obs"].detach().reshape(N, -1)  # (N, obs_dim)
        action_flat = batch["action"].detach().reshape(N)  # (N,)
        delta_flat = batch["delta"].detach().reshape(N)  # (N,)
        raw_delta_flat = batch["raw_delta"].detach().reshape(N)  # (N,)
        log_prob_old_flat = batch["log_prob"].detach().reshape(N)  # (N,)
        value_old_flat = batch["value"].detach().reshape(N)  # (N,)
        
        # Move to device
        obs_flat = obs_flat.to(self.device)
        action_flat = action_flat.to(self.device)
        delta_flat = delta_flat.to(self.device)
        raw_delta_flat = raw_delta_flat.to(self.device)
        log_prob_old_flat = log_prob_old_flat.to(self.device)
        value_old_flat = value_old_flat.to(self.device)
        advantages = advantages.detach().to(self.device)
        returns = returns.detach().to(self.device)
        
        # Training metrics
        metrics = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "kl": [],
            "clip_fraction": [],
        }
        
        # PPO update loop
        for epoch in range(epochs):
            # Shuffle indices
            indices = torch.randperm(N, device=self.device)
            
            # Minibatch updates
            for start_idx in range(0, N, minibatch_size):
                end_idx = min(start_idx + minibatch_size, N)
                mb_indices = indices[start_idx:end_idx]
                
                # Get minibatch
                mb_obs = obs_flat[mb_indices]
                mb_action = action_flat[mb_indices]
                mb_delta = delta_flat[mb_indices]
                mb_raw_delta = raw_delta_flat[mb_indices]
                mb_log_prob_old = log_prob_old_flat[mb_indices]
                mb_value_old = value_old_flat[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                
                # Get new log probs and values
                mb_log_prob_new = self.policy.log_prob(mb_obs, mb_action, mb_delta, mb_raw_delta)
                mb_value_new = self.policy.value(mb_obs).squeeze(-1)
                mb_entropy = self.policy.entropy(mb_obs)
                
                # Compute ratio
                ratio = torch.exp(mb_log_prob_new - mb_log_prob_old)  # (M,)
                
                # Clipped surrogate loss
                surr1 = ratio * mb_advantages  # (M,)
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_advantages  # (M,)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = 0.5 * (mb_value_new - mb_returns).pow(2).mean()
                
                # Entropy bonus
                entropy_bonus = mb_entropy.mean()
                
                # Total loss
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_bonus
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Compute metrics
                with torch.no_grad():
                    # Approximate KL (use absolute value for non-negative metric)
                    kl = (mb_log_prob_old - mb_log_prob_new).mean().abs()
                    
                    # Clip fraction
                    clip_fraction = ((ratio < 1 - self.clip_eps) | (ratio > 1 + self.clip_eps)).float().mean()
                
                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy_bonus.item())
                metrics["kl"].append(kl.item())
                metrics["clip_fraction"].append(clip_fraction.item())
            
            # Check KL early stopping (fix edge case when minibatch_size > N)
            n_mb = max(1, int(np.ceil(N / minibatch_size)))
            avg_kl = float(np.mean(metrics["kl"][-n_mb:]))
            if avg_kl > self.target_kl:
                break
        
        # Average metrics
        final_metrics = {k: np.mean(v) for k, v in metrics.items()}
        return final_metrics

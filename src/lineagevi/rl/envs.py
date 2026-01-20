"""RL environment for latent velocity dynamics."""

from typing import Optional, Tuple, Dict, Literal
import torch
import torch.nn as nn
import numpy as np
import scanpy as sc

from .adapter import VelocityVAEAdapter
from .gmm import SklearnGMMScorer


class LatentVelocityEnv:
    """
    Single-episode environment for latent velocity dynamics.
    
    State: (z_t, goal_embedding, t_normalized)
    Action: (a_t ∈ {0,...,d}, Δ_t ∈ R)
    Transition: z_tilde = z + u, v = velocity_decoder(z_tilde, x), z_next = z_tilde + dt * v
    Reward: λ_progress*(d_t - d_{t+1}) - λ_act*I[a≠0] - λ_mag*|Δ| + R_succ*I[success]
    Termination: ||z - centroid[g]|| < eps_success OR t >= T_max
    
    Success is terminal/absorbing: when success is reached, done=True immediately,
    and no bootstrapping occurs past success (V(s_{t+1}) = 0 when done=True).
    """
    
    def __init__(
        self,
        adapter: VelocityVAEAdapter,
        centroids: torch.Tensor,  # (n_goals, n_latent)
        goal_names: list,
        dt: float = 0.1,
        T_max: int = 100,
        eps_success: float = 0.1,
        lambda_progress: float = 1.0,
        lambda_act: float = 0.01,
        lambda_mag: float = 0.1,
        R_succ: float = 10.0,
        alpha_stay: float = 0.0,
        goal_emb_dim: Optional[int] = None,
        cluster_indices: Optional[torch.Tensor] = None,
        process_indices: Optional[torch.Tensor] = None,
        use_negative_velocity: bool = False,
        deactivate_velocity: bool = False,
        gmm_path: Optional[str] = None,
        lambda_off: float = 0.0,
    ):
        self.adapter = adapter
        self.centroids = centroids.to(adapter.device)  # (n_goals, n_latent)
        self.goal_names = goal_names
        self.n_goals = len(goal_names)
        self.n_latent = adapter.n_latent
        self.dt = dt
        self.T_max = T_max
        self.eps_success = eps_success
        self.lambda_progress = lambda_progress
        self.lambda_act = lambda_act
        self.lambda_mag = lambda_mag
        self.R_succ = R_succ
        self.alpha_stay = alpha_stay
        self.use_negative_velocity = use_negative_velocity
        self.deactivate_velocity = deactivate_velocity
        self.lambda_off = lambda_off
        
        # Initialize GMM scorer if lambda_off > 0
        self.gmm_scorer = None
        if lambda_off > 0.0:
            if gmm_path is None:
                raise ValueError("gmm_path must be provided when lambda_off > 0")
            self.gmm_scorer = SklearnGMMScorer(gmm_path)
        
        # Goal encoding: one-hot if goal_emb_dim is None, else learned embedding
        self.goal_emb_dim = goal_emb_dim if goal_emb_dim is not None else self.n_goals
        self.use_learned_goal_emb = goal_emb_dim is not None
        
        # Cluster and process indices (fixed for episode)
        self.cluster_indices = cluster_indices
        self.process_indices = process_indices
        
        # Episode state
        self.z: Optional[torch.Tensor] = None
        self.goal_idx: Optional[int] = None
        self.t: int = 0
        self.done: bool = False
        
        # Logging
        self.step_norms: Dict[str, list] = {
            "velocity_magnitude": [],
            "perturbation_magnitude": [],
            "state_change": [],
        }
    
    def reset(
        self,
        z0: torch.Tensor,
        goal_idx: int,
        x0: Optional[torch.Tensor] = None,
        cluster_idx: Optional[torch.Tensor] = None,
        process_idx: Optional[torch.Tensor] = None,
        goal_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Reset environment to initial state.
        
        Parameters
        ----------
        z0 : torch.Tensor
            Initial latent state of shape (n_latent,).
        goal_idx : int
            Index of target goal/lineage.
        x0 : torch.Tensor, optional
            Initial gene expression for fixed_x mode.
        cluster_idx : torch.Tensor, optional
            Cluster index (scalar tensor or int).
        process_idx : torch.Tensor, optional
            Process index (scalar tensor or int).
        
        Returns
        -------
        obs : torch.Tensor
            Observation (concatenated state).
        info : dict
            Additional info.
        """
        self.z = z0.to(self.adapter.device)
        self.goal_idx = goal_idx
        self.goal_state = goal_state.to(self.adapter.device) if goal_state is not None else None
        self.t = 0
        self.done = False
        self.step_norms = {
            "velocity_magnitude": [],
            "perturbation_magnitude": [],
            "state_change": [],
        }
        
        # Store per-episode cluster/process indices
        self.cluster_indices = cluster_idx
        self.process_indices = process_idx
        
        # Store goal state if provided (for sample mode), otherwise use centroid
        self.goal_state = None  # Will be set in reset if provided
        
        # Set fixed x if provided
        if x0 is not None and self.adapter.velocity_mode == "fixed_x":
            self.adapter.set_fixed_x(x0)
        
        obs = self._get_obs()
        info = {"distance": self._compute_distance().item()}
        return obs, info
    
    def _get_obs(self) -> torch.Tensor:
        """Get current observation."""
        # Goal embedding
        if self.use_learned_goal_emb:
            # For now, use one-hot (can be replaced with learned embedding)
            goal_emb = torch.zeros(self.goal_emb_dim, device=self.adapter.device)
            if self.goal_idx < self.goal_emb_dim:
                goal_emb[self.goal_idx] = 1.0
        else:
            goal_emb = torch.zeros(self.n_goals, device=self.adapter.device)
            goal_emb[self.goal_idx] = 1.0
        
        # Normalized time
        t_norm = torch.tensor(self.t / self.T_max, device=self.adapter.device)
        
        # Concatenate: [z, goal_emb, t_norm]
        obs = torch.cat([self.z, goal_emb, t_norm.unsqueeze(0)])
        return obs
    
    def _compute_distance(self) -> torch.Tensor:
        """Compute distance to target goal (centroid or sampled cell)."""
        if self.goal_state is not None:
            goal = self.goal_state  # (n_latent,)
        else:
            goal = self.centroids[self.goal_idx]  # (n_latent,)
        distance = torch.norm(self.z - goal, p=2)
        return distance
    
    def step(
        self,
        action: Tuple[int, float],
    ) -> Tuple[torch.Tensor, float, bool, Dict]:
        """
        Step environment.
        
        Parameters
        ----------
        action : tuple
            (a_t, Δ_t) where a_t ∈ {0,...,d} and Δ_t ∈ R.
        
        Returns
        -------
        obs : torch.Tensor
            Next observation.
        reward : float
            Reward for this step.
        done : bool
            Whether episode is done (terminal/absorbing on success).
        info : dict
            Additional info (distance, step norms, etc.).
        """
        if self.done:
            # Already terminal, return zero reward
            return self._get_obs(), 0.0, True, {"distance": self._compute_distance().item()}
        
        a_t, delta_t = action
        
        # Apply perturbation
        u = torch.zeros(self.n_latent, device=self.adapter.device)
        if a_t > 0 and a_t <= self.n_latent:
            u[a_t - 1] = delta_t  # a_t=1 maps to dim 0, etc.
        
        z_tilde = self.z + u
        
        # Compute velocity only if not deactivated
        if self.deactivate_velocity:
            v = torch.zeros(self.n_latent, device=self.adapter.device)
        else:
            v = self.adapter.velocity(
                z_tilde.unsqueeze(0),
                cluster_indices=self.cluster_indices,
                process_indices=self.process_indices,
            ).squeeze(0)  # (n_latent,)
            
            # Apply negative velocity if requested
            if self.use_negative_velocity:
                v = -v
        
        # Update state: z_next = z_tilde + dt * v (or just z_tilde if velocity is deactivated)
        z_old = self.z  # Save before update for state_change logging
        if self.deactivate_velocity:
            z_next = z_tilde  # Skip velocity update
        else:
            z_next = z_tilde + self.dt * v
        
        # Compute distances (linear, not squared)
        if self.goal_state is not None:
            goal_z = self.goal_state
        else:
            goal_z = self.centroids[self.goal_idx]
        d_t = self._compute_distance()  # L2 distance at t
        d1_t = torch.norm(self.z - goal_z, p=1) / np.sqrt(self.n_latent)
        
        self.z = z_next
        d_tp1 = self._compute_distance()  # L2 distance at t+1
        d1_tp1 = torch.norm(self.z - goal_z, p=1) / np.sqrt(self.n_latent)
        
        # Check success (terminal/absorbing) - use goal_z for consistency
        if self.goal_state is not None:
            goal_z_check = self.goal_state
        else:
            goal_z_check = self.centroids[self.goal_idx]
        success = torch.norm(z_next - goal_z_check, p=2) < self.eps_success
        if success:
            self.done = True
        
        # Check timeout
        self.t += 1
        timeout = self.t >= self.T_max
        
        # Compute reward using linear distance progress (instead of squared distance)
        progress = self.lambda_progress * ((d_t - d_tp1) + (d1_t - d1_tp1))
        state_cost = self.alpha_stay * d_tp1
        action_penalty = self.lambda_act if a_t != 0 else 0.0
        magnitude_penalty = self.lambda_mag * abs(delta_t)
        success_bonus = self.R_succ if success else 0.0
        
        # Off-manifold penalty
        off_manifold_penalty = 0.0
        nll = 0.0
        if self.lambda_off > 0.0 and self.gmm_scorer is not None:
            # Compute NLL for z_next (the resulting state after step)
            with torch.no_grad():
                nll_torch = self.gmm_scorer.nll_torch(self.z)  # Use current state (z_next)
                nll = nll_torch.item()
                off_manifold_penalty = self.lambda_off * nll
        
        reward = progress - state_cost - action_penalty - magnitude_penalty + success_bonus - off_manifold_penalty
        
        # Log step norms
        v_norm = torch.norm(v, p=2).item()
        u_norm = torch.norm(u, p=2).item()
        z_change_norm = torch.norm(z_next - z_old, p=2).item()
        
        self.step_norms["velocity_magnitude"].append(v_norm)
        self.step_norms["perturbation_magnitude"].append(u_norm)
        self.step_norms["state_change"].append(z_change_norm)
        
        # Build info dictionary
        info = {
            "distance": d_tp1.item(),
            "sqdist": d_tp1.item() ** 2,  # Squared distance for logging (computed from linear distance)
            "success": success,
            "timeout": timeout,
            "nll": nll,  # Store NLL for logging
            "velocity_norm": v_norm,
            "perturbation_norm": u_norm,
            "state_change_norm": z_change_norm,
        }
        
        done = self.done or timeout
        
        return self._get_obs(), reward, done, info


class VectorizedLatentVelocityEnv:
    """
    Vectorized environment for parallel rollouts.
    
    Handles batch of B environments in parallel.
    """
    
    def __init__(
        self,
        adapter: VelocityVAEAdapter,
        centroids: torch.Tensor,  # (n_goals, n_latent)
        goal_names: list,
        batch_size: int,
        dt: float = 0.1,
        T_max: int = 100,
        eps_success: float = 0.1,
        lambda_progress: float = 1.0,
        lambda_act: float = 0.01,
        lambda_mag: float = 0.1,
        R_succ: float = 10.0,
        alpha_stay: float = 0.0,
        goal_emb_dim: Optional[int] = None,
        cluster_indices: Optional[torch.Tensor] = None,
        process_indices: Optional[torch.Tensor] = None,
        use_negative_velocity: bool = False,
        deactivate_velocity: bool = False,
        gmm_path: Optional[str] = None,
        lambda_off: float = 0.0,
    ):
        self.adapter = adapter
        self.centroids = centroids.to(adapter.device)
        self.goal_names = goal_names
        self.n_goals = len(goal_names)
        self.n_latent = adapter.n_latent
        self.batch_size = batch_size
        self.dt = dt
        self.T_max = T_max
        self.eps_success = eps_success
        self.lambda_progress = lambda_progress
        self.lambda_act = lambda_act
        self.lambda_mag = lambda_mag
        self.R_succ = R_succ
        self.alpha_stay = alpha_stay
        self.use_negative_velocity = use_negative_velocity
        self.deactivate_velocity = deactivate_velocity
        self.lambda_off = lambda_off
        
        # Initialize GMM scorer if lambda_off > 0
        self.gmm_scorer = None
        if lambda_off > 0.0:
            if gmm_path is None:
                raise ValueError("gmm_path must be provided when lambda_off > 0")
            self.gmm_scorer = SklearnGMMScorer(gmm_path)
        
        self.goal_emb_dim = goal_emb_dim if goal_emb_dim is not None else self.n_goals
        self.use_learned_goal_emb = goal_emb_dim is not None
        
        self.cluster_indices = cluster_indices
        self.process_indices = process_indices
        
        # Batch state: (B, ...)
        self.z: Optional[torch.Tensor] = None  # (B, n_latent)
        self.goal_idx: Optional[torch.Tensor] = None  # (B,)
        self.t: Optional[torch.Tensor] = None  # (B,)
        self.done: Optional[torch.Tensor] = None  # (B,)
        
        # Logging
        self.step_norms: Dict[str, list] = {
            "velocity_magnitude": [],
            "perturbation_magnitude": [],
            "state_change": [],
        }
    
    def reset(
        self,
        z0: torch.Tensor,  # (B, n_latent)
        goal_idx: torch.Tensor,  # (B,)
        x0: Optional[torch.Tensor] = None,  # (B, 2*n_genes) or (2*n_genes,)
        cluster_idx: Optional[torch.Tensor] = None,  # (B,)
        process_idx: Optional[torch.Tensor] = None,  # (B,)
        goal_states: Optional[torch.Tensor] = None,  # (B, n_latent) - optional goal states for sample mode
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Reset batch of environments.
        
        Parameters
        ----------
        z0 : torch.Tensor
            Initial latent states of shape (batch_size, n_latent).
        goal_idx : torch.Tensor
            Goal indices of shape (batch_size,).
        x0 : torch.Tensor, optional
            Initial gene expression for fixed_x mode.
        cluster_idx : torch.Tensor, optional
            Cluster indices of shape (batch_size,).
        process_idx : torch.Tensor, optional
            Process indices of shape (batch_size,).
        
        Returns
        -------
        obs : torch.Tensor
            Observations of shape (batch_size, obs_dim).
        info : dict
            Additional info.
        """
        self.z = z0.to(self.adapter.device)
        self.goal_idx = goal_idx.to(self.adapter.device).long()
        # Store goal states if provided (for sample mode), otherwise use centroids
        self.goal_states = goal_states.to(self.adapter.device) if goal_states is not None else None
        self.t = torch.zeros(self.batch_size, device=self.adapter.device, dtype=torch.long)
        self.done = torch.zeros(self.batch_size, device=self.adapter.device, dtype=torch.bool)
        
        # Store per-batch cluster/process indices
        self.cluster_idx = cluster_idx.to(self.adapter.device) if cluster_idx is not None else None
        self.process_idx = process_idx.to(self.adapter.device) if process_idx is not None else None
        
        self.step_norms = {
            "velocity_magnitude": [],
            "perturbation_magnitude": [],
            "state_change": [],
        }
        
        # Set fixed x if provided (for fixed_x mode)
        if x0 is not None and self.adapter.velocity_mode == "fixed_x":
            if x0.dim() == 1:
                # Single x for all, expand to batch
                x0 = x0.unsqueeze(0).expand(self.batch_size, -1)
            # Ensure x0 is on adapter device and pass full batch
            x0 = x0.to(self.adapter.device)
            self.adapter.set_fixed_x(x0)
        
        obs = self._get_obs()
        distances = self._compute_distances()
        info = {"distances": distances.cpu().numpy()}
        return obs, info
    
    def _get_obs(self) -> torch.Tensor:
        """Get current observations for batch."""
        B = self.batch_size
        
        # Goal embeddings
        if self.use_learned_goal_emb:
            goal_emb = torch.zeros(B, self.goal_emb_dim, device=self.adapter.device)
            goal_emb.scatter_(1, self.goal_idx.unsqueeze(1), 1.0)
        else:
            goal_emb = torch.zeros(B, self.n_goals, device=self.adapter.device)
            goal_emb.scatter_(1, self.goal_idx.unsqueeze(1), 1.0)
        
        # Normalized time
        t_norm = (self.t.float() / self.T_max).unsqueeze(1)  # (B, 1)
        
        # Concatenate: [z, goal_emb, t_norm]
        obs = torch.cat([self.z, goal_emb, t_norm], dim=1)  # (B, n_latent + goal_emb_dim + 1)
        return obs
    
    def _compute_distances(self) -> torch.Tensor:
        """Compute distances to target goals (centroids or sampled cells) for batch."""
        # Get goal states for each batch element: (B, n_latent)
        if self.goal_states is not None:
            goals_batch = self.goal_states  # (B, n_latent)
        else:
            goals_batch = self.centroids[self.goal_idx]  # (B, n_latent)
        # Compute L2 distances: (B,)
        distances = torch.norm(self.z - goals_batch, p=2, dim=1)
        return distances
    
    def step(
        self,
        actions: Tuple[torch.Tensor, torch.Tensor],  # (a_t: (B,), delta_t: (B,))
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Step batch of environments.
        
        Parameters
        ----------
        actions : tuple
            (a_t, delta_t) where a_t is (B,) and delta_t is (B,).
        
        Returns
        -------
        obs : torch.Tensor
            Next observations of shape (batch_size, obs_dim).
        rewards : torch.Tensor
            Rewards of shape (batch_size,).
        done : torch.Tensor
            Done flags of shape (batch_size,).
        info : dict
            Additional info.
        """
        a_t, delta_t = actions
        a_t = a_t.to(self.adapter.device).long()
        delta_t = delta_t.to(self.adapter.device).float()
        B = self.batch_size
        
        # Mask out already-done episodes
        active_mask = ~self.done
        
        # Apply perturbations
        u = torch.zeros(B, self.n_latent, device=self.adapter.device)
        # Create indices for scatter: (B,)
        dim_indices = (a_t - 1).clamp(min=0, max=self.n_latent - 1)
        # Only apply perturbation if a_t > 0
        perturb_mask = (a_t > 0) & active_mask
        u[perturb_mask, dim_indices[perturb_mask]] = delta_t[perturb_mask]
        
        z_tilde = self.z + u
        
        # Compute velocities (batched) only if not deactivated
        if self.deactivate_velocity:
            v = torch.zeros(B, self.n_latent, device=self.adapter.device)
        else:
            v = self.adapter.velocity(
                z_tilde,
                cluster_indices=self.cluster_idx,
                process_indices=self.process_idx,
            )  # (B, n_latent)
            
            # Apply negative velocity if requested
            if self.use_negative_velocity:
                v = -v
        
        # Update states: z_next = z_tilde + dt * v (or just z_tilde if velocity is deactivated)
        z_old = self.z
        if self.deactivate_velocity:
            z_next = z_tilde  # Skip velocity update
        else:
            z_next = z_tilde + self.dt * v
        
        # Compute distances (linear, not squared) before update
        if self.goal_states is not None:
            goal_z_batch = self.goal_states  # (B, n_latent)
        else:
            goal_z_batch = self.centroids[self.goal_idx]  # (B, n_latent)
        d_t = self._compute_distances()  # (B,) L2 distance at t
        d1_t = torch.norm(self.z - goal_z_batch, p=1, dim=1) / np.sqrt(self.n_latent)
        
        # Check success (terminal/absorbing) - goal_z_batch already set above
        success = (torch.norm(z_next - goal_z_batch, p=2, dim=1) < self.eps_success) & active_mask
        self.done = self.done | success
        
        # Only update active envs (absorbing terminal states)
        self.z = torch.where(active_mask.unsqueeze(1), z_next, self.z)
        self.t = torch.where(active_mask, self.t + 1, self.t)
        
        d_tp1 = self._compute_distances()  # (B,) L2 distance at t+1
        d1_tp1 = torch.norm(self.z - goal_z_batch, p=1, dim=1) / np.sqrt(self.n_latent)
        
        # Check timeout
        timeout = self.t >= self.T_max
        
        # Compute rewards using linear distance progress (instead of squared distance)
        progress = self.lambda_progress * ((d_t - d_tp1) + (d1_t - d1_tp1)) * active_mask.float()
        state_cost = self.alpha_stay * d_tp1 * active_mask.float()  # Use linear distance for state cost
        action_penalty = self.lambda_act * (a_t != 0).float() * active_mask.float()
        magnitude_penalty = self.lambda_mag * delta_t.abs() * active_mask.float()
        success_bonus = self.R_succ * success.float()
        
        # Off-manifold penalty
        off_manifold_penalty = torch.zeros(B, device=self.adapter.device, dtype=torch.float32)
        nll_batch = torch.zeros(B, device=self.adapter.device, dtype=torch.float32)
        if self.lambda_off > 0.0 and self.gmm_scorer is not None:
            # Compute NLL for z_next (the resulting state after step)
            with torch.no_grad():
                nll_torch = self.gmm_scorer.nll_torch(self.z)  # Use current state (z_next)
                nll_batch = nll_torch
                off_manifold_penalty = self.lambda_off * nll_torch
        
        rewards = progress - state_cost - action_penalty - magnitude_penalty + success_bonus - off_manifold_penalty
        
        # Log step norms (average over active episodes)
        v_norm = torch.norm(v, p=2, dim=1)  # (B,)
        u_norm = torch.norm(u, p=2, dim=1)  # (B,)
        # State change is z_next - z_old (before update)
        z_change = z_next - z_old
        z_change_norm = torch.norm(z_change, p=2, dim=1)  # (B,)
        
        active_v_norm = v_norm[active_mask].mean().item() if active_mask.any() else 0.0
        active_u_norm = u_norm[active_mask].mean().item() if active_mask.any() else 0.0
        active_z_norm = z_change_norm[active_mask].mean().item() if active_mask.any() else 0.0
        
        self.step_norms["velocity_magnitude"].append(active_v_norm)
        self.step_norms["perturbation_magnitude"].append(active_u_norm)
        self.step_norms["state_change"].append(active_z_norm)
        
        done = self.done | timeout
        
        info = {
            "distances": d_tp1.cpu().numpy(),
            "sqdist": (d_tp1 ** 2).cpu().numpy(),  # Squared distances for logging (computed from linear distance)
            "success": success.cpu().numpy(),
            "timeout": timeout.cpu().numpy(),
            "nll": nll_batch.cpu().numpy(),  # Store as numpy for logging
            "velocity_norm": active_v_norm,
            "perturbation_norm": active_u_norm,
            "state_change_norm": active_z_norm,
        }
        
        return self._get_obs(), rewards, done, info

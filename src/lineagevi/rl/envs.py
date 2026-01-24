"""RL environment for latent velocity dynamics."""

from typing import Optional, Tuple, Dict, Literal
import torch
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
    Termination: t >= T_max; optionally terminate on success if terminate_on_success=True.
    
    Success reward is awarded at most once per episode, even if the agent remains
    within the success radius for subsequent steps.
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
        perturb_clip: Optional[float] = None,
        cluster_indices: Optional[torch.Tensor] = None,
        process_indices: Optional[torch.Tensor] = None,
        use_negative_velocity: bool = False,
        deactivate_velocity: bool = False,
        terminate_on_success: bool = False,
        milestone_rewards: bool = False,
        reward_mode: Literal["plain", "scaled", "milestone", "multi_milestone"] = "plain",
        progress_weight_p: float = 0.0,
        progress_weight_c: float = 0.1,
        milestone_decay_factor: Optional[float] = None,
        success_reward_bonus_pct: float = 0.0,
        success_reward_bonus_w: float = 0.0,
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
        self.base_R_succ = R_succ
        self.alpha_stay = alpha_stay
        self.perturb_clip = perturb_clip
        self.use_negative_velocity = use_negative_velocity
        self.deactivate_velocity = deactivate_velocity
        self.terminate_on_success = terminate_on_success
        self.reward_mode = reward_mode
        self.milestone_rewards = milestone_rewards or (reward_mode == "multi_milestone")
        self.progress_weight_p = progress_weight_p
        self.progress_weight_c = progress_weight_c
        self.milestone_decay_factor = milestone_decay_factor
        self.success_reward_bonus_pct = success_reward_bonus_pct
        self.success_reward_bonus_w = success_reward_bonus_w
        self.lambda_off = lambda_off
        
        if self.milestone_rewards and self.terminate_on_success:
            raise ValueError("milestone_rewards requires terminate_on_success=False")
        if self.success_reward_bonus_pct > 0.0 and self.success_reward_bonus_w > 0.0:
            raise ValueError("success_reward_bonus_pct and success_reward_bonus_w are mutually exclusive")
        
        if self.milestone_rewards and self.terminate_on_success:
            raise ValueError("milestone_rewards requires terminate_on_success=False")
        if self.reward_mode == "multi_milestone":
            if self.milestone_decay_factor is None or not (0.0 < self.milestone_decay_factor < 1.0):
                raise ValueError("milestone_decay_factor must be in (0, 1) when reward_mode='multi_milestone'")
        
        # Initialize GMM scorer if lambda_off > 0
        self.gmm_scorer = None
        if lambda_off > 0.0:
            if gmm_path is None:
                raise ValueError("gmm_path must be provided when lambda_off > 0")
            self.gmm_scorer = SklearnGMMScorer(gmm_path)
        
        # Goal encoding: difference vector (z_goal - z_t)
        
        # Cluster and process indices (fixed for episode)
        self.cluster_indices = cluster_indices
        self.process_indices = process_indices
        
        # Episode state
        self.z: Optional[torch.Tensor] = None
        self.goal_idx: Optional[int] = None
        self.t: int = 0
        self.done: bool = False
        self.success_awarded: bool = False
        self.milestone_level: int = 0
        self.d0: Optional[float] = None
        
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
        self.success_awarded = False
        self.milestone_level = 0
        self.d0 = None
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
        initial_distance = self._compute_distance().item()
        self.d0 = initial_distance
        info = {"distance": initial_distance}
        return obs, info
    
    def _get_obs(self) -> torch.Tensor:
        """Get current observation."""
        # Goal embedding: difference vector (z_goal - z_t)
        if self.goal_state is not None:
            goal = self.goal_state  # (n_latent,)
        else:
            goal = self.centroids[self.goal_idx]  # (n_latent,)
        goal_diff = goal - self.z  # (n_latent,)
        goal_emb = goal_diff
        
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
        if self.perturb_clip is not None:
            delta_t = float(np.clip(delta_t, -self.perturb_clip, self.perturb_clip))
        
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
        
        self.z = z_next
        d_tp1 = self._compute_distance()  # L2 distance at t+1
        
        # Check success (may or may not terminate) - use goal_z for consistency
        if self.goal_state is not None:
            goal_z_check = self.goal_state
        else:
            goal_z_check = self.centroids[self.goal_idx]
        distance_next = torch.norm(z_next - goal_z_check, p=2)
        success = distance_next < self.eps_success
        if self.milestone_rewards:
            prev_level = self.milestone_level
            if self.milestone_decay_factor is None or not (0.0 < self.milestone_decay_factor < 1.0):
                raise ValueError("milestone_decay_factor must be in (0, 1) when milestone_rewards is enabled")
            ratio = max(distance_next.item() / float(self.eps_success), 1e-12)
            if ratio < 1.0:
                k = int(np.floor(np.log(ratio) / np.log(self.milestone_decay_factor)))
                milestone_level = k + 1
            else:
                milestone_level = 0
            n_new = max(0, milestone_level - prev_level)
            success_award = n_new > 0
            if success_award:
                self.milestone_level = milestone_level
        else:
            success_award = success and (not self.success_awarded)
        if self.terminate_on_success and success:
            self.done = True
        if success:
            self.success_awarded = True
        
        # Check timeout
        self.t += 1
        timeout = self.t >= self.T_max
        
        # Compute reward using L2-only distance progress
        progress_raw = (d_t - d_tp1)
        if self.reward_mode == "scaled":
            denom = float(self.d0) if self.d0 is not None else float(d_t.item())
            d_tilde = float(d_t.item()) / (denom + 1e-8)
            weight = (1.0 / (d_tilde + self.progress_weight_c) ** self.progress_weight_p)
            progress = self.lambda_progress * (progress_raw / (denom + 1e-8)) * weight
        else:
            progress = self.lambda_progress * progress_raw
        d0 = float(self.d0) if self.d0 is not None else float(d_t.item())
        state_cost = self.alpha_stay * (d_t / (d0 + 1e-8))
        action_penalty = self.lambda_act if a_t != 0 else 0.0
        magnitude_penalty = self.lambda_mag * abs(delta_t)
        if self.milestone_rewards:
            n_new_levels = max(0, milestone_level - prev_level)
            k_start = prev_level + 1
            k_end = prev_level + n_new_levels
            sum_k = (k_start + k_end) * n_new_levels / 2.0
            if self.success_reward_bonus_pct > 0.0:
                success_bonus = self.base_R_succ * (n_new_levels + (self.success_reward_bonus_pct * sum_k))
            elif self.success_reward_bonus_w > 0.0:
                success_bonus = (self.base_R_succ * n_new_levels) + (self.success_reward_bonus_w * sum_k)
            else:
                success_bonus = self.base_R_succ * n_new_levels
        else:
            success_bonus = self.R_succ if success_award else 0.0
        
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
            "success_awarded": success_award,
            "timeout": timeout,
            "nll": nll,  # Store NLL for logging
            "velocity_norm": v_norm,
            "perturbation_norm": u_norm,
            "state_change_norm": z_change_norm,
            "progress": progress.item(),
            "action_penalty": float(action_penalty),
            "magnitude_penalty": float(magnitude_penalty),
            "success_bonus": float(success_bonus),
            "off_manifold_penalty": float(off_manifold_penalty),
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
        perturb_clip: Optional[float] = None,
        cluster_indices: Optional[torch.Tensor] = None,
        process_indices: Optional[torch.Tensor] = None,
        use_negative_velocity: bool = False,
        deactivate_velocity: bool = False,
        terminate_on_success: bool = False,
        milestone_rewards: bool = False,
        reward_mode: Literal["plain", "scaled", "milestone", "multi_milestone"] = "plain",
        progress_weight_p: float = 0.0,
        progress_weight_c: float = 0.1,
        milestone_decay_factor: Optional[float] = None,
        success_reward_bonus_pct: float = 0.0,
        success_reward_bonus_w: float = 0.0,
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
        self.base_R_succ = R_succ
        self.alpha_stay = alpha_stay
        self.perturb_clip = perturb_clip
        self.use_negative_velocity = use_negative_velocity
        self.deactivate_velocity = deactivate_velocity
        self.terminate_on_success = terminate_on_success
        self.reward_mode = reward_mode
        self.milestone_rewards = milestone_rewards or (reward_mode == "multi_milestone")
        self.progress_weight_p = progress_weight_p
        self.progress_weight_c = progress_weight_c
        self.milestone_decay_factor = milestone_decay_factor
        self.success_reward_bonus_pct = success_reward_bonus_pct
        self.success_reward_bonus_w = success_reward_bonus_w
        self.lambda_off = lambda_off
        
        if self.milestone_rewards and self.terminate_on_success:
            raise ValueError("milestone_rewards requires terminate_on_success=False")
        if self.success_reward_bonus_pct > 0.0 and self.success_reward_bonus_w > 0.0:
            raise ValueError("success_reward_bonus_pct and success_reward_bonus_w are mutually exclusive")
        
        # Initialize GMM scorer if lambda_off > 0
        self.gmm_scorer = None
        if lambda_off > 0.0:
            if gmm_path is None:
                raise ValueError("gmm_path must be provided when lambda_off > 0")
            self.gmm_scorer = SklearnGMMScorer(gmm_path)
        
        # Goal encoding: difference vector (z_goal - z_t)
        
        self.cluster_indices = cluster_indices
        self.process_indices = process_indices
        
        # Batch state: (B, ...)
        self.z: Optional[torch.Tensor] = None  # (B, n_latent)
        self.goal_idx: Optional[torch.Tensor] = None  # (B,)
        self.t: Optional[torch.Tensor] = None  # (B,)
        self.done: Optional[torch.Tensor] = None  # (B,)
        self.success_awarded: Optional[torch.Tensor] = None  # (B,)
        self.milestone_level: Optional[torch.Tensor] = None  # (B,)
        self.d0: Optional[torch.Tensor] = None  # (B,)
        
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
        self.success_awarded = torch.zeros(self.batch_size, device=self.adapter.device, dtype=torch.bool)
        self.milestone_level = torch.zeros(self.batch_size, device=self.adapter.device, dtype=torch.long)
        self.d0 = None
        
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
        self.d0 = distances.detach()
        info = {"distances": distances.cpu().numpy()}
        return obs, info
    
    def _get_obs(self) -> torch.Tensor:
        """Get current observations for batch."""
        B = self.batch_size
        
        # Goal embeddings: difference vector (z_goal - z_t)
        if self.goal_states is not None:
            goals_batch = self.goal_states  # (B, n_latent)
        else:
            goals_batch = self.centroids[self.goal_idx]  # (B, n_latent)
        goal_diff = goals_batch - self.z  # (B, n_latent)
        goal_emb = goal_diff
        
        # Normalized time
        t_norm = (self.t.float() / self.T_max).unsqueeze(1)  # (B, 1)
        
        # Concatenate: [z, goal_emb, t_norm]
        obs = torch.cat([self.z, goal_emb, t_norm], dim=1)  # (B, 2*n_latent + 1)
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
        if self.perturb_clip is not None:
            delta_t = torch.clamp(delta_t, -self.perturb_clip, self.perturb_clip)
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
        
        # Check success (may or may not terminate) - goal_z_batch already set above
        eps_success = self.eps_success
        if torch.is_tensor(eps_success):
            eps_success = eps_success.to(self.adapter.device)
        else:
            eps_success = torch.full((B,), float(eps_success), device=self.adapter.device)
        distance_next = torch.norm(z_next - goal_z_batch, p=2, dim=1)
        success = (distance_next < eps_success) & active_mask
        if self.milestone_rewards:
            if self.milestone_decay_factor is None or not (0.0 < self.milestone_decay_factor < 1.0):
                raise ValueError("milestone_decay_factor must be in (0, 1) when milestone_rewards is enabled")
            ratio = torch.clamp(distance_next / eps_success, min=1e-12)
            log_decay = float(np.log(self.milestone_decay_factor))
            k = torch.floor(torch.log(ratio) / log_decay)
            milestone_level = torch.where(ratio < 1.0, k + 1, torch.zeros_like(k))
            milestone_level = milestone_level.to(torch.long)
            current_level = self.milestone_level
            n_new = torch.clamp(milestone_level - current_level, min=0)
            success_award = (n_new > 0) & active_mask
            updated_level = torch.maximum(current_level, milestone_level)
            self.milestone_level = torch.where(active_mask, updated_level, current_level)
        else:
            success_award = success & (~self.success_awarded)
        if self.terminate_on_success:
            self.done = self.done | success
        self.success_awarded = self.success_awarded | success
        
        # Only update active envs (absorbing terminal states)
        self.z = torch.where(active_mask.unsqueeze(1), z_next, self.z)
        self.t = torch.where(active_mask, self.t + 1, self.t)
        
        d_tp1 = self._compute_distances()  # (B,) L2 distance at t+1
        
        # Check timeout
        timeout = self.t >= self.T_max
        
        # Compute rewards using L2-only distance progress
        progress_raw = (d_t - d_tp1) * active_mask.float()
        if self.reward_mode == "scaled":
            denom = self.d0 if self.d0 is not None else d_t
            d_tilde = d_t / (denom + 1e-8)
            weight = (1.0 / (d_tilde + self.progress_weight_c) ** self.progress_weight_p)
            progress = self.lambda_progress * (progress_raw / (denom + 1e-8)) * weight
        else:
            progress = self.lambda_progress * progress_raw
        denom = self.d0 if self.d0 is not None else d_t
        state_cost = self.alpha_stay * (d_t / (denom + 1e-8)) * active_mask.float()  # Use linear distance for state cost
        action_penalty = self.lambda_act * (a_t != 0).float() * active_mask.float()
        magnitude_penalty = self.lambda_mag * delta_t.abs() * active_mask.float()
        if self.milestone_rewards:
            n_new_levels = torch.clamp(milestone_level - current_level, min=0)
            k_start = current_level + 1
            k_end = current_level + n_new_levels
            sum_k = (k_start + k_end).float() * n_new_levels.float() / 2.0
            if self.success_reward_bonus_pct > 0.0:
                success_bonus = self.base_R_succ * (n_new_levels.float() + (self.success_reward_bonus_pct * sum_k))
            elif self.success_reward_bonus_w > 0.0:
                success_bonus = (self.base_R_succ * n_new_levels.float()) + (self.success_reward_bonus_w * sum_k)
            else:
                success_bonus = self.base_R_succ * n_new_levels.float()
            success_bonus = success_bonus * active_mask.float()
        else:
            success_bonus = self.R_succ * success_award.float()
        
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
            "success_awarded": success_award.cpu().numpy(),
            "timeout": timeout.cpu().numpy(),
            "nll": nll_batch.cpu().numpy(),  # Store as numpy for logging
            "velocity_norm": active_v_norm,
            "perturbation_norm": active_u_norm,
            "state_change_norm": active_z_norm,
            "progress": progress.detach().cpu().numpy(),
            "action_penalty": action_penalty.detach().cpu().numpy(),
            "magnitude_penalty": magnitude_penalty.detach().cpu().numpy(),
            "success_bonus": success_bonus.detach().cpu().numpy(),
            "off_manifold_penalty": off_manifold_penalty.detach().cpu().numpy(),
        }
        
        return self._get_obs(), rewards, done, info

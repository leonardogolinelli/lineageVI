"""Actor-critic policy network for sparse latent perturbations."""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


class ActorCriticPolicy(nn.Module):
    """
    Actor-critic policy with hybrid discrete/continuous actions.
    
    Discrete action: a_t ∈ {0, 1, ..., d} where 0 = no-op, k = perturb dim k
    Continuous action: Δ_t ∈ R (magnitude, only used if a_t != 0)
    
    Architecture:
    - Shared MLP trunk: [z, proj(z_goal - z_t), t] → hidden
    - Categorical head: hidden → logits[d+1]
    - Action embedding: action → action_emb
    - Magnitude head: [hidden, action_emb] → (μ, logσ) (conditioned on state + action)
    - Value head: hidden → V(s)
    
    The magnitude distribution is now conditioned on both the state and the chosen action,
    allowing the policy to learn different magnitude distributions for different actions.
    """
    
    def __init__(
        self,
        obs_dim: int,  # 2*n_latent + 1 (z, goal_diff, time)
        n_latent: int,
        goal_cond_dim: int = 32,
        use_t_norm: bool = False,
        allow_noop_action: bool = True,
        hidden_sizes: list = [128, 128],
        actor_hidden_sizes: Optional[list] = None,
        critic_hidden_sizes: Optional[list] = None,
        separate_trunks: bool = False,
        activation: str = "relu",
        delta_clip: Optional[float] = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_latent = n_latent
        self.delta_clip = delta_clip
        self.goal_cond_dim = goal_cond_dim
        self.allow_noop_action = allow_noop_action
        self.use_t_norm = use_t_norm
        
        # Goal conditioning projection for (z_goal - z_t)
        self.goal_proj = nn.Linear(n_latent, goal_cond_dim)
        
        def build_trunk(sizes: list) -> nn.Sequential:
            layers = []
            input_dim = n_latent + goal_cond_dim + (1 if use_t_norm else 0)
            for hidden_size in sizes:
                layers.append(nn.Linear(input_dim, hidden_size))
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                else:
                    raise ValueError(f"Unknown activation: {activation}")
                input_dim = hidden_size
            return nn.Sequential(*layers)
        
        self.use_separate_trunks = separate_trunks or actor_hidden_sizes is not None or critic_hidden_sizes is not None
        if self.use_separate_trunks:
            actor_sizes = actor_hidden_sizes if actor_hidden_sizes is not None else hidden_sizes
            critic_sizes = critic_hidden_sizes if critic_hidden_sizes is not None else hidden_sizes
            self.actor_trunk = build_trunk(actor_sizes)
            self.critic_trunk = build_trunk(critic_sizes)
            actor_trunk_output_dim = actor_sizes[-1]
            critic_trunk_output_dim = critic_sizes[-1]
        else:
            self.shared_trunk = build_trunk(hidden_sizes)
            trunk_output_dim = hidden_sizes[-1]
        
        # Categorical head (action selection)
        self.action_head = nn.Linear(
            actor_trunk_output_dim if self.use_separate_trunks else trunk_output_dim,
            n_latent + 1,
        )  # +1 for no-op
        
        # Action embedding for magnitude conditioning
        self.action_embed_dim = 16
        self.action_embedding = nn.Embedding(n_latent + 1, self.action_embed_dim)  # +1 for no-op
        
        # Magnitude head (conditioned on state + action)
        # Input: hidden_state + action_embedding
        magnitude_input_dim = (
            actor_trunk_output_dim if self.use_separate_trunks else trunk_output_dim
        ) + self.action_embed_dim
        self.magnitude_mu_head = nn.Linear(magnitude_input_dim, 1)  # Single μ output
        self.magnitude_logstd_head = nn.Linear(magnitude_input_dim, 1)  # Single logσ output
        
        # Value head
        self.value_head = nn.Linear(
            critic_trunk_output_dim if self.use_separate_trunks else trunk_output_dim, 1
        )
        
        # Log-std clamping constants
        self.LOG_STD_MIN = -5.0
        self.LOG_STD_MAX = 2.0
    
    def _get_magnitude_params(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get clamped magnitude parameters conditioned on state and action.
        
        Parameters
        ----------
        obs : torch.Tensor
            Observations of shape (batch_size, obs_dim).
        action : torch.Tensor
            Discrete actions of shape (batch_size,).
        
        Returns
        -------
        magnitude_mu : torch.Tensor
            Magnitude means of shape (batch_size,).
        magnitude_std : torch.Tensor
            Magnitude stddevs (from clamped log_std) of shape (batch_size,).
        """
        if self.use_separate_trunks:
            h = self.actor_trunk(self._encode_obs(obs))
        else:
            h = self.shared_trunk(self._encode_obs(obs))
        
        # Embed action
        action_emb = self.action_embedding(action)  # (B, action_embed_dim)
        
        # Concatenate hidden state and action embedding
        magnitude_input = torch.cat([h, action_emb], dim=1)  # (B, trunk_output_dim + action_embed_dim)
        
        # Get μ and logσ for the chosen action
        magnitude_mu = self.magnitude_mu_head(magnitude_input).squeeze(-1)  # (B,)
        magnitude_logstd = self.magnitude_logstd_head(magnitude_input).squeeze(-1)  # (B,)
        
        # Clamp log_std to prevent sigma collapse and KL blowups
        magnitude_logstd_clamped = torch.clamp(magnitude_logstd, self.LOG_STD_MIN, self.LOG_STD_MAX)
        magnitude_std = torch.exp(magnitude_logstd_clamped)
        return magnitude_mu, magnitude_std
    
    def forward(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through policy network.
        
        Note: Magnitude parameters are now conditioned on action, so this method
        returns placeholder values for magnitude_mu and magnitude_logstd.
        Use _get_magnitude_params(obs, action) to get actual magnitude params.
        
        Parameters
        ----------
        obs : torch.Tensor
            Observations of shape (batch_size, obs_dim).
        
        Returns
        -------
        action_logits : torch.Tensor
            Categorical logits of shape (batch_size, n_latent + 1).
        magnitude_mu : torch.Tensor
            Placeholder (zeros) of shape (batch_size, n_latent). 
            Use _get_magnitude_params(obs, action) for actual values.
        magnitude_logstd : torch.Tensor
            Placeholder (zeros) of shape (batch_size, n_latent).
            Use _get_magnitude_params(obs, action) for actual values.
        value : torch.Tensor
            Value estimates of shape (batch_size, 1).
        """
        if self.use_separate_trunks:
            h_actor = self.actor_trunk(self._encode_obs(obs))
            h_critic = self.critic_trunk(self._encode_obs(obs))
        else:
            h_actor = self.shared_trunk(self._encode_obs(obs))
            h_critic = h_actor
        
        action_logits = self.action_head(h_actor)  # (B, n_latent + 1)
        action_logits = self._apply_noop_mask(action_logits)
        value = self.value_head(h_critic)  # (B, 1)
        
        # Return placeholders for backward compatibility
        # Magnitude params are now action-conditioned, so we can't return them here
        batch_size = obs.shape[0]
        magnitude_mu = torch.zeros(batch_size, self.n_latent, device=obs.device)
        magnitude_logstd = torch.zeros(batch_size, self.n_latent, device=obs.device)
        
        return action_logits, magnitude_mu, magnitude_logstd, value
    
    def sample(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        n_actions: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Parameters
        ----------
        obs : torch.Tensor
            Observations of shape (batch_size, obs_dim).
        deterministic : bool, default False
            If True, take mode (argmax) for discrete action.
        
        Returns
        -------
        action : torch.Tensor
            Discrete actions of shape (batch_size,).
        delta : torch.Tensor
            Continuous magnitudes of shape (batch_size,).
        raw_delta : torch.Tensor
            Same as delta (kept for backward compatibility).
        log_prob : torch.Tensor
            Log probabilities of shape (batch_size,).
        """
        action_logits, _, _, _ = self.forward(obs)
        
        # Sample discrete action(s)
        if deterministic:
            action = torch.argmax(action_logits, dim=1)  # (B,)
            if n_actions > 1:
                action = action.unsqueeze(1).repeat(1, n_actions)  # (B, K)
        else:
            action_dist = dist.Categorical(logits=action_logits)
            if n_actions == 1:
                action = action_dist.sample()  # (B,)
            else:
                action = action_dist.sample((n_actions,)).transpose(0, 1)  # (B, K)
        
        # Sample magnitude conditioned on state and chosen action
        # If action = 0, magnitude is 0 (no-op)
        batch_size = obs.shape[0]
        if action.dim() == 1:
            action_mask = (action > 0).float()  # (B,)
            # Get magnitude params for chosen action (conditioned on state + action)
            magnitude_mu, magnitude_std = self._get_magnitude_params(obs, action)  # (B,)
            # Sample magnitude directly from normal distribution (only if action > 0)
            magnitude_dist = dist.Normal(magnitude_mu, magnitude_std)
            delta = magnitude_dist.sample()  # (B,)
            delta = delta * action_mask
        else:
            _, k_actions = action.shape
            action_flat = action.reshape(-1)  # (B*K,)
            obs_exp = obs.unsqueeze(1).repeat(1, k_actions, 1).reshape(-1, obs.shape[1])  # (B*K, obs_dim)
            action_mask = (action_flat > 0).float()  # (B*K,)
            magnitude_mu, magnitude_std = self._get_magnitude_params(obs_exp, action_flat)  # (B*K,)
            magnitude_dist = dist.Normal(magnitude_mu, magnitude_std)
            delta_flat = magnitude_dist.sample()  # (B*K,)
            delta_flat = delta_flat * action_mask
            delta = delta_flat.reshape(batch_size, k_actions)  # (B, K)
        
        # Clip magnitude if configured
        if self.delta_clip is not None:
            delta = torch.clamp(delta, -self.delta_clip, self.delta_clip)
        
        # Compute log probability
        log_prob = self.log_prob(obs, action, delta)
        
        return action, delta, delta, log_prob  # raw_delta same as delta now
    
    def log_prob(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability of action.
        
        Parameters
        ----------
        obs : torch.Tensor
            Observations of shape (batch_size, obs_dim).
        action : torch.Tensor
            Discrete actions of shape (batch_size,).
        delta : torch.Tensor
            Continuous magnitudes of shape (batch_size,).
        
        Returns
        -------
        log_prob : torch.Tensor
            Log probabilities of shape (batch_size,).
        """
        action_logits, _, _, _ = self.forward(obs)
        
        # Categorical log prob
        action_dist = dist.Categorical(logits=action_logits)
        if action.dim() == 1:
            log_prob_cat = action_dist.log_prob(action)  # (B,)
            action_mask = (action > 0).float()  # (B,)
            magnitude_mu, magnitude_std = self._get_magnitude_params(obs, action)  # (B,)
            magnitude_dist = dist.Normal(magnitude_mu, magnitude_std)
            log_prob_mag = magnitude_dist.log_prob(delta)  # (B,)
            log_prob = log_prob_cat + action_mask * log_prob_mag
        else:
            batch_size, k_actions = action.shape
            log_prob_cat = action_dist.log_prob(action)  # (B, K)
            action_flat = action.reshape(-1)  # (B*K,)
            obs_exp = obs.unsqueeze(1).repeat(1, k_actions, 1).reshape(-1, obs.shape[1])  # (B*K, obs_dim)
            action_mask = (action_flat > 0).float()  # (B*K,)
            magnitude_mu, magnitude_std = self._get_magnitude_params(obs_exp, action_flat)  # (B*K,)
            magnitude_dist = dist.Normal(magnitude_mu, magnitude_std)
            delta_flat = delta.reshape(-1)  # (B*K,)
            log_prob_mag = magnitude_dist.log_prob(delta_flat)  # (B*K,)
            log_prob = log_prob_cat.sum(dim=1) + (action_mask * log_prob_mag).reshape(batch_size, k_actions).sum(dim=1)
        
        return log_prob
    
    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of policy distribution.
        
        Only includes categorical action entropy, not magnitude entropy.
        We want to diversify action selection, but magnitude should be
        determined optimally for each chosen action.
        
        Parameters
        ----------
        obs : torch.Tensor
            Observations of shape (batch_size, obs_dim).
        
        Returns
        -------
        entropy : torch.Tensor
            Entropy of shape (batch_size,).
        """
        action_logits, _, _, _ = self.forward(obs)
        
        # Categorical entropy only
        action_dist = dist.Categorical(logits=action_logits)
        entropy_cat = action_dist.entropy()  # (B,)
        
        return entropy_cat
    
    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate.
        
        Parameters
        ----------
        obs : torch.Tensor
            Observations of shape (batch_size, obs_dim).
        
        Returns
        -------
        value : torch.Tensor
            Value estimates of shape (batch_size, 1).
        """
        _, _, _, value = self.forward(obs)
        return value

    def _encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode observation into trunk input by projecting goal_diff.
        
        obs is [z_t, goal_diff, t_norm] (if enabled), with goal_diff = z_goal - z_t.
        """
        z = obs[:, : self.n_latent]
        goal_diff = obs[:, self.n_latent : 2 * self.n_latent]
        if self.use_t_norm:
            t_norm = obs[:, 2 * self.n_latent : 2 * self.n_latent + 1]
        else:
            t_norm = None
        goal_emb = self.goal_proj(goal_diff)
        if t_norm is None:
            return torch.cat([z, goal_emb], dim=1)
        return torch.cat([z, goal_emb, t_norm], dim=1)

    def _apply_noop_mask(self, action_logits: torch.Tensor) -> torch.Tensor:
        if self.allow_noop_action:
            return action_logits
        masked_logits = action_logits.clone()
        masked_logits[:, 0] = -1e9
        return masked_logits

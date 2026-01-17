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
    - Shared MLP trunk: [z, goal_emb, t] → hidden
    - Categorical head: hidden → logits[d+1]
    - Magnitude head: hidden → (μ[d], logσ[d]) (one per latent dim)
    - Value head: hidden → V(s)
    """
    
    def __init__(
        self,
        obs_dim: int,  # n_latent + goal_emb_dim + 1 (for time)
        n_latent: int,
        hidden_sizes: list = [128, 128],
        delta_max: float = 1.0,
        activation: str = "relu",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_latent = n_latent
        self.delta_max = delta_max
        
        # Build shared trunk
        layers = []
        input_dim = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            input_dim = hidden_size
        
        self.shared_trunk = nn.Sequential(*layers)
        trunk_output_dim = hidden_sizes[-1]
        
        # Categorical head (action selection)
        self.action_head = nn.Linear(trunk_output_dim, n_latent + 1)  # +1 for no-op
        
        # Magnitude head (one μ, logσ per latent dim)
        self.magnitude_mu_head = nn.Linear(trunk_output_dim, n_latent)
        self.magnitude_logstd_head = nn.Linear(trunk_output_dim, n_latent)
        
        # Value head
        self.value_head = nn.Linear(trunk_output_dim, 1)
    
    def forward(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through policy network.
        
        Parameters
        ----------
        obs : torch.Tensor
            Observations of shape (batch_size, obs_dim).
        
        Returns
        -------
        action_logits : torch.Tensor
            Categorical logits of shape (batch_size, n_latent + 1).
        magnitude_mu : torch.Tensor
            Magnitude means of shape (batch_size, n_latent).
        magnitude_logstd : torch.Tensor
            Magnitude log-stddevs of shape (batch_size, n_latent).
        value : torch.Tensor
            Value estimates of shape (batch_size, 1).
        """
        h = self.shared_trunk(obs)
        
        action_logits = self.action_head(h)  # (B, n_latent + 1)
        magnitude_mu = self.magnitude_mu_head(h)  # (B, n_latent)
        magnitude_logstd = self.magnitude_logstd_head(h)  # (B, n_latent)
        value = self.value_head(h)  # (B, 1)
        
        return action_logits, magnitude_mu, magnitude_logstd, value
    
    def sample(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
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
            Raw (pre-tanh) magnitudes of shape (batch_size,).
        log_prob : torch.Tensor
            Log probabilities of shape (batch_size,).
        """
        action_logits, magnitude_mu, magnitude_logstd, _ = self.forward(obs)
        
        # Sample discrete action
        if deterministic:
            action = torch.argmax(action_logits, dim=1)  # (B,)
        else:
            action_dist = dist.Categorical(logits=action_logits)
            action = action_dist.sample()  # (B,)
        
        # Sample magnitude (only used if action > 0)
        magnitude_std = torch.exp(magnitude_logstd.clamp(min=-10, max=2))  # (B, n_latent)
        magnitude_dist = dist.Normal(magnitude_mu, magnitude_std)
        raw_delta = magnitude_dist.sample()  # (B, n_latent)
        
        # Apply tanh squashing
        delta = self.delta_max * torch.tanh(raw_delta)  # (B, n_latent)
        
        # Select magnitude for chosen action
        # If action = 0, delta = 0; else delta = delta[action - 1]
        batch_size = obs.shape[0]
        action_mask = (action > 0).long()  # (B,)
        action_indices = (action - 1).clamp(min=0, max=self.n_latent - 1)  # (B,)
        delta_selected = delta[torch.arange(batch_size, device=obs.device), action_indices]  # (B,)
        delta_selected = delta_selected * action_mask.float()  # Zero if action == 0
        
        raw_delta_selected = raw_delta[torch.arange(batch_size, device=obs.device), action_indices]  # (B,)
        raw_delta_selected = raw_delta_selected * action_mask.float()
        
        # Compute log probability
        log_prob = self.log_prob(obs, action, delta_selected, raw_delta_selected)
        
        return action, delta_selected, raw_delta_selected, log_prob
    
    def log_prob(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        delta: torch.Tensor,
        raw_delta: torch.Tensor,
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
            Continuous magnitudes (post-tanh) of shape (batch_size,).
        raw_delta : torch.Tensor
            Raw (pre-tanh) magnitudes of shape (batch_size,).
        
        Returns
        -------
        log_prob : torch.Tensor
            Log probabilities of shape (batch_size,).
        """
        action_logits, magnitude_mu, magnitude_logstd, _ = self.forward(obs)
        
        # Categorical log prob
        action_dist = dist.Categorical(logits=action_logits)
        log_prob_cat = action_dist.log_prob(action)  # (B,)
        
        # Magnitude log prob (only if action > 0)
        batch_size = obs.shape[0]
        action_mask = (action > 0).float()  # (B,)
        action_indices = (action - 1).clamp(min=0, max=self.n_latent - 1)  # (B,)
        
        # Get μ and σ for chosen action
        mu_selected = magnitude_mu[torch.arange(batch_size, device=obs.device), action_indices]  # (B,)
        logstd_selected = magnitude_logstd[torch.arange(batch_size, device=obs.device), action_indices]  # (B,)
        std_selected = torch.exp(logstd_selected.clamp(min=-10, max=2))  # (B,)
        
        # Gaussian log prob of raw_delta
        magnitude_dist = dist.Normal(mu_selected, std_selected)
        log_prob_mag_raw = magnitude_dist.log_prob(raw_delta)  # (B,)
        
        # Tanh squashing correction: log(1 - tanh^2(x))
        # delta = delta_max * tanh(raw_delta)
        # d/delta_max * tanh = delta_max * (1 - tanh^2)
        tanh_raw = torch.tanh(raw_delta)
        log_det_jacobian = torch.log(self.delta_max * (1 - tanh_raw**2) + 1e-8)  # (B,)
        
        log_prob_mag = log_prob_mag_raw - log_det_jacobian  # (B,) - correct change of variables
        
        # Total log prob: categorical + magnitude (only if action > 0)
        log_prob = log_prob_cat + action_mask * log_prob_mag
        
        return log_prob
    
    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of policy distribution.
        
        Parameters
        ----------
        obs : torch.Tensor
            Observations of shape (batch_size, obs_dim).
        
        Returns
        -------
        entropy : torch.Tensor
            Entropy of shape (batch_size,).
        """
        action_logits, magnitude_mu, magnitude_logstd, _ = self.forward(obs)
        
        # Categorical entropy
        action_dist = dist.Categorical(logits=action_logits)
        entropy_cat = action_dist.entropy()  # (B,)
        
        # Magnitude entropy (weighted by action probs)
        magnitude_std = torch.exp(magnitude_logstd.clamp(min=-10, max=2))  # (B, n_latent)
        magnitude_dist = dist.Normal(magnitude_mu, magnitude_std)
        entropy_mag_per_dim = magnitude_dist.entropy()  # (B, n_latent)
        
        # Weight by probability of selecting each action
        action_probs = F.softmax(action_logits, dim=1)  # (B, n_latent + 1)
        # Action 0 has no magnitude, so weight by probs of actions 1..n_latent
        action_probs_mag = action_probs[:, 1:]  # (B, n_latent)
        entropy_mag = (entropy_mag_per_dim * action_probs_mag).sum(dim=1)  # (B,)
        
        total_entropy = entropy_cat + entropy_mag
        return total_entropy
    
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

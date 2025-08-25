import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, n_input: int, n_hidden: int, n_latent: int):
        super().__init__()
        # shared encoder MLP
        self.encoder = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.ReLU(),
        )

        # project to mean and log-variance
        self.mean_layer = nn.Linear(n_hidden, n_latent)
        self.logvar_layer = nn.Linear(n_hidden, n_latent)

    def forward(self, x: torch.Tensor, *, generator: torch.Generator | None = None):
        # x is concatenated [u, s] so split then sum for the encoder signal
        u, s = torch.split(x, x.shape[1] // 2, dim=1)
        xs = u + s
        h = self.encoder(xs)
        mean = self.mean_layer(h)
        logvar = self.logvar_layer(h)
        z = self.reparametrize(mean, logvar, generator=generator)
        return z, mean, logvar

    @staticmethod
    def reparametrize(mean, logvar, *, generator=None):
        std = torch.exp(0.5 * logvar)
        eps = torch.empty_like(std).normal_(mean=0.0, std=1.0, generator=generator)
        return mean + eps * std

class MaskedLinearDecoder(nn.Module):
    """Linear decoder with hard mask on regression weights: outputs G from latent L."""
    def __init__(self, n_latent: int, n_output: int, mask: torch.Tensor):
        super().__init__()
        # mask shape must be [n_output, n_latent]
        assert mask.shape == (n_output, n_latent), \
            f"Mask shape {tuple(mask.shape)} must be (n_output={n_output}, n_latent={n_latent})"
        self.register_buffer("mask", mask)

        self.linear = nn.Linear(n_latent, n_output)

        # zero out masked positions at init
        with torch.no_grad():
            self.linear.weight.mul_(self.mask)

    def forward(self, x: torch.Tensor):
        # reapply mask every forward (in case of weight updates)
        masked_w = self.linear.weight * self.mask
        return F.linear(x, masked_w, self.linear.bias)

class VelocityDecoder(nn.Module):
    """
    Produces:
      - velocity_gp: (B, L)   latent-space GP-like velocity
      - velocity:    (B, 2G)  per-gene [u_vel, s_vel] if gene_prior=True,
                              else free (B, 2G) projection
    """

    def __init__(self, n_latent: int, n_hidden: int, n_output: int):
        super().__init__()
        self.shared_decoder = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.ReLU(),
        )

        # latent GP-ish velocity head (L-dim)
        self.gp_velocity_decoder = nn.Linear(n_hidden, n_latent)

        # outputs [alpha, beta, gamma] each in R^G; n_output here is 2G, so G = n_output//2
        G2 = n_output
        G = G2 // 2
        self.gene_velocity_decoder = nn.Sequential(
            nn.Linear(n_hidden, 3 * G),
            nn.Softplus()
        )
        self._G = G

    def forward(self, z: torch.Tensor, x: torch.Tensor):
        h = self.shared_decoder(z)
        velocity_gp = self.gp_velocity_decoder(h)

        kinetic_params = self.gene_velocity_decoder(h)  # (B, 3G)
        alpha, beta, gamma = torch.split(kinetic_params, self._G, dim=1)
        unspliced, spliced = torch.split(x, x.size(1) // 2, dim=1)
        velocity_u = alpha - beta * unspliced
        velocity_s = beta * unspliced - gamma * spliced
        velocity = torch.cat([velocity_u, velocity_s], dim=1)  # (B, 2G)
        
        return velocity, velocity_gp, alpha, beta, gamma

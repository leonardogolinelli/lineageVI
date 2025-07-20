import torch.nn as nn
import torch
from collections.abc import Iterable
import numpy as np
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, n_input: int, n_latent: int, n_hidden: int):
        super().__init__()
        # shared encoder MLP
        self.encoder = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
        )
        
        # project to mean and log-variance
        self.mean_layer   = nn.Linear(n_hidden, n_latent)
        self.logvar_layer = nn.Linear(n_hidden, n_latent)

    def forward(self, x: torch.Tensor):
        h      = self.encoder(x)
        mean   = self.mean_layer(h)
        logvar = self.logvar_layer(h)
        z      = self.reparametrize(mean, logvar)
        return z, mean, logvar

    @staticmethod
    def reparametrize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

class MaskedLinearDecoder(nn.Module):
    """Linear decoder for scVI with hard mask on its regression weights."""
    def __init__(
        self,
        n_input: int,
        n_output: int,
        mask: torch.Tensor,
    ):
        
        super().__init__()
        # 1) keep mask as a buffer (not a parameter)
        #    shape must be [out_features, in_features] == [n_output, n_input]
        self.register_buffer("mask", mask)

        # 2) build your normal 1-layer FCLayers that outputs 2*n_output units
        self.linear = nn.Linear(n_input, n_output)

        # 4) zero out masked positions at init
        with torch.no_grad():
            print(self.linear.weight.shape, self.mask.shape)
            self.linear.weight.mul_(self.mask)

    def forward(self, x: torch.Tensor):
        # reapply mask every forward (in case of weight updates)
        masked_w = self.linear.weight * self.mask
        return F.linear(x, masked_w, self.linear.bias)

class VelocityDecoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: int,
    ):

        super().__init__()
        self.shared_decoder = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
        )

        self.gene_velocity_decoder = nn.Sequential(
                nn.Linear(n_hidden, n_output)
        )

        self.gp_velocity_decoder = nn.Sequential(
                nn.Linear(n_hidden, n_input)
        )

    def forward(self, z):
        # Parameters for latent distribution
        h = self.shared_decoder(z)
        velocity = self.gene_velocity_decoder(h)
        velocity_gp = self.gp_velocity_decoder(h)

        return velocity, velocity_gp
    
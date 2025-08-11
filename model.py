import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scanpy as sc

from modules import Encoder, MaskedLinearDecoder, VelocityDecoder


def seed_everything(seed: int):
    """Seed Python, NumPy, and torch (CPU and CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # enforce determinism where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch ≥1.8: error on nondeterministic ops
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass

class lineageVIModel(nn.Module):
    def __init__(
        self,
        adata: sc.AnnData,
        n_hidden: int = 128,
        mask_key: str = "I",
        gene_prior: bool = False,
        seed: int | None = None,
    ):
        # If a seed is provided, lock all RNGs *before* instantiating any layers
        if seed is not None:
            seed_everything(seed)

        super().__init__()

        # latent dimension from AnnData mask
        mask_np = adata.varm[mask_key]  # shape: (G, L)
        n_latent = int(mask_np.shape[1])
        mask = torch.from_numpy(mask_np).to(torch.float32)

        # dimensions:
        # G = #genes; inputs are [u, s] -> 2G; recon outputs G (u+s)
        G = int(adata.shape[1])
        n_input = G
        n_output = n_input

        # submodules
        self.encoder = Encoder(n_input, n_hidden, n_latent)
        self.gene_decoder = MaskedLinearDecoder(n_latent, n_output, mask)
        self.velocity_decoder = VelocityDecoder(n_latent, n_hidden, 2 * G, gene_prior)

        # keep mask buffer around if needed elsewhere
        self.register_buffer("mask", mask)

        # regime toggle used by Trainer
        self.first_regime: bool = True

    def forward(self, x: torch.Tensor):
        z, mean, logvar = self.encoder(x)
        recon = self.gene_decoder(z)

        # only run velocity‐decoder if we're *not* in first_regime
        if not self.first_regime:
            velocity, velocity_gp = self.velocity_decoder(z, x)
        else:
            velocity, velocity_gp = None, None

        return recon, velocity, velocity_gp, mean, logvar

    def reconstruction_loss(self, recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # recon targets u+s
        u, s = torch.split(x, x.shape[1] // 2, dim=1)
        target = u + s
        return F.mse_loss(recon, target, reduction="mean")

    def kl_divergence(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # standard Gaussian prior KL
        kld_element = 1 + logvar - mean.pow(2) - logvar.exp()
        kld = -0.5 * torch.sum(kld_element, dim=1)
        return kld.mean()

    def velocity_loss(
        self,
        velocity_pred: torch.Tensor,  # (B, D)
        x: torch.Tensor,              # (B, D)
        x_neigh: torch.Tensor         # (B, K, D)
    ) -> torch.Tensor:
        """
        Velocity loss = mean over batch of (1 - max_cosine_similarity(predicted_velocity, neighbor_diff))
        Works in either gene‐expression space or concatenated spaces (e.g., [x, z]).
        """
        # diffs to neighbors
        diffs = x_neigh - x.unsqueeze(1)                              # (B, K, D)
        cos_sim = F.cosine_similarity(diffs, velocity_pred.unsqueeze(1), dim=-1)  # (B, K)
        max_sim, _ = cos_sim.max(dim=1)                               # (B,)
        return (1.0 - max_sim).mean()

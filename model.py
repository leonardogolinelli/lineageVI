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
    torch.backends.cudnn.benchmark     = False

    # PyTorch ≥1.8: error on nondeterministic ops
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass

class VAEModel(nn.Module):
    def __init__(
        self,
        adata: sc.AnnData,
        n_hidden: int = None,
        mask_key: str = 'I',
        gene_prior: bool = False,
        seed: int = None,                # ← new argument
    ):
        # If a seed is provided, lock all RNGs *before* instantiating any layers
        if seed is not None:
            seed_everything(seed)

        super().__init__()

        # latent dimension from AnnData mask
        n_latent = adata.varm[mask_key].shape[1]
        mask = adata.varm[mask_key]
        mask = torch.from_numpy(np.concatenate([mask, mask], axis=0))

        # input/output dims (unspliced+spliced)
        n_input  = adata.shape[1] * 2
        n_output = n_input

        # submodules
        self.encoder          = Encoder(n_input, n_hidden, n_latent)
        self.gene_decoder     = MaskedLinearDecoder(n_latent, n_output, mask)
        self.velocity_decoder = VelocityDecoder(n_latent, n_hidden, n_output, gene_prior)

        # keep mask buffer around if needed elsewhere
        if mask is not None:
            self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor):
        z, mean, logvar       = self.encoder(x)
        recon                 = self.gene_decoder(z)

        # only run velocity‐decoder if we're *not* in first_regime
        if not self.first_regime:
            velocity, velocity_gp = self.velocity_decoder(z, x)
        else:
            velocity, velocity_gp = None, None

        return recon, velocity, velocity_gp, mean, logvar


    def reconstruction_loss(self, recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(recon, x, reduction='mean')

    def kl_divergence(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # standard Gaussian prior KL
        kld_element = 1 + logvar - mean.pow(2) - logvar.exp()
        kld = -0.5 * torch.sum(kld_element, dim=1)
        return kld.mean()

    def velocity_loss(
        self,
        velocity_pred: torch.Tensor,      # (B, D)
        x: torch.Tensor,                  # (B, D)
        x_neigh: torch.Tensor             # (B, K, D)
    ) -> torch.Tensor:
        """
        Velocity loss = mean over batch of (1 - max_cosine_similarity(predicted_velocity, neighbor_diff))
        Works in either gene‐expression space or latent space.
        """
        # diffs to neighbors
        diffs   = x_neigh - x.unsqueeze(1)  # (B, K, D)
        cos_sim = F.cosine_similarity(diffs, velocity_pred.unsqueeze(1), dim=-1)  # (B, K)
        max_sim, _ = cos_sim.max(dim=1)    # (B,)
        return (1.0 - max_sim).mean()

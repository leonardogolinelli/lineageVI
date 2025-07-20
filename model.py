import torch
from model import Encoder, MaskedLinearDecoder, VelocityDecoder
import scanpy as sc
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F 

class VAEModel(nn.Module):
    def __init__(
        self,
        adata : sc.AnnData,
        unspliced_key: 'unspliced',
        spliced_key: 'spliced',
        n_input: int,     # dimension of input (genes)
        n_latent: int,    # latent dimensionality
        n_hidden: int,    # hidden size for all MLPs
        mask: torch.Tensor
    ):
        super().__init__()
        self.encoder           = Encoder(n_input, n_latent, n_hidden)
        self.gene_decoder      = MaskedLinearDecoder(n_latent, n_input, mask)
        self.velocity_decoder  = VelocityDecoder(n_latent, n_input, n_hidden)

        self.register_buffer("mask", mask)

        unspliced = adata.layers[unspliced_key].toarray() if sp.issparse(adata.layers[unspliced_key]) else adata.layers[unspliced_key].astype(np.float32, copy=False)
        spliced = adata.layers[spliced_key].toarray() if sp.issparse(adata.layers[spliced_key]) else adata.layers[spliced_key].astype(np.float32, copy=False)

        # concatenate full unspliced and spliced
        full_u_s = np.concatenate(
            [unspliced, spliced], axis=1
        ) # (B, G*2)

        self.register_buffer(
            "full_data",
            torch.from_numpy(np.asarray(full_u_s)).float(),
        )

    def forward(self, x: torch.Tensor):
        z, mean, logvar      = self.encoder(x)
        recon                = self.gene_decoder(z)
        velocity, velocity_gp = self.velocity_decoder(z)
        return recon, velocity, velocity_gp, mean, logvar
    

    def reconstruction_loss(self, recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(recon, x, reduction='mean')
    
    def kl_divergence(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # (B, latent)
        kld_element = 1 + logvar - mean.pow(2) - logvar.exp()
        # sum over latent dims, -0.5 factor
        kld = -0.5 * torch.sum(kld_element, dim=1)
        # mean over batch
        return kld.mean()

    def velocity_loss(
            self,
            velocity_pred: torch.Tensor,  # (B, G)
            x: torch.Tensor,              # (B, G)
            cell_idx: torch.Tensor,       # (B,)
        ) -> torch.Tensor:
            B, G = x.shape
            K = self.K
            # gather kNN indices buffer
            neigh_idx = self.nn_indices[cell_idx, :K]            # (B, K)
            flat_idx = neigh_idx.reshape(-1)                     # (B*K,)
            neigh_data = (
                self.full_data
                    .index_select(0, flat_idx)
                    .view(B, K, G)
            )

            # differences and cosine similarity
            diffs   = neigh_data - x.unsqueeze(1)                # (B, K, G)
            cos_sim = F.cosine_similarity(diffs, velocity_pred.unsqueeze(1), dim=-1)  # (B,K)
            max_sim, _ = cos_sim.max(dim=1)                      # (B,)
            return (1.0 - max_sim).mean()

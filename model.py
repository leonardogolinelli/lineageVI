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
    
    '''@torch.inference_mode()
    def get_velocity(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        gene_list: Sequence[str] | None = None,
        n_samples: int = 1,
        batch_size: int | None = None,
        return_mean: bool = True,
        return_negative_velo: bool = True,
        dataloader: Iterator[dict[str, Tensor | None]] | None = None,
    ) -> np.ndarray:
        """
        Estimate RNA velocity (cells × genes) by sampling from the posterior,
        using module.inference + module.generative directly.

        Parameters
        ----------
        adata
            AnnData to use (defaults to the one passed at initialization).
        indices
            Which cells to use (default: all).
        gene_list
            Return velocities only for this subset of genes.
        n_samples
            How many posterior samples per cell.
        batch_size
            Minibatch size (default scvi.settings.batch_size).
        return_mean
            If True, average over the n_samples per cell (→ [cells, genes]);
            else return all samples (→ [n_samples, cells, genes]).
        return_negative_velo
            If True, multiply all velocities by -1 before returning.
        dataloader
            You can pass your own iterator; otherwise one is built for you.

        Returns
        -------
        A NumPy array of shape
        - `(cells, G')` if `n_samples=1` or `return_mean=True`,
        - `(n_samples, cells, G')` otherwise,
        where `G' = len(gene_list)` if given, else `G = n_genes`.
        """
        import numpy as np
        import torch
        from scvi.data._utils import _validate_adata_dataloader_input
        from scvi.module._constants import MODULE_KEYS

        # 1) validate/train check
        self._check_if_trained(warn=False)
        _validate_adata_dataloader_input(self, adata, dataloader)

        # 2) build gene mask
        adata0 = self._validate_anndata(adata)
        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = list(adata0.var_names)
            gene_mask = [g in gene_list for g in all_genes]

        # 3) build or reuse dataloader
        if dataloader is None:
            if indices is None:
                indices = np.arange(adata0.n_obs)
            dataloader = self._make_data_loader(
                adata=adata0, indices=indices, batch_size=batch_size
            )
        else:
            for p in [indices, batch_size]:
                if p is not None:
                    Warning(f"Ignoring {p!r}; custom dataloader provided.")

        all_vels = []
        # 4) loop over minibatches
        for tensors in dataloader:
            samples: list[torch.Tensor] = []
            for _ in range(n_samples):
                inf_out = self.module.inference(
                    **self.module._get_inference_input(tensors)
                )
                gen_in = self.module._get_generative_input(tensors, inf_out)
                gen_out = self.module.generative(**gen_in)
                vel = gen_out[MODULE_KEYS.VELOCITY_KEY]  # (B, G)
                samples.append(vel.cpu())
            samp_tensor = torch.stack(samples, dim=0)  # (n_samples, B, G)

            if return_mean:
                batch_vel = samp_tensor.mean(dim=0)  # (B, G)
            else:
                batch_vel = samp_tensor         # (n_samples, B, G)

            # subset genes if requested
            if isinstance(gene_mask, list):
                if batch_vel.ndim == 3:
                    batch_vel = batch_vel[..., gene_mask]
                else:
                    batch_vel = batch_vel[:, gene_mask]

            if return_negative_velo:
                batch_vel.neg_()

            all_vels.append(batch_vel)

        # 5) stitch batches back together
        first = all_vels[0]
        if first.ndim == 3:
            final = torch.cat(all_vels, dim=1)  # (n_samples, total_cells, G')
        else:
            final = torch.cat(all_vels, dim=0)  # (total_cells, G')

        velos = final.numpy()

        velocity_u, velocity = np.split(velos, 2, axis=-1)
        
        return velocity_u, velocity'''

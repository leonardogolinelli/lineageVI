import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scanpy as sc
import pandas as pd
from joblib import Parallel, delayed
from typing import Tuple

from .modules import Encoder, MaskedLinearDecoder, VelocityDecoder


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

class LineageVIModel(nn.Module):
    def __init__(
        self,
        adata: sc.AnnData,
        n_hidden: int = 128,
        mask_key: str = "I",
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
        self.velocity_decoder = VelocityDecoder(n_latent, n_hidden, 2 * G)

        # keep mask buffer around if needed elsewhere
        self.register_buffer("mask", mask)

        # regime toggle used by Trainer
        self.first_regime: bool = True

    def forward(self, x: torch.Tensor):
        z, mean, logvar = self.encoder(x)
        recon = self.gene_decoder(z)

        if not self.first_regime:
            velocity, velocity_gp, alpha, beta, gamma = self.velocity_decoder(z, x)
        else:
            velocity = velocity_gp = alpha = beta = gamma = None

        return {
            "recon": recon,           # (B, G)
            "z": z,                   # (B, L)
            "mean": mean,             # (B, L)
            "logvar": logvar,         # (B, L)
            "velocity": velocity,     # (B, 2G) or None
            "velocity_gp": velocity_gp,  # (B, L) or None
            "alpha": alpha, "beta": beta, "gamma": gamma,  # (B, G) or None
        }

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

    def _forward_encoder(self, x, *, generator: torch.Generator | None = None):
        z, mean, logvar = self.encoder(x, generator=generator)
        return z, mean, logvar

    def _forward_gene_decoder(self, z):
        x_rec = self.gene_decoder(z)
        return x_rec

    def _forward_velocity_decoder(self, z, x):
        velocity, velocity_gp, alpha, beta, gamma = self.velocity_decoder(z, x)
        return velocity, velocity_gp, alpha, beta, gamma
    
    def latent_enrich(
            self,
            adata,
            groups,
            comparison='rest',
            n_sample=5000,
            use_directions=False,
            directions_key='directions',
            select_terms=None,
            exact=True,
            key_added='bf_scores'
        ):
            """Gene set enrichment test for the latent space. Test the hypothesis that latent scores
            for each term in one group (z_1) is bigger than in the other group (z_2).

            Puts results to `adata.uns[key_added]`. Results are a dictionary with
            `p_h0` - probability that z_1 > z_2, `p_h1 = 1-p_h0` and `bf` - bayes factors equal to `log(p_h0/p_h1)`.

            Parameters
            ----------
            groups: String or Dict
                    A string with the key in `adata.obs` to look for categories or a dictionary
                    with categories as keys and lists of cell names as values.
            comparison: String
                    The category name to compare against. If 'rest', then compares each category against all others.
            n_sample: Integer
                    Number of random samples to draw for each category.
            use_directions: Boolean
                    If 'True', multiplies the latent scores by directions in `adata`.
            directions_key: String
                    The key in `adata.uns` for directions.
            select_terms: Array
                    If not 'None', then an index of terms to select for the test. Only does the test
                    for these terms.
            adata: AnnData
                    An AnnData object to use. If 'None', uses `self.adata`.
            exact: Boolean
                    Use exact probabilities for comparisons.
            key_added: String
                    key of adata.uns where to put the results of the test.
            """
            import pandas as pd
            if isinstance(groups, str):
                cats_col = adata.obs[groups]
                cats = cats_col.unique()
            elif isinstance(groups, dict):
                cats = []
                all_cells = []
                for group, cells in groups.items():
                    cats.append(group)
                    all_cells += cells
                adata = adata[all_cells]
                cats_col = pd.Series(index=adata.obs_names, dtype=str)
                for group, cells in groups.items():
                    cats_col[cells] = group
            else:
                raise ValueError("groups should be a string or a dict.")

            if comparison != "rest" and isinstance(comparison, str):
                comparison = [comparison]

            if comparison != "rest" and not set(comparison).issubset(cats):
                raise ValueError("comparison should be 'rest' or among the passed groups")

            scores = {}

            for cat in cats:
                if cat in comparison:
                    continue

                cat_mask = cats_col == cat
                if comparison == "rest":
                    others_mask = ~cat_mask
                else:
                    others_mask = cats_col.isin(comparison)

                choice_1 = np.random.choice(cat_mask.sum(), n_sample)
                choice_2 = np.random.choice(others_mask.sum(), n_sample)

                adata_cat = adata[cat_mask][choice_1]
                adata_others = adata[others_mask][choice_2]

                if use_directions:
                    directions = adata.uns[directions_key]
                else:
                    directions = None

                u_cat = adata_cat.layers['Mu']
                s_cat = adata_cat.layers['Ms']
                u_s = np.concatenate([u_cat, s_cat], axis=1)  # (B, 2G)
                u_others = adata_others.layers['Mu']
                s_others = adata_others.layers['Ms']
                u_s_others = np.concatenate([u_others, s_others], axis=1)  # (B, 2G)

                u_s = torch.tensor(u_s, dtype=torch.float32)
                u_s_others = torch.tensor(u_s_others, dtype=torch.float32)

                with torch.no_grad():
                    z0, means0, logvars0 = self._forward_encoder(u_s)
                    z1, means1, logvars1 = self._forward_encoder(u_s_others)

                    vars0 = logvars0.exp()
                    vars1 = logvars1.exp()

                to_numpy = lambda x : x.detach().cpu().numpy()
                means0 = to_numpy(means0)
                means1 = to_numpy(means1)
                vars0 = to_numpy(vars0)
                vars1 = to_numpy(vars1)

                if not exact:
                    if directions is not None:
                        z0 *= directions
                        z1 *= directions

                    if select_terms is not None:
                        z0 = z0[:, select_terms]
                        z1 = z1[:, select_terms]

                    to_reduce = z0 > z1

                    zeros_mask = (np.abs(z0).sum(0) == 0) | (np.abs(z1).sum(0) == 0)

                else:
                    from scipy.special import erfc

                    if directions is not None:
                        means0 *= directions
                        means1 *= directions

                    if select_terms is not None:
                        means0 = means0[:, select_terms]
                        means1 = means1[:, select_terms]
                        vars0 = vars0[:, select_terms]
                        vars1 = vars1[:, select_terms]

                    to_reduce = (means1 - means0) / np.sqrt(2 * (vars0 + vars1))
                    to_reduce = 0.5 * erfc(to_reduce)

                    zeros_mask = (np.abs(means0).sum(0) == 0) | (np.abs(means1).sum(0) == 0)

                p_h0 = np.mean(to_reduce, axis=0)
                p_h1 = 1.0 - p_h0
                epsilon = 1e-12
                
                bf = np.log(p_h0 + epsilon) - np.log(p_h1 + epsilon)

                p_h0[zeros_mask] = 0
                p_h1[zeros_mask] = 0
                bf[zeros_mask] = 0

                scores[cat] = dict(p_h0=p_h0, p_h1=p_h1, bf=bf)

            adata.uns[key_added] = scores

    @torch.inference_mode()
    def _get_model_outputs(
        self,
        adata,
        n_samples: int = 1,
        return_mean: bool = True,
        return_negative_velo: bool = True,
        base_seed: int | None = None,
        save_to_adata: bool = False,
        unspliced_key: str = "Mu",
        spliced_key: str = "Ms",
        latent_key: str = "z",
        nn_key: str = "indices",
        batch_size: int = 256,
    ):
        """
        Samples the model over the dataset.

        If save_to_adata=False:
        Returns a dict of NumPy arrays (recon, z, mean, logvar, velocity_u, velocity,
        velocity_gp, alpha, beta, gamma) with shapes:
            - return_mean=True:                  (cells, F)
            - return_mean=False & n_samples>1:   (n_samples, cells, F)
            - return_mean=False & n_samples==1:  (cells, F)

        If save_to_adata=True:
        Writes to AnnData (and returns None):
            layers["recon"]        (cells, G)
            layers["velocity_u"]   (cells, G)
            layers["velocity"]   (cells, G)
            obsm["velocity_gp"]    (cells, L)
            obsm["z"]              (cells, L)
            obsm["mean"]           (cells, L)
            obsm["logvar"]         (cells, L)
            layers["alpha"/"beta"/"gamma"] if available (cells, G)

        NOTE: If n_samples > 1, averages across samples BEFORE writing,
                overriding a potential return_mean=False.
        """
        from .dataloader import make_dataloader

        dl = make_dataloader(
            adata,
            first_regime=True,     # encoder uses Mu/Ms; we decode per-batch
            K=10,
            unspliced_key=unspliced_key,
            spliced_key=spliced_key,
            latent_key=latent_key,
            nn_key=nn_key,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            seed=None,
        )

        device = next(self.parameters()).device
        gen = torch.Generator(device=device).manual_seed(base_seed) if base_seed is not None else None

        # per-output batch accumulators (lists of tensors)
        recon_batches = []
        vel_batches = []
        velgp_batches = []
        z_batches = []
        mean_batches = []
        logvar_batches = []
        alpha_batches, beta_batches, gamma_batches = [], [], []

        for x, idx, _ in iter(dl):
            # per-sample collectors (stack on dim=0 later)
            recon_s, vel_s, velgp_s = [], [], []
            z_s, mean_s, logvar_s = [], [], []
            alpha_s, beta_s, gamma_s = [], [], []

            # Store mean and logvar only from first sample (they are deterministic)
            mean_first, logvar_first = None, None
            
            for i in range(n_samples):
                z, mean, logvar = self._forward_encoder(x, generator=gen)
                recon = self._forward_gene_decoder(z)  # (B, G)
                velocity, velocity_gp, alpha, beta, gamma = self.velocity_decoder(z, x)  # (B, 2G), (B, L), (B, G)x3

                # collect CPU tensors
                recon_s.append(recon.cpu())
                vel_s.append(velocity.cpu())
                velgp_s.append(velocity_gp.cpu())
                z_s.append(z.cpu())
                alpha_s.append(alpha.cpu())
                beta_s.append(beta.cpu())
                gamma_s.append(gamma.cpu())
                
                # Store mean and logvar only from first sample
                if i == 0:
                    mean_first = mean.cpu()
                    logvar_first = logvar.cpu()

            # stack along sample axis
            recon_s   = torch.stack(recon_s,   dim=0)  # (n_samples, B, G)
            vel_s     = torch.stack(vel_s,     dim=0)  # (n_samples, B, 2G)
            velgp_s   = torch.stack(velgp_s,   dim=0)  # (n_samples, B, L)
            z_s       = torch.stack(z_s,       dim=0)  # (n_samples, B, L)
            alpha_s   = torch.stack(alpha_s,   dim=0)  # (n_samples, B, G)
            beta_s    = torch.stack(beta_s,    dim=0)  # (n_samples, B, G)
            gamma_s   = torch.stack(gamma_s,   dim=0)  # (n_samples, B, G)

            # apply sample mean where requested (for tensor outputs we might average)
            recon_b  = recon_s.mean(0)  if return_mean else recon_s
            vel_b    = vel_s.mean(0)    if return_mean else vel_s
            velgp_b  = velgp_s.mean(0)  if return_mean else velgp_s

            # mean and logvar are deterministic, so use single values
            mean_b, logvar_b = mean_first, logvar_first
            # keep per-sample tensors for z/α/β/γ (downstream may want dispersion)
            z_b = z_s
            alpha_b, beta_b, gamma_b = alpha_s, beta_s, gamma_s

            if return_negative_velo:
                vel_b.neg_()
                velgp_b.neg_()

            recon_batches.append(recon_b)
            vel_batches.append(vel_b)
            velgp_batches.append(velgp_b)
            z_batches.append(z_b)
            mean_batches.append(mean_b)
            logvar_batches.append(logvar_b)
            alpha_batches.append(alpha_b)
            beta_batches.append(beta_b)
            gamma_batches.append(gamma_b)

        # stitch over batches (dim=1 if we have a sample axis)
        def _stitch(lst: list[torch.Tensor]) -> torch.Tensor:
            first = lst[0]
            return torch.cat(lst, dim=1 if first.ndim == 3 else 0)

        recon_all  = _stitch(recon_batches)    # (n_samples?, cells, G) or (cells, G)
        vel_all    = _stitch(vel_batches)      # (n_samples?, cells, 2G) or (cells, 2G)
        velgp_all  = _stitch(velgp_batches)    # (n_samples?, cells, L)  or (cells, L)
        z_all      = _stitch(z_batches)        # (n_samples, cells, L)
        alpha_all  = _stitch(alpha_batches)    # (n_samples, cells, G)
        beta_all   = _stitch(beta_batches)     # (n_samples, cells, G)
        gamma_all  = _stitch(gamma_batches)    # (n_samples, cells, G)
        
        # mean and logvar are deterministic, so just concatenate along batch dimension
        mean_all   = torch.cat(mean_batches, dim=0)     # (cells, L)
        logvar_all = torch.cat(logvar_batches, dim=0)   # (cells, L)

        # squeeze leading n_samples dim if it's a singleton AND return_mean=False
        def _maybe_squeeze(t: torch.Tensor | None) -> torch.Tensor | None:
            if t is None:
                return None
            return t.squeeze(0) if (not return_mean and n_samples == 1 and t.ndim == 3) else t

        recon_all  = _maybe_squeeze(recon_all)
        vel_all    = _maybe_squeeze(vel_all)
        velgp_all  = _maybe_squeeze(velgp_all)
        z_all      = _maybe_squeeze(z_all)
        # mean and logvar are always 2D (cells, L), no need to squeeze
        alpha_all  = _maybe_squeeze(alpha_all)
        beta_all   = _maybe_squeeze(beta_all)
        gamma_all  = _maybe_squeeze(gamma_all)

        # split velocity into u/s in NumPy space
        vel_u_np = vel_s_np = None
        if vel_all is not None:
            vel_np = vel_all.numpy()
            vel_u_np, vel_s_np = np.split(vel_np, 2, axis=-1)

        if not save_to_adata:
            # return everything as NumPy arrays
            return {
                "recon": recon_all.numpy(),
                "z": z_all.numpy(),
                "mean": mean_all.numpy(),
                "logvar": logvar_all.numpy(),
                "velocity_u": vel_u_np,
                "velocity": vel_s_np,
                "velocity_gp": None if velgp_all is None else velgp_all.numpy(),
                "alpha": None if alpha_all is None else alpha_all.numpy(),
                "beta":  None if beta_all  is None else beta_all.numpy(),
                "gamma": None if gamma_all is None else gamma_all.numpy(),
            }

        # ---------------------------
        # save_to_adata=True path
        # ---------------------------
        # If multiple samples, FORCE averaging before writing (overrules return_mean)
        force_mean = (n_samples > 1)

        def _maybe_mean_first_axis(t: torch.Tensor | None) -> np.ndarray | None:
            if t is None:
                return None
            # t can be (n_samples, cells, F) OR (cells, F)
            if t.ndim == 3 and (force_mean or return_mean):
                t = t.mean(0)  # -> (cells, F)
            elif t.ndim == 3 and not (force_mean or return_mean):
                # n_samples==1 case keeps (cells, F) after squeeze above,
                # but if somehow still (1, cells, F), average it anyway
                t = t.mean(0)
            # ensure 2D (cells, F)
            if t.ndim == 2:
                arr = t.numpy()
            else:
                # defensive: last resort, mean again
                arr = t.mean(0).numpy()
            return arr.astype(np.float32, copy=False)

        # write recon
        adata.layers["recon"] = _maybe_mean_first_axis(recon_all)

        # write velocities: we already split in NumPy; apply forced mean if needed
        def _maybe_mean_np_first_axis(arr: np.ndarray | None) -> np.ndarray | None:
            if arr is None:
                return None
            # arr can be (n_samples, cells, F) or (cells, F)
            if arr.ndim == 3 and (force_mean or return_mean):
                arr = arr.mean(axis=0)  # -> (cells, F)
            elif arr.ndim == 3:
                arr = arr.mean(axis=0)  # defensive
            return arr.astype(np.float32, copy=False)

        adata.layers["velocity_u"] = _maybe_mean_np_first_axis(vel_u_np)
        adata.layers["velocity"] = _maybe_mean_np_first_axis(vel_s_np)

        # write velocity_gp, z, mean, logvar to .obsm
        Vgp = _maybe_mean_first_axis(velgp_all)
        Z   = _maybe_mean_first_axis(z_all)
        # mean and logvar are always 2D (cells, L), no need for _maybe_mean_first_axis
        MU  = mean_all.numpy().astype(np.float32)
        LV  = logvar_all.numpy().astype(np.float32)

        if Vgp is not None:
            adata.obsm["velocity_gp"] = Vgp
        if Z is not None:
            adata.obsm["z"] = Z
        if MU is not None:
            adata.obsm["mean"] = MU
        if LV is not None:
            adata.obsm["logvar"] = LV

        # kinetics into layers (G each)
        A = _maybe_mean_first_axis(alpha_all)
        B = _maybe_mean_first_axis(beta_all)
        G = _maybe_mean_first_axis(gamma_all)
        if A is not None:
            adata.layers["alpha"] = A
        if B is not None:
            adata.layers["beta"] = B
        if G is not None:
            adata.layers["gamma"] = G

        # function returns nothing when saving into AnnData
        return None

    @torch.inference_mode()
    def get_directional_uncertainty(
        self,
        adata,
        use_gp_velo: bool = False,
        n_samples: int = 50,
        n_jobs: int = -1,
        show_plot: bool = True,
        base_seed: int | None = None,
    ):
        # draw n_samples velocity fields in one call (no averaging)
        outs = self.get_model_outputs(
            adata=adata,
            n_samples=n_samples,
            return_mean=False,               # keep each sample
            return_negative_velo=True,
            base_seed=base_seed,
        )

        if not use_gp_velo:
            velocity = outs["velocity"]    # (n_samples, cells, genes) or (cells, genes) if n_samples==1
        else:
            velocity = outs["velocity_gp"]   # (n_samples, cells, L) or (cells, L)

        # Ensure we have a sample axis for the per-cell routine
        if velocity.ndim == 2:               # came back as (cells, F) because n_samples==1
            velocity = velocity[None, ...]   # -> (1, cells, F)

        df, cosine_sims = self._compute_directional_statistics_tensor(
            tensor=velocity, n_jobs=n_jobs, n_cells=adata.n_obs
        )
        df.index = adata.obs_names

        for c in df.columns:
            print(f"Adding {c} to adata.obs")
            adata.obs[c] = np.log10(df[c].values)

        if show_plot:
            print("Plotting directional_cosine_sim_variance")
            sc.pl.umap(adata, color="directional_cosine_sim_variance", vmin="p1", vmax="p99")

        return df, cosine_sims

    
    def _compute_directional_statistics_tensor(
        self, tensor: np.ndarray, n_jobs: int, n_cells: int
    ) -> pd.DataFrame:
        df = pd.DataFrame(index=np.arange(n_cells))
        df["directional_variance"] = np.nan
        df["directional_difference"] = np.nan
        df["directional_cosine_sim_variance"] = np.nan
        df["directional_cosine_sim_difference"] = np.nan
        df["directional_cosine_sim_mean"] = np.nan
        results = Parallel(n_jobs=n_jobs, verbose=3)(
            delayed(self._directional_statistics_per_cell)(tensor[:, cell_index, :])
            for cell_index in range(n_cells)
        )
        # cells by samples
        cosine_sims = np.stack([results[i][0] for i in range(n_cells)])
        df.loc[:, "directional_cosine_sim_variance"] = [
            results[i][1] for i in range(n_cells)
        ]
        df.loc[:, "directional_cosine_sim_difference"] = [
            results[i][2] for i in range(n_cells)
        ]
        df.loc[:, "directional_variance"] = [results[i][3] for i in range(n_cells)]
        df.loc[:, "directional_difference"] = [results[i][4] for i in range(n_cells)]
        df.loc[:, "directional_cosine_sim_mean"] = [results[i][5] for i in range(n_cells)]

        return df, cosine_sims

    def _directional_statistics_per_cell(
        self,
        tensor: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Internal function for parallelization.

        Parameters
        ----------
        tensor
            Shape of samples by genes for a given cell.
        """
        n_samples = tensor.shape[0]
        # over samples axis
        mean_velocity_of_cell = tensor.mean(0)
        cosine_sims = [
            self._cosine_sim(tensor[i, :], mean_velocity_of_cell) for i in range(n_samples)
        ]
        angle_samples = [np.arccos(el) for el in cosine_sims]
        return (
            cosine_sims,
            np.var(cosine_sims),
            np.percentile(cosine_sims, 95) - np.percentile(cosine_sims, 5),
            np.var(angle_samples),
            np.percentile(angle_samples, 95) - np.percentile(angle_samples, 5),
            np.mean(cosine_sims),
        )
    
    def _centered_unit_vector(self, vector: np.ndarray) -> np.ndarray:
        """Returns the centered unit vector of the vector."""
        vector = vector - np.mean(vector)
        return vector / np.linalg.norm(vector)

    def _cosine_sim(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Returns cosine similarity of the vectors."""
        v1_u = self._centered_unit_vector(v1)
        v2_u = self._centered_unit_vector(v2)
        return np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    
    @torch.inference_mode()
    def compute_extrinsic_uncertainty(
        self,
        adata,
        use_gp_velo: bool = False,
        n_samples: int = 25,
        n_jobs: int = -1,
        show_plot: bool = True,
        base_seed: int | None = None,   # ensures distinct ε across iterations (while reproducible)
    ) -> pd.DataFrame:
        import scvelo as scv
        import scanpy as sc
        from contextlib import redirect_stdout
        import io
        import numpy as np
        import pandas as pd

        # choose the working AnnData and state space
        if use_gp_velo:
            # work in GP space (cells × L)
            working_adata = self.build_gp_adata(adata, base_seed=base_seed)
            # state used for extrapolation in this space
            state_matrix = working_adata.layers["Ms"]               # (cells, L) == μ
            # function to fetch the matching velocity each iteration
            def _fetch_velocity(i_seed):
                outs = self.get_model_outputs(
                    adata=adata,
                    n_samples=1,
                    return_mean=True,
                    return_negative_velo=True,
                    base_seed=i_seed,
                    save_to_adata=False,
                )
                return outs["velocity_gp"]  # (cells, L)
        else:
            # work in gene space (cells × G)
            working_adata = adata
            state_matrix = adata.layers["Ms"]                       # (cells, G)
            def _fetch_velocity(i_seed):
                outs = self.get_model_outputs(
                    adata=adata,
                    n_samples=1,
                    return_mean=True,
                    return_negative_velo=True,
                    base_seed=i_seed,
                    save_to_adata=False,
                )
                return outs["velocity"]  # (cells, G)

        extrapolated_cells_list = []

        for i in range(n_samples):
            with io.StringIO() as buf, redirect_stdout(buf):
                vkey = f"velocities_lineagevi_{i}"
                i_seed = None if base_seed is None else base_seed + i

                velocity = _fetch_velocity(i_seed)
                if velocity is None or state_matrix is None:
                    raise RuntimeError("compute_extrinsic_uncertainty: required velocity/state not available.")

                # write velocity to the *correct* matrix shape for this space
                working_adata.layers[vkey] = velocity.astype(np.float32)

                # build transitions in the same space
                scv.tl.velocity_graph(working_adata, vkey=vkey, sqrt_transform=False, approx=True)
                T = scv.utils.get_transition_matrix(
                    working_adata, vkey=vkey, self_transitions=True, use_negative_cosines=True
                )

                # extrapolate the corresponding state in-place
                X_extrap = np.asarray(T @ state_matrix)  # (cells, features)
                extrapolated_cells_list.append(X_extrap.astype(np.float32))

        extrapolated_cells = np.stack(extrapolated_cells_list, axis=0)  # (n_samples, cells, features)

        # Directional stats over the samples axis
        df, _ = self._compute_directional_statistics_tensor(
            tensor=extrapolated_cells, n_jobs=n_jobs, n_cells=working_adata.n_obs
        )
        df.index = working_adata.obs_names

        # Log-transform into the *original* adata.obs for convenience/consistency
        for c in df.columns:
            adata.obs[c + "_extrinsic"] = np.log10(df[c].values.astype(np.float32))

        if show_plot:
            sc.pl.umap(
                adata,
                color="directional_cosine_sim_variance_extrinsic",
                vmin="p1",
                vmax="p99",
            )

        return 
    
    def _get_cell_type_idxs(self, adata, cell_type_key):
        ctype_indices = {}
        adata.obs['numerical_idx_linvi'] = np.arange(len(adata))
        for cluster, df in adata.obs.groupby(cell_type_key, observed=True):
            ctype_indices[cluster] = df['numerical_idx_linvi'].to_numpy()

        return ctype_indices
        
    
    def _get_gene_idxs(self, adata, genes):
        return np.where(adata.var_names.isin(genes))[0]
    
    def _get_gp_idxs(self, adata, gp_key, gps):
        return np.where(pd.Series(adata.uns[gp_key]).isin(gps))[0]
    
    @torch.inference_mode()
    def perturb_genes(
            self, 
            adata, 
            cell_type_key, 
            cell_type_to_perturb, 
            genes_to_perturb, 
            perturb_value,
            perturb_spliced = True,
            perturb_unspliced = False,
            perturb_both = False):
        
        perturbed_genes_idxs = self._get_gene_idxs(adata, genes_to_perturb)
        cell_type_idxs = self._get_cell_type_idxs(adata, cell_type_key=cell_type_key)
        cells_to_perturb_idxs = cell_type_idxs[cell_type_to_perturb]

        # allow both int and list
        cell_idx = cells_to_perturb_idxs   # could also be 0
        gene_idx = perturbed_genes_idxs   # could also be 0
        spliced = perturb_spliced
        unspliced = perturb_unspliced
        both = perturb_both

        # Always ensure we index properly
        mu_unperturbed = adata.layers['Mu'][cell_idx, :]
        ms_unperturbed = adata.layers['Ms'][cell_idx, :]

        # Convert to 2D consistently
        mu_unperturbed = np.atleast_2d(mu_unperturbed)
        ms_unperturbed = np.atleast_2d(ms_unperturbed)

        mu_perturbed = mu_unperturbed.copy()
        ms_perturbed = ms_unperturbed.copy()

        if unspliced:
            mu_perturbed[:, gene_idx] = perturb_value
        if spliced:
            ms_perturbed[:, gene_idx] = perturb_value
        if both:
            mu_perturbed[:, gene_idx] = perturb_value
            ms_perturbed[:, gene_idx] = perturb_value

        # Concatenate along feature axis
        mu_ms_unpert = np.concatenate([mu_unperturbed, ms_unperturbed], axis=1)
        mu_ms_pert = np.concatenate([mu_perturbed, ms_perturbed], axis=1)

        # Convert to torch tensors (float type for model)
        x_unpert = torch.tensor(mu_ms_unpert, dtype=torch.float32)
        x_pert = torch.tensor(mu_ms_pert, dtype=torch.float32)

        self.first_regime = False
        out_unpert = self.forward(x_unpert)
        out_pert = self.forward(x_pert)

        recon_unpert = out_unpert['recon']
        mean_unpert = out_unpert['mean']
        logvar_unpert = out_unpert['logvar']
        gp_velo_unpert = out_unpert['velocity_gp']
        velo_concat_unpert = out_unpert['velocity']
        velo_u_unpert, velo_unpert = np.split(velo_concat_unpert, 2, axis=1)
        alpha_unpert = out_unpert['alpha']
        beta_unpert = out_unpert['beta']
        gamma_unpert = out_unpert['gamma']

        recon_pert = out_pert['recon']
        mean_pert = out_pert['mean']
        logvar_pert = out_pert['logvar']
        gp_velo_pert = out_pert['velocity_gp']
        velo_concat_pert = out_pert['velocity']
        velo_u_pert, velo_pert = np.split(velo_concat_pert, 2, axis=1)
        alpha_pert = out_pert['alpha']
        beta_pert = out_pert['beta']
        gamma_pert = out_pert['gamma']


        if recon_unpert.shape[0] > 1:
            to_numpy = lambda x : x.cpu().numpy()

        else:
            to_numpy = lambda x : x.cpu().numpy().reshape(x.size(1))


        perturbed_outputs = {
            'recon' : to_numpy(recon_pert), 
            'mean' : to_numpy(mean_pert), 
            'logvar' : to_numpy(logvar_pert), 
            'velocity_gp' : to_numpy(gp_velo_pert), 
            'velo_u_pert' : to_numpy(velo_u_pert), 
            'velo_pert' : to_numpy(velo_pert),
            'alpha_pert' : to_numpy(alpha_pert), 
            'beta_pert' : to_numpy(beta_pert), 
            'gamma_pert' : to_numpy(gamma_pert), 
        }

        recon_diff = recon_pert - recon_unpert
        mean_diff = mean_pert - mean_unpert
        logvar_diff = logvar_pert - logvar_unpert
        gp_velo_diff = gp_velo_pert - gp_velo_unpert
        velo_u_diff = velo_u_pert - velo_u_unpert
        velo_diff = velo_pert - velo_unpert
        alpha_diff = alpha_pert - alpha_unpert
        beta_diff = beta_pert - beta_unpert
        gamma_diff = gamma_pert - gamma_unpert

        recon_diff = to_numpy(recon_diff).mean(0)
        mean_diff = to_numpy(mean_diff).mean(0)
        logvar_diff = to_numpy(logvar_diff).mean(0)
        gp_velo_diff = to_numpy(gp_velo_diff).mean(0)
        velo_u_diff = to_numpy(velo_u_diff).mean(0)
        velo_diff = to_numpy(velo_diff).mean(0)
        alpha_diff = to_numpy(alpha_diff).mean(0)
        beta_diff = to_numpy(beta_diff).mean(0)
        gamma_diff = to_numpy(gamma_diff).mean(0)

        df_gp = pd.DataFrame({
            'terms' : adata.uns['terms'],
            'mean' : mean_diff,
            'abs_mean' : abs(mean_diff),
            'logvar' : logvar_diff,
            'abs_logvar' : abs(logvar_diff),
            'gp_velocity' : gp_velo_diff,
            'abs_gp_velocity' : abs(gp_velo_diff),
        })

        df_genes = pd.DataFrame({
            'genes' : adata.var_names,
            'recon' : recon_diff,
            'abs_recon' : abs(recon_diff),
            'unspliced_velocity' : velo_u_diff,
            'abs_unspliced_velocity' : abs(velo_u_diff),
            'velocity' : velo_diff,
            'abs_velocity' : abs(velo_diff),
            'alpha' : alpha_diff,
            'abs_alpha' : abs(alpha_diff),
            'beta' : beta_diff,
            'abs_beta' : abs(beta_diff),
            'gamma' : gamma_diff,
            'abs_gamma' : abs(gamma_diff),
        })

        return df_genes, df_gp, perturbed_outputs

    @torch.inference_mode()
    def perturb_gps(self, adata, gp_uns_key, gps_to_perturb, cell_type_key, ctypes_to_perturb, perturb_value):
        cell_type_idxs = self._get_cell_type_idxs(adata, cell_type_key=cell_type_key)
        cell_idx = cell_type_idxs[ctypes_to_perturb]

        gp_idx = self._get_gp_idxs(adata, gp_uns_key, gps_to_perturb)

        mu = adata.layers['Mu'][cell_idx, :]
        ms = adata.layers['Ms'][cell_idx, :]

        mu_ms = torch.from_numpy(np.concatenate([mu, ms], axis=1))

        self.first_regime = False
        z_unpert, _, _ = self._forward_encoder(mu_ms)
        z_pert = z_unpert.clone()
        z_pert[:, gp_idx] = perturb_value
        velocity_unpert, velocity_gp_unpert, alpha_unpert, beta_unpert, gamma_unpert = self._forward_velocity_decoder(z_unpert, mu_ms)
        velocity_pert, velocity_gp_pert, alpha_pert, beta_pert, gamma_pert = self._forward_velocity_decoder(z_pert, mu_ms)
        x_dec_unpert = self._forward_gene_decoder(z_unpert)
        x_dec_pert = self._forward_gene_decoder(z_pert)

        to_numpy = lambda x : x.cpu().numpy()
        
        velo_u_pert, velo_pert = np.split(velocity_pert, 2, axis=1)

        perturbed_outputs = {
            'velocity_gp_pert' : to_numpy(velocity_gp_pert), 
            'velo_u_pert' : to_numpy(velo_u_pert), 
            'velo_pert' : to_numpy(velo_pert),
            'alpha_pert' : to_numpy(alpha_pert), 
            'beta_pert' : to_numpy(beta_pert), 
            'gamma_pert' : to_numpy(gamma_pert), 
            'recon' : to_numpy(x_dec_pert), 
        }

        velo_diff = to_numpy(velocity_pert - velocity_unpert)
        velo_gp_diff = to_numpy(velocity_gp_pert - velocity_gp_unpert)
        alpha_diff = to_numpy(alpha_pert - alpha_unpert)
        beta_diff = to_numpy(beta_pert - beta_unpert)
        gamma_diff = to_numpy(gamma_pert - gamma_unpert)
        x_dec_diff = to_numpy(x_dec_pert - x_dec_unpert)

        if velo_diff.shape[0] > 1:
            velo_diff = velo_diff.mean(0)
            velo_gp_diff = velo_gp_diff.mean(0)
            alpha_diff = alpha_diff.mean(0)
            beta_diff = beta_diff.mean(0)
            gamma_diff = gamma_diff.mean(0)
            x_dec_diff = x_dec_diff.mean(0)

        velo_diff_u, velo_diff_s = np.split(velo_diff, 2)

        genes_df = pd.DataFrame({
            'genes' : adata.var_names,
            'velo_diff_u' : velo_diff_u,
            'abs_velo_diff_u' : np.absolute(velo_diff_u),
            'velo_diff_s' : velo_diff_s,
            'abs_velo_diff_s' : np.absolute(velo_diff_s),
            'x_dec_diff' : x_dec_diff,
            'x_dec_diff_abs' : np.absolute(x_dec_diff),
            'alpha_diff' : alpha_diff,
            'alpha_diff_abs' : np.absolute(alpha_diff),
            'beta_diff' : beta_diff,
            'beta_diff_abs' : np.absolute(beta_diff),
            'gamma_diff' : gamma_diff,
            'gamma_diff_abs' : np.absolute(gamma_diff),
        })

        gps_df = pd.DataFrame({
                'gene_programs' : adata.uns['terms'],
                'velo_gp' : velo_gp_diff,
                'abs_velo_gp' : velo_gp_diff,
            })
        
        return genes_df, gps_df, perturbed_outputs

    @torch.inference_mode()
    def map_velocities(
        self,
        adata,
        direction: str = "gp_to_gene",
        n_samples: int = 100,
        scale: float = 10.0,
        base_seed: int | None = None,
        velocity_key: str = "mapped_velocity",
        return_gp_adata: bool = False,
        return_negative_velo: bool = True,
        unspliced_key: str = "Mu",
        spliced_key: str = "Ms",
        latent_key: str = "z",
        nn_key: str = "indices",
        batch_size: int = 256,
    ):
        """
        Map velocities between gene program space and gene expression space.
        
        Parameters
        ----------
        adata : AnnData
            The AnnData object containing the data.
        direction : str, default "gp_to_gene"
            Direction of mapping: "gp_to_gene" or "gene_to_gp".
        n_samples : int, default 100
            Number of samples for uncertainty estimation.
        scale : float, default 10.0
            Scaling factor for the mapped velocities.
        base_seed : int, optional
            Random seed for reproducibility.
        velocity_key : str, default "mapped_velocity"
            Key to store mapped velocities in adata.layers (gp_to_gene) or adata.obsm (gene_to_gp).
        return_gp_adata : bool, default False
            Whether to return the gene program AnnData object for downstream analysis.
        return_negative_velo : bool, default True
            Whether to negate the velocities (multiply by -1).
        unspliced_key : str, default "Mu"
            Key for unspliced counts in adata.layers.
        spliced_key : str, default "Ms"
            Key for spliced counts in adata.layers.
        latent_key : str, default "z"
            Key for latent representations in adata.obsm.
        nn_key : str, default "indices"
            Key for nearest neighbor indices in adata.uns.
        batch_size : int, default 256
            Batch size for processing.
            
        Returns
        -------
        AnnData or None
            If return_gp_adata=True, returns the gene program AnnData object.
            Otherwise returns None. Mapped velocities are always saved to adata.
        """
        import scvelo as scv
        import scanpy as sc
        from contextlib import redirect_stdout
        import io
        
        if direction not in ["gp_to_gene", "gene_to_gp"]:
            raise ValueError("direction must be 'gp_to_gene' or 'gene_to_gp'")
        
        if direction == "gp_to_gene":
            # Map from GP velocity to gene velocity
            return self._map_gp_to_gene_velocity(
                adata, n_samples, scale, base_seed,
                velocity_key, return_gp_adata, return_negative_velo, unspliced_key, spliced_key, latent_key, 
                nn_key, batch_size
            )
        else:
            # Map from gene velocity to GP velocity
            return self._map_gene_to_gp_velocity(
                adata, n_samples, scale, base_seed,
                velocity_key, return_gp_adata, return_negative_velo, unspliced_key, spliced_key, latent_key,
                nn_key, batch_size
            )
    
    def _map_gp_to_gene_velocity(
        self,
        adata,
        n_samples: int,
        scale: float,
        base_seed: int | None,
        velocity_key: str,
        return_gp_adata: bool,
        return_negative_velo: bool,
        unspliced_key: str,
        spliced_key: str,
        latent_key: str,
        nn_key: str,
        batch_size: int,
    ):
        """Map velocities from gene program space to gene expression space."""
        import scvelo as scv
        from contextlib import redirect_stdout
        import io
        
        # Get model outputs
        outs = self._get_model_outputs(
            adata=adata,
            n_samples=n_samples,
            return_mean=True,
            return_negative_velo=return_negative_velo,
            base_seed=base_seed,
            save_to_adata=False,
            unspliced_key=unspliced_key,
            spliced_key=spliced_key,
            latent_key=latent_key,
            nn_key=nn_key,
            batch_size=batch_size,
        )
        
        # Create GP-space AnnData directly
        mu = outs["mean"]  # (cells, L)
        v_gp = outs["velocity_gp"]  # (cells, L)
        
        # Build GP AnnData
        adata_gp = sc.AnnData(X=mu.astype(np.float32))
        adata_gp.obs = adata.obs.copy()
        
        if "terms" in adata.uns and len(adata.uns["terms"]) == mu.shape[1]:
            adata_gp.var_names = adata.uns["terms"]
        else:
            adata_gp.var_names = [f"GP_{i}" for i in range(mu.shape[1])]
        
        adata_gp.layers["Ms"] = mu.astype(np.float32)
        adata_gp.layers["velocity"] = v_gp.astype(np.float32)
        
        # Copy UMAP if available
        if "X_umap" in adata.obsm:
            adata_gp.obsm["X_umap"] = adata.obsm["X_umap"].copy()
        
        # Compute transition matrix in GP space
        with io.StringIO() as buf, redirect_stdout(buf):
            scv.tl.velocity_graph(adata_gp, vkey='velocity')
            T = scv.tl.transition_matrix(adata_gp, vkey='velocity').toarray()
        
        # Map back to gene expression space
        initial_state = adata.layers[spliced_key]  # Current gene expression state
        future_state = np.matmul(T, initial_state)  # Projected future state
        velo_mapped = (future_state - initial_state) * scale
        
        # Always write mapped velocity to AnnData
        adata.layers[velocity_key] = velo_mapped.astype(np.float32)
        
        # Optionally return GP AnnData
        if return_gp_adata:
            return adata_gp
        return None
    
    def _map_gene_to_gp_velocity(
        self,
        adata,
        n_samples: int,
        scale: float,
        base_seed: int | None,
        velocity_key: str,
        return_gp_adata: bool,
        return_negative_velo: bool,
        unspliced_key: str,
        spliced_key: str,
        latent_key: str,
        nn_key: str,
        batch_size: int,
    ):
        """Map velocities from gene expression space to gene program space."""
        import scvelo as scv
        import scanpy as sc
        from contextlib import redirect_stdout
        import io
        
        # Get gene-level velocities from model
        outs = self._get_model_outputs(
            adata=adata,
            n_samples=n_samples,
            return_mean=True,
            return_negative_velo=return_negative_velo,
            base_seed=base_seed,
            save_to_adata=False,
            unspliced_key=unspliced_key,
            spliced_key=spliced_key,
            latent_key=latent_key,
            nn_key=nn_key,
            batch_size=batch_size,
        )
        
        # Use gene-level velocity for transition matrix
        gene_velocity = outs["velocity"]  # (cells, G)
        
        # Create temporary AnnData with gene velocities
        adata_temp = adata.copy()
        adata_temp.layers["velocity"] = gene_velocity.astype(np.float32)
        
        # Compute transition matrix in gene space
        with io.StringIO() as buf, redirect_stdout(buf):
            scv.tl.velocity_graph(adata_temp, vkey='velocity')
            T = scv.tl.transition_matrix(adata_temp, vkey='velocity').toarray()
        
        # Map to GP space using latent representations from model outputs
        latent_state = outs["mean"]  # Current latent state from model (cells, L)
        future_latent_state = np.matmul(T, latent_state)  # Projected future latent state
        gp_velo_mapped = (future_latent_state - latent_state) * scale
        
        # Create GP AnnData if requested
        adata_gp = None
        if return_gp_adata:
            # Create GP-space AnnData directly using model outputs
            mu = outs["mean"]  # (cells, L)
            v_gp = outs["velocity_gp"]  # (cells, L)
            
            # Build GP AnnData
            adata_gp = sc.AnnData(X=mu.astype(np.float32))
            adata_gp.obs = adata.obs.copy()
            
            if "terms" in adata.uns and len(adata.uns["terms"]) == mu.shape[1]:
                adata_gp.var_names = adata.uns["terms"]
            else:
                adata_gp.var_names = [f"GP_{i}" for i in range(mu.shape[1])]
            
            adata_gp.layers["Ms"] = mu.astype(np.float32)
            adata_gp.layers[velocity_key] = v_gp.astype(np.float32)
            
            # Copy UMAP if available
            if "X_umap" in adata.obsm:
                adata_gp.obsm["X_umap"] = adata.obsm["X_umap"].copy()
        
        # Always write mapped velocity to AnnData
        adata.obsm[velocity_key] = gp_velo_mapped.astype(np.float32)
        
        # Optionally return GP AnnData
        if return_gp_adata:
            return adata_gp
        return None




import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scanpy as sc

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
    
    def build_gp_adata(self) -> sc.AnnData:
        """
        Returns a new AnnData with gene programs as obs.
        Useful for downstream analysis.
        """
        adata_gp = sc.AnnData(X=self.get_latent())
        adata_gp.obs = self._adata_ref.obs.copy()
        adata_gp.var_names = self._adata_ref.uns['terms']
        adata_gp.layers['velocity'] = self._adata_ref.obsm['velocity_gp']
        adata_gp.layers['spliced'] = self.get_latent()
        adata_gp.obsm['X_umap'] = self._adata_ref.obsm['X_umap']
        adata_gp.var_names_make_unique()
        return adata_gp

    def _forward_encoder(self, x):
        z, mean, logvar = self.encoder(x)
        return z, mean, logvar

    def _forward_gene_decoder(self, z):
        x_rec = self.gene_decoder(z)
        return x_rec

    def _forward_velocity_decoder(self, z, x):
        velocity, velocity_u = self.velocity_decoder(z, x)
        return velocity, velocity_u

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


    def get_velocity(
            self,
            adata,
            n_samples=1,
            return_mean = True,
            return_negative_velo=True
    ): 
        
        from .dataloader import make_dataloader

        dataloader = make_dataloader(
            adata,
            first_regime=True,
            K=10,
            unspliced_key='Mu',
            spliced_key='Ms',
            latent_key='z',
            nn_key='indices',
            batch_size=256, 
            shuffle=False,
            num_workers=0,
            seed=0
        )

        all_vels = []

        # 4) loop over minibatches
        with torch.no_grad():
            for x, idx, _ in iter(dataloader):
                samples: list[torch.Tensor] = []
                for _ in range(n_samples):
                    z, _, _ = self._forward_encoder(x)
                    vel, _ = self._forward_velocity_decoder(z, x)

                    samples.append(vel.cpu())
                samp_tensor = torch.stack(samples, dim=0)  # (n_samples, B, G)

                if return_mean:
                    batch_vel = samp_tensor.mean(dim=0)  # (B, G)
                else:
                    batch_vel = samp_tensor         # (n_samples, B, G)


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

            return velocity_u, velocity
    
    '''@torch.inference_mode()
    def get_directional_uncertainty(
        self,
        n_samples: int = 50,
        gene_list: Iterable[str] = None,
        n_jobs: int = -1,
        show_plot: bool = True,
    ):
        import scanpy as sc
        adata = self._validate_anndata(self.adata)

        velocity_u, velocity = self.get_velocity(
            n_samples=n_samples, return_mean=False, gene_list=gene_list
        )  # (n_samples, n_cells, n_genes)

        df, cosine_sims = self._compute_directional_statistics_tensor(
            tensor=velocity, n_jobs=n_jobs, n_cells=adata.n_obs
        )
        df.index = adata.obs_names

        for c in df.columns:
            print(f'Adding {c} to adata.obs')
            adata.obs[c] = np.log10(df[c].values) 

        if show_plot:
            print('Plotting directional_cosine_sim_variance')
            sc.pl.umap(
                adata, 
                color="directional_cosine_sim_variance",
                vmin="p1",
                vmax="p99",
            )

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
        logger.info("Computing the uncertainties...")
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
    
    def compute_extrinisic_uncertainty(
            self,
            n_samples=25, 
            n_jobs=-1,
            show_plot=True
        ) -> pd.DataFrame:

        import scanpy as sc
        import scvelo as scv
        from scvi.utils import track
        from contextlib import redirect_stdout
        import io

        adata = self._validate_anndata(self.adata)

        extrapolated_cells_list = []
        for i in track(range(n_samples)):
            with io.StringIO() as buf, redirect_stdout(buf):
                vkey = "velocities_velovi_{i}".format(i=i)
                velocity_u, velocity = self.get_velocity(adata, n_samples=1, return_mean=True)
                adata.layers[vkey] = velocity
                scv.tl.velocity_graph(adata, vkey=vkey, sqrt_transform=False, approx=True)
                t_mat = scv.utils.get_transition_matrix(
                    adata, vkey=vkey, self_transitions=True, use_negative_cosines=True
                )
                extrapolated_cells = np.asarray(t_mat @ adata.layers["Ms"])
                extrapolated_cells_list.append(extrapolated_cells)
        extrapolated_cells = np.stack(extrapolated_cells_list)
        df, _ = self._compute_directional_statistics_tensor(extrapolated_cells, n_jobs=n_jobs, n_cells=self.adata.n_obs)

        for c in df.columns:
            adata.obs[c + "_extrinisic"] = np.log10(df[c].values)

        if show_plot:
            sc.pl.umap(
                adata, 
                color="directional_cosine_sim_variance_extrinisic",
                vmin="p1", 
                vmax="p99", 
            )

        return df
        '''


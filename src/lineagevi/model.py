import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scanpy as sc
import pandas as pd
import scipy.sparse as sp
from scipy.stats import wilcoxon
from joblib import Parallel, delayed
from typing import Tuple, Optional, List

from .modules import Encoder, MaskedLinearDecoder, VelocityDecoder, ClusterEmbedding


def _wilcoxon_per_column(diffs: np.ndarray) -> np.ndarray:
    """Run Wilcoxon signed-rank test per column (alternative='two-sided'). Returns p-values; NaN if n_cells < 2.
    Constant columns (zero variance) get p-value 1.0 (no evidence against null)."""
    diffs = np.atleast_2d(diffs)
    n_cells, n_features = diffs.shape
    if n_cells < 2:
        return np.full(n_features, np.nan)
    pvals = np.full(n_features, np.nan)
    for j in range(n_features):
        col = diffs[:, j]
        if np.var(col) == 0 or np.allclose(col, col.flat[0]):
            pvals[j] = 1.0
            continue
        try:
            r = wilcoxon(col, alternative="two-sided")
            pvals[j] = r.pvalue if np.isfinite(r.pvalue) else 1.0
        except (ValueError, ZeroDivisionError):
            pvals[j] = 1.0
    return pvals


def _fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment. NaN p-values are preserved."""
    pvals = np.asarray(pvals, dtype=float)
    n = np.sum(~np.isnan(pvals))
    if n == 0:
        return pvals
    padj = np.full_like(pvals, np.nan)
    valid = ~np.isnan(pvals)
    padj[valid] = _fdr_bh_valid(pvals[valid])
    return padj


def _fdr_bh_valid(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg on a 1D array of valid p-values."""
    n = len(p)
    order = np.argsort(p)
    p_sorted = p[order]
    padj_sorted = np.minimum(1.0, p_sorted * n / np.arange(1, n + 1, dtype=float))
    # monotonicity
    for i in range(n - 2, -1, -1):
        padj_sorted[i] = min(padj_sorted[i], padj_sorted[i + 1])
    padj = np.empty_like(p)
    padj[order] = padj_sorted
    return padj


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


def _check_velocity_layers(adata, method_name, required_layers=None):
    """
    Check if required velocity layers exist in the AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object to check.
    method_name : str
        Name of the method calling this function (for error messages).
    required_layers : list of str, optional
        List of required layer keys. If None, checks for common velocity layers.
        
    Returns
    -------
    bool
        True if all required layers exist, False otherwise.
        
    Raises
    ------
    ValueError
        If required layers are missing.
    """
    if required_layers is None:
        required_layers = ["velocity"]
    
    missing_layers = []
    for layer in required_layers:
        if layer not in adata.layers:
            missing_layers.append(layer)
    
    if missing_layers:
        raise ValueError(
            f"{method_name} requires velocity layers that are missing from adata.layers. "
            f"Missing layers: {missing_layers}. "
            f"Please run get_model_outputs() first to generate velocities."
        )
    
    return True

class LineageVIModel(nn.Module):
    """
    LineageVI Neural Network Model for RNA Velocity and Gene Program Inference.
    
    This is the core neural network implementation of LineageVI, consisting of:
    - **Encoder**: Maps gene expression to latent space (z, mean, logvar)
    - **Masked Linear Decoder**: Reconstructs gene expression from latent space
    - **Velocity Decoder**: Predicts RNA velocity in gene and latent spaces
    
    The model uses a two-regime training approach:
    1. **Regime 1**: Expression reconstruction (encoder + gene decoder)
    2. **Regime 2**: Velocity prediction (velocity decoder)
    
    Parameters
    ----------
    adata : AnnData
        Single-cell RNA-seq data with gene program mask in adata.varm[mask_key].
    n_hidden : int, default 128
        Number of hidden units in the neural network layers.
    mask_key : str, default "I"
        Key for gene program mask in adata.varm.
    seed : int, optional
        Random seed for reproducible initialization.
    cluster_key : str, optional
        Key in adata.obs for cell clusters/lineages. If provided, cluster embeddings
        will be learned and used in velocity prediction.
    cluster_embedding_dim : int, default 32
        Dimension of cluster embeddings. Used when cluster_key is provided.

    Attributes
    ----------
    encoder : Encoder
        Neural network encoder that maps gene expression to latent space.
    gene_decoder : MaskedLinearDecoder
        Decoder that reconstructs gene expression from latent space.
    velocity_decoder : VelocityDecoder
        Decoder that predicts RNA velocity in gene and latent spaces.
    mask : torch.Tensor
        Binary mask for gene programs (G, L).
    n_genes : int
        Number of genes in the dataset.
    n_latent : int
        Number of gene programs (latent dimensions).
    n_hidden : int
        Number of hidden units.
    
    Examples
    --------
    >>> import scanpy as sc
    >>> import lineagevi as lvi
    >>> 
    >>> # Load data with gene program mask
    >>> adata = sc.read("data.h5ad")
    >>> 
    >>> # Initialize model
    >>> model = lvi.LineageVIModel(adata, n_hidden=256)
    >>> 
    >>> # Get model outputs
    >>> outputs = model._get_model_outputs(adata)
    """
    
    def __init__(
        self,
        adata: sc.AnnData,
        n_hidden: int = 128,
        mask_key: str = "I",
        seed: Optional[int] = None,
        cluster_key: Optional[str] = None,
        cluster_embedding_dim: int = 32,
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
        
        # Store dimensions as attributes (needed for RL adapter)
        self.n_latent = n_latent
        self.n_genes = G

        # Cluster embeddings (optional)
        self.cluster_key = cluster_key
        self.cluster_embedding_dim = cluster_embedding_dim
        if cluster_key is not None:
            # Get unique clusters and create mapping
            cluster_labels = adata.obs[cluster_key]
            unique_clusters = cluster_labels.unique().tolist()
            n_clusters = len(unique_clusters)
            
            # Create cluster to index mapping
            cluster_to_idx = {cluster: idx for idx, cluster in enumerate(unique_clusters)}
            self.cluster_to_idx = cluster_to_idx
            self.cluster_names = unique_clusters
            
            # Create cluster embedding module
            self.cluster_embedding = ClusterEmbedding(n_clusters, cluster_embedding_dim)
        else:
            self.cluster_embedding = None

        # Removed CLS embedding; keep stub for backward compatibility (e.g. old checkpoints or code)
        self.process_to_idx = {}

        # Velocity decoder input dimension: z + cluster_emb (if exists)
        velocity_input_dim = n_latent
        if cluster_key is not None:
            velocity_input_dim += cluster_embedding_dim

        # submodules
        self.encoder = Encoder(n_input, n_hidden, n_latent)
        self.gene_decoder = MaskedLinearDecoder(n_latent, n_output, mask)
        self.velocity_decoder = VelocityDecoder(velocity_input_dim, n_hidden, 2 * G, n_latent)

        # keep mask buffer around if needed elsewhere
        self.register_buffer("mask", mask)

        # regime toggle used by Trainer
        self.first_regime: bool = True

    def forward(self, x: torch.Tensor, cluster_indices: Optional[torch.Tensor] = None):
        """
        Forward pass through the LineageVI model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 2*n_genes) containing
            concatenated unspliced and spliced gene expression.
        cluster_indices : torch.Tensor, optional
            Cluster indices of shape (batch_size,) with integer cluster indices.
            Required when cluster_key is set and not in first_regime.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing model outputs:
            - 'z': Latent representations
            - 'mean': Encoder mean
            - 'logvar': Encoder log-variance
            - 'recon': Reconstructed gene expression
            - 'velocity': Gene-level velocities (if not first regime)
            - 'velocity_gp': Gene program velocities (if not first regime)
            - 'alpha', 'beta', 'gamma': Kinetic parameters (if not first regime)
        """
        device = next(self.parameters()).device
        if x.device != device:
            x = x.to(device)
        if cluster_indices is not None and cluster_indices.device != device:
            cluster_indices = cluster_indices.to(device)
        z, mean, logvar = self.encoder(x)
        recon = self.gene_decoder(z)

        if not self.first_regime:
            # In regime 2, concatenate cluster embeddings to z if available
            velocity_decoder_input = [z]
            if self.cluster_embedding is not None:
                if cluster_indices is None:
                    raise ValueError(
                        "cluster_indices is required when cluster_key is set. "
                        "Please provide cluster indices for each cell in the batch."
                    )
                if cluster_indices.dim() == 0:
                    cluster_indices = cluster_indices.unsqueeze(0)
                elif cluster_indices.dim() > 1:
                    cluster_indices = cluster_indices.squeeze()
                cluster_emb = self.cluster_embedding(cluster_indices)  # (B, E)
                velocity_decoder_input.append(cluster_emb)
            z_with_embeddings = torch.cat(velocity_decoder_input, dim=1)
            velocity, velocity_gp, alpha, beta, gamma = self.velocity_decoder(z_with_embeddings, x)
            velocity_gp = self._velocity_gp_from_gene_velocity(velocity)
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
        """
        Compute reconstruction loss for gene expression.
        
        Parameters
        ----------
        recon : torch.Tensor
            Reconstructed gene expression of shape (batch_size, n_genes).
        x : torch.Tensor
            Input gene expression of shape (batch_size, 2*n_genes) containing
            concatenated unspliced and spliced counts.
        
        Returns
        -------
        torch.Tensor
            Mean squared error loss between reconstruction and target (u + s).
        """
        # recon targets u+s
        u, s = torch.split(x, x.shape[1] // 2, dim=1)
        target = u + s
        return F.mse_loss(recon, target, reduction="mean")

    def kl_divergence(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between latent distribution and standard Gaussian.
        
        Parameters
        ----------
        mean : torch.Tensor
            Latent mean of shape (batch_size, n_latent).
        logvar : torch.Tensor
            Latent log-variance of shape (batch_size, n_latent).
        
        Returns
        -------
        torch.Tensor
            Mean KL divergence across the batch.
        """
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

    def _forward_encoder(self, x, *, generator: Optional[torch.Generator] = None):
        """
        Forward pass through the encoder only.
        
        Parameters
        ----------
        x : torch.Tensor
            Input gene expression of shape (batch_size, 2*n_genes).
        generator : torch.Generator, optional
            Random number generator for reproducible sampling.
        
        Returns
        -------
        z : torch.Tensor
            Sampled latent representations.
        mean : torch.Tensor
            Encoder mean.
        logvar : torch.Tensor
            Encoder log-variance.
        """
        z, mean, logvar = self.encoder(x, generator=generator)
        return z, mean, logvar

    def _forward_gene_decoder(self, z):
        """
        Forward pass through the gene decoder only.
        
        Parameters
        ----------
        z : torch.Tensor
            Latent representations of shape (batch_size, n_latent).
        
        Returns
        -------
        torch.Tensor
            Reconstructed gene expression of shape (batch_size, n_genes).
        """
        x_rec = self.gene_decoder(z)
        return x_rec

    def _velocity_gp_from_gene_velocity(self, velocity: torch.Tensor) -> torch.Tensor:
        """
        Compute gene program velocity as projection of (spliced) gene velocity
        onto gene decoder weights: velocity_gp = velocity_s @ W^T, where W is
        the masked gene decoder weight (G, L).
        """
        vel_s = velocity[:, self.n_genes:]  # (B, G)
        w = self.gene_decoder.linear.weight * self.gene_decoder.mask  # (G, L)
        return vel_s @ w  # (B, G) @ (G, L) = (B, L)

    def _forward_velocity_decoder(self, z, x, cluster_indices: Optional[torch.Tensor] = None):
        """
        Forward pass through the velocity decoder only.

        Parameters
        ----------
        z : torch.Tensor
            Latent representations of shape (batch_size, n_latent).
        x : torch.Tensor
            Input gene expression of shape (batch_size, 2*n_genes).
        cluster_indices : torch.Tensor, optional
            Cluster indices of shape (batch_size,) with integer cluster indices.
            Required when cluster_key is set.

        Returns
        -------
        velocity : torch.Tensor
            Gene-level velocities of shape (batch_size, 2*n_genes).
        velocity_gp : torch.Tensor
            Gene program velocities of shape (batch_size, n_latent).
        alpha : torch.Tensor
            Transcription rate parameters of shape (batch_size, n_genes).
        beta : torch.Tensor
            Splicing rate parameters of shape (batch_size, n_genes).
        gamma : torch.Tensor
            Degradation rate parameters of shape (batch_size, n_genes).
        """
        velocity_decoder_input = [z]
        if self.cluster_embedding is not None:
            if cluster_indices is None:
                raise ValueError(
                    "cluster_indices is required when cluster_key is set. "
                    "Please provide cluster indices for each cell in the batch."
                )
            if cluster_indices.dim() == 0:
                cluster_indices = cluster_indices.unsqueeze(0)
            elif cluster_indices.dim() > 1:
                cluster_indices = cluster_indices.squeeze()
            cluster_emb = self.cluster_embedding(cluster_indices)  # (B, E)
            velocity_decoder_input.append(cluster_emb)
        z_with_embeddings = torch.cat(velocity_decoder_input, dim=1)
        velocity, velocity_gp, alpha, beta, gamma = self.velocity_decoder(z_with_embeddings, x)
        velocity_gp = self._velocity_gp_from_gene_velocity(velocity)
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
                device = next(self.parameters()).device
                u_s = u_s.to(device)
                u_s_others = u_s_others.to(device)

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
        base_seed: Optional[int] = None,
        save_to_adata: bool = False,
        unspliced_key: str = "Mu",
        spliced_key: str = "Ms",
        latent_key: str = "z",
        nn_key: str = "indices",
        batch_size: int = 256,
        rescale_velocity_magnitude: bool = True,
        max_velocity_magnitude: float = 1.0,
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
        
        Parameters
        ----------
        rescale_velocity_magnitude : bool, default True
            Whether to rescale velocity magnitudes based on neighbor consistency.
            If True, computes cosine similarity between each cell's velocity direction
            and differences to its K nearest neighbors (matching the training loss).
            Velocities with high consistency (aligned with neighbor differences) get
            higher magnitudes, while inconsistent velocities get lower magnitudes.
            Uses the same K as the training regime.
        max_velocity_magnitude : float, default 1.0
            Maximum velocity magnitude after rescaling. Velocities with perfect
            consistency (max cosine similarity = 1.0 with neighbor differences) will
            have this magnitude. Velocities with zero consistency (max cosine similarity = -1.0)
            will have zero magnitude.
        """
        from .dataloader import make_dataloader

        # Set regime to False to enable velocity computation
        old_regime = self.first_regime
        self.first_regime = False
        
        cluster_to_idx = self.cluster_to_idx if self.cluster_key is not None else None
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
            cluster_key=self.cluster_key,
            cluster_to_idx=cluster_to_idx,
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

        for batch in iter(dl):
            # Batch structure: (x, idx, x_neigh) or (x, idx, x_neigh, cluster_idx)
            if len(batch) == 3:
                x, idx, _ = batch  # _ is x_neigh
                cluster_indices = None
            elif len(batch) == 4:
                x, idx, _, cluster_indices = batch
            else:
                raise ValueError(f"Unexpected batch size: {len(batch)}")
            x = x.to(device)
            if cluster_indices is not None:
                cluster_indices = cluster_indices.to(device)
            # per-sample collectors (stack on dim=0 later)
            recon_s, vel_s, velgp_s = [], [], []
            z_s, mean_s, logvar_s = [], [], []
            alpha_s, beta_s, gamma_s = [], [], []
            mean_first, logvar_first = None, None
            for i in range(n_samples):
                z, mean, logvar = self._forward_encoder(x, generator=gen)
                recon = self._forward_gene_decoder(z)  # (B, G)
                velocity, velocity_gp, alpha, beta, gamma = self._forward_velocity_decoder(z, x, cluster_indices)  # (B, 2G), (B, L), (B, G)x3

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
        def _maybe_squeeze(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
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

        # Rescale velocity magnitudes based on neighbor consistency (inference-time)
        if rescale_velocity_magnitude and vel_all is not None:
            # Get nearest neighbor indices from adata.uns
            nn_indices = adata.uns.get(nn_key)
            if nn_indices is None:
                import warnings
                warnings.warn(
                    f"Nearest neighbor indices not found in adata.uns['{nn_key}']. "
                    f"Skipping velocity magnitude rescaling. "
                    f"Run lineagevi.utils.get_neighbor_indices(adata) first.",
                    UserWarning
                )
            else:
                # Ensure velocities are 2D (cells, features)
                if vel_all.ndim == 3:
                    # (n_samples, cells, features) - take mean across samples
                    vel_mean = vel_all.mean(dim=0)  # (cells, features)
                else:
                    vel_mean = vel_all  # (cells, features)
                
                # Convert nn_indices to torch tensor
                nn_indices_tensor = torch.from_numpy(np.asarray(nn_indices, dtype=np.int64)).to(vel_mean.device)
                n_cells = vel_mean.shape[0]
                n_features = vel_mean.shape[1]
                K_total = nn_indices_tensor.shape[1]
                K = K_total - 1  # Exclude self (first column)
                
                # vel_all is shape (cells, 2*genes): [unspliced_vel, spliced_vel]
                # Split to get spliced velocities for comparison with differences in spliced space
                vel_u, vel_s = torch.split(vel_mean, n_features // 2, dim=1)
                vel_s = vel_s.to(vel_mean.device)  # (cells, genes) - spliced velocities
                
                # Get current spliced state from adata for computing neighbor differences
                if spliced_key not in adata.layers:
                    import warnings
                    warnings.warn(
                        f"Spliced counts not found in adata.layers['{spliced_key}']. "
                        f"Skipping velocity magnitude rescaling.",
                        UserWarning
                    )
                else:
                    # Get current state (spliced counts)
                    spliced_data = adata.layers[spliced_key]
                    # Handle sparse matrices
                    if sp.issparse(spliced_data):
                        spliced_data = spliced_data.toarray()
                    current_state = torch.from_numpy(np.asarray(spliced_data, dtype=np.float32)).to(vel_mean.device)  # (cells, genes)
                    
                    # Compute consistency scores for each cell
                    consistency_scores = torch.zeros(n_cells, device=vel_mean.device, dtype=vel_mean.dtype)
                    
                    for i in range(n_cells):
                        # Get cell's velocity direction (spliced part)
                        v_i = vel_s[i]  # (genes,)
                        v_i_norm = F.normalize(v_i.unsqueeze(0), p=2, dim=1).squeeze(0)  # (genes,)
                        
                        # Get neighbor indices (exclude self, which is at position 0)
                        neighbor_indices = nn_indices_tensor[i, 1:K_total]  # (K,)
                        # Filter out invalid indices (-1 for padding)
                        valid_mask = neighbor_indices >= 0
                        valid_indices = neighbor_indices[valid_mask]  # (K_valid,)
                        
                        if valid_indices.numel() == 0:
                            # No valid neighbors, set consistency to 0
                            consistency_scores[i] = 0.0
                            continue
                        
                        # Get current state for cell and neighbors
                        x_i = current_state[i]  # (genes,)
                        x_neigh = current_state[valid_indices]  # (K_valid, genes)
                        
                        # Compute differences to neighbors (like in velocity_loss)
                        diffs = x_neigh - x_i.unsqueeze(0)  # (K_valid, genes)
                        
                        # Normalize differences
                        diffs_norm = F.normalize(diffs, p=2, dim=1)  # (K_valid, genes)
                        
                        # Compute cosine similarities between velocity direction and neighbor differences
                        cos_sims = F.cosine_similarity(
                            v_i_norm.unsqueeze(0), 
                            diffs_norm, 
                            dim=1
                        )  # (K_valid,)
                        
                        # Use mean cosine similarity across neighbors as consistency score
                        mean_sim = cos_sims.mean()  # Mean consistency across neighbors
                        # Map from [-1, 1] to [0, 1] for magnitude scaling
                        consistency = (mean_sim + 1.0) / 2.0
                        consistency_scores[i] = consistency.clamp(0.0, 1.0)
                    
                    # Scale velocity magnitudes: consistency → magnitude
                    # Mean consistency (1.0) → max_velocity_magnitude
                    # Zero consistency (0.0) → 0.0
                    if vel_all.ndim == 3:
                        # (n_samples, cells, features)
                        for s in range(vel_all.shape[0]):
                            for i in range(n_cells):
                                # Get original velocity vector
                                v_orig = vel_all[s, i]  # (features,)
                                # Get original magnitude
                                magnitude_orig = v_orig.norm(p=2)
                                if magnitude_orig > 0:
                                    # Compute scale factor based on consistency
                                    scale_factor = consistency_scores[i] * max_velocity_magnitude / magnitude_orig
                                    # Scale velocity: maintain direction, scale magnitude
                                    vel_all[s, i] = v_orig * scale_factor
                    else:
                        # (cells, features)
                        for i in range(n_cells):
                            # Get original velocity vector
                            v_orig = vel_all[i]  # (features,)
                            # Get original magnitude
                            magnitude_orig = v_orig.norm(p=2)
                            if magnitude_orig > 0:
                                # Compute scale factor based on consistency
                                scale_factor = consistency_scores[i] * max_velocity_magnitude / magnitude_orig
                                # Scale velocity: maintain direction, scale magnitude
                                vel_all[i] = v_orig * scale_factor
                    
                    # Also rescale velocity_gp if available (using differences in latent space)
                    if velgp_all is not None:
                        # Get current latent state (mean)
                        if "mean" in adata.obsm:
                            current_latent_state = torch.from_numpy(adata.obsm["mean"].astype(np.float32)).to(vel_mean.device)  # (cells, L)
                            
                            if velgp_all.ndim == 3:
                                velgp_mean = velgp_all.mean(dim=0)  # (cells, features)
                            else:
                                velgp_mean = velgp_all  # (cells, features)
                            
                            # Compute consistency scores for GP velocities
                            gp_consistency_scores = torch.zeros(n_cells, device=vel_mean.device, dtype=vel_mean.dtype)
                            
                            for i in range(n_cells):
                                # Get cell's GP velocity direction (skip normalize if zero to avoid NaN)
                                vgp_i = velgp_mean[i]  # (L,)
                                vgp_norm_val = vgp_i.norm(p=2).item()
                                if vgp_norm_val < 1e-9:
                                    gp_consistency_scores[i] = 0.0
                                    continue
                                vgp_i_norm = F.normalize(vgp_i.unsqueeze(0), p=2, dim=1).squeeze(0)  # (L,)
                                
                                # Get neighbor indices
                                neighbor_indices = nn_indices_tensor[i, 1:K_total]
                                valid_mask = neighbor_indices >= 0
                                valid_indices = neighbor_indices[valid_mask]
                                
                                if valid_indices.numel() == 0:
                                    gp_consistency_scores[i] = 0.0
                                    continue
                                
                                # Get current latent state for cell and neighbors
                                z_i = current_latent_state[i]  # (L,)
                                z_neigh = current_latent_state[valid_indices]  # (K_valid, L)
                                
                                # Compute differences to neighbors in latent space
                                diffs_latent = z_neigh - z_i.unsqueeze(0)  # (K_valid, L)
                                
                                # Normalize differences
                                diffs_latent_norm = F.normalize(diffs_latent, p=2, dim=1)  # (K_valid, L)
                                
                                # Compare GP velocity direction with neighbor differences
                                cos_sims_gp = F.cosine_similarity(
                                    vgp_i_norm.unsqueeze(0),
                                    diffs_latent_norm,
                                    dim=1
                                )  # (K_valid,)
                                
                                # Use mean cosine similarity across neighbors as consistency score
                                mean_sim_gp = cos_sims_gp.mean()
                                consistency_gp = (mean_sim_gp + 1.0) / 2.0
                                gp_consistency_scores[i] = consistency_gp.clamp(0.0, 1.0)
                            
                            # Apply GP consistency scores to velocity_gp
                            if velgp_all.ndim == 3:
                                for s in range(velgp_all.shape[0]):
                                    for i in range(n_cells):
                                        vgp_orig = velgp_all[s, i]
                                        magnitude_orig = vgp_orig.norm(p=2)
                                        if magnitude_orig > 0:
                                            scale_factor = gp_consistency_scores[i] * max_velocity_magnitude / magnitude_orig
                                            velgp_all[s, i] = vgp_orig * scale_factor
                            else:
                                for i in range(n_cells):
                                    vgp_orig = velgp_all[i]
                                    magnitude_orig = vgp_orig.norm(p=2)
                                    if magnitude_orig > 0:
                                        scale_factor = gp_consistency_scores[i] * max_velocity_magnitude / magnitude_orig
                                        velgp_all[i] = vgp_orig * scale_factor
                        else:
                            # If mean not available, use same consistency scores as gene velocities
                            if velgp_all.ndim == 3:
                                for s in range(velgp_all.shape[0]):
                                    for i in range(n_cells):
                                        vgp_orig = velgp_all[s, i]
                                        magnitude_orig = vgp_orig.norm(p=2)
                                        if magnitude_orig > 0:
                                            scale_factor = consistency_scores[i] * max_velocity_magnitude / magnitude_orig
                                            velgp_all[s, i] = vgp_orig * scale_factor
                            else:
                                for i in range(n_cells):
                                    vgp_orig = velgp_all[i]
                                    magnitude_orig = vgp_orig.norm(p=2)
                                    if magnitude_orig > 0:
                                        scale_factor = consistency_scores[i] * max_velocity_magnitude / magnitude_orig
                                        velgp_all[i] = vgp_orig * scale_factor

        # split velocity into u/s in NumPy space
        vel_u_np = vel_s_np = None
        if vel_all is not None:
            vel_np = vel_all.numpy()
            vel_u_np, vel_s_np = np.split(vel_np, 2, axis=-1)

        # Restore original regime before returning
        self.first_regime = old_regime

        if not save_to_adata:
            # Extract cluster embeddings if available
            cluster_embeddings = None
            cluster_names = None
            if self.cluster_embedding is not None:
                with torch.no_grad():
                    cluster_embeddings = self.cluster_embedding.get_all_embeddings().cpu().numpy()
                cluster_names = self.cluster_names
            
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
                "cluster_embeddings": cluster_embeddings,
                "cluster_names": cluster_names,
            }

        # ---------------------------
        # save_to_adata=True path
        # ---------------------------
        # If multiple samples, FORCE averaging before writing (overrules return_mean)
        force_mean = (n_samples > 1)

        def _maybe_mean_first_axis(t: Optional[torch.Tensor]) -> Optional[np.ndarray]:
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
        def _maybe_mean_np_first_axis(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
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
            if Vgp.size > 0 and (np.abs(Vgp).max() < 1e-12 or np.allclose(Vgp, 0.0)):
                import warnings
                warnings.warn(
                    "velocity_gp is effectively all zeros. Possible causes: (1) get_model_outputs was not re-run "
                    "after updating the code that computes velocity_gp from gene velocity; (2) gene-level velocity "
                    "or gene decoder weights are zero. Re-run get_model_outputs(adata, save_to_adata=True) with "
                    "the updated lineagevi; if using a saved checkpoint, ensure the full model (including gene decoder) "
                    "is loaded.",
                    UserWarning,
                    stacklevel=2,
                )
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

        # Save cluster embeddings to adata.uns if cluster embeddings are enabled
        if self.cluster_embedding is not None:
            with torch.no_grad():
                cluster_embeddings = self.cluster_embedding.get_all_embeddings().cpu().numpy()
            adata.uns["cluster_embeddings"] = cluster_embeddings
            adata.uns["cluster_names"] = self.cluster_names

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
        base_seed: Optional[int] = None,
    ):
        """
        Compute directional uncertainty in velocity predictions.
        
        This method quantifies the uncertainty in velocity direction by sampling
        multiple velocity fields and computing directional statistics including
        variance in velocity directions and cosine similarities.
        
        Parameters
        ----------
        adata : AnnData
            Single-cell data to analyze.
        use_gp_velo : bool, default False
            Whether to use gene program velocities (True) or gene-level velocities (False).
        n_samples : int, default 50
            Number of velocity samples for uncertainty estimation.
        n_jobs : int, default -1
            Number of parallel jobs for computation (-1 uses all cores).
        show_plot : bool, default True
            Whether to display uncertainty plot.
        base_seed : int, optional
            Random seed for reproducible sampling.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with uncertainty metrics for each cell:
            - 'directional_variance': Variance in velocity direction
            - 'directional_difference': Mean absolute difference in velocity direction
            - 'directional_cosine_sim_variance': Variance in cosine similarity
        np.ndarray
            Cosine similarity matrix between velocity samples.
        
        Notes
        -----
        The method computes uncertainty by:
        1. Sampling n_samples velocity fields
        2. Computing directional statistics for each cell
        3. Quantifying variance in velocity directions
        4. Adding results to adata.obs with log10 transformation
        """
        # draw n_samples velocity fields in one call (no averaging)
        outs = self._get_model_outputs(
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
        """
        Compute directional statistics for velocity tensor using parallel processing.
        
        Parameters
        ----------
        tensor : np.ndarray
            Velocity tensor of shape (n_samples, n_cells, n_features).
        n_jobs : int
            Number of parallel jobs for computation.
        n_cells : int
            Number of cells to process.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with directional statistics for each cell:
            - 'directional_variance': Variance in velocity direction
            - 'directional_difference': Mean absolute difference in velocity direction
            - 'directional_cosine_sim_variance': Variance in cosine similarity
            - 'directional_cosine_sim_difference': Mean absolute difference in cosine similarity
            - 'directional_cosine_sim_mean': Mean cosine similarity
        """
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
        """
        Compute centered unit vector.
        
        Parameters
        ----------
        vector : np.ndarray
            Input vector to normalize.
        
        Returns
        -------
        np.ndarray
            Centered unit vector (vector - mean) / ||vector - mean||.
        """
        vector = vector - np.mean(vector)
        return vector / np.linalg.norm(vector)

    def _cosine_sim(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between two vectors.
        
        Parameters
        ----------
        v1 : np.ndarray
            First vector.
        v2 : np.ndarray
            Second vector.
        
        Returns
        -------
        np.ndarray
            Cosine similarity between the centered unit vectors, clipped to [-1, 1].
        """
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
        base_seed: Optional[int] = None,   # ensures distinct ε across iterations (while reproducible)
    ) -> pd.DataFrame:
        """
        Compute extrinsic uncertainty in velocity predictions.
        
        This method quantifies uncertainty by measuring how much velocity predictions
        change when the model is retrained with different random seeds, providing
        an estimate of model stability and reliability.
        
        Parameters
        ----------
        adata : AnnData
            Single-cell data to analyze.
        use_gp_velo : bool, default False
            Whether to use gene program velocities (True) or gene-level velocities (False).
        n_samples : int, default 25
            Number of model retraining samples for uncertainty estimation.
        n_jobs : int, default -1
            Number of parallel jobs for computation (-1 uses all cores).
        show_plot : bool, default True
            Whether to display uncertainty plot.
        base_seed : int, optional
            Base random seed for reproducible sampling.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with extrinsic uncertainty metrics for each cell:
            - 'extrinsic_variance': Variance across retrained models
            - 'extrinsic_difference': Mean absolute difference across models
            - 'extrinsic_cosine_sim_variance': Variance in cosine similarity
            - 'extrinsic_cosine_sim_difference': Mean absolute difference in cosine similarity
            - 'extrinsic_cosine_sim_mean': Mean cosine similarity
        
        Notes
        -----
        This method retrains the model n_samples times with different seeds and
        compares the resulting velocity predictions to quantify model stability.
        Results are added to adata.obs with log10 transformation.
        """
        import scvelo as scv
        import scanpy as sc
        from contextlib import redirect_stdout
        import io
        import numpy as np
        import pandas as pd
        from .utils import build_gp_adata

        # choose the working AnnData and state space
        if use_gp_velo:
            # work in GP space (cells × L)
            # First ensure we have model outputs in adata
            if "mean" not in adata.obsm or "velocity_gp" not in adata.obsm:
                # Get model outputs and save to adata
                self._get_model_outputs(
                    adata=adata,
                    n_samples=1,
                    return_mean=True,
                    return_negative_velo=True,
                    base_seed=base_seed,
                    save_to_adata=True,
                )
            working_adata = build_gp_adata(adata)
            # state used for extrapolation in this space
            state_matrix = working_adata.layers["Ms"]               # (cells, L) == μ
            # function to fetch the matching velocity each iteration
            def _fetch_velocity(i_seed):
                outs = self._get_model_outputs(
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
                outs = self._get_model_outputs(
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
    
    def _get_group_idxs(self, adata, groupby_key):
        """
        Get cell indices grouped by any categorical variable.
        
        Parameters
        ----------
        adata : AnnData
            Single-cell data with group annotations.
        groupby_key : str
            Key in adata.obs containing group information (e.g., 'cell_type', 'cluster', 'condition').
        
        Returns
        -------
        dict
            Dictionary mapping group names to arrays of cell indices.
        """
        group_indices = {}
        adata.obs['numerical_idx_linvi'] = np.arange(len(adata))
        for group, df in adata.obs.groupby(groupby_key, observed=True):
            group_indices[group] = df['numerical_idx_linvi'].to_numpy()

        return group_indices
        
    
    def _get_gene_idxs(self, adata, genes):
        """
        Get gene indices for specified gene names.
        
        Parameters
        ----------
        adata : AnnData
            Single-cell data with gene names in adata.var_names.
        genes : list or array-like
            Gene names to find indices for.
        
        Returns
        -------
        np.ndarray
            Array of gene indices where the specified genes are found.
        """
        return np.where(adata.var_names.isin(genes))[0]
    
    def _get_gp_idxs(self, adata, gp_key, gps):
        """
        Get gene program indices for specified gene program names.
        
        Parameters
        ----------
        adata : AnnData
            Single-cell data with gene program information in adata.uns.
        gp_key : str
            Key in adata.uns containing gene program names.
        gps : list or array-like
            Gene program names to find indices for.
        
        Returns
        -------
        np.ndarray
            Array of gene program indices where the specified programs are found.
        """
        return np.where(pd.Series(adata.uns[gp_key]).isin(gps))[0]
    
    @torch.inference_mode()
    def perturb_genes(
            self, 
            adata, 
            groupby_key, 
            group_to_perturb, 
            genes_to_perturb, 
            perturb_value,
            perturb_spliced = True,
            perturb_unspliced = False,
            perturb_both = False):
        """
        Perturb gene expression in specific groups and analyze velocity changes.
        
        This method artificially modifies gene expression levels in specified groups
        and genes, then analyzes how these perturbations affect velocity predictions.
        Useful for studying the sensitivity of velocity predictions to expression changes.
        
        Parameters
        ----------
        adata : AnnData
            Single-cell data to perturb.
        groupby_key : str
            Key in adata.obs containing group information (e.g., 'cell_type', 'cluster', 'condition').
        group_to_perturb : str
            Name of group to perturb.
        genes_to_perturb : list
            List of gene names to perturb.
        perturb_value : float
            Value to add to gene expression (can be negative for downregulation).
        perturb_spliced : bool, default True
            Whether to perturb spliced counts.
        perturb_unspliced : bool, default False
            Whether to perturb unspliced counts.
        perturb_both : bool, default False
            Whether to perturb both spliced and unspliced counts.
        
        Returns
        -------
        df_genes : pd.DataFrame
            DataFrame with gene-level differences (velocity, alpha, beta, gamma, recon, etc.)
            and Wilcoxon signed-rank p-values + FDR-adjusted p-values per entity
            (pval_*, padj_* columns).
        df_gp : pd.DataFrame
            DataFrame with GP-level differences (mean, logvar, gp_velocity) and
            Wilcoxon p-values + FDR (pval_*, padj_*).
        perturbed_outputs : dict
            Dictionary of perturbed model outputs (recon, mean, logvar, velocity_gp, velo_u_pert, velo_pert, alpha_pert, beta_pert, gamma_pert).
        
        Notes
        -----
        Perturbed outputs are stored in adata.obsm and adata.layers with ``_pert`` suffix.
        The method:
        1. Identifies cells of the specified group
        2. Perturbs expression of specified genes
        3. Computes velocity predictions on perturbed data
        4. Stores results for comparison with original predictions
        """
        
        perturbed_genes_idxs = self._get_gene_idxs(adata, genes_to_perturb)
        group_idxs = self._get_group_idxs(adata, groupby_key=groupby_key)
        cells_to_perturb_idxs = group_idxs[group_to_perturb]

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

        # Convert to torch tensors on model device (avoid CPU/CUDA mismatch)
        device = next(self.parameters()).device
        x_unpert = torch.as_tensor(mu_ms_unpert, dtype=torch.float32, device=device)
        x_pert = torch.as_tensor(mu_ms_pert, dtype=torch.float32, device=device)

        self.first_regime = False
        
        # Get cluster indices if cluster_key is set
        cluster_indices_unpert = None
        cluster_indices_pert = None
        if self.cluster_key is not None and self.cluster_key in adata.obs.columns:
            # Get cluster labels for perturbed cells
            cluster_labels_unpert = adata.obs[self.cluster_key].iloc[cell_idx]
            cluster_labels_pert = adata.obs[self.cluster_key].iloc[cell_idx]
            # Map to indices (on same device as model)
            cluster_indices_unpert = torch.tensor([
                self.cluster_to_idx.get(str(label), 0) for label in cluster_labels_unpert
            ], dtype=torch.long, device=device)
            cluster_indices_pert = torch.tensor([
                self.cluster_to_idx.get(str(label), 0) for label in cluster_labels_pert
            ], dtype=torch.long, device=device)

        out_unpert = self.forward(x_unpert, cluster_indices=cluster_indices_unpert)
        out_pert = self.forward(x_pert, cluster_indices=cluster_indices_pert)

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

        # Convert to numpy and ensure 2D (n_cells, n_features) for Wilcoxon
        def to_2d(t):
            a = t.cpu().numpy()
            return np.atleast_2d(a)

        recon_2d = to_2d(recon_diff)
        mean_2d = to_2d(mean_diff)
        logvar_2d = to_2d(logvar_diff)
        gp_velo_2d = to_2d(gp_velo_diff)
        velo_u_2d = to_2d(velo_u_diff)
        velo_2d = to_2d(velo_diff)
        alpha_2d = to_2d(alpha_diff)
        beta_2d = to_2d(beta_diff)
        gamma_2d = to_2d(gamma_diff)

        # Mean differences (for existing columns)
        mean_diff_np = mean_2d.mean(0)
        logvar_diff_np = logvar_2d.mean(0)
        gp_velo_diff_np = gp_velo_2d.mean(0)
        recon_diff_np = recon_2d.mean(0)
        velo_u_diff_np = velo_u_2d.mean(0)
        velo_diff_np = velo_2d.mean(0)
        alpha_diff_np = alpha_2d.mean(0)
        beta_diff_np = beta_2d.mean(0)
        gamma_diff_np = gamma_2d.mean(0)

        # Wilcoxon signed-rank + FDR per entity
        pval_mean = _wilcoxon_per_column(mean_2d)
        padj_mean = _fdr_bh(pval_mean)
        pval_logvar = _wilcoxon_per_column(logvar_2d)
        padj_logvar = _fdr_bh(pval_logvar)
        pval_gp_velo = _wilcoxon_per_column(gp_velo_2d)
        padj_gp_velo = _fdr_bh(pval_gp_velo)

        pval_recon = _wilcoxon_per_column(recon_2d)
        padj_recon = _fdr_bh(pval_recon)
        pval_velo_u = _wilcoxon_per_column(velo_u_2d)
        padj_velo_u = _fdr_bh(pval_velo_u)
        pval_velo = _wilcoxon_per_column(velo_2d)
        padj_velo = _fdr_bh(pval_velo)
        pval_alpha = _wilcoxon_per_column(alpha_2d)
        padj_alpha = _fdr_bh(pval_alpha)
        pval_beta = _wilcoxon_per_column(beta_2d)
        padj_beta = _fdr_bh(pval_beta)
        pval_gamma = _wilcoxon_per_column(gamma_2d)
        padj_gamma = _fdr_bh(pval_gamma)

        df_gp = pd.DataFrame({
            "terms": adata.uns["terms"],
            "mean_diff": mean_diff_np,
            "abs_mean_diff": np.abs(mean_diff_np),
            "logvar_diff": logvar_diff_np,
            "abs_logvar_diff": np.abs(logvar_diff_np),
            "gp_velocity_diff": gp_velo_diff_np,
            "abs_gp_velocity_diff": np.abs(gp_velo_diff_np),
            "pval_mean": pval_mean,
            "padj_mean": padj_mean,
            "pval_logvar": pval_logvar,
            "padj_logvar": padj_logvar,
            "pval_gp_velocity": pval_gp_velo,
            "padj_gp_velocity": padj_gp_velo,
        })

        df_genes = pd.DataFrame({
            "genes": adata.var_names,
            "recon_diff": recon_diff_np,
            "abs_recon_diff": np.abs(recon_diff_np),
            "unspliced_velocity_diff": velo_u_diff_np,
            "abs_unspliced_velocity_diff": np.abs(velo_u_diff_np),
            "velocity_diff": velo_diff_np,
            "abs_velocity_diff": np.abs(velo_diff_np),
            "alpha_diff": alpha_diff_np,
            "abs_alpha_diff": np.abs(alpha_diff_np),
            "beta_diff": beta_diff_np,
            "abs_beta_diff": np.abs(beta_diff_np),
            "gamma_diff": gamma_diff_np,
            "abs_gamma_diff": np.abs(gamma_diff_np),
            "pval_recon": pval_recon,
            "padj_recon": padj_recon,
            "pval_unspliced_velocity": pval_velo_u,
            "padj_unspliced_velocity": padj_velo_u,
            "pval_velocity": pval_velo,
            "padj_velocity": padj_velo,
            "pval_alpha": pval_alpha,
            "padj_alpha": padj_alpha,
            "pval_beta": pval_beta,
            "padj_beta": padj_beta,
            "pval_gamma": pval_gamma,
            "padj_gamma": padj_gamma,
        })

        # Store perturbed outputs in adata
        # For perturb_genes, outputs are computed only for perturbed cells
        # We need to handle full adata shape - initialize arrays if needed
        n_cells = adata.shape[0]
        n_genes = adata.shape[1]
        n_gps = len(perturbed_outputs['velocity_gp'][0]) if len(perturbed_outputs['velocity_gp'].shape) > 1 else 1
        
        # Initialize full arrays if they don't exist
        if 'velocity_gp_pert' not in adata.obsm:
            adata.obsm['velocity_gp_pert'] = np.zeros((n_cells, n_gps), dtype=np.float32)
        if 'velocity_u_pert' not in adata.layers:
            adata.layers['velocity_u_pert'] = np.zeros((n_cells, n_genes), dtype=np.float32)
        if 'velocity_pert' not in adata.layers:
            adata.layers['velocity_pert'] = np.zeros((n_cells, n_genes), dtype=np.float32)
        if 'alpha_pert' not in adata.layers:
            adata.layers['alpha_pert'] = np.zeros((n_cells, n_genes), dtype=np.float32)
        if 'beta_pert' not in adata.layers:
            adata.layers['beta_pert'] = np.zeros((n_cells, n_genes), dtype=np.float32)
        if 'gamma_pert' not in adata.layers:
            adata.layers['gamma_pert'] = np.zeros((n_cells, n_genes), dtype=np.float32)
        if 'recon_pert' not in adata.layers:
            adata.layers['recon_pert'] = np.zeros((n_cells, n_genes), dtype=np.float32)
        if 'mean_pert' not in adata.obsm:
            adata.obsm['mean_pert'] = np.zeros((n_cells, n_gps), dtype=np.float32)
        if 'logvar_pert' not in adata.obsm:
            adata.obsm['logvar_pert'] = np.zeros((n_cells, n_gps), dtype=np.float32)
        
        # Store perturbed outputs for perturbed cells
        if isinstance(cell_idx, (int, np.integer)) or (hasattr(cell_idx, '__len__') and len(cell_idx) == 1):
            # Single cell case
            idx = cell_idx if isinstance(cell_idx, (int, np.integer)) else cell_idx[0]
            # Handle both 1D and 2D outputs
            if len(perturbed_outputs['velocity_gp'].shape) == 1:
                adata.obsm['velocity_gp_pert'][idx] = perturbed_outputs['velocity_gp']
            else:
                adata.obsm['velocity_gp_pert'][idx] = perturbed_outputs['velocity_gp'][0]
            adata.layers['velocity_u_pert'][idx] = perturbed_outputs['velo_u_pert'][0] if len(perturbed_outputs['velo_u_pert'].shape) > 1 else perturbed_outputs['velo_u_pert']
            adata.layers['velocity_pert'][idx] = perturbed_outputs['velo_pert'][0] if len(perturbed_outputs['velo_pert'].shape) > 1 else perturbed_outputs['velo_pert']
            adata.layers['alpha_pert'][idx] = perturbed_outputs['alpha_pert'][0] if len(perturbed_outputs['alpha_pert'].shape) > 1 else perturbed_outputs['alpha_pert']
            adata.layers['beta_pert'][idx] = perturbed_outputs['beta_pert'][0] if len(perturbed_outputs['beta_pert'].shape) > 1 else perturbed_outputs['beta_pert']
            adata.layers['gamma_pert'][idx] = perturbed_outputs['gamma_pert'][0] if len(perturbed_outputs['gamma_pert'].shape) > 1 else perturbed_outputs['gamma_pert']
            adata.layers['recon_pert'][idx] = perturbed_outputs['recon'][0] if len(perturbed_outputs['recon'].shape) > 1 else perturbed_outputs['recon']
            if len(perturbed_outputs['mean'].shape) == 1:
                adata.obsm['mean_pert'][idx] = perturbed_outputs['mean']
            else:
                adata.obsm['mean_pert'][idx] = perturbed_outputs['mean'][0]
            if len(perturbed_outputs['logvar'].shape) == 1:
                adata.obsm['logvar_pert'][idx] = perturbed_outputs['logvar']
            else:
                adata.obsm['logvar_pert'][idx] = perturbed_outputs['logvar'][0]
        else:
            # Multiple cells case
            adata.obsm['velocity_gp_pert'][cell_idx] = perturbed_outputs['velocity_gp']
            adata.layers['velocity_u_pert'][cell_idx] = perturbed_outputs['velo_u_pert']
            adata.layers['velocity_pert'][cell_idx] = perturbed_outputs['velo_pert']
            adata.layers['alpha_pert'][cell_idx] = perturbed_outputs['alpha_pert']
            adata.layers['beta_pert'][cell_idx] = perturbed_outputs['beta_pert']
            adata.layers['gamma_pert'][cell_idx] = perturbed_outputs['gamma_pert']
            adata.layers['recon_pert'][cell_idx] = perturbed_outputs['recon']
            adata.obsm['mean_pert'][cell_idx] = perturbed_outputs['mean']
            adata.obsm['logvar_pert'][cell_idx] = perturbed_outputs['logvar']
        
        print("\nPerturbed outputs stored in adata:")
        print(f"  adata.obsm['velocity_gp_pert']: GP velocities (shape: {adata.obsm['velocity_gp_pert'].shape})")
        print(f"  adata.obsm['mean_pert']: Perturbed latent means (shape: {adata.obsm['mean_pert'].shape})")
        print(f"  adata.obsm['logvar_pert']: Perturbed latent logvars (shape: {adata.obsm['logvar_pert'].shape})")
        print(f"  adata.layers['velocity_u_pert']: Unspliced velocities (shape: {adata.layers['velocity_u_pert'].shape})")
        print(f"  adata.layers['velocity_pert']: Spliced velocities (shape: {adata.layers['velocity_pert'].shape})")
        print(f"  adata.layers['alpha_pert']: Transcription rates (shape: {adata.layers['alpha_pert'].shape})")
        print(f"  adata.layers['beta_pert']: Splicing rates (shape: {adata.layers['beta_pert'].shape})")
        print(f"  adata.layers['gamma_pert']: Degradation rates (shape: {adata.layers['gamma_pert'].shape})")
        print(f"  adata.layers['recon_pert']: Reconstructions (shape: {adata.layers['recon_pert'].shape})")
        print(f"  (Stored for {len(cell_idx) if hasattr(cell_idx, '__len__') else 1} perturbed cell(s))\n")

        return df_genes, df_gp, perturbed_outputs

    @torch.inference_mode()
    def perturb_gps(self, adata, gp_uns_key, gps_to_perturb, groupby_key, group_to_perturb, perturb_value):
        """
        Perturb gene program expression in specific groups and analyze velocity changes.
        
        This method artificially modifies gene program expression levels in specified groups
        and gene programs, then analyzes how these perturbations affect velocity predictions.
        Useful for studying the sensitivity of velocity predictions to gene program changes.
        
        Parameters
        ----------
        adata : AnnData
            Single-cell data to perturb.
        gp_uns_key : str
            Key in adata.uns containing gene program names.
        gps_to_perturb : list
            List of gene program names to perturb.
        groupby_key : str
            Key in adata.obs containing group information (e.g., 'cell_type', 'cluster', 'condition').
        group_to_perturb : str
            Name of group to perturb.
        perturb_value : float
            Value to replace gene program activation with (replaces the activation value for all cells in the group).
        
        Returns
        -------
        genes_df : pd.DataFrame
            DataFrame with gene-level differences (velocity, alpha, beta, gamma, x_dec, etc.)
            and Wilcoxon signed-rank p-values + FDR-adjusted p-values (pval_*, padj_*).
        gps_df : pd.DataFrame
            DataFrame with GP-level velocity difference and Wilcoxon p-value + FDR
            (pval_velo_gp, padj_velo_gp).
        
        Notes
        -----
        Perturbed outputs are stored in adata.obsm and adata.layers with ``_pert`` suffix.
        The method:
        1. Identifies cells of the specified group
        2. Perturbs expression of specified gene programs
        3. Computes velocity predictions on perturbed data
        4. Stores results for comparison with original predictions
        """
        group_idxs = self._get_group_idxs(adata, groupby_key=groupby_key)
        cell_idx = group_idxs[group_to_perturb]

        gp_idx = self._get_gp_idxs(adata, gp_uns_key, gps_to_perturb)

        mu = adata.layers['Mu'][cell_idx, :]
        ms = adata.layers['Ms'][cell_idx, :]

        mu_ms = torch.from_numpy(np.concatenate([mu, ms], axis=1)).float()
        
        # Get device
        device = next(self.parameters()).device
        mu_ms = mu_ms.to(device)

        self.first_regime = False
        
        # Get cluster indices if cluster_key is set
        cluster_indices = None
        if self.cluster_embedding is not None:
            # Cluster embeddings are enabled, so we need cluster indices
            if self.cluster_key is None or self.cluster_key not in adata.obs.columns:
                raise ValueError(
                    f"cluster_key '{self.cluster_key}' not found in adata.obs.columns. "
                    f"Available columns: {list(adata.obs.columns)}"
                )
            if not hasattr(self, 'cluster_to_idx') or self.cluster_to_idx is None:
                raise ValueError(
                    "cluster_to_idx not initialized. This should not happen if cluster_key was set during model initialization."
                )
            # Get cluster labels for perturbed cells
            # Handle both single index and array of indices
            if isinstance(cell_idx, (int, np.integer)) or (hasattr(cell_idx, '__len__') and len(cell_idx) == 1):
                # Single cell case
                if isinstance(cell_idx, (int, np.integer)):
                    idx = cell_idx
                else:
                    idx = cell_idx[0]
                cluster_labels = [adata.obs[self.cluster_key].iloc[idx]]
            else:
                # Multiple cells case
                cluster_labels = adata.obs[self.cluster_key].iloc[cell_idx]
                if hasattr(cluster_labels, 'values'):
                    cluster_labels = cluster_labels.values.tolist()
                else:
                    cluster_labels = list(cluster_labels)
            # Map to indices
            cluster_indices = torch.tensor([
                self.cluster_to_idx.get(str(label), 0) for label in cluster_labels
            ], dtype=torch.long, device=device)

        z_unpert, _, _ = self._forward_encoder(mu_ms)
        z_pert = z_unpert.clone()
        z_pert[:, gp_idx] = perturb_value
        velocity_unpert, velocity_gp_unpert, alpha_unpert, beta_unpert, gamma_unpert = self._forward_velocity_decoder(z_unpert, mu_ms, cluster_indices)
        velocity_pert, velocity_gp_pert, alpha_pert, beta_pert, gamma_pert = self._forward_velocity_decoder(z_pert, mu_ms, cluster_indices)
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

        velo_diff_2d = np.atleast_2d(to_numpy(velocity_pert - velocity_unpert))
        velo_gp_2d = np.atleast_2d(to_numpy(velocity_gp_pert - velocity_gp_unpert))
        alpha_2d = np.atleast_2d(to_numpy(alpha_pert - alpha_unpert))
        beta_2d = np.atleast_2d(to_numpy(beta_pert - beta_unpert))
        gamma_2d = np.atleast_2d(to_numpy(gamma_pert - gamma_unpert))
        x_dec_2d = np.atleast_2d(to_numpy(x_dec_pert - x_dec_unpert))

        velo_diff_mean = velo_diff_2d.mean(0)
        velo_gp_mean = velo_gp_2d.mean(0)
        alpha_mean = alpha_2d.mean(0)
        beta_mean = beta_2d.mean(0)
        gamma_mean = gamma_2d.mean(0)
        x_dec_mean = x_dec_2d.mean(0)

        velo_diff_u, velo_diff_s = np.split(velo_diff_mean, 2)

        n_genes = alpha_2d.shape[1]
        velo_u_2d = velo_diff_2d[:, :n_genes]
        velo_s_2d = velo_diff_2d[:, n_genes:]

        pval_velo_u = _wilcoxon_per_column(velo_u_2d)
        padj_velo_u = _fdr_bh(pval_velo_u)
        pval_velo_s = _wilcoxon_per_column(velo_s_2d)
        padj_velo_s = _fdr_bh(pval_velo_s)
        pval_x_dec = _wilcoxon_per_column(x_dec_2d)
        padj_x_dec = _fdr_bh(pval_x_dec)
        pval_alpha = _wilcoxon_per_column(alpha_2d)
        padj_alpha = _fdr_bh(pval_alpha)
        pval_beta = _wilcoxon_per_column(beta_2d)
        padj_beta = _fdr_bh(pval_beta)
        pval_gamma = _wilcoxon_per_column(gamma_2d)
        padj_gamma = _fdr_bh(pval_gamma)
        pval_velo_gp = _wilcoxon_per_column(velo_gp_2d)
        padj_velo_gp = _fdr_bh(pval_velo_gp)

        genes_df = pd.DataFrame({
            "genes": adata.var_names,
            "velo_diff_u": velo_diff_u,
            "abs_velo_diff_u": np.abs(velo_diff_u),
            "velo_diff_s": velo_diff_s,
            "abs_velo_diff_s": np.abs(velo_diff_s),
            "x_dec_diff": x_dec_mean,
            "x_dec_diff_abs": np.abs(x_dec_mean),
            "alpha_diff": alpha_mean,
            "alpha_diff_abs": np.abs(alpha_mean),
            "beta_diff": beta_mean,
            "beta_diff_abs": np.abs(beta_mean),
            "gamma_diff": gamma_mean,
            "gamma_diff_abs": np.abs(gamma_mean),
            "pval_velo_u": pval_velo_u,
            "padj_velo_u": padj_velo_u,
            "pval_velo_s": pval_velo_s,
            "padj_velo_s": padj_velo_s,
            "pval_x_dec": pval_x_dec,
            "padj_x_dec": padj_x_dec,
            "pval_alpha": pval_alpha,
            "padj_alpha": padj_alpha,
            "pval_beta": pval_beta,
            "padj_beta": padj_beta,
            "pval_gamma": pval_gamma,
            "padj_gamma": padj_gamma,
        })

        gps_df = pd.DataFrame({
            "gene_programs": adata.uns["terms"],
            "velo_gp": velo_gp_mean,
            "abs_velo_gp": np.abs(velo_gp_mean),
            "pval_velo_gp": pval_velo_gp,
            "padj_velo_gp": padj_velo_gp,
        })
        
        # Store perturbed outputs in adata
        # For perturb_gps, outputs are computed only for perturbed cells
        # We need to handle full adata shape - initialize arrays if needed
        n_cells = adata.shape[0]
        n_genes = adata.shape[1]
        n_gps = len(perturbed_outputs['velocity_gp_pert'][0]) if len(perturbed_outputs['velocity_gp_pert'].shape) > 1 else 1
        
        # Initialize full arrays if they don't exist
        if 'velocity_gp_pert' not in adata.obsm:
            adata.obsm['velocity_gp_pert'] = np.zeros((n_cells, n_gps), dtype=np.float32)
        if 'velocity_u_pert' not in adata.layers:
            adata.layers['velocity_u_pert'] = np.zeros((n_cells, n_genes), dtype=np.float32)
        if 'velocity_pert' not in adata.layers:
            adata.layers['velocity_pert'] = np.zeros((n_cells, n_genes), dtype=np.float32)
        if 'alpha_pert' not in adata.layers:
            adata.layers['alpha_pert'] = np.zeros((n_cells, n_genes), dtype=np.float32)
        if 'beta_pert' not in adata.layers:
            adata.layers['beta_pert'] = np.zeros((n_cells, n_genes), dtype=np.float32)
        if 'gamma_pert' not in adata.layers:
            adata.layers['gamma_pert'] = np.zeros((n_cells, n_genes), dtype=np.float32)
        if 'recon_pert' not in adata.layers:
            adata.layers['recon_pert'] = np.zeros((n_cells, n_genes), dtype=np.float32)
        
        # Store perturbed outputs for perturbed cells
        if isinstance(cell_idx, (int, np.integer)) or (hasattr(cell_idx, '__len__') and len(cell_idx) == 1):
            # Single cell case
            idx = cell_idx if isinstance(cell_idx, (int, np.integer)) else cell_idx[0]
            adata.obsm['velocity_gp_pert'][idx] = perturbed_outputs['velocity_gp_pert'][0] if len(perturbed_outputs['velocity_gp_pert'].shape) > 1 else perturbed_outputs['velocity_gp_pert']
            adata.layers['velocity_u_pert'][idx] = perturbed_outputs['velo_u_pert'][0] if len(perturbed_outputs['velo_u_pert'].shape) > 1 else perturbed_outputs['velo_u_pert']
            adata.layers['velocity_pert'][idx] = perturbed_outputs['velo_pert'][0] if len(perturbed_outputs['velo_pert'].shape) > 1 else perturbed_outputs['velo_pert']
            adata.layers['alpha_pert'][idx] = perturbed_outputs['alpha_pert'][0] if len(perturbed_outputs['alpha_pert'].shape) > 1 else perturbed_outputs['alpha_pert']
            adata.layers['beta_pert'][idx] = perturbed_outputs['beta_pert'][0] if len(perturbed_outputs['beta_pert'].shape) > 1 else perturbed_outputs['beta_pert']
            adata.layers['gamma_pert'][idx] = perturbed_outputs['gamma_pert'][0] if len(perturbed_outputs['gamma_pert'].shape) > 1 else perturbed_outputs['gamma_pert']
            adata.layers['recon_pert'][idx] = perturbed_outputs['recon'][0] if len(perturbed_outputs['recon'].shape) > 1 else perturbed_outputs['recon']
        else:
            # Multiple cells case
            adata.obsm['velocity_gp_pert'][cell_idx] = perturbed_outputs['velocity_gp_pert']
            adata.layers['velocity_u_pert'][cell_idx] = perturbed_outputs['velo_u_pert']
            adata.layers['velocity_pert'][cell_idx] = perturbed_outputs['velo_pert']
            adata.layers['alpha_pert'][cell_idx] = perturbed_outputs['alpha_pert']
            adata.layers['beta_pert'][cell_idx] = perturbed_outputs['beta_pert']
            adata.layers['gamma_pert'][cell_idx] = perturbed_outputs['gamma_pert']
            adata.layers['recon_pert'][cell_idx] = perturbed_outputs['recon']
        
        print("\nPerturbed outputs stored in adata:")
        print(f"  adata.obsm['velocity_gp_pert']: GP velocities (shape: {adata.obsm['velocity_gp_pert'].shape})")
        print(f"  adata.layers['velocity_u_pert']: Unspliced velocities (shape: {adata.layers['velocity_u_pert'].shape})")
        print(f"  adata.layers['velocity_pert']: Spliced velocities (shape: {adata.layers['velocity_pert'].shape})")
        print(f"  adata.layers['alpha_pert']: Transcription rates (shape: {adata.layers['alpha_pert'].shape})")
        print(f"  adata.layers['beta_pert']: Splicing rates (shape: {adata.layers['beta_pert'].shape})")
        print(f"  adata.layers['gamma_pert']: Degradation rates (shape: {adata.layers['gamma_pert'].shape})")
        print(f"  adata.layers['recon_pert']: Reconstructions (shape: {adata.layers['recon_pert'].shape})")
        print(f"  (Stored for {len(cell_idx) if hasattr(cell_idx, '__len__') and not isinstance(cell_idx, (int, np.integer)) else 1} perturbed cell(s))\n")
        
        return genes_df, gps_df

    @torch.inference_mode()
    def perturb_cluster_labels(self, adata, source_cluster, target_cluster):
        """
        Perturb cluster embeddings by swapping embeddings between two clusters.
        
        This perturbation swaps the cluster embeddings in the model, computes
        velocity predictions with the swapped embeddings, and then restores
        the original embeddings. This allows evaluation of how cluster-specific
        dynamics affect velocity predictions.
        
        Parameters
        ----------
        adata : AnnData
            Single-cell data to compute velocities on.
        source_cluster : str
            Cluster label whose embedding will be swapped.
        target_cluster : str
            Cluster label whose embedding will be swapped with source_cluster.
        
        Returns
        -------
        genes_df : pd.DataFrame
            DataFrame with gene-level differences (velocity, alpha, beta, gamma, etc.).
        gps_df : pd.DataFrame
            DataFrame with GP-level differences (GP velocity, etc.).
        
        Notes
        -----
        Perturbed outputs are stored in adata.obsm and adata.layers with ``_pert`` suffix.
        """
        import torch
        import numpy as np
        import pandas as pd
        
        if source_cluster is None or target_cluster is None:
            raise ValueError("Both source_cluster and target_cluster must be provided")
        
        if self.cluster_embedding is None:
            raise ValueError("Cluster embeddings are not enabled in this model")
        
        if self.cluster_to_idx is None:
            raise ValueError("Model does not have cluster_to_idx mapping")
        
        # Get cluster indices
        if source_cluster not in self.cluster_to_idx:
            raise ValueError(
                f"Source cluster '{source_cluster}' not found in model. "
                f"Available clusters: {list(self.cluster_to_idx.keys())}"
            )
        if target_cluster not in self.cluster_to_idx:
            raise ValueError(
                f"Target cluster '{target_cluster}' not found in model. "
                f"Available clusters: {list(self.cluster_to_idx.keys())}"
            )
        
        source_idx = self.cluster_to_idx[source_cluster]
        target_idx = self.cluster_to_idx[target_cluster]
        
        # Get required data
        mu = adata.layers['Mu']
        ms = adata.layers['Ms']
        mu_ms = torch.from_numpy(np.concatenate([mu, ms], axis=1)).float()
        
        # Get device
        device = next(self.parameters()).device
        mu_ms = mu_ms.to(device)
        
        self.first_regime = False
        
        # Get cluster indices for all cells
        cluster_labels = adata.obs[self.cluster_key]
        cluster_indices_unpert = torch.tensor([
            self.cluster_to_idx.get(str(label), 0) for label in cluster_labels
        ], dtype=torch.long, device=device)

        # Save original embeddings
        embeddings = self.cluster_embedding.embeddings.weight
        original_source_emb = embeddings[source_idx].clone()
        original_target_emb = embeddings[target_idx].clone()
        
        try:
            # Compute velocities with original embeddings
            z_unpert, _, _ = self._forward_encoder(mu_ms)
            velocity_unpert, velocity_gp_unpert, alpha_unpert, beta_unpert, gamma_unpert = self._forward_velocity_decoder(
                z_unpert, mu_ms, cluster_indices_unpert
            )
            x_dec_unpert = self._forward_gene_decoder(z_unpert)
            
            # Swap cluster embeddings
            with torch.no_grad():
                embeddings[source_idx].copy_(original_target_emb)
                embeddings[target_idx].copy_(original_source_emb)
            
            # Compute velocities with swapped embeddings
            # Note: cluster_indices remain the same, but the embeddings they point to are swapped
            velocity_pert, velocity_gp_pert, alpha_pert, beta_pert, gamma_pert = self._forward_velocity_decoder(
                z_unpert, mu_ms, cluster_indices_unpert
            )
            x_dec_pert = self._forward_gene_decoder(z_unpert)
            
            # Convert to numpy
            to_numpy = lambda x: x.cpu().numpy()
            
            velo_u_unpert, velo_unpert = np.split(to_numpy(velocity_unpert), 2, axis=1)
            velo_u_pert, velo_pert = np.split(to_numpy(velocity_pert), 2, axis=1)
            
            perturbed_outputs = {
                'velocity_gp_pert': to_numpy(velocity_gp_pert),
                'velo_u_pert': velo_u_pert,
                'velo_pert': velo_pert,
                'alpha_pert': to_numpy(alpha_pert),
                'beta_pert': to_numpy(beta_pert),
                'gamma_pert': to_numpy(gamma_pert),
                'recon': to_numpy(x_dec_pert),
            }
            
            # Compute differences
            velo_diff = to_numpy(velocity_pert - velocity_unpert)
            velo_gp_diff = to_numpy(velocity_gp_pert - velocity_gp_unpert)
            alpha_diff = to_numpy(alpha_pert - alpha_unpert)
            beta_diff = to_numpy(beta_pert - beta_unpert)
            gamma_diff = to_numpy(gamma_pert - gamma_unpert)
            x_dec_diff = to_numpy(x_dec_pert - x_dec_unpert)
            
            # Average across cells
            if velo_diff.shape[0] > 1:
                velo_diff = velo_diff.mean(0)
                velo_gp_diff = velo_gp_diff.mean(0)
                alpha_diff = alpha_diff.mean(0)
                beta_diff = beta_diff.mean(0)
                gamma_diff = gamma_diff.mean(0)
                x_dec_diff = x_dec_diff.mean(0)
            
            velo_diff_u, velo_diff_s = np.split(velo_diff, 2)
            
            # Create DataFrames
            genes_df = pd.DataFrame({
                'genes': adata.var_names,
                'velo_diff_u': velo_diff_u,
                'abs_velo_diff_u': np.absolute(velo_diff_u),
                'velo_diff_s': velo_diff_s,
                'abs_velo_diff_s': np.absolute(velo_diff_s),
                'x_dec_diff': x_dec_diff,
                'x_dec_diff_abs': np.absolute(x_dec_diff),
                'alpha_diff': alpha_diff,
                'alpha_diff_abs': np.absolute(alpha_diff),
                'beta_diff': beta_diff,
                'beta_diff_abs': np.absolute(beta_diff),
                'gamma_diff': gamma_diff,
                'gamma_diff_abs': np.absolute(gamma_diff),
            })
            
            gps_df = pd.DataFrame({
                'gene_programs': adata.uns['terms'],
                'velo_gp': velo_gp_diff,
                'abs_velo_gp': np.absolute(velo_gp_diff),
            })
            
            # Store perturbed outputs in adata
            # For perturb_cluster_labels, outputs are computed for ALL cells
            adata.obsm['velocity_gp_pert'] = perturbed_outputs['velocity_gp_pert'].astype(np.float32)
            adata.layers['velocity_u_pert'] = perturbed_outputs['velo_u_pert'].astype(np.float32)
            adata.layers['velocity_pert'] = perturbed_outputs['velo_pert'].astype(np.float32)
            adata.layers['alpha_pert'] = perturbed_outputs['alpha_pert'].astype(np.float32)
            adata.layers['beta_pert'] = perturbed_outputs['beta_pert'].astype(np.float32)
            adata.layers['gamma_pert'] = perturbed_outputs['gamma_pert'].astype(np.float32)
            adata.layers['recon_pert'] = perturbed_outputs['recon'].astype(np.float32)
            
            print("\nPerturbed outputs stored in adata:")
            print(f"  adata.obsm['velocity_gp_pert']: GP velocities (shape: {adata.obsm['velocity_gp_pert'].shape})")
            print(f"  adata.layers['velocity_u_pert']: Unspliced velocities (shape: {adata.layers['velocity_u_pert'].shape})")
            print(f"  adata.layers['velocity_pert']: Spliced velocities (shape: {adata.layers['velocity_pert'].shape})")
            print(f"  adata.layers['alpha_pert']: Transcription rates (shape: {adata.layers['alpha_pert'].shape})")
            print(f"  adata.layers['beta_pert']: Splicing rates (shape: {adata.layers['beta_pert'].shape})")
            print(f"  adata.layers['gamma_pert']: Degradation rates (shape: {adata.layers['gamma_pert'].shape})")
            print(f"  adata.layers['recon_pert']: Reconstructions (shape: {adata.layers['recon_pert'].shape})")
            print(f"  (Stored for all {adata.shape[0]} cells)\n")
            
        finally:
            # Always restore original embeddings
            with torch.no_grad():
                embeddings[source_idx].copy_(original_source_emb)
                embeddings[target_idx].copy_(original_target_emb)
        
        return genes_df, gps_df

    @torch.inference_mode()
    def map_velocities(
        self,
        adata,
        adata_gp=None,
        direction: str = "gp_to_gene",
        scale: float = 10.0,
        velocity_key: str = "mapped_velocity",
        unspliced_key: str = "Mu",
        spliced_key: str = "Ms",
    ):
        """
        Map velocities between gene program space and gene expression space.
        
        This method uses pre-computed velocities and latent representations from get_model_outputs().
        
        Parameters
        ----------
        adata : AnnData
            The AnnData object containing pre-computed velocities and latent representations.
        adata_gp : AnnData, optional
            Gene program AnnData object. Required for direction="gene_to_gp".
            Will be modified in place with mapped velocities.
        direction : str, default "gp_to_gene"
            Direction of mapping: "gp_to_gene" or "gene_to_gp".
        scale : float, default 10.0
            Scaling factor for the mapped velocities.
        velocity_key : str, default "mapped_velocity"
            Key to store mapped velocities in adata.layers (gp_to_gene) or adata.obsm (gene_to_gp).
        unspliced_key : str, default "Mu"
            Key for unspliced counts in adata.layers.
        spliced_key : str, default "Ms"
            Key for spliced counts in adata.layers.
            
        Returns
        -------
        None
            Mapped velocities are saved to adata and adata_gp (if provided).
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
                adata, scale, velocity_key, unspliced_key, spliced_key
            )
        else:
            # Map from gene velocity to GP velocity
            return self._map_gene_to_gp_velocity(
                adata, adata_gp, scale, velocity_key
            )
    
    def _map_gp_to_gene_velocity(
        self,
        adata,
        scale: float,
        velocity_key: str,
        unspliced_key: str,
        spliced_key: str,
    ):
        """Map velocities from gene program space to gene expression space."""
        import scvelo as scv
        from contextlib import redirect_stdout
        import io
        
        # Check for required pre-computed results
        required_layers = [spliced_key]
        _check_velocity_layers(adata, "map_velocities (gp_to_gene)", required_layers)
        
        # Get pre-computed GP velocities and latent representations
        if "velocity_gp" not in adata.obsm:
            raise ValueError("velocity_gp not found in adata.obsm. Please run get_model_outputs() first.")
        if "mean" not in adata.obsm:
            raise ValueError("mean not found in adata.obsm. Please run get_model_outputs() first.")
        
        mu = adata.obsm["mean"]  # (cells, L)
        v_gp = adata.obsm["velocity_gp"]  # (cells, L)
        
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
        
        return None
    
    def _map_gene_to_gp_velocity(
        self,
        adata,
        adata_gp,
        scale: float,
        velocity_key: str,
    ):
        """Map velocities from gene expression space to gene program space."""
        import scvelo as scv
        import scanpy as sc
        from contextlib import redirect_stdout
        import io
        
        # Check for required pre-computed results
        required_layers = ["velocity"]
        _check_velocity_layers(adata, "map_velocities (gene_to_gp)", required_layers)
        
        # Get pre-computed gene-level velocities
        gene_velocity = adata.layers["velocity"]  # (cells, G)
        
        # Create temporary AnnData with gene velocities
        adata_temp = adata.copy()
        adata_temp.layers["velocity"] = gene_velocity.astype(np.float32)
        
        # Compute transition matrix in gene space
        with io.StringIO() as buf, redirect_stdout(buf):
            scv.tl.velocity_graph(adata_temp, vkey='velocity')
            T = scv.tl.transition_matrix(adata_temp, vkey='velocity').toarray()
        
        # Get pre-computed latent representations
        if "mean" not in adata.obsm:
            raise ValueError("mean not found in adata.obsm. Please run get_model_outputs() first.")
        
        latent_state = adata.obsm["mean"]  # Current latent state (cells, L)
        future_latent_state = np.matmul(T, latent_state)  # Projected future latent state
        gp_velo_mapped = (future_latent_state - latent_state) * scale
        
        # Use provided GP AnnData and modify in place
        if adata_gp is not None:
            # Write the mapped velocity to the provided GP AnnData
            adata_gp.layers[velocity_key] = gp_velo_mapped.astype(np.float32)
        else:
            raise ValueError("adata_gp is required for gene_to_gp direction. Please provide a pre-computed GP AnnData object.")
        
        # Always write mapped velocity to original AnnData
        adata.obsm[velocity_key] = gp_velo_mapped.astype(np.float32)
        
        # No return value since adata_gp is modified in place
        return None



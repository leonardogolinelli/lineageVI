# api.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import torch
import scanpy as sc
import numpy as np
import pandas as pd
import copy

from .model import LineageVIModel
from .trainer import _Trainer  # internal; do NOT export
from . import utils


class LineageVI:
    """
    LineageVI: Deep learning-based RNA velocity model with gene program inference.
    
    LineageVI is a variational autoencoder that learns gene programs (GPs) and 
    predicts RNA velocity in both gene expression and gene program spaces. It 
    uses a two-regime training approach: first reconstructing expression, then 
    predicting velocity.
    
    Parameters
    ----------
    adata : AnnData
        Single-cell RNA-seq data with layers for unspliced and spliced counts.
    n_hidden : int, default 128
        Number of hidden units in the neural network.
    mask_key : str, default "I"
        Key for gene program mask in adata.uns.
    unspliced_key : str, default "unspliced"
        Key for unspliced counts in adata.layers.
    spliced_key : str, default "spliced"
        Key for spliced counts in adata.layers.
    nn_key : str, default "indices"
        Key for nearest neighbor indices in adata.uns.
    device : torch.device, optional
        Device to run computations on. Defaults to CUDA if available, else CPU.
    seed : int, optional
        Random seed for reproducibility.
    cluster_key : str, optional
        Key in adata.obs for cell clusters/lineages. If provided, cluster embeddings
        will be learned and used in velocity prediction.
    cluster_embedding_dim : int, default 32
        Dimension of cluster embeddings. Used when cluster_key is provided.
    cls_encoding_key : str, optional
        Key in adata.obs for CLS encoding (biological processes). If provided, process-specific CLS
        embeddings will be learned. If None or non-existent, a 'cls_encoding' column
        will be created with all cells assigned to 'Unspecified' (single global CLS embedding).
    cls_embedding_dim : int, default 32
        Dimension of CLS (classification) embedding. The CLS embedding learns
        process-specific global dynamics and is used in velocity prediction.
        It is learned only in regime 2 (velocity prediction) and frozen in regime 1.
    
    Attributes
    ----------
    adata : AnnData
        The input single-cell data.
    model : LineageVIModel
        The underlying neural network model.
    device : torch.device
        Device used for computations.
    
    Examples
    --------
    >>> import scanpy as sc
    >>> import lineagevi as lvi
    >>> 
    >>> # Load data
    >>> adata = sc.read("data.h5ad")
    >>> 
    >>> # Initialize model
    >>> linvi = lvi.LineageVI(adata)
    >>> 
    >>> # Train model
    >>> history = linvi.fit(epochs1=50, epochs2=50)
    >>> 
    >>> # Get model outputs
    >>> linvi.get_model_outputs()
    >>> 
    >>> # Analyze gene programs
    >>> linvi.latent_enrich(adata, groups="cell_type")
    """

    def __init__(
        self,
        adata: sc.AnnData,
        n_hidden: int = 128,
        mask_key: str = "I",
        *,
        unspliced_key: str = "unspliced",
        spliced_key: str = "spliced",
        nn_key: str = "indices",
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
        cluster_key: Optional[str] = None,
        cluster_embedding_dim: int = 32,
        cls_encoding_key: Optional[str] = None,
        cls_embedding_dim: int = 32,
    ):
        self.adata = adata
        self.model = LineageVIModel(
            adata, 
            n_hidden=n_hidden, 
            mask_key=mask_key, 
            seed=seed,
            cluster_key=cluster_key,
            cluster_embedding_dim=cluster_embedding_dim,
            cls_encoding_key=cls_encoding_key,
            cls_embedding_dim=cls_embedding_dim,
        )
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # dataloader / field names
        # latent_key is always 'z' since we use the mean (deterministic)
        self.unspliced_key = unspliced_key
        self.spliced_key = spliced_key
        self.latent_key = "z"  # Hardcoded - always uses mean from encoder
        self.nn_key = nn_key
        self.cluster_key = cluster_key

    # -------------------------
    # Training
    # -------------------------
    def fit(
        self,
        K: int = 10,
        batch_size: int = 1024,
        lr: float = 1e-3,
        epochs1: int = 50,
        epochs2: int = 50,
        seeds: Tuple[int, int, int] = (0, 1, 2),
        output_dir: Optional[str] = None,
        verbose: int = 1,
        monitor_genes: Optional[List[str]] = None,
        monitor_negative_velo: bool = True,
        monitor_every_epochs: int = 1,
    ) -> Dict[str, List[float]]:
        """
        Train the LineageVI model using two-regime training.
        
        The training process consists of two regimes:
        1. **Regime 1**: Expression reconstruction - trains encoder and gene decoder
        2. **Regime 2**: Velocity prediction - trains velocity decoder
        
        Parameters
        ----------
        K : int, default 10
            Number of nearest neighbors for velocity computation.
        batch_size : int, default 1024
            Batch size for training.
        lr : float, default 1e-3
            Learning rate for optimization.
        epochs1 : int, default 50
            Number of epochs for regime 1 (expression reconstruction).
        epochs2 : int, default 50
            Number of epochs for regime 2 (velocity prediction).
        seeds : Tuple[int, int, int], default (0, 1, 2)
            Random seeds for (model initialization, regime 1, regime 2).
        output_dir : str, optional
            Directory to save model weights. Defaults to current directory.
        verbose : int, default 1
            Verbosity level (0=silent, 1=progress, 2=detailed).
        monitor_genes : List[str], optional
            List of gene names to monitor during training. Phase plane plots will be
            generated for these genes during both regimes and saved to 
            output_dir/training_plots/regime{1|2}/ with filenames like 
            {gene_name}_regime{1|2}_epoch_{epoch:03d}.png.
        monitor_negative_velo : bool, default True
            Whether to use negative velocities in monitoring plots. If True, shows negative
            velocities (matches scVelo convention). If False, shows positive velocities.
        monitor_every_epochs : int, default 1
            Generate monitoring plots every N epochs. Plots are always generated at
            epoch 0 (before training starts) and the last epoch of each regime if 
            monitor_genes is provided.
        
        Returns
        -------
        Dict[str, List[float]]
            Training history with keys:
            - 'regime1_loss': List of reconstruction losses for regime 1
            - 'regime2_velocity_loss': List of velocity losses for regime 2
        
        Notes
        -----
        After training, call `get_model_outputs()` to annotate the AnnData object
        with velocities and latent representations.
        
        Examples
        --------
        >>> # Basic training
        >>> history = linvi.fit()
        >>> 
        >>> # Custom training parameters
        >>> history = linvi.fit(
        ...     epochs1=100, epochs2=100, 
        ...     lr=5e-4, batch_size=512
        ... )
        >>> 
        >>> # Training with monitoring
        >>> history = linvi.fit(
        ...     epochs1=50, epochs2=50,
        ...     monitor_genes=['Gnas', 'Ins1', 'Pdx1'],
        ...     output_dir='./results'
        ... )
        >>> 
        >>> # Get model outputs after training
        >>> linvi.get_model_outputs()
        """
        engine = _Trainer(
            self.model,
            self.adata,
            device=self.device,
            verbose=verbose,
            unspliced_key=self.unspliced_key,
            spliced_key=self.spliced_key,
            latent_key=self.latent_key,
            nn_key=self.nn_key,
        )
        history = engine.fit(
            K=K,
            batch_size=batch_size,
            lr=lr,
            epochs1=epochs1,
            epochs2=epochs2,
            seeds=seeds,
            output_dir=(output_dir or "."),
            monitor_genes=monitor_genes,
            monitor_negative_velo=monitor_negative_velo,
            monitor_every_epochs=monitor_every_epochs,
        )
        self.adata = engine.adata
        return history    

    
    '''def build_gp_adata(
        self,
        adata: Optional[sc.AnnData] = None,
        *,
        n_samples: int = 1,
        return_negative_velo: bool = True,
        base_seed: Optional[int] = None,
    ) -> sc.AnnData:
        """See `LineageVIModel.build_gp_adata`."""
        return self.model.build_gp_adata(
            (adata or self.adata),
            n_samples=n_samples,
            return_negative_velo=return_negative_velo,
            base_seed=base_seed,
        )'''
    
    def get_model_outputs(
        self,
        adata,
        n_samples: int = 1,
        return_mean: bool = False,
        return_negative_velo: bool = True,
        base_seed: Optional[int] = None,
        save_to_adata: bool = False,
        unspliced_key: str = "Mu",
        spliced_key: str = "Ms",
        nn_key: str = "indices",
        batch_size: int = 256,
        rescale_velocity_magnitude: bool = True,
        max_velocity_magnitude: float = 1.0,
    ):
        """
        Get model predictions including velocities and latent representations.
        
        This method runs the trained model to generate predictions including
        gene expression reconstruction, latent representations, and RNA velocities
        in both gene expression and gene program spaces.
        
        Parameters
        ----------
        adata : AnnData
            Single-cell data to process.
        n_samples : int, default 1
            Number of samples for uncertainty estimation.
        return_mean : bool, default True
            Whether to return mean predictions (averaged over samples).
        return_negative_velo : bool, default True
            Whether to negate velocities (multiply by -1).
        base_seed : int, optional
            Random seed for reproducibility.
        save_to_adata : bool, default False
            Whether to save outputs to the AnnData object.
        unspliced_key : str, default "Mu"
            Key for unspliced counts in adata.layers.
        spliced_key : str, default "Ms"
            Key for spliced counts in adata.layers.
        nn_key : str, default "indices"
            Key for nearest neighbor indices in adata.uns.
        batch_size : int, default 256
            Batch size for processing.
        rescale_velocity_magnitude : bool, default True
            Whether to rescale velocity magnitudes based on neighbor consistency.
            If True, velocities with consistent directions across neighbors get
            higher magnitudes, while inconsistent velocities get lower magnitudes.
        max_velocity_magnitude : float, default 1.0
            Maximum velocity magnitude after rescaling. Velocities with perfect
            consistency (all neighbors agree) will have this magnitude.
        
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing model outputs:
            - 'recon': Reconstructed gene expression
            - 'z': Latent representations
            - 'mean': Encoder mean
            - 'logvar': Encoder log-variance
            - 'velocity': Gene-level velocities
            - 'velocity_gp': Gene program velocities
            - 'alpha', 'beta', 'gamma': Kinetic parameters
        
        Examples
        --------
        >>> # Get basic model outputs
        >>> outputs = linvi.get_model_outputs(adata)
        >>> 
        >>> # Get outputs with uncertainty estimation
        >>> outputs = linvi.get_model_outputs(adata, n_samples=100)
        >>> 
        >>> # Save outputs to AnnData
        >>> linvi.get_model_outputs(adata, save_to_adata=True)
        """
        return self.model._get_model_outputs(
            adata,
            n_samples=n_samples,
            return_mean=return_mean,
            return_negative_velo=return_negative_velo,
            base_seed=base_seed,
            save_to_adata=save_to_adata,
            unspliced_key=unspliced_key,
            spliced_key=spliced_key,
            latent_key=self.latent_key,  # Always "z" - hardcoded
            nn_key=nn_key,
            batch_size=batch_size,
            rescale_velocity_magnitude=rescale_velocity_magnitude,
            max_velocity_magnitude=max_velocity_magnitude,
        )

    def latent_enrich(
        self,
        adata: Optional[sc.AnnData],
        groups,
        *,
        comparison: str | list[str] = "rest",
        n_sample: int = 5000,
        use_directions: bool = False,
        directions_key: str = "directions",
        select_terms=None,
        exact: bool = True,
        key_added: str = "bf_scores",
    ):
        """See `LineageVIModel.latent_enrich`."""
        return self.model.latent_enrich(
            (adata or self.adata),
            groups,
            comparison=comparison,
            n_sample=n_sample,
            use_directions=use_directions,
            directions_key=directions_key,
            select_terms=select_terms,
            exact=exact,
            key_added=key_added,
        )

    def get_directional_uncertainty(
        self,
        adata: Optional[sc.AnnData] = None,
        *,
        use_gp_velo: bool = False,
        n_samples: int = 50,
        n_jobs: int = -1,
        show_plot: bool = True,
        base_seed: Optional[int] = None,
    ):
        """See `LineageVIModel.get_directional_uncertainty`."""
        return self.model.get_directional_uncertainty(
            (adata or self.adata),
            use_gp_velo=use_gp_velo,
            n_samples=n_samples,
            n_jobs=n_jobs,
            show_plot=show_plot,
            base_seed=base_seed,
        )

    def compute_extrinsic_uncertainty(
        self,
        adata: Optional[sc.AnnData] = None,
        *,
        use_gp_velo: bool = False,
        n_samples: int = 25,
        n_jobs: int = -1,
        show_plot: bool = True,
        base_seed: Optional[int] = None,
    ):
        """See `LineageVIModel.compute_extrinsic_uncertainty`."""
        return self.model.compute_extrinsic_uncertainty(
            (adata or self.adata),
            use_gp_velo=use_gp_velo,
            n_samples=n_samples,
            n_jobs=n_jobs,
            show_plot=show_plot,
            base_seed=base_seed,
        )

    def perturb_genes(
        self,
        adata: Optional[sc.AnnData],
        *,
        groupby_key: str,
        group_to_perturb: str,
        genes_to_perturb,
        perturb_value: float,
        perturb_spliced: bool = True,
        perturb_unspliced: bool = False,
        perturb_both: bool = False,
    ):
        """See `LineageVIModel.perturb_genes`."""
        return self.model.perturb_genes(
            (adata or self.adata),
            groupby_key=groupby_key,
            group_to_perturb=group_to_perturb,
            genes_to_perturb=genes_to_perturb,
            perturb_value=perturb_value,
            perturb_spliced=perturb_spliced,
            perturb_unspliced=perturb_unspliced,
            perturb_both=perturb_both,
        )

    def perturb_gps(
        self,
        adata: Optional[sc.AnnData],
        *,
        gp_uns_key: str,
        gps_to_perturb,
        groupby_key: str,
        group_to_perturb: str,
        perturb_value: float,
    ):
        """See `LineageVIModel.perturb_gps`."""
        return self.model.perturb_gps(
            (adata or self.adata),
            gp_uns_key,
            gps_to_perturb,
            groupby_key,
            group_to_perturb,
            perturb_value,
        )

    def map_velocities(
        self,
        adata: Optional[sc.AnnData] = None,
        adata_gp: Optional[sc.AnnData] = None,
        *,
        direction: str = "gp_to_gene",
        scale: float = 10.0,
        velocity_key: str = "mapped_velocity",
        unspliced_key: str = "Mu",
        spliced_key: str = "Ms",
    ):
        """
        Map velocities between gene program space and gene expression space.
        
        This method enables bidirectional mapping of RNA velocities using pre-computed results:
        - **gp_to_gene**: Maps velocities from gene program space to gene expression space
        - **gene_to_gp**: Maps velocities from gene expression space to gene program space
        
        Note: This method requires that get_model_outputs() has been called first to generate
        the necessary velocity and latent representations.
        
        Parameters
        ----------
        adata : AnnData, optional
            Single-cell data with pre-computed velocities and latent representations.
            If None, uses self.adata.
        adata_gp : AnnData, optional
            Gene program AnnData object. Required for direction="gene_to_gp".
            Will be modified in place with mapped velocities.
        direction : str, default "gp_to_gene"
            Direction of mapping: "gp_to_gene" or "gene_to_gp".
        scale : float, default 10.0
            Scaling factor for mapped velocities.
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
        
        Examples
        --------
        >>> # First, get model outputs
        >>> linvi.get_model_outputs(adata, save_to_adata=True)
        >>> 
        >>> # Map from GP to gene space
        >>> linvi.map_velocities(adata, direction="gp_to_gene")
        >>> 
        >>> # Map from gene to GP space with pre-computed GP data
        >>> linvi.map_velocities(adata, adata_gp=gp_adata, direction="gene_to_gp")
        >>> # gp_adata is now modified with mapped velocities
        """
        return self.model.map_velocities(
            (adata or self.adata),
            adata_gp=adata_gp,
            direction=direction,
            scale=scale,
            velocity_key=velocity_key,
            unspliced_key=unspliced_key,
            spliced_key=spliced_key,
        )
    
    def compute_cluster_alignment(
        self,
        adata: Optional[sc.AnnData] = None,
        source_cluster: str = None,
        target_cluster: str = None,
        *,
        cluster_key: str = 'clusters',
        use_gp_space: bool = False,
        velocity_key: Optional[str] = None,
        state_key: Optional[str] = None,
        aggregation: str = 'mean',
    ) -> float:
        """
        Compute alignment metric: how much do velocities in source_cluster
        point toward target_cluster's state.
        
        This metric measures the directional alignment between velocity vectors
        of cells in the source cluster and the direction from those cells to
        the target cluster's centroid.
        
        Parameters
        ----------
        adata : AnnData, optional
            Single-cell data with computed velocities and states.
            If None, uses self.adata.
        source_cluster : str
            Name of the source cluster (cells with velocities).
        target_cluster : str
            Name of the target cluster (direction target).
        cluster_key : str, default 'clusters'
            Key in adata.obs containing cluster labels.
        use_gp_space : bool, default False
            If True, use gene program space (GP velocities and GP states).
            If False, use gene expression space (gene velocities and spliced states).
        velocity_key : str, optional
            Key for velocities:
            - If use_gp_space=True: key in adata.obsm (default: 'velocity_gp')
            - If use_gp_space=False: key in adata.layers (default: 'velocity')
        state_key : str, optional
            Key for current states:
            - If use_gp_space=True: key in adata.obsm (default: 'mean')
            - If use_gp_space=False: key in adata.layers (default: 'Ms')
        aggregation : str, default 'mean'
            How to aggregate alignment scores across cells in source cluster:
            - 'mean': Simple mean of all cell alignments
            - 'weighted_mean': Velocity magnitude-weighted mean
        
        Returns
        -------
        float
            Alignment score in [-1, 1]:
            - 1.0: Perfect alignment (all velocities point toward target)
            - 0.0: Orthogonal (no alignment)
            - -1.0: Opposite direction (velocities point away from target)
        
        Examples
        --------
        >>> # Compute alignment in gene expression space
        >>> alignment = linvi.compute_cluster_alignment(
        ...     source_cluster='Alpha',
        ...     target_cluster='Beta'
        ... )
        >>> 
        >>> # Compute alignment in gene program space
        >>> alignment_gp = linvi.compute_cluster_alignment(
        ...     source_cluster='Alpha',
        ...     target_cluster='Beta',
        ...     use_gp_space=True
        ... )
        >>> 
        >>> # Use weighted mean aggregation
        >>> alignment_weighted = linvi.compute_cluster_alignment(
        ...     source_cluster='Alpha',
        ...     target_cluster='Beta',
        ...     aggregation='weighted_mean'
        ... )
        """
        return self.model.compute_cluster_alignment(
            (adata or self.adata),
            source_cluster=source_cluster,
            target_cluster=target_cluster,
            cluster_key=cluster_key,
            use_gp_space=use_gp_space,
            velocity_key=velocity_key,
            state_key=state_key,
            aggregation=aggregation,
        )
    
    def compute_cluster_alignment_matrix(
        self,
        adata: Optional[sc.AnnData] = None,
        *,
        cluster_key: str = 'clusters',
        use_gp_space: bool = False,
        velocity_key: Optional[str] = None,
        state_key: Optional[str] = None,
        aggregation: str = 'mean',
        normalize: bool = False,
    ) -> pd.DataFrame:
        """
        Compute alignment matrix for all cluster pairs.
        
        Each entry (i, j) represents how much velocities in cluster i point
        toward cluster j's state.
        
        Parameters
        ----------
        adata : AnnData, optional
            Single-cell data with computed velocities and states.
            If None, uses self.adata.
        cluster_key : str, default 'clusters'
            Key in adata.obs containing cluster labels.
        use_gp_space : bool, default False
            If True, use gene program space. If False, use gene expression space.
        velocity_key : str, optional
            Key for velocities:
            - If use_gp_space=True: key in adata.obsm (default: 'velocity_gp')
            - If use_gp_space=False: key in adata.layers (default: 'velocity')
        state_key : str, optional
            Key for states:
            - If use_gp_space=True: key in adata.obsm (default: 'mean')
            - If use_gp_space=False: key in adata.layers (default: 'Ms')
        aggregation : str, default 'mean'
            Aggregation method: 'mean' or 'weighted_mean'.
        normalize : bool, default False
            If True, normalize each row (source cluster) using softmax so that
            alignment scores sum to 1. This converts alignment scores into a
            probability distribution over target clusters for each source cluster.
            Softmax preserves relative differences better than simple normalization
            and avoids many scores becoming zero.
        
        Returns
        -------
        alignment_df : pd.DataFrame
            Square DataFrame of shape (n_clusters, n_clusters) where:
            - Rows (index) = source clusters
            - Columns = target clusters (column labels appear at top when visualizing)
            - Values = alignment scores in [-1, 1] (or normalized probabilities if normalize=True)
            - Diagonal and failed pairs are NaN
        
        Examples
        --------
        >>> # Compute alignment matrix in gene expression space
        >>> df = linvi.compute_cluster_alignment_matrix()
        >>> 
        >>> # Find which cluster Alpha points toward
        >>> target_cluster = df.loc['Alpha'].idxmax()
        >>> alignment = df.loc['Alpha', target_cluster]
        >>> print(f"Alpha → {target_cluster}: {alignment:.3f}")
        >>> 
        >>> # Access specific alignment
        >>> alignment_ab = df.loc['Alpha', 'Beta']
        >>> 
        >>> # Get normalized probability distribution over targets for Alpha
        >>> df_normalized = linvi.compute_cluster_alignment_matrix(normalize=True)
        >>> alpha_probs = df_normalized.loc['Alpha']  # Probability distribution over targets
        >>> 
        >>> # Visualize the alignment matrix (raw scores)
        >>> import seaborn as sns
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots(figsize=(10, 8))
        >>> sns.heatmap(df, cmap='RdBu_r', center=0, annot=True, fmt='.2f', 
        ...             xticklabels=True, yticklabels=True, ax=ax)
        >>> ax.set_xlabel('Target Cluster')
        >>> ax.set_ylabel('Source Cluster')
        >>> ax.set_title('Cluster Alignment Matrix')
        >>> plt.tight_layout()
        >>> plt.show()
        >>> 
        >>> # Visualize normalized matrix (probabilities) with labels on top
        >>> # High probability = red, low probability = blue (using 'RdBu_r' colormap)
        >>> fig, ax = plt.subplots(figsize=(10, 8))
        >>> # 'RdBu_r' maps: low values (0) → blue, high values (1) → red
        >>> sns.heatmap(df_normalized, cmap='RdBu_r', vmin=0, vmax=1, 
        ...             annot=True, fmt='.3f', xticklabels=True, yticklabels=True, ax=ax)
        >>> # Move x-axis ticks and labels to top
        >>> ax.xaxis.tick_top()
        >>> ax.xaxis.set_label_position('top')
        >>> # Rotate x-axis labels vertically
        >>> plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
        >>> ax.set_title('Normalized Cluster Alignment Matrix (Probabilities)')
        >>> plt.tight_layout()
        >>> plt.show()
        """
        return self.model.compute_cluster_alignment_matrix(
            (adata or self.adata),
            cluster_key=cluster_key,
            use_gp_space=use_gp_space,
            velocity_key=velocity_key,
            state_key=state_key,
            aggregation=aggregation,
            normalize=normalize,
        )
    
    def _recompute_velocities_from_mean(self, adata: sc.AnnData):
        """
        Recompute velocities from a given mean (latent representation) without running the encoder.
        This is used for GP perturbations where we want to use the perturbed mean directly.
        
        This function only modifies the adata passed to it (which should be a copy).
        It does not affect standard usage of get_model_outputs().
        
        Parameters
        ----------
        adata : AnnData
            AnnData with perturbed mean in adata.obsm['mean'].
            Will be modified in place with recomputed velocities.
        """
        import torch
        
        # Get required data
        if 'mean' not in adata.obsm:
            raise ValueError("adata.obsm['mean'] not found")
        if 'Mu' not in adata.layers or 'Ms' not in adata.layers:
            raise ValueError("adata.layers['Mu'] and adata.layers['Ms'] must exist")
        
        mean = adata.obsm['mean']  # (cells, latent_dim)
        mu = adata.layers['Mu']  # (cells, genes)
        ms = adata.layers['Ms']  # (cells, genes)
        
        # Convert to torch tensors
        device = next(self.model.parameters()).device
        z = torch.from_numpy(mean.astype(np.float32)).to(device)
        mu_tensor = torch.from_numpy(np.asarray(mu, dtype=np.float32)).to(device)
        ms_tensor = torch.from_numpy(np.asarray(ms, dtype=np.float32)).to(device)
        mu_ms = torch.cat([mu_tensor, ms_tensor], dim=1)  # (cells, 2*genes)
        
        # Get cluster and process indices if needed
        cluster_indices = None
        if self.model.cluster_embedding is not None:
            if self.model.cluster_key is None or self.model.cluster_key not in adata.obs.columns:
                raise ValueError(
                    f"cluster_key '{self.model.cluster_key}' not found in adata.obs.columns"
                )
            cluster_labels = adata.obs[self.model.cluster_key]
            cluster_indices = torch.tensor([
                self.model.cluster_to_idx.get(str(label), 0) for label in cluster_labels
            ], dtype=torch.long, device=device)
        
        # Get process indices (always present)
        cls_encoding_key = self.model.cls_encoding_key
        process_labels = adata.obs[cls_encoding_key]
        process_indices = torch.tensor([
            self.model.process_to_idx.get(str(label), 0) for label in process_labels
        ], dtype=torch.long, device=device)
        
        # Set model to velocity regime
        self.model.first_regime = False
        
        # Compute velocities from the perturbed mean
        with torch.no_grad():
            velocity, velocity_gp, alpha, beta, gamma = self.model._forward_velocity_decoder(
                z, mu_ms, cluster_indices, process_indices
            )
        
        # Convert to numpy and save
        adata.obsm['velocity_gp'] = velocity_gp.cpu().numpy().astype(np.float32)
        
        # Split velocity into unspliced and spliced
        vel_u, vel_s = torch.split(velocity, velocity.shape[1] // 2, dim=1)
        adata.layers['velocity_u'] = vel_u.cpu().numpy().astype(np.float32)
        adata.layers['velocity'] = vel_s.cpu().numpy().astype(np.float32)
        
        # Also update z in obsm
        adata.obsm['z'] = z.cpu().numpy().astype(np.float32)
    
    def perturb_cluster_labels(
        self,
        adata: Optional[sc.AnnData] = None,
        *,
        source_cluster: str = None,
        target_cluster: str = None,
    ):
        """
        Perturb cluster embeddings by swapping embeddings between two clusters.
        
        This perturbation swaps the cluster embeddings in the model, computes
        velocity predictions with the swapped embeddings, and then restores
        the original embeddings. This allows evaluation of how cluster-specific
        dynamics affect velocity predictions.
        
        Parameters
        ----------
        adata : AnnData, optional
            Single-cell data to compute velocities on. If None, uses self.adata.
        source_cluster : str
            Cluster label whose embedding will be swapped.
        target_cluster : str
            Cluster label whose embedding will be swapped with source_cluster.
        
        Returns
        -------
        df_genes : pd.DataFrame
            DataFrame with gene-level differences (velocity, alpha, beta, gamma, etc.).
        df_gps : pd.DataFrame
            DataFrame with GP-level differences (GP velocity, etc.).
        
        Notes
        -----
        Perturbed outputs are stored in adata.obsm and adata.layers with ``_pert`` suffix.
        This function:
        1. Saves the original cluster embeddings
        2. Swaps embeddings between source_cluster and target_cluster
        3. Computes velocities with swapped embeddings
        4. Computes differences from original velocities
        5. Restores original embeddings
        """
        if adata is None:
            adata = self.adata
            if adata is None:
                raise ValueError("adata must be provided either as parameter or via self.adata")
        
        return self.model.perturb_cluster_labels(
            adata,
            source_cluster=source_cluster,
            target_cluster=target_cluster,
        )
    
    def evaluate_perturbation_effects(
        self,
        adata: Optional[sc.AnnData] = None,
        *,
        perturbation_type: str = 'gps',
        groupby_key: str = 'clusters',
        group_to_perturb: str = None,
        perturb_value: float = 1.0,
        genes_to_perturb: Optional[List[str]] = None,
        gps_to_perturb: Optional[List[str]] = None,
        gp_uns_key: str = 'terms',
        source_cluster: Optional[str] = None,
        target_cluster: Optional[str] = None,
        use_gp_space: bool = False,
        normalize: bool = True,
        aggregation: str = 'weighted_mean',
        compute_embedding_similarity: bool = True,
        velocity_key: Optional[str] = None,
        state_key: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Evaluate how perturbations affect cluster alignment and embedding similarity.
        
        This method computes baseline metrics, applies a perturbation, recomputes
        model outputs, and then computes post-perturbation metrics to quantify the
        effects of the perturbation on:
        - Directional alignment (how velocities point toward clusters)
        - Cluster embedding similarity (cosine similarity between cluster embeddings)
          Only computed for cluster label switching perturbations (cluster embeddings are
          model parameters, independent of input data for GP/gene perturbations).
        
        Parameters
        ----------
        adata : AnnData, optional
            Single-cell data to perturb. If None, uses self.adata.
        perturbation_type : str, default 'gps'
            Type of perturbation:
            - 'gps': Perturb gene program activations
            - 'genes': Perturb gene expression
            - 'cluster_labels': Switch cluster labels from source_cluster to target_cluster
        groupby_key : str, default 'clusters'
            Key in adata.obs containing group information.
        group_to_perturb : str, optional
            Name of group to perturb (required for 'gps' and 'genes' perturbations).
        perturb_value : float, default 1.0
            For GP perturbations: value to replace gene program activation with.
            For gene perturbations: value to add to expression.
            Not used for cluster label switching.
        genes_to_perturb : list, optional
            List of gene names to perturb (required if perturbation_type='genes').
        gps_to_perturb : list, optional
            List of gene program names to perturb (required if perturbation_type='gps').
        gp_uns_key : str, default 'terms'
            Key in adata.uns containing gene program names (for GP perturbations).
        source_cluster : str, optional
            Original cluster label to switch from (required if perturbation_type='cluster_labels').
        target_cluster : str, optional
            Target cluster label to switch to (required if perturbation_type='cluster_labels').
        use_gp_space : bool, default False
            If True, compute alignment in gene program space.
        normalize : bool, default True
            If True, normalize alignment scores using softmax.
        aggregation : str, default 'weighted_mean'
            Aggregation method: 'mean' or 'weighted_mean'.
        compute_embedding_similarity : bool, default True
            If True, compute embedding similarity changes. Only meaningful for 
            cluster label switching perturbations (for GP/gene perturbations, 
            cluster embeddings are unchanged).
        velocity_key : str, optional
            Key for velocities (see compute_cluster_alignment_matrix for defaults).
        state_key : str, optional
            Key for states (see compute_cluster_alignment_matrix for defaults).
        
        Returns
        -------
        results : dict
            Dictionary containing:
            - 'baseline_alignment': DataFrame (baseline alignment matrix)
            - 'perturbed_alignment': DataFrame (post-perturbation alignment matrix)
            - 'alignment_delta': DataFrame (difference: perturbed - baseline)
            - 'baseline_similarity': DataFrame (baseline embedding similarity, if computed)
            - 'perturbed_similarity': DataFrame (post-perturbation similarity, if computed)
            - 'similarity_delta': DataFrame (difference: perturbed - baseline, if computed)
        
        Examples
        --------
        >>> # Evaluate GP perturbation effects
        >>> results = vae.evaluate_perturbation_effects(
        ...     adata,
        ...     perturbation_type='gps',
        ...     group_to_perturb='Alpha',
        ...     gps_to_perturb=['GP_1', 'GP_2'],
        ...     perturb_value=1.0,
        ...     compute_embedding_similarity=False  # Cluster embeddings unchanged
        ... )
        >>> 
        >>> # Evaluate cluster label switching effects
        >>> results = vae.evaluate_perturbation_effects(
        ...     adata,
        ...     perturbation_type='cluster_labels',
        ...     source_cluster='Alpha',
        ...     target_cluster='Beta',
        ...     compute_embedding_similarity=True  # Cluster embeddings change
        ... )
        >>> 
        >>> # Visualize alignment changes
        >>> import lineagevi.plots as lv_plots
        >>> fig, ax = lv_plots.plot_cluster_alignment_matrix(
        ...     results['alignment_delta'],
        ...     title='Alignment Change After Perturbation'
        ... )
        """
        if adata is None:
            adata = self.adata
            if adata is None:
                raise ValueError("adata must be provided either as parameter or via self.adata")
        
        if group_to_perturb is None:
            raise ValueError("group_to_perturb must be provided")
        
        # 1. Compute baseline metrics
        print("Computing baseline metrics...")
        baseline_alignment = self.compute_cluster_alignment_matrix(
            adata,
            cluster_key=groupby_key,
            use_gp_space=use_gp_space,
            normalize=normalize,
            aggregation=aggregation,
            velocity_key=velocity_key,
            state_key=state_key,
        )
        
        # For GP/gene perturbations, cluster embeddings are unchanged (they're model parameters)
        # Only compute similarity for cluster label switching
        baseline_similarity = None
        if compute_embedding_similarity and perturbation_type == 'cluster_labels':
            baseline_similarity = utils.compute_cluster_embedding_similarity(adata)
        elif compute_embedding_similarity and perturbation_type in ['gps', 'genes']:
            print("Note: Skipping cluster embedding similarity for GP/gene perturbations "
                  "(cluster embeddings are model parameters, independent of input data).")
        
        # 2. Validate that genes/GPs exist before applying perturbation
        if perturbation_type == 'genes':
            if genes_to_perturb is None:
                raise ValueError("genes_to_perturb must be provided when perturbation_type='genes'")
            
            # Convert to list if single string
            if isinstance(genes_to_perturb, str):
                genes_to_perturb = [genes_to_perturb]
            
            # Validate that all genes exist in adata
            available_genes = set(adata.var_names)
            missing_genes = [g for g in genes_to_perturb if g not in available_genes]
            if missing_genes:
                raise ValueError(
                    f"The following genes are not present in adata.var_names: {missing_genes}. "
                    f"Available genes: {list(available_genes)[:10]}..." if len(available_genes) > 10 else f"Available genes: {list(available_genes)}"
                )
        
        elif perturbation_type == 'gps':
            if gps_to_perturb is None:
                raise ValueError("gps_to_perturb must be provided when perturbation_type='gps'")
            
            # Convert to list if single string
            if isinstance(gps_to_perturb, str):
                gps_to_perturb = [gps_to_perturb]
            
            # Validate that GP key exists in adata.uns
            if gp_uns_key not in adata.uns:
                raise KeyError(
                    f"Gene program key '{gp_uns_key}' not found in adata.uns. "
                    f"Available uns keys: {list(adata.uns.keys())}"
                )
            
            # Validate that all GPs exist
            available_gps = set(adata.uns[gp_uns_key])
            missing_gps = [gp for gp in gps_to_perturb if gp not in available_gps]
            if missing_gps:
                raise ValueError(
                    f"The following gene programs are not present in adata.uns['{gp_uns_key}']: {missing_gps}. "
                    f"Available gene programs: {list(available_gps)[:10]}..." if len(available_gps) > 10 else f"Available gene programs: {list(available_gps)}"
                )
        elif perturbation_type == 'cluster_labels':
            if source_cluster is None or target_cluster is None:
                raise ValueError(
                    "Both source_cluster and target_cluster must be provided when perturbation_type='cluster_labels'"
                )
            # Validate clusters exist in model
            if self.model.cluster_embedding is None:
                raise ValueError("Cluster embeddings are not enabled in this model")
            if self.model.cluster_to_idx is None:
                raise ValueError("Model does not have cluster_to_idx mapping")
            available_clusters = set(self.model.cluster_to_idx.keys())
            if source_cluster not in available_clusters:
                raise ValueError(
                    f"Source cluster '{source_cluster}' not found in model. "
                    f"Available clusters: {list(available_clusters)}"
                )
            if target_cluster not in available_clusters:
                raise ValueError(
                    f"Target cluster '{target_cluster}' not found in model. "
                    f"Available clusters: {list(available_clusters)}"
                )
        else:
            raise ValueError(
                f"perturbation_type must be 'genes', 'gps', or 'cluster_labels', got '{perturbation_type}'"
            )
        
        # 3. Apply perturbation (work on a copy)
        adata_perturbed = copy.deepcopy(adata)
        
        if perturbation_type == 'genes':
            print(f"Applying gene perturbation to group '{group_to_perturb}'...")
            self.perturb_genes(
                adata_perturbed,
                groupby_key=groupby_key,
                group_to_perturb=group_to_perturb,
                genes_to_perturb=genes_to_perturb,
                perturb_value=perturb_value,
            )
            # Recompute model outputs after gene perturbation
            print("Recomputing model outputs after perturbation...")
            self.get_model_outputs(adata_perturbed, save_to_adata=True)
        
        elif perturbation_type == 'gps':
            print(f"Applying GP perturbation to group '{group_to_perturb}'...")
            # For GP perturbations, we need to modify the latent representation directly
            # Get group indices
            group_idxs = self.model._get_group_idxs(adata_perturbed, groupby_key=groupby_key)
            cell_idx = group_idxs[group_to_perturb]
            
            # Get GP indices
            gp_idx = self.model._get_gp_idxs(adata_perturbed, gp_uns_key, gps_to_perturb)
            
            # Modify the latent representation (mean) in adata.obsm
            if 'mean' not in adata_perturbed.obsm:
                raise ValueError(
                    "adata.obsm['mean'] not found. Please run get_model_outputs(adata, save_to_adata=True) first."
                )
            
            # Apply perturbation: replace GP activation values
            # Make a deep copy of the array to ensure we're not modifying the original
            mean_perturbed = np.array(adata_perturbed.obsm['mean'], copy=True)
            
            # Convert cell_idx to numpy array if needed
            if isinstance(cell_idx, (int, np.integer)):
                cell_idx = np.array([cell_idx])
            elif not isinstance(cell_idx, np.ndarray):
                cell_idx = np.array(cell_idx)
            
            # Convert gp_idx to numpy array if needed
            if isinstance(gp_idx, (int, np.integer)):
                gp_idx = np.array([gp_idx])
            elif not isinstance(gp_idx, np.ndarray):
                gp_idx = np.array(gp_idx)
            
            # Use broadcasting to set perturbation value for all cells and GPs
            # mean_perturbed[cell_idx[:, None], gp_idx] broadcasts (n_cells, 1) with (n_gps,)
            # to set values for all combinations
            mean_perturbed[np.ix_(cell_idx, gp_idx)] = perturb_value
            
            # Assign the modified copy back to the perturbed adata
            adata_perturbed.obsm['mean'] = mean_perturbed
            
            # For GP perturbations, we need to manually recompute velocities from the perturbed mean
            # without running get_model_outputs (which would recompute mean from Mu/Ms)
            print("Recomputing velocities from perturbed GP activations...")
            self._recompute_velocities_from_mean(adata_perturbed)
        
        elif perturbation_type == 'cluster_labels':
            print(f"Swapping cluster embeddings: '{source_cluster}' <-> '{target_cluster}'...")
            # Perturb cluster embeddings (this function handles swapping and restoring)
            # Note: perturb_cluster_labels restores embeddings automatically, so we need to
            # manually swap again for post-perturbation metrics, or we can use the perturbed outputs
            # For now, let's swap, compute metrics, then restore
            # Actually, we should compute baseline first, then swap and compute perturbed
            
            # The function already computes everything we need, but we need baseline too
            # So we'll compute baseline first, then call perturb_cluster_labels which will
            # compute differences. But we need the actual perturbed outputs for alignment matrix.
            
            # For now, let's manually swap for this evaluation
            # Save original embeddings
            source_idx = self.model.cluster_to_idx[source_cluster]
            target_idx = self.model.cluster_to_idx[target_cluster]
            embeddings = self.model.cluster_embedding.embeddings.weight
            original_source_emb = embeddings[source_idx].clone()
            original_target_emb = embeddings[target_idx].clone()
            
            try:
                # Swap embeddings
                with torch.no_grad():
                    embeddings[source_idx].copy_(original_target_emb)
                    embeddings[target_idx].copy_(original_source_emb)
                
                # Recompute model outputs with swapped cluster embeddings
                print("Recomputing model outputs after cluster embedding swap...")
                self.get_model_outputs(adata_perturbed, save_to_adata=True)
            finally:
                # Restore original embeddings
                with torch.no_grad():
                    embeddings[source_idx].copy_(original_source_emb)
                    embeddings[target_idx].copy_(original_target_emb)
        
        # 5. Compute post-perturbation metrics
        print("Computing post-perturbation metrics...")
        perturbed_alignment = self.compute_cluster_alignment_matrix(
            adata_perturbed,
            cluster_key=groupby_key,
            use_gp_space=use_gp_space,
            normalize=normalize,
            aggregation=aggregation,
            velocity_key=velocity_key,
            state_key=state_key,
        )
        
        perturbed_similarity = None
        if compute_embedding_similarity and perturbation_type == 'cluster_labels':
            perturbed_similarity = utils.compute_cluster_embedding_similarity(adata_perturbed)
        
        # 6. Compute differences
        print("Computing differences...")
        alignment_delta = perturbed_alignment - baseline_alignment
        
        similarity_delta = None
        if compute_embedding_similarity and perturbation_type == 'cluster_labels':
            similarity_delta = perturbed_similarity - baseline_similarity
        
        return {
            'baseline_alignment': baseline_alignment,
            'perturbed_alignment': perturbed_alignment,
            'alignment_delta': alignment_delta,
            'baseline_similarity': baseline_similarity,
            'perturbed_similarity': perturbed_similarity,
            'similarity_delta': similarity_delta,
        }
    

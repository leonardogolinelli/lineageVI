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
    n_layers : int, default 1
        Number of hidden layers in the encoder and velocity decoder.
    dropout : float, default 0.0
        Dropout probability applied after each hidden layer (0 = no dropout).
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
    >>> # Differential test (e.g. latent / gene programs)
    >>> diff = linvi.differential(adata, groupby_key="cell_type", mode="latent")
    """

    def __init__(
        self,
        adata: sc.AnnData,
        n_hidden: int = 128,
        n_layers: int = 1,
        dropout: float = 0.0,
        mask_key: str = "I",
        *,
        unspliced_key: str = "unspliced",
        spliced_key: str = "spliced",
        nn_key: str = "indices",
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
        cluster_key: Optional[str] = None,
        cluster_embedding_dim: int = 32,
    ):
        self.adata = adata
        self.model = LineageVIModel(
            adata,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout=dropout,
            mask_key=mask_key,
            seed=seed,
            cluster_key=cluster_key,
            cluster_embedding_dim=cluster_embedding_dim,
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
        lr_regime1: Optional[float] = None,
        lr_regime2: Optional[float] = None,
        epochs1: int = 50,
        epochs2: int = 50,
        seeds: Tuple[int, int, int] = (0, 1, 2),
        train_size: Optional[float] = None,
        velocity_loss_weight_gene: float = 1.0,
        velocity_loss_weight_gp: float = 1.0,
        kl_weight_schedule: str = "linear",
        kl_weight: float = 1e-5,
        kl_weight_min: float = 0.0,
        kl_weight_max: float = 1e-1,
        kl_cycle_ramp_frac: float = 0.2,
        output_dir: Optional[str] = None,
        verbose: int = 1,
        monitor_genes: Optional[List[str]] = None,
        monitor_negative_velo: bool = True,
        monitor_every_epochs: int = 1,
        show_metrics: bool = True,
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
            Default learning rate for both regimes (used when lr_regime1/lr_regime2 are not set).
        lr_regime1 : float, optional
            Learning rate for regime 1 (expression reconstruction). If None, uses lr.
        lr_regime2 : float, optional
            Learning rate for regime 2 (velocity prediction). If None, uses lr.
        epochs1 : int, default 50
            Number of epochs for regime 1 (expression reconstruction).
        epochs2 : int, default 50
            Number of epochs for regime 2 (velocity prediction).
        seeds : Tuple[int, int, int], default (0, 1, 2)
            Random seeds for (model initialization, regime 1, regime 2).
        train_size : float, optional
            Fraction of cells to use for training (0 < train_size < 1). The rest is used
            for validation and both training and validation losses are logged each epoch.
            If None, all cells are used for training and no validation loss is computed.
        velocity_loss_weight_gene : float, default 1.0
            Weight for the gene-level velocity loss (expression space).
        velocity_loss_weight_gp : float, default 1.0
            Weight for the gene program velocity loss (latent space).
        kl_weight_schedule : str, default "linear"
            Schedule for KL weight in regime 1: "none" (constant kl_weight), or "linear"
            (anneal from kl_weight_min to kl_weight_max over first kl_cycle_ramp_frac of epochs, then hold at max).
        kl_weight : float, default 1e-5
            Constant KL weight when kl_weight_schedule is "none".
        kl_weight_min : float, default 0.0
            Minimum KL weight for "linear" schedule.
        kl_weight_max : float, default 1e-1
            Maximum KL weight for "linear" schedule.
        kl_cycle_ramp_frac : float, default 0.2
            For "linear": fraction of total epochs used for ramp min->max, then hold at max (1 = ramp all epochs).
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
        show_metrics : bool, default True
            If True, display the training loss curves plot as cell output (in addition to saving).
            If False, only save plots to disk.
        
        Returns
        -------
        Dict[str, List[float]]
            Training history with keys:
            - 'regime1_loss', 'regime1_recon_loss', 'regime1_kl_loss': regime 1 training losses
            - 'regime2_velocity_loss', 'regime2_velocity_loss_gene', 'regime2_velocity_loss_gp': regime 2 training losses
            - 'regime1_val_loss', 'regime1_val_recon_loss', 'regime1_val_kl_loss': regime 1 validation (if train_size set)
            - 'regime2_velocity_val_loss', 'regime2_velocity_val_loss_gene', 'regime2_velocity_val_loss_gp': regime 2 validation (if train_size set)
        
        Notes
        -----
        After training, call `get_model_outputs()` to annotate the AnnData object
        with velocities and latent representations.
        
        Examples
        --------
        >>> # Basic training
        >>> history = linvi.fit()
        >>> 
        >>> # Training with train/validation split (logs train and val loss)
        >>> history = linvi.fit(train_size=0.9)
        >>> 
        >>> # Custom training parameters
        >>> history = linvi.fit(
        ...     epochs1=100, epochs2=100, 
        ...     lr=5e-4, batch_size=512
        ... )
        >>> # Separate learning rates per regime
        >>> history = linvi.fit(lr_regime1=1e-3, lr_regime2=5e-4)
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
            lr_regime1=lr_regime1,
            lr_regime2=lr_regime2,
            epochs1=epochs1,
            epochs2=epochs2,
            seeds=seeds,
            train_size=train_size,
            velocity_loss_weight_gene=velocity_loss_weight_gene,
            velocity_loss_weight_gp=velocity_loss_weight_gp,
            kl_weight_schedule=kl_weight_schedule,
            kl_weight=kl_weight,
            kl_weight_min=kl_weight_min,
            kl_weight_max=kl_weight_max,
            kl_cycle_ramp_frac=kl_cycle_ramp_frac,
            output_dir=(output_dir or "."),
            monitor_genes=monitor_genes,
            monitor_negative_velo=monitor_negative_velo,
            monitor_every_epochs=monitor_every_epochs,
            show_metrics=show_metrics,
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

    def differential(
        self,
        adata: Optional[sc.AnnData],
        groupby_key: str,
        mode: str = "expression",
        *,
        layer: Optional[str] = None,
        velocity_layer: str = "velocity",
        ensure_model_outputs: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Differential test (Wilcoxon rank-sum 1 vs rest) by group.

        Returns a dict of DataFrames (one per group), each with columns
        'difference', 'pval', 'padj' and index = feature names.
        See LineageVIModel.differential for full documentation.
        """
        return self.model.differential(
            (adata or self.adata),
            groupby_key=groupby_key,
            mode=mode,
            layer=layer,
            velocity_layer=velocity_layer,
            ensure_model_outputs=ensure_model_outputs,
        )

    def intrinsic_uncertainty(
        self,
        adata: Optional[sc.AnnData] = None,
        *,
        use_gp_velo: bool = False,
        n_samples: int = 50,
        n_jobs: int = -1,
        show_plot: bool = True,
        base_seed: Optional[int] = None,
    ):
        """See `LineageVIModel.intrinsic_uncertainty`."""
        return self.model.intrinsic_uncertainty(
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

    def perturb(
        self,
        adata: Optional[sc.AnnData],
        mode: str,
        *,
        groupby_key: str,
        group_to_perturb: str,
        perturb_value: float,
        genes_to_perturb=None,
        gp_uns_key: Optional[str] = None,
        gps_to_perturb=None,
        perturb_spliced: bool = True,
        perturb_unspliced: bool = False,
        perturb_both: bool = False,
    ):
        """See `LineageVIModel.perturb`. Single entry point for mode='genes' or mode='gps'."""
        return self.model.perturb(
            (adata or self.adata),
            mode=mode,
            groupby_key=groupby_key,
            group_to_perturb=group_to_perturb,
            perturb_value=perturb_value,
            genes_to_perturb=genes_to_perturb,
            gp_uns_key=gp_uns_key,
            gps_to_perturb=gps_to_perturb,
            perturb_spliced=perturb_spliced,
            perturb_unspliced=perturb_unspliced,
            perturb_both=perturb_both,
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

        # Set model to velocity regime
        self.model.first_regime = False

        # Compute velocities from the perturbed mean
        with torch.no_grad():
            velocity, velocity_gp, alpha, beta, gamma = self.model._forward_velocity_decoder(
                z, mu_ms, cluster_indices
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


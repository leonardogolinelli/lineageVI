# api.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import torch
import scanpy as sc

from .model import LineageVIModel
from .trainer import _Trainer  # internal; do NOT export


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
    latent_key : str, default "z"
        Key for latent representations in adata.obsm.
    nn_key : str, default "indices"
        Key for nearest neighbor indices in adata.uns.
    device : torch.device, optional
        Device to run computations on. Defaults to CUDA if available, else CPU.
    seed : int, optional
        Random seed for reproducibility.
    
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
        latent_key: str = "z",
        nn_key: str = "indices",
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ):
        self.adata = adata
        self.model = LineageVIModel(adata, n_hidden=n_hidden, mask_key=mask_key, seed=seed)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # dataloader / field names
        self.unspliced_key = unspliced_key
        self.spliced_key = spliced_key
        self.latent_key = latent_key
        self.nn_key = nn_key

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
            generated for these genes at each epoch during regime 2 (velocity prediction)
            and saved to output_dir/training_plots/ with filenames like {gene_name}_epoch_{epoch:03d}.png.
        monitor_negative_velo : bool, default True
            Whether to use negative velocities in monitoring plots. If True, shows negative
            velocities (matches scVelo convention). If False, shows positive velocities.
        
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
        return_mean: bool = True,
        return_negative_velo: bool = True,
        base_seed: Optional[int] = None,
        save_to_adata: bool = False,
        unspliced_key: str = "Mu",
        spliced_key: str = "Ms",
        latent_key: str = "z",
        nn_key: str = "indices",
        batch_size: int = 256,
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
        latent_key : str, default "z"
            Key for latent representations in adata.obsm.
        nn_key : str, default "indices"
            Key for nearest neighbor indices in adata.uns.
        batch_size : int, default 256
            Batch size for processing.
        
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
            n_samples,
            return_mean,
            return_negative_velo,
            base_seed,
            save_to_adata,
            unspliced_key,
            spliced_key
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

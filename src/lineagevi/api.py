# api.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import torch
import scanpy as sc

from .model import LineageVIModel
from .trainer import _Trainer  # internal; do NOT export


class LineageVI:
    """
    User-facing estimator.

    - Holds the model and AnnData.
    - Provides .fit(...) to train via the internal Trainer.
    - Exposes explicit, typed wrappers for advanced model utilities so that
      IDEs and help() show the real argument lists.
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
        shuffle_regime1: bool = True,
        shuffle_regime2: bool = False,
        seeds: Tuple[int, int, int] = (0, 1, 2),
        output_dir: Optional[str] = None,
        verbose: int = 1,
    ) -> Dict[str, List[float]]:
        """Train the model via the internal Trainer; annotates self.adata on completion."""
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
            shuffle_regime1=shuffle_regime1,
            shuffle_regime2=shuffle_regime2,
            seeds=seeds,
            output_dir=(output_dir or "."),
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
        base_seed: int | None = None,
        save_to_adata: bool = False,
        unspliced_key: str = "Mu",
        spliced_key: str = "Ms",
        latent_key: str = "z",
        nn_key: str = "indices",
        batch_size: int = 256,
    ):
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
        cell_type_key: str,
        cell_type_to_perturb: str,
        genes_to_perturb,
        perturb_value: float,
        perturb_spliced: bool = True,
        perturb_unspliced: bool = False,
        perturb_both: bool = False,
    ):
        """See `LineageVIModel.perturb_genes`."""
        return self.model.perturb_genes(
            (adata or self.adata),
            cell_type_key=cell_type_key,
            cell_type_to_perturb=cell_type_to_perturb,
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
        cell_type_key: str,
        ctypes_to_perturb: str,
        perturb_value: float,
    ):
        """See `LineageVIModel.perturb_gps`."""
        return self.model.perturb_gps(
            (adata or self.adata),
            gp_uns_key,
            gps_to_perturb,
            cell_type_key,
            ctypes_to_perturb,
            perturb_value,
        )

    def map_velocities(
        self,
        adata: Optional[sc.AnnData] = None,
        *,
        direction: str = "gp_to_gene",
        n_samples: int = 100,
        scale: float = 10.0,
        base_seed: Optional[int] = None,
        save_to_adata: bool = True,
        velocity_key: str = "mapped_velocity",
        unspliced_key: str = "Mu",
        spliced_key: str = "Ms",
        latent_key: str = "z",
        nn_key: str = "indices",
        batch_size: int = 256,
    ):
        """See `LineageVIModel.map_velocities`."""
        return self.model.map_velocities(
            (adata or self.adata),
            direction=direction,
            n_samples=n_samples,
            scale=scale,
            base_seed=base_seed,
            save_to_adata=save_to_adata,
            velocity_key=velocity_key,
            unspliced_key=unspliced_key,
            spliced_key=spliced_key,
            latent_key=latent_key,
            nn_key=nn_key,
            batch_size=batch_size,
        )

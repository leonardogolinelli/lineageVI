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
    - Transparently forwards advanced methods to the underlying LineageVIModel
      using *args/**kwargs so signatures can evolve without breaking the API.
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
    ):
        self.adata = adata
        self.model = LineageVIModel(adata, n_hidden=n_hidden, mask_key=mask_key)
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
        # keep any annotations produced during training
        self.adata = engine.adata
        return history

    # -------------------------
    # Transparent forwards to the underlying model
    # (use *args/**kwargs so the model’s signature can evolve)
    # -------------------------

    def get_model_outputs(self, *args, **kwargs):
        return self.model.get_model_outputs(*args, **kwargs)

    def build_gp_adata(self, *args, **kwargs):
        return self.model.build_gp_adata(*args, **kwargs)

    def latent_enrich(self, *args, **kwargs):
        return self.model.latent_enrich(*args, **kwargs)
    
    def get_directional_uncertainty(self, *args, **kwargs):
        return self.model.get_directional_uncertainty(*args, **kwargs)

    def compute_extrinsic_uncertainty(self, *args, **kwargs):
        return self.model.compute_extrinsic_uncertainty(*args, **kwargs)

    def perturb_genes(self, *args, **kwargs):
        return self.model.perturb_genes(*args, **kwargs)

    def perturb_gps(self, *args, **kwargs):
        return self.model.perturb_gps(*args, **kwargs)

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.sparse as sp
import scanpy as sc
import random

class RegimeDataset(Dataset):
    """
    PyTorch Dataset for LineageVI two-regime training.
    
    This dataset handles both training regimes:
    - **Regime 1**: Expression reconstruction using unspliced+spliced counts
    - **Regime 2**: Velocity prediction using latent representations
    
    Parameters
    ----------
    adata : AnnData
        Single-cell data with required layers and nearest neighbor indices.
    K : int
        Number of nearest neighbors for velocity computation.
    unspliced_key : str, default 'unspliced'
        Key for unspliced counts in adata.layers.
    spliced_key : str, default 'spliced'
        Key for spliced counts in adata.layers.
    latent_key : str, default 'z'
        Key for latent representations in adata.obsm.
    nn_key : str, default 'indices'
        Key for nearest neighbor indices in adata.uns.
    
    Attributes
    ----------
    x : torch.Tensor
        Concatenated unspliced+spliced counts (cells, 2*genes).
    nn_indices : torch.Tensor
        Nearest neighbor indices (cells, K).
    latent_data : torch.Tensor or None
        Latent representations for regime 2 (cells, L).
    first_regime : bool
        Whether currently in regime 1 (expression) or regime 2 (latent).
    
    Notes
    -----
    The dataset automatically switches between regimes using set_regime().
    Latent data is loaded lazily when switching to regime 2.
    """
    
    def __init__(
        self,
        adata: sc.AnnData,
        K: int,
        unspliced_key: str = 'unspliced',
        spliced_key: str = 'spliced',
        latent_key: str = 'z',
        nn_key: str = 'indices',
    ):
        self.adata        = adata
        self.K            = K
        self.unspliced_key = unspliced_key
        self.spliced_key  = spliced_key
        self.latent_key   = latent_key

        # kNN indices from adata.uns['indices']
        indices = adata.uns.get(nn_key)
        if indices is None:
            raise KeyError(f"adata.uns['{nn_key}'] not found")
        self.nn_indices = torch.from_numpy(np.asarray(indices, dtype=np.int64))

        # full unspliced+spliced counts
        u = adata.layers[unspliced_key]
        s = adata.layers[spliced_key]
        u = u.toarray().astype(np.float32) if sp.issparse(u) else np.asarray(u, dtype=np.float32)
        s = s.toarray().astype(np.float32) if sp.issparse(s) else np.asarray(s, dtype=np.float32)
        full = np.concatenate([u, s], axis=1)
        self.x = torch.from_numpy(full)

        # will load latent z at switch to regime 2
        self.latent_data = None
        self.first_regime = True

    def set_regime(self, first: bool):
        """
        Switch between expression-based regime (first) and latent-based (second).
        
        Parameters
        ----------
        first : bool
            If True, switch to regime 1 (expression reconstruction).
            If False, switch to regime 2 (velocity prediction).
        
        Notes
        -----
        When switching to regime 2, latent data is loaded from adata.obsm[latent_key]
        if not already loaded.
        """
        self.first_regime = first
        if not first and self.latent_data is None:
            z = self.adata.obsm[self.latent_key]
            self.latent_data = torch.from_numpy(np.asarray(z, dtype=np.float32))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx: int):
        # exclude self (position 0), take next K neighbors
        neigh_idx = self.nn_indices[idx, 1:self.K+1]
        x_neigh   = self.x[neigh_idx]  # (K, G)

        if self.first_regime:
            x = self.x[idx]
            return x, idx, x_neigh      # you only need x and its neighs
        else:
            x       = self.x[idx]
            z       = self.latent_data[idx]
            z_neigh = self.latent_data[neigh_idx]
            return x, idx, x_neigh, z, z_neigh

def make_dataloader(
    adata: sc.AnnData,
    first_regime: bool = True,
    K: int = 10,
    unspliced_key: str = 'unspliced',
    spliced_key: str = 'spliced',
    latent_key: str = 'z',
    nn_key: str = 'indices',
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int | None = None,
) -> DataLoader:
    """
    Create a PyTorch DataLoader for LineageVI training.
    
    This function creates a DataLoader that can handle both training regimes
    of LineageVI with appropriate data preprocessing and batching.
    
    Parameters
    ----------
    adata : AnnData
        Single-cell data with required layers and nearest neighbor indices.
    first_regime : bool, default True
        Whether to create dataset for regime 1 (expression) or regime 2 (latent).
    K : int, default 10
        Number of nearest neighbors for velocity computation.
    unspliced_key : str, default 'unspliced'
        Key for unspliced counts in adata.layers.
    spliced_key : str, default 'spliced'
        Key for spliced counts in adata.layers.
    latent_key : str, default 'z'
        Key for latent representations in adata.obsm.
    nn_key : str, default 'indices'
        Key for nearest neighbor indices in adata.uns.
    batch_size : int, default 64
        Batch size for the DataLoader.
    shuffle : bool, default True
        Whether to shuffle the data during training.
    num_workers : int, default 0
        Number of worker processes for data loading.
    seed : int, optional
        Random seed for reproducible shuffling.
    
    Returns
    -------
    DataLoader
        PyTorch DataLoader configured for LineageVI training.
    
    Examples
    --------
    >>> # Create dataloader for regime 1 (expression reconstruction)
    >>> loader1 = make_dataloader(adata, first_regime=True, batch_size=256)
    >>> 
    >>> # Create dataloader for regime 2 (velocity prediction)
    >>> loader2 = make_dataloader(adata, first_regime=False, batch_size=128)
    >>> 
    >>> # Custom parameters
    >>> loader = make_dataloader(
    ...     adata, 
    ...     first_regime=True,
    ...     batch_size=512,
    ...     shuffle=True,
    ...     num_workers=4
    ... )
    """
    # --- DO NOT reseed global RNGs here ---

    ds = RegimeDataset(
        adata, K,
        unspliced_key=unspliced_key,
        spliced_key=spliced_key,
        latent_key=latent_key,
        nn_key=nn_key,
    )
    ds.set_regime(first_regime)

    gen = None
    if shuffle and seed is not None:
        gen = torch.Generator()
        gen.manual_seed(seed)

    # Seed numpy/python per worker deterministically (doesn't touch torch model RNG)
    def _worker_init_fn(worker_id: int):
        if seed is None:
            return
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=gen,              # controls PyTorch shuffling deterministically
        worker_init_fn=_worker_init_fn,
        persistent_workers=(num_workers > 0),
    )


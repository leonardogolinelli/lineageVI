from __future__ import annotations

from typing import Optional, Union, TYPE_CHECKING

import torch
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

if TYPE_CHECKING:
    from .api import LineageVI  # Only imported for type checking, not at runtime

def add_annotations(adata, files, min_genes=0, max_genes=None, varm_key='I', uns_key='terms',
                clean=True, genes_use_upper=True):
    """\
    Add annotations to an AnnData object from files.

    Parameters
    ----------
    adata
        Annotated data matrix.
    files
        Paths to text files with annotations. The function considers rows to be gene sets
        with name of a gene set in the first column followed by names of genes.
    min_genes
        Only include gene sets which have the total number of genes in adata
        greater than this value.
    max_genes
        Only include gene sets which have the total number of genes in adata
        less than this value.
    varm_key
        Store the binary array I of size n_vars x number of annotated terms in files
        in `adata.varm[varm_key]`. if I[i,j]=1 then the gene i is present in the annotation j.
    uns_key
        Sore gene sets' names in `adata.uns[uns_key]`.
    clean
        If 'True', removes the word before the first underscore for each term name (like ``REACTOME_``)
        and cuts the name to the first thirty symbols.
    genes_use_upper
        if 'True', converts genes' names from files and adata to uppercase for comparison.
    """
    
    files = [files] if isinstance(files, str) else files
    annot = []

    for file in files:
        with open(file) as f:
            p_f = [l.upper() for l in f] if genes_use_upper else f
            terms = [l.strip('\n').split() for l in p_f]

        if clean:
            terms = [[term[0].split('_', 1)[-1][:30]]+term[1:] for term in terms if term]
        annot+=terms

    var_names = adata.var_names.str.upper() if genes_use_upper else adata.var_names
    I = [[int(gene in term) for term in annot] for gene in var_names]
    I = np.asarray(I, dtype='int32')

    mask = I.sum(0) > min_genes
    if max_genes is not None:
        mask &= I.sum(0) < max_genes
    I = I[:, mask]
    adata.varm[varm_key] = I
    adata.uns[uns_key] = [term[0] for i, term in enumerate(annot) if i not in np.where(~mask)[0]]


def load_model(
    adata: sc.AnnData,
    model_path: str,
    *,
    map_location: str | torch.device = "cpu",
    training: bool = False,
    **kwargs,
) -> LineageVI:
    """
    Reconstruct a LineageVI instance and load trained weights.

    Parameters
    ----------
    adata : AnnData
        The AnnData to associate with the model (must match training genes/order).
    model_path : str
        Path to the saved .pt checkpoint (state_dict).
    map_location : str or torch.device, default "cpu"
        Where to map the weights when loading.
    training : bool, default False
        If True, leave the model in training mode. Otherwise call .eval().
    kwargs :
        Extra args passed to LineageVI(...) constructor, e.g. n_hidden, mask_key.

    Returns
    -------
    LineageVI
        Model with weights loaded and in eval() or train() mode depending on `training`.
    """
    # Import here to avoid circular import
    from .api import LineageVI
    
    # initialize a fresh instance
    inst = LineageVI(adata, **kwargs)

    # load weights
    state = torch.load(model_path, map_location=map_location)
    report = inst.model.load_state_dict(state)
    if len(report.missing_keys) or len(report.unexpected_keys):
        print("Warning: Incompatible keys when loading:", report)

    # set mode
    if training:
        inst.model.train()
    else:
        inst.model.eval()

    return inst

def build_gp_adata(
        adata,
    ) -> sc.AnnData:
        """
        Build an AnnData object in gene program (GP) space using pre-computed results.
        
        This function creates a new AnnData object where features are gene programs
        instead of genes, containing latent representations and velocities from
        pre-computed model outputs stored in the AnnData object.
        
        Parameters
        ----------
        adata : AnnData
            Single-cell data with pre-computed model outputs stored in adata.obsm.
            Requires that get_model_outputs() has been called with save_to_adata=True.
        
        Returns
        -------
        AnnData
            Gene program AnnData object with:
            - X and layers["Ms"]: Encoder mean μ (cells, L)
            - layers["z"]: Sampled latent representations (cells, L)
            - layers["logvar"]: Encoder log-variance (cells, L)
            - layers["velocity"]: Gene program velocities (cells, L)
            - obs: Copied from original adata
            - var_names: Gene program names from adata.uns["terms"]
        
        Examples
        --------
        >>> # First, get model outputs and save to AnnData
        >>> model.get_model_outputs(adata, save_to_adata=True)
        >>> 
        >>> # Build GP AnnData from pre-computed results
        >>> gp_adata = build_gp_adata(adata)
        >>> 
        >>> # Use for downstream analysis
        >>> sc.pp.neighbors(gp_adata)
        >>> sc.tl.umap(gp_adata)
        >>> sc.pl.umap(gp_adata, color="velocity")
        """
        import pandas as pd
        
        # Check for required pre-computed results
        required_keys = ["mean", "velocity_gp", "z", "logvar"]
        missing_keys = [key for key in required_keys if key not in adata.obsm]
        if missing_keys:
            raise ValueError(
                f"Missing required keys in adata.obsm: {missing_keys}. "
                f"Please run get_model_outputs(adata, save_to_adata=True) first."
            )
        
        # Get pre-computed results from AnnData
        mu     = np.asarray(adata.obsm["mean"])         # (cells, L)
        v_gp   = np.asarray(adata.obsm["velocity_gp"])  # (cells, L)
        z_arr  = np.asarray(adata.obsm["z"])            # (cells, L)
        lv_arr = np.asarray(adata.obsm["logvar"])       # (cells, L)

        # Build GP-space AnnData
        adata_gp = sc.AnnData(X=mu.astype(np.float32))
        adata_gp.obs = adata.obs.copy()

        if "terms" in adata.uns and len(adata.uns["terms"]) == mu.shape[1]:
            adata_gp.var_names = adata.uns["terms"]
        else:
            adata_gp.var_names = pd.Index([f"GP_{i}" for i in range(mu.shape[1])])

        # Treat μ as "Ms" (state) in this space; stash extras in layers
        adata_gp.layers["Ms"]      = mu.astype(np.float32)
        adata_gp.layers["z"]       = z_arr.astype(np.float32)
        adata_gp.layers["logvar"]  = lv_arr.astype(np.float32)

        # Velocity in GP space goes to obsm (not required by scVelo, just convenient)
        adata_gp.layers["velocity"] = v_gp.astype(np.float32)

        # Optional visuals
        if "X_umap" in adata.obsm:
            adata_gp.obsm["X_umap"] = adata.obsm["X_umap"].copy()

        adata_gp.var_names_make_unique()
        return adata_gp


def compute_nearest_neighbors(
    adata: sc.AnnData,
    K: Optional[int] = None,
    *,
    neighbors_key: str = 'neighbors',
    indices_key: str = 'indices',
) -> None:
    """
    Extract K-nearest neighbor indices from scanpy's pre-computed neighbor graph.
    
    This function extracts neighbor indices from adata.obsp['distances'] that were
    computed by sc.pp.neighbors(). This avoids recomputing neighbors and reuses
    scanpy's neighbor graph.
    
    Requires that sc.pp.neighbors() has been called on the AnnData object first.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with pre-computed neighbors from sc.pp.neighbors().
    K : int, optional
        Number of neighbors to extract (excluding self). If None, uses all
        available neighbors from the computed graph. The function will extract
        K+1 neighbors total (including self as the first neighbor).
    neighbors_key : str, default 'neighbors'
        Key in adata.uns where the neighbors structure is stored (from sc.pp.neighbors).
    indices_key : str, default 'indices'
        Key in adata.uns where to store the neighbor indices array.
    
    Examples
    --------
    >>> # First compute neighbors with scanpy
    >>> sc.pp.pca(adata)
    >>> sc.pp.neighbors(adata, n_neighbors=20)
    >>> 
    >>> # Extract indices from scanpy's neighbor graph
    >>> lineagevi.utils.compute_nearest_neighbors(adata, K=20)
    >>> # Access indices: adata.uns['indices']
    
    Notes
    -----
    The indices array will have shape (n_cells, K+1) where the first column contains
    the cell index itself, followed by its K nearest neighbors sorted by distance.
    """
    # Check if neighbors have been computed
    if neighbors_key not in adata.uns:
        raise ValueError(
            f"Neighbors not found. Please run sc.pp.neighbors(adata) first. "
            f"Key '{neighbors_key}' not found in adata.uns. "
            f"Available uns keys: {list(adata.uns.keys())}"
        )
    
    # Get the distances sparse matrix
    if 'distances' not in adata.obsp:
        raise ValueError(
            f"Key 'distances' not found in adata.obsp. "
            f"Available obsp keys: {list(adata.obsp.keys())}. "
            f"Make sure sc.pp.neighbors() has been called."
        )
    
    distances_matrix = adata.obsp['distances']
    
    # Ensure it's a sparse matrix
    if not sp.issparse(distances_matrix):
        raise TypeError(f"Expected sparse matrix in adata.obsp['distances'], got {type(distances_matrix)}")
    
    n_cells = distances_matrix.shape[0]
    indices_list = []
    
    # Extract indices from sparse CSR matrix
    # Each row contains the neighbors for that cell
    for i in range(n_cells):
        # Get the row (neighbors and distances for cell i)
        row_start = distances_matrix.indptr[i]
        row_end = distances_matrix.indptr[i + 1]
        
        # Column indices are the neighbor indices
        neighbor_indices = distances_matrix.indices[row_start:row_end]
        # Values are the distances
        neighbor_distances = distances_matrix.data[row_start:row_end]
        
        # Sort by distance (closest first)
        sorted_idx = np.argsort(neighbor_distances)
        neighbor_indices = neighbor_indices[sorted_idx]
        
        # Include self as first neighbor
        full_indices = np.concatenate([[i], neighbor_indices])
        
        # Limit to K+1 if K is specified
        if K is not None:
            full_indices = full_indices[:K + 1]
        
        indices_list.append(full_indices)
    
    # Convert to numpy array - pad with -1 if rows have different lengths
    if indices_list:
        max_neighbors = max(len(idx) for idx in indices_list)
        indices_array = np.full((n_cells, max_neighbors), -1, dtype=np.int64)
        
        for i, idx in enumerate(indices_list):
            indices_array[i, :len(idx)] = idx
    else:
        indices_array = np.array([], dtype=np.int64).reshape(n_cells, 0)
    
    adata.uns[indices_key] = indices_array

def compute_cluster_embedding_similarity(
    adata: Optional[sc.AnnData] = None,
    *,
    embeddings: Optional[np.ndarray] = None,
    cluster_names: Optional[Union[list, np.ndarray]] = None,
    embeddings_key: str = 'cluster_embeddings',
    names_key: str = 'cluster_names',
) -> pd.DataFrame:
    """
    Compute cosine similarity matrix between all cluster embeddings.
    
    This function computes pairwise cosine similarity between cluster embeddings,
    which can be used to visualize relationships between clusters in embedding space.
    
    Parameters
    ----------
    adata : AnnData, optional
        AnnData object containing cluster embeddings in adata.uns.
        If None, embeddings and cluster_names must be provided directly.
    embeddings : np.ndarray, optional
        Cluster embeddings array of shape (n_clusters, embedding_dim).
        If None, will be read from adata.uns[embeddings_key].
    cluster_names : list or np.ndarray, optional
        List of cluster names of length n_clusters.
        If None, will be read from adata.uns[names_key].
    embeddings_key : str, default 'cluster_embeddings'
        Key in adata.uns where cluster embeddings are stored.
    names_key : str, default 'cluster_names'
        Key in adata.uns where cluster names are stored.
    
    Returns
    -------
    pd.DataFrame
        Cosine similarity matrix of shape (n_clusters, n_clusters).
        Rows and columns are indexed by cluster names.
        Values range from -1 to 1, where 1 indicates identical embeddings.
        The matrix is symmetric.
    
    Examples
    --------
    >>> # Compute similarity matrix from AnnData
    >>> similarity_df = lineagevi.utils.compute_cluster_embedding_similarity(adata)
    >>> 
    >>> # Plot as heatmap
    >>> import lineagevi.plots as lv_plots
    >>> fig, ax = lv_plots.plot_cluster_alignment_matrix(similarity_df, title='Cluster Embedding Similarity')
    >>> 
    >>> # Use custom embeddings
    >>> embeddings = np.random.randn(10, 32)  # 10 clusters, 32-dim embeddings
    >>> names = ['Cluster_' + str(i) for i in range(10)]
    >>> similarity_df = lineagevi.utils.compute_cluster_embedding_similarity(
    ...     embeddings=embeddings,
    ...     cluster_names=names
    ... )
    """
    # Get embeddings and names from adata if not provided directly
    if embeddings is None:
        if adata is None:
            raise ValueError("Either adata or embeddings must be provided")
        if embeddings_key not in adata.uns:
            raise KeyError(
                f"Cluster embeddings not found in adata.uns['{embeddings_key}']. "
                f"Please run get_model_outputs(adata, save_to_adata=True) first. "
                f"Available uns keys: {list(adata.uns.keys())}"
            )
        embeddings = np.asarray(adata.uns[embeddings_key])
    
    if cluster_names is None:
        if adata is None:
            raise ValueError("Either adata or cluster_names must be provided")
        if names_key not in adata.uns:
            raise KeyError(
                f"Cluster names not found in adata.uns['{names_key}']. "
                f"Please run get_model_outputs(adata, save_to_adata=True) first. "
                f"Available uns keys: {list(adata.uns.keys())}"
            )
        cluster_names = adata.uns[names_key]
    
    # Convert to numpy arrays
    embeddings = np.asarray(embeddings)
    cluster_names = np.asarray(cluster_names)
    
    # Validate shapes
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D array (n_clusters, embedding_dim), got shape {embeddings.shape}")
    
    n_clusters = embeddings.shape[0]
    if len(cluster_names) != n_clusters:
        raise ValueError(
            f"Number of cluster names ({len(cluster_names)}) does not match "
            f"number of embeddings ({n_clusters})"
        )
    
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    # Convert cluster names to strings for DataFrame
    cluster_names_str = [str(name) for name in cluster_names]
    
    # Create DataFrame with cluster names as row and column indices
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=cluster_names_str,
        columns=cluster_names_str
    )
    
    return similarity_df


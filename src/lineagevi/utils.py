from __future__ import annotations

import json
import os
import re
from typing import Optional, Union, TYPE_CHECKING
from urllib.parse import urlencode
from urllib.request import urlopen

import torch
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

if TYPE_CHECKING:
    from .api import LineageVI  # Only imported for type checking, not at runtime

def add_annotations(adata, files, min_genes=0, max_genes=None, varm_key='mask', uns_key='terms',
                clean=True, genes_use_upper=True):
    """\
    Add annotations to an AnnData object from files.

    Parameters
    ----------
    adata
        Annotated data matrix.
    files
        Path(s) to annotation file(s). GMT format (tab-separated: name \\t description \\t gene1 \\t gene2 ...)
        is parsed so the full term name is preserved (including spaces). Space-separated lines
        are supported as fallback (first token = name, rest = genes).
    min_genes
        Only include gene sets which have the total number of genes in adata
        greater than this value.
    max_genes
        Only include gene sets which have the total number of genes in adata
        less than this value.
    varm_key
        Store the binary array I of size n_vars x number of annotated terms in files
        in `adata.varm[varm_key]`. if mask[i,j]=1 then the gene i is present in the annotation j.
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
            raw_lines = [l.rstrip("\n") for l in f]
        terms = []
        for line in raw_lines:
            if not line:
                continue
            # GMT format is tab-separated: name \t [description] \t gene1 \t gene2 ...
            if "\t" in line:
                parts = line.split("\t")
                name = parts[0].strip()
                # genes are from part 2 onward (part 1 is often empty description)
                genes = [p.strip() for p in parts[2:] if p.strip()]
                if not name:
                    continue
                if genes_use_upper:
                    genes = [g.upper() for g in genes]
                terms.append([name] + genes)
            else:
                # fallback: space-separated (first token = name, rest = genes)
                toks = line.split()
                if toks:
                    if genes_use_upper:
                        toks = [toks[0]] + [g.upper() for g in toks[1:]]
                    terms.append(toks)
        # Apply clean to term names only (strip prefix and/or truncate to 30 chars)
        if clean:
            terms = [[t[0].split("_", 1)[-1][:30]] + t[1:] for t in terms if t]
        annot += terms

    var_names = adata.var_names.str.upper() if genes_use_upper else adata.var_names
    I = [[int(gene in term) for term in annot] for gene in var_names]
    I = np.asarray(I, dtype='int32')

    mask = I.sum(0) > min_genes
    if max_genes is not None:
        mask &= I.sum(0) < max_genes
    I = I[:, mask]
    adata.varm[varm_key] = I
    adata.uns[uns_key] = [term[0] for i, term in enumerate(annot) if i not in np.where(~mask)[0]]


def preprocess_for_lineagevi(
    dataset_name: Optional[str] = None,
    adata_path: Optional[str] = None,
    adata: Optional[sc.AnnData] = None,
    annotation_file: Optional[Union[str, list[str]]] = None,
    min_shared_counts: int = 20,
    n_top_genes: int = 2000,
    min_genes_per_term: int = 12,
    n_pcs: int = 100,
    n_neighbors: int = 200,
    K_neighbors: int = 20,
    skip_if_preprocessed: bool = True,
    cluster_key: Optional[str] = None,
    clean_terms: bool = True,
) -> sc.AnnData:
    """
    Preprocess AnnData for LineageVI training.
    
    This function performs the complete preprocessing pipeline:
    1. Loads data (from scvelo datasets, file path, or provided AnnData)
    2. Sets X and counts layers from unspliced/spliced
    3. Adds gene set annotations (if provided)
    4. Filters and normalizes data
    5. Filters annotation terms
    6. Computes moments and neighbors
    
    Parameters
    ----------
    dataset_name : str, optional
        Name of dataset to load from scvelo (e.g., 'pancreas', 'gastrulation').
        If provided, will call scv.datasets.{dataset_name}().
    adata_path : str, optional
        Path to AnnData file (.h5ad). Used if dataset_name is None.
    adata : AnnData, optional
        Pre-loaded AnnData object. Used if both dataset_name and adata_path are None.
    annotation_file : str or list[str], optional
        Full path(s) to annotation file(s) (e.g., .gmt files).
        If a single string, will be converted to a list. If None, annotations are skipped.
    min_shared_counts : int, default 20
        Minimum shared counts for filtering in scv.pp.filter_and_normalize.
    n_top_genes : int, default 2000
        Number of top highly variable genes to keep.
    min_genes_per_term : int, default 12
        Minimum genes per annotation term to retain.
    n_pcs : int, default 100
        Number of PCs for moments computation.
    n_neighbors : int, default 200
        Number of neighbors for moments computation (smoothing parameter).
    K_neighbors : int, default 20
        Number of neighbors to extract for model (K+1 including self).
    skip_if_preprocessed : bool, default True
        If True, skip steps that appear already done (checks for 'Mu', 'Ms', 'mask', etc.).
    cluster_key : str, optional
        Key for storing cluster labels. If None, uses 'leiden' after clustering.
    clean_terms : bool, default True
        If True, annotation term names are shortened (prefix before first '_' removed,
        then truncated to 30 characters). If False, full term names from the file are kept.
    
    Returns
    -------
    adata : AnnData
        Preprocessed AnnData ready for LineageVI training.
    
    Examples
    --------
    >>> import lineagevi as lvi
    >>> 
    >>> # Load from scvelo dataset
    >>> adata = lvi.utils.preprocess_for_lineagevi(
    ...     dataset_name='pancreas',
    ...     annotation_file="/path/to/annotations/msigdb_development.gmt",
    ...     n_pcs=100,
    ...     n_neighbors=200,
    ... )
    >>> 
    >>> # Load from file
    >>> adata = lvi.utils.preprocess_for_lineagevi(
    ...     adata_path="data.h5ad",
    ...     annotation_file="/path/to/annotations/msigdb_development.gmt",
    ... )
    >>> 
    >>> # Use pre-loaded AnnData
    >>> import scanpy as sc
    >>> adata = sc.read_h5ad("data.h5ad")
    >>> adata = lvi.utils.preprocess_for_lineagevi(
    ...     adata=adata,
    ...     annotation_file="/path/to/annotations/msigdb_development.gmt",
    ... )
    >>> 
    >>> # Initialize and train model
    >>> vae = lvi.LineageVI(adata, ...)
    >>> vae.fit(...)
    """
    import scvelo as scv
    
    # Load data
    if dataset_name is not None:
        print(f"Loading dataset '{dataset_name}' from scvelo...")
        dataset_func = getattr(scv.datasets, dataset_name, None)
        if dataset_func is None:
            raise ValueError(
                f"Dataset '{dataset_name}' not found in scv.datasets. "
                f"Available datasets: {[x for x in dir(scv.datasets) if not x.startswith('_')]}"
            )
        adata = dataset_func()
        print(f"Loaded {adata.n_obs} cells, {adata.n_vars} genes")
    elif adata_path is not None:
        print(f"Loading data from {adata_path}...")
        adata = sc.read_h5ad(adata_path)
        print(f"Loaded {adata.n_obs} cells, {adata.n_vars} genes")
    elif adata is not None:
        print(f"Using provided AnnData: {adata.n_obs} cells, {adata.n_vars} genes")
    else:
        raise ValueError(
            "Must provide one of: dataset_name, adata_path, or adata"
        )
    
    # Check if already preprocessed
    if skip_if_preprocessed:
        has_moments = 'Mu' in adata.layers and 'Ms' in adata.layers
        has_annotations = 'mask' in adata.varm and 'terms' in adata.uns
        has_neighbors = 'indices' in adata.uns
        if has_moments and has_annotations and has_neighbors:
            print("Data appears already preprocessed. Skipping preprocessing.")
            return adata
    
    print("Starting preprocessing pipeline...")
    
    # Step 1: Set X and counts from unspliced/spliced layers
    if 'unspliced' in adata.layers and 'spliced' in adata.layers:
        print("Setting X and counts layers from unspliced/spliced...")
        adata.X = adata.layers['unspliced'].copy() + adata.layers['spliced'].copy()
        adata.layers['counts'] = adata.X.copy()
    elif 'counts' not in adata.layers:
        print("Warning: No 'unspliced'/'spliced' layers found. Using adata.X as counts.")
        adata.layers['counts'] = adata.X.copy()
    
    # Step 2: Add annotations (if provided)
    if annotation_file is not None:
        # Convert single string to list
        annotation_files = [annotation_file] if isinstance(annotation_file, str) else annotation_file
        print(f"Adding annotations from {len(annotation_files)} file(s)...")
        add_annotations(
            adata,
            files=annotation_files,
            min_genes=min_genes_per_term,
            varm_key='mask',
            uns_key='terms',
            clean=clean_terms,
            genes_use_upper=True
        )
        print(f"  Added {adata.varm['mask'].shape[1]} annotation terms")
        
        # Filter genes to only those in at least one annotation
        n_genes_before = adata.n_vars
        adata._inplace_subset_var(adata.varm['mask'].sum(1) > 0)
        n_genes_after = adata.n_vars
        print(f"  Filtered to {n_genes_after} genes present in annotations (from {n_genes_before})")
    
    # Step 3: Filter and normalize
    print(f"Filtering and normalizing (min_shared_counts={min_shared_counts}, n_top_genes={n_top_genes})...")
    n_genes_before = adata.n_vars
    scv.pp.filter_and_normalize(
        adata,
        min_shared_counts=min_shared_counts,
        n_top_genes=n_top_genes,
        subset_highly_variable=True,
        log=True
    )
    n_genes_after = adata.n_vars
    print(f"  Filtered to {n_genes_after} highly variable genes (from {n_genes_before})")
    
    # Step 4: Filter annotation terms (if annotations were added)
    if 'mask' in adata.varm:
        print(f"Filtering annotation terms (min_genes_per_term={min_genes_per_term})...")
        n_terms_before = adata.varm['mask'].shape[1]
        select_terms = adata.varm['mask'].sum(0) > min_genes_per_term
        adata.uns['terms'] = np.array(adata.uns['terms'])[select_terms].tolist()
        adata.varm['mask'] = adata.varm['mask'][:, select_terms]
        n_terms_after = adata.varm['mask'].shape[1]
        print(f"  Retained {n_terms_after} terms (from {n_terms_before})")
        # Filter genes not in any retained term
        n_genes_before = adata.n_vars
        adata._inplace_subset_var(adata.varm['mask'].sum(1) > 0)
        n_genes_after = adata.n_vars
        print(f"  Filtered to {n_genes_after} genes in retained terms (from {n_genes_before})")

    # Step 5: Compute moments and neighbors
    print(f"Computing moments (n_pcs={n_pcs}, n_neighbors={n_neighbors})...")
    scv.pp.moments(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
    print("  Moments computed: 'Mu' and 'Ms' layers added")
    
    # Cluster cells
    if cluster_key is None:
        cluster_key = 'leiden'
    print("Computing Leiden clustering...")
    sc.tl.leiden(adata, key_added=cluster_key)
    print(f"  Clustering stored in adata.obs['{cluster_key}']")
    
    # Extract neighbor indices for model
    print(f"Extracting {K_neighbors} nearest neighbors...")
    get_neighbor_indices(adata, K=K_neighbors, neighbors_key='neighbors', indices_key='indices')
    print(f"  Neighbor indices stored in adata.uns['indices']")
    
    print(f"Preprocessing complete! Final data: {adata.n_obs} cells, {adata.n_vars} genes")
    if 'mask' in adata.varm:
        print(f"  {adata.varm['mask'].shape[1]} annotation terms")
    
    return adata

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
    import json
    from pathlib import Path
    
    # Try to load saved configuration first
    model_dir = Path(model_path).parent
    config_path = model_dir / "model_config.json"
    saved_config = None
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
            print(f"Loaded model configuration from {config_path}")
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
    
    # Use saved config to override defaults, but kwargs take precedence
    if saved_config is not None:
        # Map saved config keys to kwargs (only if not already in kwargs)
        config_mapping = {
            "n_hidden": "n_hidden",
            "n_layers": "n_layers",
            "dropout": "dropout",
            "cluster_key": "cluster_key",
            "cluster_embedding_dim": "cluster_embedding_dim",
        }
        for config_key, kwarg_key in config_mapping.items():
            if config_key in saved_config and saved_config[config_key] is not None:
                if kwarg_key not in kwargs:
                    kwargs[kwarg_key] = saved_config[config_key]
    
    # Load state_dict to infer any remaining parameters
    state = torch.load(model_path, map_location=map_location)
    
    # Infer cluster_embedding_dim and cluster_key from state_dict if present
    # Check for both old and new key formats
    cluster_emb_key = None
    if "cluster_embedding.embeddings.weight" in state:
        cluster_emb_key = "cluster_embedding.embeddings.weight"
    elif "cluster_embedding.weight" in state:
        cluster_emb_key = "cluster_embedding.weight"
    
    if cluster_emb_key is not None:
        if "cluster_embedding_dim" not in kwargs:
            cluster_embedding_dim = state[cluster_emb_key].shape[1]
            kwargs["cluster_embedding_dim"] = cluster_embedding_dim
        
        # If cluster_key not provided, try to infer from common keys
        if "cluster_key" not in kwargs:
            # Try common cluster key names
            common_keys = ["leiden", "clusters", "cluster", "cell_type", "annotation"]
            for key in common_keys:
                if key in adata.obs.columns:
                    kwargs["cluster_key"] = key
                    break
            # If still not found, raise an error
            if "cluster_key" not in kwargs:
                raise ValueError(
                    f"Saved model has cluster embeddings but no cluster_key provided. "
                    f"Please specify cluster_key. Available obs columns: {list(adata.obs.columns)}"
                )
    
    # initialize a fresh instance with inferred parameters
    inst = LineageVI(adata, **kwargs)

    # load weights with strict=False to handle architecture changes gracefully
    report = inst.model.load_state_dict(state, strict=False)
    if len(report.missing_keys) or len(report.unexpected_keys):
        print("Warning: Incompatible keys when loading:")
        if len(report.missing_keys) > 0:
            print(f"  Missing keys: {report.missing_keys}")
        if len(report.unexpected_keys) > 0:
            print(f"  Unexpected keys: {report.unexpected_keys}")

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
        required_keys = ["mean", "velocity_gp", "logvar"]
        missing_keys = [key for key in required_keys if key not in adata.obsm]
        if missing_keys:
            raise ValueError(
                f"Missing required keys in adata.obsm: {missing_keys}. "
                f"Please run get_model_outputs(adata, save_to_adata=True) first."
            )
        
        # Get pre-computed results from AnnData
        mu     = np.asarray(adata.obsm["mean"])         # (cells, L)
        v_gp   = np.asarray(adata.obsm["velocity_gp"])  # (cells, L)
        lv_arr = np.asarray(adata.obsm["logvar"])       # (cells, L)

        if v_gp.size > 0 and np.allclose(v_gp, 0.0):
            raise ValueError(
                "adata.obsm['velocity_gp'] is all zeros. scvelo.tl.velocity_graph will fail. "
                "Re-run get_model_outputs(adata, save_to_adata=True) so that velocity_gp is "
                "computed from gene velocity and gene decoder weights (no kernel restart needed)."
            )

        # Build GP-space AnnData
        adata_gp = sc.AnnData(X=mu.astype(np.float32))
        adata_gp.obs = adata.obs.copy()

        if "terms" in adata.uns and len(adata.uns["terms"]) == mu.shape[1]:
            adata_gp.var_names = adata.uns["terms"]
        else:
            adata_gp.var_names = pd.Index([f"GP_{i}" for i in range(mu.shape[1])])

        # Treat μ as "Ms" (state) in this space; stash extras in layers
        adata_gp.layers["Ms"]      = mu.astype(np.float32)
        adata_gp.layers["logvar"]  = lv_arr.astype(np.float32)

        # Velocity in GP space goes to obsm (not required by scVelo, just convenient)
        adata_gp.layers["velocity"] = v_gp.astype(np.float32)

        # Optional visuals
        if "X_umap" in adata.obsm:
            adata_gp.obsm["X_umap"] = adata.obsm["X_umap"].copy()

        adata_gp.var_names_make_unique()
        return adata_gp


def get_neighbor_indices(
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
    >>> lineagevi.utils.get_neighbor_indices(adata, K=20)
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
    >>> # Plot as heatmap (e.g. with seaborn)
    >>> import seaborn as sns
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(figsize=(8, 6))
    >>> sns.heatmap(similarity_df, cmap='RdBu_r', center=0, annot=True, ax=ax)
    >>> ax.set_title('Cluster Embedding Similarity')
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


# -------------------------------------------------------------------------------
# Enrichr gene set libraries
# -------------------------------------------------------------------------------

_ENRICHR_BASE = "https://maayanlab.cloud/Enrichr"


def list_enrichr_libraries() -> pd.DataFrame:
    """\
    List available Enrichr gene set libraries.

    Fetches library metadata from the Enrichr API (datasetStatistics).
    Use this to explore library names before downloading with
    :func:`download_enrichr_library` or :func:`download_enrichr_libraries`.

    Returns
    -------
    pandas.DataFrame
        Columns include ``libraryName``, ``numTerms``, and other stats.
        Use ``df['libraryName']`` to get names for downloading.
    """
    url = f"{_ENRICHR_BASE}/datasetStatistics"
    with urlopen(url) as resp:
        data = json.load(resp)
    rows = []
    for lib in data.get("statistics", []):
        row = {
            "libraryName": lib.get("libraryName"),
            "numTerms": lib.get("numTerms"),
        }
        for k, v in lib.items():
            if k not in row:
                row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)


def _enrichr_line_to_gmt_parts(line: str) -> Optional[tuple[str, str, list[str]]]:
    """Parse one line from Enrichr geneSetLibrary text into (name, description, genes)."""
    line = line.strip()
    if not line:
        return None
    # Tab-separated: name \t [desc \t] gene1 \t gene2 ...
    if "\t" in line:
        parts = line.split("\t")
        name = parts[0].strip()
        # If 2 parts only, second might be description or single gene
        if len(parts) == 2:
            desc = ""
            genes = [parts[1].strip()] if parts[1].strip() else []
        else:
            desc = parts[1].strip() if len(parts) > 2 else ""
            genes = [p.strip() for p in parts[2:] if p.strip()]
        return (name, desc, genes)
    # Space-separated: infer boundary between set name and genes (gene-like tokens at end)
    tokens = line.split()
    if len(tokens) < 2:
        return None
    # Gene-like: alphanumeric/dash/dot, typical length
    def looks_like_gene(t: str) -> bool:
        return bool(re.match(r"^[A-Za-z0-9\-\.]+$", t) and 2 <= len(t) <= 15)

    best_i = len(tokens)
    for i in range(1, len(tokens)):
        tail = tokens[i:]
        if len(tail) >= 2 and all(looks_like_gene(t) for t in tail):
            best_i = i
            break
    name = " ".join(tokens[:best_i])
    genes = tokens[best_i:]
    return (name, "", genes)


def download_enrichr_library(
    library_name: str,
    output_path: Optional[str] = None,
    skip_existing: bool = True,
) -> str:
    """\
    Download one Enrichr gene set library and save it as a GMT file.

    Parameters
    ----------
    library_name : str
        Exact library name as returned by :func:`list_enrichr_libraries`
        (e.g. ``'KEGG_2021_Human'``).
    output_path : str, optional
        Path for the output ``.gmt`` file. If None, uses ``{library_name}.gmt``
        in the current directory (with unsafe characters replaced by underscores).
    skip_existing : bool, default True
        If True and the output file already exists, skip the download and return
        the path without overwriting.

    Returns
    -------
    str
        Path to the written (or existing) GMT file.
    """
    if output_path is None:
        safe = re.sub(r"[^\w\-.]", "_", library_name)
        output_path = f"{safe}.gmt"

    if skip_existing and os.path.isfile(output_path):
        return output_path

    params = urlencode({"mode": "text", "libraryName": library_name})
    url = f"{_ENRICHR_BASE}/geneSetLibrary?{params}"
    with urlopen(url) as resp:
        text = resp.read().decode("utf-8")

    lines_out = []
    for raw_line in text.splitlines():
        parsed = _enrichr_line_to_gmt_parts(raw_line)
        if parsed is None:
            continue
        name, desc, genes = parsed
        if not genes:
            continue
        line_gmt = "\t".join([name, desc] + genes)
        lines_out.append(line_gmt)

    with open(output_path, "w") as f:
        f.write("\n".join(lines_out))

    return output_path


def download_enrichr_libraries(
    library_names: Union[str, list[str]],
    output_dir: Optional[str] = None,
    skip_existing: bool = True,
) -> list[str]:
    """\
    Download one or more Enrichr gene set libraries and save them as GMT files.

    Parameters
    ----------
    library_names : str or list of str
        Library name(s) from :func:`list_enrichr_libraries`.
    output_dir : str, optional
        Directory to write ``.gmt`` files into. If None, files are written
        to the current directory. Filenames are derived from library names.
    skip_existing : bool, default True
        If True, skip downloading when the output GMT file already exists.

    Returns
    -------
    list of str
        Paths to the written (or existing) GMT files.
    """
    names = [library_names] if isinstance(library_names, str) else list(library_names)
    written = []
    for lib in names:
        out_path = None
        if output_dir is not None:
            safe = re.sub(r"[^\w\-.]", "_", lib)
            out_path = os.path.join(output_dir, f"{safe}.gmt")
        written.append(
            download_enrichr_library(lib, output_path=out_path, skip_existing=skip_existing)
        )
    return written


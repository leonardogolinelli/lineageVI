from __future__ import annotations

import torch
import scanpy as sc
import numpy as np

from .api import LineageVI  # avoid importing the top-level package to prevent circulars

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
        If 'True', removes the word before the first underscore for each term name (like 'REACTOME_')
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


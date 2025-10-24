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
        model: LineageVI,
        n_samples: int = 1,
        return_negative_velo: bool = True,
        base_seed: int | None = None,
    ) -> sc.AnnData:
        """
        Return an AnnData in GP space (features = L).

        - X and layers["Ms"] hold μ (encoder mean).
        - layers["z"] holds sampled z (averaged over samples if n_samples>1).
        - layers["logvar"] holds encoder log-variance (averaged if n_samples>1).
        - obsm["velocity_gp"] holds GP velocity.
        """
        outs = model._get_model_outputs(
            adata=adata,
            n_samples=n_samples,
            return_mean=True,              # recon/vel/vel_gp averaged; z/mean/logvar kept per-sample by design
            return_negative_velo=return_negative_velo,
            base_seed=base_seed,
            save_to_adata=False,
        )

        mu     = np.asarray(outs["mean"])         # (cells, L) or (n_samples, cells, L)
        v_gp   = np.asarray(outs["velocity_gp"])  # (cells, L) or (n_samples, cells, L)
        z_arr  = np.asarray(outs["z"])            # (n_samples, cells, L) or (cells, L)
        lv_arr = np.asarray(outs["logvar"])       # (n_samples, cells, L) or (cells, L)

        # collapse any sample axis defensively to (cells, L)
        def _to_2d(a: np.ndarray) -> np.ndarray:
            return a.mean(axis=0) if a.ndim == 3 else a

        mu     = _to_2d(mu)
        v_gp   = _to_2d(v_gp)
        z_arr  = _to_2d(z_arr)
        lv_arr = _to_2d(lv_arr)

        if not (mu.ndim == v_gp.ndim == z_arr.ndim == lv_arr.ndim == 2):
            raise RuntimeError(
                f"Expected 2D arrays after collapsing sample axis; got shapes "
                f"mu={mu.shape}, v_gp={v_gp.shape}, z={z_arr.shape}, logvar={lv_arr.shape}"
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
        adata_gp.layers["z"]       = z_arr.astype(np.float32)
        adata_gp.layers["logvar"]  = lv_arr.astype(np.float32)

        # Velocity in GP space goes to obsm (not required by scVelo, just convenient)
        adata_gp.layers["velocity"] = v_gp.astype(np.float32)

        # Optional visuals
        if "X_umap" in adata.obsm:
            adata_gp.obsm["X_umap"] = adata.obsm["X_umap"].copy()

        adata_gp.var_names_make_unique()
        return adata_gp


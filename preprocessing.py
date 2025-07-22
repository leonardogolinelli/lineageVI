import os
from datetime import datetime
import scanpy as sc
import scvelo as scv
from sklearn.neighbors import NearestNeighbors
from utils import add_annotations

class Preprocessor:
    def __init__(
        self,
        n_pca: int = 100,
        min_shared_counts: int = 20,
        n_top_genes: int = 2000,
        norm_log: bool = True,
        n_neighbors: int = 200,
        knn_neighbors: int = 20,
        annotation_files: list = None,
        annotation_min_genes: int = 12,
        varm_key: str = 'I',
        uns_key: str = 'terms',
        clean: bool = True,
        genes_use_upper: bool = True,
        save_path: str = None,
    ):
        """
        Downloads the pancreas dataset automatically, then:
          1) PCA
          2) filter/normalize + HVG
          3) scVelo neighbors & moments
          4) explicit KNN indices on X_pca
          5) optional gene‑set annotation (via add_annotations)
          6) optional save to timestamped folder

        If save_path is None, nothing is written to disk.
        """
        self.n_pca = n_pca
        self.min_shared_counts = min_shared_counts
        self.n_top_genes = n_top_genes
        self.norm_log = norm_log
        self.n_neighbors = n_neighbors
        self.knn_neighbors = knn_neighbors
        self.annotation_files = annotation_files or []
        self.annotation_min_genes = annotation_min_genes
        self.varm_key = varm_key
        self.uns_key = uns_key
        self.clean = clean
        self.genes_use_upper = genes_use_upper
        self.save_path = save_path

    def run(self):
        # 1) download
        adata = scv.datasets.pancreas()

        # 2) PCA
        sc.pp.pca(adata, n_comps=self.n_pca)

        # 3) filter, normalize, select HVG, log-transform
        scv.pp.filter_and_normalize(
            adata,
            min_shared_counts=self.min_shared_counts,
            n_top_genes=self.n_top_genes,
            subset_highly_variable=True,
            log=self.norm_log,
        )

        # 4) neighbors & moments (scVelo)
        scv.pp.neighbors(adata, n_neighbors=self.n_neighbors)
        scv.pp.moments(adata)

        # 5) explicit KNN on X_pca
        nbrs = NearestNeighbors(
            n_neighbors=self.knn_neighbors + 1,
            metric='euclidean',
            n_jobs=-1
        )
        nbrs.fit(adata.obsm['X_pca'])
        _, indices = nbrs.kneighbors(adata.obsm['X_pca'])
        adata.uns['indices'] = indices

        # 6) optional gene‑set annotation
        if self.annotation_files:
            add_annotations(
                adata,
                files=self.annotation_files,
                min_genes=self.annotation_min_genes,
                varm_key=self.varm_key,
                uns_key=self.uns_key,
                clean=self.clean,
                genes_use_upper=self.genes_use_upper,
            )
            # subset var to those with any annotation
            mask = adata.varm[self.varm_key].sum(axis=1) > 0
            adata._inplace_subset_var(mask)

        # 7) optional save
        if self.save_path:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(self.save_path, f"processed_input_anndatas_{ts}")
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, "pancreas.h5ad")
            adata.write_h5ad(out_file)
            print(f"→ Written processed AnnData to {out_file}")

        return adata
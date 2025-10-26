Data Preprocessing Tutorial
============================

This tutorial demonstrates how to preprocess single-cell RNA sequencing data for use with LineageVI. The preprocessing steps include data loading, quality control, normalization, and preparation for velocity analysis.

.. note::
   This tutorial uses the preprocessing notebook v2, which contains the most up-to-date preprocessing workflow.

The following notebook shows the complete preprocessing workflow:

.. code-block:: python

   # Load autoreload extension
   %load_ext autoreload
   %autoreload 2

   # Import required libraries
   import lineagevi as linvi
   import scanpy as sc
   import scvelo as scv
   import numpy as np
   import os

   # Load pancreas dataset
   raw_path = '/Users/lgolinelli/git/lineageVI/notebooks/data/inputs/anndata/raw'
   dataset_name = 'pancreas'
   raw_adata_path = os.path.join(raw_path, dataset_name + '.h5ad')
   os.makedirs(raw_path, exist_ok=True)
   adata = scv.datasets.pancreas(raw_adata_path)

   # Prepare count data
   adata.layers['counts'] = adata.layers['spliced'].copy() + adata.layers['unspliced'].copy()
   adata.X = adata.layers['counts'].copy()

   # Store original counts
   adata.layers['unspliced_counts'] = adata.layers['unspliced'].copy()
   adata.layers['spliced_counts'] = adata.layers['spliced'].copy()

   # Filter and normalize data
   scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000, subset_highly_variable=True, log=True)

   # Select gene programs
   select_terms = adata.varm['I'].sum(0)>12
   adata.uns['terms'] = np.array(adata.uns['terms'])[select_terms].tolist()
   adata.varm['I'] = adata.varm['I'][:, select_terms]
   adata._inplace_subset_var(adata.varm['I'].sum(1)>0)

   # Additional preprocessing steps
   adata.X = adata.layers['counts'].copy()
   sc.pp.normalize_total(adata)
   sc.pp.log1p(adata)
   sc.pp.scale(adata)
   sc.pp.pca(adata, n_comps=100)
   scv.pp.neighbors(adata, n_neighbors=200, use_rep='X_pca')
   scv.pp.moments(adata, n_neighbors=200)
   sc.pp.scale(adata, layer='Mu')
   sc.pp.scale(adata, layer='Ms')

   # Compute nearest neighbors for velocity analysis
   from sklearn.neighbors import NearestNeighbors
   nbrs = NearestNeighbors(n_neighbors=20 + 1, metric='euclidean', n_jobs=-1)
   nbrs.fit(adata.obsm['X_pca'])
   distances, indices = nbrs.kneighbors(adata.obsm['X_pca'])
   adata.uns['indices'] = indices

   # Add gene annotations
   annotation_path = '/Users/lgolinelli/git/lineageVI/notebooks/data/inputs/gene_sets/'
   annotation_name = 'msigdb_development_or_pancreas.gmt'
   file_path = os.path.join(annotation_path, annotation_name)
   os.makedirs(annotation_path, exist_ok=True)

   linvi.utils.add_annotations(
       adata, 
       files=[file_path],
       min_genes=12,
       varm_key='I',
       uns_key='terms',
       clean=True,
       genes_use_upper=True)

   adata._inplace_subset_var(adata.varm['I'].sum(1) > 0)

   # Save processed data
   processed_dir_path = '/Users/lgolinelli/git/lineageVI/notebooks/data/inputs/anndata/processed/'
   processed_adata_path = os.path.join(processed_dir_path, dataset_name + '.h5ad')
   os.makedirs(processed_dir_path, exist_ok=True)
   adata.write_h5ad(processed_adata_path)

   # Visualize results
   sc.tl.umap(adata)
   sc.pl.umap(adata, color='clusters')

Additional Resources
--------------------

- :doc:`../api/lineagevi` - API reference for the LineageVI class
- :doc:`../api/model` - Detailed model documentation
- :doc:`training` - Next step: model training tutorial

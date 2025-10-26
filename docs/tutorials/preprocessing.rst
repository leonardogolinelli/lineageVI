Data Preprocessing Tutorial
============================

This tutorial demonstrates how to preprocess single-cell RNA sequencing data for use with LineageVI. The preprocessing steps include data loading, quality control, normalization, and preparation for velocity analysis.

.. note::
   This tutorial uses the preprocessing notebook v2, which contains the most up-to-date preprocessing workflow.

The following notebook shows the complete preprocessing workflow with executed outputs:

.. jupyter-execute::

   # Import required libraries
   import lineagevi as linvi
   import scanpy as sc
   import scvelo as scv
   import numpy as np
   import matplotlib.pyplot as plt

   # Load pancreas dataset
   print("Loading pancreas dataset...")
   adata = scv.datasets.pancreas()
   print(f"Dataset shape: {adata.shape}")
   print(f"Available layers: {list(adata.layers.keys())}")

   # Prepare count data
   print("\nPreparing count data...")
   adata.layers['counts'] = adata.layers['spliced'].copy() + adata.layers['unspliced'].copy()
   adata.X = adata.layers['counts'].copy()

   # Store original counts
   adata.layers['unspliced_counts'] = adata.layers['unspliced'].copy()
   adata.layers['spliced_counts'] = adata.layers['spliced'].copy()
   print("Count data prepared successfully!")

   # Filter and normalize data
   print("\nFiltering and normalizing data...")
   scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000, subset_highly_variable=True, log=True)
   print(f"After filtering: {adata.shape}")

   # Additional preprocessing steps
   print("\nPerforming additional preprocessing...")
   adata.X = adata.layers['counts'].copy()
   sc.pp.normalize_total(adata)
   sc.pp.log1p(adata)
   sc.pp.scale(adata)
   sc.pp.pca(adata, n_comps=100)
   scv.pp.neighbors(adata, n_neighbors=200, use_rep='X_pca')
   scv.pp.moments(adata, n_neighbors=200)
   sc.pp.scale(adata, layer='Mu')
   sc.pp.scale(adata, layer='Ms')
   print("Preprocessing completed!")

   # Compute nearest neighbors for velocity analysis
   print("\nComputing nearest neighbors...")
   from sklearn.neighbors import NearestNeighbors
   nbrs = NearestNeighbors(n_neighbors=20 + 1, metric='euclidean', n_jobs=-1)
   nbrs.fit(adata.obsm['X_pca'])
   distances, indices = nbrs.kneighbors(adata.obsm['X_pca'])
   adata.uns['indices'] = indices
   print("Nearest neighbors computed!")

   # Visualize results
   print("\nGenerating UMAP visualization...")
   sc.tl.umap(adata)
   fig, ax = plt.subplots(1, 1, figsize=(8, 6))
   sc.pl.umap(adata, color='clusters', ax=ax, show=False)
   plt.title('Pancreas Dataset - Cell Clusters')
   plt.tight_layout()
   plt.show()
   print("Preprocessing tutorial completed successfully!")

Additional Resources
--------------------

- :doc:`../api/lineagevi` - API reference for the LineageVI class
- :doc:`../api/model` - Detailed model documentation
- :doc:`training` - Next step: model training tutorial

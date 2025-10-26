Basic Usage Tutorial
====================

This tutorial will walk you through the basic usage of LineageVI step by step.

Data Preparation
----------------

First, let's load and prepare your single-cell data:

.. code-block:: python

   import scanpy as sc
   import lineagevi as lvi
   import pandas as pd
   import numpy as np
   
   # Load your data
   adata = sc.read("your_data.h5ad")
   
   # Check data structure
   print(f"Data shape: {adata.shape}")
   print(f"Layers: {list(adata.layers.keys())}")
   print(f"Obs columns: {list(adata.obs.columns)}")

Preprocessing
-------------

LineageVI requires specific preprocessing steps:

.. code-block:: python

   # Ensure you have unspliced and spliced counts
   if 'unspliced' not in adata.layers:
       raise ValueError("Missing 'unspliced' layer")
   if 'spliced' not in adata.layers:
       raise ValueError("Missing 'spliced' layer")
   
   # Basic preprocessing
   sc.pp.normalize_total(adata, target_sum=1e4)
   sc.pp.log1p(adata)
   
   # Find highly variable genes
   sc.pp.highly_variable_genes(adata, n_top_genes=2000)
   adata = adata[:, adata.var.highly_variable]
   
   # Compute neighbors for velocity analysis
   sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)

Model Initialization
--------------------

Create a LineageVI instance:

.. code-block:: python

   # Initialize with default parameters
   linvi = lvi.LineageVI(adata)
   
   # Or with custom parameters
   linvi = lvi.LineageVI(
       adata,
       n_hidden=256,  # More hidden units for complex data
       mask_key="I",   # Gene program mask key
       device='cuda'  # Use GPU if available
   )

Training
--------

Train the model using the two-regime approach:

.. code-block:: python

   # Basic training
   history = linvi.fit()
   
   # Custom training parameters
   history = linvi.fit(
       epochs1=100,    # Expression reconstruction epochs
       epochs2=100,    # Velocity prediction epochs
       lr=5e-4,        # Learning rate
       batch_size=512  # Batch size
   )
   
   # Check training progress
   print("Regime 1 losses:", history['regime1_loss'][-5:])
   print("Regime 2 losses:", history['regime2_velocity_loss'][-5:])

Getting Model Outputs
---------------------

After training, get the model predictions:

.. code-block:: python

   # Get basic outputs
   linvi.get_model_outputs()
   
   # With uncertainty estimation
   linvi.get_model_outputs(n_samples=100, return_negative_velo=True)
   
   # Check what was added
   print("Added to obsm:", list(adata.obsm.keys()))
   print("Added to layers:", list(adata.layers.keys()))

Analyzing Gene Programs
-----------------------

Explore the learned gene programs:

.. code-block:: python

   # Gene program enrichment analysis
   linvi.latent_enrich(adata, groups="cell_type")
   
   # Check enrichment results
   print(adata.uns['latent_enrichment'].head())

Basic Visualization
-------------------

Create some basic plots:

.. code-block:: python

   # UMAP with cell types
   sc.tl.umap(adata)
   sc.pl.umap(adata, color="cell_type", title="Cell Types")
   
   # UMAP with velocities
   sc.pl.umap(adata, color="velocity", title="RNA Velocity")
   
   # Gene program phase plane
   lvi.pl.plot_gp_phase_planes(adata, program_pairs=[("GP1", "GP2")])

Velocity Mapping
----------------

Map velocities between gene and gene program spaces:

.. code-block:: python

   # Map from gene programs to genes
   linvi.map_velocities(adata, direction="gp_to_gene")
   
   # Map from genes to gene programs
   gp_adata = linvi.map_velocities(
       adata, 
       direction="gene_to_gp", 
       return_gp_adata=True
   )

Next Steps
----------

Now that you have the basics, explore:

* :doc:`gene_programs` - Understanding gene program interpretation
* :doc:`velocity_analysis` - Advanced velocity analysis
* :doc:`uncertainty_analysis` - Uncertainty quantification

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Training doesn't converge**
   - Try reducing learning rate
   - Increase number of epochs
   - Check data preprocessing

**Memory issues**
   - Reduce batch size
   - Use CPU instead of GPU
   - Subset your data

**Poor velocity predictions**
   - Ensure proper preprocessing
   - Check neighbor graph quality
   - Try different model parameters

Quick Start Guide
=================

This guide will get you up and running with LineageVI in just a few minutes.

Basic Usage
-----------

1. **Load your data**

.. code-block:: python

   import scanpy as sc
   import lineagevi as lvi
   
   # Load single-cell data
   adata = sc.read("your_data.h5ad")
   
   # Ensure you have unspliced and spliced counts
   print(adata.layers.keys())  # Should include 'unspliced' and 'spliced'

2. **Initialize LineageVI**

.. code-block:: python

   # Create LineageVI instance
   linvi = lvi.LineageVI(adata, n_hidden=128)
   
   # Optional: Set device for GPU acceleration
   # linvi = lvi.LineageVI(adata, device='cuda')

3. **Train the model**

.. code-block:: python

   # Train with default parameters
   history = linvi.fit(epochs1=50, epochs2=50)
   
   # Or with custom parameters
   history = linvi.fit(
       epochs1=100, 
       epochs2=100, 
       lr=5e-4, 
       batch_size=512
   )

4. **Get model outputs**

.. code-block:: python

   # Get velocities and latent representations
   linvi.get_model_outputs()
   
   # Check what was added to adata
   print(adata.obsm.keys())  # Should include 'z', 'velocity', etc.
   print(adata.layers.keys())  # Should include velocity layers

5. **Analyze gene programs**

.. code-block:: python

   # Enrichment analysis
   linvi.latent_enrich(adata, groups="cell_type")
   
   # Plot gene program phase planes
   lvi.pl.plot_gp_phase_planes(adata, program_pairs=[("GP1", "GP2")])

Complete Example
----------------

Here's a complete example using a simulated dataset:

.. code-block:: python

   import scanpy as sc
   import lineagevi as lvi
   import numpy as np
   
   # Create simulated data
   n_cells, n_genes = 1000, 2000
   adata = sc.AnnData(
       X=np.random.poisson(5, (n_cells, n_genes)),
       obs=pd.DataFrame({'cell_type': np.random.choice(['A', 'B', 'C'], n_cells)})
   )
   
   # Add unspliced and spliced layers
   adata.layers['unspliced'] = np.random.poisson(3, (n_cells, n_genes))
   adata.layers['spliced'] = np.random.poisson(2, (n_cells, n_genes))
   
   # Preprocess
   sc.pp.normalize_total(adata, target_sum=1e4)
   sc.pp.log1p(adata)
   sc.pp.highly_variable_genes(adata, n_top_genes=2000)
   adata = adata[:, adata.var.highly_variable]
   
   # Compute neighbors
   sc.pp.neighbors(adata)
   
   # Initialize and train LineageVI
   linvi = lvi.LineageVI(adata)
   history = linvi.fit(epochs1=30, epochs2=30)
   
   # Get outputs
   linvi.get_model_outputs()
   
   # Analyze
   linvi.latent_enrich(adata, groups="cell_type")
   
   # Plot
   sc.pl.umap(adata, color="cell_type")
   sc.pl.umap(adata, color="velocity")

Next Steps
----------

Now that you have the basics, explore:

* :doc:`notebooks/Tutorial` - Comprehensive tutorial
* :doc:`api/index` - Complete API reference
* :doc:`examples/index` - More examples

Common Parameters
-----------------

**LineageVI Constructor**
   - ``n_hidden``: Number of hidden units (default: 128)
   - ``mask_key``: Key for gene program mask (default: "I")
   - ``device``: Device for computation (default: auto-detect)

**Training (fit method)**
   - ``epochs1``: Epochs for regime 1 (default: 50)
   - ``epochs2``: Epochs for regime 2 (default: 50)
   - ``lr``: Learning rate (default: 1e-3)
   - ``batch_size``: Batch size (default: 1024)

**Model Outputs (get_model_outputs method)**
   - ``n_samples``: Number of samples for uncertainty (default: 1)
   - ``return_negative_velo``: Negate velocities (default: True)
   - ``save_to_adata``: Save to AnnData object (default: False)

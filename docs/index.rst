LineageVI Documentation
========================

LineageVI is a deep learning-based RNA velocity model that learns gene programs (GPs) and predicts RNA velocity in both gene expression and gene program spaces. It uses a two-regime training approach: first reconstructing expression, then predicting velocity.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   tutorials/index
   api/index

Features
--------

* **Gene Program Inference**: Automatically learns meaningful gene programs from single-cell data
* **Bidirectional Velocity Prediction**: Predicts velocities in both gene expression and gene program spaces
* **Two-Regime Training**: Optimized training strategy for robust model performance
* **Uncertainty Quantification**: Comprehensive uncertainty analysis for velocity predictions
* **Perturbation Analysis**: Study the sensitivity of velocity predictions to expression changes
* **Rich Visualization**: Built-in plotting functions for phase planes and gene program analysis

Quick Start
-----------

.. code-block:: python

   import scanpy as sc
   import lineagevi as lvi
   
   # Load your single-cell data
   adata = sc.read("your_data.h5ad")
   
   # Initialize and train LineageVI
   linvi = lvi.LineageVI(adata)
   history = linvi.fit(epochs1=50, epochs2=50)
   
   # Get model outputs
   linvi.get_model_outputs()
   
   # Analyze gene programs
   linvi.latent_enrich(adata, groups="cell_type")

Installation
------------

.. code-block:: bash

   pip install lineagevi

For detailed installation instructions, see :doc:`installation`.

Citation
--------

If you use LineageVI in your research, please cite:

.. code-block:: bibtex

   @article{lineagevi2024,
     title={LineageVI: Deep learning-based RNA velocity with gene program inference},
     author={LineageVI Team},
     journal={Nature Methods},
     year={2024}
   }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

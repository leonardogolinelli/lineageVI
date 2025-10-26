Examples
========

This section contains practical examples of using LineageVI on real datasets.

.. toctree::
   :maxdepth: 2

   pancreas_analysis
   brain_development
   cancer_progression
   perturbation_analysis

Real Dataset Examples
---------------------

:doc:`pancreas_analysis`
   Complete analysis of pancreatic development using LineageVI.

:doc:`brain_development`
   Study brain development trajectories with gene program analysis.

:doc:`cancer_progression`
   Analyze cancer progression and drug response using velocity analysis.

:doc:`perturbation_analysis`
   Study the effects of gene perturbations on cellular trajectories.

Getting the Data
----------------

Most examples use publicly available datasets that can be downloaded using:

.. code-block:: python

   import scanpy as sc
   
   # Download example datasets
   adata = sc.datasets.pbmc3k()  # PBMC dataset
   # or
   adata = sc.datasets.pancreas()  # Pancreas dataset

Running the Examples
--------------------

Each example is designed to be run as a Jupyter notebook. You can:

1. Download the notebook from our GitHub repository
2. Run it locally with your own data
3. Modify parameters to explore different analyses

Example Structure
-----------------

Each example follows this structure:

1. **Data Loading**: Load and inspect the dataset
2. **Preprocessing**: Prepare data for LineageVI
3. **Model Training**: Train the LineageVI model
4. **Analysis**: Analyze gene programs and velocities
5. **Visualization**: Create publication-ready plots
6. **Interpretation**: Discuss biological insights

Prerequisites
-------------

- LineageVI installed
- Jupyter notebook environment
- Basic knowledge of single-cell analysis
- Familiarity with scanpy and matplotlib

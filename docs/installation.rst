Installation
============

Prerequisites
-------------

LineageVI requires Python 3.8 or higher and the following dependencies:

* PyTorch (>= 1.9.0)
* NumPy (>= 1.19.0)
* Pandas (>= 1.3.0)
* Scanpy (>= 1.8.0)
* AnnData (>= 0.7.0)
* scVelo (>= 0.2.4)
* Matplotlib (>= 3.3.0)
* Seaborn (>= 0.11.0)
* scikit-learn (>= 0.24.0)
* joblib (>= 1.0.0)

Installation Methods
--------------------

PyPI Installation
~~~~~~~~~~~~~~~~~

The easiest way to install LineageVI is via pip:

.. code-block:: bash

   pip install lineagevi

Conda Installation
~~~~~~~~~~~~~~~~~

You can also install LineageVI using conda:

.. code-block:: bash

   conda install -c conda-forge lineagevi

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

To install LineageVI in development mode:

.. code-block:: bash

   git clone https://github.com/your-org/lineagevi.git
   cd lineagevi
   pip install -e .

Installation with GPU Support
-----------------------------

For GPU acceleration, install PyTorch with CUDA support:

.. code-block:: bash

   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # Then install LineageVI
   pip install lineagevi

Verification
------------

To verify your installation, run:

.. code-block:: python

   import lineagevi as lvi
   print(f"LineageVI version: {lvi.__version__}")

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'torch'**
   Make sure PyTorch is properly installed. Try reinstalling with the appropriate CUDA version.

**CUDA out of memory**
   Reduce batch size or use CPU-only mode by setting ``device='cpu'`` in the LineageVI constructor.

**Missing dependencies**
   Install missing packages using pip or conda.

Getting Help
~~~~~~~~~~~~

If you encounter issues:

1. Check the `GitHub Issues <https://github.com/your-org/lineagevi/issues>`_ page
2. Join our `Discord <https://discord.gg/lineagevi>`_ community
3. Email us at support@lineagevi.org

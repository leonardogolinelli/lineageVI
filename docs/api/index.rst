API Reference
=============

This section contains the complete API reference for LineageVI.

Main API
--------

.. toctree::
   :maxdepth: 2

   lineagevi
   model
   plots
   utils

Core Classes
------------

.. autosummary::
   :toctree: generated
   :template: class.rst

   lineagevi.LineageVI
   lineagevi.model.LineageVIModel

Neural Network Modules
---------------------

.. autosummary::
   :toctree: generated
   :template: class.rst

   lineagevi.modules.Encoder
   lineagevi.modules.MaskedLinearDecoder
   lineagevi.modules.VelocityDecoder

Plotting Functions
------------------

.. autosummary::
   :toctree: generated
   :template: function.rst

   lineagevi.plots.plot_phase_plane
   lineagevi.plots.plot_gp_phase_planes
   lineagevi.plots.top_features_table
   lineagevi.plots.plot_abs_bfs

Utility Functions
-----------------

.. autosummary::
   :toctree: generated
   :template: function.rst

   lineagevi.utils.build_gp_adata
   lineagevi.utils.load_model

API Reference
=============

This section contains the complete API reference for LineageVI.

.. toctree::
   :maxdepth: 2

   lineagevi
   model
   plots
   utils

.. autosummary::
   :toctree: generated
   :template: class.rst

   lineagevi.LineageVI
   lineagevi.model.LineageVIModel
   lineagevi.modules.Encoder
   lineagevi.modules.MaskedLinearDecoder
   lineagevi.modules.VelocityDecoder

.. autosummary::
   :toctree: generated
   :template: function.rst

   lineagevi.plots.plot_phase_plane
   lineagevi.plots.plot_gp_phase_planes
   lineagevi.plots.top_features_table
   lineagevi.plots.plot_abs_bfs
   lineagevi.utils.build_gp_adata
   lineagevi.utils.load_model

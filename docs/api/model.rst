LineageVI Model
===============

.. automodule:: lineagevi.model
   :members:
   :undoc-members:
   :show-inheritance:

LineageVIModel Class
--------------------

.. autoclass:: lineagevi.LineageVIModel
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. automethod:: forward
   .. automethod:: reconstruction_loss
   .. automethod:: kl_divergence
   .. automethod:: velocity_loss
   .. automethod:: _get_model_outputs
   .. automethod:: latent_enrich
   .. automethod:: get_directional_uncertainty
   .. automethod:: compute_extrinsic_uncertainty
   .. automethod:: map_velocities
   .. automethod:: perturb_genes
   .. automethod:: perturb_gps

Neural Network Modules
----------------------

.. autoclass:: lineagevi.Encoder
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: lineagevi.MaskedLinearDecoder
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: lineagevi.VelocityDecoder
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Utility Functions
-----------------

.. autofunction:: lineagevi.seed_everything

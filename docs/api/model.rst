LineageVI Model
===============

.. automodule:: lineagevi.model
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: lineagevi.model.LineageVIModel
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

.. autoclass:: lineagevi.modules.Encoder
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: lineagevi.modules.MaskedLinearDecoder
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: lineagevi.modules.VelocityDecoder
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. autofunction:: lineagevi.utils.seed_everything
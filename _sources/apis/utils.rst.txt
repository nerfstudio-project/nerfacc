Utils
===================================

Below are the basic functions that supports sampling and rendering. 

.. currentmodule:: nerfacc

.. autosummary::
   :nosignatures:
   :toctree: generated/

   inclusive_prod
   exclusive_prod
   inclusive_sum
   exclusive_sum

   pack_info

   render_visibility_from_alpha
   render_visibility_from_density
   render_weight_from_alpha
   render_weight_from_density
   render_transmittance_from_alpha
   render_transmittance_from_density
   accumulate_along_rays

   importance_sampling
   searchsorted

   ray_aabb_intersect
   traverse_grids
   
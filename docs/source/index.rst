NerfAcc Documentation
===================================

NerfAcc is a PyTorch Nerf acceleration toolbox for both training and inference. It focus on
efficient volumetric rendering of radiance fields, which is universal and plug-and-play for most of the NeRFs.

Using NerfAcc, 

- The `vanilla Nerf`_ model with 8-layer MLPs can be trained to *better quality* (+~0.5 PNSR) \
  in *1 hour* rather than *1~2 days* as in the paper.
- The `Instant-NGP Nerf`_ model can be trained to *equal quality* in *4.5 minutes*, \
  comparing to the official pure-CUDA implementation.
- The `D-Nerf`_ model for *dynamic* objects can also be trained in *1 hour* \
  rather than *2 days* as in the paper, and with *better quality* (+~2.5 PSNR).
- Both *bounded* and *unbounded* scenes are supported.

**And it is pure Python interface with flexible APIs!**

| Github: https://github.com/KAIR-BAIR/nerfacc
| Paper: https://arxiv.org/pdf/2210.04847.pdf
| Authors: `Ruilong Li`_, `Matthew Tancik`_, `Angjoo Kanazawa`_

.. note::

   This repo is focusing on the single scene situation. Generalizable Nerfs across
   multiple scenes is currently out of the scope of this repo. But you may still find
   some useful tricks in this repo. :)


Installation:
-------------

.. code-block:: console

   $ pip install nerfacc

Usage:
-------------

The idea of NerfAcc is to perform efficient ray marching and volumetric rendering. 
So NerfAcc can work with any user-defined radiance field. To plug the NerfAcc rendering
pipeline into your code and enjoy the acceleration, you only need to define two functions 
with your radience field.

- `sigma_fn`: Compute density at each sample. It will be used by :func:`nerfacc.ray_marching` to skip the empty and occluded space during ray marching, which is where the major speedup comes from. 
- `rgb_sigma_fn`: Compute color and density at each sample. It will be used by :func:`nerfacc.rendering` to conduct differentiable volumetric rendering. This function will receive gradients to update your network.

An simple example is like this:

.. code-block:: python

   import torch
   from torch import Tensor
   import nerfacc 

   radiance_field = ...  # network: a NeRF model
   rays_o: Tensor = ...  # ray origins. (n_rays, 3)
   rays_d: Tensor = ...  # ray normalized directions. (n_rays, 3)
   optimizer = ...  # optimizer

   def sigma_fn(
      t_starts: Tensor, t_ends:Tensor, ray_indices: Tensor
   ) -> Tensor:
      """ Query density values from a user-defined radiance field.
      :params t_starts: Start of the sample interval along the ray. (n_samples, 1).
      :params t_ends: End of the sample interval along the ray. (n_samples, 1).
      :params ray_indices: Ray indices that each sample belongs to. (n_samples,).
      :returns The post-activation density values. (n_samples, 1).
      """
      t_origins = rays_o[ray_indices]  # (n_samples, 3)
      t_dirs = rays_d[ray_indices]  # (n_samples, 3)
      positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
      sigmas = radiance_field.query_density(positions) 
      return sigmas  # (n_samples, 1)

   def rgb_sigma_fn(
      t_starts: Tensor, t_ends: Tensor, ray_indices: Tensor
   ) -> Tuple[Tensor, Tensor]:
      """ Query rgb and density values from a user-defined radiance field.
      :params t_starts: Start of the sample interval along the ray. (n_samples, 1).
      :params t_ends: End of the sample interval along the ray. (n_samples, 1).
      :params ray_indices: Ray indices that each sample belongs to. (n_samples,).
      :returns The post-activation rgb and density values. 
         (n_samples, 3), (n_samples, 1).
      """
      t_origins = rays_o[ray_indices]  # (n_samples, 3)
      t_dirs = rays_d[ray_indices]  # (n_samples, 3)
      positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
      rgbs, sigmas = radiance_field(positions, condition=t_dirs)  
      return rgbs, sigmas  # (n_samples, 3), (n_samples, 1)

   # Efficient Raymarching: Skip empty and occluded space, pack samples from all rays.
   # ray_indices: (n_samples,). t_starts: (n_samples, 1). t_ends: (n_samples, 1).
   with torch.no_grad():
      ray_indices, t_starts, t_ends = nerfacc.ray_marching(
         rays_o, rays_d, sigma_fn=sigma_fn, near_plane=0.2, far_plane=1.0, 
         early_stop_eps=1e-4, alpha_thre=1e-2, 
      )

   # Differentiable Volumetric Rendering.
   # colors: (n_rays, 3). opaicity: (n_rays, 1). depth: (n_rays, 1).
   color, opacity, depth = nerfacc.rendering(
      t_starts, t_ends, ray_indices, n_rays=rays_o.shape[0], rgb_sigma_fn=rgb_sigma_fn
   )

   # Optimize: Both the network and rays will receive gradients
   optimizer.zero_grad()
   loss = F.mse_loss(color, color_gt)
   loss.backward()
   optimizer.step()


Links:
-------------

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Python API

   apis/*

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Example Usages

   examples/*

.. toctree::
   :maxdepth: 1
   :caption: Projects

   nerfstudio <https://docs.nerf.studio/>


.. _`vanilla Nerf`: https://arxiv.org/abs/2003.08934
.. _`Instant-NGP Nerf`: https://arxiv.org/abs/2201.05989
.. _`D-Nerf`: https://arxiv.org/abs/2011.13961
.. _`MipNerf360`: https://arxiv.org/abs/2111.12077
.. _`pixel-Nerf`: https://arxiv.org/abs/2012.02190
.. _`Nerf++`: https://arxiv.org/abs/2010.07492

.. _`Ruilong Li`: https://www.liruilong.cn/
.. _`Matthew Tancik`: https://www.matthewtancik.com/
.. _`Angjoo Kanazawa`: https://people.eecs.berkeley.edu/~kanazawa/
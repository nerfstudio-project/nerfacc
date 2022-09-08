Usage
=====

.. _installation:

Installation
------------

To use nerfacc, first install it using pip:

.. code-block:: console

   (.venv) $ pip install git+https://github.com/liruilong940607/nerfacc

Example of use
----------------

.. code-block:: python

    from typing import Callable, List, Union
    from torch import Tensor
    import torch
    import torch.nn.function as F

    from nerfacc import OccupancyField, volumetric_rendering

    # setup the scene bounding box.
    scene_aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5]).cuda()

    # setup the scene radiance field. Assume you have a NeRF model and 
    # it has following functions:
    # - query_density(): {x} -> {density} 
    # - forward(): {x, dirs} -> {rgb, density}
    radiance_field = ...

    # setup some rendering settings
    render_n_samples = 1024
    render_bkgd = torch.ones(3).cuda()
    render_step_size = (
        (scene_aabb[3:] - scene_aabb[:3]).max() * math.sqrt(3) / render_n_samples
    )
    
    # setup occupancy field with eval function
    def occ_eval_fn(x: torch.Tensor) -> torch.Tensor:
        """Evaluate occupancy given positions.

        Args:
            x: positions with shape (N, 3).
        Returns:
            occupancy values with shape (N, 1). 
        """
        density_after_activation = radiance_field.query_density(x)
        occupancy = density_after_activation * render_step_size
        return occupancy
    occ_field = OccupancyField(occ_eval_fn=occ_eval_fn, aabb=aabb, resolution=128)

    # training
    for step in range(10_000):
        # generate rays from data and the gt pixel color
        rays = ...
        pixels = ...

        # update occupancy grid
        occ_field.every_n_step(step)        

        # rendering
        (
            accumulated_color,
            accumulated_depth,
            accumulated_weight,
            _,
        ) = volumetric_rendering(
            query_fn=radiance_field.forward,  # {x, dir} -> {rgb, density}
            rays_o=rays.origins,
            rays_d=rays.viewdirs,
            scene_aabb=aabb,
            scene_occ_binary=occupancy_field.occ_grid_binary,
            scene_resolution=occupancy_field.resolution,
            render_bkgd=render_bkgd,
            render_n_samples=render_n_samples,
            # other kwargs for `query_fn`
            ...,
        )

        # compute loss
        loss = F.mse_loss(accumulated_color, pixels)

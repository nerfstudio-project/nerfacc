
# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).
 
## [0.5.0] - 2023-04-04
 
This is a major upgrade in which 90% of the code has been rewritten. In this version
we achieves:

![Teaser](/docs/source/_static/images/teaser.jpg?raw=true)
 
Links:
- Documentation: https://www.nerfacc.com/en/v0.5.0/
- ArXiv Report: Coming Soon.

Methodologies:
- Upgrade Occupancy Grid to support multiple levels.
- Support Proposal Network from Mip-NeRF 360.
- Update examples on unbounded scenes to use Multi-level Occupancy Grid or Proposal Network.
- Contraction for Occupancy Grid is no longer supported due to it's inefficiency for ray traversal.

API Changes:
- [Changed] `OccupancyGrid()` -> `OccGridEstimator()`. 
    - [Added] Argument `levels=1` for multi-level support.
    - [Added] Function `self.sampling()` that does basically the same thing with the old `nerfacc.ray_marching`.
    - [Renamed] Function `self.every_n_step()` -> `self.update_every_n_steps()`
- [Added] `PropNetEstimator()`. With functions `self.sampling()`, `self.update_every_n_steps()`
and `self.compute_loss()`.
- [Removed] `ray_marching()`. Ray marching is now implemented through calling `sampling()` of
the `OccGridEstimator()` / `PropNetEstimator()`.
- [Changed] `ray_aabb_intersect()` now supports multiple aabb, and supports new argument `near_plane`, `far_plane`, `miss_value`.
- [Changed] `render_*_from_*()`. The input shape changes from `(all_samples, 1)` to `(all_samples)`. And the function will returns all intermediate results so it might be a tuple.
- [Changed] `rendering()`. The input shape changes from `(all_samples, 1)` to `(all_samples)`, including the shape assumption for the `rgb_sigma_fn` and `rgb_alpha_fn`. Be aware of this shape change.
- [Changed] `accumulate_along_rays()`. The shape of the `weights` in the inputs should be `(all_samples)` now.
- [Removed] `unpack_info()`, `pack_data()`, `unpack_data()` are temporally removed due to in-compatibility
with the new backend implementation. Will add them back later.
- [Added] Some basic functions that support both batched tensor and flattened tensor: `inclusive_prod()`, `inclusive_sum()`, `exclusive_prod()`, `exclusive_sum()`, `importance_sampling()`, `searchsorted()`.

Examples & Benchmarks: 
- More benchmarks and examples. See folder `examples/` and `benchmarks/`.
 
## [0.3.5] - 2023-02-23

A stable version that achieves:
- The vanilla Nerf model with 8-layer MLPs can be trained to better quality (+0.5 PNSR) in 1 hour rather than days as in the paper.
- The Instant-NGP Nerf model can be trained to equal quality in 4.5 minutes, comparing to the official pure-CUDA implementation.
- The D-Nerf model for dynamic objects can also be trained in 1 hour rather than 2 days as in the paper, and with better quality (+~2.5 PSNR).
- Both bounded and unbounded scenes are supported.

Links:
- Documentation: https://www.nerfacc.com/en/v0.3.5/
- ArXiv Report: https://arxiv.org/abs/2210.04847v2/

Methodologies:
- Single resolution `nerfacc.OccupancyGrid` for synthetic scenes.
- Contraction methods `nerfacc.ContractionType` for unbounded scenes.

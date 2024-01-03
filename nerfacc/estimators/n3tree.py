import math
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor

from ..grid import _enlarge_aabb
from ..volrend import (
    render_visibility_from_alpha,
    render_visibility_from_density,
)
from .base import AbstractEstimator

try:
    import svox
except ImportError:
    raise ImportError(
        "Please install this forked version of svox: "
        "pip install git+https://github.com/liruilong940607/svox.git"
    )


class N3TreeEstimator(AbstractEstimator):
    """Use N3Tree to implement Occupancy Grid.

    This allows more flexible topologies than the cascaded grid. However, it is
    slower to create samples from the tree than the cascaded grid. By default,
    it has the same topology as the cascaded grid but `self.tree` can be
    modified to have different topologies.
    """

    def __init__(
        self,
        roi_aabb: Union[List[int], Tensor],
        resolution: Union[int, List[int], Tensor] = 128,
        levels: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()

        if "contraction_type" in kwargs:
            raise ValueError(
                "`contraction_type` is not supported anymore for nerfacc >= 0.4.0."
            )

        # check the resolution is legal
        assert isinstance(
            resolution, int
        ), "N3Tree only supports uniform resolution!"

        # check the roi_aabb is legal
        if isinstance(roi_aabb, (list, tuple)):
            roi_aabb = torch.tensor(roi_aabb, dtype=torch.float32)
        assert isinstance(roi_aabb, Tensor), f"Invalid type: {roi_aabb}!"
        assert roi_aabb.shape[0] == 6, f"Invalid shape: {roi_aabb}!"
        roi_aabb = roi_aabb.cpu()

        # to be compatible with the OccupancyGrid
        aabbs = torch.stack(
            [_enlarge_aabb(roi_aabb, 2**i) for i in range(levels)], dim=0
        )
        self.register_buffer("aabbs", aabbs)  # [n_aabbs, 6]

        center = (roi_aabb[:3] + roi_aabb[3:]) / 2.0
        radius = (roi_aabb[3:] - roi_aabb[:3]) / 2.0 * 2 ** (levels - 1)

        tree_depth = int(math.log2(resolution)) - 1
        self.tree = svox.N3Tree(
            N=2,
            data_dim=1,
            init_refine=tree_depth,
            depth_limit=20,
            radius=radius.tolist(),
            center=center.tolist(),
        )
        _aabbs = [_enlarge_aabb(roi_aabb, 2**i) for i in range(levels - 1)]
        for aabb in _aabbs[::-1]:
            leaf_c = self.tree.corners + self.tree.lengths * 0.5
            sel = ((leaf_c > aabb[:3]) & (leaf_c < aabb[3:])).all(dim=-1)
            self.tree[sel].refine()
        # print("tree size", len(self.tree), "at resolution", resolution)

        self.thresh = 0.0

    @torch.no_grad()
    def sampling(
        self,
        # rays
        rays_o: Tensor,  # [n_rays, 3]
        rays_d: Tensor,  # [n_rays, 3]
        # sigma/alpha function for skipping invisible space
        sigma_fn: Optional[Callable] = None,
        alpha_fn: Optional[Callable] = None,
        near_plane: float = 0.0,
        far_plane: float = 1e10,
        t_min: Optional[Tensor] = None,  # [n_rays]
        t_max: Optional[Tensor] = None,  # [n_rays]
        # rendering options
        render_step_size: float = 1e-3,
        early_stop_eps: float = 1e-4,
        alpha_thre: float = 0.0,
        stratified: bool = False,
        cone_angle: float = 0.0,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Sampling with spatial skipping.

        Note:
            This function is not differentiable to any inputs.

        Args:
            rays_o: Ray origins of shape (n_rays, 3).
            rays_d: Normalized ray directions of shape (n_rays, 3).
            sigma_fn: Optional. If provided, the marching will skip the invisible space
                by evaluating the density along the ray with `sigma_fn`. It should be a
                function that takes in samples {t_starts (N,), t_ends (N,),
                ray indices (N,)} and returns the post-activation density values (N,).
                You should only provide either `sigma_fn` or `alpha_fn`.
            alpha_fn: Optional. If provided, the marching will skip the invisible space
                by evaluating the density along the ray with `alpha_fn`. It should be a
                function that takes in samples {t_starts (N,), t_ends (N,),
                ray indices (N,)} and returns the post-activation opacity values (N,).
                You should only provide either `sigma_fn` or `alpha_fn`.
            near_plane: Optional. Near plane distance. Default: 0.0.
            far_plane: Optional. Far plane distance. Default: 1e10.
            t_min: Optional. Per-ray minimum distance. Tensor with shape (n_rays).
                If profided, the marching will start from maximum of t_min and near_plane.
            t_max: Optional. Per-ray maximum distance. Tensor with shape (n_rays).
                If profided, the marching will stop by minimum of t_max and far_plane.
            render_step_size: Step size for marching. Default: 1e-3.
            early_stop_eps: Early stop threshold for skipping invisible space. Default: 1e-4.
            alpha_thre: Alpha threshold for skipping empty space. Default: 0.0.
            stratified: Whether to use stratified sampling. Default: False.
            cone_angle: Cone angle for linearly-increased step size. 0. means
                constant step size. Default: 0.0.

        Returns:
            A tuple of {LongTensor, Tensor, Tensor}:

            - **ray_indices**: Ray index of each sample. IntTensor with shape (n_samples).
            - **t_starts**: Per-sample start distance. Tensor with shape (n_samples,).
            - **t_ends**: Per-sample end distance. Tensor with shape (n_samples,).

        Examples:

        .. code-block:: python

            >>> ray_indices, t_starts, t_ends = grid.sampling(
            >>>     rays_o, rays_d, render_step_size=1e-3)
            >>> t_mid = (t_starts + t_ends) / 2.0
            >>> sample_locs = rays_o[ray_indices] + t_mid * rays_d[ray_indices]

        """

        assert (
            t_min is None and t_max is None
        ), "Do not supported per-ray min max. Please use near_plane and far_plane instead."
        if stratified:
            near_plane += torch.rand(()).item() * render_step_size

        t_starts, t_ends, packed_info, ray_indices = svox.volume_sample(
            self.tree,
            thresh=self.thresh,
            rays=svox.Rays(
                rays_o.contiguous(), rays_d.contiguous(), rays_d.contiguous()
            ),
            step_size=render_step_size,
            cone_angle=cone_angle,
            near_plane=near_plane,
            far_plane=far_plane,
        )
        packed_info = packed_info.long()
        ray_indices = ray_indices.long()

        # skip invisible space
        if (alpha_thre > 0.0 or early_stop_eps > 0.0) and (
            sigma_fn is not None or alpha_fn is not None
        ):
            alpha_thre = min(alpha_thre, self.thresh)

            # Compute visibility of the samples, and filter out invisible samples
            if sigma_fn is not None:
                if t_starts.shape[0] != 0:
                    sigmas = sigma_fn(t_starts, t_ends, ray_indices)
                else:
                    sigmas = torch.empty((0,), device=t_starts.device)
                assert (
                    sigmas.shape == t_starts.shape
                ), "sigmas must have shape of (N,)! Got {}".format(sigmas.shape)
                masks = render_visibility_from_density(
                    t_starts=t_starts,
                    t_ends=t_ends,
                    sigmas=sigmas,
                    ray_indices=ray_indices,
                    n_rays=len(rays_o),
                    early_stop_eps=early_stop_eps,
                    alpha_thre=alpha_thre,
                )
            elif alpha_fn is not None:
                if t_starts.shape[0] != 0:
                    alphas = alpha_fn(t_starts, t_ends, ray_indices)
                else:
                    alphas = torch.empty((0,), device=t_starts.device)
                assert (
                    alphas.shape == t_starts.shape
                ), "alphas must have shape of (N,)! Got {}".format(alphas.shape)
                masks = render_visibility_from_alpha(
                    alphas=alphas,
                    ray_indices=ray_indices,
                    n_rays=len(rays_o),
                    early_stop_eps=early_stop_eps,
                    alpha_thre=alpha_thre,
                )
            ray_indices, t_starts, t_ends = (
                ray_indices[masks],
                t_starts[masks],
                t_ends[masks],
            )
        return ray_indices, t_starts, t_ends

    @torch.no_grad()
    def update_every_n_steps(
        self,
        step: int,
        occ_eval_fn: Callable,
        occ_thre: float = 1e-2,
        ema_decay: float = 0.95,
        warmup_steps: int = 256,
        n: int = 16,
    ) -> None:
        """Update the estimator every n steps during training.

        Args:
            step: Current training step.
            occ_eval_fn: A function that takes in sample locations :math:`(N, 3)` and
                returns the occupancy values :math:`(N, 1)` at those locations.
            occ_thre: Threshold used to binarize the occupancy grid. Default: 1e-2.
            ema_decay: The decay rate for EMA updates. Default: 0.95.
            warmup_steps: Sample all cells during the warmup stage. After the warmup
                stage we change the sampling strategy to 1/4 uniformly sampled cells
                together with 1/4 occupied cells. Default: 256.
            n: Update the grid every n steps. Default: 16.
        """
        if not self.training:
            raise RuntimeError(
                "You should only call this function only during training. "
                "Please call _update() directly if you want to update the "
                "field during inference."
            )
        if step % n == 0 and self.training:
            self._update(
                step=step,
                occ_eval_fn=occ_eval_fn,
                occ_thre=occ_thre,
                ema_decay=ema_decay,
                warmup_steps=warmup_steps,
            )

    @torch.no_grad()
    def _sample_uniform_and_occupied_cells(self, n: int) -> List[Tensor]:
        """Samples both n uniform and occupied cells."""
        uniform_indices = torch.randint(
            len(self.tree), (n,), device=self.device
        )
        occupied_indices = torch.nonzero(self.tree[:].values >= self.thresh)[
            :, 0
        ]
        if n < len(occupied_indices):
            selector = torch.randint(
                len(occupied_indices), (n,), device=self.device
            )
            occupied_indices = occupied_indices[selector]
        indices = torch.cat([uniform_indices, occupied_indices], dim=0)
        return indices

    @torch.no_grad()
    def _update(
        self,
        step: int,
        occ_eval_fn: Callable,
        occ_thre: float = 0.01,
        ema_decay: float = 0.95,
        warmup_steps: int = 256,
    ) -> None:
        """Update the occ field in the EMA way."""
        if step < warmup_steps:
            x = self.tree.sample(1).squeeze(1)
            occ = occ_eval_fn(x).squeeze(-1)
            sel = (*self.tree._all_leaves().T,)
            self.tree.data.data[sel] = torch.maximum(
                self.tree.data.data[sel] * ema_decay, occ[:, None]
            )
        else:
            N = len(self.tree) // 4
            indices = self._sample_uniform_and_occupied_cells(N)
            x = self.tree[indices].sample(1).squeeze(1)
            occ = occ_eval_fn(x).squeeze(-1)
            self.tree[indices] = torch.maximum(
                self.tree[indices].values * ema_decay, occ[:, None]
            )
        self.thresh = min(occ_thre, self.tree[:].values.mean().item())


if __name__ == "__main__":
    roi_aabb = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
    resolution = 128
    levels = 4
    estimator = N3TreeEstimator(roi_aabb, resolution, levels)

    def occ_eval_fn(x):
        return torch.rand(len(x), 1)

    estimator.update_every_n_steps(0, occ_eval_fn, occ_thre=0.5)

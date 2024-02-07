from typing import Any, Callable, List, Mapping, Optional, Tuple, Union

fVDB_ENABLED = True
try:
    from fvdb import GridBatch, sparse_grid_from_ijk
except ImportError:
    fVDB_ENABLED = False
    GridBatch = object
    sparse_grid_from_ijk = callable
import torch
from torch import Tensor

from nerfacc.estimators.base import AbstractEstimator
from nerfacc.volrend import (
    render_visibility_from_alpha,
    render_visibility_from_density,
)


@torch.no_grad()
def traverse_vdbs(
    # rays
    rays_o: Tensor,  # [n_rays, 3]
    rays_d: Tensor,  # [n_rays, 3]
    # grids
    grids: GridBatch,
    # options
    near_planes: Optional[Tensor] = None,  # [n_rays]
    far_planes: Optional[Tensor] = None,  # [n_rays]
    step_size: Optional[float] = 1e-3,
    cone_angle: Optional[float] = 0.0,
):
    """Traverse the fVDB grids."""
    assert fVDB_ENABLED, "Please install fVDB to use this function."
    assert len(grids) == 1, "Only support one grid for now."

    if near_planes is None:
        near_planes = torch.zeros_like(rays_o[:, 0])
    if far_planes is None:
        far_planes = torch.full_like(rays_o[:, 0], float("inf"))

    _, indices, intervals = grids.uniform_ray_samples(
        rays_o,
        rays_d,
        near_planes,
        far_planes,
        step_size,
        cone_angle,
        # Use the midpoint of the sample intervals to determine occupancy.
        include_end_segments=False,
    )
    t_starts, t_ends = torch.unbind(intervals.jdata, dim=-1)
    ray_indices = indices.jdata.long()

    # TODO(ruilongli): In fvdb, we would like to restrain the endpoints of the sample
    # intervals to be within the grid boundaries.
    return t_starts, t_ends, ray_indices


class VDBEstimator(AbstractEstimator):
    """Occupancy Estimator Using A VDB."""

    def __init__(self, init_grid: GridBatch, device="cuda:0") -> None:
        super().__init__()
        assert fVDB_ENABLED, "Please install fVDB to use this class."
        assert len(init_grid) == 1, "Only support one grid for now."

        # Create a mutable grid from the initial grid.
        self.grid = sparse_grid_from_ijk(
            init_grid.ijk,
            voxel_sizes=init_grid.voxel_sizes,
            origins=init_grid.origins,
            mutable=True,
        ).to(device)

        # The buffer for float occupancy values
        self.occs = torch.nn.Parameter(
            torch.zeros([self.grid.total_voxels], device=device),
            requires_grad=False,
        )

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["grid"] = self.grid
        state_dict["occs"] = self.occs.state_dict()
        return state_dict

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True
    ):
        init_grid = state_dict["grid"]
        self.grid = sparse_grid_from_ijk(
            init_grid.ijk,
            voxel_sizes=init_grid.voxel_sizes,
            origins=init_grid.origins,
            mutable=True,
        )
        remaining_state_dict = {
            k: v for k, v in state_dict.items() if k not in ["grid", "occs"]
        }
        super().load_state_dict(remaining_state_dict, strict=strict)

    def to(self, device: Union[str, torch.device]):
        self.grid = self.grid.to(device)
        self.occs = self.occs.to(device)
        super().to(device)
        return self

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
                If provided, the marching will start from maximum of t_min and near_plane.
            t_max: Optional. Per-ray maximum distance. Tensor with shape (n_rays).
                If provided, the marching will stop by minimum of t_max and far_plane.
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

        near_planes = torch.full_like(rays_o[..., 0], fill_value=near_plane)
        far_planes = torch.full_like(rays_o[..., 0], fill_value=far_plane)

        if t_min is not None:
            near_planes = torch.clamp(near_planes, min=t_min)
        if t_max is not None:
            far_planes = torch.clamp(far_planes, max=t_max)

        if stratified:
            near_planes += torch.rand_like(near_planes) * render_step_size
        t_starts, t_ends, ray_indices = traverse_vdbs(
            rays_o,
            rays_d,
            self.grid,
            near_planes=near_planes,
            far_planes=far_planes,
            step_size=render_step_size,
            cone_angle=cone_angle,
        )

        # skip invisible space
        if (alpha_thre > 0.0 or early_stop_eps > 0.0) and (
            sigma_fn is not None or alpha_fn is not None
        ):
            alpha_thre = min(alpha_thre, self.occs.mean().item())
            n_rays = rays_o.shape[0]

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
                    n_rays=n_rays,
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
                    n_rays=n_rays,
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
    def _get_all_cells(self) -> List[Tensor]:
        """Returns all cells of the grid."""
        return self.grid.ijk.jdata

    @torch.no_grad()
    def _sample_uniform_and_occupied_cells(self) -> List[Tensor]:
        """Samples both n uniform and occupied cells."""
        n = self.grid.total_voxels // 4
        uniform_selector = torch.randint(
            0, self.grid.total_voxels, (n,), device=self.device
        )
        uniform_ijks = self.grid.ijk.jdata[uniform_selector]

        occupied_ijks = self.grid.ijk_enabled.jdata
        if n < len(occupied_ijks):
            occupied_selector = torch.randint(
                0, len(occupied_ijks), (n,), device=self.device
            )
            occupied_ijks = occupied_ijks[occupied_selector]

        ijks = torch.cat([uniform_ijks, occupied_ijks], dim=0)
        return ijks

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
        # sample cells
        if step < warmup_steps:
            ijks = self._get_all_cells()
        else:
            ijks = self._sample_uniform_and_occupied_cells()

        # update the occ buffer
        grid_coords = ijks - 0.5 + torch.rand_like(ijks, dtype=torch.float32)
        x = self.grid.grid_to_world(grid_coords).jdata
        occ = occ_eval_fn(x).squeeze(-1)
        index = self.grid.ijk_to_index(ijks).jdata
        self.occs[index] = torch.maximum(self.occs[index] * ema_decay, occ)

        # update the grid
        thre = torch.clamp(self.occs.mean(), max=occ_thre)
        active = self.occs[index] >= thre
        _ijks = ijks[active]
        if len(_ijks) > 0:
            self.grid.enable_ijk(_ijks)
        _ijks = ijks[~active]
        if len(_ijks) > 0:
            self.grid.disable_ijk(_ijks)

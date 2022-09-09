import math
import time

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from datasets.nerf_synthetic import SubjectLoader
from datasets.utils import namedtuple_map
from radiance_fields.ngp import NGPradianceField

from nerfacc import OccupancyField, volumetric_rendering


def render_image(radiance_field, rays, render_bkgd):
    """Render the pixels of an image.

    Args:
      radiance_field: the radiance field of nerf.
      rays: a `Rays` namedtuple, the rays to be rendered.

    Returns:
      rgb: torch.tensor, rendered color image.
      depth: torch.tensor, rendered depth image.
      acc: torch.tensor, rendered accumulated weights per pixel.
    """
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays)
    else:
        num_rays, _ = rays_shape
    results = []
    chunk = torch.iinfo(torch.int32).max if radiance_field.training else 81920
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        chunk_color, chunk_depth, chunk_weight, alive_ray_mask, = volumetric_rendering(
            query_fn=radiance_field.forward,  # {x, dir} -> {rgb, density}
            rays_o=chunk_rays.origins,
            rays_d=chunk_rays.viewdirs,
            scene_aabb=occ_field.aabb,
            scene_occ_binary=occ_field.occ_grid_binary,
            scene_resolution=occ_field.resolution,
            render_bkgd=render_bkgd,
            render_n_samples=render_n_samples,
        )
        results.append([chunk_color, chunk_depth, chunk_weight, alive_ray_mask])
    rgb, depth, acc, alive_ray_mask = [torch.cat(r, dim=0) for r in zip(*results)]
    return (
        rgb.view((*rays_shape[:-1], -1)),
        depth.view((*rays_shape[:-1], -1)),
        acc.view((*rays_shape[:-1], -1)),
        alive_ray_mask.view(*rays_shape[:-1]),
    )


if __name__ == "__main__":
    torch.manual_seed(42)

    device = "cuda:0"

    # setup dataset
    train_dataset = SubjectLoader(
        subject_id="lego",
        root_fp="/home/ruilongli/data/nerf_synthetic/",
        split="train",
        num_rays=8192,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=10,
        batch_size=1,
        collate_fn=getattr(train_dataset.__class__, "collate_fn"),
    )
    val_dataset = SubjectLoader(
        subject_id="lego",
        root_fp="/home/ruilongli/data/nerf_synthetic/",
        split="val",
        num_rays=None,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=10,
        batch_size=1,
        collate_fn=getattr(train_dataset.__class__, "collate_fn"),
    )

    # setup the scene bounding box.
    scene_aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5])

    # setup the scene radiance field. Assume you have a NeRF model and
    # it has following functions:
    # - query_density(): {x} -> {density}
    # - forward(): {x, dirs} -> {rgb, density}
    radiance_field = NGPradianceField(aabb=scene_aabb).to(device)

    # setup some rendering settings
    render_n_samples = 1024
    render_step_size = (
        (scene_aabb[3:] - scene_aabb[:3]).max() * math.sqrt(3) / render_n_samples
    )

    optimizer = torch.optim.Adam(radiance_field.parameters(), lr=3e-3, eps=1e-15)

    # setup occupancy field with eval function
    def occ_eval_fn(x: torch.Tensor) -> torch.Tensor:
        """Evaluate occupancy given positions.

        Args:
            x: positions with shape (N, 3).
        Returns:
            occupancy values with shape (N, 1).
        """
        density_after_activation = radiance_field.query_density(x)
        # occupancy = 1.0 - torch.exp(-density_after_activation * render_step_size)
        occupancy = density_after_activation * render_step_size
        return occupancy

    occ_field = OccupancyField(
        occ_eval_fn=occ_eval_fn, aabb=scene_aabb, resolution=128
    ).to(device)

    # training
    step = 0
    tic = time.time()
    for epoch in range(100):
        for data in train_dataloader:
            step += 1
            if step > 30_000:
                print("training stops")
                exit()

            # generate rays from data and the gt pixel color
            rays = namedtuple_map(lambda x: x.to(device), data["rays"])
            pixels = data["pixels"].to(device)
            render_bkgd = data["color_bkgd"].to(device)

            # update occupancy grid
            occ_field.every_n_step(step)

            rgb, depth, acc, alive_ray_mask = render_image(
                radiance_field, rays, render_bkgd
            )

            # compute loss
            loss = F.mse_loss(rgb, pixels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                elapsed_time = time.time() - tic
                print(
                    f"elapsed_time={elapsed_time:.2f}s | {step=} | loss={loss.item(): .5f}"
                )

            if step % 30_000 == 0 and step > 0:
                # evaluation
                radiance_field.eval()
                psnrs = []
                with torch.no_grad():
                    for data in tqdm.tqdm(val_dataloader):
                        # generate rays from data and the gt pixel color
                        rays = namedtuple_map(lambda x: x.to(device), data["rays"])
                        pixels = data["pixels"].to(device)
                        render_bkgd = data["color_bkgd"].to(device)
                        # rendering
                        rgb, depth, acc, alive_ray_mask = render_image(
                            radiance_field, rays, render_bkgd
                        )
                        mse = F.mse_loss(rgb, pixels)
                        psnr = -10.0 * torch.log(mse) / np.log(10.0)
                        psnrs.append(psnr.item())
                psnr_avg = sum(psnrs) / len(psnrs)
                print(f"evaluation: {psnr_avg=}")

# elapsed_time=320.04s | step=30000 | loss= 0.00022
# evaluation: psnr_avg=34.41712421417236 (6.13 it/s)

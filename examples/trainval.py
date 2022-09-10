import math
import time

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from datasets.nerf_synthetic import SubjectLoader, namedtuple_map
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
        split="trainval",
        num_rays=8192,
    )
    # train_dataset.images = train_dataset.images.to(device)
    # train_dataset.camtoworlds = train_dataset.camtoworlds.to(device)
    # train_dataset.K = train_dataset.K.to(device)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=4,
        batch_size=None,
        persistent_workers=True,
        shuffle=True,
    )

    test_dataset = SubjectLoader(
        subject_id="lego",
        root_fp="/home/ruilongli/data/nerf_synthetic/",
        split="test",
        num_rays=None,
    )
    # test_dataset.images = test_dataset.images.to(device)
    # test_dataset.camtoworlds = test_dataset.camtoworlds.to(device)
    # test_dataset.K = test_dataset.K.to(device)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        num_workers=4,
        batch_size=None,
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
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[20000, 30000], gamma=0.1
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
        # those two are similar when density is small.
        # occupancy = 1.0 - torch.exp(-density_after_activation * render_step_size)
        occupancy = density_after_activation * render_step_size
        return occupancy

    occ_field = OccupancyField(
        occ_eval_fn=occ_eval_fn, aabb=scene_aabb, resolution=128
    ).to(device)

    # training
    step = 0
    tic = time.time()
    data_time = 0
    tic_data = time.time()
    for epoch in range(300):
        for data in train_dataloader:
            data_time += time.time() - tic_data
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
            scheduler.step()

            if step % 50 == 0:
                elapsed_time = time.time() - tic
                print(
                    f"elapsed_time={elapsed_time:.2f}s (data={data_time:.2f}s) | {step=} | loss={loss:.5f}"
                )

            if step % 30_000 == 0 and step > 0:
                # evaluation
                radiance_field.eval()
                psnrs = []
                with torch.no_grad():
                    for data in tqdm.tqdm(test_dataloader):
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
            tic_data = time.time()

# "train"
# elapsed_time=298.27s (data=60.08s) | step=30000 | loss=0.00026
# evaluation: psnr_avg=33.305334663391115 (6.42 it/s)

# "train" batch_over_images=True
# elapsed_time=335.21s (data=68.99s) | step=30000 | loss=0.00028
# evaluation: psnr_avg=33.74970862388611 (6.23 it/s)

# "train" batch_over_images=True, schedule
# elapsed_time=296.30s (data=54.38s) | step=30000 | loss=0.00022
# evaluation: psnr_avg=34.3978275680542 (6.22 it/s)

# "trainval"
# elapsed_time=289.94s (data=51.99s) | step=30000 | loss=0.00021
# evaluation: psnr_avg=34.44980221748352 (6.61 it/s)

# "trainval" batch_over_images=True, schedule
# elapsed_time=291.42s (data=52.82s) | step=30000 | loss=0.00020
# evaluation: psnr_avg=35.41630497932434 (6.40 it/s)

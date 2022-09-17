import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from datasets.nerf_synthetic import SubjectLoader, namedtuple_map
from radiance_fields.mlp import VanillaNeRFRadianceField
from radiance_fields.ngp import NGPradianceField

from nerfacc import OccupancyField, volumetric_rendering

TARGET_SAMPLE_BATCH_SIZE = 1 << 16


def render_image(
    radiance_field, rays, render_bkgd, render_step_size, test_chunk_size=81920
):
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

    def sigma_fn(frustum_origins, frustum_dirs, frustum_starts, frustum_ends):
        positions = (
            frustum_origins + frustum_dirs * (frustum_starts + frustum_ends) / 2.0
        )
        return radiance_field.query_density(positions)

    def sigma_rgb_fn(frustum_origins, frustum_dirs, frustum_starts, frustum_ends):
        positions = (
            frustum_origins + frustum_dirs * (frustum_starts + frustum_ends) / 2.0
        )
        return radiance_field(positions, frustum_dirs)

    results = []
    chunk = torch.iinfo(torch.int32).max if radiance_field.training else test_chunk_size
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        chunk_results = volumetric_rendering(
            sigma_fn=sigma_fn,
            sigma_rgb_fn=sigma_rgb_fn,
            rays_o=chunk_rays.origins,
            rays_d=chunk_rays.viewdirs,
            scene_aabb=occ_field.aabb,
            scene_occ_binary=occ_field.occ_grid_binary,
            scene_resolution=occ_field.resolution,
            render_bkgd=render_bkgd,
            render_step_size=render_step_size,
            near_plane=0.0,
            stratified=radiance_field.training,
        )
        results.append(chunk_results)
    colors, opacities, n_marching_samples, n_rendering_samples = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        sum(n_marching_samples),
        sum(n_rendering_samples),
    )


if __name__ == "__main__":
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "method",
        type=str,
        default="ngp",
        choices=["ngp", "vanilla"],
        help="which nerf to use",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="trainval",
        choices=["train", "trainval"],
        help="which train split to use",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="lego",
        choices=[
            "chair",
            "drums",
            "ficus",
            "hotdog",
            "lego",
            "materials",
            "mic",
            "ship",
        ],
        help="which scene to use",
    )
    parser.add_argument(
        "--test_chunk_size",
        type=int,
        default=81920,
    )
    args = parser.parse_args()

    device = "cuda:0"
    scene = args.scene

    # setup the scene bounding box.
    scene_aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5])
    # setup some rendering settings
    render_n_samples = 1024
    render_step_size = (
        (scene_aabb[3:] - scene_aabb[:3]).max() * math.sqrt(3) / render_n_samples
    ).item()

    # setup dataset
    train_dataset = SubjectLoader(
        subject_id=scene,
        root_fp="/home/ruilongli/data/nerf_synthetic/",
        split=args.train_split,
        num_rays=TARGET_SAMPLE_BATCH_SIZE // render_n_samples,
        # color_bkgd_aug="random",
    )

    train_dataset.images = train_dataset.images.to(device)
    train_dataset.camtoworlds = train_dataset.camtoworlds.to(device)
    train_dataset.K = train_dataset.K.to(device)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=0,
        batch_size=None,
        # persistent_workers=True,
        shuffle=True,
    )

    test_dataset = SubjectLoader(
        subject_id=scene,
        root_fp="/home/ruilongli/data/nerf_synthetic/",
        split="test",
        num_rays=None,
    )
    test_dataset.images = test_dataset.images.to(device)
    test_dataset.camtoworlds = test_dataset.camtoworlds.to(device)
    test_dataset.K = test_dataset.K.to(device)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        num_workers=0,
        batch_size=None,
    )

    # setup the scene radiance field. Assume you have a NeRF model and
    # it has following functions:
    # - query_density(): {x} -> {density}
    # - forward(): {x, dirs} -> {rgb, density}
    if args.method == "ngp":
        radiance_field = NGPradianceField(aabb=scene_aabb).to(device)
        optimizer = torch.optim.Adam(radiance_field.parameters(), lr=1e-2, eps=1e-15)
        max_steps = 20000
        occ_field_warmup_steps = 2000
        grad_scaler = torch.cuda.amp.GradScaler(1)

    elif args.method == "vanilla":
        radiance_field = VanillaNeRFRadianceField().to(device)
        optimizer = torch.optim.Adam(radiance_field.parameters(), lr=5e-4)
        max_steps = 40000
        occ_field_warmup_steps = 256
        grad_scaler = torch.cuda.amp.GradScaler(2**10)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[max_steps // 2, max_steps * 3 // 4, max_steps * 9 // 10],
        gamma=0.33,
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

    for epoch in range(10000000):
        for i in range(len(train_dataset)):
            radiance_field.train()
            data = train_dataset[i]
            data_time += time.time() - tic_data

            # generate rays from data and the gt pixel color
            # rays = namedtuple_map(lambda x: x.to(device), data["rays"])
            # pixels = data["pixels"].to(device)
            render_bkgd = data["color_bkgd"]
            rays = data["rays"]
            pixels = data["pixels"]

            # update occupancy grid
            occ_field.every_n_step(step, warmup_steps=occ_field_warmup_steps)

            rgb, acc, counter, compact_counter = render_image(
                radiance_field, rays, render_bkgd, render_step_size
            )
            num_rays = len(pixels)
            num_rays = int(
                num_rays * (TARGET_SAMPLE_BATCH_SIZE / float(compact_counter))
            )
            train_dataset.update_num_rays(num_rays)
            alive_ray_mask = acc.squeeze(-1) > 0

            # compute loss
            loss = F.smooth_l1_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])

            optimizer.zero_grad()
            # do not unscale it because we are using Adam.
            grad_scaler.scale(loss).backward()
            optimizer.step()
            scheduler.step()

            if step % 100 == 0:
                elapsed_time = time.time() - tic
                loss = F.mse_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])
                print(
                    f"elapsed_time={elapsed_time:.2f}s (data={data_time:.2f}s) | {step=} | "
                    f"loss={loss:.5f} | "
                    f"alive_ray_mask={alive_ray_mask.long().sum():d} | "
                    f"counter={counter:d} | compact_counter={compact_counter:d} | num_rays={len(pixels):d} |"
                )

            # if time.time() - tic > 300:
            if step >= max_steps and step % max_steps == 0 and step > 0:
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
                        rgb, acc, _, _ = render_image(
                            radiance_field,
                            rays,
                            render_bkgd,
                            render_step_size,
                            test_chunk_size=args.test_chunk_size,
                        )
                        mse = F.mse_loss(rgb, pixels)
                        psnr = -10.0 * torch.log(mse) / np.log(10.0)
                        psnrs.append(psnr.item())
                psnr_avg = sum(psnrs) / len(psnrs)
                print(f"evaluation: {psnr_avg=}")
                # imageio.imwrite(
                #     "acc_binary_test.png",
                #     ((acc > 0).float().cpu().numpy() * 255).astype(np.uint8),
                # )

                # psnrs = []
                # train_dataset.training = False
                # with torch.no_grad():
                #     for data in tqdm.tqdm(train_dataloader):
                #         # generate rays from data and the gt pixel color
                #         rays = namedtuple_map(lambda x: x.to(device), data["rays"])
                #         pixels = data["pixels"].to(device)
                #         render_bkgd = data["color_bkgd"].to(device)
                #         # rendering
                #         rgb, acc, _, _ = render_image(
                #             radiance_field, rays, render_bkgd, render_step_size
                #         )
                #         mse = F.mse_loss(rgb, pixels)
                #         psnr = -10.0 * torch.log(mse) / np.log(10.0)
                #         psnrs.append(psnr.item())
                # psnr_avg = sum(psnrs) / len(psnrs)
                # print(f"evaluation on train: {psnr_avg=}")
                # imageio.imwrite(
                #     "acc_binary_train.png",
                #     ((acc > 0).float().cpu().numpy() * 255).astype(np.uint8),
                # )
                # imageio.imwrite(
                #     "rgb_train.png",
                #     (rgb.cpu().numpy() * 255).astype(np.uint8),
                # )
                train_dataset.training = True

            if step == max_steps:
                print("training stops")
                exit()
            tic_data = time.time()

            step += 1

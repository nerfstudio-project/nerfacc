import argparse
import math
import os
import random
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from nerfacc import OccupancyField, contract, volumetric_rendering_pipeline

device = "cuda:0"


def _set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def scale_aabb(aabb: torch.Tensor, scale: float) -> torch.Tensor:
    """Half the axis-aligned bounding box.

    Args:
        aabb: the scene axis-aligned bounding box {xmin, ymin, zmin, xmax, ymax, zmax}.

    Returns:
        The halfed axis-aligned bounding box.
    """
    center = (aabb[:3] + aabb[3:]) / 2.0
    half = (aabb[3:] - aabb[:3]) / 2.0
    return torch.cat([center - half * scale, center + half * scale], dim=-1)


def render_image(
    radiance_field,
    rays,
    timestamps,
    render_bkgd,
    render_step_size,
    test_chunk_size=81920,
    contraction: str = None,
    cone_angle: float = 0.0,
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

    def sigma_fn(frustum_starts, frustum_ends, ray_indices):
        ray_indices = ray_indices.long()
        frustum_origins = chunk_rays.origins[ray_indices]
        frustum_dirs = chunk_rays.viewdirs[ray_indices]
        positions = (
            frustum_origins + frustum_dirs * (frustum_starts + frustum_ends) / 2.0
        )
        positions = contract(
            positions, scale_aabb(radiance_field.aabb, 0.5), contraction=contraction
        )
        if timestamps is None:
            return radiance_field.query_density(positions)
        else:
            if radiance_field.training:
                t = timestamps[ray_indices]
            else:
                t = timestamps.expand_as(positions[:, :1])
            return radiance_field.query_density(positions, t)

    def rgb_sigma_fn(frustum_starts, frustum_ends, ray_indices):
        ray_indices = ray_indices.long()
        frustum_origins = chunk_rays.origins[ray_indices]
        frustum_dirs = chunk_rays.viewdirs[ray_indices]
        positions = (
            frustum_origins + frustum_dirs * (frustum_starts + frustum_ends) / 2.0
        )
        positions = contract(
            positions, scale_aabb(radiance_field.aabb, 0.5), contraction=contraction
        )
        if timestamps is None:
            return radiance_field(positions, frustum_dirs)
        else:
            if radiance_field.training:
                t = timestamps[ray_indices]
            else:
                t = timestamps.expand_as(positions[:, :1])
            return radiance_field(positions, t, frustum_dirs)

    results = []
    chunk = torch.iinfo(torch.int32).max if radiance_field.training else test_chunk_size
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        chunk_results = volumetric_rendering_pipeline(
            sigma_fn=sigma_fn,
            rgb_sigma_fn=rgb_sigma_fn,
            rays_o=chunk_rays.origins,
            rays_d=chunk_rays.viewdirs,
            scene_aabb=scale_aabb(
                occ_field.aabb, 0.5 if contraction == "mipnerf360" else 1.0
            ),
            scene_occ_binary=occ_field.occ_grid_binary,
            scene_resolution=occ_field.resolution,
            render_bkgd=render_bkgd,
            render_step_size=render_step_size,
            near_plane=0.2 if contraction == "mipnerf360" else None,
            far_plane=1e4 if contraction == "mipnerf360" else None,
            stratified=radiance_field.training,
            contraction=contraction,
            cone_angle=cone_angle,
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
    _set_random_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "method",
        type=str,
        default="ngp",
        choices=["ngp", "vanilla", "dnerf"],
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
            # nerf synthetic
            "chair",
            "drums",
            "ficus",
            "hotdog",
            "lego",
            "materials",
            "mic",
            "ship",
            # dnerf
            "bouncingballs",
            "hellwarrior",
            "hook",
            "jumpingjacks",
            "lego",
            "mutant",
            "standup",
            "trex",
            # 360
            "garden",
        ],
        help="which scene to use",
    )
    parser.add_argument(
        "--aabb",
        type=lambda s: [float(item) for item in s.split(",")],
        default="-1.5,-1.5,-1.5,1.5,1.5,1.5",
        help="delimited list input",
    )
    parser.add_argument(
        "--test_chunk_size",
        type=int,
        default=81920,
    )
    parser.add_argument(
        "--contraction", type=str, default=None, choices=[None, "mipnerf360"]
    )
    parser.add_argument("--cone_angle", type=float, default=0.0)
    args = parser.parse_args()

    # setup the scene bounding box.
    scene_aabb = torch.tensor(args.aabb, dtype=torch.float32)

    train_dataset_kwargs = {}
    test_dataset_kwargs = {}
    if args.method == "ngp":
        if args.scene == "garden":
            from datasets.nerf_360_v2 import SubjectLoader, namedtuple_map

            data_root_fp = "/home/ruilongli/data/360_v2/"
            target_sample_batch_size = 1 << 20
            train_dataset_kwargs = {"color_bkgd_aug": "random", "factor": 4}
            test_dataset_kwargs = {"factor": 4}
        else:
            from datasets.nerf_synthetic import SubjectLoader, namedtuple_map

            data_root_fp = "/home/ruilongli/data/nerf_synthetic/"
            target_sample_batch_size = 1 << 18
        from radiance_fields.ngp import NGPradianceField

        radiance_aabb = (
            scene_aabb if args.contraction is None else scale_aabb(scene_aabb, 2.0)
        )
        radiance_field = NGPradianceField(aabb=radiance_aabb).to(device)
        optimizer = torch.optim.Adam(radiance_field.parameters(), lr=1e-2, eps=1e-15)
        max_steps = 20000
        occ_field_warmup_steps = 256
        grad_scaler = torch.cuda.amp.GradScaler(2**10)

    elif args.method == "vanilla":
        from datasets.nerf_synthetic import SubjectLoader, namedtuple_map
        from radiance_fields.mlp import VanillaNeRFRadianceField

        radiance_field = VanillaNeRFRadianceField().to(device)
        optimizer = torch.optim.Adam(radiance_field.parameters(), lr=5e-4)
        max_steps = 40000
        occ_field_warmup_steps = 2000
        grad_scaler = torch.cuda.amp.GradScaler(1)
        data_root_fp = "/home/ruilongli/data/nerf_synthetic/"
        target_sample_batch_size = 1 << 16

    elif args.method == "dnerf":
        from datasets.dnerf_synthetic import SubjectLoader, namedtuple_map
        from radiance_fields.mlp import DNeRFRadianceField

        radiance_field = DNeRFRadianceField().to(device)
        optimizer = torch.optim.Adam(radiance_field.parameters(), lr=5e-4)
        max_steps = 40000
        occ_field_warmup_steps = 2000
        grad_scaler = torch.cuda.amp.GradScaler(1)
        data_root_fp = "/home/ruilongli/data/dnerf/"
        target_sample_batch_size = 1 << 16

    scene = args.scene

    # setup some rendering settings
    render_n_samples = 1024
    render_step_size = (
        (scene_aabb[3:] - scene_aabb[:3]).max() * math.sqrt(3) / render_n_samples
    ).item()

    # setup dataset
    train_dataset = SubjectLoader(
        subject_id=scene,
        root_fp=data_root_fp,
        split=args.train_split,
        num_rays=target_sample_batch_size // render_n_samples,
        **train_dataset_kwargs,
    )

    train_dataset.images = train_dataset.images.to(device)
    train_dataset.camtoworlds = train_dataset.camtoworlds.to(device)
    train_dataset.K = train_dataset.K.to(device)
    if hasattr(train_dataset, "timestamps"):
        train_dataset.timestamps = train_dataset.timestamps.to(device)

    sg_aabb_center = train_dataset.camtoworlds[:, :3, -1].mean(dim=0)
    sg_aabb_half = (
        (train_dataset.camtoworlds[:, :3, -1] - sg_aabb_center).norm(dim=-1).mean()
    )
    sg_aabb = torch.cat([sg_aabb_center - sg_aabb_half, sg_aabb_center + sg_aabb_half])
    print(f"suggested aabb from train dataset: {sg_aabb.tolist()}")

    test_dataset = SubjectLoader(
        subject_id=scene,
        root_fp=data_root_fp,
        split="test",
        num_rays=None,
        **test_dataset_kwargs,
    )
    test_dataset.images = test_dataset.images.to(device)
    test_dataset.camtoworlds = test_dataset.camtoworlds.to(device)
    test_dataset.K = test_dataset.K.to(device)
    if hasattr(train_dataset, "timestamps"):
        test_dataset.timestamps = test_dataset.timestamps.to(device)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[max_steps // 2, max_steps * 3 // 4, max_steps * 9 // 10],
        gamma=0.33,
    )

    # setup occupancy field with eval function
    @torch.no_grad()
    def occ_eval_fn(x: torch.Tensor) -> torch.Tensor:
        """Evaluate occupancy given positions.

        Args:
            x: positions with shape (N, 3).
        Returns:
            occupancy values with shape (N, 1).
        """
        if args.method == "dnerf":
            idxs = torch.randint(
                0, len(train_dataset.timestamps), (x.shape[0],), device=x.device
            )
            t = train_dataset.timestamps[idxs]
            density_after_activation = radiance_field.query_density(x, t)
        else:
            density_after_activation = radiance_field.query_density(x)
        if args.contraction == "mipnerf360":
            # TODO: goes into radiance field: defined by user.
            # mipnerf360 contract scene into a sphere
            x_norm = (x - occ_field.aabb[:3]) / (
                occ_field.aabb[3:] - occ_field.aabb[:3]
            )
            x_norm = x_norm * 4 - 2  # to [-2, 2]
            r = torch.linalg.norm(x_norm, dim=-1, keepdim=True)
            contraction_scaling = 1 / ((2 - r) ** 2)
            contraction_scaling[r < 1] = 1
            contraction_scaling[r > 2] = 0
            contraction_scaling = torch.clamp(contraction_scaling, max=1e10)
            step_size = render_step_size * contraction_scaling
        else:
            step_size = render_step_size
        # those two are similar when density is small.
        occupancy = 1.0 - torch.exp(-density_after_activation * step_size)
        # occupancy = density_after_activation * step_size
        return occupancy

    occ_aabb = scene_aabb if args.contraction is None else scale_aabb(scene_aabb, 2.0)
    occ_field = OccupancyField(
        occ_eval_fn=occ_eval_fn, aabb=occ_aabb, resolution=128
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

            render_bkgd = data["color_bkgd"]
            rays = data["rays"]
            pixels = data["pixels"]
            timestamps = data.get("timestamps", None)

            # update occupancy grid
            occ_field.every_n_step(step, warmup_steps=occ_field_warmup_steps)

            rgb, acc, counter, compact_counter = render_image(
                radiance_field,
                rays,
                timestamps,
                render_bkgd,
                render_step_size,
                contraction=args.contraction,
                cone_angle=args.cone_angle,
            )
            num_rays = len(pixels)
            num_rays = int(
                num_rays * (target_sample_batch_size / float(compact_counter))
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
            if step >= 0 and step % max_steps == 0 and step > 0:
                # evaluation
                radiance_field.eval()

                psnrs = []
                with torch.no_grad():
                    for i in tqdm.tqdm(range(len(test_dataset))):
                        data = test_dataset[i]
                        render_bkgd = data["color_bkgd"]
                        rays = data["rays"]
                        pixels = data["pixels"]
                        timestamps = data.get("timestamps", None)

                        # rendering
                        rgb, acc, _, _ = render_image(
                            radiance_field,
                            rays,
                            timestamps,
                            render_bkgd,
                            render_step_size,
                            test_chunk_size=args.test_chunk_size,
                            contraction=args.contraction,
                            cone_angle=args.cone_angle,
                        )
                        mse = F.mse_loss(rgb, pixels)
                        psnr = -10.0 * torch.log(mse) / np.log(10.0)
                        psnrs.append(psnr.item())
                        # if step == max_steps:
                        #     output_dir = os.path.join("./outputs/nerfacc/", scene)
                        #     os.makedirs(output_dir, exist_ok=True)
                        #     save = torch.cat([pixels, rgb], dim=1)
                        #     imageio.imwrite(
                        #         os.path.join(output_dir, "%05d.png" % i),
                        #         (save.cpu().numpy() * 255).astype(np.uint8),
                        #     )
                        # else:
                        #     imageio.imwrite(
                        #         "acc_binary_test.png",
                        #         ((acc > 0).float().cpu().numpy() * 255).astype(np.uint8),
                        #     )
                        #     imageio.imwrite(
                        #         "rgb_test.png",
                        #         (rgb.cpu().numpy() * 255).astype(np.uint8),
                        #     )
                        #     break
                psnr_avg = sum(psnrs) / len(psnrs)
                print(f"evaluation: {psnr_avg=}")
                train_dataset.training = True

            if step == max_steps:
                print("training stops")
                exit()
            tic_data = time.time()

            step += 1

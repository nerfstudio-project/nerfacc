"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import argparse
import math
import time
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from radiance_fields.tensorf import TensorEncoder, TensorRadianceField
from torch import nn
from utils import render_image, set_random_seed

from nerfacc import ContractionType, OccupancyGrid


def get_param_groups(module: nn.Module, lr: float, tensor_lr: float):
    param_groups = [
        {"params": [], "lr": lr},
        {"params": [], "lr": tensor_lr},
    ]
    for m in module.modules():
        if isinstance(m, TensorEncoder):
            param_groups[1]["params"] += list(m.parameters(recurse=False))
        else:
            param_groups[0]["params"] += list(m.parameters(recurse=False))
    return param_groups


def get_may_upsample_radiance_field_fn(
    radiance_field: nn.Module,
    max_resolution: int,
    upsample_steps: List[int],
):
    resolutions = (
        np.round(
            np.exp(
                np.linspace(
                    np.log(radiance_field.resolution),
                    np.log(max_resolution),
                    len(upsample_steps) + 1,
                )
            )
        ).astype(np.int64)
    ).tolist()[1:]

    def may_upsample_radiance_field_fn(
        optimizer: torch.optim.Optimizer, step: int
    ):
        if step in upsample_steps:
            i = upsample_steps.index(step)
            #  print(
            #      f"{step}: upsample radiance field to resolution {resolutions[i]}"
            #  )
            radiance_field.upsample(resolutions[i])
            optimizer = torch.optim.Adam(
                get_param_groups(radiance_field, 1e-3, 2e-2),
                eps=1e-15,
            )
        return optimizer

    return may_upsample_radiance_field_fn


device = "cuda:0"
set_random_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_split",
    type=str,
    default="train",
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
        # mipnerf360 unbounded
        "garden",
        "bicycle",
        "bonsai",
        "counter",
        "kitchen",
        "room",
        "stump",
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
    default=8192,
)
parser.add_argument(
    "--unbounded",
    action="store_true",
    help="whether to use unbounded rendering",
)
parser.add_argument(
    "--auto_aabb",
    action="store_true",
    help="whether to automatically compute the aabb",
)
parser.add_argument("--cone_angle", type=float, default=0.0)
args = parser.parse_args()

render_n_samples = 1024

# setup the dataset
train_dataset_kwargs = {}
test_dataset_kwargs = {}
if args.unbounded:
    from datasets.nerf_360_v2 import SubjectLoader

    data_root_fp = "/home/ruilongli/data/360_v2/"
    target_sample_batch_size = 1 << 20
    train_dataset_kwargs = {"color_bkgd_aug": "random", "factor": 4}
    test_dataset_kwargs = {"factor": 4}
    grid_resolution = 256
    # TODO(Hang Gao @ 02/20): Copy-paste from other scripts should just
    # work but it is not the priority now.
    raise NotImplementedError("Unbounded rendering is not implemented yet")
else:
    from datasets.nerf_synthetic import SubjectLoader

    #  data_root_fp = "/home/ruilongli/data/nerf_synthetic/"
    data_root_fp = "/home/hangg/datasets/nerf-blender/"
    target_sample_batch_size = 1 << 18
    grid_resolution = 128

train_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=data_root_fp,
    split=args.train_split,
    #  num_rays=target_sample_batch_size // render_n_samples,
    num_rays=4096,
    **train_dataset_kwargs,
)

train_dataset.images = train_dataset.images.to(device)
train_dataset.camtoworlds = train_dataset.camtoworlds.to(device)
train_dataset.K = train_dataset.K.to(device)

test_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=data_root_fp,
    split="test",
    num_rays=None,
    **test_dataset_kwargs,
)
test_dataset.images = test_dataset.images.to(device)
test_dataset.camtoworlds = test_dataset.camtoworlds.to(device)
test_dataset.K = test_dataset.K.to(device)

if args.auto_aabb:
    camera_locs = torch.cat(
        [train_dataset.camtoworlds, test_dataset.camtoworlds]
    )[:, :3, -1]
    args.aabb = torch.cat(
        [camera_locs.min(dim=0).values, camera_locs.max(dim=0).values]
    ).tolist()
    print("Using auto aabb", args.aabb)

# setup the scene bounding box.
if args.unbounded:
    print("Using unbounded rendering")
    contraction_type = ContractionType.UN_BOUNDED_SPHERE
    # contraction_type = ContractionType.UN_BOUNDED_TANH
    scene_aabb = None
    near_plane = 0.2
    far_plane = 1e4
    render_step_size = 1e-2
    alpha_thre = 1e-2
else:
    contraction_type = ContractionType.AABB
    scene_aabb = torch.tensor(args.aabb, dtype=torch.float32, device=device)
    near_plane = None
    far_plane = None
    render_step_size = (
        (scene_aabb[3:] - scene_aabb[:3]).max()
        * math.sqrt(3)
        / render_n_samples
    ).item()
    alpha_thre = 0.0

# setup the radiance field we want to train.
#  max_steps = 30000
max_steps = 15000
base_resolution = 128
max_resolution = 300
upsample_steps = [2000, 3000, 4000, 5500, 7000]
radiance_field = TensorRadianceField(
    aabb=args.aabb,
    resolution=base_resolution,
    unbounded=args.unbounded,
).to(device)
optimizer = torch.optim.Adam(
    get_param_groups(radiance_field, 1e-3, 2e-2),
    eps=1e-15,
)
lr_mult = 0.1 ** (1 / max_steps)
#  render_step_size *= 5
#  render_step_size_mult = 1 / 5 ** (1 / max_steps)

occupancy_grid = OccupancyGrid(
    roi_aabb=args.aabb,
    resolution=grid_resolution,
    contraction_type=contraction_type,
).to(device)

may_upsample_radiance_field_fn = get_may_upsample_radiance_field_fn(
    radiance_field, max_resolution, upsample_steps
)

# training
step = 0
run_tic = step_tic = time.time()
num_rays_per_sec = num_samples_per_sec = 0
for epoch in range(10000000):
    for i in range(len(train_dataset)):
        radiance_field.train()
        data = train_dataset[i]

        render_bkgd = data["color_bkgd"]
        rays = data["rays"]
        pixels = data["pixels"]

        def occ_eval_fn(x):
            if args.cone_angle > 0.0:
                # randomly sample a camera for computing step size.
                camera_ids = torch.randint(
                    0, len(train_dataset), (x.shape[0],), device=device
                )
                origins = train_dataset.camtoworlds[camera_ids, :3, -1]
                t = (origins - x).norm(dim=-1, keepdim=True)
                # compute actual step size used in marching, based on the distance to the camera.
                step_size = torch.clamp(
                    t * args.cone_angle, min=render_step_size
                )
                # filter out the points that are not in the near far plane.
                if (near_plane is not None) and (far_plane is not None):
                    step_size = torch.where(
                        (t > near_plane) & (t < far_plane),
                        step_size,
                        torch.zeros_like(step_size),
                    )
            else:
                step_size = render_step_size
            # compute occupancy
            density = radiance_field.query_density(x)
            return density * step_size

        # update occupancy grid
        occupancy_grid.every_n_step(step=step, occ_eval_fn=occ_eval_fn)

        # render
        rgb, acc, depth, n_rendering_samples = render_image(
            radiance_field,
            occupancy_grid,
            rays,
            scene_aabb,
            # rendering options
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            render_bkgd=render_bkgd,
            cone_angle=args.cone_angle,
            alpha_thre=alpha_thre,
        )
        num_rays_per_sec += len(pixels)
        num_samples_per_sec += n_rendering_samples
        if n_rendering_samples == 0:
            continue

        # dynamic batch size for rays to keep sample batch size constant.
        num_rays = len(pixels)
        num_rays = int(
            num_rays * (target_sample_batch_size / float(n_rendering_samples))
        )
        #  train_dataset.update_num_rays(num_rays)
        alive_ray_mask = acc.squeeze(-1) > 0

        # compute loss
        loss = F.smooth_l1_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for param_group in optimizer.param_groups:
            param_group["lr"] *= lr_mult
        #  render_step_size *= render_step_size_mult

        # upsample radiance field if needed
        optimizer = may_upsample_radiance_field_fn(optimizer, step)

        if (step % 10000 == 0 and step > 0) or step == max_steps:
            #  if (step % 1000 == 0 and step > 0) or step == max_steps:
            #  if (step % 100 == 0 and step > 0) or step == max_steps:
            elapsed_time = time.time() - run_tic
            delta_time = time.time() - step_tic
            psnr = -10.0 * torch.log(F.mse_loss(rgb, pixels)) / np.log(10.0)
            print(
                f"elapsed_time={elapsed_time:.2f}s | step={step} | "
                f"loss={loss:.4f} | "
                f"psnr={psnr:.4f} | "
                f"alive_ray_mask={alive_ray_mask.long().sum():d} | "
                #  f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} |"
                f"num_rays_per_sec={num_rays_per_sec/delta_time:.2e} | num_samples_per_sec={num_samples_per_sec/delta_time:.2e} |"
            )
            num_rays_per_sec = num_samples_per_sec = 0
            step_tic = time.time()

        if step == max_steps:
            #  if step % 1000 == 0 and step > 0:
            #  if step % 7500 == 0 and step > 0:
            # evaluation
            radiance_field.eval()

            psnrs = []
            with torch.no_grad():
                for i in tqdm.tqdm(range(len(test_dataset))):
                    data = test_dataset[i]
                    render_bkgd = data["color_bkgd"]
                    rays = data["rays"]
                    pixels = data["pixels"]

                    # rendering
                    rgb, acc, depth, _ = render_image(
                        radiance_field,
                        occupancy_grid,
                        rays,
                        scene_aabb,
                        # rendering options
                        near_plane=near_plane,
                        far_plane=far_plane,
                        render_step_size=render_step_size,
                        render_bkgd=render_bkgd,
                        cone_angle=args.cone_angle,
                        alpha_thre=alpha_thre,
                        # test options
                        test_chunk_size=args.test_chunk_size,
                    )
                    mse = F.mse_loss(rgb, pixels)
                    psnr = -10.0 * torch.log(mse) / np.log(10.0)
                    psnrs.append(psnr.item())
                    #  import imageio

                    #  #  imageio.imwrite(
                    #  #      "acc_binary_test.png",
                    #  #      ((acc > 0).float().cpu().numpy() * 255).astype(
                    #  #          np.uint8
                    #  #      ),
                    #  #  )
                    #  imageio.imwrite(
                    #      "rgb_test.png",
                    #      (rgb.cpu().numpy() * 255).astype(np.uint8),
                    #  )
                    #  break
            psnr_avg = sum(psnrs) / len(psnrs)
            print(f"evaluation: psnr_avg={psnr_avg}")
            train_dataset.training = True

        if step == max_steps:
            print("training stops")
            exit()

        step += 1

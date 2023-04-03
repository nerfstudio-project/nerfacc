"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import argparse
import math
import pathlib
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from datasets.dnerf_synthetic import SubjectLoader
from lpips import LPIPS
from radiance_fields.mlp import TNeRFRadianceField

from examples.utils import render_image_with_occgrid, set_random_seed
from nerfacc.estimators.occ_grid import OccGridEstimator

device = "cuda:0"
set_random_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root",
    type=str,
    default=str(pathlib.Path.cwd() / "data/dnerf"),
    help="the root dir of the dataset",
)
parser.add_argument(
    "--train_split",
    type=str,
    default="train",
    choices=["train"],
    help="which train split to use",
)
parser.add_argument(
    "--scene",
    type=str,
    default="lego",
    choices=[
        # dnerf
        "bouncingballs",
        "hellwarrior",
        "hook",
        "jumpingjacks",
        "lego",
        "mutant",
        "standup",
        "trex",
    ],
    help="which scene to use",
)
parser.add_argument(
    "--test_chunk_size",
    type=int,
    default=4096,
)
args = parser.parse_args()

# training parameters
max_steps = 30000
init_batch_size = 1024
target_sample_batch_size = 1 << 16
# scene parameters
aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
near_plane = 0.0
far_plane = 1.0e10
# model parameters
grid_resolution = 128
grid_nlvl = 1
# render parameters
render_step_size = 5e-3

# setup the dataset
train_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split=args.train_split,
    num_rays=init_batch_size,
    device=device,
)
test_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split="test",
    num_rays=None,
    device=device,
)

estimator = OccGridEstimator(
    roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
).to(device)

# setup the radiance field we want to train.
radiance_field = TNeRFRadianceField().to(device)
optimizer = torch.optim.Adam(radiance_field.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[
        max_steps // 2,
        max_steps * 3 // 4,
        max_steps * 5 // 6,
        max_steps * 9 // 10,
    ],
    gamma=0.33,
)

lpips_net = LPIPS(net="vgg").to(device)
lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()

# training
tic = time.time()
for step in range(max_steps + 1):
    radiance_field.train()
    estimator.train()

    i = torch.randint(0, len(train_dataset), (1,)).item()
    data = train_dataset[i]

    render_bkgd = data["color_bkgd"]
    rays = data["rays"]
    pixels = data["pixels"]
    timestamps = data["timestamps"]

    # update occupancy grid
    estimator.update_every_n_steps(
        step=step,
        occ_eval_fn=lambda x: radiance_field.query_opacity(
            x, timestamps, render_step_size
        ),
        occ_thre=1e-2,
    )

    # render
    rgb, acc, depth, n_rendering_samples = render_image_with_occgrid(
        radiance_field,
        estimator,
        rays,
        # rendering options
        near_plane=near_plane,
        render_step_size=render_step_size,
        render_bkgd=render_bkgd,
        alpha_thre=0.01 if step > 1000 else 0.00,
        # t-nerf options
        timestamps=timestamps,
    )
    if n_rendering_samples == 0:
        continue

    if target_sample_batch_size > 0:
        # dynamic batch size for rays to keep sample batch size constant.
        num_rays = len(pixels)
        num_rays = int(
            num_rays * (target_sample_batch_size / float(n_rendering_samples))
        )
        train_dataset.update_num_rays(num_rays)

    # compute loss
    loss = F.smooth_l1_loss(rgb, pixels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if step % 5000 == 0:
        elapsed_time = time.time() - tic
        loss = F.mse_loss(rgb, pixels)
        psnr = -10.0 * torch.log(loss) / np.log(10.0)
        print(
            f"elapsed_time={elapsed_time:.2f}s | step={step} | "
            f"loss={loss:.5f} | psnr={psnr:.2f} | "
            f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} | "
            f"max_depth={depth.max():.3f} | "
        )

    if step > 0 and step % max_steps == 0:
        # evaluation
        radiance_field.eval()
        estimator.eval()

        psnrs = []
        lpips = []
        with torch.no_grad():
            for i in tqdm.tqdm(range(len(test_dataset))):
                data = test_dataset[i]
                render_bkgd = data["color_bkgd"]
                rays = data["rays"]
                pixels = data["pixels"]
                timestamps = data["timestamps"]

                # rendering
                rgb, acc, depth, _ = render_image_with_occgrid(
                    radiance_field,
                    estimator,
                    rays,
                    # rendering options
                    near_plane=near_plane,
                    render_step_size=render_step_size,
                    render_bkgd=render_bkgd,
                    alpha_thre=0.01,
                    # test options
                    test_chunk_size=args.test_chunk_size,
                    # t-nerf options
                    timestamps=timestamps,
                )
                mse = F.mse_loss(rgb, pixels)
                psnr = -10.0 * torch.log(mse) / np.log(10.0)
                psnrs.append(psnr.item())
                lpips.append(lpips_fn(rgb, pixels).item())
                # if i == 0:
                #     imageio.imwrite(
                #         "rgb_test.png",
                #         (rgb.cpu().numpy() * 255).astype(np.uint8),
                #     )
                #     imageio.imwrite(
                #         "rgb_error.png",
                #         (
                #             (rgb - pixels).norm(dim=-1).cpu().numpy() * 255
                #         ).astype(np.uint8),
                #     )
        psnr_avg = sum(psnrs) / len(psnrs)
        lpips_avg = sum(lpips) / len(lpips)
        print(f"evaluation: psnr_avg={psnr_avg}, lpips_avg={lpips_avg}")

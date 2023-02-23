"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import argparse
import itertools
import time

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from datasets.nerf_360_v2 import SubjectLoader
from prop_utils import (
    compute_prop_loss,
    get_proposal_requires_grad_fn,
    render_image,
)
from radiance_fields.ngp import NGPDensityField, NGPRadianceField
from utils import set_random_seed

parser = argparse.ArgumentParser()
parser.add_argument(
    "--scene",
    type=str,
    default="garden",
    choices=[
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
    "--test_chunk_size",
    type=int,
    default=8192,
)
args = parser.parse_args()

# for reproducibility
set_random_seed(42)

# hyperparameters
device = "cuda:0"
max_steps = 30000
#  max_steps = 4500  # 1 min.
batch_size = 4096
scene_aabb = None
near_plane = 0.2
far_plane = 1e3
sampling_type = "lindisp"
num_samples = 48
num_samples_per_prop = [256, 96]
opaque_bkgd = True

# setup the dataset
train_dataset_kwargs = {}
test_dataset_kwargs = {}

#  data_root_fp = "/home/ruilongli/data/360_v2/"
data_root_fp = "/home/hangg/datasets/360_v2/"
train_dataset_kwargs = {"color_bkgd_aug": "random", "factor": 4}
test_dataset_kwargs = {"factor": 4}

train_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=data_root_fp,
    split="train",
    num_rays=batch_size,
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

# The region of interest of the scene has been normalized to [-1, 1]^3.
aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device)

# setup the radiance field we want to train.
grad_scaler = torch.cuda.amp.GradScaler(2**10)
radiance_field = NGPRadianceField(
    aabb=aabb,
    unbounded=True,
).to(device)
proposal_networks = [
    NGPDensityField(
        aabb=aabb,
        unbounded=True,
        n_levels=5,
        max_resolution=128,
    ).to(device),
    NGPDensityField(
        aabb=aabb,
        unbounded=True,
        n_levels=5,
        max_resolution=256,
    ).to(device),
]
proposal_requires_grad_fn = get_proposal_requires_grad_fn()
optimizer = torch.optim.Adam(
    itertools.chain(
        radiance_field.parameters(),
        *[p.parameters() for p in proposal_networks],
    ),
    lr=1e-2,
    eps=1e-15,
)

# training
step = 0
tic = time.time()
for epoch in range(10000000):
    for i in range(len(train_dataset)):
        radiance_field.train()
        for p in proposal_networks:
            p.train()
        data = train_dataset[i]

        render_bkgd = data["color_bkgd"]
        rays = data["rays"]
        pixels = data["pixels"]

        # render
        (
            rgb,
            acc,
            depth,
            weights_per_level,
            s_vals_per_level,
            n_rendering_samples,
        ) = render_image(
            radiance_field,
            proposal_networks,
            rays,
            scene_aabb,
            # rendering options
            num_samples=num_samples,
            num_samples_per_prop=num_samples_per_prop,
            near_plane=near_plane,
            far_plane=far_plane,
            sampling_type=sampling_type,
            opaque_bkgd=opaque_bkgd,
            render_bkgd=render_bkgd,
            # train options
            proposal_requires_grad=proposal_requires_grad_fn(step),
        )

        # compute loss
        loss = F.smooth_l1_loss(rgb, pixels)
        loss += compute_prop_loss(s_vals_per_level, weights_per_level)

        optimizer.zero_grad()
        # do not unscale it because we are using Adam.
        grad_scaler.scale(loss).backward()
        optimizer.step()

        if (step % 10000 == 0 and step > 0) or step == max_steps:
            elapsed_time = time.time() - tic
            psnr = -10.0 * torch.log(F.mse_loss(rgb, pixels)) / np.log(10.0)
            print(
                f"elapsed_time={elapsed_time:.2f}s | step={step} | "
                f"loss={loss:.4f} | "
                f"psnr={psnr:.4f} | "
                f"n_rendering_samples={n_rendering_samples:d} |"
            )

        if step == max_steps:
            # evaluation
            radiance_field.eval()
            for p in proposal_networks:
                p.eval()

            psnrs = []
            with torch.no_grad():
                for j in tqdm.tqdm(range(len(test_dataset))):
                    data = test_dataset[j]
                    render_bkgd = data["color_bkgd"]
                    rays = data["rays"]
                    pixels = data["pixels"]

                    # rendering
                    (
                        rgb,
                        acc,
                        depth,
                        weights_per_level,
                        s_vals_per_level,
                        n_rendering_samples,
                    ) = render_image(
                        radiance_field,
                        proposal_networks,
                        rays,
                        scene_aabb,
                        # rendering options
                        num_samples=num_samples,
                        num_samples_per_prop=num_samples_per_prop,
                        near_plane=near_plane,
                        far_plane=far_plane,
                        sampling_type=sampling_type,
                        opaque_bkgd=opaque_bkgd,
                        render_bkgd=render_bkgd,
                        # test options
                        test_chunk_size=args.test_chunk_size,
                    )
                    psnr = (
                        -10.0
                        * torch.log(F.mse_loss(rgb, pixels))
                        / np.log(10.0)
                    )
                    psnrs.append(psnr.item())
                    #  print(psnr.item())
                    #  import os

                    #  import imageio

                    #  os.makedirs(".cache", exist_ok=True)
                    #  imageio.imwrite(
                    #      f".cache/{j}_{step}.png",
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

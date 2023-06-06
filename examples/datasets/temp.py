"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import collections
import json
import os

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from .utils import Rays

def similarity_from_cameras(c2w, strict_scaling):
    """
    reference: nerf-factory
    Get a similarity transform to normalize dataset
    from c2w (OpenCV convention) cameras
    :param c2w: (N, 4)
    :return T (4,4) , scale (float)
    """
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    # (1) Rotate the world so that z+ is the up axis
    # we estimate the up axis by averaging the camera up axes
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)

    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
    else:
        # In the unlikely case the original data has y+ up axis,
        # rotate 180-deg about x axis
        R_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    #  R_align = np.eye(3) # DEBUG
    R = R_align @ R
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]

    # (2) Recenter the scene using camera center rays
    # find the closest point to the origin for each camera's center ray
    nearest = t + (fwds * -t).sum(-1)[:, None] * fwds

    # median for more robustness
    translate = -np.median(nearest, axis=0)

    #  translate = -np.mean(t, axis=0)  # DEBUG

    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align

    # (3) Rescale the scene using camera distances
    scale_fn = np.max if strict_scaling else np.median
    scale = 1.0 / scale_fn(np.linalg.norm(t + translate, axis=-1))

    return transform, scale


def _load_renderings():
    root_fp = "/home/ruilongl/workspace/nerfacc/examples/temp/"
    frames = np.loadtxt(os.path.join(root_fp, "datalist.txt"), dtype=str)
    
    images = []
    camtoworlds = []
    intrinsics = []

    for frame in frames:
        fname = os.path.join(root_fp, "raw_rgb", frame + ".png")
        pixels = imageio.imread(fname)
        images.append(pixels)

        annot = os.path.join(root_fp, "annotations", frame + ".json")
        with open(annot) as f:
            annot = json.load(f)
        camtoworlds.append(annot["camera_pose"])
        intrinsics.append(annot["intrinsics"])

    images = np.stack(images, axis=0)
    camtoworlds = np.stack(camtoworlds, axis=0)
    intrinsics = np.stack(intrinsics, axis=0)

    return images, camtoworlds, intrinsics


class SubjectLoader(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    WIDTH, HEIGHT = 1600, 900
    NEAR, FAR = 0.1, 15
    OPENGL_CAMERA = False

    def __init__(
        self,
        color_bkgd_aug: str = "random",
        num_rays: int = None,
        near: float = None,
        far: float = None,
        batch_over_images: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        assert color_bkgd_aug in ["white", "black", "random"]
        self.training = num_rays is not None

        self.num_rays = num_rays
        self.near = self.NEAR if near is None else near
        self.far = self.FAR if far is None else far
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images
        self.images, self.camtoworlds, self.intrinsics = _load_renderings()
        # import pdb; pdb.set_trace()

        # # normalize the scene
        # T, sscale = similarity_from_cameras(
        #     self.camtoworlds, strict_scaling=False
        # )
        # self.camtoworlds = np.einsum("nij, ki -> nkj", self.camtoworlds, T)
        # self.camtoworlds[:, :3, 3] *= sscale
        # print ("sscale", sscale)

        self.images = torch.from_numpy(self.images).to(torch.uint8)
        self.camtoworlds = torch.from_numpy(self.camtoworlds).to(torch.float32)
        self.Ks = torch.from_numpy(self.intrinsics).to(torch.float32)

        self.images = self.images.to(device)
        self.camtoworlds = self.camtoworlds.to(device)
        self.Ks = self.Ks.to(device)
        assert self.images.shape[1:3] == (self.HEIGHT, self.WIDTH)


    def __len__(self):
        return len(self.images)

    @torch.no_grad()
    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        pixels, rays = data["pixels"], data["rays"]

        if self.training:
            if self.color_bkgd_aug == "random":
                color_bkgd = torch.rand(3, device=self.images.device)
            elif self.color_bkgd_aug == "white":
                color_bkgd = torch.ones(3, device=self.images.device)
            elif self.color_bkgd_aug == "black":
                color_bkgd = torch.zeros(3, device=self.images.device)
        else:
            # just use white during inference
            color_bkgd = torch.ones(3, device=self.images.device)

        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["pixels", "rays"]},
        }

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        num_rays = self.num_rays

        if self.training:
            if self.batch_over_images:
                image_id = torch.randint(
                    0,
                    len(self.images),
                    size=(num_rays,),
                    device=self.images.device,
                )
            else:
                image_id = [index] * num_rays
            x = torch.randint(
                0, self.WIDTH, size=(num_rays,), device=self.images.device
            )
            y = torch.randint(
                0, self.HEIGHT, size=(num_rays,), device=self.images.device
            )
        else:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.WIDTH, device=self.images.device),
                torch.arange(self.HEIGHT, device=self.images.device),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()

        # generate rays
        pixels = self.images[image_id, y, x] / 255.0  # (num_rays, 4)
        c2w = self.camtoworlds[image_id]  # (num_rays, 3, 4)
        K = self.Ks[image_id]
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - K[:, 0, 2] + 0.5) / K[:, 0, 0],
                    (y - K[:, 1, 2] + 0.5)
                    / K[:, 1, 1]
                    * (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  # [num_rays, 3]

        # [n_cams, height, width, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )

        if self.training:
            origins = torch.reshape(origins, (num_rays, 3))
            viewdirs = torch.reshape(viewdirs, (num_rays, 3))
            pixels = torch.reshape(pixels, (num_rays, 3))
        else:
            origins = torch.reshape(origins, (self.HEIGHT, self.WIDTH, 3))
            viewdirs = torch.reshape(viewdirs, (self.HEIGHT, self.WIDTH, 3))
            pixels = torch.reshape(pixels, (self.HEIGHT, self.WIDTH, 3))

        rays = Rays(origins=origins, viewdirs=viewdirs)

        return {
            "pixels": pixels,  # [h, w, 3] or [num_rays, 3]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
        }


if __name__ == "__main__":
    # pip install viser==0.0.15
    import viser
    import time
    import viser.transforms as tf

    images, camtoworlds, intrinsics = _load_renderings()

    server = viser.ViserServer()

    aabb = np.array([405, 1115, 0.5, 415, 1125, 2.5])
    points = np.random.rand(10000, 3)
    points = points * (aabb[3:] - aabb[:3]) + aabb[:3]
    colors = np.zeros_like(points)

    server.add_point_cloud(
        "/points",
        points - camtoworlds[0, :3, 3],
        colors,
    )

    for i, (c2w, K) in enumerate(zip(camtoworlds, intrinsics)):
        fov = 2 * np.arctan2(images[0].shape[0] / 2, K[0, 0])
        aspect = images[0].shape[1] / images[0].shape[0]
        server.add_camera_frustum(
            f"/t{i}",
            fov=fov,
            aspect=aspect,
            scale=0.5,
            image=images[i, ::1, ::1],
            wxyz=tf.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, 3] - camtoworlds[0, :3, 3],
        )

    while True:
        time.sleep(10.0)

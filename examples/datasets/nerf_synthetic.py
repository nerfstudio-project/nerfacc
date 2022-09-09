# Copyright (c) Meta Platforms, Inc. and affiliates.
import json
import os

import imageio.v2 as imageio
import numpy as np
import torch

from .utils import Cameras, generate_rays


def _load_renderings(root_fp: str, subject_id: str, split: str):
    """Load images from disk."""
    if not root_fp.startswith("/"):
        # allow relative path. e.g., "./data/nerf_synthetic/"
        root_fp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            root_fp,
        )

    data_dir = os.path.join(root_fp, subject_id)
    with open(os.path.join(data_dir, "transforms_{}.json".format(split)), "r") as fp:
        meta = json.load(fp)
    images = []
    camtoworlds = []

    for i in range(len(meta["frames"])):
        frame = meta["frames"][i]
        fname = os.path.join(data_dir, frame["file_path"] + ".png")
        rgba = imageio.imread(fname)
        camtoworlds.append(frame["transform_matrix"])
        images.append(rgba)

    images = np.stack(images, axis=0).astype(np.float32)
    camtoworlds = np.stack(camtoworlds, axis=0).astype(np.float32)

    h, w = images.shape[1:3]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * w / np.tan(0.5 * camera_angle_x)

    return images, camtoworlds, focal


class SubjectLoader(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    SPLITS = ["train", "val", "trainval", "test"]
    SUBJECT_IDS = [
        "chair",
        "drums",
        "ficus",
        "hotdog",
        "lego",
        "materials",
        "mic",
    ]

    WIDTH, HEIGHT = 800, 800
    NEAR, FAR = 2.0, 6.0

    def __init__(
        self,
        subject_id: str,
        root_fp: str,
        split: str,
        color_bkgd_aug: str = "white",
        num_rays: int = None,
        near: float = None,
        far: float = None,
    ):
        super().__init__()
        assert split in self.SPLITS, "%s" % split
        assert subject_id in self.SUBJECT_IDS, "%s" % subject_id
        assert color_bkgd_aug in ["white", "black", "random"]
        self.split = split
        self.num_rays = num_rays
        self.near = self.NEAR if near is None else near
        self.far = self.FAR if far is None else far
        self.training = (num_rays is not None) and (split in ["train", "trainval"])
        self.color_bkgd_aug = color_bkgd_aug
        if split == "trainval":
            _images_train, _camtoworlds_train, _focal_train = _load_renderings(
                root_fp, subject_id, "train"
            )
            _images_val, _camtoworlds_val, _focal_val = _load_renderings(
                root_fp, subject_id, "val"
            )
            self.images = np.concatenate([_images_train, _images_val])
            self.camtoworlds = np.concatenate([_camtoworlds_train, _camtoworlds_val])
            self.focal = _focal_train
        else:
            self.images, self.camtoworlds, self.focal = _load_renderings(
                root_fp, subject_id, split
            )
        assert self.images.shape[1:3] == (self.HEIGHT, self.WIDTH)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        rgba, rays = data["rgba"], data["rays"]
        pixels, alpha = torch.split(rgba, [3, 1], dim=-1)

        if self.training:
            if self.color_bkgd_aug == "random":
                color_bkgd = torch.rand(3)
            elif self.color_bkgd_aug == "white":
                color_bkgd = torch.ones(3)
            elif self.color_bkgd_aug == "black":
                color_bkgd = torch.zeros(3)
        else:
            # just use white during inference
            color_bkgd = torch.ones(3)

        pixels = pixels * alpha + color_bkgd * (1.0 - alpha)
        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgba", "rays"]},
        }

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        # load data
        camera_id = index
        K = np.array(
            [
                [self.focal, 0, self.WIDTH / 2.0],
                [0, self.focal, self.HEIGHT / 2.0],
                [0, 0, 1],
            ]
        ).astype(np.float32)
        w2c = np.linalg.inv(self.camtoworlds[camera_id])
        rgba = self.images[camera_id]

        # create pixels
        rgba = torch.from_numpy(rgba).float() / 255.0

        # create rays from camera
        cameras = Cameras(
            intrins=torch.from_numpy(K).float(),
            extrins=torch.from_numpy(w2c).float(),
            distorts=None,
            width=self.WIDTH,
            height=self.HEIGHT,
        )

        if self.num_rays is not None:
            x = torch.randint(0, self.WIDTH, size=(self.num_rays,))
            y = torch.randint(0, self.HEIGHT, size=(self.num_rays,))
            pixels_xy = torch.stack([x, y], dim=-1)
            rgba = rgba[y, x, :]
        else:
            pixels_xy = None  # full image

        # Be careful: This dataset's camera coordinate is not the same as
        # opencv's camera coordinate! It is actually opengl.
        rays = generate_rays(
            cameras,
            opencv_format=False,
            pixels_xy=pixels_xy,
        )

        return {
            "camera_id": camera_id,
            "rgba": rgba,  # [h, w, 4] or [num_rays, 4]
            "rays": rays,  # [h, w] or [num_rays, 4]
        }

import torch
import numpy as np
import collections
import json
import os
import torch.nn.functional as F

from .utils import Rays

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius, mat):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return torch.Tensor(mat) @ c2w

def generateSphericalTestPoses(root_fp: str, subject_id: str, numberOfFrames: int = 30):

   if not root_fp.startswith("/"):
      root_fp = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..",root_fp) # e.g., "./data/nerf_synthetic/"

   data_dir = os.path.join(root_fp, subject_id)
    
   with open(os.path.join(data_dir, 'transforms.json'), 'r') as fp:
      meta = json.load(fp)

   frame = meta["frames"][0]
   fname = os.path.join(data_dir, frame['file_path'][2:])

   #image intrinsics
   focal, cx, cy = frame['fl_x'], frame['cx'], frame['cy']
   K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])

   w, h = frame['w'], frame['h']

   c2w = np.array(frame["transform_matrix"])
   # camtoworlds = torch.stack([pose_spherical(angle, -45.0, 2.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
   camtoworlds = torch.stack([pose_spherical(angle, 0, 0, c2w) for angle in np.linspace(-180,180,numberOfFrames+1)[:-1]], 0)

   return camtoworlds, K, int(w), int(h)


class SubjectTestPoseLoader(torch.utils.data.Dataset):

   OPENGL_CAMERA = True
   def __init__(self, subject_id: str, root_fp: str,color_bkgd_aug: str = "white", numberOfFrames: int = 30):

      super().__init__()
      
      self.color_bkgd_aug = color_bkgd_aug
      self.camtoworlds, self.K, self.WIDTH, self.HEIGHT = generateSphericalTestPoses(root_fp, subject_id, numberOfFrames)

      self.camtoworlds = (self.camtoworlds).to(torch.float32)
      self.K = torch.from_numpy(self.K).to(torch.float32)

   def __len__(self):
      return len(self.camtoworlds)

   @torch.no_grad()
   def __getitem__(self, index):
      data = self.fetch_data(index)
      return data

   def fetch_data(self, index):
      """Fetch the data (it maybe cached for multiple batches)."""

      x, y = torch.meshgrid(
            torch.arange(self.WIDTH, device=self.camtoworlds.device),
            torch.arange(self.HEIGHT, device=self.camtoworlds.device),
            indexing="xy",
      )
      x = x.flatten()
      y = y.flatten()

      # generate rays
      c2w = self.camtoworlds[[index]]  # (num_rays, 3, 4)
      K = self.K
      camera_dirs = F.pad(
         torch.stack(
               [
                  (x - K[0, 2] + 0.5) / K[0, 0],
                  (y - K[1, 2] + 0.5) / K[1, 1] * (-1.0 if self.OPENGL_CAMERA else 1.0),
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

      origins = torch.reshape(origins, (self.HEIGHT, self.WIDTH, 3))
      viewdirs = torch.reshape(viewdirs, (self.HEIGHT, self.WIDTH, 3))

      rays = Rays(origins=origins, viewdirs=viewdirs)

      color_bkgd = torch.ones(3, device=self.camtoworlds.device)

      return {
         "rays": rays,  # [h, w, 3] or [num_rays, 3]
         "color_bkgd": color_bkgd
      }

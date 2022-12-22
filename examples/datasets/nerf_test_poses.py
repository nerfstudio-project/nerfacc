import torch
import numpy as np
import collections
import json
import os
import torch.nn.functional as F

from .utils import Rays

from scipy.spatial.transform import Rotation as SR


def sphericalPoses(p0,numberOfFrames):
   """
   We first move the camera to [0,0,tz] in the world coordinate space. 
   Then we rotate the camera pos 45 degrees wrt X axis.
   Finally we rotate the camera wrt Z axis numberOfFrames times.
   Note: Camera space and world space (ENU) is actually aligned 
      X_c == X_w or E (east)
      Y_c == Y_w or N (north)
      Z_c == Z_w or U (up)
      Camera is positioned at [0,0,tz] it is actually looking down to -Z direction 
   """
   transMat = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,2500],[0,0,0,1]]).astype(float) #move camera to 0,0,1500
   
   #rotate camera 45 degrees wrt X axis
   rotMatX = np.identity(4)
   rotMatX[0:3,0:3] = SR.from_euler('X',np.pi/4).as_matrix()
   
   #first translate then rotate
   transMat = rotMatX @ transMat
   
   poses = []
   for angle in np.linspace(0,2*np.pi,numberOfFrames):

      rotMatZ = np.identity(4)
      rotMatZ[0:3,0:3] = SR.from_euler('Z',angle).as_matrix()

      myPose = rotMatZ @ transMat
      # myPose[1,3] += 130 #We needed this for non-masked shuttle imageset to move rotation center
      poses.append(myPose)

   poses = np.stack(poses, axis=0)
   return poses

def generateSphericalTestPoses(root_fp: str, subject_id: str, numberOfFrames: int, factor: int):

   if not root_fp.startswith("/"):
      root_fp = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..",root_fp) # e.g., "./data/nerf_synthetic/"

   data_dir = os.path.join(root_fp, subject_id)
    
   with open(os.path.join(data_dir, 'transforms.json'), 'r') as fp:
      meta = json.load(fp)

   frame = meta["frames"][15]
   fname = os.path.join(data_dir, frame['file_path'][2:])

   #image intrinsics
   focal, cx, cy = frame['fl_x'], frame['cx'], frame['cy']
   K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])
   K[:2, :] /= factor

   w, h = frame['w']/factor, frame['h']/factor

   c2w = np.array(frame["transform_matrix"])
   camtoworlds = sphericalPoses(c2w, numberOfFrames)

   return camtoworlds, K, int(w), int(h)


class SubjectTestPoseLoader(torch.utils.data.Dataset):

   OPENGL_CAMERA = True
   def __init__(self, subject_id: str, root_fp: str,color_bkgd_aug: str = "black", numberOfFrames: int = 120, downscale_factor: int = 4):

      super().__init__()
      
      self.color_bkgd_aug = color_bkgd_aug
      self.camtoworlds, self.K, self.WIDTH, self.HEIGHT = generateSphericalTestPoses(root_fp, subject_id, numberOfFrames, downscale_factor)

      self.camtoworlds = torch.from_numpy(self.camtoworlds).to(torch.float32)
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

      if self.color_bkgd_aug == "random":
         color_bkgd = torch.rand(3, device=self.camtoworlds.device)
      elif self.color_bkgd_aug == "white":
         color_bkgd = torch.ones(3, device=self.camtoworlds.device)
      elif self.color_bkgd_aug == "black":
         color_bkgd = torch.zeros(3, device=self.camtoworlds.device)

      return {
         "rays": rays,  # [h, w, 3] or [num_rays, 3]
         "color_bkgd": color_bkgd
      }

import torch
import tqdm

from nerfacc.occupancy_field import OccupancyField

BATCH_SIZE = 16
device = "cuda:0"


def occ_eval_fn(positions: torch.Tensor) -> torch.Tensor:
    return torch.rand_like(positions[:, :1])


def test_occ_field():
    occ_field = OccupancyField(occ_eval_fn, aabb=[0, 0, 0, 1, 1, 1]).to(device)

    for step in tqdm.tqdm(range(100000000)):
        occ_field.every_n_step(step, occ_thre=0.1)


if __name__ == "__main__":
    test_occ_field()

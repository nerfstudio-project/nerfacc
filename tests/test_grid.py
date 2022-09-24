import torch
import tqdm

from nerfacc.grid import OccupancyGrid

device = "cuda:0"


def occ_eval_fn(positions: torch.Tensor) -> torch.Tensor:
    return torch.rand_like(positions[:, :1])


def test_occ_field():
    occ_field = OccupancyGrid(aabb=[0, 0, 0, 1, 1, 1]).to(device)

    for step in tqdm.tqdm(range(50000)):
        occ_field.every_n_step(step, occ_eval_fn, occ_thre=0.1)


if __name__ == "__main__":
    test_occ_field()

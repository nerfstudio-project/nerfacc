# nerfacc
[![Core Tests.](https://github.com/KAIR-BAIR/nerfacc/actions/workflows/code_checks.yml/badge.svg)](https://github.com/KAIR-BAIR/nerfacc/actions/workflows/code_checks.yml)
[![Documentation Status](https://readthedocs.com/projects/plenoptix-nerfacc/badge/?version=latest)](https://plenoptix-nerfacc.readthedocs-hosted.com/en/latest/?badge=latest)

This is a **tiny** tootlbox  for **accelerating** NeRF training & rendering using PyTorch CUDA extensions. Plug-and-play for most of the NeRFs!

## Examples: Instant-NGP NeRF

``` bash
python examples/train_ngp_nerf.py --train_split trainval --scene lego
```

Performance:

| PSNR | Lego | Mic | Materials | Chair | Hotdog |
| - | - | - | - | - | - |
| Papers (5mins) | 36.39 | 36.22 | 29.78 | 35.00 | 37.40 |
| Ours (~5mins)  | 36.61 | 37.62 | 30.11 | 36.09 | 38.09 |
| Exact training time  | 300s  | 274s  | 266s  | 341s  | 277s  |


## Examples: Vanilla MLP NeRF

``` bash
python examples/train_mlp_nerf.py --train_split train --scene lego
```

Performance:

| PNSR | Lego | Mic | Materials | Chair | Hotdog |
| - | - | - | - | - | - |
| Paper (~2days) | 32.54 | 32.91 | 29.62 | 33.00 | 36.18 |
| Ours (~45mins) | 33.21 | 33.36 | 29.48 | 32.79 | 35.54 |

## Examples: MLP NeRF on Dynamic objects (D-NeRF)

```bash
python examples/train_mlp_dnerf.py --train_split train --scene lego
```

Performance:

|  | Lego | Stand Up |
| - | - | - |
| Paper (~2days) | 21.64 | 32.79 |
| Ours (~45mins) | 24.66 | 33.98 |


## Examples: NGP on unbounded scene

On MipNeRF360 Garden scene

```bash
python examples/train_ngp_nerf.py --train_split train --scene garden --aabb="-4,-4,-4,4,4,4" --unbounded --cone_angle=0.004
```

Performance:

|  | Garden |
| - | - |
| Ours | 25.40 |
| Time | 1246s |

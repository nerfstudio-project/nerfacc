# nerfacc
[![Core Tests.](https://github.com/KAIR-BAIR/nerfacc/actions/workflows/code_checks.yml/badge.svg)](https://github.com/KAIR-BAIR/nerfacc/actions/workflows/code_checks.yml)
[![Documentation Status](https://readthedocs.com/projects/plenoptix-nerfacc/badge/?version=latest)](https://plenoptix-nerfacc.readthedocs-hosted.com/en/latest/?badge=latest)


This is a **tiny** toolbox  for **accelerating** NeRF training & rendering using PyTorch CUDA extensions. Plug-and-play for most of the NeRFs!

## Examples: 

``` bash
# Instant-NGP Nerf
python examples/train_ngp_nerf.py --train_split trainval --scene lego
```

``` bash
# Vanilla MLP Nerf
python examples/train_mlp_nerf.py --train_split train --scene lego
```

```bash
# MLP Nerf on Dynamic objects (D-Nerf)
python examples/train_mlp_dnerf.py --train_split train --scene lego
```

```bash
# NGP on MipNeRF360 unbounded scene
python examples/train_ngp_nerf.py --train_split train --scene garden --auto_aabb --unbounded --cone_angle=0.004
```

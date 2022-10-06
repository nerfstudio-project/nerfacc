# NerfAcc
[![Core Tests.](https://github.com/KAIR-BAIR/nerfacc/actions/workflows/code_checks.yml/badge.svg)](https://github.com/KAIR-BAIR/nerfacc/actions/workflows/code_checks.yml)
[![Documentation Status](https://readthedocs.com/projects/plenoptix-nerfacc/badge/?version=latest)](https://www.nerfacc.com/en/latest/?badge=latest)

https://www.nerfacc.com/

This is a **tiny** toolbox  for **accelerating** NeRF training & rendering using PyTorch CUDA extensions. Plug-and-play for most of the NeRFs!

## Installation

```
pip install nerfacc
```

## Examples: 

Before running those example scripts, please check the script about which dataset it is needed, and download
the dataset first.

``` bash
# Instant-NGP NeRF in 4.5 minutes.
# See results at here: https://www.nerfacc.com/en/latest/examples/ngp.html
python examples/train_ngp_nerf.py --train_split trainval --scene lego
```

``` bash
# Vanilla MLP NeRF in 1 hour.
# See results at here: https://www.nerfacc.com/en/latest/examples/vanilla.html
python examples/train_mlp_nerf.py --train_split train --scene lego
```

```bash
# T-NeRF for Dynamic objects in 1 hour.
# See results at here: https://www.nerfacc.com/en/latest/examples/dnerf.html
python examples/train_mlp_dnerf.py --train_split train --scene lego
```

```bash
# Unbounded scene in 1 hour.
# See results at here: https://www.nerfacc.com/en/latest/examples/unbounded.html
python examples/train_ngp_nerf.py --train_split train --scene garden --auto_aabb --unbounded --cone_angle=0.004
```

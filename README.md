# nerfacc

This is a **tiny** tootlbox  for **accelerating** NeRF training & rendering using PyTorch CUDA extensions. Plug-and-play for most of the NeRFs!

## Examples: Instant-NGP NeRF

``` bash
python examples/trainval.py ngp --train_split trainval
```

Performance on TITAN RTX :

| trainval | Lego | Mic | Materials | Chair | Hotdog |
| - | - | - | - | - | - |
| Time | 300s  | 274s  | 266s  | 341s  | 277s  |
| PSNR | 36.61 | 37.62 | 30.11 | 36.09 | 38.09 |
| FPS  | 12.87 | 23.67 | 9.33  | 16.91 | 7.48  |

Instant-NGP paper (5 min) on 3090 (w/ mask):

| trainval | Lego | Mic | Materials | Chair | Hotdog |
| - | - | - | - | - | - |
| PSNR | 36.39 | 36.22 | 29.78 | 35.00 | 37.40 |


## Examples: Vanilla MLP NeRF

``` bash
python examples/trainval.py vanilla --train_split train
```

Performance on test set:

|  | Lego | Mic | Materials | Chair | Hotdog |
| - | - | - | - | - | - |
| Paper PSNR (train set) | 32.54 | 32.91 | 29.62 | 33.00 | 36.18 |
| Our PSNR (train set) | 33.21 | 33.36 | 29.48 | 32.79 | 35.54 |
| Our PSNR (trainval set) | 33.66  | - | - | - | - | - |
| Our train time & test FPS | 45min; 0.43FPS | 44min; 1FPS | 37min; 0.33FPS* | 44min; 0.57FPS* | 50min; 0.15 FPS* |

For reference, vanilla NeRF paper trains on V100 GPU for 1-2 days per scene. Test time rendering takes about 30 secs to render a 800x800 image. Our model is trained on a TITAN X.

Note: We only use a single MLP with more samples (1024), instead of two MLPs with coarse-to-fine sampling as in the paper. Both ways share the same spirit to do dense sampling around the surface. Our fast rendering inheritly skip samples away from the surface so we can simplly increase the number of samples with a single MLP, to achieve the same goal with coarse-to-fine sampling, without runtime or memory issue.

*FPS for some scenes are tested under `--test_chunk_size=8192` (default is `81920`) to avoid OOM.


## Examples: MLP NeRF on Dynamic objects

Here we trained something similar to D-NeRF on the dnerf dataset:

``` bash
python examples/trainval.py dnerf --train_split train --test_chunk_size=8192
```

Performance on test set:

|  | Lego | Stand Up |
| - | - | - |
| DNeRF paper PSNR (train set) | 21.64 | 32.79 |
| Our PSNR (train set) | 24.66 | 33.98 |
| Our train time & test FPS | 43min; 0.15FPS | 41min; 0.4FPS |


Note the numbers here are tested with version `v0.0.8`

<!-- 
## Tips:

1. sample rays over all images per iteration (`batch_over_images=True`) is better: `PSNR 33.31 -> 33.75`.
2. make use of scheduler (`MultiStepLR(optimizer, milestones=[20000, 30000], gamma=0.1)`) to adjust learning rate gives: `PSNR 33.75 -> 34.40`.
3. increasing chunk size (`chunk: 8192 -> 81920`) during inference gives speedup: `FPS 4.x -> 6.2`
4. random bkgd color (`color_bkgd_aug="random"`) for the `Lego` scene actually hurts: `PNSR 35.42 -> 34.38`
 -->

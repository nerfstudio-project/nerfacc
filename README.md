# nerfacc

This is a **tiny** tootlbox  for **accelerating** NeRF training & rendering using PyTorch CUDA extensions. Plug-and-play for most of the NeRFs!

## Instant-NGP example

```
python examples/trainval.py
```

## Performance Reference

Ours on TITAN RTX :

| trainval | Lego | Mic | Materials |
| - | - | - | - |
| Time | 300s  | 272s  | 258s  |
| PSNR | 36.28 | 36.16 | 29.76 |
| FPS  | 11.49 | 21.48 | 8.86  |

Instant-NGP paper (5 min) on 3090:

| trainval | Lego | Mic | Materials |
| - | - | - | - |
| PSNR | 36.39 | 36.22 | 29.78 |


<!-- 

Tested with the default settings on the Lego test set.

| Model | Split | PSNR | Train Time | Test Speed | GPU | Train Memory |
| - | - | - | - | - | - | - |
| instant-ngp (paper)            | trainval?            | 36.39  |  -   | -    | 3090    |
| instant-ngp (code)             | train (35k steps)    | 36.08  |  308 sec  | 55.32 fps  | TITAN RTX  |  1734MB |
| instant-ngp (code) w/o rng bkgd| train (35k steps)    | 34.17  |  -  | -  | - |  - |
| torch-ngp (`-O`)               | train (30K steps)    | 34.15  |  310 sec  | 7.8 fps    | V100 |
| ours                           | trainval (35K steps) | 36.22  |  378 sec  | 12.08 fps    | TITAN RTX  | -->

## Tips:

1. sample rays over all images per iteration (`batch_over_images=True`) is better: `PSNR 33.31 -> 33.75`.
2. make use of scheduler (`MultiStepLR(optimizer, milestones=[20000, 30000], gamma=0.1)`) to adjust learning rate gives: `PSNR 33.75 -> 34.40`.
3. increasing chunk size (`chunk: 8192 -> 81920`) during inference gives speedup: `FPS 4.x -> 6.2`
4. random bkgd color (`color_bkgd_aug="random"`) for the `Lego` scene actually hurts: `PNSR 35.42 -> 34.38`


# nerfacc

This is a **tiny** tootlbox  for **accelerating** NeRF training & rendering using PyTorch CUDA extensions. Plug-and-play for most of the NeRFs!

## Instant-NGP example

```
python examples/trainval.py
```

## Performance Reference

Tested with the default settings on the Lego test set.

| Model | Split | PSNR | Train Time | Test Speed | GPU |
| - | - | - | - | - | - |
| instant-ngp (paper)            | trainval?            | 36.39  |  -   | -    | 3090    |
| torch-ngp (`-O`)               | train (30K steps)    | 34.15  |  310 sec  | 7.8 fps  | V100 |
| ours                           | train (30K steps)    | 34.40  |  296 sec  | 6.2 fps | TITAN RTX  |
| ours                           | trainval (30K steps)    | 35.42  |  291 sec  | 6.4 fps | TITAN RTX  |

## Tips:

1. sample rays over all images per iteration (`batch_over_images=True`) is better: `PSNR 33.31 -> 33.75`.
2. make use of scheduler (`MultiStepLR(optimizer, milestones=[20000, 30000], gamma=0.1)`) to adjust learning rate gives: `PSNR 33.75 -> 34.40`.
3. increasing chunk size (`chunk: 8192 -> 81920`) during inference gives speedup: `FPS 4.x -> 6.2`
4. random bkgd color (`color_bkgd_aug="random"`) for the `Lego` scene actually hurts: `PNSR 35.42 -> 34.38`


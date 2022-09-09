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
| ours                           | train (30K steps)    | 33.31  |  298 sec  | 6.4 fps | TITAN RTX  |
| ours                           | trainval (30K steps)    | 34.45  |  290 sec  | 6.6 fps | TITAN RTX  |
# nerfacc

This is a **tiny** tootlbox for **accelerating** NeRF training & rendering using PyTorch CUDA extensions. Plug-and-play for most of the NeRFs!

## Instant-NGP example

```
python examples/trainval.py
```

## Performance Reference

Tested with the default settings on the Lego dataset.
Here the speed refers to the `iterations per second`.

| Model | Split | PSNR | Train Speed | Test Speed | GPU |
| - | - | - | - | - | - |
| instant-ngp (paper)            | trainval?            | 36.39  |  -   | -    | 3090    |
| torch-ngp (`-O`)               | train (30K steps)    | 34.15  |  97  | 7.8  | V100 |
| ours                           | train (30K steps)    | 34.26  |  96  | ?    | TITAN RTX  |
.. _`Instant-NGP Example`:

Instant-NGP
====================

See code `examples/train_ngp_nerf.py` at our `github repository`_ for details.

Benchmarks
------------
*updated on 2022-10-08*

Here we trained a `Instant-NGP Nerf`_ model on the `Nerf-Synthetic dataset`_. We follow the same
settings with the Instant-NGP paper, which uses trainval split for training and test split for
evaluation. Our experiments are conducted on a single NVIDIA TITAN RTX GPU. The training
memory footprint is about 3GB.

.. note::
    
    The Instant-NGP paper makes use of the alpha channel in the images to apply random background
    augmentation during training. Yet we only uses RGB values with a constant white background.

+----------------------+-------+-------+---------+-------+-------+-------+-------+-------+-------+
| PSNR                 | Lego  | Mic   |Materials| Chair |Hotdog | Ficus | Drums | Ship  | MEAN  |
|                      |       |       |         |       |       |       |       |       |       |
+======================+=======+=======+=========+=======+=======+=======+=======+=======+=======+
| Instant-NGP (5min)   | 36.39 | 36.22 | 29.78   | 35.00 | 37.40 | 33.51 | 26.02 | 31.10 | 33.18 |
+----------------------+-------+-------+---------+-------+-------+-------+-------+-------+-------+
| Ours  (~4.5min)      | 36.82 | 37.61 | 30.18   | 36.13 | 38.11 | 34.48 | 26.62 | 31.37 | 33.92 |
+----------------------+-------+-------+---------+-------+-------+-------+-------+-------+-------+
| Ours  (Training time)| 288s  | 259s  | 256s    | 324s  | 288s  | 245s  | 262s  | 257s  | 272s  |
+----------------------+-------+-------+---------+-------+-------+-------+-------+-------+-------+

.. _`Instant-NGP Nerf`: https://arxiv.org/abs/2201.05989
.. _`github repository`: https://github.com/KAIR-BAIR/nerfacc/tree/5637cc9a1565b2685c02250eb1ee1c53d3b07af1
.. _`Nerf-Synthetic dataset`: https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi

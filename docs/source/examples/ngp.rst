.. _`Instant-NGP Example`:

Instant-NGP
====================

See code `examples/train_ngp_nerf.py` at our `github repository`_ for details.

Benchmarks
------------
*updated on 2022-10-12*

Here we trained a `Instant-NGP Nerf`_ model on the `Nerf-Synthetic dataset`_. We follow the same
settings with the Instant-NGP paper, which uses train split for training and test split for
evaluation. All experiments are conducted on a single NVIDIA TITAN RTX GPU. The training
memory footprint is about 3GB.

.. note::
    
    The Instant-NGP paper makes use of the alpha channel in the images to apply random background
    augmentation during training. For fair comparision, we rerun their code with a constant white
    background during both training and testing. Also it is worth to mention that we didn't strictly
    follow the training receipe in the Instant-NGP paper, such as the learning rate schedule etc, as
    the purpose of this benchmark is to showcase instead of reproducing the paper.

+-----------------------+-------+-------+---------+-------+-------+-------+-------+-------+-------+
| PSNR                  | Lego  | Mic   |Materials| Chair |Hotdog | Ficus | Drums | Ship  | MEAN  |
|                       |       |       |         |       |       |       |       |       |       |
+=======================+=======+=======+=========+=======+=======+=======+=======+=======+=======+
|Instant-NGP 35k steps  | 35.87 | 36.22 | 29.08   | 35.10 | 37.48 | 30.61 | 23.85 | 30.62 | 32.35 |
+-----------------------+-------+-------+---------+-------+-------+-------+-------+-------+-------+
|(training time)        | 309s  | 258s  | 256s    | 316s  | 292s  | 207s  | 218s  | 250s  | 263s  |
+-----------------------+-------+-------+---------+-------+-------+-------+-------+-------+-------+
|Ours 20k steps         | 35.50 | 36.16 | 29.14   | 35.23 | 37.15 | 31.71 | 24.88 | 29.91 | 32.46 |
+-----------------------+-------+-------+---------+-------+-------+-------+-------+-------+-------+
|(training time)        | 287s  | 274s  | 269s    | 317s  | 269s  | 244s  | 249s  | 257s  | 271s  |
+-----------------------+-------+-------+---------+-------+-------+-------+-------+-------+-------+

.. _`Instant-NGP Nerf`: https://github.com/NVlabs/instant-ngp/tree/51e4107edf48338e9ab0316d56a222e0adf87143
.. _`github repository`: https://github.com/KAIR-BAIR/nerfacc/tree/76c0f9817da4c9c8b5ccf827eb069ee2ce854b75
.. _`Nerf-Synthetic dataset`: https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi

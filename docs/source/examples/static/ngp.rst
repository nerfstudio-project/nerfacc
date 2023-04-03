.. _`Instant-NGP Example`:

Instant-NGP
====================

See code `examples/train_ngp_nerf_occ.py` and `examples/train_ngp_nerf_prop.py` at our 
`github repository`_ for details.


Radiance Field
--------------
We follow the `Instant-NGP`_ paper to implement the radiance field (`examples/radiance_fields/ngp.py`),
and aligns the hyperparameters (e.g., hashencoder, mlp) with the paper. It is build on top of the
`tiny-cuda-nn`_ library.


Benchmark: Nerf-Synthetic Dataset
---------------------------------
*updated on 2023-04-04 with nerfacc==0.5.0*

Our experiments are conducted on a single NVIDIA TITAN RTX GPU. 
The training memory footprint is about 3GB.

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
| training time         | 309s  | 258s  | 256s    | 316s  | 292s  | 207s  | 218s  | 250s  | 263s  |
+-----------------------+-------+-------+---------+-------+-------+-------+-------+-------+-------+
|Ours (occ) 20k steps   | 35.67 | 36.85 | 29.60   | 35.71 | 37.37 | 33.95 | 25.44 | 30.29 | 33.11 |
+-----------------------+-------+-------+---------+-------+-------+-------+-------+-------+-------+
| training time         | 288s  | 260s  | 253s    | 326s  | 272s  | 249s  | 252s  | 251s  | 269s  |
+-----------------------+-------+-------+---------+-------+-------+-------+-------+-------+-------+
|Ours (prop) 20k steps  | 34.04 | 34.56 | 28.76   | 34.21 | 36.44 | 31.41 | 24.81 | 29.85 | 31.76 |
+-----------------------+-------+-------+---------+-------+-------+-------+-------+-------+-------+
| training time         | 225s  | 235s  | 235s    | 240s  | 239s  | 242s  | 258s  | 247s  | 240s  |
+-----------------------+-------+-------+---------+-------+-------+-------+-------+-------+-------+


Benchmark: Mip-NeRF 360 Dataset
---------------------------------
*updated on 2023-04-04 with nerfacc==0.5.0*

Our experiments are conducted on a single NVIDIA TITAN RTX GPU.

.. note:: 
    `Ours (prop)` combines the proposal network (:class:`nerfacc.PropNetEstimator`) with the 
    Instant-NGP radiance field. This is exactly what the `Nerfacto`_ model is doing in the
    `nerfstudio`_ project. In fact, the hyperparameters for `Ours (prop)` in this experiment
    are aligned with the `Nerfacto`_ model.

+-----------------------+-------+-------+---------+-------+-------+-------+-------+-------+
| PSNR                  |Bicycle| Garden|   Stump | Bonsai|Counter|Kitchen| Room  | MEAN  |
|                       |       |       |         |       |       |       |       |       |
+=======================+=======+=======+=========+=======+=======+=======+=======+=======+
|NeRF++ (~days)         | 22.64 | 24.32 | 23.34   | 29.15 | 26.38 | 27.80 | 28.87 | 26.21 |
+-----------------------+-------+-------+---------+-------+-------+-------+-------+-------+
|Mip-NeRF 360 (~days)   | 24.37 | 26.98 | 26.40   | 33.46 | 29.55 | 32.23 | 31.63 | 29.23 |
+-----------------------+-------+-------+---------+-------+-------+-------+-------+-------+
|Instant-NGP 35k steps  | 22.40 | 24.86 | 23.17   | 24.41 | 27.38 | 29.07 | 30.24 | 25.93 |
+-----------------------+-------+-------+---------+-------+-------+-------+-------+-------+
| training time         | 301s  | 339s  | 295s    | 279s  | 339s  | 366s  | 317s  | 319s  |
+-----------------------+-------+-------+---------+-------+-------+-------+-------+-------+
|Ours (occ) 20k steps   | 22.40 | 23.94 | 22.98   | 30.09 | 26.84 | 28.03 | 30.60 | 26.41 |
+-----------------------+-------+-------+---------+-------+-------+-------+-------+-------+
| training time         | 277s  | 302s  | 299s    | 278s  | 315s  | 331s  | 301s  | 300s  |
+-----------------------+-------+-------+---------+-------+-------+-------+-------+-------+
|Ours (prop) 20k steps  | 23.23 | 25.42 | 25.24   | 30.71 | 26.74 | 30.70 | 30.99 | 27.58 |
+-----------------------+-------+-------+---------+-------+-------+-------+-------+-------+
| training time         | 276s  | 293s  | 291s    | 291s  | 291s  | 295s  | 287s  | 289s  |
+-----------------------+-------+-------+---------+-------+-------+-------+-------+-------+


.. _`github repository`: https://github.com/KAIR-BAIR/nerfacc/
.. _`Instant-NGP`: https://arxiv.org/abs/2201.05989
.. _`tiny-cuda-nn`: https://github.com/NVlabs/tiny-cuda-nn
.. _`Nerfacto`: https://docs.nerf.studio/en/latest/nerfology/methods/nerfacto.html
.. _`nerfstudio`: https://docs.nerf.studio/en/latest/
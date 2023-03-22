Unbounded Scene
====================

See code `examples/train_ngp_nerf.py` at our `github repository`_ for details.

Benchmarks
------------
*updated on 2023-03-21*

Here we trained a `Instant-NGP Nerf`_  on the `MipNerf360`_ dataset. We used train 
split for training and test split for evaluation. Our experiments are conducted on a 
single NVIDIA TITAN RTX GPU. The training memory footprint is about 6-9GB.

The main difference between working with unbounded scenes and bounded scenes, is that
a contraction method is needed to map the infinite space to a finite :ref:`Occupancy Grid`.
We have difference options provided for this (see :ref:`Occupancy Grid`). The experiments
here is basically the Instant-NGP experiments (see :ref:`Instant-NGP Example`) with a contraction method
that takes from `MipNerf360`_.

.. note:: 
    Even though we are comparing with `Nerf++`_ and `MipNerf360`_, the model and everything are
    totally different with them. There are plenty of ideas from those papers that would be very
    helpful for the performance, but we didn't adopt them. As this is just a simple example to 
    show how to use the library, we didn't want to make it too complicated.


+----------------------+-------+-------+-------+-------+-------+-------+-------+-------+
| PSNR                 |Garden |Bicycle|Bonsai |Counter|Kitchen| Room  | Stump | MEAN  |
|                      |       |       |       |       |       |       |       |       |
+======================+=======+=======+=======+=======+=======+=======+=======+=======+
| Nerf++ (~days)       | 24.32 | 22.64 | 29.15 | 26.38 | 27.80 | 28.87 | 24.34 | 26.21 |
+----------------------+-------+-------+-------+-------+-------+-------+-------+-------+
| MipNerf360 (~days)   | 26.98 | 24.37 | 33.46 | 29.55 | 32.23 | 31.63 | 26.40 | 29.23 |
+----------------------+-------+-------+-------+-------+-------+-------+-------+-------+
| Ours (occ)           | 24.76 | 22.38 | 29.72 | 26.80 | 28.02 | 30.67 | 22.39 | 26.39 |
+----------------------+-------+-------+-------+-------+-------+-------+-------+-------+
| Ours (Training time) | 323s  | 302s  | 300s  | 337s  | 347s  | 320s  | 322s  | 322s  |
+----------------------+-------+-------+-------+-------+-------+-------+-------+-------+
| Ours (prop)          | 25.43 | 23.21 | 30.70 | 26.74 | 30.72 | 31.00 | 25.18 | 27.57 |
+----------------------+-------+-------+-------+-------+-------+-------+-------+-------+
| Ours (Training time) | 295s  | 299s  | 296s  | 298s  | 305s  | 295s  | 296s  | 298s  |
+----------------------+-------+-------+-------+-------+-------+-------+-------+-------+

Note `Ours (prop)` is basically a `Nerfacto_` model.

.. _`Instant-NGP Nerf`: https://arxiv.org/abs/2201.05989
.. _`MipNerf360`: https://arxiv.org/abs/2111.12077
.. _`Nerf++`: https://arxiv.org/abs/2010.07492
.. _`github repository`: https://github.com/KAIR-BAIR/nerfacc/
.. _`Nerfacto`: https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/models/nerfacto.py
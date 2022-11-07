Unbounded Scene
====================

See code `examples/train_ngp_nerf.py` at our `github repository`_ for details.

Benchmarks
------------
*updated on 2022-11-07*

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
| Ours (~20 mins)      | 25.41 | 22.97 | 30.71 | 27.34 | 30.32 | 31.00 | 23.43 | 27.31 |
+----------------------+-------+-------+-------+-------+-------+-------+-------+-------+
| Ours (Training time) | 25min | 17min | 19min | 23min | 28min | 20min | 17min | 21min |
+----------------------+-------+-------+-------+-------+-------+-------+-------+-------+

.. _`Instant-NGP Nerf`: https://arxiv.org/abs/2201.05989
.. _`MipNerf360`: https://arxiv.org/abs/2111.12077
.. _`Nerf++`: https://arxiv.org/abs/2010.07492
.. _`github repository`: https://github.com/KAIR-BAIR/nerfacc/tree/76c0f9817da4c9c8b5ccf827eb069ee2ce854b75

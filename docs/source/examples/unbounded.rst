Unbounded Scene
====================

See code `examples/train_ngp_nerf.py` at our `github repository`_ for details.

Benchmarks
------------

Here we trained a `Instant-NGP Nerf`_  on the `MipNerf360`_ dataset. We used train 
split for training and test split for evaluation. Our experiments are conducted on a 
single NVIDIA TITAN RTX GPU. 

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


+----------------------+-------+-------+------------+-------+--------+--------+--------+
|                      |Garden |Bicycle| Bonsai     |Counter|Kitchen | Room   | Stump  |
|                      |       |       |            |       |        |        |        |
+======================+=======+=======+============+=======+========+========+========+
|Nerf++(PSNR:~days)    | 24.32 | 22.64 | 29.15      | 26.38 | 27.80  | 28.87  | 24.34  |
+----------------------+-------+-------+------------+-------+--------+--------+--------+
|MipNerf360(PSNR:~days)| 26.98 | 24.37 | 33.46      | 29.55 | 32.23  | 31.63  | 28.65  |
+----------------------+-------+-------+------------+-------+--------+--------+--------+
| Ours  (PSNR:~1hr)    | 25.41 | 22.89 | 27.35      | 23.15 | 27.74  | 30.66  | 21.83  |
+----------------------+-------+-------+------------+-------+--------+--------+--------+
| Ours  (Training time)| 40min | 35min | 47min      | 39min | 60min  | 41min  | 28min  |
+----------------------+-------+-------+------------+-------+--------+--------+--------+

.. _`Instant-NGP Nerf`: https://arxiv.org/abs/2103.13497
.. _`MipNerf360`: https://arxiv.org/abs/2111.12077
.. _`Nerf++`: https://arxiv.org/abs/2010.07492
.. _`github repository`: https://github.com/KAIR-BAIR/nerfacc/

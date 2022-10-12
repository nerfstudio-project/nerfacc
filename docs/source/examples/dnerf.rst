Dynamic Scene
====================

See code `examples/train_mlp_dnerf.py` at our `github repository`_ for details.

Benchmarks
------------
*updated on 2022-10-08*

Here we trained a 8-layer-MLP for the radiance field and a 4-layer-MLP for the warping field,
(similar to the T-Nerf model in the `D-Nerf`_ paper) on the `D-Nerf dataset`_. We used train 
split for training and test split for evaluation. Our experiments are conducted on a 
single NVIDIA TITAN RTX GPU. The training memory footprint is about 11GB.

.. note::

    The :ref:`Occupancy Grid` used in this example is shared by all the frames. In other words, 
    instead of using it to indicate the opacity of an area at a single timestamp, 
    Here we use it to indicate the `maximum` opacity at this area `over all the timestamps`.
    It is not optimal but still makes the rendering very efficient.

+----------------------+----------+---------+-------+---------+-------+--------+---------+-------+-------+
| PSNR                 | bouncing | hell    | hook  | jumping | lego  | mutant | standup | trex  | MEAN  |
|                      | balls    | warrior |       | jacks   |       |        |         |       |       |
+======================+==========+=========+=======+=========+=======+========+=========+=======+=======+
| D-Nerf (~ days)      | 32.80    | 25.02   | 29.25 | 32.80   | 21.64 | 31.29  | 32.79   | 31.75 | 29.67 |
+----------------------+----------+---------+-------+---------+-------+--------+---------+-------+-------+
| Ours  (~ 1 hr)       | 39.49    | 25.58   | 31.86 | 32.73   | 24.32 | 35.55  | 35.90   | 32.33 | 32.22 |
+----------------------+----------+---------+-------+---------+-------+--------+---------+-------+-------+
| Ours  (Training time)| 37min    | 52min   | 69min | 64min   | 44min | 79min  | 79min   | 39min | 58min |
+----------------------+----------+---------+-------+---------+-------+--------+---------+-------+-------+

.. _`D-Nerf`: https://arxiv.org/abs/2011.13961
.. _`D-Nerf dataset`: https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=0
.. _`github repository`: https://github.com/KAIR-BAIR/nerfacc/tree/76c0f9817da4c9c8b5ccf827eb069ee2ce854b75


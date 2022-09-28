Dynamic Scene
====================

See code `examples/train_mlp_dnerf.py` at our `github repository`_ for details.

Benchmarks
------------

Here we trained a 8-layer-MLP for the radiance field and a 4-layer-MLP for the warping field,
(similar to the T-Nerf model in the `D-Nerf`_ paper) on the `D-Nerf dataset`_. We used train 
split for training and test split for evaluation. Our experiments are conducted on a 
single NVIDIA TITAN RTX GPU. 

.. note::

    The :ref:`Occupancy Grid` used in this example is shared by all the frames. In other words, 
    instead of using it to indicate the opacity of an area at a single timestamp, 
    Here we use it to indicate the `maximum` opacity at this area `over all the timestamps`.
    It is not optimal but still makes the rendering very efficient.

+----------------------+----------+---------+-------+---------+-------+--------+---------+-------+-------+
|                      | bouncing | hell    | hook  | jumping | lego  | mutant | standup | trex  | AVG   |
|                      | balls    | warrior |       | jacks   |       |        |         |       |       |
+======================+==========+=========+=======+=========+=======+========+=========+=======+=======+
| D-Nerf (PSNR: ~2day) | 38.93    | 25.02   | 29.25 | 32.80   | 21.64 | 31.29  | 32.79   | 31.75 | 30.43 |
+----------------------+----------+---------+-------+---------+-------+--------+---------+-------+-------+
| Ours  (PSNR: ~50min) | 39.60    | 22.41   | 30.64 | 29.79   | 24.75 | 35.20  | 34.50   | 31.83 | 31.09 |
+----------------------+----------+---------+-------+---------+-------+--------+---------+-------+-------+
| Ours  (Training time)| 45min    | 49min   | 51min | 46min   | 53min | 57min  | 49min   | 46min | 50min |
+----------------------+----------+---------+-------+---------+-------+--------+---------+-------+-------+

.. _`D-Nerf`: https://arxiv.org/abs/2104.00677
.. _`D-Nerf dataset`: https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=0
.. _`github repository`: https://github.com/KAIR-BAIR/nerfacc/


Vanilla Nerf 
====================

See code `examples/train_mlp_nerf.py` at our `github repository`_ for details.

Benchmarks
------------
*updated on 2022-10-08*

Here we trained a 8-layer-MLP for the radiance field as in the `vanilla Nerf`_. We used the 
train split for training and test split for evaluation as in the Nerf paper. Our experiments are 
conducted on a single NVIDIA TITAN RTX GPU. The training memory footprint is about 10GB.

.. note:: 
    The vanilla Nerf paper uses two MLPs for course-to-fine sampling. Instead here we only use a 
    single MLP with more samples (1024). Both ways share the same spirit to do dense sampling 
    around the surface. Our fast rendering inheritly skip samples away from the surface 
    so we can simplly increase the number of samples with a single MLP, to achieve the same goal 
    with the coarse-to-fine sampling, without runtime or memory issue.

+----------------------+-------+-------+---------+-------+-------+-------+-------+-------+-------+
| PSNR                 | Lego  | Mic   |Materials| Chair |Hotdog | Ficus | Drums | Ship  | MEAN  |
|                      |       |       |         |       |       |       |       |       |       |
+======================+=======+=======+=========+=======+=======+=======+=======+=======+=======+
| NeRF  (~ days)       | 32.54 | 32.91 | 29.62   | 33.00 | 36.18 | 30.13 | 25.01 | 28.65 | 31.00 |
+----------------------+-------+-------+---------+-------+-------+-------+-------+-------+-------+
| Ours  (~ 50min)      | 33.69 | 33.76 | 29.73   | 33.32 | 35.80 | 32.52 | 25.39 | 28.18 | 31.55 |
+----------------------+-------+-------+---------+-------+-------+-------+-------+-------+-------+
| Ours  (Training time)| 58min | 53min | 46min   | 62min | 56min | 42min | 52min | 49min | 52min |
+----------------------+-------+-------+---------+-------+-------+-------+-------+-------+-------+

.. _`github repository`: : https://github.com/KAIR-BAIR/nerfacc/tree/76c0f9817da4c9c8b5ccf827eb069ee2ce854b75
.. _`vanilla Nerf`: https://arxiv.org/abs/2003.08934

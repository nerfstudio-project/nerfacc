.. _`T-NeRF Example`:

T-NeRF 
====================
See code `examples/train_mlp_dnerf.py` at our `github repository`_ for details.

Radiance Field
--------------
Here we implement a very basic time-conditioned NeRF (T-NeRF) model (`examples/radiance_fields/mlp.py`)
for dynamic scene reconstruction.
The implementation is mostly follow the T-NeRF described in the `D-NeRF`_ paper, with a 8-layer-MLP 
for the radiance field and a 4-layer-MLP for the warping field. The only major difference is that
we reduce the max frequency of the positional encoding from 10 to 4, to respect the fact that the
motion of the object is relatively smooth.


Benchmarks: D-NeRF Dataset
---------------------------
*updated on 2022-10-08*

Our experiments are conducted on a single NVIDIA TITAN RTX GPU. 
The training memory footprint is about 11GB.

+----------------------+----------+---------+-------+---------+-------+--------+---------+-------+-------+
| PSNR                 | bouncing | hell    | hook  | jumping | lego  | mutant | standup | trex  | MEAN  |
|                      | balls    | warrior |       | jacks   |       |        |         |       |       |
+======================+==========+=========+=======+=========+=======+========+=========+=======+=======+
| D-NeRF (~ days)      | 32.80    | 25.02   | 29.25 | 32.80   | 21.64 | 31.29  | 32.79   | 31.75 | 29.67 |
+----------------------+----------+---------+-------+---------+-------+--------+---------+-------+-------+
| Ours  (~ 1 hr)       | 39.49    | 25.58   | 31.86 | 32.73   | 24.32 | 35.55  | 35.90   | 32.33 | 32.22 |
+----------------------+----------+---------+-------+---------+-------+--------+---------+-------+-------+

.. _`D-NeRF`: https://arxiv.org/abs/2011.13961
.. _`github repository`: https://github.com/KAIR-BAIR/nerfacc/


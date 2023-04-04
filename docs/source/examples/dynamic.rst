Dynamic NeRFs
===================================


The :class:`nerfacc.PropNetEstimator` can natually work with dynamic NeRFs. To make the 
:class:`nerfacc.OccGridEstimator` also work with dynamic NeRFs, we need to make some compromises.
In these examples, we use the :class:`nerfacc.OccGridEstimator` to estimate the
`maximum` opacity at each area `over all the timestamps`. This allows us to share the same estimator
across all the timestamps, including those timestamps that are not in the training set. 
In other words, we use it to cache the union of the occupancy at all timestamps.
It is not optimal but still makes the rendering very efficient if the motion is not crazyly significant.


Performance Overview
--------------------
*updated on 2023-04-04*

+----------------------+-----------+----------------------------------+-----------------------+--------------------------+
| Methods              | Dataset   | Training Time :math:`\downarrow` | PSNR :math:`\uparrow` | LPIPS :math:`\downarrow` |
+======================+===========+==================================+=======================+==========================+
| TiNeuVox `[1]`_      | HyperNeRF | 56.3min                          | 24.19                 | 0.425                    |
+----------------------+-----------+----------------------------------+-----------------------+--------------------------+
| *+nerfacc (occgrid)* |           | 33.0min                          | 24.19                 | 0.434                    |
+----------------------+-----------+----------------------------------+-----------------------+--------------------------+
| *+nerfacc (propnet)* |           | 34.3min                          | 24.26                 | 0.398                    |
+----------------------+-----------+----------------------------------+-----------------------+--------------------------+
| TiNeuVox `[1]`_      | D-NeRF    | 11.8min                          | 31.14                 | 0.050                    |
+----------------------+-----------+----------------------------------+-----------------------+--------------------------+
| *+nerfacc (occgrid)* |           | 4.2min                           | 31.75                 | 0.038                    |
+----------------------+-----------+----------------------------------+-----------------------+--------------------------+
| K-Planes `[2]`_      | D-NeRF    | 63.9min                          | 30.28                 | 0.043                    |
+----------------------+-----------+----------------------------------+-----------------------+--------------------------+
| *+nerfacc (occgrid)* |           | 38.8min                          | 30.35                 | 0.042                    |
+----------------------+-----------+----------------------------------+-----------------------+--------------------------+
| T-NeRF `[3]`_        | D-NeRF    | 20hours                          | 28.78                 | 0.069                    |
+----------------------+-----------+----------------------------------+-----------------------+--------------------------+
| *+nerfacc (occgrid)* |           | 58min                            | 32.22                 | 0.040                    |
+----------------------+-----------+----------------------------------+-----------------------+--------------------------+

Implementation Details
----------------------

.. toctree::
   :glob:
   :maxdepth: 1

   dynamic/*

|

3rd-Party Use Cases
-------------------

- `Representing Volumetric Videos as Dynamic MLP Maps, CVPR 2023 <https://github.com/zju3dv/mlp_maps>`_.

.. _`[1]`: https://arxiv.org/abs/2205.15285
.. _`[2]`: https://arxiv.org/abs/2301.10241
.. _`[3]`: https://arxiv.org/abs/2011.13961
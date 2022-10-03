NerfAcc Documentation
===================================

NerfAcc is a PyTorch Nerf acceleration toolbox for both training and inference. 

Using NerfAcc, 

- The `vanilla Nerf`_ model with 8-layer MLPs can be trained to *better quality* (+~0.5 PNSR) \
  in *1 hour* rather than *1~2 days* as in the paper.
- The `Instant-NGP Nerf`_ model can be trained to *equal quality* with *9/10th* of the training time (4.5 minutes) \
  comparing to the official pure-CUDA implementation.
- The `D-Nerf`_ model for *dynamic* objects can also be trained in *1 hour* \
  rather than *2 days* as in the paper, and with *better quality* (+~0.5 PSNR).
- Both the *bounded* and *unbounded* scenes are supported.

*And it is pure python interface with flexible apis!*

.. note::

   This repo is focusing on the single scene situation. Generalizable Nerfs across \
   multiple scenes is currently out of the scope of this repo. But you may still find
   some useful tricks in this repo. :)

Installation:
-------------

.. code-block:: console

   $ pip install nerfacc

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Python API

   apis/*

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Example Usages

   examples/*

.. toctree::
   :maxdepth: 1
   :caption: Projects

   nerfstudio <https://docs.nerf.studio/>


.. _`vanilla Nerf`: https://arxiv.org/abs/2003.08934
.. _`Instant-NGP Nerf`: https://arxiv.org/abs/2103.13497
.. _`D-Nerf`: https://arxiv.org/abs/2104.00677
.. _`MipNerf360`: https://arxiv.org/abs/2111.12077
.. _`pixel-Nerf`: https://arxiv.org/abs/2012.02190
NerfAcc Documentation
===================================

NerfAcc is a PyTorch Nerf acceleration toolbox for both training and inference. It focus on
efficient volumetric rendering of radiance fields, which is universal for most of the NeRFs.

Using NerfAcc, 

- The `vanilla Nerf`_ model with 8-layer MLPs can be trained to *better quality* (+~0.5 PNSR) \
  in *1 hour* rather than *1~2 days* as in the paper.
- The `Instant-NGP Nerf`_ model can be trained to *better quality* (+~0.7 PSNR) with *9/10th* of \
  the training time (4.5 minutes) comparing to the official pure-CUDA implementation.
- The `D-Nerf`_ model for *dynamic* objects can also be trained in *1 hour* \
  rather than *2 days* as in the paper, and with *better quality* (+~0.5 PSNR).
- The *unbounded* scenes can be trained to *better quality* (+~1.1 PSNR) than `Nerf++`_ in *20 minutes*.

**And it is pure Python interface with flexible APIs!**

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
.. _`Instant-NGP Nerf`: https://arxiv.org/abs/2201.05989
.. _`D-Nerf`: https://arxiv.org/abs/2011.13961
.. _`MipNerf360`: https://arxiv.org/abs/2111.12077
.. _`pixel-Nerf`: https://arxiv.org/abs/2012.02190
.. _`Nerf++`: https://arxiv.org/abs/2010.07492

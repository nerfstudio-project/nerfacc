NerfAcc Documentation
===================================

NerfAcc is a PyTorch Nerf acceleration toolbox for both training and inference. 

Using NerfAcc, 

- The `vanilla Nerf model`_ with 8-layer MLPs can be trained to *better quality* (+~1.0 PNSR) \
  in *45 minutes* rather than *1~2 days* as in the paper.
- The `instant-ngp Nerf model`_ can be trained to *better quality* (+~1.0 PNSR) \
  in *5 minutes* compare to the paper.
- The `D-Nerf model`_ for *dynamic* objects can also be trained in *45 minutes* \
  rather than *2 days* as in the paper, and with *better quality* (+~2.0 PSNR).
- *Unbounded scenes* from `MipNerf360`_ can also be trained in \
  *~1 hour* and get comparable quality to the paper.

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

   NeRFactory <https://plenoptix-nerfactory.readthedocs-hosted.com/>


.. _`vanilla NeRF model`: https://arxiv.org/abs/2003.08934
.. _`instant-ngp NeRF model`: https://arxiv.org/abs/2103.13497
.. _`D-Nerf model`: https://arxiv.org/abs/2104.00677
.. _`MipNerf360`: https://arxiv.org/abs/2111.12077
.. _`pixel-Nerf`: https://arxiv.org/abs/2012.02190
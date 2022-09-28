Instant-NGP
====================

See code `examples/train_ngp_nerf.py` at our `github repository`_ for details.

Benchmarks
------------

Here we trained a NGP Nerf model on the NeRF-Synthetic dataset. We follow the same
settings with the paper, which uses trainval split for training and test split for
evaluation. Our experiments are conducted on a single NVIDIA TITAN RTX GPU.

.. note::
    
    The paper makes use of the alpha channel in the images to apply random background
    augmentation during training. Yet we only uses a constant white background.

+----------------------+----------+----------+------------+-------+--------+--------+--------+--------+
|                      | Lego     | Mic      | Materials  |Chair  |Hotdog  | Ficus  | Drums  | Ship   |
|                      |          |          |            |       |        |        |        |        |
+======================+==========+==========+============+=======+========+========+========+========+
| Paper (PSNR: 5min)   | 36.39    | 36.22    | 29.78      | 35.00 | 37.40  | 33.51  | 26.02  | 31.10  |
+----------------------+----------+----------+------------+-------+--------+--------+--------+--------+
| Ours  (PSNR)         | 36.61    | 37.45    | 30.15      | 36.10 | 38.17  |        | 25.83  |        |
+----------------------+----------+----------+------------+-------+--------+--------+--------+--------+
| Ours  (Training time)| 300s     | 272s     | 258s       | 311s  | 287s   |        | 249s   |        |
+----------------------+----------+----------+------------+-------+--------+--------+--------+--------+

.. _`github repository`: : https://github.com/KAIR-BAIR/nerfacc/
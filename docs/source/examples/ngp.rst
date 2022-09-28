Instant-NGP
====================

See code `examples/train_ngp_nerf.py` at our `github repository`_ for details.

Benchmarks
------------

Here we trained a NGP Nerf model on the NeRF-Synthetic dataset. We follow the same
settings with the paper, which uses trainval split for training and test split for
evaluation. Our experiments are conducted on a single NVIDIA TITAN RTX GPU. The training
memory footprint is about 3GB.

.. note::
    
    The paper makes use of the alpha channel in the images to apply random background
    augmentation during training. Yet we only uses RGB values with a constant white background.

+----------------------+-------+-------+------------+-------+--------+--------+--------+--------+--------+
|                      | Lego  | Mic   | Materials  |Chair  |Hotdog  | Ficus  | Drums  | Ship   | AVG    |
|                      |       |       |            |       |        |        |        |        |        |
+======================+=======+=======+============+=======+========+========+========+========+========+
| Paper (PSNR: 5min)   | 36.39 | 36.22 | 29.78      | 35.00 | 37.40  | 33.51  | 26.02  | 31.10  | 33.18  |
+----------------------+-------+-------+------------+-------+--------+--------+--------+--------+--------+
| Ours  (PSNR:4.5min)  | 36.71 | 36.78 | 29.06      | 36.10 | 37.88  | 32.07  | 25.83  | 31.39  | 33.23  |
+----------------------+-------+-------+------------+-------+--------+--------+--------+--------+--------+
| Ours  (Training time)| 286s  | 251s  | 250s       | 311s  | 275s   | 254s   | 249s   | 255s   | 266s   |
+----------------------+-------+-------+------------+-------+--------+--------+--------+--------+--------+

.. _`github repository`: : https://github.com/KAIR-BAIR/nerfacc/
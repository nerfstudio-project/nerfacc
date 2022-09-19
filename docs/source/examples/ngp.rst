Instant-NGP
====================

See code at our github repository: https://github.com/KAIR-BAIR/nerfacc/tree/master/examples/

Benchmarks
------------

We trained on NeRF-Synthetic trainval set using TITAN RTX, and evaluated on testset.
Note Instant-NGP's results are taken from the paper, which is trained on a Nvidia 3090,
with random background trick using alpha channel.

+----------------------+----------+----------+------------+-------+--------+
|                      | Lego     | Mic      | Materials  |Chair  |Hotdog  |
|                      |          |          |            |       |        |
+======================+==========+==========+============+=======+========+
| Paper (PSNR: 5min)   | 36.39    | 36.22    |  29.78     |  35.00| 37.40  |
+----------------------+----------+----------+------------+-------+--------+
| Ours  (PSNR)         | 36.61    | 37.45    | 30.15      | 36.06 | 38.17  |
+----------------------+----------+----------+------------+-------+--------+
| Ours  (Training time)| 300s     | 272s     | 258s       | 331s  | 287s   |
+----------------------+----------+----------+------------+-------+--------+

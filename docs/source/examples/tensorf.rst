.. _`Instant-NGP Example`:

Instant-NGP
====================

See code `examples/train_tensorf_nerf.py` at our `github repository`_ for details.

Benchmarks
------------
*updated on 2023-02-24*

A bunch of details to be explained -- left for later. We have a version that we
apply to the original repo (s.t. with all the alpha masking and ray filtering
tricks) and another we cleanly port in the examples (s.t. with no tricks and
keep it as standard as possible).

+--------------------------+-------+
| PSNR                     | MEAN  |
|                          |       |
+==========================+=======+
|TensoRF 15k steps (orig.) | 32.52 |
+--------------------------+-------+
|(training time)           | 633s  |
+--------------------------+-------+
|Ours 15k steps            | 32.51 |
+--------------------------+-------+
|(training time)           | 386s  |
+--------------------------+-------+

+--------------------------+-------+
| PSNR                     | MEAN  |
|                          |       |
+==========================+=======+
|TensoRF 15k steps (field) | 31.19 |
+--------------------------+-------+
|(training time)           | 872s  |
+--------------------------+-------+
|Ours 15k steps            | 30.81 |
4--------------------------+-------+
|(training time)           | 204s  |
+--------------------------+-------+

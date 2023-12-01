K-Planes
====================

In this example we showcase how to plug the nerfacc library into the *official* codebase 
of `K-Planes <https://sarafridov.github.io/K-Planes/>`_. See 
`our forked repo <https://github.com/liruilong940607/kplanes/tree/b97bc2eefc18f00cd54833800e7fc1072e58be51>`_
for details.


Benchmark: D-NeRF Dataset
---------------------------------
*updated on 2023-04-04 with nerfacc==0.5.0*

Our experiments are conducted on a single NVIDIA GeForce RTX 2080 Ti. 

+----------------------+----------+---------+-------+---------+-------+--------+---------+-------+-------+
| PSNR                 | bouncing | hell    | hook  | jumping | lego  | mutant | standup | trex  | MEAN  |
|                      | balls    | warrior |       | jacks   |       |        |         |       |       |
+======================+==========+=========+=======+=========+=======+========+=========+=======+=======+
| K-Planes             | 39.10    | 23.95   | 27.76 | 31.11   | 25.18 | 32.44  | 32.51   | 30.25 | 30.29 |
+----------------------+----------+---------+-------+---------+-------+--------+---------+-------+-------+
| training time        | 68min    | 70min   | 70min | 70min   | 70min | 71min  | 72min   | 71min | 70min |
+----------------------+----------+---------+-------+---------+-------+--------+---------+-------+-------+
| Ours (occ)           | 38.95    | 24.00   | 27.74 | 30.46   | 25.25 | 32.58  | 32.84   | 30.49 | 30.29 |
+----------------------+----------+---------+-------+---------+-------+--------+---------+-------+-------+
| training time        | 41min    | 41min   | 40min | 40min   | 39min | 39min  | 40min   | 38min | 40min |
+----------------------+----------+---------+-------+---------+-------+--------+---------+-------+-------+

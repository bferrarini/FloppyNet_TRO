# How to compile lce_cnn

There are several ways to compile lce_cnn. We used the docker for cross-platform compiling available at [lce-build](https://docs.larq.dev/compute-engine/build/docker/).
Note that you don't need a PRI4 to compile the source code if you opt for the docker. You need a regular Linux box or any other OS capable of running a docker.

You will place *lce_cnn.cc* in the _example_ directory  of the _compute_engine_ repository ((check here)[https://docs.larq.dev/compute-engine/build/]).
The provided Makefile includes the required targets for lce_cnn. It should be placed in  `<path_to_your_git_clone>/\larq_compute_engine\tflite\build_make\`.

To compile lce_cnn, follow the instruction for LCE on the LCE [web page](https://docs.larq.dev/compute-engine/build/docker/).

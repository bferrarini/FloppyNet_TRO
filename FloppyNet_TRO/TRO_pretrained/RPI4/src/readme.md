# How to compile lce_cnn

There are several ways to compile lce_cnn. We useduse the docker for cross-platform compiling avaialble at [lce-build](https://docs.larq.dev/compute-engine/build/docker/).
Note that is you opt for the doker, you don't need a PRI4 to compile the source code. You just need a regular Linux box or any other OS capable to run a docker,

You will place *lce_cnn.cc* in the _example_ directory  of the _compute_engine_ repository ((check here)[https://docs.larq.dev/compute-engine/build/]).
The Makefile is modified with the required targets for lce_cnn. It should be placed in  `<path_to_your_git_clone>/\larq_compute_engine\tflite\build_make\`.

To compile, follow the instruction for LCE on the LCE [web page](https://docs.larq.dev/compute-engine/build/docker/).

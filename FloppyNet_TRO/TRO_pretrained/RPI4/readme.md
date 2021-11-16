# How to execute a LCE model on the RPI4

We provide the binary *lce_cnn*. This tool can run and benchamrck any model comepiled for LCE ot TFLITE on an ARM64 platform.
The source code is provided in _src_.

```
./lce_cnn -h
```

### Computing an image representation

```
./lce_cnn -g floppyNet_try4.tflite -i image000.jpg -o feature_file.txt
```

### Benchmarking a model

```
./lce_cnn -g floppyNet_try4.tflite -b
```

If your need a more comprehensive benchmark, you can use lce_benchmark_model (LCE custom TFLite Model Benchmark Tool), which was released withing the LCE project at https://github.com/larq/compute-engine
A typical use is as follows:

```
./lce_benchmark_model --graph=floppyNet_try4.tflite --num_threads=4 --wamup_runs=50
```

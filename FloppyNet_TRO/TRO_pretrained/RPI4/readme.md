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

The benchmark measures the image descriptor computation while not considering the time required to resize and load an input image from the file system. Thus, it reflects the actual computational efficiency (lines 260 and 264 of src/lce_cnn.cc).
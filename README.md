# FloppyNet
 FloppyNet is a BNN designed for VPR and loop closure detection in particular.
 
This repository shares several tool for training and deploying FloppyNet on a Rasperry PI4.

1. *main.py* is the only file you need for:
    *  Training FloppyNet and the other BNNs presented in our paper.
    *  Exporting a model in [Larq-Compute-Engine](https://docs.larq.dev/compute-engine/) format for ARM64 cpus (i.e. RPI4).  
    *  Computing an image descriptor
2. *TRO_pretrained* contains the model trained for the paper. Both H5 and LCE (.tflite) formats are available.
    * RPI4 includes an engine to run the LCE models: lce_cnn
3. *scripts* includes detailed instructions on how to used main.py with serverl examples.

The project has been developed within [Eclipse+PyDev](https://www.pydev.org/) but the code can be exectuded from a command line.

## Software Requirements

The main python3 packages required to use the provided code are the following.

* Tensorflow >= 2.3.1
* larq >= 0.10.2
* larq-compute-engine >= 0.4.3
* opencv >= 4.4.0
* prettytable 2.0.0

## How to cite this work
If you use this code, please cite us:

`BIBTEX HERE`

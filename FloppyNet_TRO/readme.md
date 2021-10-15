# 1) main.py
This script accepts several parameters to execute tre main function: training a model, exporting it for ARM64 deployment and computing an image representation.
main.py ralys on a configuration file, *experiment_presets.py*, where the training and modl parameters are defined.
The available parameters can be checked via: 

```
main.py -h
```

## EXPERIMENTAL PRESETS

1) alexnet_TRO (Baseline)
2) binarynet_TRO
3) shallownet_TRO
4) floppynet_TRO

## WORKING FOLDER
The default is: *./output/trained_models*
It can be changed via `-models_save_dir` parameter
For example, to train FloppyNet with the preset *floppynet_TRO* using a different working folder, the command is as follows:
```
python3 main.py -M training --preset floppynet_TRO --models_save_dir C:\mymodels
```

## Training a model ###
```
python3 main.py -M training --preset <PRESET_OF_YOUR_CHOICE>
```

## Exporting a model ###

__As a H5 keras model__
```
python3 main.py -M export --format H5 --preset <PRESET_OF_YOUR_CHOICE>
```

__As a tflite model for RPI4__
```
python3 main.py -M export --format arm64 --preset <PRESET_OF_YOUR_CHOICE>
```

__both formats at the same time__
```
python3 main.py -M export --format all --preset <PRESET_OF_YOUR_CHOICE>
```

## Computing an image representation
```
python3 main.py -M descriptor --target_images PATH_TO_YOUR_IMAGE_OR_IMAGE_FOLDER --output_features_file PATH_TO_OUT_FILE --preset <PRESET_OF_YOUR_CHOICE>
```

## Computing an image representation from an H5 model
```
python3 main.py -M descriptor --target_images PATH_TO_YOUR_IMAGE_OR_IMAGE_FOLDER --output_features_file PATH_TO_OUT_FILE --h5_model PATH_TO_YOUR_H5_MODEL_FILE
```

## Computing an image representation from an TFLITE model
You need a RPI4 installed with a 64-bit OS (e.g. Ubuntu 20.04) and the lce_cnn executable available in TRO_pretrained/RPI4 folder of this project 

## CPU_ONLY mode
Use the flag `--cpu_only` as follows:
```
python3 main.py -M descriptor --cpu_only --target_images PATH_TO_YOUR_IMAGE_OR_IMAGE_FOLDER --output_features_file PATH_TO_OUT_FILE --preset <PRESET_OF_YOUR_CHOICE>
```

More details can be found into the `scripts` folder.


# 2) lce_cnn
The pretrained model for RPI4 are available in the folder `TRO_pretrained/RPI4`.
In the same directory are available our BNN executable based on LCE and the source code. If you are interested in compiling our source code or your version, I suggest compiling it inside the Doker provided byLarq progject](https://docs.larq.dev/compute-engine/build/docker/).

# 3) 'output' folder
This directory is populated on training. If you intentend only to run pretrained model, you don't need this folder.
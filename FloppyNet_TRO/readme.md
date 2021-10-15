### EXPERIMENTAL PRESETS ###

1) alexnet_TRO (Baseline)
2) binarynet_TRO
3) shallownet_TRO
4) floppynet_TRO

### PRESET WORKING FOLDER ###
./output/trained_models
It can be changed via -models_save_dir paramter
For example, to train FloppyNet with the preset 'floppynet_TRO' using a different working folder, the command is as follows:
python3 main.py main.py -M training --preset floppynet_TRO --models_save_dir C:\mymodels

### Training a model ###
python3 main.py -M training --preset <PRESET_OF_YOUR_CHOICE>

### Exporting a model ###

* As a H5 keras model
python3 main.py -M export --format H5 --preset <PRESET_OF_YOUR_CHOICE>

* As a tflite model for RPI4
python3 main.py -M export --format arm64 --preset <PRESET_OF_YOUR_CHOICE>

* both formats at the same time
python3 main.py -M export --format all --preset <PRESET_OF_YOUR_CHOICE>

### Computing an image representation ###
python3 main.py -M descriptor --target_images PATH_TO_YOUR_IMAGE_OR_IMAGE_FOLDER --output_features_file PATH_TO_OUT_FILE --preset <PRESET_OF_YOUR_CHOICE>

### Computing an image representation from an H5 model ###
python3 main.py -M descriptor --target_images PATH_TO_YOUR_IMAGE_OR_IMAGE_FOLDER --output_features_file PATH_TO_OUT_FILE --h5_model PATH_TO_YOUR_H5_MODEL_FILE

### Computing an image representation from an TFLITE model ###
You need a RPI4 installed with a 64-bit OS (e.g. Ubuntu 20.04) and the lce_cnn executable available in TRO_pretrained/RPI4 folder of this project 

### CPU_ONLY MODE ###
Use the flag --cpu_only as follows:
python3 main.py -M descriptor --cpu_only --target_images PATH_TO_YOUR_IMAGE_OR_IMAGE_FOLDER --output_features_file PATH_TO_OUT_FILE --preset <PRESET_OF_YOUR_CHOICE>

### MODEL SAVE FOLDER ###
the parameter 


More into the scripts directory.
'''
Created on 10 Oct 2021

@author: Bruno Ferrarini
@affiliation Univeristy of Essex, UK

'''
#import config
import os
import larq
import larq_compute_engine as lce
from training_utility import train_op
from features_helpers import compute_descriptor
import model_factory as factory
import experiment_presets as EX
import config.argparsing as AP


##############################
# EXEPRIMENT CONFIGURATION ###
##############################
#from models.Lce_HybridShallow import QuantizedHShallow as network

'''
    Train the model
    @param epoch: the number of epochs
    @param training_data: path to the training dataset
    @param training_classes: number of classes in the training dataset
    @param validation_data: path to the training dataset
    @param batch_size: the size of the batch
    @param l_reate: learning rate
    @param resume: True to resume the training. Default is False. 
    @param augment: uses augmentation
    @param backup: backups the old model before starting a new training session

'''

def train(model_name,
          epochs, 
          training_data,
          training_classes,
          validation_data = None,
          val_split = 0.4,
          batch_size = 24, 
          l_rate=1e-4, 
          augment = False, 
          resume = False, 
          backup = False,
          model_save_dir = os.path.join('.','output','trained_models'),
          ):
    
    #Instantiate a model wrapper
    model_wrapper = factory.get_model(model_name, training_classes, l_rate, resume, model_save_dir)
    
    train_op(model_wrapper, 
             model_name = model_name,
             train_dir = training_data,
             val_dir = validation_data,
             val_split = val_split,
             batch_size = batch_size, 
             epochs = epochs, 
             augment = augment,
             resume = resume, backup = backup)
    
    
    
def export(
            model_name,
            training_classes, #for loading the model
            out_layer = 'pool5',
            flatten =  False, #False keeps the output layers as it is. True, returns a flatten feature map
            model_save_dir = os.path.join('.','output','trained_models'),
            out_dir = None,
            model_format = 'H5',
            verb = False
        ):
    
    import tensorflow as tf
    
    # Some of the parameters are unnecessary for exporting. Thus they are set arbitrary
    model_wrapper = factory.get_model(model_name, training_classes = training_classes, l_rate = 10, resume = False, model_save_dir = model_save_dir)
    
    out_dir_ = os.path.join('output','trained_models', model_name, 'export') if out_dir is None else out_dir
    if not os.path.exists(out_dir_):
        os.mkdir(out_dir_)
        print(f"Created {out_dir_}")
        
    model_wrapper.load()
        # import traceback
        # import sys
        # print('Weights not available in the default location')
        # print('Empty model exported\n')
        # print(traceback.format_exc())
        
    sub_model = model_wrapper.get_inner_layer_by_name(out_layer, flatten=flatten)
    if verb:
        larq.models.summary(sub_model) 
    
    
    if model_format == AP.H5 or model_format == AP.ALL:
        fn = os.path.join(out_dir_, model_name + ".h5")
        sub_model.save(fn)
        print(f"{model_name} model saved at {fn}")
        
    if model_format == AP.ARM64 or model_format == AP.ALL:
        if model_name == EX.ANet_TRO: #TFLITE for regular deployment on ARM
            converter = tf.lite.TFLiteConverter.from_keras_model(sub_model)
            # #no optimization as we want AlexNet as a 32-bit model
            # converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
            tflite_model = converter.convert() 
            
            fn = os.path.join(out_dir_, model_name + ".tflite")
            with open(fn, 'wb') as f:
                f.write(tflite_model)
                print(f"{model_name} model for RPI4 is saved at {fn}")
            
        else: #LCE for BNNs
            fn = os.path.join(out_dir_, model_name + ".tflite")
            with open(fn, "wb") as fb:
                fb_bytes = lce.convert_keras_model(sub_model)
                fb.write(fb_bytes)
                print(f"{model_name} model for RPI4 is saved at {fn}")
        
        
def descriptor(
        model_name,
        training_classes, #for loading the model
        images, 
        out_file,
        out_layer = 'pool5',
        model_save_dir = os.path.join('.','output','trained_models'),
        verb = False,
        ):

    model_wrapper = factory.get_model(model_name, training_classes = training_classes, l_rate = 10, resume = False, model_save_dir = model_save_dir)
    
    model_wrapper.load()      
    
    sub_model = model_wrapper.get_inner_layer_by_name(out_layer, flatten=False)
    if verb:
        larq.models.summary(sub_model)   
         
    compute_descriptor(sub_model, images, out_file, limit_to = None, flatten = False, batch_size = 1)
    
    print(f"Feature file written at {out_file}")


def descriptor_from_h5(
        images, 
        out_file,
        h5_fn,
        verb = False,
        ):
    
    import tensorflow as tf
    
    model = tf.keras.models.load_model(h5_fn)
    
    if verb:
        larq.models.summary(model)   
         
    compute_descriptor(model, images, out_file, limit_to = None, flatten = False, batch_size = 1)
    
    print(f"Feature file written at {out_file}")    
    


'''
Created on 11 Oct 2021

@author: Bruno Ferrarini
@affiliation Univeristy of Essex, UK

'''

import experiment_presets as E
import config
import os

def get_model(
            model_name,
            training_classes,
            l_rate=1e-4, 
            resume = False, 
            model_save_dir = os.path.join('.','output','trained_models')
            ):
    
    
    model = None
    
    if model_name == E.FNet_TRO:

        from models.Lce_HybridShallow import QuantizedHShallow as floppynet_wrapper
    
        #Instantiate a model wrapper
        model = floppynet_wrapper(
            model_name = model_name, 
            working_dir = model_save_dir, 
            model_name_2 = 'model', 
            logger_lvl = config.log_lvl,
            nClasses = training_classes,
            fc_units = 256,
            filters = (96,256,256),
            l_rate = l_rate,
            save_weights_only = True,
            enable_monitoring = True,
            tensorboard = True,
            activation_precision = 1, #BNN
            kernel_precision = 1, #BNN
            enable_history = False,
            clean_history = not resume,
            optimizer = None, #Adam will be used with the l_rate as a learning rate
            loss = 'categorical_crossentropy'
            )
        
    elif model_name == E.SNet_TRO:
        
        from models.Lce_shallowNet import QuantizedShallow

        model = QuantizedShallow(
            model_name = model_name, 
            working_dir = model_save_dir, 
            model_name_2 = 'model', 
            logger_lvl = config.log_lvl,
            nClasses = training_classes,
            fc_units = 4096,
            l_rate = l_rate,
            save_weights_only = True,
            enable_monitoring = True,
            tensorboard = True,
            activation_precision = 1, 
            kernel_precision = 1,
            enable_history = False
            )        
    
    elif model_name == E.ANet_TRO:
        
        from models.Z_AlexNet import AlexNet
        
        model = AlexNet(
            model_name = model_name, 
            working_dir =  model_save_dir, 
            model_name_2 = 'model', 
            logger_lvl = config.log_lvl,
            nClasses = training_classes,
            fc_units = 4096,
            l_rate = l_rate,
            save_weights_only = True,
            enable_monitoring = True,
            tensorboard = True,
            enable_history= False
            )
    
    elif model_name == E.BNet_TRO:
        
        from models.Lce_AlexNet import QuantizedAlexNet as BinAlex
        
        model = BinAlex(
            model_name = model_name, 
            working_dir = model_save_dir, 
            model_name_2 = 'model', 
            logger_lvl = config.log_lvl,
            nClasses = training_classes,
            fc_units = 4096,
            l_rate = l_rate,
            save_weights_only = True,
            enable_monitoring = True,
            tensorboard = True,
            activation_precision = 1, 
            kernel_precision = 1,
            enable_history = False
            )
 
    
    else:
        raise ValueError(f"Invalid model name (preset): {model_name}")
        
    return model


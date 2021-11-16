'''
Created on 9 Oct 2021

@author: Bruno Ferrarini
@affiliation Univeristy of Essex, UK

Experimental presets

'''

import os
import dataset_presets as D
import copy

########################################
### working directory for presets ######
###  Change it for your convenience ####
########################################
model_save_dir = os.path.join('.','output','trained_models')
#################################################################

#model_save_dir = os.path.join(r'C:\Users\main\Documents\eclipse-workspace\FloppyNet_TRO','output','trained_models')


experiments = dict()
#########################
### TRO Paper presets ###
#########################

params = dict()

FNet_TRO = 'floppynet_TRO'
params['model_name'] = FNet_TRO 
## TRAINING DATA ###
## CHANGE THE PATHs ACCORDINGLY WITH YOU NEEDS ##
params['training_data'] = D.training_datasets[D.PLACES365]['training_path']
# Set validation data to None to split the training data
params['validation_data'] = D.training_datasets[D.PLACES365]['validation_path']
# val split is ignored if a path to validation data is given
params['val_split'] = 0.4
####################
params['classes'] = D.training_datasets[D.PLACES365]['nClass']
params['l_rate'] = 5e-4
params['batch_size'] = 24
params['epochs'] = 150
params['model_save_dir'] = model_save_dir
params['out_layer'] = 'pool5'
#params['module'] = 'operation_modes'
experiments[FNet_TRO] = copy.copy(params)


params = dict()
SNet_TRO = 'shallownet_TRO'
params['model_name'] = SNet_TRO
params['training_data'] = D.training_datasets[D.PLACES365]['training_path']
params['validation_data'] = D.training_datasets[D.PLACES365]['validation_path']
params['classes'] = D.training_datasets[D.PLACES365]['nClass']
params['l_rate'] = 5e-4
params['batch_size'] = 24
params['val_split'] = 0.4
params['epochs'] = 150
params['model_save_dir'] = model_save_dir
params['out_layer'] = 'pool5'
#params['module'] = 'experiments.train.shallowNet'
experiments[SNet_TRO] = copy.copy(params)

params = dict()
BNet_TRO = 'binarynet_TRO'
params['model_name'] = BNet_TRO
params['training_data'] = D.training_datasets[D.PLACES365]['training_path']
params['validation_data'] = D.training_datasets[D.PLACES365]['validation_path']
params['classes'] = D.training_datasets[D.PLACES365]['nClass']
params['l_rate'] = 1e-4
params['batch_size'] = 32
params['val_split'] = 0.4
params['epochs'] = 150
params['model_save_dir'] = model_save_dir
params['out_layer'] = 'pool5'
#params['module'] = 'experiments.train.binaryNet'
experiments[BNet_TRO] = copy.copy(params)

params = dict()
ANet_TRO = 'alexnet_TRO'
params['model_name'] = ANet_TRO
params['training_data'] = D.training_datasets[D.PLACES365]['training_path']
params['validation_data'] = D.training_datasets[D.PLACES365]['validation_path']
params['classes'] = D.training_datasets[D.PLACES365]['nClass']
params['l_rate'] = 5e-4
params['batch_size'] = 24
params['val_split'] = 0.4
params['epochs'] = 25
params['model_save_dir'] = model_save_dir
params['out_layer'] = 'pool5'
#params['module'] = 'experiments.train.alexNet'
experiments[ANet_TRO] = copy.copy(params)


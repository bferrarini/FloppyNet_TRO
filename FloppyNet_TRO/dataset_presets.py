'''
Created on 9 Oct 2021

@author: Bruno Ferrarini
@affiliation Univeristy of Essex, UK

Datasets specifications.

'''


###########################
#### TRAINING DATASETS ####
###########################


PLACES365 = 'places365'
training_datasets = dict()


training_datasets[PLACES365] = dict()
training_datasets[PLACES365]['training_path'] = r"E:\fast_datasets\places365_standard\train"
training_datasets[PLACES365]['validation_path'] = r"E:\fast_datasets\places365_standard\val"
training_datasets[PLACES365]['test_path'] = r"E:\fast_datasets\places365_standard\val"
training_datasets[PLACES365]['nClass'] = 365


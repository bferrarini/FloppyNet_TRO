'''
Created on 11 Oct 2021

@author: Bruno Ferrarini
@affiliation Univeristy of Essex, UK

'''

import tensorflow as tf
import csv
import numpy as np
from pathlib import Path
import os
import cv2
import re

def basename_without_extension(filename): 
    return Path(filename).stem

def label(filename):
    return basename_without_extension(filename)

def next_batch(i, batch_size, fn_list, target_size):
    batch = list()
    batch_fn = list()
    for fn in fn_list[i*batch_size : min((i+1)*batch_size,len(fn_list))]:
        img = cv2.imread(fn, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img*(1/255)
        batch.append(img)
        batch_fn.append(fn)
    batch = np.array(batch)
    return batch, batch_fn
        

def compute_descriptor(model : tf.keras.models.Model,
              images : str,
              toFile : str,
              limit_to = None,
              flatten = True,
              batch_size = 1,          
              ):
    

    input_shape = model._feed_input_shapes[0]
    IMG_HEIGHT = input_shape[1]
    IMG_WIDTH = input_shape[2]
        
    if os.path.isdir(images):
        images_fn = [os.path.join(images, f) for f in os.listdir(images) if re.search('.*\.[jpg|png]', f)]
    elif os.path.isfile(images):
        images_fn = [images, ]
    
    
    if limit_to is None:
        nSTEPS = len(images_fn) // batch_size
    else:
        nSTEPS = limit_to
    
    
    with open(toFile, 'w') as f:
        fnames = ['file_name',
                  'label',
                  'shape',
                  'features'
                  ]
        writer = csv.DictWriter(f, fieldnames=fnames, delimiter=';', lineterminator="\n")
        writer.writeheader()
         
        for step in range (nSTEPS):
            batch, filenames = next_batch(step, batch_size, images_fn, (IMG_WIDTH, IMG_HEIGHT))
            model_out = model.predict(batch)
            shape = np.shape(model_out)
            #rid-off the batch size
            #print(shape)
            shape_str = str(shape[1:])
            for j in range(batch_size):
                flat_features = model_out[j].flatten('C')
                features = map(str,flat_features)
                row = {
                    'file_name' : filenames[j],
                    'label' : label(filenames[j]),
                    'shape' : shape_str,
                    'features' : ",".join(features)
                    }
                writer.writerow(row)



if __name__ == '__main__':
    
    images = r"C:\Users\main\Pictures\test"
    
    images_fn = [os.path.join(images, f) for f in os.listdir(images) if os.path.isfile(os.path.join(images, f))]
    
    batch, _ = next_batch(0, 2, images_fn, (277,277))
    
    print(np.shape(batch))
    
    batch, _ = next_batch(1, 2, images_fn, (277,277))
    
    print(np.shape(batch))
    
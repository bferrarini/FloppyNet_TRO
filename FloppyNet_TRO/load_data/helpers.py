'''
Created on 13 Jan 2020

@author: Bruno Ferrarini
@affiliation Univeristy of Essex, UK

'''

import time
import matplotlib.pyplot as plt

'''
    Measure the time needed to load all the images by an iterator.
    It works with Keras Generators as well.
    @parameters
        - steps: number of batches to use to collect the statistics
        - BATCH_SIZE: batch size
'''

def timeit(ds, steps=1000, BATCH_SIZE = 32):
    start = time.time()
    it = iter(ds)
    for i in range(steps):
        batch = next(it)
        if i%10 == 0:
            print('.',end='')
    print()
    end = time.time()

    duration = end-start
    print("{} batches: {} s".format(steps, duration))
    print("{:0.1f} Images/s".format(BATCH_SIZE*steps/duration))



'''
    Display 25 random images from a batch.
    @parameters
        - image_batch: a batch f images (e.g. next(keras_generator)[0])
        - label_batch: the corresponding labels (e.g. next(keras_generator)[1])
        - class_mode: can be either 'categorical' or 'sparse'
'''

def show_batch(image_batch, label_batch, class_mode='categorical'):
    plt.figure(figsize=(10,10))
    X = 5
    Y = 5
    f, ax = plt.subplots(X,Y)
    for i in range(X):
        for j in range(Y):
            ax[i,j].imshow(image_batch[i*X + j])
            if class_mode == 'categorical':
                title = list(label_batch[i*X + j]).index(max(list(label_batch[i*X + j])))
            if class_mode == 'sparse':
                title = str(int(label_batch[i*X + j]))
            ax[i,j].set_title(title)
            ax[i,j].axis('off')


    
    
    
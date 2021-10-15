'''
Created on 20 Jan 2020


Utilities to display results.
Deprecated as I use Tensorboard now.

@author: Bruno Ferrarini
@affiliation Univeristy of Essex, UK

'''

import pickle
import matplotlib.pyplot as plt
import math

'''
    Plots the history file content.
    @params
        - fn: the history filename produces by the fit method of a model wrap class
        - metrics: a list eith the metrics to plot. If it is None, all the content of the history file is shown.
'''


def display_history(fn, metrics = None):
    
    def plot(metric, history):
        
        results = history[metric]
        epochs = [i+1 for i in range(len(results))] 
        #plt.scatter(epochs, results, linewidths=2, joinstyle='-')
        plt.plot(epochs, results, '-x')
        
        try:
            val_results = history[str("val_") + metric]
            period = math.floor(len(results) / len(val_results))
            val_epochs = [(i+1)*period for i in range(len(val_results))]
            #plt.scatter(val_epochs, val_results, linewidths=2)
            plt.plot(val_epochs, val_results, '-x')
            plt.legend(['Train','Val'], loc='upper right')
        except:
            plt.legend(['Train',], loc='upper left')
            
        plt.title('Model ' + metric)
        plt.ylabel(metric)
        plt.xlabel('Epoch')        
        
        plt.show()
    
    history = pickle.load(fn)
    
    if metrics is None:
        for k in history.keys():
            if not k.startswith('val_'):
                plot(k, history)
    else:
        for metric in metrics:
            v = history(metric)
            plot(metric, v)
    
            
    print(history)
    
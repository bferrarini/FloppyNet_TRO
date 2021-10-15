'''
Created on 06 Nov 2019

@author: Bruno Ferrarini
@affiliation Univeristy of Essex, UK

'''

import os
import shutil
import tensorflow as tf
import pickle
import glob
import zipfile
from datetime import datetime
import time
from PIL import Image
import numpy as np
import json
from utils import mkdirs, clear_dir
from tensorflow.keras.callbacks import TensorBoard



'''
    Base class to handle checkpoint filenames
'''

class CheckPoint():
    
    '''
        Constructor
        #Arguments
            save_dir: is thepath were the model files will be stored
            create_is_not_exists: if true makes the save_dir directory being create by the class  constructor
    '''
    
    def __init__(self, save_dir, create_if_not_exist = True):
        self.save_dir = save_dir
        
        if create_if_not_exist == True:
            if not os.path.exists(save_dir):               
                os.makedirs(save_dir)
                
    '''
        Clear the model directory.
    '''            
                
    def clear(self):
        shutil.rmtree(self.save_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
    '''
        Save the history file along with the model
    '''
        
    def save_history(self):
        pass
    
    '''
        Restore history
    '''
   
    def load_history(self):
        pass
    
    '''
        Callback to pass to model.fit
    '''
    def callback(self):
        pass
    
    
        
'''
    Derived class of CheckPoints. Handles the filename for the single save mode.
'''      
class SingleCheckPoint(CheckPoint):
    
    '''
        Constructor
        #Arguments
            save_dir: is thepath were the model files will be stored
            model_name: is the name of the file to save. For example: model_xyz will saved as model_xyz.ckpt in save_dir
            create_is_not_exists: if true makes the save_dir directory being create by the class  constructor
    '''    
    
    def __init__(self, save_dir, model_name, filename = 'model', create_if_not_exist = True, save_weights_only = True, 
                 enable_monitoring = False, monitor_metric = 'val_accuracy'):
        
        CheckPoint.__init__(self, save_dir, create_if_not_exist = create_if_not_exist)
        
        self.save_model_dir = os.path.join(self.save_dir, model_name)
        
        self.checkpoint_fn = filename
        
        if not os.path.exists(self.save_model_dir) and create_if_not_exist == True:
            os.makedirs(self.save_model_dir)
        
        self.ckp = os.path.join(self.save_model_dir, filename + '.ckpt')
        self.hist = os.path.join(self.save_model_dir, filename + ".hst")
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        self.zip = os.path.join(self.save_model_dir, filename + "_" + date_time + '.zip')
        
        self.save_weights_only = save_weights_only
        if save_weights_only:
            self.ckp = os.path.join(self.save_model_dir, filename + '.ckpt')
        else:
            self.ckp = os.path.join(self.save_model_dir, filename + '.h5')
        
        self.monitoring = enable_monitoring
        self.monior_metric = monitor_metric
            
    '''
        Save the history along with the model
    '''
                    
    def save_history(self, history_obj):
        
        with open(self.hist, 'wb') as f:
            pickle.dump(history_obj.history,f)
        
            
    '''
        The fullpath of the model file
    '''
    def load_history(self):
        with open(self.hist, 'rb') as f:
            history = pickle.load(f)        
            return history
        
    '''
        Callback to pass to model.fit
    '''    
    def callback(self, save_weights_only=True, verbose=1):
        
        if self.monitoring:
            fn = tf.keras.callbacks.ModelCheckpoint(
                    filepath=self.ckp, 
                    verbose=verbose, 
                    save_weights_only=self.save_weights_only,
                    monitor=self.monior_metric,
                    save_best_only=True,
                    mode='max'
                    )            
        else:
        
            fn = tf.keras.callbacks.ModelCheckpoint(
                    filepath=self.ckp, 
                    verbose=verbose, 
                    save_weights_only=self.save_weights_only)
        
        
        
        return fn
    
    '''
        Checks if a model file already exists
    '''
   
    def check_model_files(self):
        ckp = glob.glob(self.ckp + "*")
        print(len(ckp))
        hst = glob.glob(self.hist + "*")
        print(len(hst))
        return len(ckp) > 0
    
    ''' 
        Backups older files
    
    '''
    def backup(self):
        if self.check_model_files():
            files_to_backup = []
            for f in list(glob.glob(self.ckp + "*")):
                files_to_backup.append(f)
            for f in list(glob.glob(self.hist + "*")):
                files_to_backup.append(f)
            for f in list(glob.glob(os.path.join(self.save_model_dir, "checkpoint") + "*")):
                files_to_backup.append(f)                
            #print(file_to_backup)
        
            
            with zipfile.ZipFile(self.zip, mode='w') as newZip:
                for f in files_to_backup:
                    newZip.write(f)
            return self.zip
                    
        
    @property
    def filename(self):
        return self.ckp
    

'''
    This checkpoint saver has been added lately to the framework to handle binary networks.
    Binary Network exhibits wide variation in validation accuracy, thus just saving the last model or using monitoring, is not always the
    the best way to spot the best model for VPR.
    This callback saves all the checkpoints in weights only format and, optionally, ad a complete model.
    
'''
   

class MultiBrenchCheckPointSaver(CheckPoint):
    
    '''
        @param save_mode: 'weights' for weights only, 'complete' for complete model. 
    '''
    
    def __init__(self, save_dir, model_name, save_mode, filename = 'model_{epoch:03d}_{val_accuracy:.2f}', create_if_not_exist = True, clean = True):
        CheckPoint.__init__(self, save_dir, create_if_not_exist=create_if_not_exist)
        
        self.save_model_dir_weights = os.path.join(self.save_dir, model_name, 'history',  'weights')
        self.save_model_dir_complete = os.path.join(self.save_dir, model_name, 'history',  'complete')
        
        self.save_mode = save_mode
        
        if clean:
            shutil.rmtree(self.save_model_dir_weights, ignore_errors = True)
            shutil.rmtree(self.save_model_dir_complete, ignore_errors = True)
        
        if self.save_mode == 'weights':
            if not os.path.exists(self.save_model_dir_weights) and create_if_not_exist == True:
                os.makedirs(self.save_model_dir_weights)
                
        if self.save_mode == 'complete':
            if not os.path.exists(self.save_model_dir_complete) and create_if_not_exist == True:
                os.makedirs(self.save_model_dir_complete)                
        
        self.ckp_w = os.path.join(self.save_model_dir_weights, filename + '.ckpt')
        self.ckp_c = os.path.join(self.save_model_dir_complete, filename + '.h5')
        
        
        
    def callback(self):
        
        if self.save_mode == 'weights':
            fn = tf.keras.callbacks.ModelCheckpoint(
                    filepath=self.ckp_w, 
                    verbose=True, 
                    save_weights_only=True
                    )
                    
        if self.save_mode == 'complete':
            fn = tf.keras.callbacks.ModelCheckpoint(
                    filepath=self.ckp_c, 
                    verbose=True, 
                    save_weights_only=False
                    )
                    
        return fn

'''
    This Abstract Class has the purpose of providing my custom callbacks
    with the method set_val_gen, which is required for particular cases such as the autoencoders.
'''

class AbstractCallback(tf.keras.callbacks.Callback):
    
    def __init__(self):
        super(AbstractCallback, self).__init__()
        self.gen = None
        
    def set_val_gen(self, val_gen):
        pass
    

class TimerCallback(AbstractCallback):
    
    def __init__(self):
        super(TimerCallback, self).__init__()
        self.start = 0
    
    def on_epoch_begin(self, epoch, logs=None):
        self.start = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        duration = time.time() - self.start
        print("Epoch {:d} ran in {:0.2f} seconds.\n".format(epoch, duration))
        
class SampleAutoencoderReconstraction(AbstractCallback):
    
    def __init__(self, out_dir, generator, create_dir = True, clear_old = False):
        super(SampleAutoencoderReconstraction, self).__init__()
        self.img_index = 0
        self.dir = out_dir    
        self.gen = generator
        
        if clear_old:
            clear_dir(out_dir)
            
        if create_dir:
            mkdirs(out_dir)
            
    def set_val_gen(self, val_gen):
        AbstractCallback.set_val_gen(self, val_gen)
        self.gen = val_gen
        
    
    def on_epoch_end(self, epoch, logs={}):
        batch = next(self.gen)
        y_pred = self.model.predict(batch)
        
        b_img = (batch[0][0][:,:,:]*255).astype(np.uint8)
        
        fn = os.path.join(self.dir, "{:04d}_sample.png".format(epoch))
        r_img = (y_pred[0][:,:,:]*255).astype(np.uint8)
        
        img = Image.fromarray(np.concatenate((b_img,r_img), axis = 1))
        img.save(fn)
        
        print('\nSample save at {:s}'.format(fn))    
    
'''
    Defines a set of general purpose operations to execute at the end of various training phases
'''
        
class MiscellaneousCallback(AbstractCallback):
    
    def __init__(self, out_dir):
        super(MiscellaneousCallback, self).__init__()    
        self.dir = out_dir
        self.epoch_fn = "epoch.ckpt"
    
    def on_epoch_end(self, epoch, logs={}):   
        self._annotate_epoch(epoch)
    
    def _annotate_epoch(self, epoch):
        fn = os.path.join(self.dir, self.epoch_fn)
        with open(fn, "w") as f:
            x = {"last_epoch_trained" : epoch}
            json.dump(x, f)
    
    def read_epoch_annotation(self):
        fn = os.path.join(self.dir, self.epoch_fn)
        if os.path.exists(fn):
            with open(fn, "r") as f:
                x = json.load(f)
                return x['last_epoch_trained']
        else:
            return 0
        
class ClassifierTensorboardCallback(tf.keras.callbacks.TensorBoard):
    
    def __init__(self, log_dir='logs', histogram_freq=0, write_graph=False, write_images=False,
                 update_freq='epoch', profile_batch=2, embeddings_freq=0,
                 embeddings_metadata=None, **kwargs):
        super(ClassifierTensorboardCallback, self).__init__(log_dir, histogram_freq, write_graph, write_images,
                 update_freq, profile_batch, embeddings_freq,
                 embeddings_metadata)
        
class TrainValTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.compat.v1.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.compat.v1.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir, **kwargs):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        #logs.update({'lr': tf.keras.backend.eval(self.model.optimizer.lr)})
        logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)
        super().on_epoch_end(epoch, logs)

    
    
'''
Created on 3 Jan 2020

@author: Bruno Ferrarini
@affiliation Univeristy of Essex, UK

'''

from functions.callback import SampleAutoencoderReconstraction, SingleCheckPoint, MiscellaneousCallback, TimerCallback, MultiBrenchCheckPointSaver, LRTensorBoard
import logging
import utils
import tensorflow as tf
import larq as lq
import os
import datetime
from utils.results import display_history
import tensorflow.keras as keras

from prettytable import PrettyTable

class ModelWrapper():
    
    '''
        Contructor
            @param model_name: the name of the model is part will of the fullpath of the complete working dir
            @Param working_dir: the root of the working folder where models, output, etc.. will be saved
            @param model_name2: is the name for the weight file
            @param logger_lvl: loggin level for the internal logger
            @param l_rate: learning rate
            @param **kwargs: is deputed to handle the parameter for the derived classes
    '''    
    
    def __init__(self,
                 model_name,
                 working_dir,
                 model_name_2 = "checkpoint",
                 logger_lvl = logging.ERROR,
                 l_rate = 0.0001,
                 verbose = True,
                 save_weights_only = True,
                 enable_monitoring = False,
                 monitor_metric = 'val_accuracy',
                 tensorboard = False,
                 enable_history = False,
                 clean_history = False,
                 **kwargs
                 
                 ):
        self.model_name = model_name
        self.working_dir = working_dir
        self.verbose = verbose
        
        self.lr = l_rate
        
        self.logger = self._setup_logger(level = logger_lvl)
        
        self.checkpoint = SingleCheckPoint(
            save_dir = self.working_dir, 
            model_name = self.model_name,
            filename = model_name_2, 
            create_if_not_exist = True,
            save_weights_only = save_weights_only,
            enable_monitoring = enable_monitoring,
            monitor_metric = monitor_metric
            )
         
        
        #Add the checkpoint callback to the callbacks to call during the training
        self.cb = [self.checkpoint.callback(),]
        
        if enable_history:
            self.history_saver = MultiBrenchCheckPointSaver(
                    save_dir = self.working_dir,
                    #this is to have 'history' as a subdirectory of model subfolder
                    model_name = self.model_name,
                    save_mode = 'weights',
                    clean = clean_history
                    )
            self._add_callbacks(self.history_saver.callback())
        
        if not save_weights_only:
            #Experimental
            self.checkpoint2 = SingleCheckPoint(
                save_dir = self.working_dir, 
                model_name = self.model_name,
                filename = model_name_2, 
                create_if_not_exist = True,
                save_weights_only = True,
                enable_monitoring = enable_monitoring,
                monitor_metric = monitor_metric
                )   
            self._add_callbacks(self.checkpoint2.callback())
        
        self.save_weights_only = save_weights_only
        
        if tensorboard:
            tb_dir_scalars = os.path.join(self.working_dir, self.model_name, r"logs\\scalars\\" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))  # @UndefinedVariable
            tensorboard_callback = keras.callbacks.TensorBoard(log_dir=tb_dir_scalars, write_graph=True)
            #tensorboard_callback = TrainValTensorBoard(log_dir=tb_dir_scalars)
            #### self._add_callbacks(tensorboard_callback)
            # NEW (24.11.2020)
            lr_log_callback = LRTensorBoard(log_dir=tb_dir_scalars, write_graph=True, profile_batch=0)
            self._add_callbacks(lr_log_callback)
                
        #Metrics
        self.metrics = []
                
        #Create the model. Later in the process, weights might be loaded
        # Let's try to commenti it out
        #self.model = self._setup_model(**kwargs)
        
        # layer map. Needs to be implemented in the subclass
        self.friendly_names = None
        self._init_friendly_names()
        
        self.inner_layers = {}
        
        self.k_bit = 0

    def input_shape(self):
        input_shape = self.model._feed_input_shapes[0]
        IMG_HEIGHT = input_shape[1]
        IMG_WIDTH = input_shape[2]
        return (IMG_HEIGHT, IMG_WIDTH)
    
    def output_shape(self):
        return self.model.layers[-1].output.shape
        
    '''
        Trains the model using Keras generators
        
        NOTE: working folder == self.checkpoint.save_model_dir, which is a subfolder of self.working_dir
        
            - epochs: training epochs
            - train_gen: training generator
            - val_gen; validation generator
            - resume_training: set True to resume from a previous checkpoint
            - weight_filename: used only when 'resume_training' is True. 
                If it is 'None' the last checkpoint in the working area is loaded, otherwise the specified file is used
            - BACKUP_OLD_MODEL_DATA: if True a backup of the old weights file is created in the working folder.
    '''    
 
 
    def fit_generator(self,
            epochs,
            train_gen, 
            val_gen,
            resume_training = False,
            weight_filename = None,
            BACKUP_OLD_MODEL_DATA = False
            ):
        
        self.fit(epochs, train_gen, val_gen, resume_training, weight_filename, BACKUP_OLD_MODEL_DATA)
    
    
    def fit(self,
                epochs,
                train_gen, 
                val_gen,
                resume_training = False,
                weight_filename = None,
                BACKUP_OLD_MODEL_DATA = False
                ):
        
        #Set the val_generators to the callbacks which needs it
        # Skip the first as it is a function and not a Callbackobject
        
        for i in range(0,len(self.cb)):
            try:
                self.cb[i].set_val_gen(val_gen)
            except:
                pass
                
        
        logger = self.logger
        
        logger.info("=> Checkpoint set at: {0}\n".format(self.checkpoint.filename))
        
        #Backup old model weights if required
        if BACKUP_OLD_MODEL_DATA:
            bkp_zip_fn = self.checkpoint.backup()
            if not bkp_zip_fn is None:
                logger.info("=> Old data saved in {:s}".format(bkp_zip_fn))
            
        
        #Epoch Annotation callback
        epoch_tracker = MiscellaneousCallback(self.checkpoint.save_model_dir)
        checkpoint_fn = os.path.join(self.checkpoint.save_model_dir,"checkpoint")
        if resume_training and os.path.exists(checkpoint_fn):
            initial_epoch = epoch_tracker.read_epoch_annotation()
            #Load weights to resume training
            self._load_weights(fn = self.checkpoint.filename)
#             try:
#                 self._load_model(fn = os.path.join(self.checkpoint.filename))
#             except:
#                 print("Load Model Failed. Trying with weights only")
#                 self._load_weights(fn = self.checkpoint.filename)
            msg = utils.assign(weight_filename, self.checkpoint.filename)
            logger.info("=> Weights file {:s}\n".format(msg))
            logger.info("=> Training will resume from epoch {0}\n".format(initial_epoch))
        else:
            initial_epoch = 0
            logger.info("=> Training will start from the beginning\n")
        
        self._add_callbacks([epoch_tracker,])
        
        #Time measure
        tcb = TimerCallback()
        self._add_callbacks([tcb,])
        
        STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
        if not val_gen is None:
            STEP_SIZE_VALID=val_gen.n//val_gen.batch_size     
        else:
            STEP_SIZE_VALID = 1
        
        logger.info("=> Start training...\n")
        
        # I need to check the version in order to call the proper method
        if self._tf_version_230():        
            
            history = self.model.fit(
                train_gen,
                epochs = epochs,
                validation_data = val_gen,
                steps_per_epoch = STEP_SIZE_TRAIN,
                validation_steps= STEP_SIZE_VALID,
                callbacks=self.cb,
                workers=2,
                validation_freq = 1,
                initial_epoch = initial_epoch,
                #metrics = self.metrics,
                )
        else:
            history = self.model.fit_generator(
                train_gen,
                epochs = epochs,
                validation_data = val_gen,
                steps_per_epoch = STEP_SIZE_TRAIN,
                validation_steps= STEP_SIZE_VALID,
                callbacks=self.cb,
                workers=2,
                validation_freq = 1,
                initial_epoch = initial_epoch,
                #metrics = self.metrics,
                )            
        
        self.checkpoint.save_history(history)            
            
    @property
    def save_model_dir(self):
        return self.checkpoint.save_model_dir
    
    def eval(self):
        pass
    
    def predict(self):
        pass    
    

    '''
        Returns a model cut at the required level
        Deprecated: use layer_output_by_name instead
    '''
 
    def layer_output(self, layer_friendly_name, k_bit = 0):
        layer = self.friendly_names[layer_friendly_name]
        return self._layer_output(layer, k_bit)
    
    '''
        Returns a model cut at the required level
        @params
            - name: layer name as defined in the model
    '''    
    
    def layer_output_by_name(self, name, k_bit = 0):
        layer = 0
        for l in self.model.layers: 
            if l.name == name: 
                return self._layer_output(layer, k_bit)
            else:
                layer += 1
        
    def _layer_output(self, layer, k_bit = 0):
        
        l = self.model.layers[layer]
        out = l.output
        
        if k_bit > 0:
            out = lq.quantizers.DoReFaQuantizer(k_bit=k_bit)(out)
        out = tf.keras.models.Model(inputs=self.model.input, outputs = out)
            
        return out
    
    def layer_index_by_name(self, name):
        layer = 0
        for l in self.model.layers: 
            if l.name == name: 
                return layer
            else:
                layer += 1        
    
    
    '''
        Interface to add an intermediate layer to the available model to extract inner features
        layer_friendly: the label of the layer.
        layer_model: the keras model built from the inner layer.
        override: if true, the layer model is replaced (if already exists).
    '''
        
    def _add_inner_layer(self, layer_friendly, layer_model, override = True):
        
        if override or not layer_friendly in self.inner_layers:
            self.inner_layers[layer_friendly] = layer_model
            return True
        else:
            return False
        
    '''
        Returns a runnable model to extract inner feature. If it not exists, then it is instantiated and added to self.inner_layers
        layer_frindly: the label identifying the layer to get
        Deprecated: use get_inner_layer_by_name instead
    '''
        
    def get_inner_layer(self, layer_friendly):
        if layer_friendly in self.inner_layers:
            return self.inner_layers[layer_friendly]
        else:
            layer_model = self.layer_output(layer_friendly, k_bit = self.k_bit)
            self._add_inner_layer(layer_friendly, layer_model)
            return layer_model 
        
    '''
        Returns a runnable model to extract inner feature. If it not exists, then it is instantiated and added to self.inner_layers
        layer_name: the layer name as defined in the model
    '''
        
    def get_inner_layer_by_name(self, layer_name, **kwargs):
        if layer_name in self.inner_layers:
            return self.inner_layers[layer_name]
        else:
            layer_model = self.layer_output_by_name(layer_name, k_bit = self.k_bit)
            self._add_inner_layer(layer_name, layer_model)
            return layer_model 
    
    '''
        Initializes the layer index mapping 
    '''
    
    def _init_friendly_names(self):
        self.friendly_names = {}
        
    
    def _set_metrics(self, metrics):
        self.metrics = metrics
    
    
    '''
        Add callbacks to use during the training phase
    '''
    
    def _add_callbacks(self, new_cbs):
        try:
            for cb in new_cbs:
                self.cb.append(cb)
        except:
            self.cb.append(new_cbs)
    '''
        Creates the Keras Model. This method should also compile the model with the proper optimizer, metrics and loss
    '''
    
    def _setup_model(self, **kwargs):
        pass

    '''
        Load weights for the model.
            - fn: the checkpoint filename. If fn is 'None', the internal checkpoint.filename will be used to load the latest model.
    '''
    def _load_weights(self, fn = None):
        if fn is None:
            fn = self.checkpoint.filename
        else:
            fn = fn
        #if os.path.exists(fn):
        self.model.load_weights(fn)
        self.logger.info("=> Weights loaded from {:s}".format(fn))
        
    def _load_model(self,  fn = None):
        
        if fn is None:
            fn = self.checkpoint.filename
        else:
            fn = fn
        #if os.path.exists(fn):
        #tf.keras.backend.clear_session()
        self.model = tf.keras.models.load_model(fn)
        self.logger.info("=> Model loaded from {:s}".format(fn))       
        
    '''
        fit_generator is deprecated since 2.3.0. I need to check the version in order to call the proper method
    '''
    def _tf_version_230(self):
        v = tf.__version__
        t = v.split(".")
        return (int(t[0]) >= 2 and int(t[1]) >= 3)
    
    
    '''
        Saves full model into a directory
        @param format: can be either 'tf' or 'h5' 
    '''
    def save_model(self, fn = None, save_format = 'tf'):
        if fn is None:
            fn = os.path.join(self.checkpoint.save_model_dir, 'best_model.' + save_format)
        else:
            fn = fn
        self.model.save(fn)
         
    '''
        Wrapper for self._load_weights
        Load weights for the model.
            - fn: the checkpoint filename. If fn is 'None', the internal checkpoint.filename will be used to load the latest model.
    '''
            
    def load_weights(self, fn = None):
        self._load_weights(fn = fn)
       
    def load(self, fn = None):
        if self.save_weights_only:
            self._load_weights(fn = fn)
        else:
            self._load_model(fn = fn)
        
    '''
        Configures the logger
    '''
    def _setup_logger(self, level=logging.ERROR):
    #Logging setup
        FORMAT = '%(message)s'
        logging.basicConfig(format = FORMAT, level=logging.INFO)
        logger = logging.getLogger('Logger')
        logger.setLevel(level=level)
        
        return logger
    
    '''
        Plots the metrics set for the model
    '''
    
    def display_history(self):
        fn = self.checkpoint.hist
        if os.path.exists(fn):
            display_history(fn = fn)
        else:
            self.logger.info("=> History file does not exists: {:s}".format(fn))
     
    '''
        This is a single conv block I frequently use . 
        Override is as convenient for your network.
        note:  CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
    '''
            
    def _conv_block(self, units, kernel_size = (3,3), activation='relu', block=1, layer=1):
        
        def layer_wrapper(inp):
            x = lq.layers.QuantConv2D(units, activation=None, kernel_size = kernel_size, padding='same', name='block{}_conv{}'.format(block, layer))(inp)
            x = keras.layers.BatchNormalization(name='block{}_bn{}'.format(block, layer))(x)
            x = keras.layers.Activation(activation, name='block{}_act{}'.format(block, layer))(x)
            return x

        return layer_wrapper
    
    '''
        This is dense block  I frequently use . 
        Override is as convenient for your network.
        note:  CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
    '''
    
    def _dense_block(self, units, activation='relu', name='fc1'):

        def layer_wrapper(inp):
            x = lq.layers.QuantDense(units, name=name, activation=None)(inp)
            x = keras.layers.BatchNormalization(name='{}_bn'.format(name))(x)
            x = keras.layers.Activation(activation, name='{}_act'.format(name))(x)
            return x

        return layer_wrapper    

            
'''
    With respect to ModelWrapper adds methods that are apecific for an autoencoder
'''
            
class AutoencoderWrapper(ModelWrapper):
    
    '''
        Contructor
            @param model_name: the name of the model is part will of the fullpath of the complete working dir
            @Param working_dir: the root of the working folder where models, output, etc.. will be saved
            @param model_name2: is the name for the weight file
            @param logger_lvl: loggin level for the internal logger
            @param **kwargs: is deputed to handle the parameter for the derived classes
    ''' 
    
    def __init__(self, model_name, working_dir, model_name_2 = "checkpoint", logger_lvl = logging.ERROR, **kwargs):
        super(AutoencoderWrapper, self).__init__(model_name, working_dir, model_name_2, logger_lvl = logger_lvl, **kwargs)
        
        #Set specific callbacks
        image_out_dir = os.path.join(self.checkpoint.save_model_dir, "reconstructed")
        #val_gen is set at the beginning of super().fit()
        auto_cb = SampleAutoencoderReconstraction(image_out_dir, generator = None, create_dir = True, clear_old = True)
        self._add_callbacks(auto_cb)
        
        self._latent_space = None
    
    '''
        Returns a Keras model to compute the latent space. The latent space the bottleneck of the model
        @return: a Keras model at the latent space layer, namely the encoder
    '''
    
    def latent_space(self):
        
        out = self._latent_space
        out = tf.keras.models.Model(inputs=self.model.input, outputs = out)
        
        return out
    
    '''
        Returns a Keras model of the decoder stage. It takes a tensor of the shape of the latent space and returns a 
        reconstruction.
        @requires: a trained model
        @return: a keras model of the decoder stage
    '''
    
    def detached_decoder(self):
        
        #create a decoder
        
        #the latent space is a Tensor. I guess it is the output of the corresponding layer
        i_shape = self._latent_space.shape
        
        i = keras.layers.Input(shape = i_shape)
        decoder_ = self._decoder(latent_tensor = i)
        
        decoder = keras.models.Model(i, decoder_)
        
        #Copy the weigths from out to in
        auto_weights = self.model.get_weights()
        decoder_weights = decoder.get_weights()
        c = 1
        for d in range (len(decoder_weights)-1, -1, -1):
#             print(d)
#             print(len(auto_weights)-c)
            decoder_weights[d] = auto_weights[len(auto_weights)-c]
            d += 1
            c += 1
                    
        decoder.set_weights(decoder_weights)  
        
        
        return decoder
    
    '''
        Builds the encoder
        @return the encoder
    '''
    
    def _encoder(self):
        pass
    
    '''
        Builds the decoder
        @return the decoder
    '''    
    
    def _decoder(self):
        pass
    
   
    '''
        Creates the Keras Model. This method should also compile the model with the proper optimizer, metrics and loss
    '''
   
    def _setup_model(self, **kwargs):
        ModelWrapper._setup_model(self, **kwargs)
        
    '''
        Display the structure of the network showing the shape of input and output tensors for each layer.
        This is recommended to debug the autoencoder network.
    '''    
    
    def _display_layers(self, m = None):
        c = 0
        t = PrettyTable(['#','Layer','in','out','Trainable'])
        if m is None:
            model_to_display = self.model
        else:
            model_to_display = m
        for l in model_to_display.layers: 
            t.add_row([str(c), l.name, l.input_shape, l.output_shape, str(l.trainable)])
            c += 1
        print(t)  
        
    
    '''
        This is a single transpose block I frequently use . 
        Override is as convenient for your network.
        note:  CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
    '''
    
    def _transpose_block(self, units, kernel_size = (3,3), activation='relu', name='fc1', block=1, layer=1):
        
        def layer_wrapper(inp):
            x = lq.layers.QuantConv2DTranspose(units, activation=None, kernel_size = kernel_size, padding='same', name='D_block{}_conv{}'.format(block, layer))(inp)
            x = keras.layers.BatchNormalization(name='D_block{}_bn{}'.format(block, layer))(x)
            x = keras.layers.Activation(activation, name='D_block{}_act{}'.format(block, layer))(x)
            return x        
            
        return layer_wrapper
    

if __name__ == '__main__':
    pass
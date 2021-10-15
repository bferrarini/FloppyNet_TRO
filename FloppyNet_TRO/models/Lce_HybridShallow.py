'''
Created on 18 Jan 2020

@author: Bruno Ferrarini
@affiliation Univeristy of Essex, UK

'''

import config

import models.abstract
import larq as lq
import tensorflow as tf
from prettytable import PrettyTable
import tensorflow.keras as keras
from models.extra.Kste import KSteSign

class QuantizedHShallow(models.abstract.ModelWrapper):
    
    def __init__(self, 
                 model_name, 
                 working_dir, 
                 nClasses, 
                 fc_units = 512, 
                 filters = (96,256,256), 
                 enable_history = False, 
                 activation_precision : int = 1, 
                 kernel_precision : int = 1, 
                 model_name_2 = "checkpoint", 
                 logger_lvl = 
                 config.log_lvl, 
                 l_rate = 0.0001, 
                 optimizer = None,
                 loss = 'categorical_crossentropy',
                 **kwargs):
        self.filters = filters
        super(QuantizedHShallow, self).__init__(model_name, working_dir, model_name_2, logger_lvl = logger_lvl, enable_history = enable_history,**kwargs)    
        
        self.nClasses = nClasses
        self.l_rate = l_rate
        self.activation_precision = activation_precision
        self.kernel_precision = kernel_precision 
        self.units = fc_units
        self.optimizer = optimizer
        self.loss = loss
        
        self.model = self._setup_model(verbose = False) 
        

    '''
        Creates a Keras Model.
    '''
    
    def _setup_model(self, **kwargs):
        
        if "verbose" in kwargs:
            verbose = kwargs['verbose']
        else:
            verbose = False
        
        #tf.debugging.set_log_device_placement(True)
        
        self.model_name = self.model_name if self.model_name is not None else 'unknown_model_name'
        
        input_img = keras.layers.Input(shape = (227, 227, 3))
        cnn = self._cnn(input_tensor = input_img)
            
        net = self._fully_connected(self.nClasses, cnn,
                        units = self.units)
    
        model = keras.models.Model(input_img, net)
            
        if self.optimizer is None:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.l_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
                amsgrad=False,
                name='Adam'
                )
            self.optimizer = optimizer
    
        model.compile(optimizer=self.optimizer,
                  loss=self.loss,
                  metrics=['accuracy'])
        
        self.model = model

        if verbose:
            self._display_layers()
        
        return self.model
    
    '''
        Returns the CNN part of the Model
    '''
    def _cnn(self,input_tensor=None, input_shape=None,
                use_bias = False,
                momentum = 0.9):
    

        img_input = keras.layers.Input(shape=input_shape) if input_tensor is None else (
            keras.layers.Input(tensor=input_tensor, shape=input_shape) if not keras.backend.is_keras_tensor(input_tensor) else input_tensor
        )
        
        #Block 1 - The input quantizer have to be set on None
        x = self._conv_block(filters = self.filters[0], kernel_size = (11,11), strides = (4,4), padding = 'valid', name='conv1',
                                            input_quantizer = None,
                                            kernel_quantizer = "ste_sign",
                                            kernel_constraint = "weight_clip",
                                            use_bias = use_bias,
                                            batch_norm = False,
                                            momentum = momentum
                                            )(img_input)
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)            
        #Block 2
        x = self._conv_block(filters = self.filters[1], kernel_size = (5,5), strides = (1,1),  padding = 'same', name='conv2',
                                            input_quantizer = "ste_sign",
                                            kernel_quantizer = "ste_sign",
                                            kernel_constraint = "weight_clip",
                                            use_bias = use_bias,
                                            momentum = momentum)(x)
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)  
        
        #Block 3
        x = self._conv_block(self.filters[2],  kernel_size = (3,3), strides = (1,1), name='conv3',
                                            input_quantizer = "ste_sign",
                                            kernel_quantizer = "ste_sign",
                                            kernel_constraint = "weight_clip",
                                            use_bias = use_bias,
                                            momentum = momentum)(x)
                                    
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)   
              
        x = keras.layers.BatchNormalization(name = 'pool5_bn', momentum = momentum)(x) 
         
        x = keras.layers.Flatten(name='flatten')(x)
        #self._latent_space = x
        
        return x
     
     
    '''
        Returns the FC part of the Model
    ''' 
    def _fully_connected(self, nClasses, cnn, units):
        
        x = self._dense_block(units = units, activation='relu', name='fc6')(cnn)
        x = self._dense_block(units = units, activation='relu', name='fc7')(x)
        x = self._dense_block(units = nClasses, activation='softmax', name='fc8')(x)
        
        return x

    '''
        Convolutional block
    '''
    def _conv_block(self, 
                    filters, 
                    name,
                    kernel_size = (3,3), 
                    strides = (1,1), 
                    padding = 'same',
                    pad_value = 1.0,
                    #activation=None, name = None,
                    input_quantizer = 'ste_sign',
                    kernel_quantizer = 'ste_sign',
                    kernel_constraint=lq.constraints.WeightClip(clip_value=1),
                    use_bias = False,
                    batch_norm = True,
                    momentum = 0.9):
        
        def layer_wrapper(inp):
            x = inp
            if batch_norm:
                x = keras.layers.BatchNormalization(name = name + '_bn', momentum = momentum)(x)
            x = lq.layers.QuantConv2D(filters, kernel_size = kernel_size, strides = strides, padding=padding, pad_values = pad_value, name=name,
                                      input_quantizer = input_quantizer,
                                      kernel_quantizer = kernel_quantizer,
                                      kernel_constraint = kernel_constraint,
                                      use_bias = use_bias)(x)
            
            #x = keras.layers.Activation(activation, name = name + '_act')(x)
            return x

        return layer_wrapper

    
    '''
        Dense block
    '''
    def _dense_block(self, units, activation='relu', name='fc1', use_batch_norm = True):

        def layer_wrapper(inp):
            x = lq.layers.QuantDense(units, name=name)(inp)
            if use_batch_norm:
                x = keras.layers.BatchNormalization(name='bn_{}'.format(name))(x)
            x = keras.layers.Activation(activation, name='act_{}'.format(name))(x)
            #x = keras.layers.Dropout(dropout, name='dropout_{}'.format(name))(x)
            return x

        return layer_wrapper  
        

    '''
        Returns a runnable model to extract inner feature. If it not exists, then it is instantiated and added to self.inner_layers
        layer_name: the layer name as defined in the model
    '''
        
    def get_inner_layer_by_name(self, layer_name, k_bit = None, activation = None, flatten = False):
        layer = self.layer_output_by_name(layer_name)
        # Quantized depends on the network. For LarqAlex is ste_sign
        if activation:
            out = keras.layers.Activation(activation)(layer.output)
        else:
            out = layer.output
        if k_bit:
            out = KSteSign(k_bit=k_bit)(out)
        if flatten:
            out = keras.layers.Flatten()(out)
        model = tf.keras.models.Model(inputs=self.model.input, outputs = out)
    
        return model
    
    def _display_layers(self):
        c = 0
        t = PrettyTable(['#','Layer','in','out','Trainable'])
        for l in self.model.layers: 
            t.add_row([str(c), l.name, l.input_shape, l.output_shape, str(l.trainable)])
            c += 1
        print(t)         

    
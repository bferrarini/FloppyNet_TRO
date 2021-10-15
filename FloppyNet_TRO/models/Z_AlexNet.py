'''
Created on 24 Jun 2020

Correct version of AlexNet with NO Overlapping pools and 227 input images

@author: Bruno Ferrarini
@affiliation Univeristy of Essex, UK

'''

import config

import models.abstract
import larq as lq
import tensorflow as tf
from prettytable import PrettyTable
import tensorflow.keras as keras

class AlexNet(models.abstract.ModelWrapper):
    
    def __init__(self, model_name, working_dir, nClasses, 
                 fc_units = 4096,
                 model_name_2 = "checkpoint", logger_lvl = config.log_lvl, l_rate = 0.0001, **kwargs):
        super(AlexNet, self).__init__(model_name, working_dir, model_name_2, logger_lvl = logger_lvl, **kwargs)
        
        #Set specific callbacs
        self.nClasses = nClasses
        self.l_rate = l_rate
        self.units = fc_units
        self.model = self._setup_model(verbose = True)


    '''
        Creates the Keras Model.
    '''
    
    def _setup_model(self, **kwargs):
        
        if "verbose" in kwargs:
            verbose = kwargs['verbose']
        else:
            verbose = False
        
        #tf.debugging.set_log_device_placement(True)
        
        self.model_name = self.model_name if self.model_name is not None else 'VGGM'
        
        input_img = keras.layers.Input(shape = (227, 227, 3))
        cnn = self._cnn(input_tensor = input_img)
        net = self._fully_connected(self.nClasses, cnn, units = self.units)
    
        model = keras.models.Model(input_img, net)
            
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.l_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
            name='Adam'
            )
    
        model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
        
        self.model = model

        if verbose:
            self._display_layers()
        
        return self.model
        
    '''
        Returns the Encoder part of the Model
    '''
    
    def _cnn(self,input_tensor=None, input_shape=None ,activation='relu'):
    

        img_input = keras.layers.Input(shape=input_shape) if input_tensor is None else (
            keras.layers.Input(tensor=input_tensor, shape=input_shape) if not keras.backend.is_keras_tensor(input_tensor) else input_tensor
        )
        
        #Block 1
        x = self._conv_block(filters = 96, kernel_size = (11,11), strides = (4,4), activation='relu', padding = 'valid', name='conv1')(img_input)
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)        
        
        #Block 2
        x = self._conv_block(filters = 256, kernel_size = (5,5), strides = (1,1), activation='relu', padding = 'same', name='conv2')(x)
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)  
        
        #Block 3
        x = self._conv_block(384,  kernel_size = (3,3), strides = (1,1), activation=activation, name='conv3')(x)
        x = self._conv_block(384, kernel_size = (3,3), strides = (1,1), activation=activation, name='conv4')(x)
        x = self._conv_block(256, kernel_size = (3,3), strides = (1,1), activation=activation, name='conv5')(x)
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)   
              
         
        x = keras.layers.Flatten(name='flatten')(x)
        self._latent_space = x
        encoder = x
        
        return encoder

                     
    def _fully_connected(self, nClasses, cnn, units):
        
        x = self._dense_block(units = units, activation='relu', name='fc6')(cnn)
        x = self._dense_block(units = units, activation='relu', name='fc7')(x)
#         x = self._dense_block(units = 512, activation='relu', name='fc6')(cnn)
#         x = self._dense_block(units = 512, activation='relu', name='fc7')(x)
        x = self._dense_block(units = nClasses, activation='softmax', name='fc8')(x)
        
        return x
                             
    
    def _conv_block(self, filters, kernel_size = (3,3), strides = (1,1), padding = 'same', activation='relu', name = None):
        
        def layer_wrapper(inp):
            x = lq.layers.QuantConv2D(filters, kernel_size = kernel_size, strides = strides, padding=padding, name=name)(inp)
            x = keras.layers.BatchNormalization(name = name + '_bn')(x)
            x = keras.layers.Activation(activation, name = name + '_act')(x)
            return x

        return layer_wrapper
    
    def _dense_block(self, units, activation='relu', name='fc1', use_batch_norm = True):

        def layer_wrapper(inp):
            x = lq.layers.QuantDense(units, name=name)(inp)
            if use_batch_norm:
                x = keras.layers.BatchNormalization(name='bn_{}'.format(name))(x)
            x = keras.layers.Activation(activation, name='act_{}'.format(name))(x)
            #x = keras.layers.Dropout(dropout, name='dropout_{}'.format(name))(x)
            return x

        return layer_wrapper    
     
    
    
    def _display_layers(self):
        c = 0
        t = PrettyTable(['#','Layer','in','out','Trainable'])
        for l in self.model.layers: 
            t.add_row([str(c), l.name, l.input_shape, l.output_shape, str(l.trainable)])
            c += 1
        print(t)         

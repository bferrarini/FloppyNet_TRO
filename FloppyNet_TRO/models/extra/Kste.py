'''
Created on 11 Apr 2020

@author: Bruno Ferrarini
@affiliation Univeristy of Essex, UK

'''

import tensorflow as tf
from larq import utils, math
from larq.quantizers import BaseQuantizer, _clipped_gradient, ste_sign
from collections import OrderedDict
import numpy as np

@utils.register_keras_custom_object
def k1_ste_activation(x):
    return math.sign(x)
         
def k_ste_sign(x, k_bit = 2, clip_value=1.0):

    #x = tf.clip_by_value(x, 0.0, clip_value)

    @tf.custom_gradient
    def _call(x):
        def grad(dy):
            return _clipped_gradient(x, dy, clip_value)

        zeros = tf.zeros_like(x)
         
        n = 2 ** (k_bit)
        n_h = 2 ** (k_bit-1)
         
        dY = 1/ (n-1)
         
        Y = OrderedDict()
        j = 0
        #for i in range(1,n_h,2):
        for i in range(0,n_h-1):
            Y[j] = (1 + 2*i) * dY * tf.math.sign(x)
            j += 1
            #print((1 + 2*i)*dY)
        Y[j] = 1.0 * tf.math.sign(x)
             
        dx = clip_value / n_h
        MASK = OrderedDict()
         
        MASK[0] = tf.math.less(tf.math.abs(x), 1*dx)
         
        for i in range(1, n_h-1):
            A = tf.math.greater(tf.math.abs(x), dx * (i))
            B = tf.math.greater(tf.math.abs(x), dx * (i+1))
            MASK[i] = tf.math.logical_xor(A, B)
            #print("{} - {}".format(dx*i, dx * (i+1)))
             
        MASK[n_h-1] = tf.math.greater(tf.math.abs(x), clip_value - dx)
             
        val = tf.where(MASK[0], Y[0], zeros)
        for j in range(1,len(Y)):
            val += tf.where(MASK[j], Y[j], zeros)
            
        
        return val, grad

    return _call(x)


@utils.register_keras_custom_object
def k_pos_ste(inputs, clip_value=1.0):
    
    #x = tf.clip_by_value(inputs, 0.0, 1.0)
    x = inputs
    
    @tf.custom_gradient
    def _call(x):
        def grad(dy):
            return _clipped_gradient(x, dy, clip_value=clip_value)
        
        zeros = tf.zeros_like(x)
        val = tf.math.maximum(zeros, math.sign(x))
    
        return val, grad
        
    return _call(x) 
    
@utils.register_keras_custom_object
def k_round_ste(inputs, clip_value=1.0):
    
    x = tf.clip_by_value(inputs, 0.0, 1.0)
    #x = inputs
    
    @tf.custom_gradient
    def _call(x):
        def grad(dy):
            return _clipped_gradient(x, dy, clip_value=clip_value)
        
        zeros = tf.zeros_like(x)
        val = tf.math.round(x)
    
        return val, grad
        
    return _call(x)

@utils.register_keras_custom_object
def k_scaled_round_ste(inputs, clip_value=1.0):
    
    x = tf.clip_by_value(inputs, 0.0, 1.0)
    #x = inputs
    
    @tf.custom_gradient
    def _call(x):
        def grad(dy):
            return _clipped_gradient(x, dy, clip_value=clip_value)
        
        #zeros = tf.zeros_like(x)
        val = tf.math.round(x) * 0.25
    
        return val, grad
        
    return _call(x)

@utils.register_keras_custom_object
class KRoundedSte(BaseQuantizer):
    
    precision = None
    
    def __init__(self, k_bit, scale = 1.0, clip_value = 1.0, clip_gradient = True, **kwargs):
        self.precision = k_bit
        self.scale = scale
        self.clip_value = clip_value
        self.clip_gradient = clip_gradient
        super().__init__(**kwargs)
   
    def call(self, inputs):
        x = tf.clip_by_value(inputs, 0.0, 1.0)

        @tf.custom_gradient
        def _call(x):
            def grad(dy):
                if self.clip_gradient:
                    return _clipped_gradient(x, dy, clip_value=self.clip_value)
                else:
                    return dy
        
            n = 2 ** self.precision - 1
            val = self.scale * tf.round(x * n) / n
    
            outputs = (val, grad)
            
            return outputs
    
        return super().call(_call(x))

    def get_config(self):
        config = super(KRoundedSte, self).get_config()
        d = {
            "k_bit": self.precision, 
            "scale" : self.scale, 
            "clip_gradient" : self.clip_gradient, 
            "clip_value" : self.clip_value
            }
        config.update(d)
        return config 
    

@utils.register_keras_custom_object
class KRoundedSte2(BaseQuantizer):
    
    precision = None
    
    def __init__(self, k_bit, scale = 1.0, clip_value = 1.0, clip_gradient = True, **kwargs):
        self.precision = k_bit
        self.scale = scale
        self.clip_value = clip_value
        self.clip_gradient = clip_gradient
        super().__init__(**kwargs)
   
    def call(self, inputs):
        #s = tf.sign(inputs)
        #x = tf.multiply(inputs, s)
        x = tf.clip_by_value(inputs, -1.0, 1.0)
        #x = inputs
        
        @tf.custom_gradient
        def _call(x):
            def grad(dy):
                if self.clip_gradient:
                    return _clipped_gradient(x, dy, clip_value=self.clip_value)
                else:
                    return dy
        
            n = 2 ** (self.precision) - 1
            #centering the quantizaton around the x-axis
            val = (tf.round(x * n) / n)
            #val = tf.multiply(s, val)
            
            #val = tf.sign(x) * val
            val = tf.clip_by_value(val, -1.0, 1.0)
    
            outputs = (val, grad)
            
            return outputs
    
        return super().call(_call(x))

    def get_config(self):
        
        config = super(KRoundedSte2, self).get_config()
        d = {
            "k_bit": self.precision, 
            "scale" : self.scale, 
            "clip_gradient" : self.clip_gradient, 
            "clip_value" : self.clip_value
            }
        config.update(d)
        return config  


def test(x):
    if x < 0:
        return x
    else:
        return 0

@utils.register_keras_custom_object
class KSteSign(BaseQuantizer):
    
    precision = None

    def __init__(self, k_bit : int, clip_value: float = 1.0, **kwargs):
        self.clip_value = clip_value
        self.precision = k_bit
        super().__init__(**kwargs)

    def call(self, inputs):
        if self.precision == 1:
            out = ste_sign(inputs, clip_value=self.clip_value)
        else:
            out = k_ste_sign(inputs, k_bit=self.precision, clip_value=self.clip_value)
        
        return super().call(out)
    

    def get_config(self):
        
        config = super(KSteSign, self).get_config()
        d = {
            "k_bit": self.precision, 
            "clip_value" : self.clip_value
            }
        config.update(d)
        return config
   

@utils.register_keras_custom_object
class MyDoReFaQuantizer(BaseQuantizer):
    
    precision = None

    def __init__(self, k_bit: int = 2, **kwargs):
        self.precision = k_bit
        super().__init__(**kwargs)

    def call(self, inputs):
        inputs = tf.clip_by_value(inputs, 0, 1.0)

        @tf.custom_gradient
        def _k_bit_with_identity_grad(x):
            def grad(dy):
                return _clipped_gradient(x, dy, clip_value=1.0)
            n = 2 ** self.precision - 1
            return (tf.round(x * n)) / n, grad

        outputs = _k_bit_with_identity_grad(inputs)
        return super().call(outputs)

    def get_config(self):
        config = super(MyDoReFaQuantizer, self).get_config()
        d = {
            "k_bit": self.precision, 
            }
        config.update(d)
        return config


@tf.custom_gradient
def scaled_gradient(x: tf.Tensor, scale: float = 1.0) -> tf.Tensor:
    def grad(dy):
        # We don't return a gradient for `scale` as it isn't trainable
        return (dy * scale, 0.0)

    return x, grad


@utils.register_alias("lsq")
@utils.register_keras_custom_object
class LSQ(tf.keras.layers.Layer):
    r"""Instantiates a serializable k_bit quantizer as in the LSQ paper.

    # Arguments
    k_bit: number of bits for the quantization.
    mode: either "signed" or "unsigned", reflects the activation quantization scheme to
        use. When using this for weights, use mode "weights" instead.
    metrics: An array of metrics to add to the layer. If `None` the metrics set in
        `larq.context.metrics_scope` are used. Currently only the `flip_ratio` metric is
        available.

    # Returns
    Quantization function

    # References
    - [Learned Step Size Quantization](https://arxiv.org/abs/1902.08153)
    """
    precision = None

    def __init__(self, k_bit: int = 2, mode="unsigned", **kwargs):
        self.precision = k_bit
        self.mode = mode

        if mode == "unsigned":
            self.q_n = 0.00
            self.q_p = float(2 ** self.precision - 1)
        elif mode in ["signed", "weights"]:
            self.q_p = float(2 ** (self.precision - 1)) - 1

            # For signed, we can use the full signed range, e.g. [-2, 1]
            if mode == "signed":
                self.q_n = -float(2 ** (self.precision - 1))
            # For weights, we use a symmetric range, e.g. [-1, 1]
            else:
                self.q_n = -float(2 ** (self.precision - 1) - 1)

        else:
            raise ValueError(f"LSQ received unknown mode: {mode}")

        super().__init__(**kwargs)

    def build(self, input_shape):
        self.s = self.add_weight(
            name="s",
            initializer="ones",
            trainable=True,
            aggregation=tf.VariableAggregation.MEAN,
        )
        self._initialized = self.add_weight(
            name="initialized",
            initializer="zeros",
            dtype=tf.dtypes.bool,
            trainable=False,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
        )

        # Assuming that by num_features they mean all the individual pixels.
        # You can also try the number of feature maps instead.
        self.g = float(1.0 / np.sqrt(np.prod(input_shape[1:]) * self.q_p))

        super().build(input_shape)

    def call(self, inputs):
        # Calculate initial value for the scale using the first batch
        self.add_update(
            self.s.assign(
                tf.cond(
                    self._initialized,
                    lambda: self.s,  # If already initialized, just use current value
                    # Otherwise, use the value below as initialization
                    lambda: (2.0 * tf.reduce_mean(tf.math.abs(inputs)))
                    / tf.math.sqrt(self.q_p),
                )
            )
        )
        self.add_update(self._initialized.assign(True))
        s = scaled_gradient(self.s, self.g)
        rescaled_inputs = inputs / s
        clipped_inputs = tf.clip_by_value(rescaled_inputs, self.q_n, self.q_p)

        @tf.custom_gradient
        def _round_ste(x):
            return tf.round(x), lambda dy: dy

        return _round_ste(clipped_inputs) * s

    def get_config(self):
        return {**super().get_config(), "k_bit": self.precision, "mode": self.mode}        

if __name__ == '__main__':
    pass
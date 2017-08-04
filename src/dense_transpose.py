import numpy as np
from keras import backend as K
from keras.engine import Layer


class DenseTranspose(Layer):
    def __init__(self, other_layer, units, **kwargs):
        super(DenseTranspose, self).__init__(**kwargs)
        self.other_layer = other_layer
        self.units = units

    def build(self, input_shape):
        self.built = self.other_layer.built

    def call(self, inputs):
        if self.other_layer.use_bias:
            output = K.bias_add(inputs, -self.other_layer.bias)
        output = K.dot(output, self.other_layer.kernel.T)
        # if self.activation is not None:
        #     output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    # def get_config(self):
    #     config = {
    #         'use_bias': self.use_bias
    #     }
    #     base_config = super(DenseTranspose, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items()))
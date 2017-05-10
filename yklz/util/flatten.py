'''Flatten layer supporting mask'''
import keras.backend as K
from keras.layers import Layer

class MaskFlatten(Layer):

    def __init__(self, **kwargs):
        super(MaskFlatten, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(MaskFlatten, self).build(input_shape)

    def compute_mask(self, inputs, mask):
        inputs_shape = K.int_shape(inputs)
        outputs_shape = self.compute_output_shape(inputs_shape)
        mask_output = K.reshape(mask, (-1, outputs_shape[1]))
        mask_tensor = K.all(mask_output, axis=1)
        return mask_tensor

    def call(self, inputs, mask=None):
        inputs_shape = K.int_shape(inputs)
        outputs_shape = self.compute_output_shape(inputs_shape)
        return K.reshape(inputs, (-1, outputs_shape[1]))

    def compute_output_shape(self, input_shape):
        feature_size = 1
        for size in input_shape[1:]:
            feature_size *= size
        return (input_shape[0], feature_size)

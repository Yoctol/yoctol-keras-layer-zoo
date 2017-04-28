'''Customed Mask2D Layer'''
from keras.layers import Masking
import keras.backend as K
import tensorflow as tf

class Mask2D(Masking):

    def __init__(self, mask_value=0., **kwargs):
        super(Mask2D, self).__init__(
            mask_value=mask_value,
            **kwargs
        )

    def compute_mask(self, inputs, mask=None):
        mask_tensor = K.any(K.not_equal(inputs, self.mask_value), axis=-1)
        mask_x = K.any(mask_tensor, axis=1, keepdims=True)
        mask_y = K.any(mask_tensor, axis=2, keepdims=True)
        return tf.logical_and(mask_x, mask_y)

    def call(self, inputs):
        inputs_shape = K.int_shape(inputs)
        masked_tensor = self.compute_mask(inputs)
        masked_tensor = K.expand_dims(masked_tensor)
        masked_tensor = K.repeat_elements(
            masked_tensor, 
            inputs_shape[3],
            3
        )
        return inputs * K.cast(masked_tensor, K.floatx())


'''Customed Mask2D Layer'''
from keras.layers import Masking
import keras.backend as K
import tensorflow as tf

class MaskConv(Masking):

    def __init__(self, mask_value=0., **kwargs):
        super(MaskConv, self).__init__(
            mask_value=mask_value,
            **kwargs
        )

    def compute_mask(self, inputs, mask=None):
        dimension = K.ndim(inputs)
        mask_tensor = K.any(K.not_equal(inputs, self.mask_value), axis=-1)
        mask_base = K.any(mask_tensor, axis=1, keepdims=True)
        for axis in range(2, dimension - 1):
            mask_axis = K.any(mask_tensor, axis=axis, keepdims=True)
            mask_base = tf.logical_and(mask_base, mask_axis)
        return mask_base

    def call(self, inputs):
        inputs_shape = K.int_shape(inputs)
        channel_axis = len(inputs_shape) - 1
        masked_tensor = self.compute_mask(inputs)
        masked_tensor = K.expand_dims(masked_tensor)
        masked_tensor = K.repeat_elements(
            masked_tensor,
            inputs_shape[channel_axis],
            channel_axis
        )
        return inputs * K.cast(masked_tensor, K.floatx())


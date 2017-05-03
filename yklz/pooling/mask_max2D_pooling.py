'''Mask Max 2D pooling'''
from keras.layers.pooling import MaxPool2D
import keras.backend as K
import tensorflow as tf

class MaskedMax2DPooling(MaxPool2D):
    def __init__(self, 
                 pool_size=(2, 2), 
                 strides=None, 
                 padding='valid',
                 data_format=None, 
                 **kwargs):
        super(MaskedMax2DPooling, self).__init__(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
        )
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        inputs_shape = K.int_shape(inputs)
        if self.padding == 'same':
            return NotImplementedError('Please use valid padding.');
        elif self.padding == 'valid':
            mask_tensor = K.cast(mask, K.floatx())
            mask_tensor = K.expand_dims(mask_tensor)
            mask_tensor = K.repeat_elements(
                mask_tensor,
                inputs_shape[3],
                3
            )
            mask_output = K.pool2d(
                mask_tensor, 
                self.pool_size,
                self.strides,
                self.padding,
                self.data_format,
                pool_mode='max'
            )
            mask_output = K.sum(mask_output, axis=3)
            next_mask_tensor = K.not_equal(mask_output, 0.0)
            return next_mask_tensor
        
    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):
        mask_inputs = K.not_equal(inputs, 0.0)

        mask_x = K.any(mask_inputs, axis=1, keepdims=True)
        mask_y = K.any(mask_inputs, axis=2, keepdims=True)
        mask_xy = tf.logical_and(mask_x, mask_y)
        mask_xy_inv = tf.logical_not(mask_xy)

        inputs_mask = K.cast(mask_xy_inv, K.floatx()) * -1e20
        inputs_tensor = inputs + inputs_mask 
        output = K.pool2d(inputs_tensor, pool_size, strides,
                          padding, data_format,
                          pool_mode='max')
        mask_inputs = K.cast(mask_xy, K.floatx())

        mask_output = K.pool2d(
            mask_inputs, 
            self.pool_size,
            self.strides,
            self.padding,
            self.data_format,
            pool_mode='max'
        )
        return output * mask_output

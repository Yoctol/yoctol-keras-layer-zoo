'''Mask pooling'''
import keras.backend as K
from keras.layers.wrappers import Wrapper
import tensorflow as tf

class MaskPooling(Wrapper):
    def __init__(self,
                 layer,
                 pool_mode='max',
                 **kwargs):
        super(MaskPooling, self).__init__(
            layer=layer,
            **kwargs
        )
        self.pool_mode = pool_mode
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        channel_axis = K.ndim(inputs) - 1
        mask_tensor = K.cast(mask, K.floatx())
        mask_tensor = K.expand_dims(mask_tensor)
        mask_output = self.layer._pooling_function(
            mask_tensor,
            self.layer.pool_size,
            self.layer.strides,
            self.layer.padding,
            self.layer.data_format,
        )
        mask_output = K.sum(mask_output, axis=channel_axis)
        next_mask_tensor = K.not_equal(mask_output, 0.0)
        return next_mask_tensor

    def build(self, input_shape):
        self.layer.build(input_shape)
        super(MaskPooling, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def call(self, inputs, mask=None):
        inputs_tensor = inputs
        mask_inputs = K.expand_dims(mask)

        inputs_shape = K.int_shape(inputs)
        channel_axis = len(inputs_shape) - 1

        if self.pool_mode == 'max':
            mask_inv = tf.logical_not(mask_inputs)
            negative_mask = K.cast(mask_inv, K.floatx()) * -1e20
            negative_mask = K.repeat_elements(
                negative_mask,
                inputs_shape[channel_axis],
                channel_axis
            )
            inputs_tensor = inputs + negative_mask

        output = self.layer._pooling_function(
            inputs_tensor,
            self.layer.pool_size,
            self.layer.strides,
            self.layer.padding,
            self.layer.data_format,
        )
        mask_inputs = K.cast(mask_inputs, K.floatx())

        mask_output = self.layer._pooling_function(
            mask_inputs,
            self.layer.pool_size,
            self.layer.strides,
            self.layer.padding,
            self.layer.data_format,
        )
        mask_output = K.repeat_elements(
            mask_output,
            inputs_shape[channel_axis],
            channel_axis
        )
        return output * mask_output

    def get_config(self):
        config = {
            'pool_mode': self.pool_mode,
        }
        base_config = super(MaskPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

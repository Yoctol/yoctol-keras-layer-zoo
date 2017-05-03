'''Customed Convolution 2D layers which provided mask function'''
import keras.backend as K
from keras.layers import Conv2D

class Convolution2D(Conv2D):

    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                **kwargs):
        super(Convolution2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self.supports_masking = True
  
    def build(self, input_shape):
        super(Convolution2D, self).build(input_shape)
        mask_kernel_shape = self.kernel_size + (1, 1)
        self.mask_kernel = K.ones(mask_kernel_shape)

    def compute_mask(self, inputs, mask):
        if self.padding == 'same':
            raise NotImplementedError('Please use valid padding.');
        elif self.padding == 'valid':
            mask_tensor = K.cast(mask, K.floatx())
            mask_tensor = K.expand_dims(mask_tensor)
            mask_output = K.conv2d(
                mask_tensor,
                self.mask_kernel,
                self.strides,
                self.padding,
                self.data_format,
                self.dilation_rate
            )
            mask_output = K.sum(mask_output, axis=3)
            next_mask_tensor = K.not_equal(mask_output, 0.0)
            return next_mask_tensor

    def call(self, inputs, mask=None):
        outputs = super(Convolution2D, self).call(inputs)
        mask_tensor = K.cast(mask, K.floatx())
        mask_tensor = K.expand_dims(mask_tensor)
        mask_output = K.conv2d(
            mask_tensor,
            self.mask_kernel,
            self.strides,
            self.padding,
            self.data_format,
            self.dilation_rate            
        )
        mask_output = K.repeat_elements(mask_output, self.filters, 3)
        return outputs * mask_output
 

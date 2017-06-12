'''Mask ConvNet'''
import keras.backend as K
from keras.layers.wrappers import Wrapper

class MaskConvNet(Wrapper):

    def __init__(self,
                 layer,
                **kwargs):
        super(MaskConvNet, self).__init__(
            layer,
            **kwargs
        )
        self.supports_masking = True

    def build(self, input_shape):
        self.layer.build(input_shape)
        mask_kernel_shape = self.layer.kernel_size + (1, 1)
        self.mask_kernel = K.ones(mask_kernel_shape)
        super(MaskConvNet, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def compute_mask(self, inputs, mask):
        channel_axis = K.ndim(inputs) - 1
        mask_tensor = K.cast(mask, K.floatx())
        mask_tensor = K.expand_dims(mask_tensor)

        mask_output = self._compute_mask_output(mask_tensor)
        mask_output = K.sum(mask_output, axis=channel_axis)
        next_mask_tensor = K.not_equal(mask_output, 0.0)
        return next_mask_tensor

    def call(self, inputs, mask=None):
        outputs = self.layer.call(inputs)
        channel_axis = K.ndim(inputs) - 1
        mask_tensor = K.cast(mask, K.floatx())
        mask_tensor = K.expand_dims(mask_tensor)

        mask_output = self._compute_mask_output(mask_tensor)
        mask_output = K.repeat_elements(
            mask_output,
            self.layer.filters,
            channel_axis
        )
        return outputs * mask_output

    def _compute_mask_output(self, mask_tensor):
        if self.layer.rank == 1:
            mask_output = K.conv1d(
                mask_tensor,
                self.mask_kernel,
                self.layer.strides[0],
                self.layer.padding,
                self.layer.data_format,
                self.layer.dilation_rate[0]
            )
        if self.layer.rank == 2:
            mask_output = K.conv2d(
                mask_tensor,
                self.mask_kernel,
                self.layer.strides,
                self.layer.padding,
                self.layer.data_format,
                self.layer.dilation_rate
            )
        if self.layer.rank == 3:
            mask_output = K.conv3d(
                mask_tensor,
                self.mask_kernel,
                self.layer.strides,
                self.layer.padding,
                self.layer.data_format,
                self.layer.dilation_rate
            )
        return mask_output

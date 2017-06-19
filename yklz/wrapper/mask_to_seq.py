'''Mask 2D tensor to time stamp sequence'''
import keras.backend as K
from keras.layers.wrappers import Wrapper

class MaskToSeq(Wrapper):

    def __init__(self, layer, time_axis=1, **kwargs):
        super(MaskToSeq, self).__init__(layer, **kwargs)
        self.time_axis = time_axis
        self.supports_masking = True

    def build(self, input_shape):
        super(MaskToSeq, self).build(input_shape)

        self.permute_pattern = [i for i in range(len(input_shape) - 1)]
        self.permute_pattern[self.time_axis] = 1
        self.permute_pattern[1] = self.time_axis

    def compute_mask(self, inputs, mask=None):
        mask_tensor = self.layer.compute_mask(inputs, mask)
        mask_shape = K.int_shape(mask_tensor)

        mask_tensor = K.permute_dimensions(
            mask_tensor,
            self.permute_pattern
        )

        reduce_time = len(mask_shape) - 2
        for _ in range(reduce_time):
            mask_tensor = K.any(mask_tensor, -1)
        return mask_tensor

    def call(self, inputs, mask=None):
        input_shape = K.int_shape(inputs)
        outputs = self.layer.call(inputs)
        outputs = K.permute_dimensions(
            outputs,
            self.permute_pattern + [len(input_shape) - 1]
        )
        outputs_shape = self.compute_output_shape(input_shape)
        outputs = K.reshape(
            outputs,
            (-1, outputs_shape[1], outputs_shape[2])
        )

        mask_tensor = self.compute_mask(
            inputs,
            mask
        )
        mask_tensor = K.cast(mask_tensor, K.floatx())
        mask_tensor = K.expand_dims(mask_tensor)
        mask_output = K.repeat_elements(
            mask_tensor,
            outputs_shape[2],
            2
        )
        return outputs * mask_output

    def compute_output_shape(self, input_shape):
        feature_size = 1
        for axis, size in enumerate(input_shape):
            if axis == 0 or axis == self.time_axis:
                continue
            feature_size *= size
        return (input_shape[0], input_shape[self.time_axis], feature_size)

    def get_config(self):
        config = {
            'time_axis': self.time_axis,
        }
        base_config = super(MaskToSeq, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

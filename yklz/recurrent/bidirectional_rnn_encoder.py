import inspect

import keras.backend as K
from keras.layers.wrappers import Bidirectional
import tensorflow as tf

class BidirectionalRNNEncoder(Bidirectional):

    def __init__(
        self,
        layer,
        merge_mode='concat',
        weights=None,
        **kwargs
    ):
        layer.return_sequences = True
        super(BidirectionalRNNEncoder, self).__init__(
            layer,
            merge_mode,
            weights,
            **kwargs
        )

    def build(self, input_shape):
        super(BidirectionalRNNEncoder, self).build(
            input_shape
        )

    def call(self, inputs, training=None, mask=None):
        kwargs = {}
        func_args = inspect.getfullargspec(self.layer.call).args
        if 'training' in func_args:
            kwargs['training'] = training
        if 'mask' in func_args:
            kwargs['mask'] = mask

        y = self.forward_layer.call(inputs, **kwargs)
        y_rev = self.backward_layer.call(inputs, **kwargs)
        if self.merge_mode == 'concat':
            output = K.concatenate([y, y_rev])
        elif self.merge_mode == 'sum':
            output = y + y_rev
        elif self.merge_mode == 'ave':
            output = (y + y_rev) / 2
        elif self.merge_mode == 'mul':
            output = y * y_rev
        elif self.merge_mode is None:
            output = [y, y_rev]

        units = self.forward_layer.units
        if self.merge_mode == 'concat' or self.merge_mode is None:
            units *= 2
        inputs_shape = K.shape(inputs)
        zeros = tf.zeros(
            shape=[
                inputs_shape[0],
                inputs_shape[1] - 1,
                units
            ]
        )
        output = K.reshape(
            tf.slice(output, [0, inputs_shape[1] - 1,0], [-1, 1, -1]),
            shape=(inputs_shape[0], 1, units)
        )
        output = K.concatenate([output, zeros], axis=1)

        # Properly set learning phase
        if 0 < self.layer.dropout + self.layer.recurrent_dropout:
            if self.merge_mode is None:
                for out in output:
                    out._uses_learning_phase = True
            else:
                output._uses_learning_phase = True
        return output

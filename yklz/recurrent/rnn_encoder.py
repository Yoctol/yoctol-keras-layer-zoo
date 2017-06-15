'''The RNNEncoder Layer wrappers recurrent Layers used in Keras to encoded
input sequence into an encoded vector with padding zero vectors. You have
to set the return_sequences parameter in your recurrent unit to true to
perform mask function.'''
import keras.backend as K
from keras.layers.wrappers import Wrapper
import tensorflow as tf

class RNNEncoder(Wrapper):
    def __init__(self, layer, **kwargs):
        super(RNNEncoder, self).__init__(
            layer,
            **kwargs
        )
        self.supports_masking = True

    def build(self, input_shape):
        self.layer.return_sequences = True
        self.layer.build(input_shape)
        super(RNNEncoder, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(
            input_shape
        )

    def compute_mask(self, inputs, mask):
        return self.layer.compute_mask(
            inputs=inputs,
            mask=mask,
        )

    def call(self, inputs, mask=None, initial_state=None, training=None):
        inputs_shape = K.shape(inputs)
        zeros = tf.zeros(
            shape=[
                inputs_shape[0],
                inputs_shape[1] - 1,
                self.layer.units
            ]
        )
        outputs = self.layer.call(
            inputs=inputs,
            mask=mask,
            initial_state=initial_state,
            training=training
        )
        outputs = K.reshape(
            tf.slice(outputs, [0, inputs_shape[1] - 1, 0], [-1, 1, -1]),
            shape=(inputs_shape[0], 1, self.layer.units)
        )
        outputs = K.concatenate([outputs, zeros], axis=1)

        if 0 < self.layer.dropout + self.layer.recurrent_dropout:
            outputs._uses_learning_phase = True
        return outputs

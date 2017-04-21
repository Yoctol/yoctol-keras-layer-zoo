import keras.backend as K
import tensorflow as tf

from .lstm_cell import LSTMCell

class LSTMEncoder(LSTMCell):
    def __init__(
            self,
            output_units,
            **kwargs
        ):
        kwargs['return_sequences'] = False
        kwargs['output_units'] = output_units
        super(LSTMEncoder, self).__init__(**kwargs)

    def call(self, inputs, mask=None, initial_state=None, training=None):
        inputs_shape = K.shape(inputs)
        zeros = tf.zeros(
            shape=[
                inputs_shape[0], 
                inputs_shape[1] - 1, 
                self.units
            ]
        )
        outputs = super(LSTMEncoder, self).call(
            inputs=inputs,
            mask=mask,
            initial_state=initial_state,
            training=training
        )
        outputs = K.reshape(
            outputs, 
            shape=(inputs_shape[0], 1, self.units)
        )
        return K.concatenate([outputs, zeros], axis=1)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        return (input_shape[0], input_shape[1], self.units)

    def compute_mask(self, inputs, mask):
        return mask

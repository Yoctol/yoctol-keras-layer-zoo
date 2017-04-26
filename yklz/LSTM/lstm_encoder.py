import keras.backend as K
import tensorflow as tf

from .lstm_cell import LSTMCell

class LSTMEncoder(LSTMCell):
    def __init__(
            self,
            output_units,
            **kwargs
        ):
        kwargs['return_sequences'] = True
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
            tf.slice(outputs, [0, inputs_shape[1] - 1,0], [-1, 1, -1]), 
            shape=(inputs_shape[0], 1, self.units)
        )
        outputs = K.concatenate([outputs, zeros], axis=1)

        if 0 < self.dropout + self.recurrent_dropout:
            outputs._uses_learning_phase = True
        return outputs

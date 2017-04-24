import keras.backend as K
import tensorflow as tf

from .lstm_cell import LSTMCell

class LSTMDecoder(LSTMCell):
    def __init__(
            self,
            output_units,
            **kwargs
        ):
        kwargs['return_sequences'] = True
        kwargs['output_units'] = output_units
        super(LSTMDecoder, self).__init__(**kwargs)

    def step(self, inputs, states):
        y_tm1 = states[0]
        
        inputs_sum = tf.reduce_sum(inputs)
        
        def inputs_f(): return inputs
        def output_f(): return y_tm1
        current_inputs = tf.case(
            [(tf.equal(inputs_sum, 0.0), output_f)], 
            default=inputs_f
        )

        return super(LSTMDecoder, self).step(
            current_inputs,
            states
        )        

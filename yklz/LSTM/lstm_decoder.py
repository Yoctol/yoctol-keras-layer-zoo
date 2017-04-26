import keras.backend as K
import tensorflow as tf

from .lstm_cell import LSTMCell

class LSTMDecoder(LSTMCell):
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 use_output_bias=True, 
                 output_activation='tanh',
                 output_dropout=0.,
                 **kwargs):
        kwargs['return_sequences'] = True
        super(LSTMDecoder, self).__init__(
            units=units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            unit_forget_bias=unit_forget_bias,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            use_output_bias=use_output_bias,
            output_activation=output_activation,
            output_dropout=output_dropout,
            **kwargs
        )

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

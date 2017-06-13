from keras.layers.recurrent import LSTM
from keras.layers import activations
from keras.engine import InputSpec
import keras.backend as K

from .lstm_peephole import LSTMPeephole

class LSTMCell(LSTMPeephole):
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
        self.use_output_bias = use_output_bias
        self.output_dropout = min(1., max(0., output_dropout))
        self.output_activation = activations.get(output_activation)

        super(LSTMCell, self).__init__(
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
            **kwargs
        )

    def build(self, input_shape):
        self.output_kernel = self.add_weight(
            shape=(self.units, self.units),
            name='output_kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )

        if self.use_output_bias:
            self.output_bias = self.add_weight(
                shape=(self.units,),
                name='output_bias',
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint
            )
        else:
            self.output_bias = None

        super(LSTMCell, self).build(input_shape)
        batch_size = input_shape[0] if self.stateful else None
        self.state_spec.append(
            InputSpec(shape=(batch_size, self.units))
        )
        self.states.append(None)


    def get_constants(self, inputs, training=None):
        constants = super(LSTMCell, self).get_constants(
            inputs=inputs,
            training=training
        )

        if 0 < self.output_dropout < 1:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))

            def dropped_inputs():
                return K.dropout(ones, self.output_dropout)
            out_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training)]
            constants.append(out_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.)])

        return constants

    def step(self, inputs, states):
        y_tm1 = states[0]
        h_tm1 = states[1]
        c_tm1 = states[2]
        dp_mask = states[3]
        rec_dp_mask = states[4]
        out_dp_mask = states[5]

        h, new_states = super(LSTMCell, self).step(
            inputs,
            [h_tm1, c_tm1, dp_mask, rec_dp_mask]
        )
        _, c = new_states

        y = K.dot(h * out_dp_mask[0], self.output_kernel)
        if self.output_bias is not None:
            y = y + self.output_bias

        y = self.output_activation(y)
        if 0 < self.dropout + self.recurrent_dropout + self.output_dropout:
            y._uses_learning_phase = True
        return y, [y, h, c]

    def get_config(self):
        config = {
            'use_output_bias': self.use_output_bias,
            'output_dropout': self.output_dropout,
            'output_activation': self.output_activation,
        }
        base_config = super(LSTMCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

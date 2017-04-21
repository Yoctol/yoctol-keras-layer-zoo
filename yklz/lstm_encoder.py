from keras.layers.recurrent import LSTM
from keras.layers import activations
from keras.engine import InputSpec
import keras.backend as K
import tensorflow as tf

class LSTMEncoder(LSTM):
    def __init__(
            self, 
            output_units, 
            use_output_bias=True, 
            output_activation='relu',
            output_dropout=0.,
            **kwargs
        ):
        self.output_units = output_units
        self.use_output_bias = use_output_bias
        self.output_dropout = min(1., max(0., output_dropout))
        self.output_activation = activations.get(output_activation)

        kwargs['return_sequences'] = False
        kwargs['units'] = output_units
        super(LSTMEncoder, self).__init__(**kwargs)

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

        super(LSTMEncoder, self).build(input_shape)
        batch_size = input_shape[0] if self.stateful else None
        self.state_spec.append(
            InputSpec(shape=(batch_size, self.units))
        )
        self.states.append(None)


    def get_constants(self, inputs, training=None):
        constants = super(LSTMEncoder, self).get_constants(
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

    def step(self, inputs, states):
        y_tm1 = states[0]
        h_tm1 = states[1]
        c_tm1 = states[2]
        dp_mask = states[3]
        rec_dp_mask = states[4]
        out_dp_mask = states[5]

        h, new_states = super(LSTMEncoder, self).step(
            inputs, 
            [h_tm1, c_tm1, dp_mask, rec_dp_mask]
        )
        _, c = new_states

        y = K.dot(h * out_dp_mask[0], self.output_kernel)
        if self.output_bias is not None:
            y = y + self.output_bias

        y = self.output_activation(y)
        return y, [y, h, c]

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        return (input_shape[0], input_shape[1], self.units)

    def compute_mask(self, inputs, mask):
        return mask

    def get_config(self):
        config = {
            'output_units': self.output_units,
            'use_output_bias': self.use_output_bias,
            'output_dropout': self.output_dropout,
            'output_activation': self.output_activation,
        }
        base_config = super(LSTMEncoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

from keras.layers.recurrent import LSTM
import keras.backend as K

class LSTMPeephole(LSTM):
    def __init__(self, **kwargs):
        super(LSTMpeephole, self).__init__(**kwargs)

    def build(self, input_shape):
        self.P = self.add_weight(
            (self.units, self.units * 4),
            name='P',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )
        self.P_i = self.P[:, :self.units]
        self.P_f = self.P[:, self.units: 2 * self.units]
        self.P_c = self.P[:, 2 * self.units: 3 * self.units]
        self.P_o = self.P[:, 3 * self.units:]
        super(LSTMpeephole, self).build(input_shape)

    def step(self, inputs, states):
        h_tm1 = states[0]
        c_tm1 = states[1]
        dp_mask = states[2]
        rec_dp_mask = states[3]

        if self.implementation == 2:
            z = K.dot(inputs * dp_mask[0], self.kernel)
            z += K.dot(h_tm1 * rec_dp_mask[0], self.recurrent_kernel)
            z += K.dot(c_tm1, self.P)
            if self.use_bias:
                z = K.bias_add(z, self.bias)

            z0 = z[:, :self.units]
            z1 = z[:, self.units: 2 * self.units]
            z2 = z[:, 2 * self.units: 3 * self.units]
            z3 = z[:, 3 * self.units:]

            i = self.recurrent_activation(z0)
            f = self.recurrent_activation(z1)
            c = f * c_tm1 + i * self.activation(z2)
            o = self.recurrent_activation(z3)
        else:
            if self.implementation == 0:
                x_i = inputs[:, :self.units]
                x_f = inputs[:, self.units: 2 * self.units]
                x_c = inputs[:, 2 * self.units: 3 * self.units]
                x_o = inputs[:, 3 * self.units:]
            elif self.implementation == 1:
                x_i = K.dot(inputs * dp_mask[0], self.kernel_i) + self.bias_i
                x_f = K.dot(inputs * dp_mask[1], self.kernel_f) + self.bias_f
                x_c = K.dot(inputs * dp_mask[2], self.kernel_c) + self.bias_c
                x_o = K.dot(inputs * dp_mask[3], self.kernel_o) + self.bias_o
            else:
                raise ValueError('Unknown `implementation` mode.')

            i = self.recurrent_activation(
                x_i + K.dot(h_tm1 * rec_dp_mask[0], self.recurrent_kernel_i)
                + K.dot(c_tm1, self.P_i)
            )
            f = self.recurrent_activation(
                x_f + K.dot(h_tm1 * rec_dp_mask[1], self.recurrent_kernel_f)
                + K.dot(c_tm1, self.P_f)
            )
            c = f * c_tm1 + i * self.activation(
                x_c + K.dot(h_tm1 * rec_dp_mask[2], self.recurrent_kernel_c)
                + K.dot(c_tm1, self.P_c)
            )
            o = self.recurrent_activation(
                x_o + K.dot(h_tm1 * rec_dp_mask[3], self.recurrent_kernel_o)
                + K.dot(c_tm1, self.P_o)
            )
        h = o * self.activation(c)
        if 0 < self.dropout + self.recurrent_dropout:
            h._uses_learning_phase = True
        return h, [h, c]

    def compute_output_shape(self, input_shape):
        return super(LSTMpeephole, self).compute_output_shape(input_shape)

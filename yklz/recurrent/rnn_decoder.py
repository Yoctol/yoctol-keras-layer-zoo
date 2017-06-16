'''The RNNDecoder Layer wrappers recurrent Layers used in Keras to decoded
output sequence from our RNNEncoder, BidirectionalRNNEncoder and ConvEncoder
wrappers.'''
import keras.backend as K
from keras.layers.wrappers import Wrapper
import tensorflow as tf

import yklz.backend as YK

class RNNDecoder(Wrapper):
    def __init__(
            self,
            layer,
            time_steps=None,
            **kwargs
        ):
        super(RNNDecoder, self).__init__(
            layer,
            **kwargs
        )
        self.layer.return_sequences = True
        self.time_steps = time_steps
        self.supports_masking = True

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)
        super(RNNDecoder, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        output_shape = self.layer.compute_output_shape(
            input_shape
        )

        if self.time_steps is None:
            return output_shape
        else:
            return (
                output_shape[0],
                self.time_steps,
                output_shape[2]
            )

    def compute_mask(self, inputs, mask):
        output_mask =  self.layer.compute_mask(
            inputs=inputs,
            mask=mask,
        )

        if self.time_steps is None:
            return output_mask
        else:
            output_mask = K.ones_like(output_mask)
            output_mask = K.any(output_mask, axis=1, keepdims=True)
            return K.tile(output_mask, [1, self.time_steps])

    def call(
        self,
        inputs,
        mask=None,
        training=None,
        initial_state=None
    ):
        if isinstance(inputs, list):
            initial_state = inputs[1:]
            inputs = inputs[0]
        elif initial_state is not None:
            pass
        elif self.layer.stateful:
            initial_state = self.layer.states
        else:
            initial_state = self.layer.get_initial_state(inputs)

        if isinstance(mask, list):
            mask = mask[0]

        if len(initial_state) != len(self.layer.states):
            raise ValueError('Layer has ' + str(len(self.layer.states)) +
                             ' states but was passed ' +
                             str(len(initial_state)) +
                             ' initial states.')
        input_shape = K.int_shape(inputs)
        if self.layer.unroll and input_shape[1] is None:
            raise ValueError('Cannot unroll a RNN if the '
                             'time dimension is undefined. \n'
                             '- If using a Sequential model, '
                             'specify the time dimension by passing '
                             'an `input_shape` or `batch_input_shape` '
                             'argument to your first layer. If your '
                             'first layer is an Embedding, you can '
                             'also use the `input_length` argument.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a `shape` '
                             'or `batch_shape` argument to your Input layer.')
        constants = self.layer.get_constants(inputs, training=None)
        preprocessed_input = self.layer.preprocess_input(inputs, training=None)
        step_function = self.step_with_training(training)
        last_output, outputs, states = YK.rnn_decoder(step_function,
                                             preprocessed_input,
                                             initial_state,
                                             go_backwards=self.layer.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.layer.unroll,
                                             input_length=input_shape[1],
                                             time_steps=self.time_steps)
        if self.layer.stateful:
            updates = []
            for i in range(len(states)):
                updates.append((self.states[i], states[i]))
            self.layer.add_update(updates, inputs)

        # Properly set learning phase
        if 0 < self.layer.dropout + self.layer.recurrent_dropout:
            last_output._uses_learning_phase = True
            outputs._uses_learning_phase = True

        return outputs

    def step_with_training(self, training=None):

        def step(inputs, states):
            input_shape = K.int_shape(inputs)
            y_tm1 = self.layer.preprocess_input(
                K.expand_dims(states[0], axis=1),
                training
            )
            y_tm1 = K.reshape(y_tm1, (-1, input_shape[-1]))

            inputs_sum = tf.reduce_sum(inputs)

            def inputs_f(): return inputs
            def output_f(): return y_tm1
            current_inputs = tf.case(
                [(tf.equal(inputs_sum, 0.0), output_f)],
                default=inputs_f
            )

            return self.layer.step(
                current_inputs,
                states
            )

        return step

    def get_config(self):
        config = {
            'time_steps': self.time_steps,
        }
        base_config = super(RNNDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

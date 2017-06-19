import numpy as np

from keras.layers import activations
from keras.layers import initializers
from keras.layers import regularizers
from keras.layers import constraints
from keras.engine import InputSpec
from keras.layers import Layer
import keras.backend as K

import yklz.backend as YK
from .lstm_peephole import LSTMPeephole

class RNNCell(Layer):
    def __init__(self,
                 recurrent_layer,
                 dense_layer,
                 dense_dropout=0.,
                 go_backwards=False,
                 **kwargs):
        super(RNNCell, self).__init__(
            **kwargs
        )
        self.recurrent_layer = recurrent_layer
        self.dense_layer = dense_layer
        self.dense_dropout = min(1., max(0., dense_dropout))
        self.supports_masking = True

        self.dense_state_spec = None
        self.dense_state = None

    @property
    def return_sequences(self):
        return self.recurrent_layer.return_sequences

    @return_sequences.setter
    def return_sequences(self, return_sequences):
        self.recurrent_layer.return_sequences = return_sequences

    @property
    def return_state(self):
        return self.recurrent_layer.return_state

    @property
    def go_backwards(self):
        return self.recurrent_layer.go_backwards

    @go_backwards.setter
    def go_backwards(self, go_backwards):
        self.recurrent_layer.go_backwards = go_backwards

    @property
    def stateful(self):
        return self.recurrent_layer.stateful

    @property
    def unroll(self):
        return self.recurrent_layer.unroll

    @property
    def implementation(self):
        return self.recurrent_layer.implementation

    @property
    def state_spec(self):
        return [self.dense_state_spec] + self.recurrent_layer.state_spec

    @property
    def dropout(self):
        return self.recurrent_layer.dropout + self.dense_dropout

    @property
    def recurrent_dropout(self):
        return self.recurrent_layer.recurrent_dropout

    @property
    def states(self):
        return [self.dense_state] + self.recurrent_layer.states

    @states.setter
    def states(self, states):
        self.recurrent_layer.states = states[1:]
        self.dense_state = states[0]

    @property
    def units(self):
        return self.dense_layer.units

    @property
    def trainable_weights(self):
        return self.recurrent_layer.trainable_weights + \
               self.dense_layer.trainable_weights

    @property
    def non_trainable_weights(self):
        return self.recurrent_layer.non_trainable_weights + \
               self.dense_layer.non_trainable_weights

    @property
    def updates(self):
        update = []
        if hasattr(self.recurrent_layer, 'updates'):
            update += self.recurrent_layer.update
        if hasattr(self.dense_layer, 'updates'):
            update += self.dense_layer.update
        return update

    def get_updates_for(self, inputs=None):
        update = []
        update += self.recurrent_layer.get_updates_for(inputs)
        update += self.dense_layer.get_updates_for(inputs)
        return update

    @property
    def losses(self):
        loss = []
        if hasattr(self.recurrent_layer.layer, 'losses'):
            loss += self.recurrent_layer.losses
        if hasattr(self.dense_layer.layer, 'losses'):
            loss += self.dense_layer.losses
        return loss

    def get_losses_for(self, inputs=None):
        losses = []
        losses += self.recurrent_layer.get_losses_for(inputs)
        losses += self.dense_layer.get_losses_for(inputs)
        return losses

    @property
    def constraints(self):
        constraints = {}
        if hasattr(self.recurrent_layer, 'constraints'):
            constraints.update(self.recurrent_layer.constraints)
        if hasattr(self.dense_layer, 'constraints'):
            constraints.update(self.dense_layer.constraints)
        return self.dense_layer.constraints

    def get_weights(self):
        return self.recurrent_layer.get_weights() + \
               self.dense_layer.get_weights()

    def set_weights(self, weights):
        self.recurrent_layer.set_weights(weights[0])
        self.dense_layer.set_weights(weights[1])

    @property
    def activity_regularizer(self):
        if hasattr(self.dense_layer.layer, 'activity_regularizer'):
            return self.dense_layer.activity_regularizer
        else:
            return None

    def compute_output_shape(self, input_shape):
        recurrent_output_shapes = self.recurrent_layer.compute_output_shape(
            input_shape
        )
        if self.return_sequences:
            time_stamp_size = recurrent_output_shapes[1]
            dense_output_shapes = self.dense_layer.compute_output_shape(
                (
                    recurrent_output_shapes[0],
                    recurrent_output_shapes[2]
                )
            )
            return (
                dense_output_shapes[0],
                time_stamp_size,
                dense_output_shapes[1]
            )
        else:
            dense_output_shapes = self.dense_layer.compute_output_shape(
                recurrent_output_shapes
            )
            return dense_output_shapes

    def compute_mask(self, inputs, mask):
        return self.recurrent_layer.compute_mask(
            inputs,
            mask
        )

    def get_initial_state(self, inputs):
        dense_initial_state = K.zeros_like(inputs)
        dense_initial_state = K.sum(dense_initial_state, axis=(1, 2))
        dense_initial_state = K.expand_dims(dense_initial_state)
        dense_initial_state = K.tile(dense_initial_state, [1, self.dense_layer.units])
        return [dense_initial_state] + self.recurrent_layer.get_initial_state(inputs)

    def preprocess_input(self, inputs, training=None):
        return self.recurrent_layer.preprocess_input(inputs, training)

    def __call__(self, inputs, initial_state=None, **kwargs):
        if initial_state is None:
            return super(RNNCell, self).__call__(inputs, **kwargs)

        if not isinstance(initial_state, (list, tuple)):
            initial_state = [initial_state]

        is_keras_tensor = hasattr(initial_state[0], '_keras_history')
        for tensor in initial_state:
            if hasattr(tensor, '_keras_history') != is_keras_tensor:
                raise ValueError('The initial state of an RNN layer cannot be'
                                 ' specified with a mix of Keras tensors and'
                                 ' non-Keras tensors')

        if is_keras_tensor:
            # Compute the full input spec, including state
            input_spec = self.recurrent_layer.input_spec
            state_spec = self.recurrent_layer.state_spec
            if not isinstance(input_spec, list):
                input_spec = [input_spec]
            if not isinstance(state_spec, list):
                state_spec = [state_spec]
            self.recurrent_layer.input_spec = input_spec + state_spec

            # Compute the full inputs, including state
            inputs = [inputs] + list(initial_state)

            # Perform the call
            output = super(RNNCell, self).__call__(inputs, **kwargs)

            # Restore original input spec
            self.recurrent_layer.input_spec = input_spec
            return output
        else:
            kwargs['initial_state'] = initial_state
            return super(RNNCell, self).__call__(inputs, **kwargs)

    def reset_states(self, states=None):
        if states is None:
            self.recurrent_layer.reset_states(states)
        else:
            self.recurrent_layer.reset_states(states[:-1])

        batch_size = self.recurrent_layer.input_spec[0].shape[0]
        if self.dense_state is None:
            self.dense_state = K.zeros((
                batch_size,
                self.dense_layer.units
            ))
        elif states is None:
            K.set_value(
                self.dense_state,
                np.zeros((batch_size, self.dense_layer.units))
            )
        else:
            K.set_value(
                self.dense_state,
                states[-1]
            )

    def build(self, input_shape):
        if not self.recurrent_layer.built:
            self.recurrent_layer.build(input_shape)

        recurrent_output_shapes = self.recurrent_layer.compute_output_shape(
            input_shape
        )
        if self.return_sequences:
            if not self.dense_layer.built:
                self.dense_layer.build((
                    recurrent_output_shapes[0],
                    recurrent_output_shapes[2]
                ))
        elif not self.dense_layer.built:
            self.dense_layer.build(recurrent_output_shapes)

        super(RNNCell, self).build(input_shape)

        batch_size = input_shape[0] if self.stateful else None
        self.dense_state_spec = InputSpec(
            shape=(batch_size, self.dense_layer.units)
        )
        self.dense_state = None

    def get_constants(self, inputs, training=None):
        constants = self.recurrent_layer.get_constants(
            inputs=inputs,
            training=training
        )

        if 0 < self.dense_dropout < 1:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.recurrent_layer.units))

            def dropped_inputs():
                return K.dropout(ones, self.dense_dropout)
            out_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training)]
            constants.append(out_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.)])

        return constants

    def call(self, inputs, mask=None, training=None, initial_state=None):
        if isinstance(inputs, list):
            initial_state = inputs[1:]
            inputs = inputs[0]
        elif initial_state is not None:
            pass
        elif self.stateful:
            initial_state = self.states
        else:
            initial_state = self.get_initial_state(inputs)

        if isinstance(mask, list):
            mask = mask[0]

        if len(initial_state) != len(self.states):
            raise ValueError('Layer has ' + str(len(self.states)) +
                             ' states but was passed ' +
                             str(len(initial_state)) +
                             ' initial states.')
        input_shape = K.int_shape(inputs)
        if self.unroll and input_shape[1] is None:
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
        constants = self.get_constants(inputs, training=None)
        preprocessed_input = self.preprocess_input(inputs, training=None)
        last_output, outputs, states = YK.rnn(self.step,
                                             preprocessed_input,
                                             initial_state,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=input_shape[1])
        if self.stateful:
            updates = []
            for i in range(len(states)):
                updates.append((self.states[i], states[i]))
            self.add_update(updates, inputs)

        # Properly set learning phase
        if 0 < self.dropout + self.recurrent_dropout:
            last_output._uses_learning_phase = True
            outputs._uses_learning_phase = True

        if self.return_sequences:
            output = outputs
        else:
            output = last_output

        if self.return_state:
            if not isinstance(states, (list, tuple)):
                states = [states]
            else:
                states = list(states)
            return [output] + states
        else:
            return output

    def step(self, inputs, states):
        h, new_states = self.recurrent_layer.step(
            inputs,
            states[1:-1]
        )

        y = self.dense_layer.call(h * states[-1][0])
        if 0 < self.dense_dropout:
            y._uses_learning_phase = True
        return y, [y] + new_states

    def get_config(self):
        config = {
            'recurrent_layer':
                {
                    'class_name': self.recurrent_layer.__class__.__name__,
                    'config': self.recurrent_layer.get_config()
                },
            'dense_layer':
                {
                    'class_name': self.dense_layer.__class__.__name__,
                    'config': self.dense_layer.get_config()
                },
            'dense_dropout': self.dense_dropout,
            'go_backwards':self.go_backwards,
        }
        base_config = super(RNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        if custom_objects is None:
            custom_objects = {}
            custom_objects['LSTMPeephole'] = LSTMPeephole

        from keras.layers import deserialize as deserialize_layer
        recurrent_layer = deserialize_layer(
            config.pop('recurrent_layer'),
            custom_objects=custom_objects
        )
        dense_layer = deserialize_layer(
            config.pop('dense_layer'),
            custom_objects=custom_objects
        )
        return cls(recurrent_layer, dense_layer, **config)

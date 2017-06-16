import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops
from keras.backend import expand_dims
from keras.backend import zeros_like
from keras.backend import reverse

def rnn(step_function, inputs, initial_states,
        go_backwards=False, mask=None, constants=None,
        unroll=False, input_length=None):
    """Iterates over the time dimension of a tensor.

    # Arguments
        step_function: RNN step function.
            Parameters:
                input: tensor with shape `(samples, ...)` (no time dimension),
                    representing input for the batch of samples at a certain
                    time step.
                states: list of tensors.
            Returns:
                output: tensor with shape `(samples, output_dim)`
                    (no time dimension).
                new_states: list of tensors, same length and shapes
                    as 'states'. The first state in the list must be the
                    output tensor at the previous timestep.
        inputs: tensor of temporal data of shape `(samples, time, ...)`
            (at least 3D).
        initial_states: tensor with shape (samples, output_dim)
            (no time dimension),
            containing the initial values for the states used in
            the step function.
        go_backwards: boolean. If True, do the iteration over the time
            dimension in reverse order and return the reversed sequence.
        mask: binary tensor with shape `(samples, time, 1)`,
            with a zero for every element that is masked.
        constants: a list of constant values passed at each step.
        unroll: whether to unroll the RNN or to use a symbolic loop (`while_loop` or `scan` depending on backend).
        input_length: not relevant in the TensorFlow implementation.
            Must be specified if using unrolling with Theano.

    # Returns
        A tuple, `(last_output, outputs, new_states)`.

            last_output: the latest output of the rnn, of shape `(samples, ...)`
            outputs: tensor with shape `(samples, time, ...)` where each
                entry `outputs[s, t]` is the output of the step function
                at time `t` for sample `s`.
            new_states: list of tensors, latest states returned by
                the step function, of shape `(samples, ...)`.

    # Raises
        ValueError: if input dimension is less than 3.
        ValueError: if `unroll` is `True` but input timestep is not a fixed number.
        ValueError: if `mask` is provided (not `None`) but states is not provided
            (`len(states)` == 0).
    """
    ndim = len(inputs.get_shape())
    if ndim < 3:
        raise ValueError('Input should be at least 3D.')
    axes = [1, 0] + list(range(2, ndim))
    inputs = tf.transpose(inputs, (axes))

    if mask is not None:
        if mask.dtype != tf.bool:
            mask = tf.cast(mask, tf.bool)
        if len(mask.get_shape()) == ndim - 1:
            mask = expand_dims(mask)
        mask = tf.transpose(mask, axes)

    if constants is None:
        constants = []

    if unroll:
        if not inputs.get_shape()[0]:
            raise ValueError('Unrolling requires a '
                             'fixed number of timesteps.')
        states = initial_states
        successive_states = []
        successive_outputs = []

        input_list = tf.unstack(inputs)
        if go_backwards:
            input_list.reverse()

        if mask is not None:
            mask_list = tf.unstack(mask)
            if go_backwards:
                mask_list.reverse()

            for inp, mask_t in zip(input_list, mask_list):
                output, new_states = step_function(inp, states + constants)

                # tf.where needs its condition tensor
                # to be the same shape as its two
                # result tensors, but in our case
                # the condition (mask) tensor is
                # (nsamples, 1), and A and B are (nsamples, ndimensions).
                # So we need to
                # broadcast the mask to match the shape of A and B.
                # That's what the tile call does,
                # it just repeats the mask along its second dimension
                # n times.
                tiled_mask_t = tf.tile(mask_t,
                                       tf.stack([1, tf.shape(output)[1]]))

                if not successive_outputs:
                    prev_output = zeros_like(output)
                else:
                    prev_output = successive_outputs[-1]

                output = tf.where(tiled_mask_t, output, prev_output)

                return_states = []
                for state, new_state in zip(states, new_states):
                    # (see earlier comment for tile explanation)
                    tiled_mask_t = tf.tile(mask_t,
                                           tf.stack([1, tf.shape(new_state)[1]]))
                    return_states.append(tf.where(tiled_mask_t,
                                                  new_state,
                                                  state))
                states = return_states
                successive_outputs.append(output)
                successive_states.append(states)
            last_output = successive_outputs[-1]
            new_states = successive_states[-1]
            outputs = tf.stack(successive_outputs)
        else:
            for inp in input_list:
                output, states = step_function(inp, states + constants)
                successive_outputs.append(output)
                successive_states.append(states)
            last_output = successive_outputs[-1]
            new_states = successive_states[-1]
            outputs = tf.stack(successive_outputs)

    else:
        if go_backwards:
            inputs = reverse(inputs, 0)

        states = tuple(initial_states)

        time_steps = tf.shape(inputs)[0]
        outputs, _ = step_function(inputs[0], initial_states + constants)
        output_ta = tensor_array_ops.TensorArray(
            dtype=outputs.dtype,
            size=time_steps,
            tensor_array_name='output_ta')
        input_ta = tensor_array_ops.TensorArray(
            dtype=inputs.dtype,
            size=time_steps,
            tensor_array_name='input_ta')
        input_ta = input_ta.unstack(inputs)
        time = tf.constant(0, dtype='int32', name='time')

        if mask is not None:
            if not states:
                raise ValueError('No initial states provided! '
                                 'When using masking in an RNN, you should '
                                 'provide initial states '
                                 '(and your step function should return '
                                 'as its first state at time `t` '
                                 'the output at time `t-1`).')
            if go_backwards:
                mask = reverse(mask, 0)

            mask_ta = tensor_array_ops.TensorArray(
                dtype=tf.bool,
                size=time_steps,
                tensor_array_name='mask_ta')
            mask_ta = mask_ta.unstack(mask)

            def _step(time, output_ta_t, *states):
                """RNN step function.

                # Arguments
                    time: Current timestep value.
                    output_ta_t: TensorArray.
                    *states: List of states.

                # Returns
                    Tuple: `(time + 1,output_ta_t) + tuple(new_states)`
                """
                current_input = input_ta.read(time)
                mask_t = tf.reshape(mask_ta.read(time), [-1])
                output, new_states = step_function(current_input,
                                                   tuple(states) +
                                                   tuple(constants))
                for state, new_state in zip(states, new_states):
                    new_state.set_shape(state.get_shape())
                #tiled_mask_t = tf.tile(mask_t,
                #                       tf.stack([1, tf.shape(output)[1]]))
                output = tf.where(mask_t, output, states[0])
                new_states = [tf.where(mask_t, new_states[i], states[i]) for i in range(len(states))]
                output_ta_t = output_ta_t.write(time, output)
                return (time + 1, output_ta_t) + tuple(new_states)
        else:
            def _step(time, output_ta_t, *states):
                """RNN step function.

                # Arguments
                    time: Current timestep value.
                    output_ta_t: TensorArray.
                    *states: List of states.

                # Returns
                    Tuple: `(time + 1,output_ta_t) + tuple(new_states)`
                """
                current_input = input_ta.read(time)
                output, new_states = step_function(current_input,
                                                   tuple(states) +
                                                   tuple(constants))
                for state, new_state in zip(states, new_states):
                    new_state.set_shape(state.get_shape())
                output_ta_t = output_ta_t.write(time, output)
                return (time + 1, output_ta_t) + tuple(new_states)

        final_outputs = control_flow_ops.while_loop(
            cond=lambda time, *_: time < time_steps,
            body=_step,
            loop_vars=(time, output_ta) + states,
            parallel_iterations=32,
            swap_memory=True)
        last_time = final_outputs[0]
        output_ta = final_outputs[1]
        new_states = final_outputs[2:]

        outputs = output_ta.stack()
        last_output = output_ta.read(last_time - 1)

    axes = [1, 0] + list(range(2, len(outputs.get_shape())))
    outputs = tf.transpose(outputs, axes)
    return last_output, outputs, new_states


def rnn_decoder(step_function, inputs, initial_states,
        go_backwards=False, mask=None, constants=None,
        unroll=False, input_length=None, time_steps=None):
    """Iterates over the time dimension of a tensor.

    # Arguments
        step_function: RNN step function.
            Parameters:
                input: tensor with shape `(samples, ...)` (no time dimension),
                    representing input for the batch of samples at a certain
                    time step.
                states: list of tensors.
            Returns:
                output: tensor with shape `(samples, output_dim)`
                    (no time dimension).
                new_states: list of tensors, same length and shapes
                    as 'states'. The first state in the list must be the
                    output tensor at the previous timestep.
        inputs: tensor of temporal data of shape `(samples, time, ...)`
            (at least 3D).
        initial_states: tensor with shape (samples, output_dim)
            (no time dimension),
            containing the initial values for the states used in
            the step function.
        go_backwards: boolean. If True, do the iteration over the time
            dimension in reverse order and return the reversed sequence.
        mask: binary tensor with shape `(samples, time, 1)`,
            with a zero for every element that is masked.
        constants: a list of constant values passed at each step.
        unroll: whether to unroll the RNN or to use a symbolic loop (`while_loop` or `scan` depending on backend).
        input_length: not relevant in the TensorFlow implementation.
            Must be specified if using unrolling with Theano.
        time_steps: a int to specify the decoding length. we use the input
                   length to decode if time_steps is None

    # Returns
        A tuple, `(last_output, outputs, new_states)`.

            last_output: the latest output of the rnn, of shape `(samples, ...)`
            outputs: tensor with shape `(samples, time, ...)` where each
                entry `outputs[s, t]` is the output of the step function
                at time `t` for sample `s`.
            new_states: list of tensors, latest states returned by
                the step function, of shape `(samples, ...)`.

    # Raises
        ValueError: if input dimension is less than 3.
        ValueError: if `unroll` is `True` but input timestep is not a fixed number.
        ValueError: if `mask` is provided (not `None`) but states is not provided
            (`len(states)` == 0).
    """
    ndim = len(inputs.get_shape())
    if ndim < 3:
        raise ValueError('Input should be at least 3D.')
    axes = [1, 0] + list(range(2, ndim))
    inputs = tf.transpose(inputs, (axes))

    inputs_shape = tf.shape(inputs)
    if time_steps is not None:
        if inputs_shape[0] != time_steps:
            zeros = tf.zeros(
                shape=[
                    time_steps - 1,
                    inputs_shape[1],
                    inputs_shape[2]
                ]
            )
            inputs = tf.concat(
                [
                    tf.slice(inputs, [0, 0, 0], [1, -1, -1]),
                    zeros
                ],
                axis=0
            )

    if mask is not None:
        if mask.dtype != tf.bool:
            mask = tf.cast(mask, tf.bool)
        if len(mask.get_shape()) == ndim - 1:
            mask = expand_dims(mask)
        mask = tf.transpose(mask, axes)

        mask_shape = mask.get_shape()
        if time_steps is not None:
            if mask.get_shape()[0] != time_steps:
                mask = tf.ones_like(
                    mask,
                    dtype=tf.bool
                )
                mask = tf.reduce_any(
                    mask,
                    axis=0,
                    keep_dims=True
                )
                mask = tf.tile(
                    mask,
                    [time_steps, 1, 1]
                )

    if constants is None:
        constants = []

    if unroll:
        if not inputs.get_shape()[0]:
            raise ValueError('Unrolling requires a '
                             'fixed number of timesteps.')
        states = initial_states
        successive_states = []
        successive_outputs = []

        input_list = tf.unstack(inputs)
        if go_backwards:
            input_list.reverse()

        if mask is not None:
            mask_list = tf.unstack(mask)
            if go_backwards:
                mask_list.reverse()

            for inp, mask_t in zip(input_list, mask_list):
                output, new_states = step_function(inp, states + constants)

                # tf.where needs its condition tensor
                # to be the same shape as its two
                # result tensors, but in our case
                # the condition (mask) tensor is
                # (nsamples, 1), and A and B are (nsamples, ndimensions).
                # So we need to
                # broadcast the mask to match the shape of A and B.
                # That's what the tile call does,
                # it just repeats the mask along its second dimension
                # n times.
                tiled_mask_t = tf.tile(mask_t,
                                       tf.stack([1, tf.shape(output)[1]]))

                if not successive_outputs:
                    prev_output = zeros_like(output)
                else:
                    prev_output = successive_outputs[-1]

                output = tf.where(tiled_mask_t, output, prev_output)

                return_states = []
                for state, new_state in zip(states, new_states):
                    # (see earlier comment for tile explanation)
                    tiled_mask_t = tf.tile(mask_t,
                                           tf.stack([1, tf.shape(new_state)[1]]))
                    return_states.append(tf.where(tiled_mask_t,
                                                  new_state,
                                                  state))
                states = return_states
                successive_outputs.append(output)
                successive_states.append(states)
            last_output = successive_outputs[-1]
            new_states = successive_states[-1]
            outputs = tf.stack(successive_outputs)
        else:
            for inp in input_list:
                output, states = step_function(inp, states + constants)
                successive_outputs.append(output)
                successive_states.append(states)
            last_output = successive_outputs[-1]
            new_states = successive_states[-1]
            outputs = tf.stack(successive_outputs)

    else:
        if go_backwards:
            inputs = reverse(inputs, 0)

        states = tuple(initial_states)

        time_steps = tf.shape(inputs)[0]
        outputs, _ = step_function(inputs[0], initial_states + constants)
        output_ta = tensor_array_ops.TensorArray(
            dtype=outputs.dtype,
            size=time_steps,
            tensor_array_name='output_ta')
        input_ta = tensor_array_ops.TensorArray(
            dtype=inputs.dtype,
            size=time_steps,
            tensor_array_name='input_ta')
        input_ta = input_ta.unstack(inputs)
        time = tf.constant(0, dtype='int32', name='time')

        if mask is not None:
            if not states:
                raise ValueError('No initial states provided! '
                                 'When using masking in an RNN, you should '
                                 'provide initial states '
                                 '(and your step function should return '
                                 'as its first state at time `t` '
                                 'the output at time `t-1`).')
            if go_backwards:
                mask = reverse(mask, 0)

            mask_ta = tensor_array_ops.TensorArray(
                dtype=tf.bool,
                size=time_steps,
                tensor_array_name='mask_ta')
            mask_ta = mask_ta.unstack(mask)

            def _step(time, output_ta_t, *states):
                """RNN step function.

                # Arguments
                    time: Current timestep value.
                    output_ta_t: TensorArray.
                    *states: List of states.

                # Returns
                    Tuple: `(time + 1,output_ta_t) + tuple(new_states)`
                """
                current_input = input_ta.read(time)
                mask_t = tf.reshape(mask_ta.read(time), [-1])
                output, new_states = step_function(current_input,
                                                   tuple(states) +
                                                   tuple(constants))
                for state, new_state in zip(states, new_states):
                    new_state.set_shape(state.get_shape())
                #tiled_mask_t = tf.tile(mask_t,
                #                       tf.stack([1, tf.shape(output)[1]]))
                output = tf.where(mask_t, output, states[0])
                new_states = [tf.where(mask_t, new_states[i], states[i]) for i in range(len(states))]
                output_ta_t = output_ta_t.write(time, output)
                return (time + 1, output_ta_t) + tuple(new_states)
        else:
            def _step(time, output_ta_t, *states):
                """RNN step function.

                # Arguments
                    time: Current timestep value.
                    output_ta_t: TensorArray.
                    *states: List of states.

                # Returns
                    Tuple: `(time + 1,output_ta_t) + tuple(new_states)`
                """
                current_input = input_ta.read(time)
                output, new_states = step_function(current_input,
                                                   tuple(states) +
                                                   tuple(constants))
                for state, new_state in zip(states, new_states):
                    new_state.set_shape(state.get_shape())
                output_ta_t = output_ta_t.write(time, output)
                return (time + 1, output_ta_t) + tuple(new_states)

        final_outputs = control_flow_ops.while_loop(
            cond=lambda time, *_: time < time_steps,
            body=_step,
            loop_vars=(time, output_ta) + states,
            parallel_iterations=32,
            swap_memory=True)
        last_time = final_outputs[0]
        output_ta = final_outputs[1]
        new_states = final_outputs[2:]

        outputs = output_ta.stack()
        last_output = output_ta.read(last_time - 1)

    axes = [1, 0] + list(range(2, len(outputs.get_shape())))
    outputs = tf.transpose(outputs, axes)
    return last_output, outputs, new_states

'''Pick specific slice of tensor from input'''
import keras.backend as K
from keras.engine.topology import Layer
import tensorflow as tf

class Pick(Layer):
    def __init__(
        self,
        timestamp=0,
        **kwargs
    ):
        super(Pick, self).__init__(**kwargs)
        self.timestamp = timestamp
        self.supports_masking = True

    def build(self, input_shape):
        super(Pick, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def compute_mask(self, inputs, mask):
        if mask is None:
            return mask
        else:
            return tf.slice(
                mask,
                [0, self.timestamp],
                [-1, 1]
            )

    def call(self, inputs, mask=None):
        input_shape = K.int_shape(inputs)
        return K.reshape(
            tf.slice(
                inputs,
                [0, self.timestamp, 0],
                [-1, 1, -1]
            ),
            (-1, input_shape[2])
        )

    def get_config(self):
        config = {
            'timestamp': self.timestamp,
        }
        base_config = super(Pick, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf

class PaddingZero(Layer):

    def __init__(self, max_length, **kwargs):
        self.max_length = max_length
        super(PaddingZero, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PaddingZero, self).build(input_shape)

    def call(self, x):
        dims = tf.shape(x)
        x = tf.reshape(x, shape=[dims[0], 1, dims[1]])
        zeros = tf.zeros(shape=[dims[0], self.max_length - 1, dims[1]])
        return K.concatenate([x, zeros], axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.max_length, input_shape[1])

    def get_config(self):
        config = {'max_length': self.max_length}
        base_config = super(PaddingZero, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

'''ConvEncoder is a layer which transforms 2D or 3D tensor into 3D
timestamp sequence. The sequence contains encoded vector at the
first stamp and zero vectors at remaining stamps. ConvEncoder provides
mask function in Keras framework and could be combined with our customed
LSTMDecoder.
'''
import tensorflow as tf
import keras.backend as K
from keras.layers import Layer

class ConvEncoder(Layer):

    def __init__(self, **kwargs):
        super(ConvEncoder, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list):
            raise ValueError('A ConvEncoder layer should be called '
                             'on a list of inputs.')
        if len(input_shape) < 2:
            raise ValueError('A ConvEncoder layer should be called '
                             'on a list of at least 2 inputs. '
                             'Got ' + str(len(input_shape)) + ' inputs.')
        super(ConvEncoder, self).build(input_shape)

    def call(self, inputs, mask=None):
        image_dims = tf.shape(inputs[0])
        inputs_shape = [K.int_shape(i) for i in inputs]
        _, max_length, feature_size = self.compute_output_shape(inputs_shape)
        encoded_vector = K.reshape(inputs[0], (-1, feature_size))
        encoded_vector = K.expand_dims(encoded_vector, 1)
        zeros = tf.zeros(
            shape=[
                image_dims[0],
                max_length - 1,
                feature_size
            ]
        )
        return K.concatenate([encoded_vector, zeros], axis=1)

    def compute_output_shape(self, input_shape):
        feature_size = 1
        for size in input_shape[0][1:]:
            feature_size *= size
        return (input_shape[0][0], input_shape[1][1], feature_size)

    def compute_mask(self, inputs, mask=None):
        return mask[1]

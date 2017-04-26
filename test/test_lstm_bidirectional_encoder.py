'''Bidirectional Encoder test case'''
from unittest import TestCase
import numpy as np

from keras.models import Input, Model
from keras.layers import Masking

from yklz import Bidirectional_Encoder
from yklz import LSTMEncoder

class TestBidirectionalEncoder(TestCase):

    def setUp(self):
        self.feature_size = 30
        self.max_length = 10
        self.encoding_size = 20
        self.batch_size = 100

        self.mask_point = 7
        self.data = np.random.rand(
            self.batch_size, 
            self.max_length, 
            self.feature_size
        )
        self.data[:, self.mask_point:, :] = 0.0

        self.model = self.create_model()

    def create_model(self):
        inputs = Input(shape=(self.max_length, self.feature_size))
        masked_inputs = Masking(0.0)(inputs)
        outputs = Bidirectional_Encoder(
            LSTMEncoder(
                self.encoding_size,
                return_sequences=True
            )
        )(masked_inputs)
        model = Model(inputs, outputs)
        model.compile('sgd', 'mean_squared_error')
        return model

    def test_output_shape(self):
        result = self.model.predict(self.data)
        self.assertEqual(
            result.shape,
            (self.batch_size, self.max_length, self.encoding_size * 2)
        )  

    def test_mask(self):
        result = self.model.predict(self.data)
        np.testing.assert_array_almost_equal(
            result[:, 1:, :],
            np.zeros((
                self.batch_size, 
                self.max_length - 1, 
                self.encoding_size * 2
            ))
        )
        np.testing.assert_equal(
            np.any(
                np.not_equal(
                    result[:, 0:1, self.encoding_size:],
                    np.zeros((self.batch_size, 1, self.encoding_size))
                )
            ), 
            True
        )


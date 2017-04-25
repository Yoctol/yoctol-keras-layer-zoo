'''LSTM base test case'''
from unittest import TestCase

import numpy as np
from keras.models import Model, Input
from keras.layers.core import Masking

class TestLSTMBaseClass(object):

    def setUp(self):
        self.max_length = 10
        self.feature_size = 30
        self.encoding_size = 20
        self.data_size = 100

        self.mask_start_point = 7
        self.data = np.random.rand(self.data_size, self.max_length, self.feature_size)
        self.data[:, self.mask_start_point:, :] = 0.0

    def create_model(self, LSTMLayer):
        inputs = Input(shape=(self.max_length, self.feature_size))
        masked_inputs = Masking(0.0)(inputs)
        outputs = LSTMLayer(
            self.encoding_size,
            return_sequences=True
        )(masked_inputs)
        model = Model(inputs, outputs)
        model.compile('sgd', 'mean_squared_error') 
        return model

    def test_output_shape(self):
        result = self.model.predict(self.data)
        self.assertEqual(
            result.shape, 
            (self.data_size, self.max_length, self.encoding_size)
        )

    def test_mask(self):
        result = self.model.predict(self.data)
        np.testing.assert_array_almost_equal(
            result[:, self.mask_start_point - 1, :],
            result[:, -1, :]
        ) 

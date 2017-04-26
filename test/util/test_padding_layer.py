'''Test case for padding layers'''
from unittest import TestCase

import numpy as np
from keras.models import Input, Model

from yklz import LSTMCell
from yklz import PaddingZero

class TestPaddingZero(TestCase):

    def setUp(self):
        self.max_length = 10
        self.encoding_size = 200
        self.feature_size = 300
        self.batch_size = 100

        self.data = np.random.rand(
            self.batch_size, 
            self.max_length, 
            self.feature_size
        )
        self.model = self.create_model()
        
    def create_model(self):
        inputs = Input(
            shape=(
                self.max_length, 
                self.feature_size
            )
        )
        encoded_seq = LSTMCell(
            self.encoding_size,
            return_sequences=False
        )(inputs)
        outputs = PaddingZero(
            self.max_length
        )(encoded_seq)

        model = Model(inputs, outputs)
        model.compile('sgd', 'mean_squared_error')
        return model 

    def test_output_shape(self):
        result = self.model.predict(self.data)
        self.assertEqual(
            result.shape,
            (self.batch_size, self.max_length, self.encoding_size)
        )

    def test_padding_zero(self):
        result = self.model.predict(self.data)
        np.testing.assert_array_almost_equal(
            result[:, 1:, :],
            np.zeros((
                self.batch_size, 
                self.max_length - 1, 
                self.encoding_size
            ))
        )
        

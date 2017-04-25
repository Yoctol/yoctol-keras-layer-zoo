'''LSTM Encoder test case'''
from unittest import TestCase

import numpy as np
from keras.models import Model, Input
from keras.layers.core import Masking

from yklz import LSTMEncoder
from .test_lstm_base import TestLSTMBaseClass

class TestLSTMEncoderClass(TestLSTMBaseClass, TestCase):

    def setUp(self):
        super(TestLSTMEncoderClass, self).setUp()
        self.model = self.create_model(LSTMEncoder)

    def test_padding_zeros(self):
        result = self.model.predict(self.data)
        np.testing.assert_array_almost_equal(
            np.zeros((self.data_size, self.max_length - 1, self.encoding_size)),
            result[:, 1:, :]
        )

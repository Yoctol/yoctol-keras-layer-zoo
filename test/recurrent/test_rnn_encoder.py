'''RNN Encoder test case'''
from unittest import TestCase

import numpy as np
from keras.models import Model, Input
from keras.layers.core import Masking, Dense
from keras.layers import LSTM, GRU, SimpleRNN

from yklz import RNNCell, LSTMPeephole
from yklz import RNNEncoder
from test import TestRNNBaseClass

class TestRNNEncoderBaseClass(TestRNNBaseClass):

    def setUp(self):
        super(TestRNNEncoderBaseClass, self).setUp()
        self.custom_objects['RNNEncoder'] = RNNEncoder

    def create_model(self, rnn_layer):
        inputs = Input(shape=(self.max_length, self.feature_size))
        masked_inputs = Masking(0.0)(inputs)
        outputs = RNNEncoder(
            rnn_layer(
                self.encoding_size,
            )
        )(masked_inputs)
        model = Model(inputs, outputs)
        model.compile('sgd', 'mean_squared_error')
        return model

    def test_padding_zeros(self):
        result = self.model.predict(self.data)
        np.testing.assert_array_almost_equal(
            np.zeros((
                self.data_size,
                self.max_length - 1,
                self.encoding_size
            )),
            result[:, 1:, :]
        )

class TestRNNEncoderWithLSTMClass(TestRNNEncoderBaseClass, TestCase):

    def setUp(self):
        super(TestRNNEncoderWithLSTMClass, self).setUp()
        self.model = self.create_model(LSTM)

class TestRNNEncoderWithGRUClass(TestRNNEncoderBaseClass, TestCase):

    def setUp(self):
        super(TestRNNEncoderWithGRUClass, self).setUp()
        self.model = self.create_model(GRU)

class TestRNNEncoderWithSimpleRNNClass(TestRNNEncoderBaseClass, TestCase):

    def setUp(self):
        super(TestRNNEncoderWithSimpleRNNClass, self).setUp()
        self.model = self.create_model(SimpleRNN)

class TestRNNEncoderWithLSTMPeepholeClass(TestRNNEncoderBaseClass, TestCase):

    def setUp(self):
        super(TestRNNEncoderWithLSTMPeepholeClass, self).setUp()
        self.custom_objects['LSTMPeephole'] = LSTMPeephole
        self.model = self.create_model(LSTMPeephole)



class TestRNNEncoderWithRNNCellBaseClass(TestRNNEncoderBaseClass):

    def setUp(self):
        super(TestRNNEncoderWithRNNCellBaseClass, self).setUp()
        self.custom_objects['RNNCell'] = RNNCell
        self.hidden_size = 25

    def create_model(self, rnn_layer):
        inputs = Input(shape=(self.max_length, self.feature_size))
        masked_inputs = Masking(0.0)(inputs)
        outputs = RNNEncoder(
            RNNCell(
                rnn_layer(
                    self.hidden_size
                ),
                Dense(
                    self.encoding_size
                ),
                dense_dropout=0.1
            )
        )(masked_inputs)
        model = Model(inputs, outputs)
        model.compile('sgd', 'mean_squared_error')
        return model

class TestRNNEncoderWithRNNCellLSTMClass(
        TestRNNEncoderWithRNNCellBaseClass,
        TestCase
):
    def setUp(self):
        super(TestRNNEncoderWithRNNCellLSTMClass, self).setUp()
        self.model = self.create_model(LSTM)

class TestRNNEncoderWithRNNCellGRUClass(
        TestRNNEncoderWithRNNCellBaseClass,
        TestCase
):
    def setUp(self):
        super(TestRNNEncoderWithRNNCellGRUClass, self).setUp()
        self.model = self.create_model(GRU)

class TestRNNEncoderWithRNNCellSimpleRNNClass(
        TestRNNEncoderWithRNNCellBaseClass,
        TestCase
):
    def setUp(self):
        super(TestRNNEncoderWithRNNCellSimpleRNNClass, self).setUp()
        self.model = self.create_model(SimpleRNN)

class TestRNNEncoderWithRNNCellLSTMPeepholeClass(
        TestRNNEncoderWithRNNCellBaseClass,
        TestCase
):
    def setUp(self):
        super(TestRNNEncoderWithRNNCellLSTMPeepholeClass, self).setUp()
        self.custom_objects['LSTMPeephole'] = LSTMPeephole
        self.model = self.create_model(LSTMPeephole)

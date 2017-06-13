'''RNN Decoder test case'''
from unittest import TestCase

import numpy as np
from keras.models import Model, Input
from keras.layers.core import Masking
from keras.layers import LSTM, GRU, SimpleRNN

from yklz import RNNEncoder, RNNDecoder
from yklz import LSTMPeephole, LSTMCell
from test import TestRNNBaseClass

class TestRNNDecoderBaseClass(TestRNNBaseClass):

    def setUp(self):
        super(TestRNNDecoderBaseClass, self).setUp()
        self.encoding_size = self.feature_size
        self.y = np.random.rand(self.data_size, self.max_length, self.encoding_size)

    def create_model(self, rnn_layer):
        inputs = Input(shape=(self.max_length, self.feature_size))
        masked_inputs = Masking(0.0)(inputs)
        encoded = RNNEncoder(
            rnn_layer(
                self.encoding_size,
            )
        )(masked_inputs)
        outputs = RNNDecoder(
            rnn_layer(
                self.encoding_size,
            )
        )(encoded)
        model = Model(inputs, outputs)
        model.compile('sgd', 'mean_squared_error')
        return model

class TestRNNDecoderWithLSTMClass(TestRNNDecoderBaseClass, TestCase):

    def setUp(self):
        super(TestRNNDecoderWithLSTMClass, self).setUp()
        self.model = self.create_model(LSTM)

class TestRNNDecoderWithGRUClass(TestRNNDecoderBaseClass, TestCase):

    def setUp(self):
        super(TestRNNDecoderWithGRUClass, self).setUp()
        self.model = self.create_model(LSTM)

class TestRNNDecoderWithSimpleRNNClass(TestRNNDecoderBaseClass, TestCase):

    def setUp(self):
        super(TestRNNDecoderWithSimpleRNNClass, self).setUp()
        self.model = self.create_model(SimpleRNN)

class TestRNNDecoderWithLSTMPeepholeClass(TestRNNDecoderBaseClass, TestCase):

    def setUp(self):
        super(TestRNNDecoderWithLSTMPeepholeClass, self).setUp()
        self.model = self.create_model(LSTMPeephole)

class TestRNNDecoderWithLSTMCellClass(TestRNNDecoderBaseClass, TestCase):

    def setUp(self):
        super(TestRNNDecoderWithLSTMCellClass, self).setUp()
        self.model = self.create_model(LSTMCell)

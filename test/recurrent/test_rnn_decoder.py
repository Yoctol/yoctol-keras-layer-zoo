'''RNN Decoder test case'''
from unittest import TestCase

import numpy as np
from keras.models import Model, Input
from keras.layers.core import Masking, Dense
from keras.layers import LSTM, GRU, SimpleRNN

from yklz import RNNEncoder, RNNDecoder
from yklz import LSTMPeephole, RNNCell
from test import TestRNNBaseClass

class TestRNNDecoderBaseClass(TestRNNBaseClass):

    def setUp(self):
        super(TestRNNDecoderBaseClass, self).setUp()
        self.custom_objects['RNNEncoder'] = RNNEncoder
        self.custom_objects['RNNDecoder'] = RNNDecoder
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
                self.feature_size,
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
        self.model = self.create_model(GRU)

class TestRNNDecoderWithSimpleRNNClass(TestRNNDecoderBaseClass, TestCase):

    def setUp(self):
        super(TestRNNDecoderWithSimpleRNNClass, self).setUp()
        self.model = self.create_model(SimpleRNN)

class TestRNNDecoderWithLSTMPeepholeClass(TestRNNDecoderBaseClass, TestCase):

    def setUp(self):
        super(TestRNNDecoderWithLSTMPeepholeClass, self).setUp()
        self.custom_objects['LSTMPeephole'] = LSTMPeephole
        self.model = self.create_model(LSTMPeephole)

class TestRNNDecoderWithRNNCellBaseClass(TestRNNDecoderBaseClass):

    def setUp(self):
        super(TestRNNDecoderWithRNNCellBaseClass, self).setUp()
        self.custom_objects['RNNCell'] = RNNCell
        self.hidden_size = 25

    def create_model(self, rnn_layer):
        inputs = Input(shape=(self.max_length, self.feature_size))
        masked_inputs = Masking(0.0)(inputs)
        encoded = RNNEncoder(
            RNNCell(
                rnn_layer(
                    self.hidden_size,
                ),
                Dense(
                    self.encoding_size
                ),
                dense_dropout=0.1
            )
        )(masked_inputs)
        outputs = RNNDecoder(
            RNNCell(
                rnn_layer(
                    self.hidden_size,
                ),
                Dense(
                    self.feature_size
                ),
                dense_dropout=0.1
            )
        )(encoded)
        model = Model(inputs, outputs)
        model.compile('sgd', 'mean_squared_error')
        return model


class TestRNNDecoderWithRNNCellLSTMClass(
    TestRNNDecoderWithRNNCellBaseClass,
    TestCase
):
    def setUp(self):
        super(TestRNNDecoderWithRNNCellLSTMClass, self).setUp()
        self.model = self.create_model(LSTM)

class TestRNNDecoderWithRNNCellGRUClass(
    TestRNNDecoderWithRNNCellBaseClass,
    TestCase
):
    def setUp(self):
        super(TestRNNDecoderWithRNNCellGRUClass, self).setUp()
        self.model = self.create_model(GRU)

class TestRNNDecoderWithRNNCellSimpleRNNClass(
    TestRNNDecoderWithRNNCellBaseClass,
    TestCase
):
    def setUp(self):
        super(TestRNNDecoderWithRNNCellSimpleRNNClass, self).setUp()
        self.model = self.create_model(SimpleRNN)

class TestRNNDecoderWithRNNCellLSTMPeepholeClass(
    TestRNNDecoderWithRNNCellBaseClass,
    TestCase
):
    def setUp(self):
        super(TestRNNDecoderWithRNNCellLSTMPeepholeClass, self).setUp()
        self.custom_objects['LSTMPeephole'] = LSTMPeephole
        self.model = self.create_model(LSTMPeephole)

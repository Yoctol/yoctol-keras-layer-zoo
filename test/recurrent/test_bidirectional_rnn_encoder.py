'''Bidirectional Encoder test case'''
from unittest import TestCase
import numpy as np

from keras.models import Input, Model
from keras.layers import Masking, Dense
from keras.layers import LSTM, GRU, SimpleRNN

from yklz import BidirectionalRNNEncoder
from yklz import RNNCell, LSTMPeephole
from test import TestRNNBaseClass

class TestBidirectionalRNNEncoderBaseClass(TestRNNBaseClass):

    def setUp(self):
        super(TestBidirectionalRNNEncoderBaseClass, self).setUp()
        self.custom_objects['BidirectionalRNNEncoder'] = BidirectionalRNNEncoder
        self.cell_units = self.encoding_size // 2

    def create_model(self, rnn_layer):
        inputs = Input(shape=(self.max_length, self.feature_size))
        masked_inputs = Masking(0.0)(inputs)
        outputs = BidirectionalRNNEncoder(
            rnn_layer(
                self.cell_units,
            )
        )(masked_inputs)
        model = Model(inputs, outputs)
        model.compile('sgd', 'mean_squared_error')
        return model

    def test_mask_value(self):
        result = self.model.predict(self.data)
        np.testing.assert_array_almost_equal(
            result[:, 1:, :],
            np.zeros((
                self.data_size,
                self.max_length - 1,
                self.encoding_size
            ))
        )
        np.testing.assert_equal(
            np.any(
                np.not_equal(
                    result[:, 0:1, self.cell_units:],
                    np.zeros((self.data_size, 1, self.cell_units))
                )
            ),
            True
        )

class TestBidirectionalRNNEncoderWithLSTMClass(
        TestBidirectionalRNNEncoderBaseClass,
        TestCase
):
    def setUp(self):
        super(TestBidirectionalRNNEncoderWithLSTMClass, self).setUp()
        self.model = self.create_model(LSTM)

class TestBidirectionalRNNEncoderWithGRUClass(
        TestBidirectionalRNNEncoderBaseClass,
        TestCase
):
    def setUp(self):
        super(TestBidirectionalRNNEncoderWithGRUClass, self).setUp()
        self.model = self.create_model(GRU)

class TestBidirectionalRNNEncoderWithSimpleRNNClass(
        TestBidirectionalRNNEncoderBaseClass,
        TestCase
):
    def setUp(self):
        super(TestBidirectionalRNNEncoderWithSimpleRNNClass, self).setUp()
        self.model = self.create_model(SimpleRNN)

class TestBidirectionalRNNEncoderWithLSTMPeepholeClass(
        TestBidirectionalRNNEncoderBaseClass,
        TestCase
):
    def setUp(self):
        super(TestBidirectionalRNNEncoderWithLSTMPeepholeClass, self).setUp()
        self.custom_objects['LSTMPeephole'] = LSTMPeephole
        self.model = self.create_model(LSTMPeephole)


class TestBidirectionalRNNEncoderWithRNNCellBaseClass(
        TestBidirectionalRNNEncoderBaseClass
):
    def setUp(self):
        super(TestBidirectionalRNNEncoderWithRNNCellBaseClass, self).setUp()
        self.custom_objects['RNNCell'] = RNNCell
        self.hidden_units = 32

    def create_model(self, rnn_layer):
        inputs = Input(shape=(self.max_length, self.feature_size))
        masked_inputs = Masking(0.0)(inputs)
        outputs = BidirectionalRNNEncoder(
            RNNCell(
                rnn_layer(
                    self.hidden_units,
                ),
                Dense(
                    self.cell_units
                ),
                dense_dropout=0.1
            )
        )(masked_inputs)
        model = Model(inputs, outputs)
        model.compile('sgd', 'mean_squared_error')
        return model

class TestBidirectionalRNNEncoderWithRNNCellLSTMClass(
        TestBidirectionalRNNEncoderWithRNNCellBaseClass,
        TestCase
):
    def setUp(self):
        super(TestBidirectionalRNNEncoderWithRNNCellLSTMClass, self).setUp()
        self.model = self.create_model(LSTM)

class TestBidirectionalRNNEncoderWithRNNCellGRUClass(
        TestBidirectionalRNNEncoderWithRNNCellBaseClass,
        TestCase
):
    def setUp(self):
        super(TestBidirectionalRNNEncoderWithRNNCellGRUClass, self).setUp()
        self.model = self.create_model(GRU)

class TestBidirectionalRNNEncoderWithRNNCellSimpleRNNClass(
        TestBidirectionalRNNEncoderWithRNNCellBaseClass,
        TestCase
):
    def setUp(self):
        super(TestBidirectionalRNNEncoderWithRNNCellSimpleRNNClass, self).setUp()
        self.model = self.create_model(SimpleRNN)

class TestBidirectionalRNNEncoderWithRNNCellLSTMPeepholeClass(
        TestBidirectionalRNNEncoderWithRNNCellBaseClass,
        TestCase
):
    def setUp(self):
        super(TestBidirectionalRNNEncoderWithRNNCellLSTMPeepholeClass, self).setUp()
        self.custom_objects['LSTMPeephole'] = LSTMPeephole
        self.model = self.create_model(LSTMPeephole)


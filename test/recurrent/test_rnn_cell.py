'''RNN Cell test case'''
from unittest import TestCase

from keras.models import Input, Model
from keras.layers import Dense, Masking
from keras.layers import LSTM, GRU, SimpleRNN

from yklz import RNNCell, LSTMPeephole
from .test_rnn_base import TestRNNBaseClass

class TestRNNCellBaseClass(TestRNNBaseClass):

    def setUp(self):
        super(TestRNNCellBaseClass, self).setUp()
        self.hidden_size = 25
        self.custom_objects['RNNCell'] = RNNCell

    def create_model(self, rnn_layer):
        inputs = Input(shape=(self.max_length, self.feature_size))
        masked_inputs = Masking(0.0)(inputs)
        outputs = RNNCell(
            recurrent_layer=rnn_layer(
                self.hidden_size,
                return_sequences=True
            ),
            dense_layer=Dense(
                units=self.encoding_size
            ),
            dense_dropout=0.1
        )(masked_inputs)
        model = Model(inputs, outputs)
        model.compile('sgd', 'mean_squared_error')
        return model

class TestRNNCellWithLSTMClass(TestRNNCellBaseClass, TestCase):
    def setUp(self):
        super(TestRNNCellWithLSTMClass, self).setUp()
        self.model = self.create_model(LSTM)

class TestRNNCellWithGRUClass(TestRNNCellBaseClass, TestCase):
    def setUp(self):
        super(TestRNNCellWithGRUClass, self).setUp()
        self.model = self.create_model(GRU)

class TestRNNCellWithSimpleRNNClass(TestRNNCellBaseClass, TestCase):
    def setUp(self):
        super(TestRNNCellWithSimpleRNNClass, self).setUp()
        self.model = self.create_model(SimpleRNN)

class TestRNNCellWithLSTMPeepholeClass(TestRNNCellBaseClass, TestCase):
    def setUp(self):
        super(TestRNNCellWithLSTMPeepholeClass, self).setUp()
        self.model = self.create_model(LSTMPeephole)
        self.custom_objects['LSTMPeephole'] = LSTMPeephole

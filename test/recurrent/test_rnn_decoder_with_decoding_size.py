from unittest import TestCase

import numpy as np
from keras.models import Model, Input
from keras.layers.core import Masking, Dense
from keras.layers import LSTM, GRU, SimpleRNN
import keras.backend as K

from yklz import RNNEncoder, RNNDecoder
from yklz import LSTMPeephole, RNNCell
from .test_rnn_decoder import TestRNNDecoderBaseClass

class TestRNNDecoderWithDecodingSizeBaseClass(TestRNNDecoderBaseClass):

    def setUp(self):
        super(TestRNNDecoderWithDecodingSizeBaseClass, self).setUp()
        self.custom_objects['RNNCell'] = RNNCell
        self.hidden_size = 25
        self.decoding_length = self.max_length * 2

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
            ),
            time_steps=self.decoding_length
        )(encoded)
        model = Model(inputs, outputs)
        model.compile('sgd', 'mean_squared_error')
        return model

    def test_mask_function(self):
        pass

    def test_mask(self):
        mask_cache_key = str(id(self.model.input)) + '_' + str(id(None))
        mask_tensor = self.model._output_mask_cache[mask_cache_key]
        mask = mask_tensor.eval(
            session=K.get_session(),
            feed_dict={self.model.input: self.data}
        )
        self.assertTrue(np.all(mask[:, :]))

class TestRNNDecoderWithDecodingSizeLSTMClass(
    TestRNNDecoderWithDecodingSizeBaseClass,
    TestCase
):
    def setUp(self):
        super(TestRNNDecoderWithDecodingSizeLSTMClass, self).setUp()
        self.model = self.create_model(LSTM)
        self.max_length = self.decoding_length

class TestRNNDecoderWithDecodingSizeGRUClass(
    TestRNNDecoderWithDecodingSizeBaseClass,
    TestCase
):
    def setUp(self):
        super(TestRNNDecoderWithDecodingSizeGRUClass, self).setUp()
        self.model = self.create_model(GRU)
        self.max_length = self.decoding_length

class TestRNNDecoderWithDecodingSizeSimpleRNNClass(
    TestRNNDecoderWithDecodingSizeBaseClass,
    TestCase
):
    def setUp(self):
        super(TestRNNDecoderWithDecodingSizeSimpleRNNClass, self).setUp()
        self.model = self.create_model(SimpleRNN)
        self.max_length = self.decoding_length

class TestRNNDecoderWithDecodingSizeLSTMPeepholeClass(
    TestRNNDecoderWithDecodingSizeBaseClass,
    TestCase
):
    def setUp(self):
        super(TestRNNDecoderWithDecodingSizeLSTMPeepholeClass, self).setUp()
        self.custom_objects['LSTMPeephole'] = LSTMPeephole
        self.model = self.create_model(LSTMPeephole)
        self.max_length = self.decoding_length

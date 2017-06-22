'''Pick layer test case'''
import os
from unittest import TestCase

import numpy as np
import keras.backend as K
from keras.models import Model, Input
from keras.layers.core import Masking
from keras.layers import LSTM
from keras.models import load_model

from yklz import RNNEncoder, Pick

class TestPickClass(TestCase):

    def setUp(self):
        self.max_length = 10
        self.feature_size = 30
        self.encoding_size = 20
        self.data_size = 100

        self.mask_start_point = 7
        self.data = np.random.rand(
            self.data_size,
            self.max_length,
            self.feature_size
        )
        self.data[:, self.mask_start_point:, :] = 0.0
        self.y = np.random.rand(
            self.data_size,
            self.encoding_size
        )
        self.custom_objects = {}
        self.custom_objects['RNNEncoder'] = RNNEncoder
        self.custom_objects['Pick'] = Pick
        self.model = self.create_model()

    def create_model(self):
        inputs = Input(shape=(self.max_length, self.feature_size))
        masked_inputs = Masking(0.0)(inputs)
        encoded = RNNEncoder(
            LSTM(
                self.encoding_size,
                return_sequences=True
            )
        )(masked_inputs)
        outputs = Pick()(encoded)
        model = Model(inputs, outputs)
        model.compile('sgd', 'mean_squared_error')
        return model

    def test_output_shape(self):
        result = self.model.predict(self.data)
        self.assertEqual(
            result.shape,
            (self.data_size, self.encoding_size)
        )

    def test_output_value_not_zero(self):
        result = self.model.predict(self.data)
        self.assertTrue(
            np.sum(result, dtype=bool)
        )

    def test_mask(self):
        mask_cache_key = str(id(self.model.input)) + '_' + str(id(None))
        mask_tensor = self.model._output_mask_cache[mask_cache_key]
        mask = mask_tensor.eval(
            session=K.get_session(),
            feed_dict={self.model.input: self.data}
        )
        self.assertTrue(
            np.all(mask)
        )

    def test_save_load(self):
        answer = self.model.predict(self.data)
        model_name = self.__class__.__name__ + '_temp.model'
        self.model.save(model_name)
        self.model = load_model(
            model_name,
            custom_objects=self.custom_objects
        )
        os.remove(model_name)
        result = self.model.predict(self.data)
        self.assertEqual(
            result.shape,
            (self.data_size, self.encoding_size)
        )
        np.testing.assert_array_almost_equal(
            answer,
            result
        )

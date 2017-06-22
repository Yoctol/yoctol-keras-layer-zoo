'''RNN base test case'''
import os
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import keras.backend as K
from keras.models import Model, Input
from keras.layers.core import Masking
from keras.models import load_model

class TestRNNBaseClass(object):

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
            self.max_length,
            self.encoding_size
        )
        self.custom_objects = {}

    def create_model(self, rnn_layer):
        inputs = Input(shape=(self.max_length, self.feature_size))
        masked_inputs = Masking(0.0)(inputs)
        outputs = rnn_layer(
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

    @patch('keras.models.Model.fit')
    def test_training(self, patch_fit):
        self.model.fit(
            self.data,
            self.y,
            epochs=2,
            validation_split=0.1,
            batch_size=10
        )
        result = self.model.predict(self.data)
        self.assertEqual(
            result.shape,
            (self.data_size, self.max_length, self.encoding_size)
        )

    def test_mask_function(self):
        result = self.model.predict(self.data)
        np.testing.assert_array_almost_equal(
            result[:, self.mask_start_point - 1, :],
            result[:, -1, :]
        )

    def test_mask(self):
        mask_cache_key = str(id(self.model.input)) + '_' + str(id(None))
        mask_tensor = self.model._output_mask_cache[mask_cache_key]
        mask = mask_tensor.eval(
            session=K.get_session(),
            feed_dict={self.model.input: self.data}
        )
        self.assertFalse(np.any(mask[:, self.mask_start_point:]))
        self.assertTrue(np.all(mask[:, :self.mask_start_point]))

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
            (self.data_size, self.max_length, self.encoding_size)
        )
        np.testing.assert_array_almost_equal(
            answer,
            result
        )

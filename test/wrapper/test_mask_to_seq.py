'''Testcases for ConvEncoder Layer'''
from unittest import TestCase

import numpy as np
import keras.backend as K
from keras.models import Input, Model
from keras.utils.conv_utils import conv_output_length

from yklz import MaskConv, Convolution2D
from yklz import MaskedMax2DPooling, MaskToSeq
from test import TestBase2DClass

class TestMaskToSeq2DClass(TestBase2DClass, TestCase):

    def setUp(self):
        super(TestMaskToSeq2DClass, self).setUp()

        self.seq_data = np.random.rand(
            self.batch_size,
            self.x,
            self.y,
            self.channel_size
        )
        self.seq_data_max_length = 18
        self.seq_data[:, 18:, :, :] = self.mask_value

        self.model = self.create_model()

    def create_model(self):
        inputs = Input(shape=(self.x, self.y, self.channel_size))
        outputs = MaskToSeq(MaskConv(self.mask_value))(inputs)
        model = Model(inputs, outputs)
        model.compile('sgd', 'mean_squared_error')
        return model

    def test_output_shape(self):
        result = self.model.predict(self.data)
        self.assertEqual(
            result.shape,
            (self.batch_size, self.x, self.y * self.channel_size)
        )

    def test_image_data_mask_value(self):
        result = self.model.predict(self.data)
        np.testing.assert_almost_equal(
            result,
            np.zeros((
                self.batch_size,
                self.x,
                self.y * self.channel_size
            ))
        )

    def test_seq_data_mask_value(self):
        result = self.model.predict(self.seq_data)
        self.assertTrue(
            np.any(result[:, :self.seq_data_max_length, :])
        )
        np.testing.assert_almost_equal(
            result[:, self.seq_data_max_length:, :],
            np.zeros((
                self.batch_size,
                self.x - self.seq_data_max_length,
                self.y * self.channel_size
            ))
        )

    def test_image_data_mask(self):
        mask_cache_key = str(id(self.model.input)) + '_' + str(id(None))
        mask_tensor = self.model._output_mask_cache[mask_cache_key]
        mask = mask_tensor.eval(
            session=K.get_session(),
            feed_dict={self.model.input: self.data}
        )
        self.assertFalse(np.any(mask))

    def test_seq_data_mask(self):
        mask_cache_key = str(id(self.model.input)) + '_' + str(id(None))
        mask_tensor = self.model._output_mask_cache[mask_cache_key]
        mask = mask_tensor.eval(
            session=K.get_session(),
            feed_dict={self.model.input: self.seq_data}
        )
        self.assertTrue(
            np.all(
                mask[:, :self.seq_data_max_length]
            )
        )
        self.assertFalse(
            np.any(
                mask[:, self.seq_data_max_length:]
            )
        )

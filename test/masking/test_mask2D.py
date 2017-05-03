'''Mask2D testcase'''
from unittest import TestCase

import numpy as np
from keras.models import Input, Model
from keras.layers import Conv2D

from yklz import Mask2D
from test import TestBase2DClass

class TestMask2DClass(TestBase2DClass, TestCase):

    def setUp(self):
        super(TestMask2DClass, self).setUp()
        self.model = self.create_model()

    def create_model(self):
        inputs = Input(shape=(self.x, self.y, self.channel_size))
        outputs = Mask2D(self.mask_value)(inputs)
        model = Model(inputs, outputs)
        model.compile('sgd', 'mean_squared_error')
        return model

    def test_output_shape(self):
        result = self.model.predict(self.data)
        self.assertEqual(
            result.shape,
            (self.batch_size, self.x, self.y, self.channel_size)
        )

    def test_masked_value(self):
        result = self.model.predict(self.data)
        np.testing.assert_array_almost_equal(
            result[:, :self.x_start, :, :],
            np.zeros(
                shape=(
                    self.batch_size,
                    self.x_start,
                    self.y,
                    self.channel_size
                )
            )
        )
        np.testing.assert_array_almost_equal(
            result[:, self.x_end:, :, :],
            np.zeros(
                shape=(
                    self.batch_size,
                    self.x - self.x_end,
                    self.y,
                    self.channel_size
                )
            )
        )
        np.testing.assert_array_almost_equal(
            result[:, :, :self.y_start, :],
            np.zeros(
                shape=(
                    self.batch_size,
                    self.x,
                    self.y_start,
                    self.channel_size
                )
            )
        )
        np.testing.assert_array_almost_equal(
            result[:, :, self.y_end:, :],
            np.zeros(
                shape=(
                    self.batch_size,
                    self.x,
                    self.y - self.y_end,
                    self.channel_size
                )
            )
        )

        fake_mask_array = np.ones(
            shape=(self.batch_size, self.channel_size)
        )
        fake_mask_array *= self.mask_value
        np.testing.assert_array_almost_equal(
            result[:, self.fake_x_mask, self.fake_y_mask, :],
            fake_mask_array
        )

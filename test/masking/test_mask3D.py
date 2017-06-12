'''Mask2D testcase'''
from unittest import TestCase

import numpy as np
from keras.models import Input, Model
from keras.layers import Conv2D

from yklz import MaskConv
from test import TestBase3DClass

class TestMask3DClass(TestBase3DClass, TestCase):

    def setUp(self):
        super(TestMask3DClass, self).setUp()
        self.model = self.create_model()

    def create_model(self):
        inputs = Input(shape=(self.x, self.y, self.z, self.channel_size))
        outputs = MaskConv(self.mask_value)(inputs)
        model = Model(inputs, outputs)
        model.compile('sgd', 'mean_squared_error')
        return model

    def test_output_shape(self):
        result = self.model.predict(self.data)
        self.assertEqual(
            result.shape,
            (self.batch_size, self.x, self.y, self.z, self.channel_size)
        )

    def test_masked_value(self):
        result = self.model.predict(self.data)
        np.testing.assert_array_almost_equal(
            result[:, :self.x_start, :, :, :],
            np.zeros(
                shape=(
                    self.batch_size,
                    self.x_start,
                    self.y,
                    self.z,
                    self.channel_size
                )
            )
        )
        np.testing.assert_array_almost_equal(
            result[:, self.x_end:, :, :, :],
            np.zeros(
                shape=(
                    self.batch_size,
                    self.x - self.x_end,
                    self.y,
                    self.z,
                    self.channel_size
                )
            )
        )
        np.testing.assert_array_almost_equal(
            result[:, :, :self.y_start, :, :],
            np.zeros(
                shape=(
                    self.batch_size,
                    self.x,
                    self.y_start,
                    self.z,
                    self.channel_size
                )
            )
        )
        np.testing.assert_array_almost_equal(
            result[:, :, self.y_end:, :, :],
            np.zeros(
                shape=(
                    self.batch_size,
                    self.x,
                    self.y - self.y_end,
                    self.z,
                    self.channel_size
                )
            )
        )
        np.testing.assert_array_almost_equal(
            result[:, :, :, :self.z_start, :],
            np.zeros(
                shape=(
                    self.batch_size,
                    self.x,
                    self.y,
                    self.z_start,
                    self.channel_size
                )
            )
        )
        np.testing.assert_array_almost_equal(
            result[:, :, :, self.z_end:, :],
            np.zeros(
                shape=(
                    self.batch_size,
                    self.x,
                    self.y,
                    self.z - self.z_end,
                    self.channel_size
                )
            )
        )

        fake_mask_array = np.ones(
            shape=(self.batch_size, self.channel_size)
        )
        fake_mask_array *= self.mask_value
        np.testing.assert_array_almost_equal(
            result[:, self.fake_x_mask, self.fake_y_mask, self.fake_z_mask, :],
            fake_mask_array
        )

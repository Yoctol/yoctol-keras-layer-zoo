'''Base test case for convolution liked layers'''

import numpy as np
from keras.utils.conv_utils import conv_output_length

from test import TestBase2DClass

class TestConvBase2DClass(TestBase2DClass):
    def setUp(self):
        super(TestConvBase2DClass, self).setUp()
        self.kernel = (2, 3)
        self.filters = 5
        self.strides = (3, 2)
        self.padding = 'valid'

    def test_output_shape(self):
        result = self.model.predict(self.data)
        x_window_length = conv_output_length(
            self.x,
            self.kernel[0],
            self.padding,
            self.strides[0]
        )
        y_window_length = conv_output_length(
            self.y,
            self.kernel[1],
            self.padding,
            self.strides[1]
        )
        self.assertEqual(
            result.shape,
            (
                self.batch_size,
                x_window_length,
                y_window_length,
                self.filters
            )
        )

    def test_mask_zero(self):
        result = self.model.predict(self.data)
        _, x, y, _ = result.shape
        x_start_mask = (self.x_start - self.kernel[0]) // self.strides[0] + 1
        y_start_mask = (self.y_start - self.kernel[1]) // self.strides[1] + 1
        x_end_mask = (
            self.x_end + self.strides[0] - 1
        ) // self.strides[0]
        y_end_mask = (
            self.y_end + self.strides[1] - 1
        ) // self.strides[1]
        np.testing.assert_array_almost_equal(
            result[:, :x_start_mask, :, :],
            np.zeros((
                self.batch_size, x_start_mask, y, self.filters
            ))
        )
        np.testing.assert_array_almost_equal(
            result[:, :, :y_start_mask, :],
            np.zeros((
                self.batch_size, x, y_start_mask, self.filters
            ))
        )
        if x_end_mask < x:
            np.testing.assert_array_almost_equal(
                result[:, x_end_mask:, :, :],
                np.zeros((
                    self.batch_size, x - x_end_mask, y, self.filters
                ))
            )
        if y_end_mask < y:
            np.testing.assert_array_almost_equal(
                result[:, :, y_end_mask:, :],
                np.zeros((
                    self.batch_size, x, y - y_end_mask, self.filters
                ))
            )

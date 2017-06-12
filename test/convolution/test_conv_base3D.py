'''Base test case for convolution liked layers'''

import numpy as np
from keras.utils.conv_utils import conv_output_length

from test import TestBase3DClass

class TestConvBase3DClass(TestBase3DClass):
    def setUp(self):
        super(TestConvBase3DClass, self).setUp()
        self.kernel = (2, 3, 4)
        self.filters = 5
        self.strides = (3, 2, 4)
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
        z_window_length = conv_output_length(
            self.z,
            self.kernel[2],
            self.padding,
            self.strides[2]
        )
        self.assertEqual(
            result.shape,
            (
                self.batch_size,
                x_window_length,
                y_window_length,
                z_window_length,
                self.filters
            )
        )

    def test_mask_zero(self):
        result = self.model.predict(self.data)
        _, x, y, z, _ = result.shape
        x_start_mask = (self.x_start - self.kernel[0]) // self.strides[0] + 1
        y_start_mask = (self.y_start - self.kernel[1]) // self.strides[1] + 1
        z_start_mask = (self.z_start - self.kernel[2]) // self.strides[2] + 1
        x_end_mask = (
            self.x_end + self.strides[0] - 1
        ) // self.strides[0]
        y_end_mask = (
            self.y_end + self.strides[1] - 1
        ) // self.strides[1]
        z_end_mask = (
            self.z_end + self.strides[2] - 1
        ) // self.strides[2]
        np.testing.assert_array_almost_equal(
            result[:, :x_start_mask, :, :, :],
            np.zeros((
                self.batch_size, x_start_mask, y, z, self.filters
            ))
        )
        np.testing.assert_array_almost_equal(
            result[:, :, :y_start_mask, :, :],
            np.zeros((
                self.batch_size, x, y_start_mask, z, self.filters
            ))
        )
        np.testing.assert_array_almost_equal(
            result[:, :, :, :z_start_mask, :],
            np.zeros((
                self.batch_size, x, y, z_start_mask, self.filters
            ))
        )
        if x_end_mask < x:
            np.testing.assert_array_almost_equal(
                result[:, x_end_mask:, :, :, :],
                np.zeros((
                    self.batch_size, x - x_end_mask, y, z, self.filters
                ))
            )
        if y_end_mask < y:
            np.testing.assert_array_almost_equal(
                result[:, :, y_end_mask:, :, :],
                np.zeros((
                    self.batch_size, x, y - y_end_mask, z, self.filters
                ))
            )
        if z_end_mask < z:
            np.testing.assert_array_almost_equal(
                result[:, :, :, z_end_mask:, :],
                np.zeros((
                    self.batch_size, x, y, z - z_end_mask, self.filters
                ))
            )

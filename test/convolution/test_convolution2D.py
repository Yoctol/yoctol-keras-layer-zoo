'''Convolution 2D class testcase'''
from unittest import TestCase

import numpy as np
from keras.models import Input, Model

from yklz import Convolution2D
from yklz import Mask2D
from test import TestBase2DClass

class TestConvolution2DClass(TestBase2DClass, TestCase):

    def setUp(self):
        super(TestConvolution2DClass, self).setUp()
        self.kernel = (2, 3)
        self.filters = 5
        self.stride = (3, 2)

        self.model = self.create_model()

    def create_model(self):
        inputs = Input(shape=(self.x, self.y, self.channel_size))
        masked_inputs = Mask2D(self.mask_value)(inputs)
        outputs = Convolution2D(
            self.filters, 
            self.kernel,
            self.stride
        )(masked_inputs)
        model = Model(inputs, outputs)
        model.compile('sgd', 'mean_squared_error')
        return model

    def test_output_shape(self):
        result = self.model.predict(self.data)
        x_window_length = (self.x - self.kernel[0]) // self.stride[0] + 1
        y_window_length = (self.y - self.kernel[1]) // self.stride[1] + 1
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
        x_start_mask = (self.x_start - self.kernel[0]) // self.stride[0] + 1
        y_start_mask = (self.y_start - self.kernel[1]) // self.stride[1] + 1
        x_end_mask = (
            self.x_end + self.stride[0] - 1
        ) // self.stride[0]
        y_end_mask = (
            self.y_end + self.stride[1] - 1
        ) // self.stride[1]
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
        if (x_end_mask < x):
            np.testing.assert_array_almost_equal(
                result[:, x_end_mask:, :, :],
                np.zeros((
                    self.batch_size, x - x_end_mask, y, self.filters
                ))
            )
        if (y_end_mask < y):
            np.testing.assert_array_almost_equal(
                result[:, :, y_end_mask:, :],
                np.zeros((
                    self.batch_size, x, y - y_end_mask, self.filters
                ))
            )

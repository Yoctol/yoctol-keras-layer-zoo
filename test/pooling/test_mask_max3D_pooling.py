'''Masked Max3D pooling testcase'''
from unittest import TestCase

import numpy as np
from keras.models import Input, Model
from keras.utils.conv_utils import conv_output_length
from keras.layers.pooling import MaxPool3D

from yklz import MaskPooling
from yklz import MaskConv
from test import TestConvBase3DClass

class TestMaskedMax3DPoolingClass(TestConvBase3DClass, TestCase):
    def setUp(self):
        super(TestMaskedMax3DPoolingClass, self).setUp()
        self.pool_size = self.kernel
        self.filters = self.channel_size

        self.data[
            :,
            self.x_start:self.x_end,
            self.y_start:self.y_end,
            self.z_start:self.z_end,
            :
        ] = -2.0

        self.model = self.create_model()

    def create_model(self):
        inputs = Input(shape=(self.x, self.y, self.z, self.channel_size))
        masked_inputs = MaskConv(self.mask_value)(inputs)
        outputs = MaskPooling(
            MaxPool3D(
                self.pool_size,
                self.strides,
                self.padding
            ),
            pool_mode='max'
        )(masked_inputs)
        model = Model(inputs, outputs)
        model.compile('sgd', 'mean_squared_error')
        return model

    def test_negative_value(self):
        result = self.model.predict(self.data)
        x_start_mask = (self.x_start - self.pool_size[0]) // self.strides[0] + 1
        y_start_mask = (self.y_start - self.pool_size[1]) // self.strides[1] + 1
        z_start_mask = (self.z_start - self.pool_size[2]) // self.strides[2] + 1
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
            result[
                :,
                x_start_mask:x_end_mask,
                y_start_mask:y_end_mask,
                z_start_mask:z_end_mask,
                :
            ],
            np.ones(shape=(
                self.batch_size,
                x_end_mask - x_start_mask,
                y_end_mask - y_start_mask,
                z_end_mask - z_start_mask,
                self.channel_size
            )) * -2.0
        )

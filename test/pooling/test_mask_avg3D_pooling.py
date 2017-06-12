'''Masked Avg3D pooling testcase'''
from unittest import TestCase

import numpy as np
from keras.models import Input, Model
from keras.utils.conv_utils import conv_output_length
from keras.layers.pooling import AvgPool3D

from yklz import MaskPooling
from yklz import MaskConv
from test import TestConvBase3DClass

class TestMaskedAvg3DPoolingClass(TestConvBase3DClass, TestCase):
    def setUp(self):
        super(TestMaskedAvg3DPoolingClass, self).setUp()
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
            AvgPool3D(
                self.pool_size,
                self.strides,
                self.padding
            ),
            pool_mode='avg'
        )(masked_inputs)
        model = Model(inputs, outputs)
        model.compile('sgd', 'mean_squared_error')
        return model

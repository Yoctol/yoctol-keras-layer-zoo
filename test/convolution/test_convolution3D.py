'''MaskConvNet with Conv3D testcase'''
from unittest import TestCase

import numpy as np
from keras.models import Input, Model
from keras.layers import Conv3D

from yklz import MaskConvNet
from yklz import MaskConv
from test import TestConvBase3DClass

class TestConvolution3DClass(TestConvBase3DClass, TestCase):

    def setUp(self):
        super(TestConvolution3DClass, self).setUp()

        self.model = self.create_model()

    def create_model(self):
        inputs = Input(shape=(self.x, self.y, self.z, self.channel_size))
        masked_inputs = MaskConv(self.mask_value)(inputs)
        outputs = MaskConvNet(
            Conv3D(
                self.filters,
                self.kernel,
                strides=self.strides
            )
        )(masked_inputs)
        model = Model(inputs, outputs)
        model.compile('sgd', 'mean_squared_error')
        return model


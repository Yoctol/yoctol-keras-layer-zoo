'''Convolution 2D class testcase'''
from unittest import TestCase

import numpy as np
from keras.models import Input, Model

from yklz import Convolution2D
from yklz import Mask2D
from test import TestConvBase2DClass

class TestConvolution2DClass(TestConvBase2DClass, TestCase):

    def setUp(self):
        super(TestConvolution2DClass, self).setUp()

        self.model = self.create_model()

    def create_model(self):
        inputs = Input(shape=(self.x, self.y, self.channel_size))
        masked_inputs = Mask2D(self.mask_value)(inputs)
        outputs = Convolution2D(
            self.filters,
            self.kernel,
            self.strides
        )(masked_inputs)
        model = Model(inputs, outputs)
        model.compile('sgd', 'mean_squared_error')
        return model


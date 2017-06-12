'''test cases for the masking flatten layer'''
from unittest import TestCase

import numpy as np
import keras.backend as K
from keras.models import Input, Model
from keras.utils.conv_utils import conv_output_length
from keras.layers.pooling import MaxPool2D
from keras.layers import Conv2D

from yklz import MaskConvNet, MaskPooling
from yklz import MaskConv, MaskFlatten
from test import TestBase2DClass

class TestMaskFlattenClass(TestBase2DClass, TestCase):

    def setUp(self):
        super(TestMaskFlattenClass, self).setUp()

        self.kernel = (2, 3)
        self.filters = 5
        self.strides = (3, 2)
        self.padding = 'valid'

        self.conv_x_length = conv_output_length(
            self.x,
            self.kernel[0],
            self.padding,
            self.strides[0]
        )

        self.conv_y_length = conv_output_length(
            self.y,
            self.kernel[1],
            self.padding,
            self.strides[1]
        )

        self.mask_kernel = (3, self.conv_y_length)
        self.mask_strides = (1, 1)

        self.model = self.create_model()

    def create_model(self):
        inputs = Input(shape=(self.x, self.y, self.channel_size))
        masked_inputs = MaskConv(self.mask_value)(inputs)
        conv_outputs = MaskConvNet(
            Conv2D(
                self.filters,
                self.kernel,
                strides=self.strides
            )
        )(masked_inputs)
        pooling_outputs = MaskPooling(
            MaxPool2D(
                self.mask_kernel,
                self.mask_strides,
                self.padding,
            )
        )(conv_outputs)
        outputs = MaskFlatten()(pooling_outputs)
        model = Model(inputs, outputs)
        model.compile('sgd', 'mean_squared_error')
        return model

    def test_output_shape(self):
        result = self.model.predict(self.data)
        final_x_length = conv_output_length(
            self.conv_x_length,
            self.mask_kernel[0],
            self.padding,
            self.mask_strides[0]
        )
        final_y_length = conv_output_length(
            self.conv_y_length,
            self.mask_kernel[1],
            self.padding,
            self.mask_strides[1]
        )
        feature_size = self.filters * final_x_length * final_y_length
        self.assertEqual(
            result.shape,
            (self.batch_size, feature_size)
        )

    def test_mask_value(self):
        mask_cache_key = str(id(self.model.input)) + '_' + str(id(None))
        mask_tensor = self.model._output_mask_cache[mask_cache_key]
        mask = mask_tensor.eval(
            session=K.get_session(),
            feed_dict={self.model.input: self.data}
        )
        self.assertFalse(np.any(mask))

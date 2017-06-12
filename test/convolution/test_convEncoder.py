'''Testcases for ConvEncoder Layer'''
from unittest import TestCase

import numpy as np
import keras.backend as K
from keras.models import Input, Model
from keras.utils.conv_utils import conv_output_length
from keras.layers.pooling import MaxPool2D
from keras.layers import Conv2D

from yklz import MaskConv, MaskConvNet
from yklz import MaskPooling, MaskToSeq
from yklz import ConvEncoder

from test import TestSeq2DClass

class TestConvEncoderClass(TestSeq2DClass, TestCase):

    def setUp(self):
        super(TestConvEncoderClass, self).setUp()
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

        self.mask_kernel = (3, 2)
        self.mask_strides = (1, 1)

        self.final_x = conv_output_length(
            self.conv_x_length,
            self.mask_kernel[0],
            self.padding,
            self.mask_strides[0]
        )

        self.final_y = conv_output_length(
            self.conv_y_length,
            self.mask_kernel[1],
            self.padding,
            self.mask_strides[1]
        )
        self.feature_size = self.final_x * self.final_y * self.filters

        self.model = self.create_model()

    def create_model(self):
        inputs = Input(shape=(self.x, self.y, self.channel_size))
        masked_inputs = MaskConv(self.mask_value)(inputs)
        masked_seq = MaskToSeq(MaskConv(self.mask_value))(inputs)
        conv_outputs = MaskConvNet(
            Conv2D(
                self.filters,
                self.kernel,
                strides=self.strides,
            )
        )(masked_inputs)
        pooling_outputs = MaskPooling(
            MaxPool2D(
                self.mask_kernel,
                self.mask_strides,
                self.padding,
            )
        )(conv_outputs)
        outputs = ConvEncoder()(
            [pooling_outputs, masked_seq]
        )
        model = Model(inputs, outputs)
        model.compile('sgd', 'mean_squared_error')
        return model

    def test_output_shape(self):
        result = self.model.predict(self.data)
        self.assertEqual(
            result.shape,
            (self.batch_size, self.x, self.feature_size)
        )

    def test_output_value(self):
        result = self.model.predict(self.data)
        self.assertTrue(
            np.any(result[:, 0:1, :])
        )

    def test_padding_zeros(self):
        result = self.model.predict(self.data)
        np.testing.assert_almost_equal(
            result[:, self.max_length:, :],
            np.zeros((
                self.batch_size,
                self.x - self.max_length,
                self.feature_size
            ))
        )

    def test_mask(self):
        mask_cache_key = str(id(self.model.input)) + '_' + str(id(None))
        mask_tensor = self.model._output_mask_cache[mask_cache_key]
        mask = mask_tensor.eval(
            session=K.get_session(),
            feed_dict={self.model.input: self.data}
        )
        self.assertFalse(
            np.any(mask[:, self.max_length:])
        )
        self.assertTrue(
            np.all(mask[:, :self.max_length])
        )

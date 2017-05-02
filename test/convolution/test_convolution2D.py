'''Convolution 2D class testcase'''
from unittest import TestCase

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

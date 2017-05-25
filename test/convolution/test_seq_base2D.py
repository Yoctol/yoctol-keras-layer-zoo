'''base class for 2D testcase'''
import numpy as np

class TestSeq2DClass(object):

    def setUp(self):
        self.mask_value = 1.0
        self.batch_size = 100
        self.x = 30
        self.y = 45
        self.channel_size = 7

        self.data = np.random.rand(
            self.batch_size,
            self.x,
            self.y,
            self.channel_size
        )

        self.max_length = 18
        self.fake_x_mask, self.fake_y_mask = 5, 7

        self.data[:, self.max_length:, :, :] = self.mask_value
        self.data[:, self.fake_x_mask, self.fake_y_mask, :] = self.mask_value


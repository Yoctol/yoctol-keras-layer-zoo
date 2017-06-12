'''base class for 3D testcase'''
import numpy as np

class TestBase3DClass(object):

    def setUp(self):
        self.mask_value = 1.0
        self.batch_size = 100
        self.x = 30
        self.y = 45
        self.z = 60
        self.channel_size = 7

        self.data = np.random.rand(
            self.batch_size,
            self.x,
            self.y,
            self.z,
            self.channel_size
        )

        self.x_start, self.x_end = 3, 20
        self.y_start, self.y_end = 4, 32
        self.z_start, self.z_end = 10, 55
        self.fake_x_mask, self.fake_y_mask, self.fake_z_mask = 5, 7, 15

        self.data[:, :self.x_start, :, :, :] = self.mask_value
        self.data[:, self.x_end:, :, :, :] = self.mask_value
        self.data[:, :, :self.y_start, :, :] = self.mask_value
        self.data[:, :, self.y_end:, :, :] = self.mask_value
        self.data[:, :, :, :self.z_start, :] = self.mask_value
        self.data[:, :, :, self.z_end:, :] = self.mask_value
        self.data[
            :,
            self.fake_x_mask,
            self.fake_y_mask,
            self.fake_z_mask,
            :
        ] = self.mask_value


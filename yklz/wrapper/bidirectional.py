import inspect

import keras.backend as K
from keras.layers.wrappers import Bidirectional

class Bidirectional_Encoder(Bidirectional):

    def call(self, inputs, training=None, mask=None):
        kwargs = {}
        func_args = inspect.getargspec(self.layer.call).args
        if 'training' in func_args:
            kwargs['training'] = training
        if 'mask' in func_args:
            kwargs['mask'] = mask

        y = self.forward_layer.call(inputs, **kwargs)
        y_rev = self.backward_layer.call(inputs, **kwargs)
        if self.merge_mode == 'concat':
            output = K.concatenate([y, y_rev])
        elif self.merge_mode == 'sum':
            output = y + y_rev
        elif self.merge_mode == 'ave':
            output = (y + y_rev) / 2
        elif self.merge_mode == 'mul':
            output = y * y_rev
        elif self.merge_mode is None:
            output = [y, y_rev]

        # Properly set learning phase
        if 0 < self.layer.dropout + self.layer.recurrent_dropout:
            if self.merge_mode is None:
                for out in output:
                    out._uses_learning_phase = True
            else:
                output._uses_learning_phase = True
        return output

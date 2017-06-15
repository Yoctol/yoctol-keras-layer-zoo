'''LSTM Peephole test case'''
from unittest import TestCase

from yklz import LSTMPeephole
from .test_rnn_base import TestRNNBaseClass

class TestLSTMCellClass(TestRNNBaseClass, TestCase):

    def setUp(self):
        super(TestLSTMCellClass, self).setUp()
        self.custom_objects['LSTMPeephole'] = LSTMPeephole
        self.model = self.create_model(LSTMPeephole)

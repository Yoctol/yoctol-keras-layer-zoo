'''LSTM Peephole test case'''
from unittest import TestCase

from yklz import LSTMPeephole
from .test_lstm_base import TestLSTMBaseClass

class TestLSTMCellClass(TestLSTMBaseClass, TestCase):

    def setUp(self):
        super(TestLSTMCellClass, self).setUp()
        self.model = self.create_model(LSTMPeephole)

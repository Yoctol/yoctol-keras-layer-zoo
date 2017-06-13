'''LSTM Cell test case'''
from unittest import TestCase

from yklz import LSTMCell
from .test_rnn_base import TestRNNBaseClass

class TestLSTMCellClass(TestRNNBaseClass, TestCase):

    def setUp(self):
        super(TestLSTMCellClass, self).setUp()
        self.model = self.create_model(LSTMCell)

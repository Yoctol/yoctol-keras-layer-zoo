'''LSTM Cell test case'''
from unittest import TestCase

from yklz import LSTMCell
from .test_lstm_base import TestLSTMBaseClass

class TestLSTMCellClass(TestLSTMBaseClass, TestCase):

    def setUp(self):
        super(TestLSTMCellClass, self).setUp()
        self.model = self.create_model(LSTMCell)

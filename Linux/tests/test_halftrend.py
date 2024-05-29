import unittest
import numpy as np
from htrfnwe import halftrend

class TestHalftrend(unittest.TestCase):
    def test_halftrend(self):
        high = np.array([1.2, 1.3, 1.4, 1.5, 1.4, 1.6, 1.7, 1.8, 1.9, 2.0], dtype=np.float32)
        low = np.array([1.0, 1.1, 1.2, 1.3, 1.2, 1.4, 1.5, 1.6, 1.7, 1.8], dtype=np.float32)
        close = np.array([1.1, 1.2, 1.3, 1.4, 1.3, 1.5, 1.6, 1.7, 1.8, 1.9], dtype=np.float32)
        tr = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
        amplitude = 2
        channel_deviation = 2.0

        result = halftrend.halftrend(high, low, close, tr, amplitude, channel_deviation)
        self.assertIn('halftrend', result)
        self.assertIn('atrHigh', result)
        self.assertIn('atrLow', result)
        self.assertIn('arrowUp', result)
        self.assertIn('arrowDown', result)
        self.assertIn('buy', result)
        self.assertIn('sell', result)

if __name__ == '__main__':
    unittest.main()
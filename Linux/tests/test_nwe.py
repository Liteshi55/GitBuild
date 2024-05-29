import unittest
import numpy as np
from htrfnwe import (
    ewma,
    ema2,
    tema3,
    halftrend,
    run_nwe,
    ema,
    range_size,
    range_filter,
    vumanchu_swing,
)

def test_ewma():
    series = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    alpha = 0.5
    result = ewma(series, alpha)
    expected = [1., 1.5, 2.25, 3.125, 4.0625]
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"

# Add more tests for other functions...
class TestHtrfnwe(unittest.TestCase):
    def test_ewma(self):
        series = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        alpha = 0.5
        result = ewma(series, alpha)
        expected = np.array([1., 1.5, 2.25, 3.125, 4.0625], dtype=np.float32)
        np.testing.assert_allclose(result, expected)

    # Add more tests for other functions...

if __name__ == "__main__":
    unittest.main()
if __name__ == "__main__":
    test_ewma()
    print("All tests passed!")


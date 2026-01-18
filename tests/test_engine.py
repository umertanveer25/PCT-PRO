import unittest
import numpy as np
from pct_pro_engine.pct_pro_core import PCTProEngine

class TestPCTProEngine(unittest.TestCase):
    def setUp(self):
        self.engine = PCTProEngine(d_model=32, n_ancestors=4, n_subclades=8)
        self.X = np.random.rand(50, 20)
        self.y = np.random.randint(0, 2, 50)

    def test_fit_predict(self):
        """Test if the engine can fit and predict without errors."""
        self.engine.fit(self.X, self.y)
        preds = self.engine.predict(self.X)
        self.assertEqual(len(preds), 50)
        self.assertTrue(np.all((preds == 0) | (preds == 1)))

    def test_single_class(self):
        """Test robustness to single-class training data."""
        y_single = np.zeros(50)
        self.engine.fit(self.X, y_single)
        preds = self.engine.predict(self.X)
        self.assertTrue(np.all(preds == 0))

if __name__ == '__main__':
    unittest.main()

import unittest
import joblib
from sklearn.linear_model import LinearRegression

class TestModelTraining(unittest.TestCase):
    def test_model_training(self):
    	model = joblib.load('model/Boston_model.pkl')
        self.assertIsInstance(model, LinearRegression)
        self.assertGreaterEqual(len(model.feature_importances_), 4)

if __name__ == '__main__':
    unittest.main()

import unittest

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from my_package import score


class TestEvaluateModel(unittest.TestCase):

    def test_evaluate_model(self):
        # Test data with more than 3 samples
        X_test = np.array([[1], [2], [3], [4], [5]])  # Test features
        y_test = np.array([1, 0, 1, 0, 1])  # Test labels

        # Create a mock prepared dataset (like housing_prepared) and labels (like housing_labels)
        housing_prepared = np.array(
            [[1], [2], [3], [4], [5]]
        )  # Same as X_test in this simple case
        housing_labels = np.array(
            [1, 0, 1, 0, 1]
        )  # Same as y_test in this simple case

        # Create and train a simple RandomForestRegressor model for testing
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(housing_prepared, housing_labels)

        # Call the function to evaluate the model
        final_rmse = score.evaluate_model(
            X_test, y_test, model, housing_prepared, housing_labels
        )

        # Calculate expected RMSE using the real predictions
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        expected_rmse = np.sqrt(mse)

        # Check if RMSE matches
        self.assertAlmostEqual(final_rmse, expected_rmse, places=2)


if __name__ == "__main__":
    unittest.main()

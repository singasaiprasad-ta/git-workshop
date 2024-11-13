import unittest

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from my_package import train


class TestTrainModels(unittest.TestCase):

    def test_prepare_data(self):
        # Mock the DataFrame to test `prepare_data`
        housing = pd.DataFrame(
            {
                "total_rooms": [1000, 2000],
                "households": [500, 1000],
                "total_bedrooms": [300, 400],
                "population": [1500, 2500],
                "ocean_proximity": ["NEAR BAY", "NEAR OCEAN"],
                "median_house_value": [500000, 600000],
            }
        )

        # Call the prepare_data function
        housing_prepared, housing_labels, imputer = train.prepare_data(housing)

        # Test if the imputer was used correctly
        self.assertIsInstance(imputer, SimpleImputer)
        self.assertEqual(
            housing_labels.shape[0], 2
        )  # Ensure labels are correctly extracted
        self.assertEqual(
            housing_prepared.shape[0], 2
        )  # Ensure prepared data has the correct number of rows

    def test_train_models(self):
        # Mock the DataFrame with more data
        housing_prepared = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [6, 7, 8, 9, 10],
                "feature3": [11, 12, 13, 14, 15],
            }
        )
        housing_labels = pd.Series([1, 0, 1, 0, 1])

        # Split the data manually for testing purposes
        X_train, X_test, y_train, y_test = train_test_split(
            housing_prepared, housing_labels, test_size=0.33, random_state=42
        )

        # Call the `train_models` function
        best_model, X_test, y_test = train.train_models(X_train, y_train)

        # You can add assertions to check the behavior
        self.assertIsNotNone(best_model)


if __name__ == "__main__":
    unittest.main()

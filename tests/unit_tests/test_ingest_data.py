# test_ingest_data.py

import os
import unittest

import pandas as pd

from my_package import ingest_data


class TestIngestData(unittest.TestCase):
    def test_fetch_housing_data(self):
        ingest_data.fetch_housing_data()
        self.assertTrue(os.path.isdir("datasets/housing"))
        self.assertTrue(os.path.isfile("datasets/housing/housing.csv"))

    def test_load_housing_data(self):
        housing = ingest_data.load_housing_data()
        self.assertIsInstance(housing, pd.DataFrame)
        columns_expected = [
            "longitude",
            "latitude",
            "housing_median_age",
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            "median_income",
            "median_house_value",
            "ocean_proximity",
        ]
        self.assertListEqual(columns_expected, list(housing.columns))


if __name__ == "__main__":
    unittest.main()

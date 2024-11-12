import os

from my_package.ingest_data import fetch_housing_data, load_housing_data


def test_fetch_housing_data():

    fetch_housing_data()
    assert os.path.exists("datasets/housing"), "The Folder not found."
    assert os.path.exists(
        "datasets/housing/housing.csv"
    ), "the housing.csv file not found."


def test_load_housing_data():

    housing = load_housing_data()
    expected_result = [
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
    assert expected_result == list(
        housing.columns
    ), "The expected result is not same as housing.columns"

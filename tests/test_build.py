import os

import pytest


def test_import_package():
    try:
        import mypackage
    except ImportError:
        pytest.fail("package could not be imported")


def test_fetch_housing_data():
    from mypackage import fetch_housing_data

    fetch_housing_data()
    assert os.path.exists("datasets/housing"), "The Folder not found."
    assert os.path.exists(
        "datasets/housing/houising.csv"
    ), "the housing.csv file not found."


def test_load_housing_data():
    from mypackage import load_housing_data

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
    assert expected_result == list(housing.cloumns)

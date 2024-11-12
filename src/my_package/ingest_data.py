# data_ingestion.py
import os
import tarfile
import urllib.request

import pandas as pd

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """Fetches and extracts the housing dataset."""
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    with tarfile.open(tgz_path) as housing_tgz:
        housing_tgz.extractall(path=housing_path)


def load_housing_data(housing_path=HOUSING_PATH):
    """Loads the housing dataset into a pandas DataFrame."""
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


if __name__ == "__main__":
    fetch_housing_data()  # Example to test if the function works
    housing = load_housing_data()
    print(housing.head())  # Print a preview of the data

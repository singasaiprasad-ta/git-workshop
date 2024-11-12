import importlib

import pytest


def test_ingest_data_import():
    try:
        ingest_data = importlib.import_module("my_package.ingest_data")

        # Ensure that the functions exist in the module
        fetching_data = getattr(ingest_data, "fetch_housing_data", None)
        loading_data = getattr(ingest_data, "load_housing_data", None)

        if fetching_data is None or loading_data is None:
            pytest.fail("Functions not found in ingest_data.py")

    except ModuleNotFoundError:
        pytest.fail(
            "Failed to import the module 'ingest_data' from my_package."
        )
    except Exception as e:
        pytest.fail(f"An unexpected error occurred: {e}")

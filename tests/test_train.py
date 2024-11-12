import importlib

import pytest


def test_train_import():
    try:
        training = importlib.import_module("my_package.train")

        # Ensure that the functions exist in the module
        fun_1 = getattr(training, "prepare_data", None)
        fun_2 = getattr(training, "train_models", None)

        if fun_1 is None or fun_2 is None:
            pytest.fail("Functions not found in train.py")

    except ModuleNotFoundError:
        pytest.fail("Failed to import the module 'train' from my_package.")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred: {e}")

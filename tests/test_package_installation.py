import importlib

import pytest


def test_package_installation():
    try:
        importlib.import_module("my_package")

    except ModuleNotFoundError:
        pytest.fail("failed to import my_package")

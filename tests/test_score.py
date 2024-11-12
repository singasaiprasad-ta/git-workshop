import importlib

import pytest


def test_scoring_import():
    try:
        evaluation = importlib.import_module("my_package.score")

        # Ensure that the functions exist in the module
        final_result = getattr(evaluation, "evaluate_model", None)

        if final_result is None:
            pytest.fail("Functions not found in score.py")

    except ModuleNotFoundError:
        pytest.fail("Failed to import the module 'score' from my_package.")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred: {e}")

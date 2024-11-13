import argparse
import os

import joblib
import pandas as pd

from my_package import score


def main():
    parser = argparse.ArgumentParser(description="Evaluate ML Model")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the saved model"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the dataset for evaluation",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to the directory where the test data is saved",
    )
    args = parser.parse_args()

    # Load model
    model = joblib.load(args.model_path)

    # Load the saved test data
    X_test = joblib.load(os.path.join(args.output_path, "X_test.csv"))
    y_test = joblib.load(os.path.join(args.output_path, "y_test.csv"))

    # Load the dataset (assuming it's already prepared)
    housing = pd.read_csv(args.data_path)

    # Directly use the raw (or already prepared) data for evaluation
    housing_prepared = housing.drop(
        "median_house_value", axis=1
    )  # Example of simple preparation (remove target)
    housing_labels = housing[
        "median_house_value"
    ]  # Example of using the target column for labels
    housing_prepared, housing_labels = score.prepare_data_for_scoring(housing)

    # Evaluate the model
    final_rmse = score.evaluate_model(
        X_test, y_test, model, housing_prepared, housing_labels
    )
    print(f"Final RMSE of the model: {final_rmse}")


if __name__ == "__main__":
    main()

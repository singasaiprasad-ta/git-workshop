import argparse
import os

import joblib
import pandas as pd

from my_package import train


def main():
    parser = argparse.ArgumentParser(description="Train ML Models")
    parser.add_argument(
        "--input-path", type=str, required=True, help="Path to the dataset"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Directory to save the model",
    )
    args = parser.parse_args()

    # Load housing data
    housing = pd.read_csv(
        args.input_path
    )  # Assuming housing data is in CSV format
    housing_prepared, housing_labels, imputer = train.prepare_data(housing)

    # Train models and save the best one
    best_model, X_test, y_test = train.train_models(
        housing_prepared, housing_labels
    )
    # Save the trained model
    joblib.dump(best_model, os.path.join(args.output_path, "best_model.pkl"))

    print(
        f"Best model saved to: {os.path.join(args.output_path, 'best_model.pkl')}"
    )


if __name__ == "__main__":
    main()

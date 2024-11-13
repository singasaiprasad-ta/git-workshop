# main.py
from my_package.ingest_data import fetch_housing_data, load_housing_data
from my_package.score import evaluate_model
from my_package.train import prepare_data, train_models


def main():
    # Step 1: Data Ingestion
    print("Fetching the housiong data")
    fetch_housing_data()
    housing = load_housing_data()
    print(housing.head())

    # Step 2: Prepare Data for Training
    print("Preparing data for training...")
    housing_prepared, housing_labels, imputer = prepare_data(housing)

    # Step 3: Train the models
    print("Training models...")
    best_model, X_test, y_test = train_models(housing_prepared, housing_labels)

    # Step 4: Evaluate the model
    print("Evaluating the model...")
    final_rmse = evaluate_model(
        X_test, y_test, best_model, housing_prepared, housing_labels
    )
    print(final_rmse)


if __name__ == "__main__":
    main()

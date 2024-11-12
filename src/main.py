# main.py
from my_package.ingest_data import fetch_housing_data, load_housing_data
from my_package.score import evaluate_model
from my_package.train import prepare_data, train_models


def main():
   """ Step 1: Data Ingestion """
    print("Fetching housing data...")
    fetch_housing_data()
    housing = load_housing_data()

    """" Step 2: Prepare Data for Training """
    print("Preparing data for training...")
    housing_prepared, housing_labels, imputer = prepare_data(housing)

    """ Step 3: Train the models """
    print("Training models...")
    best_model, X_test, y_test = train_models(housing_prepared, housing_labels)

    """  Step 4: Evaluate the model """
    print("Evaluating the model...")
    evaluate_model(X_test, y_test, best_model)


if __name__ == "__main__":
    main()

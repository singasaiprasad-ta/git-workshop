# score.py
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressord
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV


def prepare_data_for_scoring(housing):
    """Prepares the test data using the same transformations as the training data."""

    # Load the saved imputer from the training phase
    imputer = joblib.load(
        "imputer.pkl"
    )  # Load the imputer saved during training

    # Prepare numeric features as during training
    housing["rooms_per_household"] = (
        housing["total_rooms"] / housing["households"]
    )
    housing["bedrooms_per_room"] = (
        housing["total_bedrooms"] / housing["total_rooms"]
    )
    housing["population_per_household"] = (
        housing["population"] / housing["households"]
    )

    # Drop the target column to avoid including it in the features
    housing_labels = housing["median_house_value"].copy()
    housing = housing.drop("median_house_value", axis=1)

    # Handle missing values in the same way as during training using the loaded imputer
    housing_num = housing.drop(
        "ocean_proximity", axis=1
    )  # Drop categorical column
    X = imputer.transform(
        housing_num
    )  # Transform the test data using the imputer

    housing_tr = pd.DataFrame(
        X, columns=housing_num.columns, index=housing.index
    )

    # Apply one-hot encoding as done in training
    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(
        pd.get_dummies(housing_cat, drop_first=True)
    )

    return housing_prepared, housing_labels


def evaluate_model(X_test, y_test, model, housing_prepared, housing_labels):
    """Evaluate the trained model on the test set."""
    # Make predictions

    param_grid = [
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        {
            "bootstrap": [False],
            "n_estimators": [3, 10],
            "max_features": [2, 3, 4],
        },
    ]

    # Setup the RandomForestRegressor with a fixed random state for reproducibility
    forest_reg = RandomForestRegressor(random_state=42)

    # Perform GridSearchCV
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )

    # Fit the model on the training data
    grid_search.fit(housing_prepared, housing_labels)

    # Get the best hyperparameters
    # best_params = grid_search.best_params_
    # print("Best hyperparameters:", best_params)

    # Print the GridSearchCV results
    cv_results = grid_search.cv_results_
    print("\nGridSearchCV Results:")
    for mean_score, params in zip(
        cv_results["mean_test_score"], cv_results["params"]
    ):
        print(np.sqrt(-mean_score), params)

    # Feature importances for the best model
    # feature_importances = grid_search.best_estimator_.feature_importances_
    # print("\nFeature Importances:")
    # print(
    #     sorted(
    #         zip(feature_importances, housing_prepared.columns), reverse=True
    #     )
    # )
    final_predictions = model.predict(X_test)

    # Calculate RMSE
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    return final_rmse

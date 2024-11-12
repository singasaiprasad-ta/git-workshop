# score.py
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV


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
    print(f"Final RMSE: {final_rmse}")

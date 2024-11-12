import joblib

# training.py
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor


def prepare_data(housing):
    """Prepares data for model training."""
    housing["rooms_per_household"] = (
        housing["total_rooms"] / housing["households"]
    )
    housing["bedrooms_per_room"] = (
        housing["total_bedrooms"] / housing["total_rooms"]
    )
    housing["population_per_household"] = (
        housing["population"] / housing["households"]
    )

    housing_labels = housing["median_house_value"].copy()
    housing = housing.drop(
        "median_house_value", axis=1
    )  # Drop the target for training

    # Handling missing values
    imputer = SimpleImputer(strategy="median")
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(
        X, columns=housing_num.columns, index=housing.index
    )
    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )

    # One-hot encoding for categorical variables
    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(
        pd.get_dummies(housing_cat, drop_first=True)
    )

    return housing_prepared, housing_labels, imputer


def train_models(housing_prepared, housing_labels):
    """Train multiple models and tune hyperparameters."""
    # Split the data into training and testing sets (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        housing_prepared, housing_labels, test_size=0.2, random_state=42
    )

    # Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    lin_predictions = lin_reg.predict(X_train)
    lin_mse = mean_squared_error(y_train, lin_predictions)
    lin_rmse = np.sqrt(lin_mse)
    print(f"Linear Regression RMSE: {lin_rmse}")

    # Decision Tree Regressor
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(X_train, y_train)
    tree_predictions = tree_reg.predict(X_train)
    tree_mse = mean_squared_error(y_train, tree_predictions)
    tree_rmse = np.sqrt(tree_mse)
    print(f"Decision Tree RMSE: {tree_rmse}")

    # Random Forest Regressor with Hyperparameter Tuning
    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }

    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    rnd_search.fit(X_train, y_train)
    best_model = rnd_search.best_estimator_
    rnd_search.fit(housing_prepared, housing_labels)
    cvres = rnd_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    print(f"Best Random Forest Model: {best_model}")

    # Save the best model
    joblib.dump(best_model, "best_model.pkl")

    return best_model, X_test, y_test

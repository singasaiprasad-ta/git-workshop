# score.py
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error


def evaluate_model(X_test, y_test, model):
    """Evaluate the trained model on the test set."""
    # Make predictions
    final_predictions = model.predict(X_test)

    # Calculate RMSE
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print(f"Final RMSE: {final_rmse}")

    return final_rmse


# if __name__ == "__main__":
#     # Load the best trained model
#     model = joblib.load("best_model.pkl")
#     evaluate_model(X_test, y_test, model)

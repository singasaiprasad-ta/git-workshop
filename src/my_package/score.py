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


if __name__ == "__main__":
    # Load the best trained model
    model = joblib.load("best_model.pkl")

    # Assuming we already have X_test and y_test from the training.py
    # Here you could load the dataset again (e.g., from CSV or any other source)
    # But since you want to directly use the split test set from training.py,
    # We will directly call evaluate_model after loading the model.

    # Use X_test and y_test from training.py
    # Assuming you've passed them correctly from `train_models` function
    # Example:
    # X_test, y_test = ... (passed from train.py)

    # Directly use evaluate_model by passing X_test, y_test from train.py
    # For now, this is just for structure purpose, you will call this with actual values
    evaluate_model(X_test, y_test, model)

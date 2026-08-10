import numpy as np
from numpy.typing import NDArray


class LinearRegression:
    def get_model_prediction(
        self,
        feature_matrix: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        # Compute predicted outputs by taking the dot product of features and weights: y_hat = X * W
        predictions = np.dot(feature_matrix, weights)

        return np.round(predictions, 5)

    def get_error(
        self,
        model_predictions: NDArray[np.float64],
        ground_truth: NDArray[np.float64],
    ) -> float:
        # Total number of samples
        num_samples = len(model_predictions)

        # Calculate squared error for each prediction: (y_hat - y)^2
        squared_errors = np.square(model_predictions - ground_truth)

        # Mean Squared Error formula: 1/N * sum(squared_errors)
        mean_squared_error = (1 / num_samples) * np.sum(squared_errors)

        return round(float(mean_squared_error), 5)

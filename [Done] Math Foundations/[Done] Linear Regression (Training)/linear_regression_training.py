import numpy as np
from numpy.typing import NDArray


class LinearRegressionTraining:
    learning_rate: float = 0.01

    def get_derivative(
        self,
        model_predictions: NDArray[np.float64],
        ground_truth: NDArray[np.float64],
        num_samples: int,
        feature_matrix: NDArray[np.float64],
        desired_weight_index: int,
    ) -> float:
        # Extract feature column values corresponding to the specific weight index
        feature_column = feature_matrix[:, desired_weight_index]

        # Calculate prediction error residuals: (y - y_hat)
        prediction_errors = ground_truth - model_predictions

        # Partial derivative of MSE with respect to weight j: -2/N * sum((y - y_hat) * x_j)
        weight_gradient = -2 * np.dot(prediction_errors, feature_column) / num_samples

        return float(weight_gradient)

    def get_model_prediction(
        self,
        feature_matrix: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        # Compute predictions using matrix multiplication: y_hat = X * W
        predictions = np.matmul(feature_matrix, weights)

        return np.squeeze(predictions)

    def train_model(
        self,
        feature_matrix: NDArray[np.float64],
        ground_truth: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        # Initialize weights and determine dimensions
        weights = initial_weights
        num_weights = len(weights)
        num_samples = len(feature_matrix)

        # Run gradient descent training loop
        for _ in range(num_iterations):
            predictions = self.get_model_prediction(feature_matrix, weights)

            # Update each weight using its computed partial derivative
            for weight_index in range(num_weights):
                gradient = self.get_derivative(
                    predictions,
                    ground_truth,
                    num_samples,
                    feature_matrix,
                    weight_index,
                )
                weights[weight_index] -= self.learning_rate * gradient

        return np.round(weights, 5)
import numpy as np
from numpy.typing import NDArray


class CrossEntropy:
    def binary_cross_entropy(
        self,
        true_labels: NDArray[np.float64],
        predicted_probabilities: NDArray[np.float64],
    ) -> float:
        # Small epsilon to avoid log(0) numerical instability
        epsilon = 1e-15
        clipped_probabilities = np.clip(
            predicted_probabilities,
            epsilon,
            1.0 - epsilon,
        )

        # Total number of samples in the batch
        num_samples = len(true_labels)

        # Binary cross-entropy loss formula: -1/N * sum(y * log(p) + (1 - y) * log(1 - p))
        binary_loss = -(1 / num_samples) * np.sum(
            true_labels * np.log(clipped_probabilities)
            + (1 - true_labels) * np.log(1 - clipped_probabilities)
        )

        return round(float(binary_loss), 4)

    def categorical_cross_entropy(
        self,
        true_labels: NDArray[np.float64],
        predicted_probabilities: NDArray[np.float64],
    ) -> float:
        # Small epsilon to avoid log(0) numerical instability
        epsilon = 1e-15
        clipped_probabilities = np.clip(
            predicted_probabilities,
            epsilon,
            1.0 - epsilon,
        )

        # Total number of samples in the batch
        num_samples = len(true_labels)

        # Categorical cross-entropy loss formula: -1/N * sum(y * log(p))
        categorical_loss = -(1 / num_samples) * np.sum(
            true_labels * np.log(clipped_probabilities)
        )

        return round(float(categorical_loss), 4)

import numpy as np
from numpy.typing import NDArray


class ActivationFunctions:
    def sigmoid(
        self,
        input_values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        # Sigmoid activation formula: 1 / (1 + e^(-x)), squashes values into (0, 1)
        sigmoid_output = 1 / (1 + np.exp(-input_values))

        return np.round(sigmoid_output, 5)

    def relu(
        self,
        input_values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        # ReLU activation formula: max(0, x), sets negative values to 0 while keeping positives
        relu_output = np.maximum(0.0, input_values)

        return relu_output

import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class BackPropForNeuron:
    def backward(
        self,
        input_values: NDArray[np.float64],
        weights: NDArray[np.float64],
        bias: float,
        target_value: float,
    ) -> Tuple[NDArray[np.float64], float]:
        """Computing the gradients of the loss using the chain rule for a single neuron
        """

        def sigmoid(value: float) -> float:
            return 1 / (1 + np.exp(-value))

        weighted_sum = np.dot(input_values, weights) + bias
        prediction = sigmoid(weighted_sum)

        error = prediction - target_value
        sigmoid_derivative = prediction * (1 - prediction)

        gradient_with_respect_to_weights = error * sigmoid_derivative * input_values
        gradient_with_respect_to_bias = error * sigmoid_derivative

        return (
            np.round(gradient_with_respect_to_weights, 5),
            np.round(gradient_with_respect_to_bias, 5),
        )

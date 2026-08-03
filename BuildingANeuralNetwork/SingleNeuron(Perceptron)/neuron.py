import numpy as np
from numpy.typing import NDArray


class SingleNeuron:
    def forward(
        self,
        input_values: NDArray[np.float64],
        weights: NDArray[np.float64],
        bias: float,
        activation: str,
    ) -> float:
        """Passing inputs through a single neuron with a chosen activation function."""

        weighted_sum = np.dot(input_values, weights) + bias

        def sigmoid(value: float) -> float:
            return 1 / (1 + np.exp(-value))

        def relu(value: float) -> float:
            return max(0.0, value)

        if activation == "sigmoid":
            output = sigmoid(weighted_sum)
        elif activation == "relu":
            output = relu(weighted_sum)

        return round(float(output), 5)

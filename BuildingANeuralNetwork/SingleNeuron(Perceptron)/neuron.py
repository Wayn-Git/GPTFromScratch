import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        # Formula for an perceptron = input x weights + bias -> activation -> output

        

        z = np.dot(x, w) + b
        z = float(z)
        sigmoid = 1/(1+ np.exp(-z))
        relu = np.maximum(0, z)

        if activation == "sigmoid":
            output = float(sigmoid)
        else:
            output = float(relu)



        return round (output, 5)

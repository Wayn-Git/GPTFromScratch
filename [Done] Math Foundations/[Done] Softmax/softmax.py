import numpy as np
from numpy.typing import NDArray


class Softmax:
    def softmax(
        self,
        logits: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        # Step 1: Subtract maximum logit for numerical stability to prevent exponential overflow
        stabilized_logits = logits - np.max(logits)

        # Step 2: Exponentiate stabilized logits to make all values strictly positive
        exponentiated_logits = np.exp(stabilized_logits)

        # Step 3: Sum all exponentials to compute the normalization denominator
        sum_of_exponentials = np.sum(exponentiated_logits)

        # Step 4: Divide exponentials by sum to get valid probabilities summing to 1
        probabilities = exponentiated_logits / sum_of_exponentials

        return np.round(probabilities, 4)

import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))

        n = X.shape[0]
        w = np.zeros(X.shape[1])
        b = 0.0

        

        print(X.shape)
        print(np.array(w).shape)

        

        for i in range(epochs):
            y_pred = np.dot(X, w) + b

            
            loss = y_pred - y

            dW = (2 / n) * np.dot(X.T, loss)
            db = (2 / n) * np.sum(loss)



            w -= lr * dW
            b -= lr * db



        return (np.round(w, 5 ), round(b, 5))

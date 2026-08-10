# Linear Regression with Gradient Descent

Linear Regression is one of the simplest supervised machine learning algorithms used for predicting continuous values. The model learns a linear relationship between the input features and the target by finding the set of weights that minimizes the prediction error. In this notebook, every part of the learning process is implemented from scratch using only NumPy, including prediction, gradient computation, and weight updates.

## Model Prediction

The model predicts an output by computing the weighted sum of the input features.

$$
\hat{y} = XW
$$

Where:

* $X$ is the feature matrix.
* $W$ is the weight vector.
* $\hat{y}$ is the predicted output.

## Mean Squared Error (MSE)

To measure how well the model performs, we use the **Mean Squared Error (MSE)** loss function. It calculates the average squared difference between the predicted values and the actual values. A lower loss indicates that the predictions are closer to the true targets.

$$
L = \frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2
$$

Where:

* $N$ is the number of training samples.
* $y_i$ is the true value.
* $\hat{y}_i$ is the predicted value.

## Gradient

To reduce the loss, we compute the derivative of the loss with respect to each weight. The gradient tells us how much each weight contributes to the prediction error and in which direction it should be adjusted.

$$
\frac{\partial L}{\partial w_j}
===============================

-\frac{2}{N}
\sum_{i=1}^{N}
(y_i-\hat{y}*i)x*{ij}
$$

## Gradient Descent

Once the gradient has been computed, each weight is updated using Gradient Descent.

$$
w_j = w_j - \alpha\frac{\partial L}{\partial w_j}
$$

Where:

* $w_j$ is the current weight.
* $\alpha$ is the learning rate.
* $\frac{\partial L}{\partial w_j}$ is the gradient of the loss.

This process is repeated for multiple iterations. After every update, the loss decreases, allowing the model to gradually learn the weights that best fit the training data.

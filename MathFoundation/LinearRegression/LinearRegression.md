# Linear Regression

Linear Regression is one of the simplest supervised machine learning algorithms used for predicting continuous values. The model learns a linear relationship between the input features and the target by finding the set of weights that minimizes the prediction error. 

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
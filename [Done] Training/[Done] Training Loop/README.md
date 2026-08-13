# Training Loop [Basic training loop for a single neuron]

So in this module, we're going to implement the training loop for a single neuron linear regression model. We'll understand the flow of how the model processes the input, calculates the loss and the gradients, and updates the weights and bias for a number of epochs (the amount of times the training loop should run and update the weights and bias).

Firstly, we start with our model, which is a simple linear regression formula:

$$\hat{y} = X \cdot w + b$$

To build this, we need a few key components:

* **Weight & Bias:** We put in our weights and biases where the weight is an array of zeros with respect to the input's shape, and the bias is initially initialized to zero. (This is the simplest training flow, so we didn't implement complex weight initialization).
* **Features:** We have our independent features ($X$) and our dependent feature ($y$).
* **Epochs:** These are basically the number of training iterations.
* **Learning Rate:** This controls how big or how small the numbers should change. If it is too high, it'll jump across the local minimum, or if it is too small, it can just freeze or die.
* **Loss Function:** Used to calculate our error.

---

## The Training Flow

The flow is pretty simple. This sequence repeats until we reach our max epochs, watching the loss function being minimized:

1. **Forward Pass:** Pass data through our model.
2. **Calculate Loss:** Calculate the loss using either MSE, MAE, or RMSE.
3. **Calculate Gradients:** Calculate the derivative of the loss and our model once with respect to $w$ (weight) and once with respect to $b$ (bias).
4. **Update Weights (Gradient Descent):** Update the weights using the learning rate and the derivatives.

$$w = w - (lr \cdot \text{derivative of } w)$$
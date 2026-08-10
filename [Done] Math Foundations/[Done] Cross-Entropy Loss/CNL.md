# Cross-Entropy Loss

Cross-Entropy Loss is a loss function that calculates how different the model's prediction is from the true label. It works by penalizing the model heavily for a bad prediction, resulting in a **high** loss. For a correct prediction, the resulting loss is relatively **low**. This behavior is achieved by using the **logarithm** function.

## General Cross-Entropy Formula

\[
L = -\sum_{k=1}^{C} y_k \log(p_k)
\]

Where:

- \(C\) = Number of classes
- \(y_k\) = True probability (or true label)
- \(p_k\) = Predicted probability

There are mainly two types we're going to cover:

## Binary Cross-Entropy Loss

Binary Cross-Entropy Loss is used when we calculate the loss between the predicted value and the true value, where the result is binary (**0** or **1**). There are only two possible classes to predict, whether something is **True** or **False**.

### Formula

For a single sample:

\[
L = -\left[y\log(p) + (1-y)\log(1-p)\right]
\]

For a batch of \(N\) samples:

\[
L = -\frac{1}{N}\sum_{i=1}^{N}
\left[
y_i\log(p_i)
+
(1-y_i)\log(1-p_i)
\right]
\]

### Theoretical Example

Let's say we have samples of cat pictures and we want the model to predict whether it's a cat or not.

We'd give our trained model a picture of a cat, and it'll return either **1** or **0** based on what it has learned, giving us its prediction.

---

## Categorical Cross-Entropy Loss

Categorical Cross-Entropy Loss is used when there are **more than two classes** to predict. Instead of returning a single value like **0** or **1**, the model returns a probability for **every class**, and the class with the highest probability becomes the model's prediction.

### Formula

For a single sample:

\[
L = -\sum_{k=1}^{C} y_k \log(p_k)
\]

For a batch of \(N\) samples:

\[
L = -\frac{1}{N}
\sum_{i=1}^{N}
\sum_{k=1}^{C}
y_{ik}\log(p_{ik})
\]

### Theoretical Example

Let's say we have samples of different animals, and we want the model to predict whether the image contains a **Cat**, **Dog**, or **Bird**.

We'd give our trained model an image, and instead of returning **0** or **1**, it'll return a probability for each class, such as:

```text
Cat  = 0.10
Dog  = 0.85
Bird = 0.05
```

Since **Dog** has the highest probability, the model predicts that the image contains a dog. Categorical Cross-Entropy then calculates the loss by comparing these predicted probabilities with the true label.
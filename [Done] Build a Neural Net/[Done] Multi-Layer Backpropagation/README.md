# Backpropagation in a Multi-Layer Perceptron

We've already learnt how backpropagation works for a single perceptron. Now we can extend the same idea to a **Multi-Layer Perceptron (MLP)**.

The main difference is that instead of calculating the gradient for one layer, we need to calculate the gradients for **every layer** in the network.

Consider this simple MLP:

$$
x \rightarrow z_1 \rightarrow a_1 \rightarrow z_2 \rightarrow \hat{y} \rightarrow L
$$

Where:

* $z_1$ = output of the first linear layer
* $a_1$ = output after applying the activation function
* $z_2$ = output of the second linear layer
* $\hat{y}$ = prediction
* $L$ = loss

---

## 1. Forward Propagation

First, the input moves through the network normally.

### First layer

$$
z_1 = W_1x + b_1
$$

Then we apply an activation function such as ReLU:

$$
a_1 = ReLU(z_1)
$$

### Second layer

$$
z_2 = W_2a_1 + b_2
$$

The output of the second layer becomes our prediction:

$$
\hat{y} = z_2
$$

Then we calculate the loss.

For MSE:

$$
L = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)^2
$$

---

## 2. The Chain Rule

This is where backpropagation actually happens.

We want to know:

$$
\frac{\partial L}{\partial W_1}
$$

But $L$ doesn't directly depend on $W_1$.

The path is:

$$
W_1
\rightarrow
z_1
\rightarrow
a_1
\rightarrow
z_2
\rightarrow
L
$$

So we use the chain rule:

$$
\frac{\partial L}{\partial W_1}
=

\frac{\partial L}{\partial z_2}
\frac{\partial z_2}{\partial a_1}
\frac{\partial a_1}{\partial z_1}
\frac{\partial z_1}{\partial W_1}
$$

**That's the key idea behind multi-layer backpropagation.**

We calculate the derivative at each step and multiply them together.

---

## 3. Start From the Loss

For MSE:

$$
L = \frac{1}{N}\sum(\hat{y}-y)^2
$$

The derivative with respect to the prediction is:

$$
\frac{\partial L}{\partial \hat{y}}
=

\frac{2(\hat{y}-y)}{N}
$$

Since:

$$
\hat{y}=z_2
$$

we get:

$$
\frac{\partial L}{\partial z_2}
=

\frac{2(\hat{y}-y)}{N}
$$

We can call this:

$$
\delta_2 =
\frac{2(\hat{y}-y)}{N}
$$

---

## 4. Gradient of the Second Layer

The second layer is:

$$
z_2 = W_2a_1+b_2
$$

For the weights:

$$
\frac{\partial z_2}{\partial W_2}=a_1
$$

Therefore:

$$
\frac{\partial L}{\partial W_2}
=

\delta_2a_1^T
$$

For the bias:

$$
\frac{\partial z_2}{\partial b_2}=1
$$

Therefore:

$$
\frac{\partial L}{\partial b_2}
=

\delta_2
$$

---

## 5. Move Back to the First Layer

Now we need the gradient flowing into $a_1$.

Since:

$$
z_2=W_2a_1+b_2
$$

the derivative is:

$$
\frac{\partial z_2}{\partial a_1}=W_2
$$

So:

$$
\frac{\partial L}{\partial a_1}
=

\delta_2W_2
$$

But $a_1$ came from ReLU:

$$
a_1=ReLU(z_1)
$$

The derivative of ReLU is:

$$
ReLU'(z_1)=
\begin{cases}
1 & z_1>0\
0 & z_1\leq0
\end{cases}
$$

Therefore:

$$
\delta_1
=

\frac{\partial L}{\partial a_1}
\odot
ReLU'(z_1)
$$

where $\odot$ means element-wise multiplication.

---

## 6. Gradient of the First Layer

The first layer is:

$$
z_1=W_1x+b_1
$$

The derivative with respect to $W_1$ is based on the input:

$$
\frac{\partial z_1}{\partial W_1}=x
$$

Therefore:

$$
\frac{\partial L}{\partial W_1}
=

\delta_1x^T
$$

And for the bias:

$$
\frac{\partial L}{\partial b_1}
=
\delta_1
$$

---

## The Entire Backpropagation

So the process is basically:

$$
L
\rightarrow
z_2
\rightarrow
a_1
\rightarrow
z_1
$$

At every step, we apply the chain rule.

The gradients are:

$$
\delta_2
=

\frac{2(\hat{y}-y)}{N}
$$

$$
\frac{\partial L}{\partial W_2}
=

\delta_2a_1^T
$$

$$
\frac{\partial L}{\partial b_2}
=

\delta_2
$$

$$
\delta_1
=

(\delta_2W_2)
\odot
ReLU'(z_1)
$$

$$
\frac{\partial L}{\partial W_1}
=
\delta_1x^T
$$

$$
\frac{\partial L}{\partial b_1}
=
\delta_1
$$

### The important idea

Backpropagation isn't a separate mysterious algorithm.

**It's the chain rule applied repeatedly from the loss back through every operation in the network.**

Forward propagation tells us:

$$
x\rightarrow z_1\rightarrow a_1\rightarrow z_2\rightarrow L
$$

Backpropagation calculates the gradients in the opposite direction:

$$
L\rightarrow z_2\rightarrow a_1\rightarrow z_1
$$

That is what allows us to figure out **how much every weight and bias contributed to the final error**.

# Gradient Descent

Gradient Descent is an optimization algorithm whose goal is to find the minimum *(it's trying to find a minimum, not necessarily the global minimum)* of a function by repeatedly taking small steps downhill.

Gradient Descent knows where the uphill direction is so it can move in the opposite direction, but it **doesn't** know:

- where the minimum is,
- what the graph looks like,
- or how many steps remain.

It makes every decision using **local information only**.

---

# How does it know where uphill is?

It doesn't magically "know" where uphill is.

Instead, it computes the **derivative** at its current position.

For example, consider the function:

```text
f(x) = x²
```

Suppose we're currently at:

```text
x = 5
```

The derivative of `x²` is:

```text
f'(x) = 2x
```

Substituting `x = 5`:

```text
f'(5) = 2(5)
```

which gives:

```text
10
```

The number **10** tells us that we're standing on the **positive (right) side** of the graph.

Since the derivative is positive, the function is increasing as we move to the right, meaning the downhill direction is to the **left**.

This is exactly why the Gradient Descent update rule is:

```python
x = x - learning_rate * derivative
```

We **subtract** the derivative because the derivative always points uphill, and we want to move in the opposite direction.

---

# Example

## Why subtract?

Suppose you ignored the minus sign and instead wrote:

```python
x = x + learning_rate * derivative
```

Using a learning rate of `0.01`:

```text
x = 5 + 0.01 × 10
```

which becomes:

```text
x = 5.1
```

Did we get closer to zero?

**No.**

We moved:

```text
5 → 5.1
```

We actually climbed **higher up the hill**.

Now use subtraction.

```python
x = x - learning_rate * derivative
```

```text
x = 5 - 0.01 × 10
```

```text
x = 4.9
```

Now we moved:

```text
5 → 4.9
```

That's toward the minimum.

---

# What if you're on the other side?

Suppose:

```text
x = -5
```

The derivative becomes:

```text
-10
```

Now apply the same update rule:

```text
x = -5 - 0.01 × (-10)
```

Notice something interesting.

We're subtracting a negative number.

```text
-5 + 0.1
```

which becomes:

```text
-4.9
```

So we moved:

```text
-5 → -4.9
```

Which is again **toward zero**.

The exact same update rule works on both sides of the graph.

---

# Visualizing the Function

Taking the function `x²` as an example, the graph looks like this.

<img src="[Done] Math Foundations/[Done] Gradient Descent/ images/fxgraph.png" alt="x² Graph" width="400">

The minimum is located at:

```text
x = 0
```

Notice that Gradient Descent never knows this beforehand.

It simply keeps asking:

- Where am I?
- What's the derivative here?
- Take one small step downhill.

---

# Why is Gradient Descent important?

Gradient Descent is used by almost every modern machine learning model during training.

Large Language Models such as:

- GPT-3
- GPT-4
- Llama 2
- Llama 3

are all trained using Gradient Descent-based optimization algorithms, most commonly **Adam (Adaptive Moment Estimation)** or **AdamW**.

Those optimizers are essentially improved versions of Gradient Descent.

---

# The Problem

In the following problem, we'll implement Gradient Descent from scratch.

## Algorithm

1. Compute the gradient of `f(x) = x²`, which is:

```text
f'(x) = 2x
```

The derivative tells us the direction and steepness of the slope at the current position.

2. Begin at the provided starting value `init`.

This is our initial estimate of where the minimum might be.

3. On every iteration:

```text
x_new = x_old - α × f'(x_old)
```

where `α` is the learning rate.

The learning rate determines how large each step should be.

4. Repeat the update for the specified number of iterations.

With every iteration, `x` moves closer to the point where the derivative becomes:

```text
f'(x) = 0
```

which is the minimum of the function.

---


## Key Takeaways

- Gradient points toward the steepest increase.
- Gradient descent moves in the opposite direction.
- Learning rate controls step size.
- Too large a learning rate can overshoot the minimum.
- Too small a learning rate converges slowly.

### Videos to refer

[3Blue1Brown -- Gradient descent, how neural networks learn](ttps://youtu.be/IHZwWFHWa-w?si=kHoSLZUWTpUB96Nw)

[Andrej Karpathy -- micrograd: gradient descent optimization (starts at training)](https://youtu.be/VMj-3S1tku0?si=QYiqR-Ex7sgbxEz8)
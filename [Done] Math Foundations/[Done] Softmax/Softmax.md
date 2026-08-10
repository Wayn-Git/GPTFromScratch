# Softmax

Softmax is a mathematical function that converts the **raw scores (logits)** produced by a neural network into a **probability distribution**.

For example, suppose an LLM predicts the next word after:

> **"I love"**

The model might produce these logits:

```text
Pizza : 5.2
Dogs  : 3.1
Cars  : 1.4
```

These numbers are **not probabilities**. They are simply scores indicating how much the model prefers each possible token. They can be positive, negative, or any real number, and they **do not sum to 1**.

Softmax transforms these logits into probabilities such as:

```text
Pizza : 84%
Dogs  : 13%
Cars  : 3%
```

Now the outputs:

- Are all between **0 and 1**
- Sum to **1**
- Can be interpreted as probabilities

---

## How does Softmax work?

### Step 1: Subtract the largest logit

Subtract the largest logit from every logit.

This is a **numerical stability trick** that prevents overflow when computing exponentials. It does **not** change the final probabilities because every logit is shifted by the same amount.

Example:

```text
Original logits:
[2, 5, 1]

Maximum logit:
5

Shifted logits:
[-3, 0, -4]
```

---

### Step 2: Exponentiate the shifted logits

Take the exponential of each shifted logit.

This makes every value **positive** and **amplifies the differences** between logits, meaning higher-scoring tokens receive disproportionately higher probabilities.

Example:

```text
e⁻³ ≈ 0.0498
e⁰  = 1
e⁻⁴ ≈ 0.0183
```

---

### Step 3: Normalize

Divide each exponentiated value by the **sum of all exponentiated values**.

This ensures that:

- Every probability lies between **0 and 1**
- All probabilities add up to **1**

For example:

```text
Exponentials:
[0.0498, 1, 0.0183]

Sum:
1.0681

Probabilities:
0.0498 / 1.0681
1      / 1.0681
0.0183 / 1.0681
```

---

## Summary

Softmax follows three simple steps:

1. Subtract the maximum logit.
2. Exponentiate the shifted logits.
3. Divide each exponentiated value by the sum of all exponentiated values.

The result is a valid probability distribution that a neural network or LLM can use to make predictions.

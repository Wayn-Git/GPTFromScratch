# Positional Encoding

## Core idea

Self-attention looks at words in parallel and learns the relationships between them, but it has no notion of order on its own. Without extra help it treats a sentence like a bag of words.

Sequential models like RNNs got order for free because they read one word at a time. Self-attention trades that away for parallelism, so order has to be added back explicitly.

Positional encoding is the part that adds order information back, so the model can tell which word came first and which came next.

## Example

An example:

Let's say we have these two sentences:
> cat sat
> sat cat

The words are identical but the meaning changes with order. Positional encoding gives each position its own signal, so "cat" at position 0 is represented differently from "cat" at position 1.

Another one:
> dog bit man
> man bit dog

Same bag of words, totally different story. Without position the model sees the same input. With position it sees the sequence.

## How positional encoding works:

Tokenization and word embedding happen first: the sentence is broken into tokens and mapped to vectors. Those vectors carry meaning but still carry no order information.

This mechanism adds a position signal to each embedding using sine and cosine waves at different frequencies:

1. **Position (pos):** Where the token sits in the sequence, e.g. 0, 1, 2.
2. **Dimension (i):** Which slot of the vector is being filled. Even slots use sine, odd slots use cosine.

$$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

### 1. Sine and cosine pattern

Low dimensions oscillate fast and high dimensions oscillate slowly, so every position gets a unique fingerprint. The wave pattern is predictable, which lets the model reason about relative distances and generalize to sequences longer than the ones seen in training.

Together these values form a single matrix `PE` with the same shape as the embedding matrix `X`, so the two can be combined directly:

$$X_{with\_position} = X + PE$$

### 2. Adding to embeddings

The encoding is added rather than concatenated. That keeps the model width unchanged and lets meaning and position mix in the same vector space:

$$X = Embedding(tokens) + PE$$

This `X` is exactly what self-attention consumes downstream:

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

Summary:

```text
pos         = token position in the sequence (0, 1, 2, ...)
2i/d_model  = frequency control, low dims fast, high dims slow
sin / cos   = converts position into a smooth wave signal
X + PE      = meaning + order, same shape, fed to attention
```

## Where this fits

Positional encoding sits between embedding and the first attention block. In your codebase that is `model/positional_encoding.py` producing `PE`, which is added to the output of `model/embeddings.py` before `model/attention.py` computes `Q`, `K`, `V`. No position signal, and "dog bit man" looks identical to "man bit dog" to every head.

## Resources used:

- https://arxiv.org/abs/1706.03762
- https://www.youtube.com/watch?v=vkhPtpUiLd8

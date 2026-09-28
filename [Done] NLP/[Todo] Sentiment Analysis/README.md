# Sentiment Analysis

## Core idea

Sentiment analysis reads a sentence and predicts the emotion behind it, typically on a negative-to-positive scale.

Word-counting baselines lose sentence context because they treat "not good" the same as "good". An embedding-based classifier instead learns which combinations of vectors signal positive versus negative sentiment.

The output here is a single score between 0 and 1, where values near 0 read as negative and values near 1 read as positive.

## Example

An example:

Let's say we have these sentences:
> I love this movie, it was amazing
> I hate this movie, it was boring

Spelling alone means nothing to the network. Each word is mapped to a vector, the vectors are pooled into one sentence vector, and a small head maps that to a score — high (e.g. ~0.9) for the first sentence, low (e.g. ~0.1) for the second.

## How sentiment analysis works:

Tokenization: the sentence is split into tokens and each token is mapped to an ID from the vocabulary.

Word Embedding: token IDs are looked up in a learned table of shape `(vocab_size, 16)`. The table is trained from scratch alongside the classifier, so vectors for similarly-sentimented words drift together during training.

This mechanism turns variable-length token lists into one fixed sentiment score in three steps:

1. **Embedding:** Each token becomes a 16-dimensional vector carrying learned meaning.
2. **Averaging:** The sequence is mean-pooled across time, so any length collapses to one `(16,)` sentence vector.
3. **Head:** A linear layer plus sigmoid squeezes that vector to a `(1,)` probability.

$$y = \sigma(W \cdot \mathrm{mean}(E[x]) + b)$$

This mirrors `foundations/sentiment.py`: `Embedding(vocab, 16) -> mean(dim=1) -> Linear(16, 1) -> Sigmoid`, rounded to 4 decimals.

### 1. Embedding lookup

$$E = Embedding(x)$$

An input batch of IDs with shape `(B, T)` becomes `(B, T, 16)`. Words seen in positive contexts and words seen in negative contexts separate in this space as training progresses.

### 2. Averaging across time

$$h = \mathrm{mean}(E, dim=1)$$

Mean-pooling squashes `(B, T, 16)` to `(B, 16)`. It deliberately discards order, which is a fair trade for short reviews: it handles variable length with zero extra parameters. The limitation is real too — "good, not bad" and "bad, not good" pool similarly, which is exactly the kind of order sensitivity that later motivates attention.

### 3. Linear + Sigmoid

$$logit = Wh + b$$

$$y = \frac{1}{1 + e^{-logit}}$$

Near 1 means positive, near 0 means negative, and 0.5 is the unsure middle.

Summary:

```text
E = Embedding(x)      # (B, T) -> (B, T, 16)
h = mean(E, dim=1)    # (B, T, 16) -> (B, 16)
y = sigmoid(W h + b)  # (B, 16) -> (B, 1), positive vs negative
```

## Where this fits

This is a standalone supervised classifier, not a GPT sub-layer. It reuses the same embedding idea as the GPT path (`data/vocab.py`, `model/embeddings.py`) but pools to one label per sentence instead of predicting the next token. Training uses binary cross-entropy: confident wrong answers are penalized the most, which pushes the embedding table to separate positive and negative words.

## Resources used:

- https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
- https://arxiv.org/abs/1706.03762

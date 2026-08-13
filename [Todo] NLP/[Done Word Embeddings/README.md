# Word Embeddings

Before we get into word embeddings, we need to understand what tokens are. A **token** is a small piece of text—such as a word, part of a word, or a single character—split from a larger document. The process of breaking down text into these pieces is called **tokenization**

## Types of Tokens

Depending on how a system needs to process text, tokens can vary in size and structure:

* **Word Tokens:** Splitting a sentence by spaces and punctuation so every individual word becomes a single token.
* **Subword Tokens:** Breaking rare or complex words into smaller morphological pieces (e.g., splitting "unhappily" into "un", "happi", and "ly")

---

## The Problem: Networks Only Read Numbers

The digit classifier that we built earlier works great because it uses numbers (pixel intensities) to represent shapes. But we humans do not use numbers to communicate; we use words

The main drawback of neural networks is that they *only* work with numbers. Specifically, they process math structures:

* **Scalars:** 0 dimensions (a single number)
* **Vectors:** 1 dimension (an array of numbers)
* **Matrices:** 2 dimensions (rows and columns)
* **Tensors:** 3 or more dimensions

---

## The Solution: Word Embeddings

This is where word embeddings come in. An embedding translates words into a language the network can understand by mapping each token to a dense vector (an array of numbers). The magic of embeddings is that **semantically similar words end up mathematically close together** in the vector space

### The Embedding Table

An embedding table is simply a giant lookup matrix with the shape `(vocab_size, embed_dim)`. To look up the embedding vector for a specific token ID, you just index straight into the matrix: `embedding[token_id]`


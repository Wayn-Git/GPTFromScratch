# What is Natural Language Processing (NLP)?

Natural Language Processing (NLP) is a barnch of artificial intelligence focused on enabling computers to understand and generate humnan language in a meaningful way. It combines computational linguistics and machine learning and deep learning to process text and speech for data

Some common uses of NLP is language translation, sentiment analysis (analazying the emotions), voice assistants, chatbots, spelling checking etc...

![Sentiment Analysis](image.png)

## How Natural Language Processing (NLP) Works

NLP models learn to understand language by finding relationships between letters, words, and sentences. However, before a model can actually read and learn from a text dataset, the data has to be cleaned up and translated into a format a computer can process.

This preparation phase is called **Data Preprocessing**. Here are the four main techniques used to get text ready for an NLP model:

* **Stemming and Lemmatization (Finding the Root):** This reduces variations of words down to their base form. For example, converting "universities" and "university's" down to the same root word so the model knows they mean the same thing.
* **Sentence Segmentation (Breaking it Down):** Splitting a massive block of text into individual, meaningful sentences. This can be tricky because the model has to learn whether a period means the end of a sentence or if it is just an abbreviation.
* **Stop Word Removal (Cutting the Fluff):** Deleting extremely common filler words like "the", "a", or "an". These words take up space but usually don't add much actual meaning or information to the text.
* **Tokenization (Translating to Numbers):** Chopping the text into individual words or fragments and assigning them numerical tokens. Deep learning models cannot read English words—they only read numbers—so this step is required for the model to actually process the data.
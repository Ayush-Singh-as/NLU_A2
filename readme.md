# CSL 7640: Natural Language Understanding - Assignment 2

This repository contains the source code for Assignment 2, which is divided into two main problems: learning custom Word2Vec embeddings from scraped university data, and generating character-level Indian names using various Recurrent Neural Network (RNN) architectures.

## Environment Setup

**Important Note on Python Version:** This code requires Python 3.11 or 3.12.

To install all required dependencies, run the following command in your terminal:

```
pip install requests beautifulsoup4 pdfplumber nltk matplotlib wordcloud scikit-learn gensim torch
```

## Problem 1: Learning Word Embeddings from IIT Jodhpur Data

This section builds a custom textual corpus by scraping official IIT Jodhpur webpages and trains custom Word2Vec embeddings (CBOW and Skip-gram) from scratch.

### How to Run (Execute in order)

1. **Data Scraping** Run `python task1_data_prep.py`

   * Output: Generates `iitj_corpus.txt`.

2. **NLP Preprocessing** (Note: Uncomment the nltk.download lines at the top of the script for the first run to fetch tokenizer data). Run `python task1_preprocess.py`

   * Output: Generates `iitj_wordcloud.png` and saves `cleaned_tokens.txt`.

3. **Model Training** Run `python task2_train.py`

   * Output: Creates a `/models` directory containing the trained `.model` files.

4. **Semantic Analysis** Run `python task3_analysis.py`

   * Output: Prints cosine similarity scores and analogy predictions to the terminal.

5. **Visualization** Run `python task4_plot.py`

   * Output: Generates a PCA scatter plot saved as `task4_clusters.png`.

## Problem 2: Character-Level Name Generation using RNNs

This section implements three sequence models from scratch (Vanilla RNN, Bidirectional LSTM, and an RNN with Attention) to learn the statistical patterns of Indian names and generate new ones.

### How to Run (Execute in order)

(Ensure your terminal is navigated into the Problem 2 directory, or update the file paths in the scripts accordingly).

1. **Generate Dataset** We use an LLM-derived dataset of 1000 unique Indian first names.

   * Input: Ensure `Training_Names.txt` is present in the directory.

2. **Train the Models** This script reads the text file, builds the character vocabulary, and trains all three architectures. Run `python train.py`

   * Output: Saves `BasicRNN_weights.pth`, `BiLSTM_weights.pth`, `RNNAttention_weights.pth`, and `char_mappings.pkl`.

3. **Evaluate and Generate** This script loads the trained weights, hallucinates 500 new names per model, and calculates quantitative metrics. Run `python evaluate.py`

   * Output: Prints Diversity %, Novelty %, and sample generated names to the terminal for your report.

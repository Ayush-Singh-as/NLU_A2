# CSL 7640: Natural Language Understanding - Assignment 2

This repository contains the source code for Assignment 2, which is divided into two main problems: learning custom Word2Vec embeddings from scraped university data, and generating character-level Indian names using various Recurrent Neural Network (RNN) architectures.

---

## Environment Setup

**Important Note on Python Version:** This code requires Python 3.11 or 3.12.

To install all required dependencies, run the following command in your terminal:

```
pip install requests beautifulsoup4 pdfplumber nltk matplotlib wordcloud scikit-learn gensim torch
```

---

# Problem 1: Learning Word Embeddings from IIT Jodhpur Data

This section builds a custom textual corpus by scraping official IIT Jodhpur webpages and trains custom Word2Vec embeddings (CBOW and Skip-gram) from scratch.

## How to Run (Execute in order from the root directory)

### 1. Data Scraping

Run:

```
python "Problem 1/scrap.py"
```

**Output:** Generates `iitj_corpus.txt` containing the raw text inside the Problem 1 folder.

### 2. NLP Preprocessing

(Note: Uncomment the nltk.download lines at the top of the script for the first run to fetch tokenizer data).

Run:

```
python "Problem 1/preprocess.py"
```

**Output:** Generates `iitj_wordcloud.png` and saves `cleaned_tokens.txt`.

### 3. Model Training

Run:

```
python "Problem 1/train.py"
```

**Output:** Creates a `/models` directory containing the trained `.model` files.

### 4. Semantic Analysis

Run:

```
python "Problem 1/analysis.py"
```

**Output:** Prints cosine similarity scores and analogy predictions to the terminal.

### 5. Visualization

Run:

```
python "Problem 1/plot.py"
```

**Output:** Generates a PCA scatter plot saved as `task4_clusters.png`.

---

# Problem 2: Character-Level Name Generation using RNNs

This section implements three sequence models from scratch (Vanilla RNN, Bidirectional LSTM, and an RNN with Attention) to learn the statistical patterns of Indian names and generate new ones.

## How to Run (Execute in order from the root directory)

### 1. Data Requirement

Ensure your dataset of 1000 unique Indian first names is saved as `Training_Names.txt` inside the Problem 2 folder.

### 2. Train the Models

This script reads the text file, builds the character vocabulary, imports the architectural blueprints from `models.py`, and trains all three networks.

Run:

```
python "Problem 2/train.py"
```

**Output:** Saves `BasicRNN_weights.pth`, `BiLSTM_weights.pth`, `RNNAttention_weights.pth`, and `char_mappings.pkl` in the Problem 2 folder.

### 3. Evaluate and Generate

This script loads the trained weights, hallucinates 500 new names per model, and calculates quantitative metrics.

Run:

```
python "Problem 2/evaluate.py"
```

**Output:** Prints Diversity %, Novelty %, and sample generated names directly to the terminal for reporting.

---

# Results & Outputs

## Problem 1: Corpus & Embeddings

**Word Cloud of IIT Jodhpur Corpus**

**PCA Clustering of Word Vectors:**
The 2D projection below demonstrates how the trained Word2Vec models successfully map related academic concepts (like 'btech', 'mtech', 'ug', 'pg') into localized semantic neighborhoods.

---

## Problem 2: Generated Indian Names

The sequence models were evaluated on generating 500 names based on the learned character distributions.

* **Vanilla RNN:** Captures basic character frequencies but occasionally struggles with long-term phonetic structure.
* **BiLSTM:** Shows strong phonetic realism but can exhibit lower diversity due to the bidirectional constraints applied to an autoregressive generation task.
* **RNN + Attention:** Generally achieves the best balance between realistic syllable structures and high novelty/diversity scores.

(Note: Actual Diversity and Novelty percentages, along with 5 sample names per model, are printed to the terminal upon running `evaluate.py`)
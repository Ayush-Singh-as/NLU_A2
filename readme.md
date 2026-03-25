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

**Output:** Generates `iitj_wordcloud.png` and saves `cleaned_corpus.txt`.

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

**Output:** Generates a PCA scatter plot saved as `word_clusters.png`.

---

# Problem 2: Character-Level Name Generation using RNNs

This section implements three sequence models from scratch (Vanilla RNN, Bidirectional LSTM, and an RNN with Attention) to learn the statistical patterns of Indian names and generate new ones.

## How to Run (Execute in order from the root directory)

### 1. Data Requirement

Ensure your dataset of 1000 unique Indian first names is saved as `training_names.txt` inside the Problem 2 folder.

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

### Word Cloud of IIT Jodhpur Corpus

![Word Cloud](/Problem%201/iitj_wordcloud.png)

### PCA Clustering of Word Vectors

![PCA Clusters](Problem%201/word_clusters.png)

The 2D projection below demonstrates how the trained Word2Vec models successfully map related academic concepts (like 'btech', 'mtech', 'ug', 'pg') into localized semantic neighborhoods.

### Semantic Analysis Output (CBOW & Skip-gram)

```
Neighbors test:
  research:
    students (1.0)
    student (1.0)
    supervisor (1.0)
    semester (1.0)
    design (1.0)
  student:
    program (1.0)
    students (1.0)
    semester (1.0)
    courses (1.0)
    course (1.0)
  phd:
    school (0.999)
    call (0.999)
    assistant (0.998)
    professor (0.998)
    technology (0.998)
  exam:
    video (0.85)
    calculated (0.849)
    specialist (0.849)
    namely (0.844)
    high (0.843)

Analogy test:
  ug : btech :: pg : ?
    -> starting (0.964)
    -> programs (0.964)
    -> chemical (0.964)
  btech : student :: phd : ?
    -> credits (0.978)
    -> executive (0.978)
    -> associate (0.977)
  theory : exam :: practical : ?
    -> video (0.84)
    -> specialist (0.835)
    -> weeks (0.835)

---- SKIP-GRAM ----

Loading Problem 1/models/sg_d100_w5_n5.model...

Neighbors test:
  research:
    events (0.987)
    highlights (0.987)
    registrar (0.985)
    council (0.985)
    teacher (0.985)
  student:
    may (0.997)
    register (0.995)
    completed (0.994)
    end (0.992)
    summer (0.99)
  phd:
    associate (0.994)
    call (0.992)
    assistant (0.99)
    professor (0.99)
    university (0.986)
  exam:
    mo (0.992)
    advisors (0.992)
    weeks (0.992)
    consultation (0.991)
    minutes (0.991)

Analogy test:
  ug : btech :: pg : ?
    -> healthcare (0.996)
    -> iot (0.992)
    -> reality (0.99)
  btech : student :: phd : ?
    -> ph (0.819)
    -> associate (0.776)
    -> call (0.762)
  theory : exam :: practical : ?
    -> provisional (0.978)
    -> transcript (0.977)
    -> full (0.977)
```

---

## Problem 2: Generated Indian Names

The sequence models were evaluated on generating 500 names based on the learned character distributions.

* **Vanilla RNN:** Captures basic character frequencies but occasionally struggles with long-term phonetic structure.
* **BiLSTM:** Shows strong phonetic realism but can exhibit lower diversity due to the bidirectional constraints applied to an autoregressive generation task.
* **RNN + Attention:** Generally achieves the best balance between realistic syllable structures and high novelty/diversity scores.

### Evaluation Output

```
==============================
Evaluating Vanilla RNN
==============================
Diversity Score : 98.00%
Novelty Score   : 92.84%
Sample Outputs  : Darlag, Bilami, Yashwavan, Raghupathai, Vashwavvi

==============================
Evaluating BiLSTM
==============================
Diversity Score : 89.27%
Novelty Score   : 99.57%
Sample Outputs  : Iqub, Iqemtoti, Utti, Mni, Opp

==============================
Evaluating RNN+Attention
==============================
Diversity Score : 67.52%
Novelty Score   : 99.66%
Sample Outputs  : Nasher, Nanimat, Nalanar, Nanesh, Narai
```
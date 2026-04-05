# NLP Text Classification — IMDB Movie Reviews

A machine learning project that applies NLP preprocessing techniques to classify IMDB movie reviews as **positive** or **negative**.

## Overview

This notebook covers the full NLP pipeline — from raw text cleaning to training and evaluating multiple ML models on 10,000 IMDB movie reviews.

## What's Covered

- Text preprocessing: tokenization, stop word removal, stemming, lemmatization
- Text vectorization: Bag of Words (CountVectorizer) and TF-IDF
- Classification models: Naive Bayes, Logistic Regression, Random Forest
- Evaluation: accuracy, precision, recall, F1 score, confusion matrix

## Dataset

**IMDB Dataset of 50K Movie Reviews** — available on [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

Download `IMDB Dataset.csv` and place it in the same folder as the notebook before running.

## How to Run

1. Open `NLP_Assignment.ipynb` in [Google Colab](https://colab.research.google.com/) or Jupyter Notebook
2. Upload `IMDB Dataset.csv` to the session
3. Run all cells from top to bottom

## Libraries Used

- `nltk` — tokenization, stop words, stemming, lemmatization
- `scikit-learn` — TF-IDF, CountVectorizer, ML models, metrics
- `pandas`, `numpy` — data handling
- `matplotlib`, `seaborn` — visualization

## Results

| Model | Accuracy |
|---|---|
| Naive Bayes | ~85% |
| Logistic Regression | ~88% |
| Random Forest | ~84% |

*(Results may vary slightly depending on the random seed)*

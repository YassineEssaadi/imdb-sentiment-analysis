# 🎬 IMDB Movie Reviews -- Sentiment Analysis
### A Comparative Study of 5 Machine Learning Classifiers

[![Python](https://img.shields.io/badge/Language-Python%203.x-3776AB?style=flat-square&logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/Library-scikit--learn-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![NLP](https://img.shields.io/badge/Domain-NLP%20%7C%20Text%20Classification-green?style=flat-square)](https://en.wikipedia.org/wiki/Natural_language_processing)
[![Dataset](https://img.shields.io/badge/Dataset-IMDB%2050K%20Reviews-yellow?style=flat-square)](https://www.kaggle.com/datasets/vishakhdapat/imdb-movie-reviews)
[![Academic](https://img.shields.io/badge/Institution-Tunis%20Business%20School-red?style=flat-square)](https://www.tbs.u-tunis.tn/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

> **Course:** Business Analytics -- Data Mining | **Institution:** Tunis Business School | **Year:** 2025--2026

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Models](#models)
- [Results](#results)
- [How to Run](#how-to-run)
- [Repository Contents](#repository-contents)
- [Authors](#authors)

---

## 📌 Overview

This project implements a **complete end-to-end sentiment analysis pipeline** on the IMDB Movie Reviews dataset -- a benchmark corpus of **50,000 balanced movie reviews** labelled as positive or negative. Five classical machine learning classifiers are trained, evaluated, and compared under identical experimental conditions using **TF-IDF text representations**.

The goal is to build an accurate, automated binary sentiment classification system that can process raw text reviews without human intervention, while providing a rigorous comparative study of classical ML approaches.

---

## 📊 Dataset

| Feature | Details |
|---|---|
| **Source** | IMDB Movie Reviews Dataset -- [Kaggle](https://www.kaggle.com/datasets/vishakhdapat/imdb-movie-reviews) |
| **Size** | 50,000 reviews |
| **Class Balance** | Perfectly balanced: 25,000 positive / 25,000 negative |
| **Features** | `review` (raw text) + `sentiment` (binary label) |
| **Train / Test Split** | 40,000 (80%) / 10,000 (20%) -- stratified, `random_state=42` |

### Exploratory Data Analysis

- **Class distribution:** Perfectly balanced (50% / 50%) -- no oversampling required
- **Review length:** Positive reviews are marginally longer on average
- **Top positive words:** *great, film, good, best, love*
- **Top negative words:** *bad, worst, waste, boring, awful*

---

## ⚙️ Pipeline

```
Raw Text Reviews
      |
      v
TEXT PREPROCESSING
  1. Lowercasing
  2. Remove URLs, HTML tags, @mentions, #hashtags
  3. Remove punctuation & digits
  4. Stopword removal (NLTK English stopwords)
  5. Lemmatization (WordNet lemmatizer)
      |
      v
FEATURE EXTRACTION
  Primary:  TF-IDF Vectorizer
            - max_features = 5,000
            - ngram_range  = (1, 2)    ← unigrams + bigrams
            - min_df       = 5
            - sublinear_tf = True
  Baseline: Count Vectorizer (same settings, for comparison)
      |
      v
MODEL TRAINING & TUNING
  5 classifiers trained on TF-IDF features
  + GridSearchCV (2-fold stratified CV) on NB, KNN, RF
  + 5-fold Stratified Cross-Validation for all models
      |
      v
EVALUATION
  Accuracy, Precision, Recall, F1-Score
  ROC-AUC, Sensitivity (Recall), Specificity
  + Learning curves for top-2 models
  + Vectorizer comparison (TF-IDF vs. Count)
  + Feature importance (Random Forest & Decision Tree)
```

---

## 🤖 Models

| Model | Configuration | Notes |
|---|---|---|
| **Naive Bayes** | `MultinomialNB(alpha=1.0)` | Fast probabilistic baseline; near-zero training time |
| **K-Nearest Neighbors** | `KNeighborsClassifier(n_neighbors=5, metric='cosine')` | Cosine distance preferred over Euclidean for TF-IDF |
| **Decision Tree** | `DecisionTreeClassifier(max_depth=20, min_samples_split=10)` | Controlled depth to limit overfitting on sparse features |
| **Random Forest** | `RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1)` | Ensemble method; parallel training |
| **ANN (MLP)** | `MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, early_stopping=True)` | Best performer; 2 hidden layers with early stopping |

### Hyperparameter Tuning (GridSearchCV)

| Model | Search Space |
|---|---|
| Naive Bayes | `alpha` ∈ {0.1, 0.5, 1.0, 2.0} |
| KNN | `n_neighbors` ∈ {5, 7}, `metric` = cosine |
| Random Forest | `n_estimators` ∈ {50, 100}, `max_depth` ∈ {15, 20, None} |

---

## 📈 Results

| Model | **F1-Score** | **ROC-AUC** | Sensitivity | Specificity |
|---|---|---|---|---|
| **ANN (MLP)** | **0.8934** | **0.9583** | **0.9230** | **0.8568** |
| Naive Bayes | 0.8623 | 0.9341 | 0.8740 | 0.8468 |
| Random Forest | 0.8414 | 0.9175 | 0.8834 | 0.7836 |
| KNN | 0.8011 | 0.8631 | 0.8536 | 0.7226 |
| Decision Tree | 0.7678 | 0.7696 | 0.8480 | 0.6392 |

> **Note:** All metrics are extracted from the actual experimental results reported in the [project report](report/Datamining_Project_Report.pdf) (Table 5.1). Full metrics including Accuracy, Precision, Recall, and cross-validation scores are available in the report.

- **Best model:** ANN (MLP) -- F1-Score = **0.8934**, ROC-AUC = **0.9583**
- **Best baseline:** Naive Bayes -- F1 = 0.8623 at near-zero training time
- **Weakest:** Decision Tree -- F1 = 0.7678 (overfitting on sparse TF-IDF features)

---

## 🚀 How to Run

### Prerequisites

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud joblib
```

Download NLTK resources (handled automatically by the script):

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### Dataset

Download the IMDB dataset from Kaggle:
👉 [https://www.kaggle.com/datasets/vishakhdapat/imdb-movie-reviews](https://www.kaggle.com/datasets/vishakhdapat/imdb-movie-reviews)

Place the file as `IMDB Dataset.csv` in the same directory as the script.

### Running the Script

The script was originally developed as a **Google Colab notebook** and exported to Python. To run it:

**Option A — Google Colab (recommended):**
1. Upload `src/imdb_sentiment_analysis.py` to Colab or open the original notebook
2. Upload `IMDB Dataset.csv` to the Colab session or mount Google Drive
3. Run all cells sequentially

**Option B — Local Python:**
```bash
# Clone the repo
git clone https://github.com/YassineEssaadi/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis

# Place IMDB Dataset.csv in the root directory
# Then run:
python src/imdb_sentiment_analysis.py
```

> **Note:** Remove or comment out the `from google.colab import files` block at the end of the script when running locally.

### Outputs

The script produces the following files:
- `model_comparison_results.csv` -- full metrics table for all models
- `test_predictions.csv` -- per-sample predictions from the best model
- `best_model.pkl` -- serialized best model (ANN/MLP)
- `tfidf_vectorizer.pkl` -- fitted TF-IDF vectorizer
- `count_vectorizer.pkl` -- fitted Count vectorizer

---

## 📁 Repository Contents

```
imdb-sentiment-analysis/
|
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- .gitignore
|-- src/
|   |-- imdb_sentiment_analysis.py    # Full pipeline: preprocessing, training, evaluation
|-- report/
    |-- Datamining_Project_Report.pdf  # Full academic report (PDF)
```

> **Note:** This repository contains the academic report PDF and the full Python source code. The dataset (`IMDB Dataset.csv`) is not included due to size — download it from [Kaggle](https://www.kaggle.com/datasets/vishakhdapat/imdb-movie-reviews).

---

## 👥 Authors

| Name | Contribution |
|---|---|
| **Yassine Essaadi** | EDA, model training, evaluation, report writing |
| **Sarah Mneja** | Text preprocessing, feature extraction, report writing |
| **Bahe Amri** | Hyperparameter tuning, visualization, report writing |
| **Sirine Zaltni** | Literature review, cross-validation, report writing |

**Institution:** Tunis Business School
**Course:** Business Analytics -- Data Mining
**Supervisor:** Professor Lassaad El Moubarki
**Academic Year:** 2025--2026

---

## 🔮 Future Work

- [ ] Fine-tune **BERT / DistilBERT** for accuracy > 95%
- [ ] Extend to **multi-class sentiment** (1--5 star ratings)
- [ ] Test **cross-domain generalization** (Amazon, Yelp reviews)
- [ ] Add **LIME / SHAP** explainability
- [ ] Deploy as **REST API** (Flask / FastAPI)

---

*This project was conducted as part of the Business Analytics -- Data Mining course at Tunis Business School (2025--2026).*

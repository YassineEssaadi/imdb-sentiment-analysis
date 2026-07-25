# 🎬 IMDB Movie Reviews — Sentiment Analysis
### A Comparative Study of 5 Machine Learning Classifiers

[![Python](https://img.shields.io/badge/Language-Python%203.x-3776AB?style=flat-square&logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/Library-scikit--learn-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![NLP](https://img.shields.io/badge/Domain-NLP%20%7C%20Text%20Classification-green?style=flat-square)](https://en.wikipedia.org/wiki/Natural_language_processing)
[![Dataset](https://img.shields.io/badge/Dataset-IMDB%2050K%20Reviews-yellow?style=flat-square)](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
[![Academic](https://img.shields.io/badge/Institution-Tunis%20Business%20School-red?style=flat-square)](https://www.tbs.u-tunis.tn/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

> **Course:** Business Analytics – Data Mining | **Institution:** Tunis Business School | **Year:** 2025–2026  
> **Supervisor:** Professor Lassaad El Moubarki

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Models](#models)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Authors](#authors)

---

## 📌 Overview

This project implements a **complete end-to-end sentiment analysis pipeline** on the IMDB Movie Reviews dataset — a benchmark corpus of **50,000 balanced movie reviews** labelled as positive or negative. Five classical machine learning classifiers are trained, evaluated, and compared under identical experimental conditions using **TF-IDF text representations**.

The goal is to build an accurate, automated binary sentiment classification system that can process raw text reviews without human intervention, while providing a rigorous comparative study of classical ML approaches.

---

## 📊 Dataset

| Feature | Details |
|---|---|
| **Source** | IMDB Dataset (Maas et al., 2011) — [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) |
| **Size** | 50,000 reviews |
| **Class Balance** | Perfectly balanced: 25,000 positive / 25,000 negative |
| **Features** | `review` (raw text) + `sentiment` (binary label) |
| **Train / Test Split** | 40,000 (80%) / 10,000 (20%) — stratified |

### Exploratory Data Analysis

- **Class distribution:** Perfectly balanced (50% / 50%) — no oversampling required
- **Review length:** Positive reviews are marginally longer on average
- **Top positive words:** *great, film, good, best, love*
- **Top negative words:** *bad, worst, waste, boring, awful*

---

## ❉ Pipeline

```
Raw Text Reviews
      |
      v
┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐└
|          TEXT PREPROCESSING            |
|  1. Lowercasing                        |
|  2. Remove URLs, HTML tags, mentions    |
|  3. Remove punctuation & digits         |
|  4. Stopword removal (NLTK)             |
|  5. Lemmatization (WordNet)             |
└┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┘
      |
      v
┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐└
|         FEATURE EXTRACTION              |
|  TF-IDF Vectorizer                      |
|  • max_features = 5,000                |
|  • ngram_range = (1, 2)                |
|  • min_df = 5                          |
|  • sublinear_tf = True                 |
└┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┘
      |
      v
┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐└
|         MODEL TRAINING & TUNING         |
|  5 classifiers + GridSearchCV           |
└┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┘
      |
      v
┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐└
|         EVALUATION                       |
|  Accuracy, Precision, Recall, F1,       |
|  Sensitivity, Specificity, ROC-AUC      |
|  + 5-fold Cross-Validation              |
└┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┐┘
```

---

## 🤖 Models

| Model | Configuration | Notes |
|---|---|---|
| **Naive Bayes** | Multinomial, α = 1.0 (Laplace smoothing) | Fast probabilistic baseline |
| **K-Nearest Neighbors** | k = 5, cosine distance | Cosine preferred over Euclidean for TF-IDF |
| **Decision Tree** | max_depth = 20, min_samples_split = 10 | Controlled overfitting |
| **Random Forest** | 100 trees, max_depth = 20, n_jobs = -1 | Ensemble method |
| **ANN (MLP)** | 2 hidden layers (100, 50 neurons), early stopping | Best performer |

---

## 📈 Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | **F1-Score** | ROC-AUC | Sensitivity | Specificity |
|---|---|---|---|---|---|---|---|
| **ANN (MLP)** ⛐ | — | — | — | **0.8934** | **0.9583** | **0.9230** | **0.8568** |
| Naive Bayes | — | — | — | 0.8623 | — | — | — |
| Random Forest | — | — | — | 0.8414 | — | — | — |
| KNN | — | — | — | 0.8011 | — | — | — |
| Decision Tree | — | — | — | 0.7678 | — | — | — |

### Key Findings

- 🇗 **Best model:** ANN (MLP) — F1-Score = **0.8934**, ROC-AUC = **0.9583**
- ⚡ **Best baseline:** Naive Bayes — F1 = 0.8623 at near-zero training time
- ❌ **Weakest:** Decision Tree — F1 = 0.7678 (overfitting on sparse features)
- ✅ **TF-IDF consistently outperforms Count Vectorizer** across all models
- ✅ **5-fold cross-validation** results closely aligned with test set performance (no significant overfitting)

### Vectorizer Comparison

| Vectorizer | Best Model F1 | Notes |
|---|---|---|
| **TF-IDF** | **0.8934** | Down-weights frequent terms → better discrimination |
| Count Vectorizer | Lower | Raw counts less discriminative |

---

## 📁 Repository Structure

```
imdb-sentiment-analysis/
│
├── 📄 README.md
├── 📄 report/
│   └── datamining_report.pdf            # Full academic report
│
├── 📊 data/
│   ├── IMDB_Dataset.csv                 # Raw dataset (from Kaggle)
│   └── processed/
│       ├── cleaned_reviews.csv          # After preprocessing
│       └── tfidf_features.npz            # Sparse TF-IDF matrix
│
├── 📝 notebooks/
│   ├── 01_eda.ipynb                   # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb           # Text cleaning pipeline
│   ├── 03_feature_extraction.ipynb      # TF-IDF vectorization
│   ├── 04_model_training.ipynb          # All 5 classifiers
│   ├── 05_evaluation.ipynb              # Metrics & confusion matrices
│   └── 06_hyperparameter_tuning.ipynb   # GridSearchCV
│
├── 📝 src/
│   ├── preprocessing.py                 # Text cleaning functions
│   ├── feature_extraction.py            # TF-IDF vectorizer
│   ├── models.py                        # Model definitions
│   ├── evaluation.py                    # Metrics computation
│   └── train.py                         # Main training script

---

## ▶ How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

### Download NLTK Resources

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### Run the Full Pipeline

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis

# Run training
python src/train.py
```

### Quick Inference

```python
import joblib
from src.preprocessing import preprocess_text

# Load saved model
model = joblib.load('models/best_model.pkl')

# Predict sentiment
review = "This movie was absolutely fantastic! Great acting and storyline."
cleaned = preprocess_text(review)
prediction = model.predict([cleaned])
print("Sentiment:", "Positive" if prediction[0] == 1 else "Negative")
```

---

## 👥 Authors

| Name | Contribution |
|---|---|
| **Yassine Saadi** | EDA, model training, evaluation, report writing |
| **Sarah Mneja** | Text preprocessing, feature extraction, report writing |
| **Bahe Amri** | Hyperparameter tuning, visualization, report writing |
| **Sirine Zaltni** | Literature review, cross-validation, report writing |

**Institution:** Tunis Business School  
**Course:** Business Analytics – Data Mining  
**Supervisor:** Professor Lassaad El Moubarki  
**Academic Year:** 2025–2026

---

## 🔮 Future Work

- [ ] Fine-tune **BERT / DistilBERT** for accuracy > 95%
- [ ] Extend to **multi-class sentiment** (1–5 star ratings)
- [ ] Test **cross-domain generalization** (Amazon, Yelp reviews)
- [ ] Add **LIME / SHAP** explainability
- [ ] Deploy as **REST API** (Flask / FastAPI)
- [ ] Apply **data augmentation** (back-translation, synonym replacement)

---

## 🔚 References

- Maas, A.L. et al. (2011). Learning Word Vectors for Sentiment Analysis. *ACL 2011*.
- Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers. *NAACL-HLT 2019*.
- Yang, Z. et al. (2019). XLNet: Generalized Autoregressive Pretraining. *NeurIPS 2019*.
- Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. *EMNLP 2014*.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.

---

*This project was conducted as part of the Business Analytics – Data Mining course at Tunis Business School (2025–2026).*

# MindLens-AI

Explainable mental health risk detection from social media text. Built around a Logistic Regression classifier on the Reddit Mental Health Dataset (RMHD), with SHAP/LIME explanations, bias auditing, robustness testing, and a composite trust score.

## What This Project Does

Given a piece of text (e.g. a Reddit post), the model predicts whether it signals mental-health risk. Binary labels are constructed from subreddit membership: posts from r/depression and r/SuicideWatch are labelled **Risk (1)**, while posts from r/Anxiety, r/lonely, and r/mentalhealth are labelled **Control (0)**. Beyond making predictions, the project investigates three research questions:

1. **Can the model's decisions be meaningfully interpreted?** We use SHAP and LIME to extract the features driving each prediction, and check whether clinically relevant terms (e.g. "hopeless", "insomnia", "therapy") actually appear among the top contributors.

2. **Does the model treat different user groups fairly?** We split users by engagement level and sentiment polarity, then compare false positive rates across groups. A chi-square test checks whether any observed differences are statistically significant.

3. **Is the model robust to minor text changes?** We apply two perturbation strategies — removing known trigger keywords, and replacing random words with synonyms — and measure how often the prediction flips.

## Key Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **76.83%** |
| Precision | **76.17%** |
| Recall | **78.12%** |
| F1 Score | **77.13%** |
| ROC-AUC | **85.15%** |
| FPR Gap (engagement/sentiment groups) | **25.26%** |
| Keyword-removal flip rate | **3.67%** |
| Synonym-replacement flip rate | **5.00%** |

### Confusion Matrix (Test Set — 3,992 samples)

|  | Predicted Control | Predicted Risk |
|--|:-:|:-:|
| **Actual Control** | 1,507 (TN) | 488 (FP) |
| **Actual Risk** | 437 (FN) | 1,560 (TP) |

## Architecture

```
Raw Reddit CSVs (by year/month)
        │
        ▼
  Subreddit-based labelling (Risk vs Control)
        │
        ▼
  Balanced sampling (10K per class = 20K total)
        │
        ▼
  Text cleaning + Feature engineering
        │
        ▼
  TF-IDF vectorization (5,000 features)
        │
        ├──► Logistic Regression (primary)
        │         │
        │         ▼
        │    Train/Test evaluation
        │         │
        │         ├──► SHAP / LIME Explainability
        │         ├──► Bias Auditing (FPR by group)
        │         ├──► Robustness Testing (perturbations)
        │         │
        │         ▼
        │    Composite Trust Score
        │
        └──► DistilBERT (secondary/comparison)
                  │
                  ▼
        Streamlit Web Application (demo)
```

See [Architecture_Diagram.md](Architecture_Diagram.md) for the full Mermaid diagram and component descriptions.

## Project Layout

```
notebooks/
    01_eda.ipynb              Exploratory data analysis
    02_model_training.ipynb   TF-IDF + Logistic Regression training
    03_explainability.ipynb   SHAP and LIME explanations
    04_bias_testing.ipynb     Fairness audit
    05_robustness.ipynb       Perturbation tests + trust scores
src/
    preprocessing.py          Text cleaning and feature engineering
    features.py               TF-IDF vectorization
    model.py                  Training and persistence
    evaluation.py             Metrics and error analysis
    explainability.py         SHAP/LIME wrappers, interpretability scoring
    bias.py                   Group-level fairness metrics
    robustness.py             Perturbation functions
    trust.py                  Composite trust score
app/
    streamlit_app.py          Interactive demo
data/
    raw/                      RMHD raw data (subreddit CSVs by year/month)
    processed/                Cleaned data and saved model artifacts
```

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords wordnet
```

## Running the Notebooks

Open them in order (01 through 05). Each notebook builds on artifacts saved by the previous one — the EDA notebook cleans and saves the data, the training notebook saves the model, and so on.

## Running the Demo App

```bash
streamlit run app/streamlit_app.py
```

Paste in any text and the app will show the prediction, a LIME explanation, perturbation comparison, and a trust score.

## Dataset

Reddit Mental Health Dataset (RMHD) — ~1.85 million Reddit posts (2019–2022) from five mental-health subreddits: r/depression, r/SuicideWatch, r/Anxiety, r/lonely, r/mentalhealth. Binary labels are constructed via subreddit-based proxy labelling (Risk vs. No Risk). A balanced subset of 20,000 posts (10K per class) is sampled for training (15,964 train / 3,992 test after cleaning).

**Limitation:** Subreddit-based labeling may introduce contextual bias, which is evaluated through bias and robustness analysis in later stages.

See notebook 01 for full exploration.

## Trustworthy AI Analysis

| Dimension | Method | Finding |
|-----------|--------|---------|
| **Explainability** | SHAP (LinearExplainer) + LIME | Top features align with clinical mental health vocabulary (~100 terms) |
| **Fairness** | FPR comparison across engagement & sentiment groups; chi-square test | 25.26% FPR gap observed — a limitation of proxy labelling |
| **Robustness** | Keyword removal (30 trigger words) + synonym replacement (WordNet) | 3.67% and 5.00% flip rates — both below the 10% acceptability threshold |
| **Trust Score** | `confidence − 0.5·flip_rate − 0.5·bias_gap` | Categorised as High (≥0.7), Medium (≥0.4), or Low |

See [Results_Summary.md](Results_Summary.md) for the complete results report.

## Dependencies

See [requirements.txt](requirements.txt). Main libraries: scikit-learn, Hugging Face Transformers, SHAP, LIME, NLTK, Streamlit, matplotlib, seaborn.

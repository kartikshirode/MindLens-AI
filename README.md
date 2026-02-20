# MindLens-AI

Explainable mental health risk detection from social media text. Built around a logistic regression classifier on Reddit data, with SHAP/LIME explanations, bias auditing, and robustness testing.

## What This Project Does

Given a piece of text (e.g. a Reddit post), the model predicts whether it signals depression risk. Beyond making predictions, the project investigates three research questions:

1. **Can the model's decisions be meaningfully interpreted?** We use SHAP and LIME to extract the features driving each prediction, and check whether clinically relevant terms (e.g. "hopeless", "insomnia", "therapy") actually appear among the top contributors.

2. **Does the model treat different user groups fairly?** We split users by engagement level and sentiment polarity, then compare false positive rates across groups. A chi-square test checks whether any observed differences are statistically significant.

3. **Is the model robust to minor text changes?** We apply two perturbation strategies -- removing known trigger keywords, and replacing random words with synonyms -- and measure how often the prediction flips.

## Key Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 96.1% |
| F1 Score | 95.9% |
| FPR Gap (engagement groups) | 0.003 |
| Keyword-removal flip rate | 0.3% |
| Synonym-replacement flip rate | 4.0% |

The model is accurate, shows no meaningful bias between groups (p = 1.0 on chi-square), and is largely stable under text perturbation.

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
    raw/                      Original dataset
    processed/                Cleaned data and saved model
```

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords wordnet
```

## Running the Notebooks

Open them in order (01 through 05). Each notebook builds on artifacts saved by the previous one -- the EDA notebook cleans and saves the data, the training notebook saves the model, and so on.

## Running the Demo App

```bash
streamlit run app/streamlit_app.py
```

Paste in any text and the app will show the prediction, a LIME explanation, perturbation comparison, and a trust score.

## Dataset

Reddit depression dataset (cleaned), sourced from public NLP benchmarks. Binary labels: depression-indicative vs. non-depression. See notebook 01 for full exploration.

## Dependencies

See [requirements.txt](requirements.txt). Main libraries: scikit-learn, SHAP, LIME, NLTK, Streamlit, matplotlib, seaborn.

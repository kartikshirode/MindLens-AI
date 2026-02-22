# MindLens-AI

Explainable mental health risk detection from social media text. Built around a logistic regression classifier on the Reddit Mental Health Dataset (RMHD), with SHAP/LIME explanations, bias auditing, and robustness testing.

## What This Project Does

Given a piece of text (e.g. a Reddit post), the model predicts whether it signals mental-health risk. Binary labels are constructed from subreddit membership: posts from r/depression and r/SuicideWatch are labelled Risk (1), while posts from r/Anxiety, r/lonely, and r/mentalhealth are labelled Control (0). Beyond making predictions, the project investigates three research questions:

1. **Can the model's decisions be meaningfully interpreted?** We use SHAP and LIME to extract the features driving each prediction, and check whether clinically relevant terms (e.g. "hopeless", "insomnia", "therapy") actually appear among the top contributors.

2. **Does the model treat different user groups fairly?** We split users by engagement level and sentiment polarity, then compare false positive rates across groups. A chi-square test checks whether any observed differences are statistically significant.

3. **Is the model robust to minor text changes?** We apply two perturbation strategies -- removing known trigger keywords, and replacing random words with synonyms -- and measure how often the prediction flips.

## Key Results

| Metric | Value |
|--------|-------|
| Test Accuracy | TBD (re-run notebooks) |
| F1 Score | TBD |
| FPR Gap (engagement groups) | TBD |
| Keyword-removal flip rate | TBD |
| Synonym-replacement flip rate | TBD |

*Results will be populated after running notebooks 01–05 on the RMHD dataset.*

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

Reddit Mental Health Dataset (RMHD) — ~1.85 million Reddit posts (2019–2022) from five mental-health subreddits: r/depression, r/SuicideWatch, r/Anxiety, r/lonely, r/mentalhealth. Binary labels are constructed via subreddit-based proxy labelling (Risk vs. No Risk). A balanced subset of 20,000 posts (10K per class) is sampled for training.

**Limitation:** Subreddit-based labeling may introduce contextual bias, which is evaluated through bias and robustness analysis in later stages.

See notebook 01 for full exploration.

## Dependencies

See [requirements.txt](requirements.txt). Main libraries: scikit-learn, SHAP, LIME, NLTK, Streamlit, matplotlib, seaborn.

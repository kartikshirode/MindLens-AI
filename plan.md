# MindLens AI: Explainable Mental Health Risk Detection

---

## 1. Project Overview

This project focuses on building a **trustworthy AI system** that can detect mental health risks from social media text while ensuring:
- High prediction performance
- Clear interpretability (why the model made a decision)
- Fairness across different user behaviors
- Robustness to small input changes

Unlike typical ML projects that focus only on accuracy, this project evaluates **Explainability, Bias, and Robustness**.

---

## 2. Core Problem

Most AI systems for mental health detection:
- Act as black boxes
- Provide predictions without explanations
- May be biased toward certain user behaviors
- Can be unstable with small text changes

This makes them unreliable for real-world usage.

---

## 3. Research Questions (RQs)

### RQ1: Explainability vs Performance
Can explainable NLP models maintain high performance (≥80% accuracy) while providing meaningful, human-interpretable explanations aligned with mental health indicators?

### RQ2: Bias in User Behavior
Do high-engagement users (frequent or emotionally intense posts) experience higher false positive rates compared to low-engagement users?

### RQ3: Model Robustness
Are model predictions sensitive to small input perturbations (e.g., keyword removal or paraphrasing), indicating reliance on spurious correlations?

---

## 4. Hypotheses

- H1: Explainable models can achieve ≥80% accuracy without major performance loss.
- H2: High-engagement users will show higher false positive rates.
- H3: Small perturbations in input text will significantly affect predictions.

---

## 5. System Pipeline

Data → Preprocessing → Model → Explainability → Bias Testing → Robustness Testing → Output

---

## 6. Stages vs Research Questions

### Stages (Execution Flow)

#### Stage 1: Research Proposal
- Define RQs, hypothesis, dataset, and methodology

#### Stage 2: EDA & Findings
- Analyze dataset
- Identify patterns and feasibility

#### Stage 3: Final Implementation
- Train final model
- Apply explainability
- Evaluate RQs

### Research Questions (What You Answer)
- RQ1 → Performance + Explainability
- RQ2 → Bias
- RQ3 → Robustness

---

## 7. Dataset (Best Choice)

### Primary Dataset: Reddit Mental Health Dataset (Recommended)

**Name:** Reddit Depression / SuicideWatch Dataset

**Sources:**
- https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned
- https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch

### Why this dataset?
- Real user-generated text
- Contains labeled mental health signals
- Suitable for NLP classification
- Allows grouping by user behavior

### Backup Dataset:
- Twitter Mental Health Dataset (Kaggle)

---

## 8. Methodology

### Step 1: Data Preprocessing
- Clean text (remove links, symbols)
- Tokenization
- Lowercasing
- Stopword removal (optional)

### Step 2: Feature Engineering
- TF-IDF vectors
- OR use pre-trained embeddings (BERT)

### Step 3: Model Training

Baseline:
- Logistic Regression

Advanced:
- DistilBERT (if time permits)

### Step 4: Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score

---

## 9. Explainability (RQ1)

Tools:
- SHAP
- LIME

Output:
- Word importance scores
- Highlighted influential words

---

## 10. Bias Testing (RQ2)

Approach:
- Split users based on:
  - Post frequency
  - Text length
  - Sentiment intensity

Metric:
- False Positive Rate (FPR)

Compare:
- High-engagement vs Low-engagement users

---

## 11. Robustness Testing (RQ3)

Approach:
- Modify input text:
  - Remove key words
  - Replace with synonyms

Metric:
- Percentage of prediction change

---

## 12. Day-wise Execution Plan

### Day 1
- Finalize RQs
- Select dataset
- Build baseline model

### Day 2
- Perform EDA
- Improve model
- Add explainability

### Day 3
- Bias testing
- Robustness testing
- Final results + report

---

## 13. Optional Enhancement (Portfolio Boost)

- Build Streamlit UI:
  - Input text
  - Show prediction
  - Highlight important words

- Deploy using:
  - Streamlit Cloud
  - Render

---

## 14. Final Project Positioning

This project develops a transparent and trustworthy AI system for mental health risk detection, focusing on interpretability, fairness, and robustness rather than just prediction accuracy.

---

## 15. Key Advantage

Most projects: Focus only on accuracy

This project: Focuses on
- Explainability
- Bias detection
- Robustness

---

## 16. Tools & Tech Stack

- Python
- Scikit-learn
- Transformers (HuggingFace)
- SHAP / LIME
- Pandas, NumPy
- Matplotlib / Seaborn

---

## 17. Final Goal

Build a system that not only predicts mental health risk but also explains, validates, and tests its own decisions.

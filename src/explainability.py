"""
explainability.py - SHAP, LIME wrappers and quantitative interpretability score.
"""

import numpy as np
import shap
from lime.lime_text import LimeTextExplainer


# ---------------------------------------------------------------------------
# Mental health lexicon (starter set)
# ---------------------------------------------------------------------------

MH_LEXICON = {
    "sad", "hopeless", "alone", "suicide", "tired", "worthless",
    "depressed", "anxious", "empty", "pain", "die", "help",
    "crying", "lost", "hate", "suffer", "miserable", "numb",
    "angry", "scared", "hurt", "broken", "overwhelmed", "desperate",
    "lonely", "useless", "failure", "guilt", "ashamed", "isolat",
}


# ---------------------------------------------------------------------------
# LIME
# ---------------------------------------------------------------------------

def explain_lime(model, vectorizer, text: str, num_features: int = 10, class_names=None):
    """
    Generate a LIME explanation for a single text.

    Parameters
    ----------
    model : sklearn estimator with predict_proba
    vectorizer : fitted TfidfVectorizer
    text : input text string
    num_features : number of top features to show

    Returns
    -------
    explanation : lime Explanation object
    """
    if class_names is None:
        class_names = ["No Risk", "Risk"]

    explainer = LimeTextExplainer(class_names=class_names, random_state=42)

    def predict_fn(texts):
        X = vectorizer.transform(texts)
        return model.predict_proba(X)

    explanation = explainer.explain_instance(
        text,
        predict_fn,
        num_features=num_features,
        num_samples=2000,
    )
    return explanation


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def explain_shap(model, X_train_tfidf, X_test_tfidf, feature_names):
    """
    Compute SHAP values using LinearExplainer (for Logistic Regression).

    Returns
    -------
    shap_values : np.ndarray  (n_test_samples, n_features)
    explainer   : shap.LinearExplainer
    """
    explainer = shap.LinearExplainer(model, X_train_tfidf, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_test_tfidf)
    return shap_values, explainer


def shap_summary_plot(shap_values, X_test_tfidf, feature_names, max_display: int = 20):
    """Display a SHAP summary (beeswarm) plot."""
    shap.summary_plot(
        shap_values,
        X_test_tfidf,
        feature_names=feature_names,
        max_display=max_display,
        show=True,
    )


def shap_force_plot(explainer, shap_values, idx: int, feature_names):
    """Generate a SHAP force plot for a single sample."""
    return shap.force_plot(
        explainer.expected_value,
        shap_values[idx],
        feature_names=feature_names,
        matplotlib=True,
    )


# ---------------------------------------------------------------------------
# Quantitative Interpretability Score
# ---------------------------------------------------------------------------

def compute_interpretability_score(
    shap_values: np.ndarray,
    feature_names: list[str],
    lexicon: set[str] | None = None,
    k: int = 10,
) -> dict:
    """
    Measure how many of the top-k SHAP features overlap with a mental health
    lexicon.

    Interpretability Score = (# top-k features overlapping lexicon) / k

    Parameters
    ----------
    shap_values : (n_samples, n_features) array
    feature_names : list of feature names (from TF-IDF vocabulary)
    lexicon : set of mental health keywords (defaults to MH_LEXICON)
    k : number of top features to check

    Returns
    -------
    dict with keys: per_sample_scores (array), mean_score, std_score
    """
    if lexicon is None:
        lexicon = MH_LEXICON

    feature_names = np.array(feature_names)
    per_sample_scores = []

    for i in range(shap_values.shape[0]):
        abs_vals = np.abs(shap_values[i])
        top_k_idx = np.argsort(abs_vals)[-k:]
        top_k_words = set(feature_names[top_k_idx])

        overlap = sum(
            1 for w in top_k_words
            if any(lex_word in w for lex_word in lexicon)
        )
        per_sample_scores.append(overlap / k)

    scores = np.array(per_sample_scores)
    return {
        "per_sample_scores": scores,
        "mean_score": float(scores.mean()),
        "std_score": float(scores.std()),
    }

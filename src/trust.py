"""
trust.py - Composite Trust Score computation.
"""

import numpy as np


def compute_trust_score(
    confidence: float,
    flip_rate: float,
    bias_fpr_gap: float,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> float:
    """
    Compute a composite Trust Score.

    Trust Score = confidence - α * flip_rate - β * bias_fpr_gap

    Clamped to [0, 1].
    """
    score = confidence - alpha * flip_rate - beta * bias_fpr_gap
    return float(np.clip(score, 0.0, 1.0))


def categorize_trust(score: float) -> str:
    """Map numeric trust score to a human-readable category."""
    if score >= 0.7:
        return "High"
    elif score >= 0.4:
        return "Medium"
    else:
        return "Low"


def generate_trust_report(
    text: str,
    model,
    vectorizer,
    bias_fpr_gap: float,
    perturbation_fn=None,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> dict:
    """
    End-to-end trust report for a single text.

    Returns
    -------
    dict with keys: prediction, confidence, flip_rate, trust_score, trust_label
    """
    from src.robustness import remove_keywords  # default perturbation

    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]
    confidence = float(prob[pred])

    # Perturbation flip check
    if perturbation_fn is None:
        perturbation_fn = remove_keywords
    perturbed = perturbation_fn(text)
    X_pert = vectorizer.transform([perturbed])
    pert_pred = model.predict(X_pert)[0]

    flip_rate = 1.0 if pert_pred != pred else 0.0

    score = compute_trust_score(confidence, flip_rate, bias_fpr_gap, alpha, beta)
    label = categorize_trust(score)

    return {
        "prediction": int(pred),
        "prediction_label": "Risk" if pred == 1 else "No Risk",
        "confidence": confidence,
        "flip_rate": flip_rate,
        "trust_score": score,
        "trust_label": label,
    }

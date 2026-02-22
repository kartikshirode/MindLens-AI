import numpy as np


def compute_trust_score(confidence, flip_rate, bias_gap, alpha=0.5, beta=0.5):
    """Composite trust metric: confidence penalised by instability and bias."""
    raw = confidence - alpha * flip_rate - beta * bias_gap
    return float(np.clip(raw, 0.0, 1.0))


def categorize_trust(score):
    """Bucket a numeric score into High / Medium / Low."""
    if score >= 0.7:
        return "High"
    if score >= 0.4:
        return "Medium"
    return "Low"


def generate_trust_report(text, classifier, vectorizer, bias_gap,
                          perturbation_fn=None, alpha=0.5, beta=0.5):
    """End-to-end trust assessment for a single piece of text."""
    from src.robustness import remove_keywords

    x = vectorizer.transform([text])
    pred = classifier.predict(x)[0]
    prob = classifier.predict_proba(x)[0]
    conf = float(prob[pred])

    if perturbation_fn is None:
        perturbation_fn = remove_keywords

    altered = perturbation_fn(text)
    alt_pred = classifier.predict(vectorizer.transform([altered]))[0]
    flip = 1.0 if alt_pred != pred else 0.0

    score = compute_trust_score(conf, flip, bias_gap, alpha, beta)

    return {
        "prediction": int(pred),
        "prediction_label": "Risk" if pred == 1 else "Control",
        "confidence": conf,
        "flip_rate": flip,
        "trust_score": score,
        "trust_label": categorize_trust(score),
    }

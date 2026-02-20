"""
robustness.py - Text perturbation engine and flip rate computation.
"""

import re
import random
import numpy as np
import nltk

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("stopwords", quiet=True)

from nltk.corpus import wordnet, stopwords

STOPWORDS = set(stopwords.words("english"))

# Default mental health trigger keywords (same as explainability lexicon)
TRIGGER_KEYWORDS = {
    "sad", "hopeless", "alone", "suicide", "tired", "worthless",
    "depressed", "anxious", "empty", "pain", "die", "help",
    "crying", "lost", "hate", "suffer", "miserable", "numb",
    "angry", "scared", "hurt", "broken", "overwhelmed", "desperate",
    "lonely", "useless", "failure", "guilt", "ashamed",
}


# ---------------------------------------------------------------------------
# Perturbation functions
# ---------------------------------------------------------------------------

def remove_keywords(text: str, keywords: set[str] | None = None) -> str:
    """Remove all occurrences of trigger keywords from text."""
    if keywords is None:
        keywords = TRIGGER_KEYWORDS
    words = text.split()
    return " ".join(w for w in words if w.lower() not in keywords)


def synonym_replace(text: str, n: int = 3, rng_seed: int | None = None) -> str:
    """
    Replace up to n random content words with WordNet synonyms.
    """
    rng = random.Random(rng_seed)
    words = text.split()
    content_indices = [
        i for i, w in enumerate(words)
        if w.lower() not in STOPWORDS and len(w) > 2
    ]

    if not content_indices:
        return text

    replace_indices = rng.sample(content_indices, min(n, len(content_indices)))

    for idx in replace_indices:
        word = words[idx]
        syns = wordnet.synsets(word)
        if syns:
            lemmas = [
                l.name().replace("_", " ")
                for s in syns
                for l in s.lemmas()
                if l.name().lower() != word.lower()
            ]
            if lemmas:
                words[idx] = rng.choice(lemmas)

    return " ".join(words)


# ---------------------------------------------------------------------------
# Robustness test runner
# ---------------------------------------------------------------------------

def robustness_test(model, vectorizer, texts, perturbation_fn, **fn_kwargs) -> dict:
    """
    Apply a perturbation to each text, re-predict, and compute flip rate.

    Parameters
    ----------
    model : sklearn estimator
    vectorizer : fitted TfidfVectorizer
    texts : list of str
    perturbation_fn : callable(text, **kwargs) -> str
    fn_kwargs : extra kwargs for perturbation_fn

    Returns
    -------
    dict with keys:
        flip_rate        : float (0-1)
        n_flips          : int
        n_total          : int
        flip_flags       : np.ndarray of bool
        original_conf    : np.ndarray
        perturbed_conf   : np.ndarray
        original_preds   : np.ndarray
        perturbed_preds  : np.ndarray
    """
    texts = list(texts)

    # Original predictions
    X_orig = vectorizer.transform(texts)
    orig_preds = model.predict(X_orig)
    orig_probs = model.predict_proba(X_orig)[:, 1]

    # Perturbed predictions
    perturbed_texts = [perturbation_fn(t, **fn_kwargs) for t in texts]
    X_pert = vectorizer.transform(perturbed_texts)
    pert_preds = model.predict(X_pert)
    pert_probs = model.predict_proba(X_pert)[:, 1]

    flips = orig_preds != pert_preds
    return {
        "flip_rate": float(flips.mean()),
        "n_flips": int(flips.sum()),
        "n_total": len(texts),
        "flip_flags": flips,
        "original_conf": orig_probs,
        "perturbed_conf": pert_probs,
        "original_preds": orig_preds,
        "perturbed_preds": pert_preds,
    }

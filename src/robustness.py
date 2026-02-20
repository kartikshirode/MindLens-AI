import random
import numpy as np
import nltk

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("stopwords", quiet=True)

from nltk.corpus import wordnet, stopwords

STOPWORDS = set(stopwords.words("english"))

TRIGGER_KEYWORDS = {
    "sad", "hopeless", "alone", "suicide", "tired", "worthless",
    "depressed", "anxious", "empty", "pain", "die", "help",
    "crying", "lost", "hate", "suffer", "miserable", "numb",
    "angry", "scared", "hurt", "broken", "overwhelmed", "desperate",
    "lonely", "useless", "failure", "guilt", "ashamed",
}


def remove_keywords(text, keywords=None):
    """Delete every occurrence of the trigger keywords from the input."""
    if keywords is None:
        keywords = TRIGGER_KEYWORDS
    return " ".join(w for w in text.split() if w.lower() not in keywords)


def synonym_replace(text, n=3, rng_seed=None):
    """Swap up to n content words with WordNet synonyms."""
    rng = random.Random(rng_seed)
    tokens = text.split()
    eligible = [i for i, w in enumerate(tokens) if w.lower() not in STOPWORDS and len(w) > 2]

    if not eligible:
        return text

    picked = rng.sample(eligible, min(n, len(eligible)))
    for idx in picked:
        synsets = wordnet.synsets(tokens[idx])
        if synsets:
            alternatives = [
                lem.name().replace("_", " ")
                for syn in synsets for lem in syn.lemmas()
                if lem.name().lower() != tokens[idx].lower()
            ]
            if alternatives:
                tokens[idx] = rng.choice(alternatives)
    return " ".join(tokens)


def robustness_test(classifier, vectorizer, texts, perturbation_fn, **kwargs):
    """
    Apply a perturbation to every text, re-predict, and measure how
    often the label flips.
    """
    texts = list(texts)

    x_orig = vectorizer.transform(texts)
    orig_labels = classifier.predict(x_orig)
    orig_confidence = classifier.predict_proba(x_orig)[:, 1]

    altered = [perturbation_fn(t, **kwargs) for t in texts]
    x_alt = vectorizer.transform(altered)
    new_labels = classifier.predict(x_alt)
    new_confidence = classifier.predict_proba(x_alt)[:, 1]

    flipped = orig_labels != new_labels
    return {
        "flip_rate": float(flipped.mean()),
        "n_flips": int(flipped.sum()),
        "n_total": len(texts),
        "flip_flags": flipped,
        "original_conf": orig_confidence,
        "perturbed_conf": new_confidence,
        "original_preds": orig_labels,
        "perturbed_preds": new_labels,
    }

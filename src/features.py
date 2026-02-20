"""
features.py - TF-IDF and optional embedding pipelines.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def build_tfidf(texts, max_features: int = 5000):
    """
    Fit a TF-IDF vectorizer and transform texts.

    Returns
    -------
    vectorizer : TfidfVectorizer
        Fitted vectorizer (keep for inference / LIME).
    X : sparse matrix
        TF-IDF feature matrix.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=3,
        max_df=0.95,
    )
    X = vectorizer.fit_transform(texts)
    return vectorizer, X


def build_sentence_embeddings(texts, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64):
    """
    Encode texts using a SentenceTransformer model (GPU-aware).

    Returns
    -------
    embeddings : np.ndarray   (n_samples, embed_dim)
    """
    try:
        from sentence_transformers import SentenceTransformer
        import torch
    except ImportError:
        raise ImportError(
            "Install sentence-transformers and torch: "
            "pip install sentence-transformers torch"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[features] SentenceTransformer using device: {device}")

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings

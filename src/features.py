from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def build_tfidf(corpus, max_features=5000):
    """Fit TF-IDF on the corpus and return the fitted vectorizer + sparse matrix."""
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=3,
        max_df=0.95,
    )
    matrix = tfidf.fit_transform(corpus)
    return tfidf, matrix


def build_sentence_embeddings(corpus, model_name="all-MiniLM-L6-v2", batch_size=64):
    """Produce dense sentence vectors with a SentenceTransformer model."""
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(model_name, device=device)
    vectors = encoder.encode(
        list(corpus),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return vectors

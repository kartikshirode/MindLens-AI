"""
preprocessing.py - Text cleaning, feature engineering, and data loading.
"""

import re
import os
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords

# Ensure NLTK data is available
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

STOPWORDS = set(stopwords.words("english"))

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_primary_dataset(path: str) -> pd.DataFrame:
    """
    Load the Reddit Depression dataset (clean binary labels).
    Expects a CSV with at minimum a text column and a label/class column.
    Returns DataFrame with columns: [text, label]  (1 = risk, 0 = no risk).
    """
    df = pd.read_csv(path)

    # Auto-detect column names (handles common Kaggle naming conventions)
    text_col = _find_column(df, ["clean_text", "text", "post", "selftext", "body", "title"])
    label_col = _find_column(df, ["is_depression", "label", "class", "target", "depression"])

    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "label"]

    # Normalise label to int 0/1
    df["label"] = df["label"].apply(_normalise_label)
    df.dropna(subset=["text", "label"], inplace=True)
    df["label"] = df["label"].astype(int)
    return df


def load_generalization_dataset(path: str) -> pd.DataFrame:
    """
    Load the SuicideWatch dataset for out-of-domain generalization testing.
    Same output schema as load_primary_dataset.
    """
    df = pd.read_csv(path)
    text_col = _find_column(df, ["text", "clean_text", "post", "selftext", "body", "title"])
    label_col = _find_column(df, ["label", "class", "target"])

    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "label"]
    df["label"] = df["label"].apply(_normalise_label)
    df.dropna(subset=["text", "label"], inplace=True)
    df["label"] = df["label"].astype(int)
    return df


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str, remove_stopwords: bool = False) -> str:
    """Clean a single text string."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)          # URLs
    text = re.sub(r"<.*?>", "", text)                       # HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)                   # non-alpha chars
    text = re.sub(r"\s+", " ", text).strip()                # collapse whitespace
    if remove_stopwords:
        text = " ".join(w for w in text.split() if w not in STOPWORDS)
    return text


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived numeric features to the DataFrame (in place)."""
    df = df.copy()
    df["word_count"] = df["text"].apply(lambda t: len(t.split()) if isinstance(t, str) else 0)
    df["char_count"] = df["text"].apply(lambda t: len(t) if isinstance(t, str) else 0)
    df["avg_word_length"] = df.apply(
        lambda r: r["char_count"] / r["word_count"] if r["word_count"] > 0 else 0, axis=1
    )
    df["word_density"] = df.apply(
        lambda r: r["word_count"] / r["char_count"] if r["char_count"] > 0 else 0, axis=1
    )
    df["unique_word_ratio"] = df["text"].apply(
        lambda t: len(set(t.split())) / len(t.split()) if isinstance(t, str) and len(t.split()) > 0 else 0
    )
    return df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess_pipeline(df: pd.DataFrame, remove_stopwords: bool = False) -> pd.DataFrame:
    """Run full preprocessing: clean text → drop nulls/dupes → engineer features."""
    df = df.copy()
    df["text"] = df["text"].apply(lambda t: clean_text(t, remove_stopwords=remove_stopwords))
    df = df[df["text"].str.len() > 0]
    df.drop_duplicates(subset=["text"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = engineer_features(df)
    return df


def save_processed(df: pd.DataFrame, path: str) -> None:
    """Save processed DataFrame to CSV, creating directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows → {path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Find first matching column name (case-insensitive)."""
    lower_cols = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_cols:
            return lower_cols[name.lower()]
    raise ValueError(
        f"Could not find any of {candidates} in columns {list(df.columns)}"
    )


def _normalise_label(val) -> int | None:
    """Convert various label representations to 0/1."""
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        val_lower = val.strip().lower()
        if val_lower in ("1", "depression", "suicide", "suicidewatch", "yes", "true", "risk"):
            return 1
        if val_lower in ("0", "non-depression", "no", "false", "no risk", "non-suicide"):
            return 0
    return None

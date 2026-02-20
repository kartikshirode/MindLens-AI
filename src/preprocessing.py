import re
import os
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

STOPWORDS = set(stopwords.words("english"))


def load_primary_dataset(filepath):
    """Read the Reddit depression CSV into a two-column DataFrame [text, label]."""
    raw = pd.read_csv(filepath)
    tcol = _find_column(raw, ["clean_text", "text", "post", "selftext", "body", "title"])
    lcol = _find_column(raw, ["is_depression", "label", "class", "target", "depression"])

    out = raw[[tcol, lcol]].copy()
    out.columns = ["text", "label"]
    out["label"] = out["label"].apply(_normalise_label)
    out.dropna(subset=["text", "label"], inplace=True)
    out["label"] = out["label"].astype(int)
    return out


def load_generalization_dataset(filepath):
    """Same as load_primary_dataset but for the SuicideWatch out-of-domain set."""
    raw = pd.read_csv(filepath)
    tcol = _find_column(raw, ["text", "clean_text", "post", "selftext", "body", "title"])
    lcol = _find_column(raw, ["label", "class", "target"])

    out = raw[[tcol, lcol]].copy()
    out.columns = ["text", "label"]
    out["label"] = out["label"].apply(_normalise_label)
    out.dropna(subset=["text", "label"], inplace=True)
    out["label"] = out["label"].astype(int)
    return out


def clean_text(text, remove_stopwords=False):
    """Lowercase, strip URLs / HTML / non-alpha, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if remove_stopwords:
        text = " ".join(w for w in text.split() if w not in STOPWORDS)
    return text


def engineer_features(dataframe):
    """Add word_count, char_count, avg_word_length, word_density, unique_word_ratio."""
    df = dataframe.copy()
    df["word_count"] = df["text"].apply(lambda t: len(t.split()) if isinstance(t, str) else 0)
    df["char_count"] = df["text"].apply(lambda t: len(t) if isinstance(t, str) else 0)
    df["avg_word_length"] = df.apply(
        lambda row: row["char_count"] / row["word_count"] if row["word_count"] > 0 else 0, axis=1
    )
    df["word_density"] = df.apply(
        lambda row: row["word_count"] / row["char_count"] if row["char_count"] > 0 else 0, axis=1
    )
    df["unique_word_ratio"] = df["text"].apply(
        lambda t: len(set(t.split())) / len(t.split()) if isinstance(t, str) and len(t.split()) > 0 else 0
    )
    return df


def preprocess_pipeline(dataframe, remove_stopwords=False):
    """Clean text, drop empties / duplicates, compute derived features."""
    df = dataframe.copy()
    df["text"] = df["text"].apply(lambda t: clean_text(t, remove_stopwords=remove_stopwords))
    df = df[df["text"].str.len() > 0]
    df.drop_duplicates(subset=["text"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = engineer_features(df)
    return df


def save_processed(dataframe, filepath):
    """Write a DataFrame to CSV, creating parent directories when needed."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    dataframe.to_csv(filepath, index=False)
    print(f"Saved {len(dataframe)} rows -> {filepath}")


def _find_column(dataframe, candidates):
    """Return the first matching column (case-insensitive) or raise."""
    col_map = {c.lower(): c for c in dataframe.columns}
    for name in candidates:
        if name.lower() in col_map:
            return col_map[name.lower()]
    raise ValueError(f"None of {candidates} found in {list(dataframe.columns)}")


def _normalise_label(value):
    """Convert assorted label representations into 0 or 1."""
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("1", "depression", "suicide", "suicidewatch", "yes", "true", "risk"):
            return 1
        if v in ("0", "non-depression", "no", "false", "no risk", "non-suicide"):
            return 0
    return None

"""
bias.py - Engagement-group labelling and False Positive Rate analysis.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import nltk

nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# ---------------------------------------------------------------------------
# Engagement group labelling
# ---------------------------------------------------------------------------

def label_engagement_groups(df: pd.DataFrame, col: str = "word_count") -> pd.DataFrame:
    """
    Label each row as 'high', 'mid', or 'low' engagement based on quantiles
    of the specified column.

    Also adds a 'sentiment_group' column using VADER compound score.
    """
    df = df.copy()

    q25 = df[col].quantile(0.25)
    q75 = df[col].quantile(0.75)

    df["engagement_group"] = pd.cut(
        df[col],
        bins=[-np.inf, q25, q75, np.inf],
        labels=["low", "mid", "high"],
    )

    # Sentiment intensity via VADER
    sia = SentimentIntensityAnalyzer()
    df["vader_compound"] = df["text"].apply(
        lambda t: sia.polarity_scores(t)["compound"] if isinstance(t, str) else 0.0
    )
    # Bucket absolute compound into low / mid / high intensity
    abs_compound = df["vader_compound"].abs()
    sq25 = abs_compound.quantile(0.25)
    sq75 = abs_compound.quantile(0.75)
    df["sentiment_group"] = pd.cut(
        abs_compound,
        bins=[-np.inf, sq25, sq75, np.inf],
        labels=["low", "mid", "high"],
    )

    return df


# ---------------------------------------------------------------------------
# FPR by group
# ---------------------------------------------------------------------------

def compute_fpr_by_group(y_true, y_pred, groups) -> dict:
    """
    Compute False Positive Rate for each group.

    FPR = FP / (FP + TN)

    Returns
    -------
    dict  {group_label: fpr}
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    groups = np.array(groups)

    fpr_dict = {}
    for grp in np.unique(groups):
        mask = groups == grp
        yt = y_true[mask]
        yp = y_pred[mask]

        fp = np.sum((yp == 1) & (yt == 0))
        tn = np.sum((yp == 0) & (yt == 0))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fpr_dict[grp] = fpr

    return fpr_dict


# ---------------------------------------------------------------------------
# Statistical significance
# ---------------------------------------------------------------------------

def run_significance_test(y_true, y_pred, groups, group_a: str, group_b: str) -> dict:
    """
    Chi-square test comparing FPR between two groups.

    Returns
    -------
    dict with keys: chi2, p_value, group_a_fpr, group_b_fpr, significant (bool)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    groups = np.array(groups)

    def _contingency(grp):
        mask = groups == grp
        yt = y_true[mask]
        yp = y_pred[mask]
        fp = np.sum((yp == 1) & (yt == 0))
        tn = np.sum((yp == 0) & (yt == 0))
        return fp, tn

    fp_a, tn_a = _contingency(group_a)
    fp_b, tn_b = _contingency(group_b)

    # 2×2 contingency table: rows = groups, cols = [FP, TN]
    table = np.array([[fp_a, tn_a], [fp_b, tn_b]])

    # Guard against zero rows/cols
    if table.sum() == 0 or np.any(table.sum(axis=1) == 0):
        return {
            "chi2": 0.0, "p_value": 1.0,
            "group_a_fpr": 0.0, "group_b_fpr": 0.0,
            "significant": False,
        }

    chi2, p, _, _ = chi2_contingency(table, correction=True)
    fpr_a = fp_a / (fp_a + tn_a) if (fp_a + tn_a) > 0 else 0.0
    fpr_b = fp_b / (fp_b + tn_b) if (fp_b + tn_b) > 0 else 0.0

    return {
        "chi2": float(chi2),
        "p_value": float(p),
        f"{group_a}_fpr": fpr_a,
        f"{group_b}_fpr": fpr_b,
        "significant": p < 0.05,
    }

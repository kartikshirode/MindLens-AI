import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import nltk

nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def label_engagement_groups(dataframe, col="word_count"):
    """
    Split rows into low / mid / high engagement using quantile thresholds.
    Also adds a sentiment_group column based on VADER compound intensity.
    """
    df = dataframe.copy()
    lower_q, upper_q = df[col].quantile(0.25), df[col].quantile(0.75)

    df["engagement_group"] = pd.cut(
        df[col], bins=[-np.inf, lower_q, upper_q, np.inf], labels=["low", "mid", "high"],
    )

    analyser = SentimentIntensityAnalyzer()
    df["vader_compound"] = df["text"].apply(
        lambda t: analyser.polarity_scores(t)["compound"] if isinstance(t, str) else 0.0
    )
    abs_sent = df["vader_compound"].abs()
    s_lo, s_hi = abs_sent.quantile(0.25), abs_sent.quantile(0.75)
    df["sentiment_group"] = pd.cut(
        abs_sent, bins=[-np.inf, s_lo, s_hi, np.inf], labels=["low", "mid", "high"],
    )
    return df


def compute_fpr_by_group(y_true, y_pred, groups):
    """Return {group_label: false_positive_rate} for each unique group."""
    y_true, y_pred, groups = np.array(y_true), np.array(y_pred), np.array(groups)
    rates = {}
    for g in np.unique(groups):
        mask = groups == g
        fp = np.sum((y_pred[mask] == 1) & (y_true[mask] == 0))
        tn = np.sum((y_pred[mask] == 0) & (y_true[mask] == 0))
        rates[g] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return rates


def run_significance_test(y_true, y_pred, groups, group_a, group_b):
    """Chi-square comparison of false-positive rates between two groups."""
    y_true, y_pred, groups = np.array(y_true), np.array(y_pred), np.array(groups)

    def _counts(g):
        mask = groups == g
        fp = np.sum((y_pred[mask] == 1) & (y_true[mask] == 0))
        tn = np.sum((y_pred[mask] == 0) & (y_true[mask] == 0))
        return fp, tn

    fp_a, tn_a = _counts(group_a)
    fp_b, tn_b = _counts(group_b)
    table = np.array([[fp_a, tn_a], [fp_b, tn_b]])

    if table.sum() == 0 or np.any(table.sum(axis=1) == 0):
        return {"chi2": 0.0, "p_value": 1.0,
                f"{group_a}_fpr": 0.0, f"{group_b}_fpr": 0.0, "significant": False}

    chi2_stat, p_val, _, _ = chi2_contingency(table, correction=True)
    fpr_a = fp_a / (fp_a + tn_a) if (fp_a + tn_a) > 0 else 0.0
    fpr_b = fp_b / (fp_b + tn_b) if (fp_b + tn_b) > 0 else 0.0

    return {
        "chi2": float(chi2_stat), "p_value": float(p_val),
        f"{group_a}_fpr": fpr_a, f"{group_b}_fpr": fpr_b,
        "significant": p_val < 0.05,
    }

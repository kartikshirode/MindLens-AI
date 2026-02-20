"""
evaluation.py - Cross-validation, metrics, and error analysis.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


# ---------------------------------------------------------------------------
# Single evaluation
# ---------------------------------------------------------------------------

def evaluate_single(model, X_test, y_test) -> dict:
    """
    Evaluate a trained model on a test set.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc, pr_auc,
                    confusion_matrix, classification_report (str).
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
    }

    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
        metrics["pr_auc"] = average_precision_score(y_test, y_prob)

    return metrics


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_model(
    model_cls,
    X,
    y,
    n_splits: int = 5,
    model_kwargs: dict | None = None,
) -> pd.DataFrame:
    """
    Run stratified k-fold cross-validation.

    Parameters
    ----------
    model_cls : class
        Scikit-learn estimator class (e.g., LogisticRegression).
    X : array-like
        Feature matrix.
    y : array-like
        Labels.
    n_splits : int
        Number of CV folds.
    model_kwargs : dict
        Extra kwargs passed to the model constructor.

    Returns
    -------
    pd.DataFrame with one row per fold + a mean±std summary row.
    """
    if model_kwargs is None:
        model_kwargs = {}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    records = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = np.array(y)[train_idx], np.array(y)[val_idx]

        model = model_cls(**model_kwargs)
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None

        row = {
            "fold": fold,
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1": f1_score(y_val, y_pred, zero_division=0),
        }
        if y_prob is not None:
            row["roc_auc"] = roc_auc_score(y_val, y_prob)
            row["pr_auc"] = average_precision_score(y_val, y_prob)

        records.append(row)
        print(f"  Fold {fold}: F1={row['f1']:.4f}  ROC-AUC={row.get('roc_auc', 'N/A')}")

    df = pd.DataFrame(records)

    # Summary row
    summary = {"fold": "mean±std"}
    for col in df.columns:
        if col == "fold":
            continue
        summary[col] = f"{df[col].mean():.4f} ± {df[col].std():.4f}"
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

    return df


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def error_analysis(model, X_test, y_test, texts, n: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract the most confident False Positives and False Negatives.

    Returns
    -------
    (fp_df, fn_df) each with columns: [text, true_label, pred_label, confidence]
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    texts = list(texts)
    results = pd.DataFrame({
        "text": texts[: len(y_test)],
        "true_label": y_test,
        "pred_label": y_pred,
        "confidence": y_prob,
    })

    fp = results[(results["pred_label"] == 1) & (results["true_label"] == 0)]
    fn = results[(results["pred_label"] == 0) & (results["true_label"] == 1)]

    fp_top = fp.sort_values("confidence", ascending=False).head(n).reset_index(drop=True)
    fn_top = fn.sort_values("confidence", ascending=True).head(n).reset_index(drop=True)

    return fp_top, fn_top

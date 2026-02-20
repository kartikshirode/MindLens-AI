import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
)


def evaluate_single(classifier, x_test, y_test):
    """Run standard classification metrics on a held-out test set."""
    predicted = classifier.predict(x_test)
    probabilities = classifier.predict_proba(x_test)[:, 1] if hasattr(classifier, "predict_proba") else None

    out = {
        "accuracy": accuracy_score(y_test, predicted),
        "precision": precision_score(y_test, predicted, zero_division=0),
        "recall": recall_score(y_test, predicted, zero_division=0),
        "f1": f1_score(y_test, predicted, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predicted),
        "classification_report": classification_report(y_test, predicted),
    }
    if probabilities is not None:
        out["roc_auc"] = roc_auc_score(y_test, probabilities)
        out["pr_auc"] = average_precision_score(y_test, probabilities)
    return out


def cross_validate_model(estimator_class, x_all, y_all, n_splits=5, model_kwargs=None):
    """Stratified k-fold cross validation, returns a DataFrame with per-fold rows + summary."""
    if model_kwargs is None:
        model_kwargs = {}

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows = []

    for fold_num, (tr_idx, va_idx) in enumerate(kfold.split(x_all, y_all), start=1):
        x_tr, x_va = x_all[tr_idx], x_all[va_idx]
        y_tr, y_va = np.array(y_all)[tr_idx], np.array(y_all)[va_idx]

        estimator = estimator_class(**model_kwargs)
        estimator.fit(x_tr, y_tr)

        predicted = estimator.predict(x_va)
        probabilities = estimator.predict_proba(x_va)[:, 1] if hasattr(estimator, "predict_proba") else None

        row = {
            "fold": fold_num,
            "accuracy": accuracy_score(y_va, predicted),
            "precision": precision_score(y_va, predicted, zero_division=0),
            "recall": recall_score(y_va, predicted, zero_division=0),
            "f1": f1_score(y_va, predicted, zero_division=0),
        }
        if probabilities is not None:
            row["roc_auc"] = roc_auc_score(y_va, probabilities)
            row["pr_auc"] = average_precision_score(y_va, probabilities)

        rows.append(row)
        print(f"  Fold {fold_num}: F1={row['f1']:.4f}  ROC-AUC={row.get('roc_auc', 'N/A')}")

    results_df = pd.DataFrame(rows)

    summary = {"fold": "mean+/-std"}
    for col in results_df.columns:
        if col != "fold":
            summary[col] = f"{results_df[col].mean():.4f} +/- {results_df[col].std():.4f}"
    results_df = pd.concat([results_df, pd.DataFrame([summary])], ignore_index=True)
    return results_df


def error_analysis(classifier, x_test, y_test, raw_texts, n=20):
    """Pull out the n most confident false positives and false negatives."""
    predicted = classifier.predict(x_test)
    probabilities = classifier.predict_proba(x_test)[:, 1]

    df = pd.DataFrame({
        "text": list(raw_texts)[: len(y_test)],
        "true_label": y_test,
        "pred_label": predicted,
        "confidence": probabilities,
    })

    false_pos = df[(df["pred_label"] == 1) & (df["true_label"] == 0)]
    false_neg = df[(df["pred_label"] == 0) & (df["true_label"] == 1)]

    top_fp = false_pos.sort_values("confidence", ascending=False).head(n).reset_index(drop=True)
    top_fn = false_neg.sort_values("confidence", ascending=True).head(n).reset_index(drop=True)
    return top_fp, top_fn

"""
model.py - Model training, saving, and loading (GPU-aware for deep models).
"""

import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def get_device():
    """Return 'cuda' if a GPU is available, else 'cpu'."""
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    print(f"[model] Using device: {device}")
    return device


# ---------------------------------------------------------------------------
# Baseline model (Logistic Regression - CPU, but fast enough)
# ---------------------------------------------------------------------------

def train_baseline(X_train, y_train, max_iter: int = 1000):
    """
    Train a Logistic Regression baseline with balanced class weights.

    Returns
    -------
    model : LogisticRegression
    """
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=max_iter,
        solver="lbfgs",
        C=1.0,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Advanced model (DistilBERT on GPU)
# ---------------------------------------------------------------------------

def train_distilbert(
    train_texts,
    train_labels,
    val_texts=None,
    val_labels=None,
    model_name: str = "distilbert-base-uncased",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    output_dir: str = "data/processed/distilbert_model",
):
    """
    Fine-tune DistilBERT for binary classification (GPU-accelerated).

    Returns
    -------
    trainer : transformers.Trainer
    tokenizer : transformers.AutoTokenizer
    """
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
    )
    from torch.utils.data import Dataset

    device = get_device()

    # ---- Tokenize ----
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class MHDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len=256):
            self.encodings = tokenizer(
                list(texts),
                truncation=True,
                padding=True,
                max_length=max_len,
                return_tensors="pt",
            )
            self.labels = torch.tensor(list(labels), dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item["labels"] = self.labels[idx]
            return item

    train_dataset = MHDataset(train_texts, train_labels, tokenizer)
    val_dataset = MHDataset(val_texts, val_labels, tokenizer) if val_texts is not None else None

    # ---- Model ----
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    # ---- Training Args (GPU-optimised) ----
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch" if val_dataset else "no",
        save_strategy="epoch",
        load_best_model_at_end=bool(val_dataset),
        logging_steps=50,
        fp16=torch.cuda.is_available(),   # Mixed precision on GPU
        report_to="none",
        seed=42,
    )

    # ---- Metrics callback ----
    from sklearn.metrics import accuracy_score, f1_score

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="binary"),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print(f"[model] Training DistilBERT on {device} for {epochs} epochs ...")
    trainer.train()
    return trainer, tokenizer


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model, vectorizer, path: str = "data/processed/model_artifacts.joblib"):
    """Save model and vectorizer together."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"model": model, "vectorizer": vectorizer}, path)
    print(f"[model] Saved → {path}")


def load_model(path: str = "data/processed/model_artifacts.joblib"):
    """Load model and vectorizer."""
    artifacts = joblib.load(path)
    return artifacts["model"], artifacts["vectorizer"]

import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


def get_device():
    """Return 'cuda' when a GPU is available, otherwise 'cpu'."""
    try:
        import torch
        hw = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        hw = "cpu"
    print(f"Using device: {hw}")
    return hw


def train_baseline(x_train, y_train, max_iter=1000):
    """Fit a Logistic Regression with balanced class weights."""
    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=max_iter,
        solver="lbfgs",
        C=1.0,
        random_state=42,
    )
    lr.fit(x_train, y_train)
    return lr


def train_distilbert(
    train_texts, train_labels,
    val_texts=None, val_labels=None,
    model_name="distilbert-base-uncased",
    epochs=3, batch_size=16, learning_rate=2e-5,
    output_dir="data/processed/distilbert_model",
):
    """Fine-tune DistilBERT for binary text classification."""
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        Trainer, TrainingArguments,
    )
    from torch.utils.data import Dataset
    from sklearn.metrics import accuracy_score, f1_score

    hw = get_device()
    tok = AutoTokenizer.from_pretrained(model_name)

    class TextDataset(Dataset):
        def __init__(self, texts, labels, tokeniser, max_len=256):
            self.enc = tokeniser(
                list(texts), truncation=True, padding=True,
                max_length=max_len, return_tensors="pt",
            )
            self.targets = torch.tensor(list(labels), dtype=torch.long)

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.enc.items()}
            item["labels"] = self.targets[idx]
            return item

    ds_train = TextDataset(train_texts, train_labels, tok)
    ds_val = TextDataset(val_texts, val_labels, tok) if val_texts is not None else None

    classifier = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    classifier.to(hw)

    training_cfg = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch" if ds_val else "no",
        save_strategy="epoch",
        load_best_model_at_end=bool(ds_val),
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=42,
    )

    def _metrics(eval_pred):
        logits, labels = eval_pred
        predicted = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predicted),
            "f1": f1_score(labels, predicted, average="binary"),
        }

    trainer = Trainer(
        model=classifier, args=training_cfg,
        train_dataset=ds_train, eval_dataset=ds_val,
        compute_metrics=_metrics,
    )
    trainer.train()
    return trainer, tok


def save_model(trained_model, fitted_vectorizer, filepath="data/processed/model_artifacts.joblib"):
    """Bundle model + vectorizer into one joblib file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump({"model": trained_model, "vectorizer": fitted_vectorizer}, filepath)
    print(f"Model saved -> {filepath}")


def load_model(filepath="data/processed/model_artifacts.joblib"):
    """Reload the model + vectorizer bundle."""
    bundle = joblib.load(filepath)
    return bundle["model"], bundle["vectorizer"]

"""DeBERTa discourse-effectiveness classification.

Converted from the original research notebook while preserving the original code-cell execution order.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models" / "deberta_saved_model"
CHECKPOINT_DIR = OUTPUT_DIR / "deberta_discourse_effectiveness"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# %% [Original notebook cell 1]

# %% [Original notebook cell 2]

# %% [Original notebook cell 3]

# %% [Original notebook cell 4]

# %% [Original notebook cell 5]
import sys
print(sys.executable)

# %% [Original notebook cell 6]
import sys


# %% [Original notebook cell 7]
import sys


# %% [Original notebook cell 8]
import torch
print(torch.__version__)
print(torch.cuda.is_available())

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
print("Transformers import OK")

# %% [Original notebook cell 9]

# %% [Original notebook cell 10]
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from torch import nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)

# %% [Original notebook cell 11]
file_path = DATA_DIR / "persuade_main_corpus_version.csv"
df = pd.read_csv(file_path)

df = df[["full_text", "discourse_text", "discourse_type", "discourse_effectiveness"]].copy()
print("Shape:", df.shape)
df.head()

# %% [Original notebook cell 12]
label2id = {"Ineffective": 0, "Adequate": 1, "Effective": 2}
id2label = {v: k for k, v in label2id.items()}

df["label"] = df["discourse_effectiveness"].map(label2id)
print(df["discourse_effectiveness"].value_counts())

# %% [Original notebook cell 13]
gss_1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, temp_idx = next(gss_1.split(df, y=df["label"], groups=df["full_text"]))

train_df = df.iloc[train_idx].reset_index(drop=True)
temp_df  = df.iloc[temp_idx].reset_index(drop=True)

gss_2 = GroupShuffleSplit(n_splits=1, test_size=2/3, random_state=42)
val_idx, test_idx = next(gss_2.split(temp_df, y=temp_df["label"], groups=temp_df["full_text"]))

val_df  = temp_df.iloc[val_idx].reset_index(drop=True)
test_df = temp_df.iloc[test_idx].reset_index(drop=True)

print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

# %% [Original notebook cell 14]
# CHANGED: model name only — everything else stays the same
model_name = "microsoft/deberta-v3-base"  # was: distilbert-base-uncased

tokenizer = AutoTokenizer.from_pretrained(model_name)

# %% [Original notebook cell 15]
# CHANGED: sentence-pair tokenization instead of string concatenation
def tokenize_function(examples):
    return tokenizer(
        examples["discourse_type"],   # Sentence A — structural type signal
        examples["discourse_text"],   # Sentence B — actual content
        truncation=True,
        padding=False,
        max_length=256                # CHANGED: 128 → 256 (discourse text can be long)
    )

# Build datasets from the right columns now
train_dataset = Dataset.from_pandas(train_df[["discourse_type", "discourse_text", "label"]].reset_index(drop=True))
val_dataset   = Dataset.from_pandas(val_df[["discourse_type", "discourse_text", "label"]].reset_index(drop=True))
test_dataset  = Dataset.from_pandas(test_df[["discourse_type", "discourse_text", "label"]].reset_index(drop=True))

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset   = val_dataset.map(tokenize_function, batched=True)
test_dataset  = test_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.rename_column("label", "labels")
val_dataset   = val_dataset.rename_column("label", "labels")
test_dataset  = test_dataset.rename_column("label", "labels")

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch",   columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch",  columns=["input_ids", "attention_mask", "labels"])

# %% [Original notebook cell 16]
# Compute class weights from train split only
train_labels_np = train_df["label"].to_numpy()
class_ids = np.array([0, 1, 2])
class_weights = compute_class_weight(class_weight="balanced", classes=class_ids, y=train_labels_np)
class_weights = torch.tensor(class_weights, dtype=torch.float)
print("Class weights:", class_weights)


# CHANGED: Focal Loss replaces WeightedCrossEntropy
class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma: focusing parameter. Higher = more focus on hard examples.
           0 = standard cross-entropy. Recommended starting value: 2.0
    weight: per-class weights (same as class_weight in CrossEntropyLoss)
    """
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma  = gamma

    def forward(self, logits, labels):
        # Standard CE loss (per sample, unreduced)
        ce_loss = nn.functional.cross_entropy(
            logits, labels, weight=self.weight, reduction="none"
        )
        # p_t = probability of the true class
        pt = torch.exp(-ce_loss)
        # Focal weight: (1 - pt)^gamma
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class FocalTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.get("logits")

        loss_fct = FocalLoss(
            weight=class_weights.to(model.device),
            gamma=2.0
        )
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        inputs["labels"] = labels
        return (loss, outputs) if return_outputs else loss

# %% [Original notebook cell 17]
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {
        "accuracy":    accuracy_score(labels, predictions),
        "macro_f1":    f1_score(labels, predictions, average="macro"),
        "weighted_f1": f1_score(labels, predictions, average="weighted")
    }


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %% [Original notebook cell 18]
training_args = TrainingArguments(
    output_dir=str(CHECKPOINT_DIR),
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",

    # Core training setup
    learning_rate=1e-5,          # Lower LR is usually more stable for DeBERTa fine-tuning
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,

    # Regularization / overfitting control
    weight_decay=0.01,
    warmup_ratio=0.1,
    max_grad_norm=1.0,           # ADDED: gradient clipping to stabilize training
    lr_scheduler_type="cosine",  # ADDED: smoother learning-rate decay than linear

    # Best checkpoint selection
    load_best_model_at_end=True,
    metric_for_best_model="eval_macro_f1",  # ADDED: explicitly track Macro F1
    greater_is_better=True,
    save_total_limit=1,
    report_to="none"
)

# %% [Original notebook cell 19]
trainer = FocalTrainer(      # CHANGED: WeightedTrainer → FocalTrainer
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # ADDED: stop when Macro F1 stops improving
)

trainer.train()

# Confirm which checkpoint was selected at the end
print("Best checkpoint:", trainer.state.best_model_checkpoint)
print("Best metric:", trainer.state.best_metric)

# %% [Original notebook cell 20]
# Get validation logits
val_outputs = trainer.predict(val_dataset)
val_logits  = val_outputs.predictions
val_labels  = val_outputs.label_ids
val_probs   = torch.softmax(torch.tensor(val_logits), dim=1).numpy()

# Search for best threshold for Ineffective class (label 0)
best_threshold = 0.33
best_f1 = 0.0

print("Threshold search for Ineffective class:")
print(f"{'Threshold':>10} | {'Macro F1':>10} | {'Ineffective Recall':>18}")
print("-" * 45)

for threshold in np.arange(0.15, 0.50, 0.02):
    # Predict Ineffective if its probability exceeds the threshold;
    # otherwise fall back to argmax among Adequate and Effective
    preds = []
    for probs_row in val_probs:
        if probs_row[0] >= threshold:          # Ineffective probability
            preds.append(0)
        else:
            preds.append(np.argmax(probs_row[1:]) + 1)  # best of Adequate/Effective
    preds = np.array(preds)

    macro_f1 = f1_score(val_labels, preds, average="macro")
    from sklearn.metrics import recall_score
    ineff_recall = recall_score(val_labels, preds, labels=[0], average="macro")

    print(f"{threshold:>10.2f} | {macro_f1:>10.4f} | {ineff_recall:>18.4f}")

    if macro_f1 > best_f1:
        best_f1 = macro_f1
        best_threshold = threshold

print(f"\nBest threshold: {best_threshold:.2f} (Val Macro F1: {best_f1:.4f})")

# %% [Original notebook cell 21]
test_outputs = trainer.predict(test_dataset)
test_logits  = test_outputs.predictions
test_labels  = test_outputs.label_ids
test_probs   = torch.softmax(torch.tensor(test_logits), dim=1).numpy()

# Apply calibrated threshold
test_preds_calibrated = []
for probs_row in test_probs:
    if probs_row[0] >= best_threshold:
        test_preds_calibrated.append(0)
    else:
        test_preds_calibrated.append(np.argmax(probs_row[1:]) + 1)
test_preds_calibrated = np.array(test_preds_calibrated)

# Also keep argmax predictions for comparison
test_preds_argmax = np.argmax(test_logits, axis=1)

print("=" * 50)
print("ARGMAX predictions (standard):")
print(f"  Macro F1: {f1_score(test_labels, test_preds_argmax, average='macro'):.4f}")
print()
print(f"CALIBRATED predictions (threshold={best_threshold:.2f}):")
print(f"  Macro F1: {f1_score(test_labels, test_preds_calibrated, average='macro'):.4f}")
print("=" * 50)

# %% [Original notebook cell 22]
print("\nClassification Report (calibrated):")
print(classification_report(
    test_labels,
    test_preds_calibrated,
    target_names=["Ineffective", "Adequate", "Effective"]
))

# %% [Original notebook cell 23]
cm = confusion_matrix(test_labels, test_preds_calibrated)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Ineffective", "Adequate", "Effective"],
    yticklabels=["Ineffective", "Adequate", "Effective"]
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix — DeBERTa-v3 (threshold={best_threshold:.2f})")
plt.tight_layout()
plt.show()

# %% [Original notebook cell 24]
# Save results
test_results = test_df.reset_index(drop=True).copy()
test_results["true_label"]      = test_labels
test_results["pred_label"]      = test_preds_calibrated
test_results["true_label_name"] = test_results["true_label"].map(id2label)
test_results["pred_label_name"] = test_results["pred_label"].map(id2label)
test_results["prob_ineffective"] = test_probs[:, 0]
test_results["prob_adequate"]    = test_probs[:, 1]
test_results["prob_effective"]   = test_probs[:, 2]

test_results.to_csv(OUTPUT_DIR / "deberta_test_predictions.csv", index=False)
test_results.head()

# %% [Original notebook cell 25]
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

# %% [Original notebook cell 26]


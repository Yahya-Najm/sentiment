import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
)

LABEL_NAMES = ["neg", "pos"]
BATCH_SIZE  = 32
MODEL_PATH  = "./models/final"
DATA_PATH   = "data/test"

# ── 1. Load test split ────────────────────────────────────────────────────────
print("[1/6] Loading tokenized test split from data/test ...")
test_dataset = load_from_disk(DATA_PATH)
print(f"  Examples : {len(test_dataset):,}")
print(f"  Features : {list(test_dataset.features.keys())}")

# ── 2. Load model + tokenizer ─────────────────────────────────────────────────
print(f"\n[2/6] Loading model from {MODEL_PATH} ...")
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model.eval()
print(f"  Parameters : {sum(p.numel() for p in model.parameters()):,}")

# ── 3. Device setup ───────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[3/6] Device : {device}")
model.to(device)

# ── 4. Inference ──────────────────────────────────────────────────────────────
print(f"\n[4/6] Running inference in batches of {BATCH_SIZE} ...")
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
print(f"  Total batches : {len(loader)}")

all_logits = []
all_labels = []

with torch.no_grad():
    for i, batch in enumerate(loader, start=1):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        all_logits.append(outputs.logits.cpu())
        all_labels.append(labels)

        if i % 50 == 0 or i == len(loader):
            print(f"  Batch {i}/{len(loader)} ...")

all_logits = torch.cat(all_logits, dim=0)          # (N, 2)
all_labels = torch.cat(all_labels, dim=0).numpy()  # (N,)
all_probs  = F.softmax(all_logits, dim=-1).numpy() # (N, 2)
all_preds  = all_probs.argmax(axis=1)              # (N,)
print(f"  Done. {len(all_preds):,} predictions collected.")

# ── 5. Metrics ────────────────────────────────────────────────────────────────
print("\n[5/6] Computing metrics ...")

accuracy  = accuracy_score(all_labels, all_preds)
f1        = f1_score(all_labels, all_preds, average="binary")
precision = precision_score(all_labels, all_preds, average="binary")
recall    = recall_score(all_labels, all_preds, average="binary")
cm        = confusion_matrix(all_labels, all_preds)

print(f"\n  Accuracy  : {accuracy:.4f}")
print(f"  F1        : {f1:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")

print("\n  Confusion matrix  (rows = true label, cols = predicted label)")
print(f"  {'':12s}  {'Pred neg':>10s}  {'Pred pos':>10s}")
print(f"  {'True neg':12s}  {cm[0, 0]:>10d}  {cm[0, 1]:>10d}")
print(f"  {'True pos':12s}  {cm[1, 0]:>10d}  {cm[1, 1]:>10d}")

# ── 6. Sample predictions ─────────────────────────────────────────────────────
print("\n[6/6] Sample predictions (5 examples) ...")
test_dataset.reset_format()

for i in range(5):
    raw_ids    = test_dataset[i]["input_ids"]
    text       = tokenizer.decode(raw_ids, skip_special_tokens=True)
    true_label = LABEL_NAMES[all_labels[i]]
    pred_label = LABEL_NAMES[all_preds[i]]
    confidence = all_probs[i, all_preds[i]]
    correct    = "correct" if all_labels[i] == all_preds[i] else "WRONG"

    print(f"\n  -- Example {i + 1} [{correct}] --")
    print(f"  Review     : {text[:220]}{'...' if len(text) > 220 else ''}")
    print(f"  True label : {true_label}")
    print(f"  Predicted  : {pred_label}  (confidence {confidence:.2%})")

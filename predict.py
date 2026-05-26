"""
predict.py  —  run sentiment prediction on raw text.

Usage
-----
  # single review passed as a CLI argument
  python predict.py "This movie was absolutely fantastic!"

  # interactive mode (type reviews one at a time, Ctrl-C to quit)
  python predict.py
"""

import sys
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH  = "./models/final"
LABEL_NAMES = ["neg", "pos"]
MAX_LENGTH  = 512

# ── 1. Load model + tokenizer ─────────────────────────────────────────────────
print("[1/2] Loading model and tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
print(f"  Model loaded from : {MODEL_PATH}")
print(f"  Parameters        : {sum(p.numel() for p in model.parameters()):,}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"  Device            : {device}")

# ── 2. Prediction function ────────────────────────────────────────────────────
print("\n[2/2] Prediction function ready.")

def predict(text: str) -> dict:
    """Tokenize text and return label, confidence, and per-class probabilities."""
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    probs    = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
    pred_idx = int(probs.argmax())
    return {
        "label":      LABEL_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "neg_prob":   float(probs[0]),
        "pos_prob":   float(probs[1]),
    }


def print_result(text: str, result: dict) -> None:
    print("\n" + "-" * 60)
    print(f"  Review     : {text[:220]}{'...' if len(text) > 220 else ''}")
    print(f"  Sentiment  : {result['label'].upper()}")
    print(f"  Confidence : {result['confidence']:.2%}")
    print(f"  neg prob   : {result['neg_prob']:.4f}  |  pos prob : {result['pos_prob']:.4f}")
    print("-" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    # Single review passed as a CLI argument
    text   = " ".join(sys.argv[1:])
    print(f"\nPredicting for : \"{text[:80]}{'...' if len(text) > 80 else ''}\"")
    result = predict(text)
    print_result(text, result)

else:
    # Interactive mode
    print("\nInteractive mode — type a review and press Enter. Ctrl-C to quit.\n")
    while True:
        try:
            text = input("Review > ").strip()
            if not text:
                print("  (empty input, try again)")
                continue
            result = predict(text)
            print_result(text, result)
        except KeyboardInterrupt:
            print("\nExiting.")
            break

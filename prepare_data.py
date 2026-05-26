from datasets import load_dataset
from transformers import AutoTokenizer

# ── Step 1: Load IMDB dataset ─────────────────────────────────────────────────
print("\n[1/6] Loading IMDB dataset from HuggingFace...")
dataset = load_dataset("stanfordnlp/imdb")

# ── Step 2: Inspect split sizes ───────────────────────────────────────────────
print("\n[2/6] Dataset split sizes:")
print(f"  Train : {len(dataset['train']):,} examples")
print(f"  Test  : {len(dataset['test']):,} examples")

# ── Step 3: Print 3 example reviews ──────────────────────────────────────────
print("\n[3/6] Sample reviews (first 3 from train):")
label_map = {0: "negative", 1: "positive"}
for i in range(3):
    example = dataset["train"][i]
    print(f"\n  --- Example {i + 1} ---")
    print(f"  Label : {label_map[example['label']]} ({example['label']})")
    print(f"  Review: {example['text'][:300]}...")

# ── Step 4: Tokenise ──────────────────────────────────────────────────────────
print("\n[4/6] Loading tokenizer (distilbert-base-uncased) and tokenising...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    tokens = tokenizer(
        batch["text"],
        max_length=256,
        padding="max_length",
        truncation=True,
    )
    tokens["label"] = batch["label"]
    return tokens

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
print("  Tokenisation complete.")

# ── Step 5: Split 10% of train into validation ────────────────────────────────
print("\n[5/6] Splitting 10% of training data into validation set...")
train_val = tokenized["train"].train_test_split(test_size=0.1, seed=42)
train_split = train_val["train"]
val_split   = train_val["test"]
test_split  = tokenized["test"]

print(f"  Train      : {len(train_split):,} examples")
print(f"  Validation : {len(val_split):,} examples")
print(f"  Test       : {len(test_split):,} examples")

# ── Step 6: Save splits to disk ───────────────────────────────────────────────
print("\n[6/6] Saving splits to data/ folder...")
train_split.save_to_disk("data/train")
val_split.save_to_disk("data/val")
test_split.save_to_disk("data/test")
print("  Saved: data/train, data/val, data/test")

# ── Bonus: Show shape of one tokenised example ───────────────────────────────
print("\n[Done] Shape of a single tokenised example:")
sample = train_split[0]
print(f"  input_ids      : list of {len(sample['input_ids'])} ints  (first 10: {sample['input_ids'][:10]})")
print(f"  attention_mask : list of {len(sample['attention_mask'])} ints  (first 10: {sample['attention_mask'][:10]})")
print(f"  label          : {sample['label']} ({label_map[sample['label']]})")

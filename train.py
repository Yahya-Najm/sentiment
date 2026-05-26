from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

print("[1/5] Loading tokenized train and val splits...")
train_dataset = load_from_disk("data/train").select(range(4000))
val_dataset   = load_from_disk("data/val").select(range(500))
print(f"  Train : {len(train_dataset):,} examples")
print(f"  Val   : {len(val_dataset):,} examples")

print("\n[2/5] Loading distilbert-base-uncased with sequence classification head (2 labels)...")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
print(f"  Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

print("\n[3/5] Defining training arguments...")
training_args = TrainingArguments(
    output_dir="./models",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    fp16=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=100,
)
print("  Training arguments set.")

print("\n[4/5] Creating Trainer and starting training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
trainer.train()
print("  Training complete.")

print("\n[5/5] Saving final model to ./models/final ...")
trainer.save_model("./models/final")
print("  Model saved to ./models/final")

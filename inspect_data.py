from datasets import load_dataset
import pandas as pd

pd.set_option("display.max_colwidth", None)

print("Loading raw IMDB dataset...")
dataset = load_dataset("stanfordnlp/imdb")

for name in ["train", "test"]:
    df = dataset[name].to_pandas()

    print(f"\n{'='*60}")
    print(f"  Split : {name}  ({len(df):,} rows)")
    print(f"{'='*60}")
    print(f"\nColumns : {list(df.columns)}")
    print(f"Dtypes  :\n{df.dtypes}\n")
    print("First 5 reviews:\n")
    for i, row in df.head(5).iterrows():
        label = "positive" if row["label"] == 1 else "negative"
        print(f"  [{i}] Label: {label}")
        print(f"  {row['text']}")
        print()
    print(f"\nLabel distribution:\n{df['label'].value_counts().rename({0: 'negative', 1: 'positive'})}")

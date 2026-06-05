# ---------------- IMPORTS ----------------
import pandas as pd
import re

# ---------------- TEXT CLEANING ----------------
def clean_text(text):
    text = str(text)

    # Fix OCR line breaks
    text = text.replace("\n", " ")

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Keep punctuation (important)
    text = re.sub(r"[^a-z0-9\s.,!?']", "", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------- LOAD & COMBINE DATASETS ----------------
def load_and_combine():
    print("Loading datasets...")

    df1 = pd.read_csv("datasets/train_essays.csv")
    df2 = pd.read_csv("datasets/advanced_hc3_3class_small.csv")
    df3 = pd.read_csv("datasets/ai_vs_human_dataset_medium.csv")
    df4 = pd.read_csv("datasets/ai_vs_human_text.csv")

    # ---- Dataset 1 ----
    df1 = df1.rename(columns={"text": "content", "generated": "label"})
    df1 = df1[["content", "label"]]

    # ---- Dataset 2 ----
    df2 = df2.rename(columns={"text": "content", "label": "label"})
    df2["label"] = df2["label"].apply(lambda x: 1 if str(x).lower() == "ai" else 0)
    df2 = df2[["content", "label"]]

    # ---- Dataset 3 ----
    df3 = df3.rename(columns={"text": "content", "label": "label"})
    df3["label"] = df3["label"].apply(lambda x: 1 if str(x).lower() == "ai" else 0)
    df3 = df3[["content", "label"]]

    # ---- Dataset 4 ----
    df4 = df4.rename(columns={"text": "content", "label": "label"})
    df4["label"] = df4["label"].apply(lambda x: 1 if "ai" in str(x).lower() else 0)
    df4 = df4[["content", "label"]]

    # ---------------- COMBINE ----------------
    df = pd.concat([df1, df2, df3, df4], ignore_index=True)

    print("Before cleaning:", len(df))

    # ---------------- CLEAN ----------------
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # Apply text cleaning
    df["content"] = df["content"].apply(clean_text)

    # Remove very short texts (word-based, better than char)
    df = df[df["content"].apply(lambda x: len(x.split()) > 5)]

    # Optional: trim very long texts (prevents model bias)
    df["content"] = df["content"].apply(lambda x: x[:5000])

    print("After cleaning:", len(df))
    print("Class distribution:")
    print(df["label"].value_counts())

    # Save combined dataset
    df.to_csv("datasets/combined_cleaned.csv", index=False)
    print("Saved combined_cleaned.csv")

    return df


# ---------------- LOAD CLEANED ----------------
def load_cleaned():
    return pd.read_csv("datasets/combined_cleaned.csv")


# ---------------- BALANCE DATASET ----------------
def balance_dataset(df, max_per_class=None, seed=42):
    print("Balancing dataset...")

    counts = df["label"].value_counts()
    n = counts.min() if max_per_class is None else min(max_per_class, counts.min())

    df_ai = df[df["label"] == 1].sample(n=n, random_state=seed)
    df_human = df[df["label"] == 0].sample(n=n, random_state=seed)

    balanced = pd.concat([df_ai, df_human], ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=seed)

    print("Balanced distribution:")
    print(balanced["label"].value_counts())

    balanced.to_csv("datasets/final_balanced.csv", index=False)
    print("Saved final_balanced.csv")

    return balanced


# ---------------- RUN PIPELINE ----------------
if __name__ == "__main__":
    df = load_and_combine()
    balanced = balance_dataset(df)
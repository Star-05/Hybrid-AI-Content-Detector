from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import pandas as pd

# ---------------- LOAD DATA ----------------
print("Loading dataset...")

df = pd.read_csv("datasets/final_balanced.csv")

df = df[["content", "label"]]
df["content"] = df["content"].astype(str)

dataset = Dataset.from_pandas(df)

# ---------------- TOKENIZER ----------------
print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

def tokenize(example):
    return tokenizer(
        example["content"],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_token_type_ids=False
    )

dataset = dataset.map(tokenize, batched=True)

# Remove raw text column
dataset = dataset.remove_columns(["content"])

# Convert to PyTorch tensors
dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "label"
    ]
)

# ---------------- SPLIT ----------------
dataset = dataset.train_test_split(
    test_size=0.2,
    seed=42
)

print("\nTRAIN LABEL COUNTS")
print(dataset["train"].to_pandas()["label"].value_counts())

print("\nTEST LABEL COUNTS")
print(dataset["test"].to_pandas()["label"].value_counts())

# ---------------- MODEL ----------------
print("Loading model...")

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# ---------------- TRAINING ----------------
training_args = TrainingArguments(
    output_dir="models/bert_results",

    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,

    num_train_epochs=3,

    learning_rate=2e-5,
    weight_decay=0.01,

    logging_dir="logs",
    logging_steps=100,

    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

# ---------------- TRAIN ----------------
print("Training started...")
trainer.train()

# ---------------- SAVE ----------------
model.save_pretrained("models/bert_model")
tokenizer.save_pretrained("models/bert_model")

print("BERT training complete!")
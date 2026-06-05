from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score
import pandas as pd
import torch

df = pd.read_csv("datasets/final_balanced.csv")

test_df = df.sample(1000, random_state=42)

tokenizer = AutoTokenizer.from_pretrained("models/bert_model")
model = AutoModelForSequenceClassification.from_pretrained("models/bert_model")

preds = []
labels = []

for _, row in test_df.iterrows():
    text = str(row["content"])

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    pred = torch.argmax(outputs.logits, dim=1).item()

    preds.append(pred)
    labels.append(row["label"])

acc = accuracy_score(labels, preds)

print("BERT Accuracy =", acc)
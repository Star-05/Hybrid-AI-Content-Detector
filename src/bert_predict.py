from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "models/bert_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()


def predict_bert(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)

    human_prob = probs[0][0].item()
    ai_prob = probs[0][1].item()

    return {
        "score": round(ai_prob * 100),
        "human_score": round(human_prob * 100),
        "ai_score": round(ai_prob * 100),
        "label": "AI Generated" if ai_prob > 0.5 else "Human Written"
    }
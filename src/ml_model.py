# --- IMPORTS ---
import pandas as pd
import joblib
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, roc_auc_score


# --- TEXT CLEANING ---
def clean_text(text):
    text = str(text)
    text = text.replace("\n", " ")
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s.,!?']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---- TRAIN MODEL ---
def train_model():
    print("Loading dataset...")
    df = pd.read_csv("datasets/final_balanced.csv")

    df["content"] = df["content"].astype(str).apply(clean_text)

    X = df["content"]
    y = df["label"]

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Vectorizing...")
    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 4),
        stop_words="english",
        min_df=2,
        max_df=0.9,
        sublinear_tf=True
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training...")
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=2.0
    )

    model.fit(X_train_vec, y_train)

    print("Evaluating...")
    preds = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, preds)
    print("Accuracy:", accuracy)
    print(classification_report(y_test, preds))

    joblib.dump(model, "models/tfidf_model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    print("Model saved!")

    # --- GRAPHS ---
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig("models/confusion_matrix.png")
    plt.close()

    y_probs = model.predict_proba(X_test_vec)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("models/roc_curve.png")
    plt.close()

    plt.figure()
    plt.bar(["Accuracy"], [accuracy])
    plt.ylim(0, 1)
    plt.title("Model Accuracy")
    plt.savefig("models/accuracy.png")
    plt.close()

    print("AUC Score:", roc_auc)


# --- LOAD MODEL ---
tfidf_model = None
tfidf_vectorizer = None

def ensure_model_loaded():
    global tfidf_model, tfidf_vectorizer
    if tfidf_model is None:
        tfidf_model = joblib.load("models/tfidf_model.pkl")
        tfidf_vectorizer = joblib.load("models/vectorizer.pkl")


# --- PREDICTION ---
def predict(text):
    ensure_model_loaded()

    cleaned = clean_text(text)

    if len(cleaned.split()) < 15:
        return {"score": 50, "label": "Uncertain"}

    X = tfidf_vectorizer.transform([cleaned])
    prob = tfidf_model.predict_proba(X)[0][1]

    score = int(round(prob * 100))

    if score < 30:
        label = "Human Written"
    elif score > 60:
        label = "AI Generated"
    else:
        label = "Possibly AI (Uncertain)"

    return {"score": score, "label": label}


# --- EVALUATION ---
def evaluate_model():
    df = pd.read_csv("datasets/final_balanced.csv")

    X = df["content"].astype(str)
    y = df["label"]

    preds = []
    probs = []

    print("Evaluating TF-IDF Model...")

    for i, text in enumerate(X):
        if i % 100 == 0:
            print(f"Processing {i}/{len(X)}")

        result = predict(text)
        score = result["score"] / 100

        probs.append(score)
        preds.append(1 if score > 0.5 else 0)

    acc = accuracy_score(y, preds)
    auc_score = roc_auc_score(y, probs)

    print("\nFinal Accuracy:", acc)
    print("Final AUC:", auc_score)

    # --- GRAPHS ---
    cm = confusion_matrix(y, preds)
    ConfusionMatrixDisplay(cm).plot()
    plt.title("Confusion Matrix")
    plt.savefig("models/confusion_matrix.png")
    plt.close()

    fpr, tpr, _ = roc_curve(y, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("models/roc_curve.png")
    plt.close()

    plt.figure()
    plt.bar(["Accuracy"], [acc])
    plt.ylim(0, 1)
    plt.title("Model Accuracy")
    plt.savefig("models/accuracy.png")
    plt.close()

    print("Graphs saved!")


# ---- RUN ----
if __name__ == "__main__":
    # train_model()
    evaluate_model()
    # pass
from src.ml_model import predict
from src.bert_predict import predict_bert

from flask import Flask, render_template, request, jsonify
import pytesseract
from PIL import Image
import PyPDF2
import docx
from flask_cors import CORS
import traceback

# Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)
CORS(app)

# ----- FILE VALIDATION -----
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ----- HOME -----
@app.route("/")
def home():
    return render_template("index.html")


# ----- HYBRID PROCESS FUNCTION -----
def process_text(text, source="Text"):
    text = text.replace("\n", " ").strip()

    if not text:
        return {"error": "No readable text found"}

    if len(text.split()) < 15:
        return {
            "ai_score": 50,
            "human_score": 50,
            "verdict": "Uncertain",
            "explanation": [f"{source} too short for reliable prediction"]
        }

    text = text[:5000]

    # TF-IDF Prediction
    tfidf_result = predict(text)

    # BERT Prediction
    bert_result = predict_bert(text)

    # Hybrid Score
    final_ai_score = round(
        (tfidf_result["score"] + bert_result["score"]) / 2
    )

    final_human_score = 100 - final_ai_score

    if final_ai_score >= 60:
        verdict = "AI Generated"
    elif final_ai_score <= 40:
        verdict = "Human Written"
    else:
        verdict = "Possibly AI"

    return {
        "ai_score": final_ai_score,
        "human_score": final_human_score,
        "verdict": verdict,

        "tfidf_score": tfidf_result["score"],
        "bert_score": bert_result["score"],

        "explanation": [
            f"{source} + TF-IDF + DistilBERT Hybrid"
        ],

        "extracted_text": text
    }


# ----- TEXT -----
@app.route("/predict", methods=["POST"])
def predict_text():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Please enter text"})

    result = process_text(text, "Text")
    return jsonify(result)


# ----- IMAGE OCR -----
@app.route("/predict-image", methods=["POST"])
def predict_image():
    file = request.files.get("file")

    if not file or file.filename == "":
        return jsonify({"error": "No file uploaded"})

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"})

    try:
        img = Image.open(file).convert("RGB")

        text = pytesseract.image_to_string(
            img,
            config="--oem 3 --psm 6"
        )

        result = process_text(text, "OCR")
        result["filename"] = file.filename

        return jsonify(result)

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Error processing image"})


# ----- DOCUMENT -----
@app.route("/predict-doc", methods=["POST"])
def predict_doc():
    file = request.files.get("file")

    if not file or file.filename == "":
        return jsonify({"error": "No file uploaded"})

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"})

    text = ""

    try:
        if file.filename.endswith(".pdf"):

            reader = PyPDF2.PdfReader(file)

            for page in reader.pages:
                extracted = page.extract_text()

                if extracted:
                    text += extracted + " "

        elif file.filename.endswith(".docx"):

            doc = docx.Document(file)

            text = " ".join(
                [p.text for p in doc.paragraphs]
            )

        elif file.filename.endswith(".txt"):

            text = file.read().decode("utf-8")

        else:
            return jsonify({"error": "Unsupported file type"})

        result = process_text(text, "Document")
        result["filename"] = file.filename

        return jsonify(result)

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Error processing document"})


# ----- RUN -----
if __name__ == "__main__":
    app.run(debug=True)
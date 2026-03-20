# app.py
import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.models import load_model

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "trained_model.keras"

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

model = None


def get_model():
    global model

    if model is None:
        model = load_model(MODEL_PATH)

    return model


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/sign-to-english")
def sign_to_english():
    return render_template("sign-to-english.html")


@app.route("/english-to-sign")
def english_to_sign():
    return render_template("english-to-sign.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    data = payload.get("landmarks") or payload.get("input")

    if data is None:
        return jsonify({"error": "Missing 'landmarks' or 'input' array"}), 400

    if not isinstance(data, list):
        return jsonify({"error": "Prediction input must be a list"}), 400

    if len(data) != 63:
        return jsonify({"error": "Prediction input must contain exactly 63 values"}), 400

    tensor = np.array([data], dtype=np.float32)
    pred = get_model().predict(tensor, verbose=0)
    label = int(np.argmax(pred))
    confidence = float(np.max(pred))
    return jsonify({"label": label, "confidence": confidence})


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
    )

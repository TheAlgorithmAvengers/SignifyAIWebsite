# app.py
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)
model = load_model("trained_model.keras")  # or .h5


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

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    data = payload.get("landmarks") or payload.get("input")

    if data is None:
        return jsonify({"error": "Missing 'landmarks' or 'input' array"}), 400

    tensor = np.array([data], dtype=np.float32)
    pred = model.predict(tensor)
    label = int(np.argmax(pred))
    confidence = float(np.max(pred))
    return jsonify({"label": label, "confidence": confidence})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

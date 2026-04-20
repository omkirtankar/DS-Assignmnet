"""
House Price Prediction - Flask Web Application
Serves a trained ML model with a beautiful interactive UI
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import json
import os

app = Flask(__name__)

# ── Load Model & Artifacts ────────────────────────────────────
MODEL_DIR = "model"

def load_artifacts():
    """Load the trained model, scaler, and metadata. Auto-train if missing."""
    model_path = os.path.join(MODEL_DIR, "house_price_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    meta_path = os.path.join(MODEL_DIR, "metadata.json")

    # If model files are missing, train automatically
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Model files not found. Training model automatically...")
        from train_model import train_and_save_model
        train_and_save_model()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    return model, scaler, metadata

model, scaler, metadata = load_artifacts()


@app.route("/")
def index():
    """Render the main prediction page."""
    return render_template(
        "index.html",
        feature_info=metadata["feature_info"],
        feature_names=metadata["feature_names"],
        metrics=metadata["metrics"],
        importance=metadata["feature_importance"],
        dataset=metadata["dataset"],
        model_info=metadata["model"]
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction requests."""
    try:
        data = request.get_json()
        features = []

        for name in metadata["feature_names"]:
            value = float(data.get(name, 0))
            features.append(value)

        features_array = np.array([features])
        features_scaled = scaler.transform(features_array)
        prediction = model.predict(features_scaled)[0]

        # Convert from $100k units to INR (₹85 per $1)
        USD_TO_INR = 85
        price_inr = prediction * 100000 * USD_TO_INR

        # Format in Indian numbering (XX,XX,XXX)
        def format_inr(amount):
            s = f"{int(amount)}"
            if len(s) <= 3:
                return s
            last3 = s[-3:]
            remaining = s[:-3]
            groups = []
            while len(remaining) > 2:
                groups.insert(0, remaining[-2:])
                remaining = remaining[:-2]
            if remaining:
                groups.insert(0, remaining)
            return ','.join(groups) + ',' + last3

        return jsonify({
            "success": True,
            "prediction": round(prediction, 4),
            "price_formatted": f"₹{format_inr(price_inr)}",
            "price_inr": round(price_inr, 2)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)


from flask import Flask, request, jsonify
import numpy as np
import joblib
import os

from utils.audio_utils import save_base64_audio
from utils.feature_extractor import extract_voice_features

app = Flask(__name__)

API_KEY = "india-ai-voice-2026"
MODEL_PATH = "models/ml_voice_classifier.pkl"

ml_model = joblib.load(MODEL_PATH)


@app.route("/detect", methods=["POST"])
def detect_voice():
    try:
        if request.headers.get("x-api-key") != API_KEY:
            return jsonify({"error": "Invalid API key"}), 401

        data = request.get_json(force=True)
        if "audio_base64" not in data:
            return jsonify({"error": "audio_base64 missing"}), 400

        audio_path = save_base64_audio(data["audio_base64"])
        features = extract_voice_features(audio_path)

        X = np.array(list(features.values())).reshape(1, -1)
        pred = ml_model.predict(X)[0]
        prob = ml_model.predict_proba(X)[0]

        os.remove(audio_path)

        return jsonify({
            "classification": "AI-generated" if pred == 1 else "Human",
            "confidence": int(max(prob) * 100),
            "explanation": "ML-based classification using learned voice patterns"
        })

    except Exception as e:
        print("SERVER ERROR:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host="127.0.0.1", port=5000, debug=True)

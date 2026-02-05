# AI-Generated Voice Detection (Multi-Language)

## 📌 Problem Statement
Build an API-based system that determines whether a given voice sample is **AI-generated** or **human-generated**.  
Voice samples may be provided in **Tamil, English, Hindi, Malayalam, and Telugu**.  

The system must:
- Accept **Base64-encoded MP3 audio**
- Return a **structured JSON response** with:
  - classification
  - confidence score
  - explanation

---

## 🚀 Solution Overview
This project implements a **Flask-based REST API** that classifies voice samples as **Human** or **AI-generated** using a **Machine Learning model** trained on acoustic voice features.

The system is **language-agnostic**, as it relies on **audio characteristics** (pitch, MFCCs, energy) rather than textual content.

---

## 🧠 Approach

### 1️⃣ Audio Processing
- Input audio is received as **Base64-encoded MP3**
- Audio is decoded and converted to WAV format
- Audio features are extracted using `librosa`

### 2️⃣ Feature Extraction
The following features are used:
- Pitch (mean & variation)
- MFCC (voice texture)
- Energy (loudness)

### 3️⃣ Machine Learning Model
- Model: **RandomForestClassifier**
- Training data:
  - Human voice samples
  - AI-generated voice samples
- Languages included:
  - Tamil
  - English
  - Hindi
  - Malayalam
  - Telugu

The trained model is saved as:
models/ml_voice_classifier.pkl
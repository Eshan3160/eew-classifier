# EEW Intensity Classifier

**Predicting JMA seismic intensity (震度, *shindo*) from raw accelerometer data — a machine learning pipeline modeled after Japan's operational Earthquake Early Warning (EEW) system.**

🔗 **Live API:** [eew-classifier.onrender.com/docs](https://eew-classifier.onrender.com/docs)
📦 **Repo:** [github.com/Eshan3160/eew-classifier](https://github.com/Eshan3160/eew-classifier)

---

## Motivation

Japan's EEW system is one of the most advanced disaster-mitigation technologies in the world — it estimates an earthquake's eventual severity from just the first fraction of a second of P-wave data, buying critical seconds before the destructive S-wave arrives. This project is a from-scratch reproduction of that core idea: given a short window of 3-axis ground acceleration, predict the JMA shindo intensity class before the full shaking has even happened.

This project pairs with my other work, [disaster-mapping](https://github.com/Eshan3160/disaster-mapping) — a real-time evacuation routing system for Tokyo — as part of a broader interest in 防災 (*bōsai*, disaster prevention/preparedness) engineering, and in contributing to that field in Japan specifically.

## What it does

Given a 3-axis acceleration window (North-South, East-West, Up-Down — the same channels a real seismometer like Japan's K-NET stations record), the API:

1. Applies a Butterworth bandpass filter (0.5–10 Hz) to isolate the seismic signal from noise
2. Extracts 20 features per window — PGA, signal energy, and dominant frequency per axis, plus **Pd** (peak displacement) and **τc** (characteristic period), the two features central to the real Wu & Kanamori (2005) P-wave magnitude estimation method used operationally in Japan
3. Feeds those features into a LightGBM multi-class classifier, trained on 6,000 simulated earthquake events
4. Returns the predicted shindo class (0, 1, 2, 3, 4, 5-, 5+, 6-, 6+, 7) with full class probabilities

## Model performance — and an honest note on it

**Macro F1: 0.78** across 10 imbalanced classes (real earthquake catalogs are heavily skewed toward small events, which this dataset intentionally replicates rather than artificially balances).

An earlier version of this model scored 0.98 — which turned out to be a data leak: a feature derived directly from the ground-truth label had accidentally been left in the training set. I caught this during review, removed the leaking feature, and retrained. The resulting 0.78 is the honest number. The confusion matrix shows the model's errors are almost entirely between *adjacent* intensity classes (e.g. predicting shindo 3 when the truth is 4) — a pattern consistent with genuine signal-based learning rather than memorization, and one a seismologist would consider a reasonable error profile.

I'm including this in the README deliberately: catching and fixing that leak, rather than shipping the inflated number, is the part of this project I think best reflects how I actually work.

## Tech stack

| Layer | Technology |
|---|---|
| Signal processing | SciPy (Butterworth bandpass filter) |
| Feature engineering | NumPy (custom P-wave feature extraction) |
| Model | LightGBM (multi-class, class-balanced) |
| API | FastAPI + Pydantic |
| Deployment | Docker, deployed on Render |

## Architecture

```
Raw 3-axis waveform (POST /predict)
        ↓
Bandpass filter (0.5–10 Hz, zero-phase Butterworth)
        ↓
Feature extraction (PGA, energy, dominant frequency, Pd, τc — per axis + combined)
        ↓
LightGBM classifier (10-class, trained on 6,000 simulated events)
        ↓
Predicted shindo class + full probability distribution
```

## Running locally

```bash
git clone https://github.com/Eshan3160/eew-classifier.git
cd eew-classifier/api
pip install -r requirements.txt
uvicorn main:app --reload
```
Then open `http://127.0.0.1:8000/docs`.

## Running with Docker

```bash
docker build -t eew-classifier .
docker run -p 8000:8000 eew-classifier
```

## Example request

```bash
curl -X POST https://eew-classifier.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"ns": [...], "ew": [...], "ud": [...], "sampling_rate": 100}'
```

Returns:
```json
{
  "predicted_shindo": "5+",
  "predicted_class_index": 6,
  "confidence": 0.9166,
  "probabilities": {
    "0": 0.0030,
    "1": 0.0030,
    "2": 0.0030,
    "3": 0.0030,
    "4": 0.0034,
    "5-": 0.0590,
    "5+": 0.9166,
    "6-": 0.0030,
    "6+": 0.0030,
    "7": 0.0030
  }
}

```

## What's simulated vs. real

To be transparent: the training data is synthetically generated (not real seismometer recordings), since obtaining and licensing real K-NET data was outside the scope of a solo learning project. The feature extraction methodology (bandpass filtering, Pd, τc) and the target labels (JMA shindo scale) are real and match operational Japanese EEW practice. The pipeline architecture — filter → extract → classify → serve — is built to be a drop-in replacement ready for real sensor data.

## Author

**Eshan** — 2nd-year CSE & Data Science student. Built this project to demonstrate applied ML and backend engineering for a Japan-focused internship application.

GitHub: [github.com/Eshan3160](https://github.com/Eshan3160) · LinkedIn: [linkedin.com/in/eshan-shaarmaa](https://www.linkedin.com/in/eshan-shaarmaa-b76101429)

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

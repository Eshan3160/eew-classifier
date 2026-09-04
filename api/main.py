"""
api/main.py
------------
FastAPI endpoint for the Earthquake Early Warning (EEW) intensity classifier.

Takes a raw 3-axis acceleration window (NS, EW, UD), runs it through the
SAME feature extraction used in training (data/features.py), and returns
the predicted JMA shindo intensity class from the LightGBM model.

Run from inside the api/ folder:
    uvicorn main:app --reload
Then open http://127.0.0.1:8000/docs to test it interactively.
"""

import sys
from pathlib import Path

import numpy as np
import lightgbm as lgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Make data/features.py importable from here (it lives in ../data/)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_PATH = ROOT_DIR / "model" / "eew_model.txt"

sys.path.append(str(DATA_DIR))
from features import extract_features, FS as TRAIN_FS # noqa: E402

# Must match SHINDO_LABELS in model/train_model.py exactly (same order the
# model was trained on -> class index i decodes to SHINDO_LABELS[i])
SHINDO_LABELS = ["0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7"]

# The exact feature column order produced by extract_features(), i.e. the
# order build_dataset.py assembled into eew_dataset.csv (minus the target
# columns). Python dicts preserve insertion order, so this mirrors
# features.py's extract_features() precisely.
FEATURE_ORDER = []
for axis in ["ns", "ew", "ud"]:
    for stat in ["pga", "energy", "dom_freq", "pd", "tau_c"]:
        FEATURE_ORDER.append(f"{stat}_{axis}")
for stat in ["pga", "energy", "dom_freq", "pd", "tau_c"]:
    FEATURE_ORDER.append(f"{stat}_vec")

# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="EEW Intensity Classifier",
    description="Predicts JMA seismic intensity (shindo) from a 3-axis acceleration window.",
    version="1.0.0",
)

booster: lgb.Booster | None = None


@app.on_event("startup")
def load_model():
    global booster
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    booster = lgb.Booster(model_file=str(MODEL_PATH))


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class WaveformInput(BaseModel):
    ns: list[float] = Field(..., description="North-South axis acceleration samples (gal)")
    ew: list[float] = Field(..., description="East-West axis acceleration samples (gal)")
    ud: list[float] = Field(..., description="Up-Down axis acceleration samples (gal)")
    sampling_rate: int = Field(
        default=TRAIN_FS,
        description=f"Samples per second. Must match training ({TRAIN_FS} Hz) unless you know what you're doing.",
    )

    @field_validator("ew", "ud")
    @classmethod
    def same_length(cls, v, info):
        # cross-field length check happens in the endpoint (Pydantic v2
        # doesn't easily do multi-field validators here) -- kept simple.
        return v


class PredictionResponse(BaseModel):
    predicted_shindo: str
    predicted_class_index: int
    confidence: float
    probabilities: dict[str, float]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": booster is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: WaveformInput):
    if booster is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    n = len(payload.ns)
    if not (n == len(payload.ew) == len(payload.ud)):
        raise HTTPException(
            status_code=400,
            detail="ns, ew, and ud must all have the same number of samples",
        )
    if n < 10:
        raise HTTPException(status_code=400, detail="Waveform too short to extract features from")

    raw_window = np.column_stack([payload.ns, payload.ew, payload.ud])

    try:
        feats = extract_features(raw_window, fs=payload.sampling_rate)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature extraction failed: {e}")

    # Assemble feature vector in the exact order the model was trained on
    try:
        feature_vector = np.array([[feats[name] for name in FEATURE_ORDER]])
    except KeyError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Feature mismatch between features.py and API: missing {e}",
        )

    probs = booster.predict(feature_vector)[0] # shape (10,)
    pred_idx = int(np.argmax(probs))

    return PredictionResponse(
        predicted_shindo=SHINDO_LABELS[pred_idx],
        predicted_class_index=pred_idx,
        confidence=float(probs[pred_idx]),
        probabilities={label: float(p) for label, p in zip(SHINDO_LABELS, probs)},
    )
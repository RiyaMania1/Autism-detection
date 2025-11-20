import os
import json
import logging
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, conint, confloat

# -------------------------
# Logging Setup
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [api] %(levelname)s: %(message)s"
)
log = logging.getLogger("api")

# -------------------------
# FastAPI App
# -------------------------
app = FastAPI(
    title="Autism Detection API",
    version="1.0.0",
    description="Early Autism Detection System (Dynamic)"
)

# -------------------------
# CORS
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Paths
# -------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_names.json")
RESULTS_PATH = os.path.join(MODELS_DIR, "model_results.json")

# -------------------------
# Load Model, Scaler & Metadata
# -------------------------
def load_model_artifacts():
    global MODEL_PATH, SCALER_PATH, FEATURES_PATH, RESULTS_PATH
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH) or not os.path.exists(FEATURES_PATH):
        log.error("❌ Model artifacts missing. Train a dataset first.")
        return None, None, None, {}

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURES_PATH, "r") as f:
        feature_names = json.load(f)
    with open(RESULTS_PATH, "r") as f:
        model_info = json.load(f)
    log.info(f"✔ Loaded model: {MODEL_PATH}")
    return model, scaler, feature_names, model_info

model, scaler, feature_names, model_info = load_model_artifacts()

# -------------------------
# Schemas
# -------------------------
class PredictionRequest(BaseModel):
    A1_Score: conint(ge=0, le=1)
    A2_Score: conint(ge=0, le=1)
    A3_Score: conint(ge=0, le=1)
    A4_Score: conint(ge=0, le=1)
    A5_Score: conint(ge=0, le=1)
    A6_Score: conint(ge=0, le=1)
    A7_Score: conint(ge=0, le=1)
    A8_Score: conint(ge=0, le=1)
    A9_Score: conint(ge=0, le=1)
    A10_Score: conint(ge=0, le=1)
    age: confloat(gt=0)
    result: confloat(ge=0, le=10)

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probability_yes: float
    probability_no: float
    detection_rate: float
    risk_level: str

# -------------------------
# Helpers
# -------------------------
def build_input_array(req: PredictionRequest, required_features: Optional[list]):
    default_features = [
        "A1_Score","A2_Score","A3_Score","A4_Score","A5_Score",
        "A6_Score","A7_Score","A8_Score","A9_Score","A10_Score",
        "age","result"
    ]
    if not required_features:
        required_features = default_features
    mapping = req.dict()
    input_list = [float(mapping.get(f, 0)) for f in required_features]
    return np.array([input_list], dtype=float)

def get_risk_level(prob_yes: float) -> str:
    if prob_yes >= 0.8: return "High Risk"
    if prob_yes >= 0.6: return "Medium Risk"
    if prob_yes >= 0.4: return "Low Risk"
    return "Very Low Risk"

# -------------------------
# Routes
# -------------------------
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Autism Detection API is running",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.get("/model-info")
def get_model_info():
    if model is None:
        raise HTTPException(500, "Model not loaded")
    return {
        "model_name": model_info.get("best_model", "Unknown"),
        "accuracy": model_info.get("best_accuracy"),
        "features": feature_names,
        "feature_count": len(feature_names) if feature_names else 12
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if model is None or scaler is None:
        raise HTTPException(500, "Model or scaler is not loaded")
    try:
        X = build_input_array(req, feature_names)
        X_scaled = scaler.transform(X)

        pred = model.predict(X_scaled)[0]
        prob_yes = prob_no = confidence = 0.0

        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X_scaled)[0]
            prob_no, prob_yes = float(p[0]), float(p[1])
            confidence = float(max(p))
        else:
            prob_yes = 1.0 if pred == 1 else 0.0
            prob_no = 1 - prob_yes
            confidence = 1.0

        return PredictionResponse(
            prediction="YES - Autism Detected" if pred == 1 else "NO - Autism Not Detected",
            confidence=confidence,
            probability_yes=prob_yes,
            probability_no=prob_no,
            detection_rate=prob_yes * 100,
            risk_level=get_risk_level(prob_yes)
        )
    except Exception as e:
        log.exception("Prediction failed:")
        raise HTTPException(400, f"Prediction error: {str(e)}")

@app.post("/batch-predict")
def batch_predict(requests: List[PredictionRequest]):
    results = []
    for r in requests:
        try:
            resp = predict(r)
            results.append(resp.dict())
        except Exception as e:
            results.append({"error": str(e)})
    return {"predictions": results}

# -------------------------
# Run Server
# -------------------------
if __name__ == "__main__":
    import uvicorn
    log.info("🚀 Starting Autism Detection API on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

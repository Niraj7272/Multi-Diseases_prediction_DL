from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import numpy as np
import joblib

from pathlib import Path
from tensorflow.keras.models import load_model


# =========================================================
# Project Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "multi_disease_ann.h5"

LABEL_ENCODER_PATH = (
    BASE_DIR / "models" / "disease_label_encoder.pkl"
)

FEATURE_NAMES_PATH = (
    BASE_DIR / "models" / "disease_feature_names.pkl"
)


# =========================================================
# Check Required Files
# =========================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

if not LABEL_ENCODER_PATH.exists():
    raise FileNotFoundError(
        f"Label encoder not found: {LABEL_ENCODER_PATH}"
    )

if not FEATURE_NAMES_PATH.exists():
    raise FileNotFoundError(
        f"Feature names file not found: {FEATURE_NAMES_PATH}"
    )


# =========================================================
# Load Model and Preprocessing Artifacts
# =========================================================

model = load_model(MODEL_PATH)

label_encoder = joblib.load(
    LABEL_ENCODER_PATH
)

feature_names = joblib.load(
    FEATURE_NAMES_PATH
)


# =========================================================
# Verify Model Configuration
# =========================================================

print("========================================")
print("Model loaded successfully")
print("========================================")

print("Model path:", MODEL_PATH)

print(
    "Model input features:",
    model.input_shape[1]
)

print(
    "Saved feature names:",
    len(feature_names)
)

print(
    "Number of diseases:",
    len(label_encoder.classes_)
)

print("========================================")


# =========================================================
# FastAPI Application
# =========================================================

app = FastAPI(
    title="Multi-Disease Prediction API",
    description="Deep Learning based multi-disease prediction system",
    version="1.0.0"
)


# =========================================================
# Request Schema
# =========================================================

class PredictionRequest(BaseModel):

    symptoms: List[str]


# =========================================================
# Root Endpoint
# =========================================================

@app.get("/")
def home():

    return {
        "success": True,
        "message": "Multi-Disease Prediction API is running"
    }


# =========================================================
# Get Available Symptoms
# =========================================================

@app.get("/symptoms")
def get_symptoms():

    return {
        "success": True,
        "count": len(feature_names),
        "symptoms": feature_names
    }


# =========================================================
# Prediction Function
# =========================================================

def predict_disease(symptoms, top_k=5):

    # -----------------------------------------------------
    # Create empty feature vector
    # -----------------------------------------------------

    input_data = np.zeros(
        len(feature_names),
        dtype=np.float32
    )


    # -----------------------------------------------------
    # Convert selected symptoms to 1
    # -----------------------------------------------------

    for symptom in symptoms:

        index = feature_names.index(symptom)

        input_data[index] = 1


    # -----------------------------------------------------
    # Reshape input
    # -----------------------------------------------------

    input_data = input_data.reshape(1, -1)


    # -----------------------------------------------------
    # Model Prediction
    # -----------------------------------------------------

    probabilities = model.predict(
        input_data,
        verbose=0
    )[0]


    # -----------------------------------------------------
    # Get Top K Predictions
    # -----------------------------------------------------

    top_indices = np.argsort(
        probabilities
    )[-top_k:][::-1]


    # -----------------------------------------------------
    # Prepare Results
    # -----------------------------------------------------

    results = []

    for index in top_indices:

        disease = label_encoder.classes_[index]

        probability = probabilities[index]

        results.append(
            {
                "disease": disease,
                "probability": float(probability),
                "percentage": round(
                    float(probability) * 100,
                    2
                )
            }
        )


    return results


# =========================================================
# Prediction Endpoint
# =========================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    # -----------------------------------------------------
    # Remove duplicate symptoms while preserving order
    # -----------------------------------------------------

    symptoms = list(
        dict.fromkeys(request.symptoms)
    )


    # -----------------------------------------------------
    # Check if symptoms were provided
    # -----------------------------------------------------

    if not symptoms:

        return {
            "success": False,
            "message": "Please provide at least one symptom."
        }


    # -----------------------------------------------------
    # Validate Symptoms
    # -----------------------------------------------------

    invalid_symptoms = [
        symptom
        for symptom in symptoms
        if symptom not in feature_names
    ]


    if invalid_symptoms:

        return {
            "success": False,
            "message": "Some symptoms are not available.",
            "invalid_symptoms": invalid_symptoms
        }


    # -----------------------------------------------------
    # Make Prediction
    # -----------------------------------------------------

    predictions = predict_disease(
        symptoms,
        top_k=5
    )


    # -----------------------------------------------------
    # Return Response
    # -----------------------------------------------------

    return {
        "success": True,
        "selected_symptoms": symptoms,
        "predictions": predictions
    }
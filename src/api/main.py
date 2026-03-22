"""
main.py
--------
FastAPI service for PulseDirects Customer Churn Prediction.

Endpoints:
  GET  /health          — API + model health check
  GET  /model/info      — best model name, metrics, features
  POST /predict         — single customer churn prediction
  POST /predict/batch   — multiple customers churn prediction

The entire sklearn Pipeline (preprocessing + model) is loaded
from models/churn_model.pkl — no manual preprocessing needed.
"""

import json
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = Path("/app")
MODEL_PATH     = BASE_DIR / "models" / "churn_model.pkl"
BEST_INFO_PATH = BASE_DIR / "models" / "best_model_info.json"
EVAL_PATH      = BASE_DIR / "models" / "evaluation" / "evaluation_report.json"

# ── Global model state ────────────────────────────────────────────────────────
model_pipeline = None
model_info     = {}
eval_report    = {}
feature_columns = []


# ═════════════════════════════════════════════════════════════════════════════
# STARTUP — load model once when API starts
# ═════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model pipeline on startup, release on shutdown."""
    global model_pipeline, model_info, eval_report, feature_columns

    log.info("Loading churn model pipeline...")
    if not MODEL_PATH.exists():
        log.error(f"Model not found at {MODEL_PATH}")
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    model_pipeline = joblib.load(MODEL_PATH)
    log.info("Model pipeline loaded successfully.")

    if BEST_INFO_PATH.exists():
        with open(BEST_INFO_PATH) as f:
            model_info = json.load(f)
        feature_columns = model_info.get("feature_columns", [])
        log.info(f"Model: {model_info.get('best_model')} | ROC-AUC: {model_info.get('roc_auc')}")

    if EVAL_PATH.exists():
        with open(EVAL_PATH) as f:
            eval_report = json.load(f)

    yield
    log.info("Shutting down API.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PulseDirects Churn Prediction API",
    description="Real-time customer churn prediction for telecom analytics",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ═════════════════════════════════════════════════════════════════════════════

class CustomerInput(BaseModel):
    """Single customer input schema — matches feature_engineering output columns."""
    Gender              : int   = Field(..., ge=0, le=1,    description="0=Female, 1=Male")
    Senior_Citizen      : int   = Field(..., ge=0, le=1,    description="0=No, 1=Yes")
    Partner             : int   = Field(..., ge=0, le=1,    description="0=No, 1=Yes")
    Dependents          : int   = Field(..., ge=0, le=1,    description="0=No, 1=Yes")
    Tenure_Months       : int   = Field(..., ge=0,          description="Months as customer")
    Phone_Service       : int   = Field(..., ge=0, le=1,    description="0=No, 1=Yes")
    Multiple_Lines      : int   = Field(..., ge=0,          description="Encoded category")
    Internet_Service    : int   = Field(..., ge=0,          description="Encoded category")
    Online_Security     : int   = Field(..., ge=0, le=1,    description="0=No, 1=Yes")
    Online_Backup       : int   = Field(..., ge=0, le=1,    description="0=No, 1=Yes")
    Device_Protection   : int   = Field(..., ge=0, le=1,    description="0=No, 1=Yes")
    Tech_Support        : int   = Field(..., ge=0, le=1,    description="0=No, 1=Yes")
    Streaming_Tv        : int   = Field(..., ge=0, le=1,    description="0=No, 1=Yes")
    Streaming_Movies    : int   = Field(..., ge=0, le=1,    description="0=No, 1=Yes")
    Contract            : int   = Field(..., ge=0,          description="Encoded category")
    Paperless_Billing   : int   = Field(..., ge=0, le=1,    description="0=No, 1=Yes")
    Payment_Method      : int   = Field(..., ge=0,          description="Encoded category")
    Monthly_Charges     : float = Field(..., ge=0,          description="Monthly bill amount")
    Total_Charges       : float = Field(..., ge=0,          description="Total billed amount")
    Support_Ticket_Count: int   = Field(0,  ge=0,           description="Number of support tickets")
    average_charges     : float = Field(..., ge=0,          description="Total / Tenure")
    high_value_customer : int   = Field(..., ge=0, le=1,    description="Monthly > 75th percentile")

    class Config:
        json_schema_extra = {
            "example": {
                "Gender": 1,
                "Senior_Citizen": 0,
                "Partner": 1,
                "Dependents": 0,
                "Tenure_Months": 24,
                "Phone_Service": 1,
                "Multiple_Lines": 1,
                "Internet_Service": 1,
                "Online_Security": 0,
                "Online_Backup": 1,
                "Device_Protection": 0,
                "Tech_Support": 0,
                "Streaming_Tv": 1,
                "Streaming_Movies": 1,
                "Contract": 0,
                "Paperless_Billing": 1,
                "Payment_Method": 2,
                "Monthly_Charges": 85.5,
                "Total_Charges": 2052.0,
                "Support_Ticket_Count": 3,
                "average_charges": 85.5,
                "high_value_customer": 1
            }
        }


class PredictionResponse(BaseModel):
    churn_prediction   : int
    churn_probability  : float
    churn_risk_level   : str
    model_used         : str
    message            : str


class BatchPredictionResponse(BaseModel):
    total_customers    : int
    churned_count      : int
    not_churned_count  : int
    churn_rate         : float
    predictions        : List[dict]


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def get_risk_level(probability: float) -> str:
    if probability >= 0.75:
        return "High"
    elif probability >= 0.45:
        return "Medium"
    else:
        return "Low"


def customer_to_df(customer: CustomerInput) -> pd.DataFrame:
    """Convert pydantic model to dataframe matching exact training column names."""
    data = customer.model_dump()

    col_map = {
        'Gender'              : 'Gender',
        'Senior_Citizen'      : 'Senior Citizen',
        'Partner'             : 'Partner',
        'Dependents'          : 'Dependents',
        'Tenure_Months'       : 'Tenure Months',
        'Phone_Service'       : 'Phone Service',
        'Multiple_Lines'      : 'Multiple Lines',
        'Internet_Service'    : 'Internet Service',
        'Online_Security'     : 'Online Security',
        'Online_Backup'       : 'Online Backup',
        'Device_Protection'   : 'Device Protection',
        'Tech_Support'        : 'Tech Support',
        'Streaming_Tv'        : 'Streaming Tv',
        'Streaming_Movies'    : 'Streaming Movies',
        'Contract'            : 'Contract',
        'Paperless_Billing'   : 'Paperless Billing',
        'Payment_Method'      : 'Payment Method',
        'Monthly_Charges'     : 'Monthly Charges',
        'Total_Charges'       : 'Total Charges',
        'Support_Ticket_Count': 'Support Ticket Count',
        'average_charges'     : 'average_charges',
        'high_value_customer' : 'high_value_customer',
    }

    mapped = {col_map[k]: v for k, v in data.items() if k in col_map}

    # Ensure columns are in exact training order
    training_order = [
        'Gender', 'Senior Citizen', 'Partner', 'Dependents',
        'Tenure Months', 'Phone Service', 'Multiple Lines',
        'Internet Service', 'Online Security', 'Online Backup',
        'Device Protection', 'Tech Support', 'Streaming Tv',
        'Streaming Movies', 'Contract', 'Monthly Charges',
        'Total Charges', 'Payment Method', 'Paperless Billing',
        'Support Ticket Count', 'average_charges', 'high_value_customer'
    ]

    df = pd.DataFrame([mapped])
    df = df[training_order]
    return df
    """Convert pydantic model to dataframe matching training column names."""
    data = customer.model_dump()

    # Map snake_case back to training column names
    col_map = {
        'Gender'              : 'Gender',
        'Senior_Citizen'      : 'Senior Citizen',
        'Partner'             : 'Partner',
        'Dependents'          : 'Dependents',
        'Tenure_Months'       : 'Tenure Months',
        'Phone_Service'       : 'Phone Service',
        'Multiple_Lines'      : 'Multiple Lines',
        'Internet_Service'    : 'Internet Service',
        'Online_Security'     : 'Online Security',
        'Online_Backup'       : 'Online Backup',
        'Device_Protection'   : 'Device Protection',
        'Tech_Support'        : 'Tech Support',
        'Streaming_Tv'        : 'Streaming Tv',
        'Streaming_Movies'    : 'Streaming Movies',
        'Contract'            : 'Contract',
        'Paperless_Billing'   : 'Paperless Billing',
        'Payment_Method'      : 'Payment Method',
        'Monthly_Charges'     : 'Monthly Charges',
        'Total_Charges'       : 'Total Charges',
        'Support_Ticket_Count': 'Support Ticket Count',
        'average_charges'     : 'average_charges',
        'high_value_customer' : 'high_value_customer',
    }

    mapped = {col_map[k]: v for k, v in data.items() if k in col_map}
    return pd.DataFrame([mapped])


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"])
def health_check():
    """Check API and model health."""
    return {
        "status"      : "healthy",
        "model_loaded": model_pipeline is not None,
        "model_name"  : model_info.get("best_model", "unknown"),
        "api_version" : "1.0.0"
    }


@app.get("/model/info", tags=["Model"])
def get_model_info():
    """Returns best model name, metrics, and feature list."""
    if not model_info:
        raise HTTPException(status_code=404, detail="Model info not found.")
    return {
        "best_model"      : model_info.get("best_model"),
        "roc_auc"         : model_info.get("roc_auc"),
        "accuracy"        : model_info.get("accuracy"),
        "f1_score"        : model_info.get("f1_score"),
        "precision"       : model_info.get("precision"),
        "recall"          : model_info.get("recall"),
        "cv_roc_auc_mean" : model_info.get("cv_roc_auc_mean"),
        "cv_roc_auc_std"  : model_info.get("cv_roc_auc_std"),
        "feature_columns" : feature_columns,
        "confusion_matrix": eval_report.get("confusion_matrix", {}),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_single(customer: CustomerInput):
    """
    Predict churn for a single customer.
    Returns prediction, probability, and risk level.
    """
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        df          = customer_to_df(customer)
        prediction  = int(model_pipeline.predict(df)[0])
        probability = float(model_pipeline.predict_proba(df)[0][1])
        risk_level  = get_risk_level(probability)

        log.info(
            f"Prediction: {prediction} | "
            f"Probability: {probability:.4f} | "
            f"Risk: {risk_level}"
        )

        return PredictionResponse(
            churn_prediction  = prediction,
            churn_probability = round(probability, 4),
            churn_risk_level  = risk_level,
            model_used        = model_info.get("best_model", "unknown"),
            message           = (
                "Customer is likely to churn." if prediction == 1
                else "Customer is likely to stay."
            )
        )

    except Exception as e:
        log.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(customers: List[CustomerInput]):
    """
    Predict churn for multiple customers at once.
    Returns individual predictions + aggregate summary.
    """
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if len(customers) == 0:
        raise HTTPException(status_code=400, detail="No customers provided.")

    if len(customers) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Batch size exceeds limit of 1000 customers."
        )

    try:
        dfs         = [customer_to_df(c) for c in customers]
        df_batch    = pd.concat(dfs, ignore_index=True)
        predictions = model_pipeline.predict(df_batch).tolist()
        probs       = model_pipeline.predict_proba(df_batch)[:, 1].tolist()

        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probs)):
            results.append({
                "customer_index"   : i,
                "churn_prediction" : int(pred),
                "churn_probability": round(float(prob), 4),
                "churn_risk_level" : get_risk_level(float(prob)),
                "message"          : (
                    "Likely to churn." if pred == 1
                    else "Likely to stay."
                )
            })

        churned     = sum(predictions)
        not_churned = len(predictions) - churned
        churn_rate  = round(churned / len(predictions), 4)

        log.info(
            f"Batch prediction: {len(customers)} customers | "
            f"Churn rate: {churn_rate:.2%}"
        )

        return BatchPredictionResponse(
            total_customers  = len(customers),
            churned_count    = churned,
            not_churned_count= not_churned,
            churn_rate       = churn_rate,
            predictions      = results
        )

    except Exception as e:
        log.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
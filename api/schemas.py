"""Request/response models. Validation ranges follow the UCI code book and clinical limits."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EXAMPLE_PATIENT = {
    "age": 58, "sex": 1, "cp": 4, "trestbps": 140, "chol": 260, "fbs": 0, "restecg": 2,
    "thalach": 120, "exang": 1, "oldpeak": 2.4, "slope": 2, "ca": 2, "thal": 7,
}


class PatientFeatures(BaseModel):
    """One patient's 13 clinical features. `ca` and `thal` may be null (imputed)."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE_PATIENT})

    age: int = Field(..., ge=18, le=110, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="0 = female, 1 = male")
    cp: Literal[1, 2, 3, 4] = Field(
        ..., description="Chest pain: 1 typical angina, 2 atypical, 3 non-anginal, 4 asymptomatic")
    trestbps: int = Field(..., ge=60, le=250, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., ge=80, le=700, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(
        ..., description="Resting ECG: 0 normal, 1 ST-T abnormality, 2 LV hypertrophy")
    thalach: int = Field(..., ge=50, le=230, description="Maximum heart rate achieved (bpm)")
    exang: Literal[0, 1] = Field(..., description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: Literal[1, 2, 3] = Field(
        ..., description="Slope of peak exercise ST segment: 1 up, 2 flat, 3 down")
    ca: int | None = Field(None, ge=0, le=3, description="Major vessels coloured (0-3)")
    thal: Literal[3, 6, 7] | None = Field(
        None, description="Thallium test: 3 normal, 6 fixed defect, 7 reversible defect")


class Prediction(BaseModel):
    prediction: Literal[0, 1] = Field(..., description="1 = heart disease predicted")
    risk_label: Literal["high risk", "low risk"]
    probability: float = Field(..., ge=0, le=1, description="P(heart disease)")
    confidence: float = Field(..., ge=0.5, le=1, description="Probability of predicted class")


class PredictionResponse(Prediction):
    model_config = ConfigDict(protected_namespaces=())  # allow `model_*` field names

    model_version: str
    request_id: str


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patients: list[PatientFeatures] = Field(..., min_length=1, max_length=100)


class BatchResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    predictions: list[Prediction]
    count: int
    model_version: str
    request_id: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok"]
    model_name: str
    model_version: str
    model_family: str
    model_sha256: str
    uptime_seconds: float

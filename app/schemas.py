"""Pydantic request/response schemas for the inspection API."""

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Liveness plus whether weights were loaded."""

    status: str = "ok"
    model_loaded: bool


class PolicyInfo(BaseModel):
    """The frozen decision policy in use."""

    name: str
    reviewer_accuracy: float
    budget: float
    bands: Dict[str, Tuple[float, float]] = Field(description="class -> (review lower, alert upper) confidence")
    source: str


class ModelInfo(BaseModel):
    """Model identity and decision policy."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: Optional[str] = Field(None, description="sha256 prefix of the weights; null if not loaded")
    registry: Optional[str] = Field(None, description="MLflow registry alias/version if resolvable")
    model_loaded: bool
    policy: PolicyInfo


class Detection(BaseModel):
    """One detection with its maintenance action."""

    class_id: int
    class_name: str
    confidence: float
    box: Tuple[float, float, float, float] = Field(description="x1, y1, x2, y2 in pixels")
    severity: float
    action: str


class ImageResult(BaseModel):
    """Detections and overall decision for one image."""

    filename: Optional[str]
    width: int
    height: int
    detections: List[Detection]
    decision: str = Field(description="most urgent action among detections; 'ignore' if none")


class PredictResponse(BaseModel):
    """Response of POST /predict."""

    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    results: List[ImageResult]

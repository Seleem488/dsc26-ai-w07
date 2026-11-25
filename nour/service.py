
"""The shortage-risk service: validated requests, guarded prediction, health and metrics."""
from __future__ import annotations

import time
from typing import Literal

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from .fallback import rule_based_risk
from .guards import redact


class PredictRequest(BaseModel):
    """Validated prediction request."""
    model_config = ConfigDict(extra="forbid")

    branch_id: str = Field(pattern=r"^BR-\d{3}$")
    item_code: str = Field(pattern=r"^[A-Z]{2}-\d{5}$")
    weeks_of_cover: float = Field(ge=0, le=52)
    supplier_lead_time_days: int = Field(ge=0, le=180)
    orders_last_4w: int = Field(ge=0)
    supplier_fill_rate: float = Field(ge=0, le=1)
    pharmacist_note: str = Field(default="", max_length=500)


class PredictResponse(BaseModel):
    """Prediction result."""
    risk: float
    band: Literal["low", "medium", "high"]
    source: Literal["model", "fallback"]
    note_redacted: str
    model_version: str


def band_for(risk: float) -> str:
    """Map a risk score to its risk band."""
    if risk < 0.3:
        return "low"
    if risk < 0.7:
        return "medium"
    return "high"


class ShortageService:
    def __init__(
        self,
        model,
        model_version: str,
        training_ranges: dict[str, tuple[float, float]],
    ):
        """Initialize the model, metadata, training ranges, and request log."""
        self.model = model
        self.model_version = model_version
        self.training_ranges = training_ranges
        self.log: list[dict] = []

    def predict(self, req: PredictRequest) -> PredictResponse:
        """Predict shortage risk with validation guards and a safe fallback."""
        start_time = time.perf_counter()

        # Never retain or log the original pharmacist note.
        note_redacted, _ = redact(req.pharmacist_note)

        features = {
            "weeks_of_cover": req.weeks_of_cover,
            "supplier_lead_time_days": req.supplier_lead_time_days,
            "orders_last_4w": req.orders_last_4w,
            "supplier_fill_rate": req.supplier_fill_rate,
        }

        source = "model"
        reason = None
        risk = None

        # Check every configured training range before calling the model.
        out_of_range = any(
            feature_name in features
            and not (lower <= features[feature_name] <= upper)
            for feature_name, (lower, upper) in self.training_ranges.items()
        )

        if out_of_range:
            source = "fallback"
            reason = "out_of_range"
        else:
            feature_row = np.array(
                [[
                    req.weeks_of_cover,
                    req.supplier_lead_time_days,
                    req.orders_last_4w,
                    req.supplier_fill_rate,
                ]],
                dtype=float,
            )

            try:
                prediction = self.model.predict_proba(feature_row)
            except Exception:
                source = "fallback"
                reason = "model_error"
            else:
                try:
                    risk = float(prediction[0][1])

                    if not np.isfinite(risk) or not 0.0 <= risk <= 1.0:
                        risk = None
                        source = "fallback"
                        reason = "invalid_output"

                except (TypeError, ValueError, IndexError, KeyError):
                    risk = None
                    source = "fallback"
                    reason = "invalid_output"

        # Use the provided rule-based baseline whenever the model cannot be used.
        if source == "fallback":
            risk = float(
                rule_based_risk(
                    weeks_of_cover=req.weeks_of_cover,
                    supplier_fill_rate=req.supplier_fill_rate,
                    supplier_lead_time_days=req.supplier_lead_time_days,
                )
            )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Store only the redacted note in the log.
        self.log.append({
            "branch_id": req.branch_id,
            "item_code": req.item_code,
            "risk": risk,
            "source": source,
            "reason": reason,
            "note": note_redacted,
            "latency_ms": latency_ms,
        })

        return PredictResponse(
            risk=risk,
            band=band_for(risk),
            source=source,
            note_redacted=note_redacted,
            model_version=self.model_version,
        )

    def metrics(self) -> dict:
        """Return request, fallback, fallback-rate, and latency metrics."""
        requests = len(self.log)

        if requests == 0:
            return {
                "requests": 0,
                "fallbacks": 0,
                "fallback_rate": 0.0,
                "p95_latency_ms": 0.0,
            }

        fallbacks = sum(
            entry["source"] == "fallback"
            for entry in self.log
        )

        latencies = [
            entry["latency_ms"]
            for entry in self.log
        ]

        return {
            "requests": requests,
            "fallbacks": fallbacks,
            "fallback_rate": fallbacks / requests,
            "p95_latency_ms": float(np.percentile(latencies, 95)),
        }


def create_app(service: ShortageService) -> FastAPI:
    """Create the prediction API with health and metrics endpoints."""
    app = FastAPI(title="Nour Pharmacy Shortage Risk Service")

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        return service.predict(req)

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model_version": service.model_version,
        }

    @app.get("/metrics")
    def metrics() -> dict:
        return service.metrics()

    return app

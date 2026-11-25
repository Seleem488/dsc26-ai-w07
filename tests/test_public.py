
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
    """Validated prediction response."""
    risk: float
    band: Literal["low", "medium", "high"]
    source: Literal["model", "fallback"]
    note_redacted: str
    model_version: str


def band_for(risk: float) -> str:
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
        self.model = model
        self.model_version = model_version
        self.training_ranges = training_ranges
        self.log: list[dict] = []

    def predict(self, req: PredictRequest) -> PredictResponse:
        start_time = time.perf_counter()

        features = {
            "weeks_of_cover": req.weeks_of_cover,
            "supplier_lead_time_days": req.supplier_lead_time_days,
            "orders_last_4w": req.orders_last_4w,
            "supplier_fill_rate": req.supplier_fill_rate,
        }

        # Always redact the pharmacist note before logging or responding.
        note_redacted, _ = redact(req.pharmacist_note)

        source = "model"
        reason = None
        risk = None

        # Prevent inference outside the model's training ranges.
        out_of_range = any(
            name not in features
            or not lo <= features[name] <= hi
            for name, (lo, hi) in self.training_ranges.items()
        )

        if out_of_range:
            source = "fallback"
            reason = "out_of_range"
        else:
            try:
                X = np.array([[
                    req.weeks_of_cover,
                    req.supplier_lead_time_days,
                    req.orders_last_4w,
                    req.supplier_fill_rate,
                ]], dtype=float)

                probabilities = self.model.predict_proba(X)
                candidate = float(probabilities[0][1])

                if not np.isfinite(candidate) or not 0 <= candidate <= 1:
                    source = "fallback"
                    reason = "invalid_output"
                else:
                    risk = candidate

            except Exception:
                source = "fallback"
                reason = "model_error"

        if source == "fallback":
            risk = rule_based_risk(
                weeks_of_cover=req.weeks_of_cover,
                supplier_fill_rate=req.supplier_fill_rate,
                supplier_lead_time_days=req.supplier_lead_time_days,
            )

        latency_ms = (time.perf_counter() - start_time) * 1000

        self.log.append({
            "branch_id": req.branch_id,
            "item_code": req.item_code,
            "risk": float(risk),
            "source": source,
            "reason": reason,
            "note": note_redacted,
            "latency_ms": latency_ms,
        })

        return PredictResponse(
            risk=float(risk),
            band=band_for(float(risk)),
            source=source,
            note_redacted=note_redacted,
            model_version=self.model_version,
        )

    def metrics(self) -> dict:
        requests = len(self.log)
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
            "fallback_rate": fallbacks / requests if requests else 0.0,
            "p95_latency_ms": (
                float(np.percentile(latencies, 95))
                if latencies else 0.0
            ),
        }


def create_app(service: ShortageService) -> FastAPI:
    app = FastAPI(title="Nour Pharmacy Shortage Risk API")

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        return service.predict(req)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model_version": service.model_version,
        }

    @app.get("/metrics")
    def metrics():
        return service.metrics()

    return app

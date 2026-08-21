"""Request/response shapes for the inference API."""

from __future__ import annotations

import math
from typing import Dict, List

from pydantic import BaseModel, Field, field_validator

from moe.gate import EXPERT_NAMES, CLASS_NAMES


class PredictRequest(BaseModel):
    """A single observation to score.

    For Phase 1 the client supplies the per-expert probability vectors directly
    (these come bundled with sample markets). gate_feats are the standardized
    context features the gate expects for this horizon.
    """

    horizon: int = Field(..., description="One of 5, 15, 30, 60.")
    gate_feats: List[float]
    expert_probs: Dict[str, List[float]] = Field(
        ..., description="expert name -> [down, flat, up]; keys: " + ", ".join(EXPERT_NAMES)
    )

    @field_validator("gate_feats")
    @classmethod
    def validate_gate_feats(cls, value: List[float]) -> List[float]:
        if any(not math.isfinite(v) for v in value):
            raise ValueError("gate_feats must contain only finite numbers")
        return value

    @field_validator("expert_probs")
    @classmethod
    def validate_expert_probs(cls, value: Dict[str, List[float]]) -> Dict[str, List[float]]:
        normalized: Dict[str, List[float]] = {}
        for name, probs in value.items():
            if name not in EXPERT_NAMES:
                continue
            if len(probs) != 3:
                raise ValueError(f"{name} must have exactly 3 probabilities")
            if any((not math.isfinite(p)) or p < 0 for p in probs):
                raise ValueError(f"{name} probabilities must be finite and non-negative")
            total = sum(probs)
            if total <= 0:
                raise ValueError(f"{name} probabilities must sum to a positive value")
            normalized[name] = [p / total for p in probs]
        if not normalized:
            raise ValueError("expert_probs must include at least one known expert")
        return normalized


class PredictResponse(BaseModel):
    horizon: int
    prediction: str = Field(..., description="One of: " + ", ".join(CLASS_NAMES))
    probabilities: Dict[str, float]
    gate_weights: Dict[str, float]


class Market(BaseModel):
    """A bundled sample market the public can run the model on."""

    id: str
    title: str
    category: str
    description: str


class MarketDetail(Market):
    """Full bundled sample payload used by the frontend prediction button."""

    horizon: int
    gate_feats: List[float] = Field(
        default_factory=list,
        description="Standardized gate features. Empty means train-fold mean context.",
    )
    expert_probs: Dict[str, List[float]]

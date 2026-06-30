"""Request/response shapes for the inference API."""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field

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

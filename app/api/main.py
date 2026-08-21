"""FastAPI inference service for the Kalshi jump-detection demo.

Endpoints (Phase 1):
    GET  /health           liveness
    GET  /markets          list bundled sample markets (e.g. World Cup)
    GET  /markets/{id}      one market with its precomputed expert probabilities
    POST /predict          run the MoE gate on a supplied observation

Run locally (once checkpoints are in MODEL_DIR):
    uvicorn main:app --reload
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from moe.gate import EXPERT_NAMES, N_CLASSES
from moe.predictor import HORIZONS, MODEL_DIR, checkpoint_status, predict
from schemas import Market, MarketDetail, PredictRequest, PredictResponse

DATA_DIR = Path(__file__).parent / "data" / "sample_markets"

app = FastAPI(title="Kalshi Jump Detection — Demo API", version="0.1.0")

# The web frontend calls this from the browser. Set ALLOWED_ORIGINS to a
# comma-separated list (e.g. the deployed Vercel origin) in production;
# defaults to "*" for local development.
_origins = os.environ.get("ALLOWED_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _origins == "*" else [o.strip() for o in _origins.split(",")],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _load_market(market_id: str) -> dict:
    path = DATA_DIR / f"{market_id}.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Unknown market: {market_id}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid bundled market JSON: {market_id}") from e


@app.get("/health")
def health():
    status = checkpoint_status()
    return {
        "status": "ok",
        "horizons": status,
        "models_loaded": any(status.values()),
        "model_dir": str(MODEL_DIR),
        "sample_markets_loaded": DATA_DIR.is_dir(),
    }


@app.get("/markets", response_model=list[Market])
def list_markets():
    markets = []
    for f in sorted(DATA_DIR.glob("*.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        markets.append(Market(**{k: m[k] for k in ("id", "title", "category", "description")}))
    return markets


@app.get("/markets/{market_id}", response_model=MarketDetail)
def get_market(market_id: str):
    return _load_market(market_id)


@app.post("/predict", response_model=PredictResponse)
def run_prediction(req: PredictRequest):
    if req.horizon not in HORIZONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported horizon {req.horizon}. Must be one of {HORIZONS}.",
        )

    # Assemble expert tensor + availability mask in canonical expert order.
    expert_probs = np.full((len(EXPERT_NAMES), N_CLASSES), 1.0 / N_CLASSES, dtype=np.float32)
    expert_mask = np.zeros(len(EXPERT_NAMES), dtype=np.float32)
    for i, name in enumerate(EXPERT_NAMES):
        vec = req.expert_probs.get(name)
        if vec is not None:
            expert_probs[i] = vec
            expert_mask[i] = 1.0

    if not expert_mask.any():
        raise HTTPException(
            status_code=400,
            detail=f"No known experts in expert_probs. Expected keys from: {EXPERT_NAMES}.",
        )

    try:
        result = predict(req.horizon, req.gate_feats, expert_probs, expert_mask)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except (RuntimeError, ValueError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Bad input for horizon {req.horizon} (check gate_feats length): {e}",
        ) from e

    return PredictResponse(**result)

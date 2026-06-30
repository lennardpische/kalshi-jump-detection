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
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from moe.gate import EXPERT_NAMES, N_CLASSES
from moe.predictor import predict
from schemas import Market, PredictRequest, PredictResponse

DATA_DIR = Path(__file__).parent / "data" / "sample_markets"

app = FastAPI(title="Kalshi Jump Detection — Demo API", version="0.1.0")

# The web frontend (Vercel) calls this from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: lock to the deployed web origin before launch
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _load_market(market_id: str) -> dict:
    path = DATA_DIR / f"{market_id}.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Unknown market: {market_id}")
    return json.loads(path.read_text())


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/markets", response_model=list[Market])
def list_markets():
    markets = []
    for f in sorted(DATA_DIR.glob("*.json")):
        m = json.loads(f.read_text())
        markets.append(Market(**{k: m[k] for k in ("id", "title", "category", "description")}))
    return markets


@app.get("/markets/{market_id}")
def get_market(market_id: str):
    return _load_market(market_id)


@app.post("/predict", response_model=PredictResponse)
def run_prediction(req: PredictRequest):
    # Assemble expert tensor + availability mask in canonical expert order.
    expert_probs = np.full((len(EXPERT_NAMES), N_CLASSES), 1.0 / N_CLASSES, dtype=np.float32)
    expert_mask = np.zeros(len(EXPERT_NAMES), dtype=np.float32)
    for i, name in enumerate(EXPERT_NAMES):
        vec = req.expert_probs.get(name)
        if vec is not None:
            expert_probs[i] = vec
            expert_mask[i] = 1.0

    result = predict(req.horizon, req.gate_feats, expert_probs, expert_mask)
    return PredictResponse(**result)

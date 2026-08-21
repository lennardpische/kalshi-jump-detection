"""Loads MoE gate checkpoints and runs inference.

Checkpoints are looked up under MODEL_DIR (env var; defaults to
app/api/data/models), one per horizon: moe_gate_5m.pt, moe_gate_15m.pt,
moe_gate_30m.pt, moe_gate_60m.pt.

This module is intentionally minimal — it is the seam where the real .pt files
get wired in. It is NOT runnable until the checkpoints are present.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .gate import MoEGate, EXPERT_NAMES, CLASS_NAMES, N_EXPERTS, N_CLASSES

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "data" / "models"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR)).expanduser()
HORIZONS = [5, 15, 30, 60]


def _checkpoint_path(horizon: int) -> Path:
    return MODEL_DIR / f"moe_gate_{horizon}m.pt"


def checkpoint_status() -> dict[int, bool]:
    """Which horizons have a checkpoint file present under MODEL_DIR."""
    return {h: _checkpoint_path(h).is_file() for h in HORIZONS}


@lru_cache(maxsize=len(HORIZONS))
def _load(horizon: int):
    """Load and cache the gate + its normalization stats for one horizon."""
    if horizon not in HORIZONS:
        raise ValueError(f"Unsupported horizon {horizon}. Must be one of {HORIZONS}.")

    ckpt_path = _checkpoint_path(horizon)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint {ckpt_path}. Set MODEL_DIR to the folder "
            "holding moe_gate_{H}m.pt (download from the project Drive)."
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = MoEGate(n_features=len(ckpt["gate_feat_cols"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def predict(horizon: int, gate_feats, expert_probs, expert_mask):
    """Run the gate for a single trade/observation.

    Args:
        gate_feats:  (F,) already standardized with the checkpoint's stats.
        expert_probs: (6, 3) per-expert 3-class probability vectors.
        expert_mask:  (6,) 1 where an expert is available, else 0.

    Returns dict with class probabilities, predicted label, and gate weights.
    """
    model, ckpt = _load(horizon)
    n_features = len(ckpt["gate_feat_cols"])

    gate_feats = np.asarray(gate_feats, dtype=np.float32)
    if gate_feats.size == 0:
        # Empty means "use train-fold mean context" because the gate features are
        # standardized in the checkpoint convention.
        gate_feats = np.zeros(n_features, dtype=np.float32)
    if gate_feats.shape != (n_features,):
        raise ValueError(f"expected {n_features} gate features, got {gate_feats.size}")

    expert_probs = np.asarray(expert_probs, dtype=np.float32)
    expert_mask = np.asarray(expert_mask, dtype=np.float32)
    if expert_probs.shape != (N_EXPERTS, N_CLASSES):
        raise ValueError(f"expected expert_probs shape {(N_EXPERTS, N_CLASSES)}, got {expert_probs.shape}")
    if expert_mask.shape != (N_EXPERTS,):
        raise ValueError(f"expected expert_mask shape {(N_EXPERTS,)}, got {expert_mask.shape}")
    if not np.isfinite(gate_feats).all():
        raise ValueError("gate_feats contains NaN or infinite values")
    if not np.isfinite(expert_probs).all():
        raise ValueError("expert_probs contains NaN or infinite values")
    if not np.isfinite(expert_mask).all():
        raise ValueError("expert_mask contains NaN or infinite values")

    with torch.no_grad():
        gf = torch.tensor(gate_feats).unsqueeze(0)
        ep = torch.tensor(expert_probs).unsqueeze(0)
        em = torch.tensor(expert_mask).unsqueeze(0)
        final_probs, gate_weights = model(gf, ep, em)

    probs = final_probs.squeeze(0).numpy()
    weights = gate_weights.squeeze(0).numpy()
    return {
        "horizon": horizon,
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(N_CLASSES)},
        "prediction": CLASS_NAMES[int(probs.argmax())],
        "gate_weights": {EXPERT_NAMES[i]: float(weights[i]) for i in range(N_EXPERTS)},
    }

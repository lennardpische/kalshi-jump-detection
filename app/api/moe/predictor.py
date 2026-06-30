"""Loads MoE gate checkpoints and runs inference.

Checkpoints are looked up under MODEL_DIR (env var; defaults to ./data/models),
one per horizon: moe_gate_5m.pt, moe_gate_15m.pt, moe_gate_30m.pt, moe_gate_60m.pt.

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

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "data/models"))
HORIZONS = [5, 15, 30, 60]


@lru_cache(maxsize=len(HORIZONS))
def _load(horizon: int):
    """Load and cache the gate + its normalization stats for one horizon."""
    ckpt_path = MODEL_DIR / f"moe_gate_{horizon}m.pt"
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
    model, _ = _load(horizon)
    with torch.no_grad():
        gf = torch.tensor(np.asarray(gate_feats, dtype=np.float32)).unsqueeze(0)
        ep = torch.tensor(np.asarray(expert_probs, dtype=np.float32)).unsqueeze(0)
        em = torch.tensor(np.asarray(expert_mask, dtype=np.float32)).unsqueeze(0)
        final_probs, gate_weights = model(gf, ep, em)

    probs = final_probs.squeeze(0).numpy()
    weights = gate_weights.squeeze(0).numpy()
    return {
        "horizon": horizon,
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(N_CLASSES)},
        "prediction": CLASS_NAMES[int(probs.argmax())],
        "gate_weights": {EXPERT_NAMES[i]: float(weights[i]) for i in range(N_EXPERTS)},
    }

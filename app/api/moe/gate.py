"""MoE gating network — inference-only copy.

This is the same `MoEGate` architecture trained in `research/main.py`, lifted
out so the serving layer can load a checkpoint without importing the whole
training script. Keep the architecture in sync with the research code.

A checkpoint (`moe_gate_{H}m.pt`) is a dict with:
    model_state_dict, gate_feat_cols, gate_mean, gate_std, horizon, ...
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_EXPERTS = 6
N_CLASSES = 3
EXPERT_NAMES = ["lgbm", "lstm", "mamba", "moirai", "ftt", "ctts"]
CLASS_NAMES = ["down", "flat", "up"]  # maps to label -1, 0, +1


class MoEGate(nn.Module):
    """Tabular MoE gating network.

    Inputs: (gate_features, expert_probs, expert_mask).
    Output: (final 3-class probabilities, per-expert gate weights).
    """

    def __init__(self, n_features, hidden_dims=(192, 96, 48),
                 n_experts=N_EXPERTS, dropout=0.3):
        super().__init__()
        self.n_experts = n_experts

        layers = []
        prev = n_features
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h),
                       nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.gate_head = nn.Linear(prev, n_experts)

    def forward(self, gate_feats, expert_probs, expert_mask):
        h = self.backbone(gate_feats)
        gate_logits = self.gate_head(h)
        gate_logits = gate_logits.masked_fill(expert_mask == 0, -1e9)
        gate_weights = F.softmax(gate_logits, dim=-1)
        final_probs = (gate_weights.unsqueeze(-1) * expert_probs).sum(dim=1)
        final_probs = final_probs.clamp(min=1e-7, max=1 - 1e-7)
        return final_probs, gate_weights

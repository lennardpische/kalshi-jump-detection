"""Expert interface — the Phase 2 seam.

The MoE gate consumes 3-class probability vectors from 6 base models. Today the
demo serves *precomputed* expert probabilities bundled with each sample market
(see app/api/data/sample_markets/). To accept truly new raw trades, each expert
below must be implemented to turn a trade window into a (3,) probability vector.

This is deliberately just an interface + stub so the design is reviewable. None
of these are wired up yet.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Expert(ABC):
    name: str

    @abstractmethod
    def predict_proba(self, trade_window, horizon: int) -> np.ndarray:
        """Return a (3,) [down, flat, up] probability vector for one horizon."""
        raise NotImplementedError


# Phase 2: implement one class per base model, loading its checkpoint.
#   LGBMExpert    -> data/models/lgbm_*.txt
#   LSTMExpert    -> data/models/lstm_*.pt
#   MambaExpert   -> data/models/mamba_best.pt
#   MoiraiExpert  -> data/models/moirai_classifier_best.pt
#   FTTExpert     -> data/models/ftt_*.pt
#   CTTSExpert    -> data/models/ctts_*.pt
#
# The training/inference logic for each lives in the matching research notebook.

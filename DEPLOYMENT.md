# Deployment plan

Goal: a free public demo where visitors pick a market, feed in trades, and get a
down / flat / up prediction from the trained Mixture-of-Experts model.

## Architecture

```
                 ┌─────────────────────┐         ┌──────────────────────────┐
   visitor  ───▶ │  Next.js  (app/web) │  HTTPS  │  FastAPI  (app/api)      │
                 │  Vercel free tier   │ ──────▶ │  HF Spaces / Render free │
                 │  Kalshi-style UI    │         │  loads moe_gate_*.pt     │
                 └─────────────────────┘         └──────────────────────────┘
```

Why split: PyTorch + checkpoints don't fit Vercel serverless limits, so the
model runs on a small always-/often-on Python host while the UI stays on Vercel.

## Two phases

**Phase 1 — ship now (free, CPU).** Bundle a handful of sample markets with
*precomputed* expert probabilities (`app/api/data/sample_markets/*.json`). The
live MoE gate combines them and returns a prediction. Visitors can tweak inputs.
Only needs the four `moe_gate_{5,15,30,60}m.pt` checkpoints.

**Phase 2 — real new trades.** Implement the 6 base models behind
`app/api/experts/base.py` so a raw trade window → 6 probability vectors → gate.
This is the heavy path and depends on the open question below.

## Open question: do we have the trained gate?

This was previously written backwards. Current status:

- **The 6 base-model checkpoints exist** (LightGBM, LSTM, Mamba, Moirai,
  FT-Transformer, CTTS — on Drive).
- **The gate checkpoints (`moe_gate_{5,15,30,60}m.pt`) do not exist yet.**
  `research/main.py` trains and saves them itself (see its `torch.save(...)`
  around the `ckpt_path = .../moe_gate_{HORIZON}m.pt` line) — it does **not**
  need the 6 base-model checkpoints directly. It reads a single pre-merged
  `data/moe_data.parquet` that already has each base model's 3-class
  probability columns baked in (72 columns) plus labels, per the shared
  artifact bundle linked at the top of that script.

So producing the missing Phase 1 checkpoints is: confirm `moe_data.parquet` is
available under `PROJECT_ROOT/data/`, then run `research/main.py` once per
horizon (or the multi-horizon loop) so it writes `moe_gate_{H}m.pt` into
`MODEL_DIR`. No inference code for the 6 experts needs to be written for this —
that's still Phase 2's job (`app/api/experts/base.py`), needed only for scoring
genuinely new/live trades rather than the bundled sample markets.

## Steps to go live (Phase 1)

1. Confirm `data/moe_data.parquet` is available and run `research/main.py`
   (per horizon) to produce `moe_gate_{5,15,30,60}m.pt`.
2. Point `app/api` at that `MODEL_DIR` and confirm `/predict` serves real
   predictions (`/health` reports `models_loaded: true`).
3. Replace placeholder probabilities in `sample_markets/*.json` with real values.
4. Deploy `app/api` to HF Spaces or Render using the included `Dockerfile`.
5. Deploy `app/web` to Vercel; set `NEXT_PUBLIC_API_URL` to the API URL.
6. Lock CORS in `app/api/main.py` (`ALLOWED_ORIGINS`) to the Vercel origin.

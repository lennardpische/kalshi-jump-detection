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

## Checkpoint status

Current status:

- **The 6 base-model checkpoints exist** (LightGBM, LSTM, Mamba, Moirai,
  FT-Transformer, CTTS — on Drive).
- **The gate checkpoints (`moe_gate_{5,15,30,60}m.pt`) also exist** in the
  project Drive folder. They are intentionally not committed because they are
  large.

For Phase 1, download or mount only the four gate checkpoints and set
`MODEL_DIR` to that folder. The six base-model checkpoints are only needed for
Phase 2, when the API accepts genuinely new raw trades instead of bundled
precomputed expert probabilities.

## Steps to go live (Phase 1)

1. Download or mount `moe_gate_{5,15,30,60}m.pt` outside git.
2. Point `app/api` at that folder with `MODEL_DIR` and confirm `/predict` serves real
   predictions (`/health` reports `models_loaded: true`).
3. Replace placeholder probabilities in `sample_markets/*.json` with real
   values from representative rows if needed.
4. Deploy `app/api` to HF Spaces or Render using the included `Dockerfile`.
5. Deploy `app/web` to Vercel; set `NEXT_PUBLIC_API_URL` to the API URL.
6. Lock CORS in `app/api/main.py` (`ALLOWED_ORIGINS`) to the Vercel origin.

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

## Open question: do we have all 6 base-model checkpoints?

The gate alone **cannot** read raw trades — it only re-weights expert outputs.
True "feed new World Cup trades" needs all six experts runnable:

| Expert | Checkpoint | On Drive? |
|---|---|---|
| Mamba | `mamba_best.pt` | yes |
| Moirai | `moirai_classifier_best.pt` | yes |
| LightGBM | `lgbm_*` | **confirm** |
| LSTM | `lstm_*` | **confirm** |
| FT-Transformer | `ftt_*` | **confirm** |
| CTTS | `ctts_*` | **confirm** |

If some are missing, Phase 1 (sample markets) is still a complete, honest demo;
Phase 2 unlocks once the checkpoints + per-expert inference code are available.

## Steps to go live (Phase 1)

1. Place `moe_gate_*.pt` in a `MODEL_DIR` and confirm `app/api` serves `/predict`.
2. Replace placeholder probabilities in `sample_markets/*.json` with real values.
3. Deploy `app/api` to HF Spaces or Render (small Dockerfile around uvicorn).
4. Deploy `app/web` to Vercel; set `NEXT_PUBLIC_API_URL` to the API URL.
5. Lock CORS in `app/api/main.py` to the Vercel origin.

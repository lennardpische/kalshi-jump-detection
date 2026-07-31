# Inference API

FastAPI service that runs the trained MoE **gate** and serves predictions to the
web frontend.

```
api/
├── main.py                 # FastAPI app + routes
├── schemas.py              # request/response models
├── moe/
│   ├── gate.py             # MoEGate architecture (mirror of research/main.py)
│   └── predictor.py        # loads moe_gate_{H}m.pt, runs inference
├── experts/
│   └── base.py             # Phase 2 interface for the 6 base models
└── data/sample_markets/    # bundled demo markets w/ precomputed expert probs
```

## Checkpoints

Not committed. Download `moe_gate_{5,15,30,60}m.pt` from the project Drive and
point the service at them:

```bash
export MODEL_DIR=/path/to/models
```

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Until the checkpoints are present, `/predict` returns a `503` with a clear
message; `/health`, `/markets`, and `/markets/{id}` all work without them.

## Deploy (free tier)

PyTorch is too large for Vercel serverless, so host this on **Hugging Face
Spaces** or **Render free tier** using the included `Dockerfile` (CPU-only
torch, small image):

```bash
docker build -t kalshi-api .
docker run -p 8000:8000 \
  -e ALLOWED_ORIGINS=https://your-web-app.vercel.app \
  -v /path/to/checkpoints:/app/data/models \
  kalshi-api
```

The web app on Vercel calls it via `NEXT_PUBLIC_API_URL`.

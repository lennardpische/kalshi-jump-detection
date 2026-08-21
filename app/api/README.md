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

Not committed because they are large. Download the trained gate checkpoints
from the project Drive and point the service at that folder:

```bash
export MODEL_DIR=/path/to/models
```

Expected filenames:

```text
moe_gate_5m.pt
moe_gate_15m.pt
moe_gate_30m.pt
moe_gate_60m.pt
```

You can also place them in `app/api/data/models/` for local development; that
directory is gitignored.

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Until the checkpoints are present, `/predict` returns a `503` with a clear
message; `/health`, `/markets`, and `/markets/{id}` all work without them. The
bundled sample markets use empty `gate_feats`, which the API interprets as the
train-fold mean context stored by each checkpoint.

## Export real sample markets

If you have the large `moe_data.parquet`, do not commit or deploy it. Extract a
few tiny JSON rows instead:

```bash
cd app/api
python3 -m pip install pyarrow
python3 scripts/export_sample_markets.py \
  --parquet /path/to/moe_data.parquet \
  --model-dir ../../weights/drive-download-20260821T201649Z-1-001 \
  --out-dir data/sample_markets \
  --rows-per-horizon 2
```

The script streams the parquet, reads only the needed columns, uses the
checkpoint metadata to standardize `gate_feats`, and writes API-ready JSON
payloads with real `expert_probs`.

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

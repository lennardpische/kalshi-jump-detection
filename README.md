# Finding an Edge in Prediction Markets

ML-based jump detection on Kalshi prediction markets, plus a public demo.

This repo has two halves:

| Folder | What it is |
|---|---|
| [`research/`](research/) | The original study — Mixture-of-Experts pipeline, training/eval code (`main.py`), notebooks, and the paper. See [`research/README.md`](research/README.md). |
| [`app/`](app/) | A public, deployable demo of the model. **In progress — scaffold only.** |

## The demo (`app/`)

```
app/
├── web/   Next.js frontend  → Vercel (free tier), Kalshi-inspired UI
└── api/   FastAPI inference  → Hugging Face Spaces / Render (free tier)
```

The web app lists sample markets (e.g. World Cup), the visitor runs the model,
and the API returns a down/flat/up prediction with the per-expert gate weights.

See [`DEPLOYMENT.md`](DEPLOYMENT.md) for the architecture, the two-phase plan,
and the open question about base-model checkpoints.

> Status: reviewable scaffold. Reorganized layout + minimal base code; not yet
> runnable end-to-end (needs the trained checkpoints wired in).

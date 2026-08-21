"""Export small FastAPI demo market JSONs from the large MoE parquet.

The 12GB+ `moe_data.parquet` should not be bundled with the app. This script
streams a few valid rows, reads only the columns needed by the gate, and writes
small JSON payloads under app/api/data/sample_markets/.

Example:
    python scripts/export_sample_markets.py \
      --parquet /path/to/moe_data.parquet \
      --model-dir ../../weights/drive-download-20260821T201649Z-1-001 \
      --out-dir data/sample_markets \
      --rows-per-horizon 2
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import pyarrow.parquet as pq
except ImportError as exc:
    raise SystemExit(
        "Missing pyarrow. Install it in your local environment with: "
        "python3 -m pip install pyarrow"
    ) from exc


EXPERT_NAMES = ["lgbm", "lstm", "mamba", "moirai", "ftt", "ctts"]
HORIZONS = [5, 15, 30, 60]


def expert_prob_cols(model: str, horizon: int) -> list[str]:
    """Return [down, flat, up] probability columns for model + horizon."""
    if model in ["lgbm", "lstm", "mamba", "moirai"]:
        suffix = f"{horizon}minutes_{model}"
    else:
        suffix = f"{horizon}m_{model}"
    return [
        f"probability_down_{suffix}",
        f"probability_no_jump_{suffix}",
        f"probability_up_{suffix}",
    ]


def checkpoint_path(model_dir: Path, horizon: int) -> Path:
    return model_dir / f"moe_gate_{horizon}m.pt"


def load_checkpoint(model_dir: Path, horizon: int) -> dict[str, Any]:
    path = checkpoint_path(model_dir, horizon)
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def safe_slug(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value or "market"


def json_value(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def row_probs(row: dict[str, Any], horizon: int) -> dict[str, list[float]] | None:
    expert_probs: dict[str, list[float]] = {}
    for expert in EXPERT_NAMES:
        vals = [finite_float(row[col]) for col in expert_prob_cols(expert, horizon)]
        if any(v is None for v in vals):
            return None
        total = sum(vals)
        if total <= 0:
            return None
        expert_probs[expert] = [float(v / total) for v in vals]
    return expert_probs


def standardized_gate_feats(row: dict[str, Any], ckpt: dict[str, Any]) -> list[float]:
    cols = ckpt["gate_feat_cols"]
    mean = np.asarray(ckpt["gate_mean"], dtype=np.float32)
    std = np.asarray(ckpt["gate_std"], dtype=np.float32)

    values = np.empty(len(cols), dtype=np.float32)
    for idx, col in enumerate(cols):
        value = finite_float(row.get(col))
        if value is None and col.startswith("rolling_acc_"):
            value = 1.0 / 3.0
        values[idx] = np.nan if value is None else value

    feats = (values - mean) / std
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return [float(v) for v in feats]


def required_columns(ckpt: dict[str, Any], horizon: int, available: set[str]) -> list[str]:
    cols = list(ckpt["gate_feat_cols"])
    for expert in EXPERT_NAMES:
        cols.extend(expert_prob_cols(expert, horizon))
    for optional in ["ticker", "created_time", "split", f"jump3_{horizon}m"]:
        if optional in available:
            cols.append(optional)

    missing = sorted(set(cols) - available)
    if missing:
        sample = ", ".join(missing[:10])
        extra = " ..." if len(missing) > 10 else ""
        raise ValueError(f"Parquet is missing required columns for {horizon}m: {sample}{extra}")

    return sorted(set(cols))


def export_for_horizon(
    parquet_path: Path,
    model_dir: Path,
    out_dir: Path,
    horizon: int,
    rows_per_horizon: int,
    batch_size: int,
) -> int:
    ckpt = load_checkpoint(model_dir, horizon)
    parquet = pq.ParquetFile(parquet_path)
    columns = required_columns(ckpt, horizon, set(parquet.schema.names))

    written = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        rows = batch.to_pylist()
        for row in rows:
            expert_probs = row_probs(row, horizon)
            if expert_probs is None:
                continue

            ticker = str(row.get("ticker") or f"sample-{horizon}m")
            created_time = json_value(row.get("created_time"))
            label = json_value(row.get(f"jump3_{horizon}m"))
            slug = safe_slug(ticker)

            payload = {
                "id": f"{slug}-{horizon}m-{written + 1}",
                "title": f"{ticker} historical trade window",
                "category": "Historical sample",
                "description": (
                    f"Real MoE row for a {horizon}-minute prediction"
                    + (f" at {created_time}" if created_time else ".")
                ),
                "horizon": horizon,
                "gate_feats": standardized_gate_feats(row, ckpt),
                "expert_probs": expert_probs,
                "_source": {
                    "ticker": ticker,
                    "created_time": created_time,
                    "label": label,
                },
            }

            out_path = out_dir / f"{payload['id']}.json"
            out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            written += 1
            if written >= rows_per_horizon:
                return written

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True, type=Path, help="Path to moe_data.parquet")
    parser.add_argument("--model-dir", required=True, type=Path, help="Folder with moe_gate_{H}m.pt")
    parser.add_argument("--out-dir", default=Path("data/sample_markets"), type=Path)
    parser.add_argument("--rows-per-horizon", default=1, type=int)
    parser.add_argument("--batch-size", default=2048, type=int)
    parser.add_argument("--horizons", nargs="+", default=HORIZONS, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for horizon in args.horizons:
        if horizon not in HORIZONS:
            raise ValueError(f"Unsupported horizon {horizon}; expected one of {HORIZONS}")
        count = export_for_horizon(
            parquet_path=args.parquet,
            model_dir=args.model_dir,
            out_dir=args.out_dir,
            horizon=horizon,
            rows_per_horizon=args.rows_per_horizon,
            batch_size=args.batch_size,
        )
        total += count
        print(f"wrote {count} sample JSON(s) for {horizon}m")

    print(f"done: wrote {total} file(s) to {args.out_dir}")


if __name__ == "__main__":
    main()

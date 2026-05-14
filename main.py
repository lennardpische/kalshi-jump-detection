#!/usr/bin/env python
# coding: utf-8

# # Kalshi Prediction Markets: Mixture of Experts Pipeline
#
# Publication entry point for the Mixture of Experts (MoE) pipeline used to
# predict short-horizon price movements on Kalshi prediction markets.
#
# Authors: Andres Blanco Prada, Gianluca Pisa, Lennard Pische,
# Vishwesh Venkatramani, Moritz Wassermann
#
# Task: Given a trade observation, predict whether the contract `yes_price`
# will move up (+1), remain flat (0), or move down (-1) over a forward window
# of 5, 15, 30, or 60 minutes.
#
# Approach: Six independently trained base models (LightGBM, LSTM, Mamba,
# Moirai, FT-Transformer, and Conformer-Tiny Time-Series) each output a
# 3-class probability vector for a given horizon. A learned gating network
# combines those expert predictions with weights conditioned on trade-level
# context.
#
# Data: The publication artifact bundle contains `data/moe_data.parquet`,
# model checkpoints in `data/models/`, optional caches in `moe_cache/`, and
# `requirements.txt`. Set PROJECT_ROOT to the folder that contains those paths.
#
# Example:
#   PROJECT_ROOT=/path/to/project python main.py
#
# Shared artifact folder:
#   https://drive.google.com/drive/folders/1bwPVZxxEhNfPodRlXFBZV-RKXiuSnlkq?usp=sharing

# ## Runtime setup

import os
import sys
import subprocess
from pathlib import Path

try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    drive.mount("/content/drive", force_remount=False)

DEFAULT_PROJECT_ROOT = (
    Path("/content/drive/MyDrive/project")
    if IN_COLAB
    else Path(__file__).resolve().parent
)
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", DEFAULT_PROJECT_ROOT)).expanduser().resolve()

DATA_ROOT = str(PROJECT_ROOT / "data")
MODEL_DIR = os.path.join(DATA_ROOT, "models")
MOE_CACHE = os.path.join(str(PROJECT_ROOT), "moe_cache")

_data = PROJECT_ROOT / "data"
if not _data.is_dir():
    raise FileNotFoundError(
        f"Missing {_data}. Set PROJECT_ROOT to the artifact folder that "
        "contains `data`, `notebooks`, and `requirements.txt`."
    )
os.makedirs(MODEL_DIR, exist_ok=True)

_req = PROJECT_ROOT / "requirements.txt"
if _req.is_file() and os.environ.get("INSTALL_REQUIREMENTS") == "1":
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(_req)])

print("PROJECT_ROOT:", PROJECT_ROOT)
print("DATA_ROOT:   ", DATA_ROOT)
print("MODEL_DIR:   ", MODEL_DIR)
print("MOE_CACHE:   ", MOE_CACHE)


# ## 1. Data Loading
# 
# The pre-processed dataset `moe_data.parquet` contains one row per trade with 135 columns:
# 
# - **Trade features (~40 columns):** raw trade attributes (`yes_price`, `no_price`, `count`, `taker_side`, `log_volume`) and rolling statistics computed over recent trades within the same ticker.
# - **Base-model probability outputs (72 columns):** each base model's predicted 3-class probability vector (`probability_down_*`, `probability_no_jump_*`, `probability_up_*`) for each of the four prediction horizons.
# - **Labels (4 columns):** `jump3_{H}m` for H in {5, 15, 30, 60}, encoding price direction as -1, 0, or +1.
# 
# The data covers January through December 2025. Paths are under `DATA_ROOT`,
# set in the runtime setup block above.

import os
import pandas as pd

df = pd.read_parquet(os.path.join(DATA_ROOT, "moe_data.parquet"))
df.head()


# ## 2. Imports and Global Configuration
# 
# The block below loads required libraries and defines global constants.
# `HORIZON` controls which prediction window is used for the single-horizon
# training run. Change it and rerun the script to reproduce a different
# horizon. The multi-horizon analysis iterates over all four horizons.

import os, math, time, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}',
      f'GPU: {torch.cuda.get_device_name(0)}' if DEVICE == 'cuda' else '')

HORIZON = 5        # change to 5, 15, 30, or 60
N_EXPERTS = 6
N_CLASSES = 3
EXPERT_NAMES = ['lgbm', 'lstm', 'mamba', 'moirai', 'ftt', 'ctts']

LABEL_MAP = {-1.0: 0, 0.0: 1, 1.0: 2}


def expert_prob_cols(model, horizon):
    """Return [down, no_jump, up] column names for a given model + horizon."""
    if model in ['lgbm', 'lstm', 'mamba', 'moirai']:
        s = f'{horizon}minutes_{model}'
    else:  # ftt, ctts
        s = f'{horizon}m_{model}'
    return [f'probability_down_{s}',
            f'probability_no_jump_{s}',
            f'probability_up_{s}']


def build_feature_list(df, horizon):
    """Compute the list of feature columns the GATE sees for this horizon."""
    leak = {'ticker', 'created_time', 'split', 'count',
            'volume', 'no_price', 'taker_side'}
    for h in [5, 15, 30, 60]:
        leak.update([f'target_price_{h}m', f'signed_ret_{h}m',
                     f'abs_ret_{h}m', f'jump3_{h}m'])
        if h != horizon:
            for m in EXPERT_NAMES:
                leak.update(expert_prob_cols(m, h))
            # Also drop OTHER horizons' meta-features
            leak.update([c for c in df.columns
                         if c.endswith(f'_{h}m') and (
                             c.startswith('rolling_') or
                             c.startswith('has_') or
                             c.startswith('agreement_'))])

    return [c for c in df.columns if c not in leak]


# ## 3. Model Architecture
# 
# The MoE layer has two components: a `MoEDataset` that packages per-trade tensors for PyTorch training, and a `MoEGate` MLP that learns to weight the six expert predictions.
# 
# ### 3.1 Dataset Class
# 
# `MoEDataset` extracts three tensors per trade:
# 1. **Gate features** `(F,)`: standardized contextual features the gating network uses to decide which experts to trust.
# 2. **Expert probabilities** `(6, 3)`: the 3-class softmax outputs from each of the six base models.
# 3. **Expert mask** `(6,)`: a binary indicator set to 0 for any expert whose output is missing (NaN) for this horizon or trade. Missing experts receive a uniform `[1/3, 1/3, 1/3]` placeholder; the gate ignores them via a large negative pre-softmax constant.

class MoEDataset(Dataset):
    """Yields (gate_features, expert_probs, expert_mask, label)."""
    def __init__(self, df, horizon, gate_feat_cols, gate_mean, gate_std, split=None):
        if split is not None:
            df = df[df['split'] == split].reset_index(drop=True)
        self.n = len(df)

        # ---- Gate features ----

        # Special-case rolling_acc_* NaN fill: use random-guess baseline (1/3)
        # instead of letting train-fold mean leak through
        df_copy = df[gate_feat_cols].copy()
        rolling_acc_cols = [c for c in gate_feat_cols if c.startswith('rolling_acc_')]
        if rolling_acc_cols:
            df_copy[rolling_acc_cols] = df_copy[rolling_acc_cols].fillna(1.0 / 3.0)

        gate_feats = df_copy.values.astype(np.float32)
        # Standardize using train-fold stats
        gate_feats = (gate_feats - gate_mean) / gate_std
        # Remaining NaN/inf becomes 0, the train-fold mean after standardization.
        gate_feats = np.nan_to_num(gate_feats, nan=0.0, posinf=0.0, neginf=0.0)
        self.gate_feats = gate_feats
        del df_copy

        # ---- Expert probabilities and availability mask ----
        expert_probs = np.zeros((self.n, N_EXPERTS, N_CLASSES), dtype=np.float32)
        expert_mask = np.zeros((self.n, N_EXPERTS), dtype=np.float32)
        for e_idx, m in enumerate(EXPERT_NAMES):
            cols = expert_prob_cols(m, horizon)
            vals = df[cols].values.astype(np.float32)
            valid = np.isfinite(vals).all(axis=1)
            expert_probs[valid, e_idx, :] = vals[valid]
            # Uniform [1/3, 1/3, 1/3] for missing. The gate ignores via mask.
            expert_probs[~valid, e_idx, :] = 1.0 / N_CLASSES
            expert_mask[:, e_idx] = valid.astype(np.float32)
        self.expert_probs = expert_probs
        self.expert_mask = expert_mask

        # ---- Labels ----
        labels = df[f'jump3_{horizon}m'].map(LABEL_MAP).astype(np.int64).values
        self.labels = labels

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return (torch.from_numpy(self.gate_feats[idx]),
                torch.from_numpy(self.expert_probs[idx]),
                torch.from_numpy(self.expert_mask[idx]),
                int(self.labels[idx]))


# ### 3.2 Gating Network
# 
# `MoEGate` is a three-layer MLP with LayerNorm and GELU activations. It reads the gate features, produces a 6-dimensional logit vector (one per expert), and applies a masked softmax so unavailable experts receive zero weight. The final class probabilities are a convex combination of the expert probability vectors, weighted by the gate output.

class MoEGate(nn.Module):
    """Tabular MoE gating network.
    Takes (gate_features, expert_probs, expert_mask), produces class probabilities."""
    def __init__(self, n_features, hidden_dims=(192, 96, 48),
                 n_experts=6, dropout=0.3):
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
        # Mask out unavailable experts before softmax
        gate_logits = gate_logits.masked_fill(expert_mask == 0, -1e9)
        gate_weights = F.softmax(gate_logits, dim=-1)
        # Weighted combination of expert probabilities
        final_probs = (gate_weights.unsqueeze(-1) * expert_probs).sum(dim=1)
        final_probs = final_probs.clamp(min=1e-7, max=1 - 1e-7)
        return final_probs, gate_weights


# ## 4. Training
# 
# ### 4.1 Feature Selection and Data Splits
# 
# `build_feature_list` reduces the 135-column dataframe to the gate input features for a given horizon. It removes direct look-ahead columns (target prices, signed returns, absolute returns, jump labels), probability columns from other horizons to prevent cross-horizon leakage, and ticker/timestamp identifiers.
# 
# We use a chronological split:
# - **Fit set (90% of train):** trades before approximately October 29, 2025. Used for gradient updates and normalization statistics.
# - **Validation set (10% of train):** trades between October 29 and November 6, 2025. Used for early stopping only.
# - **Test set (20% of all data):** trades from November 6, 2025 onward. Held out until final evaluation.
# 
# Class weights are computed from the fit-set label distribution to offset the heavy imbalance toward the "flat" class.

print(f'=== Training MoE gate for horizon = {HORIZON}m ===')

GATE_FEAT_COLS = build_feature_list(df, HORIZON)
print(f'Gate input features: {len(GATE_FEAT_COLS)}')

# Show breakdown
n_base  = sum(1 for c in GATE_FEAT_COLS if not c.startswith(('rolling_', 'has_',
                                                              'agreement_', 'probability_')))
n_probs = sum(1 for c in GATE_FEAT_COLS if c.startswith('probability_'))
n_meta  = len(GATE_FEAT_COLS) - n_base - n_probs
print(f'  base trade features: {n_base}')
print(f'  expert probabilities: {n_probs}')
print(f'  engineered meta features: {n_meta}')

# Define fit/val/test split using the ORIGINAL 80% timestamp
# (anchored on base-model split, not on filtered df)
TEST_CUT = pd.Timestamp('2025-11-06 03:58:08.028990', tz='UTC')

# Strip tz to match df['created_time'] which is tz-naive after normalize
if df['created_time'].dt.tz is None:
    TEST_CUT = TEST_CUT.tz_localize(None)

train_mask = df['created_time'] < TEST_CUT
val_cut = df.loc[train_mask, 'created_time'].quantile(0.9)
fit_mask = train_mask & (df['created_time'] < val_cut)

print(f'Test cut (from base models): {TEST_CUT}')
print(f'Val cut (within train fold):  {val_cut}')
print(f'  train rows:    {train_mask.sum():>12,}')
print(f'  test rows:     {(~train_mask).sum():>12,}')
print(f'  fit (90%):     {fit_mask.sum():>12,}')
print(f'  val (10%):     {(train_mask & ~fit_mask).sum():>12,}')
# Z-score stats from FIT fold only. Handle rolling_acc separately.
df_for_stats = df.loc[fit_mask, GATE_FEAT_COLS].copy()
rolling_acc_cols = [c for c in GATE_FEAT_COLS if c.startswith('rolling_acc_')]
if rolling_acc_cols:
    df_for_stats[rolling_acc_cols] = df_for_stats[rolling_acc_cols].fillna(1.0 / 3.0)

gate_mean = df_for_stats.mean().values.astype(np.float32)
gate_std = df_for_stats.std().replace(0, 1.0).values.astype(np.float32)
del df_for_stats
gc.collect()

# Build datasets
print('Building datasets...')
fit_ds  = MoEDataset(df[fit_mask],                    HORIZON, GATE_FEAT_COLS, gate_mean, gate_std)
val_ds  = MoEDataset(df[train_mask & ~fit_mask],      HORIZON, GATE_FEAT_COLS, gate_mean, gate_std)
test_ds = MoEDataset(df[~train_mask],                 HORIZON, GATE_FEAT_COLS, gate_mean, gate_std)
print(f'  fit: {len(fit_ds):,}  val: {len(val_ds):,}  test: {len(test_ds):,}')

# Class weights from fit fold
counts = np.bincount(fit_ds.labels, minlength=3).astype(np.float64)
cls_w = counts.sum() / (3 * np.maximum(counts, 1))
class_weights = torch.tensor(cls_w, dtype=torch.float32, device=DEVICE)
print(f'  class weights: {cls_w.round(3)}')

# Training hyperparameters
BATCH_SIZE = 4096
LR = 1e-3
MAX_EPOCHS = 30
PATIENCE = 1
WARMUP_STEPS = 500
ENTROPY_LAMBDA = 0.01

# Instantiate model
model = MoEGate(
    n_features=len(GATE_FEAT_COLS),
    hidden_dims=(192, 96, 48),
    n_experts=N_EXPERTS,
    dropout=0.3,
).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'  trainable params: {n_params:,}')


# ### 4.2 Training Loop
# 
# We use AdamW with a cosine learning-rate schedule and a 500-step linear warmup. The objective combines a class-weighted negative log-likelihood with a small entropy bonus on the gate weights (coefficient 0.01). The entropy term prevents the gate from collapsing its weight onto a single expert. Early stopping halts training when validation NLL shows no improvement for one epoch.

fit_loader = DataLoader(fit_ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                        num_workers=2, pin_memory=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = MAX_EPOCHS * len(fit_loader)

def lr_lambda(step):
    if step < WARMUP_STEPS:
        return step / max(1, WARMUP_STEPS)
    p = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    return 0.5 * (1 + math.cos(math.pi * p))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_loss(final_probs, gate_weights, labels):
    log_probs = torch.log(final_probs)
    nll = F.nll_loss(log_probs, labels, weight=class_weights)
    entropy = -(gate_weights * torch.log(gate_weights.clamp(min=1e-9))).sum(-1).mean()
    return nll - ENTROPY_LAMBDA * entropy, nll, entropy


@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_nll, total_n = 0.0, 0
    for xb, eb, mb, yb in loader:
        xb, eb, mb, yb = (xb.to(DEVICE), eb.to(DEVICE),
                          mb.to(DEVICE), yb.to(DEVICE))
        final_probs, gate_w = model(xb, eb, mb)
        log_probs = torch.log(final_probs)
        nll = F.nll_loss(log_probs, yb, weight=class_weights, reduction='sum')
        total_nll += nll.item()
        total_n += yb.size(0)
    return total_nll / max(1, total_n)


# Training loop with early stopping
best_val = float('inf')
best_epoch = -1
patience_left = PATIENCE
os.makedirs(MODEL_DIR, exist_ok=True)
ckpt_path = os.path.join(MODEL_DIR, f'moe_gate_{HORIZON}m.pt')

for epoch in range(MAX_EPOCHS):
    model.train()
    epoch_loss, n_batches = 0.0, 0
    t0 = time.time()
    for xb, eb, mb, yb in fit_loader:
        xb, eb, mb, yb = (xb.to(DEVICE, non_blocking=True),
                           eb.to(DEVICE, non_blocking=True),
                           mb.to(DEVICE, non_blocking=True),
                           yb.to(DEVICE, non_blocking=True))
        final_probs, gate_w = model(xb, eb, mb)
        loss, nll, ent = compute_loss(final_probs, gate_w, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        n_batches += 1

    train_avg = epoch_loss / n_batches
    val_loss = evaluate(val_loader)
    print(f'epoch {epoch+1:>2}: train_loss={train_avg:.4f}  val_nll={val_loss:.4f}  '
          f'time={time.time()-t0:.1f}s', end='')

    if val_loss < best_val:
        best_val = val_loss
        best_epoch = epoch + 1
        patience_left = PATIENCE
        torch.save({
            'model_state_dict': model.state_dict(),
            'gate_feat_cols': GATE_FEAT_COLS,
            'gate_mean': gate_mean,
            'gate_std': gate_std,
            'horizon': HORIZON,
            'best_val_nll': best_val,
            'best_epoch': best_epoch,
        }, ckpt_path)
        print('  saved')
    else:
        patience_left -= 1
        print(f'  (patience {patience_left}/{PATIENCE})')
        if patience_left == 0:
            print(f'\nEarly stopping at epoch {epoch+1}. '
                  f'Best val_nll={best_val:.4f} at epoch {best_epoch}.')
            break

ckpt = torch.load(ckpt_path, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
print(f'\nReloaded best (epoch {ckpt["best_epoch"]}, val_nll={ckpt["best_val_nll"]:.4f})')


# ## 5. Evaluation on the Held-Out Test Set
# 
# We evaluate the trained 5-minute model on the held-out 20% test fold using balanced accuracy (to correct for class imbalance) and macro-averaged F1. We also inspect the mean gate weight assigned to each expert to understand which base models the gating network relies on most.

from sklearn.metrics import (classification_report, confusion_matrix,
                              balanced_accuracy_score, f1_score)

test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                          num_workers=2, pin_memory=True)

model.eval()
all_final, all_gate, all_y = [], [], []
with torch.no_grad():
    for xb, eb, mb, yb in test_loader:
        xb, eb, mb = xb.to(DEVICE), eb.to(DEVICE), mb.to(DEVICE)
        final_probs, gate_w = model(xb, eb, mb)
        all_final.append(final_probs.cpu().numpy())
        all_gate.append(gate_w.cpu().numpy())
        all_y.append(yb.numpy())

final_probs = np.concatenate(all_final)
gate_weights = np.concatenate(all_gate)
y_test = np.concatenate(all_y)
preds = final_probs.argmax(axis=1)

print(f'=== Horizon = {HORIZON}m, test set ===')
print(f'balanced acc: {balanced_accuracy_score(y_test, preds):.4f}')
print(f'macro f1    : {f1_score(y_test, preds, average="macro"):.4f}')
print(classification_report(y_test, preds,
                            target_names=['down (-1)', 'no jump (0)', 'up (+1)'],
                            digits=4))
print('Confusion matrix:')
print(confusion_matrix(y_test, preds))

print(f'\nMean gate weight per expert (across test set):')
for e_idx, name in enumerate(EXPERT_NAMES):
    print(f'  {name:<10}: {gate_weights[:, e_idx].mean():.4f}')


# ### Expert Gate Weight Dynamics
# 
# The plot below shows how the average gate weight per expert evolves over the test period. Trades are aggregated into hourly bins and smoothed with a 6-hour rolling window to reveal temporal trends without trade-level noise.

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

# ---- Pull test timestamps aligned with gate_weights ----
TEST_CUT = pd.Timestamp('2025-11-06 03:58:08.028990', tz='UTC')
if df['created_time'].dt.tz is None:
    TEST_CUT = TEST_CUT.tz_localize(None)

test_times = (df.loc[df['created_time'] >= TEST_CUT, 'created_time']
                .reset_index(drop=True))

assert len(test_times) == len(gate_weights), \
    f'length mismatch: {len(test_times)} vs {len(gate_weights)}'

# ---- Build dataframe indexed by time ----
df_weights = pd.DataFrame(gate_weights, columns=EXPERT_NAMES)
df_weights['time'] = test_times.values

# ---- Sort by time and resample to a regular hourly grid ----
# This is the key step: average all trades within each hour to get one weight
# per expert per hour, instead of interleaved across tickers.
df_weights = df_weights.sort_values('time').set_index('time')
df_hourly = df_weights.resample('1h').mean().dropna()

print(f'resampled to {len(df_hourly):,} hourly points')

# Optional: light smoothing on top of the hourly bins
df_smooth = df_hourly.rolling(6, min_periods=1, center=True).mean()  # ~6h window

# ---- Plot ----
fig, ax = plt.subplots(figsize=(14, 6))
ax.stackplot(df_smooth.index,
             [df_smooth[name].values for name in EXPERT_NAMES],
             labels=EXPERT_NAMES, alpha=0.85)

ax.set_title('Expert gate weights over test period (hourly mean, 6h smoothing)')
ax.set_xlabel('Date')
ax.set_ylabel('Weight proportion')
ax.set_ylim(0, 1)
ax.margins(x=0, y=0)
ax.legend(loc='upper right', ncol=len(EXPERT_NAMES), framealpha=0.9)
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
fig.autofmt_xdate()
plt.tight_layout()
plt.show()


# ## 6. Multi-Horizon Analysis
# 
# This section loads the saved checkpoints for all four prediction horizons
# (5, 15, 30, and 60 minutes), runs inference on the test set for each, and
# produces three diagnostic visualizations:
# 
# 1. **Permutation feature importance:** drop in balanced accuracy when each gate feature is randomly shuffled, averaged over two repetitions on a 200k-row subsample.
# 2. **Confusion matrices:** row-normalized confusion matrices for each horizon, showing where the model makes its most common errors.
# 3. **Gate weight distribution:** mean gate weight per expert across the test set, revealing how the gating strategy shifts with the prediction horizon.
# 
# ### 6.1 Permutation Importance Utility

import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score


@torch.no_grad()
def permutation_importance_moe(model, dataset, gate_feat_cols,
                                 n_repeats=2, sample_size=200_000,
                                 device='cuda'):
    """Per-feature permutation importance for the MoE gate."""
    n = len(dataset)
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=min(sample_size, n), replace=False)
    idx.sort()

    # Materialize subset
    print(f'Materializing {len(idx):,} samples...')
    X = np.stack([dataset.gate_feats[i] for i in idx])      # [N, F]
    E = np.stack([dataset.expert_probs[i] for i in idx])    # [N, E, C]
    M = np.stack([dataset.expert_mask[i] for i in idx])     # [N, E]
    y = np.array([dataset.labels[i] for i in idx])          # [N]
    print(f'X={X.shape}, E={E.shape}, M={M.shape}, y={y.shape}')

    def score(X_arr):
        model.eval()
        preds = []
        for s in range(0, len(X_arr), 8192):
            xb = torch.from_numpy(X_arr[s:s+8192]).to(device)
            eb = torch.from_numpy(E[s:s+8192]).to(device)
            mb = torch.from_numpy(M[s:s+8192]).to(device)
            final_probs, _ = model(xb, eb, mb)
            preds.append(final_probs.argmax(-1).cpu().numpy())
        return balanced_accuracy_score(y, np.concatenate(preds))

    baseline = score(X)
    print(f'baseline balanced acc: {baseline:.4f}')

    importances = {}
    for f_idx, fname in enumerate(gate_feat_cols):
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            flat = X_perm[:, f_idx].copy()
            rng.shuffle(flat)
            X_perm[:, f_idx] = flat
            drops.append(baseline - score(X_perm))
        importances[fname] = (np.mean(drops), np.std(drops))

    return importances, baseline


def plot_importance_grid(imp_per_horizon, baseline_per_horizon, top_n=25):
    """4-panel feature importance plot matching the report style."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()

    for ax, h in zip(axes, [5, 15, 30, 60]):
        imp = imp_per_horizon[h]
        base = baseline_per_horizon[h]
        series = pd.Series({k: v[0] for k, v in imp.items()}).sort_values()
        series.tail(top_n).plot.barh(ax=ax, color='#1f77b4')
        ax.set_title(f'MoE permutation importance - horizon {h}m '
                      f'(baseline bal acc = {base:.4f})', fontsize=11)
        ax.set_xlabel('Delta balanced accuracy (higher = more important)')
        ax.grid(alpha=0.3, axis='x')
        ax.axvline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    plt.show()


# ### 6.2 Visualization Utilities
# 
# The two helper functions below produce the 4-panel confusion-matrix grid and the 4-panel gate-weight bar chart used in our report. They operate on the `results_eval` and `gate_weights_per_horizon` dictionaries populated by the evaluation loop in Section 6.3.

import seaborn as sns
from sklearn.metrics import (confusion_matrix, accuracy_score,
                              precision_recall_fscore_support)


def plot_confusion_grid(results_eval, horizons=[5, 15, 30, 60]):
    """4-panel confusion-matrix figure in the report's house style."""
    class_names = ['down', 'flat', 'up']
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))

    for ax, h in zip(axes, horizons):
        y, pred = results_eval[h]['y'], results_eval[h]['pred']
        acc = accuracy_score(y, pred)
        cm = confusion_matrix(y, pred, labels=[0, 1, 2])
        cm_norm = cm / cm.sum(axis=1, keepdims=True)

        sns.heatmap(cm_norm, annot=False, cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar=False, ax=ax, square=True, vmin=0, vmax=1)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = 'white' if cm_norm[i, j] > 0.5 else 'black'
                ax.text(j + 0.5, i + 0.42, f'{cm_norm[i, j]:.2f}',
                        ha='center', va='center',
                        fontsize=18, fontweight='bold', color=color)
                ax.text(j + 0.5, i + 0.70, f'{cm[i, j]:,}',
                        ha='center', va='center',
                        fontsize=9, color=color, alpha=0.85)

        ax.set_title(f'{h}m  (acc {acc:.3f})', fontsize=12)
        ax.set_xlabel('predicted')
        ax.set_ylabel('true')

    plt.tight_layout()
    plt.show()


def plot_gate_weights_grid(gate_weights_per_horizon, horizons=[5, 15, 30, 60]):
    """4-panel bar chart of mean gate weight per expert per horizon."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(EXPERT_NAMES)))

    for ax, h in zip(axes, horizons):
        gw = gate_weights_per_horizon[h]
        means = gw.mean(axis=0)
        stds  = gw.std(axis=0)
        ax.bar(range(len(EXPERT_NAMES)), means, yerr=stds,
               color=colors, alpha=0.85, capsize=4)
        ax.set_xticks(range(len(EXPERT_NAMES)))
        ax.set_xticklabels(EXPERT_NAMES, rotation=30)
        ax.set_ylim(0, max(0.5, means.max() * 1.3))
        ax.axhline(1 / len(EXPERT_NAMES), color='gray', linestyle='--',
                    linewidth=0.8, alpha=0.5, label='uniform')
        ax.set_title(f'{h}m gate weights', fontsize=12)
        ax.set_ylabel('mean gate weight')
        ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


# ### 6.3 Evaluating All Four Horizons
# 
# The loop below iterates over all four prediction horizons. For each horizon it loads the saved checkpoint, builds the test dataset, runs inference, computes permutation importance on a 200k-row subsample, and stores the results. The three visualization functions are called once the loop completes.

imp_per_horizon = {}
baseline_per_horizon = {}
results_eval = {}
gate_weights_per_horizon = {}

# Reuse the same test cut as during training
TEST_CUT = pd.Timestamp('2025-11-06 03:58:08.028990', tz='UTC')
if df['created_time'].dt.tz is None:
    TEST_CUT = TEST_CUT.tz_localize(None)

for HORIZON in [5, 15, 30, 60]:
    print(f'\n{"="*60}')
    print(f'Horizon = {HORIZON}m')
    print(f'{"="*60}')

    # Reload trained checkpoint
    ckpt_path = os.path.join(MODEL_DIR, f'moe_gate_{HORIZON}m.pt')
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    GATE_FEAT_COLS = ckpt['gate_feat_cols']
    gate_mean = ckpt['gate_mean']
    gate_std  = ckpt['gate_std']

    model = MoEGate(
        n_features=len(GATE_FEAT_COLS),
        hidden_dims=(192, 96, 48),
        n_experts=N_EXPERTS,
        dropout=0.3,
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Rebuild test set
    test_df = df[df['created_time'] >= TEST_CUT]
    test_ds = MoEDataset(test_df, HORIZON, GATE_FEAT_COLS, gate_mean, gate_std)
    test_loader = DataLoader(test_ds, batch_size=8192, shuffle=False,
                              num_workers=2, pin_memory=True)

    # ---- 1. Test-set predictions and gate weights ----
    print('Running inference...')
    all_final, all_gate, all_y = [], [], []
    with torch.no_grad():
        for xb, eb, mb, yb in test_loader:
            xb, eb, mb = xb.to(DEVICE), eb.to(DEVICE), mb.to(DEVICE)
            final_probs, gate_w = model(xb, eb, mb)
            all_final.append(final_probs.cpu().numpy())
            all_gate.append(gate_w.cpu().numpy())
            all_y.append(yb.numpy())

    final_probs = np.concatenate(all_final)
    gate_weights = np.concatenate(all_gate)
    y_test = np.concatenate(all_y)
    preds = final_probs.argmax(axis=1)

    results_eval[HORIZON] = {'y': y_test, 'pred': preds, 'proba': final_probs}
    gate_weights_per_horizon[HORIZON] = gate_weights

    # ---- 2. Permutation importance ----
    print('Computing permutation importance (this takes a few minutes)...')
    imp, base = permutation_importance_moe(
        model, test_ds, GATE_FEAT_COLS,
        n_repeats=2, sample_size=200_000, device=DEVICE,
    )
    imp_per_horizon[HORIZON] = imp
    baseline_per_horizon[HORIZON] = base

    # Free memory before next horizon
    del model, test_ds, test_loader, final_probs, gate_weights, y_test, preds
    gc.collect()
    torch.cuda.empty_cache()


# ---- Plot everything ----
print('\nPlotting feature importance...')
plot_importance_grid(imp_per_horizon, baseline_per_horizon)

print('\nPlotting confusion matrices...')
plot_confusion_grid(results_eval)

print('\nPlotting gate weights...')
plot_gate_weights_grid(gate_weights_per_horizon)

# src/models/torch_regressors.py
from __future__ import annotations
from typing import Optional, List, Tuple, Dict
import numpy as np
import math, time, os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, RegressorMixin

# --------- utilities ---------
def _device(auto: str = "auto") -> torch.device:
    if auto != "auto":
        return torch.device(auto)
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def _pinball_loss(y_hat: torch.Tensor, y: torch.Tensor, tau: float = 0.8) -> torch.Tensor:
    # Quantile loss (pinball)
    diff = y - y_hat
    return torch.mean(torch.maximum(tau * diff, (tau - 1.0) * diff))

def _mae_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(y - y_hat))

def _detect_sequence_from_columns(columns: List[str]) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    """
    Try to detect lag sequence columns like steps_lag1..steps_lag64.
    Returns (seq_idx, static_idx): index arrays for sequence and static features.
    """
    seq_idx = []
    for i, c in enumerate(columns):
        # customize patterns if needed
        if c.startswith("steps_lag"):
            try:
                # sort by increasing lag number (lag1, lag2, ...)
                lag = int(c.replace("steps_lag", ""))
                seq_idx.append((i, lag))
            except Exception:
                pass
    if not seq_idx:
        return None, None
    # sort by lag so the sequence is [lag1, lag2, ... lagN]
    seq_idx = [i for (i, _) in sorted(seq_idx, key=lambda x: x[1])]
    static_idx = [i for i in range(len(columns)) if i not in seq_idx]
    return seq_idx, static_idx

# --------- simple TCN block ---------
class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, d=1, p=0.1):
        super().__init__()
        pad = (k - 1) * d
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=d),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=d),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
        )
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        # causal crop to keep causality after padding
        out = self.net(x)
        crop = out.size(-1) - x.size(-1)
        if crop > 0:
            out = out[..., :-crop]
        return out + self.down(x)

class TinyTCN(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 64, layers: int = 5, k: int = 3, p: float = 0.1):
        super().__init__()
        blocks = []
        ch = in_ch
        for i in range(layers):
            d = 2 ** i
            blocks.append(TemporalBlock(ch, hidden, k=k, d=d, p=p))
            ch = hidden
        self.tcn = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):  # x: (B, C, L)
        h = self.tcn(x)
        h = self.pool(h)   # (B, hidden, 1)
        out = self.head(h) # (B, 1)
        return out.squeeze(-1)

# --------- MLP for tabular fallback ---------
class TinyMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, depth: int = 3, p: float = 0.1):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(depth):
            layers += [nn.Linear(dim, hidden), nn.ReLU(inplace=True), nn.Dropout(p)]
            dim = hidden
        layers += [nn.Linear(dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

# --------- Sklearn-compatible wrapper ---------
class NeuralRegressor(BaseEstimator, RegressorMixin):
    """
    Sklearn-compatible PyTorch regressor.

    Modes:
      - model_type="mlp": works on flat tabular features (safe default)
      - model_type="tcn": auto-detects `steps_lag*` columns as a 1D sequence and
                          (optionally) concatenates static features via a small MLP head.

    Loss:
      - loss="mae" (L1)
      - loss="quantile", tau=0.8 (pinball Q80)

    Early stopping on a small tail-val split of the given train data (no change to your train.py split logic).
    """

    def __init__(self,
                 model_type: str = "mlp",          # "mlp" or "tcn"
                 loss: str = "mae",                # "mae" or "quantile"
                 tau: float = 0.80,
                 epochs: int = 40,
                 batch_size: int = 2048,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-5,
                 hidden: int = 256,                # mlp hidden or tcn hidden
                 depth: int = 3,                   # mlp depth
                 tcn_layers: int = 5,
                 tcn_kernel: int = 3,
                 dropout: float = 0.1,
                 val_frac: float = 0.1,            # last 10% of rows as tail-val
                 patience: int = 6,
                 device: str = "auto",
                 num_workers: int = 0,
                 verbose: int = 1):
        self.model_type = model_type
        self.loss = loss
        self.tau = tau
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.hidden = hidden
        self.depth = depth
        self.tcn_layers = tcn_layers
        self.tcn_kernel = tcn_kernel
        self.dropout = dropout
        self.val_frac = val_frac
        self.patience = patience
        self.device_str = device
        self.num_workers = num_workers
        self.verbose = verbose

        # learned
        self.model_: Optional[nn.Module] = None
        self.columns_: Optional[List[str]] = None
        self.is_tcn_active_: bool = False
        self.seq_idx_: Optional[List[int]] = None
        self.static_idx_: Optional[List[int]] = None
        self.device_ = _device(device)

    # ---- internal helpers ----
    def _build_model(self, n_features: int, columns: List[str]) -> nn.Module:
        if self.model_type == "tcn":
            seq_idx, static_idx = _detect_sequence_from_columns(columns)
            if seq_idx is None:
                # fallback to MLP if no sequence columns detected
                if self.verbose:
                    print("[NeuralRegressor] No steps_lag* columns found — falling back to MLP.")
                self.is_tcn_active_ = False
                return TinyMLP(n_features, hidden=self.hidden, depth=self.depth, p=self.dropout)
            self.is_tcn_active_ = True
            self.seq_idx_, self.static_idx_ = seq_idx, static_idx
            in_ch = 1  # single channel (steps sequence)
            self.static_head = None
            if static_idx:  # if there are static feats, add a small head
                self.static_head = TinyMLP(len(static_idx), hidden=self.hidden//2, depth=2, p=self.dropout)
                self.final_head = nn.Sequential(
                    nn.Linear(self.hidden + 1, self.hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.hidden, 1),
                )
            return TinyTCN(in_ch, hidden=self.hidden, layers=self.tcn_layers, k=self.tcn_kernel, p=self.dropout)

        # default MLP
        self.is_tcn_active_ = False
        return TinyMLP(n_features, hidden=self.hidden, depth=self.depth, p=self.dropout)

    def _criterion(self):
        if self.loss == "quantile":
            return lambda yhat, y: _pinball_loss(yhat, y, self.tau)
        return _mae_loss

    def _to_tensor(self, X, y=None):
        X = torch.tensor(X, dtype=torch.float32)
        if y is None:
            return X
        return X, torch.tensor(y, dtype=torch.float32)

    def _make_loaders(self, X, y):
        n = X.shape[0]
        v = max(1, int(self.val_frac * n))
        tr_X, va_X = X[:-v], X[-v:]
        tr_y, va_y = y[:-v], y[-v:]

        tr_ds = TensorDataset(torch.tensor(tr_X, dtype=torch.float32), torch.tensor(tr_y, dtype=torch.float32))
        va_ds = TensorDataset(torch.tensor(va_X, dtype=torch.float32), torch.tensor(va_y, dtype=torch.float32))

        tr_ld = DataLoader(tr_ds, batch_size=self.batch_size, shuffle=True,
                           num_workers=self.num_workers, pin_memory=True)
        va_ld = DataLoader(va_ds, batch_size=self.batch_size, shuffle=False,
                           num_workers=self.num_workers, pin_memory=True)
        return tr_ld, va_ld

    def _forward(self, model, xb, static_b=None):
        if not self.is_tcn_active_:
            return model(xb)
        # reshape sequence: pick seq columns and make (B, C=1, L)
        seq = xb[:, self.seq_idx_].unsqueeze(1)  # (B, 1, L)
        yhat_seq = model(seq)                    # (B,)
        if self.static_idx_ and self.static_head is not None:
            stat = xb[:, self.static_idx_]
            yhat_stat = self.static_head(stat)   # (B,)
            cat = torch.stack([yhat_seq, yhat_stat], dim=1)  # (B, 2)
            return self.final_head(cat)[:, 0]
        return yhat_seq

    # ---- sklearn API ----
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        self.columns_ = getattr(X, "columns", None)
        if self.columns_ is None:
            # X is ndarray; we can't detect seq columns -> use MLP unless user set model_type="tcn" explicitly
            self.columns_ = [f"f{i}" for i in range(X.shape[1])]
        self.model_ = self._build_model(X.shape[1], self.columns_)
        self.model_.to(self.device_)

        tr_ld, va_ld = self._make_loaders(X, y)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        crit = self._criterion()

        best_val = math.inf
        best_state = None
        bad = 0
        t0 = time.time()

        if self.verbose:
            total_params = sum(p.numel() for p in self.model_.parameters())
            print(f"[NeuralRegressor] start training | device={self.device_} | params={total_params:,} | "
                  f"epochs={self.epochs} | bs={self.batch_size} | loss={self.loss}{'(tau=%.2f)'%self.tau if self.loss=='quantile' else ''}")

        for ep in range(1, self.epochs + 1):
            self.model_.train()
            tr_loss = 0.0
            for xb, yb in tr_ld:
                xb = xb.to(self.device_, non_blocking=True); yb = yb.to(self.device_, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                yhat = self._forward(self.model_, xb)
                loss = crit(yhat, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                opt.step()
                tr_loss += loss.item() * xb.size(0)
            tr_loss /= len(tr_ld.dataset)

            # val
            self.model_.eval()
            va_loss = 0.0
            with torch.no_grad():
                for xb, yb in va_ld:
                    xb = xb.to(self.device_, non_blocking=True); yb = yb.to(self.device_, non_blocking=True)
                    yhat = self._forward(self.model_, xb)
                    va_loss += crit(yhat, yb).item() * xb.size(0)
            va_loss /= len(va_ld.dataset)

            if self.verbose:
                print(f"[NeuralRegressor] epoch {ep:03d}/{self.epochs}  train={tr_loss:.4f}  val={va_loss:.4f}")

            if va_loss + 1e-6 < best_val:
                best_val = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    if self.verbose:
                        print(f"[NeuralRegressor] early stopping at epoch {ep} (best val={best_val:.4f})")
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        if self.verbose:
            print(f"[NeuralRegressor] done in {time.time()-t0:.1f}s | best_val={best_val:.4f}")
        return self

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("Model not fitted")
        X = np.asarray(X, dtype=np.float32)
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
        ld = DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                        num_workers=self.num_workers, pin_memory=True)

        preds = []
        self.model_.eval()
        with torch.no_grad():
            for (xb,) in ld:
                xb = xb.to(self.device_, non_blocking=True)
                yhat = self._forward(self.model_, xb)
                preds.append(yhat.detach().cpu().numpy())
        return np.concatenate(preds, axis=0)

from __future__ import annotations
from typing import Optional, List
import numpy as np, math, time
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, RegressorMixin

# ----- losses -----
def pinball(y_hat, y, tau=0.8, reduce="mean"):
    d = y - y_hat
    l = torch.maximum(tau*d, (tau-1.0)*d)  # shape [B]
    return l.mean() if reduce=="mean" else l

def deadzone(y_hat, y, eps=50.0, p=1.0, reduce="mean"):
    e = torch.abs(y_hat - y)
    z = torch.relu(e - eps)
    l = z**p  # shape [B]
    return l.mean() if reduce=="mean" else l


# ----- tiny backbones -----
class MLP(nn.Module):
    def __init__(self, d, h=256, depth=3, p=0.1):
        super().__init__()
        layers, x = [], d
        for _ in range(depth):
            layers += [nn.Linear(x, h), nn.ReLU(inplace=True), nn.Dropout(p)]
            x = h
        layers += [nn.Linear(x, h), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class Gate(nn.Module):
    def __init__(self, d, h=64, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(inplace=True), nn.Dropout(p),
            nn.Linear(h, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x).squeeze(-1)  # (B,)

# ----- sklearn wrapper -----
class NeuralMoE(BaseEstimator, RegressorMixin):
    """
    Mixture-of-Experts:
      y_hat = g * y_hat_TOL + (1-g) * y_hat_Q80
    Loss = g * L_tol + (1-g) * L_q80 + beta * BCE(g, active_target)

    - active_target is derived from y>=thr_active (weak supervision for gate)
    - Works with tabular features (your lags/rolls); no change to train.py
    """
    def __init__(self,
                 tol_k: float = 50.0, tol_power: float = 1.0, q: float = 0.80,
                 thr_active: float = 60.0, beta_gate: float = 0.2,
                 hidden: int = 256, depth: int = 3, gate_hidden: int = 64,
                 dropout: float = 0.1,
                 lr: float = 1e-3, weight_decay: float = 1e-5,
                 epochs: int = 80, batch_size: int = 4096,
                 val_frac: float = 0.1, patience: int = 10,
                 device: str = "auto", num_workers: int = 0, verbose: int = 1):
        self.tol_k = tol_k; self.tol_power = tol_power; self.q = q
        self.thr_active = thr_active; self.beta_gate = beta_gate
        self.hidden = hidden; self.depth = depth; self.gate_hidden = gate_hidden
        self.dropout = dropout
        self.lr = lr; self.weight_decay = weight_decay
        self.epochs = epochs; self.batch_size = batch_size
        self.val_frac = val_frac; self.patience = patience
        self.device_str = device; self.num_workers = num_workers; self.verbose = verbose

        self.model_ = None
        self.device_ = torch.device("cuda") if (device=="auto" and torch.cuda.is_available()) else torch.device(device if device!="auto" else "cpu")

    def _build(self, d):
        back = MLP(d, h=self.hidden, depth=self.depth, p=self.dropout)
        self.head_tol = nn.Linear(self.hidden, 1)
        self.head_q   = nn.Linear(self.hidden, 1)
        self.gate     = Gate(d, h=self.gate_hidden, p=self.dropout)
        self.backbone = back

        model = nn.Module()
        model.backbone = self.backbone
        model.head_tol = self.head_tol
        model.head_q   = self.head_q
        model.gate     = self.gate
        return model

    def _split(self, X, y):
        n = X.shape[0]; v = max(1, int(self.val_frac*n))
        return (X[:-v], y[:-v]), (X[-v:], y[-v:])

    def fit(self, X, y):
        X = np.asarray(X, np.float32); y = np.asarray(y, np.float32).reshape(-1)
        (Xtr, ytr), (Xva, yva) = self._split(X, y)

        self.model_ = self._build(X.shape[1]).to(self.device_)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        tr_ds = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
        va_ds = TensorDataset(torch.tensor(Xva), torch.tensor(yva))
        tr_ld = DataLoader(tr_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        va_ld = DataLoader(va_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

        def forward(batch_x):
            h = self.model_.backbone(batch_x)
            y_tol = self.model_.head_tol(h).squeeze(-1)
            y_q   = self.model_.head_q(h).squeeze(-1)
            g     = self.model_.gate(batch_x)
            y_hat = g*y_tol + (1-g)*y_q
            return y_hat, y_tol, y_q, g

        best = math.inf; best_state=None; bad=0; t0=time.time()
        if self.verbose:
            total_params = sum(p.numel() for p in self.model_.parameters())
            print(f"[NeuralMoE] start | device={self.device_} | params={total_params:,} | epochs={self.epochs} | bs={self.batch_size}")

        for ep in range(1, self.epochs+1):
            self.model_.train(); tr_loss=0.0
            for xb, yb in tr_ld:
                xb = xb.to(self.device_, non_blocking=True); yb = yb.to(self.device_, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                yhat, y_tol, y_q, g = forward(xb)
                l_tol = deadzone(y_tol, yb, self.tol_k, self.tol_power, reduce="none")  # [B]
                l_q   = pinball(y_q,   yb, self.q,                     reduce="none")   # [B]
                mix   = g * l_tol + (1.0 - g) * l_q                                  # [B]
                L     = mix.mean()

                # weak supervision for gate (optional but helps)
                active = (yb >= self.thr_active).float()
                bce = nn.functional.binary_cross_entropy(g, active)
                loss = L + self.beta_gate * bce

                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                opt.step()
                tr_loss += loss.item()*xb.size(0)
            tr_loss/=len(tr_ld.dataset)

            # val
            self.model_.eval(); va_loss=0.0
            with torch.no_grad():
                for xb, yb in va_ld:
                    xb = xb.to(self.device_, non_blocking=True); yb = yb.to(self.device_, non_blocking=True)
                    yhat, y_tol, y_q, g = forward(xb)
                    l_tol = deadzone(y_tol, yb, self.tol_k, self.tol_power, reduce="none")
                    l_q   = pinball(y_q,   yb, self.q,                     reduce="none")
                    mix   = g * l_tol + (1.0 - g) * l_q
                    L     = mix.mean()
                    active = (yb >= self.thr_active).float()
                    bce = nn.functional.binary_cross_entropy(g, active)
                    va_loss += (L + self.beta_gate*bce).item()*xb.size(0)
            va_loss/=len(va_ld.dataset)

            if self.verbose:
                print(f"[NeuralMoE] epoch {ep:03d}/{self.epochs}  train={tr_loss:.4f}  val={va_loss:.4f}")

            if va_loss+1e-6 < best:
                best = va_loss; bad=0
                best_state = {k:v.detach().cpu().clone() for k,v in self.model_.state_dict().items()}
            else:
                bad+=1
                if bad>=self.patience:
                    if self.verbose: print(f"[NeuralMoE] early stop @ {ep} (best={best:.4f})")
                    break

        if best_state is not None: self.model_.load_state_dict(best_state)
        if self.verbose: print(f"[NeuralMoE] done in {time.time()-t0:.1f}s | best_val={best:.4f}")
        return self

    def predict(self, X):
        if self.model_ is None: raise RuntimeError("fit first")
        X = np.asarray(X, np.float32)
        ds = TensorDataset(torch.tensor(X))
        ld = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        preds = []
        self.model_.eval()
        with torch.no_grad():
            for (xb,) in ld:
                xb = xb.to(self.device_, non_blocking=True)
                h = self.model_.backbone(xb)
                y_tol = self.model_.head_tol(h).squeeze(-1)
                y_q   = self.model_.head_q(h).squeeze(-1)
                g     = self.model_.gate(xb)
                y_hat = g*y_tol + (1-g)*y_q
                preds.append(y_hat.detach().cpu().numpy())
        return np.concatenate(preds, 0)

"""TopK Sparse Autoencoder — bio-sae config (Gao et al. 2024 variant) for join-
compatibility with Igor's Geneformer/scGPT atlases.

Contract matched to Biodyn-AI/bio-sae src/sae_model.py:
  encode:  h = TopK_k( W_enc (x - mu) + b_enc ),  ReLU on the top-k values
  decode:  x_hat = W_dec h + b_dec
  decoder columns unit-normalised after every optimiser step
  loss = MSE (no L1; TopK enforces sparsity)
Defaults: k=32, d_sae=4*d_model, Adam lr 3e-4, batch 4096, 4 epochs, 1M positions/layer.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAECfg:
    d_model: int
    expansion: int = 4
    k: int = 32
    lr: float = 3e-4
    batch: int = 4096
    epochs: int = 4
    train_positions: int = 1_000_000
    seed: int = 0

    @property
    def d_sae(self) -> int:
        return self.expansion * self.d_model


class TopKSAE(nn.Module):
    def __init__(self, d_model: int, d_sae: int, k: int):
        super().__init__()
        self.d_model, self.d_sae, self.k = d_model, d_sae, k
        self.W_enc = nn.Linear(d_model, d_sae, bias=True)
        self.W_dec = nn.Linear(d_sae, d_model, bias=True)
        self.register_buffer("mu", torch.zeros(d_model))
        nn.init.kaiming_uniform_(self.W_enc.weight, nonlinearity="relu")
        nn.init.zeros_(self.W_enc.bias)
        with torch.no_grad():
            self.W_dec.weight.data = F.normalize(self.W_dec.weight.data, dim=0)
        nn.init.zeros_(self.W_dec.bias)

    def encode(self, x):
        h = self.W_enc(x - self.mu)
        vals, idx = torch.topk(h, self.k, dim=-1)
        vals = F.relu(vals)
        out = torch.zeros_like(h)
        out.scatter_(-1, idx, vals)
        return out

    def decode(self, h):
        return self.W_dec(h)

    def forward(self, x):
        h = self.encode(x)
        return self.decode(h), h

    @torch.no_grad()
    def unit_norm_decoder(self):
        self.W_dec.weight.data = F.normalize(self.W_dec.weight.data, dim=0)


def train_sae(X: np.ndarray, cfg: SAECfg, device: str = "cuda", log=print):
    """Train a TopK SAE on activation rows X [n_positions, d_model]. Returns (sae, stats)."""
    rng = np.random.default_rng(cfg.seed)
    n = len(X)
    sae = TopKSAE(cfg.d_model, cfg.d_sae, cfg.k).to(device)
    sae.mu.copy_(torch.as_tensor(X.mean(0), dtype=torch.float32, device=device))
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)

    sub = rng.choice(n, min(cfg.train_positions, n), replace=False)
    Xt = torch.as_tensor(X[sub], dtype=torch.float32)
    for ep in range(cfg.epochs):
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(Xt), cfg.batch):
            xb = Xt[perm[i:i + cfg.batch]].to(device)
            xh, _ = sae(xb)
            loss = ((xb - xh) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            sae.unit_norm_decoder()
            tot += loss.item() * len(xb)
        log(f"      epoch {ep+1}/{cfg.epochs}  mse {tot/len(Xt):.4f}")

    # eval on a 100k held-out split
    hold = rng.choice(n, min(100_000, n), replace=False)
    with torch.no_grad():
        Xe = torch.as_tensor(X[hold], dtype=torch.float32, device=device)
        Xh, H = sae(Xe)
        var_res = ((Xe - Xh) ** 2).mean().item()
        var_in = Xe.var().item()
        fvu = var_res / max(var_in, 1e-9)
        freq = (H > 0).float().mean(0).cpu().numpy()
        dead = float((freq == 0).mean())
        # decoder direction separation
        W = F.normalize(sae.W_dec.weight.data, dim=0)  # [d_model, d_sae]
        s = (W.T @ W).abs()
        mean_cos = float((s.sum() - s.diag().sum()) / (s.numel() - s.shape[0]))
    stats = {"fvu": fvu, "var_explained": 1 - fvu, "dead_frac": dead,
             "alive": int((freq > 0).sum()), "mean_abs_decoder_cos": mean_cos}
    return sae, stats


@torch.no_grad()
def feature_activations(sae: TopKSAE, X: np.ndarray, device: str = "cuda", batch: int = 65536):
    sae.eval()
    out = np.empty((len(X), sae.d_sae), dtype=np.float32)
    for i in range(0, len(X), batch):
        xb = torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=device)
        out[i:i + batch] = sae.encode(xb).cpu().numpy()
    return out

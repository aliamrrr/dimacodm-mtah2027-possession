"""The deep model in torch, for TRAINING only (on other games). Inference is deep_net (numpy).

Module and parameter names are the keys deep_net.NumpyNet reads; `export_numpy` writes them.
test_ali_deep.py checks that both forward passes agree.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mtah.solutions.dimacodm.deep_features import N_GLOBAL, N_TOKEN, TEAM_COLUMN
from mtah.solutions.dimacodm.deep_net import DeepConfig


class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate="tanh")


class Attention(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        n, k, d = x.shape
        h = self.heads
        qkv = self.qkv(x).reshape(n, k, 3, h, d // h).permute(2, 0, 3, 1, 4)
        # Fused kernel, same maths as deep_net (the ball token is never masked, so no row of the
        # softmax is empty): 1.4x faster than materialising the scores, 3x with torch.compile.
        out = F.scaled_dot_product_attention(
            qkv[0], qkv[1], qkv[2], attn_mask=mask[:, None, None, :]
        )
        return self.out(out.transpose(1, 2).reshape(n, k, d))


class SetBlock(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = Attention(d, heads)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), GELU(), nn.Linear(2 * d, d))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        return x + self.mlp(self.ln2(x))


class TCNBlock(nn.Module):
    def __init__(self, c: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(c)
        self.conv1 = nn.Conv1d(c, c, 3, dilation=dilation, padding=dilation)
        self.conv2 = nn.Conv1d(c, c, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        y = self.ln(h.transpose(1, 2)).transpose(1, 2)
        y = self.conv2(self.drop(F.gelu(self.conv1(y), approximate="tanh")))
        return h + y


class SetTCN(nn.Module):
    def __init__(self, cfg: DeepConfig) -> None:
        super().__init__()
        d, c = cfg.d_token, cfg.d_frame
        self.cfg = cfg
        self.tok_in = nn.Sequential(nn.Linear(N_TOKEN, d), GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([SetBlock(d, cfg.heads) for _ in range(cfg.set_layers)])
        self.ln_tok = nn.LayerNorm(d)
        self.glob_in = nn.Linear(N_GLOBAL, d)
        self.frame = nn.Sequential(nn.Linear(6 * d, c), GELU(), nn.Linear(c, c))
        self.frame_drop = nn.Dropout(cfg.dropout)
        self.convention = nn.Embedding(cfg.n_conventions, c)
        self.tcn = nn.ModuleList([TCNBlock(c, dil, cfg.dropout) for dil in cfg.dilations])
        self.ln_out = nn.LayerNorm(c)
        self.state = nn.Linear(c, 3)
        self.ball = nn.Linear(c, 2)
        self.query = nn.Linear(c, d)
        self.boundary = nn.Linear(c, 1) if cfg.boundary_head else None

    def encode_frames(
        self, tokens: torch.Tensor, mask: torch.Tensor, globals_: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(n, K, F), (n, K), (n, G) -> frame vectors (n, C), token outputs (n, K, d)."""
        x = self.tok_in(tokens)
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_tok(x)
        team = tokens[..., TEAM_COLUMN]
        pooled = []
        for side in (team > 0, team < 0):
            sel = (mask & side).unsqueeze(-1)
            count = sel.sum(dim=1)
            pooled.append((x * sel).sum(dim=1) / count.clamp(min=1))
            top = x.masked_fill(~sel, float("-inf")).amax(dim=1)
            pooled.append(torch.where(count > 0, top, torch.zeros_like(top)))
        pooled.append(x[:, -1])
        pooled.append(self.glob_in(globals_))
        return self.frame(torch.cat(pooled, dim=-1)), x

    def forward(
        self,
        tokens: torch.Tensor,  # (B, T, K, F)
        mask: torch.Tensor,  # (B, T, K) bool
        globals_: torch.Tensor,  # (B, T, G)
        convention: torch.Tensor,  # (B,) long
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """(state logits (B, T, 3), ball (B, T, 2), possessor logits (B, T, K), team-change
        logits (B, T) if the model has that head)."""
        b, t, k, f = tokens.shape
        frame, tok = self.encode_frames(
            tokens.reshape(b * t, k, f), mask.reshape(b * t, k), globals_.reshape(b * t, -1)
        )
        h = self.frame_drop(frame.reshape(b, t, -1)) + self.convention(convention)[:, None, :]
        h = h.transpose(1, 2)
        for block in self.tcn:
            h = block(h)
        h = self.ln_out(h.transpose(1, 2))
        query = self.query(h)
        tok = tok.reshape(b, t, k, -1)
        pointer = (tok * query.unsqueeze(2)).sum(-1) / math.sqrt(tok.shape[-1])
        pointer = pointer.masked_fill(~mask, -1e9)
        boundary = None if self.boundary is None else self.boundary(h)[..., 0]
        return self.state(h), self.ball(h), pointer, boundary

    @torch.no_grad()
    def state_logits_half(
        self,
        tokens: np.ndarray,
        mask: np.ndarray,
        globals_: np.ndarray,
        convention: int,
        device: str,
        chunk: int = 4096,
    ) -> np.ndarray:
        """(T, 3) logits for a whole half, like deep_net.NumpyNet.state_logits."""
        return self.heads_half(tokens, mask, globals_, convention, device, chunk)[0]

    @torch.no_grad()
    def heads_half(
        self,
        tokens: np.ndarray,
        mask: np.ndarray,
        globals_: np.ndarray,
        convention: int,
        device: str,
        chunk: int = 4096,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """(state logits (T, 3), team-change logits (T,) or None) for a whole half."""
        self.eval()
        frames = []
        for a in range(0, len(tokens), chunk):
            frame, _ = self.encode_frames(
                torch.from_numpy(tokens[a : a + chunk]).to(device),
                torch.from_numpy(mask[a : a + chunk]).to(device),
                torch.from_numpy(globals_[a : a + chunk]).to(device),
            )
            frames.append(frame)
        h = torch.cat(frames)[None] + self.convention.weight[convention][None, None, :]
        h = h.transpose(1, 2)
        for block in self.tcn:
            h = block(h)
        h = self.ln_out(h.transpose(1, 2))
        state = self.state(h)[0].float().cpu().numpy()
        if self.boundary is None:
            return state, None
        return state, self.boundary(h)[0, :, 0].float().cpu().numpy()


def export_numpy(model: SetTCN, path: Path) -> Path:
    """Weights as float32 arrays + the config, the file deep_net.NumpyNet.load reads."""
    arrays = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, __config__=np.array(model.cfg.to_json()), **arrays)
    return path

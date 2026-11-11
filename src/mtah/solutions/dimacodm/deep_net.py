"""The deep model's forward pass in numpy: what runs on BAR-ATH (no torch, no ONNX).

Architecture (mirrored exactly by deep_model.SetTCN, which is only used to train it):

    per frame   tokens (46 slots + ball) -> MLP -> set transformer (pre-norm, masked attention)
                -> pooled: home mean/max, away mean/max, ball token  +  global features
                -> frame vector
    over time   + label-convention embedding -> dilated residual TCN (kernel 3) -> LayerNorm
    heads       state logits (H, A, D) ; ball position ; possessor pointer over the tokens

`ensemble_logits` runs every fold model, twice (as is and with the teams swapped, the mirror being
averaged back, so the output is exactly symmetric), for each label convention, and returns the
mean (team log-odds H vs A, dead log-odds D vs alive) per frame.

Everything is computed in float64. Rule 3: the organisers regenerate the submission on their own
machine, whose BLAS may sum a dot product in another order. In float64 such differences stay
around 1e-14, far below the 1e-4 grid the decoder rounds its inputs to (solution.decode_logits),
so the same file comes out; in float32 (~1e-6) a few frames would round the other way.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from mtah.solutions.dimacodm.deep_features import N_GLOBAL, N_TOKEN, TEAM_COLUMN, swap_teams

FloatArray = npt.NDArray[np.float64]
InputArray = npt.NDArray[np.float32]  # model inputs, as deep_features.inputs makes them
LN_EPS = 1e-5


@dataclass(frozen=True)
class DeepConfig:
    d_token: int = 64
    heads: int = 4
    set_layers: int = 2
    d_frame: int = 128
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    n_conventions: int = 3  # Metrica events, DFL tracking flags, SkillCorner
    dropout: float = 0.1
    boundary_head: bool = False  # also predict the frames where the team in possession changes

    @property
    def receptive_field(self) -> int:
        """Frames of context on each side of a prediction."""
        return int(sum(self.dilations))

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> DeepConfig:
        data = json.loads(text)
        data["dilations"] = tuple(data["dilations"])
        return cls(**data)


# The helpers below work in place on one fresh array (the passes over memory, not the
# arithmetic, dominate in float64); the IEEE operations are the same as the plain formulas.


def gelu(x: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """0.5 x (1 + tanh(0.79788 (x + 0.044715 x^3))), as torch's gelu(approximate='tanh'). x * x * x
    rather than x**3: plain products are faster and exactly rounded on every machine, unlike pow()."""
    u = x * x
    u *= x
    u *= 0.044715
    u += x
    u *= 0.7978845608028654
    np.tanh(u, out=u)
    u += 1.0
    u *= x
    u *= 0.5
    return u


def layer_norm(x: FloatArray, weight: FloatArray, bias: FloatArray) -> FloatArray:
    centred = x - x.mean(axis=-1, keepdims=True)
    var = (centred * centred).mean(axis=-1, keepdims=True)
    centred /= np.sqrt(var + LN_EPS)
    centred *= weight
    centred += bias
    return centred


def linear(x: FloatArray, w: dict[str, FloatArray], name: str) -> FloatArray:
    y = x @ w[f"{name}.weight"].T
    y += w[f"{name}.bias"]
    return y


class NumpyNet:
    """One trained fold: weights (its state_dict, as float64 arrays) + its config."""

    def __init__(self, weights: dict[str, FloatArray], config: DeepConfig) -> None:
        self.w = weights
        self.cfg = config

    @classmethod
    def load(cls, path: Path) -> NumpyNet:
        with np.load(path) as data:
            config = DeepConfig.from_json(str(data["__config__"]))
            weights = {k: data[k].astype(np.float64) for k in data.files if k != "__config__"}
        return cls(weights, config)

    # -- per frame ---------------------------------------------------------------------------
    def _attention(self, x: FloatArray, mask: npt.NDArray[np.bool_], prefix: str) -> FloatArray:
        n, k, d = x.shape
        h = self.cfg.heads
        dh = d // h
        qkv = linear(x, self.w, f"{prefix}.qkv").reshape(n, k, 3, h, dh).transpose(2, 0, 3, 1, 4)
        q, kk, v = qkv[0], qkv[1], qkv[2]
        scores = q @ kk.transpose(0, 1, 3, 2)
        scores /= np.sqrt(dh)
        np.copyto(scores, -1e9, where=~mask[:, None, None, :])
        scores -= scores.max(axis=-1, keepdims=True)
        np.exp(scores, out=scores)
        scores /= scores.sum(axis=-1, keepdims=True)
        out = (scores @ v).transpose(0, 2, 1, 3).reshape(n, k, d)
        return linear(out, self.w, f"{prefix}.out")

    def encode_frames(
        self, tokens: InputArray, mask: npt.NDArray[np.bool_], globals_: InputArray
    ) -> tuple[FloatArray, FloatArray]:
        """tokens (n, K, F) -> (frame vectors (n, d_frame), token outputs (n, K', d_token)).

        Only the tokens present in some frame of the chunk are computed (K' <= K, present ones
        first, in their order): an absent token is never attended to nor pooled, so the frame
        vectors are those of the full computation, 2-3x faster (most of the 46 slots are empty).
        """
        w = self.w
        count = mask.sum(axis=1)
        order = np.argsort(~mask, axis=1, kind="stable")[:, : int(count.max())]
        tokens = np.take_along_axis(tokens, order[..., None], axis=1)
        mask = np.take_along_axis(mask, order, axis=1)
        ball = count - 1  # the ball token, last of all, is always present: last present one
        x = linear(gelu(linear(tokens.astype(np.float64), w, "tok_in.0")), w, "tok_in.2")
        for i in range(self.cfg.set_layers):
            p = f"blocks.{i}"
            x = x + self._attention(
                layer_norm(x, w[f"{p}.ln1.weight"], w[f"{p}.ln1.bias"]), mask, f"{p}.attn"
            )
            y = layer_norm(x, w[f"{p}.ln2.weight"], w[f"{p}.ln2.bias"])
            x = x + linear(gelu(linear(y, w, f"{p}.mlp.0")), w, f"{p}.mlp.2")
        x = layer_norm(x, w["ln_tok.weight"], w["ln_tok.bias"])
        team = tokens[..., TEAM_COLUMN]
        pooled = []
        for side in (team > 0, team < 0):
            sel = (mask & side)[..., None]
            count = sel.sum(axis=1)
            pooled.append((x * sel).sum(axis=1) / np.maximum(count, 1))
            top = np.where(sel, x, -np.inf).max(axis=1)
            pooled.append(np.where(count > 0, top, 0.0))
        pooled.append(x[np.arange(len(x)), ball])
        pooled.append(linear(globals_.astype(np.float64), w, "glob_in"))
        frame = linear(gelu(linear(np.concatenate(pooled, axis=-1), w, "frame.0")), w, "frame.2")
        return frame, x

    # -- over time ---------------------------------------------------------------------------
    def temporal(self, frame: FloatArray, convention: int) -> FloatArray:
        """(T, d_frame) -> (T, d_frame), zero padding at both ends like torch's Conv1d."""
        w = self.w
        h = frame + w["convention.weight"][convention]
        for i, dilation in enumerate(self.cfg.dilations):
            p = f"tcn.{i}"
            y = layer_norm(h, w[f"{p}.ln.weight"], w[f"{p}.ln.bias"])
            padded = np.pad(y, ((dilation, dilation), (0, 0)))
            k1 = w[f"{p}.conv1.weight"]  # (out, in, 3)
            y = (
                padded[: len(h)] @ k1[:, :, 0].T
                + padded[dilation : dilation + len(h)] @ k1[:, :, 1].T
                + padded[2 * dilation :] @ k1[:, :, 2].T
                + w[f"{p}.conv1.bias"]
            )
            y = gelu(y) @ w[f"{p}.conv2.weight"][:, :, 0].T + w[f"{p}.conv2.bias"]
            h = h + y
        return layer_norm(h, w["ln_out.weight"], w["ln_out.bias"])

    def state_logits(
        self,
        tokens: InputArray,
        mask: npt.NDArray[np.bool_],
        globals_: InputArray,
        conventions: tuple[int, ...],
        chunk: int = 512,  # frames encoded together: small enough to stay in cache
    ) -> list[FloatArray]:
        """Per convention: (T, 3) logits for a whole half (frames encoded in chunks)."""
        return [pair[0] for pair in self.heads(tokens, mask, globals_, conventions, chunk)]

    def heads(
        self,
        tokens: InputArray,
        mask: npt.NDArray[np.bool_],
        globals_: InputArray,
        conventions: tuple[int, ...],
        chunk: int = 512,
    ) -> list[tuple[FloatArray, FloatArray | None]]:
        """Per convention: ((T, 3) state logits, (T,) team-change logits or None)."""
        frames = np.concatenate(
            [
                self.encode_frames(
                    tokens[a : a + chunk], mask[a : a + chunk], globals_[a : a + chunk]
                )[0]
                for a in range(0, len(tokens), chunk)
            ]
        )
        out: list[tuple[FloatArray, FloatArray | None]] = []
        for c in conventions:
            h = self.temporal(frames, c)
            boundary = linear(h, self.w, "boundary")[:, 0] if self.cfg.boundary_head else None
            out.append((linear(h, self.w, "state"), boundary))
        return out


def log_odds(logits: FloatArray) -> tuple[FloatArray, FloatArray]:
    """(team H vs A, dead vs alive) log-odds from (T, 3) state logits."""
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    eps = 1e-6
    team = np.log(p[:, 0] + eps) - np.log(p[:, 1] + eps)
    dead = np.log(p[:, 2] + eps) - np.log(p[:, 0] + p[:, 1] + eps)
    return team, dead


def ensemble_logits(
    nets: list[NumpyNet],
    tokens: InputArray,
    mask: npt.NDArray[np.bool_],
    globals_: InputArray,
    conventions: tuple[int, ...] = (0, 1),
) -> tuple[FloatArray, FloatArray]:
    """Mean (team, dead) log-odds over folds x conventions x {as is, teams swapped}."""
    team, dead, _ = ensemble_outputs(nets, tokens, mask, globals_, conventions)
    return team, dead


def ensemble_outputs(
    nets: list[NumpyNet],
    tokens: InputArray,
    mask: npt.NDArray[np.bool_],
    globals_: InputArray,
    conventions: tuple[int, ...] = (0, 1),
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """(team log-odds, dead log-odds, team-change probability; 0 without that head), averaged
    over folds x conventions x {as is, teams swapped}.

    The (fold, mirror) passes run on threads (numpy releases the GIL in its loops); each pass is
    computed alone and the results are averaged in a fixed order: same output as one thread."""
    swapped_tokens, swapped_globals = swap_teams(tokens, globals_)
    jobs = [
        (net, sign, t, g)
        for net in nets
        for sign, t, g in ((1.0, tokens, globals_), (-1.0, swapped_tokens, swapped_globals))
    ]

    def run(job: tuple[NumpyNet, float, InputArray, InputArray]) -> list[tuple[Any, Any, Any]]:
        net, sign, t, g = job
        out = []
        for logits, boundary in net.heads(t, mask, g, conventions):
            team, dead = log_odds(logits)  # the mirror's home is our away
            change = 0.0 if boundary is None else 1.0 / (1.0 + np.exp(-boundary))
            out.append((sign * team, dead, change))
        return out

    with ThreadPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 1)) as pool:
        results = [pair for pairs in pool.map(run, jobs) for pair in pairs]
    return (
        np.mean([r[0] for r in results], axis=0),
        np.mean([r[1] for r in results], axis=0),
        np.mean([np.broadcast_to(r[2], r[0].shape) for r in results], axis=0),
    )


def check_shapes(tokens: InputArray, globals_: InputArray) -> None:
    if tokens.shape[-1] != N_TOKEN or globals_.shape[-1] != N_GLOBAL:
        raise ValueError(f"expected {N_TOKEN} token / {N_GLOBAL} global features")

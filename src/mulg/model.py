"""Figure 3 and Eq. (1)-(9) reconstruction, with documented implementation choices."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


@dataclass
class ModelConfig:
    hidden: int = 40
    heads: int = 4
    attention_layers: int = 2
    gru_layers: int = 2
    text_dropout: float = 0.2
    attention_dropout: float = 0.1
    output_dropout: float = 0.1
    residual_dropout: float = 0.1
    kernel_size: int = 3
    gru_variant: str = "paper"
    objective: str = "ce7"


def zero_padding(x, mask):
    return x.masked_fill(~mask.unsqueeze(-1), 0)


def position_encoding(length, dim, device, dtype):
    positions = torch.arange(length, device=device, dtype=dtype)[:, None]
    rates = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000) / dim))
    result = torch.zeros(length, dim, device=device, dtype=dtype)
    result[:, 0::2] = torch.sin(positions * rates)
    result[:, 1::2] = torch.cos(positions * rates[: result[:, 1::2].shape[1]])
    return result


class DirectedAttention(nn.Module):
    """Eq. (5)-(7), per-head bilinear scores, no sqrt(d) scaling.

    Query timesteps belong to the target; keys and values belong to the source.
    The 4-head partition is an explicit choice combining Eq. (5) and Table 7.
    """

    def __init__(self, dim, heads, dropout):
        super().__init__()
        if dim % heads:
            raise ValueError("hidden must be divisible by heads")
        self.heads = heads
        self.weight = nn.Parameter(torch.empty(heads, dim // heads, dim // heads))
        for weight in self.weight:
            nn.init.xavier_uniform_(weight)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, source, query_mask, source_mask):
        b, t, d = query.shape
        q = query.reshape(b, t, self.heads, d // self.heads).transpose(1, 2)
        s = source.reshape(b, source.shape[1], self.heads, d // self.heads).transpose(1, 2)
        logits = torch.matmul(torch.matmul(q, self.weight), s.transpose(-1, -2))
        logits = logits.masked_fill(~source_mask[:, None, None, :], float("-inf"))
        weights = self.dropout(logits.softmax(dim=-1))
        values = torch.matmul(weights, s).transpose(1, 2).reshape(b, t, d)
        return zero_padding(self.out(values), query_mask)


class CrossBlock(nn.Module):
    """CT reconstruction: directed attention, FFN and conventional post-LN residuals."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.hidden
        self.attention = DirectedAttention(d, cfg.heads, cfg.attention_dropout)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.dropout = nn.Dropout(cfg.residual_dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 4), nn.ReLU(), nn.Dropout(cfg.residual_dropout), nn.Linear(d * 4, d)
        )

    def forward(self, query, source, qm, sm):
        x = zero_padding(self.norm1(query + self.dropout(self.attention(query, source, qm, sm))), qm)
        return zero_padding(self.norm2(x + self.dropout(self.ffn(x))), qm)


class PaperGRU(nn.Module):
    """Exactly the reset-before-matrix candidate and update convention in Eq. (1)-(4).

    No biases are added (none appear in the equations). Invalid timesteps retain
    the previous state. This explicit recurrence is slower than fused nn.GRU.
    """

    def __init__(self, input_dim, hidden, layers, dropout):
        super().__init__()
        self.hidden = hidden
        self.x_proj = nn.ModuleList(
            [nn.Linear(input_dim if i == 0 else hidden, 3 * hidden, bias=False) for i in range(layers)]
        )
        self.h_gates = nn.ModuleList([nn.Linear(hidden, 2 * hidden, bias=False) for _ in range(layers)])
        self.h_candidate = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        for i, project in enumerate(self.x_proj):
            projected = project(x)
            state = x.new_zeros(x.shape[0], self.hidden)
            sequence = []
            for t in range(x.shape[1]):
                xr, xz, xn = projected[:, t].chunk(3, dim=-1)
                hr, hz = self.h_gates[i](state).chunk(2, dim=-1)
                reset, update = torch.sigmoid(xr + hr), torch.sigmoid(xz + hz)
                candidate = torch.tanh(xn + self.h_candidate[i](reset * state))
                next_state = (1 - update) * state + update * candidate
                state = torch.where(mask[:, t, None], next_state, state)
                sequence.append(state)
            if i < len(self.x_proj) - 1:
                x = self.dropout(torch.stack(sequence, dim=1))
        return state


class TorchGRU(nn.Module):
    """Optional fast variant; PyTorch's candidate equation differs from Eq. (3)."""

    def __init__(self, input_dim, hidden, layers, dropout):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0)

    def forward(self, x, mask):
        packed = pack_padded_sequence(x, mask.sum(1).cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return h[-1]


class MulG(nn.Module):
    def __init__(self, input_dims, modalities="tav", cfg=None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.modalities = "".join(m for m in "tav" if m in modalities)
        if not self.modalities or len(set(modalities)) != len(modalities) or set(modalities) - set("tav"):
            raise ValueError("modalities must be a nonempty subset of tav")
        c, d = self.cfg, self.cfg.hidden
        if c.kernel_size % 2 != 1 or c.kernel_size < 1:
            raise ValueError("kernel_size must be a positive odd number")
        if c.attention_layers < 1 or c.gru_layers < 1:
            raise ValueError("Layer counts must be positive")
        self.projections = nn.ModuleDict(
            {
                m: nn.Conv1d(input_dims[m], d, c.kernel_size, padding=c.kernel_size // 2)
                for m in self.modalities
            }
        )
        self.embedding_dropout = nn.ModuleDict(
            {m: nn.Dropout(c.text_dropout if m == "t" else 0) for m in self.modalities}
        )
        self.cross = nn.ModuleDict(
            {
                f"{source}_to_{target}": nn.ModuleList([CrossBlock(c) for _ in range(c.attention_layers)])
                for target in self.modalities
                for source in self.modalities
                if source != target
            }
        )
        gru_class = PaperGRU if c.gru_variant == "paper" else TorchGRU
        self.grus = nn.ModuleDict(
            {
                m: gru_class(d * max(1, len(self.modalities) - 1), d, c.gru_layers, c.residual_dropout)
                for m in self.modalities
            }
        )
        self.fusion = nn.Linear(d * len(self.modalities), d)
        self.residual = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Dropout(c.output_dropout), nn.Linear(d, d)
        )
        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(c.output_dropout)
        self.output = nn.Linear(d, 7 if c.objective == "ce7" else 1)

    def forward(self, features, masks):
        projected = {}
        for m in self.modalities:
            x = zero_padding(features[m], masks[m])
            x = self.projections[m](self.embedding_dropout[m](x).transpose(1, 2)).transpose(1, 2)
            projected[m] = zero_padding(
                x + position_encoding(x.shape[1], x.shape[2], x.device, x.dtype), masks[m]
            )
        representations = []
        for target in self.modalities:
            incoming = []
            for source in self.modalities:
                if target == source:
                    continue
                x = projected[target]
                for block in self.cross[f"{source}_to_{target}"]:
                    x = block(x, projected[source], masks[target], masks[source])
                incoming.append(x)
            # A singleton has no cross-modal path; its projected sequence enters its GRU.
            sequence = torch.cat(incoming, dim=-1) if incoming else projected[target]
            representations.append(self.grus[target](sequence, masks[target]))
        fused = torch.relu(self.fusion(torch.cat(representations, dim=-1)))
        fused = self.norm(fused + self.residual(fused))
        return self.output(self.dropout(fused))

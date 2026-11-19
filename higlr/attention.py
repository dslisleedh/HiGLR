"""Block-routed hybrid attention with pooled coarse and residual fine values."""
import math

import torch
from torch import nn

from .layers import WindowProjection, _attention_dtype, block_mean_projection


def select_routes(scores: torch.Tensor, remote_k: int):
    batch, heads, groups, _ = scores.shape
    own = torch.arange(groups, device=scores.device)[None, None, :, None]
    own = own.expand(batch, heads, -1, -1)
    count = min(remote_k, groups - 1)
    if count == 0:
        return own.int().contiguous(), scores[..., :1] * 0
    remote = scores.scatter(-1, own, -torch.inf)
    indices = remote.detach().topk(count, dim=-1).indices
    # Normalize over ALL remote blocks, not only the selected top-k blocks.
    gates = remote.log_softmax(-1).gather(-1, indices)
    ids = torch.cat((own, indices), dim=-1).int().contiguous()
    gates = torch.cat((torch.zeros_like(gates[..., :1]), gates), dim=-1).contiguous()
    return ids, gates


def fine_reference(q, k, v, pooled, ids, gates, bias, block=8, chunk=4):
    """Differentiable selected-block oracle. Never forms token-square attention."""
    batch, tokens, heads, dim = q.shape
    area = block * block
    groups, routes = tokens // area, ids.shape[-1]
    keys = k.float().reshape(batch, groups, area, heads, dim).permute(0, 3, 1, 2, 4)
    values = v.reshape(batch, groups, area, heads, dim).permute(0, 3, 1, 2, 4)
    values = values - pooled[:, :, :, None, :]
    queries = q.float().reshape(batch, groups, area, heads, dim).permute(0, 3, 1, 2, 4)
    bi = torch.arange(batch, device=q.device)[:, None, None, None]
    hi = torch.arange(heads, device=q.device)[None, :, None, None]
    p = torch.arange(area, device=q.device)
    row, col = p // block, p % block
    delta_row = row[:, None] - row[None, :] + block - 1
    delta_col = col[:, None] - col[None, :] + block - 1
    relative = delta_row * (2 * block - 1) + delta_col
    relative_bias = bias[:, relative][None, :, None, :, None, :]
    outputs = []
    for start in range(0, groups, chunk):
        stop = min(start + chunk, groups)
        index = ids[:, :, start:stop].long()
        selected_k = keys[bi, hi, index].flatten(-3, -2)
        selected_v = values[bi, hi, index].flatten(-3, -2)
        scores = queries[:, :, start:stop] @ selected_k.transpose(-1, -2) / math.sqrt(dim)
        scores = scores.reshape(batch, heads, stop - start, area, routes, area)
        scores = scores + relative_bias + gates[:, :, start:stop, None, :, None]
        outputs.append(scores.flatten(-2).softmax(-1) @ selected_v)
    out = torch.cat(outputs, dim=2).permute(0, 2, 3, 1, 4)
    return out.reshape(batch, tokens, heads, dim).contiguous()


class BRHA(nn.Module):
    def __init__(
        self, channels: int, heads: int, attention_dtype: str = 'auto',
        backend: str = 'triton', remote_k: int = 32,
    ):
        super().__init__()
        self.channels, self.heads, self.dim = channels, heads, 64
        self.block, self.router_dim, self.topk = 8, 16, remote_k
        self.backend, self.attention_dtype = backend, attention_dtype
        self.routing_backend = 'recompute'

        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

        self.reduce_q = WindowProjection(heads, 64, 8, 'dwc')
        self.reduce_k = WindowProjection(heads, 64, 8, 'dwc')
        self.q_summary_norm = nn.LayerNorm(64)
        self.k_summary_norm = nn.LayerNorm(64)
        self.global_feature_projection = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, heads * 16))
        self.query_fusion = nn.Sequential(nn.Linear(80, 16), nn.GELU(), nn.Linear(16, 16))
        self.key_fusion = nn.Sequential(nn.Linear(80, 16), nn.GELU(), nn.Linear(16, 16))
        self.relative_bias = nn.Parameter(torch.empty(heads, 225))
        nn.init.trunc_normal_(self.relative_bias, std=0.02)
        self.coarse_gain = nn.Parameter(torch.ones(heads))
        self.coarse_v = block_mean_projection(channels, self.block, bias=False)

    def window_scores(self, q, k, condition, height, width):
        query_summary = self.reduce_q(q, height, width)
        batch, _, groups, _ = query_summary.shape
        condition = self.global_feature_projection(condition).reshape(batch, groups, self.heads, 16)
        condition = condition.permute(0, 2, 1, 3)
        query = self.query_fusion(torch.cat((self.q_summary_norm(query_summary), condition), -1))
        key_summary = self.reduce_k(k, height, width)
        key = self.key_fusion(torch.cat((self.k_summary_norm(key_summary), condition), -1))
        return query @ key.transpose(-1, -2) * 0.25

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        block = self.block
        if height % block or width % block:
            raise ValueError('BRHA feature dimensions must be divisible by 8')
        tokens = height * width
        rows, cols = height // block, width // block
        windows = x.permute(0, 2, 3, 1).reshape(batch, rows, block, cols, block, channels)
        windows = windows.permute(0, 1, 3, 2, 4, 5).reshape(batch, tokens, channels)
        q, k, v = [
            t.reshape(batch, tokens, self.heads, self.dim).contiguous()
            for t in self.qkv(windows).chunk(3, dim=-1)
        ]
        dtype = _attention_dtype(x, self.attention_dtype)
        if self.backend == 'triton' and not self.training and not torch.is_grad_enabled():
            from .kernels.brha_inference import routed_attention
            mixed = routed_attention(self, q, k, v, condition, dtype)
        else:
            mixed = self._differentiable_attention(q, k, v, condition, height, width, dtype)
        output = self.proj(mixed.reshape(batch, tokens, channels))
        output = output.reshape(batch, rows, cols, block, block, channels)
        output = output.permute(0, 5, 1, 3, 2, 4)
        return output.reshape(batch, channels, height, width).contiguous()

    def _differentiable_attention(self, q, k, v, condition, height, width, dtype):
        batch, tokens = q.shape[:2]
        area = self.block * self.block
        groups = tokens // area
        scores = self.window_scores(q, k, condition, height, width).contiguous()
        weights = self.coarse_v.weight.reshape(self.heads, self.dim, area)
        values = v.reshape(batch, groups, area, self.heads, self.dim)
        pooled = torch.einsum('bgthd,hdt->bhgd', values, weights).contiguous()
        coarse = scores.softmax(-1) @ pooled

        ids, gates = select_routes(scores, self.topk)
        qb = q.to(dtype).contiguous()
        kb = (k - k.mean(1, keepdim=True)).to(dtype).contiguous()
        if self.backend == 'triton':
            if not q.is_cuda:
                raise ValueError('Triton requires CUDA; use backend="torch" for CPU')
            from .kernels.b8_backward_v3 import Fine
            fine = Fine.apply(qb, kb, v, pooled, ids, gates, self.relative_bias, 8, True)
        else:
            fine = fine_reference(qb, kb, v, pooled, ids, gates, self.relative_bias)
        fine = fine.reshape(batch, groups, area, self.heads, self.dim)
        coarse = coarse.permute(0, 2, 1, 3)[:, :, None]
        return fine + coarse * self.coarse_gain[None, None, None, :, None]

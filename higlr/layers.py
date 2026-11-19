"""DSFFN and dense attention, ported from the validated standalone model."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

if triton is not None:
    @triton.jit
    def _gelu(x):
        return .5 * x * (1. + tl.erf(x * .7071067811865476))


    @triton.jit
    def _gelu_grad(x):
        return .5 * (1. + tl.erf(x * .7071067811865476)) + x * .3989422804014327 * tl.exp(-.5 * x * x)


    @triton.jit
    def _down_forward(X, W, B, RW, P, A, N: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
                      EPS: tl.constexpr, PREC: tl.constexpr, BT: tl.constexpr):
        t = tl.program_id(0) * BT + tl.arange(0, BT)
        d = tl.arange(0, D); r = tl.arange(0, R)
        x = tl.load(X + t[:, None] * D + d[None, :], t[:, None] < N, 0.)
        w = tl.load(W + r[None, :] * D + d[:, None])
        p = tl.dot(x, w, input_precision=PREC) + tl.load(B + r)[None, :]
        inv = tl.rsqrt(tl.sum(p * p, 1) / R + EPS)
        a = _gelu(p * inv[:, None] * tl.load(RW + r)[None, :])
        tl.store(P + t[:, None] * R + r[None, :], p, t[:, None] < N)
        tl.store(A + t[:, None] * R + r[None, :], a, t[:, None] < N)


    @triton.jit
    def _main_values(X,C,NW,NB,UW,UB,SW,SB,t,N:tl.constexpr,D:tl.constexpr,R:tl.constexpr,H:tl.constexpr,EPS:tl.constexpr,PREC:tl.constexpr,start,BH:tl.constexpr):
        d=tl.arange(0,D);r=tl.arange(0,R);h=start+tl.arange(0,BH)
        x=tl.load(X+t[:,None]*D+d[None,:],t[:,None]<N,0.)
        center=x-(tl.sum(x,1)/D)[:,None];inv=tl.rsqrt(tl.sum(center*center,1)/D+EPS)
        z=center*inv[:,None];norm=z*tl.load(NW+d)[None,:]+tl.load(NB+d)[None,:]
        value=tl.dot(norm,tl.load(UW+h[None,:]*D+d[:,None]),input_precision=PREC)
        gate=tl.dot(norm,tl.load(UW+(H+h[None,:])*D+d[:,None]),input_precision=PREC)
        c=tl.load(C+t[:,None]*R+r[None,:],t[:,None]<N,0.)
        spatial=tl.dot(c,tl.load(SW+h[None,:]*R+r[:,None]),input_precision=PREC)
        u=value+tl.load(UB+h)[None,:]+spatial+tl.load(SB+h)[None,:]
        sig=tl.sigmoid(gate+tl.load(UB+H+h)[None,:]);hidden=_gelu(u)*sig
        return x,z,inv,norm,u,sig,hidden

    @triton.jit
    def _main_forward(X,C,NW,NB,UW,UB,SW,SB,OW,OB,SCALE,Y,N:tl.constexpr,D:tl.constexpr,R:tl.constexpr,H:tl.constexpr,EPS:tl.constexpr,PREC:tl.constexpr,BT:tl.constexpr):
        t=tl.program_id(0)*BT+tl.arange(0,BT);d=tl.arange(0,D)
        branch=tl.full((BT,D),0,tl.float32)
        for start in range(0,H,32):
            h=start+tl.arange(0,32)
            x,z,inv,norm,u,sig,hidden=_main_values(X,C,NW,NB,UW,UB,SW,SB,t,N,D,R,H,EPS,PREC,start,32)
            branch+=tl.dot(hidden,tl.load(OW+d[None,:]*H+h[:,None]),input_precision=PREC)
        branch+=tl.load(OB+d)[None,:]
        x=tl.load(X+t[:,None]*D+d[None,:],t[:,None]<N,0.)
        y=x+branch*tl.load(SCALE+d)[None,:]
        tl.store(Y+t[:,None]*D+d[None,:],y,t[:,None]<N)

    @triton.jit
    def _main_backward(X,C,NW,NB,UW,UB,SW,SB,OW,OB,SCALE,DY,DX,DC,DB,DVG,HID,NORM,PART,N:tl.constexpr,D:tl.constexpr,R:tl.constexpr,H:tl.constexpr,EPS:tl.constexpr,PREC:tl.constexpr,BT:tl.constexpr):
        pid=tl.program_id(0);t=pid*BT+tl.arange(0,BT);d=tl.arange(0,D);r=tl.arange(0,R)
        dy=tl.load(DY+t[:,None]*D+d[None,:],t[:,None]<N,0.)
        db=dy*tl.load(SCALE+d)[None,:]
        dn=tl.full((BT,D),0,tl.float32);dc=tl.full((BT,R),0,tl.float32);branch=tl.full((BT,D),0,tl.float32)
        for start in range(0,H,32):
            h=start+tl.arange(0,32)
            x,z,inv,norm,u,sig,hidden=_main_values(X,C,NW,NB,UW,UB,SW,SB,t,N,D,R,H,EPS,PREC,start,32)
            dh=tl.dot(db,tl.load(OW+d[:,None]*H+h[None,:]),input_precision=PREC)
            dv=dh*sig*_gelu_grad(u);dg=dh*_gelu(u)*sig*(1.-sig)
            dn+=tl.dot(dv,tl.load(UW+h[:,None]*D+d[None,:]),input_precision=PREC)
            dn+=tl.dot(dg,tl.load(UW+(H+h[:,None])*D+d[None,:]),input_precision=PREC)
            dc+=tl.dot(dv,tl.load(SW+h[:,None]*R+r[None,:]),input_precision=PREC)
            branch+=tl.dot(hidden,tl.load(OW+d[None,:]*H+h[:,None]),input_precision=PREC)
            tl.store(DVG+t[:,None]*(2*H)+h[None,:],dv,t[:,None]<N)
            tl.store(DVG+t[:,None]*(2*H)+H+h[None,:],dg,t[:,None]<N)
            tl.store(HID+t[:,None]*H+h[None,:],hidden,t[:,None]<N)
        x=tl.load(X+t[:,None]*D+d[None,:],t[:,None]<N,0.)
        center=x-(tl.sum(x,1)/D)[:,None];inv=tl.rsqrt(tl.sum(center*center,1)/D+EPS);z=center*inv[:,None]
        norm=z*tl.load(NW+d)[None,:]+tl.load(NB+d)[None,:]
        dz=dn*tl.load(NW+d)[None,:]
        dx=dy+inv[:,None]*(dz-(tl.sum(dz,1)/D)[:,None]-z*(tl.sum(dz*z,1)/D)[:,None])
        branch+=tl.load(OB+d)[None,:]
        tl.store(DX+t[:,None]*D+d[None,:],dx,t[:,None]<N)
        tl.store(DC+t[:,None]*R+r[None,:],dc,t[:,None]<N)
        tl.store(DB+t[:,None]*D+d[None,:],db,t[:,None]<N)
        tl.store(NORM+t[:,None]*D+d[None,:],norm,t[:,None]<N)
        tl.store(PART+pid*(3*D)+d,tl.sum(tl.where(t[:,None]<N,dn*z,0.),0))
        tl.store(PART+pid*(3*D)+D+d,tl.sum(tl.where(t[:,None]<N,dn,0.),0))
        tl.store(PART+pid*(3*D)+2*D+d,tl.sum(dy*branch,0))


    @triton.jit
    def _down_backward(P, DA, RW, W, DXMAIN, DP, DX, PART,
                       N: tl.constexpr, D: tl.constexpr, R: tl.constexpr, EPS: tl.constexpr,
                       PREC: tl.constexpr, BT: tl.constexpr):
        pid = tl.program_id(0); t = pid * BT + tl.arange(0, BT)
        r = tl.arange(0, R); d = tl.arange(0, D)
        p = tl.load(P + t[:, None] * R + r[None, :], t[:, None] < N, 0.)
        da = tl.load(DA + t[:, None] * R + r[None, :], t[:, None] < N, 0.)
        inv = tl.rsqrt(tl.sum(p * p, 1) / R + EPS)
        norm = p * inv[:, None]; rw = tl.load(RW + r)
        du = da * _gelu_grad(norm * rw[None, :])
        dn = du * rw[None, :]
        dp = inv[:, None] * (dn - norm * (tl.sum(dn * norm, 1) / R)[:, None])
        dx = tl.dot(dp, tl.load(W + r[:, None] * D + d[None, :]), input_precision=PREC)
        dx += tl.load(DXMAIN + t[:, None] * D + d[None, :], t[:, None] < N, 0.)
        tl.store(DP + t[:, None] * R + r[None, :], dp, t[:, None] < N)
        tl.store(DX + t[:, None] * D + d[None, :], dx, t[:, None] < N)
        tl.store(PART + pid * R + r, tl.sum(du * norm, 0))


    def _flat(x):
        return x.permute(0, 2, 3, 1).contiguous().view(-1, x.shape[1])


    def _image(x, shape):
        return x.view(shape[0], shape[2], shape[3], -1).permute(0, 3, 1, 2)


    class _DSFFNTriton(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, nw, nb, dw, db, rw, cw, cb, sw, sb, uw, ub, ow, ob, scale, eps):
            shape = x.shape; d = shape[1]; h = ow.shape[1]; xf = _flat(x); n = xf.shape[0]; rank = rw.numel()
            # Use tf32x3 for wide Triton dots; library GEMMs follow caller TF32 policy.
            prec = 'tf32x3'  # bounded shared memory and high-accuracy FP32 dot decomposition
            p = xf.new_empty((n, rank)); a = torch.empty_like(p)
            _down_forward[(triton.cdiv(n, 32),)](xf, dw, db, rw, p, a, n, d, rank, eps, prec, 32)
            ci = F.conv2d(_image(a, shape), cw, cb, padding=1, groups=rank)
            c = _flat(ci); y = torch.empty_like(xf)
            _main_forward[(triton.cdiv(n, 16),)](xf, c, nw, nb, uw, ub, sw, sb, ow, ob, scale, y,
                                              n, d, rank, h, eps, prec, 16, num_warps=8, num_stages=1)
            ctx.save_for_backward(xf, nw, nb, dw, db, rw, cw, cb, sw, sb, uw, ub, ow, ob, scale, p, a, c)
            ctx.shape = shape; ctx.eps = eps; ctx.prec = prec
            return _image(y, shape)

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, grad):
            x, nw, nb, dw, db, rw, cw, cb, sw, sb, uw, ub, ow, ob, scale, p, a, c = ctx.saved_tensors
            rank = rw.numel(); n, d = x.shape; h = ow.shape[1]; dy = _flat(grad)
            dxmain = torch.empty_like(x); dc = torch.empty_like(c); dbranch = torch.empty_like(x)
            dvg = x.new_empty((n, 2*h)); hidden = x.new_empty((n, h)); norm = torch.empty_like(x)
            part = x.new_empty((triton.cdiv(n, 16), 3*d))
            _main_backward[(triton.cdiv(n, 16),)](x, c, nw, nb, uw, ub, sw, sb, ow, ob, scale, dy,
                dxmain, dc, dbranch, dvg, hidden, norm, part, n, d, rank, h, ctx.eps, ctx.prec, 16, num_warps=8, num_stages=1)
            gnw, gnb, gscale = part.sum(0).split(d)
            gow = (dbranch.T @ hidden).view_as(ow); gob = dbranch.sum(0)
            guw = (dvg.T @ norm).view_as(uw); gub = dvg.sum(0)
            gsw = (dvg[:, :h].T @ c).view_as(sw); gsb = gub[:h].clone()
            del dbranch, hidden, dvg, norm, part
            ga, gcw, gcb = torch.ops.aten.convolution_backward.default(
                _image(dc, ctx.shape), _image(a, ctx.shape), cw, [rank], [1, 1], [1, 1], [1, 1],
                False, [0, 0], rank, [True, True, True])
            dp = torch.empty_like(p); dx = torch.empty_like(x)
            rpart = x.new_empty((triton.cdiv(n, 32), rank))
            _down_backward[(triton.cdiv(n, 32),)](p, _flat(ga), rw, dw, dxmain, dp, dx, rpart,
                                                n, d, rank, ctx.eps, ctx.prec, 32)
            gdw = (dp.T @ x).view_as(dw); gdb = dp.sum(0); grw = rpart.sum(0)
            grads = (_image(dx, ctx.shape), gnw, gnb, gdw, gdb, grw, gcw, gcb, gsw, gsb, guw, gub, gow, gob, gscale)
            return (*grads, None)




class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(1, keepdim=True)
        z = (x - mean) * torch.rsqrt((x - mean).square().mean(1, keepdim=True) + self.eps)
        return z * self.weight[None, :, None, None] + self.bias[None, :, None, None]


def _linear_conv(x, conv):
    return F.linear(x, conv.weight[:, :, 0, 0], conv.bias)


class DSFFN(nn.Module):
    """Gated FFN with a low-rank depthwise spatial branch."""

    def __init__(self, dim: int, backend: str = 'triton'):
        super().__init__()
        self.dim = dim
        self.rank = 16 if dim <= 64 else dim // 4
        self.hidden_dim = 3 * dim
        self.norm_eps = 1e-6
        self.backend = backend
        self._packed_inference = None

        self.norm = nn.LayerNorm(dim, eps=self.norm_eps)
        self.down = nn.Conv2d(dim, self.rank, 1)
        self.rms_weight = nn.Parameter(torch.ones(self.rank))
        self.dwc = nn.Conv2d(self.rank, self.rank, 3, padding=1, groups=self.rank)
        self.spatial_up = nn.Conv2d(self.rank, self.hidden_dim, 1)
        self.group_up = nn.Conv2d(dim, 2 * self.hidden_dim, 1)
        self.group_down = nn.Conv2d(self.hidden_dim, dim, 1)
        self.residual_scale = nn.Parameter(torch.full((dim,), 0.01))

    def train(self, mode=True):
        if mode:
            self._packed_inference = None
        return super().train(mode)

    def _apply(self, fn, recurse=True):
        # Packed weights are disposable and live on the current parameter device.
        self._packed_inference = None
        return super()._apply(fn, recurse=recurse)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        use_triton = self.backend != 'torch' and x.is_cuda and triton is not None
        if self.backend == 'triton' and not use_triton:
            raise RuntimeError('FFN triton backend requires CUDA and installed Triton; use torch or auto.')
        if use_triton:
            if not self.training and not torch.is_grad_enabled() and self.dim != 64:
                from .kernels.dsffn_inference import halo_dsffn, wide_dsffn
                return halo_dsffn(self, x) if self.dim == 32 else wide_dsffn(self, x)
            params = (self.norm.weight, self.norm.bias, self.down.weight, self.down.bias,
                      self.rms_weight, self.dwc.weight, self.dwc.bias, self.spatial_up.weight,
                      self.spatial_up.bias, self.group_up.weight, self.group_up.bias,
                      self.group_down.weight, self.group_down.bias, self.residual_scale)
            return _DSFFNTriton.apply(x, *params, self.norm_eps)
        x = x.permute(0, 2, 3, 1).contiguous()
        spatial = _linear_conv(x, self.down)
        inv_rms = torch.rsqrt(spatial.square().mean(-1, keepdim=True) + self.norm_eps)
        spatial = F.gelu(spatial * inv_rms * self.rms_weight)
        spatial = self.dwc(spatial.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        spatial = _linear_conv(spatial, self.spatial_up)

        value, gate = _linear_conv(self.norm(x), self.group_up).chunk(2, -1)
        feat = F.gelu(value + spatial) * gate.sigmoid()
        x = x + _linear_conv(feat, self.group_down) * self.residual_scale
        return x.permute(0, 3, 1, 2)


def _attention_dtype(x, policy):
    if policy == 'float32' or x.device.type != 'cuda':
        return torch.float32
    with torch.cuda.device(x.device):
        supported = torch.cuda.is_bf16_supported()
    if policy == 'bfloat16' and not supported:
        raise RuntimeError('This GPU does not support BF16; use attention_dtype="float32" or "auto".')
    return torch.bfloat16 if supported else torch.float32


def block_mean_projection(channels, block, *, bias=True):
    """Learnable nonoverlapping depthwise reduction, initialized as average pooling."""
    projection = nn.Conv2d(channels, channels, block, stride=block, groups=channels, bias=bias)
    nn.init.constant_(projection.weight, 1 / (block * block))
    if bias:
        nn.init.zeros_(projection.bias)
    return projection


class WindowProjection(nn.Module):
    """B,N,heads,D window-major tokens -> B,heads,windows,D descriptors."""
    def __init__(self, heads: int = 4, dim: int = 64, window: int = 4, kind: str = 'dwc'):
        super().__init__()
        self.heads, self.dim, self.window, self.kind = heads, dim, window, kind
        if kind == 'dwc':
            self.projection = block_mean_projection(heads * dim, window)
        elif kind == 'linear':
            # Full spatial/channel linear reduction within each head, shared over windows.
            self.weight = nn.Parameter(torch.empty(heads, dim, window * window * dim))
            self.bias = nn.Parameter(torch.zeros(heads, dim))
            for weight in self.weight:
                nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        else:
            raise ValueError('reduction must be dwc or linear')

    def forward(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, n, heads, dim = tokens.shape
        w = self.window
        assert n == height * width and heads == self.heads and dim == self.dim
        assert height % w == width % w == 0
        if self.kind == 'linear':
            area = w * w
            windows = tokens.reshape(batch, n // area, area, heads, dim)
            windows = windows.permute(0, 3, 1, 2, 4).reshape(batch, heads, n // area, area * dim)
            return torch.einsum('bhgs,hds->bhgd', windows, self.weight) + self.bias[None, :, None]
        feature = tokens.reshape(batch, height // w, width // w, w, w, heads * dim)
        feature = feature.permute(0, 5, 1, 3, 2, 4).reshape(batch, heads * dim, height, width)
        reduced = self.projection(feature)
        return reduced.reshape(batch, heads, dim, -1).permute(0, 1, 3, 2)


class DenseSA(nn.Module):
    """H/8 spatial attention with centered keys."""

    def __init__(self, channels: int, heads: int, attention_dtype: str = 'auto'):
        super().__init__()
        self.heads, self.attention_dtype = heads, attention_dtype
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        q, k, v = [
            t.reshape(b, h * w, self.heads, c // self.heads)
            for t in self.qkv(tokens).chunk(3, dim=-1)
        ]
        k = k - k.mean(1, keepdim=True)
        dtype = _attention_dtype(x, self.attention_dtype)
        q, k, v = [a.to(dtype).transpose(1, 2).contiguous() for a in (q, k, v)]
        if x.is_cuda and dtype == torch.bfloat16:
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                y = F.scaled_dot_product_attention(q, k, v)
        else:
            y = F.scaled_dot_product_attention(q, k, v)
        y = self.proj(y.transpose(1, 2).reshape(b, h * w, c).float())
        return y.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()



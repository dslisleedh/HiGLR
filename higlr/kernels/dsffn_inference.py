"""Full-resolution halo and packed wide DSFFN kernels used for evaluation."""

import torch
import triton
import triton.language as tl

from torch.nn import functional as F


@triton.jit
def _halo(
    X,
    NW,
    NB,
    DW,
    DB,
    RW,
    CW,
    CB,
    SW,
    SB,
    UW,
    UB,
    OW,
    OB,
    S,
    Y,
    H: tl.constexpr,
    W: tl.constexpr,
    HD: tl.constexpr,
    EPS: tl.constexpr,
    TY: tl.constexpr,
    TX: tl.constexpr,
):
    pid = tl.program_id(0)
    ny = triton.cdiv(H, TY)
    nx = triton.cdiv(W, TX)
    b = pid // (ny * nx)
    oy = pid // nx % ny * TY
    ox = pid % nx * TX
    p = tl.arange(0, TY * TX)
    yy = oy + p // TX
    xx = ox + p % TX
    v = (yy < H) & (xx < W)
    t = b * H * W + yy * W + xx
    RH: tl.constexpr = triton.next_power_of_2((TY + 2) * (TX + 2))
    r = tl.arange(0, RH)
    hy = oy + r // (TX + 2) - 1
    hx = ox + r % (TX + 2) - 1
    hv = (r < (TY + 2) * (TX + 2)) & (hy >= 0) & (hy < H) & (hx >= 0) & (hx < W)
    d = tl.arange(0, 32)
    k = tl.arange(0, 16)
    halo = tl.load(
        X + (b * H * W + hy[:, None] * W + hx[:, None]) * 32 + d[None, :],
        hv[:, None],
        0.0,
    )
    down = (
        tl.dot(halo, tl.load(DW + k[None, :] * 32 + d[:, None]), input_precision="tf32")
        + tl.load(DB + k)[None, :]
    )
    down = (
        down
        * tl.rsqrt(tl.sum(down * down, 1) / 16 + EPS)[:, None]
        * tl.load(RW + k)[None, :]
    )
    act = 0.5 * down * (1.0 + tl.erf(down * 0.7071067811865476))
    act = tl.where(hv[:, None], act, 0.0)
    c = tl.full((TY * TX, 16), 0.0, tl.float32)
    for j in tl.static_range(9):
        idx = (p // TX + j // 3) * (TX + 2) + p % TX + j % 3
        a = tl.gather(act, tl.broadcast_to(idx[:, None], (TY * TX, 16)), 0)
        c += a * tl.load(CW + k * 9 + j)[None, :]
    c += tl.load(CB + k)[None, :]
    x = tl.load(X + t[:, None] * 32 + d[None, :], v[:, None], 0.0)
    z = x - tl.sum(x, 1)[:, None] / 32
    z = z * tl.rsqrt(tl.sum(z * z, 1) / 32 + EPS)[:, None]
    z = z * tl.load(NW + d)[None, :] + tl.load(NB + d)[None, :]
    out = tl.full((TY * TX, 32), 0.0, tl.float32)
    for off in range(HD // 32):
        hh = off * 32 + d
        spatial = (
            tl.dot(
                c, tl.load(SW + hh[None, :] * 16 + k[:, None]), input_precision="tf32"
            )
            + tl.load(SB + hh)[None, :]
        )
        value = (
            tl.dot(
                z, tl.load(UW + hh[None, :] * 32 + d[:, None]), input_precision="tf32"
            )
            + tl.load(UB + hh)[None, :]
        )
        gate = (
            tl.dot(
                z,
                tl.load(UW + (HD + hh[None, :]) * 32 + d[:, None]),
                input_precision="tf32",
            )
            + tl.load(UB + HD + hh)[None, :]
        )
        u = value + spatial
        hidden = 0.5 * u * (1.0 + tl.erf(u * 0.7071067811865476)) * tl.sigmoid(gate)
        out += tl.dot(
            hidden, tl.load(OW + d[None, :] * HD + hh[:, None]), input_precision="tf32"
        )
    out += tl.load(OB + d)[None, :]
    tl.store(
        Y + t[:, None] * 32 + d[None, :], x + out * tl.load(S + d)[None, :], v[:, None]
    )


@triton.jit
def _gelu_erf(x):
    return 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))


_DOWN_CONFIGS = [
    triton.Config({"BLOCK_T": 16}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_T": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_T": 32}, num_warps=8, num_stages=2),
]


@triton.autotune(
    configs=_DOWN_CONFIGS,
    key=["T_BUCKET", "DEVICE_KEY", "DOT_PRECISION", "D", "R"],
)
@triton.jit
def _down_rms_predwc_gelu_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    rms_weight_ptr,
    output_ptr,
    tokens,
    T_BUCKET,
    DEVICE_KEY,
    EPS: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    offs_t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, triton.next_power_of_2(D))
    offs_r = tl.arange(0, R)
    valid_t = offs_t < tokens

    x = tl.load(
        x_ptr + offs_t[:, None] * D + offs_d[None, :],
        mask=valid_t[:, None] & (offs_d[None, :] < D),
        other=0.0,
    )
    weight = tl.load(
        weight_ptr + offs_d[:, None] * R + offs_r[None, :],
        mask=offs_d[:, None] < D,
        other=0.0,
    )
    projected = tl.dot(x, weight, input_precision=DOT_PRECISION)
    projected += tl.load(bias_ptr + offs_r)[None, :]

    inverse_rms = tl.rsqrt(tl.sum(projected * projected, axis=1) / R + EPS)
    scale = tl.load(rms_weight_ptr + offs_r)
    normalized = projected * inverse_rms[:, None] * scale[None, :]
    activated = _gelu_erf(normalized)
    tl.store(
        output_ptr + offs_t[:, None] * R + offs_r[None, :],
        activated,
        mask=valid_t[:, None],
    )


@triton.jit
def _tiled_owner(
    x_ptr,
    spatial_rank_ptr,
    spatial_up_weight_ptr,
    spatial_up_bias_ptr,
    value_weight_ptr,
    value_bias_ptr,
    gate_weight_ptr,
    gate_bias_ptr,
    out_weight_ptr,
    out_bias_ptr,
    residual_scale_ptr,
    output_ptr,
    tokens,
    T_BUCKET,
    DEVICE_KEY,
    EPS: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    GD: tl.constexpr,
    GH: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    offs_t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    group = tl.program_id(1)
    valid_t = offs_t < tokens
    offs_d = tl.arange(0, triton.next_power_of_2(D))
    offs_gd = tl.arange(0, triton.next_power_of_2(GD))
    offs_r = tl.arange(0, R)
    hs = tl.arange(0, BLOCK_H)
    global_d = group * GD + offs_gd

    # LayerNorm statistics are global across all D32 channels.  Affine terms
    # are folded into the packed value/gate weights and biases.
    x_full = tl.load(
        x_ptr + offs_t[:, None] * D + offs_d[None, :],
        mask=valid_t[:, None] & (offs_d[None, :] < D),
        other=0.0,
    ).to(tl.float32)
    mean = tl.sum(x_full, axis=1) / D
    centered = tl.where(offs_d[None, :] < D, x_full - mean[:, None], 0.0)
    variance = tl.sum(centered * centered, axis=1) / D
    inverse_std = tl.rsqrt(variance + EPS)
    x_group = tl.load(
        x_ptr + offs_t[:, None] * D + global_d[None, :],
        mask=valid_t[:, None] & (offs_gd[None, :] < GD),
        other=0.0,
    )
    normalized = tl.where(
        offs_gd[None, :] < GD, (x_group - mean[:, None]) * inverse_std[:, None], 0.0
    )
    normalized = normalized.to(x_ptr.dtype.element_ty)

    result = tl.full((BLOCK_T, triton.next_power_of_2(GD)), 0, tl.float32)
    for start in range(triton.cdiv(GH, BLOCK_H)):
        offs_h = start * BLOCK_H + hs
        main_offsets = group * GD * GH + offs_gd[:, None] * GH + offs_h[None, :]
        value_weight = tl.load(
            value_weight_ptr + main_offsets,
            mask=(offs_gd[:, None] < GD) & (offs_h[None, :] < GH),
            other=0.0,
        )
        gate_weight = tl.load(
            gate_weight_ptr + main_offsets,
            mask=(offs_gd[:, None] < GD) & (offs_h[None, :] < GH),
            other=0.0,
        )
        value = tl.dot(normalized, value_weight, input_precision=DOT_PRECISION)
        gate = tl.dot(normalized, gate_weight, input_precision=DOT_PRECISION)
        value += tl.load(
            value_bias_ptr + group * GH + offs_h, mask=offs_h < GH, other=0.0
        )[None, :]
        gate += tl.load(
            gate_bias_ptr + group * GH + offs_h, mask=offs_h < GH, other=0.0
        )[None, :]

        spatial_rank = tl.load(
            spatial_rank_ptr + offs_t[:, None] * R + offs_r[None, :],
            mask=valid_t[:, None],
            other=0.0,
        )
        spatial_offsets = group * R * GH + offs_r[:, None] * GH + offs_h[None, :]
        spatial_weight = tl.load(
            spatial_up_weight_ptr + spatial_offsets,
            mask=offs_h[None, :] < GH,
            other=0.0,
        )
        spatial = tl.dot(
            spatial_rank,
            spatial_weight,
            input_precision=DOT_PRECISION,
        )
        spatial += tl.load(
            spatial_up_bias_ptr + group * GH + offs_h, mask=offs_h < GH, other=0.0
        )[None, :]

        hidden = _gelu_erf(value + spatial) * tl.sigmoid(gate)
        hidden = hidden.to(x_ptr.dtype.element_ty)
        out_offsets = group * GH * GD + offs_h[:, None] * GD + offs_gd[None, :]
        out_weight = tl.load(
            out_weight_ptr + out_offsets,
            mask=(offs_h[:, None] < GH) & (offs_gd[None, :] < GD),
            other=0.0,
        )
        result += tl.dot(hidden, out_weight, input_precision=DOT_PRECISION)
    result += tl.load(
        out_bias_ptr + group * GD + offs_gd, mask=offs_gd < GD, other=0.0
    )[None, :]
    scale = tl.load(residual_scale_ptr + global_d, mask=offs_gd < GD, other=0.0)[
        None, :
    ]

    residual = tl.load(
        x_ptr + offs_t[:, None] * D + global_d[None, :],
        mask=valid_t[:, None] & (offs_gd[None, :] < GD),
        other=0.0,
    )
    output = residual + result * scale
    tl.store(
        output_ptr + offs_t[:, None] * D + global_d[None, :],
        output,
        mask=valid_t[:, None] & (offs_gd[None, :] < GD),
    )


def _token_bucket(tokens: int) -> int:
    for boundary in (256, 1024, 4096, 16384, 65536, 262144):
        if tokens <= boundary:
            return boundary
    return 1048576


def _device_key(device: torch.device) -> int:
    properties = torch.cuda.get_device_properties(device)
    return (
        properties.major * 10 + properties.minor
    ) * 1000 + properties.multi_processor_count


def halo_dsffn(module, x):
    batch, channels, height, width = x.shape
    x = x.contiguous(memory_format=torch.channels_last)
    output = torch.empty_like(x)
    grid = (batch * triton.cdiv(height, 4) * triton.cdiv(width, 8),)
    _halo[grid](
        x,
        module.norm.weight,
        module.norm.bias,
        module.down.weight,
        module.down.bias,
        module.rms_weight,
        module.dwc.weight,
        module.dwc.bias,
        module.spatial_up.weight,
        module.spatial_up.bias,
        module.group_up.weight,
        module.group_up.bias,
        module.group_down.weight,
        module.group_down.bias,
        module.residual_scale,
        output,
        height,
        width,
        module.hidden_dim,
        module.norm_eps,
        4,
        8,
        num_warps=2,
        num_stages=1,
    )
    return output


def _pack(module):
    # Fold the affine part of LayerNorm into the value/gate projections.
    weight = module.group_up.weight[:, :, 0, 0]
    value, gate = (weight * module.norm.weight).chunk(2)
    vb, gb = (module.group_up.bias + (weight * module.norm.bias).sum(dim=1)).chunk(2)
    return (
        module.down.weight[:, :, 0, 0].T.contiguous(),
        module.spatial_up.weight[:, :, 0, 0].T.contiguous(),
        value.T.contiguous(),
        vb,
        gate.T.contiguous(),
        gb,
        module.group_down.weight[:, :, 0, 0].T.contiguous(),
    )


def _packed_weights(module, device):
    sources = (
        module.down.weight,
        module.spatial_up.weight,
        module.group_up.weight,
        module.group_up.bias,
        module.norm.weight,
        module.norm.bias,
        module.group_down.weight,
    )
    if any(p.is_inference() for p in sources):
        # Inference tensors have no version counter to detect in-place updates.
        return _pack(module)
    key = (
        device,
        torch.cuda.current_stream(device).cuda_stream,
        tuple((id(p), p.data_ptr(), p._version) for p in sources),
    )
    cached = module._packed_inference
    if cached is None or cached[0] != key:
        cached = (key, _pack(module))
        module._packed_inference = cached
    return cached[1]


def wide_dsffn(module, x):
    down, spatial_weight, value, vb, gate, gb, out_weight = _packed_weights(
        module, x.device
    )
    x = x.contiguous(memory_format=torch.channels_last)
    b, d, h, w = x.shape
    n = b * h * w
    r = module.rank
    compact = torch.empty(
        (b, r, h, w), device=x.device, dtype=x.dtype, memory_format=torch.channels_last
    )
    _down_rms_predwc_gelu_kernel[lambda meta: (triton.cdiv(n, meta["BLOCK_T"]),)](
        x,
        down,
        module.down.bias,
        module.rms_weight,
        compact,
        n,
        _token_bucket(n),
        _device_key(x.device),
        module.norm_eps,
        "tf32",
        d,
        r,
    )
    compact = F.conv2d(
        compact, module.dwc.weight, module.dwc.bias, padding=1, groups=r
    ).contiguous(memory_format=torch.channels_last)
    output = torch.empty_like(x)
    _tiled_owner[(triton.cdiv(n, 16), 1)](
        x,
        compact,
        spatial_weight,
        module.spatial_up.bias,
        value,
        vb,
        gate,
        gb,
        out_weight,
        module.group_down.bias,
        module.residual_scale,
        output,
        n,
        0,
        0,
        module.norm_eps,
        "tf32",
        d,
        r,
        d,
        module.hidden_dim,
        16,
        32,
        num_warps=4,
        num_stages=1,
    )
    return output.contiguous()

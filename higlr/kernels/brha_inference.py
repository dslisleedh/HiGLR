"""B8 evaluation kernels from the paired store/recompute benchmark."""

import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_norm_window(
    Q,
    K,
    QW,
    KW,
    QB,
    KB,
    QN,
    KN,
    QNB,
    KNB,
    Z,
    A,
    B,
    G: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    QE: tl.constexpr,
    KE: tl.constexpr,
    BR: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
):
    g = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    t = tl.arange(0, T)
    d = tl.arange(0, D)
    off = ((b * G + g) * T + t[:, None]) * H * D + h * D + d[None, :]
    woff = (h * D + d[None, :]) * T + t[:, None]
    q = tl.sum(tl.load(Q + off) * tl.load(QW + woff), axis=0) + tl.load(QB + h * D + d)
    k = tl.sum(tl.load(K + off) * tl.load(KW + woff), axis=0) + tl.load(KB + h * D + d)
    qm = tl.sum(q, 0) / D
    km = tl.sum(k, 0) / D
    qz = q - qm
    kz = k - km
    qn = qz * tl.rsqrt(tl.sum(qz * qz, 0) / D + QE) * tl.load(QN + d) + tl.load(QNB + d)
    kn = kz * tl.rsqrt(tl.sum(kz * kz, 0) / D + KE) * tl.load(KN + d) + tl.load(KNB + d)
    base = ((b * H + h) * G + g) * (D + R)
    tl.store(A + base + d, qn)
    tl.store(B + base + d, kn)
    r = tl.arange(0, BR)
    z = tl.load(Z + ((b * G + g) * H + h) * R + r, r < R, other=0)
    tl.store(A + base + D + r, z, r < R)
    tl.store(B + base + D + r, z, r < R)


@triton.jit
def _pool_window(
    V,
    W,
    P,
    G: tl.constexpr,
    H: tl.constexpr,
    WEIGHTED: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
):
    g, bh = tl.program_id(0), tl.program_id(1)
    t, d = tl.arange(0, T), tl.arange(0, D)
    offsets = ((bh // H * G + g) * T + t[:, None]) * H * D + bh % H * D + d[None, :]
    v = tl.load(V + offsets)
    if WEIGHTED:
        w = tl.load(W + (bh % H * D + d[None, :]) * T + t[:, None])
        v = v * w
    else:
        v = v * (1.0 / T)
    tl.store(P + (bh * G + g) * D + d, tl.sum(v, axis=0))


@triton.jit
def _merge_window(
    F,
    C,
    A,
    O,
    N: tl.constexpr,
    G: tl.constexpr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    head = i // D % H
    token = i // (D * H)
    group, batch = token // T % G, token // (T * G)
    c = tl.load(C + ((batch * H + head) * G + group) * D + i % D, i < N, other=0.0)
    fine = tl.load(F + i, i < N, other=0.0).to(tl.float32)
    gain = tl.load(A + head)
    tl.store(O + i, fine + gain * c, i < N)


@triton.jit
def _pack_score_id(score, ids, valid):
    score = tl.where(score == 0.0, 0.0, score)
    bits = score.to(tl.uint32, bitcast=True)
    ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
    packed = (ordered.to(tl.uint64) << 32) | (0xFFFFFFFF - ids.to(tl.uint32)).to(
        tl.uint64
    )
    return tl.where(valid, packed, 0).to(tl.uint64)


@triton.jit
def _unpack_score_id(packed):
    ordered = (packed >> 32).to(tl.uint32)
    bits = tl.where((ordered & 0x80000000) != 0, ordered ^ 0x80000000, ~ordered).to(
        tl.uint32
    )
    score = bits.to(tl.float32, bitcast=True)
    ids = (0xFFFFFFFF - packed.to(tl.uint32)).to(tl.int32)
    return score, ids


@triton.jit
def _stream(
    Q,
    K,
    V,
    C,
    I,
    L,
    G: tl.constexpr,
    D: tl.constexpr,
    NR: tl.constexpr,
    RETAIN: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    ROUTE: tl.constexpr,
    COARSE: tl.constexpr,
    QK_PRECISION: tl.constexpr,
):
    """Register-resident bitonic top-k merge + online softmax accumulators."""
    bh = tl.program_id(1)
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    nk = tl.arange(0, BN)
    dr = tl.arange(0, 16)
    q = tl.load(
        Q + (bh * G + rows[:, None]) * 16 + dr[None, :], rows[:, None] < G, other=0.0
    )
    if COARSE:
        dv = tl.arange(0, D)
        cmax = tl.full((BM,), -float("inf"), tl.float32)
        csum = tl.full((BM,), 0.0, tl.float32)
        acc = tl.full((BM, D), 0.0, tl.float32)
    if ROUTE and NR > 1:
        rmax = tl.full((BM,), -float("inf"), tl.float32)
        rsum = tl.full((BM,), 0.0, tl.float32)
        best = tl.full((BM, RETAIN), 0, tl.uint64)

    for start in range(tl.cdiv(G, BN)):
        keys = start * BN + nk
        k = tl.load(
            K + (bh * G + keys[None, :]) * 16 + dr[:, None],
            keys[None, :] < G,
            other=0.0,
        )
        score = tl.dot(q, k, input_precision=QK_PRECISION) * 0.25
        if COARSE:
            cs = tl.where(keys[None, :] < G, score, -float("inf"))
            newmax = tl.maximum(cmax, tl.max(cs, 1))
            alpha = tl.exp(cmax - newmax)
            p = tl.exp(cs - newmax[:, None])
            v = tl.load(
                V + (bh * G + keys[:, None]) * D + dv[None, :],
                keys[:, None] < G,
                other=0.0,
            )
            acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="tf32x3")
            csum = csum * alpha + tl.sum(p, 1)
            cmax = newmax
        if ROUTE and NR > 1:
            valid = (keys[None, :] < G) & (keys[None, :] != rows[:, None])
            rs = tl.where(valid, score, -float("inf"))
            newmax = tl.maximum(rmax, tl.max(rs, 1))
            alpha = tl.exp(rmax - newmax)
            rsum = rsum * alpha + tl.sum(tl.exp(rs - newmax[:, None]), 1)
            rmax = newmax
            candidate = _pack_score_id(score, keys[None, :], valid)
            tile_best = tl.topk(candidate, RETAIN)
            merged = tl.reshape(tl.join(best, tile_best), (BM, 2 * RETAIN))
            best = tl.topk(merged, RETAIN)

    if COARSE:
        tl.store(
            C + (bh * G + rows[:, None]) * D + dv[None, :],
            acc / csum[:, None],
            rows[:, None] < G,
        )
    if ROUTE:
        tl.store(I + (bh * G + rows) * NR, rows, rows < G)
        tl.store(L + (bh * G + rows) * NR, 0.0, rows < G)
        if NR > 1:
            selected, ids = _unpack_score_id(best)
            lse = rmax + tl.log(rsum)
            ranks = tl.arange(0, RETAIN)
            off = (bh * G + rows[:, None]) * NR + ranks[None, :] + 1
            mask = (rows[:, None] < G) & (ranks[None, :] < NR - 1)
            tl.store(I + off, ids, mask)
            tl.store(L + off, selected - lse[:, None], mask)


@triton.jit
def _materialize_scores(
    Q, K, S, M: tl.constexpr, G: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr
):
    bh = tl.program_id(2)
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    keys = tl.program_id(1) * BN + tl.arange(0, BN)
    dr = tl.arange(0, 16)
    q = tl.load(
        Q + (bh * M + rows[:, None]) * 16 + dr[None, :], rows[:, None] < M, other=0.0
    )
    k = tl.load(
        K + (bh * G + keys[None, :]) * 16 + dr[:, None], keys[None, :] < G, other=0.0
    )
    score = tl.dot(q, k, input_precision="tf32x3") * 0.25
    tl.store(
        S + (bh * M + rows[:, None]).to(tl.int64) * G + keys[None, :],
        score,
        (rows[:, None] < M) & (keys[None, :] < G),
    )


@triton.jit
def _routes(
    S, I, L, G: tl.constexpr, R: tl.constexpr, ENABLE: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    own = row % G
    col = tl.arange(0, BLOCK)
    values = tl.load(S + row.to(tl.int64) * G + col, col < G, other=-float("inf"))
    values = tl.where(col == own, -float("inf"), values)
    if ENABLE and G > 1:
        maximum = tl.max(values, 0)
        lse = maximum + tl.log(tl.sum(tl.exp(values - maximum), 0))
    else:
        lse = 0.0
    tl.store(I + row * R, own)
    tl.store(L + row * R, 0.0)
    for j in range(1, R):
        maximum = tl.max(values, 0)
        index = tl.min(tl.where((col < G) & (values == maximum), col, 2147483647), 0)
        tl.store(I + row * R + j, index)
        tl.store(L + row * R + j, maximum - lse if ENABLE else 0.0)
        values = tl.where(col == index, -float("inf"), values)


@triton.jit
def _flash_coarse_forward(
    S, V, C, LSE, G: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr
):
    bh = tl.program_id(1)
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols, dims = tl.arange(0, BK), tl.arange(0, 64)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denom = tl.full((BM,), 0.0, tl.float32)
    acc = tl.full((BM, 64), 0.0, tl.float32)
    for start in range(tl.cdiv(G, BK)):
        keys = start * BK + cols
        s = tl.load(
            S + (bh * G + rows[:, None]).to(tl.int64) * G + keys[None, :],
            (rows[:, None] < G) & (keys[None, :] < G),
            other=-float("inf"),
        )
        s = tl.where(rows[:, None] < G, s, 0.0)
        newmax = tl.maximum(maximum, tl.max(s, 1))
        rescale = tl.exp(maximum - newmax)
        p = tl.exp(s - newmax[:, None])
        v = tl.load(
            V + (bh * G + keys[:, None]) * 64 + dims[None, :],
            keys[:, None] < G,
            other=0.0,
        )
        acc = acc * rescale[:, None] + tl.dot(p, v, input_precision="tf32x3")
        denom = denom * rescale + tl.sum(p, 1)
        maximum = newmax
    tl.store(
        C + (bh * G + rows[:, None]) * 64 + dims[None, :],
        acc / denom[:, None],
        rows[:, None] < G,
    )
    tl.store(LSE + bh * G + rows, maximum + tl.log(denom), rows < G)


def route_and_pool(query, key, pooled, remote_k, routing_backend):
    batch, heads, groups, _ = query.shape
    routes = min(remote_k, groups - 1) + 1
    coarse = torch.empty_like(pooled)
    ids = torch.empty(
        (batch, heads, groups, routes), device=query.device, dtype=torch.int32
    )
    gates = torch.empty_like(ids, dtype=torch.float32)
    grid = (triton.cdiv(groups, 16), batch * heads)
    # Recompute was slightly faster on RTX 4090. Storing the scores may win
    # on GPUs with more memory bandwidth (e.g. H200); benchmark the target shape.
    if routing_backend == "store":
        scores = query.new_empty((batch, heads, groups, groups))
        _materialize_scores[(grid[0], triton.cdiv(groups, 128), grid[1])](
            query, key, scores, groups, groups, 16, 128, num_warps=4, num_stages=1
        )
        lse = query.new_empty((batch, heads, groups))
        _flash_coarse_forward[grid](scores, pooled, coarse, lse, groups, 16, 64)
        _routes[(batch * heads * groups,)](
            scores,
            ids,
            gates,
            groups,
            routes,
            True,
            triton.next_power_of_2(groups),
            num_warps=4,
        )
    else:
        retain = triton.next_power_of_2(max(routes - 1, 1))
        args = (
            query,
            key,
            pooled,
            coarse,
            ids,
            gates,
            groups,
            64,
            routes,
            retain,
            16,
            128,
        )
        _stream[grid](*args, True, False, "tf32x3", num_warps=4, num_stages=1)
        _stream[grid](*args, False, True, "tf32x3", num_warps=4, num_stages=1)
    return coarse, ids, gates


def routed_attention(module, q, k, v, condition, dtype):
    from .b8_backward_v3 import Fine

    batch, tokens, heads, dim = q.shape
    area = module.block**2
    groups = tokens // area
    grid = (groups, batch * heads)
    pooled = v.new_empty((batch, heads, groups, dim))
    _pool_window[grid](
        v, module.coarse_v.weight, pooled, groups, heads, True, area, dim
    )
    query_input = q.new_empty((batch, heads, groups, dim + module.router_dim))
    key_input = torch.empty_like(query_input)
    qw, kw = module.reduce_q.projection, module.reduce_k.projection
    qn, kn = module.q_summary_norm, module.k_summary_norm
    embedding = module.global_feature_projection(condition).contiguous()
    _reduce_norm_window[grid](
        q,
        k,
        qw.weight,
        kw.weight,
        qw.bias,
        kw.bias,
        qn.weight,
        kn.weight,
        qn.bias,
        kn.bias,
        embedding,
        query_input,
        key_input,
        groups,
        heads,
        module.router_dim,
        qn.eps,
        kn.eps,
        module.router_dim,
        area,
        dim,
        num_warps=4,
    )
    coarse, ids, gates = route_and_pool(
        module.query_fusion(query_input).contiguous(),
        module.key_fusion(key_input).contiguous(),
        pooled,
        module.topk,
        module.routing_backend,
    )
    fine = Fine.apply(
        q.to(dtype),
        (k - k.mean(1, keepdim=True)).to(dtype).contiguous(),
        v,
        pooled,
        ids,
        gates,
        module.relative_bias,
        module.block,
        True,
    )
    mixed = torch.empty_like(fine)
    _merge_window[(triton.cdiv(fine.numel(), 512),)](
        fine,
        coarse,
        module.coarse_gain,
        mixed,
        fine.numel(),
        groups,
        heads,
        512,
        area,
        dim,
    )
    return mixed

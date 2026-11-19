from copy import deepcopy

import torch

from higlr import HiGLR
from higlr.attention import BRHA, fine_reference, select_routes
from higlr.layers import DenseSA, DSFFN


torch.set_num_threads(2)


def test_architecture_and_backward():
    model = HiGLR(backend='torch')
    assert sum(p.numel() for p in model.parameters()) == 7190940
    assert sum(isinstance(m, BRHA) for m in model.modules()) == 10
    assert sum(isinstance(m, DenseSA) for m in model.modules()) == 5
    assert sum(isinstance(m, DSFFN) for m in model.modules()) == 35
    x = torch.rand(1, 3, 33, 41)
    restored, logits = model(x, return_scores=True)
    assert restored.shape == x.shape and logits.shape == (1, 11)
    (restored.square().mean() + logits.square().mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    assert all(m.topk == 16 for m in model.modules() if isinstance(m, BRHA))
    model.train()
    assert all(m.topk == 32 for m in model.modules() if isinstance(m, BRHA))


def test_sparse_values_against_independent_dense_mask():
    torch.manual_seed(7)
    b, h, g, t, d = 1, 2, 4, 64, 64
    q, k, v = [torch.randn(b, g*t, h, d, requires_grad=True) for _ in range(3)]
    pooled = torch.randn(b, h, g, d, requires_grad=True)
    scores = torch.randn(b, h, g, g, requires_grad=True)
    bias = torch.randn(h, 225, requires_grad=True)
    ids, gates = select_routes(scores, 2)
    actual = fine_reference(q, k, v, pooled, ids, gates, bias)
    mask = torch.full_like(scores, -torch.inf).scatter(-1, ids.long(), gates)
    mask = mask.repeat_interleave(t, -2).repeat_interleave(t, -1)
    p = torch.arange(g*t) % t
    rp = (p[:, None]//8 - p[None, :]//8 + 7)*15 + p[:, None]%8 - p[None, :]%8 + 7
    logits = q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2) / 8
    residual_v = v.transpose(1, 2) - pooled.repeat_interleave(t, 2)
    expected = ((logits + mask + bias[:, rp][None]).softmax(-1) @ residual_v).transpose(1, 2)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    variables = (q, k, v, pooled, scores, bias)
    upstream = torch.randn_like(actual)
    a = torch.autograd.grad(actual, variables, upstream, retain_graph=True)
    e = torch.autograd.grad(expected, variables, upstream)
    for av, ev in zip(a, e):
        torch.testing.assert_close(av, ev, atol=8e-6, rtol=8e-5)


def test_one_block_routes_and_all_task_heads():
    score = torch.randn(1, 1, 1, 1, requires_grad=True)
    ids, gates = select_routes(score, 32)
    assert ids.item() == gates.item() == 0
    for classes in (3, 5):
        model = HiGLR(backend='torch', num_classes=classes).eval()
        with torch.no_grad():
            image, logits = model(torch.rand(1, 3, 17, 25), return_scores=True)
        assert image.shape == (1, 3, 17, 25)
        assert logits.shape == (1, classes)


def test_initial_block_summaries_are_trainable_means():
    model = HiGLR(backend='torch', routing_backend='store')
    reductions = list(model.net.encoder.reducers.values())
    for module in model.modules():
        if isinstance(module, BRHA):
            assert module.routing_backend == 'store'
            reductions.extend((module.reduce_q.projection, module.reduce_k.projection, module.coarse_v))
    assert len(reductions) == 32
    for conv in reductions:
        assert conv.weight.requires_grad
        torch.testing.assert_close(conv.weight, torch.full_like(conv.weight, 1 / 64), atol=0, rtol=0)
        if conv.bias is not None:
            torch.testing.assert_close(conv.bias, torch.zeros_like(conv.bias), atol=0, rtol=0)

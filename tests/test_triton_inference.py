import pytest
import torch

from higlr.layers import DSFFN

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")


@pytest.mark.parametrize("groups,budget", [(1, 32), (17, 0), (145, 16), (145, 64)])
def test_store_and_recompute_match_dense_blocks(groups, budget):
    from higlr.kernels.brha_inference import route_and_pool
    generator = torch.Generator(device='cuda').manual_seed(29)
    # Small integers make scores exact, including ties across different tiles.
    q, k = [torch.randint(-3, 4, (2, 2, groups, 16), device='cuda',
                          generator=generator).float() for _ in range(2)]
    pooled = torch.randn(2, 2, groups, 64, device='cuda', generator=generator)
    scores = q @ k.transpose(-1,-2) / 4
    ref = (scores.double().softmax(-1) @ pooled.double()).float()
    scores.diagonal(dim1=-2,dim2=-1).fill_(-torch.inf)
    count = min(budget, groups - 1)
    selected = scores.argsort(dim=-1, descending=True, stable=True)[...,:count]
    with torch.inference_mode():
        for routing in ('store','recompute'):
            coarse, ids, gates = route_and_pool(q,k,pooled,budget,routing)
            torch.testing.assert_close(coarse, ref, atol=8e-6, rtol=4e-5)
            torch.testing.assert_close(ids[...,0].long(), torch.arange(groups,device='cuda').expand(2,2,-1))
            assert torch.count_nonzero(gates[...,0]) == 0
            torch.testing.assert_close(ids[...,1:].long(), selected)
            torch.testing.assert_close(gates[...,1:], scores.log_softmax(-1).gather(-1,selected),
                                       atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("width", [32,128,256])
def test_inference_uses_current_weights_and_eval_supports_gradients(width):
    fused = DSFFN(width, backend='triton').cuda().eval()
    ref = DSFFN(width, backend='torch').cuda().eval()
    x = torch.randn(2, width, 9, 13, device='cuda')
    with torch.no_grad():
        for _ in range(2):
            ref.load_state_dict(fused.state_dict())
            torch.testing.assert_close(fused(x), ref(x), rtol=.001, atol=2e-4)
            # Force cache invalidation with in-place optimizer-style weight updates.
            fused.norm.bias.add_(1)
            fused.group_up.weight.mul_(1.5)
        fused.load_state_dict(DSFFN(width).state_dict())
        ref.load_state_dict(fused.state_dict())
        torch.testing.assert_close(fused(x), ref(x), rtol=.001, atol=2e-4)
    fused(x).mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in fused.parameters())

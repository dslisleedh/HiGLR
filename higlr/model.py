"""Four-stage HiGLR restoration network."""
import torch
from torch import nn
from torch.nn import functional as F

from .attention import BRHA
from .layers import DSFFN, DenseSA, LayerNorm2d, block_mean_projection, triton


class AttentionDSFFN(nn.Module):
    def __init__(self, channels: int, backend: str, attention_dtype: str, remote_k: int):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        heads = channels // 64
        if channels == 256:
            self.attention = DenseSA(channels, heads, attention_dtype)
        else:
            self.attention = BRHA(channels, heads, attention_dtype, backend, remote_k)
        self.sa_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.01))
        self.rms = DSFFN(channels, backend)

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        feat = self.norm1(x)
        if condition is None:
            feat = self.attention(feat)
        else:
            feat = self.attention(feat, condition)
        x = x + feat * self.sa_scale
        return self.rms(x).contiguous()


class DSFFNStack(nn.Module):
    def __init__(self, depth: int, backend: str, dense5: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList([DSFFN(32, backend) for _ in range(depth)])
        self.dense5 = dense5
        if dense5:
            self.projections = nn.ModuleList([nn.Conv2d(32, 32, 1, bias=False) for _ in range(5)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous(memory_format=torch.channels_last)
        if self.dense5:
            fused = self.projections[0](x)
        for i, block in enumerate(self.blocks, 1):
            x = block(x)
            if self.dense5 and i % 4 == 0:
                fused = fused + self.projections[i // 4](x)
        if self.dense5:
            return fused.contiguous()
        return x.contiguous()


class Encoder(nn.Module):
    def __init__(self, backend: str, attention_dtype: str, remote_k: int):
        super().__init__()
        self.stages = nn.ModuleList([nn.Identity() for _ in range(4)])
        self.rms_h = DSFFNStack(4, backend)
        self.downs = nn.ModuleList([
            nn.Conv2d(c, 2 * c, 2, stride=2) for c in (32, 64, 128)
        ])
        self.embeddings = nn.ModuleDict({
            '1': nn.Conv2d(64, 128, 3, padding=1),
            '2': nn.Conv2d(128, 128, 3, padding=1),
        })
        self.reducers = nn.ModuleDict({stage: block_mean_projection(128, 8) for stage in self.embeddings})
        self.attention = nn.ModuleDict({'0': nn.ModuleList()})
        for i, channels in enumerate((64, 128, 256), 1):
            self.attention[str(i)] = nn.ModuleList([
                AttentionDSFFN(channels, backend, attention_dtype, remote_k) for _ in range(4)
            ])

    def forward(self, x: torch.Tensor, need_scores: bool):
        x = self.rms_h(x)
        skips = [x]
        conditions = []
        pooled = []
        for i in (1, 2):
            stage = str(i)
            x = self.downs[i - 1](x)
            embedding = self.embeddings[stage](x)
            if need_scores:
                pooled.append(embedding.mean((-2, -1)))
            condition = self.reducers[stage](embedding).flatten(2).transpose(1, 2).contiguous()
            for block in self.attention[stage]:
                x = block(x, condition)
            skips.append(x)
            conditions.append(condition)

        x = self.downs[2](x)
        for block in self.attention['3']:
            x = block(x)
        return x, skips, conditions, pooled


class DecoderStage(nn.Module):
    def __init__(self, channels: int, backend: str, attention_dtype: str, remote_k: int):
        super().__init__()
        if channels == 256:
            self.up = nn.Identity()
        else:
            self.up = nn.Sequential(
                nn.Conv2d(2 * channels, 4 * channels, 1, bias=False),
                nn.PixelShuffle(2),
            )
        self.naf = nn.Identity()
        self.blocks = nn.ModuleList()
        if channels == 32:
            self.rms = DSFFNStack(16, backend, dense5=True)
        else:
            self.blocks.append(AttentionDSFFN(channels, backend, attention_dtype, remote_k))
            self.rms = nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(x) + skip
        for block in self.blocks:
            x = block(x, condition)
        return self.rms(x)


class Backbone(nn.Module):
    def __init__(self, backend: str, attention_dtype: str, remote_k: int):
        super().__init__()
        self.intro = nn.Conv2d(3, 32, 3, padding=1)
        self.encoder = Encoder(backend, attention_dtype, remote_k)
        self.bottleneck = nn.Identity()
        self.decoder_h8 = DecoderStage(256, backend, attention_dtype, remote_k)
        self.decoder_h4 = DecoderStage(128, backend, attention_dtype, remote_k)
        self.decoder_h2 = DecoderStage(64, backend, attention_dtype, remote_k)
        self.decoder_h = DecoderStage(32, backend, attention_dtype, remote_k)
        self.ending = nn.Conv2d(32, 3, 3, padding=1)
        nn.init.constant_(self.ending.bias, 0.5)

    def forward(self, x: torch.Tensor, need_scores: bool):
        x = self.intro(x)
        x, (skip_h, skip_h2, skip_h4), (cond_h2, cond_h4), pooled = self.encoder(x, need_scores)

        # Checkpoints store the fifth H/8 encoder block under decoder_h8.
        for block in self.decoder_h8.blocks:
            x = block(x)
        x = self.decoder_h4(x, skip_h4, cond_h4)
        x = self.decoder_h2(x, skip_h2, cond_h2)
        x = self.decoder_h(x, skip_h)
        return self.ending(x), pooled


class HiGLR(nn.Module):
    class_order = tuple(sorted((
        'low', 'haze', 'rain', 'snow', 'low_haze', 'low_rain', 'low_snow',
        'haze_rain', 'haze_snow', 'low_haze_rain', 'low_haze_snow',
    )))

    def __init__(
        self, *, num_classes: int = 11, remote_k: int = 32, eval_remote_k: int | None = 16,
        backend: str = 'triton', attention_dtype: str = 'auto', route_chunk_size: int = 512,
        routing_backend: str = 'recompute',
    ):
        super().__init__()
        if backend not in ('torch', 'triton'):
            raise ValueError('backend must be torch or triton')
        if attention_dtype not in ('auto', 'float32', 'bfloat16'):
            raise ValueError('Invalid attention dtype')
        if type(num_classes) is not int or num_classes < 0:
            raise ValueError('num_classes must be nonnegative')
        if eval_remote_k is None:
            eval_remote_k = remote_k
        if any(type(k) is not int or not 0 <= k <= 64 for k in (remote_k, eval_remote_k)):
            raise ValueError('remote_k and eval_remote_k must be integers in [0,64]')
        if type(route_chunk_size) is not int or route_chunk_size < 1:
            raise ValueError('Positive route_chunk_size required')
        if routing_backend not in ('recompute', 'store'):
            raise ValueError('routing_backend must be recompute or store')

        self.net = Backbone(backend, attention_dtype, remote_k)
        for layer in self.net.modules():
            if isinstance(layer, BRHA):
                layer.routing_backend = routing_backend

        self.num_classes = num_classes
        self.fusion = nn.Sequential(nn.Linear(256, 128), nn.GELU()) if num_classes else nn.Identity()
        self.classifier = nn.Linear(128, num_classes) if num_classes else nn.Identity()
        self.config = dict(
            model_variant='higlr_d32_gamma3_b8_dense5', width=32, rms_expansion=3,
            rms_stage_ranks=[16, 16, 32, 64], ffn_count=35, encoder_depths=[4, 4, 4, 4],
            decoder_depths=[16, 1, 1, 1], head_dim=64, window=8, remote_k=remote_k,
            eval_remote_k=eval_remote_k, backend=backend, attention_dtype=attention_dtype,
            routing='8x8 block-shared routing', route_chunk_size=route_chunk_size,
            routing_backend=routing_backend,
            coarse_routing='original pooled Q/K block attention; broadcast output to64 queries',
            log_gate='per-block all-remote logsoftmax; own=0', fine_value='V-pool(V)',
            dense5_taps=[0, 4, 8, 12, 16], degradation='original H2/H4 conditioning + optional CE',
            num_classes=num_classes, ending_bias_init=0.5, activation_checkpoint=False, ema=False,
            bank=False, image_skip=False, padding_multiple=32, initialization='standalone draw order',
        )
        self.variant = self.config['model_variant']

    def train(self, mode: bool = True):
        super().train(mode)
        k = self.config['remote_k'] if mode else self.config['eval_remote_k']
        for layer in self.net.modules():
            if isinstance(layer, BRHA):
                layer.topk = k
        return self

    def forward(
        self, x: torch.Tensor, clean: torch.Tensor | None = None, ids: torch.Tensor | None = None,
        *, mode: str = 'inference', return_scores: bool = False,
    ):
        if x.ndim != 4 or x.shape[1] != 3 or min(x.shape) < 1:
            raise ValueError('Nonempty Bx3xHxW required')
        if x.dtype != torch.float32 or next(self.parameters()).dtype != torch.float32:
            raise ValueError('Keep input and model FP32')
        if torch.is_autocast_enabled(x.device.type):
            raise ValueError('Outer autocast is unsupported')
        if self.config['backend'] == 'triton' and (not x.is_cuda or triton is None):
            raise ValueError('Default training backend requires CUDA/Triton; CPU checks use backend="torch"')
        if mode not in ('inference', 'train256', 'pretrain', 'finetune'):
            raise ValueError('Unknown mode')
        if mode != 'inference' and (clean is None or ids is None):
            raise ValueError('Training API requires targets and IDs')
        need_scores = return_scores or mode != 'inference'
        if need_scores and not self.num_classes:
            raise ValueError('Scores require num_classes>0')

        h, w = x.shape[-2:]
        with torch.cuda.device_of(x):
            if h % 32 or w % 32:
                x = F.pad(x, (0, (-w) % 32, 0, (-h) % 32))
            restored, pooled = self.net(x, need_scores)
            restored = restored[:, :, :h, :w]
            if not need_scores:
                return restored
            scores = self.classifier(self.fusion(torch.cat(pooled, dim=1)))

        if mode == 'inference':
            return restored, scores
        return dict(restored=restored, clean_target=clean, scores=scores, query_class=ids)

    def load_restoration_weights(self, state):
        """Load matching B8/gamma3 weights with strict key and shape checks."""
        current = self.state_dict()
        if set(state) != set(current):
            raise ValueError('Expected the exact gamma3 parent state keys, including coarse query reducers')
        for name, value in current.items():
            if state[name].shape != value.shape:
                raise ValueError('Incompatible tensor: ' + name)
        return self.load_state_dict(state, strict=True)


def build_model(**kwargs):
    return HiGLR(**kwargs)

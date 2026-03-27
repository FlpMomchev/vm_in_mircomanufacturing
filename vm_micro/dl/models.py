from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .config import TrainConfig
from .frontends import build_frontend


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, pool: bool = True) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.GELU(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.GELU(),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvNormAct(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1, drop: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(drop) if drop > 0.0 else nn.Identity()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.act(out + identity)
        return out


class SpecAugment2D(nn.Module):
    def __init__(
        self,
        time_masks: int = 2,
        time_width: int = 8,
        freq_masks: int = 2,
        freq_width: int = 12,
    ) -> None:
        super().__init__()
        self.time_masks = int(time_masks)
        self.time_width = int(time_width)
        self.freq_masks = int(freq_masks)
        self.freq_width = int(freq_width)

    def _mask_along_axis(
        self, x: torch.Tensor, axis: int, n_masks: int, max_width: int
    ) -> torch.Tensor:
        if n_masks <= 0 or max_width <= 0:
            return x

        out = x.clone()
        size = out.shape[axis]
        if size <= 2:
            return out

        for _ in range(n_masks):
            width = int(torch.randint(0, max_width + 1, (1,), device=out.device).item())
            if width <= 0:
                continue
            width = min(width, size)
            start_max = max(1, size - width + 1)
            start = int(torch.randint(0, start_max, (1,), device=out.device).item())
            slicer = [slice(None)] * out.ndim
            slicer[axis] = slice(start, start + width)
            out[tuple(slicer)] = 0.0
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._mask_along_axis(x, axis=-1, n_masks=self.time_masks, max_width=self.time_width)
        x = self._mask_along_axis(x, axis=-2, n_masks=self.freq_masks, max_width=self.freq_width)
        return x


class SinCos2DPositionalEncoding(nn.Module):
    def __init__(self, dim: int, h_tokens: int, w_tokens: int) -> None:
        super().__init__()
        pe = self._build(dim=dim, h_tokens=h_tokens, w_tokens=w_tokens)
        self.register_buffer("pe", pe, persistent=True)

    @staticmethod
    def _build(dim: int, h_tokens: int, w_tokens: int) -> torch.Tensor:
        if dim % 4 != 0:
            raise ValueError("Transformer dim must be divisible by 4 for 2D sin-cos encoding.")

        half = dim // 2
        quarter = half // 2

        y = torch.arange(h_tokens, dtype=torch.float32)
        x = torch.arange(w_tokens, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        omega = torch.arange(quarter, dtype=torch.float32)
        omega = 1.0 / (10000 ** (omega / max(1, quarter)))

        out_y = yy.reshape(-1, 1) * omega.reshape(1, -1)
        out_x = xx.reshape(-1, 1) * omega.reshape(1, -1)
        pos = torch.cat([out_y.sin(), out_y.cos(), out_x.sin(), out_x.cos()], dim=1)
        return pos.unsqueeze(0)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.pe[:, : tokens.shape[1], :]


class AttentionPooling(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn = torch.softmax(self.score(tokens).squeeze(-1), dim=1)
        pooled = torch.sum(tokens * attn.unsqueeze(-1), dim=1)
        return pooled, attn


class SmallSpecCNNBackbone(nn.Module):
    def __init__(self, embed_dim: int, dropout: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 16, pool=True),
            ConvBlock(16, 32, pool=True),
            ConvBlock(32, 64, pool=True),
            ConvBlock(64, 96, pool=False),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, spec: torch.Tensor) -> dict[str, torch.Tensor]:
        fmap = self.features(spec)
        embedding = self.proj(self.pool(fmap))
        return {
            "embedding": embedding,
            "feature_map": fmap,
        }


class SpecResNetBackbone(nn.Module):
    def __init__(self, embed_dim: int, dropout: float) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(1, 32, stride=2),
            ResidualConvBlock(32, 32, stride=1, drop=dropout * 0.25),
            ResidualConvBlock(32, 64, stride=2, drop=dropout * 0.25),
            ResidualConvBlock(64, 96, stride=2, drop=dropout * 0.50),
            ResidualConvBlock(96, 128, stride=2, drop=dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, spec: torch.Tensor) -> dict[str, torch.Tensor]:
        fmap = self.stem(spec)
        embedding = self.proj(self.pool(fmap))
        return {
            "embedding": embedding,
            "feature_map": fmap,
        }


class HybridSpecTransformerBackbone(nn.Module):
    def __init__(
        self, embed_dim: int, token_grid_size: int, transformer_layers: int, dropout: float
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(1, 32, stride=2),
            ResidualConvBlock(32, 64, stride=2, drop=dropout * 0.5),
            ResidualConvBlock(64, 96, stride=2, drop=dropout * 0.5),
            ResidualConvBlock(96, embed_dim, stride=1, drop=dropout),
        )
        self.token_pool = nn.AdaptiveAvgPool2d((token_grid_size, token_grid_size))
        self.positional = SinCos2DPositionalEncoding(embed_dim, token_grid_size, token_grid_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=max(2, embed_dim // 32),
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-LN: more stable gradients on small datasets
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.pool = AttentionPooling(embed_dim)

    def forward(self, spec: torch.Tensor) -> dict[str, torch.Tensor | tuple[int, int]]:
        fmap = self.stem(spec)
        tokens_2d = self.token_pool(fmap)
        bsz, channels, h_tokens, w_tokens = tokens_2d.shape
        tokens = tokens_2d.flatten(2).transpose(1, 2)
        tokens = self.positional(tokens)
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        embedding, attn = self.pool(tokens)
        return {
            "embedding": embedding,
            "token_attention": attn,
            "token_grid_hw": (h_tokens, w_tokens),
            "feature_map": fmap,
            "tokens": tokens,
        }


class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.head(embedding)


class RegressionHead(nn.Module):
    """
    Deeper regression head with physically constrained output.

    Output is passed through sigmoid and rescaled to (DEPTH_MIN, DEPTH_MAX),
    which for this dataset is (0.1, 1.0) mm.  This prevents the model from
    ever predicting physically impossible depths (< 0 or > plate thickness)
    and gives the final linear layer a bounded target, which stabilises
    training compared to an unconstrained scalar.

    Architecture: embed_dim  hidden  hidden//2  1
    Dropout is applied only on the first two layers; the final projection is
    kept clean so the sigmoid transform is not regularised away.
    """

    DEPTH_MIN: float = 0.1
    DEPTH_MAX: float = 1.0

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        mid_dim = max(32, hidden_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),  # lighter dropout near output
            nn.Linear(mid_dim, 1),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        raw = self.net(embedding).squeeze(-1)
        # sigmoid maps (-inf, inf)  (0, 1); rescale to physical depth range
        depth_range = self.DEPTH_MAX - self.DEPTH_MIN
        return torch.sigmoid(raw) * depth_range + self.DEPTH_MIN


class DepthModel(nn.Module):
    def __init__(self, cfg: TrainConfig, out_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.frontend = build_frontend(cfg)
        self.specaugment = SpecAugment2D(
            time_masks=cfg.specaugment_time_masks,
            time_width=cfg.specaugment_time_width,
            freq_masks=cfg.specaugment_freq_masks,
            freq_width=cfg.specaugment_freq_width,
        )

        if cfg.model_type == "small_cnn":
            self.backbone = SmallSpecCNNBackbone(
                embed_dim=cfg.backbone_embed_dim,
                dropout=cfg.dropout,
            )
        elif cfg.model_type == "spec_resnet":
            self.backbone = SpecResNetBackbone(
                embed_dim=cfg.backbone_embed_dim,
                dropout=cfg.dropout,
            )
        elif cfg.model_type == "hybrid_spec_transformer":
            self.backbone = HybridSpecTransformerBackbone(
                embed_dim=cfg.backbone_embed_dim,
                token_grid_size=cfg.transformer_token_grid_size,
                transformer_layers=cfg.transformer_layers,
                dropout=cfg.dropout,
            )
        else:
            raise ValueError(f"Unsupported model_type: {cfg.model_type}")

        hidden_dim = max(64, cfg.backbone_embed_dim)
        if cfg.task == "classification":
            self.head = ClassificationHead(
                in_dim=cfg.backbone_embed_dim,
                hidden_dim=hidden_dim,
                num_classes=out_dim,
                dropout=cfg.dropout,
            )
        elif cfg.task == "regression":
            self.head = RegressionHead(
                in_dim=cfg.backbone_embed_dim,
                hidden_dim=hidden_dim,
                dropout=cfg.dropout,
            )
        else:
            raise ValueError(f"Unsupported task: {cfg.task}")

    def extract_spec(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = self.frontend(waveform.float())
        if self.training and (
            self.cfg.specaugment_time_masks > 0 or self.cfg.specaugment_freq_masks > 0
        ):
            spec = self.specaugment(spec)
        if self.cfg.channels_last:
            spec = spec.contiguous(memory_format=torch.channels_last)
        else:
            spec = spec.contiguous()
        return spec

    def forward_from_spec(
        self,
        spec: torch.Tensor,
        return_extras: bool = False,
    ) -> dict[str, Any]:
        encoded = self.backbone(spec)
        embedding = encoded["embedding"]
        out: dict[str, Any] = {"embedding": embedding}

        if self.cfg.task == "classification":
            logits = self.head(embedding)
            probs = torch.softmax(logits, dim=1)
            out.update(
                {
                    "logits": logits,
                    "class_probs": probs,
                    "pred_class": probs.argmax(dim=1),
                }
            )
        else:
            regression = self.head(embedding)
            out.update({"regression": regression})

        if return_extras:
            for key, value in encoded.items():
                if key != "embedding":
                    out[key] = value
        return out

    def forward(self, waveform: torch.Tensor, return_extras: bool = False) -> dict[str, Any]:
        spec = self.extract_spec(waveform)
        out = self.forward_from_spec(spec, return_extras=return_extras)
        if return_extras:
            out["spec"] = spec
        return out

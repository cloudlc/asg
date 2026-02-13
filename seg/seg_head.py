from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_group_norm(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    groups = min(max_groups, num_channels)
    while groups > 1 and (num_channels % groups) != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)

class UPerNetTokenHead(nn.Module):
    """
    UPerNet-style head for ViT token features (multi-level).

    Expected features order: low-level (highest resolution) -> high-level.
    Each entry can be tokens (B, N, C) or feature maps (B, C, H, W).
    """
    def __init__(
        self,
        embed_dims: Sequence[int],
        num_classes: int,
        *,
        grid_size: Optional[tuple] = None,
        grid_sizes: Optional[Sequence[tuple]] = None,
        out_size: Optional[tuple] = None,
        fpn_channels: int = 256,
        ppm_bins=(1, 2, 3, 6),
        dropout: float = 0.1,
        norm: str = "gn",
        align_corners: bool = False,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.grid_sizes = grid_sizes
        self.out_size = out_size
        self.align_corners = align_corners
        self.ppm_bins = tuple(ppm_bins)

        def norm2d(c: int):
            if norm == "bn":
                return nn.BatchNorm2d(c)
            if norm == "gn":
                return _make_group_norm(c)
            raise ValueError(f"Unknown norm='{norm}', use 'bn' or 'gn'.")

        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_c, fpn_channels, kernel_size=1, bias=False),
                norm2d(fpn_channels),
                nn.ReLU(inplace=True),
            )
            for in_c in embed_dims
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
                norm2d(fpn_channels),
                nn.ReLU(inplace=True),
            )
            for _ in embed_dims
        ])

        self.ppm = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, kernel_size=1, bias=False),
                norm2d(fpn_channels),
                nn.ReLU(inplace=True),
            )
            for _ in self.ppm_bins
        ])
        ppm_in = fpn_channels * (1 + len(self.ppm_bins))
        self.ppm_bottleneck = nn.Sequential(
            nn.Conv2d(ppm_in, fpn_channels, kernel_size=3, padding=1, bias=False),
            norm2d(fpn_channels),
            nn.ReLU(inplace=True),
        )

        fuse_in = fpn_channels * len(embed_dims)
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_in, fpn_channels, kernel_size=3, padding=1, bias=False),
            norm2d(fpn_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.classifier = nn.Conv2d(fpn_channels, num_classes, kernel_size=1)

    def _tokens_to_map(self, x: torch.Tensor, grid_hw: tuple) -> torch.Tensor:
        if x.dim() == 4:
            return x
        B, N, C = x.shape
        Hp, Wp = grid_hw
        assert N == Hp * Wp, f"Expected N={Hp*Wp} tokens, got N={N}"
        return x.transpose(1, 2).contiguous().view(B, C, Hp, Wp)

    def forward(self, features, *, grid_sizes=None, out_size=None) -> torch.Tensor:
        if grid_sizes is None:
            grid_sizes = self.grid_sizes
        maps = []
        for i, feat in enumerate(features):
            if feat.dim() == 4:
                maps.append(feat)
                continue
            if grid_sizes is not None:
                grid_hw = grid_sizes[i]
            elif self.grid_size is not None:
                grid_hw = self.grid_size
            else:
                raise ValueError("grid_size(s) required for token inputs.")
            maps.append(self._tokens_to_map(feat, grid_hw))

        laterals = [conv(m) for conv, m in zip(self.lateral_convs, maps)]

        top = laterals[-1]
        ppm_outs = [top]
        for bin_sz, ppm_conv in zip(self.ppm_bins, self.ppm):
            pooled = F.adaptive_avg_pool2d(top, output_size=(bin_sz, bin_sz))
            pooled = ppm_conv(pooled)
            up = F.interpolate(pooled, size=top.shape[2:], mode="bilinear", align_corners=self.align_corners)
            ppm_outs.append(up)
        laterals[-1] = self.ppm_bottleneck(torch.cat(ppm_outs, dim=1))

        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i - 1].shape[2:], mode="bilinear",
                               align_corners=self.align_corners)
            laterals[i - 1] = laterals[i - 1] + up

        fpn_outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        base_size = fpn_outs[0].shape[2:]
        fused = [fpn_outs[0]]
        for feat in fpn_outs[1:]:
            fused.append(F.interpolate(feat, size=base_size, mode="bilinear", align_corners=self.align_corners))
        x = torch.cat(fused, dim=1)
        x = self.fuse(x)
        logits = self.classifier(x)

        out_size = out_size if out_size is not None else self.out_size
        if out_size is not None:
            logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=self.align_corners)
        return logits

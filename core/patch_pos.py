# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cong Liu, Yancheng Institute of Technology

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional, Dict, Tuple

LossType = Literal["mse", "smooth_l1", "l1"]
class PatchRowColRegressionCriterion(nn.Module):
    def __init__(
        self,
        feat_dim,
        grid_h,
        grid_w,
        normalize=True,
        huber_beta=None,
        loss_type: LossType = "smooth_l1",
    ):
        """
        Predict row and column index of each patch via regression (single resolution).

        Args:
            feat_dim (int): Dimension of patch features (D)
            grid_h (int): Number of patch rows (fixed)
            grid_w (int): Number of patch columns (fixed)
            normalize (bool): If True, normalize row/col targets to [0, 1]
            huber_beta (float|None): SmoothL1 beta (only used when loss_type="smooth_l1")
            loss_type (str): "l1", "smooth_l1", or "mse"
        """
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.normalize = normalize

        self.row_mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.col_mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        if loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        elif loss_type == "smooth_l1":
            if huber_beta is None:
                self.loss_fn = nn.SmoothL1Loss()
            else:
                self.loss_fn = nn.SmoothL1Loss(beta=huber_beta)
        elif loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        rows_2d = torch.arange(grid_h, dtype=torch.float32).unsqueeze(1).repeat(1, grid_w)
        cols_2d = torch.arange(grid_w, dtype=torch.float32).unsqueeze(0).repeat(grid_h, 1)

        if normalize:
            rows_2d = rows_2d / (grid_h - 1)
            cols_2d = cols_2d / (grid_w - 1)

        row_targets = rows_2d.flatten()
        col_targets = cols_2d.flatten()

        self.register_buffer("row_targets", row_targets, persistent=False)
        self.register_buffer("col_targets", col_targets, persistent=False)

    def forward(self, feats):
        """
        Args:
            feats: (B, N, D) patch features, N = grid_h * grid_w

        Returns:
            avg_loss: scalar, average of row and column regression losses
        """
        B, N, D = feats.shape
        assert N == self.grid_h * self.grid_w, f"Expected N = grid_h * grid_w = {self.grid_h * self.grid_w}, got N = {N}"

        x = feats.reshape(-1, D)

        row_targets = self.row_targets.repeat(B)
        col_targets = self.col_targets.repeat(B)

        row_pred = self.row_mlp(x).squeeze(-1)
        col_pred = self.col_mlp(x).squeeze(-1)

        loss_row = self.loss_fn(row_pred, row_targets)
        loss_col = self.loss_fn(col_pred, col_targets)

        return (loss_row + loss_col) / 2.0

class PatchRowColRegressionCriterionDynamic(nn.Module):
    def __init__(
        self,
        feat_dim,
        grid_h,
        grid_w,
        normalize=True,
        loss_type: LossType = "smooth_l1",
    ):
        """
        Predict row and column index of each patch via regression,
        supporting dynamic training resolutions.

        Args:
            feat_dim (int): Dimension of patch features (D)
            grid_h (int): Max number of patch rows (upper bound)
            grid_w (int): Max number of patch columns (upper bound)
            normalize (bool): If True, normalize row/col targets to [0, 1]
                              based on the *current* hp/wp for each batch.
            loss_type (str): "l1", "smooth_l1", or "mse"
        """
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.normalize = normalize

        self.row_mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.col_mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        if loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        elif loss_type == "smooth_l1":
            self.loss_fn = nn.SmoothL1Loss()
        elif loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        rows = torch.arange(grid_h, dtype=torch.float32).unsqueeze(1).repeat(1, grid_w)
        cols = torch.arange(grid_w, dtype=torch.float32).unsqueeze(0).repeat(grid_h, 1)

        self.register_buffer("row_index_full", rows, persistent=False)
        self.register_buffer("col_index_full", cols, persistent=False)

    def forward(self, feats, hp=None, wp=None):
        """
        Args:
            feats: (B, N, D) patch features, with N = hp * wp for this batch.
            hp, wp: number of patch rows / columns used for this batch
                    (single scalar each; one resolution per batch).

        Returns:
            avg_loss: scalar, average of row and column regression losses.
        """
        B, N, D = feats.shape

        if hp is None:
            hp = self.grid_h
        if wp is None:
            wp = self.grid_w

        assert N == hp * wp, f"Expected N = hp * wp = {hp * wp}, got N = {N}"

        x = feats.reshape(-1, D)

        row_idx_2d = self.row_index_full[:hp, :wp]
        col_idx_2d = self.col_index_full[:hp, :wp]

        if self.normalize:
            row_idx_2d = row_idx_2d / max(hp - 1, 1)
            col_idx_2d = col_idx_2d / max(wp - 1, 1)

        row_targets = row_idx_2d.flatten().repeat(B)
        col_targets = col_idx_2d.flatten().repeat(B)

        row_pred = self.row_mlp(x).squeeze(-1)
        col_pred = self.col_mlp(x).squeeze(-1)

        loss_row = self.loss_fn(row_pred, row_targets)
        loss_col = self.loss_fn(col_pred, col_targets)

        return (loss_row + loss_col) / 2.0

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_4d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        return x.unsqueeze(1)
    if x.dim() == 4:
        return x
    raise ValueError(f"Expected (B,H,W) or (B,1,H,W); got {tuple(x.shape)}")


def _default_mask(gt: torch.Tensor, eps: float) -> torch.Tensor:
    return (torch.isfinite(gt) & (gt > eps)).float()

def _masked_mean_std(x: torch.Tensor, mask: torch.Tensor, eps: float):
    denom = mask.sum(dim=(1, 2)).clamp_min(1.0)
    mean = (x * mask).sum(dim=(1, 2)) / denom
    var = ((x - mean[:, None, None]) ** 2 * mask).sum(dim=(1, 2)) / denom
    std = torch.sqrt(var.clamp_min(0.5))
    return mean, std, denom


def compute_scale_and_shift(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    det_threshold: float = 1e-6,
    return_stats: bool = False,
):
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det > float(det_threshold)

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    x_0[~valid] = 1.0
    x_1[~valid] = 0.0

    if return_stats:
        return x_0, x_1, det, valid
    return x_0, x_1


def _reduction_batch(image_loss: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    denom = torch.sum(M)
    if denom == 0:
        return image_loss.sum() * 0.0
    return torch.sum(image_loss) / denom


def _gradient_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                   reduction=_reduction_batch) -> torch.Tensor:
    M = torch.sum(mask, (1, 2))
    diff = (prediction - target) * mask

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = mask[:, :, 1:] * mask[:, :, :-1]
    grad_x = grad_x * mask_x

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = mask[:, 1:, :] * mask[:, :-1, :]
    grad_y = grad_y * mask_y

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))
    return reduction(image_loss, M)


def _l1_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
            reduction=_reduction_batch) -> torch.Tensor:
    M = torch.sum(mask, (1, 2))
    res = torch.abs(prediction - target)
    image_loss = torch.sum(mask * res, (1, 2))
    return reduction(image_loss, M)


class SILogLoss(nn.Module):
    """SILog loss (pixel-wise)."""
    def __init__(self, beta: float = 0.15, correction: int = 1, per_image: bool = True):
        super().__init__()
        self.beta = float(beta)
        self.correction = int(correction)
        self.per_image = bool(per_image)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        pred = _ensure_4d(pred).float()
        target = _ensure_4d(target).float()
        mask = _ensure_4d(mask).float()

        with torch.amp.autocast(device_type=pred.device.type, enabled=False):
            valid = (mask > 0.5) & torch.isfinite(pred) & torch.isfinite(target)
            mask = valid.float()
            pred = torch.where(valid, pred, torch.ones_like(pred))
            target = torch.where(valid, target, torch.ones_like(target))
            lambda_floor = 1e-7
            pred = torch.clamp(pred, min=lambda_floor)
            target = torch.clamp(target, min=lambda_floor)

            g = torch.log(pred) - torch.log(target)
            if self.per_image:
                B = g.shape[0]
                g_flat = g.reshape(B, -1)
                m_flat = mask.reshape(B, -1)
                mask_sum = m_flat.sum(dim=1)
                valid_img = mask_sum > 0

                mask_sum_safe = mask_sum.clamp_min(1.0)
                mean = (g_flat * m_flat).sum(dim=1) / mask_sum_safe
                denom_var = (mask_sum_safe - float(self.correction)).clamp_min(1.0)
                var = ((g_flat - mean[:, None]) ** 2 * m_flat).sum(dim=1) / denom_var
                Dg = var + self.beta * mean.pow(2)
                loss_per_img = 10.0 * torch.sqrt(Dg)
                if valid_img.any():
                    return loss_per_img[valid_img].mean()
                return pred.sum() * 0.0

            denom = mask.sum().clamp_min(1.0)
            mean = (g * mask).sum() / denom
            denom_var = (denom - float(self.correction)).clamp_min(1.0)
            var = ((g - mean) ** 2 * mask).sum() / denom_var
            Dg = var + self.beta * mean.pow(2)
            return 10.0 * torch.sqrt(Dg)


class MonocularDepthHybridLoss(nn.Module):
    """
    Hybrid monocular depth loss:
      - Scale-and-shift invariant L1 on aligned prediction (MiDaS-style)
      - Multi-scale gradient loss on aligned prediction (MiDaS-style)
      - Optional SILog on aligned prediction (ZoeDepth/AdaBins-style)
    """
    def __init__(
        self,
        l1_w: float = 1.0,
        grad_w: float = 0.5,
        silog_w: float = 0.0,
        silog_beta: float = 0.15,
        scales: int = 4,
        reduction: str = "batch-based",
        eps: float = 1e-8,
        silog_on_aligned: bool = False,
        det_threshold: float = 1e-6,
        min_valid_pixels: int = 0,
        align_mode: str = "scale_shift",
    ):
        super().__init__()
        self.l1_w = float(l1_w)
        self.grad_w = float(grad_w)
        self.silog_w = float(silog_w)
        self.scales = int(scales)
        self.eps = float(eps)
        self.silog_on_aligned = bool(silog_on_aligned)
        self.det_threshold = float(det_threshold)
        self.min_valid_pixels = int(min_valid_pixels)
        align_mode = str(align_mode)
        if align_mode not in ("scale_shift", "mean_std"):
            raise ValueError(f"Unsupported align_mode='{align_mode}'. Use 'scale_shift' or 'mean_std'.")
        self.align_mode = align_mode
        if reduction == "batch-based":
            self._reduction = _reduction_batch
        else:
            raise ValueError("Only 'batch-based' reduction is supported.")

        self._silog = SILogLoss(beta=silog_beta)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None,
                interpolate: bool = True):
        prediction = _ensure_4d(prediction)
        target = _ensure_4d(target)

        if interpolate and prediction.shape[-2:] != target.shape[-2:]:
            prediction = F.interpolate(prediction, target.shape[-2:], mode="bilinear", align_corners=True)

        if mask is None:
            mask = _default_mask(target, self.eps)
        else:
            mask = _ensure_4d(mask).float()

        pred_hw = prediction[:, 0]
        tgt_hw = target[:, 0]
        m_hw = mask[:, 0]

        valid_count = None
        if self.min_valid_pixels > 0:
            valid_count = m_hw.sum(dim=(1, 2))
            keep = valid_count >= float(self.min_valid_pixels)
            if not torch.all(keep):
                keep_f = keep.to(dtype=m_hw.dtype)
                m_hw = m_hw * keep_f[:, None, None]
                mask = mask * keep_f[:, None, None, None]

        if self.align_mode == "mean_std":
            pred_mean, pred_std, denom = _masked_mean_std(pred_hw, m_hw, self.eps)
            tgt_mean, tgt_std, _ = _masked_mean_std(tgt_hw, m_hw, self.eps)
            pred_aligned = (prediction - pred_mean.view(-1, 1, 1, 1)) / pred_std.view(-1, 1, 1, 1)
            target_aligned = (target - tgt_mean.view(-1, 1, 1, 1)) / tgt_std.view(-1, 1, 1, 1)

        else:
            scale, shift = compute_scale_and_shift(
                pred_hw, tgt_hw, m_hw, det_threshold=self.det_threshold, return_stats=False
            )
            pred_aligned = scale.view(-1, 1, 1, 1) * prediction + shift.view(-1, 1, 1, 1)
            target_aligned = target

        total = pred_aligned.new_zeros(())
        if self.l1_w > 0:
            total = total + self.l1_w * _l1_loss(pred_aligned[:, 0], target_aligned[:, 0], m_hw, self._reduction)

        if self.grad_w > 0:
            grad_total = pred_aligned.new_zeros(())
            for scale_i in range(self.scales):
                step = 2 ** scale_i
                grad_total = grad_total + _gradient_loss(
                    pred_aligned[:, 0, ::step, ::step],
                    target_aligned[:, 0, ::step, ::step],
                    m_hw[:, ::step, ::step],
                    reduction=self._reduction,
                )
            total = total + self.grad_w * grad_total

        if self.silog_w > 0:
            silog_pred = pred_aligned if self.silog_on_aligned else prediction
            total = total + self.silog_w * self._silog(silog_pred, target, mask)

        return total

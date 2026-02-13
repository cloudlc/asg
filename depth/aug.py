import random
from typing import Tuple, Optional, Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from torchvision.transforms import ColorJitter, GaussianBlur
import torchvision.transforms.functional as TF


ImageLike = Union[Image.Image, np.ndarray, torch.Tensor]
DepthLike = Union[np.ndarray, torch.Tensor]


def _to_pil_rgb(image: ImageLike) -> Image.Image:
    """Convert input to PIL RGB."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, torch.Tensor):
        x = image.detach().cpu()
        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = x.permute(1, 2, 0)
        x = x.numpy()
        image = x
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0.0, 1.0) if arr.max() <= 1.5 else np.clip(arr / 255.0, 0.0, 1.0)
            arr = (arr * 255.0).round().astype(np.uint8)
        return Image.fromarray(arr)
    raise TypeError(f"Unsupported image type: {type(image)}")


def _to_depth_1chw(depth: DepthLike) -> torch.Tensor:
    """
    Convert depth to torch float32 [1,H,W]. Invalid depth may be <=0 or non-finite.
    """
    if isinstance(depth, torch.Tensor):
        d = depth.detach().float().cpu()
        if d.ndim == 2:
            d = d.unsqueeze(0)
        elif d.ndim == 3 and d.shape[-1] == 1:
            d = d.permute(2, 0, 1)
        if d.ndim != 3 or d.shape[0] != 1:
            raise ValueError(f"Depth must be [H,W] or [H,W,1] or [1,H,W]; got {tuple(d.shape)}")
        return d
    if isinstance(depth, np.ndarray):
        d = torch.from_numpy(depth).float()
        if d.ndim == 2:
            d = d.unsqueeze(0)
        elif d.ndim == 3 and d.shape[-1] == 1:
            d = d.permute(2, 0, 1)
        if d.ndim != 3 or d.shape[0] != 1:
            raise ValueError(f"Depth must be [H,W] or [H,W,1]; got {depth.shape}")
        return d
    raise TypeError(f"Unsupported depth type: {type(depth)}")


def _pil_to_tensor01(pil_img: Image.Image) -> torch.Tensor:
    """PIL RGB -> torch float32 [3,H,W] in [0,1]."""
    x = torch.from_numpy(np.array(pil_img)).float() / 255.0
    return x.permute(2, 0, 1).contiguous()


def _normalize_img(img_t: torch.Tensor, mean, std) -> torch.Tensor:
    mean_v = mean if isinstance(mean, (list, tuple)) else [float(mean)]
    std_v = std if isinstance(std, (list, tuple)) else [float(std)]
    if len(mean_v) == 1:
        mean_v = mean_v * 3
    if len(std_v) == 1:
        std_v = std_v * 3
    mean_t = torch.tensor(mean_v, dtype=img_t.dtype, device=img_t.device).view(3, 1, 1)
    std_t = torch.tensor(std_v, dtype=img_t.dtype, device=img_t.device).view(3, 1, 1)
    return (img_t - mean_t) / std_t


def _resize_depth_with_mask(
    depth_1chw: torch.Tensor,
    size_hw: Tuple[int, int],
    *,
    valid_thresh: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Resize depth with masked renormalization to avoid zero-bleed bias.
    depth_1chw: [1,H,W] -> returns [1,Ht,Wt] with invalid set to 0.
    """
    d = depth_1chw.unsqueeze(0).float()
    valid = torch.isfinite(d) & (d > 0)
    m = valid.float()
    dm = d * m

    H, W = d.shape[-2:]
    Ht, Wt = size_hw
    is_down = (Ht <= H) and (Wt <= W)

    if is_down:
        dm_rs = F.interpolate(dm, size=size_hw, mode="area")
        m_rs = F.interpolate(m, size=size_hw, mode="area")
    else:
        dm_rs = F.interpolate(dm, size=size_hw, mode="bilinear", align_corners=False)
        m_rs = F.interpolate(m, size=size_hw, mode="bilinear", align_corners=False)

    d_rs = dm_rs / (m_rs + eps)
    valid_rs = m_rs > valid_thresh
    d_rs = torch.where(valid_rs, d_rs, torch.zeros_like(d_rs))

    return d_rs.squeeze(0)


def _round_to_multiple(x: int, m: int) -> int:
    return int(round(x / m) * m)


def train_aug_depth_ar_resize_random_crop(
    image: ImageLike,
    depth: DepthLike,
    target_size: Tuple[int, int],
    *,
    hflip_prob: float = 0.5,
    scale_jitter: Optional[Tuple[float, Optional[float]]] = (1.0, None),
    color_jitter: Optional[Dict[str, float]] = None,
    color_jitter_prob: float = 0.8,
    gamma_jitter: Optional[Tuple[float, float]] = (0.9, 1.1),
    gamma_jitter_prob: float = 0.0,
    grayscale_prob: float = 0.05,
    blur_prob: float = 0.0,
    blur_kernel: Tuple[int, int] = (5, 5),
    blur_sigma: Tuple[float, float] = (0.1, 1.0),
    noise_std: Optional[Tuple[float, float]] = (0.0, 0.0),
    normalize: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ensure_multiple_of: Optional[int] = None,
    depth_valid_thresh: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Training augmentation (no padding):
      1) Resize with aspect ratio (optionally mild scale jitter), ensuring resized dims >= target
      2) Random crop to target_size
      3) Random horizontal flip
      4) Color + gamma jitter, optional grayscale, blur, and noise (RGB only)

    Returns:
      img_t:   [3,Ht,Wt] float32 in [0,1]
      depth_t: [1,Ht,Wt] float32 (invalid -> 0)
    """
    Ht, Wt = target_size
    pil = _to_pil_rgb(image)
    d = _to_depth_1chw(depth)

    W0, H0 = pil.size

    base = max(Ht / H0, Wt / W0)
    if scale_jitter is None:
        jitter = 1.0
    else:
        j0, j1 = scale_jitter
        j0 = max(1.0, float(j0))
        if j1 is None:
            j1 = (1.0 / base) if base < 1.0 else j0
        j1 = max(j0, float(j1))
        jitter = random.uniform(j0, j1)
    scale = base * jitter

    newH = int(round(H0 * scale))
    newW = int(round(W0 * scale))

    if newH < Ht or newW < Wt:
        scale = base
        newH = int(round(H0 * scale))
        newW = int(round(W0 * scale))

    if ensure_multiple_of is not None and ensure_multiple_of > 1:
        newH = max(Ht, _round_to_multiple(newH, ensure_multiple_of))
        newW = max(Wt, _round_to_multiple(newW, ensure_multiple_of))

    resample = Image.BOX if scale < 1.0 else Image.BICUBIC
    pil = pil.resize((newW, newH), resample=resample)
    d = _resize_depth_with_mask(d, (newH, newW), valid_thresh=depth_valid_thresh)

    top = 0 if newH == Ht else random.randint(0, newH - Ht)
    left = 0 if newW == Wt else random.randint(0, newW - Wt)

    pil = TF.crop(pil, top=top, left=left, height=Ht, width=Wt)
    d = d[:, top:top + Ht, left:left + Wt]

    if random.random() < hflip_prob:
        pil = TF.hflip(pil)
        d = torch.flip(d, dims=[2])

    if (
        color_jitter is not None
        and any(v > 0 for v in color_jitter.values())
        and (color_jitter_prob > 0)
        and (random.random() < color_jitter_prob)
    ):
        pil = ColorJitter(**color_jitter)(pil)

    if gamma_jitter is not None and gamma_jitter_prob > 0 and random.random() < gamma_jitter_prob:
        g0, g1 = gamma_jitter
        if g0 != 1.0 or g1 != 1.0:
            gamma = random.uniform(g0, g1)
            pil = TF.adjust_gamma(pil, gamma=gamma)

    if grayscale_prob > 0 and random.random() < grayscale_prob:
        pil = TF.to_grayscale(pil, num_output_channels=3)

    if blur_prob > 0 and random.random() < blur_prob:
        pil = GaussianBlur(kernel_size=blur_kernel, sigma=blur_sigma)(pil)

    img_t = _pil_to_tensor01(pil)

    if noise_std is not None:
        n0, n1 = noise_std
        if n1 > 0:
            std = random.uniform(n0, n1)
            if std > 0:
                img_t = (img_t + torch.randn_like(img_t) * std).clamp(0.0, 1.0)

    if normalize:
        img_t = _normalize_img(img_t, mean, std)

    return img_t, d.contiguous().float()


def eval_preprocess_depth_keep_ar(
    image: ImageLike,
    depth: DepthLike,
    target_size: Tuple[int, int],
    *,
    target_by: str = "height",
    eval_crop_mode: str = "pad",
    eval_prescale: float = 1.0,
    ensure_multiple_of: Optional[int] = 32,
    normalize: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    depth_valid_thresh: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Validation/Test preprocessing:
      - Deterministic resize keeping aspect ratio
      - Optional center crop/pad to target_size
      - Optional padding to ensure multiple-of for network stride
      - Single forward pass on the resized tensor
      - (Typically) upsample prediction back to original (H,W) externally for metrics

    Interpretation of target_size:
      - If target_by="height": uses target_size[0] as the inference height (recommended default).
      - If target_by="long_side": uses max(target_size) as the inference long side.

    Returns:
      img_t:   [3,H',W'] float32 in [0,1]
      depth_t: [1,H',W'] float32 resized with mask (invalid -> 0)
      meta: dict with orig/resized sizes and scale (useful for resizing predictions back):
            {
              "orig_h","orig_w","resized_h","resized_w",
              "scale_h","scale_w"
            }
    """
    pil = _to_pil_rgb(image)
    d = _to_depth_1chw(depth)

    W0, H0 = pil.size
    Ht, Wt = target_size

    prescale = max(1e-6, float(eval_prescale))
    if target_by == "height":
        newH = int(round(Ht * prescale))
        scale = newH / H0
        newW = int(round(W0 * scale))
    elif target_by == "long_side":
        long_target = int(round(max(Ht, Wt) * prescale))
        long0 = max(H0, W0)
        scale = long_target / long0
        newH = int(round(H0 * scale))
        newW = int(round(W0 * scale))
    else:
        raise ValueError(f"target_by must be 'height' or 'long_side', got {target_by}")

    resample = Image.BOX if scale < 1.0 else Image.BICUBIC
    pil_rs = pil.resize((newW, newH), resample=resample)
    d_rs = _resize_depth_with_mask(d, (newH, newW), valid_thresh=depth_valid_thresh)
    resize_h = newH
    resize_w = newW

    crop_top = crop_left = 0
    pad_left = pad_top = pad_right = pad_bottom = 0
    if eval_crop_mode not in ("pad", "crop", "crop_or_pad"):
        raise ValueError(f"eval_crop_mode must be 'pad', 'crop', or 'crop_or_pad', got {eval_crop_mode}")
    if eval_crop_mode == "crop" or (eval_crop_mode == "crop_or_pad" and newH >= Ht and newW >= Wt):
        top = max(0, (newH - Ht) // 2)
        left = max(0, (newW - Wt) // 2)
        pil_rs = TF.crop(pil_rs, top=top, left=left, height=Ht, width=Wt)
        d_rs = d_rs[:, top:top + Ht, left:left + Wt]
        crop_top = top
        crop_left = left
        newH, newW = Ht, Wt
    elif eval_crop_mode in ("pad", "crop_or_pad"):
        if newH < Ht or newW < Wt:
            pad_h = max(0, Ht - newH)
            pad_w = max(0, Wt - newW)
            pad_top = pad_h // 2
            pad_left = pad_w // 2
            pad_bottom = pad_h - pad_top
            pad_right = pad_w - pad_left
            mean_v = mean if isinstance(mean, (list, tuple)) else [float(mean)]
            if len(mean_v) == 1:
                mean_v = mean_v * 3
            pad_fill = tuple(int(round(m * 255.0)) for m in mean_v[:3])
            pil_rs = TF.pad(pil_rs, padding=(pad_left, pad_top, pad_right, pad_bottom), fill=pad_fill)
            d_rs = F.pad(d_rs, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)
            newH = newH + pad_top + pad_bottom
            newW = newW + pad_left + pad_right

    pad_h = 0
    pad_w = 0
    if ensure_multiple_of is not None and ensure_multiple_of > 1:
        pad_h = (ensure_multiple_of - (newH % ensure_multiple_of)) % ensure_multiple_of
        pad_w = (ensure_multiple_of - (newW % ensure_multiple_of)) % ensure_multiple_of
        if pad_h or pad_w:
            mean_v = mean if isinstance(mean, (list, tuple)) else [float(mean)]
            if len(mean_v) == 1:
                mean_v = mean_v * 3
            pad_fill = tuple(int(round(m * 255.0)) for m in mean_v[:3])
            pil_rs = TF.pad(pil_rs, padding=(0, 0, pad_w, pad_h), fill=pad_fill)
            d_rs = F.pad(d_rs, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

    meta = {
        "orig_h": float(H0), "orig_w": float(W0),
        "resized_h": float(newH), "resized_w": float(newW),
        "resize_h": float(resize_h), "resize_w": float(resize_w),
        "scale_h": float(resize_h) / float(H0),
        "scale_w": float(resize_w) / float(W0),
        "pad_h": float(pad_h), "pad_w": float(pad_w),
        "padded_h": float(newH + pad_h), "padded_w": float(newW + pad_w),
        "crop_top": float(crop_top), "crop_left": float(crop_left),
        "pad_left": float(pad_left), "pad_top": float(pad_top),
        "pad_right": float(pad_right), "pad_bottom": float(pad_bottom),
    }
    img_t = _pil_to_tensor01(pil_rs)
    if normalize:
        img_t = _normalize_img(img_t, mean, std)
    return img_t, d_rs.contiguous().float(), meta


def eval_preprocess_depth_no_resize(
    image: ImageLike,
    depth: DepthLike,
    *,
    ensure_multiple_of: Optional[int] = None,
    normalize: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Validation/Test preprocessing without resizing:
      - Convert to tensor, keep original resolution
      - Optional padding to ensure multiple-of for network stride
    """
    pil = _to_pil_rgb(image)
    d = _to_depth_1chw(depth)

    W0, H0 = pil.size
    img_t = _pil_to_tensor01(pil)

    pad_h = 0
    pad_w = 0
    if ensure_multiple_of is not None and ensure_multiple_of > 1:
        pad_h = (ensure_multiple_of - (H0 % ensure_multiple_of)) % ensure_multiple_of
        pad_w = (ensure_multiple_of - (W0 % ensure_multiple_of)) % ensure_multiple_of
        if pad_h or pad_w:
            img_t = TF.pad(img_t, padding=(0, 0, pad_w, pad_h), fill=0)
            d = F.pad(d, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

    if normalize:
        img_t = _normalize_img(img_t, mean, std)

    meta = {
        "orig_h": float(H0), "orig_w": float(W0),
        "resized_h": float(H0), "resized_w": float(W0),
        "scale_h": 1.0, "scale_w": 1.0,
        "pad_h": float(pad_h), "pad_w": float(pad_w),
        "padded_h": float(H0 + pad_h), "padded_w": float(W0 + pad_w),
    }
    return img_t, d.contiguous().float(), meta


class TrainDepthAug:
    def __init__(
        self,
        target_size: Tuple[int, int],
        *,
        hflip_prob: float = 0.5,
        scale_jitter: Optional[Tuple[float, Optional[float]]] = (1.0, None),
        color_jitter: Optional[Dict[str, float]] = None,
        color_jitter_prob: float = 0.5,
        gamma_jitter: Optional[Tuple[float, float]] = (0.9, 1.1),
        gamma_jitter_prob: float = 0.0,
        grayscale_prob: float = 0.05,
        blur_prob: float = 0.0,
        blur_kernel: Tuple[int, int] = (5, 5),
        blur_sigma: Tuple[float, float] = (0.1, 1.0),
        noise_std: Optional[Tuple[float, float]] = (0.0, 0.0),
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        ensure_multiple_of: Optional[int] = None,
        depth_valid_thresh: float = 0.1,
    ) -> None:
        self.target_size = target_size
        self.hflip_prob = hflip_prob
        self.scale_jitter = scale_jitter
        if color_jitter is None:
            color_jitter = dict(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
        self.color_jitter = dict(color_jitter)
        self.color_jitter_prob = color_jitter_prob
        self.gamma_jitter = gamma_jitter
        self.gamma_jitter_prob = gamma_jitter_prob
        self.grayscale_prob = grayscale_prob
        self.blur_prob = blur_prob
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma
        self.noise_std = noise_std
        self.normalize = normalize
        mean_v = mean if isinstance(mean, (list, tuple)) else [float(mean)]
        std_v = std if isinstance(std, (list, tuple)) else [float(std)]
        if len(mean_v) == 1:
            mean_v = mean_v * 3
        if len(std_v) == 1:
            std_v = std_v * 3
        self._mean_t = torch.tensor(mean_v).view(3, 1, 1)
        self._std_t = torch.tensor(std_v).view(3, 1, 1)
        self.ensure_multiple_of = ensure_multiple_of
        self.depth_valid_thresh = depth_valid_thresh

    def __call__(self, image: ImageLike, depth: DepthLike) -> Tuple[torch.Tensor, torch.Tensor]:
        img_t, depth_t = train_aug_depth_ar_resize_random_crop(
            image,
            depth,
            self.target_size,
            hflip_prob=self.hflip_prob,
            scale_jitter=self.scale_jitter,
            color_jitter=self.color_jitter,
            color_jitter_prob=self.color_jitter_prob,
            gamma_jitter=self.gamma_jitter,
            gamma_jitter_prob=self.gamma_jitter_prob,
            grayscale_prob=self.grayscale_prob,
            blur_prob=self.blur_prob,
            blur_kernel=self.blur_kernel,
            blur_sigma=self.blur_sigma,
            noise_std=self.noise_std,
            normalize=False,
            ensure_multiple_of=self.ensure_multiple_of,
            depth_valid_thresh=self.depth_valid_thresh,
        )
        if self.normalize:
            mean_t = self._mean_t.to(dtype=img_t.dtype)
            std_t = self._std_t.to(dtype=img_t.dtype)
            img_t = (img_t - mean_t) / std_t
        return img_t, depth_t


class EvalDepthPreprocess:
    def __init__(
        self,
        target_size: Tuple[int, int],
        *,
        target_by: str = "height",
        eval_crop_mode: str = "pad",
        eval_prescale: float = 1.0,
        ensure_multiple_of: Optional[int] = 32,
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        depth_valid_thresh: float = 0.0,
    ) -> None:
        self.target_size = target_size
        self.target_by = target_by
        self.eval_crop_mode = eval_crop_mode
        self.eval_prescale = eval_prescale
        self.ensure_multiple_of = ensure_multiple_of
        self.normalize = normalize
        mean_v = mean if isinstance(mean, (list, tuple)) else [float(mean)]
        std_v = std if isinstance(std, (list, tuple)) else [float(std)]
        if len(mean_v) == 1:
            mean_v = mean_v * 3
        if len(std_v) == 1:
            std_v = std_v * 3
        self._mean_t = torch.tensor(mean_v).view(3, 1, 1)
        self._std_t = torch.tensor(std_v).view(3, 1, 1)
        self.depth_valid_thresh = depth_valid_thresh

    def __call__(
        self, image: ImageLike, depth: DepthLike
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        img_t, depth_t, meta = eval_preprocess_depth_keep_ar(
            image,
            depth,
            self.target_size,
            target_by=self.target_by,
            eval_crop_mode=self.eval_crop_mode,
            eval_prescale=self.eval_prescale,
            ensure_multiple_of=self.ensure_multiple_of,
            normalize=False,
            depth_valid_thresh=self.depth_valid_thresh,
        )
        if self.normalize:
            mean_t = self._mean_t.to(dtype=img_t.dtype)
            std_t = self._std_t.to(dtype=img_t.dtype)
            img_t = (img_t - mean_t) / std_t
        return img_t, depth_t, meta


class EvalDepthPreprocessNoResize:
    def __init__(
        self,
        *,
        ensure_multiple_of: Optional[int] = None,
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.ensure_multiple_of = ensure_multiple_of
        self.normalize = normalize
        mean_v = mean if isinstance(mean, (list, tuple)) else [float(mean)]
        std_v = std if isinstance(std, (list, tuple)) else [float(std)]
        if len(mean_v) == 1:
            mean_v = mean_v * 3
        if len(std_v) == 1:
            std_v = std_v * 3
        self._mean_t = torch.tensor(mean_v).view(3, 1, 1)
        self._std_t = torch.tensor(std_v).view(3, 1, 1)

    def __call__(
        self, image: ImageLike, depth: DepthLike
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        img_t, depth_t, meta = eval_preprocess_depth_no_resize(
            image,
            depth,
            ensure_multiple_of=self.ensure_multiple_of,
            normalize=False,
        )
        if self.normalize:
            mean_t = self._mean_t.to(dtype=img_t.dtype)
            std_t = self._std_t.to(dtype=img_t.dtype)
            img_t = (img_t - mean_t) / std_t
        return img_t, depth_t, meta

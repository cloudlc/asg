import random
from typing import Tuple, Optional, Dict, Union

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter

ImageLike = Union[Image.Image, np.ndarray, torch.Tensor]
MaskLike = Union[Image.Image, np.ndarray, torch.Tensor]


def _to_pil_rgb(image: ImageLike) -> Image.Image:
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
        return Image.fromarray(arr, mode="RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def _to_pil_mask(mask: MaskLike) -> Image.Image:
    if isinstance(mask, Image.Image):
        return mask.convert("I")
    if isinstance(mask, torch.Tensor):
        m = mask.detach().cpu()
        if m.ndim == 3 and m.shape[0] == 1:
            m = m.squeeze(0)
        if m.ndim == 3 and m.shape[-1] == 1:
            m = m.squeeze(-1)
        mask = m.numpy()
    if isinstance(mask, np.ndarray):
        arr = mask
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        if arr.ndim != 2:
            raise ValueError(f"Mask must be [H,W]; got {arr.shape}")
        if arr.dtype != np.int32:
            arr = arr.astype(np.int32, copy=False)
        return Image.fromarray(arr, mode="I")
    raise TypeError(f"Unsupported mask type: {type(mask)}")


def _pil_to_tensor01(pil_img: Image.Image) -> torch.Tensor:
    x = torch.from_numpy(np.array(pil_img)).float() / 255.0
    return x.permute(2, 0, 1).contiguous()


def _mask_to_tensor(mask_pil: Image.Image) -> torch.Tensor:
    arr = np.array(mask_pil, dtype=np.int64)
    return torch.from_numpy(arr)


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


def train_aug_seg_resize_random_crop(
    image: ImageLike,
    mask: MaskLike,
    target_size: Tuple[int, int],
    *,
    hflip_prob: float = 0.5,
    scale_jitter: Optional[Tuple[float, Optional[float]]] = (1.0, None),
    cat_max_ratio: Optional[float] = None,
    cat_max_ratio_tries: int = 10,
    ignore_index: Optional[int] = None,
    color_jitter: Optional[Dict[str, float]] = None,
    color_jitter_prob: float = 0.0,
    normalize: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Tuple[torch.Tensor, torch.Tensor]:
    Ht, Wt = target_size
    pil = _to_pil_rgb(image)
    m = _to_pil_mask(mask)

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
        newH = int(round(H0 * base))
        newW = int(round(W0 * base))

    resample = Image.BOX if scale < 1.0 else Image.BICUBIC
    pil = pil.resize((newW, newH), resample=resample)
    m = m.resize((newW, newH), resample=Image.NEAREST)

    def _is_crop_ok(mask_crop: Image.Image) -> bool:
        if cat_max_ratio is None or cat_max_ratio >= 1.0:
            return True
        mask_np = np.array(mask_crop, dtype=np.int64)
        if ignore_index is not None:
            mask_np = mask_np[mask_np != ignore_index]
        if mask_np.size == 0:
            return True
        counts = np.bincount(mask_np.reshape(-1))
        max_ratio = counts.max() / counts.sum()
        return max_ratio < cat_max_ratio

    top = 0
    left = 0
    m_crop = None
    tries = max(1, int(cat_max_ratio_tries))
    for _ in range(tries):
        top = 0 if newH == Ht else random.randint(0, newH - Ht)
        left = 0 if newW == Wt else random.randint(0, newW - Wt)
        m_crop = TF.crop(m, top=top, left=left, height=Ht, width=Wt)
        if _is_crop_ok(m_crop):
            break
    pil = TF.crop(pil, top=top, left=left, height=Ht, width=Wt)
    if m_crop is None:
        m_crop = TF.crop(m, top=top, left=left, height=Ht, width=Wt)
    m = m_crop

    if random.random() < hflip_prob:
        pil = TF.hflip(pil)
        m = TF.hflip(m)

    if (
        color_jitter is not None
        and any(v > 0 for v in color_jitter.values())
        and (color_jitter_prob > 0)
        and (random.random() < color_jitter_prob)
    ):
        pil = ColorJitter(**color_jitter)(pil)

    img_t = _pil_to_tensor01(pil)
    if normalize:
        img_t = _normalize_img(img_t, mean, std)
    mask_t = _mask_to_tensor(m)
    return img_t, mask_t


def eval_preprocess_seg_keep_ar(
    image: ImageLike,
    mask: MaskLike,
    target_size: Tuple[int, int],
    *,
    target_by: str = "shorter",
    eval_crop_mode: str = "pad",
    normalize: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    pil = _to_pil_rgb(image)
    m = _to_pil_mask(mask)
    W0, H0 = pil.size
    Ht, Wt = target_size

    if target_by == "shorter":
        scale = min(Ht, Wt) / min(H0, W0)
    elif target_by == "longer":
        scale = max(Ht, Wt) / max(H0, W0)
    else:
        raise ValueError(f"target_by must be 'shorter' or 'longer', got {target_by}")

    newH = int(round(H0 * scale))
    newW = int(round(W0 * scale))
    resample = Image.BOX if scale < 1.0 else Image.BICUBIC
    pil = pil.resize((newW, newH), resample=resample)
    m = m.resize((newW, newH), resample=Image.NEAREST)

    top = max(0, (newH - Ht) // 2)
    left = max(0, (newW - Wt) // 2)
    pad = (0, 0, 0, 0)
    mean_v = mean if isinstance(mean, (list, tuple)) else [float(mean)]
    if len(mean_v) == 1:
        mean_v = mean_v * 3
    pad_fill = tuple(int(round(m * 255.0)) for m in mean_v[:3])

    if eval_crop_mode == "crop":
        pil = TF.crop(pil, top=top, left=left, height=Ht, width=Wt)
        m = TF.crop(m, top=top, left=left, height=Ht, width=Wt)
    elif eval_crop_mode == "pad":
        if newH < Ht or newW < Wt:
            pad_h = max(0, Ht - newH)
            pad_w = max(0, Wt - newW)
            pad_top = pad_h // 2
            pad_left = pad_w // 2
            pad_bottom = pad_h - pad_top
            pad_right = pad_w - pad_left
            pil = TF.pad(pil, padding=(pad_left, pad_top, pad_right, pad_bottom), fill=pad_fill)
            m = TF.pad(m, padding=(pad_left, pad_top, pad_right, pad_bottom), fill=0)
            pad = (pad_left, pad_top, pad_right, pad_bottom)
    elif eval_crop_mode == "crop_or_pad":
        if newH >= Ht and newW >= Wt:
            pil = TF.crop(pil, top=top, left=left, height=Ht, width=Wt)
            m = TF.crop(m, top=top, left=left, height=Ht, width=Wt)
        else:
            pad_h = max(0, Ht - newH)
            pad_w = max(0, Wt - newW)
            pad_top = pad_h // 2
            pad_left = pad_w // 2
            pad_bottom = pad_h - pad_top
            pad_right = pad_w - pad_left
            pil = TF.pad(pil, padding=(pad_left, pad_top, pad_right, pad_bottom), fill=pad_fill)
            m = TF.pad(m, padding=(pad_left, pad_top, pad_right, pad_bottom), fill=0)
            pad = (pad_left, pad_top, pad_right, pad_bottom)
    else:
        raise ValueError(f"eval_crop_mode must be 'pad', 'crop', or 'crop_or_pad', got {eval_crop_mode}")

    img_t = _pil_to_tensor01(pil)
    if normalize:
        img_t = _normalize_img(img_t, mean, std)
    mask_t = _mask_to_tensor(m)

    meta = {
        "orig_h": float(H0), "orig_w": float(W0),
        "resized_h": float(newH), "resized_w": float(newW),
        "scale_h": float(newH) / float(H0),
        "scale_w": float(newW) / float(W0),
        "crop_top": float(top), "crop_left": float(left),
        "pad_left": float(pad[0]), "pad_top": float(pad[1]),
        "pad_right": float(pad[2]), "pad_bottom": float(pad[3]),
        "out_h": float(pil.size[1]), "out_w": float(pil.size[0]),
    }
    return img_t, mask_t, meta


class TrainSegAug:
    def __init__(
        self,
        target_size: Tuple[int, int],
        *,
        hflip_prob: float = 0.5,
        scale_jitter: Optional[Tuple[float, Optional[float]]] = (1.0, None),
        cat_max_ratio: Optional[float] = None,
        cat_max_ratio_tries: int = 10,
        ignore_index: Optional[int] = None,
        color_jitter: Optional[Dict[str, float]] = None,
        color_jitter_prob: float = 0.0,
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.target_size = target_size
        self.hflip_prob = hflip_prob
        self.scale_jitter = scale_jitter
        self.cat_max_ratio = cat_max_ratio
        self.cat_max_ratio_tries = cat_max_ratio_tries
        self.ignore_index = ignore_index
        self.color_jitter = None if color_jitter is None else dict(color_jitter)
        self.color_jitter_prob = color_jitter_prob
        self.normalize = normalize
        mean_v = mean if isinstance(mean, (list, tuple)) else [float(mean)]
        std_v = std if isinstance(std, (list, tuple)) else [float(std)]
        if len(mean_v) == 1:
            mean_v = mean_v * 3
        if len(std_v) == 1:
            std_v = std_v * 3
        self._mean_t = torch.tensor(mean_v).view(3, 1, 1)
        self._std_t = torch.tensor(std_v).view(3, 1, 1)

    def __call__(self, image: ImageLike, mask: MaskLike) -> Tuple[torch.Tensor, torch.Tensor]:
        img_t, mask_t = train_aug_seg_resize_random_crop(
            image,
            mask,
            self.target_size,
            hflip_prob=self.hflip_prob,
            scale_jitter=self.scale_jitter,
            cat_max_ratio=self.cat_max_ratio,
            cat_max_ratio_tries=self.cat_max_ratio_tries,
            ignore_index=self.ignore_index,
            color_jitter=self.color_jitter,
            color_jitter_prob=self.color_jitter_prob,
            normalize=False,
        )
        if self.normalize:
            mean_t = self._mean_t.to(dtype=img_t.dtype)
            std_t = self._std_t.to(dtype=img_t.dtype)
            img_t = (img_t - mean_t) / std_t
        return img_t, mask_t


class EvalSegPreprocess:
    def __init__(
        self,
        target_size: Tuple[int, int],
        *,
        target_by: str = "shorter",
        eval_crop_mode: str = "pad",
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.target_size = target_size
        self.target_by = target_by
        self.eval_crop_mode = eval_crop_mode
        self.normalize = normalize
        mean_v = mean if isinstance(mean, (list, tuple)) else [float(mean)]
        std_v = std if isinstance(std, (list, tuple)) else [float(std)]
        if len(mean_v) == 1:
            mean_v = mean_v * 3
        if len(std_v) == 1:
            std_v = std_v * 3
        self._mean_t = torch.tensor(mean_v).view(3, 1, 1)
        self._std_t = torch.tensor(std_v).view(3, 1, 1)

    def __call__(self, image: ImageLike, mask: MaskLike) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        img_t, mask_t, meta = eval_preprocess_seg_keep_ar(
            image,
            mask,
            self.target_size,
            target_by=self.target_by,
            eval_crop_mode=self.eval_crop_mode,
            normalize=False,
        )
        if self.normalize:
            mean_t = self._mean_t.to(dtype=img_t.dtype)
            std_t = self._std_t.to(dtype=img_t.dtype)
            img_t = (img_t - mean_t) / std_t
        return img_t, mask_t, meta


def eval_preprocess_seg_multiscale(
    image: ImageLike,
    mask: MaskLike,
    base_target_size: Tuple[int, int],
    *,
    scales: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5),
    flip: bool = True,
    target_by: str = "shorter",
    eval_crop_mode: str = "pad",
    normalize: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> list[Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]]:
    out_list = []
    for s in scales:
        ts = (int(round(base_target_size[0] * s)), int(round(base_target_size[1] * s)))
        img_t, mask_t, meta = eval_preprocess_seg_keep_ar(
            image,
            mask,
            ts,
            target_by=target_by,
            eval_crop_mode=eval_crop_mode,
            normalize=normalize,
            mean=mean,
            std=std,
        )
        meta["scale"] = float(s)
        meta["flip"] = False
        out_list.append((img_t, mask_t, meta))
        if flip:
            img_f = torch.flip(img_t, dims=[2])
            mask_f = torch.flip(mask_t, dims=[1])
            meta_f = dict(meta)
            meta_f["flip"] = True
            out_list.append((img_f, mask_f, meta_f))
    return out_list


class EvalSegPreprocessMSFlip:
    def __init__(
        self,
        base_target_size: Tuple[int, int],
        *,
        scales: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5),
        flip: bool = True,
        target_by: str = "shorter",
        eval_crop_mode: str = "pad",
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.base_target_size = base_target_size
        self.scales = scales
        self.flip = flip
        self.target_by = target_by
        self.eval_crop_mode = eval_crop_mode
        self.normalize = normalize
        self.mean = mean
        self.std = std

    def __call__(
        self, image: ImageLike, mask: MaskLike
    ) -> list[Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]]:
        return eval_preprocess_seg_multiscale(
            image,
            mask,
            self.base_target_size,
            scales=self.scales,
            flip=self.flip,
            target_by=self.target_by,
            eval_crop_mode=self.eval_crop_mode,
            normalize=self.normalize,
            mean=self.mean,
            std=self.std,
        )

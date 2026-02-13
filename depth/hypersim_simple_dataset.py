import glob
import os
import random
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from torch.utils.data import Dataset

from depth.aug import TrainDepthAug, EvalDepthPreprocess, EvalDepthPreprocessNoResize


def imread_cv2(path, options=cv2.IMREAD_COLOR):
    if path.endswith((".exr", "EXR")):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f"Could not load image={path} with {options=}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class HyperSimSimple(Dataset):
    def __init__(
        self,
        roots: Union[str, Sequence[str]],
        resolution: Tuple[int, int],
        split: Optional[str] = None,
        sample_rate: float = 1.0,
        pair_transform=None,
        **kwargs,
    ):
        super().__init__()
        if isinstance(roots, (list, tuple)):
            root_list = list(roots)
        else:
            root_list = [roots]

        if split is not None:
            root_list = [os.path.join(r, split) if os.path.isdir(os.path.join(r, split)) else r for r in root_list]

        self.roots = root_list
        self.resolution = resolution
        self._setup_resolution()
        self.dataset_label = "HyperSimSimple"
        self.is_train = (split == "train")

        if pair_transform is None:
            target_size = (self.resolution[1], self.resolution[0]) if isinstance(self.resolution, (list, tuple)) else (self.resolution, self.resolution)
            if self.is_train:
                self.pair_transform = TrainDepthAug(
                    target_size=target_size,
                    normalize=True,
                    depth_valid_thresh=0.1,
                )
            else:
                self.pair_transform = EvalDepthPreprocess(
                    target_size=target_size,
                    target_by="height",
                    ensure_multiple_of=32,
                    normalize=True,
                    depth_valid_thresh=0.0,
                )
        else:
            self.pair_transform = pair_transform

        self.image_paths: List[str] = []
        for root in self.roots:
            if not os.path.isdir(root):
                continue
            self.image_paths.extend(glob.glob(os.path.join(root, "**", "*_rgb.png"), recursive=True))

        if not self.image_paths:
            raise FileNotFoundError(f"No '*_rgb.png' files found in roots: {self.roots}")

        if sample_rate < 1.0:
            num_samples = int(len(self.image_paths) * sample_rate)
            self.image_paths = random.sample(self.image_paths, num_samples)

        self.image_paths.sort()

    def _setup_resolution(self):
        if isinstance(self.resolution, int):
            self.resolution = (self.resolution, self.resolution)
        elif isinstance(self.resolution, (list, tuple)):
            assert len(self.resolution) == 2, "Resolution must be an int or a (width, height) tuple."

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if idx >= len(self.image_paths):
            raise IndexError("Index out of range")

        impath = self.image_paths[idx]
        depthpath = impath.replace("_rgb.png", "_depth.npy")

        rgb_image = imread_cv2(impath)
        depthmap = np.load(depthpath)
        depthmap[~np.isfinite(depthmap)] = 0.0
        depthmap = depthmap.astype(np.float32)

        out = self.pair_transform(rgb_image, depthmap)
        if isinstance(out, tuple) and len(out) == 3:
            img_t, depth_t, meta = out
        else:
            img_t, depth_t = out
            h, w = depth_t.shape[-2], depth_t.shape[-1]
            meta = {
                "orig_h": float(h),
                "orig_w": float(w),
                "resized_h": float(h),
                "resized_w": float(w),
                "scale_h": 1.0,
                "scale_w": 1.0,
                "pad_h": 0.0,
                "pad_w": 0.0,
                "padded_h": float(h),
                "padded_w": float(w),
            }
        return img_t, depth_t, meta


if __name__ == "__main__":
    dataset = HyperSimSimple(
        roots="/lc/data/3D/hypersim_processed/train",
        split="train",
        resolution=224,
    )
    print(f"Found {len(dataset)} total frames.")
    img_t, depth_t, _ = dataset[0]
    print("\nSample shapes:")
    print("Image shape:", img_t.shape)
    print("Depthmap shape:", depth_t.shape)

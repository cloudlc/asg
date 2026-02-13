# Active Spatial Guidance (ASG)

Official code repository for **Active Spatial Guidance: Eliminating Injected Positional Mechanisms in Vision Transformers**.

ASG is a training-only objective for ViTs:
- Remove injected positional mechanisms (AbsPE/RoPE/etc.) from the encoder.
- Add an auxiliary row/column coordinate regression loss on final-layer patch tokens during training.
- Discard the auxiliary head at inference, keeping a PE-free encoder with no extra inference cost.

## Repository Structure

```text
.
├── dinov3_cls.py                    # ImageNet-100 classification training
├── seg/
│   ├── dinov3_seg.py                # ADE20K segmentation training
│   ├── seg_head.py                  # UPerNet-style token decoder
│   ├── seg_aug.py                   # Segmentation augmentations/preprocess
│   └── seg_loss.py                  # Segmentation CE loss wrapper
├── depth/
│   ├── dinov3_depth.py              # Hypersim depth training/eval
│   ├── depth_head.py                # DPT-style depth decoder
│   ├── depth_loss.py                # Hybrid depth loss + metrics helpers
│   ├── hypersim_simple_dataset.py   # Hypersim dataset loader
│   └── aug.py                       # Depth augmentations/preprocess
├── core/
│   └── patch_pos.py                 # ASG row/col regression criteria
├── data/
│   ├── MultiScaleImageDataset.py    # Classification dataset wrappers
│   └── DynamicResolutionBatchSampler.py
└── timm_pe/
    ├── eva_relpos.py                # Relative-position variants
    └── eva_alibi.py                 # ALiBi variants
```
 
## Environment

Python dependencies used by this repo:
- `torch`, `torchvision`, `timm`
- `numpy`, `pandas`, `tqdm`, `Pillow`, `matplotlib`
- `opencv-python` (for depth data loading/processing)

This code is script-based (no packaged CLI yet). Configure experiments by editing the `args = SimpleNamespace(...)` blocks inside each training script.

## Dataset Layout

Update dataset roots in each script:
- `dinov3_cls.py`: `BASE_PATH = "/path/to/imagenet100"`
- `seg/dinov3_seg.py`: `BASE_PATH = "/path/to/ADEChallengeData2016"`
- `depth/dinov3_depth.py`: `BASE_PATH = "/path/to/3d_data"`

### 1) ImageNet-100 (classification)

Expected layout:

```text
/path/to/imagenet100/
├── train.X1/<class_name>/*.jpg
├── train.X2/<class_name>/*.jpg
├── train.X3/<class_name>/*.jpg
├── train.X4/<class_name>/*.jpg
└── val.X/<class_name>/*.jpg
```

### 2) ADE20K (segmentation)

Expected layout:

```text
/path/to/ADEChallengeData2016/
├── images/
│   ├── training/*.jpg
│   └── validation/*.jpg
└── annotations/
    ├── training/*.png
    └── validation/*.png
```

### 3) Hypersim processed (depth)

Expected layout under `"/path/to/3d_data"`:

```text
hypersim_processed/
├── train/**/**/*_rgb.png
├── train/**/**/*_depth.npy
├── test/**/**/*_rgb.png
└── test/**/**/*_depth.npy
```

## Run

From repository root:

```bash
python dinov3_cls.py
python seg/dinov3_seg.py
python depth/dinov3_depth.py
```

## Positional Strategy Switches

Set these in each script's `args`:

| Strategy | `use_abs_pos_emb` | `use_rot_pos_emb` | `use_rc_loss` |
|---|---:|---:|---:|
| No-PE | `False` | `False` | `False` |
| AbsPE | `True`  | `False` | `False` |
| RoPE  | `False` | `True`  | `False` |
| Guidance (ASG) | `False` | `False` | `True` |

Guidance weight is controlled by `rc_lambda` (with warmup via `lambda_min` and `warmup_steps_for_aux`).

For additional injected baselines in classification, set `pos_type` in `dinov3_cls.py`:
- `pos_type="relpos"` (from `timm_pe/eva_relpos.py`)
- `pos_type="alibi"` (from `timm_pe/eva_alibi.py`)

## License

MIT License. See `LICENSE`.

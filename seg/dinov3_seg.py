import math
import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import pandas as pd
import numpy as np
import random
from PIL import Image
from torch.nn import functional as F
from types import SimpleNamespace
import gc
import logging
from seg.seg_aug import TrainSegAug, EvalSegPreprocess
from seg.seg_head import UPerNetTokenHead
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import timm
BASE_PATH = "/path/to/ADEChallengeData2016"
OUTPUT_ROOT = "outputs"
data_root_default = BASE_PATH
base_path_default = data_root_default
args = SimpleNamespace(
    model_type= "dinov3",
    use_abs_pos_emb=False,
    use_rot_pos_emb=True,
    model_size='base',
    num_classes=150,
    batch_size=16,
    grad_accum_steps=1,
    train_img_size=336,
    eval_img_size=368,
    scale_jitter=(1.0, 1.3),
    use_cat_max_ratio=True,
    cat_max_ratio=0.70,
    cat_max_ratio_tries=10,
    eval_crop_mode="crop_or_pad",
    lr=7e-5,
    eta_min=1e-8,
    composite_lr=True,
    warmup_steps=3000,
    weight_decay=0.01,
    epochs=130,
    start_epoch=0,
    seed=55,
    use_rc_loss=False,
    rc_lambda=30.0,
    warmup_steps_for_aux=60,
    lambda_min=10,
    feature_layers=[2, 5, 8, 11],
    workers=5,
    color_jitter={"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.05},
    color_jitter_prob=0.1,
    train=True,
    ckpt_path=None,
    save_artifacts=False,
    clip_value=1.0,
    output_dir=os.path.join(OUTPUT_ROOT, "seg"),
    log_interval=50,
    csv_interval=3,
    compile_model=False,
    save_full_ckpt=False,
    resume_full_ckpt=False,
    resume_ckpt_path=None,
    resume_scheduler=True,
    resume_optimizer=True,
    resume_bs=False,
    base_path=base_path_default,
)
ckpt = None
if args.resume_full_ckpt and args.resume_ckpt_path:
    skip_keys = [
        "resume_full_ckpt",
        "resume_ckpt_path",
        "resume_bs",
        "resume_scheduler",
        "resume_optimizer",
    ]
    if not args.resume_scheduler:
        skip_keys.extend([
            "epochs",
            "warmup_steps",
            "warmup_ratio",
            "eta_min",
            "composite_lr",
        ])
    if not args.resume_bs:
        skip_keys.extend(["batch_size", "grad_accum_steps"])
    ckpt = torch.load(args.resume_ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", None)
    if ckpt_args is not None:
        for k, v in vars(ckpt_args).items():
            if k not in skip_keys:
                setattr(args, k, v)
if args.use_abs_pos_emb or args.use_rot_pos_emb:
    args.use_rc_loss = False
if args.eval_img_size != args.train_img_size:
    print("Best practice is to keep eval_img_size == train_img_size; overriding.", flush=True)
    args.eval_img_size = args.train_img_size

MODEL_NAME = f"vit_{args.model_size}_patch16_{args.model_type}"
TRAIN_IMAGE_PATH = os.path.join(args.base_path, 'images', 'training')
TRAIN_ANNOTATION_PATH = os.path.join(args.base_path, 'annotations', 'training')
VALID_IMAGE_PATH = os.path.join(args.base_path, 'images', 'validation')
VALID_ANNOTATION_PATH = os.path.join(args.base_path, 'annotations', 'validation')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_amp = torch.cuda.is_available()
use_bf16 = use_amp and torch.cuda.is_bf16_supported(including_emulation=False)
autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def _seed_worker(worker_id):
    worker_seed = args.seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

data_rng = torch.Generator()
data_rng.manual_seed(args.seed)
output_dir = args.output_dir
ckpt_output_dir = os.path.join(output_dir, "ckpt")
if args.save_artifacts:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_output_dir, exist_ok=True)
last_ckpt_path = os.path.join(ckpt_output_dir, "last.pth")
if args.resume_full_ckpt and args.resume_ckpt_path is None:
    args.resume_ckpt_path = last_ckpt_path

handlers = [logging.StreamHandler()]
if args.save_artifacts:
    log_file_path = os.path.join(output_dir, "run.log")
    handlers.append(logging.FileHandler(log_file_path))
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=handlers,
)
logger = logging.getLogger()

logger.info(f"Using device: {DEVICE}")
logger.info(f"Using mixed precision: {'disabled' if not use_amp else ('bfloat16' if use_bf16 else 'float16')}")
logger.info(f"Arguments: {args}")
logger.info(output_dir)

logger.info(output_dir)
if args.resume_full_ckpt and args.resume_ckpt_path and ckpt is not None:
    rng_state = ckpt.get("rng_state", None)
    if isinstance(rng_state, dict):
        try:
            if "python" in rng_state:
                random.setstate(rng_state["python"])
            if "numpy" in rng_state:
                np.random.set_state(rng_state["numpy"])
            if "torch" in rng_state:
                torch.set_rng_state(rng_state["torch"])
            if torch.cuda.is_available() and rng_state.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng_state["cuda"])
            if rng_state.get("data_rng") is not None:
                data_rng.set_state(rng_state["data_rng"])
            elif rng_state.get("train_generator") is not None:
                data_rng.set_state(rng_state["train_generator"])
            logger.info("Restored RNG states from checkpoint.")
        except Exception as exc:
            logger.warning("Failed to restore RNG states from checkpoint: %s", exc)
class SegmentationDataset(Dataset):
    """
    Custom PyTorch Dataset for semantic segmentation.

    Reads images and their corresponding segmentation masks, and applies
    appropriate data augmentation for training and validation.
    """
    def __init__(self, image_dir, annotation_dir, pair_transform):
        """
        Args:
            image_dir (str): Directory with all the images.
            annotation_dir (str): Directory with all the segmentation masks.
            img_size (int): The target size for the images and masks.
            is_train (bool): If true, applies training augmentations.
            mean (list): Mean for normalization.
            std (list): Standard deviation for normalization.
        """
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.pair_transform = pair_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        ann_path = os.path.join(self.annotation_dir, img_name.replace('.jpg', '.png'))

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(ann_path).convert('L')

        image_t, mask_t = self.pair_transform(image, mask)
        mask_t = mask_t.long() - 1
        return image_t, mask_t
img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]

train_dataset = SegmentationDataset(
    TRAIN_IMAGE_PATH,
    TRAIN_ANNOTATION_PATH,
    pair_transform=TrainSegAug(
        target_size=(args.train_img_size, args.train_img_size),
        scale_jitter=args.scale_jitter,
        cat_max_ratio=(args.cat_max_ratio if args.use_cat_max_ratio else None),
        cat_max_ratio_tries=args.cat_max_ratio_tries,
        ignore_index=0 if args.use_cat_max_ratio else None,
        color_jitter=args.color_jitter,
        color_jitter_prob=args.color_jitter_prob,
        normalize=True,
    ),
)
valid_dataset = SegmentationDataset(
    VALID_IMAGE_PATH,
    VALID_ANNOTATION_PATH,
    pair_transform=EvalSegPreprocess(
        target_size=(args.eval_img_size, args.eval_img_size),
        target_by="shorter",
        eval_crop_mode=args.eval_crop_mode,
        normalize=True,
    ),
)

loader_kwargs = dict(
    num_workers=args.workers,
    pin_memory=True,
    worker_init_fn=_seed_worker,
    generator=data_rng,
    persistent_workers=(args.workers > 0),
)
if args.workers > 0:
    loader_kwargs["prefetch_factor"] = 2

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    **loader_kwargs,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False,
    **loader_kwargs,
)

steps_per_epoch = len(train_loader)
accum_steps = max(1, int(getattr(args, "grad_accum_steps", 1)))
optimizer_steps_per_epoch = math.ceil(steps_per_epoch / accum_steps)
logger.info(f"✅ DataLoaders created successfully.")
logger.info(f"   - Training samples: {len(train_dataset)}, Batches per epoch: {len(train_loader)}")
logger.info(f"   - Validation samples: {len(valid_dataset)}, Batches per epoch: {len(valid_loader)}")
    

    
    
        


logger.info(f"🤖 Initializing model: {MODEL_NAME} for {args.num_classes} classes...")
model = timm.create_model(
    MODEL_NAME,
    pretrained=False,
    use_abs_pos_emb=args.use_abs_pos_emb,
    use_rot_pos_emb=args.use_rot_pos_emb,
    num_classes=0, 
    dynamic_img_size=True,
    img_size=args.train_img_size,
).to(DEVICE)


grid_h, grid_w = model.patch_embed.grid_size
embed_dims = [model.embed_dim] * len(args.feature_layers)
decoder = UPerNetTokenHead(
    embed_dims=embed_dims,
    num_classes=args.num_classes,
    grid_size=(grid_h, grid_w),
    out_size=(args.train_img_size, args.train_img_size),
    fpn_channels=256,
    ppm_bins=(1, 2, 3, 6),
    dropout=0.1,
    norm="gn",
).to(DEVICE)



logger.info(f'model.patch_embed.proj {model.patch_embed.proj}')
 
    



if args.compile_model:
    if hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile (mode='reduce-overhead').")
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        decoder = torch.compile(decoder)
    else:
        logger.warning("torch.compile not available; skipping compilation.")

dynamic = True
training_parameters = list(model.parameters()) + list(decoder.parameters())
param_groups = []
if args.use_rc_loss:
    grid_h, grid_w = model.patch_embed.grid_size
    dynamic = False
    from core.patch_pos import PatchRowColRegressionCriterion
    rowcol_loss = PatchRowColRegressionCriterion(
        feat_dim=model.embed_dim,
        grid_h=grid_h,
        grid_w=grid_w,
    ).to(DEVICE)
    training_parameters += list(rowcol_loss.parameters())
    param_groups.append({"params": rowcol_loss.parameters(), "weight_decay": 0.0, "lr": args.lr})


decay_params = []
no_decay_params = []

for n, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if n.endswith(".bias") or ("norm" in n.lower()):
        no_decay_params.append(p)
    else:
        decay_params.append(p)

for n, p in decoder.named_parameters():
    if not p.requires_grad:
        continue
    if n.endswith(".bias") or ("norm" in n.lower()):
        no_decay_params.append(p)
    else:
        decay_params.append(p)

param_groups.append({
    "params": decay_params,
    "lr": args.lr,
    "weight_decay": args.weight_decay,
})
param_groups.append({
    "params": no_decay_params,
    "lr": args.lr,
    "weight_decay": 0.0,
})

from seg_loss import MMSegCrossEntropyLoss
ce_criterion = MMSegCrossEntropyLoss(ignore_index=-1, avg_non_ignore=True)

optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
total_steps = args.epochs * optimizer_steps_per_epoch
if args.composite_lr:
    warmup_steps = min(args.warmup_steps, max(1, total_steps - 1))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-7 / args.lr,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.eta_min
    )
logger.info("✅ Initialized Loss, Optimizer, and LR Scheduler.")




def _infer_grid_hw(model, inputs):
    patch_size = model.patch_embed.patch_size
    if isinstance(patch_size, tuple):
        ph, pw = patch_size
    else:
        ph = pw = patch_size
    return (inputs.shape[-2] // ph, inputs.shape[-1] // pw)


def _round_to_multiple(x: int, m: int) -> int:
    return max(m, int(round(x / m) * m))

def _strip_prefix_tokens(features, grid_hw, num_prefix_tokens):
    if num_prefix_tokens <= 0:
        return features
    tokens_needed = grid_hw[0] * grid_hw[1]
    stripped = []
    for feat in features:
        if feat.dim() == 3 and feat.shape[1] == tokens_needed + num_prefix_tokens:
            stripped.append(feat[:, num_prefix_tokens:, :])
        else:
            stripped.append(feat)
    return stripped

def _forward_upernet(model, decoder, inputs, feature_layers):
    if not feature_layers:
        raise ValueError("feature_layers must be set.")
    grid_hw = _infer_grid_hw(model, inputs)
    features = model.forward_intermediates(
        inputs,
        indices=feature_layers,
        norm=False,
        intermediates_only=True,
        output_fmt="NLC",
    )
    features = _strip_prefix_tokens(features, grid_hw, model.num_prefix_tokens)
    outputs = decoder(features, grid_sizes=[grid_hw] * len(features), out_size=inputs.shape[-2:])
    return outputs, features, grid_hw


@torch.no_grad()
def fast_confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = -1):
    """
    pred/target: (B, H, W) or any same-shape tensors.
    Returns: confmat [C, C] on pred.device
    """
    pred = pred.view(-1).to(torch.int64)
    target = target.view(-1).to(torch.int64)

    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]

    idx = target * num_classes + pred
    conf = torch.bincount(idx, minlength=num_classes * num_classes)
    return conf.view(num_classes, num_classes)

ckpt_path = None
if args.train:
    use_scaler = use_amp and (autocast_dtype == torch.float16)
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=use_scaler)

    logger.info(f"\n🚀 Starting training for {MODEL_NAME}...")
    train_start_time = time.time()
    start_epoch = 0
    if args.resume_full_ckpt and args.resume_ckpt_path:
        if ckpt is not None:
            model.load_state_dict(ckpt.get("model", {}), strict=False)
            decoder.load_state_dict(ckpt.get("decoder", {}), strict=False)
            if args.resume_optimizer:
                if "optimizer" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer"])
            else:
                logger.info("Skipping optimizer state load (resume_optimizer=False).")
            if args.resume_scheduler:
                start_epoch = int(ckpt.get("epoch", 0))
                if "scheduler" in ckpt and ckpt["scheduler"] is not None:
                    scheduler.load_state_dict(ckpt["scheduler"])
            else:
                logger.info("Skipping scheduler state load (resume_scheduler=False).")
            if "scaler" in ckpt and ckpt["scaler"] is not None:
                scaler.load_state_dict(ckpt["scaler"])
            if args.use_rc_loss and "rowcol_loss" in ckpt and ckpt["rowcol_loss"] is not None:
                for k in ["row_targets", "col_targets", "row_index_full", "col_index_full"]:
                    if k in ckpt["rowcol_loss"]:
                        ckpt["rowcol_loss"].pop(k)
                rowcol_loss.load_state_dict(ckpt["rowcol_loss"])
            logger.info(f"Resumed full checkpoint from '{args.resume_ckpt_path}' at epoch {start_epoch}")
            training_history = ckpt.get("training_history", None)

    if not isinstance(locals().get("training_history", None), dict):
        if args.use_rc_loss:
            training_history = {
                'train_loss': [],
                'train_acc': [],
                'valid_acc': [],
                'valid_miou': [],
                'train_time': [],
                'val_time': [],
                'epoch': [],
                'step': [],
                'base_loss': [],
                'aux_loss': [],
            }
        else:
            training_history = {
                'train_loss': [],
                'train_acc': [],
                'valid_acc': [],
                'valid_miou': [],
                'train_time': [],
                'val_time': [],
                'epoch': [],
                'step': [],
            }
    if isinstance(locals().get("training_history", None), dict):
        training_history.setdefault('train_time', [])
        training_history.setdefault('val_time', [])
    def _pad_history(hist, fill_value=None):
        keys = [k for k, v in hist.items() if isinstance(v, list)]
        if not keys:
            return
        max_len = max(len(hist[k]) for k in keys)
        for k in keys:
            if len(hist[k]) < max_len:
                hist[k].extend([fill_value] * (max_len - len(hist[k])))
    if args.resume_full_ckpt and isinstance(locals().get("training_history", None), dict):
        _pad_history(training_history)
    step = 0
    if isinstance(locals().get("training_history", None), dict):
        step = int(training_history.get("step", [0])[-1]) if training_history.get("step") else 0
    best_acc = 0.0
    log_interval = getattr(args, "log_interval", 50)
    csv_interval = getattr(args, "csv_interval", 1) 
    last_trained_epoch = int(start_epoch)
    for epoch in range(start_epoch, args.epochs):
        epoch_train_start = time.time()
        model.train()
        decoder.train()
        running_loss_t = torch.zeros((), device=DEVICE)
        base_loss_t    = torch.zeros((), device=DEVICE)
        aux_loss_sum_t = torch.zeros((), device=DEVICE)
        train_correct_t = torch.zeros((), device=DEVICE)
        train_total_t   = torch.zeros((), device=DEVICE)
        train_samples_t = torch.zeros((), device=DEVICE)
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]", mininterval=0.5)
        
        optimizer.zero_grad(set_to_none=True)
        total_batches = len(train_loader)
        for batch_idx, (inputs, labels) in enumerate(train_pbar):
            if (batch_idx % accum_steps) == 0 and (total_batches - batch_idx) < accum_steps:
                break
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            bs = inputs.size(0)
            aux_loss = None
            with torch.amp.autocast(device_type=DEVICE.type, dtype=autocast_dtype, enabled=use_amp):
                outputs, features, grid_hw = _forward_upernet(
                    model, decoder, inputs, args.feature_layers
                )
                last_tokens = features[-1]

                loss = ce_criterion(outputs, labels)
                base_loss = loss              

                if args.use_rc_loss:
                    aux_loss = rowcol_loss(last_tokens)
                    aux_loss_sum_t += aux_loss.detach() * bs
                    t = min(1.0, (step + 1) / args.warmup_steps_for_aux)
                    lambda_t = args.lambda_min + (args.rc_lambda - args.lambda_min) * t
                    loss = base_loss + lambda_t * aux_loss
            
            loss_scaled = loss / accum_steps
            scaler.scale(loss_scaled).backward()

            do_step = ((batch_idx + 1) % accum_steps == 0) or (batch_idx + 1 == len(train_loader))
            if do_step:
                if args.clip_value is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(training_parameters, max_norm=args.clip_value)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            
            with torch.no_grad():
                pred = outputs.detach().argmax(dim=1)
                mask = (labels >= 0)
                valid_pixels = mask.sum()
                train_correct_t += ((pred == labels) & mask).sum()
                train_total_t   += valid_pixels
                train_samples_t += bs

            running_loss_t += loss.detach() * valid_pixels
            if args.use_rc_loss:
                base_loss_t += base_loss.detach() * valid_pixels

            if (step + 1) % log_interval == 0:
                avg_loss = (running_loss_t / (train_total_t.clamp_min(1))).float().item()
                avg_acc  = (train_correct_t / train_total_t.clamp_min(1)).float().item()

                if args.use_rc_loss:
                    avg_aux  = (aux_loss_sum_t / train_samples_t.clamp_min(1)).float().item()
                    train_pbar.set_postfix_str(f"loss={avg_loss:.4f} acc={avg_acc:.3f} aux={avg_aux:.4f}")
                else:
                    train_pbar.set_postfix_str(f"loss={avg_loss:.4f} acc={avg_acc:.3f}")

            step += 1

        
        train_time = time.time() - epoch_train_start
        model.eval()
        decoder.eval()
        val_correct_t = torch.zeros((), device=DEVICE)
        val_total_t   = torch.zeros((), device=DEVICE)
        confmat = torch.zeros((args.num_classes, args.num_classes), device=DEVICE, dtype=torch.int64)

        val_pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Validation]", mininterval=0.5)
        val_start = time.time()

        with torch.inference_mode():
            for inputs, labels in val_pbar:
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                with torch.amp.autocast(device_type=DEVICE.type, dtype=autocast_dtype, enabled=use_amp):
                    outputs, _, grid_hw = _forward_upernet(
                        model, decoder, inputs, args.feature_layers
                    )

                pred = outputs.argmax(dim=1)
                mask = (labels >= 0)

                val_correct_t += ((pred == labels) & mask).sum()
                val_total_t   += mask.sum()

                confmat += fast_confusion_matrix(pred, labels, args.num_classes, ignore_index=-1)

        val_time = time.time() - val_start
        confmat_f = confmat.to(torch.float32)
        intersection = torch.diag(confmat_f)
        union = confmat_f.sum(dim=1) + confmat_f.sum(dim=0) - intersection
        valid = union > 0
        epoch_val_miou = (intersection[valid] / union[valid]).mean().item() if valid.any() else 0.0

        epoch_val_acc = (val_correct_t / val_total_t.clamp_min(1)).float().item()
        epoch_train_acc = (train_correct_t / train_total_t.clamp_min(1)).float().item()

        denom_pixels = train_total_t.clamp_min(1).float()
        denom_samples = train_samples_t.clamp_min(1).float()
        epoch_train_loss = (running_loss_t / denom_pixels).float().item()
        if best_acc < epoch_val_acc:
            best_acc = epoch_val_acc

        logger.info(f"\nEpoch {epoch+1+args.start_epoch}/{args.epochs} Summary:")
        logger.info(f"Step {step} Summary:")

        if args.use_rc_loss:
            epoch_aux_loss  = (aux_loss_sum_t / denom_samples).float().item()
            epoch_base_loss = (base_loss_t / denom_pixels).float().item()
            logger.info(
                f"  Train Loss: {epoch_train_loss:.4f} | Aux Loss: {epoch_aux_loss:.4f} | Base Loss: {epoch_base_loss:.4f} | "
                f"Train Acc: {epoch_train_acc:.4f} | Valid Acc: {epoch_val_acc:.4f} | Valid mIoU: {epoch_val_miou:.4f} | "
                f"train_time: {train_time:.1f}s | val_time: {val_time:.1f}s\n"
            )
            training_history["aux_loss"].append(epoch_aux_loss)
            training_history["base_loss"].append(epoch_base_loss)
        else:
            logger.info(
                f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | "
                f"Valid Acc: {epoch_val_acc:.4f} | Valid mIoU: {epoch_val_miou:.4f} | "
                f"train_time: {train_time:.1f}s | val_time: {val_time:.1f}s\n"
            )

        training_history["train_loss"].append(epoch_train_loss)
        training_history["train_acc"].append(epoch_train_acc)
        training_history["valid_acc"].append(epoch_val_acc)
        training_history["valid_miou"].append(epoch_val_miou)
        training_history["train_time"].append(train_time)
        training_history["val_time"].append(val_time)
        training_history["epoch"].append(epoch + 1)
        training_history["step"].append(step)
        last_trained_epoch = epoch + 1
        if args.save_artifacts and (epoch + 1) % csv_interval == 0:
            pd.DataFrame(training_history).to_csv(os.path.join(output_dir, "train.csv"), index=False)
        if args.save_artifacts and args.save_full_ckpt:
            ckpt = {
                "epoch": epoch + 1,
                "step": step,
                "model": model.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict() if scaler is not None else None,
                "rowcol_loss": rowcol_loss.state_dict() if args.use_rc_loss else None,
                "training_history": training_history,
                "rng_state": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "data_rng": data_rng.get_state(),
                },
                "args": args,
            }
            torch.save(ckpt, last_ckpt_path)
            logger.info(f"Saved full checkpoint to '{last_ckpt_path}'")


    logger.info("🏁 Training complete.")
    logger.info(f"Best Accuracy: {best_acc:.4f}")
    logger.info(output_dir)
    history_df = pd.DataFrame(training_history)
    if args.save_artifacts:
        history_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)

    best_miou = history_df['valid_miou'].max()
    best_epoch = history_df.loc[history_df['valid_miou'].idxmax(), 'epoch']
    logger.info(f"Best miou: {best_miou:.4f} at epoch {best_epoch}")

    best_miou_row = history_df.loc[history_df['valid_miou'].idxmax()]
    best_miou_epoch = int(best_miou_row['epoch'])
    best_miou_val = best_miou_row['valid_miou']

    best_acc_row = history_df.loc[history_df['valid_acc'].idxmax()]
    best_acc_epoch = int(best_acc_row['epoch'])
    best_acc_val = best_acc_row['valid_acc']

    logger.info("\n--- Best Validation Metrics from History ---")
    logger.info(f"  Best miou:      {best_miou_val:.4f} (Epoch {best_miou_epoch})")
    logger.info(f"  Best acc:  {best_acc_val:.4f} (Epoch {best_acc_epoch})")
    logger.info("------------------------------------------")

del model, decoder
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


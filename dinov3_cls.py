import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset,TensorDataset, DataLoader
from data.MultiScaleImageDataset import MultiScaleImageDataset, CustomImageDataset
from data.DynamicResolutionBatchSampler import DynamicResolutionBatchSampler

from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import csv
import pickle
import numpy as np
import random
from PIL import Image
from torch.nn import functional as F
import torchvision.transforms.functional as TF
import sys
import timm
from types import SimpleNamespace
import gc
import time
import argparse
import logging

BASE_PATH = "/path/to/imagenet100"
OUTPUT_ROOT = "outputs"
args = SimpleNamespace(
    pos_type = None,
    dynamic_img_size=True,
    model_type= "dinov3",
    use_abs_pos_emb=True,
    use_rot_pos_emb=False,
    model_size='base',
    num_classes=100,
    patch_size = 16,
    grad_accum_steps=1,
    batch_size=32, 
    img_sizes=[224],
    val_img_sizes=[160, 176, 192, 208,224, 256, 272, 288, 320, 336, 352, 368, 384, 400, 416],
    lr=7e-5,
    eta_min=0.0,
    weight_decay=0.01,
    epochs=130,
    seed=29,
    use_patch_position_loss=False,
    use_rc_loss=True,
    rc_lambda=300.0,
    warmup_steps_for_aux=60,
    lambda_min=10,
    workers=5,
    train=True,
    val=True,
    ckpt_path=None,
    save_artifacts=False,
    save_full_ckpt=False,
    resume_full_ckpt=False,
    resume_ckpt_path=None,
    resume_scheduler=True,
    resume_optimizer=True,
    resume_bs=False,
    composite_lr=True,
    warmup_steps=3000,
    clip_value=1.0,
    log_interval=100,
    csv_interval=1,
    show_peak_gpu_mem=True,
    compile_model=False,
    data_root=BASE_PATH,
    output_root=OUTPUT_ROOT,
)
resume_ckpt=None
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
            "eta_min",
            "composite_lr",
        ])
    if not args.resume_bs:
        skip_keys.extend(["batch_size", "grad_accum_steps"])
    resume_ckpt = torch.load(args.resume_ckpt_path, map_location="cpu", weights_only=False)
    print(f"Resumed args from '{args.resume_ckpt_path}'")
    ckpt_args = resume_ckpt.get("args", None)
    if ckpt_args is not None:
        for k, v in vars(ckpt_args).items():
            if k not in skip_keys:
                setattr(args, k, v)
if args.pos_type is not None:
    args.has_pos = True
    args.use_rc_loss=False
    args.use_patch_position_loss=False
    args.dynamic_img_size=False
    args.val=False
if args.use_abs_pos_emb or args.use_rot_pos_emb:
    args.use_patch_position_loss=False
    args.use_rc_loss = False
offset = 0
MODEL_NAME = f"vit_{f'{args.pos_type}_' if args.pos_type is not None else ""}{args.model_size}_patch16_{args.model_type}"
output_dir = args.output_root
ckpt_output_dir = os.path.join(output_dir, "ckpt")

TRAIN_PATHS = [
    os.path.join(args.data_root, 'train.X1'),
    os.path.join(args.data_root, 'train.X2'),
    os.path.join(args.data_root, 'train.X3'),
    os.path.join(args.data_root, 'train.X4'),
]

VALID_PATH = os.path.join(args.data_root, 'val.X')
LABEL_PATH = os.path.join(BASE_PATH, 'Labels.json')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_amp = torch.cuda.is_available()
use_bf16 = use_amp and torch.cuda.is_bf16_supported(including_emulation=False)
autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

print(f"Using device: {DEVICE}", use_bf16, autocast_dtype)


if args.pos_type is not None:
    sys.path.append(r".")
    from timm_pe.eva_relpos import *
    from timm_pe.eva_alibi import *


np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
pos_prefix = ""
if args.pos_type is not None:
    pos_prefix = f"{args.pos_type}_"

abs_pos = ""
if args.use_abs_pos_emb:
    abs_pos = "_abs_pos"

rot_pos = ""
if args.use_rot_pos_emb:
    rot_pos = "_rot_pos"

patch_pos = ""
if args.use_patch_position_loss:
    patch_pos = "_patch_pos"

handlers = [logging.StreamHandler()]
if args.save_artifacts:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, "run.log")
    last_ckpt_path = os.path.join(ckpt_output_dir, f'last.pth')
    handlers.append(logging.FileHandler(log_file_path))
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=handlers,
)
logger = logging.getLogger()

logger.info(f"Using device: {DEVICE}")
logger.info(f"Using mixed precision: {'bfloat16' if use_bf16 else 'float16'}")
logger.info(args)
logger.info(output_dir)

logger.info("Cleaning up memory...")
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
logger.info("Memory cleanup complete.")

logger.info(args)
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import collections




all_class_dirs = [
    d
    for train_path in TRAIN_PATHS
    for d in os.listdir(train_path)
    if os.path.isdir(os.path.join(train_path, d))
]
selected_class_dirs = sorted(list(set(all_class_dirs)))[offset:args.num_classes+offset]
class_to_idx = {cls_name: i for i, cls_name in enumerate(selected_class_dirs)}

logger.info(f"✅ Efficiently loading the following {len(selected_class_dirs)} classes: {selected_class_dirs}")
args.num_classes = len(selected_class_dirs)
train_samples = []
for train_path_part in TRAIN_PATHS:
    for class_name in selected_class_dirs:
        class_idx = class_to_idx[class_name]
        class_dir = os.path.join(train_path_part, class_name)
        if os.path.isdir(class_dir):
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(class_dir, fname)
                    item = (path, class_idx)
                    train_samples.append(item)

valid_samples = []
for class_name in selected_class_dirs:
    class_idx = class_to_idx[class_name]
    class_dir = os.path.join(VALID_PATH, class_name)
    if os.path.isdir(class_dir):
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(class_dir, fname)
                item = (path, class_idx)
                valid_samples.append(item)

import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

img_mean = [0.485, 0.456, 0.406]
img_std  = [0.229, 0.224, 0.225]


def make_train_transform(size: int):
    t_list = [
        T.RandomResizedCrop(size, interpolation=InterpolationMode.BICUBIC, antialias=True),
        T.RandomHorizontalFlip(),
    ]
    t_list.extend([
        T.ToTensor(),
        T.Normalize(mean=img_mean, std=img_std),
    ])
    return T.Compose(t_list)

size_to_transform = {
    s: make_train_transform(s) for s in args.img_sizes
}

def make_valid_transform(img_size):
    return transforms.Compose([
        transforms.Resize(
            size=int(img_size * 1.143),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean, std=img_std),
    ])

valid_transforms = make_valid_transform(args.img_sizes[0])

valid_dataset = CustomImageDataset(valid_samples, transform=valid_transforms)

logger.info(f"Total validation images ({args.num_classes} classes): {len(valid_dataset)}")

batch_sampler = None
prefetch_kwargs = {"prefetch_factor": 2} if args.workers > 0 else {}
train_generator = torch.Generator()
train_generator.manual_seed(args.seed)
if args.resume_full_ckpt and args.resume_ckpt_path and resume_ckpt is not None:
    rng_state = resume_ckpt.get("rng_state", None)
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
                train_generator.set_state(rng_state["data_rng"])
            elif rng_state.get("train_generator") is not None:
                train_generator.set_state(rng_state["train_generator"])
            logger.info("Restored RNG states from checkpoint.")
        except Exception as exc:
            logger.warning("Failed to restore RNG states from checkpoint: %s", exc)
if len(args.img_sizes) == 1:
    train_dataset = CustomImageDataset(train_samples, transform=size_to_transform[args.img_sizes[0]])
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        **prefetch_kwargs,
    )
else:
    train_dataset = MultiScaleImageDataset(
        samples=train_samples,
        size_to_transform=size_to_transform
    )
    batch_sampler = DynamicResolutionBatchSampler(
        dataset=train_dataset,
        image_sizes=args.img_sizes,
        base_batch_size=args.batch_size,
        base_img_size=224,
        shuffle=True,
        drop_last=True,
        seed=42,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
    )
logger.info(f"Total training images ({args.num_classes} classes): {len(train_dataset)}")
valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.workers,
    pin_memory=True,
    persistent_workers=False,
)
steps_per_epoch = len(train_loader)
accum_steps = max(1, int(getattr(args, "grad_accum_steps", 1)))
optimizer_steps_per_epoch = math.ceil(steps_per_epoch / accum_steps)
logger.info(f"✅ DataLoaders for {args.num_classes} classes created successfully.")
logger.info(f"{steps_per_epoch=}, val_steps: {len(valid_loader)}")
logger.info(f"Effective batch size: {args.batch_size * accum_steps}")

import matplotlib.pyplot as plt
import numpy as np
import torchvision

def imshow(inp, title=None):
    """A helper function to denormalize and display an image tensor."""
    mean = np.array(img_mean)
    std = np.array(img_std)
    
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    
    plt.imshow(inp)
    if title is not None:
        plt.title(title, fontsize=10)
    plt.axis('off')

    

    
        




logger.info(f"🤖 Initializing model: {MODEL_NAME} for {args.num_classes} classes...")
model = timm.create_model(
    MODEL_NAME,
    pretrained=False,
    use_abs_pos_emb=args.use_abs_pos_emb,
    use_rot_pos_emb=args.use_rot_pos_emb,
    num_classes=args.num_classes,
    dynamic_img_size=args.dynamic_img_size,
    img_size=args.img_sizes[0],
).to(DEVICE)



logger.info(f'model.patch_embed.proj{model.patch_embed.proj}')
    



if args.compile_model and len(args.img_sizes)==1:
    if hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile (mode='reduce-overhead').")
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    else:
        logger.warning("torch.compile not available; skipping compilation.")

dynamic = True
training_parameters = list(model.parameters()) 
param_groups = []
if args.use_rc_loss:
    if len(args.img_sizes)==1:
        grid_h, grid_w = model.patch_embed.grid_size
        dynamic = False
        from core.patch_pos import PatchRowColRegressionCriterion
        rowcol_loss = PatchRowColRegressionCriterion(
            feat_dim=model.embed_dim,
            grid_h=grid_h,
            grid_w=grid_w,
        ).to(DEVICE)
    else:
        grid_h = grid_w = max(args.img_sizes)//args.patch_size
        from core.patch_pos import PatchRowColRegressionCriterionDynamic
        rowcol_loss = PatchRowColRegressionCriterionDynamic(
            feat_dim=model.embed_dim,
            grid_h=grid_h,
            grid_w=grid_w,
        ).to(DEVICE)
    training_parameters += list(rowcol_loss.parameters())
    param_groups.append({"params": rowcol_loss.parameters(), "weight_decay": 0.0, "lr": args.lr})
if args.use_patch_position_loss:
    if len(args.img_sizes)==1:
        from core.patch_pos import PatchPositionRegressionCriterion
        position_loss = PatchPositionRegressionCriterion(
            feat_dim=model.embed_dim,
            num_classes=model.patch_embed.num_patches
        ).to(DEVICE)
    else:
        max_grid = max(args.img_sizes)//args.patch_size
        max_patch_count = max_grid * max_grid
        from core.patch_pos import PatchPositionRegressionCriterionDynamic
        position_loss = PatchPositionRegressionCriterionDynamic(
            feat_dim=model.embed_dim,
            max_patch_count=max_patch_count
        ).to(DEVICE)
    training_parameters += list(position_loss.parameters())
    param_groups.append({"params": position_loss.parameters(), "weight_decay": 0.0, "lr": args.lr})

decay_params = []
no_decay_params = []

for n, p in model.named_parameters():
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
criterion = nn.CrossEntropyLoss()
if args.composite_lr:
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    total = sum(p.numel() for p in model.parameters())
    opt_total = sum(p.numel() for g in optimizer.param_groups for p in g["params"])
    print("model params:", total, "optimizer params:", opt_total)

    seen = set()
    dups = 0
    for g in optimizer.param_groups:
        for p in g["params"]:
            pid = id(p)
            if pid in seen:
                dups += 1
            seen.add(pid)
    print("duplicate params in groups:", dups)

    total_steps = args.epochs * optimizer_steps_per_epoch

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-7 / args.lr,
        end_factor=1.0,
        total_iters=args.warmup_steps
    )

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - args.warmup_steps,
        eta_min=1e-8
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[args.warmup_steps]
    )
else:
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    logger.info("✅ Model, Loss Function, and Optimizer are ready.")


    total_steps = args.epochs * optimizer_steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.eta_min)
    logger.info("✅ Step-based LR Scheduler is ready.")


    
sys.stdout.flush()
def get_patch_numbers(img_size, patch_size):
    """
    Calculate the number of patches in an image.

    Args:
        img_size (int or tuple): Size of the input image (H, W)
        patch_size (int): Size of the patch

    Returns:
        tuple: Number of patches in the image (H, W)
    """
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    assert 2 == len(img_size)
    hp, wp = img_size[0] // patch_size, img_size[1] // patch_size  
    return hp, wp


import csv

ckpt_path = None
if args.train:
    use_scaler = use_amp and (autocast_dtype == torch.float16)
    scaler = torch.amp.GradScaler(enabled=use_scaler)
    start_epoch = 0
    step = 0
    best_acc = 0.0
    if args.resume_full_ckpt and args.resume_ckpt_path:
        model.load_state_dict(resume_ckpt["model"])
        if args.resume_optimizer:
            if "optimizer" in resume_ckpt:
                optimizer.load_state_dict(resume_ckpt["optimizer"])
        else:
            logger.info("Skipping optimizer state load (resume_optimizer=False).")
        if args.resume_scheduler:
            start_epoch = resume_ckpt.get("epoch", 0)
            step = resume_ckpt.get("step", 0)
            if resume_ckpt.get("scheduler") is not None:
                scheduler.load_state_dict(resume_ckpt["scheduler"])
        else:
            logger.info("Skipping scheduler state load (resume_scheduler=False).")
        if resume_ckpt.get("scaler") is not None:
            scaler.load_state_dict(resume_ckpt["scaler"])
        if args.use_rc_loss and resume_ckpt.get("rowcol_loss") is not None:
            for k in ["row_targets", "col_targets", "row_index_full", "col_index_full"]:
                if k in resume_ckpt["rowcol_loss"]:
                    resume_ckpt["rowcol_loss"].pop(k)
            rowcol_loss.load_state_dict(resume_ckpt["rowcol_loss"])
        if args.use_patch_position_loss and resume_ckpt.get("position_loss") is not None:
            position_loss.load_state_dict(resume_ckpt["position_loss"])
        best_acc = resume_ckpt.get("best_acc", 0.0)
        logger.info(f"Resumed full checkpoint from '{args.resume_ckpt_path}' at epoch={start_epoch}, step={step}")
    logger.info(f"\n🚀 Starting training for {MODEL_NAME}...")

    if args.use_rc_loss or args.use_patch_position_loss:
        training_history = {
            'train_loss': [],
            'train_acc': [],
            'valid_acc': [],
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
            'train_time': [],
            'val_time': [],
            'epoch': [],
            'step': [],
        }
    if resume_ckpt is not None and resume_ckpt.get("training_history") is not None:
        training_history = resume_ckpt["training_history"]
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
    if args.resume_full_ckpt:
        _pad_history(training_history)
    log_interval = getattr(args, "log_interval", 50)
    csv_interval = getattr(args, "csv_interval", 1) 
    train_start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        epoch_train_start = time.time()
        model.train()

        aux_loss = None

        running_loss_t = torch.zeros((), device=DEVICE)
        aux_loss_sum_t = torch.zeros((), device=DEVICE)
        base_loss_t = torch.zeros((), device=DEVICE)
        train_correct_t = torch.zeros((), device=DEVICE)
        train_total = 0

        train_total = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]")
        if batch_sampler is not None:
            batch_sampler.set_epoch(epoch)
        
        total_batches = len(train_loader)
        optimizer.zero_grad(set_to_none=True)
        for step_in_epoch, (inputs, labels) in enumerate(train_pbar):
            if (step_in_epoch % accum_steps) == 0 and (total_batches - step_in_epoch) < accum_steps:
                break
            inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            bs = inputs.size(0)
            if args.show_peak_gpu_mem and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            aux_loss = None
            with torch.amp.autocast(device_type=DEVICE.type, dtype=autocast_dtype, enabled=use_amp):
                feats = model.forward_features(inputs)
                outputs = model.forward_head(feats)
                loss = criterion(outputs, labels)
                if args.use_rc_loss:
                    base_loss_t += loss.detach() * bs
                    if dynamic:
                        hp, wp = get_patch_numbers(inputs.shape[-2:], model.patch_embed.patch_size[0])
                        aux_loss = rowcol_loss(feats[:, model.num_prefix_tokens:, :], hp, wp)
                    else:
                        aux_loss = rowcol_loss(feats[:, model.num_prefix_tokens:, :])
                    

                    aux_loss_sum_t += aux_loss.detach() * bs
                    t = min(1.0, (step + 1) / args.warmup_steps_for_aux)
                    lambda_t = args.lambda_min + (args.rc_lambda - args.lambda_min) * t
                    loss = loss + lambda_t * aux_loss
                
                if args.use_patch_position_loss:
                    base_loss_t += loss.detach() * bs
                    aux_loss = position_loss(feats[:, model.num_prefix_tokens:, :])
                    aux_loss_sum_t += aux_loss.detach() * bs
                    t = min(1.0, (step + 1) / args.warmup_steps_for_aux)
                    lambda_t = args.lambda_min + (args.rc_lambda - args.lambda_min) * t
                    loss = loss + lambda_t * aux_loss
            
            loss_scaled = loss / accum_steps
            scaler.scale(loss_scaled).backward()

            do_step = ((step_in_epoch + 1) % accum_steps == 0) or (step_in_epoch + 1 == len(train_loader))
            if do_step:
                if args.clip_value is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(training_parameters, max_norm=args.clip_value)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss_t += loss.detach() * bs
            train_total += bs

            with torch.no_grad():
                pred = outputs.detach().argmax(dim=1)
                train_correct_t += (pred == labels).sum()

            if (step + 1) % log_interval == 0:
                avg_loss = (running_loss_t / train_total).float().item()
                avg_acc = (train_correct_t / train_total).float().item()
                peak_mb = None
                if args.show_peak_gpu_mem and torch.cuda.is_available():
                    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                msg = f"Epoch {epoch+1}/{args.epochs} step {step+1}: loss={avg_loss:.4f} acc={avg_acc:.3f}"
                if aux_loss is not None:
                    avg_aux = (aux_loss_sum_t / train_total).float().item()
                    msg += f" aux={avg_aux:.4f}"
                if peak_mb is not None:
                    msg += f" peak_mem={peak_mb:.0f}MB"
                train_pbar.set_postfix_str(msg)

            step += 1

        train_time = time.time() - epoch_train_start
        model.eval()
        val_correct_t = torch.zeros((), device=DEVICE)
        val_total = 0
        val_start = time.time()
        
        with torch.inference_mode():
            for inputs, labels in valid_loader:
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                with torch.amp.autocast(device_type=DEVICE.type, dtype=autocast_dtype, enabled=use_amp):
                    outputs = model(inputs)
                pred = outputs.argmax(dim=1)
                val_correct_t += (pred == labels).sum()
                val_total += labels.size(0)

        val_time = time.time() - val_start
        epoch_val_acc = (val_correct_t / val_total).item()
        if best_acc < epoch_val_acc:
            best_acc = epoch_val_acc

        epoch_train_acc  = (train_correct_t / train_total).item()
        epoch_train_loss = (running_loss_t / train_total).item()
        logger.info(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
        logger.info(f"\nStep {step} Summary:")

        if aux_loss is not None:
            epoch_aux_loss   = (aux_loss_sum_t / train_total).item()
            epoch_base_loss  = (base_loss_t / train_total).item()
            training_history['aux_loss'].append(epoch_aux_loss)
            training_history['base_loss'].append(epoch_base_loss)
            logger.info(
                f"  Train Loss: {epoch_train_loss:.4f} | Aux Loss: {epoch_aux_loss:.4f} | Base Loss: {epoch_base_loss:.4f} | "
                f"Train Acc: {epoch_train_acc:.4f} | Valid Acc: {epoch_val_acc:.4f} | "
                f"train_time: {train_time:.1f}s | val_time: {val_time:.1f}s\n"
            )
        else:
            logger.info(
                f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | "
                f"Valid Acc: {epoch_val_acc:.4f} | train_time: {train_time:.1f}s | val_time: {val_time:.1f}s\n"
            )
        

        
        training_history['train_loss'].append(epoch_train_loss)
        training_history['train_acc'].append(epoch_train_acc)
        training_history['valid_acc'].append(epoch_val_acc)  
        training_history['train_time'].append(train_time)
        training_history['val_time'].append(val_time)
        training_history['epoch'].append(epoch+1)
        training_history['step'].append(step+1)
        if args.save_artifacts and (epoch + 1) % csv_interval == 0:
            pd.DataFrame(training_history).to_csv(os.path.join(output_dir, "train.csv"), index=False)
        if args.save_artifacts and args.save_full_ckpt:
            ckpt = {
                "epoch": epoch + 1,
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict() if scaler is not None else None,
                "rowcol_loss": rowcol_loss.state_dict() if args.use_rc_loss else None,
                "position_loss": position_loss.state_dict() if args.use_patch_position_loss else None,
                "training_history": training_history,
                "rng_state": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "data_rng": train_generator.get_state(),
                },
                "args": args,
                "best_acc": best_acc,
            }
            torch.save(ckpt, last_ckpt_path)
            logger.info(f"Saved full checkpoint to '{last_ckpt_path}'")

        if (time.time() - train_start_time) >= (11.0 * 3600):
            logger.info("Stopping training: total running time exceeded 11 hours.")
            break


    logger.info("🏁 Training complete.")
    logger.info(f"Best Accuracy: {best_acc:.4f}")
    logger.info(output_dir)




    if args.save_artifacts:
        pd.DataFrame(training_history).to_csv(os.path.join(output_dir, "train.csv"), index=False)

if args.val:    
    val_results = {
        'img_size': [],
        'valid_acc': []
    }

    if not args.train:
        if ckpt_path is None:
            ckpt_path = args.ckpt_path
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
    model.to(DEVICE)
    model.eval()
    for img_size in args.val_img_sizes:
        valid_dataset.set_transform(make_valid_transform(img_size))
        batch_size = max(1, int((args.batch_size * 0.8 * 224 * 224) / (img_size * img_size)))
        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=(args.workers > 0),
            **prefetch_kwargs,
        )
        val_correct = 0
        val_total = 0
        with torch.inference_mode():
            for inputs, labels in valid_loader:
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                with torch.amp.autocast(device_type=DEVICE.type, dtype=autocast_dtype, enabled=use_amp):
                    outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_acc = val_correct / val_total
        val_results['img_size'].append(img_size)
        val_results['valid_acc'].append(epoch_val_acc)
        val_df = pd.DataFrame(val_results)
        if args.save_artifacts:
            val_df.to_csv(os.path.join(output_dir, "eval.csv"), index=False)
        logger.info(f"{img_size=}: {epoch_val_acc=}")

del model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


    
    
    



    
    
    



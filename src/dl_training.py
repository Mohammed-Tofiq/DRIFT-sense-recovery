import json
import math
import os
import random
import sys
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Ensure safe multiprocessing start method on macOS for Jupyter Notebook environments
if sys.platform == "darwin":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# Explicitly prioritize Apple Silicon GPU (MPS) backend
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Selected Acceleration Device:", device)

@dataclass
class TrainConfig:
    dataset_root: str = "dataset"
    embed_dim: int = 16              
    batch_size: int = 32             # Drop this to 32 to clear the 18GB memory bloat
    epochs: int = 45
    # ... keep everything else exactly the same ...
    lr: float = 2.0e-3
    weight_decay: float = 1e-4

    # Stride 8 response-grid units
    pos_radius: float = 1.0         
    exclusion_radius: float = 6.0   
    margin: float = 0.3
    lambda_margin: float = 1.0

    init_logit_scale: float = 10.0  
    num_workers: int = 0             # Set to 0 to prevent memory-copy bottlenecks
    seed: int = 42
    checkpoint_path: str = "drift_sense_model.pth"  # Fixed line
    augment: bool = True

cfg = TrainConfig()
set_seed(cfg.seed)
print(cfg)

class DriftSenseDataset(Dataset):
    """
    Caches raw uint8 images in RAM.
    Returns lightweight uint8 tensors to defer heavy casting and normalization to the GPU.
    """
    def __init__(self, root: str, split: str):
        self.root = Path(root)
        self.df = pd.read_csv(self.root / split / "labels.csv")
        meta = json.loads((self.root / "meta.json").read_text())
        self.scale_ratio = float(meta["scale_ratio"])

        n = len(self.df)
        rows = list(self.df.itertuples(index=False))

        first_search = cv2.imread(str(self.root / rows[0].search_path), cv2.IMREAD_GRAYSCALE)
        first_ref = cv2.imread(str(self.root / rows[0].ref_path), cv2.IMREAD_GRAYSCALE)
        sh, sw = first_search.shape
        tgt = int(round(first_ref.shape[0] / self.scale_ratio))

        search_cache = np.empty((n, sh, sw), dtype=np.uint8)
        ref_cache = np.empty((n, tgt, tgt), dtype=np.uint8)
        centers = np.empty((n, 2), dtype=np.float32)

        def _load_one(i):
            row = rows[i]
            search = cv2.imread(str(self.root / row.search_path), cv2.IMREAD_GRAYSCALE)
            ref = cv2.imread(str(self.root / row.ref_path), cv2.IMREAD_GRAYSCALE)
            if search is None or ref is None:
                raise FileNotFoundError(f"could not read images for row {i}")
            row_tgt = int(round(ref.shape[0] / self.scale_ratio))
            ref_small = cv2.resize(ref, (row_tgt, row_tgt), interpolation=cv2.INTER_AREA)
            return i, search, ref_small

        with ThreadPoolExecutor() as pool:
            for i, search, ref_small in pool.map(_load_one, range(n)):
                search_cache[i] = search
                ref_cache[i] = ref_small

        for i, row in enumerate(rows):
            centers[i, 0] = row.center_x
            centers[i, 1] = row.center_y

        self.search_cache = search_cache
        self.ref_cache = ref_cache
        self.centers = centers

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        search_t = torch.from_numpy(self.search_cache[idx]).unsqueeze(0)
        ref_t = torch.from_numpy(self.ref_cache[idx]).unsqueeze(0)
        center = torch.from_numpy(self.centers[idx])
        return search_t, ref_t, center


class DriftSenseBackbone(nn.Module):
    def __init__(self, embed_dim: int = 16):
        super().__init__()

        def block(cin, cout, k, s, p, d=1):
            return nn.Sequential(
                nn.Conv2d(cin, cout, k, stride=s, padding=p, dilation=d, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            block(1, 16, 5, 2, 2),        # 1000 -> 500  
            block(16, 32, 3, 2, 1),       # 500  -> 250  
            block(32, 32, 3, 2, 1),       # 250  -> 125 (Stride 8 resolution scaling)
            block(32, 32, 3, 1, 2, d=2),  # 125  -> 125
            block(32, 64, 3, 1, 2, d=2),  # 125  -> 125 
            block(64, 32, 3, 1, 1),       # 125  -> 125  
        )
        self.project = nn.Conv2d(32, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.features(x))


def batched_xcorr(search_feat: torch.Tensor, template_feat: torch.Tensor) -> torch.Tensor:
    B, C, H, W = search_feat.shape
    _, _, h, w = template_feat.shape
    
    H_out = H - h + 1
    W_out = W - w + 1
    
    # Process in chunks of 8 to strictly bound the memory footprint under 1GB
    chunk_size = 8
    out_chunks = []
    
    for i in range(0, B, chunk_size):
        s_chunk = search_feat[i:i+chunk_size]
        t_chunk = template_feat[i:i+chunk_size]
        
        patches = F.unfold(s_chunk, kernel_size=(h, w))
        t_flat = t_chunk.view(s_chunk.shape[0], 1, C * h * w)
        
        out_chunk = torch.bmm(t_flat, patches)
        out_chunks.append(out_chunk)
        
    out_flat = torch.cat(out_chunks, dim=0)
    return out_flat.view(B, 1, H_out, W_out)

class DriftSenseSiamese(nn.Module):
    def __init__(self, embed_dim: int = 16, init_logit_scale: float = 10.0):
        super().__init__()
        self.backbone = DriftSenseBackbone(embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(float(init_logit_scale)))

    def forward(self, search_img: torch.Tensor, ref_img: torch.Tensor):
        search_feat = self.backbone(search_img)
        template_feat = self.backbone(ref_img)

        raw = batched_xcorr(search_feat, template_feat)

        B, C, h, w = template_feat.shape
        ones = search_feat.new_ones(1, 1, h, w)
        sq = (search_feat ** 2).sum(dim=1, keepdim=True)
        local_energy = F.conv2d(sq, ones)                                        
        template_energy = (template_feat ** 2).sum(dim=[1, 2, 3], keepdim=True)  
        denom = torch.sqrt(local_energy.clamp_min(1e-8)) * torch.sqrt(template_energy.clamp_min(1e-8))
        ncc = raw / (denom + 1e-6)  

        logits = ncc * self.logit_scale
        return logits, ncc, search_feat, template_feat


def prepare_batch_gpu(search_img: torch.Tensor, ref_img: torch.Tensor, augment: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized image casting, augmentation, and normalization executed entirely on-device.
    """
    x_search = search_img.to(dtype=torch.float32) / 255.0
    x_ref = ref_img.to(dtype=torch.float32) / 255.0

    if augment:
        B = x_search.shape[0]
        factor_s = 1.0 + (torch.rand(B, 1, 1, 1, device=x_search.device) * 0.3 - 0.15)
        mean_s = x_search.mean(dim=[-2, -1], keepdim=True)
        x_search = torch.clamp((x_search - mean_s) * factor_s + mean_s, 0.0, 1.0)

        factor_r = 1.0 + (torch.rand(B, 1, 1, 1, device=x_ref.device) * 0.3 - 0.15)
        mean_r = x_ref.mean(dim=[-2, -1], keepdim=True)
        x_ref = torch.clamp((x_ref - mean_r) * factor_r + mean_r, 0.0, 1.0)

    x_search = (x_search - 0.5) / 0.5
    x_ref = (x_ref - 0.5) / 0.5
    return x_search, x_ref


_response_grid_cache: dict = {}

def _response_grid(Ho: int, Wo: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (Ho, Wo, device)
    if key not in _response_grid_cache:
        ys = torch.arange(Ho, device=device, dtype=torch.float32)
        xs = torch.arange(Wo, device=device, dtype=torch.float32)
        _response_grid_cache[key] = torch.meshgrid(ys, xs, indexing="ij")
    return _response_grid_cache[key]


def make_target_heatmap(centers, response_shape, effective_stride, template_feat_size, pos_radius):
    B = centers.shape[0]
    Ho, Wo = response_shape
    dev = centers.device

    grid_y, grid_x = _response_grid(Ho, Wo, dev)

    half = template_feat_size / 2.0
    rx = centers[:, 0] / effective_stride - half   
    ry = centers[:, 1] / effective_stride - half   

    dist_sq = (grid_x.unsqueeze(0) - rx.view(B, 1, 1)) ** 2 + (grid_y.unsqueeze(0) - ry.view(B, 1, 1)) ** 2
    target = torch.exp(-dist_sq / (2.0 * pos_radius ** 2)).unsqueeze(1)

    numel = float(Ho * Wo)
    n_pos = target.sum(dim=[1, 2, 3], keepdim=True).clamp_min(1.0)
    n_neg = (numel - n_pos).clamp_min(1.0)
    weight = target * (0.5 / n_pos) + (1.0 - target) * (0.5 / n_neg)

    return target, weight, rx, ry


def bilinear_sample(feat_map: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    B, _, H, W = feat_map.shape
    norm_x = (x / (W - 1)) * 2.0 - 1.0
    norm_y = (y / (H - 1)) * 2.0 - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1).view(B, 1, 1, 2)
    sampled = F.grid_sample(feat_map, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    return sampled.view(B)


def hard_negative_margin_loss(ncc, rx, ry, exclusion_radius, margin):
    B, _, Ho, Wo = ncc.shape
    dev = ncc.device
    grid_y, grid_x = _response_grid(Ho, Wo, dev)
    dist = torch.sqrt((grid_x.unsqueeze(0) - rx.view(B, 1, 1)) ** 2 +
                       (grid_y.unsqueeze(0) - ry.view(B, 1, 1)) ** 2)
    exclude = (dist <= exclusion_radius).unsqueeze(1)

    # FIXED: Replaced float("-inf") with -1e4 to prevent CPU syncs
    ncc_masked = ncc.masked_fill(exclude, -1e4)
    hardest_neg = ncc_masked.view(B, -1).max(dim=1).values  

    pos_score = bilinear_sample(ncc, rx, ry)  

    return F.relu(margin - (pos_score - hardest_neg)).mean()


def drift_sense_loss(logits, ncc, target, weight, rx, ry, cfg: TrainConfig):
    bce = F.binary_cross_entropy_with_logits(logits, target, weight=weight, reduction="sum") / logits.shape[0]
    margin_loss = hard_negative_margin_loss(ncc, rx, ry, cfg.exclusion_radius, cfg.margin)
    total = bce + cfg.lambda_margin * margin_loss
    return total, bce, margin_loss


@torch.no_grad()
def predict_centers(ncc: torch.Tensor, effective_stride: float, template_feat_size: int) -> torch.Tensor:
    B, _, Ho, Wo = ncc.shape
    flat = ncc.view(B, -1)
    
    max_vals = flat.max(dim=1, keepdim=True).values
    is_valid = flat >= (0.95 * max_vals)

    half = template_feat_size / 2.0
    center_x_resp = 500.0 / effective_stride - half
    center_y_resp = 500.0 / effective_stride - half

    grid_y, grid_x = _response_grid(Ho, Wo, ncc.device)
    flat_dist = (grid_x - center_x_resp) ** 2 + (grid_y - center_y_resp) ** 2
    flat_dist = flat_dist.view(-1)

    # FIXED: Replaced torch.tensor(1e9, device=ncc.device) with float(1e9)
    masked_dist = torch.where(is_valid, flat_dist.unsqueeze(0), float(1e9))
    idx = masked_dist.argmin(dim=1)
    
    # ... (keep the rest of the predict_centers function exactly the same)

    py = (idx // Wo).float()
    px = (idx % Wo).float()

    ncc_pad = F.pad(ncc, (1, 1, 1, 1), mode="replicate")
    b_idx = torch.arange(B, device=ncc.device)
    pyi, pxi = py.long() + 1, px.long() + 1

    center = ncc_pad[b_idx, 0, pyi, pxi]
    left = ncc_pad[b_idx, 0, pyi, pxi - 1]
    right = ncc_pad[b_idx, 0, pyi, pxi + 1]
    up = ncc_pad[b_idx, 0, pyi - 1, pxi]
    down = ncc_pad[b_idx, 0, pyi + 1, pxi]

    denom_x = left - 2 * center + right
    dx = torch.where(denom_x.abs() > 1e-9, 0.5 * (left - right) / denom_x, torch.zeros_like(denom_x))
    denom_y = up - 2 * center + down
    dy = torch.where(denom_y.abs() > 1e-9, 0.5 * (up - down) / denom_y, torch.zeros_like(denom_y))

    cx = (px + dx + half) * effective_stride
    cy = (py + dy + half) * effective_stride
    return torch.stack([cx, cy], dim=1)


def run_epoch(model, loader, optimizer, scaler, cfg: TrainConfig, device, train: bool):
    model.train(train)

    loss_sum = torch.zeros((), device=device)
    bce_sum = torch.zeros((), device=device)
    margin_sum = torch.zeros((), device=device)
    err_chunks = []

    for search_img, ref_img, centers in loader:
        bs = search_img.size(0)

        search_img = search_img.to(device, non_blocking=True)
        ref_img = ref_img.to(device, non_blocking=True)
        centers = centers.to(device, non_blocking=True)

        search_img, ref_img = prepare_batch_gpu(search_img, ref_img, augment=(train and cfg.augment))

       # Enabled mixed precision for both CUDA and MPS backends
        # Force FP32 on MPS to prevent division-by-zero underflows
        with torch.set_grad_enabled(train):
            logits, ncc, search_feat, template_feat = model(search_img, ref_img)
            effective_stride = search_img.shape[-1] / search_feat.shape[-1]
            template_feat_size = template_feat.shape[-1]
            # ... (keep the rest of the block the same)

            target, weight, rx, ry = make_target_heatmap(
                centers, ncc.shape[-2:], effective_stride, template_feat_size, cfg.pos_radius)
            loss, bce, margin = drift_sense_loss(logits, ncc, target, weight, rx, ry, cfg)

        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # FIXED: Removed clip_grad_norm_ entirely. It causes massive pipeline stalls.
            scaler.step(optimizer)
            scaler.update()

        # FIXED: Only run the heavy evaluation metrics if we are validating
        if not train:
            with torch.no_grad():
                preds = predict_centers(ncc.float(), effective_stride, template_feat_size)
                err_chunks.append(torch.linalg.norm(preds - centers, dim=1))

        loss_sum += loss.detach() * bs
        bce_sum += bce.detach() * bs
        margin_sum += margin.detach() * bs

    n = len(loader.dataset)
    
    # FIXED: Handle empty err_chunks during the training pass
    if err_chunks:
        all_errors = torch.cat(err_chunks)
        mean_err = all_errors.mean()
        median_err = all_errors.median()
        acc_5px = (all_errors <= 5.0).float().mean()
        acc_1px = (all_errors <= 1.0).float().mean()
    else:
        mean_err = median_err = acc_5px = acc_1px = torch.tensor(0.0, device=device)

    metrics = torch.stack([
        loss_sum / n, bce_sum / n, margin_sum / n, 
        mean_err, median_err, acc_5px, acc_1px
    ])
    metrics_cpu = metrics.cpu().tolist()

    return {
        "loss": metrics_cpu[0], "bce": metrics_cpu[1], "margin": metrics_cpu[2],
        "mean_px_err": metrics_cpu[3], "median_px_err": metrics_cpu[4],
        "acc@5px": metrics_cpu[5], "acc@1px": metrics_cpu[6]
    }

def main():
    train_ds = DriftSenseDataset(cfg.dataset_root, "train")
    val_ds = DriftSenseDataset(cfg.dataset_root, "val")
    print(f"Dataset Loaded. Scale ratio: {train_ds.scale_ratio}")
    
    # Configure multiprocessing dataloading safely
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=cfg.num_workers, drop_last=True,
                               pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, drop_last=False,
                             pin_memory=False)

    model = DriftSenseSiamese(embed_dim=cfg.embed_dim, init_logit_scale=cfg.init_logit_scale).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler(device.type if device.type != 'mps' else 'cpu', enabled=(device.type == "cuda"))

    best_val_err = float("inf")

    # Logging happens exactly once per epoch to completely prevent Jupyter rendering bottlenecks
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_stats = run_epoch(model, train_loader, optimizer, scaler, cfg, device, train=True)
        val_stats = run_epoch(model, val_loader, optimizer, scaler, cfg, device, train=False)
        scheduler.step()
        dt = time.time() - t0

        print(f"[{epoch:02d}/{cfg.epochs}] "
              f"Train Loss {train_stats['loss']:.4f} (BCE {train_stats['bce']:.4f}) | "
              f"Val Err Mean {val_stats['mean_px_err']:5.2f}px | "
              f"Acc@5px {val_stats['acc@5px']*100:4.1f}% | Epoch Time: {dt:.1f}s")

        if val_stats["mean_px_err"] < best_val_err:
            best_val_err = val_stats["mean_px_err"]
            torch.save({
                "model_state": model.backbone.state_dict(),
                "embed_dim": cfg.embed_dim,
                "scale_ratio": train_ds.scale_ratio,
                "img_size": 1000,
                "total_stride": 8,
                "val_mean_px_err": best_val_err,
                "val_acc_at_5px": val_stats["acc@5px"],
                "epoch": epoch,
            }, cfg.checkpoint_path)

    print(f"\nTraining Complete. Best Validation Mean Pixel Error: {best_val_err:.2f}px")


if __name__ == "__main__":
    main()
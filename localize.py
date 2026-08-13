"""
inference.py
============
DL-powered evaluation harness for Drift-Sense.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob 

DEFAULT_WEIGHTS = "drift_sense_model.pth"
DEFAULT_DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
ACCURACY_THRESHOLD_PX = 5.0
PSR_WARNING_THRESHOLD = 4.0

if DEFAULT_DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True

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
            block(32, 32, 3, 2, 1),       # 250  -> 125 (Scaled down to Stride 8)
            block(32, 32, 3, 1, 2, d=2),  # 125  -> 125
            block(32, 64, 3, 1, 2, d=2),  # 125  -> 125 
            block(64, 32, 3, 1, 1),       # 125  -> 125  
        )
        self.project = nn.Conv2d(32, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.features(x))

@torch.no_grad()
def extract_features(model: nn.Module, img_t: torch.Tensor) -> torch.Tensor:
    x = (img_t / 255.0 - 0.5) / 0.5
    return model(x)

def batched_xcorr(search_feat: torch.Tensor, template_feat: torch.Tensor) -> torch.Tensor:
    B, C, H, W = search_feat.shape
    if search_feat.device.type == 'mps':
        out_chunks = []
        for i in range(B):
            out_chunks.append(F.conv2d(search_feat[i:i+1], template_feat[i:i+1]))
        return torch.cat(out_chunks, dim=0)
    else:
        search_r = search_feat.reshape(1, B * C, H, W)
        out = F.conv2d(search_r, template_feat, groups=B)
        return out.reshape(B, 1, out.shape[-2], out.shape[-1])

def compute_ncc(search_feat: torch.Tensor, template_feat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    numerator = batched_xcorr(search_feat, template_feat)
    _, _, h, w = template_feat.shape
    ones = search_feat.new_ones(1, 1, h, w)
    sq = (search_feat ** 2).sum(dim=1, keepdim=True)
    local_energy = F.conv2d(sq, ones)
    template_energy = (template_feat ** 2).sum(dim=[1, 2, 3], keepdim=True)
    denom = torch.sqrt(local_energy.clamp_min(1e-8)) * torch.sqrt(template_energy.clamp_min(1e-8))
    return numerator / (denom + eps)

_response_grid_cache: dict = {}

def _response_grid(Ho: int, Wo: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (Ho, Wo, device)
    if key not in _response_grid_cache:
        ys = torch.arange(Ho, device=device, dtype=torch.float32)
        xs = torch.arange(Wo, device=device, dtype=torch.float32)
        _response_grid_cache[key] = torch.meshgrid(ys, xs, indexing="ij")
    return _response_grid_cache[key]

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

    masked_dist = torch.where(is_valid, flat_dist.unsqueeze(0), torch.tensor(1e9, device=ncc.device))
    idx = masked_dist.argmin(dim=1)

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

def peak_to_sidelobe_ratio(ncc: torch.Tensor) -> torch.Tensor:
    flat = ncc.view(ncc.shape[0], -1)
    peak = flat.max(dim=1).values
    mean = flat.mean(dim=1)
    std = flat.std(dim=1)
    return (peak - mean) / std.clamp_min(1e-6)

_MODEL_CACHE: dict = {}
_MODEL_CACHE_MAX = 4

def _load_model(weights_path: str = DEFAULT_WEIGHTS, device: torch.device = DEFAULT_DEVICE):
    key = (str(weights_path), str(device))
    if key not in _MODEL_CACHE:
        if len(_MODEL_CACHE) >= _MODEL_CACHE_MAX:
            _MODEL_CACHE.pop(next(iter(_MODEL_CACHE)))
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        model = DriftSenseBackbone(embed_dim=ckpt.get("embed_dim", 16))
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()
        _MODEL_CACHE[key] = {
            "model": model,
            "scale_ratio": float(ckpt.get("scale_ratio", 10.0)),
        }
    entry = _MODEL_CACHE[key]
    return entry["model"], entry["scale_ratio"]

def match_drift(reference_img: np.ndarray, search_img: np.ndarray,
                 weights_path: str = DEFAULT_WEIGHTS,
                 device: torch.device = DEFAULT_DEVICE) -> Tuple[float, float]:

    model, _ = _load_model(weights_path, device)
    
    # HARDCODED NOMINAL SCALE: Forces the model to learn scale robustness 
    # instead of relying on a fixed known scale from the checkpoint.
    scale_ratio = 10.0 

    with torch.inference_mode():
        ref_small_size = int(round(reference_img.shape[0] / scale_ratio))

        search_t = torch.from_numpy(search_img).to(device, non_blocking=True).unsqueeze(0).unsqueeze(0).float()
        ref_t = torch.from_numpy(reference_img).to(device, non_blocking=True).unsqueeze(0).unsqueeze(0).float()

        if ref_t.shape[-1] != ref_small_size:
            ref_t = F.interpolate(ref_t, size=(ref_small_size, ref_small_size), mode="area")

        search_feat = extract_features(model, search_t)
        template_feat = extract_features(model, ref_t)

        ncc = compute_ncc(search_feat, template_feat)

        effective_stride = search_img.shape[-1] / search_feat.shape[-1]
        template_feat_size = template_feat.shape[-1]

        psr_t = peak_to_sidelobe_ratio(ncc)
        centers_t = predict_centers(ncc, effective_stride, template_feat_size)[0]

        psr = psr_t.item()
        cx, cy = centers_t.tolist()

        if psr < PSR_WARNING_THRESHOLD:
            print(f"[low-confidence] PSR={psr:.2f} < {PSR_WARNING_THRESHOLD:.1f} -- "
                  f"peak isn't well-separated from the background response (possible periodic alias)")

    return float(cx), float(cy)

def _timed_call(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms

def save_failure_overlay(search_img: np.ndarray, pred_xy, gt_xy, error_px: float,
                          save_path: str, pair_id=None) -> str:
    vis = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR) if search_img.ndim == 2 else search_img.copy()
    gt = tuple(np.round(gt_xy).astype(int))
    pred = tuple(np.round(pred_xy).astype(int))
    cv2.drawMarker(vis, gt, (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=25, thickness=2)
    cv2.drawMarker(vis, pred, (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=25, thickness=2)
    cv2.circle(vis, gt, int(ACCURACY_THRESHOLD_PX), (255, 255, 0), 1)
    cv2.line(vis, gt, pred, (0, 165, 255), 1)
    label = f"error={error_px:.2f}px" + (f"  pair={pair_id}" if pair_id is not None else "")
    cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(save_path, vis)
    return save_path

def _load_pair(root: Path, row: dict):
    search = cv2.imread(str(root / row["search_path"]), cv2.IMREAD_GRAYSCALE)
    ref = cv2.imread(str(root / row["ref_path"]), cv2.IMREAD_GRAYSCALE)
    if search is None or ref is None:
        raise FileNotFoundError(f"Could not read images for pair {row.get('id')}")
    return ref, search

def evaluate(data_dir: str, split: str, weights_path: str, out_dir: str,
             device: torch.device = DEFAULT_DEVICE) -> dict:
    root = Path(data_dir)
    with open(root / split / "labels.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows found in {root / split / 'labels.csv'}")

    errors, latencies = [], []
    worst = {"error": -1.0}
    
    # NEW: List to hold all data for the required CSV manifest
    manifest_data = []

    with ThreadPoolExecutor(max_workers=1) as pool:
        next_pair = pool.submit(_load_pair, root, rows[0])
        for i, row in enumerate(rows):
            ref_img, search_img = next_pair.result()
            if i + 1 < len(rows):
                next_pair = pool.submit(_load_pair, root, rows[i + 1])

            gt = (float(row["center_x"]), float(row["center_y"]))

            (pred_x, pred_y), elapsed_ms = _timed_call(match_drift, ref_img, search_img, weights_path, device)
            err = math.hypot(pred_x - gt[0], pred_y - gt[1])

            errors.append(err)
            latencies.append(elapsed_ms)
            
            # NEW: Append all required fields for the judging CSV
            manifest_row = {
                "id": row.get("id", i),
                "ref_path": row.get("ref_path", ""),
                "search_path": row.get("search_path", ""),
                "gt_x": gt[0],
                "gt_y": gt[1],
                "pred_x": pred_x,
                "pred_y": pred_y,
                "error_px": err,
                "latency_ms": elapsed_ms
            }
            # Catch any extra metadata generated during dataset creation
            for k, v in row.items():
                if k not in manifest_row and k not in ["center_x", "center_y"]:
                    manifest_row[k] = v
            manifest_data.append(manifest_row)

            if err > worst["error"]:
                worst = {"error": err, "search_img": search_img, "pred": (pred_x, pred_y),
                          "gt": gt, "id": row.get("id", i)}

    errors_arr = np.array(errors)
    latencies_arr = np.array(latencies)
    
    pass_5px = float((errors_arr <= 5.0).mean())
    pass_4px = float((errors_arr <= 4.0).mean())
    pass_2px = float((errors_arr <= 2.0).mean())
    pass_1px = float((errors_arr <= 1.0).mean())

    # NEW: Explicit hardware and timing method prints required by checklist
    print(f"Hardware           : {device.type.upper()} Acceleration")
    print(f"Timing Method      : Python time.perf_counter()")
    print("-" * 40)
    print(f"n_pairs            : {len(rows)}")
    print(f"mean error (px)    : {errors_arr.mean():.3f}")
    print(f"median error (px)  : {np.median(errors_arr):.3f}")
    print(f"p95 error (px)     : {np.percentile(errors_arr, 95):.3f}")
    print(f"pass @ 5px         : {pass_5px * 100:.1f}%")
    print(f"pass @ 4px         : {pass_4px * 100:.1f}%")
    print(f"pass @ 2px         : {pass_2px * 100:.1f}%")
    print(f"pass @ 1px         : {pass_1px * 100:.1f}%")
    print(f"mean latency (ms)  : {latencies_arr.mean():.3f}")
    print(f"median latency(ms) : {np.median(latencies_arr):.3f}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # NEW: Write the predictions CSV manifest
    csv_path = out_path / "predictions_manifest.csv"
    if manifest_data:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=manifest_data[0].keys())
            writer.writeheader()
            writer.writerows(manifest_data)
        print(f"Predictions CSV    -> {csv_path}")

    overlay_path = out_path / f"worst_pair_{worst['id']}_err{worst['error']:.1f}px.png"
    save_failure_overlay(worst["search_img"], worst["pred"], worst["gt"], worst["error"],
                          str(overlay_path), pair_id=worst["id"])
    print(f"worst-pair overlay -> {overlay_path}")

    summary = {
        "hardware": device.type.upper(),
        "timing_method": "time.perf_counter()",
        "n_pairs": len(rows),
        "mean_error_px": float(errors_arr.mean()),
        "median_error_px": float(np.median(errors_arr)),
        "p95_error_px": float(np.percentile(errors_arr, 95)),
        "pass_rate_at_5px": pass_5px,
        "pass_rate_at_4px": pass_4px,
        "pass_rate_at_2px": pass_2px,
        "pass_rate_at_1px": pass_1px,
        "mean_latency_ms": float(latencies_arr.mean()),
        "median_latency_ms": float(np.median(latencies_arr)),
        "worst_pair_id": worst["id"],
        "worst_pair_error_px": float(worst["error"]),
    }
    with open(out_path / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary

def main():
    parser = argparse.ArgumentParser(description="Drift-Sense DL-powered evaluation")
    parser.add_argument("--data-dir", type=str, default="dataset")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS)
    parser.add_argument("--out-dir", type=str, default="eval_out")
    
    # CLI ARGUMENTS FOR REAL-WORLD INFERENCE
    parser.add_argument("--ref-img", type=str, default=None,
                         help="Path to a reference image")
    parser.add_argument("--search-img", type=str, default=None,
                         help="Path to a search image")
    
    # CLI ARGUMENT FOR BATCH FOLDER (No CSV)
    parser.add_argument("--batch-dir", type=str, default=None,
                         help="Path to a folder of images to evaluate without a CSV")
                         
    args = parser.parse_args()

    # 1. REAL-WORLD INFERENCE MODE (Single Pair, No CSV)
    if args.ref_img or args.search_img:
        if not (args.ref_img and args.search_img):
            parser.error("--ref-img and --search-img must be provided together")

        ref_img = cv2.imread(args.ref_img, cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(args.search_img, cv2.IMREAD_GRAYSCALE)
        
        if ref_img is None or search_img is None:
            raise FileNotFoundError(
                f"Could not read one of the provided images: ref={args.ref_img}, search={args.search_img}"
            )

        cx, cy = match_drift(ref_img, search_img, args.weights)
        print(f"predicted (x, y): ({cx:.3f}, {cy:.3f})")
        return
    
    # 2. BATCH FOLDER MODE (No CSV)
    if args.batch_dir:
        if not os.path.exists(args.batch_dir):
            print(f"Error: Directory {args.batch_dir} not found.")
            return
            
        print(f"Evaluating folder: {args.batch_dir} (No CSV mode)")
        ref_images = glob.glob(os.path.join(args.batch_dir, "*ref*.png")) 
        
        for ref_path in ref_images:
            search_path = ref_path.replace("ref", "search") 
            
            if os.path.exists(search_path):
                r_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
                s_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
                
                cx, cy = match_drift(r_img, s_img, args.weights)
                print(f"Pair {os.path.basename(ref_path)} -> predicted (x, y): ({cx:.3f}, {cy:.3f})")
            else:
                print(f"Warning: Could not find matching search image for {ref_path}")
                
        return

    # 3. STANDARD EVALUATION MODE (Uses CSV) - WITH SAFETY CHECK
    expected_csv_path = Path(args.data_dir) / args.split / "labels.csv"
    
    if not expected_csv_path.exists():
        print(f"\n ERROR: Could not find the dataset CSV at '{expected_csv_path}'")
        print("Please ensure your custom dataset follows this exact directory structure:")
        print(f"  {args.data_dir}/")
        print(f"    └── {args.split}/")
        print("        ├── labels.csv")
        print("        ├── (your reference images)")
        print("        └── (your search images)\n")
        print("Alternatively, use --batch-dir to evaluate a folder without a CSV.")
        return 

    evaluate(args.data_dir, args.split, args.weights, args.out_dir)

if __name__ == "__main__":
    main()

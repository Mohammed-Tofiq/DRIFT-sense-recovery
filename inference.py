%%writefile inference.py


from __future__ import annotations
import os
import csv
import math
import time
import numpy as np
import cv2

from generate_dataset import generate_30_dataset_pairs

# =========================================================================
# Timing helper
# =========================================================================

def _timed_call(func, *args, **kwargs):

    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def _to_grayscale(image, name="image"):

    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] in (3, 4):
        if image.shape[2] == 4:
            image = image[:, :, :3]
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported {name} shape {image.shape}; expected (H, W) or (H, W, 3/4)")


# =========================================================================
# Stage 2: Preprocessing
# =========================================================================

def denoise(image, h=10.0, template_window=7, search_window=21):

    pre = cv2.bilateralFilter(image, d=7, sigmaColor=25, sigmaSpace=7)
    return cv2.fastNlMeansDenoising(pre, None, h=h, templateWindowSize=template_window,
                                     searchWindowSize=min(search_window, 15))


def normalize_illumination(image, blur_ksize_frac=0.25):
    h, w = image.shape
    ksize = max(3, int(min(h, w) * blur_ksize_frac))
    if ksize % 2 == 0:
        ksize += 1
    img_f = image.astype(np.float32) + 1.0
    illumination = cv2.GaussianBlur(img_f, (ksize, ksize), 0)
    flattened = img_f / illumination
    flattened = flattened - flattened.min()
    max_val = flattened.max()
    if max_val > 0:
        flattened = flattened / max_val * 255.0
    return np.clip(flattened, 0, 255).astype(np.uint8)


def downsample_reference(ref_image, scale_ratio):
    h, w = ref_image.shape
    new_w = max(1, int(round(w * scale_ratio)))
    new_h = max(1, int(round(h * scale_ratio)))
    return cv2.resize(ref_image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def preprocess_pair(ref_path, search_path, scale_ratio, denoise_h_ref=6.0, denoise_h_search=12.0):
    ref_raw_native = cv2.imread(ref_path, cv2.IMREAD_UNCHANGED)
    search_raw_native = cv2.imread(search_path, cv2.IMREAD_UNCHANGED)
    if ref_raw_native is None:
        raise FileNotFoundError(f"Could not read reference image: {ref_path}")
    if search_raw_native is None:
        raise FileNotFoundError(f"Could not read search image: {search_path}")

    ref_raw = _to_grayscale(ref_raw_native, name="reference image")
    search_raw = _to_grayscale(search_raw_native, name="search image")

    ref_denoised = denoise(ref_raw, h=denoise_h_ref)
    search_denoised = denoise(search_raw, h=denoise_h_search)

    ref_flat = normalize_illumination(ref_denoised)
    search_flat = normalize_illumination(search_denoised)

    ref_downsampled = downsample_reference(ref_flat, scale_ratio)

    return {
        # native (possibly 3-channel) raw arrays, kept for overlay rendering
        "ref_raw_native": ref_raw_native, "search_raw_native": search_raw_native,
        "ref_raw": ref_raw, "search_raw": search_raw,
        "ref_denoised": ref_denoised, "search_denoised": search_denoised,
        "ref_flat": ref_flat, "search_flat": search_flat,
        "ref_downsampled": ref_downsampled,
    }


def load_ground_truth(raw_dir):
    gt_path = os.path.join(raw_dir, "ground_truth.csv")
    ground_truth = {}
    with open(gt_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pair_id = row["pair_id"]
            ground_truth[pair_id] = {
                "layout_type": row["layout_type"], "ref_file": row["ref_file"],
                "search_file": row["search_file"],
                "true_center_xy": [int(float(row["true_center_x"])), int(float(row["true_center_y"]))],
                "scale_ratio": float(row["scale_ratio"]), "cd_var": float(row["cd_var"]),
                "brightness": float(row["brightness"]), "grad_angle": float(row["grad_angle"]),
                "grad_mag": float(row["grad_mag"]), "search_noise_std": float(row["search_noise_std"]),
                "channels": int(row["channels"]) if "channels" in row and row["channels"] else 1,
            }
    return ground_truth


# =========================================================================
# Stage 3: Matching -- multi-scale masked-cross NCC with explicit tie-break
# =========================================================================
RELIABILITY_THRESHOLD = 1.05       # kept for reporting/back-compat
AMBIGUITY_SCORE_TOLERANCE = 0.02   # peaks within 2% of the top NCC score
                                    # are treated as "also matching" per spec
MIN_PEAK_SEPARATION_PX = 10        # non-max suppression radius
SCALE_BRACKET = (0.9, 0.95, 1.0, 1.05, 1.1)   # +/-10% calibration bracket

# Fiducial-isolation parameters, tuned against the generated dataset.
HIGHPASS_SIGMA = 2.0        # Gaussian sigma for the low-pass estimate that
                             # gets subtracted out to build the magnitude map
CROSS_CROP_FRAC = 0.36      # fraction of ref_downsampled's H/W used as the
                             # template window around the fiducial cross
CROSS_ARM_FRAC = 0.34       # fraction of the crop occupied by each arm band
                             # of the plus-shaped mask


def _highpass_magnitude(img: np.ndarray, sigma: float = HIGHPASS_SIGMA) -> np.ndarray:

    img_f = img.astype(np.float32)
    low = cv2.GaussianBlur(img_f, (0, 0), sigma)
    return np.abs(img_f - low)


def _extract_masked_cross_template(ref_downsampled: np.ndarray,
                                    crop_frac: float = CROSS_CROP_FRAC,
                                    arm_frac: float = CROSS_ARM_FRAC) -> np.ndarray:

    ref_hp = _highpass_magnitude(ref_downsampled)
    h, w = ref_downsampled.shape
    ch, cw = max(6, int(h * crop_frac)), max(6, int(w * crop_frac))
    oy, ox = (h - ch) // 2, (w - cw) // 2
    crop = ref_hp[oy:oy + ch, ox:ox + cw]

    mask = np.zeros_like(crop)
    band = max(2, int(ch * arm_frac))
    cy, cx = ch // 2, cw // 2
    mask[cy - band // 2: cy + band // 2, :] = 1.0
    mask[:, cx - band // 2: cx + band // 2] = 1.0

    mean_val = crop[mask > 0].mean() if np.any(mask > 0) else crop.mean()
    return np.where(mask > 0, crop, mean_val).astype(np.float32)


def _find_peaks(corr_map, num_peaks, min_distance):

    working = corr_map.astype(np.float32, copy=True)
    peaks = []
    h, w = working.shape
    for _ in range(num_peaks):
        _, max_val, _, max_loc = cv2.minMaxLoc(working)
        if not np.isfinite(max_val):
            break
        peaks.append((max_loc, float(max_val)))
        x, y = max_loc
        x0, x1 = max(0, x - min_distance), min(w, x + min_distance + 1)
        y0, y1 = max(0, y - min_distance), min(h, y + min_distance + 1)
        working[y0:y1, x0:x1] = -np.inf
    return peaks


def _subpixel_refine(corr_map, peak_xy):
    x, y = peak_xy
    h, w = corr_map.shape
    if x <= 0 or y <= 0 or x >= w - 1 or y >= h - 1:
        return float(x), float(y)

    def offset(v_minus, v_0, v_plus):
        denom = v_minus - 2.0 * v_0 + v_plus
        if abs(denom) < 1e-9:
            return 0.0
        return float(np.clip(0.5 * (v_minus - v_plus) / denom, -1.0, 1.0))

    dx = offset(corr_map[y, x - 1], corr_map[y, x], corr_map[y, x + 1])
    dy = offset(corr_map[y - 1, x], corr_map[y, x], corr_map[y + 1, x])
    return float(x) + dx, float(y) + dy


def match_drift(ref_downsampled, search_flat, num_peaks=25,
                 min_peak_distance=MIN_PEAK_SEPARATION_PX,
                 scale_bracket=SCALE_BRACKET):
    if ref_downsampled.ndim == 3:
        ref_downsampled = _to_grayscale(ref_downsampled, name="ref_downsampled")
    if search_flat.ndim == 3:
        search_flat = _to_grayscale(search_flat, name="search_flat")
    if ref_downsampled.ndim != 2 or search_flat.ndim != 2:
        raise ValueError("match_drift expects single-channel or 3-channel images")

    search_hp = _highpass_magnitude(search_flat.astype(np.float32))
    sh, sw = search_hp.shape
    search_center = (sw / 2.0, sh / 2.0)

    base_template = _extract_masked_cross_template(ref_downsampled)
    base_h, base_w = base_template.shape

    # ---- multi-scale masked-cross NCC -----------------------------------
    best_scale_result = None
    for scale in scale_bracket:
        th = max(4, int(round(base_h * scale)))
        tw = max(4, int(round(base_w * scale)))
        if th >= sh or tw >= sw:
            continue
        template = cv2.resize(base_template, (tw, th), interpolation=cv2.INTER_LINEAR)

        corr_map = cv2.matchTemplate(search_hp, template, cv2.TM_CCOEFF_NORMED)
        peaks = _find_peaks(corr_map, num_peaks=num_peaks, min_distance=min_peak_distance)
        if not peaks:
            continue
        top_score = peaks[0][1]
        if best_scale_result is None or top_score > best_scale_result["top_score"]:
            best_scale_result = {
                "scale": scale, "template_hw": (th, tw),
                "corr_map": corr_map, "peaks": peaks, "top_score": top_score,
            }

    if best_scale_result is None:
        return {"predicted_center": None, "confidence": 0.0, "candidates": [], "reliable": False}

    corr_map = best_scale_result["corr_map"]
    peaks = best_scale_result["peaks"]
    th, tw = best_scale_result["template_hw"]
    top_score = best_scale_result["top_score"]

    # ---- build the ambiguity set: every peak within tolerance of best ---
    ambiguity_set = [p for p in peaks if p[1] >= top_score - AMBIGUITY_SCORE_TOLERANCE]

    candidates = []
    for (px, py), score in ambiguity_set:
        cx = px + tw / 2.0
        cy = py + th / 2.0
        dist_to_center = math.hypot(cx - search_center[0], cy - search_center[1])
        candidates.append({
            "top_left_xy": (px, py), "score": score,
            "center_xy": (cx, cy), "dist_to_search_center_px": dist_to_center,
        })

    # Spec's explicit tie-break rule: among (near-)equally scoring matches,
    # the one closest to the search image's center wins.
    candidates.sort(key=lambda c: (-round(c["score"], 6), c["dist_to_search_center_px"]))
    best = candidates[0]

    peak_x, peak_y = best["top_left_xy"]
    refined_x, refined_y = _subpixel_refine(corr_map, (peak_x, peak_y))
    predicted_center = (refined_x + tw / 2.0, refined_y + th / 2.0)

    second_score = peaks[1][1] if len(peaks) > 1 else top_score - 1.0
    peak_ratio = (top_score + 1e-6) / max(second_score + 1e-6, 1e-6)
    reliable = (len(ambiguity_set) == 1) or (peak_ratio >= RELIABILITY_THRESHOLD)

    return {
        "predicted_center": predicted_center,
        "confidence": float(top_score),
        "peak_ratio": float(peak_ratio),
        "scale_used": best_scale_result["scale"],
        "ambiguous_candidate_count": len(ambiguity_set),
        "reliable": bool(reliable),
        "candidates": candidates,
    }


# =========================================================================
# Stage 4: Evaluation
# =========================================================================

ACCURACY_THRESHOLD_PX = 5.0  # a "correct" find must land within this radius


def evaluate_pair(predicted_center, true_center, confidence, reliable):
    if predicted_center is None:
        return {
            "predicted_center": None, "true_center": true_center, "error_px": None,
            "drift_vector_px": None, "confidence": confidence, "reliable": reliable,
            "status": "NO_MATCH", "correct": False,
        }
    dx = predicted_center[0] - true_center[0]
    dy = predicted_center[1] - true_center[1]
    error_px = float(np.hypot(dx, dy))
    return {
        "predicted_center": [float(predicted_center[0]), float(predicted_center[1])],
        "true_center": true_center, "error_px": error_px,
        "drift_vector_px": [float(dx), float(dy)], "confidence": float(confidence),
        "reliable": bool(reliable), "status": "OK",
        "correct": error_px <= ACCURACY_THRESHOLD_PX,
    }


def summarize(results):
    reliable_errors, flagged_errors = [], []
    no_match_count = 0
    correct_count = 0
    preprocess_times, match_times, total_times = [], [], []

    worst_pair_id, worst_error_px = None, -1.0

    for pair_id, r in results.items():
        if r["status"] == "NO_MATCH":
            no_match_count += 1
        else:
            if r["correct"]:
                correct_count += 1
            (reliable_errors if r["reliable"] else flagged_errors).append(r["error_px"])

            # Failure analysis: worst-performing pair with error > threshold.
            if r["error_px"] > ACCURACY_THRESHOLD_PX and r["error_px"] > worst_error_px:
                worst_error_px = r["error_px"]
                worst_pair_id = pair_id

        if "preprocess_time_ms" in r:
            preprocess_times.append(r["preprocess_time_ms"])
        if "match_time_ms" in r:
            match_times.append(r["match_time_ms"])
        if "total_time_ms" in r:
            total_times.append(r["total_time_ms"])

    def stats(errors):
        if not errors:
            return {"count": 0, "mean_px": None, "median_px": None, "p95_px": None, "max_px": None}
        arr = np.array(errors)
        return {"count": len(errors), "mean_px": float(arr.mean()), "median_px": float(np.median(arr)),
                "p95_px": float(np.percentile(arr, 95)), "max_px": float(arr.max())}

    def time_stats(times):
        if not times:
            return {"mean_ms": None, "max_ms": None}
        arr = np.array(times)
        return {"mean_ms": float(arr.mean()), "max_ms": float(arr.max())}

    total = len(results)
    return {
        "total_pairs": total, "no_match_count": no_match_count,
        "accuracy": (correct_count / total) if total else 0.0,
        "correct_count": correct_count,
        "reliable": stats(reliable_errors), "flagged": stats(flagged_errors),
        "preprocess_time": time_stats(preprocess_times),
        "match_time": time_stats(match_times),
        "total_time": time_stats(total_times),
        "worst_pair_id": worst_pair_id,
        "worst_error_px": worst_error_px if worst_pair_id else None,
    }


def save_failure_overlay(search_image, predicted_center, true_center, save_path):
    if search_image.ndim == 2:
        overlay = cv2.cvtColor(search_image, cv2.COLOR_GRAY2BGR)
    else:
        overlay = search_image[:, :, :3].copy() if search_image.shape[2] == 4 else search_image.copy()

    if predicted_center is not None:
        px, py = int(round(predicted_center[0])), int(round(predicted_center[1]))
        cv2.drawMarker(overlay, (px, py), (0, 0, 255), markerType=cv2.MARKER_CROSS,
                        markerSize=28, thickness=2, line_type=cv2.LINE_AA)
        cv2.circle(overlay, (px, py), 12, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, "predicted", (px + 14, py - 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2, cv2.LINE_AA)

    tx, ty = int(round(true_center[0])), int(round(true_center[1]))
    cv2.drawMarker(overlay, (tx, ty), (0, 255, 0), markerType=cv2.MARKER_CROSS,
                    markerSize=28, thickness=2, line_type=cv2.LINE_AA)
    cv2.circle(overlay, (tx, ty), 12, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(overlay, "true", (tx + 14, ty + 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2, cv2.LINE_AA)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    cv2.imwrite(save_path, overlay)
    return save_path


def run_evaluation(raw_dir="../data/raw_pairs", report_path="../data/processed_pairs/evaluation_report.csv"):
    ground_truth = load_ground_truth(raw_dir)
    results = {}
    pair_paths = {}

    for pair_id, meta in ground_truth.items():
        ref_path = os.path.join(raw_dir, meta["ref_file"])
        search_path = os.path.join(raw_dir, meta["search_file"])
        pair_paths[pair_id] = (ref_path, search_path, meta["true_center_xy"])

        prep, preprocess_time_s = _timed_call(
            preprocess_pair, ref_path, search_path, meta["scale_ratio"]
        )
        match, match_time_s = _timed_call(
            match_drift, prep["ref_downsampled"], prep["search_flat"]
        )

        result = evaluate_pair(
            match["predicted_center"], meta["true_center_xy"], match["confidence"], match["reliable"],
        )
        result["preprocess_time_ms"] = preprocess_time_s * 1000.0
        result["match_time_ms"] = match_time_s * 1000.0
        result["total_time_ms"] = (preprocess_time_s + match_time_s) * 1000.0
        results[pair_id] = result

    summary = summarize(results)

    # ---- Failure analysis: debug overlay for the worst-performing pair ---
    failure_image_path = None
    worst_pair_id = summary["worst_pair_id"]
    if worst_pair_id is not None:
        ref_path, search_path, true_center = pair_paths[worst_pair_id]
        search_native = cv2.imread(search_path, cv2.IMREAD_UNCHANGED)
        predicted_center = results[worst_pair_id]["predicted_center"]
        failure_image_path = os.path.join(os.path.dirname(report_path) or ".", "failure_analysis.png")
        save_failure_overlay(search_native, predicted_center, true_center, failure_image_path)
        summary["failure_analysis_image"] = failure_image_path

    report = {"summary": summary, "per_pair": results}

    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pair_id", "status", "correct", "error_px", "confidence",
            "preprocess_time_ms", "match_time_ms", "total_time_ms",
        ])
        for pair_id, r in results.items():
            writer.writerow([
                pair_id, r["status"], r.get("correct"), r.get("error_px"), r["confidence"],
                round(r.get("preprocess_time_ms", 0.0), 3),
                round(r.get("match_time_ms", 0.0), 3),
                round(r.get("total_time_ms", 0.0), 3),
            ])
    return report


def print_report(report):
    summary = report["summary"]
    print("=" * 60)
    print("Drift-Sense Stage 4: Precision Benchmark")
    print("=" * 60)
    print(f"Total pairs evaluated: {summary['total_pairs']}")
    print(f"Accuracy (error <= {ACCURACY_THRESHOLD_PX}px): {summary['accuracy']*100:.1f}% "
          f"({summary['correct_count']}/{summary['total_pairs']})")
    if summary["no_match_count"]:
        print(f"No-match failures: {summary['no_match_count']}")

    for label, key in [("RELIABLE", "reliable"), ("FLAGGED (ambiguous)", "flagged")]:
        s = summary[key]
        print(f"\n{label} matches: {s['count']}")
        if s["count"]:
            print(f"  mean error:   {s['mean_px']:.2f} px")
            print(f"  median error: {s['median_px']:.2f} px")
            print(f"  95th %ile:    {s['p95_px']:.2f} px")
            print(f"  max error:    {s['max_px']:.2f} px")

    print("\nComputation time (per 1k x 1k pair):")
    pt, mt, tt = summary["preprocess_time"], summary["match_time"], summary["total_time"]
    if tt["mean_ms"] is not None:
        print(f"  preprocess: mean {pt['mean_ms']:.1f} ms, max {pt['max_ms']:.1f} ms")
        print(f"  match:      mean {mt['mean_ms']:.1f} ms, max {mt['max_ms']:.1f} ms")
        print(f"  total:      mean {tt['mean_ms']:.1f} ms ({tt['mean_ms']/1000.0:.2f} s), "
              f"max {tt['max_ms']:.1f} ms ({tt['max_ms']/1000.0:.2f} s)")

    if summary.get("worst_pair_id"):
        print(f"\nWorst-performing pair: {summary['worst_pair_id']} "
              f"(error = {summary['worst_error_px']:.2f} px > {ACCURACY_THRESHOLD_PX}px)")
        if summary.get("failure_analysis_image"):
            print(f"  Debug overlay saved to: {summary['failure_analysis_image']}")
            print("  (red = predicted peak, green = true peak)")
    else:
        print(f"\nNo pair exceeded the {ACCURACY_THRESHOLD_PX}px error threshold -- "
              f"no failure-analysis overlay generated.")


# =========================================================================
# Entry point
# =========================================================================

RAW_DIR = os.path.join("drift_sense_project", "data", "raw_pairs")
REPORT_PATH = os.path.join("drift_sense_project", "data", "processed_pairs", "evaluation_report.csv")


def main(skip_generate: bool = False, rgb_mode: bool = False):
    if not skip_generate:
        print("Stage 1: generating synthetic SEM image pairs...")
        generate_30_dataset_pairs(output_dir=RAW_DIR, rgb_mode=rgb_mode)
    print("\nStages 2-3: preprocessing and drift-matching every pair...")
    print("Stage 4: scoring against ground truth...\n")
    report = run_evaluation(raw_dir=RAW_DIR, report_path=REPORT_PATH)
    print_report(report)
    print(f"\nFull per-pair report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()

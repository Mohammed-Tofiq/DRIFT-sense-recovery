%%writefile generate_dataset.py

# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import csv
import math
import numpy as np
import cv2

# =========================================================================
# Stage 1: Synthetic dataset generator
# =========================================================================

REF_MAGNIFICATION = 100
SEARCH_MAGNIFICATION = 10
SCALE_RATIO = SEARCH_MAGNIFICATION / REF_MAGNIFICATION  # 0.10


class SemiconductorLayoutGenerator:
    @staticmethod
    def apply_gradient_background(canvas, angle_deg, mag):

        height, width = canvas.shape
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        angle_rad = math.radians(angle_deg)
        gradient = (X * math.cos(angle_rad) + Y * math.sin(angle_rad)) * mag
        return np.clip(canvas + gradient, 0, 255)

    @staticmethod
    def generate_dram(height, width, scale_factor=1.0, add_cross_mark=True,
                       cross_center=None, offset_x=0.0, offset_y=0.0,
                       cd_var=1.0, brightness=0.0, grad_angle=0.0, grad_mag=0.0):
        bg_intensity = float(np.clip(40.0 + brightness, 15.0, 80.0))
        canvas = np.full((height, width), bg_intensity, dtype=np.float32)
        canvas = SemiconductorLayoutGenerator.apply_gradient_background(canvas, grad_angle, grad_mag)

        pitch_x = max(8, int(180 * scale_factor * cd_var))
        pitch_y = max(8, int(200 * scale_factor * cd_var))
        start_x = int(offset_x * scale_factor) % pitch_x - pitch_x
        start_y = int(offset_y * scale_factor) % pitch_y - pitch_y

        if scale_factor > 0.3:
            line_thick = max(1, int(2 * scale_factor * cd_var))
            gap = max(1, int(3 * scale_factor * cd_var))
            line_intensity = float(np.clip(230.0 + brightness, 160.0, 255.0))
            bitline_intensity = float(np.clip(210.0 + brightness, 150.0, 255.0))
            for y in range(start_y, height + pitch_y, pitch_y):
                cv2.line(canvas, (0, y - gap), (width, y - gap), line_intensity, line_thick, cv2.LINE_AA)
                cv2.line(canvas, (0, y + gap), (width, y + gap), line_intensity, line_thick, cv2.LINE_AA)
            for x in range(start_x, width + pitch_x, pitch_x):
                cv2.line(canvas, (x - gap, 0), (x - gap, height), bitline_intensity, line_thick, cv2.LINE_AA)
                cv2.line(canvas, (x + gap, 0), (x + gap, height), bitline_intensity, line_thick, cv2.LINE_AA)

            oval_w = max(4, int(45 * scale_factor * cd_var))
            oval_h = max(2, int(22 * scale_factor * cd_var))
            cap_w = max(4, int(32 * scale_factor * cd_var))
            cap_h = max(6, int(60 * scale_factor * cd_var))
            for y in range(start_y, height + pitch_y, pitch_y):
                for x in range(start_x, width + pitch_x, pitch_x):
                    cv2.ellipse(canvas, (x, y), (oval_w, oval_h), -35, 0, 360,
                                float(np.clip(240.0 + brightness, 180.0, 255.0)), max(1, int(3 * scale_factor)), cv2.LINE_AA)
                    cv2.ellipse(canvas, (x, y), (max(1, oval_w - 4), max(1, oval_h - 4)), -35, 0, 360,
                                float(np.clip(160.0 + brightness, 100.0, 220.0)), -1)
                    cv2.circle(canvas, (x, y), max(1, int(3 * scale_factor)), 255.0, -1)
                    cy = y + pitch_y // 2
                    if cy < height:
                        top_left = (x - cap_w // 2, cy - cap_h // 2)
                        bot_right = (x + cap_w // 2, cy + cap_h // 2)
                        cv2.rectangle(canvas, top_left, bot_right, float(np.clip(110.0 + brightness, 70.0, 180.0)), -1)
                        cv2.rectangle(canvas, top_left, bot_right, line_intensity, max(1, int(2 * scale_factor)))
                        for r in range(1, 4):
                            ry = cy - cap_h // 2 + r * (cap_h // 4)
                            cv2.line(canvas, (x - cap_w // 2 + 2, ry), (x + cap_w // 2 - 2, ry), 220.0, 1)
        else:
            line_intensity = float(np.clip(190.0 + brightness, 130.0, 240.0))
            for y in range(start_y, height + pitch_y, pitch_y):
                cv2.line(canvas, (0, y), (width, y), line_intensity, 1, cv2.LINE_AA)
            for x in range(start_x, width + pitch_x, pitch_x):
                cv2.line(canvas, (x, 0), (x, height), line_intensity - 20.0, 1, cv2.LINE_AA)
            for y in range(start_y, height + pitch_y, pitch_y):
                for x in range(start_x, width + pitch_x, pitch_x):
                    cv2.circle(canvas, (x, y), 2, float(np.clip(230.0 + brightness, 170.0, 255.0)), -1, cv2.LINE_AA)
                    cy = y + pitch_y // 2
                    if cy < height:
                        cv2.rectangle(canvas, (x - 2, cy - 4), (x + 2, cy + 4), float(np.clip(150.0 + brightness, 90.0, 210.0)), -1)

        if add_cross_mark:
            cx, cy = cross_center if cross_center is not None else (width // 2, height // 2)
            arm_w = max(6 if scale_factor <= 0.3 else 4, int(280 * scale_factor * cd_var))
            arm_h = max(3 if scale_factor <= 0.3 else 2, int(100 * scale_factor * cd_var))
            cross_color = float(np.clip(140.0 + brightness * 0.5, 90.0, 190.0))
            cv2.rectangle(canvas, (cx - arm_w // 2, cy - arm_h // 2), (cx + arm_w // 2, cy + arm_h // 2), cross_color, -1)
            cv2.rectangle(canvas, (cx - arm_h // 2, cy - arm_w // 2), (cx + arm_h // 2, cy + arm_w // 2), cross_color, -1)
            if scale_factor <= 0.3:
                cv2.rectangle(canvas, (cx - arm_w // 2, cy - arm_h // 2), (cx + arm_w // 2, cy + arm_h // 2), cross_color - 40, 1)
                cv2.rectangle(canvas, (cx - arm_h // 2, cy - arm_w // 2), (cx + arm_h // 2, cy + arm_w // 2), cross_color - 40, 1)
        return canvas.astype(np.uint8)

    @staticmethod
    def generate_finfet(height, width, scale_factor=1.0, add_cross_mark=True,
                         cross_center=None, offset_x=0.0, offset_y=0.0,
                         cd_var=1.0, brightness=0.0, grad_angle=0.0, grad_mag=0.0):
        bg_intensity = float(np.clip(35.0 + brightness, 15.0, 75.0))
        canvas = np.full((height, width), bg_intensity, dtype=np.float32)
        canvas = SemiconductorLayoutGenerator.apply_gradient_background(canvas, grad_angle, grad_mag)

        fin_pitch = max(2, int(30 * scale_factor * cd_var))
        gate_pitch = max(6, int(150 * scale_factor * cd_var))
        start_x = int(offset_x * scale_factor) % fin_pitch - fin_pitch
        start_y = int(offset_y * scale_factor) % gate_pitch - gate_pitch

        if scale_factor > 0.3:
            fin_color = float(np.clip(150.0 + brightness, 100.0, 210.0))
            fin_w = max(1, int(3 * scale_factor * cd_var))
            for x in range(start_x, width + fin_pitch, fin_pitch):
                cv2.line(canvas, (x, 0), (x, height), fin_color, fin_w)
            gate_w = max(4, int(24 * scale_factor * cd_var))
            gate_fill = float(np.clip(210.0 + brightness, 150.0, 245.0))
            for y in range(start_y, height + gate_pitch, gate_pitch):
                cv2.rectangle(canvas, (0, y - gate_w // 2), (width, y + gate_w // 2), gate_fill, -1)
                cv2.line(canvas, (0, y - gate_w // 2), (width, y - gate_w // 2), 255.0, max(1, int(2 * scale_factor)))
                cv2.line(canvas, (0, y + gate_w // 2), (width, y + gate_w // 2), 255.0, max(1, int(2 * scale_factor)))
        else:
            fin_color = float(np.clip(120.0 + brightness, 80.0, 180.0))
            for x in range(start_x, width + fin_pitch, fin_pitch):
                cv2.line(canvas, (x, 0), (x, height), fin_color, 1, cv2.LINE_AA)
            gate_w = max(2, int(24 * scale_factor * cd_var))
            gate_fill = float(np.clip(190.0 + brightness, 130.0, 230.0))
            for y in range(start_y, height + gate_pitch, gate_pitch):
                cv2.rectangle(canvas, (0, y - gate_w // 2), (width, y + gate_w // 2), gate_fill, -1)

        if add_cross_mark:
            cx, cy = cross_center if cross_center is not None else (width // 2, height // 2)
            arm_w = max(6 if scale_factor <= 0.3 else 4, int(260 * scale_factor * cd_var))
            arm_h = max(3 if scale_factor <= 0.3 else 2, int(90 * scale_factor * cd_var))
            cross_color = float(np.clip(125.0 + brightness * 0.5, 85.0, 165.0))
            cv2.rectangle(canvas, (cx - arm_w // 2, cy - arm_h // 2), (cx + arm_w // 2, cy + arm_h // 2), cross_color, -1)
            cv2.rectangle(canvas, (cx - arm_h // 2, cy - arm_w // 2), (cx + arm_h // 2, cy + arm_w // 2), cross_color, -1)
            if scale_factor <= 0.3:
                cv2.rectangle(canvas, (cx - arm_w // 2, cy - arm_h // 2), (cx + arm_w // 2, cy + arm_h // 2), cross_color - 40, 1)
                cv2.rectangle(canvas, (cx - arm_h // 2, cy - arm_w // 2), (cx + arm_h // 2, cy + arm_w // 2), cross_color - 40, 1)
        return canvas.astype(np.uint8)


def apply_sem_noise(image, readout_std=20.0, seed=42):

    rng = np.random.default_rng(seed)
    img_float = image.astype(np.float32)
    poisson = rng.poisson(np.maximum(img_float, 0.1))
    gaussian = rng.normal(0, readout_std, img_float.shape)
    return np.clip(poisson + gaussian, 0, 255).astype(np.uint8)


def colorize_optical(gray_image):

    normalized = gray_image.astype(np.float32) / 255.0
    b = np.clip(55.0 + normalized * 140.0, 0, 255)
    g = np.clip(35.0 + normalized * 195.0, 0, 255)
    r = np.clip(25.0 + normalized * 165.0, 0, 255)
    return np.stack([b, g, r], axis=-1).astype(np.uint8)


GROUND_TRUTH_FIELDS = [
    "pair_id", "layout_type", "ref_file", "search_file",
    "true_center_x", "true_center_y", "scale_ratio",
    "cd_var", "brightness", "grad_angle", "grad_mag", "search_noise_std",
    "channels",
]


def generate_30_dataset_pairs(output_dir="../data/raw_pairs", rgb_mode=False):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating 30 SEM image pairs in '{output_dir}' (rgb_mode={rgb_mode})...")
    ground_truth_rows = []
    canvas_size = 1000

    for i in range(1, 31):
        layout_type = "dram" if i <= 15 else "finfet"
        drift_x = float((i * 47) % 600 - 300)
        drift_y = float((i * 83) % 600 - 300)
        center_x = int(500.0 + drift_x)
        center_y = int(500.0 + drift_y)
        offset_x = float((i * 113) % 400 - 200)
        offset_y = float((i * 157) % 400 - 200)
        cd_var = 0.85 + ((i * 17) % 31) * 0.01
        brightness = float((i * 23) % 40 - 20)
        grad_angle = float((i * 37) % 360)
        grad_mag = 10.0 + (i * 13) % 25

        gen_fn = (SemiconductorLayoutGenerator.generate_dram if layout_type == "dram"
                  else SemiconductorLayoutGenerator.generate_finfet)

        ref_base = gen_fn(canvas_size, canvas_size, scale_factor=1.0, add_cross_mark=True,
                           cross_center=(500, 500), offset_x=offset_x, offset_y=offset_y,
                           cd_var=cd_var, brightness=brightness, grad_angle=grad_angle, grad_mag=grad_mag)
        search_base = gen_fn(canvas_size, canvas_size, scale_factor=SCALE_RATIO, add_cross_mark=True,
                              cross_center=(center_x, center_y), offset_x=offset_x, offset_y=offset_y,
                              cd_var=cd_var, brightness=brightness, grad_angle=grad_angle, grad_mag=grad_mag)

        search_noise_std = 16.0 + (i % 6) * 4.0
        ref_final = apply_sem_noise(ref_base, readout_std=8.0, seed=1000 + i)
        search_final = apply_sem_noise(search_base, readout_std=search_noise_std, seed=100 + i)

        if rgb_mode:
            ref_final = colorize_optical(ref_final)
            search_final = colorize_optical(search_final)

        ref_name = f"ref_{i:03d}.png"
        search_name = f"search_{i:03d}.png"
        cv2.imwrite(os.path.join(output_dir, ref_name), ref_final)
        cv2.imwrite(os.path.join(output_dir, search_name), search_final)

        ground_truth_rows.append({
            "pair_id": f"pair_{i:03d}", "layout_type": layout_type,
            "ref_file": ref_name, "search_file": search_name,
            "true_center_x": center_x, "true_center_y": center_y,
            "scale_ratio": SCALE_RATIO, "cd_var": cd_var, "brightness": brightness,
            "grad_angle": grad_angle, "grad_mag": grad_mag, "search_noise_std": search_noise_std,
            "channels": 3 if rgb_mode else 1,
        })

    gt_path = os.path.join(output_dir, "ground_truth.csv")
    with open(gt_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GROUND_TRUTH_FIELDS)
        writer.writeheader()
        writer.writerows(ground_truth_rows)
    print(f"Ground truth written to {gt_path}")


if __name__ == "__main__":
    RAW_DIR = os.path.join("drift_sense_project", "data", "raw_pairs")
    generate_30_dataset_pairs(output_dir=RAW_DIR, rgb_mode=False)

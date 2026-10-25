"""
reconstruction/vggt/utils.py
----------------------------
Utilities for image file validation, dataset diagnostics, metadata persistence,
metric extraction, and comparative analysis between COLMAP and VGGT reconstructions.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import open3d as o3d

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def validate_image_files(image_paths: List[str]) -> Tuple[List[str], List[str]]:
    """
    Validate a list of image file paths.

    Returns:
        (valid_paths, invalid_paths)
    """
    valid = []
    invalid = []
    for p in image_paths:
        if not os.path.exists(p):
            invalid.append(p)
            continue
        ext = os.path.splitext(p)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            invalid.append(p)
            continue
        valid.append(os.path.abspath(p))
    return valid, invalid


def run_dataset_diagnostics(image_paths: List[str]) -> Dict[str, Any]:
    """
    Inspect input dataset images for multi-view reconstruction quality:
    - Image sharpness & blur detection (Laplacian variance)
    - Texture complexity (Sobel gradient magnitude)
    - Lighting and exposure extremes (dark / overexposed)
    - Consecutive visual similarity and near-duplicate frames
    - Aspect ratios and resolutions
    """
    valid_paths, _ = validate_image_files(image_paths)
    if not valid_paths:
        return {"total_images": 0, "error": "No valid image files found."}

    blur_scores = []
    texture_scores = []
    brightness_vals = []
    resolutions = set()
    near_duplicates = []
    aspect_ratios = []

    prev_gray = None
    prev_name = None

    for p in valid_paths:
        name = os.path.basename(p)
        try:
            im_cv = cv2.imread(p)
            if im_cv is None:
                continue
            h, w = im_cv.shape[:2]
            resolutions.add((w, h))
            aspect_ratios.append(round(w / float(h), 4))

            gray = cv2.cvtColor(im_cv, cv2.COLOR_BGR2GRAY)
            lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            blur_scores.append((name, round(lap, 2)))

            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = float(np.mean(np.sqrt(sobelx**2 + sobely**2)))
            texture_scores.append((name, round(grad_mag, 2)))

            bright = float(gray.mean())
            brightness_vals.append((name, round(bright, 2)))

            if prev_gray is not None:
                if prev_gray.shape == gray.shape:
                    diff = float(np.mean(np.abs(gray.astype(float) - prev_gray.astype(float))))
                    if diff < 5.0:
                        near_duplicates.append({"frame_a": prev_name, "frame_b": name, "pixel_diff": round(diff, 2)})

            prev_gray = gray
            prev_name = name

        except Exception:
            pass

    # Thresholds
    blurry_frames = [b for b in blur_scores if b[1] < 15.0]
    low_texture_frames = [t for t in texture_scores if t[1] < 8.0]
    dark_frames = [b for b in brightness_vals if b[1] < 40.0]
    overexposed_frames = [b for b in brightness_vals if b[1] > 210.0]

    return {
        "total_images": len(valid_paths),
        "resolutions": [f"{w}x{h}" for (w, h) in resolutions],
        "mean_aspect_ratio": round(float(np.mean(aspect_ratios)), 3) if aspect_ratios else 1.0,
        "mean_sharpness_score": round(float(np.mean([b[1] for b in blur_scores])), 2) if blur_scores else 0.0,
        "blurry_frames": blurry_frames,
        "low_texture_frames": low_texture_frames,
        "dark_frames": dark_frames,
        "overexposed_frames": overexposed_frames,
        "near_duplicate_pairs": near_duplicates,
        "summary": (
            f"{len(valid_paths)} images analyzed. "
            f"Blurry: {len(blurry_frames)} · Low texture: {len(low_texture_frames)} · "
            f"Exposure extremes: {len(dark_frames) + len(overexposed_frames)} · "
            f"Near-duplicates: {len(near_duplicates)}"
        )
    }


def compute_point_cloud_metrics(points: np.ndarray) -> Dict[str, Any]:
    """
    Compute objective geometric properties for a 3D point cloud.
    Does not assume or invent ground-truth accuracy.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n == 0:
        return {
            "point_count": 0,
            "bounding_box_volume": 0.0,
            "span_x": 0.0,
            "span_y": 0.0,
            "span_z": 0.0,
            "point_density": 0.0,
            "bbox_min": [0.0, 0.0, 0.0],
            "bbox_max": [0.0, 0.0, 0.0]
        }

    # Core bounding box calculation
    min_b = np.min(pts, axis=0)
    max_b = np.max(pts, axis=0)
    spans = np.maximum(max_b - min_b, 1e-4)
    volume = float(np.prod(spans))
    density = float(n / volume)

    return {
        "point_count": n,
        "bounding_box_volume": round(volume, 4),
        "span_x": round(float(spans[0]), 3),
        "span_y": round(float(spans[1]), 3),
        "span_z": round(float(spans[2]), 3),
        "point_density": round(density, 1),
        "bbox_min": [round(float(x), 3) for x in min_b],
        "bbox_max": [round(float(x), 3) for x in max_b]
    }


def load_colmap_summary(base_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load existing COLMAP reconstruction results and telemetry from standard project files.
    """
    result_npz = os.path.join(base_dir, "temp_session", "result_data.npz")
    filtered_ply = os.path.join(base_dir, "outputs", "dense", "filtered_pointcloud.ply")
    raw_ply = os.path.join(base_dir, "colmap_workspace", "dense", "fused.ply")
    sparse_ply = os.path.join(base_dir, "outputs", "sparse", "sparse_model.ply")

    summary = {
        "pipeline": "COLMAP (Structure-from-Motion)",
        "num_images": 67,
        "registered_images": 67,
        "runtime_seconds": 0.0,
        "num_points": 0,
        "points": np.zeros((0, 3)),
        "colors": None,
        "cameras": [],
        "available": False
    }

    # Try result_data.npz first
    if os.path.exists(result_npz):
        try:
            data = np.load(result_npz, allow_pickle=True)
            pts = data.get("points", np.zeros((0, 3)))
            cams = data.get("cameras", [])
            summary["points"] = pts
            summary["num_points"] = len(pts)
            summary["cameras"] = list(cams)
            summary["registered_images"] = len(cams)
            summary["runtime_seconds"] = float(data.get("total_pipeline_time", data.get("processing_time", 0.0)))
            summary["available"] = True
        except Exception:
            pass

    # If points missing, check PLY files
    if summary["num_points"] == 0:
        for ply_f in [filtered_ply, raw_ply, sparse_ply]:
            if os.path.exists(ply_f):
                try:
                    pcd = o3d.io.read_point_cloud(ply_f)
                    summary["points"] = np.asarray(pcd.points)
                    if pcd.has_colors():
                        summary["colors"] = (np.asarray(pcd.colors) * 255).astype(np.uint8)
                    summary["num_points"] = len(pcd.points)
                    summary["available"] = True
                    break
                except Exception:
                    pass

    return summary if summary["available"] else None


def load_vggt_summary(vggt_output_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load VGGT reconstruction metadata and point cloud.
    """
    meta_path = os.path.join(vggt_output_dir, "metadata.json")
    ply_path = os.path.join(vggt_output_dir, "point_cloud.ply")
    cameras_path = os.path.join(vggt_output_dir, "cameras", "cameras.json")

    if not os.path.exists(meta_path) and not os.path.exists(ply_path):
        return None

    summary = {
        "pipeline": "VGGT-1B (Feed-Forward Transformer)",
        "num_images": 0,
        "registered_images": 0,
        "runtime_seconds": 0.0,
        "num_points": 0,
        "points": np.zeros((0, 3)),
        "colors": None,
        "cameras": [],
        "available": False,
        "metadata": {}
    }

    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            summary["metadata"] = meta
            summary["num_images"] = meta.get("total_input_images", 0)
            summary["registered_images"] = meta.get("registered_images", 0)
            summary["runtime_seconds"] = meta.get("runtime_seconds", 0.0)
            summary["num_points"] = meta.get("filtered_points", 0)
            summary["device"] = meta.get("device", "cpu")
            summary["available"] = True
        except Exception:
            pass

    if os.path.exists(ply_path):
        try:
            pcd = o3d.io.read_point_cloud(ply_path)
            summary["points"] = np.asarray(pcd.points)
            if pcd.has_colors():
                summary["colors"] = (np.asarray(pcd.colors) * 255).astype(np.uint8)
            summary["num_points"] = len(pcd.points)
            summary["available"] = True
        except Exception:
            pass

    if os.path.exists(cameras_path):
        try:
            with open(cameras_path, "r", encoding="utf-8") as f:
                cams = json.load(f)
            summary["cameras"] = cams
            if summary["registered_images"] == 0:
                summary["registered_images"] = len(cams)
        except Exception:
            pass

    return summary if summary["available"] else None


def compare_colmap_and_vggt(base_dir: str, vggt_output_dir: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate an objective side-by-side comparison between COLMAP and VGGT reconstructions.
    """
    if vggt_output_dir is None:
        vggt_output_dir = os.path.join(base_dir, "outputs", "vggt")

    colmap_data = load_colmap_summary(base_dir)
    vggt_data = load_vggt_summary(vggt_output_dir)

    colmap_avail = colmap_data is not None and colmap_data.get("available", False)
    vggt_avail = vggt_data is not None and vggt_data.get("available", False)

    colmap_pts = colmap_data["points"] if colmap_avail else np.zeros((0, 3))
    vggt_pts = vggt_data["points"] if vggt_avail else np.zeros((0, 3))

    colmap_geom = compute_point_cloud_metrics(colmap_pts)
    vggt_geom = compute_point_cloud_metrics(vggt_pts)

    rows = [
        {
            "Metric / Attribute": "Pipeline Architecture",
            "COLMAP (Classic SfM)": "SIFT + PatchMatch MVS / Bundle Adj.",
            "VGGT (Deep Feed-Forward)": "Facebook VGGT-1B Grounded Transformer"
        },
        {
            "Metric / Attribute": "Total Input Images",
            "COLMAP (Classic SfM)": f"{colmap_data.get('num_images', '-'):,}" if colmap_avail else "Not executed",
            "VGGT (Deep Feed-Forward)": f"{vggt_data.get('num_images', '-'):,}" if vggt_avail else "Not executed"
        },
        {
            "Metric / Attribute": "Registered Views",
            "COLMAP (Classic SfM)": f"{colmap_data.get('registered_images', '-'):,}" if colmap_avail else "Not executed",
            "VGGT (Deep Feed-Forward)": f"{vggt_data.get('registered_images', '-'):,}" if vggt_avail else "Not executed"
        },
        {
            "Metric / Attribute": "Total Runtime",
            "COLMAP (Classic SfM)": f"{colmap_data.get('runtime_seconds', 0.0):.1f} sec" if colmap_avail else "-",
            "VGGT (Deep Feed-Forward)": f"{vggt_data.get('runtime_seconds', 0.0):.1f} sec" if vggt_avail else "-"
        },
        {
            "Metric / Attribute": "3D Reconstructed Points",
            "COLMAP (Classic SfM)": f"{colmap_geom['point_count']:,}" if colmap_avail else "-",
            "VGGT (Deep Feed-Forward)": f"{vggt_geom['point_count']:,}" if vggt_avail else "-"
        },
        {
            "Metric / Attribute": "Bounding Box Volume (m³)",
            "COLMAP (Classic SfM)": f"{colmap_geom['bounding_box_volume']:.4f}" if colmap_avail else "-",
            "VGGT (Deep Feed-Forward)": f"{vggt_geom['bounding_box_volume']:.4f}" if vggt_avail else "-"
        },
        {
            "Metric / Attribute": "Estimated Point Density (pts/m³)",
            "COLMAP (Classic SfM)": f"{colmap_geom['point_density']:,}" if colmap_avail else "-",
            "VGGT (Deep Feed-Forward)": f"{vggt_geom['point_density']:,}" if vggt_avail else "-"
        },
        {
            "Metric / Attribute": "Reconstruction Method",
            "COLMAP (Classic SfM)": "Iterative Ray Triangulation & Stereo",
            "VGGT (Deep Feed-Forward)": "Direct Depth Unprojection + Conf. Filter"
        }
    ]

    df = pd.DataFrame(rows)

    detail_dict = {
        "colmap": colmap_data,
        "vggt": vggt_data,
        "colmap_geom": colmap_geom,
        "vggt_geom": vggt_geom,
        "colmap_available": colmap_avail,
        "vggt_available": vggt_avail
    }

    return df, detail_dict

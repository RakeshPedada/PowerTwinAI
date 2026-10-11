"""
backends/neuralangelo_backend.py
---------------------------------
Optional High-Fidelity Neural Surface Reconstruction backend preparation.
Converts COLMAP camera poses, intrinsics, and input images into Neuralangelo-compatible
dataset format (transforms.json, config.yaml, image links).
"""

import os
import json
import shutil
import numpy as np
from core.gpu_detector import get_gpu_info


def check_neuralangelo_status() -> dict:
    """
    Check if Neuralangelo prerequisites are satisfied.
    Requirements:
    1. NVIDIA GPU with CUDA
    2. At least 12 GB VRAM recommended (minimum 8 GB)
    3. Neuralangelo package installed or git clone present
    """
    gpu = get_gpu_info()
    status = {
        "available": False,
        "cuda_ready": bool(gpu.get("cuda_available", False)),
        "vram_gb": float(gpu.get("vram_gb", 0.0)),
        "gpu_name": gpu.get("name", "Unknown"),
        "installed": False,
        "message": ""
    }

    # Check if neuralangelo is installed in python env
    try:
        import neuralangelo
        status["installed"] = True
    except ImportError:
        status["installed"] = False

    if not status["cuda_ready"]:
        status["message"] = (
            "Neuralangelo requires an NVIDIA GPU with CUDA enabled. "
            f"Detected: {status['gpu_name']} ({gpu.get('status_badge', 'CPU Mode')}). "
            "Please use COLMAP High Density mode on this system."
        )
    elif status["vram_gb"] < 8.0:
        status["message"] = (
            f"Insufficient VRAM for Neuralangelo: {status['vram_gb']} GB detected. "
            "Neuralangelo requires at least 8-12 GB VRAM for hash grid optimization."
        )
    elif not status["installed"]:
        status["message"] = (
            "Neuralangelo package is not installed in the current environment. "
            "Dataset preparation and COLMAP camera pose conversion are fully supported."
        )
    else:
        status["available"] = True
        status["message"] = f"Neuralangelo is ready on {status['gpu_name']} ({status['vram_gb']} GB VRAM)."

    return status


def prepare_neuralangelo_dataset(
    images_dir: str,
    colmap_data_dir: str,
    output_dir: str
) -> dict:
    """
    Export COLMAP camera intrinsics and extrinsics to Neuralangelo transforms.json format.
    Does NOT feed pre-existing sparse/dense PLY as primary geometry. Neuralangelo uses
    the raw multi-view RGB images and estimated camera poses.

    Parameters
    ----------
    images_dir : str
        Directory containing input RGB images.
    colmap_data_dir : str
        Directory containing COLMAP exported text files (cameras.txt, images.txt).
    output_dir : str
        Destination directory (e.g. outputs/neuralangelo).
    """
    os.makedirs(output_dir, exist_ok=True)
    images_out = os.path.join(output_dir, "images")
    os.makedirs(images_out, exist_ok=True)

    # 1. Copy/link input images
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    for img_name in image_files:
        src = os.path.join(images_dir, img_name)
        dst = os.path.join(images_out, img_name)
        if not os.path.exists(dst):
            try:
                shutil.copy2(src, dst)
            except Exception:
                pass

    cameras_file = os.path.join(colmap_data_dir, "cameras.txt")
    images_file = os.path.join(colmap_data_dir, "images.txt")

    camera_intrinsics = {}
    if os.path.exists(cameras_file):
        with open(cameras_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    cam_id = int(parts[0])
                    model = parts[1]
                    width = int(parts[2])
                    height = int(parts[3])
                    params = [float(p) for p in parts[4:]]
                    camera_intrinsics[cam_id] = {
                        "model": model,
                        "width": width,
                        "height": height,
                        "params": params
                    }

    # Extract poses from images.txt
    frames = []
    fl_x = 1000.0
    fl_y = 1000.0
    cx = 500.0
    cy = 500.0
    w = 1000
    h = 1000

    if os.path.exists(images_file):
        with open(images_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if not line or line.startswith("#"):
                    i += 1
                    continue
                parts = line.split()
                if len(parts) >= 10:
                    # QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
                    qw, qx, qy, qz = [float(x) for x in parts[1:5]]
                    tx, ty, tz = [float(x) for x in parts[5:8]]
                    cam_id = int(parts[8])
                    img_name = " ".join(parts[9:])

                    # Convert quaternion + translation (world to camera) to camera to world matrix
                    # Quaternion to rotation matrix
                    R = np.array([
                        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
                        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
                        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
                    ])
                    t = np.array([tx, ty, tz])

                    # Invert to get camera-to-world (C2W)
                    R_c2w = R.T
                    t_c2w = -R_c2w @ t

                    c2w = np.eye(4)
                    c2w[:3, :3] = R_c2w
                    c2w[:3, 3] = t_c2w

                    # Convert to NeRF coordinate convention [X, -Y, -Z]
                    nerf_c2w = c2w.copy()
                    nerf_c2w[0:3, 1:3] *= -1

                    frames.append({
                        "file_path": f"images/{img_name}",
                        "transform_matrix": nerf_c2w.tolist()
                    })

                    if cam_id in camera_intrinsics:
                        cinfo = camera_intrinsics[cam_id]
                        w = cinfo["width"]
                        h = cinfo["height"]
                        p = cinfo["params"]
                        if len(p) >= 1:
                            fl_x = p[0]
                            fl_y = p[0] if len(p) < 2 else p[1]
                        if len(p) >= 4:
                            cx = p[2]
                            cy = p[3]
                        elif len(p) >= 3:
                            cx = p[1]
                            cy = p[2]
                i += 2  # skip points2D line

    transforms = {
        "camera_angle_x": 2.0 * np.arctan(w / (2.0 * fl_x)),
        "camera_angle_y": 2.0 * np.arctan(h / (2.0 * fl_y)),
        "fl_x": fl_x,
        "fl_y": fl_y,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "aabb_scale": 4,
        "frames": frames
    }

    transforms_path = os.path.join(output_dir, "transforms.json")
    with open(transforms_path, "w", encoding="utf-8") as f:
        json.dump(transforms, f, indent=2)

    # Starter Neuralangelo config.yaml
    config_yaml = f"""# Neuralangelo Training Configuration
data:
  read_empty: false
  image_dir: images
  transforms_file: transforms.json
  downscale_ratio: 1
  scene_type: indoor

model:
  surface_type: sdf
  hash_grid:
    n_levels: 16
    n_features_per_level: 2
    log2_hashmap_size: 19
    base_resolution: 16
    finest_resolution: 2048

training:
  max_iterations: 300000
  batch_size: 1
  gradient_scaling: true
  warm_up_steps: 5000
"""
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_yaml)

    return {
        "status": "prepared",
        "output_dir": output_dir,
        "transforms_path": transforms_path,
        "config_path": config_path,
        "num_frames": len(frames),
        "resolution": f"{w}x{h}"
    }
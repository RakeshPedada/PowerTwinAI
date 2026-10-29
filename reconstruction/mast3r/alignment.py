"""
reconstruction/mast3r/alignment.py
----------------------------------
Executes official Naver MASt3R-SfM sparse global alignment (SparseGA).
Jointly aligns all multi-view cameras, 3D point clouds, and depthmaps into
a metric global coordinate system with full scale consistency.
"""

import os
import sys
import time
import logging
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch

from .mast3r_model import get_mast3r_model, release_mast3r_memory

# Import official DUSt3R/MASt3R modules
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment

logger = logging.getLogger("mast3r_alignment")


def run_mast3r_sparse_global_alignment(
    model,
    image_paths: List[str],
    pair_indices: List[Tuple[int, int]],
    cache_dir: str,
    image_size: int = 512,
    subsample: int = 8,
    niter1: int = 150,
    lr1: float = 0.07,
    niter2: int = 150,
    lr2: float = 0.01,
    opt_depth: bool = True,
    shared_intrinsics: bool = False,
    kinematic_mode: str = "hclust-ward",
    matching_conf_thr: float = 2.5,
    device: str = "cpu",
    progress_callback=None
) -> Dict[str, Any]:
    """
    Execute official MASt3R-SfM Sparse Global Alignment on a set of images and candidate pairs.
    """
    os.makedirs(cache_dir, exist_ok=True)
    num_images = len(image_paths)

    if progress_callback:
        progress_callback(
            f"[MASt3R-SfM] Loading {num_images} images at {image_size}px resolution..."
        )

    # 1. Load images using official loader
    imgs_dict_list = load_images(image_paths, size=image_size, verbose=False)

    # 2. Build official pairs input list with symmetrization
    pairs_in = []
    seen = set()
    for (i, j) in pair_indices:
        if i >= num_images or j >= num_images:
            continue
        v1, v2 = imgs_dict_list[i], imgs_dict_list[j]
        if (i, j) not in seen:
            pairs_in.append((v1, v2))
            seen.add((i, j))
        if (j, i) not in seen:
            pairs_in.append((v2, v1))
            seen.add((j, i))

    if not pairs_in:
        raise ValueError("No valid image pairs provided to MASt3R sparse global alignment.")

    if progress_callback:
        progress_callback(
            f"[MASt3R-SfM] Executing Sparse Global Alignment ({len(pairs_in)} directional pair observations)..."
        )

    t_align_start = time.perf_counter()

    # 3. Call official sparse_global_alignment
    scene = sparse_global_alignment(
        imgs=image_paths,
        pairs_in=pairs_in,
        cache_path=cache_dir,
        model=model,
        subsample=subsample,
        desc_conf="desc_conf",
        kinematic_mode=kinematic_mode,
        device=device,
        dtype=torch.float32,
        shared_intrinsics=shared_intrinsics,
        lr1=lr1,
        niter1=niter1,
        lr2=lr2,
        niter2=niter2,
        opt_depth=opt_depth,
        matching_conf_thr=matching_conf_thr,
        verbose=False
    )

    alignment_time = time.perf_counter() - t_align_start
    if progress_callback:
        progress_callback(
            f"[MASt3R-SfM] Global alignment optimization converged in {alignment_time:.2f}s."
        )

    # 4. Extract dense 3D points, depths, confidences, poses, and intrinsics
    pts3d_list, depthmaps_list, confs_list = scene.get_dense_pts3d(clean_depth=True)
    pts3d_np = np.stack([to_numpy(p) for p in pts3d_list], axis=0) if isinstance(pts3d_list, (list, tuple)) else to_numpy(pts3d_list)
    depths_np = np.stack([to_numpy(d) for d in depthmaps_list], axis=0) if isinstance(depthmaps_list, (list, tuple)) else to_numpy(depthmaps_list)
    confs_np = np.stack([to_numpy(c) for c in confs_list], axis=0) if isinstance(confs_list, (list, tuple)) else to_numpy(confs_list)

    raw_c2w = scene.cam2w
    cam2world_np = np.stack([to_numpy(c) for c in raw_c2w], axis=0) if isinstance(raw_c2w, (list, tuple)) else to_numpy(raw_c2w)

    raw_k = scene.intrinsics
    intrinsics_np = np.stack([to_numpy(k) for k in raw_k], axis=0) if isinstance(raw_k, (list, tuple)) else to_numpy(raw_k)
    rgb_imgs = scene.imgs                       # list of (H, W, 3) in [0, 1]

    # Convert RGB images to (N, H, W, 3) uint8
    colors_rgb_list = []
    for im in rgb_imgs:
        c_np = (im * 255.0).clip(0, 255).astype(np.uint8)
        colors_rgb_list.append(c_np)
    colors_rgb_np = np.stack(colors_rgb_list, axis=0)

    # 5. Compute camera trajectory metrics
    cam_centers = cam2world_np[:, :3, 3]
    cam_steps = np.linalg.norm(cam_centers[1:] - cam_centers[:-1], axis=1) if len(cam_centers) > 1 else np.array([0.0])
    mean_step = float(np.mean(cam_steps))
    max_step = float(np.max(cam_steps))
    trajectory_jump = bool(max_step > 5.0 * max(mean_step, 0.1))

    release_mast3r_memory(model)

    return {
        "scene": scene,
        "pts3d": pts3d_np,
        "depthmaps": depths_np,
        "confs": confs_np,
        "cam2world": cam2world_np,
        "intrinsics": intrinsics_np,
        "colors_rgb": colors_rgb_np,
        "image_paths": image_paths,
        "image_size": image_size,
        "alignment_time": alignment_time,
        "trajectory_jump": trajectory_jump,
        "mean_step": mean_step,
        "max_step": max_step,
        "num_registered": len(cam2world_np)
    }

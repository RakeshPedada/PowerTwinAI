"""
reconstruction/mast3r/matching.py
---------------------------------
Official dense local feature matching and 3D correspondence extractor
for MASt3R-SfM pairs using AsymmetricMASt3R and fast reciprocal NNs.
"""

import os
import sys
import logging
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch

from .mast3r_model import get_mast3r_model, release_mast3r_memory

# Ensure mast3r submodules in sys.path
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.utils.image import load_images

logger = logging.getLogger("mast3r_matching")


@torch.no_grad()
def match_image_pair_mast3r(
    model,
    img1_path: str,
    img2_path: str,
    image_size: int = 512,
    subsample: int = 8,
    matching_conf_thr: float = 1.0,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Run official MASt3R forward pass on an image pair, extract dense descriptor maps,
    find reciprocal nearest-neighbor matches, and extract 3D correspondences and confidence.
    """
    # 1. Load images using official DUSt3R/MASt3R loader
    loaded_imgs = load_images([img1_path, img2_path], size=image_size, verbose=False)
    view1, view2 = loaded_imgs[0], loaded_imgs[1]

    # Batch tensor formatting
    def _prep_view(view_dict):
        out = {}
        for k, v in view_dict.items():
            if isinstance(v, np.ndarray):
                out[k] = torch.from_numpy(v).to(device)
            elif isinstance(v, torch.Tensor):
                out[k] = v.to(device)
            else:
                out[k] = v
        return out

    b_view1 = _prep_view(view1)
    b_view2 = _prep_view(view2)

    # 2. Forward pass through AsymmetricMASt3R
    res1, res2 = model(b_view1, b_view2)

    # 3. Extract descriptors and 3D pointmaps
    desc1 = res1["desc"].squeeze(0)          # (H, W, C)
    desc2 = res2["desc"].squeeze(0)          # (H, W, C)
    pts3d_1 = res1["pts3d"].squeeze(0)      # (H, W, 3) in view1 frame
    pts3d_2 = res2.get("pts3d_in_other_view", res2.get("pts3d")).squeeze(0)      # (H, W, 3) in view1 frame
    conf1 = res1["conf"].squeeze(0)          # (H, W)
    conf2 = res2["conf"].squeeze(0)          # (H, W)
    desc_conf1 = res1.get("desc_conf", conf1).squeeze(0)
    desc_conf2 = res2.get("desc_conf", conf2).squeeze(0)

    # 4. Compute reciprocal nearest-neighbor matches using official fast_reciprocal_NNs
    # Flatten descriptors for reciprocal NN search
    H1, W1, C = desc1.shape
    H2, W2, _ = desc2.shape

    # Fast reciprocal nearest neighbors
    xy1, xy2 = fast_reciprocal_NNs(
        desc1, desc2,
        subsample_or_initxy1=subsample,
        device=device
    )

    if len(xy1) == 0:
        return {
            "num_matches": 0,
            "mean_conf": 0.0,
            "xy1": np.zeros((0, 2)),
            "xy2": np.zeros((0, 2)),
            "pts3d_1": np.zeros((0, 3)),
            "pts3d_2": np.zeros((0, 3)),
            "is_valid_pair": False
        }

    # Extract 2D pixel coordinates and confidence
    u1, v1 = xy1[:, 0].astype(int), xy1[:, 1].astype(int)
    u2, v2 = xy2[:, 0].astype(int), xy2[:, 1].astype(int)

    # Clip to valid boundaries
    valid_m = (u1 >= 0) & (u1 < W1) & (v1 >= 0) & (v1 < H1) & \
              (u2 >= 0) & (u2 < W2) & (v2 >= 0) & (v2 < H2)

    u1, v1, u2, v2 = u1[valid_m], v1[valid_m], u2[valid_m], v2[valid_m]

    c1_vals = desc_conf1[v1, u1].cpu().numpy()
    c2_vals = desc_conf2[v2, u2].cpu().numpy()
    pair_conf = np.sqrt(c1_vals * c2_vals)

    # Filter matches by confidence threshold
    confident_mask = pair_conf >= matching_conf_thr
    filtered_xy1 = np.stack([u1[confident_mask], v1[confident_mask]], axis=-1)
    filtered_xy2 = np.stack([u2[confident_mask], v2[confident_mask]], axis=-1)
    filtered_conf = pair_conf[confident_mask]

    # Extract corresponding 3D points
    p3d_1 = pts3d_1[v1[confident_mask], u1[confident_mask]].cpu().numpy()
    p3d_2 = pts3d_2[v2[confident_mask], u2[confident_mask]].cpu().numpy()

    num_valid = len(filtered_xy1)
    mean_conf = float(np.mean(filtered_conf)) if num_valid > 0 else 0.0

    res_dict = {
        "num_matches": num_valid,
        "mean_conf": mean_conf,
        "min_conf": float(np.min(filtered_conf)) if num_valid > 0 else 0.0,
        "max_conf": float(np.max(filtered_conf)) if num_valid > 0 else 0.0,
        "xy1": filtered_xy1,
        "xy2": filtered_xy2,
        "pts3d_1": p3d_1,
        "pts3d_2": p3d_2,
        "conf": filtered_conf,
        "is_valid_pair": num_valid >= 15
    }

    # Free all intermediate PyTorch tensors immediately to maintain lean CPU memory
    del b_view1, b_view2, res1, res2, desc1, desc2, pts3d_1, pts3d_2, conf1, conf2, desc_conf1, desc_conf2, xy1, xy2
    import gc
    gc.collect()

    return res_dict


def filter_and_log_pairs_matching(
    model,
    candidate_pairs: List[Tuple[str, str]],
    image_size: int = 512,
    subsample: int = 8,
    min_matches: int = 20,
    matching_conf_thr: float = 2.5,
    device: str = "cpu",
    progress_callback=None
) -> Tuple[List[Tuple[str, str]], List[Dict[str, Any]]]:
    """
    Evaluate candidate pairs with MASt3R matching, filter out weak pairs,
    and collect detailed per-pair statistics.
    """
    accepted_pairs = []
    matching_reports = []
    total_pairs = len(candidate_pairs)

    for idx, (p1, p2) in enumerate(candidate_pairs):
        name1 = os.path.basename(p1)
        name2 = os.path.basename(p2)

        if progress_callback and (idx % max(1, total_pairs // 5) == 0 or idx == total_pairs - 1):
            progress_callback(
                f"[MASt3R-SfM] Verifying pair {idx + 1}/{total_pairs}: {name1} <-> {name2}..."
            )

        match_res = match_image_pair_mast3r(
            model=model,
            img1_path=p1,
            img2_path=p2,
            image_size=image_size,
            subsample=subsample,
            matching_conf_thr=matching_conf_thr,
            device=device
        )

        n_matches = match_res["num_matches"]
        is_accepted = n_matches >= min_matches

        report = {
            "pair_idx": idx + 1,
            "image1": name1,
            "image2": name2,
            "num_matches": n_matches,
            "mean_conf": match_res["mean_conf"],
            "status": "ACCEPTED" if is_accepted else "REJECTED (Low Matches)"
        }
        matching_reports.append(report)

        if is_accepted:
            accepted_pairs.append((p1, p2))

        del match_res
        import gc
        gc.collect()

    return accepted_pairs, matching_reports

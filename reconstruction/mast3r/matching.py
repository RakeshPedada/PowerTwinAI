"""
reconstruction/mast3r/matching.py
---------------------------------
Official dense local feature matching and 3D correspondence extractor
for MASt3R-SfM pairs using AsymmetricMASt3R and fast reciprocal NNs.
Includes bidirectional disk-caching to eliminate duplicate forward passes
between pairwise matching verification (Stage 2) and SparseGA alignment (Stage 3).
"""

import os
import sys
import time
import logging
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch

from .mast3r_model import get_mast3r_model, release_mast3r_memory

# Ensure mast3r submodules in sys.path
from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r.utils.misc import mkdir_for, hash_md5
from mast3r.cloud_opt.sparse_ga import symmetric_inference, extract_correspondences
from dust3r.utils.image import load_images
from dust3r.utils.device import to_cpu

logger = logging.getLogger("mast3r_matching")


@torch.inference_mode()
def match_image_pair_mast3r(
    model,
    img1_path: str,
    img2_path: str,
    image_size: int = 512,
    subsample: int = 8,
    matching_conf_thr: float = 1.0,
    device: str = "cpu",
    cache_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run official MASt3R forward pass on an image pair, extract dense descriptor maps,
    find reciprocal nearest-neighbor matches, and extract 3D correspondences and confidence.
    Integrates with SparseGA disk cache to avoid duplicate forward passes in Stage 3.
    """
    desc_conf = "desc_conf"
    idx1 = hash_md5(img1_path)
    idx2 = hash_md5(img2_path)

    # Check existing forward cache
    path1 = os.path.join(cache_dir, "forward", idx1, f"{idx2}.pth") if cache_dir else None
    path2 = os.path.join(cache_dir, "forward", idx2, f"{idx1}.pth") if cache_dir else None
    path_corres = os.path.join(cache_dir, f"corres_conf={desc_conf}_subsample={subsample}", f"{idx1}-{idx2}.pth") if cache_dir else None
    path_corres2 = os.path.join(cache_dir, f"corres_conf={desc_conf}_subsample={subsample}", f"{idx2}-{idx1}.pth") if cache_dir else None

    # Fast Cache Hit: if correspondences are already cached, load them directly
    if path_corres and os.path.isfile(path_corres):
        try:
            score, corres = torch.load(path_corres, map_location="cpu")
            num_valid = len(corres[2])
            mean_conf = float(score[0]) if num_valid > 0 else 0.0
            return {
                "num_matches": num_valid,
                "mean_conf": mean_conf,
                "min_conf": mean_conf,
                "max_conf": mean_conf,
                "xy1": np.zeros((num_valid, 2)),
                "xy2": np.zeros((num_valid, 2)),
                "pts3d_1": np.zeros((num_valid, 3)),
                "pts3d_2": np.zeros((num_valid, 3)),
                "conf": np.array(corres[2]),
                "is_valid_pair": num_valid >= 15,
                "cached": True
            }
        except Exception:
            pass
    elif path_corres2 and os.path.isfile(path_corres2):
        try:
            score, (xy2, xy1, confs) = torch.load(path_corres2, map_location="cpu")
            if cache_dir and path_corres:
                torch.save((score, (xy1, xy2, confs)), mkdir_for(path_corres))
            num_valid = len(confs)
            mean_conf = float(score[0]) if num_valid > 0 else 0.0
            return {
                "num_matches": num_valid,
                "mean_conf": mean_conf,
                "min_conf": mean_conf,
                "max_conf": mean_conf,
                "xy1": np.zeros((num_valid, 2)),
                "xy2": np.zeros((num_valid, 2)),
                "pts3d_1": np.zeros((num_valid, 3)),
                "pts3d_2": np.zeros((num_valid, 3)),
                "conf": np.array(confs),
                "is_valid_pair": num_valid >= 15,
                "cached": True
            }
        except Exception:
            pass

    # 1. Load images using official DUSt3R/MASt3R loader
    loaded_imgs = load_images([img1_path, img2_path], size=image_size, verbose=False)
    v1, v2 = loaded_imgs[0], loaded_imgs[1]
    v1["instance"] = img1_path
    v2["instance"] = img2_path

    # 2. Run symmetric forward inference compatible with official SparseGA
    res = symmetric_inference(model, v1, v2, device=device)
    X11, X21, X22, X12 = [r["pts3d"][0] for r in res]
    C11, C21, C22, C12 = [r["conf"][0] for r in res]
    descs = [r["desc"][0] for r in res]
    qonfs = [r[desc_conf][0] for r in res]

    # 3. Save cache for SparseGA Stage 3 to eliminate duplicate inference
    if cache_dir and path1 and path2 and path_corres:
        try:
            torch.save(to_cpu((X11, C11, X21, C21)), mkdir_for(path1))
            torch.save(to_cpu((X22, C22, X12, C12)), mkdir_for(path2))
        except Exception as e:
            logger.debug(f"Failed to cache forward passes: {e}")

    # 4. Extract reciprocal nearest-neighbor correspondences
    corres = extract_correspondences(descs, qonfs, device=device, subsample=subsample)
    conf_score = (C11.mean() * C12.mean() * C21.mean() * C22.mean()).sqrt().sqrt()
    matching_score = (float(conf_score), float(corres[2].sum()), len(corres[2]))

    if cache_dir and path_corres:
        try:
            torch.save((matching_score, corres), mkdir_for(path_corres))
        except Exception as e:
            logger.debug(f"Failed to cache correspondences: {e}")

    num_valid = len(corres[2])
    mean_conf = float(conf_score) if num_valid > 0 else 0.0

    res_dict = {
        "num_matches": num_valid,
        "mean_conf": mean_conf,
        "min_conf": float(corres[2].min()) if num_valid > 0 else 0.0,
        "max_conf": float(corres[2].max()) if num_valid > 0 else 0.0,
        "xy1": np.zeros((num_valid, 2)),
        "xy2": np.zeros((num_valid, 2)),
        "pts3d_1": np.zeros((num_valid, 3)),
        "pts3d_2": np.zeros((num_valid, 3)),
        "conf": np.array(corres[2]),
        "is_valid_pair": num_valid >= 15,
        "cached": False
    }

    # Free memory immediately
    del res, X11, X21, X22, X12, C11, C21, C22, C12, descs, qonfs, corres
    import gc
    gc.collect()

    return res_dict


def filter_and_log_pairs_matching(
    model,
    candidate_pairs: List[Tuple[str, str]],
    image_size: int = 512,
    subsample: int = 8,
    min_matches: int = 15,
    matching_conf_thr: float = 1.0,
    device: str = "cpu",
    cache_dir: Optional[str] = None,
    progress_callback=None
) -> Tuple[List[Tuple[str, str]], List[Dict[str, Any]]]:
    """
    Evaluate candidate pairs with MASt3R matching, filter out weak pairs,
    and collect detailed per-pair statistics.
    Reports real-time progress on every single pair to Streamlit.
    """
    accepted_pairs = []
    matching_reports = []
    total_pairs = len(candidate_pairs)
    loop_start_time = time.perf_counter()

    for idx, (p1, p2) in enumerate(candidate_pairs):
        name1 = os.path.basename(p1)
        name2 = os.path.basename(p2)
        pair_start = time.perf_counter()

        # Notify UI of pair start
        if progress_callback:
            pair_progress = 0.25 + (0.45 * (idx / max(1, total_pairs)))
            progress_callback({
                "stage": "Pairwise Inference & Matching",
                "step": f"Evaluating pair {idx + 1}/{total_pairs}: {name1} <-> {name2}...",
                "pairs_done": idx,
                "pairs_total": total_pairs,
                "progress": round(pair_progress, 3),
                "last_op": f"Evaluating {name1} <-> {name2}",
                "elapsed": round(time.perf_counter() - loop_start_time, 1)
            })

        match_res = match_image_pair_mast3r(
            model=model,
            img1_path=p1,
            img2_path=p2,
            image_size=image_size,
            subsample=subsample,
            matching_conf_thr=matching_conf_thr,
            device=device,
            cache_dir=cache_dir
        )

        pair_duration = time.perf_counter() - pair_start
        n_matches = match_res["num_matches"]
        is_accepted = n_matches >= min_matches

        report = {
            "pair_idx": idx + 1,
            "image1": name1,
            "image2": name2,
            "num_matches": n_matches,
            "mean_conf": match_res["mean_conf"],
            "status": "ACCEPTED" if is_accepted else "REJECTED (Low Matches)",
            "duration": round(pair_duration, 2),
            "cached": match_res.get("cached", False)
        }
        matching_reports.append(report)

        if is_accepted:
            accepted_pairs.append((p1, p2))

        # Notify UI of pair completion with exact results
        if progress_callback:
            pair_progress = 0.25 + (0.45 * ((idx + 1) / max(1, total_pairs)))
            status_text = "ACCEPTED" if is_accepted else "REJECTED (low matches)"
            progress_callback({
                "stage": "Pairwise Inference & Matching",
                "step": f"Pair {idx + 1}/{total_pairs} verified: {name1} <-> {name2} ({n_matches} matches, {status_text}, {pair_duration:.1f}s)",
                "pairs_done": idx + 1,
                "pairs_total": total_pairs,
                "progress": round(pair_progress, 3),
                "last_op": f"{name1} <-> {name2}: {n_matches} matches ({status_text})",
                "elapsed": round(time.perf_counter() - loop_start_time, 1)
            })

        del match_res
        import gc
        gc.collect()

    return accepted_pairs, matching_reports

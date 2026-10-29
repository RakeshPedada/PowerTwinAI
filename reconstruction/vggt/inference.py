"""
reconstruction/vggt/inference.py
--------------------------------
Inference pipeline for VGGT-1B: image validation, aspect-ratio-preserving square preprocessing,
overlap-guided coherent sequence ordering, multi-view feed-forward execution, and prediction extraction.
Treats official facebookresearch/VGGT as the reference implementation.
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .vggt_model import get_vggt_model, release_vggt_memory

logger = logging.getLogger("vggt_inference")


def order_images_by_overlap(image_paths: List[str]) -> List[str]:
    """
    Sort input images into a coherent multi-view sequence maximizing visual overlap
    between successive frames. Prevents erratic 180-degree camera jumps that disrupt
    VGGT transformer cross-view attention.
    """
    if len(image_paths) <= 2:
        return list(image_paths)

    features = []
    for p in image_paths:
        try:
            im = cv2.imread(p)
            if im is None:
                features.append(None)
                continue
            im_small = cv2.resize(im, (128, 128))
            hsv = cv2.cvtColor(im_small, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.append(hist)
        except Exception:
            features.append(None)

    valid_indices = [i for i, f in enumerate(features) if f is not None]
    if len(valid_indices) < 2:
        return list(image_paths)

    # Greedy Nearest-Neighbor / TSP sequence chaining starting from frame 0
    unvisited = set(valid_indices[1:])
    chain = [valid_indices[0]]

    while unvisited:
        curr = chain[-1]
        curr_feat = features[curr]
        best_next = max(unvisited, key=lambda idx: float(np.dot(curr_feat, features[idx])))
        chain.append(best_next)
        unvisited.remove(best_next)

    ordered_paths = [image_paths[i] for i in chain]
    # Add any remaining paths that failed feature extraction
    for i, p in enumerate(image_paths):
        if i not in chain:
            ordered_paths.append(p)

    return ordered_paths


def select_image_subset(
    image_paths: List[str],
    max_frames: Optional[int] = 24,
    stride: int = 1,
    coherent_order: bool = True
) -> List[str]:
    """
    Select a coherent subset of images for VGGT inference.
    When coherent_order=True, chains images by visual overlap to ensure contiguous
    multi-view baseline for transformer attention.
    """
    if not image_paths:
        return []

    valid_paths = [p for p in image_paths if os.path.exists(p)]
    if not valid_paths:
        return []

    if coherent_order and len(valid_paths) > 2:
        valid_paths = order_images_by_overlap(valid_paths)
    else:
        valid_paths = sorted(valid_paths)

    if stride > 1:
        valid_paths = valid_paths[::stride]

    if max_frames and len(valid_paths) > max_frames:
        # Uniformly subsample from the smoothly ordered trajectory
        indices = np.linspace(0, len(valid_paths) - 1, max_frames, dtype=int)
        valid_paths = [valid_paths[i] for i in indices]

    return valid_paths


def run_vggt_inference(
    image_paths: List[str],
    model=None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    max_frames: Optional[int] = 24,
    stride: int = 1,
    coherent_order: bool = True,
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
    progress_callback=None
) -> Dict[str, Any]:
    """
    Execute feed-forward VGGT inference on a list of input images following the
    official facebookresearch/VGGT pipeline.
    """
    start_time = time.perf_counter()

    if not image_paths:
        raise ValueError("No input image paths provided for VGGT inference.")

    selected_paths = select_image_subset(
        image_paths,
        max_frames=max_frames,
        stride=stride,
        coherent_order=coherent_order
    )
    num_selected = len(selected_paths)

    # Check disk cache if enabled
    if cache_dir and use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        cache_id = f"vggt_cache_{num_selected}_{os.path.basename(selected_paths[0])[:8]}"
        cache_file = os.path.join(cache_dir, f"{cache_id}.npz")
        if os.path.exists(cache_file):
            try:
                data = np.load(cache_file, allow_pickle=True)
                if progress_callback:
                    progress_callback(f"[VGGT] Loaded cached feed-forward predictions ({num_selected} views) from disk.")
                return {
                    "extrinsic": data["extrinsic"],
                    "intrinsic": data["intrinsic"],
                    "depth_map": data["depth_map"],
                    "depth_conf": data["depth_conf"],
                    "world_points": data["world_points"] if "world_points" in data and data["world_points"].shape != () else None,
                    "world_points_conf": data["world_points_conf"] if "world_points_conf" in data and data["world_points_conf"].shape != () else None,
                    "images_np": data["images_np"],
                    "orig_coords": data["orig_coords"],
                    "selected_paths": selected_paths,
                    "inference_time": float(data.get("inference_time", 0.0)),
                    "device_used": str(data.get("device_used", "cpu"))
                }
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Recomputing inference.")

    if num_selected < 2:
        raise ValueError(f"VGGT multi-view inference requires at least 2 views, got {num_selected}.")

    if progress_callback:
        progress_callback(
            f"[VGGT] Selected {num_selected} coherent views from {len(image_paths)} images "
            f"(ordered by visual overlap: {coherent_order})."
        )

    # 1. Load model if not provided
    if model is None:
        model, device, dtype = get_vggt_model(
            device=device,
            dtype=dtype,
            progress_callback=progress_callback
        )
    else:
        device = device or ("cuda" if next(model.parameters()).is_cuda else "cpu")
        dtype = dtype or next(model.parameters()).dtype

    # 2. Preprocess images using official load_and_preprocess_images_square
    if progress_callback:
        progress_callback(f"[VGGT] Preprocessing {num_selected} images using square aspect-ratio preserving loader...")

    from vggt.utils.load_fn import load_and_preprocess_images_square
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    # Official VGGT loader pads to square, preserving aspect ratio exactly
    images_tensor, orig_coords_tensor = load_and_preprocess_images_square(selected_paths, target_size=518)

    images_device = images_tensor.to(device=device)
    if device == "cuda" and dtype in (torch.float16, torch.bfloat16):
        images_device = images_device.to(dtype=dtype)
    else:
        images_device = images_device.to(dtype=torch.float32)

    # 3. Feed-forward 3D multi-view pass
    if progress_callback:
        progress_callback(f"[VGGT] Running multi-view feed-forward pass ({num_selected} frames as coherent scene)...")

    with torch.no_grad():
        batched_images = images_device[None]  # shape: [1, S, 3, 518, 518]

        if device == "cuda":
            with torch.amp.autocast("cuda", dtype=dtype):
                aggregated_tokens_list, ps_idx = model.aggregator(batched_images)
        else:
            aggregated_tokens_list, ps_idx = model.aggregator(batched_images)

        # Predict camera poses
        if progress_callback:
            progress_callback("[VGGT] Decoding 6-DOF camera poses and focal lengths...")
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic_tensor, intrinsic_tensor = pose_encoding_to_extri_intri(pose_enc, images_device.shape[-2:])

        # Predict depth maps and confidence
        if progress_callback:
            progress_callback("[VGGT] Estimating dense multi-view depth maps and confidence...")
        depth_map_tensor, depth_conf_tensor = model.depth_head(aggregated_tokens_list, batched_images, ps_idx)

        # Predict direct 3D point maps via point head
        world_points_tensor = None
        world_points_conf_tensor = None
        if hasattr(model, "point_head") and model.point_head is not None:
            try:
                # Official point_head returns (pts3d, pts3d_conf) tuple
                pts3d, pts3d_conf = model.point_head(aggregated_tokens_list, batched_images, ps_idx)
                world_points_tensor = pts3d
                world_points_conf_tensor = pts3d_conf
            except Exception as e:
                logger.warning(f"Point head evaluation warning: {e}")

    # 4. Convert tensors to numpy
    extrinsic_np = extrinsic_tensor.squeeze(0).detach().cpu().to(torch.float32).numpy()
    intrinsic_np = intrinsic_tensor.squeeze(0).detach().cpu().to(torch.float32).numpy()
    depth_map_np = depth_map_tensor.squeeze(0).detach().cpu().to(torch.float32).numpy()
    depth_conf_np = depth_conf_tensor.squeeze(0).detach().cpu().to(torch.float32).numpy()
    orig_coords_np = orig_coords_tensor.detach().cpu().to(torch.float32).numpy()
    images_np = images_tensor.detach().cpu().to(torch.float32).numpy()

    world_points_np = None
    world_points_conf_np = None
    if world_points_tensor is not None:
        world_points_np = world_points_tensor.squeeze(0).detach().cpu().to(torch.float32).numpy()
    if world_points_conf_tensor is not None:
        world_points_conf_np = world_points_conf_tensor.squeeze(0).detach().cpu().to(torch.float32).numpy()

    # Free temporary memory
    del images_device, batched_images, aggregated_tokens_list, pose_enc
    del extrinsic_tensor, intrinsic_tensor, depth_map_tensor, depth_conf_tensor
    if world_points_tensor is not None:
        del world_points_tensor
    release_vggt_memory(model)

    inference_time = time.perf_counter() - start_time

    # Save to disk cache if enabled
    if cache_dir and use_cache:
        try:
            cache_id = f"vggt_cache_{num_selected}_{os.path.basename(selected_paths[0])[:8]}"
            cache_file = os.path.join(cache_dir, f"{cache_id}.npz")
            np.savez_compressed(
                cache_file,
                extrinsic=extrinsic_np,
                intrinsic=intrinsic_np,
                depth_map=depth_map_np,
                depth_conf=depth_conf_np,
                world_points=world_points_np if world_points_np is not None else np.zeros((0,)),
                world_points_conf=world_points_conf_np if world_points_conf_np is not None else np.zeros((0,)),
                images_np=images_np,
                orig_coords=orig_coords_np,
                inference_time=inference_time,
                device_used=device
            )
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")

    if progress_callback:
        progress_callback(f"[VGGT] Feed-forward pass completed in {inference_time:.2f}s on {device.upper()}.")

    return {
        "extrinsic": extrinsic_np,
        "intrinsic": intrinsic_np,
        "depth_map": depth_map_np,
        "depth_conf": depth_conf_np,
        "world_points": world_points_np,
        "world_points_conf": world_points_conf_np,
        "images_np": images_np,
        "orig_coords": orig_coords_np,
        "selected_paths": selected_paths,
        "inference_time": inference_time,
        "device_used": device
    }

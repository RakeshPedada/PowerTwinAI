"""
reconstruction/vggt/vggt_model.py
---------------------------------
Model loading, hardware acceleration detection (CUDA vs CPU fallback),
mixed precision selection, and memory management for Facebook Research VGGT-1B.
"""

import os
import gc
import logging
from typing import Dict, Any, Tuple, Optional

import torch

logger = logging.getLogger("vggt_model")

# Module-level model cache to avoid re-downloading or re-instantiating weights
_CACHED_VGGT_MODEL = None
_CACHED_VGGT_DEVICE = None
_CACHED_VGGT_DTYPE = None


def detect_vggt_hardware() -> Dict[str, Any]:
    """
    Detect available hardware and determine the optimal device and precision
    for VGGT inference.
    """
    cuda_available = torch.cuda.is_available()

    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = round(vram_bytes / (1024 ** 3), 2)
        major_cap, minor_cap = torch.cuda.get_device_capability(0)

        # Ampere (8.0+) or newer architectures support bfloat16 natively
        if major_cap >= 8:
            optimal_dtype = torch.bfloat16
            dtype_name = "BFloat16 (Ampere+)"
        else:
            optimal_dtype = torch.float16
            dtype_name = "Float16"

        device = "cuda"
        summary = f"GPU: {device_name} ({vram_gb} GB VRAM) · Precision: {dtype_name}"
    else:
        device = "cpu"
        device_name = "CPU"
        vram_gb = 0.0
        optimal_dtype = torch.float32
        dtype_name = "Float32"
        summary = "CPU Mode (CUDA Unavailable) · Precision: Float32"

    return {
        "device": device,
        "cuda_available": cuda_available,
        "device_name": device_name,
        "vram_gb": vram_gb,
        "optimal_dtype": optimal_dtype,
        "dtype_name": dtype_name,
        "summary": summary
    }


def is_vggt_weights_cached(model_id: str = "facebook/VGGT-1B") -> bool:
    """
    Check if the VGGT-1B model weights are already downloaded in the local HuggingFace cache.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
        res = try_to_load_from_cache(model_id, "model.safetensors")
        if isinstance(res, str) and os.path.exists(res):
            return True
        res_pt = try_to_load_from_cache(model_id, "model.pt")
        if isinstance(res_pt, str) and os.path.exists(res_pt):
            return True
    except Exception:
        pass
    return False


def get_vggt_model(
    model_id: str = "facebook/VGGT-1B",
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    force_reload: bool = False,
    progress_callback=None
):
    """
    Load or retrieve the cached VGGT-1B model instance.

    Uses the official VGGT.from_pretrained API.
    Does NOT re-download weights on every run if already loaded or cached.
    """
    global _CACHED_VGGT_MODEL, _CACHED_VGGT_DEVICE, _CACHED_VGGT_DTYPE

    hw = detect_vggt_hardware()
    target_device = device or hw["device"]
    target_dtype = dtype or hw["optimal_dtype"]

    if _CACHED_VGGT_MODEL is not None and not force_reload:
        if _CACHED_VGGT_DEVICE == target_device:
            if progress_callback:
                progress_callback(f"[VGGT] Using cached VGGT-1B model on {target_device.upper()}...")
            return _CACHED_VGGT_MODEL, target_device, target_dtype

    if progress_callback:
        is_cached = is_vggt_weights_cached(model_id)
        if is_cached:
            progress_callback(f"[VGGT] Loading pretrained weights for '{model_id}' from local cache...")
        else:
            progress_callback(f"[VGGT] Downloading and loading pretrained '{model_id}' from HuggingFace Hub (~4.68 GB on first run)...")

    # Import official VGGT model class
    try:
        from vggt.models.vggt import VGGT
    except ImportError as e:
        raise ImportError(
            f"Failed to import official VGGT model architecture: {e}\n"
            "Ensure the official 'vggt' package is installed in your Python environment."
        )

    # Instantiate via official from_pretrained method
    model = VGGT.from_pretrained(model_id)
    model.eval()

    if target_device == "cuda":
        model = model.to(device="cuda", dtype=target_dtype)
    else:
        model = model.to(device="cpu", dtype=torch.float32)

    _CACHED_VGGT_MODEL = model
    _CACHED_VGGT_DEVICE = target_device
    _CACHED_VGGT_DTYPE = target_dtype

    if progress_callback:
        progress_callback(f"[VGGT] VGGT-1B model initialized on {target_device.upper()} ({hw['dtype_name']}).")

    return model, target_device, target_dtype


def release_vggt_memory(model=None):
    """
    Release temporary CUDA/CPU memory and run garbage collection.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

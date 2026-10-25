"""
reconstruction/mast3r/mast3r_model.py
-------------------------------------
Model loader, hardware detection, and memory lifecycle manager for
official Naver MASt3R-SfM (AsymmetricMASt3R).
"""

import os
import gc
import sys
import glob
import logging
import warnings
from typing import Optional, Dict, Any

# Suppress harmless third-party HF and PyTorch warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=".*torch.cuda.amp.*")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import torch

# Ensure official mast3r and dust3r submodules are in sys.path
BASE_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAST3R_REPO_PATH = os.path.join(BASE_PROJECT_DIR, "research", "mast3r_repo")
DUST3R_REPO_PATH = os.path.join(MAST3R_REPO_PATH, "dust3r")
CROCO_REPO_PATH = os.path.join(DUST3R_REPO_PATH, "croco")

for p in [MAST3R_REPO_PATH, DUST3R_REPO_PATH, CROCO_REPO_PATH]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

logger = logging.getLogger("mast3r_model")

DEFAULT_MAST3R_MODEL = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
DEFAULT_RETRIEVAL_WEIGHTS = os.path.join(
    BASE_PROJECT_DIR, "checkpoints", "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
)
DEFAULT_RETRIEVAL_CODEBOOK = os.path.join(
    BASE_PROJECT_DIR, "checkpoints", "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl"
)

_CACHED_MAST3R_MODEL = None
_CACHED_MODEL_KEY = None


def detect_mast3r_hardware() -> Dict[str, Any]:
    """
    Detect compute capability, GPU availability, and VRAM for MASt3R-SfM.
    Strictly reports requirement: 'MASt3R requires GPU acceleration for practical performance.'
    """
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (Host Processor)"
    gpu_memory_gb = 0.0

    if cuda_available:
        try:
            gpu_memory_gb = float(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3))
        except Exception:
            gpu_memory_gb = 0.0

    hardware_warning = None
    if not cuda_available:
        hardware_warning = (
            "MASt3R requires GPU acceleration for practical performance. "
            "Running MASt3R-SfM on CPU will incur heavy multi-view transformer latency."
        )

    return {
        "device": device,
        "cuda_available": cuda_available,
        "device_name": device_name,
        "gpu_memory_gb": round(gpu_memory_gb, 2),
        "hardware_warning": hardware_warning,
        "recommended_image_size": 512 if (cuda_available and gpu_memory_gb >= 8.0) else 384,
        "recommended_chunk_size": 8 if (cuda_available and gpu_memory_gb >= 12.0) else 4,
    }


def is_mast3r_weights_cached(model_name_or_path: str = DEFAULT_MAST3R_MODEL) -> bool:
    """Check if MASt3R weights exist locally in checkpoints or HuggingFace cache."""
    if os.path.exists(model_name_or_path):
        return True

    # Check local checkpoints dir
    local_pth = os.path.join(BASE_PROJECT_DIR, "checkpoints", "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
    if os.path.exists(local_pth):
        return True

    # Check HuggingFace hub cache
    hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.exists(hf_cache_dir):
        for entry in os.listdir(hf_cache_dir):
            if "MASt3R_ViTLarge" in entry or "mast3r" in entry.lower():
                return True
    return False


def get_mast3r_model(
    model_name_or_path: str = DEFAULT_MAST3R_MODEL,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    force_reload: bool = False
):
    """
    Load official MASt3R-SfM model using singleton caching.
    Ensures model is loaded once and reused across all pair evaluations.
    """
    global _CACHED_MAST3R_MODEL, _CACHED_MODEL_KEY

    hw = detect_mast3r_hardware()
    target_device = device or hw["device"]
    target_dtype = dtype or (torch.float16 if target_device == "cuda" else torch.float32)

    cache_key = f"{model_name_or_path}_{target_device}_{target_dtype}"
    if not force_reload and _CACHED_MAST3R_MODEL is not None and _CACHED_MODEL_KEY == cache_key:
        return _CACHED_MAST3R_MODEL

    from mast3r.model import AsymmetricMASt3R

    # Ensure robust safetensors loading on memory-constrained systems (fallback to pread)
    try:
        import safetensors.torch
        if not getattr(safetensors.torch, "_ptw_patched", False):
            _orig_load_model = safetensors.torch.load_model

            def _safe_load_model(m, filename, strict=True, device="cpu", **kw):
                try:
                    return _orig_load_model(m, filename, strict=strict, device=device, **kw)
                except OSError:
                    logger.info("Retrying safetensors loading with pread backend (paging file fallback)...")
                    return _orig_load_model(m, filename, strict=strict, device=device, backend="pread")

            safetensors.torch.load_model = _safe_load_model
            safetensors.torch._ptw_patched = True
    except Exception:
        pass

    # Check if a direct local checkpoint file or cached snapshot is available
    local_pth = os.path.join(BASE_PROJECT_DIR, "checkpoints", "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
    actual_path = None
    if os.path.isfile(local_pth):
        actual_path = local_pth
    else:
        hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
        snapshots = glob.glob(os.path.join(hf_cache, "models--naver--MASt3R*", "snapshots", "*"))
        if snapshots and os.path.isdir(snapshots[0]):
            actual_path = snapshots[0]
        else:
            actual_path = model_name_or_path

    logger.info(f"Loading official MASt3R model from '{actual_path}' on {target_device.upper()} ({target_dtype})...")
    model = AsymmetricMASt3R.from_pretrained(actual_path)
    model = model.to(target_device)
    if target_device == "cuda" and target_dtype == torch.float16:
        model = model.half()
    model.eval()

    _CACHED_MAST3R_MODEL = model
    _CACHED_MODEL_KEY = cache_key
    return model


def release_mast3r_memory(model=None):
    """Clean up memory and empty CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

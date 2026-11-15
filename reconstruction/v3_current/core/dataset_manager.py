"""
core/dataset_manager.py
-----------------------
Single Source of Truth for Dataset Management & Persistence across all
reconstruction backends (COLMAP, VGGT-1B, MASt3R-SfM).

Key Architecture:
1. Streamlit widget lifecycle: UploadedFile objects are ephemeral and lose stream
   state or disappear on page switches/reruns.
2. Canonical persistent disk storage: All uploaded files are saved into
   `temp_session/uploaded_dataset/` with verified read/write integrity.
3. Multi-backend adaptation: Returns plain absolute file paths (list[str]) on disk
   that survive widget unmounting, page navigation, and backend switching.
4. Comprehensive diagnostics: Emits the exact required debug report at execution start:
   ========== POWER TWIN AI DATASET DEBUG ==========
   Selected backend: <colmap / vggt / mast3r>
   Selected image count: <number>
   Valid image count: <number>
   Saved image count: <number>
   Dataset directory: <path>
   Validation result: <PASS / FAIL>
   Image filenames:
   1. <file1>
   2. <file2>
   ...
   File existence check: <ALL EXIST / MISSING FILES>
   File readability check: <ALL READABLE / CORRUPT FILES>
   ===============================================
"""

import os
import sys
import glob
import json
import shutil
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image

try:
    import streamlit as st
except ImportError:
    st = None

try:
    from config.paths import TEMP_DIR, LOG_FILE
except ImportError:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    TEMP_DIR = os.path.join(BASE_DIR, "temp_session")
    LOG_FILE = os.path.join(TEMP_DIR, "reconstruction.log")

MIN_REQUIRED_IMAGES = 3
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")
SHARED_UPLOAD_DIR = os.path.abspath(os.path.join(TEMP_DIR, "uploaded_dataset"))


def is_valid_image_file(file_path: str) -> bool:
    """Check if file exists and can be successfully decoded by PIL."""
    if not isinstance(file_path, str) or not os.path.exists(file_path) or not os.path.isfile(file_path):
        return False
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def scan_image_directory(folder_path: str) -> List[str]:
    """Scan folder and return all valid, readable image paths."""
    if not os.path.isdir(folder_path):
        return []
    found: List[str] = []
    for ext in SUPPORTED_EXTENSIONS:
        found.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
        found.extend(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))
    unique_paths = sorted(list(set(os.path.abspath(p) for p in found)))
    valid_paths = [p for p in unique_paths if is_valid_image_file(p)]
    return valid_paths


def get_shared_dataset() -> Dict[str, Any]:
    """
    Get the canonical shared dataset dictionary from st.session_state.
    Ensures persistent initialization and validates on-disk availability.
    """
    default_dataset: Dict[str, Any] = {
        "mode": "folder",
        "image_paths": [],
        "filenames": [],
        "count": 0,
        "valid_count": 0,
        "total_bytes": 0,
        "directory": "",
        "valid": False,
        "validation_msg": "No dataset loaded."
    }

    if st is None:
        # Running in standalone script or test environment
        if not hasattr(get_shared_dataset, "_standalone_cache"):
            get_shared_dataset._standalone_cache = default_dataset.copy()
        return get_shared_dataset._standalone_cache

    if "shared_dataset" not in st.session_state:
        st.session_state["shared_dataset"] = default_dataset.copy()

    ds = st.session_state["shared_dataset"]

    # If image_paths are recorded, verify their active on-disk presence
    if ds.get("image_paths"):
        existing = [p for p in ds["image_paths"] if os.path.exists(p)]
        if len(existing) != len(ds["image_paths"]):
            ds["image_paths"] = existing
            ds["filenames"] = [os.path.basename(p) for p in existing]
            ds["count"] = len(existing)
            ds["valid_count"] = sum(1 for p in existing if is_valid_image_file(p))
            ds["total_bytes"] = sum(os.path.getsize(p) for p in existing)
            ds["valid"] = ds["valid_count"] >= MIN_REQUIRED_IMAGES

    # Synchronize legacy session state variables if dataset_files exists and shared_dataset is empty
    if not ds.get("image_paths") and st.session_state.get("dataset_files"):
        legacy_files = st.session_state["dataset_files"]
        if isinstance(legacy_files, list) and len(legacy_files) > 0:
            if isinstance(legacy_files[0], str) and os.path.exists(legacy_files[0]):
                ds["image_paths"] = [os.path.abspath(p) for p in legacy_files if os.path.exists(p)]
                ds["filenames"] = [os.path.basename(p) for p in ds["image_paths"]]
                ds["count"] = len(ds["image_paths"])
                ds["valid_count"] = sum(1 for p in ds["image_paths"] if is_valid_image_file(p))
                ds["total_bytes"] = sum(os.path.getsize(p) for p in ds["image_paths"])
                ds["directory"] = os.path.dirname(ds["image_paths"][0]) if ds["image_paths"] else ""
                ds["mode"] = st.session_state.get("dataset_mode", "folder")
                ds["valid"] = ds["valid_count"] >= MIN_REQUIRED_IMAGES

    return ds


def set_shared_dataset_from_uploaded_files(uploaded_files: list) -> Dict[str, Any]:
    """
    Save uploaded files to canonical directory `temp_session/uploaded_dataset/`
    immediately upon upload, validate readability, and store in session state.
    """
    os.makedirs(SHARED_UPLOAD_DIR, exist_ok=True)

    saved_paths: List[str] = []
    seen_names = set()

    for item in uploaded_files:
        if item is None:
            continue

        # Case A: Streamlit UploadedFile object
        if hasattr(item, "read") and hasattr(item, "name"):
            name = os.path.basename(item.name)
            target_path = os.path.join(SHARED_UPLOAD_DIR, name)

            try:
                if hasattr(item, "seek"):
                    item.seek(0)
                data = item.read()
                if hasattr(item, "seek"):
                    item.seek(0)

                with open(target_path, "wb") as f_out:
                    f_out.write(data)

                saved_paths.append(os.path.abspath(target_path))
                seen_names.add(name)
            except Exception as e:
                print(f"[DATASET] Error saving uploaded file {name}: {e}", file=sys.stderr)

        # Case B: Already a file path string
        elif isinstance(item, str):
            if os.path.exists(item):
                name = os.path.basename(item)
                target_path = os.path.join(SHARED_UPLOAD_DIR, name)
                # Copy into persistent uploaded directory if not already there
                if os.path.abspath(item) != os.path.abspath(target_path):
                    try:
                        shutil.copy2(item, target_path)
                        saved_paths.append(os.path.abspath(target_path))
                    except Exception:
                        saved_paths.append(os.path.abspath(item))
                else:
                    saved_paths.append(os.path.abspath(target_path))
                seen_names.add(name)

    # Validate saved files
    valid_paths = [p for p in saved_paths if is_valid_image_file(p)]
    total_bytes = sum(os.path.getsize(p) for p in valid_paths)
    is_valid = len(valid_paths) >= MIN_REQUIRED_IMAGES

    dataset_state: Dict[str, Any] = {
        "mode": "upload",
        "image_paths": valid_paths,
        "filenames": [os.path.basename(p) for p in valid_paths],
        "count": len(valid_paths),
        "valid_count": len(valid_paths),
        "total_bytes": total_bytes,
        "directory": SHARED_UPLOAD_DIR,
        "valid": is_valid,
        "validation_msg": "Ready for 3D reconstruction." if is_valid else f"Need at least {MIN_REQUIRED_IMAGES} images (have {len(valid_paths)})."
    }

    if st is not None:
        st.session_state["shared_dataset"] = dataset_state
        st.session_state["dataset_mode"] = "upload"
        st.session_state["dataset_files"] = valid_paths
        st.session_state["dataset_folder"] = SHARED_UPLOAD_DIR
    else:
        get_shared_dataset._standalone_cache = dataset_state

    return dataset_state


def set_shared_dataset_from_folder(folder_path: str) -> Dict[str, Any]:
    """
    Scan local directory, validate image files, and store canonical paths.
    """
    clean_path = folder_path.strip().strip('"').strip("'")
    if not os.path.isdir(clean_path):
        dataset_state = {
            "mode": "folder",
            "image_paths": [],
            "filenames": [],
            "count": 0,
            "valid_count": 0,
            "total_bytes": 0,
            "directory": clean_path,
            "valid": False,
            "validation_msg": f"Directory not found: {clean_path}"
        }
    else:
        valid_paths = scan_image_directory(clean_path)
        total_bytes = sum(os.path.getsize(p) for p in valid_paths)
        is_valid = len(valid_paths) >= MIN_REQUIRED_IMAGES
        dataset_state = {
            "mode": "folder",
            "image_paths": valid_paths,
            "filenames": [os.path.basename(p) for p in valid_paths],
            "count": len(valid_paths),
            "valid_count": len(valid_paths),
            "total_bytes": total_bytes,
            "directory": os.path.abspath(clean_path),
            "valid": is_valid,
            "validation_msg": "Ready for 3D reconstruction." if is_valid else f"Need at least {MIN_REQUIRED_IMAGES} images in folder (found {len(valid_paths)})."
        }

    if st is not None:
        st.session_state["shared_dataset"] = dataset_state
        st.session_state["dataset_mode"] = "folder"
        st.session_state["dataset_files"] = dataset_state["image_paths"]
        st.session_state["dataset_folder"] = clean_path
    else:
        get_shared_dataset._standalone_cache = dataset_state

    return dataset_state


def remove_image_by_index(idx: int) -> Dict[str, Any]:
    """Remove image at specified index from active dataset."""
    ds = get_shared_dataset()
    paths = ds.get("image_paths", [])
    if 0 <= idx < len(paths):
        removed_path = paths.pop(idx)
        # If in uploaded dataset, optionally remove file
        if ds.get("mode") == "upload" and os.path.dirname(removed_path) == SHARED_UPLOAD_DIR:
            try:
                if os.path.exists(removed_path):
                    os.remove(removed_path)
            except Exception:
                pass
        ds["image_paths"] = paths
        ds["filenames"] = [os.path.basename(p) for p in paths]
        ds["count"] = len(paths)
        ds["valid_count"] = sum(1 for p in paths if is_valid_image_file(p))
        ds["total_bytes"] = sum(os.path.getsize(p) for p in paths if os.path.exists(p))
        ds["valid"] = ds["valid_count"] >= MIN_REQUIRED_IMAGES

        if st is not None:
            st.session_state["shared_dataset"] = ds
            st.session_state["dataset_files"] = paths

    return ds


def clear_shared_dataset() -> Dict[str, Any]:
    """Reset dataset and wipe temporary uploaded dataset files."""
    if os.path.exists(SHARED_UPLOAD_DIR):
        try:
            shutil.rmtree(SHARED_UPLOAD_DIR)
        except Exception:
            pass
    os.makedirs(SHARED_UPLOAD_DIR, exist_ok=True)

    empty_state: Dict[str, Any] = {
        "mode": "upload",
        "image_paths": [],
        "filenames": [],
        "count": 0,
        "valid_count": 0,
        "total_bytes": 0,
        "directory": SHARED_UPLOAD_DIR,
        "valid": False,
        "validation_msg": "Dataset cleared."
    }

    if st is not None:
        st.session_state["shared_dataset"] = empty_state
        st.session_state["dataset_files"] = []
    else:
        get_shared_dataset._standalone_cache = empty_state

    return empty_state


def validate_dataset_for_reconstruction(backend: str = "colmap") -> Tuple[bool, str, List[str]]:
    """
    Validate that the current dataset satisfies requirements for reconstruction.
    Returns (is_valid, message, list_of_valid_paths).
    """
    ds = get_shared_dataset()
    paths = ds.get("image_paths", [])
    if not paths:
        return False, f"Please provide at least {MIN_REQUIRED_IMAGES} images in Step 1 to enable reconstruction.", []

    existing = [p for p in paths if os.path.exists(p)]
    if len(existing) < MIN_REQUIRED_IMAGES:
        return False, f"Only {len(existing)} images exist on disk (minimum {MIN_REQUIRED_IMAGES} required).", existing

    readable = [p for p in existing if is_valid_image_file(p)]
    if len(readable) < MIN_REQUIRED_IMAGES:
        return False, f"Only {len(readable)} images are readable (minimum {MIN_REQUIRED_IMAGES} required).", readable

    return True, f"Dataset validated: {len(readable)} valid images ready for {backend.upper()}.", readable


def print_dataset_debug_report(backend: str) -> Dict[str, Any]:
    """
    Print the exact required debug report at the reconstruction entry point
    and append it to the log file.
    """
    backend_clean = backend.lower()
    ds = get_shared_dataset()
    paths = ds.get("image_paths", [])

    selected_count = len(paths)
    exist_flags = [os.path.exists(p) for p in paths]
    readable_flags = [is_valid_image_file(p) for p in paths]

    valid_count = sum(1 for r in readable_flags if r)
    saved_count = sum(1 for e in exist_flags if e)
    dataset_dir = ds.get("directory") or (os.path.dirname(paths[0]) if paths else SHARED_UPLOAD_DIR)

    all_exist = all(exist_flags) if paths else False
    all_readable = all(readable_flags) if paths else False
    pass_validation = (valid_count >= MIN_REQUIRED_IMAGES) and all_exist and all_readable

    validation_result = "PASS" if pass_validation else "FAIL"
    existence_check = "ALL EXIST" if all_exist else f"MISSING FILES ({selected_count - saved_count} missing)"
    readability_check = "ALL READABLE" if all_readable else f"CORRUPT FILES ({selected_count - valid_count} corrupt)"

    # Format filenames list
    filenames_lines = []
    for idx, p in enumerate(paths, start=1):
        fn = os.path.basename(p)
        filenames_lines.append(f"{idx}. {fn}")
    filenames_str = "\n".join(filenames_lines) if filenames_lines else "None"

    report_text = f"""========== POWER TWIN AI DATASET DEBUG ==========
Selected backend: {backend_clean}
Selected image count: {selected_count}
Valid image count: {valid_count}
Saved image count: {saved_count}
Dataset directory: {dataset_dir}
Validation result: {validation_result}
Image filenames:
{filenames_str}
File existence check: {existence_check}
File readability check: {readability_check}
==============================================="""

    # Emit to stdout immediately
    print(report_text, flush=True)

    # Write to log file if available
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as lf:
            lf.write(report_text + "\n\n")
    except Exception:
        pass

    return {
        "backend": backend_clean,
        "selected_image_count": selected_count,
        "valid_image_count": valid_count,
        "saved_image_count": saved_count,
        "dataset_directory": dataset_dir,
        "validation_result": validation_result,
        "existence_check": existence_check,
        "readability_check": readability_check,
        "paths": paths
    }

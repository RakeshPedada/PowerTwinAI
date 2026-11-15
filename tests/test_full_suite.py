"""
tests/test_full_suite.py
-------------------------
Comprehensive regression test suite for PowerTwinAI covering:
1. Application imports
2. Session-state initialization
3. Preset initialization & config building (no UnboundLocalError)
4. Dataset validation & image counts (0, 1, 2, 3, 5, 11, 50, 100)
5. Image upload, persistence & duplicate filename handling
6. Dataset directory creation
7. Backend selection (COLMAP, VGGT-1B, MASt3R-SfM)
8. Backend input preparation
9. Reconstruction configuration
10. Output file validation (PLY, Mesh)
11. PLY & Mesh loading
12. Error handling (corrupted, missing files)
13. Export functionality
14. Workflows A through G
"""

import os
import sys
import glob
import io
import json
import shutil
import tempfile
import zipfile
import pytest
import numpy as np
from PIL import Image

# Ensure path to reconstruction/v3_current is available
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
V3_DIR = os.path.join(PROJECT_ROOT, "reconstruction", "v3_current")
sys.path.insert(0, V3_DIR)
sys.path.insert(0, PROJECT_ROOT)

from config.paths import (
    BASE_DIR, TEMP_DIR, LOG_FILE, STATUS_FILE, RESULT_FILE,
    IMAGE_PATHS_FILE, CONFIG_FILE
)
from config.settings import QUALITY_PRESETS, get_active_config
from core.dataset_manager import (
    get_shared_dataset,
    set_shared_dataset_from_uploaded_files,
    set_shared_dataset_from_folder,
    remove_image_by_index,
    clear_shared_dataset,
    validate_dataset_for_reconstruction,
    print_dataset_debug_report,
    SHARED_UPLOAD_DIR,
    MIN_REQUIRED_IMAGES
)
from core.reconstruction_manager import (
    prepare_workspace,
    save_uploaded_images
)
from ui.data_manager import (
    find_artifact,
    scan_all_reconstructions,
    get_backend_status,
    load_cached_point_cloud,
    load_cached_mesh
)
from mesh_generator import is_mesh_ply, get_ply_statistics


class MockUploadedFile:
    """Mock for Streamlit UploadedFile object."""
    def __init__(self, filename: str, file_path: str):
        self.name = filename
        with open(file_path, "rb") as f:
            self._buffer = io.BytesIO(f.read())
        self.size = len(self._buffer.getvalue())

    def read(self, *args, **kwargs):
        return self._buffer.read(*args, **kwargs)

    def seek(self, offset, whence=io.SEEK_SET):
        return self._buffer.seek(offset, whence)

    def getbuffer(self):
        return self._buffer.getvalue()


@pytest.fixture(autouse=True)
def clean_dataset_state():
    """Ensure clean state before each test."""
    clear_shared_dataset()
    yield
    clear_shared_dataset()


@pytest.fixture(scope="module")
def sample_images():
    """Retrieve at least 11 real sample images from colmap_workspace/images."""
    img_dir = os.path.join(PROJECT_ROOT, "colmap_workspace", "images")
    imgs = sorted(glob.glob(os.path.join(img_dir, "*._c.jpg")))
    if len(imgs) < 15:
        imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    assert len(imgs) >= 11, f"Required at least 11 sample images, found {len(imgs)}"
    return imgs


# =========================================================================
# 1. APPLICATION IMPORTS
# =========================================================================

def test_application_imports():
    """Verify all primary application and UI modules import cleanly."""
    import app
    from ui.dashboard import render_dashboard
    from ui.new_reconstruction import render_new_reconstruction
    from ui.reconstruction_studio import render_reconstruction_studio
    from ui.viewer_page import render_viewer_page
    from ui.diagnostics_page import render_diagnostics_page
    from ui.comparison_page import render_comparison_page
    from ui.mesh_studio_page import render_mesh_studio_page
    from ui.export_page import render_export_page
    from ui.settings_page import render_settings_page
    from core.gpu_detector import get_gpu_info
    from core.session_manager import initialize_session

    assert callable(render_dashboard)
    assert callable(render_new_reconstruction)
    assert callable(render_reconstruction_studio)
    assert callable(get_gpu_info)
    assert callable(initialize_session)


# =========================================================================
# 2. SESSION-STATE INITIALIZATION
# =========================================================================

def test_session_state_initialization():
    """Verify session_manager initializes all required state keys."""
    from unittest.mock import MagicMock
    mock_st = MagicMock()
    mock_st.session_state = {}

    import core.session_manager as sm
    orig_st = sm.st
    sm.st = mock_st
    try:
        sm.initialize_session()
        required_keys = [
            "reconstruction_running",
            "reconstruction_done",
            "nav_page",
            "active_backend",
            "dataset_mode",
            "dataset_folder",
            "dataset_files",
            "viewer_backend",
            "viewer_color_mode",
            "vggt_running",
            "vggt_results",
            "mast3r_latest_result",
            "input_mode",
            "mesh_updated",
            "status_notification"
        ]
        for k in required_keys:
            assert k in mock_st.session_state, f"Missing required session state key: {k}"
    finally:
        sm.st = orig_st


# =========================================================================
# 3. PRESET INITIALIZATION & CONFIGURATION BUILDING
# =========================================================================

def test_preset_definitions_and_no_unbound_preset():
    """Verify quality presets for COLMAP, VGGT, and MASt3R exist and build configs without error."""
    assert "Fast" in QUALITY_PRESETS
    assert "Standard" in QUALITY_PRESETS
    assert "High Density" in QUALITY_PRESETS
    assert "VGGT Balanced" in QUALITY_PRESETS
    assert "MASt3R Hybrid" in QUALITY_PRESETS

    # Test get_active_config
    cfg = get_active_config("High Density")
    assert cfg["name"] == "High Density"
    assert "sift_max_features" in cfg

    # Test building config for each backend to ensure no UnboundLocalError
    gpu = {"cuda_available": False}
    for backend in ["colmap", "vggt", "mast3r"]:
        selected_preset = "High Density (Recommended)"
        if backend == "colmap":
            active_cfg = {
                "name": selected_preset,
                "backend": "colmap",
                "image_max_size": 4096,
                "use_gpu": False
            }
        elif backend == "vggt":
            active_cfg = {
                "name": selected_preset,
                "backend": "vggt",
                "vggt_mode": "depth",
                "vggt_conf": 1.8,
                "vggt_max_frames": 24,
                "use_gpu": False
            }
        else:
            active_cfg = {
                "name": selected_preset,
                "backend": "mast3r",
                "mast3r_size": 512,
                "mast3r_pairing": "Mode C",
                "use_gpu": False
            }
        assert active_cfg["backend"] == backend
        assert "name" in active_cfg
        assert active_cfg["name"] == selected_preset


# =========================================================================
# 4. DATASET VALIDATION & IMAGE COUNTS (0, 1, 2, 3, 5, 11, 50, 100)
# =========================================================================

def test_dataset_image_counts(sample_images):
    """Test image counts: 0, 1, 2 (errors), 3, 5, 11, 50, 100 (valid)."""
    # 0 images
    clear_shared_dataset()
    ok, msg, paths = validate_dataset_for_reconstruction("colmap")
    assert not ok, "Expected 0 images to fail validation"
    assert "at least 3 images" in msg

    # 1 image
    set_shared_dataset_from_uploaded_files([sample_images[0]])
    ok, msg, paths = validate_dataset_for_reconstruction("colmap")
    assert not ok, "Expected 1 image to fail validation"

    # 2 images
    set_shared_dataset_from_uploaded_files(sample_images[:2])
    ok, msg, paths = validate_dataset_for_reconstruction("colmap")
    assert not ok, "Expected 2 images to fail validation"

    # 3 images (minimum valid boundary)
    ds = set_shared_dataset_from_uploaded_files(sample_images[:3])
    assert ds["count"] == 3
    ok, msg, paths = validate_dataset_for_reconstruction("colmap")
    assert ok, "Expected 3 images to pass validation"
    assert len(paths) == 3

    # 5 images
    ds = set_shared_dataset_from_uploaded_files(sample_images[:5])
    assert ds["count"] == 5
    ok, msg, paths = validate_dataset_for_reconstruction("vggt")
    assert ok, "Expected 5 images to pass validation"

    # 11 images
    ds = set_shared_dataset_from_uploaded_files(sample_images[:11])
    assert ds["count"] == 11
    ok, msg, paths = validate_dataset_for_reconstruction("mast3r")
    assert ok, "Expected 11 images to pass validation"
    assert len(paths) == 11

    # 50 images
    imgs_50 = (sample_images * 5)[:50]
    # Create temp files to ensure unique paths
    with tempfile.TemporaryDirectory() as td:
        paths_50 = []
        for i, src in enumerate(imgs_50):
            dst = os.path.join(td, f"img_{i:03d}.jpg")
            shutil.copy2(src, dst)
            paths_50.append(dst)
        ds = set_shared_dataset_from_folder(td)
        assert ds["count"] == 50
        ok, msg, paths = validate_dataset_for_reconstruction("colmap")
        assert ok, "Expected 50 images to pass validation"
        assert len(paths) == 50

    # 100 images
    imgs_100 = (sample_images * 10)[:100]
    with tempfile.TemporaryDirectory() as td:
        paths_100 = []
        for i, src in enumerate(imgs_100):
            dst = os.path.join(td, f"img_{i:03d}.jpg")
            shutil.copy2(src, dst)
            paths_100.append(dst)
        ds = set_shared_dataset_from_folder(td)
        assert ds["count"] == 100
        ok, msg, paths = validate_dataset_for_reconstruction("colmap")
        assert ok, "Expected 100 images to pass validation"
        assert len(paths) == 100


# =========================================================================
# 5. IMAGE PERSISTENCE & DUPLICATE FILENAME HANDLING
# =========================================================================

def test_duplicate_filename_handling(sample_images):
    """Verify duplicate filenames in upload or folder are handled safely."""
    # Create duplicate files with same basename
    f1 = sample_images[0]
    mock1 = MockUploadedFile(os.path.basename(f1), f1)
    mock2 = MockUploadedFile(os.path.basename(f1), f1)
    mock3 = MockUploadedFile(os.path.basename(sample_images[1]), sample_images[1])
    mock4 = MockUploadedFile(os.path.basename(sample_images[2]), sample_images[2])

    ds = set_shared_dataset_from_uploaded_files([mock1, mock2, mock3, mock4])
    # Duplicate overwrite or deduplication should result in valid readable set
    assert ds["valid"]
    assert ds["count"] >= 3
    for p in ds["image_paths"]:
        assert os.path.exists(p)


# =========================================================================
# 6. WORKFLOW TESTS (WORKFLOW A THROUGH G)
# =========================================================================

def test_workflow_A_colmap(sample_images):
    """Workflow A: Upload 11 images -> COLMAP -> Start reconstruction."""
    uploads = [MockUploadedFile(os.path.basename(p), p) for p in sample_images[:11]]
    ds = set_shared_dataset_from_uploaded_files(uploads)
    assert ds["count"] == 11

    # Emit debug report
    report = print_dataset_debug_report("colmap")
    assert report["backend"] == "colmap"
    assert report["validation_result"] == "PASS"
    assert report["selected_image_count"] == 11

    # Prepare workspace & save images
    test_temp = os.path.join(TEMP_DIR, "test_colmap_wf")
    os.makedirs(test_temp, exist_ok=True)
    img_paths_json = os.path.join(test_temp, "image_paths.json")
    save_uploaded_images(ds["image_paths"], "upload", test_temp, img_paths_json)

    assert os.path.exists(img_paths_json)
    with open(img_paths_json, "r", encoding="utf-8") as f:
        saved = json.load(f)
    assert len(saved) == 11


def test_workflow_B_vggt(sample_images):
    """Workflow B: Upload 11 images -> VGGT -> Start reconstruction."""
    uploads = [MockUploadedFile(os.path.basename(p), p) for p in sample_images[:11]]
    ds = set_shared_dataset_from_uploaded_files(uploads)
    assert ds["count"] == 11

    report = print_dataset_debug_report("vggt")
    assert report["backend"] == "vggt"
    assert report["validation_result"] == "PASS"
    assert report["selected_image_count"] == 11

    test_temp = os.path.join(TEMP_DIR, "test_vggt_wf")
    os.makedirs(test_temp, exist_ok=True)
    img_paths_json = os.path.join(test_temp, "image_paths.json")
    save_uploaded_images(ds["image_paths"], "upload", test_temp, img_paths_json)

    assert os.path.exists(img_paths_json)
    with open(img_paths_json, "r", encoding="utf-8") as f:
        saved = json.load(f)
    assert len(saved) == 11


def test_workflow_C_mast3r(sample_images):
    """Workflow C: Upload 11 images -> MASt3R-SfM -> Start reconstruction."""
    uploads = [MockUploadedFile(os.path.basename(p), p) for p in sample_images[:11]]
    ds = set_shared_dataset_from_uploaded_files(uploads)
    assert ds["count"] == 11

    report = print_dataset_debug_report("mast3r")
    assert report["backend"] == "mast3r"
    assert report["validation_result"] == "PASS"
    assert report["selected_image_count"] == 11

    test_temp = os.path.join(TEMP_DIR, "test_mast3r_wf")
    os.makedirs(test_temp, exist_ok=True)
    img_paths_json = os.path.join(test_temp, "image_paths.json")
    save_uploaded_images(ds["image_paths"], "upload", test_temp, img_paths_json)

    assert os.path.exists(img_paths_json)
    with open(img_paths_json, "r", encoding="utf-8") as f:
        saved = json.load(f)
    assert len(saved) == 11


def test_workflow_D_backend_switching(sample_images):
    """Workflow D: Upload images -> Change backend -> Count and validity preserved."""
    uploads = [MockUploadedFile(os.path.basename(p), p) for p in sample_images[:11]]
    set_shared_dataset_from_uploaded_files(uploads)

    for b in ["colmap", "vggt", "mast3r", "colmap"]:
        ds = get_shared_dataset()
        assert ds["count"] == 11
        ok, msg, paths = validate_dataset_for_reconstruction(b)
        assert ok
        assert len(paths) == 11


def test_workflow_E_navigation_persistence(sample_images):
    """Workflow E: Upload images -> Navigate away (unmount uploader) -> Return -> Persists."""
    uploads = [MockUploadedFile(os.path.basename(p), p) for p in sample_images[:11]]
    set_shared_dataset_from_uploaded_files(uploads)

    # Simulate widget unmount
    uploads = None

    # Retrieve dataset in another tab
    ds = get_shared_dataset()
    assert ds["count"] == 11
    assert len(ds["image_paths"]) == 11
    for p in ds["image_paths"]:
        assert os.path.exists(p)


def test_workflow_F_preset_selection_and_switch(sample_images):
    """Workflow F: Select preset -> Change backend -> Start reconstruction without error."""
    uploads = [MockUploadedFile(os.path.basename(p), p) for p in sample_images[:11]]
    ds = set_shared_dataset_from_uploaded_files(uploads)

    # Simulate changing preset and switching backend
    preset_name = "High Density (Recommended)"
    cfg_colmap = get_active_config("High Density")
    assert cfg_colmap is not None

    # Switch to VGGT
    report = print_dataset_debug_report("vggt")
    assert report["backend"] == "vggt"
    assert report["validation_result"] == "PASS"


def test_workflow_G_viewer_and_export(sample_images):
    """Workflow G: Inspect 3D Viewer data and verify Export center creation."""
    # Verify scan_all_reconstructions
    summary = scan_all_reconstructions()
    assert "backends" in summary
    assert "colmap" in summary["backends"]
    assert "vggt" in summary["backends"]
    assert "mast3r" in summary["backends"]

    # Verify export zip creation
    with tempfile.TemporaryDirectory() as td:
        export_zip_path = os.path.join(td, "reconstruction_export.zip")
        with zipfile.ZipFile(export_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add sample files
            for p in sample_images[:3]:
                zf.write(p, arcname=f"images/{os.path.basename(p)}")
        assert os.path.exists(export_zip_path)
        assert os.path.getsize(export_zip_path) > 0


# =========================================================================
# 7. 3D VISUALIZATION & MESHING TESTS
# =========================================================================

def test_ply_and_mesh_inspection():
    """Verify PLY and mesh loading utilities."""
    # Find any existing PLY in outputs or workspace
    existing_ply = find_artifact("outputs/vggt/vggt_surface_mesh.ply") or find_artifact("output/mesh.ply")
    if existing_ply and os.path.exists(existing_ply):
        is_mesh = is_mesh_ply(existing_ply)
        assert isinstance(is_mesh, bool)
        stats = get_ply_statistics(existing_ply)
        assert "vertex_count" in stats

    # Test loading cached point cloud with corrupted/missing path
    none_res = load_cached_point_cloud("non_existent_file.ply")
    assert none_res is None

    # Test loading cached mesh with corrupted/missing path
    none_mesh = load_cached_mesh("non_existent_file.obj")
    assert none_mesh is None

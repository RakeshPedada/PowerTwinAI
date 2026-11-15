"""
tests/test_dataset_lifecycle.py
--------------------------------
Comprehensive test suite verifying the Shared Dataset Manager across
all reconstruction backends (COLMAP, VGGT-1B, MASt3R-SfM).

Executes Tests A through G as specified:
- Test A: 11 images -> select COLMAP -> verify recognition & debug report
- Test B: 11 images -> select VGGT-1B -> verify recognition & debug report
- Test C: 11 images -> select MASt3R-SfM -> verify recognition & debug report
- Test D: 11 images -> switch between COLMAP, VGGT, MASt3R -> verify count=11 preserved
- Test E: 11 images -> simulate page navigation & widget unmount -> verify persistence
- Test F: Remove 1 image -> verify count updates to 10 across all modes
- Test G: Add images -> verify count updates dynamically
"""

import os
import sys
import glob
import io
import json
import shutil
import unittest
from unittest.mock import MagicMock
from PIL import Image

# Ensure path to reconstruction/v3_current is available
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
V3_DIR = os.path.join(PROJECT_ROOT, "reconstruction", "v3_current")
sys.path.insert(0, V3_DIR)
sys.path.insert(0, PROJECT_ROOT)

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


class TestSharedDatasetLifecycle(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Locate sample images in colmap_workspace/images
        cls.img_dir = os.path.join(PROJECT_ROOT, "colmap_workspace", "images")
        all_imgs = sorted(glob.glob(os.path.join(cls.img_dir, "*._c.jpg")))
        if len(all_imgs) < 15:
            all_imgs = sorted(glob.glob(os.path.join(cls.img_dir, "*.jpg")))

        assert len(all_imgs) >= 12, f"Expected at least 12 sample images, found {len(all_imgs)}"
        cls.sample_11_paths = all_imgs[:11]
        cls.extra_image_path = all_imgs[11]

    def setUp(self):
        clear_shared_dataset()

    def tearDown(self):
        clear_shared_dataset()

    def test_A_colmap_11_images(self):
        """Test A: 11 uploaded images -> COLMAP mode recognition & validation."""
        print("\n--- RUNNING TEST A: 11 Images -> COLMAP ---")
        mock_uploads = [MockUploadedFile(os.path.basename(p), p) for p in self.sample_11_paths]
        self.assertEqual(len(mock_uploads), 11)

        ds = set_shared_dataset_from_uploaded_files(mock_uploads)
        self.assertEqual(ds["count"], 11)
        self.assertEqual(ds["valid_count"], 11)
        self.assertTrue(ds["valid"])

        # Check validation
        ok, msg, paths = validate_dataset_for_reconstruction("colmap")
        self.assertTrue(ok)
        self.assertEqual(len(paths), 11)

        # Emits debug report
        rep = print_dataset_debug_report("colmap")
        self.assertEqual(rep["backend"], "colmap")
        self.assertEqual(rep["selected_image_count"], 11)
        self.assertEqual(rep["valid_image_count"], 11)
        self.assertEqual(rep["validation_result"], "PASS")
        self.assertEqual(rep["existence_check"], "ALL EXIST")
        self.assertEqual(rep["readability_check"], "ALL READABLE")
        print("Test A Passed: 11 images verified and recognized for COLMAP.")

    def test_B_vggt_11_images(self):
        """Test B: 11 uploaded images -> VGGT mode recognition & validation."""
        print("\n--- RUNNING TEST B: 11 Images -> VGGT ---")
        mock_uploads = [MockUploadedFile(os.path.basename(p), p) for p in self.sample_11_paths]

        set_shared_dataset_from_uploaded_files(mock_uploads)
        ok, msg, paths = validate_dataset_for_reconstruction("vggt")
        self.assertTrue(ok)
        self.assertEqual(len(paths), 11)

        rep = print_dataset_debug_report("vggt")
        self.assertEqual(rep["backend"], "vggt")
        self.assertEqual(rep["selected_image_count"], 11)
        self.assertEqual(rep["valid_image_count"], 11)
        self.assertEqual(rep["validation_result"], "PASS")
        print("Test B Passed: 11 images verified and recognized for VGGT.")

    def test_C_mast3r_11_images(self):
        """Test C: 11 uploaded images -> MASt3R mode recognition & validation."""
        print("\n--- RUNNING TEST C: 11 Images -> MASt3R-SfM ---")
        mock_uploads = [MockUploadedFile(os.path.basename(p), p) for p in self.sample_11_paths]

        set_shared_dataset_from_uploaded_files(mock_uploads)
        ok, msg, paths = validate_dataset_for_reconstruction("mast3r")
        self.assertTrue(ok)
        self.assertEqual(len(paths), 11)

        rep = print_dataset_debug_report("mast3r")
        self.assertEqual(rep["backend"], "mast3r")
        self.assertEqual(rep["selected_image_count"], 11)
        self.assertEqual(rep["valid_image_count"], 11)
        self.assertEqual(rep["validation_result"], "PASS")
        print("Test C Passed: 11 images verified and recognized for MASt3R-SfM.")

    def test_D_backend_switching(self):
        """Test D: Switch between COLMAP -> VGGT -> MASt3R -> count remains 11."""
        print("\n--- RUNNING TEST D: Backend Switching ---")
        mock_uploads = [MockUploadedFile(os.path.basename(p), p) for p in self.sample_11_paths]
        set_shared_dataset_from_uploaded_files(mock_uploads)

        for backend in ["colmap", "vggt", "mast3r", "colmap"]:
            ds = get_shared_dataset()
            self.assertEqual(ds["count"], 11, f"Count changed during switch to {backend}")
            ok, msg, paths = validate_dataset_for_reconstruction(backend)
            self.assertTrue(ok)
            self.assertEqual(len(paths), 11)
        print("Test D Passed: Image count remained 11 through all backend transitions.")

    def test_E_persistence_across_reruns(self):
        """Test E: Navigate away/rerun -> verify disk paths and validity survive."""
        print("\n--- RUNNING TEST E: Persistence Across Widget Lifecycles ---")
        mock_uploads = [MockUploadedFile(os.path.basename(p), p) for p in self.sample_11_paths]
        set_shared_dataset_from_uploaded_files(mock_uploads)

        # Simulate Streamlit unmounting uploader and resetting widget
        mock_uploads = None

        # Call get_shared_dataset from a new routine
        ds_after = get_shared_dataset()
        self.assertEqual(ds_after["count"], 11)
        self.assertEqual(len(ds_after["image_paths"]), 11)
        for p in ds_after["image_paths"]:
            self.assertTrue(os.path.exists(p), f"Persistent file vanished: {p}")
        print("Test E Passed: Dataset persisted cleanly across widget unmount.")

    def test_F_remove_image(self):
        """Test F: Remove one image -> verify count updates to 10 across all modes."""
        print("\n--- RUNNING TEST F: Image Removal ---")
        mock_uploads = [MockUploadedFile(os.path.basename(p), p) for p in self.sample_11_paths]
        set_shared_dataset_from_uploaded_files(mock_uploads)

        ds = remove_image_by_index(0)
        self.assertEqual(ds["count"], 10)
        self.assertEqual(len(ds["image_paths"]), 10)

        for backend in ["colmap", "vggt", "mast3r"]:
            ok, _, paths = validate_dataset_for_reconstruction(backend)
            self.assertTrue(ok)
            self.assertEqual(len(paths), 10)
        print("Test F Passed: Count updated to 10 across all modes after deletion.")

    def test_G_add_images(self):
        """Test G: Add new images -> verify count updates dynamically."""
        print("\n--- RUNNING TEST G: Dynamic Image Addition ---")
        mock_uploads = [MockUploadedFile(os.path.basename(p), p) for p in self.sample_11_paths]
        set_shared_dataset_from_uploaded_files(mock_uploads)

        # Add 12th image
        extra_upload = MockUploadedFile(os.path.basename(self.extra_image_path), self.extra_image_path)
        combined_uploads = mock_uploads + [extra_upload]

        ds = set_shared_dataset_from_uploaded_files(combined_uploads)
        self.assertEqual(ds["count"], 12)
        self.assertEqual(len(ds["image_paths"]), 12)

        for backend in ["colmap", "vggt", "mast3r"]:
            ok, _, paths = validate_dataset_for_reconstruction(backend)
            self.assertTrue(ok)
            self.assertEqual(len(paths), 12)
        print("Test G Passed: Image addition successfully updated count to 12.")


if __name__ == "__main__":
    unittest.main()

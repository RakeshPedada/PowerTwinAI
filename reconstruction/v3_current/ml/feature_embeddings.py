"""
ml/feature_embeddings.py
------------------------
Visual embedding extraction engine for ML image pair similarity ranking.
Provides fast, lightweight, pure-CPU visual feature descriptors with
zero GPU dependency.
"""

import os
from typing import List, Optional, Tuple
import cv2
import numpy as np


class SpatialPyramidVisualEmbedding:
    """
    Lightweight, deterministic visual descriptor combining spatial pyramid
    color moments and directional gradient texture histograms.

    Captures both global scene color layout and localized structural boundaries
    without requiring heavy neural network downloads or CUDA.
    """

    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        self.target_size = target_size

    def extract(self, image_input) -> np.ndarray:
        """
        Extract a normalized 200-dimensional spatial pyramid visual descriptor.

        Parameters
        ----------
        image_input : str or np.ndarray
            Path to image or pre-loaded BGR image numpy array.

        Returns
        -------
        np.ndarray
            1D float32 normalized embedding vector of shape (200,).
        """
        if isinstance(image_input, (str, bytes, os.PathLike)):
            img = cv2.imread(str(image_input), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Could not read image from path: {image_input}")
        else:
            img = image_input

        # Canonical resize
        img_small = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)

        # Multi-color space conversions
        hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img_small, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)

        h, w = self.target_size
        h_half, w_half = h // 2, w // 2

        # 2-Level Spatial Pyramid: Level 0 (1x1 global), Level 1 (2x2 quadrants)
        grids = [
            (0, 0, h, w),               # Global
            (0, 0, h_half, w_half),     # Top-Left
            (0, w_half, h_half, w),     # Top-Right
            (h_half, 0, h, w_half),     # Bottom-Left
            (h_half, w_half, h, w),     # Bottom-Right
        ]

        features = []
        for y1, x1, y2, x2 in grids:
            hsv_cell = hsv[y1:y2, x1:x2]
            lab_cell = lab[y1:y2, x1:x2]
            gray_cell = gray[y1:y2, x1:x2]

            # 1. Color distribution: Hue (16 bins), Saturation (8 bins), Chroma-B (8 bins)
            h_hist = cv2.calcHist([hsv_cell], [0], None, [16], [0, 180]).flatten()
            s_hist = cv2.calcHist([hsv_cell], [1], None, [8], [0, 256]).flatten()
            b_hist = cv2.calcHist([lab_cell], [2], None, [8], [0, 256]).flatten()

            # 2. Structural texture: Sobel gradient orientation histogram (8 bins)
            gx = cv2.Sobel(gray_cell, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray_cell, cv2.CV_32F, 0, 1, ksize=3)
            mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
            g_hist = np.histogram(ang, bins=8, range=(0, 360), weights=mag)[0].astype(np.float32)

            cell_feat = np.concatenate([h_hist, s_hist, b_hist, g_hist]).astype(np.float32)
            norm = np.linalg.norm(cell_feat)
            if norm > 1e-7:
                cell_feat /= norm
            features.append(cell_feat)

        vec = np.concatenate(features).astype(np.float32)
        total_norm = np.linalg.norm(vec)
        if total_norm > 1e-7:
            vec /= total_norm
        return vec


class ONNXVisualEmbedding:
    """
    Optional deep learning visual embedding extractor using ONNX Runtime CPU.
    """

    def __init__(self, model_path: str):
        import onnxruntime as ort
        self.model_path = model_path
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def extract(self, image_input) -> np.ndarray:
        if isinstance(image_input, (str, bytes, os.PathLike)):
            img = cv2.imread(str(image_input), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Could not read image: {image_input}")
        else:
            img = image_input

        # Standard ImageNet preprocessing
        img_rgb = cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_rgb - mean) / std
        inp = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]

        out = self.session.run(None, {self.input_name: inp})[0].flatten()
        norm = np.linalg.norm(out)
        if norm > 1e-7:
            out /= norm
        return out


class VisualEmbeddingExtractor:
    """
    Unified manager for extracting image feature embeddings across an image list.
    """

    def __init__(self, method: str = "auto", model_path: Optional[str] = None):
        self.method = method
        self.backend = None

        if method in ("onnx", "auto") and model_path and os.path.exists(model_path):
            try:
                self.backend = ONNXVisualEmbedding(model_path)
            except Exception as e:
                print(f"[ML-EMBEDDING] Warning initializing ONNX backend: {e}. Falling back to spatial pyramid.")

        if self.backend is None:
            self.backend = SpatialPyramidVisualEmbedding()

    def extract_single(self, image_path: str) -> np.ndarray:
        return self.backend.extract(image_path)

    def extract_batch(
        self,
        image_paths: List[str],
        progress_callback=None
    ) -> np.ndarray:
        """
        Extract normalized embedding vectors for an entire list of images.
        """
        embeddings = []
        total = len(image_paths)
        for idx, path in enumerate(image_paths):
            emb = self.extract_single(path)
            embeddings.append(emb)
            if progress_callback and (idx % max(1, total // 5) == 0 or idx == total - 1):
                progress_callback(f"[ML-EMBEDDING] Extracted {idx + 1}/{total} visual embeddings...")

        return np.asarray(embeddings, dtype=np.float32)

"""
tests/test_ml_pair_ranker.py
-----------------------------
Unit tests for the ML-based visual embedding extraction, pair ranking,
graph connectivity enforcement, and bridge generation.
"""

import os
import sys
import unittest
import numpy as np
from PIL import Image
import tempfile
import shutil

# Add reconstruction/v3_current to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.config import MLPairRankingConfig
from ml.feature_embeddings import SpatialPyramidVisualEmbedding
from ml.image_pair_ranker import ImagePairRanker, save_colmap_pairs, compute_graph_metrics


class TestMLPairRanker(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_ml_pair_")
        self.image_paths = []

        # Create 6 synthetic test images (3 red-ish, 3 blue-ish)
        for i in range(6):
            img_path = os.path.join(self.temp_dir, f"img_{i:02d}.jpg")
            if i < 3:
                # Red cluster
                arr = np.zeros((100, 100, 3), dtype=np.uint8)
                arr[:, :, 0] = 200 + i * 15
                arr[:, :, 1] = 20 + i * 5
                arr[:, :, 2] = 20
            else:
                # Blue cluster
                arr = np.zeros((100, 100, 3), dtype=np.uint8)
                arr[:, :, 0] = 20
                arr[:, :, 1] = 30 + i * 5
                arr[:, :, 2] = 200 + (i - 3) * 15

            img = Image.fromarray(arr)
            img.save(img_path)
            self.image_paths.append(img_path)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_spatial_pyramid_embedding_properties(self):
        """Test embedding dimensionality, L2 normalization, and determinism."""
        extractor = SpatialPyramidVisualEmbedding()
        
        # Test dimensionality
        vec1 = extractor.extract(self.image_paths[0])
        self.assertEqual(vec1.ndim, 1)
        self.assertEqual(vec1.shape[0], 200)

        # Test L2 normalization
        norm = np.linalg.norm(vec1)
        self.assertAlmostEqual(norm, 1.0, places=5)

        # Test determinism
        vec2 = extractor.extract(self.image_paths[0])
        np.testing.assert_allclose(vec1, vec2, rtol=1e-6)

        # Test intra-cluster similarity > inter-cluster similarity
        vec_red2 = extractor.extract(self.image_paths[1])
        vec_blue = extractor.extract(self.image_paths[3])
        sim_red_red = float(np.dot(vec1, vec_red2))
        sim_red_blue = float(np.dot(vec1, vec_blue))
        self.assertGreater(sim_red_red, sim_red_blue)

    def test_pair_ranking_selection(self):
        """Test candidate pair selection with top-k and min-pairs constraints."""
        config = MLPairRankingConfig(
            enabled=True,
            top_k=2,
            min_pairs_per_image=2,
            similarity_threshold=0.0,
            auto_bridge_components=True,
            ensure_connected=True
        )
        ranker = ImagePairRanker(config=config)
        pairs, telemetry = ranker.rank_images(self.image_paths)

        # Verify no self-pairs (i == j)
        for name_a, name_b in pairs:
            self.assertNotEqual(name_a, name_b)

        # Verify canonical ordering (i < j or alphabetical)
        for name_a, name_b in pairs:
            self.assertLess(name_a, name_b)

        # Verify no duplicate pairs
        pair_set = set(pairs)
        self.assertEqual(len(pair_set), len(pairs))

        # Baseline exhaustive for 6 images: 6 * 5 / 2 = 15
        self.assertEqual(telemetry["exhaustive_pairs_baseline"], 15)
        self.assertEqual(telemetry["total_images"], 6)
        self.assertGreater(telemetry["candidate_pairs_selected"], 0)
        self.assertLessEqual(telemetry["candidate_pairs_selected"], 15)

    def test_graph_connectivity_and_bridging(self):
        """Test that ranker guarantees exactly 1 connected component and no orphan nodes."""
        config = MLPairRankingConfig(
            enabled=True,
            top_k=1,
            min_pairs_per_image=1,
            auto_bridge_components=True,
            ensure_connected=True
        )
        ranker = ImagePairRanker(config=config)
        pairs, telemetry = ranker.rank_images(self.image_paths)

        image_basenames = [os.path.basename(p) for p in self.image_paths]
        metrics = compute_graph_metrics(image_basenames, pairs)

        # Must have exactly 1 connected component
        self.assertEqual(metrics["connected_components"], 1)
        self.assertTrue(metrics["is_connected"])
        self.assertGreaterEqual(metrics["min_degree"], 1)

    def test_save_colmap_pairs(self):
        """Test saving candidate pairs in COLMAP matches_importer format."""
        pairs = [("img_00.jpg", "img_01.jpg"), ("img_01.jpg", "img_02.jpg")]
        out_path = os.path.join(self.temp_dir, "test_pairs.txt")
        save_colmap_pairs(pairs, out_path)

        self.assertTrue(os.path.exists(out_path))
        with open(out_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0], "img_00.jpg img_01.jpg")
        self.assertEqual(lines[1], "img_01.jpg img_02.jpg")

    def test_config_and_env_overrides(self):
        """Test configuration defaults and environment variable overrides."""
        orig_top_k = os.environ.get("ML_TOP_K")
        try:
            os.environ["ML_TOP_K"] = "14"
            cfg = MLPairRankingConfig()
            self.assertEqual(cfg.top_k, 14)
        finally:
            if orig_top_k is not None:
                os.environ["ML_TOP_K"] = orig_top_k
            else:
                os.environ.pop("ML_TOP_K", None)

    def test_multi_cluster_bridging(self):
        """Test that ranker connects 3 distinct visual clusters into 1 single connected component."""
        # Create 3 green images
        for i in range(3):
            green_path = os.path.join(self.temp_dir, f"green_{i:02d}.jpg")
            arr = np.zeros((100, 100, 3), dtype=np.uint8)
            arr[:, :, 1] = 200 + i * 15
            Image.fromarray(arr).save(green_path)
            self.image_paths.append(green_path)

        # 9 total images now: 3 red, 3 blue, 3 green
        config = MLPairRankingConfig(
            enabled=True,
            top_k=2,
            min_pairs_per_image=1,
            auto_bridge_components=True,
            ensure_connected=True
        )
        ranker = ImagePairRanker(config=config)
        pairs, telemetry = ranker.rank_images(self.image_paths)

        image_basenames = [os.path.basename(p) for p in self.image_paths]
        metrics = compute_graph_metrics(image_basenames, pairs)

        self.assertEqual(metrics["total_nodes"], 9)
        self.assertEqual(metrics["connected_components"], 1)
        self.assertTrue(metrics["is_connected"])
        self.assertGreaterEqual(metrics["min_degree"], 1)


if __name__ == "__main__":
    unittest.main()


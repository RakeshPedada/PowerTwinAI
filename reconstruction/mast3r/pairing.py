"""
reconstruction/mast3r/pairing.py
--------------------------------
Multi-mode image pair selection engine for MASt3R-SfM.
Supports:
- Mode A: Official MASt3R retrieval (ASMK + make_pairs_fps)
- Mode B: Existing PowerTwinAI ML overlap model (VisualEmbeddingExtractor + graph ranking)
- Mode C: Hybrid (MASt3R retrieval + existing overlap model fusion)
"""

import os
import sys
import logging
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import networkx as nx
import torch

from .mast3r_model import (
    DEFAULT_RETRIEVAL_WEIGHTS,
    DEFAULT_RETRIEVAL_CODEBOOK,
    BASE_PROJECT_DIR
)

# Ensure ML overlap model can be imported
V3_DIR = os.path.join(BASE_PROJECT_DIR, "reconstruction", "v3_current")
if os.path.exists(V3_DIR) and V3_DIR not in sys.path:
    sys.path.insert(0, V3_DIR)

logger = logging.getLogger("mast3r_pairing")


def compute_official_mast3r_retrieval_sim(
    image_paths: List[str],
    retrieval_model_path: Optional[str] = None,
    backbone=None,
    device: str = "cpu",
    progress_callback=None
) -> Optional[np.ndarray]:
    """
    Execute official MASt3R retrieval (ASMK codebook + local descriptors)
    to compute an N x N pairwise similarity matrix.
    """
    retrieval_path = retrieval_model_path or DEFAULT_RETRIEVAL_WEIGHTS
    if not os.path.isfile(retrieval_path):
        logger.warning(f"Retrieval model checkpoint not found at: {retrieval_path}")
        return None

    codebook_path = os.path.splitext(retrieval_path)[0].replace("_trainingfree", "_codebook") + ".pkl"
    if not os.path.isfile(codebook_path):
        codebook_path = DEFAULT_RETRIEVAL_CODEBOOK

    if not os.path.isfile(codebook_path):
        logger.warning(f"ASMK retrieval codebook not found at: {codebook_path}")
        return None

    try:
        from mast3r.retrieval.processor import Retriever
        if progress_callback:
            progress_callback(f"[MASt3R-SfM] Initializing official ASMK Retriever ({len(image_paths)} images)...")

        retriever = Retriever(retrieval_path, backbone=backbone, device=device)
        with torch.no_grad():
            sim_matrix = retriever(image_paths)

        del retriever
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if isinstance(sim_matrix, torch.Tensor):
            sim_matrix = sim_matrix.cpu().numpy()

        return sim_matrix
    except Exception as e:
        logger.warning(f"Official MASt3R retrieval failed: {e}. Falling back to visual feature embeddings.")
        return None


def compute_existing_ml_overlap_sim(
    image_paths: List[str],
    progress_callback=None
) -> np.ndarray:
    """
    Execute existing PowerTwinAI ML visual embedding extractor to compute
    an N x N pairwise cosine similarity matrix.
    """
    try:
        from ml.feature_embeddings import VisualEmbeddingExtractor
        if progress_callback:
            progress_callback(f"[MASt3R-SfM] Computing existing PowerTwinAI ML overlap embeddings...")

        extractor = VisualEmbeddingExtractor()
        emb_dict = extractor.extract_batch(image_paths)
        del extractor
        import gc
        gc.collect()
        valid_paths = [p for p in image_paths if p in emb_dict]

        num_images = len(image_paths)
        sim_mat = np.eye(num_images, dtype=np.float32)

        vectors = []
        for p in image_paths:
            if p in emb_dict:
                v = emb_dict[p]
                norm = np.linalg.norm(v)
                vectors.append(v / norm if norm > 1e-6 else v)
            else:
                vectors.append(np.zeros(512, dtype=np.float32))

        vectors = np.array(vectors)
        sim_mat = np.clip(np.matmul(vectors, vectors.T), 0.0, 1.0)
        np.fill_diagonal(sim_mat, 1.0)
        return sim_mat
    except Exception as e:
        logger.warning(f"Failed to compute ML overlap similarity: {e}. Generating color histogram matrix.")
        # Fallback to normalized RGB histogram cosine similarity
        import cv2
        num_images = len(image_paths)
        hists = []
        for p in image_paths:
            im = cv2.imread(p)
            if im is not None:
                h = cv2.calcHist([im], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                h = h.flatten() / (h.sum() + 1e-6)
                hists.append(h)
            else:
                hists.append(np.zeros(512, dtype=np.float32))
        hists = np.array(hists)
        sim_mat = np.matmul(hists, hists.T)
        norms = np.linalg.norm(hists, axis=1, keepdims=True)
        sim_mat = sim_mat / (np.matmul(norms, norms.T) + 1e-6)
        np.fill_diagonal(sim_mat, 1.0)
        return sim_mat


def select_candidate_pairs(
    image_paths: List[str],
    pairing_mode: str = "Mode C (Hybrid)",
    num_key_images: int = 20,
    num_neighbors: int = 8,
    retrieval_model_path: Optional[str] = None,
    backbone=None,
    device: str = "cpu",
    progress_callback=None
) -> Dict[str, Any]:
    """
    Unified multi-mode pair selection adhering to official MASt3R-SfM:
    Mode A: MASt3R retrieval
    Mode B: Existing PowerTwinAI overlap model
    Mode C: MASt3R retrieval + existing overlap model (Hybrid)
    """
    num_images = len(image_paths)
    exhaustive_count = (num_images * (num_images - 1)) // 2

    if num_images < 2:
        return {
            "candidate_pairs": [],
            "sim_matrix": np.zeros((num_images, num_images)),
            "telemetry": {
                "total_images": num_images,
                "exhaustive_pairs": 0,
                "candidate_pairs": 0,
                "rejected_pairs": 0,
                "reduction_percentage": 0.0,
                "is_connected": True,
                "pairing_mode": pairing_mode
            }
        }

    from mast3r.retrieval.graph import make_pairs_fps

    sim_mast3r = None
    sim_overlap = None
    final_sim_mat = None

    effective_key_images = min(num_key_images, num_images)
    effective_neighbors = min(num_neighbors, num_images - 1)

    # 1. Evaluate Mode A (MASt3R retrieval)
    if "Mode A" in pairing_mode or "Mode C" in pairing_mode or "Retrieval" in pairing_mode:
        sim_mast3r = compute_official_mast3r_retrieval_sim(
            image_paths,
            retrieval_model_path=retrieval_model_path,
            backbone=backbone,
            device=device,
            progress_callback=progress_callback
        )

    # 2. Evaluate Mode B (Existing ML overlap model)
    if "Mode B" in pairing_mode or "Mode C" in pairing_mode or "Overlap" in pairing_mode or sim_mast3r is None:
        sim_overlap = compute_existing_ml_overlap_sim(
            image_paths,
            progress_callback=progress_callback
        )

    # 3. Fuse or select final similarity matrix
    if "Mode C" in pairing_mode or "Hybrid" in pairing_mode:
        if sim_mast3r is not None and sim_overlap is not None:
            final_sim_mat = 0.5 * sim_mast3r + 0.5 * sim_overlap
        elif sim_mast3r is not None:
            final_sim_mat = sim_mast3r
        else:
            final_sim_mat = sim_overlap
    elif "Mode A" in pairing_mode and sim_mast3r is not None:
        final_sim_mat = sim_mast3r
    else:
        final_sim_mat = sim_overlap

    np.fill_diagonal(final_sim_mat, 1.0)

    # 4. Use official make_pairs_fps
    fps_pairs, anchor_idxs = make_pairs_fps(
        final_sim_mat,
        Na=effective_key_images,
        tokK=effective_neighbors,
        dist_thresh=None
    )

    # Construct canonical candidate pairs (i < j)
    candidate_pairs_indices = set()
    for i, j in fps_pairs:
        if i != j:
            p = (min(i, j), max(i, j))
            candidate_pairs_indices.add(p)

    # Guarantee graph connectivity (MST backbone)
    dist_mat = 1.0 - final_sim_mat
    G_complete = nx.complete_graph(num_images)
    for u, v in G_complete.edges():
        G_complete[u][v]["weight"] = float(dist_mat[u, v])
    mst = nx.minimum_spanning_tree(G_complete)
    for u, v in mst.edges():
        candidate_pairs_indices.add((min(u, v), max(u, v)))

    # Also add adjacent chronological sequence pairs to support smooth trajectory
    for i in range(num_images - 1):
        candidate_pairs_indices.add((i, i + 1))

    # Convert indices to file path tuples
    candidate_pairs = [
        (image_paths[i], image_paths[j])
        for i, j in sorted(candidate_pairs_indices)
    ]

    candidate_count = len(candidate_pairs)
    rejected_count = max(0, exhaustive_count - candidate_count)
    reduction_pct = round((rejected_count / max(1, exhaustive_count)) * 100.0, 2)

    # Check graph connectivity
    G = nx.Graph()
    G.add_nodes_from(range(num_images))
    G.add_edges_from(list(candidate_pairs_indices))
    is_connected = nx.is_connected(G)

    telemetry = {
        "total_images": num_images,
        "exhaustive_pairs": exhaustive_count,
        "candidate_pairs": candidate_count,
        "rejected_pairs": rejected_count,
        "reduction_percentage": reduction_pct,
        "is_connected": is_connected,
        "pairing_mode": pairing_mode,
        "num_key_images": effective_key_images,
        "num_neighbors": effective_neighbors,
    }

    if progress_callback:
        progress_callback(
            f"[MASt3R-SfM] Pair Selection ({pairing_mode}): {candidate_count:,} candidate pairs "
            f"selected from {exhaustive_count:,} possible pairs ({reduction_pct}% reduction, Connected: {is_connected})."
        )

    return {
        "candidate_pairs": candidate_pairs,
        "pair_indices": list(candidate_pairs_indices),
        "sim_matrix": final_sim_mat,
        "telemetry": telemetry
    }

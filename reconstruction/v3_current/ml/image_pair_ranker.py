"""
ml/image_pair_ranker.py
-----------------------
Core ML candidate image pair selection and graph connectivity verification engine.
Replaces O(N^2) all-to-all matching with top-K visual similarity candidate pairs
while guaranteeing graph connectivity and multi-view reconstruction integrity.
"""

import os
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import networkx as nx

from ml.config import MLPairRankingConfig
from ml.feature_embeddings import VisualEmbeddingExtractor


class ImagePairRanker:
    """
    Ranks image pairs by visual feature similarity, constructs a symmetric candidate graph,
    enforces minimum degree and connectivity, and exports COLMAP-compatible pair lists.
    """

    def __init__(self, config: Optional[MLPairRankingConfig] = None):
        self.config = config or MLPairRankingConfig()
        self.embedding_extractor = VisualEmbeddingExtractor(
            method=self.config.embedding_method
        )

    def rank_pairs(
        self,
        image_paths: List[str],
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Extract embeddings, rank image pairs by visual similarity, enforce
        graph connectivity, and return candidate pairs and diagnostics.

        Parameters
        ----------
        image_paths : List[str]
            Full paths or filenames of the input images.
        progress_callback : callable, optional
            Function to receive status messages.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - "candidate_pairs": List[Tuple[str, str]] (unique canonical pairs)
            - "pair_scores": Dict[Tuple[str, str], float]
            - "diagnostics": Dict of graph connectivity, degree, and reduction stats
            - "is_connected": bool
        """
        import time
        t_start = time.perf_counter()
        total_images = len(image_paths)
        exhaustive_count = (total_images * (total_images - 1)) // 2

        if total_images < 2:
            return {
                "candidate_pairs": [],
                "pair_scores": {},
                "diagnostics": {
                    "total_images": total_images,
                    "exhaustive_pairs": 0,
                    "candidate_pairs": 0,
                    "pair_reduction_pct": 0.0,
                    "connected_components": 0,
                    "is_connected": True,
                    "min_degree": 0,
                    "avg_degree": 0.0,
                },
                "is_connected": True
            }

        # Canonical basenames used by COLMAP
        image_names = [os.path.basename(p) for p in image_paths]

        if progress_callback:
            progress_callback(f"[ML-RANKER] Extracting visual embeddings for {total_images} images...")

        embeddings = self.embedding_extractor.extract_batch(
            image_paths,
            progress_callback=progress_callback
        )

        # ── 1. Cosine Similarity Matrix ──────────────────────────────────────────
        # Since embeddings are L2-normalized, matrix dot product gives cosine similarity
        sim_matrix = np.dot(embeddings, embeddings.T)
        np.clip(sim_matrix, -1.0, 1.0, out=sim_matrix)

        # ── 2. Top-K Candidate Graph Construction ────────────────────────────────
        G = nx.Graph()
        for name in image_names:
            G.add_node(name)

        pair_scores: Dict[Tuple[str, str], float] = {}

        top_k = max(1, min(self.config.top_k, total_images - 1))
        min_pairs = max(1, min(self.config.min_pairs_per_image, total_images - 1))
        thresh = self.config.similarity_threshold

        for i in range(total_images):
            src_name = image_names[i]
            scores = sim_matrix[i].copy()
            scores[i] = -999.0  # mask self-similarity

            sorted_indices = np.argsort(-scores)

            # Filter by threshold, but guarantee at least min_pairs per image
            chosen_indices = []
            for idx in sorted_indices:
                score = float(scores[idx])
                if len(chosen_indices) < min_pairs:
                    chosen_indices.append((idx, score))
                elif len(chosen_indices) < top_k and score >= thresh:
                    chosen_indices.append((idx, score))
                elif len(chosen_indices) >= top_k:
                    break

            for dst_idx, score in chosen_indices:
                dst_name = image_names[dst_idx]
                if src_name != dst_name:
                    canonical_pair = tuple(sorted([src_name, dst_name]))
                    G.add_edge(canonical_pair[0], canonical_pair[1], weight=score)
                    pair_scores[canonical_pair] = max(pair_scores.get(canonical_pair, -1.0), score)

        # ── 3. Graph Connectivity Validation & Component Bridging (Stage 5) ─────
        num_components = nx.number_connected_components(G)

        if num_components > 1 and self.config.auto_bridge_components:
            if progress_callback:
                progress_callback(
                    f"[ML-RANKER] Initial candidate graph has {num_components} components. "
                    "Bridging disconnected components via highest similarity inter-cluster edges..."
                )

            components = list(nx.connected_components(G))
            # Sort components by size descending — component 0 is the main giant component
            components.sort(key=len, reverse=True)
            main_comp = set(components[0])

            name_to_idx = {name: idx for idx, name in enumerate(image_names)}

            for other_comp in components[1:]:
                best_score = -999.0
                best_pair = None

                for u_name in other_comp:
                    u_idx = name_to_idx[u_name]
                    for v_name in main_comp:
                        v_idx = name_to_idx[v_name]
                        score = float(sim_matrix[u_idx, v_idx])
                        if score > best_score:
                            best_score = score
                            best_pair = (u_name, v_name)

                if best_pair:
                    canon = tuple(sorted(best_pair))
                    G.add_edge(canon[0], canon[1], weight=best_score)
                    pair_scores[canon] = best_score
                    main_comp.update(other_comp)

            num_components = nx.number_connected_components(G)

        # ── 4. Max Total Pairs Trimming (if requested) ───────────────────────────
        if self.config.max_total_pairs and G.number_of_edges() > self.config.max_total_pairs:
            # Sort edges by weight descending, but ensure spanning tree connectivity is preserved
            mst = nx.maximum_spanning_tree(G, weight="weight")
            non_mst_edges = [
                (u, v, data["weight"])
                for u, v, data in G.edges(data=True)
                if not mst.has_edge(u, v)
            ]
            non_mst_edges.sort(key=lambda x: x[2], reverse=True)

            budget_remaining = max(0, self.config.max_total_pairs - mst.number_of_edges())
            keep_edges = set(mst.edges())
            for u, v, _ in non_mst_edges[:budget_remaining]:
                keep_edges.add((u, v))

            # Reconstruct G with kept edges
            G_trimmed = nx.Graph()
            for n in image_names:
                G_trimmed.add_node(n)
            for u, v in keep_edges:
                G_trimmed.add_edge(u, v, weight=pair_scores[tuple(sorted([u, v]))])
            G = G_trimmed

        candidate_pairs = [tuple(sorted([u, v])) for u, v in G.edges()]
        # Sort canonically for determinism
        candidate_pairs.sort()

        # ── 5. Telemetry & Metrics ───────────────────────────────────────────────
        degrees = [d for _, d in G.degree()]
        min_deg = min(degrees) if degrees else 0
        max_deg = max(degrees) if degrees else 0
        avg_deg = float(np.mean(degrees)) if degrees else 0.0
        final_components = nx.number_connected_components(G)
        reduction_pct = round((1.0 - (len(candidate_pairs) / max(exhaustive_count, 1))) * 100.0, 2)

        elapsed_sec = round(float(time.perf_counter() - t_start), 3)

        diagnostics = {
            "total_images": total_images,
            "exhaustive_pairs": exhaustive_count,
            "exhaustive_pairs_baseline": exhaustive_count,
            "candidate_pairs": len(candidate_pairs),
            "candidate_pairs_selected": len(candidate_pairs),
            "pair_reduction_pct": reduction_pct,
            "pairs_reduction_percent": reduction_pct,
            "connected_components": final_components,
            "is_connected": (final_components == 1),
            "min_degree": min_deg,
            "max_degree": max_deg,
            "avg_degree": round(avg_deg, 2),
            "mean_degree": round(avg_deg, 2),
            "top_k": top_k,
            "min_pairs_per_image": min_pairs,
            "similarity_threshold": thresh,
            "embedding_method": self.config.embedding_method,
            "computation_time_seconds": elapsed_sec,
        }

        if progress_callback:
            progress_callback(
                f"[ML-RANKER] Selected {len(candidate_pairs):,} candidate pairs "
                f"from {exhaustive_count:,} exhaustive pairs ({reduction_pct}% reduction). "
                f"Min degree: {min_deg}, Avg degree: {avg_deg:.1f}, Components: {final_components}."
            )

        return {
            "candidate_pairs": candidate_pairs,
            "pair_scores": pair_scores,
            "diagnostics": diagnostics,
            "is_connected": (final_components == 1)
        }

    def rank_images(
        self,
        image_paths: List[str],
        progress_callback=None
    ) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
        """
        Convenience wrapper returning (candidate_pairs, diagnostics_dict).
        """
        result = self.rank_pairs(image_paths, progress_callback)
        # Add baseline metric alias if needed
        diag = result["diagnostics"]
        if "exhaustive_pairs_baseline" not in diag:
            diag["exhaustive_pairs_baseline"] = diag.get("exhaustive_pairs", 0)
        if "candidate_pairs_selected" not in diag:
            diag["candidate_pairs_selected"] = diag.get("candidate_pairs", 0)
        return result["candidate_pairs"], diag

    def save_colmap_pairs(
        self,
        candidate_pairs: List[Tuple[str, str]],
        output_file_path: str
    ) -> str:
        """
        Export candidate pairs to a COLMAP-compatible match list text file.
        """
        return save_colmap_pairs(candidate_pairs, output_file_path)


def compute_graph_metrics(
    image_names: List[str],
    pairs: List[Tuple[str, str]]
) -> Dict[str, Any]:
    """
    Compute graph topological properties including degrees, connectivity, and components.
    """
    G = nx.Graph()
    for name in image_names:
        G.add_node(name)
    for u, v in pairs:
        G.add_edge(u, v)

    degrees = [d for _, d in G.degree()]
    comps = nx.number_connected_components(G) if len(image_names) > 0 else 0
    return {
        "total_nodes": len(image_names),
        "total_edges": len(pairs),
        "connected_components": comps,
        "is_connected": (comps == 1) if len(image_names) > 0 else True,
        "min_degree": min(degrees) if degrees else 0,
        "max_degree": max(degrees) if degrees else 0,
        "avg_degree": round(float(np.mean(degrees)), 2) if degrees else 0.0,
    }


def save_colmap_pairs(
    candidate_pairs: List[Tuple[str, str]],
    output_file_path: str
) -> str:
    """
    Export candidate pairs to a COLMAP-compatible match list text file.
    Each line contains two space-separated image names.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_file_path)), exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as f:
        for img1, img2 in candidate_pairs:
            f.write(f"{img1} {img2}\n")

    return output_file_path


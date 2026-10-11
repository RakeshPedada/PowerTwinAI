"""
core/coverage_analyzer.py
-------------------------
Reconstruction coverage and completeness analysis.
Evaluates nearest-neighbor spatial distribution, voxel occupancy,
and identifies low-density or gap regions without fabricating points.
"""

import numpy as np
import open3d as o3d


def analyze_point_cloud_coverage(points, min_expected_density=None) -> dict:
    """
    Analyze the spatial completeness and coverage of a reconstructed point cloud.

    Parameters
    ----------
    points : np.ndarray | o3d.geometry.PointCloud
        Point coordinates (N, 3).
    min_expected_density : float | None
        Optional density baseline.

    Returns
    -------
    dict
        {
            "total_points": int,
            "mean_nn_dist": float,
            "median_nn_dist": float,
            "max_nn_dist": float,
            "coverage_score": float,  # 0.0 to 100.0%
            "has_coverage_warning": bool,
            "warning_message": str | None,
            "sparse_regions_ratio": float,
            "voxel_occupancy_ratio": float
        }
    """
    if isinstance(points, o3d.geometry.PointCloud):
        pcd = points
        pts = np.asarray(pcd.points)
    else:
        pts = np.asarray(points, dtype=np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

    if len(pts) < 10:
        return {
            "total_points": len(pts),
            "mean_nn_dist": 0.0,
            "median_nn_dist": 0.0,
            "max_nn_dist": 0.0,
            "coverage_score": 0.0,
            "has_coverage_warning": True,
            "warning_message": "Critical: Insufficient points to evaluate reconstruction coverage.",
            "sparse_regions_ratio": 1.0,
            "voxel_occupancy_ratio": 0.0
        }

    # 1. Compute Nearest Neighbor Distances
    nn_dists = np.asarray(pcd.compute_nearest_neighbor_distance())
    mean_dist = float(np.mean(nn_dists)) if len(nn_dists) > 0 else 0.0
    median_dist = float(np.median(nn_dists)) if len(nn_dists) > 0 else 0.0
    p95_dist = float(np.percentile(nn_dists, 95)) if len(nn_dists) > 0 else 0.0

    # Points with NN distance > 3x median distance are classified as gap / low-density edge points
    gap_threshold = median_dist * 3.0
    sparse_points_count = int(np.sum(nn_dists > gap_threshold))
    sparse_ratio = float(sparse_points_count / len(pts))

    # 2. Voxel Occupancy Coverage across bounding box
    bbox_extent = np.ptp(pts, axis=0)
    extent_mean = float(np.mean(bbox_extent))
    voxel_size = max(extent_mean / 30.0, 1e-4)

    min_coords = np.min(pts, axis=0)
    voxel_indices = np.floor((pts - min_coords) / voxel_size).astype(np.int32)
    unique_voxels = len(np.unique(voxel_indices, axis=0))

    grid_dims = np.ceil(bbox_extent / voxel_size).astype(np.int32)
    total_bounding_voxels = max(int(np.prod(grid_dims)), 1)
    voxel_occupancy = float(unique_voxels / min(total_bounding_voxels, 5000))

    # Coverage score: weighted combination of uniform density (low sparse ratio) and point quantity
    # Normalized 0 - 100%
    uniformity_score = max(0.0, 1.0 - sparse_ratio) * 60.0
    density_factor = min(1.0, len(pts) / 30000.0) * 40.0
    coverage_score = round(min(100.0, max(10.0, uniformity_score + density_factor)), 1)

    has_warning = sparse_ratio > 0.15 or len(pts) < 15000 or coverage_score < 65.0
    warning_msg = None
    if has_warning:
        warning_msg = (
            "Low coverage detected in some regions. "
            "Additional viewpoints and higher overlap may improve surface reconstruction completeness."
        )

    return {
        "total_points": len(pts),
        "mean_nn_dist": round(mean_dist, 5),
        "median_nn_dist": round(median_dist, 5),
        "max_nn_dist": round(p95_dist, 5),
        "coverage_score": coverage_score,
        "has_coverage_warning": has_warning,
        "warning_message": warning_msg,
        "sparse_regions_ratio": round(sparse_ratio * 100.0, 1),
        "voxel_occupancy_ratio": round(voxel_occupancy * 100.0, 1)
    }
"""
core/reconstruction_diagnostics.py
-----------------------------------
Comprehensive 20-metric quantitative diagnostic engine for PowerTwinAI.
Evaluates the entire pipeline from 2D image observations to final 3D surface mesh.
"""

import os
import json
import sqlite3
import numpy as np
import open3d as o3d
from collections import defaultdict
from typing import Dict, Any, Optional

from backends.ml_geometry_analyzer import MLGeometryAnalyzer


def generate_diagnostics_report(
    colmap_dir: str = "colmap_data",
    colmap_workspace: str = "colmap_workspace",
    output_dir: str = "output",
    mesh_path: Optional[str] = None,
    pointcloud_path: Optional[str] = None,
    total_input_images: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate the complete 20-metric diagnostic report covering all stages
    of the 3D reconstruction pipeline.
    """
    mesh_file = mesh_path or os.path.join(output_dir, "mesh.ply")
    pcd_file = pointcloud_path or os.path.join(output_dir, "reconstruction.ply")

    # ── 1-5. Camera & Image Telemetry ─────────────────────────────────────────
    cameras_file = os.path.join(colmap_dir, "cameras.txt")
    images_file = os.path.join(colmap_dir, "images.txt")

    cam_model = "UNKNOWN"
    img_w, img_h = 0, 0
    if os.path.exists(cameras_file):
        try:
            with open(cameras_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        parts = line.split()
                        cam_model = parts[1]
                        img_w = int(parts[2])
                        img_h = int(parts[3])
                        break
        except Exception:
            pass

    registered_cams = 0
    if os.path.exists(images_file):
        try:
            with open(images_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        parts = line.split()
                        if len(parts) >= 10 and any(parts[9].lower().endswith(ext) for ext in [".jpg", ".png", ".jpeg", ".webp", ".bmp"]):
                            registered_cams += 1
        except Exception:
            pass

    input_count = total_input_images if total_input_images and total_input_images > 0 else max(registered_cams, 1)
    reg_ratio = min(100.0, round((registered_cams / input_count) * 100.0, 2))

    # ── 6-8. SIFT Feature Extraction & Pair Inliers ───────────────────────────
    db_file = os.path.join(colmap_workspace, "database.db")
    total_features = 0
    mean_features = 0.0
    verified_inliers = 0

    if os.path.exists(db_file):
        try:
            conn = sqlite3.connect(db_file)
            cur = conn.cursor()
            cur.execute("SELECT SUM(rows), AVG(rows) FROM keypoints")
            res = cur.fetchone()
            if res and res[0] is not None:
                total_features = int(res[0])
                mean_features = round(float(res[1]), 1)

            # Check two_view_geometries or matches
            cur.execute("SELECT SUM(rows) FROM two_view_geometries WHERE rows > 0")
            res_geom = cur.fetchone()
            if res_geom and res_geom[0] is not None:
                verified_inliers = int(res_geom[0])
            else:
                cur.execute("SELECT SUM(rows) FROM matches WHERE rows > 0")
                res_m = cur.fetchone()
                if res_m and res_m[0] is not None:
                    verified_inliers = int(res_m[0])
            conn.close()
        except Exception:
            pass

    # If pair_logs.csv exists, fallback or cross-reference inliers
    pair_logs_file = os.path.join(output_dir, "pair_logs.csv")
    if verified_inliers == 0 and os.path.exists(pair_logs_file):
        try:
            with open(pair_logs_file, "r", encoding="utf-8") as f:
                lines = f.readlines()[1:]
                for line in lines:
                    parts = line.strip().split(",")
                    if len(parts) >= 4 and parts[3].isdigit():
                        verified_inliers += int(parts[3])
        except Exception:
            pass

    # ── 9-12. 3D Points, Track Lengths & Reprojection Errors ─────────────────
    points3d_file = os.path.join(colmap_dir, "points3D.txt")
    raw_points_count = 0
    mean_track_length = 0.0
    median_track_length = 0.0
    mean_reproj_error = 0.0
    raw_pts_arr = []

    if os.path.exists(points3d_file):
        try:
            errors = []
            tracks = []
            with open(points3d_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 8:
                        xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
                        err = float(parts[7])
                        obs = (len(parts) - 8) // 2
                        raw_pts_arr.append(xyz)
                        errors.append(err)
                        tracks.append(obs)

            if errors:
                raw_points_count = len(errors)
                mean_reproj_error = round(float(np.mean(errors)), 3)
                mean_track_length = round(float(np.mean(tracks)), 2)
                median_track_length = round(float(np.median(tracks)), 1)
        except Exception:
            pass

    # ── 13-16. Cleaned Point Cloud & Density ─────────────────────────────────
    cleaned_points_count = 0
    mean_nn_dist = 0.0
    median_nn_dist = 0.0
    bb_min, bb_max, bb_ext = [0.0]*3, [0.0]*3, [0.0]*3
    bb_diag = 0.0

    if os.path.exists(pcd_file):
        try:
            pcd = o3d.io.read_point_cloud(pcd_file)
            pts = np.asarray(pcd.points)
            cleaned_points_count = len(pts)
            if cleaned_points_count > 0:
                bb_min = np.min(pts, axis=0).tolist()
                bb_max = np.max(pts, axis=0).tolist()
                bb_ext = (np.max(pts, axis=0) - np.min(pts, axis=0)).tolist()
                bb_diag = float(np.linalg.norm(np.array(bb_ext)))

                # Nearest-neighbor distribution
                dists = pcd.compute_nearest_neighbor_distance()
                if len(dists) > 0:
                    mean_nn_dist = round(float(np.mean(dists)), 5)
                    median_nn_dist = round(float(np.median(dists)), 5)
        except Exception:
            pass

    retained_ratio = round((cleaned_points_count / max(raw_points_count, 1)) * 100.0, 2)

    # ── 17-20. Mesh Topology, Holes & ML Gap Classification ──────────────────
    mesh_verts = 0
    mesh_tris = 0
    boundary_edges_count = 0
    connected_components = 0
    is_watertight = False
    gaps_summary = []

    if os.path.exists(mesh_file):
        try:
            mesh = o3d.io.read_triangle_mesh(mesh_file)
            verts = np.asarray(mesh.vertices)
            tris = np.asarray(mesh.triangles)
            mesh_verts = len(verts)
            mesh_tris = len(tris)
            is_watertight = bool(mesh.is_watertight())

            if mesh_tris > 0:
                # Count boundary edges
                edge_counts = defaultdict(int)
                for tri in tris:
                    for j in range(3):
                        v0 = tri[j]
                        v1 = tri[(j + 1) % 3]
                        edge = tuple(sorted((v0, v1)))
                        edge_counts[edge] += 1
                boundary_edges_count = sum(1 for c in edge_counts.values() if c == 1)

                # Connected components
                try:
                    _, num_tris, _ = mesh.cluster_connected_triangles()
                    connected_components = len(num_tris)
                except Exception:
                    connected_components = 1

                # ML gap classification
                analyzer = MLGeometryAnalyzer(colmap_data_dir=colmap_dir)
                gaps_summary = analyzer.analyze_mesh_gaps(mesh, max_analyze_holes=15)
        except Exception as e:
            print(f"[DIAGNOSTICS] Mesh inspection note: {e}")

    total_hole_area = sum(g["area"] for g in gaps_summary)

    # ── Assemble 20 Metrics ──────────────────────────────────────────────────
    report = {
        # 1. Input Images
        "total_input_images": input_count,
        # 2. Registered Cameras
        "registered_cameras": registered_cams,
        # 3. Registration Ratio
        "registration_ratio": reg_ratio,
        # 4. Camera Model
        "camera_model": cam_model,
        # 5. Image Resolution
        "image_resolution": {
            "width": img_w,
            "height": img_h,
            "megapixels": round((img_w * img_h) / 1e6, 2)
        },
        # 6. SIFT Features
        "total_sift_features": total_features,
        # 7. Mean Features Per Image
        "mean_features_per_image": mean_features,
        # 8. Verified Pair Inliers
        "verified_pair_matches": verified_inliers,
        # 9. Mean Track Length
        "mean_track_length": mean_track_length,
        # 10. Median Track Length
        "median_track_length": median_track_length,
        # 11. Raw 3D Triangulated Points
        "raw_colmap_points": raw_points_count,
        # 12. Mean Reprojection Error (px)
        "mean_reprojection_error_px": mean_reproj_error,
        # 13. Cleaned Points Count
        "cleaned_points": cleaned_points_count,
        # 14. Retained Points Ratio (%)
        "retained_points_ratio": retained_ratio,
        # 15. Point Cloud Density (NN dist)
        "point_cloud_density": {
            "mean_nn_distance": mean_nn_dist,
            "median_nn_distance": median_nn_dist
        },
        # 16. Bounding Box Extents & Diagonal
        "bounding_box": {
            "min": [round(x, 4) for x in bb_min],
            "max": [round(x, 4) for x in bb_max],
            "extent": [round(x, 4) for x in bb_ext],
            "diagonal": round(bb_diag, 4)
        },
        # 17. Mesh Vertices Count
        "mesh_vertices": mesh_verts,
        # 18. Mesh Faces Count
        "mesh_faces": mesh_tris,
        # 19. Open Boundary Edges Count
        "boundary_edges": boundary_edges_count,
        # 20. Detected Holes & ML Visibility Classification
        "num_detected_holes": len(gaps_summary),
        "total_hole_surface_area": round(total_hole_area, 5),
        "is_watertight": is_watertight,
        "connected_components": connected_components,
        "detected_holes_summary": gaps_summary
    }

    # Save to output/reconstruction_diagnostics.json and temp_session/diagnostics.json
    for out_p in [
        os.path.join(output_dir, "reconstruction_diagnostics.json"),
        os.path.join("temp_session", "diagnostics.json")
    ]:
        try:
            os.makedirs(os.path.dirname(out_p), exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        except Exception:
            pass

    return report

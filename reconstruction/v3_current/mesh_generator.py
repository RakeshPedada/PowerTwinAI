import os
import numpy as np
import open3d as o3d


def generate_mesh(
    points,
    colors,
    output_dir="output"
):
    """
    Experimental Poisson mesh generation.

    Uses the supplied COLMAP dense point cloud and applies
    stronger density filtering to reduce unsupported Poisson
    surface regions.
    """

    print("[MESH] Creating Point Cloud...")

    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float64)

    if len(points) == 0:
        raise RuntimeError("Point cloud is empty.")

    if len(points) != len(colors):
        raise ValueError(
            "Points and colors must contain the same number of elements."
        )

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print("[MESH] Estimating Normals...")

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.05,
            max_nn=30
        )
    )

    pcd.normalize_normals()

    print("[MESH] Running Poisson Reconstruction...")

    mesh, densities = (
        o3d.geometry.TriangleMesh
        .create_from_point_cloud_poisson(
            pcd,
            depth=7
        )
    )

    densities = np.asarray(densities)

    print(
        f"[MESH] Raw Poisson vertices: "
        f"{len(mesh.vertices):,}"
    )

    print(
        f"[MESH] Raw Poisson triangles: "
        f"{len(mesh.triangles):,}"
    )

    # ------------------------------------------------------
    # STRONGER DENSITY FILTER
    # ------------------------------------------------------

    density_percentile = 10.0

    density_threshold = np.percentile(
        densities,
        density_percentile
    )

    print(
        f"[MESH] Density threshold "
        f"({density_percentile:.1f} percentile): "
        f"{density_threshold:.6f}"
    )

    vertices_to_remove = (
        densities < density_threshold
    )

    removed_vertices = int(
        np.count_nonzero(vertices_to_remove)
    )

    print(
        f"[MESH] Removing low-density vertices: "
        f"{removed_vertices:,}"
    )

    mesh.remove_vertices_by_mask(
        vertices_to_remove
    )

    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    # ------------------------------------------------------
    # SAVE
    # ------------------------------------------------------

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    ply_mesh = os.path.join(
        output_dir,
        "mesh.ply"
    )

    obj_mesh = os.path.join(
        output_dir,
        "mesh.obj"
    )

    stl_mesh = os.path.join(
        output_dir,
        "mesh.stl"
    )

    print("[MESH] Saving Mesh Files...")

    o3d.io.write_triangle_mesh(
        ply_mesh,
        mesh
    )

    o3d.io.write_triangle_mesh(
        obj_mesh,
        mesh
    )

    o3d.io.write_triangle_mesh(
        stl_mesh,
        mesh
    )

    print(
        f"[MESH] Final vertices: "
        f"{len(mesh.vertices):,}"
    )

    print(
        f"[MESH] Final triangles: "
        f"{len(mesh.triangles):,}"
    )

    return {
        "mesh": mesh,
        "ply": ply_mesh,
        "obj": obj_mesh,
        "stl": stl_mesh,
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles)
    }
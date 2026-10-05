import os
import numpy as np
import open3d as o3d


def generate_mesh(
    points,
    colors,
    output_dir="output"
):
    """
    Generate mesh from point cloud using
    Poisson Surface Reconstruction.
    """

    print("[MESH] Creating Point Cloud...")

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(
        points
    )

    pcd.colors = o3d.utility.Vector3dVector(
        colors
    )

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

    densities = np.asarray(
        densities
    )

    density_threshold = np.quantile(
        densities,
        0.02
    )

    vertices_to_remove = (
        densities < density_threshold
    )

    mesh.remove_vertices_by_mask(
        vertices_to_remove
    )

    mesh.compute_vertex_normals()

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
        f"[MESH] Vertices: "
        f"{len(mesh.vertices):,}"
    )

    print(
        f"[MESH] Triangles: "
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
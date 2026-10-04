import os
import numpy as np
import open3d as o3d

def generate_mesh(
    points,
    colors,
    camera_positions=None,
    output_dir="output"
):
    print("[MESH] Creating Point Cloud...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print("[MESH] Estimating Normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1,
            max_nn=50
        )
    )

    print("[MESH] Orienting Normals Consistently...")
    # Mathematically align normals to form a contiguous surface
    pcd.orient_normals_consistent_tangent_plane(k=50)

    # Use actual camera tracking data to ensure normals point OUTWARDS
    if camera_positions is not None and len(camera_positions) > 0:
        mean_camera = np.mean(camera_positions, axis=0)
        pcd.orient_normals_towards_camera_location(mean_camera)

    pcd.normalize_normals()

    print("[MESH] Running High-Res Poisson Reconstruction (Depth=9)...")
    # Depth 9 yields exponentially higher resolution than depth 7
    mesh, densities = (
        o3d.geometry.TriangleMesh
        .create_from_point_cloud_poisson(
            pcd,
            depth=9
        )
    )

    # Aggressive Density Clipping (Removes floating bubbles/artifacts)
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.05)
    vertices_to_remove = (densities < density_threshold)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    print("[MESH] Applying Laplacian Smoothing...")
    # Smooths noise spikes without destroying geometry edges
    mesh = mesh.filter_smooth_laplacian(number_of_iterations=10)
    mesh.compute_vertex_normals()

    print("[MESH] Projecting High-Res Colors via KD-Tree...")
    # Poisson meshing washes out vertex colors. We rebuild them by 
    # snapping to the absolute nearest original 3D point via a KD-Tree.
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    mesh_vertices = np.asarray(mesh.vertices)
    mesh_colors = np.zeros_like(mesh_vertices)
    
    original_colors = np.asarray(pcd.colors)
    
    for i, vertex in enumerate(mesh_vertices):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(vertex, 1)
        mesh_colors[i] = original_colors[idx[0]]
        
    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)

    os.makedirs(output_dir, exist_ok=True)
    ply_mesh = os.path.join(output_dir, "mesh.ply")
    obj_mesh = os.path.join(output_dir, "mesh.obj")
    stl_mesh = os.path.join(output_dir, "mesh.stl")

    print("[MESH] Saving High-Quality Mesh Files...")
    o3d.io.write_triangle_mesh(ply_mesh, mesh)
    o3d.io.write_triangle_mesh(obj_mesh, mesh)
    o3d.io.write_triangle_mesh(stl_mesh, mesh)

    print(f"[MESH] Vertices: {len(mesh.vertices):,}")
    print(f"[MESH] Triangles: {len(mesh.triangles):,}")

    return {
        "mesh": mesh,
        "ply": ply_mesh,
        "obj": obj_mesh,
        "stl": stl_mesh,
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles)
    }

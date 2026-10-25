"""
reconstruction/mast3r/visualization.py
--------------------------------------
Interactive 3D visualization utilities for MASt3R-SfM point clouds,
camera trajectories with viewing frustums and optical axes,
and multi-backend comparison (COLMAP vs. VGGT vs. MASt3R-SfM).
"""

import os
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import plotly.graph_objects as go


def create_mast3r_point_cloud_figure(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    title: str = "🌐 MASt3R-SfM 3D Point Cloud",
    max_display_points: int = 50000,
    color_mode: str = "rgb",
    center_scene: bool = True
) -> go.Figure:
    """
    Generate an interactive Plotly 3D scatter plot for the MASt3R-SfM point cloud.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        fig = go.Figure()
        fig.update_layout(title="No points available")
        return fig

    # Subsample if exceeds display budget
    if len(pts) > max_display_points:
        step = len(pts) // max_display_points
        pts = pts[::step]
        if colors is not None:
            colors = np.asarray(colors)[::step]

    # Center scene if requested (translation only, NO scaling distortion)
    if center_scene:
        scene_center = np.mean(pts, axis=0)
        display_pts = pts - scene_center
    else:
        display_pts = pts

    if colors is not None and color_mode == "rgb":
        cols = np.asarray(colors)
        marker_color = [f"rgb({int(c[0])},{int(c[1])},{int(c[2])})" for c in cols]
        colorscale = None
    else:
        marker_color = display_pts[:, 2]
        colorscale = "Viridis"

    scatter = go.Scatter3d(
        x=display_pts[:, 0],
        y=display_pts[:, 1],
        z=display_pts[:, 2],
        mode="markers",
        marker=dict(
            size=2,
            color=marker_color,
            colorscale=colorscale,
            opacity=0.85
        ),
        name="MASt3R Points",
        hoverinfo="skip"
    )

    fig = go.Figure(data=[scatter])
    fig.update_layout(
        title=dict(text=title, font=dict(color="#ffffff", size=15)),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        scene=dict(
            xaxis=dict(title="X (m)", color="#94a3b8", gridcolor="#1e293b", showbackground=False),
            yaxis=dict(title="Y (m)", color="#94a3b8", gridcolor="#1e293b", showbackground=False),
            zaxis=dict(title="Z (m)", color="#94a3b8", gridcolor="#1e293b", showbackground=False),
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig


def create_mast3r_camera_pointcloud_combined_figure(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    cameras_list: List[Dict[str, Any]],
    title: str = "📷 MASt3R-SfM Cameras & 3D Reconstruction",
    max_display_points: int = 40000,
    frustum_scale: float = 0.15
) -> go.Figure:
    """
    Render 3D point cloud along with camera positions, continuous trajectory line,
    and 3D viewing frustums in a shared, un-distorted world frame.
    """
    fig = go.Figure()

    pts = np.asarray(points, dtype=np.float64)
    if len(pts) > 0:
        if len(pts) > max_display_points:
            step = len(pts) // max_display_points
            pts = pts[::step]
            if colors is not None:
                colors = np.asarray(colors)[::step]

        if colors is not None:
            cols = np.asarray(colors)
            marker_color = [f"rgb({int(c[0])},{int(c[1])},{int(c[2])})" for c in cols]
        else:
            marker_color = pts[:, 2]

        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(size=2, color=marker_color, opacity=0.8),
            name="MASt3R Surface",
            hoverinfo="skip"
        ))

    # Add Camera Centers
    if cameras_list:
        centers = np.array([c["center"] for c in cameras_list])
        cam_names = [c.get("image_name", f"Cam {i+1}") for i, c in enumerate(cameras_list)]

        # Trajectory Path
        fig.add_trace(go.Scatter3d(
            x=centers[:, 0], y=centers[:, 1], z=centers[:, 2],
            mode="lines+markers",
            line=dict(color="#3b82f6", width=4),
            marker=dict(size=5, color="#60a5fa", symbol="circle"),
            name="Camera Trajectory",
            text=cam_names,
            hoverinfo="text"
        ))

        # First camera highlight (start of orbit)
        fig.add_trace(go.Scatter3d(
            x=[centers[0, 0]], y=[centers[0, 1]], z=[centers[0, 2]],
            mode="markers",
            marker=dict(size=8, color="#22c55e", symbol="diamond"),
            name="Origin Camera",
            hoverinfo="text",
            text=[f"Origin: {cam_names[0]}"]
        ))

        # Add 3D Frustums
        for i, c in enumerate(cameras_list):
            c2w = np.array(c["cam2world"])
            R = c2w[:3, :3]
            C = c2w[:3, 3]

            # Frustum corners in camera frame (+Z forward)
            w, h = 0.5 * frustum_scale, 0.35 * frustum_scale
            d = frustum_scale
            corners_cam = np.array([
                [-w, -h, d],
                [w, -h, d],
                [w, h, d],
                [-w, h, d]
            ])
            corners_world = (R @ corners_cam.T).T + C

            # Frustum outline lines
            fx = [C[0], corners_world[0, 0], corners_world[1, 0], C[0],
                  corners_world[2, 0], corners_world[3, 0], C[0],
                  corners_world[0, 0], corners_world[3, 0], corners_world[2, 0], corners_world[1, 0]]
            fy = [C[1], corners_world[0, 1], corners_world[1, 1], C[1],
                  corners_world[2, 1], corners_world[3, 1], C[1],
                  corners_world[0, 1], corners_world[3, 1], corners_world[2, 1], corners_world[1, 1]]
            fz = [C[2], corners_world[0, 2], corners_world[1, 2], C[2],
                  corners_world[2, 2], corners_world[3, 2], C[2],
                  corners_world[0, 2], corners_world[3, 2], corners_world[2, 2], corners_world[1, 2]]

            fig.add_trace(go.Scatter3d(
                x=fx, y=fy, z=fz,
                mode="lines",
                line=dict(color="#f59e0b", width=1.5),
                showlegend=(i == 0),
                name="Viewing Frustums",
                hoverinfo="skip"
            ))

    fig.update_layout(
        title=dict(text=title, font=dict(color="#ffffff", size=15)),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        scene=dict(
            xaxis=dict(title="X (m)", color="#94a3b8", gridcolor="#1e293b", showbackground=False),
            yaxis=dict(title="Y (m)", color="#94a3b8", gridcolor="#1e293b", showbackground=False),
            zaxis=dict(title="Z (m)", color="#94a3b8", gridcolor="#1e293b", showbackground=False),
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig


def create_three_way_comparison_figure(
    colmap_points: Optional[np.ndarray],
    vggt_points: Optional[np.ndarray],
    mast3r_points: Optional[np.ndarray],
    max_pts_per_model: int = 25000
) -> go.Figure:
    """
    Render comparative point clouds from COLMAP, VGGT, and MASt3R-SfM.
    """
    fig = go.Figure()

    # 1. COLMAP Trace
    if colmap_points is not None and len(colmap_points) > 0:
        c_pts = np.asarray(colmap_points)
        if len(c_pts) > max_pts_per_model:
            c_pts = c_pts[::(len(c_pts) // max_pts_per_model)]
        fig.add_trace(go.Scatter3d(
            x=c_pts[:, 0], y=c_pts[:, 1], z=c_pts[:, 2],
            mode="markers",
            marker=dict(size=2, color="#3b82f6", opacity=0.7),
            name="COLMAP (Baseline)",
            hoverinfo="skip"
        ))

    # 2. VGGT Trace
    if vggt_points is not None and len(vggt_points) > 0:
        v_pts = np.asarray(vggt_points)
        if len(v_pts) > max_pts_per_model:
            v_pts = v_pts[::(len(v_pts) // max_pts_per_model)]
        fig.add_trace(go.Scatter3d(
            x=v_pts[:, 0], y=v_pts[:, 1], z=v_pts[:, 2],
            mode="markers",
            marker=dict(size=2, color="#a855f7", opacity=0.7),
            name="VGGT-1B",
            hoverinfo="skip"
        ))

    # 3. MASt3R-SfM Trace
    if mast3r_points is not None and len(mast3r_points) > 0:
        m_pts = np.asarray(mast3r_points)
        if len(m_pts) > max_pts_per_model:
            m_pts = m_pts[::(len(m_pts) // max_pts_per_model)]
        fig.add_trace(go.Scatter3d(
            x=m_pts[:, 0], y=m_pts[:, 1], z=m_pts[:, 2],
            mode="markers",
            marker=dict(size=2, color="#10b981", opacity=0.7),
            name="MASt3R-SfM",
            hoverinfo="skip"
        ))

    fig.update_layout(
        title=dict(text="📊 3-Way Reconstruction Overlay (COLMAP vs. VGGT vs. MASt3R)", font=dict(color="#ffffff", size=15)),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        scene=dict(
            xaxis=dict(title="X (m)", color="#94a3b8", gridcolor="#1e293b", showbackground=False),
            yaxis=dict(title="Y (m)", color="#94a3b8", gridcolor="#1e293b", showbackground=False),
            zaxis=dict(title="Z (m)", color="#94a3b8", gridcolor="#1e293b", showbackground=False),
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig


def create_mast3r_mesh_figure(
    mesh_path: str,
    title: str = "📐 MASt3R-SfM 3D Surface Mesh",
    max_faces: int = 60000,
    color_mode: str = "rgb",
    default_color: str = "#38bdf8"
) -> go.Figure:
    """
    Generate an interactive Plotly 3D Mesh figure (go.Mesh3d) from a .ply, .obj, or .stl mesh file.
    Loads vertices and triangular faces and creates a high-fidelity rendering.
    """
    fig = go.Figure()
    if not mesh_path or not os.path.exists(mesh_path):
        fig.update_layout(
            title=dict(text="Mesh file not found", font=dict(color="#ef4444", size=14)),
            paper_bgcolor="#0e1117"
        )
        return fig

    try:
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        v_arr = np.asarray(mesh.vertices)
        f_arr = np.asarray(mesh.triangles)

        if len(v_arr) == 0 or len(f_arr) == 0:
            fig.update_layout(
                title=dict(text="Mesh contains 0 vertices or 0 faces", font=dict(color="#ef4444", size=14)),
                paper_bgcolor="#0e1117"
            )
            return fig

        # Quadric decimation for fluid in-browser interactive rotation if face count exceeds budget
        if len(f_arr) > max_faces:
            try:
                mesh_simp = mesh.simplify_quadric_decimation(target_number_of_triangles=max_faces)
                if len(mesh_simp.triangles) > 0:
                    mesh = mesh_simp
                    v_arr = np.asarray(mesh.vertices)
                    f_arr = np.asarray(mesh.triangles)
                else:
                    step = len(f_arr) // max_faces
                    f_arr = f_arr[::step]
            except Exception:
                step = len(f_arr) // max_faces
                f_arr = f_arr[::step]

        mesh3d_kwargs = dict(
            x=v_arr[:, 0],
            y=v_arr[:, 1],
            z=v_arr[:, 2],
            i=f_arr[:, 0],
            j=f_arr[:, 1],
            k=f_arr[:, 2],
            name="Surface Mesh",
            hoverinfo="skip",
            flatshading=False,
            lighting=dict(
                ambient=0.45,
                diffuse=0.6,
                roughness=0.4,
                specular=0.25,
                fresnel=0.1
            ),
            lightposition=dict(x=100, y=200, z=150)
        )

        if mesh.has_vertex_colors() and color_mode == "rgb":
            cols = np.asarray(mesh.vertex_colors)
            if cols.max() <= 1.0:
                cols = (cols * 255.0).astype(np.uint8)
            mesh3d_kwargs["vertexcolor"] = [f"rgb({int(c[0])},{int(c[1])},{int(c[2])})" for c in cols]
        else:
            mesh3d_kwargs["color"] = default_color
            mesh3d_kwargs["opacity"] = 0.95

        fig.add_trace(go.Mesh3d(**mesh3d_kwargs))

        fig.update_layout(
            title=dict(text=title, font=dict(color="#ffffff", size=15)),
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            scene=dict(
                xaxis=dict(title="X (m)", color="#94a3b8", gridcolor="#1e293b", showbackground=False),
                yaxis=dict(title="Y (m)", color="#94a3b8", gridcolor="#1e293b", showbackground=False),
                zaxis=dict(title="Z (m)", color="#94a3b8", gridcolor="#1e293b", showbackground=False),
                aspectmode="data"
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        return fig

    except Exception as e:
        fig.update_layout(
            title=dict(text=f"Error rendering mesh: {str(e)}", font=dict(color="#ef4444", size=13)),
            paper_bgcolor="#0e1117"
        )
        return fig


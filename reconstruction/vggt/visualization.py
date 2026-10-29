"""
reconstruction/vggt/visualization.py
------------------------------------
Interactive 3D visualization utilities for VGGT point clouds,
camera trajectories with viewing frustums and optical axes,
and side-by-side metric comparison with COLMAP.
Never distorts or hides scale/coordinate discrepancies.
"""

import os
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from vggt.utils.rotation import quat_to_mat


def create_vggt_point_cloud_figure(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    title: str = "🌐 VGGT 3D Point Cloud",
    max_display_points: int = 50000,
    color_mode: str = "rgb",
    center_scene: bool = True
) -> go.Figure:
    """
    Generate an interactive Plotly 3D scatter plot for the VGGT point cloud.
    Uses exact metric aspect ratio without artificial non-linear distortion.
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

    # Build marker color
    if colors is not None and color_mode == "rgb":
        cols = np.asarray(colors)
        marker_color = [f"rgb({int(c[0])},{int(c[1])},{int(c[2])})" for c in cols]
        colorscale = None
    else:
        marker_color = display_pts[:, 2]
        colorscale = "Turbo"

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
        name="VGGT 3D Points",
        hoverinfo="skip"
    )

    fig = go.Figure(data=[scatter])
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b> ({len(pts):,} points)",
            font=dict(color="#f1f5f9", size=15)
        ),
        paper_bgcolor="#0a0614",
        plot_bgcolor="#0a0614",
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis=dict(showbackground=False, visible=True, title="X (m)", color="#64748b"),
            yaxis=dict(showbackground=False, visible=True, title="Y (m)", color="#64748b"),
            zaxis=dict(showbackground=False, visible=True, title="Z (m)", color="#64748b"),
            bgcolor="#05030a",
            aspectmode="data"
        ),
        showlegend=False,
        height=700
    )
    return fig


def create_vggt_camera_trajectory_figure(
    cameras: List[Dict[str, Any]],
    points: Optional[np.ndarray] = None,
    colors: Optional[np.ndarray] = None,
    title: str = "📷 VGGT Camera Poses, Frustums & Point Cloud"
) -> go.Figure:
    """
    Generate an interactive 3D view showing camera positions, viewing rays (optical axes),
    and reconstructed 3D points in the EXACT SAME metric coordinate system.
    """
    fig = go.Figure()

    if not cameras:
        fig.update_layout(title="No camera data")
        return fig

    centers = np.array([cam["center"] for cam in cameras], dtype=float)

    # 1. Add Point Cloud in same metric coordinates
    if points is not None and len(points) > 0:
        pts = np.asarray(points, dtype=float)
        cols = np.asarray(colors) if colors is not None else None
        if len(pts) > 25000:
            step = len(pts) // 25000
            pts = pts[::step]
            if cols is not None:
                cols = cols[::step]

        if cols is not None and len(cols) == len(pts):
            p_color = [f"rgb({int(c[0])},{int(c[1])},{int(c[2])})" for c in cols]
            colorscale = None
        else:
            p_color = pts[:, 2]
            colorscale = "Viridis"

        fig.add_trace(go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            marker=dict(size=1.5, color=p_color, colorscale=colorscale, opacity=0.6),
            name=f"Reconstructed Points ({len(pts):,})",
            hoverinfo="skip"
        ))

    # 2. Camera centers
    fig.add_trace(go.Scatter3d(
        x=centers[:, 0],
        y=centers[:, 1],
        z=centers[:, 2],
        mode="markers+text",
        marker=dict(size=6, color="#ec4899", symbol="diamond"),
        text=[cam.get("image_name", f"Cam {cam.get('camera_id')}") for cam in cameras],
        textposition="top center",
        name="Camera Positions",
        hoverinfo="text"
    ))

    # 3. Camera flight path line
    fig.add_trace(go.Scatter3d(
        x=centers[:, 0],
        y=centers[:, 1],
        z=centers[:, 2],
        mode="lines",
        line=dict(color="rgba(236, 72, 153, 0.6)", width=3),
        name="Flight Trajectory",
        hoverinfo="skip"
    ))

    # 4. Camera viewing direction vectors (optical z-axis pointing forward into scene)
    ray_x, ray_y, ray_z = [], [], []
    arrow_length = 0.25  # 25 cm viewing vector

    for cam in cameras:
        ext = np.array(cam.get("extrinsic", np.eye(4)[:3]))
        R_w2c = ext[:3, :3]
        R_c2w = R_w2c.T
        center = np.array(cam["center"])

        # In OpenCV convention, camera looks down +Z axis: [0, 0, 1]
        look_dir = R_c2w @ np.array([0.0, 0.0, 1.0])
        look_end = center + look_dir * arrow_length

        ray_x.extend([center[0], look_end[0], None])
        ray_y.extend([center[1], look_end[1], None])
        ray_z.extend([center[2], look_end[2], None])

    fig.add_trace(go.Scatter3d(
        x=ray_x,
        y=ray_y,
        z=ray_z,
        mode="lines",
        line=dict(color="#38bdf8", width=4),
        name="Viewing Rays (Optical Axes)",
        hoverinfo="skip"
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b> ({len(cameras)} cameras)",
            font=dict(color="#f1f5f9", size=15)
        ),
        paper_bgcolor="#0a0614",
        plot_bgcolor="#0a0614",
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis=dict(showbackground=False, visible=True, title="X (m)", color="#64748b"),
            yaxis=dict(showbackground=False, visible=True, title="Y (m)", color="#64748b"),
            zaxis=dict(showbackground=False, visible=True, title="Z (m)", color="#64748b"),
            bgcolor="#05030a",
            aspectmode="data"
        ),
        legend=dict(
            x=0.02, y=0.98,
            font=dict(color="#cbd5e1"),
            bgcolor="rgba(15, 23, 42, 0.85)"
        ),
        height=700
    )
    return fig


def create_comparison_point_cloud_figure(
    colmap_points: np.ndarray,
    vggt_points: np.ndarray,
    title: str = "🔍 COLMAP vs VGGT Geometry Comparison"
) -> go.Figure:
    """
    Create an overlay comparison figure where COLMAP points are rendered in Cyan
    and VGGT points are rendered in Purple.
    """
    fig = go.Figure()

    def sample_cloud(pts, max_pts=30000):
        if len(pts) == 0:
            return np.zeros((0, 3))
        if len(pts) > max_pts:
            step = len(pts) // max_pts
            pts = pts[::step]
        return pts

    c_pts = sample_cloud(colmap_points)
    v_pts = sample_cloud(vggt_points)

    if len(c_pts) > 0:
        fig.add_trace(go.Scatter3d(
            x=c_pts[:, 0],
            y=c_pts[:, 1],
            z=c_pts[:, 2],
            mode="markers",
            marker=dict(size=2, color="#06b6d4", opacity=0.7),
            name=f"COLMAP ({len(colmap_points):,} pts)",
            hoverinfo="skip"
        ))

    if len(v_pts) > 0:
        fig.add_trace(go.Scatter3d(
            x=v_pts[:, 0],
            y=v_pts[:, 1],
            z=v_pts[:, 2],
            mode="markers",
            marker=dict(size=2, color="#d946ef", opacity=0.7),
            name=f"VGGT ({len(vggt_points):,} pts)",
            hoverinfo="skip"
        ))

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(color="#f1f5f9", size=15)
        ),
        paper_bgcolor="#0a0614",
        plot_bgcolor="#0a0614",
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis=dict(showbackground=False, visible=True, title="X (m)", color="#64748b"),
            yaxis=dict(showbackground=False, visible=True, title="Y (m)", color="#64748b"),
            zaxis=dict(showbackground=False, visible=True, title="Z (m)", color="#64748b"),
            bgcolor="#05030a",
            aspectmode="data"
        ),
        legend=dict(
            x=0.02, y=0.98,
            font=dict(color="#cbd5e1"),
            bgcolor="rgba(15, 23, 42, 0.85)"
        ),
        height=700
    )
    return fig


def save_html_debug_visualizations(
    output_dir: str,
    cameras: List[Dict[str, Any]],
    points: np.ndarray,
    colors: Optional[np.ndarray] = None
):
    """
    Save standalone HTML visual debug artifacts into the output debug folder.
    """
    debug_dir = os.path.join(output_dir, "debug_visuals")
    os.makedirs(debug_dir, exist_ok=True)

    try:
        traj_fig = create_vggt_camera_trajectory_figure(cameras, points, colors, title="VGGT Cameras and 3D Structure")
        traj_fig.write_html(os.path.join(debug_dir, "camera_pointcloud_combined.html"), include_plotlyjs="cdn")
    except Exception:
        pass

    try:
        pc_fig = create_vggt_point_cloud_figure(points, colors, title="VGGT Reconstructed Point Cloud")
        pc_fig.write_html(os.path.join(debug_dir, "point_cloud_interactive.html"), include_plotlyjs="cdn")
    except Exception:
        pass

def create_vggt_mesh_figure(
    mesh_path: str,
    title: str = "📐 VGGT 3D Surface Mesh",
    max_faces: int = 60000,
    color_mode: str = "rgb",
    default_color: str = "#a855f7"
) -> go.Figure:
    """
    Generate an interactive Plotly 3D Mesh figure (go.Mesh3d) from a VGGT .ply, .obj, or .stl mesh file.
    Loads vertices and triangular faces and renders a clean high-fidelity surface representation.
    """
    fig = go.Figure()
    if not mesh_path or not os.path.exists(mesh_path):
        fig.update_layout(
            title=dict(text="VGGT Surface Mesh file not found", font=dict(color="#ef4444", size=14)),
            paper_bgcolor="#05030a"
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
                paper_bgcolor="#05030a"
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
            name="VGGT Surface Mesh",
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
            paper_bgcolor="#05030a",
            plot_bgcolor="#05030a",
            scene=dict(
                xaxis=dict(title="X (m)", color="#64748b", gridcolor="#1e1b4b", showbackground=False),
                yaxis=dict(title="Y (m)", color="#64748b", gridcolor="#1e1b4b", showbackground=False),
                zaxis=dict(title="Z (m)", color="#64748b", gridcolor="#1e1b4b", showbackground=False),
                aspectmode="data"
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        return fig

    except Exception as e:
        fig.update_layout(
            title=dict(text=f"Error rendering VGGT mesh: {str(e)}", font=dict(color="#ef4444", size=13)),
            paper_bgcolor="#05030a"
        )
        return fig


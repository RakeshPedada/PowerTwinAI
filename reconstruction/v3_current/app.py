# =========================================================
# WINDOWS ASYNCIO FIX
# Must be the very first thing — before streamlit or any
# other import that touches the event loop.
# =========================================================

import sys
import asyncio
import logging

if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )

logging.getLogger("asyncio").setLevel(logging.CRITICAL)

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd 
import subprocess
import json
import os
import glob
import open3d as o3d

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

from ui.styles import load_styles
from ui.header import show_header
from ui.input_section import render_input_section
from ui.dataset_preview import render_dataset_preview
from ui.reconstruction_controls import render_reconstruction_controls
from ui.progress_panel import render_progress_panel
from ui.results_dashboard import render_results_dashboard
from ui.point_cloud_viewer import render_point_cloud
from ui.mesh_viewer import render_mesh_viewer
from ui.camera_trajectory import render_camera_trajectory
from ui.reconstruction_summary import render_reconstruction_summary
from ui.pair_statistics import render_pair_statistics
from core.session_manager import initialize_session
from mesh_generator import is_mesh_ply, get_ply_statistics


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Advanced SfM Reconstruction Engine",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# LOAD UI STYLES
# =========================================================

load_styles()

# =========================================================
# PATHS
# =========================================================

from config.paths import *
from config.settings import *

# =========================================================
# SESSION STATE
# =========================================================

initialize_session()

# =========================================================
# HEADER
# =========================================================

show_header()

# =========================================================
# UPLOAD SECTION
# =========================================================

uploaded_files, mode = render_input_section()

# =========================================================
# RECONSTRUCTION CONTROLS (QUALITY PRESETS & ADVANCED SETTINGS)
# =========================================================

render_reconstruction_controls(
    uploaded_files,
    mode,
    TEMP_DIR,
    LOG_FILE,
    STATUS_FILE,
    RESULT_FILE,
    IMAGE_PATHS_FILE
)

# =========================================================
# LIVE LOGS
# =========================================================

if render_progress_panel(
    LOG_FILE,
    STATUS_FILE
):
    st.rerun()

# =========================================================
# RESULTS & 5-TAB VISUALIZATION
# =========================================================

results = render_results_dashboard(
    RESULT_FILE
)

if results:
    points = results["points"]
    cameras = results["cameras"]
    sparse_points = results.get("sparse_points", np.zeros((0, 3)))
    raw_points = results.get("raw_points", np.zeros((0, 3)))
    colors = results.get("colors", st.session_state.get("reconstruction_colors"))
    analytics = results["analytics"]
    processing_time = results["processing_time"]
    colmap_time = results["colmap_time"]
    reconstruction_time = results["reconstruction_time"]
    total_pipeline_time = results["total_pipeline_time"]
    ply_path = results.get("ply_path", os.path.join("output", "reconstruction.ply"))
    mesh_path = results.get("mesh_path", os.path.join("output", "mesh.ply"))

    # ── Hole / Coverage Alert (Requirement 5) ────────────────────────────────
    has_cov_warning = analytics.get("has_coverage_warning", False)
    cov_msg = analytics.get("coverage_warning")
    if has_cov_warning and cov_msg:
        st.warning(f"⚠️ **Coverage Warning**: {cov_msg}")

    # =====================================================
    # 5-TAB VISUALIZATION INTERFACE (Requirement 8)
    # =====================================================
    st.markdown(
        "<div class='section-title'>🎯 Multi-Stage Reconstruction Visualizer</div>",
        unsafe_allow_html=True
    )

    tab_input, tab_sparse, tab_raw_dense, tab_cleaned, tab_final_mesh = st.tabs([
        "📷 1. Input Images",
        "🌐 2. Sparse Reconstruction",
        "☁️ 3. Dense Point Cloud",
        "✨ 4. Cleaned Point Cloud",
        "🔺 5. Final Mesh"
    ])

    # ── Tab 1: Input Images ──────────────────────────────────────────────────
    with tab_input:
        st.markdown("<p style='color: #94a3b8; font-size: 13px;'>Inspection gallery of multi-view input images used for Structure-from-Motion.</p>", unsafe_allow_html=True)
        render_dataset_preview(uploaded_files, mode)

    # ── Tab 2: Sparse Reconstruction ─────────────────────────────────────────
    with tab_sparse:
        st.markdown("<p style='color: #94a3b8; font-size: 13px;'>Recovered 6-DOF camera trajectory and initial multi-view triangulated sparse feature cloud.</p>", unsafe_allow_html=True)
        col_s1, col_s2 = st.columns([1, 1])
        with col_s1:
            render_camera_trajectory(cameras)
        with col_s2:
            if len(sparse_points) > 0:
                render_point_cloud(sparse_points, title="🌐 Sparse SfM Feature Points", key="pc_sparse")
            else:
                st.info("Sparse point cloud file not populated.")
        render_pair_statistics()

    # ── Tab 3: Dense Point Cloud ─────────────────────────────────────────────
    with tab_raw_dense:
        st.markdown("<p style='color: #94a3b8; font-size: 13px;'>Raw dense point cloud generated via multi-view stereo fusion before outlier filtering.</p>", unsafe_allow_html=True)
        raw_ply_file = os.path.join("outputs", "dense", "raw_pointcloud.ply")
        if len(raw_points) > 0:
            render_point_cloud(raw_points, title="☁️ Raw Dense Point Cloud (Pre-Cleaning)", key="pc_raw_dense")
        elif os.path.exists(raw_ply_file):
            try:
                raw_pcd_obj = o3d.io.read_point_cloud(raw_ply_file)
                render_point_cloud(
                    np.asarray(raw_pcd_obj.points),
                    np.asarray(raw_pcd_obj.colors) if raw_pcd_obj.has_colors() else None,
                    title=f"☁️ Raw Dense Point Cloud ({len(raw_pcd_obj.points):,} points)",
                    key="pc_raw_dense_file"
                )
            except Exception:
                render_point_cloud(points, colors, title="☁️ Dense Point Cloud", key="pc_dense_fallback_1")
        else:
            render_point_cloud(points, colors, title="☁️ Dense Point Cloud", key="pc_dense_fallback_2")

    # ── Tab 4: Cleaned Point Cloud ───────────────────────────────────────────
    with tab_cleaned:
        st.markdown("<p style='color: #94a3b8; font-size: 13px;'>Filtered dense point cloud with statistical outlier removal, normal estimation, and surface orientation.</p>", unsafe_allow_html=True)
        render_point_cloud(points, colors, title="✨ Cleaned Point Cloud (Filtered & Normal-Oriented)", key="pc_cleaned")

    # ── Tab 5: Final Mesh ────────────────────────────────────────────────────
    with tab_final_mesh:
        st.markdown("<p style='color: #94a3b8; font-size: 13px;'>Watertight 3D triangular surface mesh generated via Poisson reconstruction / Ball Pivoting with photorealistic colors.</p>", unsafe_allow_html=True)
        mesh_file = (
            analytics.get("mesh_path") 
            if analytics and "mesh_path" in analytics 
            else os.path.join("output", "mesh.ply")
        )
        render_mesh_viewer(
            points=points, 
            colors=colors, 
            default_mesh_path=mesh_file,
            key="mesh_tab"
        )

    # =====================================================
    # PLY STATISTICS INSPECTOR (Requirement 7)
    # =====================================================
    with st.expander("🔍 PLY Structural Geometry Inspector (`get_ply_statistics`)", expanded=False):
        inspect_options = [
            os.path.join("outputs", "mesh", "final_mesh.ply"),
            os.path.join("output", "mesh.ply"),
            os.path.join("outputs", "dense", "filtered_pointcloud.ply"),
            os.path.join("output", "reconstruction.ply"),
            os.path.join("outputs", "dense", "raw_pointcloud.ply"),
            os.path.join("outputs", "sparse", "sparse_model.ply")
        ]
        valid_options = [p for p in inspect_options if os.path.exists(p)]
        if valid_options:
            selected_path = st.selectbox("Select PLY artifact to inspect:", valid_options, key="ply_inspector_select")
            ply_stats = get_ply_statistics(selected_path)
            stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
            stat_c1.metric("Object Type", ply_stats["type"])
            stat_c2.metric("Vertices", f"{ply_stats['vertex_count']:,}")
            stat_c3.metric("Triangular Faces", f"{ply_stats['face_count']:,}")
            stat_c4.metric("File Size", f"{ply_stats['file_size_mb']} MB")

            st.caption(f"Normals: {'✅ Available' if ply_stats['has_normals'] else '❌ Missing'} | RGB Colors: {'✅ Available' if ply_stats['has_colors'] else '❌ Missing'} | Verified element face: {'✅ Yes' if ply_stats['has_faces'] else '❌ No'}")
            st.json(ply_stats)
        else:
            st.info("No generated PLY files found to inspect.")

    # =====================================================
    # RECONSTRUCTION SUMMARY & FINAL REPORT (Requirement 14)
    # =====================================================
    render_reconstruction_summary(
        analytics,
        uploaded_files,
        cameras,
        points,
        processing_time,
        colmap_time,
        reconstruction_time,
        total_pipeline_time
    )
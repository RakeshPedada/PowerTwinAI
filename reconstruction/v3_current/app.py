import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd 
import subprocess
import json
import os
import glob
from streamlit_autorefresh import st_autorefresh
from PowerTwinAI.reconstruction.v3_current.ui.pair_statistics import render_pair_statistics
from dense_reconstruction import DenseReconstructor
from ui.styles import load_styles
from ui.header import show_header
from ui.input_section import render_input_section
from ui.dataset_preview import render_dataset_preview
from ui.reconstruction_controls import render_reconstruction_controls
from ui.progress_panel import render_progress_panel
from ui.results_dashboard import render_results_dashboard
from ui.point_cloud_viewer import render_point_cloud
from ui.camera_trajectory import render_camera_trajectory
from ui.pair_statistics import render_pair_statistics
from core.session_manager import initialize_session


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

load_styles ()

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
# UPLOAD
# =========================================================

uploaded_files, mode = render_input_section()


# =========================================================
# IMAGE PREVIEW
# =========================================================

render_dataset_preview(uploaded_files, mode)


# =========================================================
# START BUTTON
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
# RESULTS
# =========================================================
results = render_results_dashboard(
    RESULT_FILE
)

if results:

    (
        points,
        cameras,
        analytics,
        processing_time,
        colmap_time,
        reconstruction_time,
        total_pipeline_time
    ) = results
    # =====================================================
    # POINT CLOUD VISUALIZATION
    # =====================================================

    render_point_cloud(points)

    # =====================================================
    # CAMERA TRAJECTORY
    # =====================================================

    render_camera_trajectory(cameras)

    # =====================================================
    # PAIR STATISTICS
    # =====================================================

    render_pair_statistics()

    # =====================================================
    # RECONSTRUCTION SUMMARY
    # =====================================================

    st.markdown(
        "<div class='section-title'>📋 Reconstruction Summary</div>",
        unsafe_allow_html=True
    )

    input_images = int(
        analytics.get(
            "total_input_images",
            len(uploaded_files)
        )
    )

    registered_images = int(
        analytics.get(
            "registered_images",
            len(cameras)
        )
    )

    registration_ratio = float(
        analytics.get(
            "registration_ratio",
            (
                registered_images
                / max(input_images, 1)
                * 100.0
            )
        )
    )

    sparse_points = int(
        analytics.get(
            "total_sparse_points",
            0
        )
    )

    dense_points = int(
        analytics.get(
            "total_dense_points",
            0
        )
    )

    health_score = float(
        analytics.get(
            "health_score",
            registration_ratio
        )
    )

    st.markdown(
        f"""
### Final Reconstruction Statistics

- Input Images: **{input_images:,}**
- COLMAP Registered Images: **{registered_images:,}**
- Registration Ratio: **{registration_ratio:.2f}%**
- Sparse Points: **{sparse_points:,}**
- Dense Points Generated: **{dense_points:,}**
- Final Cleaned Points: **{len(points):,}**
- Camera Positions: **{len(cameras):,}**
- Registration Health Score: **{health_score:.2f}%**
- Reconstruction Stage Time: **{processing_time:.2f} sec**
- COLMAP Time: **{colmap_time:.2f} sec**
- Dense + Cleaning + Mesh Time: **{reconstruction_time:.2f} sec**
- **Total Pipeline Time: {total_pipeline_time:.2f} sec**
- Sparse Reconstruction: **COLMAP SfM + Bundle Adjustment**
- Dense Reconstruction: **Stage-1 StereoSGBM Prototype**
"""
    )
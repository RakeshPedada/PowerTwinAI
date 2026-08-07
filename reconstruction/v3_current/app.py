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
from ui.reconstruction_summary import render_reconstruction_summary
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
# =========================================================
# WINDOWS ASYNCIO FIX
# Must be the very first thing — before streamlit or any
# other import that touches the event loop.
# =========================================================

import sys
import os
import asyncio
import logging
import warnings

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

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
import json
import glob
import open3d as o3d

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

from config.paths import (
    BASE_DIR, TEMP_DIR, OUTPUT_DIR, OUTPUTS_DIR,
    LOG_FILE, STATUS_FILE, RESULT_FILE,
    IMAGE_PATHS_FILE, CONFIG_FILE
)
from config.settings import QUALITY_PRESETS, get_active_config
from core.session_manager import initialize_session
from ui.styles import load_styles
from ui.sidebar import render_sidebar
from ui.dashboard import render_dashboard
from ui.new_reconstruction import render_new_reconstruction
from ui.reconstruction_studio import render_reconstruction_studio
from ui.viewer_page import render_viewer_page
from ui.diagnostics_page import render_diagnostics_page
from ui.comparison_page import render_comparison_page
from ui.mesh_studio_page import render_mesh_studio_page
from ui.export_page import render_export_page
from ui.settings_page import render_settings_page

# Keep legacy imports intact for backward compatibility
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
from ui.vggt_section import render_vggt_section
from ui.mast3r_section import render_mast3r_section
from ui.comparison_section import render_comparison_section
from mesh_generator import is_mesh_ply, get_ply_statistics


# =========================================================
# PAGE CONFIGURATION (Expanded Sidebar Engineering Layout)
# =========================================================

st.set_page_config(
    page_title="PowerTwinAI – 3D Reconstruction Workstation",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# INJECT PROFESSIONAL DESIGN SYSTEM
# =========================================================

load_styles()

# =========================================================
# INITIALIZE REACTIVE SESSION STATE
# =========================================================

initialize_session()

# =========================================================
# PERSISTENT SIDEBAR NAVIGATION
# =========================================================

active_page = render_sidebar()

# =========================================================
# PAGE ROUTER
# =========================================================

if active_page == "Overview":
    render_dashboard()

elif active_page == "New Reconstruction":
    render_new_reconstruction()

elif active_page == "Reconstruction Studio":
    render_reconstruction_studio()

elif active_page == "3D Viewer":
    render_viewer_page()

elif active_page == "AI Diagnostics":
    render_diagnostics_page()

elif active_page == "Backend Comparison":
    render_comparison_page()

elif active_page == "Mesh Studio":
    render_mesh_studio_page()

elif active_page == "Export Center":
    render_export_page()

elif active_page == "Settings":
    render_settings_page()

else:
    render_dashboard()

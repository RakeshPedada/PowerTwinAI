"""
ui/settings_page.py
-------------------
Enterprise Hardware Telemetry, Engine Configurations, and Workstation Settings.
Displays verified hardware acceleration status, local weights cache paths,
processing thread configurations, and storage directories.
"""

import os
import sys
import streamlit as st
import torch
import open3d as o3d

from core.gpu_detector import get_gpu_info
from vggt_pipeline import is_vggt_weights_cached
from reconstruction.mast3r import is_mast3r_weights_cached
from config.paths import BASE_DIR, TEMP_DIR, OUTPUT_DIR, OUTPUTS_DIR


def render_settings_page():
    """Render the Settings & Configuration page."""

    st.markdown(
        """
<div class="eng-card" style="padding: 16px 22px; margin-bottom: 20px;">
  <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
    <div>
      <div style="display: flex; align-items: center; gap: 10px;">
        <span style="font-size: 22px;">⚙️</span>
        <h2 style="margin: 0; font-size: 20px; font-weight: 700; color: #fff;">Workstation Environment &amp; Engine Settings</h2>
      </div>
      <p style="margin: 4px 0 0; font-size: 13px; color: #94A3B8;">
        Hardware acceleration telemetry, local model checkpoint caches, and workstation runtime preferences.
      </p>
    </div>
    <span class="badge-pill badge-primary">System Telemetry</span>
  </div>
</div>
""",
        unsafe_allow_html=True
    )

    gpu = get_gpu_info()
    vggt_cached = is_vggt_weights_cached("facebook/VGGT-1B")
    mast3r_cached = is_mast3r_weights_cached()

    # ── 1. System & Hardware Telemetry ────────────────────────────────────────
    st.markdown("<div class='section-sub-label' style='margin-bottom: 12px; font-weight: 700; color: #94A3B8;'>SYSTEM RUNTIME &amp; HARDWARE TELEMETRY</div>", unsafe_allow_html=True)

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Host Processor / GPU", gpu.get("name", "Intel UHD Graphics"))
    h2.metric("CUDA Acceleration", "Available (GPU)" if gpu.get("cuda_available") else "CPU Fallback Mode", delta="CUDA Ready" if gpu.get("cuda_available") else "Active", delta_color="normal")
    h3.metric("PyTorch Engine", f"v{torch.__version__}")
    h4.metric("Open3D Geometric Engine", f"v{o3d.__version__}")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # ── 2. Local Weights Cache Status ─────────────────────────────────────────
    st.markdown("<div class='section-sub-label' style='margin-bottom: 12px; font-weight: 700; color: #94A3B8;'>DEEP LEARNING MODEL CACHE STATUS</div>", unsafe_allow_html=True)

    c_col1, c_col2 = st.columns(2)
    with c_col1:
        st.markdown(
            f"""
<div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; padding: 16px 20px;">
  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
    <h4 style="margin: 0; font-size: 15px; color: #fff;">facebook/VGGT-1B</h4>
    <span class="badge-pill {'badge-success' if vggt_cached else 'badge-warning'}">{'● Cached (~4.68 GB)' if vggt_cached else '● Not Cached'}</span>
  </div>
  <p style="font-size: 12px; color: #94A3B8; line-height: 1.5; margin: 0 0 10px;">
    Visual Geometry Grounded Transformer model weights stored in local Hugging Face Hub cache.
  </p>
  <div style="font-size: 11px; color: #cbd5e1; font-family: 'JetBrains Mono', monospace; background: var(--bg-elevated); padding: 6px 10px; border-radius: 6px;">
    ~/.cache/huggingface/hub/models--facebook--VGGT-1B
  </div>
</div>
""",
            unsafe_allow_html=True
        )

    with c_col2:
        st.markdown(
            f"""
<div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; padding: 16px 20px;">
  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
    <h4 style="margin: 0; font-size: 15px; color: #fff;">naver/MASt3R_ViTLarge_512</h4>
    <span class="badge-pill {'badge-success' if mast3r_cached else 'badge-warning'}">{'● Cached (~2.6 GB)' if mast3r_cached else '● Not Cached'}</span>
  </div>
  <p style="font-size: 12px; color: #94A3B8; line-height: 1.5; margin: 0 0 10px;">
    Naver Labs MASt3R model checkpoints and retrieval codebook.
  </p>
  <div style="font-size: 11px; color: #cbd5e1; font-family: 'JetBrains Mono', monospace; background: var(--bg-elevated); padding: 6px 10px; border-radius: 6px;">
    checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
  </div>
</div>
""",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # ── 3. Workspace Directories & Storage Paths ──────────────────────────────
    st.markdown("<div class='section-sub-label' style='margin-bottom: 12px; font-weight: 700; color: #94A3B8;'>WORKSPACE PATHS &amp; STORAGE ARCHITECTURE</div>", unsafe_allow_html=True)

    paths_info = [
        ("Base Workspace Root", BASE_DIR),
        ("Temporary Session Working Dir", TEMP_DIR),
        ("COLMAP Outputs Directory", OUTPUT_DIR),
        ("Structured Multi-Backend Outputs", OUTPUTS_DIR),
    ]

    for label, p in paths_info:
        st.markdown(
            f"""
<div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 10px 16px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
  <span style="font-size: 12px; font-weight: 600; color: #94A3B8;">{label}:</span>
  <code style="font-size: 11px; color: #38BDF8;">{p}</code>
</div>
""",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # ── 4. Visualization & Thread Preferences ─────────────────────────────────
    st.markdown("<div class='section-sub-label' style='margin-bottom: 12px; font-weight: 700; color: #94A3B8;'>VISUALIZATION PREFERENCES</div>", unsafe_allow_html=True)

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.selectbox("Default 3D Background Color:", ["#0B0F19 (Deep Obsidian)", "#070A11 (Carbon)", "#111827 (Dark Gray)"], index=0)
        st.slider("Plotly Point Budget for Fluid Rendering:", 20000, 200000, 80000, 10000)
    with p_col2:
        st.selectbox("Default Camera Frustum Mode:", ["Orbit / Free Turntable", "Top-Down Orthogonal", "Isometric 45°"], index=0)
        st.checkbox("Auto-Reload 3D Point Cloud upon Job Completion", value=True)

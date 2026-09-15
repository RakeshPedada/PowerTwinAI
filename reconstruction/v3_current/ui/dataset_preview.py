"""
ui/dataset_preview.py
---------------------
Dataset preview and thumbnail gallery matching the Figma design.
"""

import os
import streamlit as st


def render_dataset_preview(uploaded_files, mode):
    if not uploaded_files:
        return

    st.markdown(
        """
<div class="figma-card">
  <div class="card-header-bar">
    <div class="card-header-icon">📸</div>
    <div class="card-header-titles">
      <h3>Dataset Preview &amp; Gallery</h3>
      <p>Inspection of input frames and dataset footprint</p>
    </div>
  </div>
""",
        unsafe_allow_html=True,
    )

    total_size = 0
    preview_limit = 8

    cols = st.columns(4)

    for idx, file in enumerate(uploaded_files):
        if mode == "upload" or mode == "Upload Images":
            size_mb = len(file.getvalue()) / (1024 * 1024)
            caption = file.name
            image = file
        else:
            size_mb = os.path.getsize(file) / (1024 * 1024)
            caption = os.path.basename(file)
            image = file

        total_size += size_mb

        if idx < preview_limit:
            with cols[idx % 4]:
                st.image(
                    image,
                    caption=caption,
                    width='stretch'
                )

    if len(uploaded_files) > preview_limit:
        st.caption(f"Showing first {preview_limit} of {len(uploaded_files)} images...")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    st.markdown(
        f"""
<div class="glass-stat-grid">
  <div class="glass-stat-card">
    <div class="glass-stat-label">Total Images</div>
    <div class="glass-stat-value">{len(uploaded_files)}</div>
    <div class="glass-stat-sub">Valid frames loaded</div>
  </div>
  <div class="glass-stat-card">
    <div class="glass-stat-label">Dataset Size</div>
    <div class="glass-stat-value">{total_size:.2f} MB</div>
    <div class="glass-stat-sub">Uncompressed memory footprint</div>
  </div>
</div>
</div>
""",
        unsafe_allow_html=True,
    )
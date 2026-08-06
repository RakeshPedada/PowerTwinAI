"""
Dataset preview section for the PowerTwinAI Streamlit application.

Responsibilities:
- Preview uploaded images
- Display dataset statistics
"""

import os
import streamlit as st


def render_dataset_preview(uploaded_files, mode):
    """
    Display dataset preview and statistics.

    Parameters
    ----------
    uploaded_files : list
        Uploaded files or dataset image paths.

    mode : str
        Input mode selected by the user.
    """

    if not uploaded_files:
        return

    st.markdown(
        "<div class='section-title'>📸 Dataset Preview</div>",
        unsafe_allow_html=True
    )

    total_size = 0
    preview_limit = 8

    cols = st.columns(4)

    for idx, file in enumerate(uploaded_files):

        # --------------------------------------------------
        # Calculate image size
        # --------------------------------------------------

        if mode == "Upload Images":
            size_mb = len(file.getvalue()) / (1024 * 1024)
            caption = file.name
            image = file

        else:
            size_mb = os.path.getsize(file) / (1024 * 1024)
            caption = os.path.basename(file)
            image = file

        total_size += size_mb

        # --------------------------------------------------
        # Preview only first few images
        # --------------------------------------------------

        if idx < preview_limit:

            cols[idx % 4].image(
                image,
                caption=caption,
                use_container_width=True
            )

    # --------------------------------------------------
    # Dataset Statistics
    # --------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Images</div>
            <div class="metric-value">{len(uploaded_files)}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Dataset Size</div>
            <div class="metric-value">{total_size:.2f} MB</div>
        </div>
        """, unsafe_allow_html=True)
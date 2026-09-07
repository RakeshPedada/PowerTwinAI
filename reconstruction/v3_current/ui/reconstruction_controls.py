"""
ui/reconstruction_controls.py
-----------------------------
Launch and execution controls matching the Figma design.
"""

import streamlit as st

from core.reconstruction_manager import (
    prepare_workspace,
    save_uploaded_images,
    launch_reconstruction
)


def render_reconstruction_controls(
    uploaded_files,
    mode,
    temp_dir,
    log_file,
    status_file,
    result_file,
    image_paths_file
):
    if not uploaded_files:
        return

    if st.session_state.reconstruction_running:
        st.markdown(
            """
<div class="status-box-success" style="border-color: rgba(168, 85, 247, 0.4); color: #d8b4fe;">
  ⚡ &nbsp; <b>Reconstruction in progress...</b> COLMAP feature extractor and stereo pipeline active.
</div>
""",
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        """
<div class="figma-card" style="padding: 20px 28px;">
  <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
    <div>
      <h4 style="margin: 0; font-size: 16px; font-weight: 700; color: #fff;">Ready to Reconstruct</h4>
      <p style="margin: 4px 0 0; font-size: 12px; color: #94a3b8;">Execute high-precision 3D Structure-from-Motion pipeline</p>
    </div>
  </div>
  <div style="margin-top: 16px;">
""",
        unsafe_allow_html=True,
    )

    if st.button("🚀 Start 3D Reconstruction Pipeline"):
        prepare_workspace(
            temp_dir,
            log_file,
            status_file,
            result_file
        )

        save_uploaded_images(
            uploaded_files,
            mode,
            temp_dir,
            image_paths_file
        )

        launch_reconstruction(log_file=log_file)

        st.session_state.reconstruction_running = True
        st.session_state.reconstruction_done = False
        st.rerun()

    st.markdown("</div></div>", unsafe_allow_html=True)
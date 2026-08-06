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
    """
    Render reconstruction controls.
    """

    if not uploaded_files:
        return

    if st.session_state.reconstruction_running:
        return

    if st.button("🚀 Start Reconstruction"):

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

        launch_reconstruction()

        st.session_state.reconstruction_running = True
        st.session_state.reconstruction_done = False
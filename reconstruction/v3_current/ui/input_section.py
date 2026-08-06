"""
Input section for the PowerTwinAI Streamlit application.

Responsibilities:
- Select image source
- Load uploaded images
- Load images from an existing dataset folder
"""

import os
import glob
import streamlit as st


def render_input_section():
    """
    Render the input section.

    Returns
    -------
    uploaded_files : list
        List of uploaded images or dataset image paths.

    mode : str
        Selected input mode.
    """

    st.markdown(
        "<div class='section-title'>📂 Choose Inspection Data Source</div>",
        unsafe_allow_html=True
    )

    mode = st.radio(
        "Choose Image Source",
        (
            "Upload Images",
            "Existing Dataset Folder"
        )
    )

    uploaded_files = []

    # -----------------------------------------------------
    # Upload Images
    # -----------------------------------------------------
    if mode == "Upload Images":

        uploaded_files = st.file_uploader(
            "📂 Upload Images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

    # -----------------------------------------------------
    # Existing Dataset Folder
    # -----------------------------------------------------
    else:

        dataset_path = st.text_input(
            "Dataset Folder",
            placeholder=r"Example: E:\Datasets\Your_Dataset"
        )

        if dataset_path:

            if os.path.isdir(dataset_path):

                extensions = ("*.jpg", "*.jpeg", "*.png")

                for ext in extensions:
                    uploaded_files.extend(
                        glob.glob(os.path.join(dataset_path, ext))
                    )

                if not uploaded_files:
                    st.info("ℹ️ No supported images found in this folder.")

            else:

                st.warning("⚠️ Dataset folder does not exist.")

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} images loaded successfully.")
    else:
        st.info("No images selected.")

    return uploaded_files, mode
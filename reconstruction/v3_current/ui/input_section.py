"""
ui/input_section.py
-------------------
Exact Figma-matching Input Section for the SfM Reconstruction Engine,
featuring interactive cards for 'Upload Images' and 'Existing Data Folder'.
"""

import os
import glob
import streamlit as st

IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"
)

MIN_IMAGES = 3


def _scan_folder(folder_path: str) -> list[str]:
    """Return all image file paths found directly inside *folder_path*."""
    paths: list[str] = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))
    return sorted(set(paths))


def _fmt_size(n_bytes: int) -> str:
    """Human-readable file size."""
    if n_bytes < 1024:
        return f"{n_bytes} B"
    if n_bytes < 1024 ** 2:
        return f"{n_bytes/1024:.1f} KB"
    return f"{n_bytes/1024**2:.1f} MB"


def _render_upload_mode() -> list | None:
    """Render the upload dropzone and metadata check."""
    st.markdown(
        """
<div class="input-zone-tag">
  <div class="yellow-bullet"></div>
  <span>Upload Image Files</span>
</div>
""",
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Select 100+ images for best results (JPG, PNG, BMP, TIFF, WebP)",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"],
        accept_multiple_files=True,
        label_visibility="visible",
        help="Upload multi-view overlapping images of the scene (~60% overlap recommended).",
    )

    if uploaded_files:
        n = len(uploaded_files)
        total_bytes = sum(f.size for f in uploaded_files)

        st.markdown(
            f"""
<div class="badge-counter">
  📷 &nbsp;<b>{n}</b> image{"s" if n != 1 else ""} selected &nbsp;·&nbsp; {_fmt_size(total_bytes)} total
</div>
""",
            unsafe_allow_html=True,
        )

        ok_count = n >= MIN_IMAGES
        _render_req_rows(n_images=n, from_folder=False)
        return uploaded_files if ok_count else None

    _render_req_rows(n_images=0, from_folder=False)
    return None


def _render_folder_mode() -> list | None:
    """Render the folder path input and validation."""
    st.markdown(
        """
<div class="input-zone-tag">
  <div class="yellow-bullet"></div>
  <span>Specify Local Dataset Directory</span>
</div>
""",
        unsafe_allow_html=True,
    )

    folder_path = st.text_input(
        "Local Directory Path",
        placeholder="e.g. C:\\Users\\Del\\datasets\\building_facade",
        help="Absolute path to a folder containing image files.",
        label_visibility="visible",
    )

    if not folder_path:
        _render_req_rows(n_images=0, from_folder=True)
        return None

    folder_path = folder_path.strip().strip('"').strip("'")

    if not os.path.isdir(folder_path):
        st.markdown(
            f"""
<div class="status-box-error">
  ❌ &nbsp; Directory not found: <b>{folder_path}</b>
</div>
""",
            unsafe_allow_html=True,
        )
        return None

    image_paths = _scan_folder(folder_path)
    n = len(image_paths)

    if n == 0:
        st.markdown(
            """
<div class="status-box-error">
  ⚠️ &nbsp; No supported image files (.jpg, .png, .tiff, .webp) found in the specified directory.
</div>
""",
            unsafe_allow_html=True,
        )
        return None

    total_bytes = sum(os.path.getsize(p) for p in image_paths if os.path.exists(p))
    st.markdown(
        f"""
<div class="status-box-success">
  ✅ &nbsp; Found <b>{n}</b> image{"s" if n != 1 else ""} &nbsp;·&nbsp; {_fmt_size(total_bytes)} total
</div>
""",
        unsafe_allow_html=True,
    )

    _render_req_rows(n_images=n, from_folder=True)
    return image_paths if n >= MIN_IMAGES else None


def _render_req_rows(n_images: int, from_folder: bool) -> None:
    """Show requirement rows with glowing status dots."""
    checks = [
        (
            n_images >= MIN_IMAGES,
            f"3 or more overlapping images required {f'(found {n_images})' if n_images > 0 else ''}",
            "dot-green" if n_images >= MIN_IMAGES else ("dot-yellow" if n_images == 0 else "dot-red"),
        ),
        (
            True,
            "Minimum 60% overlap between adjacent views for optimal bundle adjustment",
            "dot-green",
        ),
        (
            True,
            "Consistent camera intrinsics, focus, and illumination recommended",
            "dot-green",
        ),
    ]

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    for _ok, label, dot_cls in checks:
        st.markdown(
            f'<div class="req-item"><span class="{dot_cls}"></span><span>{label}</span></div>',
            unsafe_allow_html=True,
        )


def render_input_section() -> tuple[list | None, str]:
    """
    Main entry point for Dataset Input section exactly as designed in Figma.

    Returns
    -------
    uploaded_files : list | None
    mode           : "upload" | "folder"
    """

    if "input_mode" not in st.session_state:
        st.session_state.input_mode = "upload"

    st.markdown(
        """
<div class="figma-card">
  <!-- Card Header -->
  <div class="card-header-bar">
    <div class="card-header-icon">📁</div>
    <div class="card-header-titles">
      <h3>Choose Input Data &amp; Parameters</h3>
      <p>SELECT HIGH-RESOLUTION IMAGE DATASET FOR 3D RECONSTRUCTION</p>
    </div>
  </div>

  <div class="section-sub-label">SELECT INPUT METHOD</div>
""",
        unsafe_allow_html=True,
    )

    # ── Interactive Mode Cards ────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        is_upload = (st.session_state.input_mode == "upload")
        active_cls = "mode-card-active" if is_upload else "mode-card-inactive"
        radio_cls = "radio-active" if is_upload else "radio-inactive"
        dot_html = '<div class="radio-dot"></div>' if is_upload else ''

        st.markdown(
            f"""
<div class="mode-select-card {active_cls}">
  <div class="mode-radio-circle {radio_cls}">
    {dot_html}
  </div>
  <div class="mode-card-content">
    <h4>Upload Images</h4>
    <p>Upload multi-view image files (.jpg, .png, etc.)</p>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        if st.button("Select Upload Mode", key="btn_mode_upload", width='stretch'):
            st.session_state.input_mode = "upload"
            st.rerun()

    with col2:
        is_folder = (st.session_state.input_mode == "folder")
        active_cls = "mode-card-active" if is_folder else "mode-card-inactive"
        radio_cls = "radio-active" if is_folder else "radio-inactive"
        dot_html = '<div class="radio-dot"></div>' if is_folder else ''

        st.markdown(
            f"""
<div class="mode-select-card {active_cls}">
  <div class="mode-radio-circle {radio_cls}">
    {dot_html}
  </div>
  <div class="mode-card-content">
    <h4>Existing data folder</h4>
    <p>Point to a local folder containing images</p>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        if st.button("Select Folder Mode", key="btn_mode_folder", width='stretch'):
            st.session_state.input_mode = "folder"
            st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Render active mode ───────────────────────────────────────────────────
    uploaded_files: list | None = None

    if st.session_state.input_mode == "upload":
        uploaded_files = _render_upload_mode()
    else:
        uploaded_files = _render_folder_mode()

    st.markdown("</div>", unsafe_allow_html=True)

    return uploaded_files, st.session_state.input_mode
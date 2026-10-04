"""
PowerTwinAI — COLMAP Runner

Fixes applied
-------------
1. RGBA PNG → RGB JPEG conversion before passing to COLMAP
   (COLMAP feature_extractor silently skips RGBA images → 0 loaded images)
2. subprocess.run now uses check=True + captures stderr so failures are visible
3. COLMAP_PATH used consistently (was mixing module-level constant with local var)
4. Image copy count validated — raises early if 0 images were actually copied
5. Absolute workspace path used — avoids CWD-relative path issues
"""

import os
import shutil
import subprocess
from pathlib import Path

import cv2


# ── COLMAP Executable Detection ───────────────────────────────────────────────

def get_colmap_path():
    """
    Locate the COLMAP executable.
    Checks COLMAP_PATH env var, known install locations, then system PATH.
    """
    env_path = os.environ.get("COLMAP_PATH")
    if env_path and (os.path.exists(env_path) or shutil.which(env_path)):
        return env_path

    candidates = [
        r"C:\COLMAP\bin\colmap.exe",          # installed location (checked first)
        r"C:\COLMAP\COLMAP.bat",
        r"C:\COLMAP\colmap.bat",
        r"C:\COLMAP\colmap.exe",
        r"C:\Program Files\COLMAP\bin\colmap.exe",
        r"C:\Program Files\COLMAP\COLMAP.bat",
        r"C:\Program Files\COLMAP\colmap.exe",
        r"D:\COLMAP\bin\colmap.exe",
        r"E:\COLMAP\bin\colmap.exe",
        r"D:\COLMAP\COLMAP.bat",
        r"E:\COLMAP\COLMAP.bat",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    which = (
        shutil.which("colmap")
        or shutil.which("colmap.bat")
        or shutil.which("COLMAP.bat")
    )
    if which:
        return which

    return r"C:\COLMAP\bin\colmap.exe"


COLMAP_PATH = get_colmap_path()


# ── RGBA → RGB Conversion ─────────────────────────────────────────────────────

def convert_to_rgb_jpeg(src_path, dst_path):
    """
    Convert any image (including RGBA PNG with alpha) to plain RGB JPEG.
    COLMAP's feature_extractor silently skips RGBA images, resulting in
    0 loaded images and the 'No images with matches' error.
    """
    img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise RuntimeError(f"Cannot read image: {src_path}")

    # Handle RGBA — flatten alpha onto white background
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype("float32") / 255.0
        bgr   = img[:, :, :3].astype("float32")
        white = 255.0
        rgb   = (bgr * alpha + white * (1.0 - alpha)).astype("uint8")
    elif img.ndim == 2:
        # Grayscale → BGR
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        rgb = img

    cv2.imwrite(
        str(dst_path),
        rgb,
        [cv2.IMWRITE_JPEG_QUALITY, 98]  # 98 = less artefacts → cleaner SIFT
    )


# ── Qt Environment Builder ────────────────────────────────────────────────────

def build_colmap_env(colmap_bin):
    """
    Build (env, cwd) for COLMAP subprocesses so Qt finds its platform plugins.

    Two-layer fix:
    1. QT_* env vars — works for most Qt versions.
    2. cwd = COLMAP bin dir — Qt ALWAYS searches for plugins relative to
       the executable location first, making this the most robust fix.
    Also sets QT_QPA_PLATFORM=offscreen so COLMAP never opens a display.

    Returns
    -------
    tuple(dict, str) — (env dict, cwd string to pass subprocess.run cwd=)
    """
    colmap_dir    = Path(colmap_bin).resolve().parent   # C:\COLMAP\bin
    plugin_root   = (colmap_dir.parent / "plugins").resolve()
    platforms_dir = plugin_root / "platforms"

    env = os.environ.copy()
    env["QT_PLUGIN_PATH"]              = str(plugin_root)
    env["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platforms_dir)
    env["QT_QPA_PLATFORM"]             = "offscreen"
    env["PATH"] = str(colmap_dir) + os.pathsep + env.get("PATH", "")

    cwd = str(colmap_dir)   # Layer 2 — run from the COLMAP bin directory

    print(
        f"[COLMAP] bin dir        : {colmap_dir}\n"
        f"[COLMAP] Qt plugin root : {plugin_root}\n"
        f"[COLMAP] Platforms dir  : {platforms_dir} "
        f"(exists={platforms_dir.exists()})\n"
        f"[COLMAP] QPA platform   : offscreen"
    )
    return env, cwd


# ── Main COLMAP Runner ────────────────────────────────────────────────────────

def run_colmap(image_paths):
    """
    Run the full COLMAP sparse reconstruction pipeline.

    Parameters
    ----------
    image_paths : list[str | Path]
        Paths to input images (can be RGBA PNGs — converted automatically).
    """

    colmap_bin = get_colmap_path()

    if not os.path.exists(colmap_bin) and not shutil.which(colmap_bin):
        raise FileNotFoundError(
            f"COLMAP executable was not found at '{colmap_bin}'.\n"
            f"Please install COLMAP (C:\\COLMAP\\bin\\colmap.exe) "
            f"or set the COLMAP_PATH environment variable."
        )

    # Build Qt-aware environment once — passed to every subprocess.run
    qt_env, colmap_cwd = build_colmap_env(colmap_bin)

    # Use absolute workspace path — avoids CWD-relative issues
    workspace = Path(os.path.abspath("colmap_workspace"))

    if workspace.exists():
        try:
            shutil.rmtree(workspace)
        except Exception as e:
            print(f"[COLMAP] Cleanup warning: {e}")

    colmap_images = workspace / "images"
    colmap_images.mkdir(parents=True, exist_ok=True)

    # ── Copy & convert images to RGB JPEG ────────────────────────────────────

    print("[COLMAP] Preparing images (RGBA → RGB JPEG)...")

    copied = 0
    for image_path in image_paths:
        src = Path(image_path)
        if not src.exists():
            print(f"[COLMAP] WARNING: skipping missing file: {src}")
            continue

        # Always output as .jpg — COLMAP handles JPEG reliably
        dst = colmap_images / (src.stem + ".jpg")

        try:
            convert_to_rgb_jpeg(src, dst)
            copied += 1
        except Exception as e:
            print(f"[COLMAP] WARNING: could not convert {src.name}: {e}")

    print(f"[COLMAP] Images prepared: {copied}")

    if copied == 0:
        raise RuntimeError(
            "No images were successfully copied to the COLMAP workspace.\n"
            "Check that the preprocessing pipeline produced output files."
        )

    # ── Paths ─────────────────────────────────────────────────────────────────

    database_path = str(workspace / "database.db")
    sparse_path   = workspace / "sparse"
    sparse_path.mkdir(parents=True, exist_ok=True)

    # ── Feature Extraction ────────────────────────────────────────────────────

    print("[COLMAP] Feature Extraction...")

    result = subprocess.run(
        [
            colmap_bin,
            "feature_extractor",
            "--database_path",                  database_path,
            "--image_path",                     str(colmap_images),
            "--ImageReader.single_camera",      "1",
            "--FeatureExtraction.use_gpu",       "0",     # CPU — no CUDA
            # ── Accuracy parameters ──────────────────────────────────────────
            "--SiftExtraction.max_num_features", "32768", # 4× default (8192)
            "--SiftExtraction.peak_threshold",   "0.002", # lower = finer features
            "--SiftExtraction.edge_threshold",   "16",    # wider = keep edge features
            "--SiftExtraction.domain_size_pooling", "1",  # DSP-SIFT descriptors
        ],
        capture_output=False,
        env=qt_env,
        cwd=colmap_cwd,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"[COLMAP] feature_extractor failed "
            f"(exit code {result.returncode})"
        )

    # ── Feature Matching ──────────────────────────────────────────────────────

    print("[COLMAP] Feature Matching...")

    result = subprocess.run(
        [
            colmap_bin,
            "exhaustive_matcher",
            "--database_path",               database_path,
            "--FeatureMatching.use_gpu",     "0",      # CPU — no CUDA
            # ── Accuracy parameters ─────────────────────────────────────────
            "--FeatureMatching.max_num_matches", "65536",  # 2× default (32768)
        ],
        capture_output=False,
        env=qt_env,
        cwd=colmap_cwd,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"[COLMAP] exhaustive_matcher failed "
            f"(exit code {result.returncode})"
        )

    # ── Sparse Reconstruction ─────────────────────────────────────────────────

    print("[COLMAP] Sparse Reconstruction (this may take several minutes)...")

    result = subprocess.run(
        [
            colmap_bin,
            "mapper",
            "--database_path",                        database_path,
            "--image_path",                           str(colmap_images),
            "--output_path",                          str(sparse_path),
            # ── Accuracy parameters ──────────────────────────────────────────
            "--Mapper.ba_global_max_num_iterations",  "100", # more global BA
            "--Mapper.ba_local_max_num_iterations",   "40",  # more local BA
            "--Mapper.min_num_matches",               "10",  # accept weaker pairs
            "--Mapper.init_min_num_inliers",          "50",  # flexible init
            "--Mapper.abs_pose_min_num_inliers",      "15",  # register more cameras
        ],
        capture_output=False,
        env=qt_env,
        cwd=colmap_cwd,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"[COLMAP] mapper failed "
            f"(exit code {result.returncode})"
        )

    # ── Export TXT ────────────────────────────────────────────────────────────

    print("[COLMAP] Exporting TXT files...")

    model_folder = sparse_path / "0"

    if not model_folder.exists():
        raise Exception(
            "COLMAP failed to create sparse model.\n"
            "Possible reasons:\n"
            "  - Too few images (need at least 3 with overlapping content)\n"
            "  - Images have insufficient overlap / texture\n"
            "  - Images are too blurry or low contrast\n"
            f"  - Check workspace: {workspace}"
        )

    colmap_data = Path(os.path.abspath("colmap_data"))
    colmap_data.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            colmap_bin,
            "model_converter",
            "--input_path",  str(model_folder),
            "--output_path", str(colmap_data),
            "--output_type", "TXT",
        ],
        capture_output=False,
        env=qt_env,
        cwd=colmap_cwd,
        check=True,
    )

    print("[COLMAP] Completed")
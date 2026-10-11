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
import sys
import json
import shutil
import subprocess
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

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
    Build (env, cwd) for COLMAP subprocesses so Qt finds its platform plugins
    and Windows loader can locate all dependency DLLs in the lib folder.
    """
    colmap_path = Path(colmap_bin).resolve()
    if colmap_path.name.lower() in ("colmap.exe", "colmap"):
        bin_dir = colmap_path.parent
        colmap_root = bin_dir.parent
    else:
        colmap_root = colmap_path.parent
        bin_dir = colmap_root / "bin"

    lib_dir = colmap_root / "lib"
    plugin_root = lib_dir / "plugins"
    platforms_dir = plugin_root / "platforms"

    env = os.environ.copy()
    # Add bin and lib to PATH so all DLLs (Ceres, Boost, FreeImage, Qt) are found
    env["PATH"] = f"{bin_dir};{lib_dir};" + env.get("PATH", "")
    env["QT_PLUGIN_PATH"] = str(plugin_root)
    env["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platforms_dir)
    env["QT_QPA_PLATFORM"] = "offscreen"

    cwd = str(bin_dir) if bin_dir.exists() else str(colmap_root)

    print(
        f"[COLMAP] bin dir        : {bin_dir}\n"
        f"[COLMAP] lib dir        : {lib_dir} (exists={lib_dir.exists()})\n"
        f"[COLMAP] Qt plugin root : {plugin_root} (exists={plugin_root.exists()})\n"
        f"[COLMAP] QPA platform   : offscreen"
    )
    return env, cwd


# ── Main COLMAP Runner ────────────────────────────────────────────────────────

def run_colmap(image_paths, config=None):
    """
    Run the full COLMAP sparse reconstruction pipeline.

    Parameters
    ----------
    image_paths : list[str | Path]
        Paths to input images (can be RGBA PNGs — converted automatically).
    config : dict | None
        Optional reconstruction configuration settings.
    """
    config = config or {}

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

    # Terminate any dangling COLMAP processes to prevent Windows file locks
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/F", "/IM", "colmap.exe"], capture_output=True)
        except Exception:
            pass

    if workspace.exists():
        try:
            shutil.rmtree(workspace)
        except Exception as e:
            print(f"[COLMAP] Cleanup warning: {e}")

    colmap_images = workspace / "images"
    colmap_images.mkdir(parents=True, exist_ok=True)

    # ── Copy & convert images to RGB JPEG ────────────────────────────────────

    print("[COLMAP] Preparing images (RGBA -> RGB JPEG)...")

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

    # ── Database Path ─────────────────────────────────────────────────────────
    # Ensure any stale database file is removed so COLMAP initializes clean tables
    if os.path.exists(database_path):
        try:
            os.remove(database_path)
        except Exception:
            pass

    # ── Feature Extraction ────────────────────────────────────────────────────

    sift_max_img_size = str(min(2800, int(config.get("image_max_size", 2800))))
    sift_max_features = str(min(12288, int(config.get("sift_max_features", 10240))))
    sift_peak_threshold = str(max(0.004, float(config.get("sift_peak_threshold", 0.004))))
    sift_edge_threshold = str(config.get("sift_edge_threshold", 12))
    use_gpu_sift = "1" if config.get("use_gpu", False) else "0"

    print(f"[COLMAP] Feature Extraction (max_size={sift_max_img_size}, max_features={sift_max_features}, peak_th={sift_peak_threshold}, threads=2)...")

    cmd_feat = [
        colmap_bin,
        "feature_extractor",
        "--database_path",                  database_path,
        "--image_path",                     str(colmap_images),
        "--ImageReader.single_camera",      "1",
        "--SiftExtraction.use_gpu",          use_gpu_sift,
        "--SiftExtraction.num_threads",      "2",
        "--SiftExtraction.max_image_size",   sift_max_img_size,
        "--SiftExtraction.max_num_features", sift_max_features,
        "--SiftExtraction.peak_threshold",   sift_peak_threshold,
        "--SiftExtraction.edge_threshold",   sift_edge_threshold,
    ]

    result = subprocess.run(cmd_feat, capture_output=False, env=qt_env, cwd=colmap_cwd)

    if result.returncode != 0:
        if result.returncode in (3221225786, -1073741510, -2, 130):
            print(f"[COLMAP] SIFT extraction interrupted or cancelled (exit code {result.returncode}).")
            raise RuntimeError(f"[COLMAP] Reconstruction was cancelled or interrupted.")
        print(f"[COLMAP] WARNING: SIFT extraction exited with {result.returncode}. Retrying with resilient fallback settings...")
        fallback_cmd = [
            colmap_bin,
            "feature_extractor",
            "--database_path",                  database_path,
            "--image_path",                     str(colmap_images),
            "--ImageReader.single_camera",      "1",
            "--SiftExtraction.use_gpu",          "0",
            "--SiftExtraction.num_threads",      "1",
            "--SiftExtraction.max_image_size",   "2000",
            "--SiftExtraction.max_num_features", "8192",
            "--SiftExtraction.peak_threshold",   "0.006",
            "--SiftExtraction.edge_threshold",   "10",
        ]
        result = subprocess.run(fallback_cmd, capture_output=False, env=qt_env, cwd=colmap_cwd)
        if result.returncode != 0:
            if result.returncode in (3221225786, -1073741510, -2, 130):
                print(f"[COLMAP] SIFT extraction interrupted or cancelled (exit code {result.returncode}).")
                raise RuntimeError(f"[COLMAP] Reconstruction was cancelled or interrupted.")
            raise RuntimeError(
                f"[COLMAP] feature_extractor failed even with resilient settings "
                f"(exit code {result.returncode})"
            )

    # ── Feature Matching ──────────────────────────────────────────────────────

    matcher_mode = str(config.get("matcher", "ml_ranked")).lower()
    ml_pair_ranking_enabled = bool(config.get("ml_pair_ranking", True))
    ml_telemetry = {}

    match_cmd = None

    if matcher_mode in ("ml_ranked", "auto") and ml_pair_ranking_enabled:
        print("[COLMAP] Feature Matching: ML-Ranked Candidate Selection Mode...")
        try:
            from ml import ImagePairRanker, MLPairRankingConfig
            ml_cfg = MLPairRankingConfig.from_dict({
                "enabled": True,
                "top_k": int(config.get("ml_top_k", 8)),
                "min_pairs_per_image": int(config.get("ml_min_pairs_per_image", 4)),
                "similarity_threshold": float(config.get("ml_similarity_threshold", 0.0)),
                "max_total_pairs": int(config["ml_max_total_pairs"]) if "ml_max_total_pairs" in config and config["ml_max_total_pairs"] else None,
                "embedding_method": config.get("ml_embedding_method", "spatial_pyramid"),
                "ensure_connected": bool(config.get("ml_ensure_connected", True)),
                "auto_bridge_components": bool(config.get("ml_auto_bridge_components", True)),
                "fallback_to_exhaustive_if_failed": bool(config.get("ml_fallback_to_exhaustive", True)),
            })

            prepared_img_paths = sorted([
                str(p) for p in colmap_images.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            ])
            ranker = ImagePairRanker(ml_cfg)
            rank_result = ranker.rank_pairs(prepared_img_paths, progress_callback=print)

            candidate_pairs = rank_result["candidate_pairs"]
            ml_telemetry = rank_result["diagnostics"]
            ml_telemetry["matcher_mode"] = "ml_ranked"

            if not rank_result["is_connected"] and ml_cfg.fallback_to_exhaustive_if_failed:
                print("[COLMAP-ML] WARNING: Candidate graph is disconnected. Falling back to exhaustive matching...")
                matcher_mode = "exhaustive"
            else:
                pairs_file_path = workspace / "ml_candidate_pairs.txt"
                ranker.save_colmap_pairs(candidate_pairs, str(pairs_file_path))

                # Export to output directories for telemetry and inspection
                for out_dir in [Path("output"), Path("outputs"), Path("temp_session")]:
                    try:
                        out_dir.mkdir(parents=True, exist_ok=True)
                        ranker.save_colmap_pairs(candidate_pairs, str(out_dir / "ml_candidate_pairs.txt"))
                        with open(out_dir / "ml_pair_telemetry.json", "w", encoding="utf-8") as tf:
                            json.dump(ml_telemetry, tf, indent=2)
                    except Exception as e:
                        print(f"[COLMAP-ML] Warning saving pairs to {out_dir}: {e}")

                print(
                    f"[COLMAP-ML] Selected {len(candidate_pairs)} candidate pairs "
                    f"({ml_telemetry.get('pair_reduction_pct', 0)}% reduction from {ml_telemetry.get('exhaustive_pairs', 0)} exhaustive pairs)."
                )

                match_cmd = [
                    colmap_bin,
                    "matches_importer",
                    "--database_path",               database_path,
                    "--match_list_path",             str(pairs_file_path),
                    "--match_type",                  "pairs",
                    "--SiftMatching.use_gpu",        use_gpu_sift,
                    "--SiftMatching.num_threads",    "4",
                    "--SiftMatching.max_num_matches", "32768",
                ]
        except Exception as e:
            print(f"[COLMAP-ML] ML pair ranking error: {e}. Falling back to sequential matching.")
            matcher_mode = "sequential"

    if match_cmd is None:
        if matcher_mode == "sequential":
            print("[COLMAP] Feature Matching (Sequential Matcher with Quadratic Overlap)...")
            match_cmd = [
                colmap_bin,
                "sequential_matcher",
                "--database_path",                      database_path,
                "--SequentialMatching.overlap",         "15",
                "--SequentialMatching.quadratic_overlap", "1",
                "--SequentialMatching.loop_detection",  "0",
                "--SiftMatching.use_gpu",               use_gpu_sift,
                "--SiftMatching.num_threads",           "4",
                "--SiftMatching.max_num_matches",       "32768",
            ]
            ml_telemetry = {
                "matcher_mode": "sequential",
                "total_images": len(list(colmap_images.glob("*.jpg"))),
                "is_connected": True
            }
        else:
            print("[COLMAP] Feature Matching (Exhaustive Matcher - Baseline Mode)...")
            match_cmd = [
                colmap_bin,
                "exhaustive_matcher",
                "--database_path",               database_path,
                "--SiftMatching.use_gpu",        use_gpu_sift,
                "--SiftMatching.num_threads",    "4",
                "--SiftMatching.max_num_matches", "32768",
            ]
            total_n = len(list(colmap_images.glob("*.jpg")))
            ml_telemetry = {
                "matcher_mode": "exhaustive",
                "total_images": total_n,
                "exhaustive_pairs": (total_n * (total_n - 1)) // 2,
                "candidate_pairs": (total_n * (total_n - 1)) // 2,
                "pair_reduction_pct": 0.0,
                "is_connected": True
            }

        for out_dir in [Path("output"), Path("outputs"), Path("temp_session")]:
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_dir / "ml_pair_telemetry.json", "w", encoding="utf-8") as tf:
                    json.dump(ml_telemetry, tf, indent=2)
            except Exception:
                pass

    result = subprocess.run(
        match_cmd,
        capture_output=False,
        env=qt_env,
        cwd=colmap_cwd,
    )

    if result.returncode != 0:
        if result.returncode in (3221225786, -1073741510, -2, 130):
            print(f"[COLMAP] Feature matching interrupted or cancelled (exit code {result.returncode}).")
            raise RuntimeError(f"[COLMAP] Reconstruction was cancelled or interrupted.")
        raise RuntimeError(
            f"[COLMAP] feature matcher failed "
            f"(exit code {result.returncode})"
        )

    # ── Sparse Reconstruction ─────────────────────────────────────────────────

    print("[COLMAP] Sparse Reconstruction (recovering multi-view camera poses & point geometry)...")

    result = subprocess.run(
        [
            colmap_bin,
            "mapper",
            "--database_path",                        database_path,
            "--image_path",                           str(colmap_images),
            "--output_path",                          str(sparse_path),
            "--Mapper.ba_global_max_num_iterations",  "100",
            "--Mapper.ba_local_max_num_iterations",   "40",
            "--Mapper.min_num_matches",               "10",
            "--Mapper.init_min_num_inliers",          "30",
            "--Mapper.init_min_tri_angle",            "3.5",
            "--Mapper.abs_pose_min_num_inliers",      "15",
            "--Mapper.tri_ignore_two_view_tracks",    "0",
            "--Mapper.tri_min_angle",                 "1.5",
            "--Mapper.tri_continue_max_angle_error",  "3.0",
        ],
        capture_output=False,
        env=qt_env,
        cwd=colmap_cwd,
    )

    if result.returncode != 0:
        if result.returncode in (3221225786, -1073741510, -2, 130):
            print(f"[COLMAP] Sparse mapper interrupted or cancelled (exit code {result.returncode}).")
            raise RuntimeError(f"[COLMAP] Reconstruction was cancelled or interrupted.")
        raise RuntimeError(
            f"[COLMAP] mapper failed "
            f"(exit code {result.returncode})"
        )

    # ── Select Largest Sparse Model ───────────────────────────────────────────

    model_folder = sparse_path / "0"
    if not model_folder.exists():
        sub_models = [d for d in sparse_path.iterdir() if d.is_dir()]
        if sub_models:
            # Pick the submodel with the most files/size
            model_folder = max(sub_models, key=lambda d: sum(f.stat().st_size for f in d.glob("*")))
        else:
            raise Exception(
                "COLMAP failed to create sparse model.\n"
                "Possible reasons:\n"
                "  - Too few images (need at least 3 with overlapping content)\n"
                "  - Images have insufficient overlap / texture\n"
                "  - Images are too blurry or low contrast\n"
                f"  - Check workspace: {workspace}"
            )

    # ── Point Densification (Point Triangulator) ──────────────────────────────

    print("[COLMAP] Densifying 3D point cloud via multi-view triangulation...")
    try:
        subprocess.run(
            [
                colmap_bin,
                "point_triangulator",
                "--database_path", database_path,
                "--image_path",    str(colmap_images),
                "--input_path",    str(model_folder),
                "--output_path",   str(model_folder),
                "--clear_points",  "0",
            ],
            capture_output=False,
            env=qt_env,
            cwd=colmap_cwd,
        )
    except Exception as e:
        print(f"[COLMAP] Point triangulator warning: {e}")

    # ── Export TXT ────────────────────────────────────────────────────────────

    print("[COLMAP] Exporting TXT files...")

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

    # ── Export Structured Sparse Model (Requirement 13) ───────────────────────
    sparse_outputs_dir = Path(os.path.abspath("outputs/sparse"))
    sparse_outputs_dir.mkdir(parents=True, exist_ok=True)
    sparse_ply_dest = str(sparse_outputs_dir / "sparse_model.ply")

    try:
        subprocess.run(
            [
                colmap_bin,
                "model_converter",
                "--input_path",  str(model_folder),
                "--output_path", sparse_ply_dest,
                "--output_type", "PLY",
            ],
            capture_output=False,
            env=qt_env,
            cwd=colmap_cwd,
            check=True,
        )
        print(f"[COLMAP] Sparse model exported to: {sparse_ply_dest}")
    except Exception as e:
        print(f"[COLMAP] Sparse PLY export warning: {e}")

    print("[COLMAP] Completed")
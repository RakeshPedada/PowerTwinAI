import os
import sys
import shutil
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

COLMAP_PATH = r"E:\COLMAP\COLMAP.bat"

INPUT_IMAGES = (
    PROJECT_ROOT
    / "results"
    / "baselines"
    / "reconstruction_baseline_v1"
    / "artifacts"
    / "input_images"
)

RUN_ROOT = (
    PROJECT_ROOT
    / "results"
    / "baselines"
    / "reconstruction_baseline_v1"
    / "artifacts"
    / "controlled_run"
)

WORKSPACE = RUN_ROOT / "colmap_workspace"
COLMAP_DATA = RUN_ROOT / "colmap_data"
LOG_DIR = (
    PROJECT_ROOT
    / "results"
    / "baselines"
    / "reconstruction_baseline_v1"
    / "logs"
)


def run_command(command):
    print("\n" + "=" * 70)
    print("RUNNING:")
    print(" ".join(str(x) for x in command))
    print("=" * 70)

    subprocess.run(
        [str(x) for x in command],
        check=True
    )


def count_registered_images(model_path):
    images_bin = model_path / "images.bin"

    if not images_bin.exists():
        raise FileNotFoundError(
            f"images.bin not found: {images_bin}"
        )

    return images_bin


def select_largest_sparse_model(sparse_path):
    best_model = None
    best_size = -1

    for item in sparse_path.iterdir():
        if not item.is_dir():
            continue

        images_bin = item / "images.bin"

        if not images_bin.exists():
            continue

        size = images_bin.stat().st_size

        if size > best_size:
            best_size = size
            best_model = item

    if best_model is None:
        raise RuntimeError(
            "Unable to determine the largest sparse model."
        )

    print(f"[BASELINE] Selected sparse model: {best_model}")

    return best_model


def main():
    start_time = time.perf_counter()

    if not INPUT_IMAGES.is_dir():
        raise FileNotFoundError(
            f"Input images not found: {INPUT_IMAGES}"
        )

    image_files = [
        p for p in INPUT_IMAGES.iterdir()
        if p.is_file()
    ]

    print(f"[BASELINE] Input images: {len(image_files)}")

    if len(image_files) != 67:
        raise RuntimeError(
            f"Expected 67 images, found {len(image_files)}"
        )

    if RUN_ROOT.exists():
        raise RuntimeError(
            f"Controlled run directory already exists: {RUN_ROOT}\n"
            "Do not overwrite an existing baseline run."
        )

    RUN_ROOT.mkdir(parents=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    WORKSPACE.mkdir()

    workspace_images = WORKSPACE / "images"

    shutil.copytree(
        INPUT_IMAGES,
        workspace_images
    )

    database_path = WORKSPACE / "database.db"
    sparse_path = WORKSPACE / "sparse"

    sparse_path.mkdir()

    print("[BASELINE] Feature Extraction")

    run_command([
        COLMAP_PATH,
        "feature_extractor",
        "--database_path",
        database_path,
        "--image_path",
        workspace_images,
    ])

    print("[BASELINE] Feature Matching")

    run_command([
        COLMAP_PATH,
        "exhaustive_matcher",
        "--database_path",
        database_path,
    ])

    print("[BASELINE] Sparse Reconstruction")

    run_command([
        COLMAP_PATH,
        "mapper",
        "--database_path",
        database_path,
        "--image_path",
        workspace_images,
        "--output_path",
        sparse_path,
    ])

    sparse_model = select_largest_sparse_model(
        sparse_path
    )

    print("[BASELINE] Exporting sparse model")

    COLMAP_DATA.mkdir()

    run_command([
        COLMAP_PATH,
        "model_converter",
        "--input_path",
        sparse_model,
        "--output_path",
        COLMAP_DATA,
        "--output_type",
        "TXT",
    ])

    dense_path = WORKSPACE / "dense"

    print("[BASELINE] Image Undistortion")

    run_command([
        COLMAP_PATH,
        "image_undistorter",
        "--image_path",
        workspace_images,
        "--input_path",
        sparse_model,
        "--output_path",
        dense_path,
        "--output_type",
        "COLMAP",
    ])

    print("[BASELINE] PatchMatch Stereo")

    run_command([
        COLMAP_PATH,
        "patch_match_stereo",
        "--workspace_path",
        dense_path,
        "--workspace_format",
        "COLMAP",
        "--PatchMatchStereo.geom_consistency",
        "true",
    ])

    fused_path = dense_path / "fused_geometric.ply"

    print("[BASELINE] Stereo Fusion")

    run_command([
        COLMAP_PATH,
        "stereo_fusion",
        "--workspace_path",
        dense_path,
        "--workspace_format",
        "COLMAP",
        "--input_type",
        "geometric",
        "--output_type",
        "PLY",
        "--output_path",
        fused_path,
        "--StereoFusion.min_num_pixels",
        "2",
    ])

    if not fused_path.exists():
        raise RuntimeError(
            "fused_geometric.ply was not generated."
        )

    total_time = (
        time.perf_counter()
        - start_time
    )

    print("\n" + "=" * 70)
    print("[BASELINE] CONTROLLED COLMAP RUN COMPLETE")
    print("=" * 70)
    print(f"[BASELINE] Dense output: {fused_path}")
    print(f"[BASELINE] Total time: {total_time:.2f} sec")


if __name__ == "__main__":
    main()
"""
PowerTwinAI

Preprocessing Manager

Runs all preprocessing modules in sequence.
"""

import shutil
from pathlib import Path

from preprocessing.background_removal import BackgroundRemover
from preprocessing.image_resize import ImageResizer
from preprocessing.image_quality import analyze_image_quality


def preprocess_images(
    image_paths,
    progress_callback=print
):
    progress_callback(
        "[PREPROCESS] Initializing..."
    )

    temp_root = Path("temp_session")
    original_dir = temp_root / "original"
    resized_dir = temp_root / "resized"
    processed_dir = temp_root / "processed"

    for folder in [
        original_dir,
        resized_dir,
        processed_dir
    ]:
        if folder.exists():
            shutil.rmtree(folder)

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

    progress_callback(
        "[PREPROCESS] Image Quality Assessment..."
    )

    copied_paths = []
    quality_results = []

    for image_path in image_paths:
        quality_result = analyze_image_quality(image_path)

        quality_results.append(quality_result)

        progress_callback(
            f"[QUALITY] {Path(image_path).name}: "
            f"{quality_result['status']}"
        )

        for warning in quality_result["warnings"]:
            progress_callback(
                f"[QUALITY WARNING] "
                f"{Path(image_path).name}: {warning}"
            )

        for error in quality_result["errors"]:
            progress_callback(
                f"[QUALITY FAIL] "
                f"{Path(image_path).name}: {error}"
            )

        if quality_result["status"] == "FAIL":
            progress_callback(
                f"[QUALITY] Skipping "
                f"{Path(image_path).name}"
            )
            continue

        destination = (
            original_dir /
            Path(image_path).name
        )

        shutil.copy2(
            image_path,
            destination
        )

        copied_paths.append(
            str(destination)
        )

    progress_callback(
        "[PREPROCESS] Image Resize..."
    )

    ImageResizer().process_folder(
        original_dir,
        resized_dir
    )

    progress_callback(
        "[PREPROCESS] Background Removal..."
    )

    processed_paths = (
        BackgroundRemover().process_folder(
            resized_dir,
            processed_dir
        )
    )

    progress_callback(
        "[PREPROCESS] Completed"
    )

    return processed_paths
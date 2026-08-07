"""
PowerTwinAI
Preprocessing Manager

Runs all preprocessing modules in sequence.
"""

import os
import shutil
from pathlib import Path

from preprocessing.background_removal import BackgroundRemover
from preprocessing.image_resize import ImageResizer


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

    # Clean previous run

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
        "[PREPROCESS] Copying images..."
    )

    copied_paths = []

    for image_path in image_paths:

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
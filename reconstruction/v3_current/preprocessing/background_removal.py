"""
PowerTwinAI
Background Removal Module

Author:
PowerTwinAI Team

Description
-----------
Removes image backgrounds using rembg and saves the
processed images for reconstruction.

This module is independent of the reconstruction pipeline
and can be reused anywhere.

Enhancements
------------
- Cached rembg session (singleton) to avoid reloading ONNX model
- Force PNG output to preserve alpha channel
- progress_callback support for UI integration
- Per-image error handling with fallback (skip bad image)
- Sorted + deterministic file ordering
- WebP support added
"""

from pathlib import Path

import cv2
import numpy as np
from rembg import remove, new_session


SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]


class BackgroundRemover:

    # Class-level session cache — loaded once, reused across all instances
    _session = None

    def __init__(
        self,
        model_name: str = "u2net",
        progress_callback=print
    ):
        self.progress_callback = progress_callback

        if BackgroundRemover._session is None:
            self.progress_callback(
                f"[BG] Loading rembg model: {model_name} ..."
            )
            BackgroundRemover._session = new_session(model_name)
            self.progress_callback(
                "[BG] rembg model loaded and cached"
            )
        else:
            self.progress_callback(
                "[BG] Reusing cached rembg session"
            )

        self.session = BackgroundRemover._session

    def remove_background(self, image):

        if image is None:
            raise ValueError("Input image is None.")

        success, encoded = cv2.imencode(".png", image)

        if not success:
            raise RuntimeError("Unable to encode image.")

        # Pass cached session — no ONNX reload
        output = remove(
            encoded.tobytes(),
            session=self.session
        )

        array = np.frombuffer(
            output,
            dtype=np.uint8
        )

        result = cv2.imdecode(
            array,
            cv2.IMREAD_UNCHANGED
        )

        return result

    def process_image(
        self,
        input_path,
        output_path
    ):

        image = cv2.imread(
            str(input_path),
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise RuntimeError(
                f"Cannot read image:\n{input_path}"
            )

        result = self.remove_background(image)

        # Force PNG output to preserve alpha transparency
        output_path = Path(output_path).with_suffix(".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(output_path), result)

        return str(output_path)

    def process_folder(
        self,
        input_folder,
        output_folder
    ):

        input_folder = Path(input_folder)
        output_folder = Path(output_folder)

        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        processed_paths = []
        failed_paths = []

        image_files = sorted([
            f
            for ext in SUPPORTED_EXTENSIONS
            for f in input_folder.glob(f"*{ext}")
        ])

        total = len(image_files)

        self.progress_callback(
            f"[BG] Images Found: {total}"
        )

        for index, image_path in enumerate(image_files):

            # Always output as PNG for alpha channel
            output_path = (
                output_folder /
                (image_path.stem + ".png")
            )

            self.progress_callback(
                f"[BG] ({index + 1}/{total}) "
                f"{image_path.name}"
            )

            try:
                self.process_image(
                    image_path,
                    output_path
                )
                processed_paths.append(str(output_path))

            except Exception as e:
                self.progress_callback(
                    f"[BG ERROR] {image_path.name}: {e} — skipping"
                )
                failed_paths.append(str(image_path))

        self.progress_callback(
            f"[BG] Completed — "
            f"{len(processed_paths)} succeeded, "
            f"{len(failed_paths)} failed"
        )

        return processed_paths, failed_paths
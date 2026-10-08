"""
PowerTwinAI
Image Resize Module

Author:
PowerTwinAI Team

Description
-----------
Resizes large images while preserving aspect ratio.
Smaller images are kept unchanged.

Enhancements
------------
- max_height parameter added alongside max_width
- Scale computed from both axes — whichever is the limiting constraint
- Sorted file ordering for deterministic reconstruction
- WebP support added to extensions
- progress_callback support for UI integration
- Per-image error handling — bad image is skipped, not a crash
"""

from pathlib import Path

import cv2


SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]


class ImageResizer:

    def __init__(
        self,
        max_width: int = 4096,
        max_height: int = 4096,
        progress_callback=print
    ):
        self.max_width = max_width
        self.max_height = max_height
        self.progress_callback = progress_callback

        self.progress_callback(
            f"[RESIZE] Max Size Constraint: {self.max_width}x{self.max_height}"
        )

    def resize_image(self, image):

        height, width = image.shape[:2]

        # Scale factor limited by both axes — takes the smaller scale
        scale = min(
            self.max_width / width,
            self.max_height / height,
            1.0           # Never upscale
        )

        if scale == 1.0:
            return image

        new_width = int(width * scale)
        new_height = int(height * scale)

        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

        return resized

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

        resized = self.resize_image(image)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        extension = output_path.suffix.lower()

        if extension in [".jpg", ".jpeg"]:
            cv2.imwrite(
                str(output_path),
                resized,
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )

        elif extension == ".webp":
            cv2.imwrite(
                str(output_path),
                resized,
                [cv2.IMWRITE_WEBP_QUALITY, 95]
            )

        else:
            cv2.imwrite(str(output_path), resized)

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

        # Sorted for deterministic ordering across runs
        image_files = sorted([
            f
            for ext in SUPPORTED_EXTENSIONS
            for f in input_folder.glob(f"*{ext}")
        ])

        total = len(image_files)

        self.progress_callback(
            f"[RESIZE] Images Found: {total}"
        )

        for index, image_path in enumerate(image_files):

            output_path = output_folder / image_path.name

            self.progress_callback(
                f"[RESIZE] ({index + 1}/{total}) "
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
                    f"[RESIZE ERROR] {image_path.name}: {e} — skipping"
                )
                failed_paths.append(str(image_path))

        self.progress_callback(
            f"[RESIZE] Completed — "
            f"{len(processed_paths)} succeeded, "
            f"{len(failed_paths)} failed"
        )

        return processed_paths, failed_paths
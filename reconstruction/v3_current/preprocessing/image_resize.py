"""
PowerTwinAI
Image Resize Module

Author:
PowerTwinAI Team

Description
-----------
Resizes large images while preserving aspect ratio.
Smaller images are kept unchanged.
"""

import os
from pathlib import Path

import cv2


class ImageResizer:

    def __init__(self, max_width=1600):

        self.max_width = max_width

        print(
            f"[RESIZE] Max Width : {self.max_width}"
        )

    def resize_image(self, image):

        height, width = image.shape[:2]

        if width <= self.max_width:

            return image

        scale = self.max_width / float(width)

        new_width = self.max_width
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

        os.makedirs(
            Path(output_path).parent,
            exist_ok=True
        )

        cv2.imwrite(
            str(output_path),
            resized,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95
            ]
        )

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

        extensions = [
            ".jpg",
            ".jpeg",
            ".png"
        ]

        image_files = []

        for ext in extensions:

            image_files.extend(
                input_folder.glob(f"*{ext}")
            )

        print(
            f"[RESIZE] Images Found : {len(image_files)}"
        )

        for index, image_path in enumerate(image_files):

            output_path = (
                output_folder /
                image_path.name
            )

            print(
                f"[RESIZE] ({index+1}/{len(image_files)}) "
                f"{image_path.name}"
            )

            self.process_image(
                image_path,
                output_path
            )

            processed_paths.append(
                str(output_path)
            )

        print(
            "[RESIZE] Completed"
        )

        return processed_paths
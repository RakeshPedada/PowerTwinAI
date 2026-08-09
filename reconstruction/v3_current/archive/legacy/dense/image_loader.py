"""
Image Loader Module

Objective
---------
This module is responsible for loading and validating image pairs
used during dense reconstruction.

Responsibilities
----------------
- Load two images from disk.
- Verify that both images exist.
- Ensure both images have identical dimensions.
- Return validated image pairs.

This separation keeps dense_reconstruction.py focused on the
dense reconstruction workflow rather than file I/O operations.

Part of:
PowerTwinAI - Phase 6 Modularization
"""

import cv2


def load_image_pair(image1_path, image2_path):
    """
    Load and validate an image pair.

    Parameters
    ----------
    image1_path : str
        Path to the first image.

    image2_path : str
        Path to the second image.

    Returns
    -------
    tuple
        (img1, img2)

    Raises
    ------
    FileNotFoundError
        If either image cannot be loaded.

    ValueError
        If the images have different dimensions.
    """

    img1 = cv2.imread(
        image1_path,
        cv2.IMREAD_COLOR
    )

    img2 = cv2.imread(
        image2_path,
        cv2.IMREAD_COLOR
    )

    if img1 is None:

        raise FileNotFoundError(
            f"Unable to load image:\n{image1_path}"
        )

    if img2 is None:

        raise FileNotFoundError(
            f"Unable to load image:\n{image2_path}"
        )

    if img1.shape[:2] != img2.shape[:2]:

        raise ValueError(
            "Image pair dimensions do not match."
        )

    return img1, img2
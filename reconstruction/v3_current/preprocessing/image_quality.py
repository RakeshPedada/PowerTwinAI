"""
PowerTwinAI
Image Quality Module

Author:
PowerTwinAI Team

Description
------ check the quality of images for preprocessing and reconstruction.-----
"""

from pathlib import Path

import numpy as np
from PIL import Image


VALID_FORMATS = {"jpg", "jpeg", "png"}

MIN_WIDTH = 640
MIN_HEIGHT = 480

SHARPNESS_FAIL = 20
SHARPNESS_WARNING = 50

BRIGHTNESS_FAIL_LOW = 30
BRIGHTNESS_WARNING_LOW = 50
BRIGHTNESS_WARNING_HIGH = 220
BRIGHTNESS_FAIL_HIGH = 245


def check_image_readable(image_path):
    try:
        with Image.open(image_path) as image:
            image.verify()
        return True
    except Exception:
        return False


def check_image_format(image_path):
    extension = Path(image_path).suffix.lower().replace(".", "")
    return extension in VALID_FORMATS


def get_sharpness(image):
    gray = np.asarray(image.convert("L"), dtype=np.float32)

    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0

    laplacian = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4 * gray[1:-1, 1:-1]
    )

    return float(np.var(laplacian))


def get_brightness(image):
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    return float(np.mean(gray))


def analyze_image_quality(image_path):
    result = {
        "image": str(image_path),
        "status": "PASS",
        "readable": False,
        "format_valid": False,
        "width": 0,
        "height": 0,
        "sharpness": 0.0,
        "brightness": 0.0,
        "warnings": [],
        "errors": []
    }

    path = Path(image_path)

    if not path.exists():
        result["status"] = "FAIL"
        result["errors"].append("Image file does not exist")
        return result

    if not check_image_format(path):
        result["status"] = "FAIL"
        result["errors"].append("Invalid image format")
        return result

    result["format_valid"] = True

    if not check_image_readable(path):
        result["status"] = "FAIL"
        result["errors"].append("Image cannot be read")
        return result

    result["readable"] = True

    try:
        with Image.open(path) as image:
            width, height = image.size
            sharpness = get_sharpness(image)
            brightness = get_brightness(image)

        result["width"] = width
        result["height"] = height
        result["sharpness"] = round(sharpness, 2)
        result["brightness"] = round(brightness, 2)

    except Exception:
        result["status"] = "FAIL"
        result["errors"].append("Image analysis failed")
        return result

    if width < MIN_WIDTH or height < MIN_HEIGHT:
        result["warnings"].append("Low image resolution")

    if sharpness < SHARPNESS_FAIL:
        result["errors"].append("Image is too blurry")
    elif sharpness < SHARPNESS_WARNING:
        result["warnings"].append("Image may be blurry")

    if brightness < BRIGHTNESS_FAIL_LOW:
        result["errors"].append("Image is too dark")
    elif brightness < BRIGHTNESS_WARNING_LOW:
        result["warnings"].append("Image is dark")
    elif brightness > BRIGHTNESS_FAIL_HIGH:
        result["errors"].append("Image is overexposed")
    elif brightness > BRIGHTNESS_WARNING_HIGH:
        result["warnings"].append("Image is bright")

    if result["errors"]:
        result["status"] = "FAIL"
    elif result["warnings"]:
        result["status"] = "WARNING"

    return result
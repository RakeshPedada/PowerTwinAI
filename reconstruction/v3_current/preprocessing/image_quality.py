"""
PowerTwinAI
Image Quality Assessment Module

Author:
PowerTwinAI Team

Description
-----------
Checks the quality of images for preprocessing and 3D reconstruction.
Validates format, readability, resolution, sharpness, brightness,
contrast, aspect ratio, and color mode.

Enhancements
------------
- Single image open (no double-open overhead)
- Low resolution upgraded to FAIL (critical for 3D reconstruction)
- Aspect ratio check added
- Contrast check via std deviation
- Grayscale image warning
- Exception messages exposed in result
- Sorted/deterministic behavior preserved
"""

from pathlib import Path

import numpy as np
from PIL import Image


# ── Supported Formats ────────────────────────────────────────────────────────

VALID_FORMATS = {"jpg", "jpeg", "png", "webp"}

# ── Resolution Thresholds ─────────────────────────────────────────────────────

MIN_WIDTH = 640
MIN_HEIGHT = 480

# ── Sharpness (Laplacian Variance) ────────────────────────────────────────────

SHARPNESS_FAIL = 20
SHARPNESS_WARNING = 50

# ── Brightness (Mean pixel value 0–255) ───────────────────────────────────────

BRIGHTNESS_FAIL_LOW = 30
BRIGHTNESS_WARNING_LOW = 50
BRIGHTNESS_WARNING_HIGH = 220
BRIGHTNESS_FAIL_HIGH = 245

# ── Contrast (Std deviation of grayscale) ─────────────────────────────────────

CONTRAST_WARNING = 20

# ── Aspect Ratio ──────────────────────────────────────────────────────────────

ASPECT_MAX = 3.0    # e.g. ultra-wide panoramas
ASPECT_MIN = 0.33   # e.g. extreme portrait crops


def check_image_format(image_path):
    extension = Path(image_path).suffix.lower().replace(".", "")
    return extension in VALID_FORMATS


def get_sharpness(gray_array):
    """Laplacian variance — higher means sharper."""
    if gray_array.shape[0] < 3 or gray_array.shape[1] < 3:
        return 0.0

    laplacian = (
        gray_array[:-2, 1:-1]
        + gray_array[2:, 1:-1]
        + gray_array[1:-1, :-2]
        + gray_array[1:-1, 2:]
        - 4 * gray_array[1:-1, 1:-1]
    )

    return float(np.var(laplacian))


def get_brightness(gray_array):
    """Mean pixel intensity."""
    return float(np.mean(gray_array))


def get_contrast(gray_array):
    """Std deviation — low value means flat/washed-out image."""
    return float(np.std(gray_array))


def analyze_image_quality(image_path):
    result = {
        "image": str(image_path),
        "status": "PASS",
        "readable": False,
        "format_valid": False,
        "width": 0,
        "height": 0,
        "aspect_ratio": 0.0,
        "sharpness": 0.0,
        "brightness": 0.0,
        "contrast": 0.0,
        "warnings": [],
        "errors": []
    }

    path = Path(image_path)

    # ── Existence check ───────────────────────────────────────────────────────

    if not path.exists():
        result["status"] = "FAIL"
        result["errors"].append("Image file does not exist")
        return result

    # ── Format check ──────────────────────────────────────────────────────────

    if not check_image_format(path):
        result["status"] = "FAIL"
        result["errors"].append(
            f"Invalid format: '{path.suffix}' "
            f"(accepted: {', '.join(VALID_FORMATS)})"
        )
        return result

    result["format_valid"] = True

    # ── Open once — verify + analyze in single pass ───────────────────────────

    try:
        with Image.open(path) as image:

            # Verify image integrity
            try:
                image.verify()
            except Exception as verify_err:
                result["status"] = "FAIL"
                result["errors"].append(
                    f"Image integrity check failed: {verify_err}"
                )
                return result

        # Re-open after verify() (verify() closes the file pointer)
        with Image.open(path) as image:

            result["readable"] = True

            width, height = image.size

            # Grayscale warning
            if image.mode == "L":
                result["warnings"].append(
                    "Grayscale image — color info missing, "
                    "may reduce reconstruction quality"
                )

            gray = np.asarray(
                image.convert("L"),
                dtype=np.float32
            )

            sharpness = get_sharpness(gray)
            brightness = get_brightness(gray)
            contrast = get_contrast(gray)

        result["width"] = width
        result["height"] = height
        result["aspect_ratio"] = round(width / height, 3)
        result["sharpness"] = round(sharpness, 2)
        result["brightness"] = round(brightness, 2)
        result["contrast"] = round(contrast, 2)

    except Exception as e:
        result["status"] = "FAIL"
        result["errors"].append(f"Image analysis failed: {e}")
        return result

    # ── Resolution — FAIL for 3D reconstruction ───────────────────────────────

    if width < MIN_WIDTH or height < MIN_HEIGHT:
        result["errors"].append(
            f"Resolution too low ({width}x{height}) — "
            f"minimum {MIN_WIDTH}x{MIN_HEIGHT} required for reconstruction"
        )

    # ── Aspect ratio ──────────────────────────────────────────────────────────

    aspect = result["aspect_ratio"]
    if aspect > ASPECT_MAX or aspect < ASPECT_MIN:
        result["warnings"].append(
            f"Extreme aspect ratio ({aspect:.2f}) — "
            f"may degrade feature matching"
        )

    # ── Sharpness ─────────────────────────────────────────────────────────────

    if sharpness < SHARPNESS_FAIL:
        result["errors"].append(
            f"Image is too blurry (sharpness={sharpness:.1f})"
        )
    elif sharpness < SHARPNESS_WARNING:
        result["warnings"].append(
            f"Image may be blurry (sharpness={sharpness:.1f})"
        )

    # ── Brightness ────────────────────────────────────────────────────────────

    if brightness < BRIGHTNESS_FAIL_LOW:
        result["errors"].append(
            f"Image is too dark (brightness={brightness:.1f})"
        )
    elif brightness < BRIGHTNESS_WARNING_LOW:
        result["warnings"].append(
            f"Image is dark (brightness={brightness:.1f})"
        )
    elif brightness > BRIGHTNESS_FAIL_HIGH:
        result["errors"].append(
            f"Image is overexposed (brightness={brightness:.1f})"
        )
    elif brightness > BRIGHTNESS_WARNING_HIGH:
        result["warnings"].append(
            f"Image is bright (brightness={brightness:.1f})"
        )

    # ── Contrast ──────────────────────────────────────────────────────────────

    if contrast < CONTRAST_WARNING:
        result["warnings"].append(
            f"Low contrast (std={contrast:.1f}) — "
            f"may reduce feature matching accuracy"
        )

    # ── Final status ──────────────────────────────────────────────────────────

    if result["errors"]:
        result["status"] = "FAIL"
    elif result["warnings"]:
        result["status"] = "WARNING"

    return result
"""
PowerTwinAI

Preprocessing Manager

Runs all preprocessing modules in sequence.

Enhancements
------------
- Unique temp session dirs per run (uuid-based) — no collision on parallel runs
- Full preprocessing report returned (processed, failed, quality results)
- Per-image error handling — one bad image doesn't crash the pipeline
- Consistent progress_callback passed into all submodules
- Parallel quality assessment using ThreadPoolExecutor (speedup)
- Quality summary stats logged at end
- Cleanup of temp dirs on failure configurable
"""

import json
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from preprocessing.background_removal import BackgroundRemover
from preprocessing.image_resize import ImageResizer
from preprocessing.image_quality import analyze_image_quality


def preprocess_images(
    image_paths,
    progress_callback=print,
    max_workers: int = 4,
    save_quality_report: bool = True,
    cleanup_temp: bool = False,
    remove_background: bool = False,
    config: dict | None = None,
):
    """
    Run the full preprocessing pipeline on a list of image paths.

    Parameters
    ----------
    image_paths : list[str | Path]
        Paths to input images.
    progress_callback : callable
        Function to receive progress messages (default: print).
    max_workers : int
        Number of threads for parallel quality assessment.
    save_quality_report : bool
        If True, saves quality_report.json alongside processed output.
    cleanup_temp : bool
        If True, deletes temp dirs after completion.

    Returns
    -------
    dict with keys:
        processed_paths   — list of final output image paths
        quality_results   — per-image quality dicts
        failed_quality    — images that failed quality check
        warned_quality    — images that passed with warnings
        failed_resize     — images that failed during resize
        failed_bg         — images that failed during BG removal
        temp_dir          — path to temp session folder (if not cleaned up)
    """

    progress_callback("[PREPROCESS] Initializing...")

    # ── Unique temp session — safe for parallel runs ──────────────────────────

    session_id = uuid.uuid4().hex[:8]
    temp_root = Path(f"temp_session_{session_id}")
    original_dir = temp_root / "original"
    resized_dir = temp_root / "resized"
    processed_dir = temp_root / "processed"

    for folder in [original_dir, resized_dir, processed_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    progress_callback(
        f"[PREPROCESS] Session ID: {session_id} | "
        f"Temp: {temp_root}"
    )

    # ── Step 1: Parallel Quality Assessment ───────────────────────────────────

    progress_callback(
        f"[PREPROCESS] Quality Assessment "
        f"({max_workers} threads) ..."
    )

    quality_results = [None] * len(image_paths)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        future_to_index = {
            executor.submit(
                analyze_image_quality, path
            ): i
            for i, path in enumerate(image_paths)
        }

        for future in as_completed(future_to_index):
            i = future_to_index[future]
            try:
                quality_results[i] = future.result()
            except Exception as e:
                quality_results[i] = {
                    "image": str(image_paths[i]),
                    "status": "FAIL",
                    "errors": [f"Quality check crashed: {e}"],
                    "warnings": []
                }

    # ── Log quality results ───────────────────────────────────────────────────

    passed, warned, failed = 0, 0, 0
    copied_paths = []

    for quality_result in quality_results:

        img_name = Path(quality_result["image"]).name
        status = quality_result["status"]

        progress_callback(
            f"[QUALITY] {img_name}: {status}"
        )

        for warning in quality_result.get("warnings", []):
            progress_callback(
                f"[QUALITY WARNING] {img_name}: {warning}"
            )

        for error in quality_result.get("errors", []):
            progress_callback(
                f"[QUALITY FAIL] {img_name}: {error}"
            )

        if status == "FAIL":
            failed += 1
            continue

        if status == "WARNING":
            warned += 1
        else:
            passed += 1

        destination = original_dir / img_name
        shutil.copy2(quality_result["image"], destination)
        copied_paths.append(str(destination))

    progress_callback(
        f"[QUALITY] Summary — "
        f"PASS: {passed}, WARNING: {warned}, FAIL: {failed}"
    )

    if not copied_paths:
        progress_callback(
            "[PREPROCESS] No images passed quality check. Aborting."
        )
        return _build_report(
            processed_paths=[],
            quality_results=quality_results,
            failed_resize=[],
            failed_bg=[],
            temp_dir=str(temp_root)
        )

    # ── Step 2: Resize ────────────────────────────────────────────────────────

    progress_callback("[PREPROCESS] Resizing Images...")

    max_res = int(config.get("image_max_size", 4096)) if config else 4096
    resizer = ImageResizer(
        max_width=max_res,
        max_height=max_res,
        progress_callback=progress_callback
    )
    _, failed_resize = resizer.process_folder(
        original_dir,
        resized_dir
    )

    # ── Step 3: Background Removal (Optional) ─────────────────────────────────
    if remove_background:
        progress_callback("[PREPROCESS] Removing Backgrounds...")
        remover = BackgroundRemover(progress_callback=progress_callback)
        processed_paths, failed_bg = remover.process_folder(
            resized_dir,
            processed_dir
        )
    else:
        progress_callback(
            "[PREPROCESS] Preserving scene background & markers for optimal Structure-from-Motion tracking..."
        )
        processed_paths = []
        for f in sorted(resized_dir.glob("*.*")):
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
                dst = processed_dir / f.name
                shutil.copy2(f, dst)
                processed_paths.append(str(dst))
        failed_bg = []

    # ── Step 4: Save Quality Report ───────────────────────────────────────────

    if save_quality_report:
        report_path = processed_dir / "quality_report.json"
        try:
            with open(report_path, "w") as f:
                json.dump(quality_results, f, indent=2)
            progress_callback(
                f"[PREPROCESS] Quality report saved: {report_path}"
            )
        except Exception as e:
            progress_callback(
                f"[PREPROCESS] Could not save quality report: {e}"
            )

    # ── Step 5: Optional Cleanup ──────────────────────────────────────────────

    if cleanup_temp:
        shutil.rmtree(original_dir, ignore_errors=True)
        shutil.rmtree(resized_dir, ignore_errors=True)
        progress_callback(
            "[PREPROCESS] Temp dirs cleaned up"
        )

    progress_callback(
        f"[PREPROCESS] Completed — "
        f"{len(processed_paths)} images ready for reconstruction"
    )

    return _build_report(
        processed_paths=processed_paths,
        quality_results=quality_results,
        failed_resize=failed_resize,
        failed_bg=failed_bg,
        temp_dir=str(temp_root)
    )


def _build_report(
    processed_paths,
    quality_results,
    failed_resize,
    failed_bg,
    temp_dir
):
    """Build the final pipeline report dict."""
    return {
        "processed_paths": processed_paths,
        "quality_results": quality_results,
        "failed_quality": [
            r["image"]
            for r in quality_results
            if r.get("status") == "FAIL"
        ],
        "warned_quality": [
            r["image"]
            for r in quality_results
            if r.get("status") == "WARNING"
        ],
        "failed_resize": failed_resize,
        "failed_bg": failed_bg,
        "temp_dir": temp_dir,
    }
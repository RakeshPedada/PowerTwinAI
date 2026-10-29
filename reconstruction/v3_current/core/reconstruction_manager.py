import os
import json
import subprocess


def prepare_workspace(
    temp_dir,
    log_file,
    status_file,
    result_file
):
    # Terminate any lingering COLMAP processes to prevent SQLite file locks
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/F", "/IM", "colmap.exe"], capture_output=True)
        except Exception:
            pass

    os.makedirs(temp_dir, exist_ok=True)

    with open(
        log_file,
        "w",
        encoding="utf-8",
        errors="ignore"
    ) as f:
        f.write("")

    if os.path.exists(result_file):
        try:
            os.remove(result_file)
        except Exception:
            pass

    # Immediately initialize status file to RUNNING to guarantee UI state consistency
    try:
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump({
                "status": "RUNNING",
                "step": "Initializing Pipeline Environment...",
                "progress": 0.03,
                "message": "Starting background reconstruction worker..."
            }, f, indent=2)
    except Exception:
        pass


def save_uploaded_images(
    uploaded_files,
    mode,
    temp_dir,
    image_paths_file
):
    """
    Save images to *temp_dir* and write their paths to *image_paths_file*.

    Parameters
    ----------
    uploaded_files : list[UploadedFile] | list[str]
        Streamlit UploadedFile objects (upload mode) or plain file paths
        (folder mode).
    mode           : str  — "upload" or "folder"
    temp_dir       : str  — working directory for saved images
    image_paths_file : str — JSON file that will hold the list of paths
    """

    saved_paths: list[str] = []

    # ── Upload mode: UploadedFile objects → save bytes to disk ───────────────
    if mode == "upload":

        os.makedirs(temp_dir, exist_ok=True)

        for file in uploaded_files:

            save_path = os.path.join(temp_dir, file.name)

            with open(save_path, "wb") as f:
                f.write(file.getbuffer())

            saved_paths.append(save_path)   # always a plain str

    # ── Folder mode: already plain file paths (list[str]) ────────────────────
    else:

        for item in uploaded_files:
            # Guard: reject any stray UploadedFile that slipped through
            if isinstance(item, str):
                saved_paths.append(item)
            else:
                # Fallback: save it the same way as upload mode
                save_path = os.path.join(temp_dir, item.name)
                with open(save_path, "wb") as f:
                    f.write(item.getbuffer())
                saved_paths.append(save_path)

    # ── Persist paths as JSON (list of plain strings only) ───────────────────
    with open(image_paths_file, "w", encoding="utf-8") as f:
        json.dump(saved_paths, f, indent=2)


def launch_reconstruction(log_file=None):
    """
    Launch reconstruction_runner.py in a background process using the best
    available Python interpreter and redirect stdout/stderr to log_file.
    """
    import sys
    from config.paths import BASE_DIR, LOG_FILE

    target_log_file = log_file or LOG_FILE

    # Prefer Python 3.10 where all SfM packages (open3d, rembg, opencv) are installed
    py310_path = r"C:\Users\Del\AppData\Local\Programs\Python\Python310\python.exe"
    if os.path.exists(py310_path):
        python_bin = py310_path
    else:
        python_bin = sys.executable

    script_path = os.path.join(BASE_DIR, "reconstruction_runner.py")

    with open(target_log_file, "a", encoding="utf-8") as f:
        f.write(f"[SYSTEM] Launching Reconstruction Pipeline...\n")
        f.write(f"[SYSTEM] Python Interpreter: {python_bin}\n")
        f.write(f"[SYSTEM] Runner Script: {script_path}\n")

    log_handle = open(target_log_file, "a", encoding="utf-8")

    creation_flags = 0
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        creation_flags = subprocess.CREATE_NO_WINDOW

    proc = subprocess.Popen(
        [python_bin, "-u", script_path],
        cwd=BASE_DIR,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        creationflags=creation_flags
    )

    # Save PID so watchdog and UI can track active process
    pid_file = os.path.join(os.path.dirname(target_log_file), "runner.pid")
    try:
        with open(pid_file, "w", encoding="utf-8") as pf:
            pf.write(str(proc.pid))
    except Exception:
        pass

    return proc


def cancel_reconstruction(temp_dir=None):
    """
    Safely terminate active reconstruction processes (COLMAP and Python runner).
    """
    from config.paths import TEMP_DIR
    target_dir = temp_dir or TEMP_DIR
    pid_file = os.path.join(target_dir, "runner.pid")

    if os.path.exists(pid_file):
        try:
            with open(pid_file, "r") as pf:
                pid = int(pf.read().strip())
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
            else:
                os.kill(pid, 9)
        except Exception:
            pass
        try:
            os.remove(pid_file)
        except Exception:
            pass

    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/F", "/IM", "colmap.exe"], capture_output=True)
        except Exception:
            pass

    status_file = os.path.join(target_dir, "status.json")
    try:
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump({
                "status": "FAILED",
                "step": "Pipeline cancelled by user.",
                "progress": 0.0
            }, f, indent=2)
    except Exception:
        pass
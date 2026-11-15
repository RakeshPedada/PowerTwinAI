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
    Robustly handles both Streamlit UploadedFile objects and plain file path strings.
    """
    import shutil

    saved_paths: list[str] = []
    os.makedirs(temp_dir, exist_ok=True)

    for item in uploaded_files:
        if item is None:
            continue

        if isinstance(item, str):
            if os.path.exists(item):
                # If in upload mode and not in temp_dir, copy to temp_dir
                dest_path = os.path.join(temp_dir, os.path.basename(item))
                if os.path.abspath(item) != os.path.abspath(dest_path):
                    try:
                        shutil.copy2(item, dest_path)
                        saved_paths.append(os.path.abspath(dest_path))
                    except Exception:
                        saved_paths.append(os.path.abspath(item))
                else:
                    saved_paths.append(os.path.abspath(item))
        elif hasattr(item, "name"):
            filename = getattr(item, "name", "image.jpg")
            save_path = os.path.join(temp_dir, filename)
            try:
                if hasattr(item, "seek"):
                    item.seek(0)
                if hasattr(item, "getbuffer"):
                    buf = item.getbuffer()
                else:
                    buf = item.read()
                with open(save_path, "wb") as f:
                    f.write(buf)
                if hasattr(item, "seek"):
                    item.seek(0)
                saved_paths.append(os.path.abspath(save_path))
            except Exception as e:
                print(f"[RECON_MGR] Warning saving image {filename}: {e}")

    # Persist paths as JSON (list of plain strings only)
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
import os
import shutil
import subprocess


def get_colmap_path():
    """
    Locate the COLMAP executable or batch script.
    Checks environment variable COLMAP_PATH, candidate paths on C: and other drives,
    and system PATH.
    """
    env_path = os.environ.get("COLMAP_PATH")
    if env_path and (os.path.exists(env_path) or shutil.which(env_path)):
        return env_path

    candidates = [
        r"C:\COLMAP\COLMAP.bat",
        r"C:\COLMAP\colmap.bat",
        r"C:\COLMAP\bin\colmap.exe",
        r"C:\COLMAP\colmap.exe",
        r"C:\Program Files\COLMAP\COLMAP.bat",
        r"C:\Program Files\COLMAP\bin\colmap.exe",
        r"C:\Program Files\COLMAP\colmap.exe",
        r"E:\COLMAP\COLMAP.bat",
        r"D:\COLMAP\COLMAP.bat",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    which = shutil.which("colmap") or shutil.which("colmap.bat") or shutil.which("COLMAP.bat")
    if which:
        return which

    return r"C:\COLMAP\COLMAP.bat"


COLMAP_PATH = get_colmap_path()


def run_colmap(image_paths):

    colmap_bin = get_colmap_path()
    if not os.path.exists(colmap_bin) and not shutil.which(colmap_bin):
        raise FileNotFoundError(
            f"COLMAP executable was not found at '{colmap_bin}'.\n"
            f"Please install COLMAP (recommended location: C:\\COLMAP\\COLMAP.bat) "
            f"or set the COLMAP_PATH environment variable."
        )

    workspace = "colmap_workspace"

    if os.path.exists(workspace):

        try:
            shutil.rmtree(workspace)

        except Exception as e:

            print(
                f"[COLMAP] Cleanup failed: {e}"
            )

    os.makedirs(
        workspace,
        exist_ok=True
    )

    # =====================================
    # CREATE IMAGE-ONLY FOLDER
    # =====================================

    colmap_images = os.path.join(
        workspace,
        "images"
    )
  

    os.makedirs(colmap_images)

    for image_path in image_paths:

        if os.path.exists(image_path):

            shutil.copy(
                image_path,
                os.path.join(
                    colmap_images,
                    os.path.basename(image_path)
                )
            )
    print(f"[DEBUG] Images copied: {len(os.listdir(colmap_images))}")

    database_path = os.path.join(
        workspace,
        "database.db"
    )

    sparse_path = os.path.join(
        workspace,
        "sparse"
    )

    os.makedirs(sparse_path)

    print("[COLMAP] Feature Extraction...")

    subprocess.run([
        COLMAP_PATH,
        "feature_extractor",
        "--database_path",
        database_path,
        "--image_path",
        colmap_images
    ])

    print("[COLMAP] Feature Matching...")

    subprocess.run([
        COLMAP_PATH,
        "exhaustive_matcher",
        "--database_path",
        database_path
    ])

    print("[COLMAP] Sparse Reconstruction...")

    subprocess.run([
        COLMAP_PATH,
        "mapper",
        "--database_path",
        database_path,
        "--image_path",
        colmap_images,
        "--output_path",
        sparse_path
    ])

    print("[COLMAP] Exporting TXT files...")

    model_folder = os.path.join(
        sparse_path,
        "0"
    )

    if not os.path.exists(model_folder):
        raise Exception(
            "COLMAP failed to create sparse model"
        )

    os.makedirs(
        "colmap_data",
        exist_ok=True
    )

    subprocess.run([
        COLMAP_PATH,
        "model_converter",
        "--input_path",
        model_folder,
        "--output_path",
        "colmap_data",
        "--output_type",
        "TXT"
    ])

    print("[COLMAP] Completed")
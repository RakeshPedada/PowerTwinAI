import os
import shutil
import subprocess


COLMAP_PATH = r"E:\COLMAP\COLMAP.bat"

def select_largest_sparse_model(sparse_path):
    """
    Select the COLMAP sparse model containing
    the largest number of registered images.

    COLMAP may generate multiple reconstruction
    models when Mapper.multiple_models is enabled.

    Returns
    -------
    str
        Path to the largest sparse model.
    """

    model_folders = [
        os.path.join(
            sparse_path,
            name
        )
        for name in os.listdir(sparse_path)
        if os.path.isdir(
            os.path.join(
                sparse_path,
                name
            )
        )
    ]

    if not model_folders:
        raise RuntimeError(
            "COLMAP did not create any sparse models."
        )

    best_model = None
    best_registered_images = -1

    print(
        f"[COLMAP] Found "
        f"{len(model_folders)} sparse model(s)."
    )

    for model_folder in model_folders:

        result = subprocess.run(
            [
                COLMAP_PATH,
                "model_analyzer",
                "--path",
                model_folder
            ],
            capture_output=True,
            text=True
        )

        output = (
            result.stdout
            + "\n"
            + result.stderr
        )

        registered_images = None

        for line in output.splitlines():

            if "Registered images:" in line:

                try:

                    registered_images = int(
                        line.split(
                            "Registered images:"
                        )[1].strip()
                    )

                except ValueError:

                    registered_images = None

                break

        if registered_images is None:

            print(
                f"[COLMAP] Could not determine "
                f"model size: {model_folder}"
            )

            continue

        print(
            f"[COLMAP] Model "
            f"{os.path.basename(model_folder)}: "
            f"{registered_images} registered images"
        )

        if registered_images > best_registered_images:

            best_registered_images = (
                registered_images
            )

            best_model = model_folder

    if best_model is None:

        raise RuntimeError(
            "Unable to determine the largest "
            "COLMAP sparse model."
        )

    print(
        f"[COLMAP] Selected sparse model: "
        f"{best_model}"
    )

    print(
        f"[COLMAP] Registered images: "
        f"{best_registered_images}"
    )

    return best_model

def run_colmap(image_paths):

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

    subprocess.run(
        [
            COLMAP_PATH,
            "feature_extractor",
            "--database_path",
            database_path,
            "--image_path",
            colmap_images
        ],
        check=True
    )

    print("[COLMAP] Feature Matching...")

    subprocess.run(
        [
            COLMAP_PATH,
            "exhaustive_matcher",
            "--database_path",
            database_path
        ],
        check=True
    )
    print("[COLMAP] Sparse Reconstruction...")

    subprocess.run(
        [
            COLMAP_PATH,
            "mapper",
            "--database_path",
            database_path,
            "--image_path",
            colmap_images,
            "--output_path",
            sparse_path
        ],
        check=True
    )

    print("[COLMAP] Selecting best sparse model...")

    model_folder = select_largest_sparse_model(
        sparse_path
    )
    
    if os.path.exists("colmap_data"):

        shutil.rmtree(
            "colmap_data"
        )

    os.makedirs(
        "colmap_data",
        exist_ok=True
    )

    subprocess.run(
        [
            COLMAP_PATH,
            "model_converter",
            "--input_path",
            model_folder,
            "--output_path",
            "colmap_data",
            "--output_type",
            "TXT"
        ],
        check=True
    )

    print("[COLMAP] Completed")

    return model_folder
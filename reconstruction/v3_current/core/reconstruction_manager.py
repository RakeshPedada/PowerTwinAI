import os
import json
import subprocess


def prepare_workspace(
    temp_dir,
    log_file,
    status_file,
    result_file
):

    os.makedirs(temp_dir, exist_ok=True)

    with open(
        log_file,
        "w",
        encoding="utf-8",
        errors="ignore"
    ) as f:
        f.write("")

    for file in [status_file, result_file]:

        if os.path.exists(file):
            os.remove(file)


def save_uploaded_images(
    uploaded_files,
    mode,
    temp_dir,
    image_paths_file
):

    saved_paths = []

    if mode == "Upload Images":

        for file in uploaded_files:

            save_path = os.path.join(
                temp_dir,
                file.name
            )

            with open(save_path, "wb") as f:
                f.write(file.getbuffer())

            saved_paths.append(save_path)

    else:

        saved_paths = uploaded_files

    with open(image_paths_file, "w") as f:
        json.dump(saved_paths, f)


def launch_reconstruction():

    subprocess.Popen(
        ["python", "reconstruction_runner.py"],
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
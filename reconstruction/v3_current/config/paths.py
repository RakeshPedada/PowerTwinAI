"""
Project paths used throughout the application.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

TEMP_DIR = os.path.join(BASE_DIR, "temp_session")

OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Requirement 13: Structured outputs directory
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
SPARSE_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "sparse")
DENSE_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "dense")
MESH_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "mesh")
NEURALANGELO_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "neuralangelo")

LOG_FILE = os.path.join(TEMP_DIR, "reconstruction.log")

STATUS_FILE = os.path.join(TEMP_DIR, "status.json")

RESULT_FILE = os.path.join(TEMP_DIR, "result_data.npz")

IMAGE_PATHS_FILE = os.path.join(TEMP_DIR, "image_paths.json")

CONFIG_FILE = os.path.join(TEMP_DIR, "reconstruction_config.json")
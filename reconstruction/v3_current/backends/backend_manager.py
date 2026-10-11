"""
Backend Manager

Objective
---------
Provide a unified interface for selecting and executing
PowerTwinAI reconstruction backends.

Responsibilities
----------------
- Select the requested reconstruction backend.
- Execute the selected backend.
- Keep the reconstruction pipeline independent from
  backend-specific implementations.

Supported Backends
------------------
- COLMAP Backend
- PowerTwin Backend (Future)

Part of
-------
PowerTwinAI Phase 6 Architecture
"""

from backends.colmap_backend import run_colmap_backend


class BackendManager:
    """
    Controls reconstruction backend selection and execution.
    """

    def __init__(self, backend="colmap"):

        self.backend = backend.lower()

    def run(self, image_paths, config=None):
        """
        Execute the selected reconstruction backend with active configuration.
        """

        if self.backend == "colmap":

            return run_colmap_backend(
                image_paths,
                config=config
            )

        raise ValueError(
            f"Unsupported reconstruction backend: "
            f"{self.backend}"
        )
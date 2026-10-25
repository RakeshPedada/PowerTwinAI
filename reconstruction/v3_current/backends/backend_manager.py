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
- VGGT Backend (Facebook Research Visual Geometry Grounded Transformer)
- PowerTwin Backend (Future)

Part of
-------
PowerTwinAI Architecture
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
        config = config or {}

        if self.backend == "colmap":
            return run_colmap_backend(
                image_paths,
                config=config
            )

        if self.backend == "vggt":
            from vggt_pipeline import run_vggt_reconstruction
            return run_vggt_reconstruction(
                image_paths=image_paths,
                output_dir=config.get("output_dir"),
                reconstruction_mode=config.get("reconstruction_mode", "depth"),
                conf_threshold_val=config.get("conf_threshold_val", 3.5),
                max_frames=config.get("max_frames", 16),
                stride=config.get("stride", 1),
                coherent_order=config.get("coherent_order", True),
                remove_outliers=config.get("remove_outliers", True),
                voxel_size=config.get("voxel_size", None),
                export_colmap=config.get("export_colmap", True),
                use_ba=config.get("use_ba", False),
                progress_callback=config.get("progress_callback", None)
            )

        if self.backend in ["mast3r", "mast3r-sfm", "mast3r_sfm"]:
            from reconstruction.mast3r import run_mast3r_reconstruction
            return run_mast3r_reconstruction(
                image_paths=image_paths,
                output_dir=config.get("output_dir"),
                pairing_mode=config.get("pairing_mode", "Mode C (Hybrid)"),
                image_size=config.get("image_size", 512),
                subsample=config.get("subsample", 8),
                min_matches=config.get("min_matches", 15),
                matching_conf_thr=config.get("matching_conf_thr", 1.0),
                min_conf_thr=config.get("min_conf_thr", 1.1),
                num_key_images=config.get("num_key_images", 20),
                num_neighbors=config.get("num_neighbors", 8),
                niter1=config.get("niter1", 150),
                lr1=config.get("lr1", 0.07),
                niter2=config.get("niter2", 150),
                lr2=config.get("lr2", 0.01),
                opt_depth=config.get("opt_depth", True),
                shared_intrinsics=config.get("shared_intrinsics", False),
                kinematic_mode=config.get("kinematic_mode", "hclust-ward"),
                remove_outliers=config.get("remove_outliers", True),
                voxel_size=config.get("voxel_size", None),
                export_colmap=config.get("export_colmap", True),
                use_ba=config.get("use_ba", False),
                generate_mesh_flag=config.get("generate_mesh_flag", True),
                mesh_method=config.get("mesh_method", "Poisson"),
                poisson_depth=config.get("poisson_depth", 8),
                density_quantile=config.get("density_quantile", 0.01),
                max_frames=config.get("max_frames", None),
                progress_callback=config.get("progress_callback", None)
            )

        raise ValueError(
            f"Unsupported reconstruction backend: "
            f"{self.backend}"
        )

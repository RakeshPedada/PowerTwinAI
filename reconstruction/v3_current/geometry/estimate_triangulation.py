"""
Phase 3 - EstimateTriangulation

COLMAP-style triangulation API layer.

This module converts camera/image observations into the data
required by robust triangulation and returns the estimated 3D
point together with its inlier mask.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .robust_triangulation import RobustTriangulation
from .triangulation_estimator import (
    TriangulationResidualType,
)


@dataclass
class TriangulationPointData:
    """
    Observation in image coordinates and normalized camera
    coordinates.
    """

    img_point: np.ndarray
    cam_point: np.ndarray


@dataclass
class TriangulationPoseData:
    """
    Camera pose information used during triangulation.
    """

    cam_from_world: np.ndarray
    proj_center: np.ndarray
    camera: object = None


@dataclass
class EstimateTriangulationResult:
    """
    Result of robust triangulation.
    """

    success: bool
    point3D: np.ndarray | None
    inlier_mask: np.ndarray
    points: list
    poses: list


def _normalize_image_point(
    image_point,
    camera=None,
):
    """
    Convert an image point to normalized camera coordinates.

    Supported camera interfaces:

    1. camera.cam_from_img(point)
    2. camera.CamFromImg(point)

    If no camera model is supplied, the image point is treated
    as already normalized.
    """

    point = np.asarray(
        image_point,
        dtype=float,
    ).reshape(-1)

    if point.size != 2:
        raise ValueError(
            "Image point must contain exactly two values."
        )

    if not np.all(
        np.isfinite(point)
    ):
        return np.zeros(
            2,
            dtype=float,
        )

    if camera is None:
        return point.copy()

    method = getattr(
        camera,
        "cam_from_img",
        None,
    )

    if method is None:
        method = getattr(
            camera,
            "CamFromImg",
            None,
        )

    if method is None:
        # No camera conversion interface.
        # Preserve the supplied coordinates.
        return point.copy()

    try:
        normalized = np.asarray(
            method(point),
            dtype=float,
        ).reshape(-1)
    except Exception:
        return np.zeros(
            2,
            dtype=float,
        )

    if normalized.size < 2:
        return np.zeros(
            2,
            dtype=float,
        )

    normalized = normalized[:2]

    if not np.all(
        np.isfinite(normalized)
    ):
        return np.zeros(
            2,
            dtype=float,
        )

    return normalized


def _extract_pose(
    pose,
    camera=None,
):
    """
    Extract cam_from_world and projection center.

    Supported pose representations:

    - dictionary
    - object attributes
    - direct 4x4 matrix
    """

    cam_from_world = None
    proj_center = None

    if isinstance(pose, dict):
        cam_from_world = pose.get(
            "cam_from_world"
        )

        if cam_from_world is None:
            cam_from_world = pose.get(
                "CamFromWorld"
            )

        proj_center = pose.get(
            "proj_center"
        )

        if proj_center is None:
            proj_center = pose.get(
                "projection_center"
            )

    else:
        cam_from_world = getattr(
            pose,
            "cam_from_world",
            None,
        )

        if cam_from_world is None:
            cam_from_world = getattr(
                pose,
                "CamFromWorld",
                None,
            )

        proj_center = getattr(
            pose,
            "proj_center",
            None,
        )

        if proj_center is None:
            proj_center = getattr(
                pose,
                "projection_center",
                None,
            )

    if cam_from_world is None:
        candidate = np.asarray(
            pose,
            dtype=float,
        )

        if candidate.shape == (4, 4):
            cam_from_world = candidate

    if cam_from_world is None:
        raise ValueError(
            "Unable to extract cam_from_world."
        )

    cam_from_world = np.asarray(
        cam_from_world,
        dtype=float,
    )

    if cam_from_world.shape != (4, 4):
        raise ValueError(
            "cam_from_world must have shape (4, 4)."
        )

    if not np.all(
        np.isfinite(cam_from_world)
    ):
        raise ValueError(
            "cam_from_world contains non-finite values."
        )

    if proj_center is None:
        # For X_cam = R X_world + t:
        #
        # C = -R^T t
        #
        rotation = cam_from_world[
            :3,
            :3,
        ]

        translation = cam_from_world[
            :3,
            3,
        ]

        proj_center = (
            -rotation.T
            @ translation
        )

    proj_center = np.asarray(
        proj_center,
        dtype=float,
    ).reshape(-1)

    if proj_center.size != 3:
        raise ValueError(
            "proj_center must contain three values."
        )

    return (
        cam_from_world,
        proj_center,
    )


def _projection_matrix(
    cam_from_world,
    camera=None,
):
    """
    Construct the projection matrix used by the
    triangulation estimator.

    EstimateTriangulation stores observations as normalized
    camera coordinates. Therefore the corresponding projection
    matrix must be the camera extrinsic matrix:

        P = [R | t]

    Calibration is already applied by Camera.CamFromImg().
    Multiplying by K here would apply calibration twice.
    """

    extrinsic = np.asarray(
        cam_from_world,
        dtype=float,
    )

    if extrinsic.shape != (4, 4):
        raise ValueError(
            "cam_from_world must have shape (4, 4)."
        )

    return extrinsic[:3, :]


def EstimateTriangulation(
    points: Sequence[Sequence[float]],
    cam_from_world: Sequence,
    cameras: Sequence = None,
    min_tri_angle: float = 0.0,
    residual_type=TriangulationResidualType.ANGULAR_ERROR,
    max_error: float = np.deg2rad(2.0),
    confidence: float = 0.9999,
    min_inlier_ratio: float = 0.02,
    max_num_trials: int = 10000,
    random_seed: int = -1,
):
    """
    COLMAP-style robust triangulation entry point.

    Angular-error mode operates on normalized camera
    coordinates and [R|t] projection matrices.

    Reprojection-error mode operates on original image
    coordinates and K[R|t] projection matrices.
    """

    points = list(points)
    cam_from_world = list(
        cam_from_world
    )

    if cameras is None:
        cameras = [
            None
            for _ in points
        ]
    else:
        cameras = list(cameras)

    if len(points) != len(
        cam_from_world
    ):
        raise ValueError(
            "points and cam_from_world must have "
            "the same length."
        )

    if len(points) != len(
        cameras
    ):
        raise ValueError(
            "points and cameras must have "
            "the same length."
        )

    if len(points) < 2:
        return EstimateTriangulationResult(
            success=False,
            point3D=None,
            inlier_mask=np.zeros(
                len(points),
                dtype=bool,
            ),
            points=[],
            poses=[],
        )

    point_data = []
    pose_data = []

    observations = []
    projection_matrices = []

    for image_point, pose, camera in zip(
        points,
        cam_from_world,
        cameras,
    ):

        image_point = np.asarray(
            image_point,
            dtype=float,
        ).reshape(-1)

        if image_point.size != 2:
            raise ValueError(
                "Each image point must contain "
                "exactly two values."
            )

        normalized_point = (
            _normalize_image_point(
                image_point,
                camera,
            )
        )

        pose_matrix, proj_center = (
            _extract_pose(
                pose,
                camera,
            )
        )

        point_data.append(
            TriangulationPointData(
                img_point=image_point,
                cam_point=normalized_point,
            )
        )

        pose_data.append(
            TriangulationPoseData(
                cam_from_world=pose_matrix,
                proj_center=proj_center,
                camera=camera,
            )
        )

        if (
            residual_type
            == TriangulationResidualType.REPROJECTION_ERROR
        ):
            observations.append(
                image_point
            )

            extrinsic = pose_matrix[
                :3,
                :,
            ]

            K = None

            if camera is not None:

                for name in (
                    "K",
                    "k",
                    "calibration_matrix",
                    "CalibrationMatrix",
                ):

                    candidate = getattr(
                        camera,
                        name,
                        None,
                    )

                    if candidate is not None:

                        candidate = np.asarray(
                            candidate,
                            dtype=float,
                        )

                        if candidate.shape == (3, 3):
                            K = candidate
                            break

            if K is None:
                # Without a calibration matrix, the supplied
                # coordinates are treated as the camera/image
                # coordinates represented by the extrinsic model.
                projection_matrices.append(
                    extrinsic
                )

            else:
                projection_matrices.append(
                    K @ extrinsic
                )

        else:
            observations.append(
                normalized_point
            )

            projection_matrices.append(
                _projection_matrix(
                    pose_matrix,
                    camera,
                )
            )

    triangulator = RobustTriangulation(
        max_error=max_error,
        confidence=confidence,
        min_inlier_ratio=min_inlier_ratio,
        max_num_trials=max_num_trials,
        min_tri_angle=min_tri_angle,
        random_seed=random_seed,
        residual_type=residual_type,
    )

    model, inlier_mask = (
        triangulator.estimate(
            observations,
            projection_matrices,
        )
    )

    success = (
        model is not None
        and int(
            np.count_nonzero(
                inlier_mask
            )
        ) >= 2
    )

    return EstimateTriangulationResult(
        success=success,
        point3D=(
            None
            if model is None
            else np.asarray(
                model,
                dtype=float,
            )
        ),
        inlier_mask=np.asarray(
            inlier_mask,
            dtype=bool,
        ),
        points=point_data,
        poses=pose_data,
    )



# Compatibility alias.
estimate_triangulation = (
    EstimateTriangulation
)

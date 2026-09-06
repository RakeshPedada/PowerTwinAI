"""
Phase 3 - Track Operations

Feature-track representation and COLMAP-style robust
track triangulation helpers.

This module keeps the existing FeatureTrack abstraction while
adding TriangulateTrack(), which is used by the incremental
triangulation operations.
"""

from dataclasses import dataclass, field
from math import comb
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .estimate_triangulation import EstimateTriangulation
from .triangulation_estimator import (
    TriangulationResidualType,
)


# ============================================================
# Feature Track
# ============================================================


@dataclass
class FeatureTrack:
    """
    A feature track represented as:

        image_id -> feature_id
    """

    observations: Dict[int, int] = field(
        default_factory=dict
    )

    def add_observation(
        self,
        image_id: int,
        feature_id: int,
    ) -> bool:
        """
        Add an observation.

        Returns False if the image already has an
        observation in this track.
        """

        image_id = int(image_id)
        feature_id = int(feature_id)

        if image_id in self.observations:
            return False

        self.observations[
            image_id
        ] = feature_id

        return True

    def add(
        self,
        image_id: int,
        feature_id: int,
    ) -> bool:
        """Compatibility alias."""

        return self.add_observation(
            image_id,
            feature_id,
        )

    def has_image(
        self,
        image_id: int,
    ) -> bool:
        return int(image_id) in self.observations

    def has(
        self,
        image_id: int,
    ) -> bool:
        """Compatibility alias."""

        return self.has_image(
            image_id
        )

    def remove_observation(
        self,
        image_id: int,
    ) -> bool:

        image_id = int(image_id)

        if image_id not in self.observations:
            return False

        del self.observations[
            image_id
        ]

        return True

    def size(self) -> int:
        return len(
            self.observations
        )

    def __len__(self):
        return self.size()

    def items(self):
        return self.observations.items()

    def image_ids(self):
        return self.observations.keys()

    def feature_ids(self):
        return self.observations.values()


# ============================================================
# Track Triangulation Result
# ============================================================


@dataclass
class TriangulateTrackResult:
    """
    Result of robust track triangulation.
    """

    success: bool
    point3D: np.ndarray | None
    inlier_mask: np.ndarray
    num_observations: int
    num_inliers: int
    min_num_trials: int


# COLMAP Phase 3 specification.
kExhaustiveSamplingThreshold = 15


# ============================================================
# Utility
# ============================================================


def _n_choose_2(
    num_observations: int,
) -> int:

    if num_observations < 2:
        return 0

    return int(
        comb(
            num_observations,
            2,
        )
    )


# ============================================================
# TriangulateTrack
# ============================================================


def TriangulateTrack(
    points: Sequence[Sequence[float]],
    cam_from_world: Sequence,
    cameras: Sequence = None,
    min_angle: float = 1.5,
    create_max_angle_error: float = 2.0,
    random_seed: int = -1,
    confidence: float = 0.9999,
    min_inlier_ratio: float = 0.02,
    max_num_trials: int = 10000,
) -> TriangulateTrackResult:
    """
    Robustly triangulate one feature track.

    Parameters
    ----------
    points
        Image observations.

    cam_from_world
        Camera poses corresponding to observations.

    cameras
        Camera models corresponding to observations.

    min_angle
        Minimum triangulation angle in degrees.

    create_max_angle_error
        Maximum angular reprojection error in degrees.

    random_seed
        RANSAC random seed.

    confidence
        RANSAC confidence.

    min_inlier_ratio
        Minimum expected inlier ratio.

    max_num_trials
        Maximum RANSAC trials.

    Returns
    -------
    TriangulateTrackResult
    """

    points = list(points)
    cam_from_world = list(
        cam_from_world
    )

    num_observations = len(
        points
    )

    if cameras is None:
        cameras = [
            None
            for _ in range(
                num_observations
            )
        ]
    else:
        cameras = list(cameras)

    # --------------------------------------------------------
    # Validate correspondence counts.
    # --------------------------------------------------------

    if len(cam_from_world) != num_observations:
        raise ValueError(
            "points and cam_from_world must have "
            "the same length."
        )

    if len(cameras) != num_observations:
        raise ValueError(
            "points and cameras must have "
            "the same length."
        )

    if num_observations < 2:
        return TriangulateTrackResult(
            success=False,
            point3D=None,
            inlier_mask=np.zeros(
                num_observations,
                dtype=bool,
            ),
            num_observations=num_observations,
            num_inliers=0,
            min_num_trials=0,
        )

    # --------------------------------------------------------
    # COLMAP Track Creation settings:
    #
    #   min_angle
    #   ANGULAR_ERROR
    #   create_max_angle_error
    #   random_seed
    # --------------------------------------------------------

    min_tri_angle = np.deg2rad(
        float(min_angle)
    )

    max_error = np.deg2rad(
        float(create_max_angle_error)
    )

    # --------------------------------------------------------
    # Exhaustive sampling for small tracks.
    #
    # For N <= 15:
    #
    #     min_num_trials = N choose 2
    # --------------------------------------------------------

    if (
        num_observations
        <= kExhaustiveSamplingThreshold
    ):
        min_num_trials = _n_choose_2(
            num_observations
        )
    else:
        min_num_trials = 0

    effective_max_trials = max(
        int(max_num_trials),
        min_num_trials,
    )

    # --------------------------------------------------------
    # Robust triangulation.
    # --------------------------------------------------------

    result = EstimateTriangulation(
        points=points,
        cam_from_world=cam_from_world,
        cameras=cameras,
        min_tri_angle=min_tri_angle,
        residual_type=(
            TriangulationResidualType
            .ANGULAR_ERROR
        ),
        max_error=max_error,
        confidence=confidence,
        min_inlier_ratio=min_inlier_ratio,
        max_num_trials=effective_max_trials,
        random_seed=random_seed,
    )

    inlier_mask = np.asarray(
        result.inlier_mask,
        dtype=bool,
    )

    num_inliers = int(
        np.count_nonzero(
            inlier_mask
        )
    )

    success = bool(
        result.success
        and num_inliers >= 2
        and result.point3D is not None
    )

    return TriangulateTrackResult(
        success=success,
        point3D=result.point3D,
        inlier_mask=inlier_mask,
        num_observations=num_observations,
        num_inliers=num_inliers,
        min_num_trials=min_num_trials,
    )


# ============================================================
# Lower-case compatibility alias
# ============================================================


triangulate_track = TriangulateTrack


K_EXHAUSTIVE_SAMPLING_THRESHOLD = (
    kExhaustiveSamplingThreshold
)
"""
Phase 3 - Triangulation Estimator

Robust geometric triangulation primitives for PowerTwinAI.

Implements the geometry required by the Phase 3 specification:

- two-view triangulation,
- multi-view linear triangulation,
- positive-depth / cheirality validation,
- triangulation-angle validation,
- normalized-camera-coordinate support,
- angular and reprojection residual calculation,
- minimum sample requirement of two observations.

The estimator remains compatible with the existing
PowerTwinAI projection-matrix interface.
"""

from enum import Enum
from typing import Sequence

import numpy as np


class TriangulationResidualType(Enum):
    """
    Residual types defined by the Phase 3 specification.
    """

    ANGULAR_ERROR = "angular_error"
    REPROJECTION_ERROR = "reprojection_error"


class TriangulationEstimator:
    """
    Estimate a 3D point from corresponding observations.

    The projection matrices are expected to be 3x4 matrices
    of the form:

        P = K [R | t]

    where the camera coordinate system is represented by
    the projection matrix.

    Minimum sample count:

        kMinNumSamples = 2
    """

    kMinNumSamples = 2

    # =====================================================
    # PUBLIC ESTIMATE
    # =====================================================

    def estimate(
        self,
        observations: Sequence[Sequence[float]],
        projection_matrices: Sequence[np.ndarray],
        min_tri_angle: float = 0.0,
        residual_type: TriangulationResidualType = (
            TriangulationResidualType.ANGULAR_ERROR
        ),
    ) -> np.ndarray | None:
        """
        Estimate and geometrically validate a 3D point.

        Parameters
        ----------
        observations
            Image observations [x, y].

        projection_matrices
            Corresponding 3x4 camera projection matrices.

        min_tri_angle
            Minimum triangulation angle in radians.

        residual_type
            Residual type requested by the caller.

        Returns
        -------
        numpy.ndarray | None
            Valid 3D point, or None when the candidate fails
            geometric validation.
        """

        self._validate_inputs(
            observations,
            projection_matrices,
        )

        if len(observations) < self.kMinNumSamples:
            return None

        if min_tri_angle < 0:
            raise ValueError(
                "min_tri_angle must not be negative."
            )

        if not isinstance(
            residual_type,
            TriangulationResidualType,
        ):
            raise ValueError(
                "Invalid triangulation residual type."
            )

        if len(observations) == 2:
            point_3d = self._triangulate_two_view(
                observations[0],
                observations[1],
                projection_matrices[0],
                projection_matrices[1],
            )
        else:
            point_3d = self._triangulate_multi_view(
                observations,
                projection_matrices,
            )

        if point_3d is None:
            return None

        # -------------------------------------------------
        # CHEIRALITY
        # -------------------------------------------------

        if not self.has_positive_depth_all(
            point_3d,
            projection_matrices,
        ):
            return None

        # -------------------------------------------------
        # TRIANGULATION ANGLE
        # -------------------------------------------------

        if not self.has_sufficient_triangulation_angle(
            point_3d,
            projection_matrices,
            min_tri_angle,
        ):
            return None

        return point_3d

    # =====================================================
    # INPUT VALIDATION
    # =====================================================

    @staticmethod
    def _validate_inputs(
        observations,
        projection_matrices,
    ) -> None:

        if len(observations) != len(
            projection_matrices
        ):
            raise ValueError(
                "Observations and projection matrices must "
                "have the same length."
            )

        for observation in observations:

            if len(observation) != 2:
                raise ValueError(
                    "Each observation must contain [x, y]."
                )

            values = np.asarray(
                observation,
                dtype=float,
            )

            if not np.all(
                np.isfinite(values)
            ):
                raise ValueError(
                    "Observations must contain finite values."
                )

        for projection_matrix in projection_matrices:

            matrix = np.asarray(
                projection_matrix,
                dtype=float,
            )

            if matrix.shape != (3, 4):
                raise ValueError(
                    "Each projection matrix must have shape "
                    "(3, 4)."
                )

            if not np.all(
                np.isfinite(matrix)
            ):
                raise ValueError(
                    "Projection matrices must contain "
                    "finite values."
                )

    # =====================================================
    # TWO-VIEW TRIANGULATION
    # =====================================================

    def _triangulate_two_view(
        self,
        observation_1,
        observation_2,
        projection_matrix_1,
        projection_matrix_2,
    ) -> np.ndarray | None:
        """
        Linear two-view triangulation.

        This constructs the standard four-row homogeneous
        DLT system and returns the Euclidean 3D point.
        """

        rows = []

        for observation, projection_matrix in (
            (
                observation_1,
                projection_matrix_1,
            ),
            (
                observation_2,
                projection_matrix_2,
            ),
        ):

            matrix = np.asarray(
                projection_matrix,
                dtype=float,
            )

            x, y = observation

            rows.append(
                x * matrix[2]
                - matrix[0]
            )

            rows.append(
                y * matrix[2]
                - matrix[1]
            )

        return self._solve_dlt(rows)

    # =====================================================
    # MULTI-VIEW TRIANGULATION
    # =====================================================

    def _triangulate_multi_view(
        self,
        observations,
        projection_matrices,
    ) -> np.ndarray | None:
        """
        Linear multi-view triangulation.

        All supplied observations contribute to the DLT
        system.
        """

        rows = []

        for observation, projection_matrix in zip(
            observations,
            projection_matrices,
        ):

            matrix = np.asarray(
                projection_matrix,
                dtype=float,
            )

            x, y = observation

            rows.append(
                x * matrix[2]
                - matrix[0]
            )

            rows.append(
                y * matrix[2]
                - matrix[1]
            )

        return self._solve_dlt(rows)

    # =====================================================
    # DLT SOLVER
    # =====================================================

    @staticmethod
    def _solve_dlt(
        rows,
    ) -> np.ndarray | None:

        matrix_a = np.asarray(
            rows,
            dtype=float,
        )

        if matrix_a.shape[0] < 4:
            return None

        try:

            _, _, vt = np.linalg.svd(
                matrix_a
            )

        except np.linalg.LinAlgError:
            return None

        homogeneous_point = vt[-1]

        if (
            homogeneous_point.shape != (4,)
            or not np.all(
                np.isfinite(
                    homogeneous_point
                )
            )
        ):
            return None

        if abs(
            homogeneous_point[3]
        ) < 1e-12:
            return None

        point_3d = (
            homogeneous_point[:3]
            /
            homogeneous_point[3]
        )

        if not np.all(
            np.isfinite(point_3d)
        ):
            return None

        return point_3d

    # =====================================================
    # POSITIVE DEPTH
    # =====================================================

    @staticmethod
    def has_positive_depth(
        point_3d,
        projection_matrix,
    ) -> bool:
        """
        Check whether a 3D point lies in front of a camera.

        For P = K[R|t], the third component of P X is
        proportional to the camera-coordinate depth.
        """

        matrix = np.asarray(
            projection_matrix,
            dtype=float,
        )

        homogeneous_point = np.append(
            np.asarray(
                point_3d,
                dtype=float,
            ),
            1.0,
        )

        projected = (
            matrix
            @ homogeneous_point
        )

        if not np.all(
            np.isfinite(projected)
        ):
            return False

        return projected[2] > 0.0

    @classmethod
    def has_positive_depth_all(
        cls,
        point_3d,
        projection_matrices,
    ) -> bool:
        """
        Require positive depth in every observation.
        """

        return all(
            cls.has_positive_depth(
                point_3d,
                projection_matrix,
            )
            for projection_matrix
            in projection_matrices
        )

    # =====================================================
    # CAMERA CENTER
    # =====================================================

    @staticmethod
    def camera_center(
        projection_matrix,
    ) -> np.ndarray | None:
        """
        Calculate the projection center of a 3x4 camera
        matrix as the null-space solution P C = 0.
        """

        matrix = np.asarray(
            projection_matrix,
            dtype=float,
        )

        if matrix.shape != (3, 4):
            return None

        try:

            _, _, vt = np.linalg.svd(
                matrix
            )

        except np.linalg.LinAlgError:
            return None

        center_h = vt[-1]

        if abs(center_h[3]) < 1e-12:
            return None

        center = (
            center_h[:3]
            /
            center_h[3]
        )

        if not np.all(
            np.isfinite(center)
        ):
            return None

        return center

    # =====================================================
    # TRIANGULATION ANGLE
    # =====================================================

    @classmethod
    def calculate_triangulation_angle(
        cls,
        point_3d,
        projection_matrix_1,
        projection_matrix_2,
    ) -> float:
        """
        Calculate the angle between the two viewing rays.

        Returns radians.
        """

        center_1 = cls.camera_center(
            projection_matrix_1
        )

        center_2 = cls.camera_center(
            projection_matrix_2
        )

        if center_1 is None or center_2 is None:
            return 0.0

        ray_1 = (
            np.asarray(point_3d)
            - center_1
        )

        ray_2 = (
            np.asarray(point_3d)
            - center_2
        )

        norm_1 = np.linalg.norm(
            ray_1
        )

        norm_2 = np.linalg.norm(
            ray_2
        )

        if (
            norm_1 <= 1e-12
            or norm_2 <= 1e-12
        ):
            return 0.0

        cosine = (
            np.dot(
                ray_1,
                ray_2,
            )
            /
            (norm_1 * norm_2)
        )

        cosine = np.clip(
            cosine,
            -1.0,
            1.0,
        )

        return float(
            np.arccos(cosine)
        )

    @classmethod
    def has_sufficient_triangulation_angle(
        cls,
        point_3d,
        projection_matrices,
        min_tri_angle,
    ) -> bool:
        """
        For multiple views, accept the point when at least
        one observation pair reaches the minimum angle.

        For two views, that pair must reach the threshold.
        """

        if len(projection_matrices) < 2:
            return False

        for index_1 in range(
            len(projection_matrices)
        ):

            for index_2 in range(
                index_1 + 1,
                len(projection_matrices),
            ):

                angle = (
                    cls.calculate_triangulation_angle(
                        point_3d,
                        projection_matrices[index_1],
                        projection_matrices[index_2],
                    )
                )

                if angle >= min_tri_angle:
                    return True

        return False

    # =====================================================
    # ANGULAR REPROJECTION ERROR
    # =====================================================

    @staticmethod
    def calculate_angular_reprojection_error(
        point_3d,
        observation,
        projection_matrix,
    ):
        """
        Calculate squared angular reprojection error.

        The angular triangulation path uses:
            normalized image observation + [R | t]

        where [R | t] maps world coordinates into the camera frame.
        The observation therefore represents the normalized camera ray
        [x, y, 1].
        """
        matrix = np.asarray(projection_matrix, dtype=float)

        if matrix.shape == (4, 4):
            R = matrix[:3, :3]
            t = matrix[:3, 3]
        elif matrix.shape == (3, 4):
            R = matrix[:, :3]
            t = matrix[:, 3]
        else:
            raise ValueError(
                "Angular reprojection requires a 3x4 or 4x4 pose matrix"
            )

        observation = np.asarray(observation, dtype=float).reshape(-1)
        if observation.size < 2:
            return np.inf

        camera_ray = np.array(
            [observation[0], observation[1], 1.0],
            dtype=float,
        )

        # Convert the normalized camera ray into the world frame.
        world_ray = R.T @ camera_ray
        world_ray_norm = np.linalg.norm(world_ray)

        if world_ray_norm <= 1e-15:
            return np.inf

        world_ray /= world_ray_norm

        # Camera center: C = -R^T t.
        camera_center = -R.T @ t

        point_ray = np.asarray(point_3d, dtype=float) - camera_center
        point_ray_norm = np.linalg.norm(point_ray)

        if point_ray_norm <= 1e-15:
            return np.inf

        point_ray /= point_ray_norm

        cosine = float(np.clip(np.dot(world_ray, point_ray), -1.0, 1.0))
        angle = float(np.arccos(cosine))

        return angle * angle


    @staticmethod
    def calculate_squared_reprojection_error(
        point_3d,
        observation,
        projection_matrix,
    ) -> float:
        """
        Calculate squared pixel reprojection error.
        """

        matrix = np.asarray(
            projection_matrix,
            dtype=float,
        )

        homogeneous_point = np.append(
            np.asarray(
                point_3d,
                dtype=float,
            ),
            1.0,
        )

        projected = (
            matrix
            @ homogeneous_point
        )

        if (
            abs(projected[2])
            < 1e-12
        ):
            return float("inf")

        projected_xy = (
            projected[:2]
            /
            projected[2]
        )

        observation_xy = np.asarray(
            observation,
            dtype=float,
        )

        difference = (
            projected_xy
            - observation_xy
        )

        return float(
            np.dot(
                difference,
                difference,
            )
        )
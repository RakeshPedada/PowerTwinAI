"""
Phase 3 - Triangulation Residuals

Residual definitions used by robust triangulation.

Important:
COLMAP's triangulation estimator works with squared residuals.
Therefore the returned values from this module are squared
angular or squared reprojection errors.
"""

from typing import Sequence

import numpy as np

from .triangulation_estimator import (
    TriangulationEstimator,
    TriangulationResidualType,
)


class TriangulationResidual:
    """
    Evaluate triangulation residuals for one 3D point.
    """

    def __init__(
        self,
        residual_type=TriangulationResidualType.ANGULAR_ERROR,
    ):
        if not isinstance(
            residual_type,
            TriangulationResidualType,
        ):
            raise ValueError(
                "Invalid triangulation residual type."
            )

        self.residual_type = residual_type

    def evaluate(
        self,
        point_3d,
        observations: Sequence[Sequence[float]],
        projection_matrices: Sequence[np.ndarray],
    ) -> np.ndarray:
        """
        Return one squared residual per observation.
        """

        if len(observations) != len(
            projection_matrices
        ):
            raise ValueError(
                "Observations and projection matrices "
                "must have the same length."
            )

        residuals = []

        for observation, projection_matrix in zip(
            observations,
            projection_matrices,
        ):

            if (
                self.residual_type
                == TriangulationResidualType.ANGULAR_ERROR
            ):
                residual = (
                    TriangulationEstimator
                    .calculate_angular_reprojection_error(
                        point_3d,
                        observation,
                        projection_matrix,
                    )
                )

            elif (
                self.residual_type
                == TriangulationResidualType.REPROJECTION_ERROR
            ):
                residual = (
                    TriangulationEstimator
                    .calculate_squared_reprojection_error(
                        point_3d,
                        observation,
                        projection_matrix,
                    )
                )

            else:
                residual = float("inf")

            residuals.append(
                float(residual)
            )

        return np.asarray(
            residuals,
            dtype=float,
        )

    def evaluate_single(
        self,
        point_3d,
        observation,
        projection_matrix,
    ) -> float:
        """
        Evaluate one observation.
        """

        if (
            self.residual_type
            == TriangulationResidualType.ANGULAR_ERROR
        ):
            return float(
                TriangulationEstimator
                .calculate_angular_reprojection_error(
                    point_3d,
                    observation,
                    projection_matrix,
                )
            )

        if (
            self.residual_type
            == TriangulationResidualType.REPROJECTION_ERROR
        ):
            return float(
                TriangulationEstimator
                .calculate_squared_reprojection_error(
                    point_3d,
                    observation,
                    projection_matrix,
                )
            )

        return float("inf")


def calculate_angular_residuals(
    point_3d,
    observations,
    projection_matrices,
):
    """
    Convenience function for squared angular errors.
    """

    return TriangulationResidual(
        TriangulationResidualType.ANGULAR_ERROR
    ).evaluate(
        point_3d,
        observations,
        projection_matrices,
    )


def calculate_reprojection_residuals(
    point_3d,
    observations,
    projection_matrices,
):
    """
    Convenience function for squared reprojection errors.
    """

    return TriangulationResidual(
        TriangulationResidualType.REPROJECTION_ERROR
    ).evaluate(
        point_3d,
        observations,
        projection_matrices,
    )
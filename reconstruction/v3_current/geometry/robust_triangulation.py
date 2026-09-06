"""
Phase 3 - Robust Triangulation

Robust triangulation using RANSAC followed by LO-RANSAC-style
local optimization.

The implementation operates on corresponding observations and
3x4 projection matrices.
"""

from typing import Sequence

import numpy as np

from .lo_ransac import LORANSAC
from .ransac import RANSAC, RANSACOptions, CombinationSampler
from .support_measurement import InlierSupportMeasurer
from .triangulation_estimator import (
    TriangulationEstimator,
    TriangulationResidualType,
)
from .triangulation_residual import TriangulationResidual


class RobustTriangulation:
    """
    Robust 3D point estimation from multiple observations.
    """

    def __init__(
        self,
        max_error: float = np.deg2rad(2.0),
        confidence: float = 0.9999,
        min_inlier_ratio: float = 0.02,
        max_num_trials: int = 10000,
        min_tri_angle: float = 0.0,
        random_seed: int = -1,
        residual_type=TriangulationResidualType.ANGULAR_ERROR,
    ):
        if max_error <= 0:
            raise ValueError(
                "max_error must be positive."
            )

        if not 0.0 <= min_inlier_ratio <= 1.0:
            raise ValueError(
                "min_inlier_ratio must be between 0 and 1."
            )

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "confidence must be between 0 and 1."
            )

        if max_num_trials < 1:
            raise ValueError(
                "max_num_trials must be positive."
            )

        if min_tri_angle < 0:
            raise ValueError(
                "min_tri_angle must not be negative."
            )

        self.max_error = float(max_error)
        self.confidence = float(confidence)
        self.min_inlier_ratio = float(
            min_inlier_ratio
        )
        self.max_num_trials = int(
            max_num_trials
        )
        self.min_tri_angle = float(
            min_tri_angle
        )
        self.random_seed = int(
            random_seed
        )

        self.residual_type = residual_type

        self.estimator = TriangulationEstimator()

        self.residual = TriangulationResidual(
            residual_type
        )

    # =====================================================
    # MAIN API
    # =====================================================

    def estimate(
        self,
        observations: Sequence[Sequence[float]],
        projection_matrices: Sequence[np.ndarray],
    ):
        """
        Estimate a robust 3D point.

        Returns
        -------
        model, inlier_mask

        model
            Estimated 3D point or None.

        inlier_mask
            Boolean mask corresponding to all observations.
        """

        observations = list(
            observations
        )

        projection_matrices = [
            np.asarray(
                matrix,
                dtype=float,
            )
            for matrix in projection_matrices
        ]

        if len(observations) < 2:
            return None, np.zeros(
                len(observations),
                dtype=bool,
            )

        if len(observations) != len(
            projection_matrices
        ):
            raise ValueError(
                "Observations and projection matrices "
                "must have the same length."
            )

        data = list(
            zip(
                observations,
                projection_matrices,
            )
        )

        # -------------------------------------------------
        # COLMAP-style squared threshold semantics.
        # -------------------------------------------------

        max_residual = (
            self.max_error
            * self.max_error
        )

        options = RANSACOptions(
            max_error=self.max_error,
            min_inlier_ratio=self.min_inlier_ratio,
            confidence=self.confidence,
            max_num_trials=self.max_num_trials,
            random_seed=self.random_seed,
        )

        ransac = RANSAC(
            options=options
        )

        estimator_function = (
            self._estimate_model
        )

        residual_function = (
            self._residual_function
        )

        sampler = None

        if len(data) <= 15:
            sampler = CombinationSampler(
                num_samples=len(data),
                sample_size=2,
            )

        result = ransac.estimate(
            data=data,
            estimator=estimator_function,
            residual_function=residual_function,
            sample_size=2,
            max_residual=max_residual,
            sampler=sampler,
        )

        if (
            result.model is None
            or not result.success
        ):
            return (
                None,
                result.inlier_mask,
            )

        # -------------------------------------------------
        # LO-RANSAC refinement.
        #
        # Pass the complete data set so local optimization
        # can expand the inlier set rather than optimizing
        # only over the original inliers.
        # -------------------------------------------------

        lo_ransac = LORANSAC(
            max_local_trials=10
        )

        optimized = lo_ransac.optimize(
            model=result.model,
            inlier_data=result.inlier_data,
            residual_function=residual_function,
            estimator=estimator_function,
            max_residual=max_residual,
            all_data=data,
        )

        if optimized is not None:
            optimized_model = optimized[0]
        else:
            optimized_model = result.model

        # -------------------------------------------------
        # Final complete residual evaluation.
        # -------------------------------------------------

        final_residuals = np.asarray(
            residual_function(
                optimized_model,
                data,
            ),
            dtype=float,
        )

        final_mask = (
            final_residuals
            <= max_residual
        )

        # -------------------------------------------------
        # Final geometric validation.
        # -------------------------------------------------

        final_observations = [
            observations[index]
            for index, is_inlier
            in enumerate(final_mask)
            if is_inlier
        ]

        final_projection_matrices = [
            projection_matrices[index]
            for index, is_inlier
            in enumerate(final_mask)
            if is_inlier
        ]

        if len(final_observations) < 2:
            return (
                None,
                final_mask,
            )

        # Re-estimate using the complete final inlier set.
        final_model = self.estimator.estimate(
            final_observations,
            final_projection_matrices,
            min_tri_angle=self.min_tri_angle,
            residual_type=self.residual_type,
        )

        if final_model is None:
            return (
                None,
                final_mask,
            )

        # -------------------------------------------------
        # Recompute final mask after geometric validation.
        # -------------------------------------------------

        final_residuals = np.asarray(
            residual_function(
                final_model,
                data,
            ),
            dtype=float,
        )

        final_mask = (
            final_residuals
            <= max_residual
        )

        return (
            final_model,
            final_mask,
        )

    # =====================================================
    # MODEL ESTIMATION
    # =====================================================

    def _estimate_model(
        self,
        sample,
    ):
        """
        Estimate a model from a minimal sample.

        Two observations are sufficient for triangulation.
        """

        observations = [
            item[0]
            for item in sample
        ]

        projection_matrices = [
            item[1]
            for item in sample
        ]

        return self.estimator.estimate(
            observations,
            projection_matrices,
            min_tri_angle=self.min_tri_angle,
            residual_type=self.residual_type,
        )

    # =====================================================
    # RESIDUAL EVALUATION
    # =====================================================

    def _residual_function(
        self,
        model,
        data,
    ):
        """
        Evaluate squared residuals for all observations.
        """

        if model is None:
            return np.full(
                len(data),
                np.inf,
                dtype=float,
            )

        observations = [
            item[0]
            for item in data
        ]

        projection_matrices = [
            item[1]
            for item in data
        ]

        return self.residual.evaluate(
            model,
            observations,
            projection_matrices,
        )


def triangulate_robust(
    observations,
    projection_matrices,
    max_error=np.deg2rad(2.0),
    confidence=0.9999,
    min_inlier_ratio=0.02,
    max_num_trials=10000,
    min_tri_angle=0.0,
    random_seed=-1,
    residual_type=TriangulationResidualType.ANGULAR_ERROR,
):
    """
    Convenience wrapper around RobustTriangulation.
    """

    triangulator = RobustTriangulation(
        max_error=max_error,
        confidence=confidence,
        min_inlier_ratio=min_inlier_ratio,
        max_num_trials=max_num_trials,
        min_tri_angle=min_tri_angle,
        random_seed=random_seed,
        residual_type=residual_type,
    )

    return triangulator.estimate(
        observations,
        projection_matrices,
    )

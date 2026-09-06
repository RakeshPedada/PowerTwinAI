"""
Phase 3 - LO-RANSAC

Local-optimization RANSAC refinement.

LO-RANSAC extends the core RANSAC infrastructure while
providing the local optimization stage used by robust
geometric estimation.

Behavior follows the Phase 3 specification:

- maximum of 10 local trials,
- re-estimate from the current inlier set,
- evaluate against the complete data set,
- accept only an improved support,
- continue only while the inlier count increases.
"""

from typing import Any, Callable, Sequence

import numpy as np

from .ransac import RANSAC, RANSACOptions
from .support_measurement import InlierSupportMeasurer


class LORANSAC(RANSAC):
    """
    RANSAC with local optimization.

    This class extends the core RANSAC infrastructure while
    retaining the explicit local-optimization API used by
    PowerTwinAI's triangulation layer.
    """

    kMaxNumLocalTrials = 10

    def __init__(
        self,
        max_local_trials: int = kMaxNumLocalTrials,
        options: RANSACOptions | None = None,
        max_iterations: int | None = None,
        random_seed: int | None = None,
    ):
        if max_local_trials < 1:
            raise ValueError(
                "max_local_trials must be positive."
            )

        self.max_local_trials = min(
            int(max_local_trials),
            self.kMaxNumLocalTrials,
        )

        super().__init__(
            max_iterations=max_iterations,
            random_seed=random_seed,
            options=options,
        )

        self.support_measurer = (
            InlierSupportMeasurer()
        )

    # =====================================================
    # LOCAL OPTIMIZATION
    # =====================================================

    def optimize(
        self,
        model: Any,
        inlier_data: Sequence[Any],
        residual_function: Callable,
        estimator: Callable,
        max_residual: float,
        all_data: Sequence[Any] | None = None,
    ):
        """
        Locally optimize a RANSAC model.

        Local optimization evaluates every iteration against
        the complete data set when all_data is supplied.
        """

        if model is None:
            return None

        if all_data is None:
            all_data = list(inlier_data)
        else:
            all_data = list(all_data)

        current_inlier_data = list(
            inlier_data
        )

        if len(current_inlier_data) == 0:
            return None

        # -------------------------------------------------
        # Initial support over complete data.
        # -------------------------------------------------

        current_residuals = np.asarray(
            residual_function(
                model,
                all_data,
            ),
            dtype=float,
        )

        if len(current_residuals) != len(all_data):
            raise ValueError(
                "residual_function must return one "
                "residual for each data element."
            )

        current_mask = (
            current_residuals
            <= max_residual
        )

        current_support = (
            self._evaluate_support(
                current_residuals,
                current_mask,
            )
        )

        best_model = model
        best_residuals = current_residuals
        best_mask = current_mask
        best_support = current_support

        # -------------------------------------------------
        # Local optimization.
        # -------------------------------------------------

        for _ in range(
            self.max_local_trials
        ):

            current_indices = np.flatnonzero(
                best_mask
            )

            # A triangulation model requires at least
            # two observations. The generic RANSAC base
            # may use a different estimator-specific
            # minimum, so prefer the estimator's declared
            # value when available.
            min_num_samples = getattr(
                estimator,
                "kMinNumSamples",
                2,
            )

            if (
                len(current_indices)
                < min_num_samples
            ):
                break

            current_inlier_data = [
                all_data[index]
                for index in current_indices
            ]

            # ---------------------------------------------
            # Re-estimate from current complete inliers.
            # ---------------------------------------------

            try:
                optimized_model = estimator(
                    current_inlier_data
                )
            except Exception:
                break

            if optimized_model is None:
                break

            # ---------------------------------------------
            # Evaluate optimized model against ALL data.
            # ---------------------------------------------

            optimized_residuals = np.asarray(
                residual_function(
                    optimized_model,
                    all_data,
                ),
                dtype=float,
            )

            if len(optimized_residuals) != len(
                all_data
            ):
                raise ValueError(
                    "residual_function must return one "
                    "residual for each data element."
                )

            optimized_mask = (
                optimized_residuals
                <= max_residual
            )

            optimized_support = (
                self._evaluate_support(
                    optimized_residuals,
                    optimized_mask,
                )
            )

            # ---------------------------------------------
            # Support must improve.
            # ---------------------------------------------

            if not self._is_better(
                optimized_support,
                best_support,
            ):
                break

            old_num_inliers = (
                best_support.num_inliers
            )

            best_model = optimized_model
            best_residuals = (
                optimized_residuals
            )
            best_mask = optimized_mask
            best_support = optimized_support

            # ---------------------------------------------
            # Continue only if the number of inliers
            # actually increased.
            # ---------------------------------------------

            if (
                best_support.num_inliers
                <= old_num_inliers
            ):
                break

        optimized_indices = np.flatnonzero(
            best_mask
        )

        optimized_inlier_data = [
            all_data[index]
            for index in optimized_indices
        ]

        return (
            best_model,
            optimized_inlier_data,
        )

    # =====================================================
    # RESULT-COMPATIBLE API
    # =====================================================

    def optimize_result(
        self,
        model,
        data,
        residual_function,
        estimator,
        max_residual,
    ):
        """
        Return optimized model, support, residuals and mask.
        """

        data = list(data)

        residuals = np.asarray(
            residual_function(
                model,
                data,
            ),
            dtype=float,
        )

        if len(residuals) != len(data):
            raise ValueError(
                "residual_function must return one "
                "residual for each data element."
            )

        mask = (
            residuals
            <= max_residual
        )

        inlier_data = [
            data[index]
            for index, is_inlier
            in enumerate(mask)
            if is_inlier
        ]

        optimized = self.optimize(
            model=model,
            inlier_data=inlier_data,
            residual_function=residual_function,
            estimator=estimator,
            max_residual=max_residual,
            all_data=data,
        )

        if optimized is None:
            return (
                model,
                self._evaluate_support(
                    residuals,
                    mask,
                ),
                residuals,
                mask,
            )

        optimized_model, _ = optimized

        optimized_residuals = np.asarray(
            residual_function(
                optimized_model,
                data,
            ),
            dtype=float,
        )

        optimized_mask = (
            optimized_residuals
            <= max_residual
        )

        optimized_support = (
            self._evaluate_support(
                optimized_residuals,
                optimized_mask,
            )
        )

        return (
            optimized_model,
            optimized_support,
            optimized_residuals,
            optimized_mask,
        )

    # =====================================================
    # SUPPORT
    # =====================================================

    def _evaluate_support(
        self,
        residuals,
        inlier_mask,
    ):
        """
        Construct support from complete residual data.
        """

        residuals = np.asarray(
            residuals,
            dtype=float,
        )

        inlier_mask = np.asarray(
            inlier_mask,
            dtype=bool,
        )

        inlier_residuals = (
            residuals[inlier_mask]
        )

        return self.support_measurer.evaluate(
            inlier_residuals
        )

    @staticmethod
    def _is_better(
        candidate,
        current,
    ) -> bool:
        """
        Compare supports.

        Primary:
            larger inlier count.

        Secondary:
            smaller residual sum.
        """

        if (
            candidate.num_inliers
            != current.num_inliers
        ):
            return (
                candidate.num_inliers
                > current.num_inliers
            )

        return (
            candidate.residual_sum
            < current.residual_sum
        )


__all__ = [
    "LORANSAC",
]

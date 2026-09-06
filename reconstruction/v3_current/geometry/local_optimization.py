"""
Phase 3 - Local Optimization

Provides a generic local-optimization interface used by
LO-RANSAC after a promising model is found.
"""

from typing import Any, Callable, Sequence


class LocalOptimizer:
    """
    Perform local optimization on a candidate model.

    The estimator function receives the current inlier data
    and returns an improved model.
    """

    def optimize(
        self,
        model: Any,
        inlier_data: Sequence[Any],
        estimator: Callable[[Sequence[Any]], Any],
    ) -> Any:
        """
        Re-estimate a model using its current inlier data.

        Parameters
        ----------
        model
            Current candidate model.

        inlier_data
            Data classified as inliers for the model.

        estimator
            Function that estimates an improved model from
            the supplied inlier data.

        Returns
        -------
        Any
            Improved model returned by the estimator.
        """

        if not inlier_data:
            return model

        optimized_model = estimator(
            inlier_data
        )

        if optimized_model is None:
            return model

        return optimized_model
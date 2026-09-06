"""
Phase 3 - Support Measurement

RANSAC support comparison.

Supported COLMAP-style support strategies:

1. InlierSupport
   - maximize number of inliers
   - minimize residual sum as tie-breaker

2. UniqueInlierSupport
   - maximize unique inliers
   - maximize total inliers as secondary criterion
   - minimize residual sum as final criterion

3. MEstimatorSupport
   - maximize number of inliers
   - minimize MSAC-style truncated residual score
"""

from dataclasses import dataclass

import numpy as np


# ============================================================
# STANDARD INLIER SUPPORT
# ============================================================


@dataclass
class InlierSupport:
    """
    Standard RANSAC support.

    The primary criterion is the number of inliers.
    The secondary criterion is the sum of inlier residuals.
    """

    num_inliers: int = 0
    residual_sum: float = float("inf")

    def __lt__(self, other):
        """
        Return True when this support is worse than `other`.
        """

        if self.num_inliers != other.num_inliers:
            return (
                self.num_inliers
                < other.num_inliers
            )

        return (
            self.residual_sum
            > other.residual_sum
        )


class InlierSupportMeasurer:
    """
    Standard inlier support measurer.

    A candidate is better when:

    1. it has more inliers, or
    2. it has the same number of inliers but a lower
       total residual.
    """

    def evaluate(
        self,
        residuals,
        max_residual=None,
    ) -> InlierSupport:

        residuals = np.asarray(
            residuals,
            dtype=float,
        ).reshape(-1)

        finite = np.isfinite(
            residuals
        )

        if max_residual is None:
            inlier_mask = finite
        else:
            inlier_mask = (
                finite
                &
                (
                    residuals
                    <= max_residual
                )
            )

        inlier_residuals = (
            residuals[inlier_mask]
        )

        if len(inlier_residuals) == 0:
            return InlierSupport(
                num_inliers=0,
                residual_sum=float("inf"),
            )

        return InlierSupport(
            num_inliers=int(
                len(inlier_residuals)
            ),
            residual_sum=float(
                np.sum(
                    inlier_residuals
                )
            ),
        )

    def is_left_better(
        self,
        left: InlierSupport,
        right: InlierSupport,
    ) -> bool:

        if (
            left.num_inliers
            != right.num_inliers
        ):
            return (
                left.num_inliers
                > right.num_inliers
            )

        return (
            left.residual_sum
            < right.residual_sum
        )

    # Compatibility alias.
    IsLeftBetter = is_left_better


# ============================================================
# UNIQUE INLIER SUPPORT
# ============================================================


@dataclass
class UniqueInlierSupport:
    """
    Support that distinguishes unique inliers from
    total inlier observations.

    Primary criterion:
        unique inlier count

    Secondary criterion:
        total inlier count

    Final criterion:
        residual sum
    """

    num_unique_inliers: int = 0
    num_inliers: int = 0
    residual_sum: float = float("inf")

    def __lt__(self, other):
        if (
            self.num_unique_inliers
            != other.num_unique_inliers
        ):
            return (
                self.num_unique_inliers
                < other.num_unique_inliers
            )

        if (
            self.num_inliers
            != other.num_inliers
        ):
            return (
                self.num_inliers
                < other.num_inliers
            )

        return (
            self.residual_sum
            > other.residual_sum
        )


class UniqueInlierSupportMeasurer:
    """
    Unique-inlier support measurer.

    `unique_ids` identifies the geometric observation
    represented by each residual.

    Duplicate observations with the same ID contribute
    to total inliers but only once to unique inliers.
    """

    def evaluate(
        self,
        residuals,
        max_residual=None,
        unique_ids=None,
    ) -> UniqueInlierSupport:

        residuals = np.asarray(
            residuals,
            dtype=float,
        ).reshape(-1)

        if unique_ids is None:
            unique_ids = list(
                range(len(residuals))
            )
        else:
            unique_ids = list(unique_ids)

        if len(unique_ids) != len(residuals):
            raise ValueError(
                "unique_ids and residuals "
                "must have the same length."
            )

        finite = np.isfinite(
            residuals
        )

        if max_residual is None:
            inlier_mask = finite
        else:
            inlier_mask = (
                finite
                &
                (
                    residuals
                    <= max_residual
                )
            )

        inlier_indices = np.flatnonzero(
            inlier_mask
        )

        if len(inlier_indices) == 0:
            return UniqueInlierSupport(
                num_unique_inliers=0,
                num_inliers=0,
                residual_sum=float("inf"),
            )

        inlier_ids = {
            unique_ids[int(index)]
            for index in inlier_indices
        }

        residual_sum = float(
            np.sum(
                residuals[inlier_mask]
            )
        )

        return UniqueInlierSupport(
            num_unique_inliers=len(
                inlier_ids
            ),
            num_inliers=int(
                len(inlier_indices)
            ),
            residual_sum=residual_sum,
        )

    def is_left_better(
        self,
        left: UniqueInlierSupport,
        right: UniqueInlierSupport,
    ) -> bool:

        if (
            left.num_unique_inliers
            != right.num_unique_inliers
        ):
            return (
                left.num_unique_inliers
                > right.num_unique_inliers
            )

        if (
            left.num_inliers
            != right.num_inliers
        ):
            return (
                left.num_inliers
                > right.num_inliers
            )

        return (
            left.residual_sum
            < right.residual_sum
        )

    # Compatibility alias.
    IsLeftBetter = is_left_better


# ============================================================
# M-ESTIMATOR / MSAC SUPPORT
# ============================================================


@dataclass
class MEstimatorSupport:
    """
    M-estimator / MSAC-style support.

    The primary criterion is the number of inliers.
    The secondary criterion is the truncated residual score.
    """

    num_inliers: int = 0
    score: float = float("inf")

    @property
    def residual_sum(self):
        """
        Compatibility alias.

        The M-estimator score is the relevant residual
        objective for this support type.
        """

        return self.score

    def __lt__(self, other):
        if self.num_inliers != other.num_inliers:
            return (
                self.num_inliers
                < other.num_inliers
            )

        return (
            self.score
            > other.score
        )


class MEstimatorSupportMeasurer:
    """
    MSAC-style support measurer.

    Residuals larger than the inlier threshold are
    truncated to the threshold, preventing extreme
    outliers from dominating the score.
    """

    def evaluate(
        self,
        residuals,
        max_residual=None,
    ) -> MEstimatorSupport:

        residuals = np.asarray(
            residuals,
            dtype=float,
        ).reshape(-1)

        finite = np.isfinite(
            residuals
        )

        if max_residual is None:
            valid_residuals = np.where(
                finite,
                residuals,
                0.0,
            )

            score = float(
                np.sum(
                    valid_residuals
                )
            )

            num_inliers = int(
                np.sum(finite)
            )

            return MEstimatorSupport(
                num_inliers=num_inliers,
                score=score,
            )

        threshold = float(
            max_residual
        )

        if threshold < 0:
            raise ValueError(
                "max_residual must be non-negative."
            )

        truncated = np.where(
            finite,
            np.minimum(
                residuals,
                threshold,
            ),
            threshold,
        )

        score = float(
            np.sum(truncated)
        )

        inlier_mask = (
            finite
            &
            (
                residuals
                <= threshold
            )
        )

        return MEstimatorSupport(
            num_inliers=int(
                np.sum(inlier_mask)
            ),
            score=score,
        )

    def is_left_better(
        self,
        left: MEstimatorSupport,
        right: MEstimatorSupport,
    ) -> bool:

        if (
            left.num_inliers
            != right.num_inliers
        ):
            return (
                left.num_inliers
                > right.num_inliers
            )

        return (
            left.score
            < right.score
        )

    # Compatibility alias.
    IsLeftBetter = is_left_better


# ============================================================
# COLMAP-STYLE CLASS ALIASES
# ============================================================


UniqueInlierSupportMeasurer.Evaluate = (
    UniqueInlierSupportMeasurer.evaluate
)

MEstimatorSupportMeasurer.Evaluate = (
    MEstimatorSupportMeasurer.evaluate
)

InlierSupportMeasurer.Evaluate = (
    InlierSupportMeasurer.evaluate
)
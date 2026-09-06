"""
Phase 3 - Core RANSAC Infrastructure

PowerTwinAI implementation of the Phase 3 RANSAC infrastructure.

Includes:

- RANSACOptions
- Dynamic trial calculation
- RandomSampler
- CombinationSampler
- Support-based model selection
- Squared residual threshold semantics
- Final residual/inlier-mask calculation
- Deterministic sampling when a seed is supplied
- Exhaustive minimal-sample enumeration for small problems
"""

import math
import random

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable, Iterator, Sequence

from .support_measurement import (
    InlierSupport,
    InlierSupportMeasurer,
)


# =========================================================
# RANSAC OPTIONS
# =========================================================


@dataclass
class RANSACOptions:
    """
    Configuration for RANSAC.

    Values follow the Phase 3 specification.
    """

    max_error: float = 0.0
    min_inlier_ratio: float = 0.1
    confidence: float = 0.99
    dyn_num_trials_multiplier: float = 3.0
    min_num_trials: int = 0
    max_num_trials: int = 2**31 - 1
    random_seed: int = -1
    num_threads: int = 1

    def check(self) -> None:

        if self.max_error <= 0:
            raise ValueError(
                "max_error must be greater than zero."
            )

        if not 0.0 <= self.min_inlier_ratio <= 1.0:
            raise ValueError(
                "min_inlier_ratio must be in [0, 1]."
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "confidence must be in [0, 1]."
            )

        if self.dyn_num_trials_multiplier < 0:
            raise ValueError(
                "dyn_num_trials_multiplier must not be negative."
            )

        if self.min_num_trials < 0:
            raise ValueError(
                "min_num_trials must not be negative."
            )

        if self.max_num_trials < 0:
            raise ValueError(
                "max_num_trials must not be negative."
            )

        if self.min_num_trials > self.max_num_trials:
            raise ValueError(
                "min_num_trials must not exceed max_num_trials."
            )

        if self.random_seed < -1:
            raise ValueError(
                "random_seed must be >= -1."
            )

        if self.num_threads == 0 or self.num_threads < -1:
            raise ValueError(
                "num_threads must be -1 or a positive integer."
            )


# =========================================================
# RANSAC RESULT
# =========================================================


@dataclass
class RANSACResult:
    """
    Result produced by RANSAC.
    """

    model: Any = None
    support: InlierSupport | None = None
    inlier_data: list[Any] | None = None
    iterations: int = 0

    success: bool = False
    inlier_mask: list[bool] | None = None
    num_trials: int = 0


# =========================================================
# TRIAL CALCULATION
# =========================================================


def compute_num_trials(
    num_inliers: int,
    num_samples: int,
    min_num_samples: int,
    confidence: float,
    num_trials_multiplier: float,
) -> int:
    """
    Compute dynamic RANSAC trial count.

    The probability of obtaining an all-inlier minimal
    sample is computed sequentially.
    """

    if num_samples <= 0:
        return 0

    if min_num_samples <= 0:
        return 0

    if num_inliers < min_num_samples:
        return 2**31 - 1

    if num_inliers > num_samples:
        num_inliers = num_samples

    if confidence <= 0:
        return 0

    if confidence >= 1:
        return 2**31 - 1

    probability_inlier = 1.0

    for i in range(min_num_samples):

        numerator = num_inliers - i
        denominator = num_samples - i

        if numerator <= 0 or denominator <= 0:
            probability_inlier = 0.0
            break

        probability_inlier *= (
            numerator / denominator
        )

    probability_outlier = 1.0 - probability_inlier
    probability_failure = 1.0 - confidence

    if probability_outlier <= 0.0:
        return 1

    if probability_failure <= 0.0:
        return 2**31 - 1

    if probability_outlier >= 1.0:
        return 2**31 - 1

    denominator = math.log(
        probability_outlier
    )

    if denominator == 0.0:
        return 2**31 - 1

    trials = math.ceil(
        (
            math.log(probability_failure)
            / denominator
        )
        * num_trials_multiplier
    )

    return max(1, trials)


# Compatibility alias matching the documented naming.
ComputeNumTrials = compute_num_trials


# =========================================================
# RANDOM SAMPLER
# =========================================================


class RandomSampler:
    """
    Random minimal-sample sampler.

    This is the normal sampler for non-exhaustive RANSAC.
    """

    def __init__(
        self,
        num_samples: int,
        sample_size: int,
        random_seed: int = -1,
    ):
        if num_samples < 0:
            raise ValueError(
                "num_samples must be non-negative."
            )

        if sample_size <= 0:
            raise ValueError(
                "sample_size must be positive."
            )

        if sample_size > num_samples:
            raise ValueError(
                "sample_size must not exceed num_samples."
            )

        self.num_samples = int(
            num_samples
        )

        self.sample_size = int(
            sample_size
        )

        self.random_generator = random.Random(
            None
            if random_seed == -1
            else random_seed
        )

    def sample_indices(self) -> list[int]:

        return self.random_generator.sample(
            range(self.num_samples),
            self.sample_size,
        )

    def __iter__(self) -> Iterator[list[int]]:
        while True:
            yield self.sample_indices()


# =========================================================
# COMBINATION SAMPLER
# =========================================================


class CombinationSampler:
    """
    Exhaustive minimal-sample sampler.

    Generates every combination of `sample_size` indices
    exactly once.

    For triangulation, sample_size is normally 2.

    Example for N=4, sample_size=2:

        [0, 1]
        [0, 2]
        [0, 3]
        [1, 2]
        [1, 3]
        [2, 3]
    """

    def __init__(
        self,
        num_samples: int,
        sample_size: int,
    ):
        if num_samples < 0:
            raise ValueError(
                "num_samples must be non-negative."
            )

        if sample_size <= 0:
            raise ValueError(
                "sample_size must be positive."
            )

        if sample_size > num_samples:
            raise ValueError(
                "sample_size must not exceed num_samples."
            )

        self.num_samples = int(
            num_samples
        )

        self.sample_size = int(
            sample_size
        )

        self._combinations = list(
            combinations(
                range(self.num_samples),
                self.sample_size,
            )
        )

        self._index = 0

    @property
    def num_combinations(self) -> int:
        return len(
            self._combinations
        )

    def sample_indices(self) -> list[int]:

        if self._index >= len(
            self._combinations
        ):
            raise StopIteration

        sample = list(
            self._combinations[
                self._index
            ]
        )

        self._index += 1

        return sample

    def reset(self) -> None:
        self._index = 0

    def __iter__(self) -> Iterator[list[int]]:
        self.reset()

        for combination in self._combinations:
            yield list(combination)


# =========================================================
# RANSAC
# =========================================================


class RANSAC:
    """
    Generic robust model estimator.

    The estimator repeatedly:

    1. samples a minimal subset,
    2. estimates a candidate model,
    3. evaluates residuals,
    4. measures support,
    5. retains the best candidate,
    6. dynamically updates the stopping criterion.
    """

    def __init__(
        self,
        max_iterations: int | None = None,
        random_seed: int | None = None,
        options: RANSACOptions | None = None,
    ):

        if options is None:

            seed = (
                -1
                if random_seed is None
                else random_seed
            )

            if max_iterations is None:
                max_iterations = 100

            if max_iterations <= 0:
                raise ValueError(
                    "max_iterations must be greater than zero."
                )

            options = RANSACOptions(
                max_error=1.0,
                max_num_trials=max_iterations,
                min_num_trials=0,
                random_seed=seed,
            )

        options.check()

        self.options = options

        self.max_iterations = (
            options.max_num_trials
        )

        self.random_seed = (
            options.random_seed
        )

        self.random_generator = random.Random(
            None
            if options.random_seed == -1
            else options.random_seed
        )

        self.support_measurer = (
            InlierSupportMeasurer()
        )

    # -----------------------------------------------------
    # SAMPLER FACTORY
    # -----------------------------------------------------

    def _create_sampler(
        self,
        num_samples: int,
        sample_size: int,
        sampler: Any = None,
    ):

        if sampler is not None:

            if isinstance(
                sampler,
                type,
            ):
                return sampler(
                    num_samples,
                    sample_size,
                )

            return sampler

        return RandomSampler(
            num_samples=num_samples,
            sample_size=sample_size,
            random_seed=self.options.random_seed,
        )

    # -----------------------------------------------------
    # ESTIMATE
    # -----------------------------------------------------

    def estimate(
        self,
        data: Sequence[Any],
        sample_size: int,
        estimator: Callable[
            [Sequence[Any]],
            Any,
        ],
        residual_function: Callable[
            [Any, Sequence[Any]],
            Sequence[float],
        ],
        max_residual: float | None = None,
        sampler: Any = None,
    ) -> RANSACResult:
        """
        Estimate a robust model.

        `max_residual` is the residual threshold.

        When omitted, the squared RANSAC error threshold
        from RANSACOptions is used.

        `sampler` may be:

        - None: RandomSampler
        - CombinationSampler instance
        - sampler class
        - compatible sampler object exposing sample_indices()
        """

        if sample_size <= 0:
            raise ValueError(
                "sample_size must be greater than zero."
            )

        if len(data) < sample_size:
            return RANSACResult(
                iterations=0,
                num_trials=0,
                success=False,
                inlier_data=[],
                inlier_mask=[],
            )

        if max_residual is None:
            max_residual = (
                self.options.max_error
                * self.options.max_error
            )

        if max_residual < 0:
            raise ValueError(
                "max_residual must not be negative."
            )

        num_samples = len(data)

        # -------------------------------------------------
        # INITIAL DYNAMIC TRIAL COUNT
        # -------------------------------------------------

        # COLMAP initializes the dynamic trial estimate
        # using a fixed hypothetical sample population.
        # This avoids making the initial estimate dependent
        # on the actual dataset size.
        kNumSamples = 100000

        dynamic_max_trials = compute_num_trials(
            num_inliers=max(
                0,
                math.ceil(
                    self.options.min_inlier_ratio
                    * kNumSamples
                ),
            ),
            num_samples=kNumSamples,
            min_num_samples=sample_size,
            confidence=self.options.confidence,
            num_trials_multiplier=(
                self.options.dyn_num_trials_multiplier
            ),
        )

        dynamic_max_trials = min(
            dynamic_max_trials,
            self.options.max_num_trials,
        )

        dynamic_max_trials = max(
            dynamic_max_trials,
            self.options.min_num_trials,
        )

        if dynamic_max_trials <= 0:
            dynamic_max_trials = 1

        # -------------------------------------------------
        # SAMPLER
        # -------------------------------------------------

        sampler_object = self._create_sampler(
            num_samples=num_samples,
            sample_size=sample_size,
            sampler=sampler,
        )

        exhaustive_sampler = isinstance(
            sampler_object,
            CombinationSampler,
        )

        # -------------------------------------------------
        # WORKING STATE
        # -------------------------------------------------

        best_model = None
        best_support = None
        best_inlier_data: list[Any] = []
        best_inlier_mask: list[bool] = []

        iterations_performed = 0

        # -------------------------------------------------
        # RANSAC LOOP
        # -------------------------------------------------

        while (
            iterations_performed
            < dynamic_max_trials
            and iterations_performed
            < self.options.max_num_trials
        ):

            try:

                if hasattr(
                    sampler_object,
                    "sample_indices",
                ):
                    sample_indices = (
                        sampler_object.sample_indices()
                    )

                else:
                    sample_indices = next(
                        iter(sampler_object)
                    )

            except StopIteration:

                # Exhaustive sampler has been
                # completely consumed.
                break

            iterations_performed += 1

            sample_data = [
                data[index]
                for index in sample_indices
            ]

            candidate_model = estimator(
                sample_data
            )

            if candidate_model is None:
                continue

            residuals = residual_function(
                candidate_model,
                data,
            )

            if len(residuals) != num_samples:
                raise ValueError(
                    "residual_function must return one "
                    "residual for each data element."
                )

            support = (
                self.support_measurer.evaluate(
                    residuals,
                    max_residual,
                )
            )

            inlier_mask = [
                residual <= max_residual
                for residual in residuals
            ]

            inlier_data = [
                item
                for item, is_inlier in zip(
                    data,
                    inlier_mask,
                )
                if is_inlier
            ]

            # -------------------------------------------------
            # BEST MODEL
            # -------------------------------------------------

            if (
                best_support is None
                or self.support_measurer.is_left_better(
                    support,
                    best_support,
                )
            ):

                best_model = candidate_model
                best_support = support
                best_inlier_data = inlier_data
                best_inlier_mask = inlier_mask

                # ---------------------------------------------
                # DYNAMIC TRIAL UPDATE
                #
                # For CombinationSampler the minimum number
                # of trials remains authoritative. This
                # guarantees all combinations are considered.
                # ---------------------------------------------

                dynamic_max_trials = compute_num_trials(
                    num_inliers=support.num_inliers,
                    num_samples=num_samples,
                    min_num_samples=sample_size,
                    confidence=self.options.confidence,
                    num_trials_multiplier=(
                        self.options.dyn_num_trials_multiplier
                    ),
                )

                dynamic_max_trials = min(
                    dynamic_max_trials,
                    self.options.max_num_trials,
                )

                dynamic_max_trials = max(
                    dynamic_max_trials,
                    self.options.min_num_trials,
                )

                if dynamic_max_trials <= 0:
                    dynamic_max_trials = 1

        # -------------------------------------------------
        # FINAL MODEL VALIDATION
        # -------------------------------------------------

        if best_model is None:
            return RANSACResult(
                iterations=iterations_performed,
                num_trials=iterations_performed,
                success=False,
                inlier_data=[],
                inlier_mask=[],
            )

        final_residuals = residual_function(
            best_model,
            data,
        )

        if len(final_residuals) != num_samples:
            raise ValueError(
                "residual_function must return one "
                "residual for each data element."
            )

        final_support = (
            self.support_measurer.evaluate(
                final_residuals,
                max_residual,
            )
        )

        final_inlier_mask = [
            residual <= max_residual
            for residual in final_residuals
        ]

        final_inlier_data = [
            item
            for item, is_inlier in zip(
                data,
                final_inlier_mask,
            )
            if is_inlier
        ]

        success = (
            final_support.num_inliers
            >= sample_size
        )

        return RANSACResult(
            model=best_model,
            support=final_support,
            inlier_data=final_inlier_data,
            iterations=iterations_performed,
            success=success,
            inlier_mask=final_inlier_mask,
            num_trials=iterations_performed,
        )


# =========================================================
# LOWER-CASE COMPATIBILITY
# =========================================================


ransac = RANSAC


__all__ = [
    "RANSACOptions",
    "RANSACResult",
    "RandomSampler",
    "CombinationSampler",
    "RANSAC",
    "ransac",
    "compute_num_trials",
    "ComputeNumTrials",
]

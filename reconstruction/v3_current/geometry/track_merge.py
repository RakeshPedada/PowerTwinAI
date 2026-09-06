"""
Phase 3 - Track Merge

Provides safe merging of two feature tracks during
incremental reconstruction.
"""

from .track_operations import (
    FeatureTrack,
)


class TrackMerger:
    """
    Merge compatible feature tracks.

    Tracks are incompatible when they contain different
    feature observations for the same image.
    """

    def can_merge(
        self,
        left: FeatureTrack,
        right: FeatureTrack,
    ) -> bool:
        """
        Check whether two tracks can be merged safely.
        """

        for image_id, feature_id in right.observations.items():

            if image_id in left.observations:

                if left.observations[image_id] != feature_id:
                    return False

        return True

    def merge(
        self,
        left: FeatureTrack,
        right: FeatureTrack,
    ) -> FeatureTrack | None:
        """
        Merge two compatible tracks.

        Returns None when the tracks conflict.
        """

        if not self.can_merge(left, right):
            return None

        merged_observations = dict(
            left.observations
        )

        merged_observations.update(
            right.observations
        )

        return FeatureTrack(
            observations=merged_observations
        )

from collections import deque
"""
Phase 3 - Incremental Triangulator

Geometry-focused incremental triangulation operations:

    Create()
    Continue()
    Complete()
    Merge()
    Retriangulate()

The implementation accepts either:
    - a camera object providing CamFromImg()/cam_from_img(),
      optionally providing K/calibration_matrix, or
    - an Observation containing a projection_matrix.

Database, matching, mapper initialization, and bundle adjustment
remain outside Phase 3.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .track_operations import TriangulateTrack
from .estimate_triangulation import EstimateTriangulation
from .triangulation_estimator import TriangulationResidualType


# ============================================================
# Data structures
# ============================================================


@dataclass
class Observation:
    image_id: int
    point2D_idx: int
    xy: np.ndarray
    camera: object = None
    cam_from_world: np.ndarray | None = None
    point3D_id: int | None = None
    projection_matrix: np.ndarray | None = None


@dataclass
class Point3D:
    point3D_id: int
    xyz: np.ndarray
    observations: List[Tuple[int, int]] = field(
        default_factory=list
    )


@dataclass
class TriangulatorOptions:
    min_angle: float = 1.5
    create_max_angle_error: float = 2.0
    continue_max_angle_error: float = 2.0
    merge_max_reproj_error: float = 4.0
    complete_max_reproj_error: float = 4.0
    complete_max_transitivity: int = 5

    re_max_angle_error: float = 5.0
    re_min_ratio: float = 0.2
    re_max_trials: int = 1

    ignore_two_view_tracks: bool = True

    min_focal_length_ratio: float = 0.1
    max_focal_length_ratio: float = 10.0
    max_extra_param: float = 1.0

    random_seed: int = -1


# ============================================================
# IncrementalTriangulator
# ============================================================


class IncrementalTriangulator:

    def __init__(
        self,
        options: TriangulatorOptions | None = None,
    ):
        self.options = (
            options
            if options is not None
            else TriangulatorOptions()
        )

        self.points3D: Dict[int, Point3D] = {}

        self.next_point3D_id = 1

        self.correspondences: Dict[
            Tuple[int, int],
            List[Tuple[int, int]],
        ] = {}

        self.camera_validity_cache: Dict[
            int,
            bool,
        ] = {}

        self._observation_registry: Dict[
            Tuple[int, int],
            Observation,
        ] = {}

        self.retriangulation_trials = 0

    # ========================================================
    # CREATE
    # ========================================================

    def Create(
        self,
        observations: Sequence[Observation],
    ) -> Optional[Point3D]:

        observations = [
            observation
            for observation in observations
            if observation.point3D_id is None
        ]

        if len(observations) < 2:
            return None

        if (
            self.options.ignore_two_view_tracks
            and len(observations) == 2
        ):
            return None

        valid = self._filter_valid_observations(
            observations
        )

        if len(valid) < 2:
            return None

        points = [
            self._normalized_point(
                observation
            )
            for observation in valid
        ]

        poses = [
            observation.cam_from_world
            for observation in valid
        ]

        cameras = [
            None
            for _ in valid
        ]

        result = TriangulateTrack(
            points=points,
            cam_from_world=poses,
            cameras=cameras,
            min_angle=self.options.min_angle,
            create_max_angle_error=(
                self.options.create_max_angle_error
            ),
            random_seed=self.options.random_seed,
        )

        if not result.success:
            return None

        inlier_indices = np.flatnonzero(
            result.inlier_mask
        )

        if len(inlier_indices) < 2:
            return None

        point3D_id = self.next_point3D_id
        self.next_point3D_id += 1

        track_observations = []

        for index in inlier_indices:

            observation = valid[
                int(index)
            ]

            observation.point3D_id = (
                point3D_id
            )

            key = (
                observation.image_id,
                observation.point2D_idx,
            )

            track_observations.append(
                key
            )

        point3D = Point3D(
            point3D_id=point3D_id,
            xyz=np.asarray(
                result.point3D,
                dtype=float,
            ),
            observations=track_observations,
        )

        self.points3D[
            point3D_id
        ] = point3D

        return point3D

    def _CreateReprojectionTrack(
        self,
        observations: Sequence[Observation],
    ) -> Optional[Point3D]:
        """Create a new track using reprojection-error RANSAC.

        This is the CompleteImage() new-track path. Unlike Create(),
        which uses angular error for normal triangulation, completion
        uses pixel reprojection error as required by the Phase 3
        specification.
        """

        valid = self._filter_valid_observations(
            observations
        )

        if len(valid) < 2:
            return None

        points = [
            np.asarray(
                observation.xy,
                dtype=float,
            )
            for observation in valid
        ]

        poses = [
            observation.cam_from_world
            for observation in valid
        ]

        cameras = [
            observation.camera
            for observation in valid
        ]

        if any(
            pose is None
            for pose in poses
        ):
            return None

        result = EstimateTriangulation(
            points=points,
            cam_from_world=poses,
            cameras=cameras,
            min_tri_angle=np.deg2rad(
                self.options.min_angle
            ),
            residual_type=(
                TriangulationResidualType.REPROJECTION_ERROR
            ),
            max_error=(
                float(
                    self.options.complete_max_reproj_error
                )
            ),
            random_seed=(
                self.options.random_seed
            ),
        )

        if not result.success:
            return None

        inlier_indices = np.flatnonzero(
            result.inlier_mask
        )

        if len(inlier_indices) < 2:
            return None

        point3D_id = self.next_point3D_id
        self.next_point3D_id += 1

        track_observations = []

        for index in inlier_indices:

            observation = valid[
                int(index)
            ]

            observation.point3D_id = (
                point3D_id
            )

            key = (
                observation.image_id,
                observation.point2D_idx,
            )

            if key not in track_observations:
                track_observations.append(
                    key
                )

        point3D = Point3D(
            point3D_id=point3D_id,
            xyz=np.asarray(
                result.point3D,
                dtype=float,
            ),
            observations=track_observations,
        )

        self.points3D[
            point3D_id
        ] = point3D

        return point3D

    # ========================================================
    # CONTINUE
    # ========================================================

    def Continue(
            self,
            reference: Observation,
            candidates: Sequence[Observation],
            max_angle_error: float | None = None,
            ) -> bool:

            if reference.point3D_id is None:
                return False

            point3D = self.points3D.get(
                reference.point3D_id
            )

            if point3D is None:
                return False

            threshold_degrees = (
                self.options.continue_max_angle_error
                if max_angle_error is None
                else max_angle_error
            )

            best_candidate = None
            best_error = float("inf")

            for candidate in candidates:

                if candidate.point3D_id is not None:
                    continue

                if not self._observation_is_valid(
                    candidate
                ):
                    continue

                error = self._angular_error(
                    point3D.xyz,
                    candidate,
                )

                if error < best_error:
                    best_error = error
                    best_candidate = candidate

            if best_candidate is None:
                return False

            if best_error > np.deg2rad(
                threshold_degrees
            ):
                return False

            best_candidate.point3D_id = (
                point3D.point3D_id
            )

            key = (
                best_candidate.image_id,
                best_candidate.point2D_idx,
            )

            if key not in point3D.observations:
                point3D.observations.append(
                    key
                )

            return True



        # ========================================================
    # COMPLETE
        # ========================================================

    def Complete(
        self,
        point_or_observations,
        max_transitivity: Optional[int] = None,
        max_reproj_error: Optional[float] = None,
    ) -> bool:
        """Complete an existing track through the correspondence graph."""

        if isinstance(point_or_observations, Point3D):
            point = point_or_observations
            initial_keys = list(point.observations)

            initial_observations = [
                self._observation_registry[key]
                for key in initial_keys
                if key in self._observation_registry
            ]
        else:
            initial_observations = list(point_or_observations)

            point = None
            initial_keys = [
                (obs.image_id, obs.point2D_idx)
                for obs in initial_observations
            ]

            for candidate in self.points3D.values():
                if any(
                    key in candidate.observations
                    for key in initial_keys
                ):
                    point = candidate
                    break

        if point is None or not initial_observations:
            return False

        if max_transitivity is None:
            max_transitivity = self.options.complete_max_transitivity

        if max_reproj_error is None:
            max_reproj_error = self.options.complete_max_reproj_error

        queue = deque(
            (obs, 0)
            for obs in initial_observations
        )

        visited = {
            (obs.image_id, obs.point2D_idx)
            for obs in initial_observations
        }

        added = False
        max_error_sq = float(max_reproj_error) ** 2

        while queue:
            current_obs, depth = queue.popleft()

            if depth >= max_transitivity:
                continue

            current_key = (
                current_obs.image_id,
                current_obs.point2D_idx,
            )

            for candidate_entry in self.correspondences.get(
                current_key, []
            ):
                if isinstance(candidate_entry, Observation):
                    candidate = candidate_entry
                else:
                    candidate_key = tuple(candidate_entry)
                    candidate = self._observation_registry.get(candidate_key)
                    if candidate is None:
                        continue

                candidate_key = (
                    candidate.image_id,
                    candidate.point2D_idx,
                )

                if candidate_key in visited:
                    continue

                visited.add(candidate_key)

                if candidate.point3D_id is not None:
                    continue

                if candidate.cam_from_world is None:
                    continue

                if candidate.camera is None:
                    continue

                error = self._reprojection_error(
                    point.xyz,
                    candidate,
                )

                if not np.isfinite(error):
                    continue

                if error * error > max_error_sq:
                    continue

                candidate.point3D_id = point.point3D_id

                key = (
                    candidate.image_id,
                    candidate.point2D_idx,
                )

                if key not in point.observations:
                    point.observations.append(key)
                    added = True

                queue.append(
                    (candidate, depth + 1)
                )

        return added
    # ========================================================
    # MERGE
    # ========================================================

    def Merge(
        self,
        point3D_id1: int,
        point3D_id2: int,
    ) -> Optional[Point3D]:

        if point3D_id1 == point3D_id2:
            return self.points3D.get(
                point3D_id1
            )

        canonical_id1 = min(
            point3D_id1,
            point3D_id2,
        )

        canonical_id2 = max(
            point3D_id1,
            point3D_id2,
        )

        point1 = self.points3D.get(
            canonical_id1
        )

        point2 = self.points3D.get(
            canonical_id2
        )

        if point1 is None or point2 is None:
            return None

        length1 = len(
            point1.observations
        )

        length2 = len(
            point2.observations
        )

        if length1 == 0 or length2 == 0:
            return None

        merged_xyz = (
            length1 * point1.xyz
            +
            length2 * point2.xyz
        ) / (
            length1 + length2
        )

        merged_observations = []

        for key in (
            point1.observations
            +
            point2.observations
        ):

            if key in merged_observations:
                continue

            observation = (
                self._find_observation(
                    key
                )
            )

            if observation is None:
                return None

            error = self._reprojection_error(
                merged_xyz,
                observation,
            )

            if (
                error
                > self.options.merge_max_reproj_error
            ):
                return None

            merged_observations.append(
                key
            )

        point1.xyz = merged_xyz
        point1.observations = (
            merged_observations
        )

        for key in merged_observations:

            observation = (
                self._find_observation(
                    key
                )
            )

            if observation is not None:
                observation.point3D_id = (
                    point1.point3D_id
                )

        del self.points3D[
            point2.point3D_id
        ]

        return point1

    # ========================================================
    # RETRIANGULATE
    # ========================================================

    def Retriangulate(
        self,
        correspondences,
        num_tri_corrs: int,
        num_total_corrs: int,
    ) -> int:

        if num_total_corrs <= 0:
            return 0

        tri_ratio = (
            float(num_tri_corrs)
            / float(num_total_corrs)
        )

        if (
            tri_ratio
            >= self.options.re_min_ratio
        ):
            return 0

        if (
            self.retriangulation_trials
            >= self.options.re_max_trials
        ):
            return 0

        self.retriangulation_trials += 1

        changed = 0

        for observation1, observation2 in (
            correspondences
        ):

            if not self._observation_is_valid(
                observation1
            ):
                continue

            if not self._observation_is_valid(
                observation2
            ):
                continue

            has_point1 = (
                observation1.point3D_id
                is not None
            )

            has_point2 = (
                observation2.point3D_id
                is not None
            )

            if has_point1 and has_point2:
                continue

            if has_point1 != has_point2:

                reference = (
                    observation1
                    if has_point1
                    else observation2
                )

                candidate = (
                    observation2
                    if has_point1
                    else observation1
                )

                if self.Continue(
                    reference,
                    [candidate],
                    max_angle_error=(
                        self.options.re_max_angle_error
                    ),
                ):
                    changed += 1

                continue

            point = self.Create(
                [
                    observation1,
                    observation2,
                ]
            )

            if point is not None:
                changed += 1

        return changed

    # ========================================================
    # CORRESPONDENCES
    # ========================================================

    def AddCorrespondence(
        self,
        observation1: Observation,
        observation2: Observation,
    ) -> None:

        self.RegisterObservation(
            observation1
        )

        self.RegisterObservation(
            observation2
        )

        key1 = (
            observation1.image_id,
            observation1.point2D_idx,
        )

        key2 = (
            observation2.image_id,
            observation2.point2D_idx,
        )

        if key2 not in self.correspondences.get(
            key1,
            [],
        ):
            self.correspondences.setdefault(
                key1,
                [],
            ).append(key2)

        if key1 not in self.correspondences.get(
            key2,
            [],
        ):
            self.correspondences.setdefault(
                key2,
                [],
            ).append(key1)

    # ========================================================
    # OBSERVATION REGISTRY
    # ========================================================

    def RegisterObservation(
        self,
        observation: Observation,
    ) -> None:

        self._observation_registry[
            (
                observation.image_id,
                observation.point2D_idx,
            )
        ] = observation

    def RegisterObservations(
        self,
        observations: Iterable[Observation],
    ) -> None:

        for observation in observations:
            self.RegisterObservation(
                observation
            )

    def _find_observation(
        self,
        key,
    ):
        return self._observation_registry.get(
            key
        )

    # ========================================================
    # CAMERA NORMALIZATION
    # ========================================================

    @staticmethod
    def _normalized_point(
        observation: Observation,
    ) -> np.ndarray:

        point = np.asarray(
            observation.xy,
            dtype=float,
        ).reshape(-1)

        if point.size != 2:
            raise ValueError(
                "Observation xy must contain two values."
            )

        camera = observation.camera

        if camera is not None:

            method = getattr(
                camera,
                "CamFromImg",
                None,
            )

            if method is None:
                method = getattr(
                    camera,
                    "cam_from_img",
                    None,
                )

            if method is not None:
                try:
                    normalized = np.asarray(
                        method(point),
                        dtype=float,
                    ).reshape(-1)

                    if normalized.size >= 2:
                        return normalized[:2]

                except Exception:
                    pass

            # Common K fallback.
            K = None

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

            if K is not None:

                try:
                    ray = (
                        np.linalg.inv(K)
                        @ np.array(
                            [
                                point[0],
                                point[1],
                                1.0,
                            ]
                        )
                    )

                    if abs(ray[2]) > 1e-12:
                        return (
                            ray[:2]
                            / ray[2]
                        )

                except np.linalg.LinAlgError:
                    pass

        # If no camera is available, assume the caller supplied
        # normalized coordinates.
        return point

    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def _observation_is_valid(
        observation,
    ) -> bool:

        if observation is None:
            return False

        if observation.cam_from_world is None:
            return False

        matrix = np.asarray(
            observation.cam_from_world,
            dtype=float,
        )

        if matrix.shape != (4, 4):
            return False

        if not np.all(
            np.isfinite(matrix)
        ):
            return False

        xy = np.asarray(
            observation.xy,
            dtype=float,
        )

        return bool(
            xy.size == 2
            and np.all(
                np.isfinite(xy)
            )
        )

    def _filter_valid_observations(
        self,
        observations,
    ):

        return [
            observation
            for observation in observations
            if self._observation_is_valid(
                observation
            )
        ]

    # ========================================================
    # GEOMETRIC ERRORS
    # ========================================================

    def _reprojection_error(
        self,
        point3D,
        observation,
    ) -> float:

        X = np.append(
            np.asarray(
                point3D,
                dtype=float,
            ),
            1.0,
        )

        pose = np.asarray(
            observation.cam_from_world,
            dtype=float,
        )[:3, :]

        projected = (
            pose
            @ X
        )

        if abs(projected[2]) < 1e-12:
            return float("inf")

        projected_xy = (
            projected[:2]
            / projected[2]
        )

        # -------------------------------------------------
        # Pixel-space reprojection when a calibrated camera
        # is available. Merge/Complete thresholds are
        # specified in pixels.
        # -------------------------------------------------

        camera = getattr(
            observation,
            "camera",
            None,
        )

        if camera is not None:

            K = getattr(
                camera,
                "K",
                None,
            )

            if K is None:
                K = getattr(
                    camera,
                    "k",
                    None,
                )

            if K is None:
                K = getattr(
                    camera,
                    "calibration_matrix",
                    None,
                )

            if K is None:
                K = getattr(
                    camera,
                    "CalibrationMatrix",
                    None,
                )

            if K is not None:

                K = np.asarray(
                    K,
                    dtype=float,
                )

                if K.shape == (3, 3):

                    homogeneous = (
                        K
                        @ np.array(
                            [
                                projected_xy[0],
                                projected_xy[1],
                                1.0,
                            ],
                            dtype=float,
                        )
                    )

                    if abs(
                        homogeneous[2]
                    ) > 1e-12:

                        projected_pixel = (
                            homogeneous[:2]
                            / homogeneous[2]
                        )

                        observed_pixel = np.asarray(
                            observation.xy,
                            dtype=float,
                        )

                        return float(
                            np.linalg.norm(
                                projected_pixel
                                - observed_pixel
                            )
                        )

        # -------------------------------------------------
        # Fallback for camera-less observations:
        # compare normalized coordinates.
        # -------------------------------------------------

        point = self._normalized_point(
            observation
        )

        return float(
            np.linalg.norm(
                projected_xy - point
            )
        )

    def _angular_error(
        self,
        point3D,
        observation,
    ) -> float:

        center = self._camera_center(
            observation.cam_from_world
        )

        if center is None:
            return float("inf")

        world_ray = (
            np.asarray(
                point3D,
                dtype=float,
            )
            - center
        )

        world_norm = np.linalg.norm(
            world_ray
        )

        if world_norm <= 1e-12:
            return float("inf")

        world_ray /= world_norm

        # Transform the normalized camera ray into world space.
        point = self._normalized_point(
            observation
        )

        camera_ray = np.array(
            [
                point[0],
                point[1],
                1.0,
            ],
            dtype=float,
        )

        camera_norm = np.linalg.norm(
            camera_ray
        )

        if camera_norm <= 1e-12:
            return float("inf")

        camera_ray /= camera_norm

        pose = np.asarray(
            observation.cam_from_world,
            dtype=float,
        )

        rotation = pose[
            :3,
            :3,
        ]

        world_camera_ray = (
            rotation.T
            @ camera_ray
        )

        ray_norm = np.linalg.norm(
            world_camera_ray
        )

        if ray_norm <= 1e-12:
            return float("inf")

        world_camera_ray /= ray_norm

        cosine = np.clip(
            np.dot(
                world_ray,
                world_camera_ray,
            ),
            -1.0,
            1.0,
        )

        return float(
            np.arccos(
                cosine
            )
        )

    @staticmethod
    def _camera_center(
        cam_from_world,
    ):

        matrix = np.asarray(
            cam_from_world,
            dtype=float,
        )

        if matrix.shape != (4, 4):
            return None

        R = matrix[
            :3,
            :3,
        ]

        t = matrix[
            :3,
            3,
        ]

        center = (
            -R.T
            @ t
        )

        if not np.all(
            np.isfinite(center)
        ):
            return None

        return center

    # ========================================================
    # HIGH-LEVEL IMAGE / TRACK OPERATIONS
    # ========================================================

    def TriangulateImage(
        self,
        image_id: int,
        observations: Sequence[Observation],
    ) -> List[Point3D]:
        """Triangulate tracks associated with one image."""

        self.RegisterObservations(observations)

        created = []

        for observation in observations:

            if observation.point3D_id is not None:
                continue

            if not self._observation_is_valid(
                observation
            ):
                continue

            candidate_keys = self.correspondences.get(
                (
                    observation.image_id,
                    observation.point2D_idx,
                ),
                [],
            )

            candidates = []

            for key in candidate_keys:
                candidate = self._find_observation(key)

                if candidate is None:
                    continue

                if candidate.point3D_id is not None:
                    continue

                candidates.append(candidate)

            if not candidates:
                continue

            point = self.Create(
                [observation] + candidates
            )

            if point is not None:
                created.append(point)

        return created

    def CompleteImage(
        self,
        image_id: int,
        observations: Sequence[Observation],
    ) -> int:
        """Complete existing tracks and create new tracks for an image."""

        self.RegisterObservations(observations)

        completed = 0

        for observation in observations:

            if observation.point3D_id is None:
                continue

            point = self.points3D.get(
                observation.point3D_id
            )

            if point is None:
                continue

            if self.Complete(
                point,
                max_transitivity=(
                    self.options.complete_max_transitivity
                ),
                max_reproj_error=(
                    self.options.complete_max_reproj_error
                ),
            ):
                completed += 1

        # New tracks are created from remaining
        # correspondence groups.
        visited = set()

        for observation in observations:

            key = (
                observation.image_id,
                observation.point2D_idx,
            )

            if key in visited:
                continue

            if observation.point3D_id is not None:
                continue

            candidates = []

            for candidate_key in self.correspondences.get(
                key,
                [],
            ):

                candidate = self._find_observation(
                    candidate_key
                )

                if candidate is None:
                    continue

                if candidate.point3D_id is not None:
                    continue

                candidates.append(candidate)

            if len(candidates) < 1:
                continue

            group = [
                observation
            ] + candidates

            point = self._CreateReprojectionTrack(group)

            if point is None:
                continue

            completed += 1

            for candidate in group:
                visited.add(
                    (
                        candidate.image_id,
                        candidate.point2D_idx,
                    )
                )

        return completed

    def CompleteTracks(
        self,
        observations: Sequence[Observation],
    ) -> int:
        """Complete a collection of existing tracks."""

        added = 0

        for observation in observations:

            before = None

            if observation.point3D_id is not None:
                point = self.points3D.get(
                    observation.point3D_id
                )

                if point is not None:
                    before = len(
                        point.observations
                    )

            self.Complete(
                [observation]
            )

            if (
                before is not None
                and observation.point3D_id is not None
            ):
                point = self.points3D.get(
                    observation.point3D_id
                )

                if point is not None:
                    added += max(
                        0,
                        len(point.observations)
                        - before,
                    )

        return added

    def CompleteAllTracks(
        self,
        observations: Sequence[Observation] | None = None,
    ) -> int:
        """Complete all registered tracks."""

        if observations is None:
            observations = list(
                self._observation_registry.values()
            )

        return self.CompleteTracks(
            observations
        )

    def MergeTracks(
        self,
        point_pairs,
    ) -> int:
        """Merge the supplied 3D point pairs."""

        merged = 0

        for point_id1, point_id2 in point_pairs:

            if self.Merge(
                point_id1,
                point_id2,
            ) is not None:
                merged += 1

        return merged

    # ========================================================
    # ALL TRACK OPERATIONS
    # ========================================================

    def MergeAllTracks(
        self,
        point_pairs,
    ) -> int:

        merged = 0

        for point_id1, point_id2 in point_pairs:

            if self.Merge(
                point_id1,
                point_id2,
            ) is not None:
                merged += 1

        return merged

    # ========================================================
    # CAMERA PARAMETER VALIDATION
    # ========================================================

    def IsCameraValid(
        self,
        camera_id: int,
        camera,
    ) -> bool:

        camera_id = int(
            camera_id
        )

        if camera_id in (
            self.camera_validity_cache
        ):
            return self.camera_validity_cache[
                camera_id
            ]

        if camera is None:
            valid = True

        else:

            method = getattr(
                camera,
                "HasBogusParams",
                None,
            )

            if method is None:
                method = getattr(
                    camera,
                    "has_bogus_params",
                    None,
                )

            if method is None:
                valid = True

            else:

                try:

                    valid = not bool(
                        method(
                            self.options.min_focal_length_ratio,
                            self.options.max_focal_length_ratio,
                            self.options.max_extra_param,
                        )
                    )

                except Exception:
                    valid = False

        self.camera_validity_cache[
            camera_id
        ] = valid

        return valid


# ============================================================
# Factory
# ============================================================


def create_incremental_triangulator(
    options=None,
):
    return IncrementalTriangulator(
        options=options
    )

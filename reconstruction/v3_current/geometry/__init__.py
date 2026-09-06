"""
PowerTwinAI Geometry Package
Phase 3 - Robust Geometric Estimation and Triangulation
"""

from .ransac import (
    RANSAC,
    RANSACOptions,
    RANSACResult,
)

from .lo_ransac import (
    LORANSAC,
)

from .support_measurement import (
    InlierSupport,
    InlierSupportMeasurer,
)

from .triangulation_estimator import (
    TriangulationEstimator,
    TriangulationResidualType,
)

from .triangulation_residual import (
    TriangulationResidual,
    calculate_angular_residuals,
    calculate_reprojection_residuals,
)

from .robust_triangulation import (
    RobustTriangulation,
    triangulate_robust,
)

from .estimate_triangulation import (
    EstimateTriangulation,
    estimate_triangulation,
    EstimateTriangulationResult,
    TriangulationPointData,
    TriangulationPoseData,
)

from .track_operations import (
    FeatureTrack,
    TriangulateTrack,
    TriangulateTrackResult,
)

from .track_merge import (
    TrackMerger,
)

from .incremental_triangulator import (
    IncrementalTriangulator,
    TriangulatorOptions,
    Observation,
    Point3D,
    create_incremental_triangulator,
)

from .geometry_manager import (
    GeometryManager,
)

__all__ = [
    "RANSAC",
    "RANSACOptions",
    "RANSACResult",
    "LORANSAC",
    "InlierSupport",
    "InlierSupportMeasurer",
    "TriangulationEstimator",
    "TriangulationResidualType",
    "TriangulationResidual",
    "calculate_angular_residuals",
    "calculate_reprojection_residuals",
    "RobustTriangulation",
    "triangulate_robust",
    "EstimateTriangulation",
    "estimate_triangulation",
    "EstimateTriangulationResult",
    "TriangulationPointData",
    "TriangulationPoseData",
    "FeatureTrack",
    "TriangulateTrack",
    "TriangulateTrackResult",
    "TrackMerger",
    "IncrementalTriangulator",
    "TriangulatorOptions",
    "Observation",
    "Point3D",
    "create_incremental_triangulator",
    "GeometryManager",
]
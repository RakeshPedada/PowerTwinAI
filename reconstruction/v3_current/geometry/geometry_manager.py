"""
Phase 3 - Geometry Manager

Provides a single integration interface between the Phase 3
geometry modules and the PowerTwinAI reconstruction pipeline.
"""

from .robust_triangulation import RobustTriangulation


class GeometryManager:
    """
    Manage robust geometry operations for PowerTwinAI.
    """

    def __init__(self):
        self.robust_triangulation = RobustTriangulation()

    def triangulate(
        self,
        observations,
        projection_matrices,
        max_residual=1.0,
    ):
        """
        Estimate a robust 3D point from multiple observations.
        """

        return self.robust_triangulation.estimate(
            observations,
            projection_matrices,
            max_residual,
        )
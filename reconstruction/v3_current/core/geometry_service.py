"""
Phase 3 - Geometry Service

Connects the Phase 3 geometry subsystem to the PowerTwinAI
core architecture.
"""

from geometry import GeometryManager


class GeometryService:
    """
    Core service for robust geometry operations.
    """

    def __init__(self):
        self.geometry_manager = GeometryManager()

    def triangulate_point(
        self,
        observations,
        projection_matrices,
        max_residual=1.0,
    ):
        """
        Triangulate a robust 3D point.
        """

        return self.geometry_manager.triangulate(
            observations,
            projection_matrices,
            max_residual,
        )

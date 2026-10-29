"""
ml/config.py
------------
Configuration dataclass and environment variable loader for the ML-based
image-pair candidate selection layer in PowerTwinAI.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MLPairRankingConfig:
    """
    Configuration options for ML candidate image pair selection.
    """
    enabled: bool = field(
        default_factory=lambda: os.environ.get("ML_PAIR_RANKING_ENABLED", "true").lower() in ("true", "1", "yes")
    )
    top_k: int = field(
        default_factory=lambda: int(os.environ.get("ML_TOP_K", "8"))
    )
    min_pairs_per_image: int = field(
        default_factory=lambda: int(os.environ.get("ML_MIN_PAIRS_PER_IMAGE", "4"))
    )
    similarity_threshold: float = field(
        default_factory=lambda: float(os.environ.get("ML_SIMILARITY_THRESHOLD", "0.0"))
    )
    max_total_pairs: Optional[int] = field(
        default_factory=lambda: (
            int(os.environ["ML_MAX_TOTAL_PAIRS"]) if "ML_MAX_TOTAL_PAIRS" in os.environ else None
        )
    )
    embedding_method: str = "spatial_pyramid"  # "spatial_pyramid", "onnx", or "auto"
    ensure_connected: bool = True
    auto_bridge_components: bool = True
    fallback_to_exhaustive_if_failed: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "MLPairRankingConfig":
        """Build config from a dictionary (e.g. from reconstruction preset or JSON)."""
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "top_k": self.top_k,
            "min_pairs_per_image": self.min_pairs_per_image,
            "similarity_threshold": self.similarity_threshold,
            "max_total_pairs": self.max_total_pairs,
            "embedding_method": self.embedding_method,
            "ensure_connected": self.ensure_connected,
            "auto_bridge_components": self.auto_bridge_components,
            "fallback_to_exhaustive_if_failed": self.fallback_to_exhaustive_if_failed,
        }

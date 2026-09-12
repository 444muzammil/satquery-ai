"""
SatQuery AI — RS VLM Provider Abstraction Layer

Provider-agnostic interface for remote sensing vision-language models.
Supports RunPod, local HuggingFace, and classical CV fallback providers.
"""

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

logger = logging.getLogger("satquery.providers")


class TaskType(str, Enum):
    """Supported RS analysis tasks."""
    VQA = "vqa"
    CAPTION = "caption"
    GROUNDING = "grounding"
    CHANGE_DETECTION = "change_detection"
    CHANGE_VQA = "change_vqa"
    OPTICAL_SAR_FUSION = "fusion"


class Modality(str, Enum):
    """Image modality types."""
    OPTICAL_RGB = "rgb"
    MULTISPECTRAL = "multispectral"
    SAR = "sar"
    UNKNOWN = "unknown"


@dataclass
class ImageInput:
    """Standardized image input for provider calls."""
    data: str                        # Base64-encoded image data
    role: str = "primary"            # "primary" | "temporal" | "sar"
    modality: Modality = Modality.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)

    def clean_base64(self) -> str:
        """Return base64 data without data URI prefix."""
        if "," in self.data:
            return self.data.split(",", 1)[1]
        return self.data


@dataclass
class RegionBox:
    """A detected spatial region."""
    box: List[float]                 # [ymin_pct, xmin_pct, ymax_pct, xmax_pct]
    label: str = ""
    actual_pct: float = 0.0
    area_px: int = 0
    confidence: float = -1.0        # -1 means not reported


@dataclass
class RSVLMResponse:
    """Standardized response from any RS VLM provider."""
    text: str                        # Model's textual response
    regions: List[Dict[str, Any]] = field(default_factory=list)
    contours: List[Any] = field(default_factory=list)
    confidence: float = -1.0        # Model-reported confidence (0-1), -1 = not available
    model_name: str = "Unknown"     # Name of model that produced this
    is_fallback: bool = False       # Whether fallback was used
    latency_ms: float = 0.0        # Inference time in milliseconds
    raw_response: Optional[Dict[str, Any]] = None  # Full response for debugging
    error: Optional[str] = None     # Error message if failed
    coverage_pct: Optional[str] = None  # Coverage percentage if applicable

    @property
    def success(self) -> bool:
        return self.error is None and bool(self.text)


class RSVLMProvider(ABC):
    """
    Abstract base class for Remote Sensing VLM providers.

    Implementations:
    - RunPodProvider: HTTP client to RunPod-hosted models (EarthDial, GeoChat, etc.)
    - LocalProvider: HuggingFace transformers local inference
    - ClassicalFallbackProvider: Deterministic CV analysis (honest about limitations)
    """

    @abstractmethod
    async def analyze(
        self,
        images: List[ImageInput],
        query: str,
        task: TaskType,
        config: Optional[Dict[str, Any]] = None,
    ) -> RSVLMResponse:
        """
        Analyze one or more images with a natural-language query.

        Args:
            images: List of ImageInput objects (1 for single-image, 2 for pairs)
            query: Natural-language query
            task: Task type enum
            config: Optional task-specific parameters

        Returns:
            RSVLMResponse with text, optional regions, confidence, and metadata
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is available and healthy."""
        ...

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Return this provider's capability declaration."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""
        ...

    def _measure_latency(self, start_time: float) -> float:
        """Calculate elapsed time in milliseconds."""
        return round((time.time() - start_time) * 1000, 1)

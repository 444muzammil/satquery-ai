"""
SatQuery AI — Local HuggingFace Provider Stub

Infrastructure for future local Remote Sensing Vision-Language Model loading.
Provides a stubbed implementation of RSVLMProvider that does not load model
weights by default and adheres to local deployment configuration.
"""

import time
import logging
from typing import Any, Dict, List, Optional

from providers.base import RSVLMProvider, RSVLMResponse, ImageInput, TaskType
from config import settings
from registry import get_model_info

logger = logging.getLogger("satquery.providers.local")


class LocalProvider(RSVLMProvider):
    """
    Local HuggingFace provider stub for remote sensing VLM inference.

    Allows future on-device execution of models like EarthDial when
    RS_VLM_LOCAL is enabled and deep learning runtimes are present.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize LocalProvider stub without loading weights into memory.

        Args:
            model_name: Model identifier or local checkpoint path.
            device: Target execution device ('cpu', 'cuda', etc.).
        """
        self.model_name: str = model_name or settings.RS_VLM_MODEL
        self.device: str = device or "cpu"
        self._model: Any = None
        self._tokenizer: Any = None
        self._processor: Any = None
        self._is_loaded: bool = False

    @property
    def name(self) -> str:
        """Human-readable provider name."""
        return "Local/HuggingFace"

    async def health_check(self) -> bool:
        """Return True only if the model is loaded."""
        return self._is_loaded and self._model is not None

    def get_capabilities(self) -> Dict[str, Any]:
        """Return earthdial capabilities from registry."""
        model_info = get_model_info("earthdial")
        if model_info and "capabilities" in model_info:
            return dict(model_info["capabilities"])
        return {
            "vqa": True,
            "caption": True,
            "grounding": "partial",
            "change_detection": False,
            "change_vqa": False,
            "sar_analysis": True,
            "multispectral": True,
        }

    def _load_model(self) -> Optional[str]:
        """
        Protected method to structure local model loading.

        Checks configuration and verifies transformers and torch availability.
        Does not load actual heavy weights in this stub implementation.

        Returns:
            Optional[str]: Error message string on failure, or None on success.
        """
        if not settings.RS_VLM_LOCAL:
            error_msg = "Local model loading is disabled. Set RS_VLM_LOCAL=true to enable."
            logger.warning(error_msg)
            return error_msg

        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            error_msg = f"Missing dependencies: {exc}. Ensure transformers + torch are installed."
            logger.error(error_msg)
            return error_msg

        try:
            logger.info(
                "Initializing local model structure for %s on %s",
                self.model_name,
                self.device,
            )
            # Model loading structure (for future implementation):
            # self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
            # self._processor = transformers.AutoProcessor.from_pretrained(self.model_name)
            # self._model = transformers.AutoModelForVision2Seq.from_pretrained(
            #     self.model_name, torch_dtype=torch.float16
            # ).to(self.device)
            self._model = object()
            self._is_loaded = True
            return None
        except Exception as exc:
            error_msg = f"Failed to initialize local model {self.model_name}: {exc}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    async def analyze(
        self,
        images: List[ImageInput],
        query: str,
        task: TaskType,
        config: Optional[Dict[str, Any]] = None,
    ) -> RSVLMResponse:
        """
        Analyze images with a query using the local model.

        If model is not loaded, returns an informative RSVLMResponse error.
        """
        start_time = time.time()

        if not self._is_loaded or self._model is None:
            logger.warning("Local model %s requested but not loaded.", self.model_name)
            return RSVLMResponse(
                text="",
                model_name=self.name,
                latency_ms=self._measure_latency(start_time),
                error="Local model not loaded. Set RS_VLM_LOCAL=true and ensure transformers + torch are installed.",
            )

        # Structure for future local inference execution
        return RSVLMResponse(
            text="Local inference completed.",
            model_name=self.name,
            latency_ms=self._measure_latency(start_time),
        )

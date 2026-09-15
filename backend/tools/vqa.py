"""
SatQuery AI — VQA Tool

Visual Question Answering via RS VLM provider or classical fallback.
"""

import logging
from typing import Any, Dict

from providers.base import RSVLMProvider, RSVLMResponse, ImageInput, TaskType

logger = logging.getLogger("satquery.tools.vqa")


async def execute_vqa(
    query: str,
    image_base64: str,
    metadata: Dict[str, Any],
    provider: RSVLMProvider,
) -> Dict[str, Any]:
    """
    Execute Visual Question Answering.

    Routes to RS VLM provider (EarthDial/GeoChat) or falls back to classical CV.

    Args:
        query: Natural-language question about the image
        image_base64: Base64-encoded image
        metadata: Image metadata
        provider: RS VLM provider instance

    Returns:
        Tool result dict
    """
    image = ImageInput(data=image_base64, role="primary", metadata=metadata)

    response = await provider.analyze(
        images=[image],
        query=query,
        task=TaskType.VQA,
    )

    return {
        "status": "success" if response.success else "error",
        "summary": response.text if response.success else f"System was unable to analyze due to API Error: {response.error}",
        "fallback": response.is_fallback,
        "model": response.model_name,
        "confidence": response.confidence,
        "latency_ms": response.latency_ms,
    }

"""
SatQuery AI — Captioning / Scene Description Tool

Scene description via RS VLM provider or classical fallback.
"""

import logging
from typing import Any, Dict

from providers.base import RSVLMProvider, RSVLMResponse, ImageInput, TaskType

logger = logging.getLogger("satquery.tools.caption")

# Standard RS captioning prompt
CAPTION_PROMPT = (
    "Provide a comprehensive remote-sensing scene description detailing "
    "land cover, spatial layout, and infrastructure."
)


async def execute_caption(
    query: str,
    image_base64: str,
    metadata: Dict[str, Any],
    provider: RSVLMProvider,
) -> Dict[str, Any]:
    """
    Execute scene captioning / description.

    Routes to RS VLM provider or falls back to classical spectral analysis.

    Args:
        query: User's query (or default captioning prompt)
        image_base64: Base64-encoded image
        metadata: Image metadata
        provider: RS VLM provider instance

    Returns:
        Tool result dict
    """
    # Use the standard caption prompt if query is generic
    effective_query = query if query.strip() else CAPTION_PROMPT

    image = ImageInput(data=image_base64, role="primary", metadata=metadata)

    response = await provider.analyze(
        images=[image],
        query=effective_query,
        task=TaskType.CAPTION,
    )

    return {
        "status": "success" if response.success else "error",
        "summary": response.text,
        "fallback": response.is_fallback,
        "model": response.model_name,
        "confidence": response.confidence,
        "latency_ms": response.latency_ms,
    }

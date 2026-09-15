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

    if not response.success:
        try:
            import cv2
            import numpy as np
            import base64
            
            clean = image_base64.split(",")[1] if "," in image_base64 else image_base64
            arr = np.frombuffer(base64.b64decode(clean), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            
            if img is not None:
                h, w = img.shape[:2]
                total_pixels = h * w
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                
                veg_mask = cv2.inRange(hsv, np.array([30, 20, 20]), np.array([90, 255, 255]))
                water_mask = cv2.inRange(hsv, np.array([45, 15, 10]), np.array([145, 255, 230]))
                urban_mask = cv2.inRange(hsv, np.array([0, 0, 30]), np.array([180, 80, 240]))
                urban_mask = cv2.bitwise_and(urban_mask, cv2.bitwise_not(cv2.bitwise_or(veg_mask, water_mask)))
                
                veg_pct = (cv2.countNonZero(veg_mask) / total_pixels) * 100
                water_pct = (cv2.countNonZero(water_mask) / total_pixels) * 100
                urban_pct = (cv2.countNonZero(urban_mask) / total_pixels) * 100
                
                fallback_text = (
                    f"Due to temporary API limitations, the system automatically engaged the Classical Spectral Heuristic Fallback. "
                    f"Telemetry decomposition reveals the scene comprises approximately {veg_pct:.1f}% vegetation, "
                    f"{water_pct:.1f}% hydrological features, and {urban_pct:.1f}% structural or urban development. "
                    f"The remaining {(100 - (veg_pct + water_pct + urban_pct)):.1f}% consists of undefined or barren land cover."
                )
                
                return {
                    "status": "success",
                    "summary": fallback_text,
                    "fallback": True,
                    "model": "Classical Spectral Heuristic (Fallback)",
                    "confidence": 0.5,
                    "latency_ms": response.latency_ms,
                }
        except Exception as e:
            logger.error(f"Classical caption fallback failed: {e}")

    return {
        "status": "success" if response.success else "error",
        "summary": response.text if response.success else f"System was unable to analyze due to API Error: {response.error}",
        "fallback": response.is_fallback,
        "model": response.model_name,
        "confidence": response.confidence,
        "latency_ms": response.latency_ms,
    }

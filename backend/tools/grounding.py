"""
SatQuery AI — Grounding Tool

Text-guided spatial region localization via external grounding model,
RS VLM, or classical CV HSV-based fallback.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from providers.base import RSVLMProvider, ImageInput, TaskType

logger = logging.getLogger("satquery.tools.grounding")

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


async def execute_grounding(
    target: str,
    image_base64: str,
    metadata: Dict[str, Any],
    provider: RSVLMProvider,
    grounding_api_url: str = "",
    grounding_api_key: str = "",
) -> Dict[str, Any]:
    """
    Execute text-guided region grounding.

    Tries: external grounding model → RS VLM → classical CV fallback.

    Args:
        target: Object/feature to locate
        image_base64: Base64-encoded image
        metadata: Image metadata
        provider: RS VLM provider
        grounding_api_url: Optional dedicated grounding model endpoint
        grounding_api_key: Optional API key for grounding model

    Returns:
        Tool result dict with regions and contours
    """
    # Try dedicated grounding model first
    if grounding_api_url:
        result = _try_external_grounding(image_base64, target, grounding_api_url, grounding_api_key)
        if result:
            return result

    # Try RS VLM for text-based bbox extraction
    image = ImageInput(data=image_base64, role="primary", metadata=metadata)
    response = await provider.analyze(
        images=[image],
        query=f"Locate and provide bounding box coordinates for: {target}",
        task=TaskType.GROUNDING,
    )

    if response.success and response.regions:
        return {
            "status": "success",
            "summary": f"RS VLM identified spatial boundaries for '{target}'.",
            "regions": response.regions,
            "contours": response.contours,
            "fallback": response.is_fallback,
            "model": response.model_name,
        }

    # Classical CV fallback — always available
    if OPENCV_AVAILABLE:
        img = _decode_base64_to_cv2(image_base64)
        if img is not None:
            regions, contours = classical_grounding(img, target)
            summary = (
                f"Classical spatial heuristic localized {len(regions)} "
                f"regions for '{target}'."
            )
            return {
                "status": "success",
                "summary": summary,
                "regions": regions,
                "contours": contours,
                "fallback": True,
                "model": "Classical CV HSV Segmentation (Fallback)",
            }

    return {
        "status": "success",
        "summary": f"No spatial regions identified for '{target}'.",
        "regions": [],
        "contours": [],
        "fallback": True,
        "model": "No Grounding Model Available",
    }


def classical_grounding(
    img: np.ndarray, target: str
) -> Tuple[List[Dict[str, Any]], List[list]]:
    """
    Classical CV grounding via HSV color-space segmentation.

    Keyword-matched thresholding for common RS targets:
    water, vegetation, urban/built-up, barren/soil.
    Falls back to edge-based saliency for unknown targets.

    This is a deterministic heuristic — not a learned model.

    Args:
        img: OpenCV BGR image
        target: Target feature to locate

    Returns:
        Tuple of (regions list, contours list)
    """
    h, w = img.shape[:2]
    image_area = h * w
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    t_low = target.lower()

    # Keyword-matched HSV thresholding
    if any(kw in t_low for kw in ["water", "river", "lake", "canal", "flood", "ocean", "pond"]):
        m1 = cv2.inRange(hsv, np.array([85, 30, 30]), np.array([140, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([5, 10, 40]), np.array([30, 220, 245]))
        mask = cv2.bitwise_or(m1, m2)
    elif any(kw in t_low for kw in ["vegetation", "forest", "crop", "tree", "agricultural", "farm", "green"]):
        mask = cv2.inRange(hsv, np.array([25, 20, 20]), np.array([95, 255, 255]))
    elif any(kw in t_low for kw in ["build", "urban", "road", "railway", "structure", "development", "industrial"]):
        mask = cv2.inRange(hsv, np.array([0, 0, 85]), np.array([180, 50, 255]))
    elif any(kw in t_low for kw in ["barren", "soil", "sand", "dirt", "desert"]):
        mask = cv2.inRange(hsv, np.array([10, 15, 90]), np.array([35, 130, 255]))
    else:
        # Edge-based saliency for arbitrary targets
        edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 70, 170)
        mask = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions, valid_contours = [], []
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:30]:
        area = cv2.contourArea(c)
        if area > max(90, int(image_area * 0.0008)):
            x, y, bw, bh = cv2.boundingRect(c)
            regions.append({
                "box": [
                    round((y / h) * 100, 1),
                    round((x / w) * 100, 1),
                    round(((y + bh) / h) * 100, 1),
                    round(((x + bw) / w) * 100, 1),
                ],
                "label": target.title(),
                "actual_pct": (area / image_area) * 100.0,
                "area_px": area,
            })
            valid_contours.append(c.tolist())

    return regions, valid_contours


def _try_external_grounding(
    image_base64: str, target: str, url: str, api_key: str
) -> Optional[Dict[str, Any]]:
    """Try a dedicated external grounding model endpoint."""
    import requests

    clean = image_base64.split(",")[1] if "," in image_base64 else image_base64
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = requests.post(
            url,
            json={"image": clean, "target": target},
            headers=headers,
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            if "regions" in data:
                return {
                    "status": "success",
                    "summary": f"Identified spatial boundaries for '{target}' via Grounding Model.",
                    "regions": data["regions"],
                    "contours": data.get("contours", []),
                    "fallback": False,
                    "model": "Grounding DINO-RS",
                }
    except Exception as exc:
        logger.warning(f"External grounding model failed: {exc}")

    return None


def _decode_base64_to_cv2(b64_str: str) -> Optional[np.ndarray]:
    """Decode base64 string to OpenCV image."""
    import base64

    if not OPENCV_AVAILABLE or not b64_str:
        return None
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        img_bytes = base64.b64decode(b64_str)
        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    except Exception:
        return None

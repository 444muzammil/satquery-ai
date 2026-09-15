"""
SatQuery AI — Change Detection Tool

Bi-temporal change analysis via specialist model, agentic VLM orchestration,
or classical CV pixel differencing.
"""

import base64
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from providers.base import RSVLMProvider, ImageInput, TaskType

logger = logging.getLogger("satquery.tools.change")

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


async def execute_change_detection(
    target: str,
    image_a_base64: str,
    image_b_base64: str,
    metadata_a: Dict[str, Any],
    metadata_b: Dict[str, Any],
    provider: RSVLMProvider,
    change_model_url: str = "",
    change_model_key: str = "",
) -> Dict[str, Any]:
    """
    Execute bi-temporal change analysis.

    Strategy:
    1. Try dedicated change specialist model (if configured)
    2. Try agentic VLM approach (describe each image → compare)
    3. Fall back to classical CV pixel differencing

    Args:
        target: What to look for in changes
        image_a_base64: Before image (base64)
        image_b_base64: After image (base64)
        metadata_a: Before image metadata
        metadata_b: After image metadata
        provider: RS VLM provider
        change_model_url: Optional dedicated change model endpoint
        change_model_key: Optional API key for change model

    Returns:
        Tool result dict with regions, contours, and change description
    """
    # Strategy 1: Dedicated change specialist
    if change_model_url:
        result = _try_external_change_model(
            image_a_base64, image_b_base64, target,
            change_model_url, change_model_key,
        )
        if result:
            return result

    # Strategy 2: Agentic VLM — Single-Shot Architecture
    # Re-enabled: Now downsamples images and passes both in a single network call.
    vlm_result = await _agentic_change_analysis(
        target, image_a_base64, image_b_base64,
        metadata_a, metadata_b, provider,
    )

    # Strategy 3: Classical CV pixel differencing (always run for spatial evidence)
    cv_result = _classical_change_detection(image_a_base64, image_b_base64, target)

    # Merge: VLM provides semantic description, CV provides spatial evidence
    if vlm_result and not vlm_result.get("fallback", True):
        # VLM succeeded — use its description, supplement with CV spatial evidence
        if cv_result.get("regions"):
            vlm_result["regions"] = cv_result["regions"]
            vlm_result["contours"] = cv_result["contours"]
            vlm_result["coverage_pct"] = cv_result.get("coverage_pct", "")
        else:
            # SAFETY GUARD: VLM produced text but CV found zero spatial evidence
            original_summary = vlm_result.get("summary", "")
            vlm_result["summary"] = f"VLM inferred changes: '{original_summary}'. HOWEVER, spatial verification found zero pixel-level evidence. This is likely an AI hallucination."
            vlm_result["fallback"] = True
            vlm_result["status"] = "warning"
        return vlm_result

    # VLM unavailable — use classical CV with honest limitations
    return cv_result


def _downsample_base64(b64: str, max_dim: int = 800) -> str:
    """Downsample base64 image to speed up VLM network upload."""
    import base64
    if not OPENCV_AVAILABLE:
        return b64
    try:
        clean = b64.split(",")[1] if "," in b64 else b64
        arr = np.frombuffer(base64.b64decode(clean), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None: return b64
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')
    except Exception:
        return b64


async def _agentic_change_analysis(
    target: str,
    image_a_base64: str,
    image_b_base64: str,
    metadata_a: Dict[str, Any],
    metadata_b: Dict[str, Any],
    provider: RSVLMProvider,
) -> Optional[Dict[str, Any]]:
    """
    Lightning-fast Single-Shot VLM strategy.
    Downsamples the images to tiny payloads, then passes BOTH simultaneously to Gemini.
    """
    caps = provider.get_capabilities()
    if caps.get("is_fallback"):
        return None

    try:
        small_a = _downsample_base64(image_a_base64)
        small_b = _downsample_base64(image_b_base64)

        img_a = ImageInput(data=small_a, role="primary", metadata=metadata_a)
        img_b = ImageInput(data=small_b, role="primary", metadata=metadata_b)

        query = (
            f"You are viewing two satellite images of the exact same area at different times. "
            f"The first image is BEFORE. The second image is AFTER. "
            f"Analyze what changed regarding '{target}'. "
            f"Be highly specific about the visual changes you see (e.g. 'A new road was built', 'Forest was cleared'). "
            f"Strictly limit your response to a concise, professional paragraph of 4 to 7 lines."
        )

        resp = await provider.analyze(
            images=[img_a, img_b],
            query=query,
            task=TaskType.VQA,
        )

        if resp.success:
            return {
                "status": "success",
                "summary": resp.text,
                "regions": [],
                "contours": [],
                "fallback": False,
                "model": f"Single-Shot VLM via {provider.name}",
            }

    except Exception as exc:
        logger.warning(f"Single-Shot VLM change analysis failed: {exc}")

    return None


def _classical_change_detection(
    image_a_base64: str, image_b_base64: str, target: str,
) -> Dict[str, Any]:
    """
    Classical pixel-differencing change detection with morphological filtering.

    Honest about limitations: reports spatial changes but cannot interpret
    what changed semantically.
    """
    if not OPENCV_AVAILABLE:
        return {
            "status": "error",
            "summary": "OpenCV not available for change detection.",
            "regions": [], "contours": [], "fallback": True,
            "model": "None",
        }

    imgA = _decode(image_a_base64)
    imgB = _decode(image_b_base64)

    if imgA is None or imgB is None:
        return {
            "status": "error",
            "summary": "Failed to decode one or both images for change detection.",
            "regions": [], "contours": [], "fallback": True,
            "model": "None",
        }

    # Resize B to match A if needed
    hA, wA = imgA.shape[:2]
    if imgB.shape[:2] != (hA, wA):
        imgB = cv2.resize(imgB, (wA, hA), interpolation=cv2.INTER_LINEAR)

    # Multi-channel absolute difference
    diff = cv2.absdiff(
        cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY),
    )

    # Adaptive thresholding for better noise rejection
    # Raise threshold slightly (22) to avoid seasonal/shadow noise
    _, thresh = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)
    
    # 1. Stronger MORPH_OPEN to remove scattered seasonal/vegetation noise
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small)
    
    # 2. Moderate MORPH_CLOSE to group adjacent changes accurately,
    # ensuring boundaries hug the actual construction sites instead of swallowing everything.
    kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_med)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_area = hA * wA
    regions, valid_contours = [], []
    total_change_px = 0

    # Increase limit from 10 to 50 to capture all changes
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:50]:
        area = cv2.contourArea(c)
        if area > max(80, int(image_area * 0.001)):
            total_change_px += area
            x, y, bw, bh = cv2.boundingRect(c)
            
            # Approximate polygon for precise GIS-style masking
            epsilon = 0.005 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            poly_pct = []
            for pt in approx:
                px, py = pt[0]
                poly_pct.append([round((px / wA) * 100, 2), round((py / hA) * 100, 2)])

            regions.append({
                "box": [
                    round((y / hA) * 100, 1),
                    round((x / wA) * 100, 1),
                    round(((y + bh) / hA) * 100, 1),
                    round(((x + bw) / wA) * 100, 1),
                ],
                "polygon": poly_pct,
                "label": "Bi-Temporal Change Zone",
                "actual_pct": (area / image_area) * 100.0,
                "area_px": area,
            })
            valid_contours.append(c.tolist())

    pct_changed = (total_change_px / image_area) * 100.0

    summary = (
        f"Classical pixel-differencing detected surface changes across "
        f"{pct_changed:.2f}% of the scene ({len(regions)} distinct change zones). "
        f"Semantic interpretation of what changed regarding '{target}' requires "
        f"a remote-sensing change specialist model. "
        f"Configure CHANGE_MODEL_ENDPOINT or RS_VLM_PROVIDER to enable."
    )

    return {
        "status": "success",
        "summary": summary,
        "regions": regions,
        "contours": valid_contours,
        "coverage_pct": f"{pct_changed:.2f}%",
        "fallback": True,
        "model": "Classical Bi-Temporal Image Differencing (Fallback)",
    }


def _try_external_change_model(
    img_a_b64: str, img_b_b64: str, target: str,
    url: str, api_key: str,
) -> Optional[Dict[str, Any]]:
    """Try dedicated external change detection model."""
    import requests

    ca = img_a_b64.split(",")[1] if "," in img_a_b64 else img_a_b64
    cb = img_b_b64.split(",")[1] if "," in img_b_b64 else img_b_b64
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = requests.post(
            url,
            json={"image_a": ca, "image_b": cb, "target": target},
            headers=headers,
            timeout=50,
        )
        if resp.status_code == 200:
            data = resp.json()
            if "summary" in data:
                return {
                    "status": "success",
                    "summary": data["summary"],
                    "regions": data.get("regions", []),
                    "contours": data.get("contours", []),
                    "fallback": False,
                    "model": "Change Detection Specialist",
                }
    except Exception as exc:
        logger.warning(f"External change model failed: {exc}")

    return None


def _decode(b64: str) -> Optional[np.ndarray]:
    """Decode base64 to OpenCV image."""
    if not OPENCV_AVAILABLE or not b64:
        return None
    try:
        clean = b64.split(",")[1] if "," in b64 else b64
        arr = np.frombuffer(base64.b64decode(clean), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None

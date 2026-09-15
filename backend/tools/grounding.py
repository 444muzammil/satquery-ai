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


def _grabcut_polygon(img: np.ndarray, box_pct: List[float]) -> Optional[List[List[float]]]:
    """Use OpenCV GrabCut to extract precise polygon from a bounding box."""
    h, w = img.shape[:2]
    ymin, xmin, ymax, xmax = box_pct
    x1 = max(0, int(xmin / 100 * w))
    y1 = max(0, int(ymin / 100 * h))
    x2 = min(w, int(xmax / 100 * w))
    y2 = min(h, int(ymax / 100 * h))
    
    bw = x2 - x1
    bh = y2 - y1
    if bw < 5 or bh < 5:
        return None
        
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    rect = (x1, y1, bw, bh)
    
    try:
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0), 0, 1).astype("uint8")
        
        contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            epsilon = 0.005 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            
            poly_pct = []
            for pt in approx:
                px, py = pt[0]
                poly_pct.append([round((px / w) * 100, 2), round((py / h) * 100, 2)])
                
            if len(poly_pct) >= 3:
                return poly_pct
    except Exception:
        pass
    return None


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

    Tries: external grounding model → Dense Mapping Interceptor → RS VLM → CV fallback.
    """
    # Try dedicated grounding model first
    if grounding_api_url:
        result = _try_external_grounding(image_base64, target, grounding_api_url, grounding_api_key)
        if result:
            return result

    # Dense Mapping Interceptor
    # For widespread land-cover categories, VLMs hit token limits (drawing ~10 boxes max).
    # We intercept these queries and map them using Dense CV Segmentation to cover the whole city/forest.
    t_low = target.lower()
    dense_keywords = ["water", "vegetation", "forest", "urban", "building", "buildings", "road", "barren"]
    is_dense = any(kw in t_low for kw in dense_keywords)

    if is_dense and OPENCV_AVAILABLE:
        img = _decode_base64_to_cv2(image_base64)
        if img is not None:
            # We request up to 50 regions and lower the area threshold for tiny buildings
            # (Limiting to 50 prevents SVG rendering glitches in the frontend where borders appear/disappear)
            regions, contours = classical_grounding(img, target, max_regions=50, min_area_pct=0.0002)
            if len(regions) > 5:
                return {
                    "status": "success",
                    "summary": f"Dense Semantic Mapping intercept triggered. Mapped {len(regions)} zones for widespread '{target}'.",
                    "regions": regions,
                    "contours": [],
                    "fallback": False,
                    "model": "Dense CV Segmentation",
                }

    # Try RS VLM for specific text-based bbox extraction (e.g. "Find the airplane")
    image = ImageInput(data=image_base64, role="primary", metadata=metadata)
    response = await provider.analyze(
        images=[image],
        query=f"Locate and provide bounding box coordinates for: {target}",
        task=TaskType.GROUNDING,
    )

    if response.success and response.regions:
        if OPENCV_AVAILABLE:
            img = _decode_base64_to_cv2(image_base64)
            if img is not None:
                for r in response.regions:
                    if "polygon" not in r:
                        poly = _grabcut_polygon(img, r["box"])
                        if poly:
                            r["polygon"] = poly
                        else:
                            ymin, xmin, ymax, xmax = r["box"]
                            r["polygon"] = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
                    r["label"] = target.replace("mark ", "").title()

        return {
            "status": "success",
            "summary": f"RS VLM identified spatial boundaries for '{target}'. GrabCut semantic segmentation contoured the boundaries.",
            "regions": response.regions,
            "contours": response.contours,
            "fallback": response.is_fallback,
            "model": f"{response.model_name} + Agentic GrabCut",
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
    img: np.ndarray, target: str, max_regions: int = 30, min_area_pct: float = 0.0008
) -> Tuple[List[Dict[str, Any]], List[list]]:
    """
    Classical CV grounding via HSV color-space segmentation.
    """
    h, w = img.shape[:2]
    image_area = h * w
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    t_low = target.lower()

    if any(kw in t_low for kw in ["water", "river", "lake", "canal", "flood", "ocean", "pond", "coast"]):
        # Turbid and clear water. Avoids pure gray clouds/concrete (Sat > 15) and avoids bright clouds (Val < 230)
        mask = cv2.inRange(hsv, np.array([45, 15, 10]), np.array([145, 255, 230]))
    elif any(kw in t_low for kw in ["vegetation", "forest", "crop", "tree", "agricultural", "farm", "green"]):
        # Healthy green vegetation. Avoids yellowish barren land (Hue > 35).
        mask = cv2.inRange(hsv, np.array([35, 30, 20]), np.array([85, 255, 255]))
    elif any(kw in t_low for kw in ["build", "urban", "road", "railway", "structure", "development", "industrial", "buildings"]):
        # Concrete/Urban areas. Low saturation (gray), but capped brightness to avoid bright white clouds.
        mask = cv2.inRange(hsv, np.array([0, 0, 40]), np.array([180, 45, 215]))
    elif any(kw in t_low for kw in ["barren", "soil", "sand", "dirt", "desert", "runway", "airport"]):
        # Brown, tan, and orange soil.
        mask = cv2.inRange(hsv, np.array([10, 20, 50]), np.array([35, 150, 240]))
    else:
        edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 70, 170)
        mask = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # 1. Remove tiny noise (salt and pepper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    
    # 2. Merge nearby structures/pixels into larger "zones" or "blocks"
    # Using 13x13 prevents polygons from aggressively cutting across natural curves (like rivers)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions, valid_contours = [], []
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:max_regions]:
        area = cv2.contourArea(c)
        if area > max(40, int(image_area * min_area_pct)):
            x, y, bw, bh = cv2.boundingRect(c)
            
            # Tighter epsilon (0.003) hugs natural curves like rivers perfectly without creating huge jagged polygons
            epsilon = 0.003 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            poly_pct = []
            for pt in approx:
                px, py = pt[0]
                poly_pct.append([round((px / w) * 100, 2), round((py / h) * 100, 2)])

            regions.append({
                "box": [
                    round((y / h) * 100, 1),
                    round((x / w) * 100, 1),
                    round(((y + bh) / h) * 100, 1),
                    round(((x + bw) / w) * 100, 1),
                ],
                "polygon": poly_pct,
                "label": target.title(),
                "actual_pct": (area / image_area) * 100.0,
                "area_px": area,
            })
            valid_contours.append(c.tolist())

    return regions, valid_contours


def _try_external_grounding(
    image_base64: str, target: str, url: str, api_key: str
) -> Optional[Dict[str, Any]]:
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
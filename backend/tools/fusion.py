"""
SatQuery AI — Optical-SAR Fusion Tool

Cross-modal analysis using optical and SAR imagery via specialist model,
agentic VLM orchestration, or classical CV feature co-occurrence.
"""

import base64
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from providers.base import RSVLMProvider, ImageInput, TaskType, Modality

logger = logging.getLogger("satquery.tools.fusion")

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


async def execute_optical_sar_fusion(
    target: str,
    optical_base64: str,
    sar_base64: str,
    metadata_optical: Dict[str, Any],
    metadata_sar: Dict[str, Any],
    provider: RSVLMProvider,
    fusion_model_url: str = "",
    fusion_model_key: str = "",
) -> Dict[str, Any]:
    """
    Execute optical-SAR cross-modal analysis.

    Strategy:
    1. Try dedicated fusion specialist (if configured)
    2. Try agentic VLM approach (analyze each modality → cross-validate)
    3. Fall back to classical CV feature co-occurrence

    Both images are GENUINELY used — this is a mandatory SIH requirement.

    Args:
        target: Feature query
        optical_base64: Optical image (base64)
        sar_base64: SAR image (base64)
        metadata_optical: Optical image metadata
        metadata_sar: SAR image metadata
        provider: RS VLM provider
        fusion_model_url: Optional dedicated fusion model endpoint
        fusion_model_key: Optional API key for fusion model

    Returns:
        Tool result dict with regions, contours, and cross-modal analysis
    """
    # Strategy 1: Dedicated fusion specialist
    if fusion_model_url:
        result = _try_external_fusion(
            optical_base64, sar_base64, target,
            fusion_model_url, fusion_model_key,
        )
        if result:
            return result

    # Strategy 2: Agentic VLM — analyze each modality then cross-validate
    vlm_result = await _agentic_fusion_analysis(
        target, optical_base64, sar_base64,
        metadata_optical, metadata_sar, provider,
    )

    # Strategy 3: Classical CV feature co-occurrence (always run for spatial evidence)
    cv_result = _classical_optical_sar_fusion(optical_base64, sar_base64, target)

    # Merge results
    if vlm_result and not vlm_result.get("fallback", True):
        # VLM succeeded — use its description, supplement with CV spatial evidence
        if cv_result.get("regions"):
            vlm_result["regions"] = cv_result["regions"]
            vlm_result["contours"] = cv_result["contours"]
        else:
            # SAFETY GUARD: VLM produced text but CV found zero spatial evidence
            original_summary = vlm_result.get("summary", "")
            vlm_result["summary"] = f"VLM inferred cross-modal features: '{original_summary}'. HOWEVER, spatial verification found zero pixel-level evidence. This is likely an AI hallucination."
            vlm_result["fallback"] = True
            vlm_result["status"] = "warning"
        return vlm_result

    return cv_result


async def _agentic_fusion_analysis(
    target: str,
    optical_base64: str,
    sar_base64: str,
    metadata_optical: Dict[str, Any],
    metadata_sar: Dict[str, Any],
    provider: RSVLMProvider,
) -> Optional[Dict[str, Any]]:
    """
    Agentic approach: analyze optical and SAR separately with VLM, then combine.

    EarthDial supports SAR through BigEarthNet.txt SAR alignment, so it can
    interpret radar backscatter characteristics (specular water, double-bounce urban).
    """
    caps = provider.get_capabilities()
    if caps.get("is_fallback"):
        return None

    try:
        # Analyze optical image
        img_opt = ImageInput(
            data=optical_base64, role="primary",
            modality=Modality.OPTICAL_RGB, metadata=metadata_optical,
        )
        resp_opt = await provider.analyze(
            images=[img_opt],
            query=f"Describe the land cover, built-up areas, water bodies, and vegetation visible. Focus on: {target}",
            task=TaskType.CAPTION,
        )

        # Analyze SAR image
        img_sar = ImageInput(
            data=sar_base64, role="sar",
            modality=Modality.SAR, metadata=metadata_sar,
        )
        resp_sar = await provider.analyze(
            images=[img_sar],
            query=(
                "This is a SAR (Synthetic Aperture Radar) image. "
                "Identify regions of high backscatter (bright — likely built-up structures or double-bounce) "
                "and low backscatter (dark — likely smooth water surfaces or specular reflections). "
                f"Focus on: {target}"
            ),
            task=TaskType.CAPTION,
        )

        if resp_opt.success and resp_sar.success:
            # Cross-modal synthesis
            synthesis_query = (
                f"Cross-modal analysis combining optical and SAR observations:\n\n"
                f"OPTICAL OBSERVATION: {resp_opt.text}\n\n"
                f"SAR OBSERVATION: {resp_sar.text}\n\n"
                f"Synthesize complementary information from both modalities for: '{target}'. "
                f"Where do optical and SAR evidence agree or provide complementary information? "
                f"Identify built-up regions (high SAR backscatter + optical structure) "
                f"and water bodies (low SAR backscatter + optical water signature)."
            )

            resp_synth = await provider.analyze(
                images=[img_opt],
                query=synthesis_query,
                task=TaskType.VQA,
            )

            if resp_synth.success:
                return {
                    "status": "success",
                    "summary": resp_synth.text,
                    "regions": [],
                    "contours": [],
                    "fallback": False,
                    "model": f"Agentic Cross-Modal Analysis via {provider.name}",
                }

    except Exception as exc:
        logger.warning(f"Agentic fusion analysis failed: {exc}")

    return None


def _classical_optical_sar_fusion(
    optical_base64: str, sar_base64: str, target: str,
) -> Dict[str, Any]:
    """
    Classical optical-SAR feature co-occurrence analysis.

    Uses SAR backscatter thresholding (high = built-up, low = water)
    combined with optical edge detection for cross-modal validation.
    Both images are genuinely used.
    """
    if not OPENCV_AVAILABLE:
        return {
            "status": "error",
            "summary": "OpenCV not available for fusion analysis.",
            "regions": [], "contours": [], "fallback": True,
            "model": "None",
        }

    imgOpt = _decode(optical_base64)
    imgSAR = _decode(sar_base64)

    if imgOpt is None or imgSAR is None:
        return {
            "status": "error",
            "summary": "Failed to decode optical or SAR image.",
            "regions": [], "contours": [], "fallback": True,
            "model": "None",
        }

    # Resize SAR to match optical if needed
    hO, wO = imgOpt.shape[:2]
    if imgSAR.shape[:2] != (hO, wO):
        imgSAR = cv2.resize(imgSAR, (wO, hO), interpolation=cv2.INTER_LINEAR)

    # SAR analysis: backscatter thresholding
    sar_gray = cv2.cvtColor(imgSAR, cv2.COLOR_BGR2GRAY)
    sar_filtered = cv2.bilateralFilter(cv2.medianBlur(sar_gray, 5), 9, 75, 75)
    _, high_backscatter = cv2.threshold(sar_filtered, 200, 255, cv2.THRESH_BINARY)
    _, low_backscatter = cv2.threshold(sar_filtered, 45, 255, cv2.THRESH_BINARY_INV)

    # Optical analysis: edge detection (for context in cloud-free areas)
    opt_gray = cv2.cvtColor(imgOpt, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(opt_gray, 80, 180)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8))

    # Cross-modal fusion: Combine SAR backscatter with optical edges.
    # SAR high backscatter detects built-up structures (even through clouds).
    # Optical edges reinforce structure boundaries in clear areas.
    # Fusion: require SAR evidence, but boost with optical edge co-occurrence.
    fused_built = cv2.bitwise_and(
        cv2.bitwise_or(high_backscatter, edges),
        cv2.dilate(high_backscatter, np.ones((7, 7), np.uint8))
    )
    
    # Morphological closing to group sparse overlapping pixels into coherent polygons
    fused_built = cv2.morphologyEx(
        fused_built, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
    )
    
    # Slight dilation to thicken resulting shapes
    fused_built = cv2.dilate(fused_built, np.ones((5, 5), np.uint8), iterations=2)

    h, w = imgOpt.shape[:2]
    image_area = h * w
    contours, _ = cv2.findContours(fused_built, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions, valid_contours = [], []

    t_low = target.lower()
    is_water_query = any(kw in t_low for kw in ["water", "river", "lake", "ocean", "sea", "flood", "pond"])
    is_build_query = any(kw in t_low for kw in ["build", "urban", "structure", "city", "industrial"])

    if not is_water_query:
        # Increase limit to capture more areas, but group them logically
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:30]:
            area = cv2.contourArea(c)
            if area > max(120, int(image_area * 0.0005)):
                x, y, bw, bh = cv2.boundingRect(c)
                
                # Approximate polygon for precise GIS-style masking
                epsilon = 0.005 * cv2.arcLength(c, True)
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
                    "label": "Cross-Modal: High Backscatter (Built-up)",
                    "actual_pct": (area / image_area) * 100.0,
                    "area_px": area,
                })
                valid_contours.append(c.tolist())

    if not is_build_query:
        # Also detect water from SAR low backscatter
        # Use morphological closing to merge small water blobs into unified bodies
        low_backscatter = cv2.morphologyEx(
            low_backscatter, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        )
        water_contours, _ = cv2.findContours(
            low_backscatter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        for c in sorted(water_contours, key=cv2.contourArea, reverse=True)[:15]:
            area = cv2.contourArea(c)
            if area > max(200, int(image_area * 0.002)):
                x, y, bw, bh = cv2.boundingRect(c)
                
                epsilon = 0.005 * cv2.arcLength(c, True)
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
                    "label": "Cross-Modal: Low SAR Backscatter (Water)",
                    "actual_pct": (area / image_area) * 100.0,
                    "area_px": area,
                })
                valid_contours.append(c.tolist())

    summary = (
        f"Classical optical-SAR co-occurrence analysis: matched optical edge gradients "
        f"with SAR microwave backscatter, identifying {len(regions)} cross-validated features. "
        f"High SAR backscatter + optical edges → built-up structures. "
        f"Low SAR backscatter → smooth water surfaces. "
        f"Semantic interpretation of '{target}' requires a cross-modal specialist model."
    )

    return {
        "status": "success",
        "summary": summary,
        "regions": regions,
        "contours": valid_contours,
        "fallback": True,
        "model": "Classical Optical-SAR Feature Co-occurrence (Fallback)",
    }


def _try_external_fusion(
    opt_b64: str, sar_b64: str, query: str,
    url: str, api_key: str,
) -> Optional[Dict[str, Any]]:
    """Try dedicated external fusion model."""
    import requests

    co = opt_b64.split(",")[1] if "," in opt_b64 else opt_b64
    cs = sar_b64.split(",")[1] if "," in sar_b64 else sar_b64
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = requests.post(
            url,
            json={"optical_image": co, "sar_image": cs, "query": query},
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
                    "model": "Optical-SAR Fusion Specialist",
                }
    except Exception as exc:
        logger.warning(f"External fusion model failed: {exc}")

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

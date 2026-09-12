"""
SatQuery AI — Calibrated Confidence Estimation

Evidence-based confidence scoring that reflects actual output quality.
Never starts at 98% — bases score on real evidence factors.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger("satquery.evidence.confidence")


def estimate_confidence(
    tools_executed: List[str],
    models_used: List[str],
    has_fallback: bool,
    has_vlm_response: bool,
    regions_found: int,
    input_quality: Dict[str, Any],
    penalties: Dict[str, bool],
) -> float:
    """
    Calculate calibrated confidence score based on evidence quality.

    Range: 10% (no model, no evidence) to 90% (VLM + verified spatial evidence).
    Never 95%+ to reflect inherent uncertainty in RS interpretation.

    Args:
        tools_executed: List of tool names that ran
        models_used: List of model names that produced results
        has_fallback: Whether any tool used a fallback
        has_vlm_response: Whether a real VLM (not classical CV) produced the answer
        regions_found: Number of spatial regions identified
        input_quality: Dict with 'is_geotiff', 'has_georeference', etc.
        penalties: Dict of penalty flags

    Returns:
        Confidence score as float (10.0 to 90.0)
    """
    # Base score depends on model type
    if has_vlm_response and not has_fallback:
        base = 72.0   # Real VLM response
    elif has_vlm_response and has_fallback:
        base = 55.0   # VLM partially + fallback
    elif has_fallback:
        base = 38.0   # Pure classical CV fallback
    else:
        base = 15.0   # No model produced output

    # Input quality adjustments
    if input_quality.get("is_geotiff"):
        base += 5.0
    if input_quality.get("has_georeference"):
        base += 3.0

    # Spatial evidence quality
    if regions_found > 0:
        base += min(8.0, regions_found * 2.0)
    elif any(t in ["RS_GROUNDING", "CHANGE_DETECTION", "OPTICAL_SAR_FUSION"]
             for t in tools_executed):
        # Expected spatial evidence but got none
        base -= 8.0

    # Penalties
    if penalties.get("resampling"):
        base -= 5.0    # Spatial alignment uncertainty
    if penalties.get("policy_rejected"):
        base -= 10.0   # Some tools were rejected
    if penalties.get("vlm_failed"):
        base -= 12.0   # Primary VLM failed, used fallback

    # Clamp to valid range
    return max(10.0, min(90.0, round(base, 1)))

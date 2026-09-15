"""
SatQuery AI — GIS Area Calculator Tool

Deterministic area calculation using GeoTIFF affine transform and resolution metadata.
Extracted from original main.py (unchanged logic — this tool works correctly).
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("satquery.tools.gis")


def calculate_area(
    regions: List[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Calculate spatial footprint from accumulated regions and GeoTIFF metadata.

    Uses GeoTIFF pixel resolution for real-world area conversion.
    This is a deterministic tool — no ML model involved.

    Args:
        regions: List of region dicts with 'actual_pct' and 'area_px' keys
        metadata: Image metadata with 'has_georeference', 'resolution' keys

    Returns:
        Dict with status, summary, coverage_pct, real_world_area
    """
    if not regions:
        return {
            "status": "success",
            "summary": "No localized spatial footprints to measure.",
            "coverage_pct": "0.0%",
            "fallback": False,
            "model": "Deterministic Rasterio/Affine GIS Engine",
        }

    total_pct = min(100.0, sum(r.get("actual_pct", 0.0) for r in regions))

    # Real-world metric calculation via GeoTIFF metadata
    real_world_str = ""
    res = metadata.get("resolution", [])
    if metadata.get("has_georeference") and res and len(res) >= 2:
        res_x, res_y = abs(res[0]), abs(res[1])
        total_px = sum(r.get("area_px", 0) for r in regions)
        
        # Check if CRS is geographic (degrees)
        crs = str(metadata.get("crs", "")).upper()
        if "4326" in crs or res_x < 0.1:
            # Approximate conversion: 1 degree ~ 111,320 meters at equator
            meter_per_deg = 111320.0
            sq_meters = total_px * (res_x * meter_per_deg) * (res_y * meter_per_deg)
        else:
            sq_meters = total_px * (res_x * res_y)
            
        if sq_meters >= 1_000_000:
            real_world_str = f" ({sq_meters / 1_000_000:.2f} km²)"
        else:
            real_world_str = f" ({sq_meters / 10_000:.2f} hectares)"

    summary = f"Spatial footprint computed at {total_pct:.2f}% of AOI extent{real_world_str}."
    return {
        "status": "success",
        "summary": summary,
        "coverage_pct": f"{total_pct:.2f}%",
        "real_world_area": real_world_str.strip(" ()"),
        "fallback": False,
        "model": "Deterministic Rasterio/Affine GIS Engine",
    }

"""
SatQuery AI — Spatial Comparator Tool

Deterministic quadrant-density analysis for comparative spatial queries.
Extracted from original main.py (unchanged logic — this tool works correctly).
"""

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("satquery.tools.spatial")


def compare_spatial_density(
    target: str,
    regions: List[Dict[str, Any]],
    img_shape: Tuple[int, int],
) -> Dict[str, Any]:
    """
    Calculate spatial quadrant density distributions to resolve comparative questions.
    (e.g., 'Spot the area where there is more structural development')

    This is a deterministic tool — no ML model involved.

    Args:
        target: The feature being compared
        regions: List of region dicts with 'box' and 'area_px' keys
        img_shape: (height, width) of the image

    Returns:
        Dict with status, summary, regions, contours
    """
    if not regions:
        return {
            "status": "success",
            "summary": (
                f"No localized features available to compute "
                f"spatial distribution for '{target}'."
            ),
            "regions": [],
            "contours": [],
            "fallback": False,
            "model": "Deterministic Spatial Density Comparator",
        }

    h, w = img_shape[:2]
    quadrants = {
        "Northeastern": {"box": [0, 50, 50, 100], "score": 0.0, "boxes": []},
        "Northwestern": {"box": [0, 0, 50, 50], "score": 0.0, "boxes": []},
        "Southeastern": {"box": [50, 50, 100, 100], "score": 0.0, "boxes": []},
        "Southwestern": {"box": [50, 0, 100, 50], "score": 0.0, "boxes": []},
        "Central": {"box": [25, 25, 75, 75], "score": 0.0, "boxes": []},
    }

    total_area = sum(r.get("area_px", 1.0) for r in regions)
    for r in regions:
        ymin, xmin, ymax, xmax = r["box"]
        cy, cx = (ymin + ymax) / 2.0, (xmin + xmax) / 2.0
        weight = r.get("area_px", 1.0)

        if 25 <= cy <= 75 and 25 <= cx <= 75:
            quadrants["Central"]["score"] += weight
            quadrants["Central"]["boxes"].append(r)
        if cy <= 50 and cx >= 50:
            quadrants["Northeastern"]["score"] += weight
            quadrants["Northeastern"]["boxes"].append(r)
        elif cy <= 50 and cx < 50:
            quadrants["Northwestern"]["score"] += weight
            quadrants["Northwestern"]["boxes"].append(r)
        elif cy > 50 and cx >= 50:
            quadrants["Southeastern"]["score"] += weight
            quadrants["Southeastern"]["boxes"].append(r)
        else:
            quadrants["Southwestern"]["score"] += weight
            quadrants["Southwestern"]["boxes"].append(r)

    top_quad = max(quadrants.items(), key=lambda q: q[1]["score"])
    quad_name = top_quad[0]
    quad_pct = (top_quad[1]["score"] / total_area) * 100.0 if total_area > 0 else 0.0

    if top_quad[1]["boxes"]:
        min_y = min(b["box"][0] for b in top_quad[1]["boxes"])
        min_x = min(b["box"][1] for b in top_quad[1]["boxes"])
        max_y = max(b["box"][2] for b in top_quad[1]["boxes"])
        max_x = max(b["box"][3] for b in top_quad[1]["boxes"])
        focus_box = [min_y, min_x, max_y, max_x]
    else:
        focus_box = top_quad[1]["box"]

    rep_region = [{
        "box": focus_box,
        "label": f"Peak Concentration: {quad_name} ({target.title()})",
        "actual_pct": round(quad_pct, 2),
    }]

    abs_y = int((focus_box[0] / 100.0) * h)
    abs_x = int((focus_box[1] / 100.0) * w)
    abs_ymax = int((focus_box[2] / 100.0) * h)
    abs_xmax = int((focus_box[3] / 100.0) * w)
    box_contour = [
        [[abs_x, abs_y]],
        [[abs_xmax, abs_y]],
        [[abs_xmax, abs_ymax]],
        [[abs_x, abs_ymax]],
    ]

    summary = (
        f"Spatial density analysis confirmed that the {quad_name} sector contains the "
        f"highest concentration of '{target}' ({quad_pct:.1f}% of identified features)."
    )

    return {
        "status": "success",
        "summary": summary,
        "regions": rep_region,
        "contours": [box_contour],
        "fallback": False,
        "model": "Deterministic Spatial Density Comparator",
    }

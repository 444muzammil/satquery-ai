"""
SatQuery AI — GeoJSON / GIS Export

Converts contour data to GeoJSON features using GeoTIFF affine transforms.
Extracted from main.py (logic unchanged).
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("satquery.evidence.export")

try:
    import geopandas as gpd
    from shapely.geometry import Polygon
    GIS_EXPORT_AVAILABLE = True
except ImportError:
    GIS_EXPORT_AVAILABLE = False


def generate_gis_export(
    contours: List[Any],
    metadata: Dict[str, Any],
    task_label: str,
) -> Optional[Dict[str, Any]]:
    """
    Convert contour data to GeoJSON using GeoTIFF affine transform.

    Args:
        contours: List of contour point arrays
        metadata: Image metadata with 'transform' and 'crs'
        task_label: Task label for feature properties

    Returns:
        GeoJSON dict or None if export not possible
    """
    if not GIS_EXPORT_AVAILABLE or not contours:
        return None

    try:
        transform = metadata.get("transform")
        crs = metadata.get("crs", "EPSG:4326")
        if not transform:
            transform = [
                1.0, 0.0, 0.0,
                0.0, -1.0, float(metadata.get("height", 512)),
            ]

        a, b, c, d, e, f = transform
        polygons = []
        for contour in contours:
            if len(contour) < 3:
                continue
            coords = []
            for pt in contour:
                x, y = pt[0]
                coords.append((
                    (x * a) + (y * b) + c,
                    (x * d) + (y * e) + f,
                ))
            coords.append(coords[0])  # Close polygon
            polygons.append(Polygon(coords))

        if polygons:
            gdf = gpd.GeoDataFrame(
                {"feature": [task_label] * len(polygons)},
                geometry=polygons,
                crs=crs,
            )
            if crs and "4326" not in str(crs):
                try:
                    gdf = gdf.to_crs("EPSG:4326")
                except Exception:
                    pass
            return json.loads(gdf.to_json())
    except Exception as exc:
        logger.warning(f"GIS export failed: {exc}")

    return None

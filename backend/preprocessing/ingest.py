"""
SatQuery AI — Image Ingestion & Normalization

Upload handling, GeoTIFF parsing, radiometric normalization.
Extracted from main.py upload endpoint.
"""

import io
import base64
import logging
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image
import rasterio

logger = logging.getLogger("satquery.preprocessing")


def process_upload(content: bytes, filename: str) -> Dict[str, Any]:
    """
    Process an uploaded image file: extract metadata and generate preview.

    Handles GeoTIFF/TIFF with rasterio (CRS, bounds, transform, resolution),
    and PNG/JPEG with PIL.

    Args:
        content: Raw file bytes
        filename: Original filename

    Returns:
        Dict with 'metadata' and 'preview' keys

    Raises:
        ValueError: If file cannot be parsed
    """
    is_tiff = filename.lower().endswith((".tif", ".tiff", ".geotiff"))
    metadata: Dict[str, Any] = {
        "filename": filename,
        "format": "GeoTIFF/TIFF" if is_tiff else "PNG/JPEG",
        "has_georeference": False,
    }
    preview_base64 = ""

    if is_tiff:
        metadata, preview_base64 = _process_geotiff(content, metadata)
    else:
        metadata, preview_base64 = _process_standard_image(content, metadata)

    return {"metadata": metadata, "preview": preview_base64}


def _process_geotiff(
    content: bytes, metadata: Dict[str, Any]
) -> Tuple[Dict[str, Any], str]:
    """Process GeoTIFF/TIFF file with rasterio."""
    with rasterio.MemoryFile(content) as memfile:
        with memfile.open() as dataset:
            metadata.update({
                "width": dataset.width,
                "height": dataset.height,
                "bands": dataset.count,
                "crs": str(dataset.crs) if dataset.crs else "EPSG:4326",
                "bounds": [
                    dataset.bounds.left, dataset.bounds.bottom,
                    dataset.bounds.right, dataset.bounds.top,
                ],
                "resolution": list(dataset.res) if dataset.res else [1.0, 1.0],
                "has_georeference": bool(dataset.crs and dataset.transform),
                "dtype": str(dataset.dtypes[0]) if dataset.dtypes else "unknown",
            })

            if dataset.transform:
                metadata["transform"] = [
                    dataset.transform.a, dataset.transform.b, dataset.transform.c,
                    dataset.transform.d, dataset.transform.e, dataset.transform.f,
                ]

            # Compute dynamic range stats for modality detection
            band1 = dataset.read(1).astype(np.float32)
            p2, p98 = np.percentile(band1, (2, 98))
            metadata["dynamic_range_ratio"] = float(p98 / p2) if p2 > 0 else 0.0

            # Generate preview with 2-98% percentile stretch
            if dataset.count in [1, 2]:
                raw_data = dataset.read(1)
            else:
                raw_data = np.moveaxis(dataset.read([1, 2, 3]), 0, -1)

            raw_float = raw_data.astype(np.float32)
            p2_all, p98_all = np.percentile(raw_float, (2, 98))
            if p98_all > p2_all:
                img_array = np.clip(
                    (raw_float - p2_all) / (p98_all - p2_all) * 255.0, 0, 255
                ).astype(np.uint8)
            else:
                img_array = np.zeros_like(raw_data, dtype=np.uint8)

            img = Image.fromarray(img_array)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            preview_base64 = (
                f"data:image/png;base64,"
                f"{base64.b64encode(buffered.getvalue()).decode()}"
            )

    return metadata, preview_base64


def _process_standard_image(
    content: bytes, metadata: Dict[str, Any]
) -> Tuple[Dict[str, Any], str]:
    """Process standard image (PNG/JPEG) with PIL."""
    img = Image.open(io.BytesIO(content))
    fmt = img.format or "PNG"
    metadata.update({
        "width": img.width,
        "height": img.height,
        "bands": len(img.getbands()),
    })
    preview_base64 = (
        f"data:image/{fmt.lower()};base64,"
        f"{base64.b64encode(content).decode()}"
    )
    return metadata, preview_base64

import io
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from providers.base import Modality

try:
    import rasterio
except ImportError:
    rasterio = None

try:
    from PIL import Image
except ImportError:
    Image = None

logger = logging.getLogger(__name__)

MAX_IMAGE_SIZE_MB = 100.0

@dataclass
class ValidationResult:
    valid: bool
    error: str = ''
    warnings: List[str] = field(default_factory=list)
    detected_modality: Modality = Modality.UNKNOWN
    detected_format: str = ''

@dataclass 
class PairValidation:
    compatible: bool
    error: str = ''
    warnings: List[str] = field(default_factory=list)
    overlap_pct: float = 0.0
    needs_reprojection: bool = False
    needs_resampling: bool = False

@dataclass
class QueryValidation:
    valid: bool
    error: str = ''

@dataclass
class TaskValidation:
    valid: bool
    error: str = ''

def validate_upload(content: bytes, filename: str) -> ValidationResult:
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        return ValidationResult(valid=False, error=f"File exceeds maximum size of {MAX_IMAGE_SIZE_MB}MB")

    if not content:
        return ValidationResult(valid=False, error="File is empty")

    warnings = []
    
    # Check magic bytes
    if content.startswith(b'II*\x00') or content.startswith(b'MM\x00*'):
        detected_format = 'TIFF'
        if rasterio:
            try:
                with rasterio.MemoryFile(content) as memfile:
                    with memfile.open() as dataset:
                        pass
            except Exception as e:
                return ValidationResult(valid=False, error=f"Invalid TIFF file: {str(e)}")
        else:
            warnings.append("rasterio not installed, skipping TIFF deep validation")
    elif content.startswith(b'\x89PNG\r\n\x1a\n') or content.startswith(b'\x89PNG'):
        detected_format = 'PNG'
        if Image:
            try:
                img = Image.open(io.BytesIO(content))
                img.verify()
            except Exception as e:
                return ValidationResult(valid=False, error=f"Invalid PNG file: {str(e)}")
        else:
            warnings.append("PIL not installed, skipping PNG deep validation")
    elif content.startswith(b'\xff\xd8\xff'):
        detected_format = 'JPEG'
        if Image:
            try:
                img = Image.open(io.BytesIO(content))
                img.verify()
            except Exception as e:
                return ValidationResult(valid=False, error=f"Invalid JPEG file: {str(e)}")
        else:
            warnings.append("PIL not installed, skipping JPEG deep validation")
    else:
        return ValidationResult(valid=False, error="Unsupported file format")

    return ValidationResult(
        valid=True,
        warnings=warnings,
        detected_format=detected_format
    )

def detect_modality(metadata: dict) -> Modality:
    bands = metadata.get('bands', 0)
    dtype = metadata.get('dtype', 'uint8')
    p2 = metadata.get('p2', 0.0)
    p98 = metadata.get('p98', 1.0)
    
    dynamic_range_ratio = (p98 / p2) if p2 > 0 else float('inf')

    if bands == 1:
        if dtype in ('uint16', 'float32') and dynamic_range_ratio > 50:
            return Modality.SAR
        elif dtype == 'uint8':
            return Modality.OPTICAL_RGB  # Fallback for optical grayscale

    elif bands == 3:
        if dtype in ('uint8', 'uint16'):
            return Modality.OPTICAL_RGB
            
    elif bands > 3:
        return Modality.MULTISPECTRAL

    return Modality.UNKNOWN
    
def validate_pair_compatibility(meta_a: dict, meta_b: dict) -> PairValidation:
    warnings = []
    
    crs_a = meta_a.get('crs')
    crs_b = meta_b.get('crs')
    needs_reprojection = False
    if crs_a and crs_b and crs_a != crs_b:
        needs_reprojection = True
        warnings.append(f"CRS mismatch: {crs_a} vs {crs_b}")

    res_a = meta_a.get('resolution', [1.0, 1.0])
    res_b = meta_b.get('resolution', [1.0, 1.0])
    needs_resampling = False
    
    res_diff_x = max(res_a[0]/res_b[0], res_b[0]/res_a[0]) if res_a[0] and res_b[0] else 1.0
    res_diff_y = max(res_a[1]/res_b[1], res_b[1]/res_a[1]) if res_a[1] and res_b[1] else 1.0
    
    if res_diff_x > 2.0 or res_diff_y > 2.0:
        needs_resampling = True
        warnings.append(f"Resolution difference > 2x: {res_a} vs {res_b}")

    dim_a = (meta_a.get('width'), meta_a.get('height'))
    dim_b = (meta_b.get('width'), meta_b.get('height'))
    if dim_a != dim_b:
        needs_resampling = True
        warnings.append(f"Dimension mismatch: {dim_a} vs {dim_b}")

    bounds_a = meta_a.get('bounds')
    bounds_b = meta_b.get('bounds')
    overlap_pct = 0.0
    
    if bounds_a and bounds_b:
        left = max(bounds_a[0], bounds_b[0])
        bottom = max(bounds_a[1], bounds_b[1])
        right = min(bounds_a[2], bounds_b[2])
        top = min(bounds_a[3], bounds_b[3])
        
        if right > left and top > bottom:
            intersection = (right - left) * (top - bottom)
            area_a = (bounds_a[2] - bounds_a[0]) * (bounds_a[3] - bounds_a[1])
            if area_a > 0:
                overlap_pct = (intersection / area_a) * 100.0
                
        if overlap_pct < 50.0:
            return PairValidation(compatible=False, error="Geographic overlap < 50%", overlap_pct=overlap_pct, needs_reprojection=needs_reprojection, needs_resampling=needs_resampling, warnings=warnings)
    
    return PairValidation(compatible=True, warnings=warnings, overlap_pct=overlap_pct, needs_reprojection=needs_reprojection, needs_resampling=needs_resampling)

def validate_query(query: str) -> QueryValidation:
    if not query or not query.strip():
        return QueryValidation(valid=False, error="Query cannot be empty")
    if len(query) > 2000:
        return QueryValidation(valid=False, error="Query length exceeds 2000 characters")
    return QueryValidation(valid=True)

def validate_task_inputs(task: str, has_image_a: bool, has_image_b: bool, has_sar: bool) -> TaskValidation:
    if task == "change_detection":
        if not (has_image_a and has_image_b):
            return TaskValidation(valid=False, error="Change detection requires two images")
    elif task == "fusion":
        if not (has_image_a and has_sar):
            return TaskValidation(valid=False, error="Fusion requires an optical image and a SAR image")
    elif task in ("vqa", "caption", "grounding", "change_vqa"):
        if not has_image_a:
            return TaskValidation(valid=False, error="Task requires at least one image")
    else:
        if not has_image_a:
            return TaskValidation(valid=False, error="Task requires at least one image")
            
    return TaskValidation(valid=True)

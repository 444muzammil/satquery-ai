"""
Classical Computer Vision Fallback Provider for SatQuery AI.
Deterministic CV fallback when deep learning models are unavailable.
"""
import base64
import logging
from typing import Any, Dict, List, Optional
import time

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

from providers.base import (
    RSVLMProvider,
    RSVLMResponse,
    ImageInput,
    TaskType,
)

logger = logging.getLogger("satquery.providers.fallback")


class ClassicalFallbackProvider(RSVLMProvider):
    """
    Deterministic classical computer vision provider for remote sensing.
    Acts as an honest fallback when VLM endpoints are unavailable.
    """

    @property
    def name(self) -> str:
        return "Classical CV Fallback"

    async def health_check(self) -> bool:
        """Classical CV is always available if OpenCV is installed."""
        return cv2 is not None and np is not None

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "classical_fallback",
            "supported_tasks": [
                TaskType.VQA.value,
                TaskType.CAPTION.value,
                TaskType.GROUNDING.value,
                TaskType.CHANGE_DETECTION.value,
                TaskType.OPTICAL_SAR_FUSION.value,
            ],
            "limitations": "Deterministic heuristic-based visual analysis. Lacks semantic reasoning.",
        }

    def _decode_base64_to_cv2(self, b64_str: str) -> Optional[Any]:
        """Decode base64 image data to an OpenCV BGR image."""
        if cv2 is None or np is None or not b64_str:
            return None
        try:
            if "," in b64_str:
                b64_str = b64_str.split(",")[1]
            img_bytes = base64.b64decode(b64_str)
            img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            logger.error(f"Failed to decode base64 image: {e}")
            return None

    def _classical_spectral_analysis(self, img: Any, query: str) -> RSVLMResponse:
        """Classical GLI and albedo stats for VQA."""
        h, w = img.shape[:2]
        b, g, r = cv2.split(img.astype(np.float32))
        denom = (2 * g + r + b)
        denom[denom == 0] = 1.0
        gli = (2 * g - r - b) / denom
        veg_coverage = np.sum(gli > 0.05) / (h * w) * 100.0
        albedo = np.mean(img)
        soil_type = "high-albedo reflective substrate" if albedo > 130 else "moderate-albedo loamy/fallow terrain"
        
        text = (f"Classical spectral analysis (RS VLM unavailable): Observation shows approximately {veg_coverage:.1f}% "
                f"active photosynthetic vegetation canopy surrounded by {soil_type}. "
                f"Note: Semantic question answering requires a remote-sensing VLM. Configure RS_VLM_PROVIDER to enable.")
        
        return RSVLMResponse(
            text=text,
            is_fallback=True,
            model_name="Classical CV Analysis (Fallback)"
        )

    def _classical_scene_summary(self, img: Any) -> RSVLMResponse:
        """Classical spectral composition for Captioning."""
        h, w = img.shape[:2]
        b, g, r = cv2.split(img.astype(np.float32))
        denom = (2 * g + r + b)
        denom[denom == 0] = 1.0
        gli = (2 * g - r - b) / denom
        veg_coverage = np.sum(gli > 0.05) / (h * w) * 100.0
        albedo = np.mean(img)
        soil_type = "high-albedo reflective substrate" if albedo > 130 else "moderate-albedo loamy/fallow terrain"

        text = (f"Classical scene summary (RS VLM unavailable): The scene is composed of {veg_coverage:.1f}% "
                f"vegetation and predominately {soil_type}. "
                f"Note: Semantic scene captioning requires a remote-sensing VLM. Configure RS_VLM_PROVIDER to enable.")
        
        return RSVLMResponse(
            text=text,
            is_fallback=True,
            model_name="Classical CV Analysis (Fallback)"
        )

    def _classical_grounding(self, img: Any, target: str) -> RSVLMResponse:
        """HSV color thresholding for grounding."""
        h, w = img.shape[:2]
        image_area = h * w
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        t_low = target.lower()

        # Open-vocabulary heuristic adaptation
        if any(w_kw in t_low for w_kw in ["water", "river", "lake", "canal", "flood", "ocean"]):
            m1 = cv2.inRange(hsv, np.array([85, 30, 30]), np.array([140, 255, 255]))
            m2 = cv2.inRange(hsv, np.array([5, 10, 40]), np.array([30, 220, 245]))
            mask = cv2.bitwise_or(m1, m2)
        elif any(w_kw in t_low for w_kw in ["vegetation", "forest", "crop", "tree", "agricultural", "farm"]):
            mask = cv2.inRange(hsv, np.array([25, 20, 20]), np.array([95, 255, 255]))
        elif any(w_kw in t_low for w_kw in ["build", "urban", "road", "railway", "structure", "development", "industrial"]):
            mask = cv2.inRange(hsv, np.array([0, 0, 85]), np.array([180, 50, 255]))
        elif any(w_kw in t_low for w_kw in ["barren", "soil", "sand", "dirt"]):
            mask = cv2.inRange(hsv, np.array([10, 15, 90]), np.array([35, 130, 255]))
        else:
            # Saliency and edge detection for arbitrary targets
            edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 70, 170)
            mask = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7,7), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions, valid_contours = [], []
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:30]:
            area = cv2.contourArea(c)
            if area > max(90, int(image_area * 0.0008)):
                x, y, bw, bh = cv2.boundingRect(c)
                regions.append({
                    "box": [round((y/h)*100, 1), round((x/w)*100, 1), round(((y+bh)/h)*100, 1), round(((x+bw)/w)*100, 1)],
                    "label": target.title(),
                    "actual_pct": (area / image_area) * 100.0,
                    "area_px": area
                })
                valid_contours.append(c.tolist())
                
        text = f"Classical heuristic localized {len(regions)} spatial boundaries for '{target}'."
        return RSVLMResponse(
            text=text,
            regions=regions,
            contours=valid_contours,
            is_fallback=True,
            model_name="Classical CV Analysis (Fallback)"
        )

    def _classical_change_detection(self, img_a: Any, img_b: Any) -> RSVLMResponse:
        """cv2.absdiff with morphological filtering."""
        # Ensure dimensions match for absolute difference
        h, w = img_a.shape[:2]
        if img_b.shape[:2] != (h, w):
            img_b = cv2.resize(img_b, (w, h), interpolation=cv2.INTER_LINEAR)

        diff = cv2.absdiff(cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY))
        _, thresh = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Enhanced morphological filtering
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        image_area = h * w
        regions, valid_contours = [], []
        total_change_px = 0

        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
            area = cv2.contourArea(c)
            if area > max(80, int(image_area * 0.001)):
                total_change_px += area
                x, y, bw, bh = cv2.boundingRect(c)
                regions.append({
                    "box": [round((y/h)*100, 1), round((x/w)*100, 1), round(((y+bh)/h)*100, 1), round(((x+bw)/w)*100, 1)],
                    "label": "Bi-Temporal Variance Zone",
                    "actual_pct": (area / image_area) * 100.0,
                    "area_px": area
                })
                valid_contours.append(c.tolist())

        pct_changed = (total_change_px / image_area) * 100.0 if image_area > 0 else 0.0
        
        text = (f"Pixel-level intensity differencing detected changes across {pct_changed:.2f}% of scene. "
                f"Semantic interpretation of what changed requires a change specialist model.")

        return RSVLMResponse(
            text=text,
            regions=regions,
            contours=valid_contours,
            coverage_pct=f"{pct_changed:.2f}%",
            is_fallback=True,
            model_name="Classical CV Analysis (Fallback)"
        )

    def _classical_optical_sar_fusion(self, opt_img: Any, sar_img: Any) -> RSVLMResponse:
        """SAR backscatter thresholding + optical edge analysis."""
        h, w = opt_img.shape[:2]
        if sar_img.shape[:2] != (h, w):
            sar_img = cv2.resize(sar_img, (w, h), interpolation=cv2.INTER_LINEAR)

        sar_gray = cv2.cvtColor(sar_img, cv2.COLOR_BGR2GRAY)
        sar_filtered = cv2.bilateralFilter(cv2.medianBlur(sar_gray, 5), 9, 75, 75)
        _, high_backscatter = cv2.threshold(sar_filtered, 200, 255, cv2.THRESH_BINARY)
        
        # Enhanced edge detection for optical image
        opt_gray = cv2.cvtColor(opt_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(opt_gray, 80, 180)
        edges = cv2.dilate(edges, np.ones((3,3), np.uint8))

        fused_built = cv2.bitwise_and(edges, high_backscatter)

        image_area = h * w
        contours, _ = cv2.findContours(fused_built, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions, valid_contours = [], []

        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            area = cv2.contourArea(c)
            if area > 120:
                x, y, bw, bh = cv2.boundingRect(c)
                regions.append({
                    "box": [round((y/h)*100, 1), round((x/w)*100, 1), round(((y+bh)/h)*100, 1), round(((x+bw)/w)*100, 1)],
                    "label": "Fused High Backscatter / Optical Boundary",
                    "actual_pct": (area / image_area) * 100.0,
                    "area_px": area
                })
                valid_contours.append(c.tolist())

        text = (f"Classical cross-modal feature co-occurrence analysis. "
                f"Identified {len(regions)} structural features from SAR high-backscatter / optical edge correlation.")

        return RSVLMResponse(
            text=text,
            regions=regions,
            contours=valid_contours,
            is_fallback=True,
            model_name="Classical CV Analysis (Fallback)"
        )

    async def analyze(
        self,
        images: List[ImageInput],
        query: str,
        task: TaskType,
        config: Optional[Dict[str, Any]] = None,
    ) -> RSVLMResponse:
        start_time = time.time()
        
        if cv2 is None or np is None:
            return RSVLMResponse(
                text="Error: OpenCV or NumPy is not installed. Classical fallback is unavailable.",
                error="Missing CV dependencies",
                is_fallback=True,
                model_name="Classical CV Analysis (Fallback)"
            )
            
        if not images:
            return RSVLMResponse(
                text="Error: No images provided for analysis.",
                error="Missing images",
                is_fallback=True,
                model_name="Classical CV Analysis (Fallback)"
            )
            
        img_a = self._decode_base64_to_cv2(images[0].clean_base64())
        if img_a is None:
            return RSVLMResponse(
                text="Error: Failed to decode primary image.",
                error="Image decoding failed",
                is_fallback=True,
                model_name="Classical CV Analysis (Fallback)"
            )

        try:
            if task == TaskType.VQA:
                resp = self._classical_spectral_analysis(img_a, query)
            elif task == TaskType.CAPTION:
                resp = self._classical_scene_summary(img_a)
            elif task == TaskType.GROUNDING:
                resp = self._classical_grounding(img_a, query)
            elif task == TaskType.CHANGE_DETECTION:
                if len(images) < 2:
                    resp = RSVLMResponse(
                        text="Error: Change detection requires two images.",
                        error="Missing temporal image",
                        is_fallback=True,
                        model_name="Classical CV Analysis (Fallback)"
                    )
                else:
                    img_b = self._decode_base64_to_cv2(images[1].clean_base64())
                    if img_b is None:
                        resp = RSVLMResponse(
                            text="Error: Failed to decode temporal image.",
                            error="Image decoding failed",
                            is_fallback=True,
                            model_name="Classical CV Analysis (Fallback)"
                        )
                    else:
                        resp = self._classical_change_detection(img_a, img_b)
            elif task == TaskType.OPTICAL_SAR_FUSION:
                if len(images) < 2:
                    resp = RSVLMResponse(
                        text="Error: Fusion requires optical and SAR images.",
                        error="Missing SAR image",
                        is_fallback=True,
                        model_name="Classical CV Analysis (Fallback)"
                    )
                else:
                    img_sar = self._decode_base64_to_cv2(images[1].clean_base64())
                    if img_sar is None:
                        resp = RSVLMResponse(
                            text="Error: Failed to decode SAR image.",
                            error="Image decoding failed",
                            is_fallback=True,
                            model_name="Classical CV Analysis (Fallback)"
                        )
                    else:
                        resp = self._classical_optical_sar_fusion(img_a, img_sar)
            else:
                resp = RSVLMResponse(
                    text=f"Classical fallback cannot handle task type: {task.value}",
                    error="Unsupported task type",
                    is_fallback=True,
                    model_name="Classical CV Analysis (Fallback)"
                )
                
        except Exception as e:
            logger.error(f"Error during classical analysis: {e}")
            resp = RSVLMResponse(
                text="Error occurred during classical analysis.",
                error=str(e),
                is_fallback=True,
                model_name="Classical CV Analysis (Fallback)"
            )
            
        resp.latency_ms = self._measure_latency(start_time)
        return resp

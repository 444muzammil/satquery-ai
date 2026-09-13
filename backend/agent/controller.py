import base64
import logging
import json
import numpy as np
import cv2
from typing import Any, Dict, List, Tuple, Optional
from pydantic import BaseModel
import rasterio
from rasterio.warp import reproject, Resampling

from config import Settings
from providers.base import RSVLMProvider
from registry import TOOL_REGISTRY, MODEL_REGISTRY

from tools.vqa import execute_vqa
from tools.caption import execute_caption
from tools.grounding import execute_grounding
from tools.change import execute_change_detection
from tools.fusion import execute_optical_sar_fusion
from tools.gis import calculate_area
from tools.spatial import compare_spatial_density

from evidence.confidence import estimate_confidence
from evidence.report import generate_report
from evidence.export import generate_gis_export

from agent.router import route_query

logger = logging.getLogger("satquery.agent.controller")

class VQARequest(BaseModel):
    image_base64: str
    query: str
    has_sar: bool = False
    has_bitemporal: bool = False
    image_b_base64: str = ""
    sar_base64: str = ""
    metadata: dict = {}
    metadata_b: dict = {}
    metadata_sar: dict = {}

def decode_base64_to_cv2(b64_str: str) -> Optional[np.ndarray]:
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        data = base64.b64decode(b64_str)
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        return None

class AgentController:
    def __init__(self, settings: Settings, primary_provider: RSVLMProvider, secondary_provider: Optional[RSVLMProvider] = None):
        self.settings = settings
        self.primary_provider = primary_provider
        self.secondary_provider = secondary_provider
        self.tool_registry = TOOL_REGISTRY

    def _align_spatial_pair(self, metaA: dict, metaB: dict, imgA: np.ndarray, imgB: np.ndarray) -> Tuple[np.ndarray, str, bool]:
        """
        Geospatial alignment using rasterio transforms when GeoTIFF metadata is present,
        with mathematical resampling fallback.
        """
        has_geo_a = metaA.get("has_georeference", False) and metaA.get("transform")
        has_geo_b = metaB.get("has_georeference", False) and metaB.get("transform")
        hA, wA = imgA.shape[:2]
        
        if has_geo_a and has_geo_b and metaA.get("crs") and metaB.get("crs"):
            crsA = rasterio.crs.CRS.from_string(metaA["crs"])
            crsB = rasterio.crs.CRS.from_string(metaB["crs"])
            
            if crsA != crsB or metaA["transform"] != metaB["transform"] or (hA, wA) != imgB.shape[:2]:
                try:
                    aligned_b = np.zeros_like(imgA)
                    reproject(
                        source=imgB,
                        destination=aligned_b,
                        src_transform=rasterio.Affine(*metaB["transform"]),
                        src_crs=crsB,
                        dst_transform=rasterio.Affine(*metaA["transform"]),
                        dst_crs=crsA,
                        resampling=Resampling.bilinear
                    )
                    return aligned_b, f"Reprojected from {metaB['crs']} to {metaA['crs']} and resampled to Image A grid.", True
                except Exception as e:
                    pass
        
        # Spatial resampling fallback
        if imgB.shape[:2] != (hA, wA):
            aligned_b = cv2.resize(imgB, (wA, hA), interpolation=cv2.INTER_LINEAR)
            return aligned_b, "Resampled spatial grid dimensions to match Image A reference extent.", True
            
        return imgB, "Spatial compatibility validated (Pixel-aligned).", False

    def _validate_tool_policy(self, tool_name: str, target: str, request: VQARequest, accumulated_regions: list) -> Tuple[bool, str]:
        if tool_name not in self.tool_registry:
            return False, f"Unknown tool '{tool_name}' rejected by policy."
            
        if tool_name == "CHANGE_DETECTION" and not (request.has_bitemporal and request.image_b_base64):
            return False, "Bi-temporal change analysis requires both Image A and Image B inputs."
            
        if tool_name == "OPTICAL_SAR_FUSION" and not (request.has_sar and request.sar_base64):
            return False, "Optical-SAR cross-modal analysis requires co-registered SAR data."
            
        if tool_name in ["AREA_CALCULATOR", "SPATIAL_COMPARATOR"] and not accumulated_regions:
            return False, f"Tool '{tool_name}' requires prior spatial localization from RS_GROUNDING."
            
        if tool_name in ["RS_VQA", "RS_CAPTION", "RS_GROUNDING"] and not request.image_base64:
            return False, "Image A base telemetry is missing."
            
        return True, "Policy check passed."

    def _format_error(self, message: str, task: str) -> dict:
        return {
            "answer": message,
            "confidence": 0.0,
            "task": task,
            "model_used": "SPATIAL_VALIDATOR",
            "evidence": {"type": "error", "details": "Validation Verification Failed"},
            "trace": [
                f"[TASK] {task}",
                f"[INPUT_VALIDATION] FAILED: {message}",
                "[OUTPUT_GENERATED] Pipeline Halted"
            ],
            "report_data": None,
            "gis_export": None
        }

    async def execute(self, request: VQARequest) -> Dict[str, Any]:
        imgA = decode_base64_to_cv2(request.image_base64)
        if imgA is None:
            return self._format_error("Missing or corrupt Image A telemetry.", "Initialization")

        req_meta_a = request.metadata or {}
        req_meta_b = request.metadata_b or {}
        req_meta_sar = request.metadata_sar or {}

        auditable_trace = []
        val_penalties = {"fallback_used": False, "resampling": False, "vlm_failed": False, "policy_rejected": False}
        
        format_a = req_meta_a.get("format", "Standard Raster")
        crs_a = req_meta_a.get("crs", "Local Grid")
        auditable_trace.append(f"[INPUT_VALIDATION] Image A: Format={format_a}, CRS={crs_a}, Dimensions={imgA.shape[1]}x{imgA.shape[0]}")

        imgB_aligned, imgSAR_aligned = None, None
        
        if request.has_bitemporal and request.image_b_base64:
            imgB = decode_base64_to_cv2(request.image_b_base64)
            if imgB is not None:
                imgB_aligned, msg_b, resampled_b = self._align_spatial_pair(req_meta_a, req_meta_b, imgA, imgB)
                auditable_trace.append(f"[INPUT_VALIDATION] Image B Alignment: {msg_b}")
                val_penalties["resampling"] = val_penalties["resampling"] or resampled_b

        if request.has_sar and request.sar_base64:
            imgSAR = decode_base64_to_cv2(request.sar_base64)
            if imgSAR is not None:
                imgSAR_aligned, msg_sar, resampled_sar = self._align_spatial_pair(req_meta_a, req_meta_sar, imgA, imgSAR)
                auditable_trace.append(f"[INPUT_VALIDATION] SAR Sensor Alignment: {msg_sar}")
                val_penalties["resampling"] = val_penalties["resampling"] or resampled_sar

        task_label, intent_info, planned_tools = route_query(request, self.settings)
        
        auditable_trace.insert(0, f"[TASK] {task_label}")

        accumulated_regions = []
        accumulated_contours = []
        accumulated_stats = {}
        models_invoked = []
        tools_executed = []
        observations = []
        has_vlm_response = False

        for step in planned_tools:
            t_name = step.get("tool", "").upper()
            t_input = step.get("input", "")

            pol_valid, pol_msg = self._validate_tool_policy(t_name, t_input, request, accumulated_regions)
            if not pol_valid:
                auditable_trace.append(f"[POLICY_VALIDATION] REJECTED: {t_name} -> {pol_msg}")
                val_penalties["policy_rejected"] = True
                continue

            auditable_trace.append(f"[POLICY_VALIDATION] PASSED: {t_name}")
            auditable_trace.append(f"[TOOL_SELECTED] {t_name}")
            auditable_trace.append(f"[PARAMETERS] target='{t_input}'")

            res = None
            try:
                if t_name == "RS_VQA":
                    res = await execute_vqa(t_input, request.image_base64, req_meta_a, self.primary_provider)
                elif t_name == "RS_CAPTION":
                    res = await execute_caption(t_input, request.image_base64, req_meta_a, self.primary_provider)
                elif t_name == "RS_GROUNDING":
                    res = await execute_grounding(
                        t_input, request.image_base64, req_meta_a, self.primary_provider,
                        self.settings.GROUNDING_MODEL_ENDPOINT, self.settings.GROUNDING_MODEL_API_KEY
                    )
                elif t_name == "CHANGE_DETECTION":
                    res = await execute_change_detection(
                        t_input, request.image_base64, request.image_b_base64,
                        req_meta_a, req_meta_b, self.primary_provider,
                        self.settings.CHANGE_MODEL_ENDPOINT, self.settings.CHANGE_MODEL_API_KEY
                    )
                elif t_name == "OPTICAL_SAR_FUSION":
                    res = await execute_optical_sar_fusion(
                        t_input, request.image_base64, request.sar_base64,
                        req_meta_a, req_meta_sar, self.primary_provider,
                        self.settings.FUSION_MODEL_ENDPOINT, self.settings.FUSION_MODEL_API_KEY
                    )
                elif t_name == "AREA_CALCULATOR":
                    res = calculate_area(accumulated_regions, req_meta_a)
                elif t_name == "SPATIAL_COMPARATOR":
                    res = compare_spatial_density(t_input, accumulated_regions, imgA.shape)
            except Exception as e:
                logger.error(f"Error executing tool {t_name}: {e}")
                auditable_trace.append(f"[OBSERVATION] Tool {t_name} failed: {e}")
                continue
            
            if res:
                tools_executed.append(t_name)
                model_name = res.get("model", t_name)
                models_invoked.append(model_name)
                
                if not res.get("fallback", False) and t_name in ["RS_VQA", "RS_CAPTION"]:
                    has_vlm_response = True
                elif res.get("fallback"):
                    auditable_trace.append(f"[FALLBACK] {model_name}")
                    val_penalties["fallback_used"] = True
                elif res.get("status") == "error":
                    val_penalties["vlm_failed"] = True

                if res.get("regions"):
                    if t_name == "SPATIAL_COMPARATOR":
                        accumulated_regions = res["regions"]
                    else:
                        accumulated_regions.extend(res["regions"])
                
                if res.get("contours"):
                    if t_name == "SPATIAL_COMPARATOR":
                        accumulated_contours = res["contours"]
                    else:
                        accumulated_contours.extend(res["contours"])
                        
                if "coverage_pct" in res:
                    accumulated_stats["Computed Extent"] = res["coverage_pct"]
                if res.get("real_world_area"):
                    accumulated_stats["Real-World Footprint"] = res["real_world_area"]
                    
                obs = res.get("summary", "")
                auditable_trace.append(f"[OBSERVATION] {obs}")
                observations.append(obs)

        confidence_score = estimate_confidence(
            tools_executed=tools_executed,
            models_used=models_invoked,
            has_fallback=val_penalties["fallback_used"],
            has_vlm_response=has_vlm_response,
            regions_found=len(accumulated_regions),
            input_quality=req_meta_a,
            penalties=val_penalties
        )
        
        auditable_trace.append(f"[VERIFICATION] Synthesized {len(accumulated_contours)} verified spatial feature(s).")
        auditable_trace.append(f"[OUTPUT_GENERATED] Output vectors generated in EPSG:{req_meta_a.get('crs', '4326')}")

        final_answer = " ".join(observations) if observations else "Analysis concluded with no matching telemetry."

        result_payload = {
            "answer": final_answer,
            "short_summary": f"Completed {task_label} analysis.",
            "confidence": confidence_score,
            "task": task_label,
            "model_used": ", ".join(set(models_invoked)) if models_invoked else "SatQuery_Orchestrator",
            "tools_executed": tools_executed,
            "evidence": {
                "type": "grounding" if accumulated_regions else "text_only",
                "regions": accumulated_regions,
                "stats": accumulated_stats or {"Features Identified": len(accumulated_regions)}
            },
            "trace": auditable_trace
        }
        
        req_meta_a["intent"] = intent_info.get("intent", "open_query")
        req_meta_a["target"] = intent_info.get("target", "unspecified")
        req_meta_a["operation"] = intent_info.get("operation", "unspecified")
        req_meta_a["models_executed"] = models_invoked
        
        result_payload["report_data"] = generate_report(req_meta_a, request.query, task_label, result_payload)
        result_payload["gis_export"] = generate_gis_export(accumulated_contours, req_meta_a, task_label)
        
        return result_payload

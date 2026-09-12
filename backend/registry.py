"""
SatQuery AI — Model and Tool Registry

Declares capabilities, modalities, and fallback strategies for each specialist.
The agent controller consults this registry to select appropriate tools.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("satquery.registry")


# =============================================================================
# MODEL REGISTRY — Declares each specialist's verified capabilities
# =============================================================================

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {

    "earthdial": {
        "name": "EarthDial",
        "provider": "runpod",
        "rs_adapted": True,
        "adaptation_evidence": {
            "dataset": "BigEarthNet.txt (arXiv:2603.29630)",
            "training_data": (
                "Sentinel-1 SAR (VV/VH) + Sentinel-2 multispectral (12 bands) "
                "+ Corine Land Cover labels + text annotations"
            ),
            "additional_datasets": ["RSVQA-LR", "RSVQA-HR", "VRSBench", "RSICD"],
            "method": (
                "Supervised fine-tuning of vision encoder + LLM backbone "
                "(Qwen2-VL / Vicuna) on remote sensing instruction-following data"
            ),
        },
        "capabilities": {
            "vqa": True,
            "caption": True,
            "grounding": "partial",     # Outputs bbox as text tokens — not pixel-precise
            "change_detection": False,  # Single-image model — needs agentic orchestration
            "change_vqa": False,
            "sar_analysis": True,       # Via BigEarthNet.txt SAR alignment
            "multispectral": True,
        },
        "supported_modalities": ["rgb", "multispectral", "sar"],
        "supported_input_count": [1],
        "supported_tasks": ["vqa", "caption", "grounding"],
        "supported_formats": ["geotiff", "tiff", "png", "jpeg"],
        "gpu_requirements": {
            "EarthDial_7B": {"fp16": "16-24 GB VRAM", "int4": "8-12 GB VRAM"},
            "EarthDial_4B": {"fp16": "8-10 GB VRAM", "int4": "4-6 GB VRAM"},
        },
        "config_keys": {
            "endpoint": "RS_VLM_ENDPOINT",
            "api_key": "RS_VLM_API_KEY",
            "model": "RS_VLM_MODEL",
        },
        "fallback": "classical_cv",
    },

    "geochat": {
        "name": "GeoChat",
        "provider": "runpod",
        "rs_adapted": True,
        "adaptation_evidence": {
            "dataset": "GeoChat-Instruct (507k instruction pairs)",
            "training_data": "12 aerial RS datasets: RSVQA, RSICD, UCM, DOTA, DIOR, FAIR1M",
            "method": "Fine-tuned LLaVA-1.5 (Vicuna-7B + CLIP-ViT-L/14)",
        },
        "capabilities": {
            "vqa": True,
            "caption": True,
            "grounding": True,          # Native bbox generation in dialogue
            "change_detection": False,
            "change_vqa": False,
            "sar_analysis": False,      # RGB optical only
            "multispectral": False,
        },
        "supported_modalities": ["rgb"],
        "supported_input_count": [1],
        "supported_tasks": ["vqa", "caption", "grounding"],
        "supported_formats": ["geotiff", "tiff", "png", "jpeg"],
        "config_keys": {
            "endpoint": "SECONDARY_VLM_ENDPOINT",
            "api_key": "SECONDARY_VLM_API_KEY",
        },
        "fallback": "classical_cv",
    },

    "change_specialist": {
        "name": "Change Detection Specialist",
        "provider": "runpod",
        "rs_adapted": True,
        "capabilities": {
            "change_detection": True,
            "change_vqa": True,
        },
        "supported_modalities": ["rgb", "multispectral"],
        "supported_input_count": [2],
        "supported_tasks": ["change_detection", "change_vqa"],
        "supported_formats": ["geotiff", "tiff", "png", "jpeg"],
        "config_keys": {
            "endpoint": "CHANGE_MODEL_ENDPOINT",
            "api_key": "CHANGE_MODEL_API_KEY",
        },
        "fallback": "classical_change_cv",
    },

    "fusion_specialist": {
        "name": "Optical-SAR Fusion Specialist",
        "provider": "runpod",
        "rs_adapted": True,
        "capabilities": {
            "optical_sar_fusion": True,
        },
        "supported_modalities": ["rgb+sar"],
        "supported_input_count": [2],
        "supported_tasks": ["fusion"],
        "supported_formats": ["geotiff", "tiff"],
        "config_keys": {
            "endpoint": "FUSION_MODEL_ENDPOINT",
            "api_key": "FUSION_MODEL_API_KEY",
        },
        "fallback": "classical_fusion_cv",
    },

    "gis_engine": {
        "name": "Deterministic GIS Engine",
        "provider": "local",
        "rs_adapted": False,            # Deterministic, no ML
        "capabilities": {
            "area_calculation": True,
            "spatial_comparison": True,
            "raster_operations": True,
            "crs_validation": True,
        },
        "supported_modalities": ["any"],
        "supported_input_count": [1, 2],
        "supported_tasks": ["area_calculator", "spatial_comparator"],
        "supported_formats": ["geotiff", "tiff"],
        "config_keys": {},
        "fallback": None,               # Deterministic — always available
    },
}


# =============================================================================
# TOOL REGISTRY — Maps tool names to their specialist model and configuration
# =============================================================================

TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "RS_VQA": {
        "description": "Visual Question Answering via RS-adapted VLM",
        "primary_model": "earthdial",
        "secondary_model": "geochat",
        "modality": "any_single",
        "requires_vlm": True,
    },
    "RS_CAPTION": {
        "description": "Scene Description via RS-adapted VLM",
        "primary_model": "earthdial",
        "secondary_model": "geochat",
        "modality": "any_single",
        "requires_vlm": True,
    },
    "RS_GROUNDING": {
        "description": "Text-Guided Spatial Region Localization",
        "primary_model": "earthdial",
        "secondary_model": "geochat",
        "modality": "any_single",
        "requires_vlm": False,          # Has classical CV fallback
    },
    "SPATIAL_COMPARATOR": {
        "description": "Comparative Regional Density Analysis",
        "primary_model": "gis_engine",
        "modality": "any_single",
        "requires_vlm": False,
    },
    "AREA_CALCULATOR": {
        "description": "Deterministic GIS Area Calculation",
        "primary_model": "gis_engine",
        "modality": "any",
        "requires_vlm": False,
    },
    "CHANGE_DETECTION": {
        "description": "Bi-Temporal Change Analysis",
        "primary_model": "change_specialist",
        "secondary_model": "earthdial",  # Agentic: analyze each separately
        "modality": "bitemporal",
        "requires_vlm": False,          # Has classical CV fallback
    },
    "OPTICAL_SAR_FUSION": {
        "description": "Optical-SAR Cross-Modal Analysis",
        "primary_model": "fusion_specialist",
        "secondary_model": "earthdial",  # Agentic: analyze each modality
        "modality": "optical+sar",
        "requires_vlm": False,          # Has classical CV fallback
    },
}


def get_model_info(model_id: str) -> Optional[Dict[str, Any]]:
    """Look up a model's declaration in the registry."""
    return MODEL_REGISTRY.get(model_id)


def get_tool_info(tool_name: str) -> Optional[Dict[str, Any]]:
    """Look up a tool's declaration in the registry."""
    return TOOL_REGISTRY.get(tool_name)


def get_available_tools() -> List[str]:
    """Return names of all registered tools."""
    return list(TOOL_REGISTRY.keys())


def model_supports_task(model_id: str, task: str) -> bool:
    """Check if a model supports a specific task."""
    model = MODEL_REGISTRY.get(model_id)
    if not model:
        return False
    caps = model.get("capabilities", {})
    return caps.get(task, False) is True


def model_supports_modality(model_id: str, modality: str) -> bool:
    """Check if a model supports a specific modality."""
    model = MODEL_REGISTRY.get(model_id)
    if not model:
        return False
    supported = model.get("supported_modalities", [])
    return modality in supported or "any" in supported

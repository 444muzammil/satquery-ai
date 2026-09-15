"""
SatQuery AI — FastAPI Backend Entrypoint
"""

import logging
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from providers.fallback import ClassicalFallbackProvider
from providers.runpod import RunPodProvider
from providers.local import LocalProvider
from providers.openai_vision import OpenAIVisionProvider
from providers.gemini_vision import GeminiVisionProvider
from preprocessing.ingest import process_upload
from agent.controller import AgentController, VQARequest
from agent.validator import validate_upload

logger = logging.getLogger("satquery.main")

app = FastAPI(title="SatQuery AI - Geospatial Intelligence Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize primary RS VLM provider based on settings
if settings.RS_VLM_PROVIDER == "runpod":
    primary_provider = RunPodProvider(settings)
elif settings.RS_VLM_PROVIDER == "openai":
    primary_provider = OpenAIVisionProvider(settings)
elif settings.RS_VLM_PROVIDER == "gemini":
    primary_provider = GeminiVisionProvider(settings)
elif settings.RS_VLM_PROVIDER == "local":
    primary_provider = LocalProvider(
        settings.RS_VLM_MODEL, 
        device="cuda" if settings.RS_VLM_LOCAL else "cpu"
    )
else:
    primary_provider = ClassicalFallbackProvider()

# Initialize the agent controller
controller = AgentController(
    settings=settings,
    primary_provider=primary_provider,
    secondary_provider=None  # Can be mapped to GeoChat in config later
)


@app.get("/health")
async def health_check():
    """System health check."""
    provider_status = await primary_provider.health_check()
    return {
        "status": "healthy",
        "provider": primary_provider.name,
        "provider_healthy": provider_status
    }


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """Handle image upload, validation, and metadata extraction."""
    try:
        content = await file.read()
        
        # 1. Validate format & size
        val_result = validate_upload(content, file.filename)
        if not val_result.valid:
            raise HTTPException(status_code=400, detail=val_result.error)
            
        # 2. Extract metadata and create preview
        result = process_upload(content, file.filename)
        
        # 3. Inject detected modality into metadata (from validator heuristic if available)
        if hasattr(val_result, 'detected_modality') and val_result.detected_modality and val_result.detected_modality.value != "unknown":
            result["metadata"]["modality"] = val_result.detected_modality.value
            
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")


@app.post("/api/vqa")
async def analyze_query(request: VQARequest) -> Dict[str, Any]:
    """Execute AI analysis based on user query and images."""
    try:
        # Pass to the agent controller pipeline
        result = await controller.execute(request)
        return result
    except Exception as e:
        logger.error(f"VQA execution failed: {e}", exc_info=True)
        # Return fallback error schema matching the expected frontend API contract
        return {
            "answer": f"System Error: {str(e)}",
            "short_summary": "System error occurred during execution.",
            "confidence": 0.0,
            "task": "UNKNOWN",
            "model_used": "None",
            "tools_executed": [],
            "evidence": {"type": "error", "regions": [], "stats": {}},
            "trace": [
                "[SYSTEM_ERROR] Fatal error during execution.", 
                f"[DETAIL] {str(e)}"
            ],
            "report_data": {},
            "gis_export": None
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
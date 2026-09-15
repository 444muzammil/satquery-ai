"""
SatQuery AI - Gemini Vision Mock Provider

Temporary provider mapping RS VLM calls to Gemini 2.5 Flash for internal demos.
"""
import httpx
import logging
import time
import asyncio
from typing import List, Optional

from .base import RSVLMProvider, RSVLMResponse, ImageInput, TaskType
from config import Settings

logger = logging.getLogger("satquery.providers.gemini_vision")

class GeminiVisionProvider(RSVLMProvider):
    def __init__(self, settings: Settings):
        self.model_name = "gemini-3.6-flash"
        self.timeout = settings.RS_VLM_TIMEOUT
        self.api_key = getattr(settings, "GEMINI_API_KEY", "")
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        
    async def analyze(self, images: List[ImageInput], query: str, task: TaskType, config=None) -> RSVLMResponse:
        start_t = time.time()
        
        if not self.api_key:
            return RSVLMResponse(
                text="",
                error="Gemini API key missing. Please configure GEMINI_API_KEY.",
                is_fallback=False,
                model_name=self.model_name
            )
        system_prompt = (
            "You are the Vision Module for SatQuery AI, an expert geospatial intelligence system. "
            "Analyze the provided satellite imagery with a highly professional, technical, and remote-sensing specific tone. "
            "Provide detailed observations about land use, infrastructure, and geography, but be highly concise. "
            "Strictly limit your response to a single paragraph of 4 to 7 sentences.\n\n"
            f"User Query: {query}"
        )
        parts = [{"text": system_prompt}]
        
        for img in images:
            b64 = img.clean_base64()
            parts.append({
                "inline_data": {
                    "mime_type": "image/png" if img.data.startswith("data:image/png") else "image/jpeg",
                    "data": b64
                }
            })
            
        payload = {
            "contents": [
                {
                    "parts": parts
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 250
            }
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    resp = await client.post(self.endpoint, json=payload, headers=headers)
                    if resp.status_code == 429:
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2.0 ** attempt)
                            continue
                    resp.raise_for_status()
                    data = resp.json()
                    
                    try:
                        answer = data["candidates"][0]["content"]["parts"][0]["text"]
                    except (KeyError, IndexError):
                        answer = "Could not extract text from Gemini response."
                    
                    return RSVLMResponse(
                        text=answer,
                        is_fallback=False,
                        model_name=self.model_name,
                        latency_ms=int((time.time() - start_t) * 1000)
                    )
            except Exception as e:
                # If it's a 429 from raise_for_status
                if attempt < max_retries - 1 and getattr(e, 'response', None) and getattr(e.response, 'status_code', None) == 429:
                    await asyncio.sleep(2.0 ** attempt)
                    continue
                
                logger.error(f"Gemini Vision API Error: {e}", exc_info=True)
                return RSVLMResponse(
                    text="",
                    error=f"API Error: {str(e)}",
                    is_fallback=True,
                    model_name="Fallback_CV",
                    latency_ms=int((time.time() - start_t) * 1000)
                )
    async def health_check(self) -> bool:
        return bool(self.api_key)

    def get_capabilities(self) -> dict:
        return {
            "vqa": True,
            "caption": True,
            "grounding": False
        }

    @property
    def name(self) -> str:
        return "Gemini 3.6 Flash"
